//! Count ops by type in register-based IR

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::env;
use std::fs;
use wasmparser::{Parser, Payload};

use reg_analyzer::regvm::{WasmToVReg, VRegInst};
use reg_analyzer::regalloc::{compute_live_intervals, linear_scan_alloc, count_spill_ops};

fn classify_op(inst: &VRegInst) -> &'static str {
    use VRegInst::*;
    match inst {
        I32Const { .. } | I64Const { .. } => "const",
        
        I32Add { .. } | I32Sub { .. } | I64Add { .. } | I64Sub { .. } => "add/sub",
        I32Mul { .. } | I64Mul { .. } => "mul",
        I32DivS { .. } | I32DivU { .. } | I32RemS { .. } | I32RemU { .. } |
        I64DivS { .. } | I64DivU { .. } | I64RemS { .. } | I64RemU { .. } => "div/rem",
        
        I32And { .. } | I32Or { .. } | I32Xor { .. } |
        I64And { .. } | I64Or { .. } | I64Xor { .. } => "bitwise",
        
        I32Shl { .. } | I32ShrU { .. } | I32ShrS { .. } |
        I32Rotl { .. } | I32Rotr { .. } |
        I64Shl { .. } | I64ShrU { .. } | I64ShrS { .. } => "shift",
        
        I32Eq { .. } | I32Ne { .. } | I32LtS { .. } | I32LtU { .. } |
        I32GtS { .. } | I32GtU { .. } | I32LeS { .. } | I32LeU { .. } |
        I32GeS { .. } | I32GeU { .. } |
        I64Eq { .. } | I64Ne { .. } | I64LtS { .. } | I64LtU { .. } |
        I64GtS { .. } | I64GtU { .. } | I64LeS { .. } | I64LeU { .. } |
        I64GeS { .. } | I64GeU { .. } => "compare",
        
        I32Eqz { .. } | I64Eqz { .. } | I32Clz { .. } | I32Ctz { .. } | 
        I32Popcnt { .. } | I64Clz { .. } | I64Ctz { .. } => "unary",
        
        I32WrapI64 { .. } | I64ExtendI32S { .. } | I64ExtendI32U { .. } |
        I32Extend8S { .. } | I32Extend16S { .. } => "convert",
        
        I32Load { .. } | I64Load { .. } | I32Load8U { .. } | I32Load8S { .. } |
        I32Load16U { .. } | I32Load16S { .. } | I64Load8U { .. } | I64Load8S { .. } |
        I64Load16U { .. } | I64Load16S { .. } | I64Load32U { .. } | I64Load32S { .. } => "load",
        
        I32Store { .. } | I64Store { .. } | I32Store8 { .. } | I32Store16 { .. } |
        I64Store8 { .. } | I64Store16 { .. } | I64Store32 { .. } => "store",
        
        LocalGet { .. } | LocalSet { .. } | LocalTee { .. } => "local (eliminated)",
        GlobalGet { .. } | GlobalSet { .. } => "global",
        
        Call { .. } | CallIndirect { .. } => "call",
        
        Block { .. } | Loop { .. } | If { .. } | Else { .. } | End { .. } => "block/end",
        Br { .. } | BrIf { .. } | BrTable { .. } => "branch",
        
        Return { .. } => "return",
        Select { .. } => "select",
        Unreachable | Nop => "nop/unreachable",
        Drop { .. } => "drop",
        MemorySize { .. } | MemoryGrow { .. } | MemoryCopy { .. } | MemoryFill { .. } => "memory",
        Mov { .. } => "mov",
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let wasm_path = args.get(1).map(|s| s.as_str()).unwrap_or("../target/wasm32-unknown-unknown/release/json_wasm_bench.wasm");
    let num_regs: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);

    let wasm_bytes = fs::read(wasm_path).context("Failed to read WASM file")?;

    let mut func_types: Vec<wasmparser::FuncType> = Vec::new();
    let mut type_indices: Vec<u32> = Vec::new();
    
    let mut op_counts: HashMap<&'static str, u32> = HashMap::new();
    let mut total_spill_loads = 0u32;
    let mut total_spill_stores = 0u32;
    let mut func_count = 0u32;

    for payload in Parser::new(0).parse_all(&wasm_bytes) {
        let payload = payload?;

        match &payload {
            Payload::TypeSection(reader) => {
                for rec_group in reader.clone() {
                    let rec_group = rec_group?;
                    for sub_type in rec_group.types() {
                        if let wasmparser::CompositeInnerType::Func(ft) = &sub_type.composite_type.inner {
                            func_types.push(ft.clone());
                        }
                    }
                }
            }
            Payload::FunctionSection(reader) => {
                for func in reader.clone() {
                    type_indices.push(func?);
                }
            }
            Payload::CodeSectionEntry(body) => {
                let type_idx = type_indices.get(func_count as usize).copied().unwrap_or(0);
                let func_type = func_types.get(type_idx as usize);
                let num_params = func_type.map(|ft| ft.params().len() as u32).unwrap_or(0);

                let mut num_locals = 0u32;
                for local in body.get_locals_reader()? {
                    let (count, _) = local?;
                    num_locals += count;
                }

                let mut converter = WasmToVReg::new(num_params, num_locals);
                for op in body.get_operators_reader()? {
                    converter.convert_op(&op?);
                }

                // Count ops by type
                for inst in &converter.instructions {
                    let cat = classify_op(inst);
                    *op_counts.entry(cat).or_insert(0) += 1;
                }

                // Count spills
                let intervals = compute_live_intervals(&converter.instructions);
                let alloc = linear_scan_alloc(&intervals, num_regs);
                let (loads, stores) = count_spill_ops(&converter.instructions, &alloc);
                total_spill_loads += loads;
                total_spill_stores += stores;

                func_count += 1;
            }
            _ => {}
        }
    }

    // Remove "local (eliminated)" from counts for final tally
    let local_ops = op_counts.remove("local (eliminated)").unwrap_or(0);
    
    // Add spill ops
    op_counts.insert("spill_load", total_spill_loads);
    op_counts.insert("spill_store", total_spill_stores);

    let total: u32 = op_counts.values().sum();

    // Sort by count descending
    let mut sorted: Vec<_> = op_counts.into_iter().collect();
    sorted.sort_by_key(|(_, count)| std::cmp::Reverse(*count));

    println!("=== Register-Based IR Op Breakdown ({} registers) ===\n", num_regs);
    println!("{:<20} {:>10} {:>10}", "Op Type", "Count", "%");
    println!("{}", "-".repeat(42));
    
    for (op, count) in &sorted {
        let pct = (*count as f64 / total as f64) * 100.0;
        println!("{:<20} {:>10} {:>9.1}%", op, count, pct);
    }
    
    println!("{}", "-".repeat(42));
    println!("{:<20} {:>10}", "TOTAL", total);
    println!("\n(Eliminated {} local.get/set/tee ops)", local_ops);

    Ok(())
}
