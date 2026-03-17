//! Compare original WASM instructions vs register-based IR
//!
//! Shows which instruction types dominate in both representations.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use wasmparser::{Parser, Payload, Operator};

use reg_analyzer::regvm::{WasmToVReg, VRegInst, FuncSig};
use reg_analyzer::regalloc::{compute_live_intervals, linear_scan_alloc};
use reg_analyzer::preg_vm::lower_to_preg;

fn classify_wasm_op(op: &Operator) -> &'static str {
    use Operator::*;
    match op {
        I32Const { .. } | I64Const { .. } | F32Const { .. } | F64Const { .. } => "const",

        I32Add | I32Sub | I64Add | I64Sub | F32Add | F32Sub | F64Add | F64Sub => "add/sub",
        I32Mul | I64Mul | F32Mul | F64Mul => "mul",
        I32DivS | I32DivU | I32RemS | I32RemU |
        I64DivS | I64DivU | I64RemS | I64RemU |
        F32Div | F64Div => "div/rem",

        I32And | I32Or | I32Xor | I64And | I64Or | I64Xor => "bitwise",

        I32Shl | I32ShrU | I32ShrS | I32Rotl | I32Rotr |
        I64Shl | I64ShrU | I64ShrS | I64Rotl | I64Rotr => "shift",

        I32Eq | I32Ne | I32LtS | I32LtU | I32GtS | I32GtU | I32LeS | I32LeU | I32GeS | I32GeU |
        I64Eq | I64Ne | I64LtS | I64LtU | I64GtS | I64GtU | I64LeS | I64LeU | I64GeS | I64GeU |
        F32Eq | F32Ne | F32Lt | F32Gt | F32Le | F32Ge |
        F64Eq | F64Ne | F64Lt | F64Gt | F64Le | F64Ge => "compare",

        I32Eqz | I64Eqz | I32Clz | I32Ctz | I32Popcnt | I64Clz | I64Ctz | I64Popcnt => "unary",

        I32WrapI64 | I64ExtendI32S | I64ExtendI32U |
        I32Extend8S | I32Extend16S | I64Extend8S | I64Extend16S | I64Extend32S |
        I32TruncF32S | I32TruncF32U | I32TruncF64S | I32TruncF64U |
        I64TruncF32S | I64TruncF32U | I64TruncF64S | I64TruncF64U |
        F32ConvertI32S | F32ConvertI32U | F32ConvertI64S | F32ConvertI64U |
        F64ConvertI32S | F64ConvertI32U | F64ConvertI64S | F64ConvertI64U |
        F32DemoteF64 | F64PromoteF32 |
        I32ReinterpretF32 | I64ReinterpretF64 | F32ReinterpretI32 | F64ReinterpretI64 => "convert",

        I32Load { .. } | I64Load { .. } | F32Load { .. } | F64Load { .. } |
        I32Load8S { .. } | I32Load8U { .. } | I32Load16S { .. } | I32Load16U { .. } |
        I64Load8S { .. } | I64Load8U { .. } | I64Load16S { .. } | I64Load16U { .. } |
        I64Load32S { .. } | I64Load32U { .. } => "load",

        I32Store { .. } | I64Store { .. } | F32Store { .. } | F64Store { .. } |
        I32Store8 { .. } | I32Store16 { .. } |
        I64Store8 { .. } | I64Store16 { .. } | I64Store32 { .. } => "store",

        LocalGet { .. } => "local.get",
        LocalSet { .. } => "local.set",
        LocalTee { .. } => "local.tee",
        GlobalGet { .. } | GlobalSet { .. } => "global",

        Call { .. } | CallIndirect { .. } => "call",

        Block { .. } | Loop { .. } | If { .. } | Else | End => "block/end",
        Br { .. } | BrIf { .. } | BrTable { .. } => "branch",

        Return => "return",
        Select => "select",
        Unreachable | Nop => "nop/unreachable",
        Drop => "drop",
        MemorySize { .. } | MemoryGrow { .. } | MemoryCopy { .. } | MemoryFill { .. } => "memory",

        _ => "other",
    }
}

fn classify_vreg_op(inst: &VRegInst) -> &'static str {
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

        LocalGet { .. } => "local.get",
        LocalSet { .. } => "local.set",
        LocalTee { .. } => "local.tee",
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

fn classify_preg_op(inst: &reg_analyzer::regvm::RegInst) -> &'static str {
    use reg_analyzer::regvm::RegInst::*;
    match inst {
        I32Const { .. } | I64Const { .. } => "const",
        I32Add { .. } | I32Sub { .. } | I64Add { .. } | I64Sub { .. } => "add/sub",
        I32Mul { .. } | I64Mul { .. } => "mul",
        I32DivU { .. } => "div/rem",
        I32And { .. } | I32Or { .. } | I32Xor { .. } |
        I64And { .. } | I64Or { .. } | I64Xor { .. } => "bitwise",
        I32Shl { .. } | I32ShrU { .. } | I32ShrS { .. } | I64Shl { .. } => "shift",
        I32Eq { .. } | I32Ne { .. } | I32LtS { .. } | I32LtU { .. } |
        I32GtS { .. } | I32GtU { .. } | I32LeS { .. } | I32LeU { .. } |
        I32GeS { .. } | I32GeU { .. } => "compare",
        I32Eqz { .. } | I32Clz { .. } | I64Eqz { .. } | I32WrapI64 { .. } => "unary/convert",
        I32Load { .. } | I64Load { .. } | I32Load8U { .. } | I32Load8S { .. } |
        I32Load16U { .. } | I32Load16S { .. } => "load",
        I32Store { .. } | I64Store { .. } | I32Store8 { .. } | I32Store16 { .. } => "store",
        LocalGet { .. } => "local.get",
        LocalSet { .. } => "local.set",
        GlobalGet { .. } | GlobalSet { .. } => "global",
        Call { .. } => "call",
        Block { .. } | Loop { .. } | If { .. } | Else { .. } | End { .. } | Label { .. } => "block/end",
        Br { .. } | BrIf { .. } | Return => "branch",
        Move { .. } => "move",
        Spill { .. } => "spill",
        Reload { .. } => "reload",
        Select { .. } => "select",
        Unreachable | Nop | Drop { .. } => "nop/drop",
    }
}

fn main() -> Result<()> {
    let wasm_path = "../json-crate-wasm/target/wasm32-unknown-unknown/release/json_crate_wasm.wasm";
    let num_regs = 8u32;

    let wasm_bytes = fs::read(wasm_path).context("Failed to read WASM file")?;

    let mut func_types: Vec<wasmparser::FuncType> = Vec::new();
    let mut type_indices: Vec<u32> = Vec::new();

    let mut wasm_counts: HashMap<&'static str, u32> = HashMap::new();
    let mut vreg_counts: HashMap<&'static str, u32> = HashMap::new();
    let mut preg_counts: HashMap<&'static str, u32> = HashMap::new();

    let mut func_count = 0u32;
    let mut total_wasm_ops = 0u32;

    // First pass: collect types
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
            _ => {}
        }
    }

    // Build function signature table
    let func_sigs: Vec<FuncSig> = type_indices.iter().map(|&type_idx| {
        let func_type = func_types.get(type_idx as usize);
        let num_params = func_type.map(|ft| ft.params().len() as u32).unwrap_or(0);
        let num_results = func_type.map(|ft| ft.results().len() as u32).unwrap_or(0);
        (num_params, num_results)
    }).collect();

    // Second pass: analyze functions
    for payload in Parser::new(0).parse_all(&wasm_bytes) {
        let payload = payload?;
        if let Payload::CodeSectionEntry(body) = &payload {
            let type_idx = type_indices.get(func_count as usize).copied().unwrap_or(0);
            let func_type = func_types.get(type_idx as usize);
            let num_params = func_type.map(|ft| ft.params().len() as u32).unwrap_or(0);

            let mut num_locals = 0u32;
            for local in body.get_locals_reader()? {
                let (count, _) = local?;
                num_locals += count;
            }

            // Count original WASM ops
            for op in body.get_operators_reader()? {
                let op = op?;
                let cat = classify_wasm_op(&op);
                *wasm_counts.entry(cat).or_insert(0) += 1;
                total_wasm_ops += 1;
            }

            // Convert to VReg IR
            let mut converter = WasmToVReg::new_with_sigs(num_params, num_locals, func_sigs.clone());
            for op in body.get_operators_reader()? {
                converter.convert_op(&op?);
            }

            // Count VReg ops
            for inst in &converter.instructions {
                let cat = classify_vreg_op(inst);
                *vreg_counts.entry(cat).or_insert(0) += 1;
            }

            // Do register allocation and lower to PReg
            let intervals = compute_live_intervals(&converter.instructions);
            let alloc = linear_scan_alloc(&intervals, num_regs);
            let preg_insts = lower_to_preg(&converter.instructions, &alloc);

            // Count PReg ops
            for inst in &preg_insts {
                let cat = classify_preg_op(inst);
                *preg_counts.entry(cat).or_insert(0) += 1;
            }

            func_count += 1;
        }
    }

    let total_vreg: u32 = vreg_counts.values().sum();
    let total_preg: u32 = preg_counts.values().sum();

    // Collect all categories
    let mut all_cats: std::collections::BTreeSet<&'static str> = std::collections::BTreeSet::new();
    all_cats.extend(wasm_counts.keys().copied());
    all_cats.extend(vreg_counts.keys().copied());
    all_cats.extend(preg_counts.keys().copied());

    // Sort by WASM count descending
    let mut sorted_cats: Vec<_> = all_cats.into_iter().collect();
    sorted_cats.sort_by_key(|cat| std::cmp::Reverse(wasm_counts.get(cat).copied().unwrap_or(0)));

    println!("=== Instruction Analysis: WASM vs Register-Based IR ===\n");
    println!("WASM file: {}", wasm_path);
    println!("Functions: {}", func_count);
    println!("Physical registers: {}\n", num_regs);

    println!("{:<15} {:>10} {:>8} {:>10} {:>8} {:>10} {:>8}",
             "Category", "WASM", "%", "VReg", "%", "PReg", "%");
    println!("{}", "=".repeat(75));

    for cat in &sorted_cats {
        let wasm = wasm_counts.get(cat).copied().unwrap_or(0);
        let vreg = vreg_counts.get(cat).copied().unwrap_or(0);
        let preg = preg_counts.get(cat).copied().unwrap_or(0);

        let wasm_pct = if total_wasm_ops > 0 { (wasm as f64 / total_wasm_ops as f64) * 100.0 } else { 0.0 };
        let vreg_pct = if total_vreg > 0 { (vreg as f64 / total_vreg as f64) * 100.0 } else { 0.0 };
        let preg_pct = if total_preg > 0 { (preg as f64 / total_preg as f64) * 100.0 } else { 0.0 };

        println!("{:<15} {:>10} {:>7.1}% {:>10} {:>7.1}% {:>10} {:>7.1}%",
                 cat, wasm, wasm_pct, vreg, vreg_pct, preg, preg_pct);
    }

    println!("{}", "=".repeat(75));
    println!("{:<15} {:>10} {:>8} {:>10} {:>8} {:>10} {:>8}",
             "TOTAL", total_wasm_ops, "100%", total_vreg, "100%", total_preg, "100%");

    // Summary
    println!("\n=== Key Observations ===\n");

    let local_get_wasm = wasm_counts.get("local.get").copied().unwrap_or(0);
    let local_set_wasm = wasm_counts.get("local.set").copied().unwrap_or(0);
    let local_tee_wasm = wasm_counts.get("local.tee").copied().unwrap_or(0);
    let total_local_wasm = local_get_wasm + local_set_wasm + local_tee_wasm;

    let local_get_vreg = vreg_counts.get("local.get").copied().unwrap_or(0);
    let local_set_vreg = vreg_counts.get("local.set").copied().unwrap_or(0);

    let spill_preg = preg_counts.get("spill").copied().unwrap_or(0);
    let reload_preg = preg_counts.get("reload").copied().unwrap_or(0);
    let move_preg = preg_counts.get("move").copied().unwrap_or(0);

    println!("1. Local variable operations in WASM: {} ({:.1}% of total)",
             total_local_wasm, (total_local_wasm as f64 / total_wasm_ops as f64) * 100.0);
    println!("   - local.get: {}", local_get_wasm);
    println!("   - local.set: {}", local_set_wasm);
    println!("   - local.tee: {}", local_tee_wasm);

    println!("\n2. Register VM overhead:");
    println!("   - Spill operations: {}", spill_preg);
    println!("   - Reload operations: {}", reload_preg);
    println!("   - Move operations: {}", move_preg);
    println!("   - Total overhead: {} ({:.1}% of PReg total)",
             spill_preg + reload_preg + move_preg,
             ((spill_preg + reload_preg + move_preg) as f64 / total_preg as f64) * 100.0);

    println!("\n3. Instruction count comparison:");
    println!("   - WASM: {} instructions", total_wasm_ops);
    println!("   - VReg: {} instructions ({:+.1}%)",
             total_vreg,
             ((total_vreg as f64 - total_wasm_ops as f64) / total_wasm_ops as f64) * 100.0);
    println!("   - PReg: {} instructions ({:+.1}%)",
             total_preg,
             ((total_preg as f64 - total_wasm_ops as f64) / total_wasm_ops as f64) * 100.0);

    // Top 5 categories
    println!("\n4. Top 5 instruction categories (by WASM count):");
    for (i, cat) in sorted_cats.iter().take(5).enumerate() {
        let wasm = wasm_counts.get(cat).copied().unwrap_or(0);
        let pct = (wasm as f64 / total_wasm_ops as f64) * 100.0;
        println!("   {}. {}: {} ({:.1}%)", i + 1, cat, wasm, pct);
    }

    Ok(())
}
