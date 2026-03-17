//! Full instruction-level breakdown of physical register IR
//!
//! Counts every individual PReg instruction (not categories) across the
//! entire WASM JSON parser module with 8 physical registers.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use wasmparser::{Parser, Payload};

use reg_analyzer::regvm::{WasmToVReg, RegInst, FuncSig};
use reg_analyzer::regalloc::{compute_live_intervals, linear_scan_alloc};
use reg_analyzer::preg_vm::lower_to_preg;

fn inst_name(inst: &RegInst) -> &'static str {
    use RegInst::*;
    match inst {
        I32Const { .. } => "i32.const",
        I64Const { .. } => "i64.const",
        I32Add { .. } => "i32.add",
        I32Sub { .. } => "i32.sub",
        I32Mul { .. } => "i32.mul",
        I32And { .. } => "i32.and",
        I32Or { .. } => "i32.or",
        I32Xor { .. } => "i32.xor",
        I32Shl { .. } => "i32.shl",
        I32ShrU { .. } => "i32.shr_u",
        I32ShrS { .. } => "i32.shr_s",
        I64Add { .. } => "i64.add",
        I64Sub { .. } => "i64.sub",
        I64Mul { .. } => "i64.mul",
        I64And { .. } => "i64.and",
        I64Or { .. } => "i64.or",
        I64Xor { .. } => "i64.xor",
        I64Shl { .. } => "i64.shl",
        I32DivU { .. } => "i32.div_u",
        I32Eqz { .. } => "i32.eqz",
        I32WrapI64 { .. } => "i32.wrap_i64",
        I32Clz { .. } => "i32.clz",
        I64Eqz { .. } => "i64.eqz",
        I32Eq { .. } => "i32.eq",
        I32Ne { .. } => "i32.ne",
        I32LtS { .. } => "i32.lt_s",
        I32LtU { .. } => "i32.lt_u",
        I32GtS { .. } => "i32.gt_s",
        I32GtU { .. } => "i32.gt_u",
        I32LeS { .. } => "i32.le_s",
        I32LeU { .. } => "i32.le_u",
        I32GeS { .. } => "i32.ge_s",
        I32GeU { .. } => "i32.ge_u",
        I32Load { .. } => "i32.load",
        I64Load { .. } => "i64.load",
        I32Load8U { .. } => "i32.load8_u",
        I32Load8S { .. } => "i32.load8_s",
        I32Load16U { .. } => "i32.load16_u",
        I32Load16S { .. } => "i32.load16_s",
        I32Store { .. } => "i32.store",
        I64Store { .. } => "i64.store",
        I32Store8 { .. } => "i32.store8",
        I32Store16 { .. } => "i32.store16",
        BrIf { .. } => "br_if",
        Br { .. } => "br",
        Label { .. } => "label",
        Call { .. } => "call",
        Move { .. } => "move",
        Spill { .. } => "spill",
        Reload { .. } => "reload",
        Select { .. } => "select",
        Unreachable => "unreachable",
        Nop => "nop",
        Drop { .. } => "drop",
        Block { .. } => "block",
        Loop { .. } => "loop",
        End => "end",
        LocalGet { .. } => "local.get",
        LocalSet { .. } => "local.set",
        GlobalGet { .. } => "global.get",
        GlobalSet { .. } => "global.set",
    }
}

fn main() -> Result<()> {
    let wasm_path = std::env::args().nth(1).unwrap_or_else(||
        "../json-crate-wasm/target/wasm32-unknown-unknown/release/json_crate_wasm.wasm".to_string());
    let num_regs: u32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(8);

    let wasm_bytes = fs::read(&wasm_path).context("Failed to read WASM file")?;

    let mut func_types: Vec<wasmparser::FuncType> = Vec::new();
    let mut type_indices: Vec<u32> = Vec::new();
    let mut func_names: HashMap<u32, String> = HashMap::new();

    // First pass: collect types, signatures, exports
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
                for func in reader.clone() { type_indices.push(func?); }
            }
            Payload::ExportSection(reader) => {
                for export in reader.clone() {
                    let export = export?;
                    if let wasmparser::ExternalKind::Func = export.kind {
                        func_names.insert(export.index, export.name.to_string());
                    }
                }
            }
            _ => {}
        }
    }

    let func_sigs: Vec<FuncSig> = type_indices.iter().map(|&type_idx| {
        let func_type = func_types.get(type_idx as usize);
        let num_params = func_type.map(|ft| ft.params().len() as u32).unwrap_or(0);
        let num_results = func_type.map(|ft| ft.results().len() as u32).unwrap_or(0);
        (num_params, num_results)
    }).collect();

    // Second pass: convert each function and count PReg instructions
    let mut total_counts: HashMap<&'static str, u32> = HashMap::new();
    let mut per_func: Vec<(String, HashMap<&'static str, u32>, u32)> = Vec::new();
    let mut func_count = 0u32;

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

            let mut converter = WasmToVReg::new_with_sigs(num_params, num_locals, func_sigs.clone());
            for op in body.get_operators_reader()? {
                converter.convert_op(&op?);
            }

            let intervals = compute_live_intervals(&converter.instructions);
            let alloc = linear_scan_alloc(&intervals, num_regs);
            let preg_insts = lower_to_preg(&converter.instructions, &alloc);

            let mut func_counts: HashMap<&'static str, u32> = HashMap::new();
            for inst in &preg_insts {
                let name = inst_name(inst);
                *func_counts.entry(name).or_insert(0) += 1;
                *total_counts.entry(name).or_insert(0) += 1;
            }

            let fname = func_names.get(&func_count)
                .cloned()
                .unwrap_or_else(|| format!("func_{}", func_count));
            let total: u32 = func_counts.values().sum();
            per_func.push((fname, func_counts, total));
            func_count += 1;
        }
    }

    let grand_total: u32 = total_counts.values().sum();

    // Ensure every defined instruction appears (even with 0 count)
    let all_instructions: &[&str] = &[
        "i32.const", "i64.const",
        "i32.add", "i32.sub", "i32.mul", "i32.and", "i32.or", "i32.xor",
        "i32.shl", "i32.shr_u", "i32.shr_s",
        "i64.add", "i64.sub", "i64.mul", "i64.and", "i64.or", "i64.xor", "i64.shl",
        "i32.div_u",
        "i32.eqz", "i32.wrap_i64", "i32.clz", "i64.eqz",
        "i32.eq", "i32.ne", "i32.lt_s", "i32.lt_u", "i32.gt_s", "i32.gt_u",
        "i32.le_s", "i32.le_u", "i32.ge_s", "i32.ge_u",
        "i32.load", "i64.load", "i32.load8_u", "i32.load8_s", "i32.load16_u", "i32.load16_s",
        "i32.store", "i64.store", "i32.store8", "i32.store16",
        "br_if", "br", "label", "call",
        "move", "spill", "reload", "select",
        "unreachable", "nop", "drop",
        "block", "loop", "end",
        "local.get", "local.set", "global.get", "global.set",
    ];
    for name in all_instructions {
        total_counts.entry(name).or_insert(0);
    }

    // Sort by count descending, then alphabetical for ties
    let mut sorted: Vec<_> = total_counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));

    // Print header
    println!("=== PReg Instruction Table ({} physical registers) ===\n", num_regs);
    println!("WASM: {}", wasm_path);
    println!("Functions: {}", func_count);
    println!("Total PReg instructions: {}\n", grand_total);

    println!("{:<20} {:>10} {:>10} {:>10}", "Instruction", "Count", "%", "Cumul%");
    println!("{}", "=".repeat(52));

    let mut cumulative = 0.0f64;
    for (name, count) in &sorted {
        let pct = (**count as f64 / grand_total as f64) * 100.0;
        cumulative += pct;
        println!("{:<20} {:>10} {:>9.2}% {:>9.2}%", name, count, pct, cumulative);
    }
    println!("{}", "=".repeat(52));
    println!("{:<20} {:>10} {:>9}  {:>9}", "TOTAL", grand_total, "100%", "");

    // Unique instruction count
    let used = sorted.iter().filter(|(_, c)| **c > 0).count();
    println!("\nUnique instructions used: {} / {} total", used, sorted.len());

    // Top 5 functions by instruction count
    per_func.sort_by(|a, b| b.2.cmp(&a.2));
    println!("\n--- Top 10 functions by PReg instruction count ---\n");
    println!("{:<30} {:>10} {:>10}", "Function", "Insts", "% of total");
    println!("{}", "-".repeat(52));
    for (name, _, total) in per_func.iter().take(10) {
        let pct = (*total as f64 / grand_total as f64) * 100.0;
        println!("{:<30} {:>10} {:>9.2}%", name, total, pct);
    }

    Ok(())
}
