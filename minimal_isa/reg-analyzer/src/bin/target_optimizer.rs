//! Register targeting optimizer.
//!
//! Approach: run parse_json_deep with unconstrained allocation, capture every
//! executed specialized opcode (e.g. "i32.add.r0.r1.r2"), count frequencies,
//! then pick the top-N most frequent as the ISA. Instructions not in the ISA
//! need moves to remap to a compatible variant.
//!
//! Usage: target_optimizer [wasm_path]

use anyhow::{Context, Result, anyhow};
use std::collections::{HashMap, HashSet};
use std::fs;
use wasmparser::{Parser, Payload};
use reg_analyzer::regvm::{
    WasmToVReg, VRegInst, FuncSig, SlotType,
    GLOBALS_MEM_BASE, FRAME_SP_ADDR, FRAME_STACK_BASE, SLOT_SIZE,
};
use reg_analyzer::regalloc::{compute_live_intervals, linear_scan_alloc, RegAllocResult};
use reg_analyzer::preg_vm::PRegVM;
use reg_analyzer::interpreter::Value;

fn val_type_to_slot(vt: &wasmparser::ValType) -> SlotType {
    match vt {
        wasmparser::ValType::I64 => SlotType::I64,
        _ => SlotType::I32,
    }
}

fn generate_test_json(target_size: usize) -> String {
    let mut json = String::from(r#"{"data":{"users":["#);
    let user_template = r#"{"name":"user_XXX","email":"user_XXX@example.com","age":25,"active":true,"tags":["a","b","c"]}"#;
    let mut i = 0;
    while json.len() < target_size {
        if i > 0 { json.push(','); }
        json.push_str(&user_template.replace("XXX", &format!("{:04}", i)));
        i += 1;
    }
    json.push_str(r#"]},"meta":{"count":"#);
    json.push_str(&i.to_string());
    json.push_str(r#"}}"#);
    json
}

fn main() -> Result<()> {
    let num_regs: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let wasm_path = std::env::args().nth(2).unwrap_or_else(||
        "../json-crate-wasm/target/wasm32-unknown-unknown/release/json_crate_wasm.wasm".to_string());

    let wasm_bytes = fs::read(&wasm_path).context("Failed to read WASM file")?;

    // Parse WASM
    let mut func_types: Vec<wasmparser::FuncType> = Vec::new();
    let mut type_indices: Vec<u32> = Vec::new();
    let mut func_names: HashMap<u32, String> = HashMap::new();
    let mut code_bodies: Vec<wasmparser::FunctionBody> = Vec::new();
    let mut global_inits: Vec<i32> = Vec::new();
    let mut global_types: Vec<SlotType> = Vec::new();
    let mut data_segments: Vec<(u32, Vec<u8>)> = Vec::new();

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
            Payload::GlobalSection(reader) => {
                for global in reader.clone() {
                    let global = global?;
                    let init_expr = global.init_expr.get_binary_reader();
                    let mut init_val = 0i32;
                    for op in wasmparser::OperatorsReader::new(init_expr) {
                        if let Ok(wasmparser::Operator::I32Const { value }) = op {
                            init_val = value;
                            break;
                        }
                    }
                    global_inits.push(init_val);
                    global_types.push(val_type_to_slot(&global.ty.content_type));
                }
            }
            Payload::ExportSection(reader) => {
                for export in reader.clone() {
                    let export = export?;
                    if let wasmparser::ExternalKind::Func = export.kind {
                        func_names.insert(export.index, export.name.to_string());
                    }
                }
            }
            Payload::CodeSectionEntry(body) => { code_bodies.push(body.clone()); }
            Payload::DataSection(reader) => {
                for data in reader.clone() {
                    let data = data?;
                    if let wasmparser::DataKind::Active { memory_index: 0, offset_expr } = data.kind {
                        let mut offset = 0u32;
                        for op in wasmparser::OperatorsReader::new(offset_expr.get_binary_reader()) {
                            if let Ok(wasmparser::Operator::I32Const { value }) = op {
                                offset = value as u32;
                                break;
                            }
                        }
                        data_segments.push((offset, data.data.to_vec()));
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

    // Compile all functions with ML + N regs
    let mut vreg_funcs: Vec<(Vec<VRegInst>, RegAllocResult, u32, u32)> = Vec::new();

    for (func_count, body) in code_bodies.iter().enumerate() {
        let type_idx = type_indices.get(func_count).copied().unwrap_or(0);
        let func_type = func_types.get(type_idx as usize);
        let num_params = func_type.map(|ft| ft.params().len() as u32).unwrap_or(0);
        let mut num_locals = 0u32;
        let mut local_types: Vec<SlotType> = Vec::new();
        if let Some(ft) = func_type {
            for p in ft.params() { local_types.push(val_type_to_slot(p)); }
        }
        for local in body.get_locals_reader()? {
            let (count, vt) = local?;
            num_locals += count;
            let st = val_type_to_slot(&vt);
            for _ in 0..count { local_types.push(st); }
        }

        let mut converter = WasmToVReg::new_memory_lowered(
            num_params, num_locals, func_sigs.clone(),
            local_types, global_types.clone(),
        );
        for op in body.get_operators_reader()? {
            converter.convert_op(&op?);
        }

        let intervals = compute_live_intervals(&converter.instructions);
        let alloc = linear_scan_alloc(&intervals, num_regs);
        vreg_funcs.push((converter.instructions, alloc, num_params, num_locals));
    }

    // Set up VM and run
    let mut max_spill_slots = 0u32;
    for (_, alloc, _, _) in &vreg_funcs {
        max_spill_slots = max_spill_slots.max(alloc.num_spill_slots);
    }
    let mut vm = PRegVM::new(num_regs as usize, max_spill_slots as usize + 64, 256);

    for (vreg_insts, alloc, num_params, num_locals) in &vreg_funcs {
        vm.add_vreg_function_ml(
            vreg_insts.clone(),
            RegAllocResult {
                vreg_to_preg: alloc.vreg_to_preg.clone(),
                spilled: alloc.spilled.clone(),
                spill_slots: alloc.spill_slots.clone(),
                num_spill_slots: alloc.num_spill_slots,
            },
            *num_params, *num_locals,
        );
    }

    for (i, val) in global_inits.iter().enumerate() {
        if i < vm.globals.len() { vm.globals[i] = Value::I32(*val); }
    }
    for (offset, data) in &data_segments {
        vm.write_memory(*offset as usize, data);
    }

    let test_json = generate_test_json(2048);
    let json_data = test_json.as_bytes();
    vm.write_memory(0, json_data);
    vm.globals[0] = Value::I32(global_inits.first().copied().unwrap_or(1048576));

    vm.write_memory(FRAME_SP_ADDR as usize, &FRAME_STACK_BASE.to_le_bytes());
    for (i, val) in global_inits.iter().enumerate() {
        let addr = GLOBALS_MEM_BASE as usize + (i as u32 * SLOT_SIZE) as usize;
        vm.write_memory(addr, &(*val as u32).to_le_bytes());
    }
    let frame_base = FRAME_STACK_BASE as usize;
    vm.write_memory(frame_base, &0u32.to_le_bytes());
    vm.write_memory(frame_base + SLOT_SIZE as usize, &(json_data.len() as u32).to_le_bytes());

    vm.enable_reg_trace();

    eprintln!("Executing parse_json_deep ({} regs, ML)...", num_regs);

    let func_idx = func_names.iter()
        .find(|(_, name)| *name == "parse_json_deep")
        .map(|(idx, _)| *idx)
        .ok_or_else(|| anyhow!("No parse_json_deep"))?;

    let func = &vreg_funcs[func_idx as usize];
    let result = vm.execute_vreg(&func.0, &func.1);
    let nodes = result.map(|v| v.as_i32() as u32).unwrap_or(0);
    let reg_trace = vm.reg_trace.take().unwrap_or_default();
    let trace_len = reg_trace.len();

    eprintln!("Trace: {} instructions, {} nodes", trace_len, nodes);

    // === Build specialized opcodes and count frequencies ===
    let mut freq: HashMap<String, u64> = HashMap::new();
    for (name, dsts, srcs) in &reg_trace {
        let mut parts = vec![name.to_string()];
        for r in dsts { parts.push(format!("r{}", r)); }
        for r in srcs { parts.push(format!("r{}", r)); }
        let spec = parts.join(".");
        *freq.entry(spec).or_insert(0) += 1;
    }

    // Sort by frequency
    let mut sorted: Vec<_> = freq.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));

    let total_unique = sorted.len();

    println!("=== Hot Opcode ISA Selection ({} regs, ML) ===\n", num_regs);
    println!("WASM:  {}", wasm_path);
    println!("Nodes: {}", nodes);
    println!("Dynamic trace: {} instructions", trace_len);
    println!("Total unique specialized ops: {}\n", total_unique);

    // === Show top-N coverage sweep ===
    println!("{:>6} {:>12} {:>10} {:>10} {:>12}", "ISA sz", "Covered", "Cover%", "Uncov%", "Uncov insts");
    println!("{}", "=".repeat(55));

    let budgets = [50, 75, 100, 125, 150, 175, 188, 200, 220, 250, 300, 350, total_unique];
    let mut cumul = 0u64;
    let mut prev_printed = 0;

    for (i, (name, count)) in sorted.iter().enumerate() {
        cumul += **count;
        let isa_size = i + 1 + 12; // +12 for move instructions

        for &budget in &budgets {
            if isa_size >= budget && prev_printed < budget && budget <= total_unique + 12 {
                let uncovered = trace_len as u64 - cumul;
                let cover_pct = cumul as f64 / trace_len as f64 * 100.0;
                let uncov_pct = uncovered as f64 / trace_len as f64 * 100.0;
                println!("{:>6} {:>12} {:>9.2}% {:>9.2}% {:>12}",
                    budget, cumul, cover_pct, uncov_pct, uncovered);
                prev_printed = budget;
            }
        }
    }
    // Always print the full ISA line
    if prev_printed < total_unique + 12 {
        println!("{:>6} {:>12} {:>9.2}% {:>9.2}% {:>12}",
            total_unique + 12, trace_len, 100.0, 0.0, 0);
    }
    println!("{}", "=".repeat(55));

    // === Detailed: top 50 hottest opcodes ===
    println!("\n--- Top 50 hottest specialized opcodes ---");
    println!("{:<40} {:>10} {:>8} {:>8}", "Specialized Op", "Count", "%", "Cumul%");
    println!("{}", "-".repeat(68));
    let mut c = 0u64;
    for (name, count) in sorted.iter().take(50) {
        c += **count;
        let pct = **count as f64 / trace_len as f64 * 100.0;
        let cpct = c as f64 / trace_len as f64 * 100.0;
        println!("{:<40} {:>10} {:>7.2}% {:>7.1}%", name, count, pct, cpct);
    }
    println!("{}", "-".repeat(68));

    // === Per base-opcode breakdown ===
    println!("\n--- Variants per base opcode (dynamic) ---");
    let mut by_base: HashMap<&str, (usize, u64)> = HashMap::new(); // (variants, total_count)
    for (name, dsts, srcs) in &reg_trace {
        let mut parts = vec![name.to_string()];
        for r in dsts { parts.push(format!("r{}", r)); }
        for r in srcs { parts.push(format!("r{}", r)); }
        let spec = parts.join(".");
        let e = by_base.entry(name).or_insert((0, 0));
        e.1 += 1;
    }
    // Count unique variants per base
    let mut base_variants: HashMap<&str, HashSet<String>> = HashMap::new();
    for (name, dsts, srcs) in &reg_trace {
        let mut parts = vec![name.to_string()];
        for r in dsts { parts.push(format!("r{}", r)); }
        for r in srcs { parts.push(format!("r{}", r)); }
        base_variants.entry(name).or_default().insert(parts.join("."));
    }
    let mut base_sorted: Vec<_> = base_variants.iter().map(|(name, vars)| {
        let total: u64 = vars.iter().map(|v| freq.get(v).copied().unwrap_or(0)).sum();
        (*name, vars.len(), total)
    }).collect();
    base_sorted.sort_by(|a, b| b.2.cmp(&a.2));

    println!("{:<25} {:>8} {:>12} {:>8}", "Base Opcode", "Variants", "Dyn Count", "% trace");
    println!("{}", "-".repeat(55));
    for (name, variants, total) in &base_sorted {
        let pct = *total as f64 / trace_len as f64 * 100.0;
        println!("{:<25} {:>8} {:>12} {:>7.1}%", name, variants, total, pct);
    }
    println!("{}", "-".repeat(55));
    println!("{:<25} {:>8} {:>12}", "TOTAL", total_unique, trace_len);

    Ok(())
}
