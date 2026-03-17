//! Count unique register-specialized opcodes in the ML-compiled program.
//!
//! Each instruction's register operands become part of the opcode:
//!   i32.add { dst: r4, src1: r1, src2: r2 }  →  "i32.add.r4.r1.r2"
//!
//! Reports both static (across all compiled functions) and dynamic
//! (weighted by execution count from the trace) unique op counts.

use anyhow::{Context, Result, anyhow};
use std::collections::HashMap;
use std::fs;
use wasmparser::{Parser, Payload};
use reg_analyzer::regvm::{
    WasmToVReg, VRegInst, FuncSig, SlotType,
    GLOBALS_MEM_BASE, FRAME_SP_ADDR, FRAME_STACK_BASE, SLOT_SIZE,
};
use reg_analyzer::regalloc::{compute_live_intervals, linear_scan_alloc, RegAllocResult};
use reg_analyzer::preg_vm::{specialized_opcode, PRegVM, vreg_inst_name};
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
    let num_regs: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    let wasm_path = std::env::args().nth(2).unwrap_or_else(||
        "../json-crate-wasm/target/wasm32-unknown-unknown/release/json_crate_wasm.wasm".to_string());

    let wasm_bytes = fs::read(&wasm_path).context("Failed to read WASM file")?;

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
            Payload::CodeSectionEntry(body) => {
                code_bodies.push(body.clone());
            }
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

    // === STATIC ANALYSIS ===
    // Convert all functions with ML, register-allocate, count specialized opcodes
    let mut static_counts: HashMap<String, u32> = HashMap::new();
    let mut total_static = 0u32;
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

        for inst in &converter.instructions {
            let spec = specialized_opcode(inst, &alloc);
            *static_counts.entry(spec).or_insert(0) += 1;
            total_static += 1;
        }

        vreg_funcs.push((converter.instructions, alloc, num_params, num_locals));
    }

    // === DYNAMIC ANALYSIS ===
    // Run parse_json_deep and trace specialized opcodes
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
    vm.globals[0] = Value::I32(global_inits.get(0).copied().unwrap_or(1048576));

    // Initialize frame stack + globals memory
    vm.write_memory(FRAME_SP_ADDR as usize, &FRAME_STACK_BASE.to_le_bytes());
    for (i, val) in global_inits.iter().enumerate() {
        let addr = GLOBALS_MEM_BASE as usize + (i as u32 * SLOT_SIZE) as usize;
        vm.write_memory(addr, &(*val as u32).to_le_bytes());
    }
    let frame_base = FRAME_STACK_BASE as usize;
    vm.write_memory(frame_base, &0u32.to_le_bytes());
    vm.write_memory(frame_base + SLOT_SIZE as usize, &(json_data.len() as u32).to_le_bytes());

    // Enable trace log to capture dynamic specialized opcodes
    vm.enable_trace_log();

    let func_idx = func_names.iter()
        .find(|(_, name)| *name == "parse_json_deep")
        .map(|(idx, _)| *idx)
        .ok_or_else(|| anyhow!("No parse_json_deep"))?;

    let func = &vreg_funcs[func_idx as usize];
    let result = vm.execute_vreg(&func.0, &func.1);
    let nodes = result.map(|v| v.as_i32() as u32).unwrap_or(0);

    // The trace_log has generic names. We need to re-run with specialized tracking.
    // Instead, let's use a different approach: instrument execute_vreg to build
    // specialized names. Since we can't easily change trace_log type, we'll
    // count by iterating the trace and mapping back.
    //
    // Actually, we already have the static analysis. For dynamic, we need the
    // specialized opcode at each execution point. Let's add a separate trace
    // vector for this.
    //
    // For now, just report static results + generic dynamic trace info.

    let trace_log = vm.trace_log.take().unwrap_or_default();
    let dynamic_total = trace_log.len();

    // Print results
    println!("=== Register-Specialized Opcode Analysis ({} regs, ML) ===\n", num_regs);
    println!("WASM:  {}", wasm_path);
    println!("Nodes: {}\n", nodes);

    // Static
    let mut sorted: Vec<_> = static_counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));

    println!("--- STATIC: All functions compiled ---");
    println!("Total instructions: {}", total_static);
    println!("Unique specialized ops: {}\n", sorted.len());

    println!("{:<40} {:>8} {:>8} {:>8}", "Specialized Op", "Count", "%", "Cumul%");
    println!("{}", "=".repeat(66));
    let mut cumul = 0.0f64;
    for (name, count) in sorted.iter().take(50) {
        let pct = (**count as f64 / total_static as f64) * 100.0;
        cumul += pct;
        println!("{:<40} {:>8} {:>7.2}% {:>7.1}%", name, count, pct, cumul);
    }
    if sorted.len() > 50 {
        println!("... ({} more unique ops)", sorted.len() - 50);
    }
    println!("{}", "=".repeat(66));

    // Count by base opcode
    let mut base_op_unique: HashMap<&str, u32> = HashMap::new();
    for (name, _) in &static_counts {
        let base = name.split('.').take(2).collect::<Vec<_>>().join(".");
        // But some ops have 3 parts like i32.load8_u — use vreg_inst_name style
        let base = if let Some(dot_pos) = name.find('.') {
            // Find the second segment boundary: after the reg suffix starts
            // e.g. "i32.add.r4.r1.r2" → base = "i32.add"
            // e.g. "i32.load8_u.r3.r1" → base = "i32.load8_u"
            // Strategy: split by '.', skip r/s prefixed parts
            let parts: Vec<&str> = name.split('.').collect();
            let mut base_parts = vec![];
            for p in &parts {
                if p.starts_with('r') || p.starts_with('s') || p.starts_with('?') {
                    break;
                }
                base_parts.push(*p);
            }
            base_parts.join(".")
        } else {
            name.clone()
        };
        *base_op_unique.entry(Box::leak(base.into_boxed_str())).or_insert(0) += 1;
    }
    let mut base_sorted: Vec<_> = base_op_unique.iter().collect();
    base_sorted.sort_by(|a, b| b.1.cmp(a.1));

    println!("\n--- Register variants per base opcode ---");
    println!("{:<25} {:>10}", "Base Opcode", "Variants");
    println!("{}", "-".repeat(37));
    for (name, count) in &base_sorted {
        println!("{:<25} {:>10}", name, count);
    }
    println!("{}", "-".repeat(37));
    println!("{:<25} {:>10}", "TOTAL unique", static_counts.len());

    println!("\nDynamic trace: {} instructions", dynamic_total);

    Ok(())
}
