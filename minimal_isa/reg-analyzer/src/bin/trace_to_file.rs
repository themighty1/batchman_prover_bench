//! Run parse_json_deep on ~2KB generated JSON and write full instruction
//! trace to a file (one instruction per line) for offline analysis.
//!
//! Usage: trace_to_file [num_regs] [wasm_path] [json_size] [output_file] [--ml]
//!   --ml  Enable memory-lowered mode (locals/globals become memory loads/stores)

use anyhow::{Context, Result, anyhow};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use wasmparser::{Parser, Payload};
use reg_analyzer::regvm::{
    WasmToVReg, VRegInst, RegInst, FuncSig, SlotType,
    GLOBALS_MEM_BASE, FRAME_SP_ADDR, FRAME_STACK_BASE, SLOT_SIZE,
};
use reg_analyzer::regalloc::{compute_live_intervals, linear_scan_alloc, RegAllocResult};
use reg_analyzer::preg_vm::{lower_to_preg, PRegVM};
use reg_analyzer::interpreter::Value;

/// Generate test JSON of approximately the given size
fn generate_test_json(target_size: usize) -> String {
    let mut json = String::from(r#"{"data":{"users":["#);
    let user_template = r#"{"name":"user_XXX","email":"user_XXX@example.com","age":25,"active":true,"tags":["a","b","c"]}"#;

    let mut i = 0;
    while json.len() < target_size {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&user_template.replace("XXX", &format!("{:04}", i)));
        i += 1;
    }
    json.push_str(r#"]},"meta":{"count":"#);
    json.push_str(&i.to_string());
    json.push_str(r#"}}"#);
    json
}

/// Convert a wasmparser ValType to our SlotType
fn val_type_to_slot(vt: &wasmparser::ValType) -> SlotType {
    match vt {
        wasmparser::ValType::I64 => SlotType::I64,
        _ => SlotType::I32, // i32, f32, f64, etc. all fit in i32 for our purposes
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let memory_lowered = args.iter().any(|a| a == "--ml");
    // Filter out --ml from positional args
    let positional: Vec<&str> = args[1..].iter()
        .filter(|a| *a != "--ml")
        .map(|s| s.as_str())
        .collect();

    let num_regs: u32 = positional.first().and_then(|s| s.parse().ok()).unwrap_or(8);
    let wasm_path = positional.get(1).copied().unwrap_or(
        "../json-crate-wasm/target/wasm32-unknown-unknown/release/json_crate_wasm.wasm");
    let json_size: usize = positional.get(2).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let output_file = positional.get(3).copied().unwrap_or("trace.log");

    let test_json = generate_test_json(json_size);
    let json_data = test_json.as_bytes();

    let wasm_bytes = fs::read(wasm_path).context("Failed to read WASM file")?;

    // Parse WASM module
    let mut func_types: Vec<wasmparser::FuncType> = Vec::new();
    let mut type_indices: Vec<u32> = Vec::new();
    let mut func_names: HashMap<String, u32> = HashMap::new();
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
                        func_names.insert(export.name.to_string(), export.index);
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

    // Convert all functions
    let mut max_spill_slots = 0u32;
    let mut vreg_funcs: Vec<(Vec<VRegInst>, RegAllocResult, u32, u32)> = Vec::new();
    let mut preg_funcs: Vec<(Vec<RegInst>, u32, u32)> = Vec::new();

    for (func_count, body) in code_bodies.iter().enumerate() {
        let type_idx = type_indices.get(func_count).copied().unwrap_or(0);
        let func_type = func_types.get(type_idx as usize);
        let num_params = func_type.map(|ft| ft.params().len() as u32).unwrap_or(0);
        let mut num_locals = 0u32;

        // Collect local types: params from FuncType, then declared locals from body
        let mut local_types: Vec<SlotType> = Vec::new();
        if let Some(ft) = func_type {
            for p in ft.params() {
                local_types.push(val_type_to_slot(p));
            }
        }
        for local in body.get_locals_reader()? {
            let (count, vt) = local?;
            num_locals += count;
            let st = val_type_to_slot(&vt);
            for _ in 0..count {
                local_types.push(st);
            }
        }

        let mut converter = if memory_lowered {
            WasmToVReg::new_memory_lowered(
                num_params, num_locals, func_sigs.clone(),
                local_types, global_types.clone(),
            )
        } else {
            WasmToVReg::new_with_sigs(num_params, num_locals, func_sigs.clone())
        };
        for op in body.get_operators_reader()? {
            converter.convert_op(&op?);
        }

        let intervals = compute_live_intervals(&converter.instructions);
        let alloc = linear_scan_alloc(&intervals, num_regs);
        let preg_insts = lower_to_preg(&converter.instructions, &alloc);

        max_spill_slots = max_spill_slots.max(alloc.num_spill_slots);
        vreg_funcs.push((converter.instructions, alloc, num_params, num_locals));
        preg_funcs.push((preg_insts, num_params, num_locals));
    }

    // Set up VM
    let mut vm = PRegVM::new(num_regs as usize, max_spill_slots as usize + 64, 256);

    for (preg_insts, num_params, num_locals) in &preg_funcs {
        vm.add_function(preg_insts.clone(), *num_params, *num_locals);
    }
    for (vreg_insts, alloc, num_params, num_locals) in &vreg_funcs {
        let alloc_clone = RegAllocResult {
            vreg_to_preg: alloc.vreg_to_preg.clone(),
            spilled: alloc.spilled.clone(),
            spill_slots: alloc.spill_slots.clone(),
            num_spill_slots: alloc.num_spill_slots,
        };
        if memory_lowered {
            vm.add_vreg_function_ml(vreg_insts.clone(), alloc_clone, *num_params, *num_locals);
        } else {
            vm.add_vreg_function(vreg_insts.clone(), alloc_clone, *num_params, *num_locals);
        }
    }

    // Initialize globals (into vm.globals for non-lowered, into memory for lowered)
    for (i, val) in global_inits.iter().enumerate() {
        if i < vm.globals.len() {
            vm.globals[i] = Value::I32(*val);
        }
    }

    // Copy WASM data sections into memory (e.g. .rodata for allocator state, string constants)
    for (offset, data) in &data_segments {
        eprintln!("Copying data segment: {} bytes at offset 0x{:x}", data.len(), offset);
        vm.write_memory(*offset as usize, data);
    }

    // Set up memory - write JSON at offset 0 (in unused stack area, below __stack_pointer)
    vm.write_memory(0, json_data);
    vm.globals[0] = Value::I32(global_inits.get(0).copied().unwrap_or(1048576));

    // Find parse_json_deep
    let func_idx = *func_names.get("parse_json_deep")
        .ok_or_else(|| anyhow!("No parse_json_deep function"))?;

    if memory_lowered {
        // Memory-lowered mode: initialize frame stack and write globals + params to memory
        eprintln!("Memory-lowered mode enabled");

        // Initialize FRAME_SP_ADDR → FRAME_STACK_BASE
        vm.write_memory(FRAME_SP_ADDR as usize, &FRAME_STACK_BASE.to_le_bytes());

        // Write globals to GLOBALS_MEM_BASE
        for (i, val) in global_inits.iter().enumerate() {
            let addr = GLOBALS_MEM_BASE as usize + (i as u32 * SLOT_SIZE) as usize;
            vm.write_memory(addr, &(*val as u32).to_le_bytes());
        }

        // Write initial params (ptr=0, len=json_data.len()) to frame
        let frame_base = FRAME_STACK_BASE as usize;
        vm.write_memory(frame_base, &0u32.to_le_bytes());  // ptr = 0
        vm.write_memory(frame_base + SLOT_SIZE as usize, &(json_data.len() as u32).to_le_bytes());  // len
    } else {
        // Original mode: set up locals directly
        let func = &vreg_funcs[func_idx as usize];
        let total_locals = (func.2 + func.3) as usize;
        vm.locals = vec![Value::I32(0); total_locals];
        vm.locals[0] = Value::I32(0);  // ptr
        vm.locals[1] = Value::I32(json_data.len() as i32);  // len
    }

    // Enable both counting and logging
    vm.enable_tracing();
    vm.enable_trace_log();

    eprintln!("Executing parse_json_deep on {} bytes of JSON...", json_data.len());

    let func = &vreg_funcs[func_idx as usize];
    let result = vm.execute_vreg(&func.0, &func.1);

    let nodes = result.map(|v| v.as_i32() as u32).unwrap_or(0);

    // Get trace data
    let counts = vm.trace_counts.take().unwrap_or_default();
    let trace_log = vm.trace_log.take().unwrap_or_default();
    let grand_total = trace_log.len() as u64;

    // Write trace to file
    {
        let mut f = fs::File::create(output_file).context("Failed to create trace file")?;
        for inst in &trace_log {
            writeln!(f, "{}", inst)?;
        }
    }

    // Print summary
    let mut sorted: Vec<_> = counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));

    let mode_str = if memory_lowered { " [memory-lowered]" } else { "" };
    println!("=== Dynamic Trace: parse_json_deep ({} regs){} ===\n", num_regs, mode_str);
    println!("WASM:       {}", wasm_path);
    println!("JSON size:  {} bytes", json_data.len());
    println!("Nodes:      {}", nodes);
    println!("Trace file: {}", output_file);
    println!("Total instructions: {}\n", grand_total);

    println!("{:<20} {:>12} {:>10} {:>10}", "Instruction", "Count", "%", "Cumul%");
    println!("{}", "=".repeat(54));

    let mut cumulative = 0.0f64;
    for (name, count) in &sorted {
        let pct = (**count as f64 / grand_total as f64) * 100.0;
        cumulative += pct;
        println!("{:<20} {:>12} {:>9.2}% {:>9.2}%", name, count, pct, cumulative);
    }
    println!("{}", "=".repeat(54));
    println!("{:<20} {:>12} {:>9}  {:>9}", "TOTAL", grand_total, "100%", "");
    println!("\nInst/byte: {:.1}", grand_total as f64 / json_data.len() as f64);

    Ok(())
}
