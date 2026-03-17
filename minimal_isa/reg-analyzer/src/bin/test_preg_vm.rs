//! Test physical register VM against wasmi reference implementation
//!
//! This test uses actual register allocation with a fixed number of registers
//! and spill slots, unlike test_regvm which uses unlimited virtual registers.

use anyhow::{Context, Result, anyhow};
use std::collections::HashMap;
use std::fs;
use wasmi::{Engine, Linker, Module, Store, Memory, MemoryType};
use wasmparser::{Parser, Payload};
use reg_analyzer::regvm::{WasmToVReg, VRegInst, RegInst, FuncSig};
use reg_analyzer::regalloc::{compute_live_intervals, linear_scan_alloc, RegAllocResult};
use reg_analyzer::preg_vm::{lower_to_preg, PRegVM};
use reg_analyzer::interpreter::Value;

/// Converted function with both VReg and PReg IR
struct ConvertedFunc {
    vreg_instructions: Vec<VRegInst>,
    preg_instructions: Vec<RegInst>,
    alloc: RegAllocResult,
    num_params: u32,
    num_locals: u32,
    num_spill_slots: u32,
}

fn load_and_convert(wasm_bytes: &[u8], num_regs: u32) -> Result<(Vec<ConvertedFunc>, HashMap<String, u32>, Vec<i32>)> {
    let mut func_types: Vec<wasmparser::FuncType> = Vec::new();
    let mut type_indices: Vec<u32> = Vec::new();
    let mut func_names: HashMap<String, u32> = HashMap::new();
    let mut code_bodies: Vec<wasmparser::FunctionBody> = Vec::new();
    let mut global_inits: Vec<i32> = Vec::new();

    // First pass: collect types, function indices, exports, globals, and code bodies
    for payload in Parser::new(0).parse_all(wasm_bytes) {
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

    // Convert each function
    let mut functions = Vec::new();
    for (func_count, body) in code_bodies.iter().enumerate() {
        let type_idx = type_indices.get(func_count).copied().unwrap_or(0);
        let func_type = func_types.get(type_idx as usize);
        let num_params = func_type.map(|ft| ft.params().len() as u32).unwrap_or(0);
        let mut num_locals = 0u32;
        for local in body.get_locals_reader()? {
            let (count, _) = local?;
            num_locals += count;
        }

        // Convert to VReg IR
        let mut converter = WasmToVReg::new_with_sigs(num_params, num_locals, func_sigs.clone());
        for op in body.get_operators_reader()? {
            converter.convert_op(&op?);
        }
        let vreg_instructions = converter.instructions;

        // Do register allocation
        let intervals = compute_live_intervals(&vreg_instructions);
        let alloc = linear_scan_alloc(&intervals, num_regs);

        // Lower to PReg IR with spill/reload
        let preg_instructions = lower_to_preg(&vreg_instructions, &alloc);

        let num_spill_slots = alloc.num_spill_slots;
        functions.push(ConvertedFunc {
            vreg_instructions,
            preg_instructions,
            alloc,
            num_params,
            num_locals,
            num_spill_slots,
        });
    }

    Ok((functions, func_names, global_inits))
}

fn run_wasmi_query(wasm_bytes: &[u8], json_data: &[u8], query: &[u8]) -> Result<String> {
    let config = wasmi::Config::default();
    let engine = Engine::new(&config);
    let module = Module::new(&engine, wasm_bytes).context("Failed to parse WASM module")?;
    let mut store = Store::new(&engine, ());
    let mut linker = Linker::new(&engine);
    let memory_type = MemoryType::new(1, Some(256)).unwrap();
    let memory = Memory::new(&mut store, memory_type).unwrap();
    linker.define("env", "memory", memory.clone())?;
    let instance = linker.instantiate(&mut store, &module)?.start(&mut store)?;
    let memory = instance.get_memory(&store, "memory").unwrap_or(memory);

    let json_ptr = 0u32;
    let query_ptr = json_data.len() as u32;
    let output_ptr = query_ptr + query.len() as u32 + 100;

    memory.write(&mut store, json_ptr as usize, json_data)?;
    memory.write(&mut store, query_ptr as usize, query)?;

    let json_query_fn = instance.get_func(&store, "json_query")
        .ok_or_else(|| anyhow!("No json_query function"))?;

    let result_len = json_query_fn.typed::<(i32, i32, i32, i32, i32), i32>(&store)?
        .call(&mut store, (json_ptr as i32, json_data.len() as i32, query_ptr as i32, query.len() as i32, output_ptr as i32))?;

    if result_len == 0 {
        return Ok("(not found)".to_string());
    }

    let mut output = vec![0u8; result_len as usize];
    memory.read(&store, output_ptr as usize, &mut output)?;
    Ok(String::from_utf8_lossy(&output).to_string())
}

fn main() -> Result<()> {
    let num_regs = 8u32;

    let json_data = include_bytes!("../../../guest-programs/json-query/fixtures/test_input.json");
    let query = include_bytes!("../../../guest-programs/json-query/fixtures/query.txt");
    let query_trimmed: Vec<u8> = query.iter().copied().take_while(|&c| c != b'\n' && c != b'\r').collect();
    let expected_result = "Bob Smith";

    println!("=== Physical Register VM Test ({} registers) ===\n", num_regs);
    println!("JSON fixture: {} bytes", json_data.len());
    println!("Query: {}", String::from_utf8_lossy(&query_trimmed));
    println!("Expected: \"{}\"\n", expected_result);

    let wasm_path = "../pure-wasm/target/wasm32-unknown-unknown/release/pure_json_wasm.wasm";
    let wasm_bytes = fs::read(wasm_path).context("Failed to read WASM file")?;
    println!("WASM size: {} bytes\n", wasm_bytes.len());

    // Test with wasmi
    println!("--- wasmi (reference) ---");
    let wasmi_result = run_wasmi_query(&wasm_bytes, json_data, &query_trimmed)?;
    println!("Result: \"{}\"", wasmi_result);
    let wasmi_pass = wasmi_result == expected_result;
    println!("Check: {}\n", if wasmi_pass { "PASS" } else { "FAIL" });

    // Convert and allocate registers
    println!("--- Physical Register VM ({} regs) ---", num_regs);
    let (functions, func_names, global_inits) = load_and_convert(&wasm_bytes, num_regs)?;

    println!("Converted {} functions", functions.len());

    let mut total_vreg_insts = 0;
    let mut total_preg_insts = 0;
    let mut total_spill_slots = 0u32;
    for func in functions.iter() {
        total_vreg_insts += func.vreg_instructions.len();
        total_preg_insts += func.preg_instructions.len();
        total_spill_slots = total_spill_slots.max(func.num_spill_slots);
    }
    println!("VReg instructions: {}", total_vreg_insts);
    println!("PReg instructions: {} (with spill/reload)", total_preg_insts);
    println!("Max spill slots needed: {}", total_spill_slots);

    // Set up the physical register VM
    let mut vm = PRegVM::new(num_regs as usize, total_spill_slots as usize + 64, 256);

    // Register all functions with the VM for nested calls
    for func in functions.iter() {
        vm.add_function(
            func.preg_instructions.clone(),
            func.num_params,
            func.num_locals,
        );
        // Also register VReg functions for execute_vreg
        vm.add_vreg_function(
            func.vreg_instructions.clone(),
            reg_analyzer::regalloc::RegAllocResult {
                vreg_to_preg: func.alloc.vreg_to_preg.clone(),
                spilled: func.alloc.spilled.clone(),
                spill_slots: func.alloc.spill_slots.clone(),
                num_spill_slots: func.alloc.num_spill_slots,
            },
            func.num_params,
            func.num_locals,
        );
    }
    println!("Registered {} functions with VM", vm.functions.len());

    // Initialize globals
    for (i, val) in global_inits.iter().enumerate() {
        if i < vm.globals.len() {
            vm.globals[i] = Value::I32(*val);
        }
    }

    // Set up memory layout (same as wasmi reference)
    let json_ptr = 0u32;
    let query_ptr = json_data.len() as u32;
    let output_ptr = query_ptr + query_trimmed.len() as u32 + 100;

    vm.write_memory(json_ptr as usize, json_data);
    vm.write_memory(query_ptr as usize, &query_trimmed);

    // Find and execute json_query function
    if let Some(&func_idx) = func_names.get("json_query") {
        let func = &functions[func_idx as usize];

        println!("\njson_query function:");
        println!("  VReg instructions: {}", func.vreg_instructions.len());
        println!("  PReg instructions: {}", func.preg_instructions.len());
        println!("  Spill slots: {}", func.num_spill_slots);

        // Check for CallIndirect
        let has_call_indirect = func.vreg_instructions.iter().any(|inst| {
            matches!(inst, reg_analyzer::regvm::VRegInst::CallIndirect { .. })
        });
        println!("  Uses CallIndirect: {}", has_call_indirect);

        // Check for Return instructions
        let return_count = func.vreg_instructions.iter().filter(|inst| {
            matches!(inst, reg_analyzer::regvm::VRegInst::Return { .. })
        }).count();
        println!("  Return instructions: {}", return_count);

        // Set up arguments in registers (calling convention: r3, r4, r5, ...)
        // But we also need them in locals since the function uses LocalGet
        let total_locals = (func.num_params + func.num_locals) as usize;
        vm.locals = vec![Value::I32(0); total_locals];
        vm.locals[0] = Value::I32(json_ptr as i32);
        vm.locals[1] = Value::I32(json_data.len() as i32);
        vm.locals[2] = Value::I32(query_ptr as i32);
        vm.locals[3] = Value::I32(query_trimmed.len() as i32);
        vm.locals[4] = Value::I32(output_ptr as i32);

        // Initialize global 0 (stack pointer)
        vm.globals[0] = Value::I32(global_inits.get(0).copied().unwrap_or(1048576));

        // Debug: show allocation info
        println!("\n  Allocation info:");
        println!("    vregs with pregs: {}", func.alloc.vreg_to_preg.len());
        println!("    vregs with spill slots: {}", func.alloc.spill_slots.len());

        // Execute the VReg instructions with register allocation
        println!("\nExecuting with {} physical registers...", num_regs);

        let result = vm.execute_vreg(&func.vreg_instructions, &func.alloc);

        match result {
            Some(val) => {
                let result_len = val.as_i32() as usize;
                println!("  Return value: {} (0x{:x})", result_len, result_len);
                if result_len == 0 {
                    println!("Result: (not found)");
                    println!("Check: FAIL\n");
                } else {
                    let output = vm.read_memory(output_ptr as usize, result_len);
                    let preg_result = String::from_utf8_lossy(output).to_string();
                    println!("Result: \"{}\"", preg_result);
                    let preg_pass = preg_result == expected_result;
                    println!("Check: {}\n", if preg_pass { "PASS" } else { "FAIL" });

                    println!("=== Final Comparison ===");
                    if wasmi_result == preg_result && preg_result == expected_result {
                        println!("PASS: Both wasmi and PRegVM return correct result \"{}\"", expected_result);
                    } else {
                        println!("FAIL: Results differ");
                        println!("  wasmi:  \"{}\"", wasmi_result);
                        println!("  pregvm: \"{}\"", preg_result);
                        println!("  expected: \"{}\"", expected_result);
                    }
                }
            }
            None => {
                println!("Result: execution failed (unreachable or error)");
                println!("Check: FAIL");
            }
        }
    } else {
        println!("json_query function not found");
    }

    Ok(())
}
