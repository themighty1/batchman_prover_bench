//! Dynamic instruction trace of the PReg VM
//!
//! Actually executes the JSON parser on the fixture input and counts
//! every instruction that fires at runtime (including across function calls).

use anyhow::{Context, Result, anyhow};
use std::collections::HashMap;
use std::fs;
use wasmparser::{Parser, Payload};
use reg_analyzer::regvm::{WasmToVReg, VRegInst, RegInst, FuncSig};
use reg_analyzer::regalloc::{compute_live_intervals, linear_scan_alloc, RegAllocResult};
use reg_analyzer::preg_vm::{lower_to_preg, PRegVM};
use reg_analyzer::interpreter::Value;

fn main() -> Result<()> {
    let num_regs = 8u32;

    let json_data = include_bytes!("../../../guest-programs/json-query/fixtures/test_input.json");

    let wasm_path = "../json-crate-wasm/target/wasm32-unknown-unknown/release/json_crate_wasm.wasm";
    let wasm_bytes = fs::read(wasm_path).context("Failed to read WASM file")?;

    // Parse WASM module
    let mut func_types: Vec<wasmparser::FuncType> = Vec::new();
    let mut type_indices: Vec<u32> = Vec::new();
    let mut func_names: HashMap<String, u32> = HashMap::new();
    let mut code_bodies: Vec<wasmparser::FunctionBody> = Vec::new();
    let mut global_inits: Vec<i32> = Vec::new();
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
        vm.add_vreg_function(
            vreg_insts.clone(),
            RegAllocResult {
                vreg_to_preg: alloc.vreg_to_preg.clone(),
                spilled: alloc.spilled.clone(),
                spill_slots: alloc.spill_slots.clone(),
                num_spill_slots: alloc.num_spill_slots,
            },
            *num_params,
            *num_locals,
        );
    }

    // Initialize globals
    for (i, val) in global_inits.iter().enumerate() {
        if i < vm.globals.len() {
            vm.globals[i] = Value::I32(*val);
        }
    }

    // Copy WASM data sections into memory
    for (offset, data) in &data_segments {
        vm.write_memory(*offset as usize, data);
    }

    // Set up memory
    let json_ptr = 0u32;
    vm.write_memory(json_ptr as usize, json_data);
    vm.globals[0] = Value::I32(global_inits.get(0).copied().unwrap_or(1048576));

    // Find parse_json_deep(ptr, len) -> u32
    let func_idx = *func_names.get("parse_json_deep")
        .ok_or_else(|| anyhow!("No parse_json_deep function"))?;

    let func = &vreg_funcs[func_idx as usize];
    let total_locals = (func.2 + func.3) as usize;
    vm.locals = vec![Value::I32(0); total_locals];
    vm.locals[0] = Value::I32(json_ptr as i32);
    vm.locals[1] = Value::I32(json_data.len() as i32);

    // Enable tracing and execute
    vm.enable_tracing();

    let result = vm.execute_vreg(&func.0, &func.1);

    let nodes = result.map(|v| v.as_i32() as u32).unwrap_or(0);
    let correct = nodes > 0;

    // Print results
    let counts = vm.trace_counts.take().unwrap_or_default();
    let grand_total: u64 = counts.values().sum();

    let mut sorted: Vec<_> = counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));

    // Cross-check with wasmi fuel
    let wasmi_fuel = {
        use wasmi::{Engine, Linker, Module, Store, Memory, MemoryType};
        let mut config = wasmi::Config::default();
        config.consume_fuel(true);
        let engine = Engine::new(&config);
        let module = Module::new(&engine, &wasm_bytes[..]).unwrap();
        let mut store = Store::new(&engine, ());
        store.set_fuel(u64::MAX).unwrap();
        let mut linker = Linker::new(&engine);
        let memory_type = MemoryType::new(1, Some(256)).unwrap();
        let memory = Memory::new(&mut store, memory_type).unwrap();
        linker.define("env", "memory", memory.clone()).unwrap();
        let instance = linker.instantiate(&mut store, &module).unwrap().start(&mut store).unwrap();
        let memory = instance.get_memory(&store, "memory").unwrap_or(memory);

        memory.write(&mut store, 0, json_data).unwrap();

        let fuel_before = store.get_fuel().unwrap();
        let deep_fn = instance.get_func(&store, "parse_json_deep").unwrap();
        let _r = deep_fn.typed::<(i32, i32), u32>(&store).unwrap()
            .call(&mut store, (0i32, json_data.len() as i32)).unwrap();
        fuel_before - store.get_fuel().unwrap()
    };

    println!("=== Dynamic PReg Instruction Trace ({} physical registers) ===\n", num_regs);
    println!("Function: parse_json_deep");
    println!("Input:    fixtures/test_input.json ({} bytes)", json_data.len());
    println!("Nodes:    {}", nodes);
    println!("Correct:  {}\n", if correct { "PASS" } else { "FAIL" });
    println!("Total dynamic instructions: {}", grand_total);
    println!("wasmi fuel (reference):     {}", wasmi_fuel);
    println!("Ratio (fuel/traced):        {:.2}x\n", wasmi_fuel as f64 / grand_total as f64);

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

    let used = sorted.iter().filter(|(_, c)| **c > 0).count();
    println!("\nUnique instructions executed: {}", used);
    println!("Instructions per JSON byte: {:.1}", grand_total as f64 / json_data.len() as f64);

    Ok(())
}
