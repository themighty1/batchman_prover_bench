//! Dynamic tracing of WASM execution
//!
//! Runs the pure WASM JSON parser and traces executed instructions.

use anyhow::{Context, Result, anyhow};
use std::collections::HashMap;
use std::env;
use std::fs;
use wasmi::{Engine, Linker, Module, Store, Memory, MemoryType};

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

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    let wasm_path = args.get(1).map(|s| s.as_str())
        .unwrap_or("../pure-wasm/target/wasm32-unknown-unknown/release/pure_json_wasm.wasm");
    let json_size: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let num_regs: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);

    println!("=== Dynamic WASM Trace ===");
    println!("WASM: {}", wasm_path);
    println!("JSON size: ~{} bytes", json_size);
    println!("Registers: {}\n", num_regs);

    // Generate test data
    let test_json = generate_test_json(json_size);
    println!("Generated JSON: {} bytes", test_json.len());

    // Load WASM
    let wasm_bytes = fs::read(wasm_path).context("Failed to read WASM file")?;

    // Setup wasmi engine with fuel metering
    let mut config = wasmi::Config::default();
    config.consume_fuel(true);
    let engine = Engine::new(&config);

    let module = Module::new(&engine, &wasm_bytes[..])
        .context("Failed to parse WASM module")?;

    // Create store with fuel
    let mut store = Store::new(&engine, ());
    store.set_fuel(u64::MAX).unwrap();

    // Create linker with memory
    let mut linker = Linker::new(&engine);

    // WASM needs memory - provide it
    let memory_type = MemoryType::new(1, Some(256)).unwrap(); // 1-256 pages (64KB-16MB)
    let memory = Memory::new(&mut store, memory_type).unwrap();
    linker.define("env", "memory", memory.clone())?;

    // Instantiate
    let instance = linker.instantiate(&mut store, &module)
        .context("Failed to instantiate WASM")?
        .start(&mut store)
        .context("Failed to start WASM instance")?;

    // Get memory (might be exported or imported)
    let memory = instance.get_memory(&store, "memory")
        .unwrap_or(memory);

    // Get exported functions
    let parse_json = instance.get_func(&store, "parse_json")
        .ok_or_else(|| anyhow!("No parse_json function"))?;
    let parse_json_deep = instance.get_func(&store, "parse_json_deep")
        .ok_or_else(|| anyhow!("No parse_json_deep function"))?;

    // Write JSON to memory at offset 0
    let json_bytes = test_json.as_bytes();
    memory.write(&mut store, 0, json_bytes)
        .context("Failed to write JSON to memory")?;

    let json_ptr = 0i32;
    let json_len = json_bytes.len() as i32;

    // Get fuel before
    let fuel_before = store.get_fuel().unwrap();

    // Call parse_json(ptr, len) -> u64
    let result = parse_json.typed::<(i32, i32), u64>(&store)
        .context("Failed to type parse_json")?
        .call(&mut store, (json_ptr, json_len))
        .context("Failed to call parse_json")?;

    let fuel_after_simple = store.get_fuel().unwrap();
    let fuel_simple = fuel_before - fuel_after_simple;

    // Decode result
    let objects = (result >> 48) as u32;
    let arrays = ((result >> 32) & 0xFFFF) as u32;
    let strings = ((result >> 16) & 0xFFFF) as u32;
    let numbers = (result & 0xFFFF) as u32;

    println!("\nparse_json results:");
    println!("  Objects: {}, Arrays: {}, Strings: {}, Numbers: {}", objects, arrays, strings, numbers);

    // Re-write JSON and call parse_json_deep
    memory.write(&mut store, 0, json_bytes).unwrap();

    let fuel_before_deep = store.get_fuel().unwrap();

    let nodes = parse_json_deep.typed::<(i32, i32), u32>(&store)
        .context("Failed to type parse_json_deep")?
        .call(&mut store, (json_ptr, json_len))
        .context("Failed to call parse_json_deep")?;

    let fuel_after_deep = store.get_fuel().unwrap();
    let fuel_deep = fuel_before_deep - fuel_after_deep;

    println!("\n=== Execution Results ===");
    println!("parse_json:      {:>10} WASM instructions", fuel_simple);
    println!("parse_json_deep: {:>10} WASM instructions", fuel_deep);
    println!("Nodes parsed:    {:>10}", nodes);
    println!("Inst/byte:       {:>10.1}", fuel_deep as f64 / json_bytes.len() as f64);

    // Now do static analysis with register allocation
    println!("\n=== Static Analysis with {} Registers ===", num_regs);

    let func_ops = analyze_functions(&wasm_bytes, num_regs)?;

    // Calculate scale factor: dynamic instructions / static instructions
    let total_static: u32 = func_ops.get("__TOTAL__")
        .map(|m| m.iter()
            .filter(|(k, _)| *k != "local")
            .map(|(_, v)| v)
            .sum())
        .unwrap_or(1);

    let scale = fuel_deep as f64 / total_static as f64;

    println!("\nDynamic instruction estimate (scaled by {:.2}x):", scale);
    print_weighted_results(&func_ops, num_regs, scale);

    // Compare register counts
    println!("\n=== Spill Overhead by Register Count ===");
    println!("{:<10} {:>12} {:>12} {:>12} {:>10}", "Registers", "Spill Ops", "Base Ops", "Total Ops", "Overhead%");
    println!("{}", "-".repeat(60));

    for regs in [8, 12, 16, 20, 24, 32, 48, 64] {
        let ops = analyze_functions(&wasm_bytes, regs)?;
        let total_map = ops.get("__TOTAL__").unwrap();

        let spill_loads = *total_map.get("spill_load").unwrap_or(&0);
        let spill_stores = *total_map.get("spill_store").unwrap_or(&0);
        let spill_total = spill_loads + spill_stores;

        let base_ops: u32 = total_map.iter()
            .filter(|(k, _)| *k != "spill_load" && *k != "spill_store" && *k != "local")
            .map(|(_, v)| v)
            .sum();

        let total_ops = base_ops + spill_total;
        let overhead = if base_ops > 0 {
            (spill_total as f64 / base_ops as f64) * 100.0
        } else {
            0.0
        };

        // Scale to dynamic
        let dyn_spill = (spill_total as f64 * scale) as u64;
        let dyn_base = (base_ops as f64 * scale) as u64;
        let dyn_total = (total_ops as f64 * scale) as u64;

        println!("{:<10} {:>12} {:>12} {:>12} {:>9.1}%", regs, dyn_spill, dyn_base, dyn_total, overhead);
    }

    Ok(())
}

/// Analyze all functions and return op counts
fn analyze_functions(wasm_bytes: &[u8], num_regs: u32) -> Result<HashMap<String, HashMap<String, u32>>> {
    use wasmparser::{Parser, Payload};
    use reg_analyzer::regvm::WasmToVReg;
    use reg_analyzer::regalloc::{compute_live_intervals, linear_scan_alloc, count_spill_ops};

    let mut func_types: Vec<wasmparser::FuncType> = Vec::new();
    let mut type_indices: Vec<u32> = Vec::new();
    let mut func_names: HashMap<u32, String> = HashMap::new();
    let mut results: HashMap<String, HashMap<String, u32>> = HashMap::new();
    let mut func_count = 0u32;
    let mut total_ops: HashMap<String, u32> = HashMap::new();

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
                for func in reader.clone() {
                    type_indices.push(func?);
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

                let mut op_counts: HashMap<String, u32> = HashMap::new();
                for inst in &converter.instructions {
                    let cat = classify_op(inst);
                    *op_counts.entry(cat.to_string()).or_insert(0) += 1;
                    *total_ops.entry(cat.to_string()).or_insert(0) += 1;
                }

                // Add spill counts
                let intervals = compute_live_intervals(&converter.instructions);
                let alloc = linear_scan_alloc(&intervals, num_regs);
                let (loads, stores) = count_spill_ops(&converter.instructions, &alloc);
                *op_counts.entry("spill_load".to_string()).or_insert(0) += loads;
                *op_counts.entry("spill_store".to_string()).or_insert(0) += stores;
                *total_ops.entry("spill_load".to_string()).or_insert(0) += loads;
                *total_ops.entry("spill_store".to_string()).or_insert(0) += stores;

                let func_name = func_names.get(&func_count)
                    .cloned()
                    .unwrap_or_else(|| format!("func_{}", func_count));
                results.insert(func_name, op_counts);

                func_count += 1;
            }
            _ => {}
        }
    }

    results.insert("__TOTAL__".to_string(), total_ops);
    Ok(results)
}

fn classify_op(inst: &reg_analyzer::regvm::VRegInst) -> &'static str {
    use reg_analyzer::regvm::VRegInst::*;
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
        LocalGet { .. } | LocalSet { .. } | LocalTee { .. } => "local",
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

fn print_weighted_results(func_ops: &HashMap<String, HashMap<String, u32>>, _num_regs: u32, scale: f64) {
    let total_ops = func_ops.get("__TOTAL__").unwrap();

    let mut sorted: Vec<_> = total_ops.iter()
        .filter(|(k, _)| *k != "local")
        .collect();
    sorted.sort_by_key(|(_, count)| std::cmp::Reverse(**count));

    let total: u32 = sorted.iter().map(|(_, c)| **c).sum();
    let scaled_total = (total as f64 * scale) as u64;

    println!("\n{:<20} {:>12} {:>12} {:>10}", "Op Type", "Static", "Dynamic*", "%");
    println!("{}", "-".repeat(56));

    for (op, count) in &sorted {
        let scaled = (**count as f64 * scale) as u64;
        let pct = (**count as f64 / total as f64) * 100.0;
        println!("{:<20} {:>12} {:>12} {:>9.1}%", op, count, scaled, pct);
    }

    println!("{}", "-".repeat(56));
    println!("{:<20} {:>12} {:>12}", "TOTAL", total, scaled_total);
    println!("\n* Dynamic estimates based on fuel consumption");

    let local_count = total_ops.get("local").unwrap_or(&0);
    println!("(Eliminated {} local.get/set/tee ops)", local_count);
}
