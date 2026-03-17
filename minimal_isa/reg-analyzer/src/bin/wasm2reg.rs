//! Convert WASM to register-based IR with N registers

use anyhow::{Context, Result};
use std::env;
use std::fs;
use wasmparser::{Parser, Payload};

// Import from library
use reg_analyzer::regvm::WasmToVReg;
use reg_analyzer::regalloc::{compute_live_intervals, linear_scan_alloc, rewrite_with_allocation, count_spill_ops};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    let wasm_path = args.get(1).map(|s| s.as_str()).unwrap_or("../target/wasm32-unknown-unknown/release/json_wasm_bench.wasm");
    let num_regs: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let verbose = args.iter().any(|a| a == "-v" || a == "--verbose");

    let wasm_bytes = fs::read(wasm_path).context("Failed to read WASM file")?;

    println!("Converting {} with {} registers", wasm_path, num_regs);
    println!("File size: {} bytes\n", wasm_bytes.len());

    let mut func_types: Vec<wasmparser::FuncType> = Vec::new();
    let mut type_indices: Vec<u32> = Vec::new();

    let mut total_wasm_ops = 0u32;
    let mut total_vreg_ops = 0u32;
    let mut total_spill_loads = 0u32;
    let mut total_spill_stores = 0u32;
    let mut total_vregs = 0u32;
    let mut total_spilled_vregs = 0u32;
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
                let func_idx = func_count;
                let type_idx = type_indices.get(func_idx as usize).copied().unwrap_or(0);
                let func_type = func_types.get(type_idx as usize);

                let num_params = func_type.map(|ft| ft.params().len() as u32).unwrap_or(0);

                // Count locals
                let mut num_locals = 0u32;
                for local in body.get_locals_reader()? {
                    let (count, _) = local?;
                    num_locals += count;
                }

                // Convert WASM to virtual register IR
                let mut converter = WasmToVReg::new(num_params, num_locals);
                let mut wasm_op_count = 0u32;

                for op in body.get_operators_reader()? {
                    let op = op?;
                    converter.convert_op(&op);
                    wasm_op_count += 1;
                }

                total_wasm_ops += wasm_op_count;
                total_vreg_ops += converter.instructions.len() as u32;
                total_vregs += converter.num_vregs();

                // Allocate registers
                let intervals = compute_live_intervals(&converter.instructions);
                let alloc = linear_scan_alloc(&intervals, num_regs);
                let (loads, stores) = count_spill_ops(&converter.instructions, &alloc);

                total_spill_loads += loads;
                total_spill_stores += stores;
                total_spilled_vregs += alloc.spilled.len() as u32;

                if verbose && !alloc.spilled.is_empty() {
                    println!("=== Function {} ===", func_idx);
                    println!("  WASM ops: {}", wasm_op_count);
                    println!("  Virtual regs: {}", converter.num_vregs());
                    println!("  Spilled vregs: {}", alloc.spilled.len());
                    println!("  Spill loads: {}, stores: {}", loads, stores);

                    // Show first few instructions
                    let output = rewrite_with_allocation(&converter.instructions, &alloc, num_regs);
                    println!("  First 20 instructions:");
                    for (i, line) in output.iter().take(20).enumerate() {
                        println!("    {:4}: {}", i, line);
                    }
                    println!();
                }

                func_count += 1;
            }
            _ => {}
        }
    }

    // Summary
    println!("=== Summary ({} registers) ===", num_regs);
    println!("Functions: {}", func_count);
    println!("Original WASM ops: {}", total_wasm_ops);
    println!("Register IR ops: {}", total_vreg_ops);
    println!("Total virtual registers: {}", total_vregs);
    println!("Spilled virtual registers: {}", total_spilled_vregs);
    println!();
    println!("Spill operations:");
    println!("  Loads (reload from stack): {}", total_spill_loads);
    println!("  Stores (spill to stack): {}", total_spill_stores);
    println!("  Total spill ops: {}", total_spill_loads + total_spill_stores);
    println!();

    let final_ops = total_vreg_ops + total_spill_loads + total_spill_stores;
    println!("Final instruction count: {} (IR) + {} (spills) = {}",
             total_vreg_ops, total_spill_loads + total_spill_stores, final_ops);
    println!("Overhead from spills: {:.1}%",
             (total_spill_loads + total_spill_stores) as f64 / total_vreg_ops as f64 * 100.0);

    Ok(())
}
