//! Execute a compiled ISA-VM program from file with inputs.
//!
//! Usage:
//!   rv32_run [program.bin] [json_file] [path] [expected_string]
//!   rv32_run [program.bin] [int_input] [expected_u32]
//!
//! Defaults: program.bin, fixture.json, data.5.v, val5

use anyhow::Result;
use reg_analyzer::rv32_isa_vm::{Rv32FuncInfo, Rv32IsaVm, CompiledProgram, MAILBOX_BASE, IO_OUTPUT_LEN};
use std::collections::HashMap;
use std::fs;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let program_path = args.get(1)
        .cloned()
        .unwrap_or_else(|| "program.bin".to_string());

    // Load compiled program
    let encoded = fs::read(&program_path)?;
    let program: CompiledProgram = bincode::deserialize(&encoded)?;

    println!("=== rv32_run ({} regs) ===", program.num_regs);
    println!("  Program:   {} ({} bytes)", program_path, encoded.len());
    println!("  Functions: {}", program.functions.len());
    println!("  Entry:     0x{:x}", program.entry_addr);

    // Count instructions and ISA size
    let total_insts: usize = program.functions.iter().map(|f| f.rewritten.len()).sum();
    let mut all_opcodes = std::collections::HashSet::new();
    for f in &program.functions {
        for inst in &f.rewritten {
            all_opcodes.insert(inst.specialized.clone());
        }
    }
    println!("  Instructions: {}", total_insts);
    println!("  ISA size:  {} unique opcodes", all_opcodes.len());

    // Detect input mode from args
    let input_arg = args.get(2).cloned();
    let arg3 = args.get(3).cloned();
    let arg4 = args.get(4).cloned();

    // Heuristic: if we have 5 args (prog, json, path, expected), it's query mode
    // If we have 4 args (prog, int, expected), it's legacy int mode
    // If we have 3 args (prog, file, expected_u32), it's legacy file mode
    let query_mode = args.len() >= 5;
    let input_is_int = !query_mode && input_arg.as_ref().map_or(false, |s| s.parse::<u32>().is_ok());

    // Build function table
    let mut func_table: HashMap<u32, Rv32FuncInfo> = HashMap::new();
    for f in program.functions {
        let entry = f.rewritten.iter().find(|i| i.addr != 0).map(|i| i.addr).unwrap_or(0);
        func_table.insert(entry, f);
    }

    // Create VM and load segments
    let mut vm = Rv32IsaVm::new(program.num_regs as usize);
    for seg in &program.segments {
        vm.memory.write_bytes(seg.vaddr, &seg.data);
    }

    // Write inputs
    if query_mode {
        let json_path = input_arg.as_deref().unwrap_or("../guest-programs/json-query/fixtures/test_input.json");
        let path = arg3.as_deref().unwrap_or("data.5.v");
        let expected = arg4.as_deref().unwrap_or("val5");
        println!("  JSON:      {}", json_path);
        println!("  Path:      {}", path);
        println!("  Expected:  {:?}", expected);

        let fixture = fs::read(json_path)?;
        vm.memory.write_input(&fixture);
        vm.memory.write_path(path.as_bytes());
    } else if input_is_int {
        let n: u32 = input_arg.as_ref().unwrap().parse().unwrap();
        let expected: u32 = arg3.as_ref().and_then(|s| s.parse().ok()).unwrap_or(0);
        println!("  Input:     n={}", n);
        println!("  Expected:  {}", expected);
        vm.memory.write_input_u32(n);
    } else if let Some(ref file_path) = input_arg {
        let expected: u32 = arg3.as_ref().and_then(|s| s.parse().ok()).unwrap_or(0);
        println!("  Input:     {}", file_path);
        println!("  Expected:  {}", expected);
        let fixture = fs::read(file_path)?;
        vm.memory.write_input(&fixture);
    }

    // Initialize SP in mailbox
    vm.memory.write_u32(MAILBOX_BASE + 2 * 4, 0x7FFF_0000);
    vm.frame_reg = 0x8000_1000;

    println!();

    // Execute
    let entry_func = func_table.get(&program.entry_addr)
        .ok_or_else(|| anyhow::anyhow!("entry 0x{:x} not in function table", program.entry_addr))?;
    vm.execute_function(entry_func, &func_table, &program.addr_to_func, None)?;

    println!("  Steps:     {}", vm.steps);
    let pages = vm.memory.num_pages();
    println!("  Memory:    {} pages = {} KB", pages, pages * 4);

    // Op breakdown
    let mut ops: Vec<_> = vm.op_counts.iter().collect();
    ops.sort_by(|a, b| b.1.cmp(a.1));
    let mem_ops = ["lw", "lh", "lhu", "lb", "lbu", "sw", "sh", "sb", "conv_load", "conv_store", "spill", "reload", "lw_frame", "sw_frame"];
    let mut mem_count: u64 = 0;
    let mut non_mem_count: u64 = 0;
    println!("\n  Op breakdown (top 20):");
    for (i, (op, count)) in ops.iter().enumerate() {
        if i < 20 {
            let pct = **count as f64 / vm.steps as f64 * 100.0;
            let is_mem = mem_ops.contains(&op.as_str());
            println!("    {:>12}: {:>9} ({:5.1}%){}", op, count, pct, if is_mem { " [mem]" } else { "" });
        }
        if mem_ops.contains(&op.as_str()) {
            mem_count += **count;
        } else {
            non_mem_count += **count;
        }
    }
    let total = mem_count + non_mem_count;
    println!("\n  Memory ops:     {:>9} ({:.1}%)", mem_count, mem_count as f64 / total as f64 * 100.0);
    println!("  Non-memory ops: {:>9} ({:.1}%)", non_mem_count, non_mem_count as f64 / total as f64 * 100.0);

    // Per-base-op: how many distinct specialized opcodes were dynamically used?
    let mut by_base: HashMap<String, (u64, usize)> = HashMap::new(); // (dyn_count, num_specializations)
    for (spec, count) in &vm.spec_counts {
        // Extract base op: everything before the first '.'
        let base = spec.split('.').next().unwrap_or(spec).to_string();
        let e = by_base.entry(base).or_insert((0, 0));
        e.0 += count;
        e.1 += 1;
    }
    let mut by_base_vec: Vec<_> = by_base.iter().collect();
    by_base_vec.sort_by(|a, b| b.1.0.cmp(&a.1.0));
    println!("\n  Specializations per base op (dynamic):");
    for (base, (dyn_count, num_specs)) in &by_base_vec {
        let pct = *dyn_count as f64 / vm.steps as f64 * 100.0;
        println!("    {:>12}: {:>3} specializations, {:>9} dyn ({:5.1}%)", base, num_specs, dyn_count, pct);
    }

    // Dump all specialized opcodes with imm diversity
    println!("\n  Immediate analysis:");
    // Group spec_counts by base op, collect distinct imms per specialization
    // We need the actual instructions for this - scan function table
    let mut spec_imms: HashMap<String, std::collections::HashSet<i32>> = HashMap::new();
    for f in func_table.values() {
        for inst in &f.rewritten {
            if let Some(imm) = inst.imm {
                spec_imms.entry(inst.specialized.clone()).or_default().insert(imm);
            }
        }
    }
    // Count how many specialized opcodes have >1 distinct imm (static)
    let mut multi_imm: Vec<_> = spec_imms.iter().filter(|(_, imms)| imms.len() > 1).collect();
    multi_imm.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    if multi_imm.is_empty() {
        println!("    All specialized opcodes have at most 1 distinct immediate value.");
        println!("    => Immediate is fully determined by opcode. No runtime imm needed.");
    } else {
        println!("    Specialized opcodes with MULTIPLE distinct immediates (static sites):");
        for (spec, imms) in &multi_imm {
            let mut iv: Vec<_> = imms.iter().collect();
            iv.sort();
            println!("      {}: {} distinct imms: {:?}", spec, imms.len(), &iv[..iv.len().min(10)]);
        }
    }

    // Per-op: top 10 largest absolute immediates
    let mut op_imms: HashMap<String, Vec<i32>> = HashMap::new();
    for f in func_table.values() {
        for inst in &f.rewritten {
            if let Some(imm) = inst.imm {
                op_imms.entry(inst.op.clone()).or_default().push(imm);
            }
        }
    }
    println!("\n  Top 10 largest |imm| per op (excluding lui):");
    let mut op_list: Vec<_> = op_imms.iter().filter(|(op, _)| op.as_str() != "lui").collect();
    op_list.sort_by_key(|(op, _)| op.clone());
    for (op, imms) in &op_list {
        let mut unique: Vec<i32> = imms.iter().copied().collect::<std::collections::HashSet<_>>().into_iter().collect();
        unique.sort_by_key(|v| std::cmp::Reverse(v.unsigned_abs()));
        let top10: Vec<_> = unique.iter().take(10).map(|v| format!("{}", v)).collect();
        let abs_max = unique.first().map(|v| v.unsigned_abs()).unwrap_or(0);
        println!("    {:>12}: |max|={:>12}  top10: [{}]", op, abs_max, top10.join(", "));
    }

    // Check result
    if query_mode {
        let result = vm.memory.read_output_string();
        let expected = arg4.as_deref().unwrap_or("val5");
        println!("  Result:    {:?} (expected {:?})", result, expected);
        if result == expected {
            println!("\n  PASS");
        } else {
            println!("\n  FAIL");
            std::process::exit(1);
        }
    } else {
        let expected: u32 = if input_is_int {
            arg3.as_ref().and_then(|s| s.parse().ok()).unwrap_or(0)
        } else {
            arg3.as_ref().and_then(|s| s.parse().ok()).unwrap_or(0)
        };
        let result = vm.memory.read_u32(IO_OUTPUT_LEN);
        println!("  Result:    {} (expected {})", result, expected);
        if result == expected {
            println!("\n  PASS");
        } else {
            println!("\n  FAIL");
            std::process::exit(1);
        }
    }

    Ok(())
}
