use anyhow::Result;
use reg_analyzer::rv32::{decode_elf, get_elf_functions_named, build_cfg, classify_jalr_x0};
use reg_analyzer::rv32::legacy::inline_outlined_functions;
use reg_analyzer::rv32_regalloc::run_regalloc_with_symbols;
use reg_analyzer::rv32_isa_vm::{Rv32FuncInfo, Rv32IsaVm};
use std::collections::HashMap;
use std::fs;

fn main() -> Result<()> {
    let num_regs: u32 = std::env::args().nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let elf_path = std::env::args().nth(2)
        .unwrap_or_else(|| "../rv32-build/target/riscv32i-unknown-none-elf/release/json_query".to_string());
    let fixture_path = std::env::args().nth(3)
        .unwrap_or_else(|| "../guest-programs/json-query/fixtures/test_input.json".to_string());

    let data = fs::read(&elf_path)?;

    // Step 1: Decode and allocate
    println!("=== RV32 Rewritten-ISA Execution ({} regs) ===\n", num_regs);

    let (decoded_raw, _text_addr, _text_len) = decode_elf(&data)?;
    let elf_funcs_named = get_elf_functions_named(&data)?;
    let (mut decoded, elf_funcs) = inline_outlined_functions(&decoded_raw, &elf_funcs_named);
    println!("  ELF function symbols: {} (after inlining)", elf_funcs.len());
    let (jump_table_targets, _jump_table_bases) = classify_jalr_x0(&mut decoded, &data, &elf_funcs_named);
    let blocks = build_cfg(&decoded, &jump_table_targets);
    let alloc_result = run_regalloc_with_symbols(&decoded, &blocks, num_regs, &elf_funcs);

    let ok_funcs = alloc_result.func_results.iter().filter(|r| r.ok).count();
    let total_funcs = alloc_result.func_results.len();
    println!("  Functions: {}/{} OK", ok_funcs, total_funcs);

    for r in &alloc_result.func_results {
        if !r.ok {
            eprintln!("  FAIL 0x{:x}: {:?}", r.entry_addr, r.error);
        }
    }

    // Count total opcodes
    let mut all_opcodes = std::collections::HashSet::new();
    for r in &alloc_result.func_results {
        if r.ok {
            for inst in &r.rewritten {
                all_opcodes.insert(inst.specialized.clone());
            }
        }
    }
    println!("  Static ISA: {} unique opcodes", all_opcodes.len());

    // Step 2: Build function table and address→function map
    let mut func_table: HashMap<u32, Rv32FuncInfo> = HashMap::new();
    let mut addr_to_func: HashMap<u32, u32> = HashMap::new();  // any addr → func entry

    for r in &alloc_result.func_results {
        if !r.ok { continue; }

        // Map real addresses in this function to its entry
        // (skip synthetic trampoline addresses in 0xF0xxxxxx range)
        for inst in &r.rewritten {
            if inst.addr != 0 && inst.addr < 0xF000_0000 {
                addr_to_func.insert(inst.addr, r.entry_addr);
            }
        }

        func_table.insert(r.entry_addr, Rv32FuncInfo {
            rewritten: r.rewritten.clone(),
            num_spill_slots: r.num_spill_slots,
            entry_reg_map: r.entry_reg_map.clone(),
            jr_table_redirects: r.jr_table_redirects.clone(),
        });
    }

    println!("  Function table: {} entries", func_table.len());

    // Step 3: Load ELF and initialize VM
    let fixture = fs::read(&fixture_path)?;
    let mut vm = Rv32IsaVm::new(num_regs as usize);
    let entry_addr = vm.load_elf(&data)?;
    vm.write_input(&fixture);
    println!("  Entry: 0x{:x}\n", entry_addr);

    // Initialize conv_regs with SP (x2) — conv_regs bridges calling convention
    vm.conv_regs[2] = 0x7FFF0000;  // x2 = SP

    // Initialize frame pointer to spill area base (0x8000_1000..0x800F_FFFF)
    // Input data lives at 0x8010_0000 — no overlap
    vm.frame_reg = 0x8000_1000;

    // Dump functions 0x13d28 and 0x16c8c
    for &addr in &[0x13d28u32, 0x16c8c] {
        if let Some(f) = func_table.get(&addr) {
            println!("\n  Function 0x{:x} ({} insts, {} spills):", addr, f.rewritten.len(), f.num_spill_slots);
            for (i, inst) in f.rewritten.iter().enumerate() {
                println!("    {:3}. 0x{:06x}  {:35} rd={:?} rs1={:?} rs2={:?} imm={:?} o_rd={:?} o_rs1={:?}{}",
                    i, inst.addr, inst.specialized,
                    inst.rd, inst.rs1, inst.rs2, inst.imm, inst.orig_rd, inst.orig_rs1,
                    if inst.is_move { " [mv]" } else { "" });
            }
        }
    }

    // Step 4: Execute
    println!("\n  Executing...");

    if let Some(entry_func) = func_table.get(&entry_addr) {
        vm.execute_function(entry_func, &func_table, &addr_to_func, None)?;
    } else {
        anyhow::bail!("Entry function 0x{:x} not in function table (allocation failed?)", entry_addr);
    }

    println!("  Steps: {}", vm.steps);
    println!("  Halted: {}", vm.halted);

    // Print top instruction types
    let mut counts: Vec<_> = vm.op_counts.iter().collect();
    counts.sort_by_key(|(_, &count)| std::cmp::Reverse(count));
    println!("\n  Top instruction types:");
    for (op, count) in counts.iter().take(10) {
        println!("    {}: {} ({:.1}%)", op, count, 100.0 * (**count as f64) / (vm.steps as f64));
    }

    // Step 5: Check result
    let result = vm.memory.read_u32(0x80000000);
    let expected: u32 = std::env::args().nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(281);
    println!("\n  Result at 0x80000000: {}", result);
    if result == expected {
        println!("  PASS (expected {})", expected);
    } else {
        println!("  FAIL (expected {}, got {})", expected, result);
    }

    Ok(())
}
