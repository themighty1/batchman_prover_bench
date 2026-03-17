//! Unified ISA-VM verifier for any RV32 ELF using the shared MMIO convention:
//!   IO_OUTPUT_LEN  — u32 result written by guest (or output length for query mode)
//!   IO_INPUT_LEN   — u32 (byte length for json / iteration count for toy)
//!   IO_INPUT_DATA  — raw bytes (json programs only)
//!
//! Usage:
//!   rv32_verify [num_regs] [elf_path] [input] [expected]
//!   rv32_verify [num_regs] [elf_path] [json_file] [path] [expected_string]
//!
//! Mode 1 (legacy):
//!   [input] is either:
//!     - a path to a fixture file  → write_input(bytes) (len + data)
//!     - an integer                → write_input_u32(n) (len field only)
//!   [expected] is the u32 value expected at OUTPUT_ADDR after execution.
//!
//! Mode 2 (json query — 5 args):
//!   [json_file] is a path to a JSON fixture file
//!   [path] is a dot-notation JSON path (e.g. "data.5.v")
//!   [expected_string] is the expected output string (e.g. "val5")
//!
//! Defaults: 4 regs, json_query.elf, fixture.json, expected=281

use anyhow::Result;
use reg_analyzer::rv32::{decode_elf, get_elf_functions_named, build_cfg, classify_jalr_x0};
use reg_analyzer::rv32_regalloc::{build_branch_target_map, run_regalloc_with_symbols};
use reg_analyzer::rv32_isa_vm::{Rv32FuncInfo, Rv32IsaVm, MAILBOX_BASE, IO_OUTPUT_LEN};
use std::collections::HashMap;
use std::fs;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let num_regs: u32 = args.get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let elf_path = args.get(2)
        .cloned()
        .unwrap_or_else(|| "../rv32-build/target/riscv32i-unknown-none-elf/release/json_query".to_string());

    let input_arg = args.get(3)
        .cloned()
        .unwrap_or_else(|| "../guest-programs/json-query/fixtures/test_input.json".to_string());

    // Detect query mode: if we have 6 args total (program + 5), it's query mode
    let query_mode = args.len() >= 6;

    let query_path = if query_mode { args.get(4).cloned() } else { None };
    let expected_string = if query_mode { args.get(5).cloned() } else { None };

    let expected_u32: u32 = if !query_mode {
        args.get(4).and_then(|s| s.parse().ok()).unwrap_or(281)
    } else {
        0
    };

    // Detect input type (legacy mode): integer → write_input_u32, otherwise file → write_input
    let input_is_int = !query_mode && input_arg.parse::<u32>().is_ok();

    println!("=== rv32_verify ({} regs) ===", num_regs);
    println!("  ELF:      {}", elf_path);
    if query_mode {
        println!("  JSON:     {}", input_arg);
        println!("  Path:     {}", query_path.as_deref().unwrap_or(""));
        println!("  Expected: {:?}\n", expected_string.as_deref().unwrap_or(""));
    } else if input_is_int {
        println!("  Input:    n={}", input_arg);
        println!("  Expected: {}\n", expected_u32);
    } else {
        println!("  Input:    {}", input_arg);
        println!("  Expected: {}\n", expected_u32);
    }

    // Step 1: decode + regalloc
    let data = fs::read(&elf_path)?;
    let (decoded_raw, _text_addr, _text_len) = decode_elf(&data)?;
    let elf_funcs_named = get_elf_functions_named(&data)?;
    let mut decoded = decoded_raw;
    let elf_funcs: Vec<(u32, u32)> = elf_funcs_named.iter().map(|(a, s, _)| (*a, *s)).collect();
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

    // Count unique opcodes
    let mut all_opcodes = std::collections::HashSet::new();
    for r in &alloc_result.func_results {
        if r.ok {
            for inst in &r.rewritten {
                all_opcodes.insert(inst.specialized.clone());
            }
        }
    }
    println!("  ISA size:  {} unique opcodes", all_opcodes.len());

    // Step 2: build function table
    let mut func_table: HashMap<u32, Rv32FuncInfo> = HashMap::new();
    let mut addr_to_func: HashMap<u32, u32> = HashMap::new();

    for r in &alloc_result.func_results {
        if !r.ok { continue; }
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

    // Step 3: load ELF and write input
    let mut vm = Rv32IsaVm::new(num_regs as usize);
    let entry_addr = vm.load_elf(&data)?;

    if query_mode {
        // Query mode: write JSON data + path
        let fixture = fs::read(&input_arg)?;
        vm.write_input(&fixture);
        if let Some(ref path) = query_path {
            vm.memory.write_path(path.as_bytes());
        }
    } else if input_is_int {
        let n: u32 = input_arg.parse().unwrap();
        vm.memory.write_input_u32(n);
    } else {
        let fixture = fs::read(&input_arg)?;
        vm.write_input(&fixture);
    }

    vm.memory.write_u32(MAILBOX_BASE + 2 * 4, 0x7FFF_0000);  // SP in mailbox
    vm.frame_reg = 0x8000_1000;

    println!("  Entry:     0x{:x}\n", entry_addr);

    // Dump function if DUMP_FUNC env var is set
    if let Ok(dump_addr_str) = std::env::var("DUMP_FUNC") {
        let dump_addr = u32::from_str_radix(&dump_addr_str, 16).unwrap();
        if let Some(f) = func_table.get(&dump_addr) {
            eprintln!("=== Rewritten function 0x{:x} ({} instructions, {} spill slots) ===",
                dump_addr, f.rewritten.len(), f.num_spill_slots);
            for (i, inst) in f.rewritten.iter().enumerate() {
                eprintln!("  [{:3}] addr=0x{:08x} {} rd={:?} rs1={:?} rs2={:?} imm={:?} orig_rd={:?} orig_rs1={:?} orig_rs2={:?} {}{}",
                    i, inst.addr, inst.op,
                    inst.rd, inst.rs1, inst.rs2, inst.imm,
                    inst.orig_rd, inst.orig_rs1, inst.orig_rs2,
                    inst.specialized,
                    if inst.is_move { " [MOVE]" } else { "" });
            }
            // Rebuild addr_to_idx locally for debug dump (not stored in Rv32FuncInfo)
            let addr_to_idx = build_branch_target_map(&f.rewritten);
            eprintln!("=== addr_to_idx (debug, {} entries) ===", addr_to_idx.len());
            let mut entries: Vec<_> = addr_to_idx.iter().collect();
            entries.sort_by_key(|(_, &idx)| idx);
            for (&addr, &idx) in &entries {
                eprintln!("  0x{:08x} -> idx {}", addr, idx);
            }
        }
    }

    // Step 4: execute
    if let Some(entry_func) = func_table.get(&entry_addr) {
        vm.execute_function(entry_func, &func_table, &addr_to_func, None)?;
    } else {
        anyhow::bail!("entry 0x{:x} not in function table", entry_addr);
    }

    println!("  Steps:     {}", vm.steps);

    // Step 5: check result
    if query_mode {
        let result = vm.memory.read_output_string();
        let expected = expected_string.as_deref().unwrap_or("");
        println!("  Result:    {:?} (expected {:?})", result, expected);

        if result == expected {
            println!("\n  PASS");
        } else {
            println!("\n  FAIL");
            std::process::exit(1);
        }
    } else {
        let result = vm.memory.read_u32(IO_OUTPUT_LEN);
        println!("  Result:    {} (expected {})", result, expected_u32);

        if result == expected_u32 {
            println!("\n  PASS");
        } else {
            println!("\n  FAIL");
            std::process::exit(1);
        }
    }

    Ok(())
}
