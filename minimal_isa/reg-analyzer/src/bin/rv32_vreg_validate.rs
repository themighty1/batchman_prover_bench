/// VReg Validation — Step 3 correctness check.
///
/// Runs the plain Rv32Vm (ground truth) and the lifted vreg IR side-by-side.
/// For every executed instruction:
///   - Before execution: each source vreg's recorded value must match the
///     physical register in the Rv32Vm.
///   - After execution: record the destination vreg's new value from Rv32Vm.
///
/// At each block entry, the block's entry_vregs are initialised from the
/// current Rv32Vm register state (connecting cross-block value flow).
///
/// A mismatch means lift_to_vregs assigned the wrong vreg at a use site —
/// it is reading a stale definition instead of the current live value.

use anyhow::Result;
use reg_analyzer::rv32::{
    decode_elf, get_elf_functions_named, classify_jalr_x0, build_cfg, lift_to_vregs,
};
use reg_analyzer::rv32_vm::Rv32Vm;
use std::collections::{HashMap, HashSet};
use std::fs;

fn main() -> Result<()> {
    let elf_path = std::env::args().nth(1)
        .unwrap_or_else(|| "../rv32-build/target/riscv32i-unknown-none-elf/release/json_query".to_string());
    let fixture_path = std::env::args().nth(2)
        .unwrap_or_else(|| "../guest-programs/json-query/fixtures/test_input.json".to_string());
    let expected_result: u32 = std::env::args().nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(281);

    let data = fs::read(&elf_path)?;

    // -----------------------------------------------------------------------
    // Build vreg IR (Steps 1-3)
    // -----------------------------------------------------------------------
    let (decoded_raw, _, _) = decode_elf(&data)?;
    let elf_funcs_named = get_elf_functions_named(&data)?;
    let mut decoded = decoded_raw;
    let (jump_table_targets, _jump_table_bases) = classify_jalr_x0(&mut decoded, &data, &elf_funcs_named);
    let blocks = build_cfg(&decoded, &jump_table_targets);
    let (block_infos, total_vregs) = lift_to_vregs(&decoded, &blocks);

    println!("=== VReg Validation (Step 3 check) ===\n");
    println!("  ELF:    {}", elf_path);
    println!("  Blocks: {},  VRegs: {}\n", blocks.len(), total_vregs);

    // -----------------------------------------------------------------------
    // Index vreg IR
    // -----------------------------------------------------------------------

    // instruction addr → index into block_infos (bi_idx)
    let mut addr_to_bi: HashMap<u32, usize> = HashMap::new();
    // instruction addr → index within that block's insts vec
    let mut addr_to_inst_idx: HashMap<u32, usize> = HashMap::new();

    for (bi_idx, bi) in block_infos.iter().enumerate() {
        for (inst_idx, inst) in bi.insts.iter().enumerate() {
            addr_to_bi.insert(inst.addr, bi_idx);
            addr_to_inst_idx.insert(inst.addr, inst_idx);
        }
    }

    // Addresses that start a new block (trigger entry_vreg initialisation)
    let block_start_addrs: HashSet<u32> = blocks.iter().map(|b| b.start_addr).collect();

    // -----------------------------------------------------------------------
    // Execute + validate
    // -----------------------------------------------------------------------
    let fixture = fs::read(&fixture_path)?;
    let mut vm = Rv32Vm::new();
    let entry = vm.load_elf(&data)?;
    vm.write_input(&fixture);
    vm.pc = entry;

    // vreg → its currently recorded value (populated from Rv32Vm ground truth)
    let mut vreg_values: HashMap<u32, u32> = HashMap::new();
    vreg_values.insert(0, 0u32); // vreg 0 == x0 == always zero

    let mut checked = 0usize;
    let mut mismatches = 0usize;
    let mut steps = 0u64;
    let max_steps = 50_000_000u64;
    let max_report = 10usize; // stop printing after this many mismatches

    'run: while !vm.halted && steps < max_steps {
        let pc = vm.pc;
        steps += 1;

        // --- Block entry: seed entry_vregs from current physical registers ---
        if block_start_addrs.contains(&pc) {
            if let Some(&bi_idx) = addr_to_bi.get(&pc) {
                let bi = &block_infos[bi_idx];
                for r in 0..32usize {
                    let val = if r == 0 { 0 } else { vm.regs[r] };
                    vreg_values.insert(bi.entry_vregs[r], val);
                }
            }
        }

        // --- Vreg validation for this instruction ---
        if let (Some(&bi_idx), Some(&inst_idx)) =
            (addr_to_bi.get(&pc), addr_to_inst_idx.get(&pc))
        {
            let vi = &block_infos[bi_idx].insts[inst_idx];

            // Check each source vreg matches the physical register
            for &(vreg_opt, preg_opt) in &[
                (vi.rs1, vi.orig_rs1_preg),
                (vi.rs2, vi.orig_rs2_preg),
            ] {
                if let (Some(vreg), Some(preg)) = (vreg_opt, preg_opt) {
                    let phys = if preg == 0 { 0 } else { vm.regs[preg as usize] };
                    match vreg_values.get(&vreg) {
                        Some(&recorded) if recorded != phys => {
                            println!(
                                "  MISMATCH 0x{:x} {}: v{} (x{})  \
                                 recorded={:#010x}  actual={:#010x}",
                                pc, vi.op, vreg, preg, recorded, phys
                            );
                            mismatches += 1;
                            if mismatches >= max_report {
                                println!("  (stopped reporting after {} mismatches)", max_report);
                                break 'run;
                            }
                        }
                        None => {
                            // First use of this vreg (e.g. post-call fresh vreg).
                            // Initialise lazily from Rv32Vm — we trust the call
                            // boundary but will catch any stale use after this point.
                            vreg_values.insert(vreg, phys);
                        }
                        _ => {} // recorded == phys: correct
                    }
                    checked += 1;
                }
            }

            // Execute in Rv32Vm
            vm.step()?;

            // Record destination vreg after execution
            if let (Some(rd_vreg), Some(rd_preg)) = (vi.rd, vi.orig_rd_preg) {
                let val = if rd_preg == 0 { 0 } else { vm.regs[rd_preg as usize] };
                vreg_values.insert(rd_vreg, val);
            }
        } else {
            // Instruction outside our decoded stream — execute but don't track
            vm.step()?;
        }

        if vm.pc == pc { vm.halted = true; } // spin-loop halt
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!("  Steps:        {}", steps);
    println!("  Uses checked: {}", checked);
    println!("  Mismatches:   {}", mismatches);

    let result = vm.memory.read_u32(0x80000000);
    println!("\n  Result: {} (expected {})", result, expected_result);

    if mismatches == 0 && result == expected_result {
        println!("\n  PASS: vreg lifting is correct.");
    } else if mismatches > 0 {
        println!("\n  FAIL: lifting has mismatches — wrong vreg assigned at use site.");
    } else {
        println!("\n  FAIL: wrong result (lifting may be correct; bug is in Step 4/5/6).");
    }

    Ok(())
}
