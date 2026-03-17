//! Analyze liveness to determine how many immediate-elimination replacements
//! would require register spilling.
//!
//! Usage: analyze_liveness [program.bin]

use anyhow::Result;
use reg_analyzer::rv32_flat_vm::*;
use std::collections::{HashMap, HashSet};
use std::fs;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let program_path = args.get(1).map(|s| s.as_str()).unwrap_or("/tmp/flat3.bin");

    let encoded = fs::read(program_path)?;
    let program: FlatProgram = bincode::deserialize(&encoded)?;

    let num_regs = program.num_regs as usize;
    let num_insts = program.code_segment.len();

    println!("Program: {} ({} instructions, {} regs)", program_path, num_insts, num_regs);

    // Build opcode table
    let opcode_table: Vec<OpcodeInfo> = program.opcode_table.iter().map(|s| {
        OpcodeInfo {
            name: s.name.clone(),
            base_op: s.base_op.clone(),
            rd: s.rd,
            rs1: s.rs1,
            rs2: s.rs2,
            orig_rd: s.orig_rd,
            orig_rs1: None,
            orig_rs2: None,
        }
    }).collect();

    // Decode all instructions
    let mut insts = Vec::with_capacity(num_insts);
    for i in 0..num_insts {
        let opcode_id = program.code_segment[i];
        let imm = program.imm_table[i];
        let info = &opcode_table[opcode_id as usize];
        insts.push(Inst {
            base_op: info.base_op.clone(),
            rd: info.rd,
            rs1: info.rs1,
            rs2: info.rs2,
            imm,
        });
    }

    // --- Build basic blocks ---
    let branch_ops: HashSet<&str> = ["beq", "bne", "blt", "bge", "bltu", "bgeu"].into();
    let term_ops: HashSet<&str> = ["jal", "jal_call", "jalr", "jr_computed", "jr_table_idx", "ret", "halt", "ecall"].into();

    let mut block_starts: HashSet<usize> = HashSet::new();
    block_starts.insert(0);
    block_starts.insert(program.entry_pc as usize);

    for (i, inst) in insts.iter().enumerate() {
        let op = inst.base_op.as_str();
        if branch_ops.contains(op) {
            let target = inst.imm as usize;
            if target < num_insts { block_starts.insert(target); }
            if i + 1 < num_insts { block_starts.insert(i + 1); }
        } else if term_ops.contains(op) {
            if matches!(op, "jal" | "jal_call" | "jr_computed") {
                let target = inst.imm as usize;
                if target < num_insts { block_starts.insert(target); }
            }
            if i + 1 < num_insts { block_starts.insert(i + 1); }
        }
    }

    let mut block_starts_sorted: Vec<usize> = block_starts.into_iter().collect();
    block_starts_sorted.sort();

    // Map instruction index → block index
    let mut idx_to_block: HashMap<usize, usize> = HashMap::new();
    let mut block_ranges: Vec<(usize, usize)> = Vec::new(); // (start, end exclusive)

    for (bi, &start) in block_starts_sorted.iter().enumerate() {
        let end = if bi + 1 < block_starts_sorted.len() {
            block_starts_sorted[bi + 1]
        } else {
            num_insts
        };
        idx_to_block.insert(start, bi);
        block_ranges.push((start, end));
    }

    let num_blocks = block_ranges.len();
    println!("  Basic blocks: {}", num_blocks);

    // Build successor graph
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];

    for bi in 0..num_blocks {
        let (start, end) = block_ranges[bi];
        if end <= start { continue; }
        let last_idx = end - 1;
        let op = insts[last_idx].base_op.as_str();

        if branch_ops.contains(op) {
            let target = insts[last_idx].imm as usize;
            if let Some(&tbi) = idx_to_block.get(&target) {
                successors[bi].push(tbi);
            }
            if end < num_insts {
                if let Some(&nbi) = idx_to_block.get(&end) {
                    successors[bi].push(nbi);
                }
            }
        } else if matches!(op, "jal" | "jr_computed") {
            let target = insts[last_idx].imm as usize;
            if let Some(&tbi) = idx_to_block.get(&target) {
                successors[bi].push(tbi);
            }
        } else if op == "jal_call" {
            // Call target + fall-through (return comes back)
            let target = insts[last_idx].imm as usize;
            if let Some(&tbi) = idx_to_block.get(&target) {
                successors[bi].push(tbi);
            }
            if end < num_insts {
                if let Some(&nbi) = idx_to_block.get(&end) {
                    successors[bi].push(nbi);
                }
            }
        } else if matches!(op, "jalr" | "jr_table_idx") {
            // Indirect: conservatively, could go anywhere.
            // For liveness, treat as "all regs live" at block exit.
            // We handle this by not adding successors (live_out stays conservative).
        } else if matches!(op, "ret" | "halt" | "ecall") {
            // No successors
        } else {
            // Fall-through
            if end < num_insts {
                if let Some(&nbi) = idx_to_block.get(&end) {
                    successors[bi].push(nbi);
                }
            }
        }
    }

    // --- Liveness analysis ---
    let all_regs: u8 = (1u8 << num_regs) - 1;

    // defs/uses for each instruction
    fn get_defs_uses(inst: &Inst, all_regs: u8) -> (u8, u8) {
        match inst.base_op.as_str() {
            "save_context" => (0, all_regs),       // reads all GP regs
            "restore_context" => (all_regs, 0),     // writes all GP regs
            _ => {
                let mut defs = 0u8;
                let mut uses = 0u8;
                // rd = written register
                if let Some(rd) = inst.rd { defs |= 1 << rd; }
                // rs1, rs2 = read registers
                if let Some(rs1) = inst.rs1 { uses |= 1 << rs1; }
                if let Some(rs2) = inst.rs2 { uses |= 1 << rs2; }
                (defs, uses)
            }
        }
    }

    let mut live_in: Vec<u8> = vec![0; num_blocks];
    // For blocks with indirect jumps (jalr, jr_table_idx), conservatively assume all regs live out
    let mut live_out: Vec<u8> = vec![0; num_blocks];
    for bi in 0..num_blocks {
        let (start, end) = block_ranges[bi];
        if end > start {
            let last_op = insts[end - 1].base_op.as_str();
            if matches!(last_op, "jalr" | "jr_table_idx") {
                live_out[bi] = all_regs;
            }
        }
    }

    // Iterate until convergence
    let mut iterations = 0;
    loop {
        let mut changed = false;
        iterations += 1;
        for bi in (0..num_blocks).rev() {
            // live_out = union of live_in of successors (plus conservative bits)
            let mut new_out = live_out[bi] & all_regs; // keep conservative bits for indirect
            for &sbi in &successors[bi] {
                new_out |= live_in[sbi];
            }
            live_out[bi] = new_out;

            // Backward through block to compute live_in
            let (start, end) = block_ranges[bi];
            let mut live = new_out;
            for idx in (start..end).rev() {
                let (defs, uses) = get_defs_uses(&insts[idx], all_regs);
                live = (live & !defs) | uses;
            }
            if live != live_in[bi] {
                live_in[bi] = live;
                changed = true;
            }
        }
        if !changed { break; }
    }
    println!("  Liveness iterations: {}", iterations);

    // --- Analyze each immediate-using instruction ---
    // Build per-instruction extra cost: 0=not imm, 1=no spill (load_const), 3=spill (save+load_const+restore)
    let imm_ops: HashSet<&str> = [
        "addi", "lui", "slti", "sltiu", "xori", "ori", "andi",
        "slli", "srli", "srai",
        "lw", "sw", "lb", "lbu", "lh", "lhu", "sb", "sh",
        "lw_frame", "sw_frame", "addi_frame",
    ].into();

    let mut extra_cost: Vec<u8> = vec![0; num_insts]; // 0, 1, or 3
    let mut extra_cost_r2: Vec<u8> = vec![0; num_insts]; // fixed-r2 variant

    let mut total_imm = 0usize;
    let mut no_spill = 0usize;
    let mut need_spill = 0usize;
    let mut by_op: HashMap<String, (usize, usize)> = HashMap::new();

    let mut fixed_r2_ok = 0usize;
    let mut fixed_r2_spill = 0usize;
    let mut by_op_r2: HashMap<String, (usize, usize)> = HashMap::new();

    for bi in 0..num_blocks {
        let (start, end) = block_ranges[bi];

        // Backward pass to get live_after for each instruction
        let mut live = live_out[bi];
        let block_len = end - start;
        let mut live_after: Vec<u8> = vec![0; block_len];

        for local_idx in (0..block_len).rev() {
            live_after[local_idx] = live;
            let idx = start + local_idx;
            let (defs, uses) = get_defs_uses(&insts[idx], all_regs);
            live = (live & !defs) | uses;
        }

        // Check each immediate instruction
        for local_idx in 0..block_len {
            let idx = start + local_idx;
            let inst = &insts[idx];
            if !imm_ops.contains(inst.base_op.as_str()) { continue; }

            total_imm += 1;
            let la = live_after[local_idx];
            let spill = !can_avoid_spill(inst, la, num_regs);

            if spill {
                need_spill += 1;
                extra_cost[idx] = 3;
                by_op.entry(inst.base_op.clone()).or_insert((0, 0)).1 += 1;
            } else {
                no_spill += 1;
                extra_cost[idx] = 1;
                by_op.entry(inst.base_op.clone()).or_insert((0, 0)).0 += 1;
            }

            // Fixed-r2 analysis: can we use r2 specifically?
            // r2 must be dead after the instruction AND not be a source register
            // (clobbering r2 before the op would destroy the source)
            let r2_bit = 1u8 << 2;
            let r2_live_after = la & r2_bit != 0;
            let r2_is_source = inst.rs1 == Some(2) || inst.rs2 == Some(2);
            let r2_is_dest = inst.rd == Some(2);
            // If r2 is the destination and not a source, the old r2 value is dead anyway
            let r2_ok = (!r2_live_after && !r2_is_source) || (r2_is_dest && !r2_is_source);
            if r2_ok {
                fixed_r2_ok += 1;
                extra_cost_r2[idx] = 1;
                by_op_r2.entry(inst.base_op.clone()).or_insert((0, 0)).0 += 1;
            } else {
                fixed_r2_spill += 1;
                extra_cost_r2[idx] = 3;
                by_op_r2.entry(inst.base_op.clone()).or_insert((0, 0)).1 += 1;
            }
        }
    }

    println!("\n=== Static: Immediate elimination spill analysis ===");
    println!("  Non-branch immediate instructions: {}", total_imm);
    println!("  No spill needed:  {:>6} ({:.1}%)", no_spill, 100.0 * no_spill as f64 / total_imm as f64);
    println!("  Spill needed:     {:>6} ({:.1}%)", need_spill, 100.0 * need_spill as f64 / total_imm as f64);

    println!("\n  Per-op breakdown:");
    let mut ops: Vec<_> = by_op.into_iter().collect();
    ops.sort_by_key(|(_, (ns, s))| std::cmp::Reverse(*ns + *s));
    for (op, (ns, s)) in &ops {
        let total = ns + s;
        let pct = if *s > 0 { format!("{:.1}% spill", 100.0 * *s as f64 / total as f64) } else { "0% spill".into() };
        println!("    {:<12} {:>5} total  {:>5} no_spill  {:>5} spill  ({})", op, total, ns, s, pct);
    }

    println!("\n=== Static: Fixed-r2 convention ===");
    println!("  r2 available:     {:>6} ({:.1}%)", fixed_r2_ok, 100.0 * fixed_r2_ok as f64 / total_imm as f64);
    println!("  r2 needs spill:   {:>6} ({:.1}%)", fixed_r2_spill, 100.0 * fixed_r2_spill as f64 / total_imm as f64);

    println!("\n  Per-op breakdown:");
    let mut ops_r2: Vec<_> = by_op_r2.into_iter().collect();
    ops_r2.sort_by_key(|(_, (ns, s))| std::cmp::Reverse(*ns + *s));
    for (op, (ns, s)) in &ops_r2 {
        let total = ns + s;
        let pct = if *s > 0 { format!("{:.1}% spill", 100.0 * *s as f64 / total as f64) } else { "0% spill".into() };
        println!("    {:<12} {:>5} total  {:>5} r2_ok  {:>5} spill  ({})", op, total, ns, s, pct);
    }

    // --- Dynamic analysis: run VM with profiling ---
    let json_path = args.get(2).map(|s| s.as_str()).unwrap_or("../guest-programs/json-query/fixtures/test_input.json");
    let path = args.get(3).map(|s| s.as_str()).unwrap_or("data.5.v");
    let expected = args.get(4).map(|s| s.as_str()).unwrap_or("val5");

    let mut vm = FlatVm::new(num_regs, program.code_segment.clone(), program.imm_table.clone());
    vm.inst_hits = Some(vec![0u64; num_insts]);

    for seg in &program.segments {
        vm.memory.write_bytes(seg.vaddr, &seg.data);
    }

    let fixture = fs::read(json_path)?;
    vm.memory.write_input(&fixture);
    vm.memory.write_path(path.as_bytes());
    vm.memory.write_u32(reg_analyzer::rv32_isa_vm::MAILBOX_BASE + 2 * 4, 0x7FFF_0000);
    vm.pc = program.entry_pc * 4;

    vm.execute(&opcode_table)?;

    let result = vm.memory.read_output_string();
    assert_eq!(result, expected, "VM produced wrong result: {:?} != {:?}", result, expected);

    let hits = vm.inst_hits.as_ref().unwrap();
    let total_steps = vm.steps;

    // Compute dynamic extra instructions
    let mut dyn_no_spill: u64 = 0;   // hits on no-spill sites (each adds +1)
    let mut dyn_spill: u64 = 0;       // hits on spill sites (each adds +3)
    let mut dyn_by_op: HashMap<String, (u64, u64)> = HashMap::new();

    for idx in 0..num_insts {
        let h = hits[idx];
        if h == 0 || extra_cost[idx] == 0 { continue; }
        let op = &insts[idx].base_op;
        let entry = dyn_by_op.entry(op.clone()).or_insert((0, 0));
        if extra_cost[idx] == 1 {
            dyn_no_spill += h;
            entry.0 += h;
        } else {
            dyn_spill += h;
            entry.1 += h;
        }
    }

    let extra_insts = dyn_no_spill * 1 + dyn_spill * 3;

    println!("\n=== Dynamic: execution trace impact ===");
    println!("  Original trace:   {:>10} steps", total_steps);
    println!("  Imm hits (no spill): {:>10} × 1 = {:>10} extra", dyn_no_spill, dyn_no_spill);
    println!("  Imm hits (spill):    {:>10} × 3 = {:>10} extra", dyn_spill, dyn_spill * 3);
    println!("  Total extra instr:   {:>10} (+{:.1}%)", extra_insts, 100.0 * extra_insts as f64 / total_steps as f64);
    println!("  New trace size:      {:>10}", total_steps + extra_insts);

    println!("\n  Per-op dynamic breakdown:");
    let mut dops: Vec<_> = dyn_by_op.into_iter().collect();
    dops.sort_by_key(|(_, (ns, s))| std::cmp::Reverse(*ns + *s));
    for (op, (ns, s)) in &dops {
        let extra = ns * 1 + s * 3;
        println!("    {:<12} {:>10} no_spill  {:>10} spill  {:>10} extra instr", op, ns, s, extra);
    }

    // Full trace breakdown by base_op
    let mut op_hits: HashMap<String, u64> = HashMap::new();
    for idx in 0..num_insts {
        let h = hits[idx];
        if h == 0 { continue; }
        *op_hits.entry(insts[idx].base_op.clone()).or_insert(0) += h;
    }
    let mut op_list: Vec<_> = op_hits.iter().collect();
    op_list.sort_by_key(|(_, h)| std::cmp::Reverse(**h));

    println!("\n=== Dynamic: full trace breakdown by base_op ===");
    println!("  Total steps: {}", total_steps);

    let mem_ops: HashSet<&str> = [
        "lw", "sw", "lb", "lbu", "lh", "lhu", "sb", "sh",
        "lw_frame", "sw_frame",
    ].into();
    let branch_ops_set: HashSet<&str> = ["beq", "bne", "blt", "bge", "bltu", "bgeu"].into();
    let jump_ops_set: HashSet<&str> = ["jal", "jal_call", "jalr", "jr_computed", "jr_table_idx", "ret"].into();
    let conv_ops: HashSet<&str> = ["conv_load", "conv_store"].into();

    let mut cat_mem: u64 = 0;
    let mut cat_branch: u64 = 0;
    let mut cat_jump: u64 = 0;
    let mut cat_alu: u64 = 0;
    let mut cat_conv: u64 = 0;
    let mut cat_other: u64 = 0;

    for (op, h) in &op_list {
        let pct = 100.0 * **h as f64 / total_steps as f64;
        println!("    {:<18} {:>10} ({:>5.1}%)", op, h, pct);
        if mem_ops.contains(op.as_str()) { cat_mem += **h; }
        else if branch_ops_set.contains(op.as_str()) { cat_branch += **h; }
        else if jump_ops_set.contains(op.as_str()) { cat_jump += **h; }
        else if conv_ops.contains(op.as_str()) { cat_conv += **h; }
        else if matches!(op.as_str(), "save_context" | "restore_context" | "halt" | "ecall" | "addi_frame") { cat_other += **h; }
        else { cat_alu += **h; }
    }

    println!("\n  Categories:");
    println!("    Memory (lw/sw/lb/sb/..):  {:>10} ({:.1}%)", cat_mem, 100.0 * cat_mem as f64 / total_steps as f64);
    println!("    Conv (mailbox):           {:>10} ({:.1}%)", cat_conv, 100.0 * cat_conv as f64 / total_steps as f64);
    println!("    ALU (add/sub/shift/..):   {:>10} ({:.1}%)", cat_alu, 100.0 * cat_alu as f64 / total_steps as f64);
    println!("    Branch (beq/bne/..):      {:>10} ({:.1}%)", cat_branch, 100.0 * cat_branch as f64 / total_steps as f64);
    println!("    Jump (jal/ret/..):        {:>10} ({:.1}%)", cat_jump, 100.0 * cat_jump as f64 / total_steps as f64);
    println!("    Other (save/restore/..):  {:>10} ({:.1}%)", cat_other, 100.0 * cat_other as f64 / total_steps as f64);

    // Fixed-r2 dynamic analysis
    let mut dyn_r2_ok: u64 = 0;
    let mut dyn_r2_spill: u64 = 0;
    let mut dyn_by_op_r2: HashMap<String, (u64, u64)> = HashMap::new();

    for idx in 0..num_insts {
        let h = hits[idx];
        if h == 0 || extra_cost_r2[idx] == 0 { continue; }
        let op = &insts[idx].base_op;
        let entry = dyn_by_op_r2.entry(op.clone()).or_insert((0, 0));
        if extra_cost_r2[idx] == 1 {
            dyn_r2_ok += h;
            entry.0 += h;
        } else {
            dyn_r2_spill += h;
            entry.1 += h;
        }
    }

    let extra_r2 = dyn_r2_ok + dyn_r2_spill * 3;

    println!("\n=== Dynamic: Fixed-r2 convention ===");
    println!("  Original trace:   {:>10} steps", total_steps);
    println!("  r2 ok (no spill):    {:>10} × 1 = {:>10} extra", dyn_r2_ok, dyn_r2_ok);
    println!("  r2 needs spill:      {:>10} × 3 = {:>10} extra", dyn_r2_spill, dyn_r2_spill * 3);
    println!("  Total extra instr:   {:>10} (+{:.1}%)", extra_r2, 100.0 * extra_r2 as f64 / total_steps as f64);
    println!("  New trace size:      {:>10}", total_steps + extra_r2);

    println!("\n  Per-op dynamic breakdown:");
    let mut dops_r2: Vec<_> = dyn_by_op_r2.into_iter().collect();
    dops_r2.sort_by_key(|(_, (ns, s))| std::cmp::Reverse(*ns + *s));
    for (op, (ns, s)) in &dops_r2 {
        let extra = s * 3 + ns;
        println!("    {:<12} {:>10} r2_ok  {:>10} spill  {:>10} extra instr", op, ns, s, extra);
    }

    Ok(())
}

/// Check if an immediate instruction can be replaced without spilling.
fn can_avoid_spill(inst: &Inst, live_after: u8, num_regs: usize) -> bool {
    let op = inst.base_op.as_str();

    match op {
        // lui rD, imm → just becomes load_const.rD, always safe
        "lui" => true,

        // lw_frame rD, imm → load const into rD, then lw from frame_reg+rD.
        // lw_frame only reads frame_reg (not a GP reg), so rD is always safe to clobber first.
        "lw_frame" => true,

        // addi_frame imm → need any dead GP reg for the constant
        "addi_frame" => {
            for r in 0..num_regs as u8 {
                if live_after & (1 << r) == 0 { return true; }
            }
            false
        }

        // sw_frame rVal, imm → need a GP reg (not rVal) for offset
        "sw_frame" => {
            let val_reg = inst.rs2;
            for r in 0..num_regs as u8 {
                if Some(r) == val_reg { continue; }
                if live_after & (1 << r) == 0 { return true; }
            }
            false
        }

        // sw/sb/sh: no rd, has rs1 (base) + rs2 (value). Need a 3rd reg for offset.
        "sw" | "sb" | "sh" => {
            for r in 0..num_regs as u8 {
                if Some(r) == inst.rs1 { continue; }
                if Some(r) == inst.rs2 { continue; }
                if live_after & (1 << r) == 0 { return true; }
            }
            false
        }

        // Everything else: has rd and rs1 (and sometimes rs2).
        // Case A: rD ≠ rS1 (and rD ≠ rS2) → use rD for constant
        // Case B: find any dead register not used as source
        _ => {
            let rd = inst.rd;
            let rs1 = inst.rs1;
            let rs2 = inst.rs2;

            // Case A: destination not same as any source → safe to use rD
            if rd.is_some() && rd != rs1 && rd != rs2 {
                return true;
            }

            // Case B: find a dead register not used as source
            for r in 0..num_regs as u8 {
                if Some(r) == rs1 { continue; }
                if Some(r) == rs2 { continue; }
                if live_after & (1 << r) == 0 { return true; }
            }

            false
        }
    }
}

struct Inst {
    base_op: String,
    rd: Option<u8>,
    rs1: Option<u8>,
    rs2: Option<u8>,
    #[allow(dead_code)]
    imm: i32,
}
