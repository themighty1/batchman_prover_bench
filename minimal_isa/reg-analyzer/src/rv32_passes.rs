//! Post-rewrite passes for the RV32 ISA-VM pipeline.
//!
//! These passes transform the rewritten instruction stream after regalloc
//! and spill lowering. Each pass is a pure function:
//! `&[RewrittenInst] -> Vec<RewrittenInst>`.

use std::collections::{HashMap, HashSet, BTreeSet};
use crate::rv32_regalloc::{RewrittenInst, build_branch_target_map};
use crate::rv32_isa_vm::MemSegment;

/// Returns true if the instruction is a function call (jal or jalr with orig_rd != 0).
pub fn is_call(inst: &RewrittenInst) -> bool {
    if inst.op == "jal" && inst.orig_rd.map_or(false, |r| r != 0) {
        return true;
    }
    if inst.op == "jalr" && inst.orig_rd.map_or(false, |r| r != 0) {
        return true;
    }
    false
}

// ---------------------------------------------------------------------------
// Pass: lower spill/reload into concrete sw/lw
// ---------------------------------------------------------------------------

/// Lower spill/reload instructions into real sw/lw using the frame register.
///
/// `frame_reg` is the physical register ID used as the frame pointer
/// (typically `num_regs`, i.e. one above the allocatable range).
///
/// Transforms:
/// - `spill.rX slot=N` → `sw rX, offset(frame_reg)`
/// - `reload.rX slot=N` → `lw rX, offset(frame_reg)`
/// - Inserts `addi frame_reg, frame_reg, frame_size` at function entry
/// - Inserts `addi frame_reg, frame_reg, -frame_size` before each return
pub fn lower_spills(rewritten: &[RewrittenInst], num_spill_slots: usize, frame_reg: u8) -> Vec<RewrittenInst> {
    let frame_size = (num_spill_slots * 4) as i32;
    let mut result = Vec::with_capacity(rewritten.len() + 4);

    // Function prologue: advance frame pointer
    if frame_size > 0 {
        result.push(RewrittenInst {
            addr: 0,
            op: "addi".into(),
            rd: Some(frame_reg),
            rs1: Some(frame_reg),
            rs2: None,
            imm: Some(frame_size),
            is_move: false,
            specialized: format!("addi.r{}.r{}", frame_reg, frame_reg),
            orig_rd: None, orig_rs1: None, orig_rs2: None,
        });
    }

    for inst in rewritten {
        match inst.op.as_str() {
            "spill" => {
                // spill.rX slot=N → sw rX, (N*4 - frame_size)(frame_reg)
                // Uses negative offsets so spill area is BELOW frame_reg,
                // preventing overlap between caller and callee spill areas.
                let src = inst.rs1.unwrap();
                let slot_offset = inst.imm.unwrap() * 4 - frame_size;
                result.push(RewrittenInst {
                    addr: inst.addr,
                    op: "sw".into(),
                    rd: None,
                    rs1: Some(frame_reg),
                    rs2: Some(src),
                    imm: Some(slot_offset),
                    is_move: false,
                    specialized: format!("sw.r{}.r{}", frame_reg, src),
                    orig_rd: None, orig_rs1: None, orig_rs2: None,
                });
            }
            "reload" => {
                // reload.rX slot=N → lw rX, (N*4 - frame_size)(frame_reg)
                let dst = inst.rd.unwrap();
                let slot_offset = inst.imm.unwrap() * 4 - frame_size;
                result.push(RewrittenInst {
                    addr: inst.addr,
                    op: "lw".into(),
                    rd: Some(dst),
                    rs1: Some(frame_reg),
                    rs2: None,
                    imm: Some(slot_offset),
                    is_move: false,
                    specialized: format!("lw.r{}.r{}", dst, frame_reg),
                    orig_rd: None, orig_rs1: None, orig_rs2: None,
                });
            }
            "mov" => {
                // mov is a first-class instruction in our custom ISA
                result.push(inst.clone());
            }
            "restack" => {
                // restack = spill-to-spill move. Should be rare.
                // For now, panic — we'll handle it if it actually occurs.
                panic!("restack not yet supported in lower_spills");
            }
            "nop_move" => {
                // Skip — no-op moves are eliminated
            }
            "ret" => {
                // Return instruction — insert frame restore before it.
                // Use ret's addr so that jumps targeting ret land here first.
                if frame_size > 0 {
                    result.push(RewrittenInst {
                        addr: inst.addr,
                        op: "addi".into(),
                        rd: Some(frame_reg),
                        rs1: Some(frame_reg),
                        rs2: None,
                        imm: Some(-frame_size),
                        is_move: false,
                        specialized: format!("addi.r{}.r{}", frame_reg, frame_reg),
                        orig_rd: None, orig_rs1: None, orig_rs2: None,
                    });
                }
                result.push(inst.clone());
            }
            _ => {
                result.push(inst.clone());
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Pass: split frame-relative ops
// ---------------------------------------------------------------------------

/// Splits instructions that use the frame register into `_frame` variants.
///
/// For example:
/// - `lw.r0.rF`  → op="lw_frame",  specialized="lw_frame.r0"
/// - `sw.rF.r1`  → op="sw_frame",  specialized="sw_frame.r1"
/// - `addi.rF.rF` → op="addi_frame", specialized="addi_frame"
///
/// This eliminates the runtime branch in the VM that checks `rs1 == frame_reg_id`.
pub fn pass_split_frame_ops(rewritten: &[RewrittenInst], frame_reg: u8) -> Vec<RewrittenInst> {
    let mut result = Vec::with_capacity(rewritten.len());

    for inst in rewritten {
        let uses_frame_rd = inst.rd == Some(frame_reg);
        let uses_frame_rs1 = inst.rs1 == Some(frame_reg);
        let uses_frame_rs2 = inst.rs2 == Some(frame_reg);

        if !uses_frame_rd && !uses_frame_rs1 && !uses_frame_rs2 {
            result.push(inst.clone());
            continue;
        }

        // Build new op name and specialized string without the frame register
        let new_op = format!("{}_frame", inst.op);
        let mut spec_parts = vec![format!("{}_frame", inst.op)];

        // rd — keep if not frame reg
        let new_rd = if uses_frame_rd { None } else { inst.rd };
        if let Some(rd) = new_rd {
            spec_parts.push(format!("r{}", rd));
        }

        // rs1 — keep if not frame reg
        let new_rs1 = if uses_frame_rs1 { None } else { inst.rs1 };
        if let Some(rs1) = new_rs1 {
            spec_parts.push(format!("r{}", rs1));
        }

        // rs2 — keep if not frame reg
        let new_rs2 = if uses_frame_rs2 { None } else { inst.rs2 };
        if let Some(rs2) = new_rs2 {
            spec_parts.push(format!("r{}", rs2));
        }

        result.push(RewrittenInst {
            addr: inst.addr,
            op: new_op,
            rd: new_rd,
            rs1: new_rs1,
            rs2: new_rs2,
            imm: inst.imm,
            is_move: inst.is_move,
            specialized: spec_parts.join("."),
            orig_rd: inst.orig_rd,
            orig_rs1: inst.orig_rs1,
            orig_rs2: inst.orig_rs2,
        });
    }

    result
}

// ---------------------------------------------------------------------------
// Pass: resolve branch targets to instruction indices
// ---------------------------------------------------------------------------

/// Replaces branch/jump address offsets with direct instruction indices.
///
/// Before: `beq r0, r1, imm=offset` → VM computes `inst.addr + offset`, looks up `addr_to_idx`
/// After:  `beq r0, r1, imm=target_idx` → VM just does `pc = target_idx`
///
/// This eliminates the `addr_to_idx` HashMap lookup on every branch.
/// Only applies to statically-resolvable targets (conditional branches, jal, jr_computed).
pub fn pass_resolve_branches(rewritten: &[RewrittenInst]) -> Vec<RewrittenInst> {
    // Build addr → index map
    let addr_to_idx = build_branch_target_map(rewritten);

    let mut result = Vec::with_capacity(rewritten.len());

    for (idx, inst) in rewritten.iter().enumerate() {
        match inst.op.as_str() {
            "beq" | "bne" | "blt" | "bge" | "bltu" | "bgeu" => {
                // Conditional branch: target = inst.addr + offset
                let offset = inst.imm.unwrap();
                let target_addr = (inst.addr as i64 + offset as i64) as u32;
                let target_idx = addr_to_idx.get(&target_addr)
                    .unwrap_or_else(|| panic!("{} target 0x{:x} not in addr_to_idx (at idx {})", inst.op, target_addr, idx));
                let mut new_inst = inst.clone();
                new_inst.imm = Some(*target_idx as i32);
                result.push(new_inst);
            }
            "jal" => {
                let orig_rd = inst.orig_rd.unwrap_or(0);
                if orig_rd == 0 {
                    // Unconditional jump: target = inst.addr + offset
                    let offset = inst.imm.unwrap_or(0);
                    let target_addr = (inst.addr as i64 + offset as i64) as u32;
                    let target_idx = addr_to_idx.get(&target_addr)
                        .unwrap_or_else(|| panic!("jal target 0x{:x} not in addr_to_idx (at idx {})", target_addr, idx));
                    let mut new_inst = inst.clone();
                    new_inst.imm = Some(*target_idx as i32);
                    result.push(new_inst);
                } else {
                    // Call — target is a function entry, resolved via func_table at runtime
                    result.push(inst.clone());
                }
            }
            "jr_computed" => {
                // Already stores target addr in imm, resolve to index
                let target_addr = inst.imm.unwrap() as u32;
                let target_idx = addr_to_idx.get(&target_addr)
                    .unwrap_or_else(|| panic!("jr_computed target 0x{:x} not in addr_to_idx (at idx {})", target_addr, idx));
                let mut new_inst = inst.clone();
                new_inst.imm = Some(*target_idx as i32);
                result.push(new_inst);
            }
            _ => {
                result.push(inst.clone());
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Post-compile pass: rewrite jump tables in memory to instruction indices
// ---------------------------------------------------------------------------

/// Patches jump table entries in memory segments from code addresses to instruction indices.
///
/// For each jr_table instruction, the jump table in memory contains code addresses.
/// This pass replaces each address with the corresponding instruction index (after
/// applying jr_table_redirects). After this, `jr_table` becomes `jr_table_idx`
/// and the VM handler is just `pc = regs[rs1]`.
///
/// Parameters:
/// - `functions`: mutable slice of Rv32FuncInfo (jr_table ops are renamed in-place)
/// - `segments`: mutable slice of MemSegment (jump table entries are patched)
/// - `jump_table_bases`: map of jr_table instruction addr → (table_base_addr, num_entries)
pub fn pass_rewrite_jump_tables(
    functions: &mut [crate::rv32_isa_vm::Rv32FuncInfo],
    segments: &mut [MemSegment],
    jump_table_bases: &HashMap<u32, (u32, usize)>,
) {
    // Helper: read u32 little-endian from memory segments
    fn read_seg_u32(segments: &[MemSegment], addr: u32) -> Option<u32> {
        for seg in segments.iter() {
            let end = seg.vaddr.wrapping_add(seg.data.len() as u32);
            if addr >= seg.vaddr && addr.wrapping_add(4) <= end {
                let off = (addr - seg.vaddr) as usize;
                return Some(u32::from_le_bytes([
                    seg.data[off], seg.data[off + 1], seg.data[off + 2], seg.data[off + 3],
                ]));
            }
        }
        None
    }

    // Helper: write u32 little-endian to memory segments
    fn write_seg_u32(segments: &mut [MemSegment], addr: u32, val: u32) {
        for seg in segments.iter_mut() {
            let end = seg.vaddr.wrapping_add(seg.data.len() as u32);
            if addr >= seg.vaddr && addr.wrapping_add(4) <= end {
                let off = (addr - seg.vaddr) as usize;
                seg.data[off..off + 4].copy_from_slice(&val.to_le_bytes());
                return;
            }
        }
        panic!("write_seg_u32: address 0x{:x} not in any segment", addr);
    }

    let mut patched = 0;
    for func in functions.iter_mut() {
        let addr_to_idx = build_branch_target_map(&func.rewritten);

        for inst in func.rewritten.iter_mut() {
            if inst.op != "jr_table" { continue; }

            let jr_addr = inst.addr;
            if let Some(&(table_base, num_entries)) = jump_table_bases.get(&jr_addr) {
                // Patch each entry in the jump table
                for entry_idx in 0..num_entries {
                    let entry_addr = table_base.wrapping_add((entry_idx * 4) as u32);
                    let code_addr = read_seg_u32(segments, entry_addr)
                        .unwrap_or_else(|| panic!("jr_table at 0x{:x}: can't read table entry at 0x{:x}", jr_addr, entry_addr));
                    // Apply redirect if needed
                    let final_addr = func.jr_table_redirects
                        .get(&(jr_addr, code_addr))
                        .copied()
                        .unwrap_or(code_addr);
                    // Resolve to instruction index
                    let idx = addr_to_idx.get(&final_addr)
                        .unwrap_or_else(|| panic!("jr_table at 0x{:x}: target 0x{:x} (redirect of 0x{:x}) not in addr_to_idx",
                            jr_addr, final_addr, code_addr));
                    write_seg_u32(segments, entry_addr, *idx as u32);
                }
                // Rename the op so the VM knows the register holds an index
                inst.op = "jr_table_idx".to_string();
                inst.specialized = inst.specialized.replace("jr_table", "jr_table_idx");
                patched += 1;
            }
        }
    }

    if patched > 0 {
        eprintln!("  pass_rewrite_jump_tables: patched {} jr_table instructions", patched);
    }
}

// ---------------------------------------------------------------------------
// Pass: dead conv_store elimination (per-function)
// ---------------------------------------------------------------------------

/// Eliminates redundant conv_store instructions within a function.
///
/// A conv_store for orig X is "dead" if, on every path from it to a sync point
/// (call, ret, or function end), X is re-conv_stored before that sync point.
///
/// Returns a `Vec<bool>` of keep flags (true = keep, false = strip).
pub fn pass_dead_conv_store(rewritten: &[RewrittenInst]) -> Vec<bool> {
    let n = rewritten.len();
    let mut keep = vec![true; n];

    let branch_ops: HashSet<&str> = [
        "beq", "bne", "blt", "bge", "bltu", "bgeu", "jal", "jalr",
        "jr_computed", "jr_table_idx", "jr_table", "ret",
    ].into_iter().collect();

    // Collect all origs that have conv_stores
    let mut all_origs: HashSet<u8> = HashSet::new();
    for inst in rewritten {
        if inst.op == "conv_store" {
            if let Some(orig) = inst.orig_rd {
                all_origs.insert(orig);
            }
        }
    }
    if all_origs.is_empty() {
        return keep;
    }

    // Build basic blocks
    let mut is_bb_start = vec![false; n];
    is_bb_start[0] = true;
    for (i, inst) in rewritten.iter().enumerate() {
        if branch_ops.contains(inst.op.as_str()) {
            if let Some(target) = inst.imm {
                let target = target as usize;
                if target < n {
                    is_bb_start[target] = true;
                }
            }
            if i + 1 < n { is_bb_start[i + 1] = true; }
        }
        if is_call(inst) && i + 1 < n {
            is_bb_start[i + 1] = true;
        }
    }

    let mut blocks: Vec<(usize, usize)> = Vec::new();
    let mut block_of: Vec<usize> = vec![0; n];
    {
        let mut start = 0;
        for i in 1..=n {
            if i == n || is_bb_start[i] {
                let block_idx = blocks.len();
                for j in start..i { block_of[j] = block_idx; }
                blocks.push((start, i));
                start = i;
            }
        }
    }
    let num_blocks = blocks.len();

    // Build successor map
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];
    for (bi, &(_start, end)) in blocks.iter().enumerate() {
        let last = end - 1;
        let last_inst = &rewritten[last];
        let last_op = last_inst.op.as_str();

        if !["ret", "jal", "jalr", "jr_computed", "jr_table_idx", "jr_table"].contains(&last_op)
            || (last_op == "jal" && last_inst.orig_rd.map_or(false, |rd| rd != 0))
        {
            if end < n {
                succs[bi].push(block_of[end]);
            }
        }
        if branch_ops.contains(last_op) && !matches!(last_op, "jalr" | "jr_table") {
            // jalr.imm is a register offset (not a branch target index),
            // jr_table.imm is also from the original jalr encoding (not a target),
            // so exclude both from branch-target successor computation.
            if let Some(target) = last_inst.imm {
                let target = target as usize;
                if target < n {
                    succs[bi].push(block_of[target]);
                }
            }
        }
        if is_call(last_inst) {
            if end < n && !succs[bi].contains(&block_of[end]) {
                succs[bi].push(block_of[end]);
            }
        }
    }

    // Dataflow using bitsets (origs are 0-31, fits in u32)
    let mut orig_to_bit: HashMap<u8, u32> = HashMap::new();
    for (i, &orig) in all_origs.iter().enumerate() {
        orig_to_bit.insert(orig, 1u32 << i);
    }
    let all_bits: u32 = (1u32 << all_origs.len()) - 1;

    let mut killed_at_entry: Vec<u32> = vec![0; num_blocks];
    let mut killed_at_exit: Vec<u32> = vec![0; num_blocks];

    // Iterative fixpoint
    let mut changed = true;
    while changed {
        changed = false;
        for bi in (0..num_blocks).rev() {
            let (start, end) = blocks[bi];
            let last = end - 1;
            let last_inst = &rewritten[last];

            let new_exit = if is_call(last_inst) || last_inst.op == "ret" {
                0u32
            } else if succs[bi].is_empty() {
                0u32
            } else {
                let mut bits = all_bits;
                for &s in &succs[bi] {
                    bits &= killed_at_entry[s];
                }
                bits
            };

            let mut bits = new_exit;
            for i in (start..end).rev() {
                if rewritten[i].op == "conv_store" {
                    if let Some(orig) = rewritten[i].orig_rd {
                        if let Some(&bit) = orig_to_bit.get(&orig) {
                            bits |= bit;
                        }
                    }
                }
                if is_call(&rewritten[i]) {
                    bits = 0;
                }
            }

            if bits != killed_at_entry[bi] {
                killed_at_entry[bi] = bits;
                changed = true;
            }
            killed_at_exit[bi] = new_exit;
        }
    }

    if std::env::var("DSE_BLOCKS").is_ok() {
        let func_addr = rewritten.iter().find(|r| r.addr != 0).map(|r| r.addr).unwrap_or(0);
        for (bi, &(start, end)) in blocks.iter().enumerate() {
            let last = &rewritten[end-1];
            // decode killed_at_entry bits
            let entry_origs: Vec<u8> = orig_to_bit.iter()
                .filter(|(_, &bit)| killed_at_entry[bi] & bit != 0)
                .map(|(&orig, _)| orig)
                .collect();
            let exit_origs: Vec<u8> = orig_to_bit.iter()
                .filter(|(_, &bit)| killed_at_exit[bi] & bit != 0)
                .map(|(&orig, _)| orig)
                .collect();
            eprintln!("    DSE_BLOCKS func=0x{:x} block {} [{}-{}] last={} succs={:?} killed_entry={:?} killed_exit={:?}",
                func_addr, bi, start, end, last.op, succs[bi], entry_origs, exit_origs);
        }
    }

    // Strip dead conv_stores
    let mut stripped = 0u64;
    for (bi, &(start, end)) in blocks.iter().enumerate() {
        let mut will_be_killed = killed_at_exit[bi];
        for i in (start..end).rev() {
            if rewritten[i].op == "conv_store" {
                if let Some(orig) = rewritten[i].orig_rd {
                    if let Some(&bit) = orig_to_bit.get(&orig) {
                        if will_be_killed & bit != 0 {
                            keep[i] = false;
                            stripped += 1;
                            if std::env::var("DSE_DEBUG").is_ok() {
                                let func_addr = rewritten.iter().find(|r| r.addr != 0).map(|r| r.addr).unwrap_or(0);
                                let (bstart, bend) = blocks[bi];
                                let last_op = &rewritten[bend-1].op;
                                eprintln!("    DSE strip: func=0x{:x} idx={} {} (orig x{}, block {} [{}-{}], last_op={}, succs={:?})",
                                    func_addr, i, rewritten[i].specialized, orig, bi, bstart, bend, last_op, succs[bi]);
                            }
                        } else {
                            will_be_killed |= bit;
                        }
                    }
                }
            }
            if is_call(&rewritten[i]) {
                will_be_killed = 0;
            }
        }
    }

    if stripped > 0 {
        eprintln!("    conv_store DSE: stripped {}", stripped);
    }

    keep
}

// ---------------------------------------------------------------------------
// Pass: expand save_context/restore_context into explicit ops
// ---------------------------------------------------------------------------

/// Expands `save_context` and `restore_context` compound instructions into
/// individual `sw_save`, `lw_save`, `sw_save_frame`, `lw_save_frame`, and
/// `addi_save` instructions. Returns the expanded instruction stream and an
/// old→new index mapping for remapping branch targets and segment patches.
pub fn pass_expand_save_restore(
    flat_insts: &[RewrittenInst],
    num_regs: usize,
) -> (Vec<RewrittenInst>, Vec<usize>) {
    let n = flat_insts.len();
    let frame_size = ((num_regs + 1) * 4) as i32;

    // Ops whose imm holds a branch/jump target index (needs remapping).
    let target_imm_ops: HashSet<&str> = [
        "beq", "bne", "blt", "bge", "bltu", "bgeu",
        "jal", "jal_call", "jr_computed",
    ].into();

    // Pass 1: compute old→new mapping
    let mut old_to_new: Vec<usize> = Vec::with_capacity(n);
    let mut new_len = 0usize;
    for inst in flat_insts {
        old_to_new.push(new_len);
        match inst.op.as_str() {
            "save_context" => new_len += num_regs + 2,   // N sw_save + sw_save_frame + addi_save
            "restore_context" => new_len += num_regs + 2, // addi_save + N lw_save + lw_save_frame
            _ => new_len += 1,
        }
    }

    // Pass 2: expand and remap
    let mut result: Vec<RewrittenInst> = Vec::with_capacity(new_len);
    for (_i, inst) in flat_insts.iter().enumerate() {
        match inst.op.as_str() {
            "save_context" => {
                // sw_save rK, imm=K*4 for each GP reg
                for r in 0..num_regs as u8 {
                    result.push(RewrittenInst {
                        addr: 0,
                        op: "sw_save".into(),
                        rd: None,
                        rs1: None,
                        rs2: Some(r),
                        imm: Some((r as i32) * 4),
                        is_move: false,
                        specialized: format!("sw_save.r{}", r),
                        orig_rd: None, orig_rs1: None, orig_rs2: None,
                    });
                }
                // sw_save_frame imm=num_regs*4
                result.push(RewrittenInst {
                    addr: 0,
                    op: "sw_save_frame".into(),
                    rd: None, rs1: None, rs2: None,
                    imm: Some((num_regs as i32) * 4),
                    is_move: false,
                    specialized: "sw_save_frame".into(),
                    orig_rd: None, orig_rs1: None, orig_rs2: None,
                });
                // addi_save imm=frame_size
                result.push(RewrittenInst {
                    addr: 0,
                    op: "addi_save".into(),
                    rd: None, rs1: None, rs2: None,
                    imm: Some(frame_size),
                    is_move: false,
                    specialized: "addi_save".into(),
                    orig_rd: None, orig_rs1: None, orig_rs2: None,
                });
            }
            "restore_context" => {
                // addi_save imm=-frame_size
                result.push(RewrittenInst {
                    addr: 0,
                    op: "addi_save".into(),
                    rd: None, rs1: None, rs2: None,
                    imm: Some(-frame_size),
                    is_move: false,
                    specialized: "addi_save".into(),
                    orig_rd: None, orig_rs1: None, orig_rs2: None,
                });
                // lw_save rK, imm=K*4 for each GP reg
                for r in 0..num_regs as u8 {
                    result.push(RewrittenInst {
                        addr: 0,
                        op: "lw_save".into(),
                        rd: Some(r),
                        rs1: None,
                        rs2: None,
                        imm: Some((r as i32) * 4),
                        is_move: false,
                        specialized: format!("lw_save.r{}", r),
                        orig_rd: None, orig_rs1: None, orig_rs2: None,
                    });
                }
                // lw_save_frame imm=num_regs*4
                result.push(RewrittenInst {
                    addr: 0,
                    op: "lw_save_frame".into(),
                    rd: None, rs1: None, rs2: None,
                    imm: Some((num_regs as i32) * 4),
                    is_move: false,
                    specialized: "lw_save_frame".into(),
                    orig_rd: None, orig_rs1: None, orig_rs2: None,
                });
            }
            _ => {
                let mut new_inst = inst.clone();
                // Remap branch/jump target imm
                if target_imm_ops.contains(inst.op.as_str()) {
                    if let Some(target) = inst.imm {
                        let tidx = target as usize;
                        if tidx < n {
                            new_inst.imm = Some(old_to_new[tidx] as i32);
                        }
                    }
                }
                result.push(new_inst);
            }
        }
    }

    let expanded = result.len() - n;
    if expanded > 0 {
        eprintln!("  Expand save/restore: {} → {} instructions (+{})",
            n, result.len(), expanded);
    }

    (result, old_to_new)
}

// ---------------------------------------------------------------------------
// Pass: dead code elimination on flat instruction stream
// ---------------------------------------------------------------------------

/// Eliminates instructions whose only effect is writing a dead register.
///
/// Uses backward register liveness analysis across a CFG built from the flat
/// instruction stream. Returns the filtered instructions and an old→new index
/// mapping for remapping branch targets and segment patches.
pub fn pass_dce(
    flat_insts: &[RewrittenInst],
    num_regs: usize,
) -> (Vec<RewrittenInst>, Vec<usize>) {
    let n = flat_insts.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let all_regs: u64 = (1u64 << num_regs) - 1;

    // Pure ops: only write rd, no side effects — safe to eliminate if rd dead.
    let pure_ops: HashSet<&str> = [
        "addi", "add", "sub", "mov",
        "mul", "mulh", "mulhsu", "mulhu", "div", "divu", "rem", "remu",
        "sll", "srl", "sra", "slli", "srli", "srai",
        "xor", "or", "and", "xori", "ori", "andi",
        "slt", "sltu", "slti", "sltiu",
        "lui", "auipc",
        "lw", "lb", "lbu", "lh", "lhu",
        "lw_frame",
        // lw_save is NOT pure: it restores registers from the save stack.
        // Even if the restored value appears dead, it may be needed by a
        // future save_context (which saves ALL registers unconditionally).
        "conv_load",
    ].into();

    let branch_ops: HashSet<&str> = ["beq", "bne", "blt", "bge", "bltu", "bgeu"].into();
    let term_ops: HashSet<&str> = [
        "jal", "jal_call", "jalr", "jr_computed", "jr_table_idx", "jr_table",
        "ret", "halt", "ecall",
    ].into();

    // Ops whose imm holds a branch/jump target index (needs remapping).
    let target_imm_ops: HashSet<&str> = [
        "beq", "bne", "blt", "bge", "bltu", "bgeu",
        "jal", "jal_call", "jr_computed",
    ].into();

    // --- Build CFG ---
    let mut block_starts: BTreeSet<usize> = BTreeSet::new();
    block_starts.insert(0);

    for (i, inst) in flat_insts.iter().enumerate() {
        let op = inst.op.as_str();
        if branch_ops.contains(op) {
            let target = inst.imm.unwrap_or(0) as usize;
            if target < n { block_starts.insert(target); }
            if i + 1 < n { block_starts.insert(i + 1); }
        } else if term_ops.contains(op) {
            if matches!(op, "jal" | "jal_call" | "jr_computed") {
                let target = inst.imm.unwrap_or(0) as usize;
                if target < n { block_starts.insert(target); }
            }
            if i + 1 < n { block_starts.insert(i + 1); }
        }
    }

    let starts: Vec<usize> = block_starts.iter().copied().collect();
    let num_blocks = starts.len();
    let mut block_ranges: Vec<(usize, usize)> = Vec::with_capacity(num_blocks);
    let mut start_to_block: HashMap<usize, usize> = HashMap::with_capacity(num_blocks);

    for (bi, &s) in starts.iter().enumerate() {
        let end = if bi + 1 < num_blocks { starts[bi + 1] } else { n };
        start_to_block.insert(s, bi);
        block_ranges.push((s, end));
    }

    // Successor edges
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];
    // Track blocks ending with indirect jumps (conservative: all regs live)
    let mut indirect_blocks: Vec<bool> = vec![false; num_blocks];

    for bi in 0..num_blocks {
        let (start, end) = block_ranges[bi];
        if end <= start { continue; }
        let last = end - 1;
        let op = flat_insts[last].op.as_str();

        if branch_ops.contains(op) {
            let target = flat_insts[last].imm.unwrap_or(0) as usize;
            if let Some(&tbi) = start_to_block.get(&target) {
                successors[bi].push(tbi);
            }
            if let Some(&nbi) = start_to_block.get(&end) {
                successors[bi].push(nbi);
            }
        } else if op == "jal" {
            let target = flat_insts[last].imm.unwrap_or(0) as usize;
            if let Some(&tbi) = start_to_block.get(&target) {
                successors[bi].push(tbi);
            }
        } else if op == "jal_call" {
            // jal_call transfers control to the callee. The callee may depend on
            // registers set up by pre-call shuffles (for conv_load stripping).
            // Mark as indirect so all regs are conservatively live, preventing
            // DCE from eliminating shuffle movs whose destination appears dead
            // in the callee's entry block.
            indirect_blocks[bi] = true;
            let target = flat_insts[last].imm.unwrap_or(0) as usize;
            if let Some(&tbi) = start_to_block.get(&target) {
                successors[bi].push(tbi);
            }
            if let Some(&nbi) = start_to_block.get(&end) {
                successors[bi].push(nbi);
            }
        } else if matches!(op, "jalr" | "jr_table_idx" | "jr_table" | "jr_computed") {
            indirect_blocks[bi] = true;
        } else if matches!(op, "ret" | "halt" | "ecall") {
            // Terminal — no successors
        } else {
            // Fall-through
            if let Some(&nbi) = start_to_block.get(&end) {
                successors[bi].push(nbi);
            }
        }
    }

    // Build predecessor lists for efficient backward iteration
    let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];
    for bi in 0..num_blocks {
        for &sbi in &successors[bi] {
            predecessors[sbi].push(bi);
        }
    }

    // --- Iterative liveness + DCE ---
    let mut dead = vec![false; n];

    loop {
        // Compute liveness (skipping dead instructions)
        let mut live_in: Vec<u64> = vec![0; num_blocks];
        let mut live_out: Vec<u64> = vec![0; num_blocks];

        // Initialize indirect blocks with all regs live out
        for bi in 0..num_blocks {
            if indirect_blocks[bi] {
                live_out[bi] = all_regs;
            }
        }

        // Fixpoint: backward dataflow
        loop {
            let mut changed = false;
            for bi in (0..num_blocks).rev() {
                // live_out = union of live_in of successors, preserving indirect conservative bits
                let mut new_out = if indirect_blocks[bi] { all_regs } else { 0u64 };
                for &sbi in &successors[bi] {
                    new_out |= live_in[sbi];
                }

                let (start, end) = block_ranges[bi];
                let mut live = new_out;

                // Walk backward through block
                for idx in (start..end).rev() {
                    if dead[idx] { continue; }
                    let inst = &flat_insts[idx];
                    let op = inst.op.as_str();

                    let mut d = 0u64;
                    let mut u = 0u64;
                    if let Some(rd) = inst.rd { d |= 1 << rd; }
                    if let Some(rs1) = inst.rs1 { u |= 1 << rs1; }
                    if let Some(rs2) = inst.rs2 { u |= 1 << rs2; }
                    // swap defines and uses both rd and rs1
                    if op == "swap" {
                        if let Some(rd) = inst.rd { u |= 1 << rd; }
                        if let Some(rs1) = inst.rs1 { d |= 1 << rs1; }
                    }

                    live = (live & !d) | u;
                }

                live_out[bi] = new_out;
                if live != live_in[bi] {
                    live_in[bi] = live;
                    changed = true;
                }
            }
            if !changed { break; }
        }

        // Mark dead: combined backward sweep with chain elimination
        let mut new_dead = dead.clone();
        let mut any_new = false;

        for bi in 0..num_blocks {
            let (start, end) = block_ranges[bi];
            let mut live = live_out[bi];

            for idx in (start..end).rev() {
                if new_dead[idx] { continue; }
                let inst = &flat_insts[idx];
                let op = inst.op.as_str();

                let mut defs = 0u64;
                let mut uses = 0u64;
                if let Some(rd) = inst.rd { defs |= 1 << rd; }
                if let Some(rs1) = inst.rs1 { uses |= 1 << rs1; }
                if let Some(rs2) = inst.rs2 { uses |= 1 << rs2; }
                // swap defines and uses both rd and rs1
                if op == "swap" {
                    if let Some(rd) = inst.rd { uses |= 1 << rd; }
                    if let Some(rs1) = inst.rs1 { defs |= 1 << rs1; }
                }

                if pure_ops.contains(op) && defs != 0 && (live & defs) == 0 {
                    // rd is dead — eliminate
                    new_dead[idx] = true;
                    any_new = true;
                    // Don't add uses to live (chain elimination)
                } else {
                    live = (live & !defs) | uses;
                }
            }
        }

        dead = new_dead;
        if !any_new { break; }
    }

    // --- Re-index ---
    let eliminated = dead.iter().filter(|&&d| d).count();
    let mut old_to_new: Vec<usize> = vec![0; n];
    let mut new_idx = 0usize;
    for i in 0..n {
        old_to_new[i] = new_idx;
        if !dead[i] {
            new_idx += 1;
        }
    }

    let mut result: Vec<RewrittenInst> = Vec::with_capacity(new_idx);
    for (i, inst) in flat_insts.iter().enumerate() {
        if dead[i] { continue; }
        let mut new_inst = inst.clone();

        // Remap branch/jump target imm
        if target_imm_ops.contains(inst.op.as_str()) {
            if let Some(target) = inst.imm {
                let tidx = target as usize;
                if tidx < n {
                    new_inst.imm = Some(old_to_new[tidx] as i32);
                }
            }
        }

        result.push(new_inst);
    }

    eprintln!("  DCE: {} → {} instructions ({} eliminated, {:.1}%)",
        n, result.len(), eliminated, 100.0 * eliminated as f64 / n as f64);

    (result, old_to_new)
}
