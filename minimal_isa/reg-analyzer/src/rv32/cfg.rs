//! CFG construction: basic blocks and control-flow edges.
//!
//! Call order:
//!   1. `classify_jalr_x0` — tag indirect jumps as ret / jr_table / jr_computed
//!   2. `build_cfg`        — build basic blocks and edges (reads those tags)

use std::collections::{HashMap, HashSet, BTreeSet};
use super::decode::DecodedInst;

// ---------------------------------------------------------------------------
// CFG: basic blocks with successor/predecessor edges
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id: usize,
    pub start: usize,       // instruction index (inclusive)
    pub end: usize,         // instruction index (exclusive)
    pub start_addr: u32,
    pub succs: Vec<usize>,  // successor block ids
    pub preds: Vec<usize>,  // predecessor block ids
}

pub(super) fn is_branch(op: &str) -> bool {
    matches!(op, "beq"|"bne"|"blt"|"bge"|"bltu"|"bgeu")
}

fn _is_unconditional_jump(d: &DecodedInst) -> bool {
    // jal x0, offset  (j pseudo)
    (d.op == "jal" && d.rd == Some(0))
    // jalr x0, rs, offset  (jr/ret pseudo)
    || (d.op == "jalr" && d.rd == Some(0))
    // Classified indirect jumps
    || d.op == "ret" || d.op == "jr_table" || d.op == "jr_computed"
}

/// Returns true if `d` is a function call (saves a return address).
pub fn is_call(d: &DecodedInst) -> bool {
    // jal rd, offset or jalr rd, rs, offset with rd != x0
    (d.op == "jal" || d.op == "jalr") && d.rd.map_or(false, |r| r != 0)
}

fn is_block_terminator(d: &DecodedInst) -> bool {
    is_branch(&d.op) || d.op == "jal" || d.op == "jalr"
    || d.op == "ret" || d.op == "jr_table" || d.op == "jr_computed"
}

/// Classify `jalr x0` instructions in the decoded stream (must run before `build_cfg`).
///
/// - `jalr x0, x1, 0` → `ret` (function return)
/// - `jalr x0, xN, 0` with lw-from-table pattern → `jr_table` (jump table dispatch)
/// - `jalr x0, xN, offset` with auipc pattern → `jr_computed` (computed intra-function jump)
///
/// Returns:
/// - map of instruction address → jump table target addresses (for `jr_table`)
/// - map of instruction address → (table_base_addr, num_entries) for memory patching
pub fn classify_jalr_x0(
    result: &mut Vec<DecodedInst>,
    elf_data: &[u8],
    elf_funcs: &[(u32, u32, String)],
) -> (HashMap<u32, Vec<u32>>, HashMap<u32, (u32, usize)>) {
    // Build ELF memory map for reading jump table entries
    let seg_data: Vec<(u32, Vec<u8>)> = {
        use object::elf::*;
        use object::read::elf::FileHeader as _;
        use object::Endianness;
        let mut segments = Vec::new();
        if let Ok(elf) = FileHeader32::<Endianness>::parse(elf_data) {
            if let Ok(endian) = elf.endian() {
                if let Ok(segs) = elf.program_headers(endian, elf_data) {
                    for seg in segs {
                        if seg.p_type.get(endian) != PT_LOAD { continue; }
                        let vaddr = seg.p_vaddr.get(endian);
                        let filesz = seg.p_filesz.get(endian) as usize;
                        let offset = seg.p_offset.get(endian) as usize;
                        if filesz > 0 && offset + filesz <= elf_data.len() {
                            segments.push((vaddr, elf_data[offset..offset + filesz].to_vec()));
                        }
                    }
                }
            }
        }
        segments
    };

    let read_u32_mem = |addr: u32| -> Option<u32> {
        for (base, data) in &seg_data {
            if addr >= *base && (addr - base) as usize + 4 <= data.len() {
                let off = (addr - base) as usize;
                return Some(u32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]));
            }
        }
        None
    };

    // Build addr set for validating jump table entries
    let result_addrs: HashSet<u32> = result.iter().map(|inst| inst.addr).collect();

    let mut jump_table_targets: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut jump_table_bases: HashMap<u32, (u32, usize)> = HashMap::new();
    let mut _num_ret = 0;
    let mut _num_jr_table = 0;
    let mut _num_jr_computed = 0;

    for i in 0..result.len() {
        if result[i].op != "jalr" || result[i].rd != Some(0) { continue; }

        // Case 1: ret (jalr x0, x1, 0)
        if result[i].rs1 == Some(1) && result[i].imm == Some(0) {
            result[i].op = "ret".to_string();
            _num_ret += 1;
            continue;
        }

        let target_reg = result[i].rs1.unwrap();
        let jalr_offset = result[i].imm.unwrap_or(0);

        // Scan backward to find what sets target_reg
        let mut classified = false;
        for j in (0..i).rev() {
            if result[j].rd != Some(target_reg) { continue; }

            if result[j].op == "auipc" {
                // Computed target = auipc_addr + auipc_imm + jalr_offset
                let target = result[j].addr
                    .wrapping_add(result[j].imm.unwrap_or(0) as u32)
                    .wrapping_add(jalr_offset as u32);
                // Only classify as jr_computed if target is within the same function
                // (cross-function targets are tail calls, handled by the jalr path)
                let inst_addr = result[i].addr;
                let containing = elf_funcs.iter().find(|(faddr, fsize, _)| {
                    inst_addr >= *faddr && inst_addr < faddr + fsize
                });
                let is_intra_func = containing.map_or(false, |(faddr, fsize, _)| {
                    target >= *faddr && target < faddr + fsize
                });
                if is_intra_func {
                    result[i].op = "jr_computed".to_string();
                    result[i].imm = Some(target as i32);
                    _num_jr_computed += 1;
                    classified = true;
                }
                // else: leave as jalr for tail-call handling
            } else if result[j].op == "lw" && jalr_offset == 0 {
                // Potential jump table: lw target_reg, offset(base_reg)
                // Trace backward through add → lui/addi to find table base
                let load_base_reg = result[j].rs1.unwrap();
                let mut table_base = None;
                for k in (0..j).rev() {
                    if result[k].rd != Some(load_base_reg) { continue; }
                    if result[k].op == "add" {
                        let r1 = result[k].rs1.unwrap();
                        let r2 = result[k].rs2.unwrap();
                        for base_cand in [r1, r2] {
                            for l in (0..k).rev() {
                                if result[l].rd != Some(base_cand) { continue; }
                                if result[l].op == "addi" {
                                    let addi_src = result[l].rs1.unwrap_or(0);
                                    let addi_imm = result[l].imm.unwrap_or(0);
                                    for m in (0..l).rev() {
                                        if result[m].rd != Some(addi_src) { continue; }
                                        if result[m].op == "lui" {
                                            let upper = result[m].imm.unwrap_or(0) as u32;
                                            table_base = Some(upper.wrapping_add(addi_imm as u32));
                                        } else if result[m].op == "auipc" {
                                            let upper = result[m].addr.wrapping_add(result[m].imm.unwrap_or(0) as u32);
                                            table_base = Some(upper.wrapping_add(addi_imm as u32));
                                        }
                                        break;
                                    }
                                }
                                break;
                            }
                            if table_base.is_some() { break; }
                        }
                    }
                    break;
                }

                if let Some(base) = table_base {
                    let mut offset = 0u32;
                    let mut targets = Vec::new();
                    loop {
                        if let Some(target_addr) = read_u32_mem(base.wrapping_add(offset)) {
                            if result_addrs.contains(&target_addr) {
                                targets.push(target_addr);
                                offset += 4;
                                if offset > 256 { break; }
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                    if !targets.is_empty() {
                        result[i].op = "jr_table".to_string();
                        jump_table_targets.insert(result[i].addr, targets.clone());
                        jump_table_bases.insert(result[i].addr, (base, targets.len()));
                        _num_jr_table += 1;
                        classified = true;
                    }
                }
            }
            break; // Stop at first write to target_reg
        }

        if !classified {
            // Unclassified indirect jumps are handled as jr_computed at runtime.
        }
    }

    // eprintln!("  Classified jalr x0: {} ret, {} jr_table, {} jr_computed",
    //     num_ret, num_jr_table, num_jr_computed);

    (jump_table_targets, jump_table_bases)
}

pub fn build_cfg(decoded: &[DecodedInst], jump_table_targets: &HashMap<u32, Vec<u32>>) -> Vec<BasicBlock> {
    if decoded.is_empty() {
        return Vec::new();
    }

    // Collect branch targets
    let mut branch_targets: HashSet<u32> = HashSet::new();
    for d in decoded {
        if is_branch(&d.op) || d.op == "jal" {
            if let Some(imm) = d.imm {
                let target = (d.addr as i64 + imm as i64) as u32;
                branch_targets.insert(target);
            }
        }
    }

    // Block starts: entry, branch targets, instruction after terminator
    let mut block_starts: BTreeSet<usize> = BTreeSet::new();
    block_starts.insert(0);
    let addr_to_idx: HashMap<u32, usize> = decoded.iter().enumerate()
        .map(|(i, d)| (d.addr, i)).collect();

    for &target in &branch_targets {
        if let Some(&idx) = addr_to_idx.get(&target) {
            block_starts.insert(idx);
        }
    }
    for (i, d) in decoded.iter().enumerate() {
        if is_block_terminator(d) && i + 1 < decoded.len() {
            block_starts.insert(i + 1);
        }
    }

    // Create blocks
    let starts: Vec<usize> = block_starts.into_iter().collect();
    let mut blocks: Vec<BasicBlock> = Vec::new();
    for w in 0..starts.len() {
        let start = starts[w];
        let end = if w + 1 < starts.len() { starts[w + 1] } else { decoded.len() };
        blocks.push(BasicBlock {
            id: w,
            start,
            end,
            start_addr: decoded[start].addr,
            succs: Vec::new(),
            preds: Vec::new(),
        });
    }

    // Map address → block id for branch targets
    let addr_to_block: HashMap<u32, usize> = blocks.iter()
        .map(|b| (b.start_addr, b.id)).collect();

    // Build edges
    for b in 0..blocks.len() {
        let last = &decoded[blocks[b].end - 1];

        if is_branch(&last.op) {
            // Conditional: taken + fallthrough
            if let Some(imm) = last.imm {
                let target = (last.addr as i64 + imm as i64) as u32;
                if let Some(&tid) = addr_to_block.get(&target) {
                    blocks[b].succs.push(tid);
                }
            }
            if b + 1 < blocks.len() {
                blocks[b].succs.push(b + 1);
            }
        } else if last.op == "jal" {
            if last.rd == Some(0) {
                // Unconditional jump (j): single target
                if let Some(imm) = last.imm {
                    let target = (last.addr as i64 + imm as i64) as u32;
                    if let Some(&tid) = addr_to_block.get(&target) {
                        blocks[b].succs.push(tid);
                    }
                }
            } else {
                // Call (jal ra, ...): falls through after return
                if b + 1 < blocks.len() {
                    blocks[b].succs.push(b + 1);
                }
            }
        } else if last.op == "ret" {
            // Return: no successors
        } else if last.op == "jr_computed" {
            // Computed intra-function jump: single target from imm
            let target = last.imm.unwrap() as u32;
            if let Some(&tid) = addr_to_block.get(&target) {
                blocks[b].succs.push(tid);
            }
        } else if last.op == "jr_table" {
            // Jump table dispatch: edges from resolved targets
            if let Some(targets) = jump_table_targets.get(&last.addr) {
                for &target in targets {
                    if let Some(&tid) = addr_to_block.get(&target) {
                        if !blocks[b].succs.contains(&tid) {
                            blocks[b].succs.push(tid);
                        }
                    }
                }
            }
        } else if last.op == "jalr" {
            if last.rd == Some(0) {
                // Unclassified indirect jump: no successors
            } else {
                // Indirect call: falls through
                if b + 1 < blocks.len() {
                    blocks[b].succs.push(b + 1);
                }
            }
        } else {
            // Non-terminator: falls through
            if b + 1 < blocks.len() {
                blocks[b].succs.push(b + 1);
            }
        }
    }

    // Build predecessor lists
    for b in 0..blocks.len() {
        let succs = blocks[b].succs.clone();
        for s in succs {
            blocks[s].preds.push(b);
        }
    }

    blocks
}
