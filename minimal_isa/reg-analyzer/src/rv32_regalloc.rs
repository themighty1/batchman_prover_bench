//! RV32IM regalloc2 integration: liveness analysis, Function impl, rewriter.

use std::collections::{HashMap, HashSet};
use regalloc2 as ra2;
use ra2::{Operand, OperandConstraint, OperandKind, OperandPos};
use crate::rv32::{BasicBlock, DecodedInst};

// ---------------------------------------------------------------------------
// Liveness analysis
// ---------------------------------------------------------------------------

/// Per-block register usage: which physical registers are used-before-def and which are def'd.
struct BlockRegInfo {
    use_before_def: HashSet<u8>,
    defs: HashSet<u8>,
}

fn block_reg_info(decoded: &[DecodedInst], block: &BasicBlock) -> BlockRegInfo {
    let mut use_before_def = HashSet::new();
    let mut defs = HashSet::new();
    for i in block.start..block.end {
        let d = &decoded[i];
        for r in [d.rs1, d.rs2].into_iter().flatten() {
            if r != 0 && !defs.contains(&r) {
                use_before_def.insert(r);
            }
        }
        if let Some(r) = d.rd {
            if r != 0 { defs.insert(r); }
        }
        // Note: we don't model call clobbers here — for ISA counting,
        // we just let values flow through calls naturally.
    }
    BlockRegInfo { use_before_def, defs }
}

/// Iterative dataflow: compute live-in and live-out register sets per block.
pub fn compute_liveness(
    decoded: &[DecodedInst],
    blocks: &[BasicBlock],
) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let n = blocks.len();
    let infos: Vec<BlockRegInfo> = blocks.iter().map(|b| block_reg_info(decoded, b)).collect();

    let mut live_in: Vec<HashSet<u8>> = infos.iter().map(|i| i.use_before_def.clone()).collect();
    let mut live_out: Vec<HashSet<u8>> = vec![HashSet::new(); n];

    let mut changed = true;
    while changed {
        changed = false;
        for bi in (0..n).rev() {
            let mut new_out = HashSet::new();
            for &si in &blocks[bi].succs {
                for &r in &live_in[si] { new_out.insert(r); }
            }
            let mut new_in = infos[bi].use_before_def.clone();
            for &r in &new_out {
                if !infos[bi].defs.contains(&r) { new_in.insert(r); }
            }
            if new_in != live_in[bi] || new_out != live_out[bi] {
                changed = true;
                live_in[bi] = new_in;
                live_out[bi] = new_out;
            }
        }
    }

    let li: Vec<Vec<u8>> = live_in.into_iter().map(|s| { let mut v: Vec<u8> = s.into_iter().collect(); v.sort(); v }).collect();
    let lo: Vec<Vec<u8>> = live_out.into_iter().map(|s| { let mut v: Vec<u8> = s.into_iter().collect(); v.sort(); v }).collect();
    (li, lo)
}

// ---------------------------------------------------------------------------
// Function boundary detection
// ---------------------------------------------------------------------------

/// Find connected components from the CFG: each component is a "function".
/// Returns a list of (entry_block_id, vec_of_block_ids) per function.
pub fn find_functions(blocks: &[BasicBlock]) -> Vec<(usize, Vec<usize>)> {
    let n = blocks.len();
    let mut visited = vec![false; n];
    let mut functions = Vec::new();

    for entry in 0..n {
        if visited[entry] { continue; }
        // BFS from this entry
        let mut queue = vec![entry];
        let mut component = Vec::new();
        visited[entry] = true;
        while let Some(bi) = queue.pop() {
            component.push(bi);
            for &si in &blocks[bi].succs {
                if !visited[si] {
                    visited[si] = true;
                    queue.push(si);
                }
            }
            for &pi in &blocks[bi].preds {
                if !visited[pi] {
                    visited[pi] = true;
                    queue.push(pi);
                }
            }
        }
        component.sort();
        functions.push((entry, component));
    }
    functions
}

/// Find functions using ELF symbol boundaries instead of CFG connectivity.
/// Each ELF function symbol defines a range [addr, addr+size).
/// Blocks whose start_addr falls within a function's range belong to that function.
pub fn find_functions_from_symbols(
    blocks: &[BasicBlock],
    elf_funcs: &[(u32, u32)],  // (addr, size) sorted by addr
) -> Vec<(usize, Vec<usize>)> {
    // Map each block to its containing ELF function
    let mut functions: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut block_assigned = vec![false; blocks.len()];

    for &(func_addr, func_size) in elf_funcs {
        let func_end = func_addr + func_size;

        // Find blocks whose start_addr falls within [func_addr, func_end)
        let mut component = Vec::new();
        let mut entry_block = None;

        for (bi, block) in blocks.iter().enumerate() {
            if block_assigned[bi] { continue; }
            if block.start_addr >= func_addr && block.start_addr < func_end {
                if entry_block.is_none() || block.start_addr == func_addr {
                    entry_block = Some(bi);
                }
                component.push(bi);
                block_assigned[bi] = true;
            }
        }

        if !component.is_empty() {
            component.sort();
            let entry = entry_block.unwrap_or(component[0]);
            functions.push((entry, component));
        }
    }

    // Handle any unassigned blocks (shouldn't happen with good symbol tables)
    for bi in 0..blocks.len() {
        if !block_assigned[bi] {
            functions.push((bi, vec![bi]));
        }
    }

    functions
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Caller-saved registers: ra, t0-t2, a0-a7, t3-t6
const _CALLER_SAVED: [u8; 16] = [1, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 28, 29, 30, 31];

fn is_call_inst(d: &DecodedInst) -> bool {
    (d.op == "jal" || d.op == "jalr") && d.rd.map_or(false, |r| r != 0)
}

fn is_branch_inst(d: &DecodedInst) -> bool {
    matches!(d.op.as_str(), "beq"|"bne"|"blt"|"bge"|"bltu"|"bgeu")
}

fn is_return_inst(d: &DecodedInst) -> bool {
    // "ret" = classified return, "jalr" rd=0 = unclassified tail call (also a function exit)
    d.op == "ret" || (d.op == "jalr" && d.rd == Some(0))
}

fn _is_terminator(d: &DecodedInst) -> bool {
    is_branch_inst(d) || d.op == "jal" || d.op == "jalr"
    || d.op == "ret" || d.op == "jr_table" || d.op == "jr_computed"
}

// ---------------------------------------------------------------------------
// Rv32Ra2Func — regalloc2 Function implementation
// ---------------------------------------------------------------------------

/// Info about each instruction in the flat ra2 array, for rewriting later.
#[derive(Debug, Clone)]
pub struct Ra2InstInfo {
    pub orig_block: usize,   // original block index (in the function's local numbering)
    pub orig_idx: usize,     // index into decoded[] (global)
    pub is_synthetic: bool,  // synthetic fallthrough branch
}

pub struct Rv32Ra2Func {
    operands: Vec<Vec<Operand>>,
    is_branch_flag: Vec<bool>,
    is_ret_flag: Vec<bool>,
    _is_call_flag: Vec<bool>,
    _num_regs: u32,

    block_inst_ranges: Vec<(usize, usize)>,
    block_succs: Vec<Vec<ra2::Block>>,
    block_preds: Vec<Vec<ra2::Block>>,
    block_params_storage: Vec<Vec<ra2::VReg>>,

    // branch_blockparams[block_id][succ_idx] = vregs to pass
    branch_args: Vec<Vec<Vec<ra2::VReg>>>,

    pub num_vregs: usize,
    pub inst_info: Vec<Ra2InstInfo>,
    pub entry_block_id: usize,

    // For rewriting: map vreg back to (block_local_id, preg)
    // and per-block entry/exit vreg mappings
    pub vreg_to_preg_origin: HashMap<usize, u8>,  // ra2 vreg index → original preg
    pub block_exit_vregs: Vec<[Option<usize>; 32]>,  // per block: preg → ra2 vreg index

    // Trampoline info for critical edge splitting: (src_block, succ_idx_in_src, target_block)
    // Block indices are in RPO-remapped space.
    pub trampoline_blocks: Vec<(usize, usize, usize)>,
    // Per-block: which trampoline index is this block? None for real/wrapper blocks.
    pub block_trampoline_idx: Vec<Option<usize>>,
    // local block index → start address of original block (for patching branches)
    pub local_block_addr: Vec<u32>,
}

impl Rv32Ra2Func {
    /// Build a regalloc2 Function for a single function (subset of blocks).
    ///
    /// `func_blocks` is the list of global block indices in this function.
    /// `entry` is the global block index of the entry block.
    pub fn build(
        decoded: &[DecodedInst],
        all_blocks: &[BasicBlock],
        func_blocks: &[usize],
        entry: usize,
        live_in: &[Vec<u8>],
        _live_out: &[Vec<u8>],
        num_regs: u32,
    ) -> Option<Self> {
        if func_blocks.is_empty() { return None; }

        // Map global block id → local block id
        let mut global_to_local: HashMap<usize, usize> = HashMap::new();
        // Entry block must be local block 0
        let entry_local = 0usize;
        global_to_local.insert(entry, entry_local);
        let mut local_idx = 1;
        for &gbi in func_blocks {
            if gbi == entry { continue; }
            global_to_local.insert(gbi, local_idx);
            local_idx += 1;
        }
        let num_local_blocks = global_to_local.len();

        // Order: local 0 = entry, rest in order
        let mut local_to_global: Vec<usize> = vec![0; num_local_blocks];
        for (&gbi, &lbi) in &global_to_local {
            local_to_global[lbi] = gbi;
        }

        // Build succs/preds in local numbering
        let mut local_succs: Vec<Vec<usize>> = vec![Vec::new(); num_local_blocks];
        let mut local_preds: Vec<Vec<usize>> = vec![Vec::new(); num_local_blocks];
        for lbi in 0..num_local_blocks {
            let gbi = local_to_global[lbi];
            for &gs in &all_blocks[gbi].succs {
                if let Some(&ls) = global_to_local.get(&gs) {
                    local_succs[lbi].push(ls);
                }
            }
        }
        for lbi in 0..num_local_blocks {
            let succs = local_succs[lbi].clone();
            for ls in succs {
                local_preds[ls].push(lbi);
            }
        }

        // Split critical edges
        let mut trampoline_blocks: Vec<(usize, usize, usize)> = Vec::new(); // (src, succ_idx_in_src, target)
        {
            let mut to_split = Vec::new();
            for lbi in 0..num_local_blocks {
                if local_succs[lbi].len() <= 1 { continue; }
                for (si, &target) in local_succs[lbi].iter().enumerate() {
                    if local_preds[target].len() > 1 {
                        to_split.push((lbi, si, target));
                    }
                }
            }
            for (src, si, target) in to_split {
                let tramp = num_local_blocks + trampoline_blocks.len();
                trampoline_blocks.push((src, si, target));

                // Fix edges
                local_succs[src][si] = tramp;
                if let Some(pos) = local_preds[target].iter().position(|&p| p == src) {
                    local_preds[target][pos] = tramp;
                }

                // Trampoline block
                local_succs.push(vec![target]);
                local_preds.push(vec![src]);
                local_to_global.push(usize::MAX); // no original block
            }
        }
        // If entry block (local 0) has predecessors (loop backedges), we need a
        // synthetic wrapper entry block, because regalloc2 forbids block params on entry.
        // Insert the wrapper as a new block that branches to the original entry.
        let entry_has_preds = !local_preds[entry_local].is_empty();
        let real_entry: usize;  // the block we return from entry_block()

        if entry_has_preds {
            // Create wrapper: new block at end of lists
            let wrapper = local_succs.len();
            local_succs.push(vec![entry_local]); // wrapper → old entry
            local_preds.push(vec![]);            // wrapper has no preds
            local_preds[entry_local].push(wrapper); // old entry gains wrapper as pred
            local_to_global.push(usize::MAX);
            real_entry = wrapper;
        } else {
            real_entry = entry_local;
        }

        let total_blocks = local_succs.len();

        // --- Compute effective live-in per block ---
        // Extend global live_in to cover pass-through needs across the entire function
        // (including trampoline blocks from critical edge splitting).
        let mut effective_live_in: Vec<HashSet<u8>> = vec![HashSet::new(); total_blocks];
        for lbi in 0..num_local_blocks {
            let gbi = local_to_global[lbi];
            for &r in &live_in[gbi] {
                effective_live_in[lbi].insert(r);
            }
        }
        // Trampoline blocks initially inherit from their target's live-in
        for (ti, &(_src, _si, target)) in trampoline_blocks.iter().enumerate() {
            let lbi = num_local_blocks + ti;
            if target < num_local_blocks {
                effective_live_in[lbi] = effective_live_in[target].clone();
            }
        }
        // Iterate: propagate successor needs back through non-defining blocks
        {
            let block_defs: Vec<HashSet<u8>> = (0..total_blocks).map(|lbi| {
                if lbi < num_local_blocks {
                    let gbi = local_to_global[lbi];
                    let block = &all_blocks[gbi];
                    let mut defs = HashSet::new();
                    for i in block.start..block.end {
                        if let Some(r) = decoded[i].rd {
                            if r != 0 { defs.insert(r); }
                        }
                    }
                    defs
                } else {
                    HashSet::new() // trampolines don't define anything
                }
            }).collect();

            let mut changed = true;
            while changed {
                changed = false;
                for lbi in (0..total_blocks).rev() {
                    let mut needed = HashSet::new();
                    for &ls in &local_succs[lbi] {
                        for &r in &effective_live_in[ls] {
                            needed.insert(r);
                        }
                    }
                    for r in needed {
                        if !block_defs[lbi].contains(&r) && !effective_live_in[lbi].contains(&r) {
                            effective_live_in[lbi].insert(r);
                            changed = true;
                        }
                    }
                }
            }
        }

        // --- VReg allocation ---
        let mut next_vreg = 0usize;
        let mut vreg_to_preg_origin: HashMap<usize, u8> = HashMap::new();

        let alloc_vreg = |next: &mut usize, preg: u8, map: &mut HashMap<usize, u8>| -> usize {
            let v = *next;
            *next += 1;
            map.insert(v, preg);
            v
        };

        // Per-block: entry vregs (block params) and exit vregs
        let mut block_entry_vregs: Vec<[Option<usize>; 32]> = vec![[None; 32]; total_blocks];
        let mut block_exit_vregs: Vec<[Option<usize>; 32]> = vec![[None; 32]; total_blocks];
        let mut block_params_storage: Vec<Vec<ra2::VReg>> = vec![Vec::new(); total_blocks];

        // Per-block lifted instructions + operands
        let mut flat_operands: Vec<Vec<Operand>> = Vec::new();
        let mut flat_is_branch: Vec<bool> = Vec::new();
        let mut flat_is_ret: Vec<bool> = Vec::new();
        let mut flat_is_call: Vec<bool> = Vec::new();
        let mut flat_info: Vec<Ra2InstInfo> = Vec::new();
        let mut block_inst_ranges: Vec<(usize, usize)> = Vec::new();

        // Helper: push a synthetic instruction
        let push_synthetic = |ops: Vec<Operand>, is_br: bool, is_r: bool, lbi: usize,
                              flat_ops: &mut Vec<Vec<Operand>>,
                              flat_br: &mut Vec<bool>, flat_ret: &mut Vec<bool>,
                              flat_call: &mut Vec<bool>,
                              flat_inf: &mut Vec<Ra2InstInfo>| {
            flat_ops.push(ops);
            flat_br.push(is_br);
            flat_ret.push(is_r);
            flat_call.push(false);
            flat_inf.push(Ra2InstInfo {
                orig_block: lbi,
                orig_idx: usize::MAX,
                is_synthetic: true,
            });
        };

        // Process real blocks
        for lbi in 0..num_local_blocks {
            let gbi = local_to_global[lbi];
            let block = &all_blocks[gbi];
            let li: Vec<u8> = {
                let mut v: Vec<u8> = effective_live_in[lbi].iter().copied().collect();
                v.sort();
                v
            };

            let is_ra2_entry = lbi == real_entry;

            if !is_ra2_entry {
                // All non-entry blocks get block params for live-in regs
                // (including the original entry block when we have a wrapper)
                for &r in &li {
                    let v = alloc_vreg(&mut next_vreg, r, &mut vreg_to_preg_origin);
                    block_entry_vregs[lbi][r as usize] = Some(v);
                    block_params_storage[lbi].push(ra2::VReg::new(v, ra2::RegClass::Int));
                }
            } else {
                // RA2 entry block: allocate vregs but NO block params.
                for &r in &li {
                    let v = alloc_vreg(&mut next_vreg, r, &mut vreg_to_preg_origin);
                    block_entry_vregs[lbi][r as usize] = Some(v);
                }
            }

            // Current vreg mapping: preg → vreg
            let mut current = block_entry_vregs[lbi];

            let inst_start = flat_operands.len();

            // RA2 entry block: one synthetic def per live-in vreg
            if is_ra2_entry {
                for &r in &li {
                    if let Some(v) = block_entry_vregs[lbi][r as usize] {
                        let ops = vec![Operand::new(
                            ra2::VReg::new(v, ra2::RegClass::Int),
                            OperandConstraint::Reg,
                            OperandKind::Def,
                            OperandPos::Late,
                        )];
                        push_synthetic(ops, false, false, lbi,
                            &mut flat_operands, &mut flat_is_branch, &mut flat_is_ret, &mut flat_is_call, &mut flat_info);
                    }
                }
            }

            // Debug: trace block boundaries
            if std::env::var("DUMP_BUILD").is_ok() {
                let first_addr = decoded[block.start].addr;
                let last_addr = decoded[block.end - 1].addr;
                if first_addr <= 0x011660 && last_addr >= 0x011640 {
                    eprintln!("  BUILD BLOCK lbi={} gbi={} addrs=0x{:06x}..0x{:06x} entry_vregs[8]=v{:?} entry_vregs[11]=v{:?}",
                        lbi, gbi, first_addr, last_addr,
                        block_entry_vregs[lbi][8], block_entry_vregs[lbi][11]);
                }
            }

            for i in block.start..block.end {
                let d = &decoded[i];
                let is_ret = is_return_inst(d);
                // A "pure branch" is a control-flow instruction that doesn't define a register:
                // conditional branches (beq etc) or unconditional jump (jal x0) or return or indirect jump
                let is_pure_branch = is_branch_inst(d) || (d.op == "jal" && d.rd == Some(0))
                    || d.op == "jr_table" || d.op == "jr_computed";

                // Build operands
                let mut ops = Vec::new();

                // Destination (def)
                let rd_vreg = d.rd.and_then(|r| {
                    if r == 0 { return None; } // x0 writes are sinks
                    let v = alloc_vreg(&mut next_vreg, r, &mut vreg_to_preg_origin);
                    ops.push(Operand::new(
                        ra2::VReg::new(v, ra2::RegClass::Int),
                        OperandConstraint::Reg,
                        OperandKind::Def,
                        OperandPos::Late,
                    ));
                    Some(v)
                });

                // For calls: also define x10 (a0) as return value register.
                // This ensures regalloc2 knows x10 is redefined by the call,
                // preventing stale pre-call values from being used as the return value.
                let is_call = is_call_inst(d);
                let mut x10_ret_vreg: Option<usize> = None;
                if is_call && d.rd != Some(10) {
                    let v = alloc_vreg(&mut next_vreg, 10, &mut vreg_to_preg_origin);
                    ops.push(Operand::new(
                        ra2::VReg::new(v, ra2::RegClass::Int),
                        OperandConstraint::Reg,
                        OperandKind::Def,
                        OperandPos::Late,
                    ));
                    x10_ret_vreg = Some(v);
                }

                // Sources (use)
                for r in [d.rs1, d.rs2].into_iter().flatten() {
                    if let Some(v) = current[r as usize] {
                        ops.push(Operand::new(
                            ra2::VReg::new(v, ra2::RegClass::Int),
                            OperandConstraint::Reg,
                            OperandKind::Use,
                            OperandPos::Early,
                        ));
                    } else if r != 0 {
                        panic!(
                            "BUG: Use of x{} at 0x{:x} (op={}) has no vreg in current[] \
                             (block={}, inst_idx={}). Missing live-in or def?",
                            r, d.addr, d.op, lbi, i
                        );
                    }
                }

                // Calls: mark as non-branch (they fall through, and may define rd)
                // Pure branches/returns: mark as branch/ret
                flat_operands.push(ops);
                flat_is_branch.push(is_pure_branch || is_ret);
                flat_is_ret.push(is_ret);
                flat_is_call.push(is_call);
                flat_info.push(Ra2InstInfo {
                    orig_block: lbi,
                    orig_idx: i,
                    is_synthetic: false,
                });

                // Debug: trace vreg assignments in build()
                if std::env::var("DUMP_BUILD").is_ok() {
                    if d.addr >= 0x011640 && d.addr <= 0x011660 {
                        let use_vregs: Vec<_> = [d.rs1, d.rs2].iter()
                            .filter_map(|r| r.map(|r| (r, current[r as usize])))
                            .collect();
                        eprintln!("  BUILD block={} 0x{:06x} op={} rd={:?}→v{:?} uses={:?} current[8]=v{:?} current[11]=v{:?}",
                            lbi, d.addr, d.op, d.rd, rd_vreg,
                            use_vregs, current[8], current[11]);
                    }
                }

                // Update current vreg after def
                if let (Some(r), Some(v)) = (d.rd, rd_vreg) {
                    if r != 0 { current[r as usize] = Some(v); }
                }
                if let Some(v) = x10_ret_vreg {
                    current[10] = Some(v);
                }
            }

            // Every block must end with is_branch=true or is_ret=true.
            // If the last real instruction wasn't a pure branch or return, add synthetic branch.
            let last_d = &decoded[block.end - 1];
            let last_is_pure_branch = is_branch_inst(last_d) || (last_d.op == "jal" && last_d.rd == Some(0))
                || last_d.op == "jr_table" || last_d.op == "jr_computed";
            let last_is_ret = is_return_inst(last_d);
            if !last_is_pure_branch && !last_is_ret {
                push_synthetic(vec![], true, false, lbi,
                    &mut flat_operands, &mut flat_is_branch, &mut flat_is_ret, &mut flat_is_call, &mut flat_info);
            }

            let inst_end = flat_operands.len();
            block_inst_ranges.push((inst_start, inst_end));

            block_exit_vregs[lbi] = current;
        }

        // Process trampoline blocks (empty, just forward block params)
        for (ti, &(_src, _si, target)) in trampoline_blocks.iter().enumerate() {
            let lbi = num_local_blocks + ti;

            // Trampoline needs same block params as target block
            // Use target's block_params_storage to determine which regs
            let target_regs: Vec<u8> = block_params_storage[target].iter()
                .map(|v| vreg_to_preg_origin[&v.vreg()])
                .collect();

            for &r in &target_regs {
                let v = alloc_vreg(&mut next_vreg, r, &mut vreg_to_preg_origin);
                block_entry_vregs[lbi][r as usize] = Some(v);
                block_params_storage[lbi].push(ra2::VReg::new(v, ra2::RegClass::Int));
            }

            // Copy entry to exit (pass-through)
            block_exit_vregs[lbi] = block_entry_vregs[lbi];

            // Two synthetic instructions: nop (for S→T edge moves) + branch (for T→B edge moves).
            // This gives regalloc2 two ProgPoints so it can place S→T and T→B moves separately.
            let inst_start = flat_operands.len();
            push_synthetic(vec![], false, false, lbi,
                &mut flat_operands, &mut flat_is_branch, &mut flat_is_ret, &mut flat_is_call, &mut flat_info);
            push_synthetic(vec![], true, false, lbi,
                &mut flat_operands, &mut flat_is_branch, &mut flat_is_ret, &mut flat_is_call, &mut flat_info);
            let inst_end = flat_operands.len();
            block_inst_ranges.push((inst_start, inst_end));
        }

        // Process wrapper entry block (if created)
        if entry_has_preds {
            let lbi = real_entry;
            // Wrapper needs to define all live-in regs of the original entry block,
            // then branch to the original entry.
            let orig_entry_li: Vec<u8> = {
                let mut v: Vec<u8> = effective_live_in[entry_local].iter().copied().collect();
                v.sort();
                v
            };

            let inst_start = flat_operands.len();

            // One synthetic def per live-in vreg
            for &r in &orig_entry_li {
                let v = alloc_vreg(&mut next_vreg, r, &mut vreg_to_preg_origin);
                block_entry_vregs[lbi][r as usize] = Some(v);
                let ops = vec![Operand::new(
                    ra2::VReg::new(v, ra2::RegClass::Int),
                    OperandConstraint::Reg,
                    OperandKind::Def,
                    OperandPos::Late,
                )];
                push_synthetic(ops, false, false, lbi,
                    &mut flat_operands, &mut flat_is_branch, &mut flat_is_ret, &mut flat_is_call, &mut flat_info);
            }
            // Synthetic branch to original entry
            push_synthetic(vec![], true, false, lbi,
                &mut flat_operands, &mut flat_is_branch, &mut flat_is_ret, &mut flat_is_call, &mut flat_info);

            let inst_end = flat_operands.len();
            block_inst_ranges.push((inst_start, inst_end));

            block_exit_vregs[lbi] = block_entry_vregs[lbi];
        }

        // --- Build branch_blockparams ---
        // For each block's branch, for each successor, pass exit_vregs matching successor's block params
        let mut branch_args: Vec<Vec<Vec<ra2::VReg>>> = vec![Vec::new(); total_blocks];
        for lbi in 0..total_blocks {
            for &succ in &local_succs[lbi] {
                let mut args = Vec::new();
                // Successor's block params are for its live-in regs
                for &param_vreg in &block_params_storage[succ] {
                    let preg = vreg_to_preg_origin[&param_vreg.vreg()];
                    if let Some(exit_v) = block_exit_vregs[lbi][preg as usize] {
                        args.push(ra2::VReg::new(exit_v, ra2::RegClass::Int));
                    } else {
                        // Missing exit vreg — create a placeholder.
                        let v = alloc_vreg(&mut next_vreg, preg, &mut vreg_to_preg_origin);
                        block_exit_vregs[lbi][preg as usize] = Some(v);
                        args.push(ra2::VReg::new(v, ra2::RegClass::Int));
                    }
                }
                branch_args[lbi].push(args);
            }
        }

        // Compact vregs: renumber to eliminate gaps (orphaned vregs cause regalloc2 panic)
        let (compact_num_vregs, compact_map) = {
            let mut used: HashSet<usize> = HashSet::new();
            for ops in &flat_operands {
                for op in ops { used.insert(op.vreg().vreg()); }
            }
            for params in &block_params_storage {
                for p in params { used.insert(p.vreg()); }
            }
            for succ_args in &branch_args {
                for args in succ_args {
                    for v in args { used.insert(v.vreg()); }
                }
            }
            let mut sorted_used: Vec<usize> = used.into_iter().collect();
            sorted_used.sort();
            let mut map: HashMap<usize, usize> = HashMap::new();
            for (new_idx, &old_idx) in sorted_used.iter().enumerate() {
                map.insert(old_idx, new_idx);
            }
            (sorted_used.len(), map)
        };

        // Apply compact mapping to all vreg references
        let remap = |v: usize| -> usize { compact_map[&v] };
        for ops in &mut flat_operands {
            for op in ops.iter_mut() {
                let old = op.vreg();
                *op = Operand::new(
                    ra2::VReg::new(remap(old.vreg()), old.class()),
                    op.constraint(),
                    op.kind(),
                    op.pos(),
                );
            }
        }
        for params in &mut block_params_storage {
            for p in params.iter_mut() {
                *p = ra2::VReg::new(remap(p.vreg()), p.class());
            }
        }
        for succ_args in &mut branch_args {
            for args in succ_args.iter_mut() {
                for v in args.iter_mut() {
                    *v = ra2::VReg::new(remap(v.vreg()), v.class());
                }
            }
        }
        // Remap vreg_to_preg_origin
        let remapped_origin: HashMap<usize, u8> = vreg_to_preg_origin.iter()
            .filter_map(|(&old_v, &preg)| compact_map.get(&old_v).map(|&new_v| (new_v, preg)))
            .collect();
        vreg_to_preg_origin = remapped_origin;
        // Remap block_exit_vregs
        for exits in &mut block_exit_vregs {
            for slot in exits.iter_mut() {
                if let Some(v) = slot {
                    if let Some(&new_v) = compact_map.get(v) {
                        *v = new_v;
                    }
                }
            }
        }

        // --- RPO reordering ---
        // The regalloc2 checker processes blocks in index order during its
        // fixpoint analysis. If forward-edge predecessors (e.g. trampolines)
        // have higher indices than their successors (e.g. the loop header),
        // the checker's fixpoint may not converge correctly for pass-through
        // block params. We reorder all blocks into RPO so that forward-edge
        // predecessors always have lower indices.
        let rpo_order = {
            let mut order = Vec::new();
            let mut visited = vec![false; total_blocks];
            fn dfs_post(b: usize, succs: &[Vec<usize>], visited: &mut [bool], order: &mut Vec<usize>) {
                if visited[b] { return; }
                visited[b] = true;
                for &s in &succs[b] {
                    dfs_post(s, succs, visited, order);
                }
                order.push(b);
            }
            dfs_post(real_entry, &local_succs, &mut visited, &mut order);
            // Visit any unreachable blocks (e.g. infinite loop tail)
            for b in 0..total_blocks {
                if !visited[b] {
                    dfs_post(b, &local_succs, &mut visited, &mut order);
                }
            }
            order.reverse(); // post-order reversed = RPO
            order
        };
        // Build old→new block index mapping
        let mut rpo_map = vec![0usize; total_blocks]; // old index → new index
        for (new_idx, &old_idx) in rpo_order.iter().enumerate() {
            rpo_map[old_idx] = new_idx;
        }
        // Reorder per-block arrays AND repack flat instruction arrays so
        // that instruction indices are globally increasing in block order.
        // regalloc2's checker requires inst IDs to be monotonically increasing
        // across blocks when blocks are iterated in index order.
        let reorder_succs = |v: &[Vec<usize>]| -> Vec<Vec<usize>> {
            let mut out = vec![Vec::new(); total_blocks];
            for (new_idx, &old_idx) in rpo_order.iter().enumerate() {
                out[new_idx] = v[old_idx].iter().map(|&i| rpo_map[i]).collect();
            }
            out
        };
        let local_succs = reorder_succs(&local_succs);
        let local_preds = reorder_succs(&local_preds);
        let block_params_storage: Vec<Vec<ra2::VReg>> = rpo_order.iter().map(|&old| block_params_storage[old].clone()).collect();
        let branch_args: Vec<Vec<Vec<ra2::VReg>>> = rpo_order.iter().map(|&old| branch_args[old].clone()).collect();
        let block_exit_vregs: Vec<[Option<usize>; 32]> = rpo_order.iter().map(|&old| block_exit_vregs[old]).collect();
        let real_entry = rpo_map[real_entry];
        // Repack flat instruction arrays in new block order
        let total_insts = flat_operands.len();
        let mut new_operands = Vec::with_capacity(total_insts);
        let mut new_is_branch = Vec::with_capacity(total_insts);
        let mut new_is_ret = Vec::with_capacity(total_insts);
        let mut new_is_call = Vec::with_capacity(total_insts);
        let mut new_info = Vec::with_capacity(total_insts);
        let mut new_block_inst_ranges = Vec::with_capacity(total_blocks);
        for &old_bi in &rpo_order {
            let (old_start, old_end) = block_inst_ranges[old_bi];
            let new_start = new_operands.len();
            for old_idx in old_start..old_end {
                new_operands.push(flat_operands[old_idx].clone());
                new_is_branch.push(flat_is_branch[old_idx]);
                new_is_ret.push(flat_is_ret[old_idx]);
                new_is_call.push(flat_is_call[old_idx]);
                new_info.push(flat_info[old_idx].clone());
            }
            let new_end = new_operands.len();
            new_block_inst_ranges.push((new_start, new_end));
        }
        let flat_operands = new_operands;
        let flat_is_branch = new_is_branch;
        let flat_is_ret = new_is_ret;
        let flat_is_call = new_is_call;
        let flat_info = new_info;
        let block_inst_ranges = new_block_inst_ranges;

        // Convert succs/preds to ra2::Block
        let block_succs_ra2: Vec<Vec<ra2::Block>> = local_succs.iter()
            .map(|s| s.iter().map(|&i| ra2::Block::new(i)).collect()).collect();
        let block_preds_ra2: Vec<Vec<ra2::Block>> = local_preds.iter()
            .map(|p| p.iter().map(|&i| ra2::Block::new(i)).collect()).collect();

        // Build local block → start address map
        let mut local_block_addr = vec![0u32; total_blocks];
        for (new_idx, &old_idx) in rpo_order.iter().enumerate() {
            if old_idx < num_local_blocks {
                let gbi = local_to_global[old_idx];
                local_block_addr[new_idx] = all_blocks[gbi].start_addr;
            }
        }

        // Remap trampoline_blocks indices and build per-block trampoline index map
        let trampoline_blocks: Vec<(usize, usize, usize)> = trampoline_blocks.iter()
            .map(|&(src, si, target)| (rpo_map[src], si, rpo_map[target]))
            .collect();
        let mut block_trampoline_idx = vec![None; total_blocks];
        for (ti, &(_src, _si, _target)) in trampoline_blocks.iter().enumerate() {
            // The trampoline's RPO block index: the old index was num_local_blocks + ti
            let old_tramp_block = num_local_blocks + ti;
            let new_tramp_block = rpo_map[old_tramp_block];
            block_trampoline_idx[new_tramp_block] = Some(ti);
        }

        Some(Rv32Ra2Func {
            operands: flat_operands,
            is_branch_flag: flat_is_branch,
            is_ret_flag: flat_is_ret,
            _is_call_flag: flat_is_call,
            _num_regs: num_regs,
            block_inst_ranges,
            block_succs: block_succs_ra2,
            block_preds: block_preds_ra2,
            block_params_storage,
            branch_args,
            num_vregs: compact_num_vregs,
            inst_info: flat_info,
            entry_block_id: real_entry,
            vreg_to_preg_origin,
            block_exit_vregs,
            trampoline_blocks,
            block_trampoline_idx,
            local_block_addr,
        })
    }
}

impl ra2::Function for Rv32Ra2Func {
    fn num_insts(&self) -> usize { self.operands.len() }
    fn num_blocks(&self) -> usize { self.block_inst_ranges.len() }
    fn entry_block(&self) -> ra2::Block { ra2::Block::new(self.entry_block_id) }

    fn block_insns(&self, block: ra2::Block) -> ra2::InstRange {
        let (s, e) = self.block_inst_ranges[block.index()];
        ra2::InstRange::new(ra2::Inst::new(s), ra2::Inst::new(e))
    }

    fn block_succs(&self, block: ra2::Block) -> &[ra2::Block] {
        &self.block_succs[block.index()]
    }
    fn block_preds(&self, block: ra2::Block) -> &[ra2::Block] {
        &self.block_preds[block.index()]
    }
    fn block_params(&self, block: ra2::Block) -> &[ra2::VReg] {
        &self.block_params_storage[block.index()]
    }

    fn is_ret(&self, insn: ra2::Inst) -> bool { self.is_ret_flag[insn.index()] }
    fn is_branch(&self, insn: ra2::Inst) -> bool { self.is_branch_flag[insn.index()] }

    fn branch_blockparams(&self, block: ra2::Block, _insn: ra2::Inst, succ_idx: usize) -> &[ra2::VReg] {
        let bi = block.index();
        if succ_idx < self.branch_args[bi].len() {
            &self.branch_args[bi][succ_idx]
        } else {
            &[]
        }
    }

    fn inst_operands(&self, insn: ra2::Inst) -> &[Operand] {
        &self.operands[insn.index()]
    }

    fn inst_clobbers(&self, _insn: ra2::Inst) -> ra2::PRegSet {
        // No clobbers: the interpreter saves/restores ALL physical registers
        // around calls (Rust-level recursion in execute_function), so from
        // regalloc2's perspective registers survive calls. The only register
        // that changes is x10 (return value), modeled via a Def operand.
        ra2::PRegSet::empty()
    }
    fn num_vregs(&self) -> usize { self.num_vregs }
    fn spillslot_size(&self, _: ra2::RegClass) -> usize { 1 }
}

// ---------------------------------------------------------------------------
// Run regalloc2 on all functions and collect ISA stats
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Rewritten instruction stream
// ---------------------------------------------------------------------------

/// A rewritten instruction with physical registers from the reduced ISA.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RewrittenInst {
    pub addr: u32,           // original address (0 for synthetic moves)
    pub op: String,          // base opcode (e.g. "addi", "mov", "spill", "reload")
    pub rd: Option<u8>,      // allocated physical register index (0..num_regs-1)
    pub rs1: Option<u8>,     // allocated physical register index
    pub rs2: Option<u8>,     // allocated physical register index
    pub imm: Option<i32>,    // immediate value (preserved from original, or spill slot)
    pub is_move: bool,       // true if regalloc2-inserted move/spill/reload
    pub specialized: String, // fully specialized opcode name
    pub orig_rd: Option<u8>,  // original RV32 physical register for rd (from DecodedInst)
    pub orig_rs1: Option<u8>, // original RV32 physical register for rs1
    pub orig_rs2: Option<u8>, // original RV32 physical register for rs2
}

fn make_move_inst(from: &ra2::Allocation, to: &ra2::Allocation) -> RewrittenInst {
    let from_reg = from.as_reg().map(|p| p.hw_enc() as u8);
    let to_reg = to.as_reg().map(|p| p.hw_enc() as u8);
    let from_spill = from.as_stack().map(|s| s.index() as u32);
    let to_spill = to.as_stack().map(|s| s.index() as u32);

    match (to_reg, from_reg, to_spill, from_spill) {
        (Some(d), Some(s), _, _) => RewrittenInst {
            addr: 0, op: "mov".into(),
            rd: Some(d), rs1: Some(s), rs2: None,
            imm: None, is_move: true,
            specialized: format!("mov.r{}.r{}", d, s),
            orig_rd: None, orig_rs1: None, orig_rs2: None,
        },
        (_, Some(s), Some(slot), _) => RewrittenInst {
            addr: 0, op: "spill".into(),
            rd: None, rs1: Some(s), rs2: None,
            imm: Some(slot as i32), is_move: true,
            specialized: format!("spill.r{}", s),
            orig_rd: None, orig_rs1: None, orig_rs2: None,
        },
        (Some(d), _, _, Some(slot)) => RewrittenInst {
            addr: 0, op: "reload".into(),
            rd: Some(d), rs1: None, rs2: None,
            imm: Some(slot as i32), is_move: true,
            specialized: format!("reload.r{}", d),
            orig_rd: None, orig_rs1: None, orig_rs2: None,
        },
        (_, _, Some(to_s), Some(from_s)) => RewrittenInst {
            addr: 0, op: "restack".into(),
            rd: None, rs1: None, rs2: None,
            imm: None, is_move: true,
            specialized: format!("restack.s{}.s{}", to_s, from_s),
            orig_rd: None, orig_rs1: None, orig_rs2: None,
        },
        _ => RewrittenInst {
            addr: 0, op: "nop_move".into(),
            rd: None, rs1: None, rs2: None,
            imm: None, is_move: true,
            specialized: "nop_move".into(),
            orig_rd: None, orig_rs1: None, orig_rs2: None,
        },
    }
}

fn rewrite_one_inst(d: &DecodedInst, allocs: &[ra2::Allocation]) -> RewrittenInst {
    let mut name = d.op.clone();
    let mut rd_new = None;
    let mut rs1_new = None;
    let mut rs2_new = None;
    let mut ai = 0;

    // Def first (rd) — skip x0
    if d.rd.is_some() && d.rd != Some(0) {
        if ai < allocs.len() && allocs[ai].is_reg() {
            let preg = allocs[ai].as_reg().unwrap();
            let hw = preg.hw_enc() as u8;
            rd_new = Some(hw);
            name = format!("{}.r{}", name, hw);
        }
        ai += 1;
    }

    // Uses (rs1, rs2) — match build() iteration:
    // [d.rs1, d.rs2].into_iter().flatten() skips None,
    // build() skips x0 because current[0] = None.
    for (src_idx, r_opt) in [(0usize, d.rs1), (1, d.rs2)] {
        if let Some(r) = r_opt {
            if r == 0 { continue; } // x0 never gets a vreg operand
            if ai < allocs.len() && allocs[ai].is_reg() {
                let preg = allocs[ai].as_reg().unwrap();
                let hw = preg.hw_enc() as u8;
                if src_idx == 0 { rs1_new = Some(hw); } else { rs2_new = Some(hw); }
                name = format!("{}.r{}", name, hw);
            } else if ai < allocs.len() && allocs[ai].is_stack() {
                name = format!("{}.spill", name);
            }
            ai += 1;
        }
    }

    RewrittenInst {
        addr: d.addr,
        op: d.op.clone(),
        rd: rd_new,
        rs1: rs1_new,
        rs2: rs2_new,
        imm: d.imm,
        is_move: false,
        specialized: name,
        orig_rd: d.rd,
        orig_rs1: d.rs1,
        orig_rs2: d.rs2,
    }
}

// ---------------------------------------------------------------------------
// Parallel move helpers
// ---------------------------------------------------------------------------



/// Emit regalloc2 edits as sequential instructions (in order).
///
/// regalloc2's parallel-move resolver already serializes moves into a correct
/// sequential order (with cycle-breaking via scratch registers/spill slots).
/// We must NOT reorder them — just convert each (from, to) pair into the
/// appropriate mov/spill/reload instruction.
fn emit_moves_sequential(
    moves: &[(ra2::Allocation, ra2::Allocation)],
) -> Vec<RewrittenInst> {
    moves.iter().map(|(from, to)| make_move_inst(from, to)).collect()
}

/// Rewrite the instruction stream for one function.
///
/// Returns `(instructions, jr_table_redirects)`.
/// `jr_table_redirects` maps original target addr → trampoline synth_addr
/// for jr_table targets that need critical-edge trampoline redirection.
pub fn rewrite_function(
    decoded: &[DecodedInst],
    ra2_func: &Rv32Ra2Func,
    output: &ra2::Output,
) -> (Vec<RewrittenInst>, HashMap<(u32, u32), u32>) {
    let mut result = Vec::new();

    // Collect edits by instruction index
    let mut edits_before: HashMap<usize, Vec<(ra2::Allocation, ra2::Allocation)>> = HashMap::new();
    let mut edits_after: HashMap<usize, Vec<(ra2::Allocation, ra2::Allocation)>> = HashMap::new();
    for &(ref pp, ref edit) in &output.edits {
        let ra2::Edit::Move { from, to } = edit;
        let idx = pp.inst().index();
        match pp.pos() {
            ra2::InstPosition::Before => edits_before.entry(idx).or_default().push((*from, *to)),
            ra2::InstPosition::After => edits_after.entry(idx).or_default().push((*from, *to)),
        }
    }

    // Build trampoline info.
    // succ_idx=0 → taken branch, succ_idx=1 → fallthrough.
    // For each trampoline, assign a synthetic address so addr_to_idx can find it.
    struct TrampInfo {
        synth_addr: u32,
        target_addr: u32,
        src: usize,          // source local block
        succ_idx: usize,     // 0=taken, 1=fallthrough
    }
    let mut tramp_info: Vec<TrampInfo> = Vec::new();
    // (src_local_block, original_target_addr) → (synth_addr, succ_idx)
    let mut branch_redirects: HashMap<(usize, u32), (u32, usize)> = HashMap::new();
    let mut seen_synth_addrs: HashSet<u32> = HashSet::new();

    for (ti, &(src, succ_idx, target)) in ra2_func.trampoline_blocks.iter().enumerate() {
        let target_addr = if target < ra2_func.local_block_addr.len() {
            ra2_func.local_block_addr[target]
        } else { 0 };
        // Use high address range (0xF0xxxxxx) to avoid collisions with real .text addresses.
        // Incorporate the function entry address for uniqueness across functions.
        let func_entry = ra2_func.local_block_addr.first().copied().unwrap_or(0);
        let synth_addr = 0xF000_0000_u32
            .wrapping_add(func_entry & 0x00FF_FFFF)
            .wrapping_add(ti as u32 * 4);
        assert!(
            seen_synth_addrs.insert(synth_addr),
            "trampoline synth_addr collision: 0x{:08x} for func=0x{:x} ti={}",
            synth_addr, func_entry, ti
        );
        tramp_info.push(TrampInfo { synth_addr, target_addr, src, succ_idx });
        branch_redirects.insert((src, target_addr), (synth_addr, succ_idx));
    }

    // For each source block, collect: which fallthrough trampolines to emit after it
    let mut fallthrough_tramps_after: HashMap<usize, Vec<usize>> = HashMap::new(); // block_idx → [tramp_idx]
    for (ti, t) in tramp_info.iter().enumerate() {
        if t.succ_idx == 1 {
            // Fallthrough trampoline: needs jal inserted after source block
            fallthrough_tramps_after.entry(t.src).or_default().push(ti);
        }
    }

    // Walk blocks in order, emit instructions with edits
    for (block_idx, &(inst_start, inst_end)) in ra2_func.block_inst_ranges.iter().enumerate() {
        // Check if this is a trampoline block
        let tramp_idx = ra2_func.block_trampoline_idx[block_idx];

        // If trampoline, mark the first emitted instruction with the synthetic address
        let mut is_first_tramp_inst = tramp_idx.is_some();
        // Remember result length so we can verify synth_addr coverage afterward.
        let block_emit_start = result.len();

        for flat_idx in inst_start..inst_end {
            let info = &ra2_func.inst_info[flat_idx];

            // Edits before this instruction.
            // The first emitted move gets the target instruction's address so that
            // build_branch_target_map can resolve branch targets directly without
            // a backward walk.  For trampoline block entries the synthetic address
            // is used instead; for all other real instructions the decoded addr.
            if let Some(moves) = edits_before.get(&flat_idx) {
                let emitted = emit_moves_sequential(moves);
                let mut first = true;
                for mut mv in emitted {
                    if first {
                        first = false;
                        if is_first_tramp_inst {
                            mv.addr = tramp_info[tramp_idx.unwrap()].synth_addr;
                            is_first_tramp_inst = false;
                        } else if !info.is_synthetic {
                            mv.addr = decoded[info.orig_idx].addr;
                        }
                    }
                    result.push(mv);
                }
            }

            // The instruction itself
            if !info.is_synthetic {
                let d = &decoded[info.orig_idx];
                let allocs = output.inst_allocs(ra2::Inst::new(flat_idx));

                if is_call_inst(d) && d.rd != Some(10) {
                    let has_rd = d.rd.is_some() && d.rd != Some(0);
                    let x10_idx = if has_rd { 1 } else { 0 };
                    let mut filtered: Vec<ra2::Allocation> = Vec::new();
                    for (i, &a) in allocs.iter().enumerate() {
                        if i != x10_idx { filtered.push(a); }
                    }
                    let mut inst = rewrite_one_inst(d, &filtered);
                    // Patch taken-branch redirects (succ_idx=0)
                    if matches!(inst.op.as_str(), "beq"|"bne"|"blt"|"bge"|"bltu"|"bgeu") {
                        if let Some(imm) = inst.imm {
                            let target = (inst.addr as i64 + imm as i64) as u32;
                            if let Some(&(synth, 0)) = branch_redirects.get(&(block_idx, target)) {
                                inst.imm = Some((synth as i64 - inst.addr as i64) as i32);
                            }
                        }
                    }
                    // Emit conv_store if instruction writes to an original register
                    let conv_orig = inst.orig_rd;
                    let conv_phys = inst.rd;
                    result.push(inst);
                    if let (Some(orig), Some(phys)) = (conv_orig, conv_phys) {
                        if orig != 0 {
                            result.push(RewrittenInst {
                                addr: 0, op: "conv_store".into(),
                                rd: None, rs1: Some(phys), rs2: None, imm: None,
                                is_move: false,
                                specialized: format!("conv_store.r{}.x{}", phys, orig),
                                orig_rd: Some(orig), orig_rs1: None, orig_rs2: None,
                            });
                        }
                    }

                    if x10_idx < allocs.len() && allocs[x10_idx].is_reg() {
                        let phys = allocs[x10_idx].as_reg().unwrap().hw_enc() as u8;
                        result.push(RewrittenInst {
                            addr: 0, op: "conv_load".into(),
                            rd: Some(phys), rs1: None, rs2: None, imm: None,
                            is_move: false,
                            specialized: format!("conv_load.r{}.x10", phys),
                            orig_rd: Some(10), orig_rs1: None, orig_rs2: None,
                        });
                    }
                } else {
                    // Guard: if this is a call with d.rd == Some(10), the conv_load
                    // for the return value (x10) is silently skipped above, which
                    // means conv_regs[10] never propagates to the physical register
                    // after the call.  We believe this case never occurs in practice
                    // (normal calls use rd=x1); panic here so it can't go unnoticed.
                    assert!(
                        !is_call_inst(d) || d.rd != Some(10),
                        "call at 0x{:x} has rd=x10: conv_load for return value \
                         would be silently dropped — calling convention broken",
                        d.addr
                    );
                    // Debug: dump allocs for specific address
                    if std::env::var("DUMP_ALLOC").is_ok() {
                        if d.addr >= 0x0115f0 && d.addr <= 0x011680 {
                            eprintln!("  ALLOC 0x{:x}: op={} rd={:?} rs1={:?} rs2={:?} allocs={:?} ops={:?}",
                                d.addr, d.op, d.rd, d.rs1, d.rs2, allocs,
                                &ra2_func.operands[flat_idx]);
                        }
                    }
                    let mut inst = rewrite_one_inst(d, allocs);
                    // Patch taken-branch redirects (succ_idx=0)
                    if matches!(inst.op.as_str(), "beq"|"bne"|"blt"|"bge"|"bltu"|"bgeu") {
                        if let Some(imm) = inst.imm {
                            let target = (inst.addr as i64 + imm as i64) as u32;
                            if let Some(&(synth, 0)) = branch_redirects.get(&(block_idx, target)) {
                                inst.imm = Some((synth as i64 - inst.addr as i64) as i32);
                            }
                        }
                    }
                    // Emit conv_store if instruction writes to an original register
                    let conv_orig = inst.orig_rd;
                    let conv_phys = inst.rd;
                    result.push(inst);
                    if let (Some(orig), Some(phys)) = (conv_orig, conv_phys) {
                        if orig != 0 {
                            result.push(RewrittenInst {
                                addr: 0, op: "conv_store".into(),
                                rd: None, rs1: Some(phys), rs2: None, imm: None,
                                is_move: false,
                                specialized: format!("conv_store.r{}.x{}", phys, orig),
                                orig_rd: Some(orig), orig_rs1: None, orig_rs2: None,
                            });
                        }
                    }
                }
            } else {
                // Synthetic entry def: emit conv_load to populate register from conv_regs
                let ops = &ra2_func.operands[flat_idx];
                if ops.len() == 1 && ops[0].kind() == OperandKind::Def {
                    let vreg_idx = ops[0].vreg().vreg();
                    if let Some(&orig_preg) = ra2_func.vreg_to_preg_origin.get(&vreg_idx) {
                        let allocs = output.inst_allocs(ra2::Inst::new(flat_idx));
                        if !allocs.is_empty() && allocs[0].is_reg() {
                            let phys = allocs[0].as_reg().unwrap().hw_enc() as u8;
                            // If this conv_load is the first thing emitted in a
                            // trampoline block, give it the synth_addr so the map
                            // finds the block without a backward walk.
                            let addr = if is_first_tramp_inst {
                                is_first_tramp_inst = false;
                                tramp_info[tramp_idx.unwrap()].synth_addr
                            } else { 0 };
                            result.push(RewrittenInst {
                                addr,
                                op: "conv_load".into(),
                                rd: Some(phys),
                                rs1: None, rs2: None,
                                imm: None,
                                is_move: false,
                                specialized: format!("conv_load.r{}.x{}", phys, orig_preg),
                                orig_rd: Some(orig_preg),
                                orig_rs1: None, orig_rs2: None,
                            });
                        }
                    }
                }
                // Synthetic branch instructions emit nothing (just structural)
            }

            // Edits after this instruction
            if let Some(moves) = edits_after.get(&flat_idx) {
                let emitted = emit_moves_sequential(moves);
                for mut mv in emitted {
                    if is_first_tramp_inst {
                        mv.addr = tramp_info[tramp_idx.unwrap()].synth_addr;
                        is_first_tramp_inst = false;
                    }
                    result.push(mv);
                }
            }
        }

        // For non-trampoline blocks ending with a synthetic branch (single-
        // successor fallthrough), emit an explicit jal when the successor is
        // not the next block in layout order.  Without this, execution falls
        // through to the wrong block.
        if tramp_idx.is_none() && inst_start < inst_end {
            let last_flat = inst_end - 1;
            if ra2_func.inst_info[last_flat].is_synthetic
                && ra2_func.is_branch_flag[last_flat]
            {
                let succs = &ra2_func.block_succs[block_idx];
                if succs.len() == 1 {
                    let succ = succs[0].index();
                    let next_block = block_idx + 1;
                    if succ != next_block
                        || next_block >= ra2_func.block_inst_ranges.len()
                    {
                        let target_addr =
                            if let Some(ti) = ra2_func.block_trampoline_idx[succ] {
                                tramp_info[ti].synth_addr
                            } else {
                                ra2_func.local_block_addr[succ]
                            };
                        result.push(RewrittenInst {
                            addr: 0,
                            op: "jal".into(),
                            rd: None, rs1: None, rs2: None,
                            imm: Some(target_addr as i32),
                            is_move: false,
                            specialized: "jal".into(),
                            orig_rd: Some(0), orig_rs1: None, orig_rs2: None,
                        });
                    }
                }
            }
        }

        // If this is a trampoline block, add a jump to the real target
        if let Some(ti) = tramp_idx {
            let t = &tramp_info[ti];
            let jal_addr = if is_first_tramp_inst { t.synth_addr } else { 0 };
            result.push(RewrittenInst {
                addr: jal_addr,
                op: "jal".into(),
                rd: None, rs1: None, rs2: None,
                imm: Some((t.target_addr as i64 - jal_addr as i64) as i32),
                is_move: false,
                specialized: "jal".into(),
                orig_rd: Some(0), orig_rs1: None, orig_rs2: None,
            });

            // Invariant: exactly one instruction in this block must carry the
            // synth_addr so that build_branch_target_map can find the trampoline.
            // If the synth_addr was placed on the first edits_before move but a
            // future refactor drops or resets it, no instruction would bear it and
            // the ISA VM would silently jump to the wrong place.
            let synth_addr = t.synth_addr;
            let has_synth = result[block_emit_start..]
                .iter()
                .any(|i| i.addr == synth_addr);
            assert!(
                has_synth,
                "trampoline ti={} synth_addr=0x{:08x} not found in any emitted instruction \
                 (block_idx={}, emit_range={}..{}); build_branch_target_map will miss it",
                ti, synth_addr, block_idx, block_emit_start, result.len()
            );
        }

        // If this is an original block with fallthrough trampolines, insert jal to trampoline
        if let Some(ft_tramps) = fallthrough_tramps_after.get(&block_idx) {
            for &ti in ft_tramps {
                let t = &tramp_info[ti];
                result.push(RewrittenInst {
                    addr: 0,
                    op: "jal".into(),
                    rd: None, rs1: None, rs2: None,
                    imm: Some(t.synth_addr as i32),
                    is_move: false,
                    specialized: "jal".into(),
                    orig_rd: Some(0), orig_rs1: None, orig_rs2: None,
                });
            }
        }
    }

    // Build jr_table redirect map: for each trampoline, if the source block
    // contains a jr_table instruction, add (jr_table_addr, target) → synth_addr.
    // Keyed per-instruction so two jr_tables targeting the same block through
    // different trampolines don't silently overwrite each other.
    let mut jr_table_redirects: HashMap<(u32, u32), u32> = HashMap::new();
    for (&(src_block, target_addr), &(synth_addr, _succ_idx)) in &branch_redirects {
        // Find if this source block has a jr_table as its terminator
        let (inst_start, inst_end) = ra2_func.block_inst_ranges[src_block];
        for flat_idx in (inst_start..inst_end).rev() {
            let info = &ra2_func.inst_info[flat_idx];
            if info.is_synthetic { continue; }
            let d = &decoded[info.orig_idx];
            if d.op == "jr_table" {
                jr_table_redirects.insert((d.addr, target_addr), synth_addr);
            }
            break; // only check last real instruction
        }
    }

    (result, jr_table_redirects)
}

// ---------------------------------------------------------------------------
// Run regalloc2 on all functions and collect ISA stats
// ---------------------------------------------------------------------------

pub struct Rv32AllocResult {
    /// Per-function results
    pub func_results: Vec<FuncAllocResult>,
}

pub struct FuncAllocResult {
    pub entry_addr: u32,
    pub num_blocks: usize,
    pub num_insts: usize,
    pub num_vregs: usize,
    pub num_spills: usize,
    /// Map of specialized opcode → count
    pub specialized: HashMap<String, usize>,
    /// Rewritten instruction stream (empty if allocation failed)
    pub rewritten: Vec<RewrittenInst>,
    pub ok: bool,
    pub error: Option<String>,
    /// Mapping of (original_preg, physical_reg) at function entry
    pub entry_reg_map: Vec<(u8, u8)>,
    /// Number of spill slots allocated by regalloc2
    pub num_spill_slots: usize,
    /// jr_table redirect map: (jr_table_inst_addr, target_addr) → trampoline synth_addr
    pub jr_table_redirects: HashMap<(u32, u32), u32>,
    /// Physical register holding x10 (return value) at ret sites
    pub ret_x10_reg: Option<u8>,
}

/// Build the MachineEnv for N integer registers.
pub fn make_machine_env(num_regs: u32) -> ra2::MachineEnv {
    let mut preferred = Vec::new();
    for r in 0..num_regs {
        preferred.push(ra2::PReg::new(r as usize, ra2::RegClass::Int));
    }
    ra2::MachineEnv {
        preferred_regs_by_class: [preferred, vec![], vec![]],
        non_preferred_regs_by_class: [vec![], vec![], vec![]],
        scratch_by_class: [None, None, None],
        fixed_stack_slots: vec![],
    }
}

/// Run regalloc2 on all functions in the binary.
pub fn run_regalloc(
    decoded: &[DecodedInst],
    blocks: &[BasicBlock],
    num_regs: u32,
) -> Rv32AllocResult {
    run_regalloc_inner(decoded, blocks, num_regs, None)
}

pub fn run_regalloc_with_symbols(
    decoded: &[DecodedInst],
    blocks: &[BasicBlock],
    num_regs: u32,
    elf_funcs: &[(u32, u32)],
) -> Rv32AllocResult {
    run_regalloc_inner(decoded, blocks, num_regs, Some(elf_funcs))
}

fn run_regalloc_inner(
    decoded: &[DecodedInst],
    blocks: &[BasicBlock],
    num_regs: u32,
    elf_funcs: Option<&[(u32, u32)]>,
) -> Rv32AllocResult {
    let functions = if let Some(syms) = elf_funcs {
        find_functions_from_symbols(blocks, syms)
    } else {
        find_functions(blocks)
    };
    let (live_in, live_out) = compute_liveness(decoded, blocks);
    let env = make_machine_env(num_regs);
    let opts = ra2::RegallocOptions {
        verbose_log: false,
        validate_ssa: false,
        algorithm: ra2::Algorithm::Ion,
    };

    let mut func_results = Vec::new();

    for (entry, func_blocks) in &functions {
        let entry_addr = if *entry < blocks.len() { blocks[*entry].start_addr } else { 0 };

        let ra2_func = match Rv32Ra2Func::build(decoded, blocks, func_blocks, *entry, &live_in, &live_out, num_regs) {
            Some(f) => f,
            None => {
                func_results.push(FuncAllocResult {
                    entry_addr,
                    num_blocks: func_blocks.len(),
                    num_insts: 0, num_vregs: 0, num_spills: 0,
                    specialized: HashMap::new(),
                    rewritten: Vec::new(),
                    ok: false,
                    error: Some("build failed".into()),
                    entry_reg_map: Vec::new(),
                    num_spill_slots: 0,
                    jr_table_redirects: HashMap::new(),
                    ret_x10_reg: None,
                });
                continue;
            }
        };

        let num_blocks = ra2_func.block_inst_ranges.len();
        let num_insts = ra2_func.operands.len();
        let num_vregs = ra2_func.num_vregs;

        // Use catch_unwind to handle regalloc2 internal panics
        // Try Ion first, fall back to Fastalloc on panic
        let env_clone = env.clone();
        let ra2_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ra2::run(&ra2_func, &env_clone, &opts)
        }));

        // If Ion panicked or returned an error, retry with Fastalloc
        let should_retry = match &ra2_result {
            Err(_) => true,                   // panic
            Ok(Err(_)) => true,               // regalloc2 error (e.g. TooManyLiveRegs)
            Ok(Ok(_)) => false,               // success
        };
        let ra2_result = if should_retry {
            let fast_opts = ra2::RegallocOptions {
                verbose_log: false,
                validate_ssa: false,
                algorithm: ra2::Algorithm::Fastalloc,
            };
            let env_clone2 = env.clone();
            eprintln!("  Ion failed for 0x{:x}, retrying with Fastalloc", entry_addr);
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                ra2::run(&ra2_func, &env_clone2, &fast_opts)
            }))
        } else {
            ra2_result
        };

        match ra2_result {
            Ok(Ok(output)) => {
                // Dump RA2 function structure for debugging
                if std::env::var("DUMP_RA2").is_ok() {
                    eprintln!("=== RA2 FUNC 0x{:x} ===", entry_addr);
                    eprintln!("  num_vregs={}, num_blocks={}, entry=block{}",
                        ra2_func.num_vregs, ra2_func.block_inst_ranges.len(),
                        ra2_func.entry_block_id);
                    for bi in 0..ra2_func.block_inst_ranges.len() {
                        let (s, e) = ra2_func.block_inst_ranges[bi];
                        let params: Vec<String> = ra2_func.block_params_storage[bi].iter()
                            .map(|v| format!("v{}(x{})", v.vreg(), ra2_func.vreg_to_preg_origin.get(&v.vreg()).unwrap_or(&99)))
                            .collect();
                        let preds: Vec<usize> = ra2_func.block_preds[bi].iter().map(|b| b.index()).collect();
                        let succs: Vec<usize> = ra2_func.block_succs[bi].iter().map(|b| b.index()).collect();
                        eprintln!("  block{} insts=[{}..{}) params=[{}] preds={:?} succs={:?}",
                            bi, s, e, params.join(", "), preds, succs);
                        for inst_idx in s..e {
                            let ops: Vec<String> = ra2_func.operands[inst_idx].iter().map(|op| {
                                let vr = op.vreg();
                                let kind = if op.kind() == ra2::OperandKind::Def { "Def" } else { "Use" };
                                let pos = if op.pos() == ra2::OperandPos::Early { "E" } else { "L" };
                                let preg = ra2_func.vreg_to_preg_origin.get(&vr.vreg()).unwrap_or(&99);
                                format!("{}:{} v{}(x{}) ", kind, pos, vr.vreg(), preg)
                            }).collect();
                            let br = if ra2_func.is_branch_flag[inst_idx] { " BR" } else { "" };
                            let ret = if ra2_func.is_ret_flag[inst_idx] { " RET" } else { "" };
                            let info = &ra2_func.inst_info[inst_idx];
                            let addr_str = if info.is_synthetic { "synth".to_string() }
                                else { format!("0x{:x}", decoded[info.orig_idx].addr) };
                            let op_name = if info.is_synthetic { "---".to_string() }
                                else { decoded[info.orig_idx].op.clone() };
                            // Show allocation
                            let allocs: Vec<String> = ra2_func.operands[inst_idx].iter().enumerate().map(|(oi, _op)| {
                                let a = output.inst_allocs(ra2::Inst::new(inst_idx));
                                if oi < a.len() {
                                    format!("{}", a[oi])
                                } else { "?".into() }
                            }).collect();
                            eprintln!("    inst{:3} {} {:8} ops=[{}] allocs=[{}]{}{}",
                                inst_idx, addr_str, op_name,
                                ops.join(", "), allocs.join(", "), br, ret);
                        }
                        // branch_args
                        for (si, succ) in ra2_func.block_succs[bi].iter().enumerate() {
                            let args: Vec<String> = ra2_func.branch_args[bi][si].iter()
                                .map(|v| format!("v{}(x{})", v.vreg(), ra2_func.vreg_to_preg_origin.get(&v.vreg()).unwrap_or(&99)))
                                .collect();
                            if !args.is_empty() {
                                eprintln!("    → block{}: branch_args=[{}]", succ.index(), args.join(", "));
                            }
                        }
                    }
                    // Also print edits_before/after
                    eprintln!("  --- Edits ---");
                    for bi in 0..ra2_func.block_inst_ranges.len() {
                        let (s, e) = ra2_func.block_inst_ranges[bi];
                        for inst_idx in s..e {
                            let inst = ra2::Inst::new(inst_idx);
                            for edit in output.edits.iter() {
                                let (pp, edit_data) = edit;
                                if pp.inst() == inst {
                                    eprintln!("    inst{} {:?}: {:?}", inst_idx, pp.pos(), edit_data);
                                }
                            }
                        }
                    }
                }

                // Run regalloc2's built-in checker to verify allocation correctness.
                {
                    let mut checker = ra2::checker::Checker::new(&ra2_func, &env);
                    checker.prepare(&output);
                    if let Err(e) = checker.run() {
                        eprintln!("  CHECKER FAILED for 0x{:x}: {:?}", entry_addr, e);
                    }
                }

                // Rewrite instruction stream with allocations.
                let (rewritten_raw, jr_table_redirects) =
                    rewrite_function(decoded, &ra2_func, &output);

                // Post-rewrite passes (see rv32_passes.rs)
                use crate::rv32_passes::{lower_spills, pass_split_frame_ops, pass_resolve_branches};
                let rewritten_spills = lower_spills(&rewritten_raw, output.num_spillslots, frame_reg_id(num_regs));
                let rewritten_frame = pass_split_frame_ops(&rewritten_spills, frame_reg_id(num_regs));
                let rewritten = pass_resolve_branches(&rewritten_frame);

                // Build specialized opcodes from the lowered stream
                let mut specialized = HashMap::new();
                for inst in &rewritten {
                    *specialized.entry(inst.specialized.clone()).or_default() += 1;
                }

                let num_spills = output.num_spillslots;

                // Compute entry_reg_map: (orig_preg, phys_reg) at function entry
                let mut entry_reg_map = Vec::new();
                {
                    let entry_block = ra2_func.entry_block_id;
                    let (inst_start, inst_end) = ra2_func.block_inst_ranges[entry_block];
                    for flat_idx in inst_start..inst_end {
                        let info = &ra2_func.inst_info[flat_idx];
                        if !info.is_synthetic { continue; }
                        let ops = &ra2_func.operands[flat_idx];
                        if ops.len() == 1 && ops[0].kind() == OperandKind::Def {
                            let vreg_idx = ops[0].vreg().vreg();
                            if let Some(&orig_preg) = ra2_func.vreg_to_preg_origin.get(&vreg_idx) {
                                let allocs = output.inst_allocs(ra2::Inst::new(flat_idx));
                                if !allocs.is_empty() && allocs[0].is_reg() {
                                    let phys = allocs[0].as_reg().unwrap().hw_enc() as u8;
                                    entry_reg_map.push((orig_preg, phys));
                                }
                            }
                        }
                    }
                }

                // Find which physical register holds x10 at ret sites
                let mut ret_x10_reg: Option<u8> = None;
                let mut last_x10_phys: Option<u8> = None;
                for inst in &rewritten {
                    if inst.op == "conv_store" && inst.orig_rd == Some(10) {
                        last_x10_phys = inst.rs1;
                    }
                    if inst.op == "ret" {
                        if let Some(p) = last_x10_phys {
                            ret_x10_reg = Some(p);
                        }
                    }
                }

                func_results.push(FuncAllocResult {
                    entry_addr,
                    num_blocks,
                    num_insts,
                    num_vregs,
                    num_spills,
                    specialized,
                    rewritten,
                    ok: true,
                    error: None,
                    entry_reg_map,
                    num_spill_slots: output.num_spillslots,
                    jr_table_redirects,
                    ret_x10_reg,
                });
            }
            Ok(Err(e)) => {
                func_results.push(FuncAllocResult {
                    entry_addr,
                    num_blocks,
                    num_insts,
                    num_vregs,
                    num_spills: 0,
                    specialized: HashMap::new(),
                    rewritten: Vec::new(),
                    ok: false,
                    error: Some(format!("{:?}", e)),
                    entry_reg_map: Vec::new(),
                    num_spill_slots: 0,
                    jr_table_redirects: HashMap::new(),
                    ret_x10_reg: None,
                });
            }
            Err(panic) => {
                let msg = if let Some(s) = panic.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = panic.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    "unknown panic".into()
                };
                func_results.push(FuncAllocResult {
                    entry_addr,
                    num_blocks,
                    num_insts,
                    num_vregs,
                    num_spills: 0,
                    specialized: HashMap::new(),
                    rewritten: Vec::new(),
                    ok: false,
                    error: Some(format!("panic: {}", msg)),
                    entry_reg_map: Vec::new(),
                    num_spill_slots: 0,
                    jr_table_redirects: HashMap::new(),
                    ret_x10_reg: None,
                });
            }
        }
    }

    Rv32AllocResult { func_results }
}

// ---------------------------------------------------------------------------
// Spill lowering: convert spill/reload into real sw/lw using frame register
// ---------------------------------------------------------------------------

/// Return the dedicated frame pointer register ID for a given number of GP regs.
/// regalloc2 uses r0..r(N-1) for GP; the frame register is always r_N,
/// just above the allocatable range, so it can never be clobbered by regalloc2.
pub fn frame_reg_id(num_regs: u32) -> u8 {
    num_regs as u8
}

/// Build a map from instruction address to index in the rewritten stream.
/// Used for branch target resolution.
///
/// rewrite_function guarantees that the first emitted instruction of every
/// block (real or trampoline) carries the block's entry address — either the
/// decoded instruction's original .text address (for original blocks) or the
/// pre-assigned synth_addr (for trampoline blocks).  Therefore a simple
/// first-occurrence scan is sufficient; no backward walk is needed.
pub fn build_branch_target_map(rewritten: &[RewrittenInst]) -> HashMap<u32, usize> {
    let mut map = HashMap::new();

    for (idx, inst) in rewritten.iter().enumerate() {
        if inst.addr != 0 {
            map.entry(inst.addr).or_insert(idx);
        }
    }

    map
}

