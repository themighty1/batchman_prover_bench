//! Multi-block CFG construction and regalloc2 integration.
//!
//! Extracted from isa_regalloc.rs — provides:
//! - CFG construction from VRegInst streams
//! - Ra2FuncMulti: regalloc2 Function impl with FixedReg ISA constraints
//! - Multi-block rewriter: applies regalloc2 allocations to produce new VRegInst stream

use std::collections::{BTreeSet, HashMap, HashSet};

use regalloc2::{
    self as ra2, MachineEnv, Operand, OperandConstraint, OperandKind, OperandPos,
    RegallocOptions, Algorithm,
};

use crate::regvm::{VRegInst, VReg, PReg as OurPReg, SpillSlot};
use crate::regalloc::RegAllocResult;
use crate::preg_vm::{vreg_dst_regs, vreg_src_regs, specialized_opcode, rewrite_inst_vregs};

// ============================================================
// CFG types
// ============================================================

/// Basic block in the control flow graph.
#[derive(Debug, Clone)]
pub struct CfgBlock {
    /// Indices into original VRegInst stream for this block's ra2 instructions.
    /// Includes data instructions and branch terminators (If/BrIf/Br/Return).
    /// Does NOT include pure markers (Block/Loop/Else/End).
    pub ra2_insts: Vec<usize>,
    /// Successor block indices
    pub succs: Vec<usize>,
    /// Predecessor block indices
    pub preds: Vec<usize>,
    /// Range [start, end) of original VRegInst indices covered by this block
    pub orig_start: usize,
    pub orig_end: usize,
    /// Is this a synthetic trampoline block?
    pub is_trampoline: bool,
}

// ============================================================
// Helpers
// ============================================================

/// Returns true if the instruction is a pure structural marker (no vreg operands).
pub fn is_pure_marker(inst: &VRegInst) -> bool {
    matches!(inst, VRegInst::Block { .. } | VRegInst::Loop { .. } |
             VRegInst::Else { .. } | VRegInst::End { .. })
}

/// Returns true if the instruction is a branch terminator for a basic block.
pub fn is_block_terminator(inst: &VRegInst) -> bool {
    matches!(inst, VRegInst::If { .. } | VRegInst::BrIf { .. } | VRegInst::Br { .. } |
             VRegInst::BrTable { .. } | VRegInst::Return { .. } | VRegInst::Unreachable)
}

/// Parse an ISA opcode string like "i32.add.r0.r1.r2" into ("i32.add", [0, 1, 2]).
pub fn parse_isa_opcode(s: &str) -> (&str, Vec<u8>) {
    let parts: Vec<&str> = s.split('.').collect();
    let mut base_end = 0;
    let mut regs = Vec::new();
    for (i, p) in parts.iter().enumerate() {
        if p.starts_with('r') && p.len() > 1 && p[1..].chars().all(|c| c.is_ascii_digit()) {
            if base_end == 0 { base_end = i; }
            regs.push(p[1..].parse::<u8>().unwrap());
        } else if p.starts_with('s') && p.len() > 1 && p[1..].chars().all(|c| c.is_ascii_digit()) {
            if base_end == 0 { base_end = i; }
            regs.push(255);
        } else if *p == "?" {
            if base_end == 0 { base_end = i; }
            regs.push(254);
        }
    }
    if base_end == 0 { base_end = parts.len(); }
    let mut byte_pos = 0;
    for i in 0..base_end {
        byte_pos += parts[i].len();
        if i < base_end - 1 { byte_pos += 1; }
    }
    (&s[..byte_pos], regs)
}

// ============================================================
// CFG construction
// ============================================================

/// Build a CFG from the VRegInst stream.
pub fn build_cfg(insts: &[VRegInst]) -> (
    Vec<CfgBlock>,
    HashMap<u32, usize>,  // block_ends: label -> End index
    HashMap<u32, usize>,  // loop_starts: label -> Loop index
    HashMap<u32, usize>,  // else_positions: label -> Else index
) {
    if insts.is_empty() {
        return (vec![], HashMap::new(), HashMap::new(), HashMap::new());
    }

    // Step 1: Build control flow maps
    let mut block_ends: HashMap<u32, usize> = HashMap::new();
    let mut loop_starts: HashMap<u32, usize> = HashMap::new();
    let mut else_positions: HashMap<u32, usize> = HashMap::new();
    let mut cf_stack: Vec<u32> = Vec::new();

    for (i, inst) in insts.iter().enumerate() {
        match inst {
            VRegInst::Block { label } => cf_stack.push(*label),
            VRegInst::Loop { label } => { loop_starts.insert(*label, i); cf_stack.push(*label); }
            VRegInst::If { label, .. } => cf_stack.push(*label),
            VRegInst::Else { label } => { else_positions.insert(*label, i); }
            VRegInst::End { .. } => { if let Some(l) = cf_stack.pop() { block_ends.insert(l, i); } }
            _ => {}
        }
    }

    // Helper: resolve a WASM label to the instruction index it branches to
    let branch_target = |label: u32| -> Option<usize> {
        if let Some(&start) = loop_starts.get(&label) {
            Some(start + 1)
        } else if let Some(&end) = block_ends.get(&label) {
            if end + 1 < insts.len() { Some(end + 1) } else { None }
        } else {
            None
        }
    };

    // Step 2: Collect split points (positions where a new basic block starts)
    let mut splits: BTreeSet<usize> = BTreeSet::new();
    splits.insert(0);

    for (i, inst) in insts.iter().enumerate() {
        match inst {
            VRegInst::Loop { .. } => {
                if i + 1 < insts.len() { splits.insert(i + 1); }
            }
            VRegInst::If { label, .. } => {
                if i + 1 < insts.len() { splits.insert(i + 1); }
                if let Some(&ep) = else_positions.get(label) {
                    if ep + 1 < insts.len() { splits.insert(ep + 1); }
                }
                if let Some(&end) = block_ends.get(label) {
                    if end + 1 < insts.len() { splits.insert(end + 1); }
                }
            }
            VRegInst::Else { .. } => {
                if i + 1 < insts.len() { splits.insert(i + 1); }
            }
            VRegInst::End { .. } => {
                if i + 1 < insts.len() { splits.insert(i + 1); }
            }
            VRegInst::BrIf { label, .. } => {
                if i + 1 < insts.len() { splits.insert(i + 1); }
                if let Some(t) = branch_target(*label) { splits.insert(t); }
            }
            VRegInst::Br { label } => {
                if i + 1 < insts.len() { splits.insert(i + 1); }
                if let Some(t) = branch_target(*label) { splits.insert(t); }
            }
            VRegInst::BrTable { labels, default, .. } => {
                if i + 1 < insts.len() { splits.insert(i + 1); }
                for l in labels {
                    if let Some(t) = branch_target(*l) { splits.insert(t); }
                }
                if let Some(t) = branch_target(*default) { splits.insert(t); }
            }
            VRegInst::Return { .. } | VRegInst::Unreachable => {
                if i + 1 < insts.len() { splits.insert(i + 1); }
            }
            _ => {}
        }
    }

    let split_vec: Vec<usize> = splits.into_iter().filter(|&s| s < insts.len()).collect();

    // Step 3: Build block contents
    let mut blocks: Vec<CfgBlock> = Vec::new();
    let mut split_to_block: HashMap<usize, usize> = HashMap::new();

    for (bi, &start) in split_vec.iter().enumerate() {
        let end = if bi + 1 < split_vec.len() { split_vec[bi + 1] } else { insts.len() };
        split_to_block.insert(start, bi);

        let mut ra2_insts = Vec::new();
        for j in start..end {
            if !is_pure_marker(&insts[j]) {
                ra2_insts.push(j);
            }
        }

        blocks.push(CfgBlock {
            ra2_insts,
            succs: Vec::new(),
            preds: Vec::new(),
            orig_start: start,
            orig_end: end,
            is_trampoline: false,
        });
    }

    // Step 4: Compute successor edges
    for bi in 0..blocks.len() {
        let orig_end = blocks[bi].orig_end;
        let last_ra2 = blocks[bi].ra2_insts.last().copied();

        let succs = if let Some(last_idx) = last_ra2 {
            match &insts[last_idx] {
                VRegInst::Return { .. } | VRegInst::Unreachable => vec![],

                VRegInst::Br { label } => {
                    branch_target(*label)
                        .and_then(|t| split_to_block.get(&t).copied())
                        .into_iter().collect()
                }

                VRegInst::BrIf { label, .. } => {
                    let mut s = vec![];
                    if let Some(t) = branch_target(*label) {
                        if let Some(&tbi) = split_to_block.get(&t) { s.push(tbi); }
                    }
                    if let Some(&fbi) = split_to_block.get(&orig_end) { s.push(fbi); }
                    s
                }

                VRegInst::If { label, .. } => {
                    let mut s = vec![];
                    if let Some(&tbi) = split_to_block.get(&(last_idx + 1)) { s.push(tbi); }
                    if let Some(&ep) = else_positions.get(label) {
                        if let Some(&ebi) = split_to_block.get(&(ep + 1)) { s.push(ebi); }
                    } else if let Some(&end_pos) = block_ends.get(label) {
                        if let Some(&abi) = split_to_block.get(&(end_pos + 1)) { s.push(abi); }
                    }
                    s
                }

                VRegInst::BrTable { labels, default, .. } => {
                    let mut s = vec![];
                    for l in labels {
                        if let Some(t) = branch_target(*l) {
                            if let Some(&tbi) = split_to_block.get(&t) {
                                if !s.contains(&tbi) { s.push(tbi); }
                            }
                        }
                    }
                    if let Some(t) = branch_target(*default) {
                        if let Some(&tbi) = split_to_block.get(&t) {
                            if !s.contains(&tbi) { s.push(tbi); }
                        }
                    }
                    s
                }

                _ => {
                    compute_fallthrough(&blocks[bi], insts, &else_positions, &block_ends, &split_to_block)
                }
            }
        } else {
            compute_fallthrough(&blocks[bi], insts, &else_positions, &block_ends, &split_to_block)
        };

        blocks[bi].succs = succs;
    }

    // Step 5: Compute predecessors
    for bi in 0..blocks.len() {
        let succs = blocks[bi].succs.clone();
        for si in succs {
            if si < blocks.len() { blocks[si].preds.push(bi); }
        }
    }

    (blocks, block_ends, loop_starts, else_positions)
}

/// Compute the fallthrough successor for a block ending with a non-branch instruction.
fn compute_fallthrough(
    block: &CfgBlock,
    insts: &[VRegInst],
    _else_positions: &HashMap<u32, usize>,
    block_ends: &HashMap<u32, usize>,
    split_to_block: &HashMap<usize, usize>,
) -> Vec<usize> {
    let last_data = block.ra2_insts.last().copied().unwrap_or(block.orig_start);

    for j in (last_data + 1)..block.orig_end {
        if let VRegInst::Else { label } = &insts[j] {
            if let Some(&end_pos) = block_ends.get(label) {
                if let Some(&tbi) = split_to_block.get(&(end_pos + 1)) {
                    return vec![tbi];
                }
            }
        }
    }

    if block.orig_end < insts.len() {
        if let VRegInst::Else { label } = &insts[block.orig_end] {
            if let Some(&end_pos) = block_ends.get(label) {
                if let Some(&tbi) = split_to_block.get(&(end_pos + 1)) {
                    return vec![tbi];
                }
            }
        }
    }

    split_to_block.get(&block.orig_end).copied().into_iter().collect()
}

/// Split critical edges by inserting trampoline blocks.
pub fn split_critical_edges(blocks: &mut Vec<CfgBlock>) {
    let mut to_split: Vec<(usize, usize, usize)> = Vec::new();

    for bi in 0..blocks.len() {
        if blocks[bi].succs.len() <= 1 { continue; }
        for si in 0..blocks[bi].succs.len() {
            let target = blocks[bi].succs[si];
            if blocks[target].preds.len() > 1 {
                to_split.push((bi, si, target));
            }
        }
    }

    for (src, succ_idx, target) in to_split {
        let tramp_bi = blocks.len();
        blocks.push(CfgBlock {
            ra2_insts: vec![],
            succs: vec![target],
            preds: vec![src],
            orig_start: usize::MAX,
            orig_end: usize::MAX,
            is_trampoline: true,
        });
        blocks[src].succs[succ_idx] = tramp_bi;
        if let Some(pos) = blocks[target].preds.iter().position(|&p| p == src) {
            blocks[target].preds[pos] = tramp_bi;
        }
    }
}

// ============================================================
// Ra2FuncMulti — regalloc2 Function implementation
// ============================================================

/// Multi-block regalloc2 function.
pub struct Ra2FuncMulti {
    // Flat ra2 instruction data (contiguous per block)
    pub orig_indices: Vec<usize>,
    pub operands: Vec<Vec<Operand>>,
    is_branch_flag: Vec<bool>,
    is_ret_flag: Vec<bool>,

    // Per-block data
    pub block_inst_ranges: Vec<(usize, usize)>,
    block_succs_storage: Vec<Vec<ra2::Block>>,
    block_preds_storage: Vec<Vec<ra2::Block>>,
    entry_params: Vec<ra2::VReg>,

    // VReg mapping
    pub num_vregs: usize,
    _vreg_map: HashMap<VReg, usize>,
    pub vreg_reverse: Vec<VReg>,

    // CFG info (for rewriter)
    pub cfg_blocks: Vec<CfgBlock>,
    pub block_ends_map: HashMap<u32, usize>,
    pub loop_starts_map: HashMap<u32, usize>,
    pub else_positions_map: HashMap<u32, usize>,
}

impl Ra2FuncMulti {
    /// Build a multi-block regalloc2 function from a VRegInst stream.
    /// Returns None if the function contains BrTable (fall back to linear scan).
    pub fn build(
        insts: &[VRegInst],
        num_regs: u32,
        isa_by_base: &HashMap<String, Vec<Vec<u8>>>,
        unconstrained_alloc: &RegAllocResult,
    ) -> Option<Self> {
        for inst in insts {
            if matches!(inst, VRegInst::BrTable { .. }) { return None; }
        }

        let (mut blocks, block_ends_map, loop_starts_map, else_positions_map) = build_cfg(insts);
        if blocks.is_empty() { return None; }

        split_critical_edges(&mut blocks);

        // Build VReg mapping
        let mut vreg_set: HashSet<VReg> = HashSet::new();
        for inst in insts {
            for v in vreg_dst_regs(inst) { vreg_set.insert(v); }
            for v in vreg_src_regs(inst) { vreg_set.insert(v); }
        }
        let mut vreg_list: Vec<VReg> = vreg_set.into_iter().collect();
        vreg_list.sort_by_key(|v| v.0);
        let mut vreg_map: HashMap<VReg, usize> = HashMap::new();
        let mut vreg_reverse: Vec<VReg> = Vec::new();
        for (i, v) in vreg_list.iter().enumerate() {
            vreg_map.insert(*v, i);
            vreg_reverse.push(*v);
        }
        let num_vregs = vreg_reverse.len();

        let to_ra2_vreg = |v: VReg| -> ra2::VReg {
            ra2::VReg::new(vreg_map[&v], ra2::RegClass::Int)
        };

        // Build flat instruction array
        let mut orig_indices: Vec<usize> = Vec::new();
        let mut operands_list: Vec<Vec<Operand>> = Vec::new();
        let mut is_branch_flag: Vec<bool> = Vec::new();
        let mut is_ret_flag: Vec<bool> = Vec::new();
        let mut block_inst_ranges: Vec<(usize, usize)> = Vec::new();

        for block in &blocks {
            let start = orig_indices.len();

            for &orig_idx in &block.ra2_insts {
                let inst = &insts[orig_idx];
                let dsts = vreg_dst_regs(inst);
                let srcs = vreg_src_regs(inst);

                let is_ret = matches!(inst, VRegInst::Return { .. });
                let is_branch = is_block_terminator(inst) && !is_ret;

                let mut ops = Vec::new();
                if is_branch || is_ret {
                    for v in &dsts { ops.push(Operand::reg_def(to_ra2_vreg(*v))); }
                    for v in &srcs { ops.push(Operand::reg_use(to_ra2_vreg(*v))); }
                } else {
                    let spec = specialized_opcode(inst, unconstrained_alloc);
                    let (base, _) = parse_isa_opcode(&spec);
                    let base_str = base.to_string();
                    let target_regs = isa_by_base.get(&base_str).and_then(|v| v.first().cloned());
                    let use_fixed = target_regs.as_ref()
                        .map(|t| t.len() == dsts.len() + srcs.len() && t.iter().all(|&r| r < num_regs as u8))
                        .unwrap_or(false);

                    if use_fixed {
                        let target = target_regs.as_ref().unwrap();
                        for (j, v) in dsts.iter().enumerate() {
                            let preg = ra2::PReg::new(target[j] as usize, ra2::RegClass::Int);
                            ops.push(Operand::new(to_ra2_vreg(*v), OperandConstraint::FixedReg(preg),
                                OperandKind::Def, OperandPos::Late));
                        }
                        for (j, v) in srcs.iter().enumerate() {
                            let preg = ra2::PReg::new(target[dsts.len() + j] as usize, ra2::RegClass::Int);
                            ops.push(Operand::new(to_ra2_vreg(*v), OperandConstraint::FixedReg(preg),
                                OperandKind::Use, OperandPos::Early));
                        }
                    } else {
                        for v in &dsts { ops.push(Operand::reg_def(to_ra2_vreg(*v))); }
                        for v in &srcs { ops.push(Operand::reg_use(to_ra2_vreg(*v))); }
                    }
                }

                orig_indices.push(orig_idx);
                operands_list.push(ops);
                is_branch_flag.push(is_branch || is_ret);
                is_ret_flag.push(is_ret);
            }

            // Synthetic fallthrough branch if needed
            let need_synthetic = if block.ra2_insts.is_empty() {
                true
            } else {
                let last = *block.ra2_insts.last().unwrap();
                !is_block_terminator(&insts[last])
            };

            if need_synthetic {
                orig_indices.push(usize::MAX);
                operands_list.push(vec![]);
                is_branch_flag.push(true);
                is_ret_flag.push(false);
            }

            let end = orig_indices.len();
            block_inst_ranges.push((start, end));
        }

        // Build block succs/preds as ra2::Block
        let block_succs_storage: Vec<Vec<ra2::Block>> = blocks.iter()
            .map(|b| b.succs.iter().map(|&s| ra2::Block::new(s)).collect())
            .collect();
        let block_preds_storage: Vec<Vec<ra2::Block>> = blocks.iter()
            .map(|b| b.preds.iter().map(|&p| ra2::Block::new(p)).collect())
            .collect();

        // Compute entry_params: vregs Used before Def'd in block 0
        let mut defined: HashSet<usize> = HashSet::new();
        let mut live_in: Vec<ra2::VReg> = Vec::new();
        let mut live_in_set: HashSet<usize> = HashSet::new();
        let (b0_start, b0_end) = block_inst_ranges[0];
        for idx in b0_start..b0_end {
            for op in &operands_list[idx] {
                if op.kind() == OperandKind::Use {
                    let vi = op.vreg().vreg();
                    if !defined.contains(&vi) && !live_in_set.contains(&vi) {
                        live_in.push(op.vreg());
                        live_in_set.insert(vi);
                    }
                }
            }
            for op in &operands_list[idx] {
                if op.kind() == OperandKind::Def { defined.insert(op.vreg().vreg()); }
            }
        }

        Some(Ra2FuncMulti {
            orig_indices,
            operands: operands_list,
            is_branch_flag,
            is_ret_flag,
            block_inst_ranges,
            block_succs_storage,
            block_preds_storage,
            entry_params: live_in,
            num_vregs,
            _vreg_map: vreg_map,
            vreg_reverse,
            cfg_blocks: blocks,
            block_ends_map,
            loop_starts_map,
            else_positions_map,
        })
    }
}

impl ra2::Function for Ra2FuncMulti {
    fn num_insts(&self) -> usize { self.orig_indices.len() }
    fn num_blocks(&self) -> usize { self.block_inst_ranges.len() }
    fn entry_block(&self) -> ra2::Block { ra2::Block::new(0) }

    fn block_insns(&self, block: ra2::Block) -> ra2::InstRange {
        let (start, end) = self.block_inst_ranges[block.index()];
        ra2::InstRange::new(ra2::Inst::new(start), ra2::Inst::new(end))
    }

    fn block_succs(&self, block: ra2::Block) -> &[ra2::Block] {
        &self.block_succs_storage[block.index()]
    }
    fn block_preds(&self, block: ra2::Block) -> &[ra2::Block] {
        &self.block_preds_storage[block.index()]
    }
    fn block_params(&self, block: ra2::Block) -> &[ra2::VReg] {
        if block.index() == 0 { &self.entry_params } else { &[] }
    }

    fn is_ret(&self, insn: ra2::Inst) -> bool { self.is_ret_flag[insn.index()] }
    fn is_branch(&self, insn: ra2::Inst) -> bool { self.is_branch_flag[insn.index()] }

    fn branch_blockparams(&self, _block: ra2::Block, _insn: ra2::Inst, _succ_idx: usize) -> &[ra2::VReg] {
        &[]
    }

    fn inst_operands(&self, insn: ra2::Inst) -> &[Operand] {
        &self.operands[insn.index()]
    }

    fn inst_clobbers(&self, _insn: ra2::Inst) -> ra2::PRegSet { ra2::PRegSet::empty() }
    fn num_vregs(&self) -> usize { self.num_vregs }
    fn spillslot_size(&self, _regclass: ra2::RegClass) -> usize { 1 }
}

// ============================================================
// Convenience builders
// ============================================================

/// Build a MachineEnv for the given number of integer registers.
pub fn make_machine_env(num_regs: u32) -> MachineEnv {
    let mut preferred_regs = Vec::new();
    for r in 0..num_regs {
        preferred_regs.push(ra2::PReg::new(r as usize, ra2::RegClass::Int));
    }
    MachineEnv {
        preferred_regs_by_class: [preferred_regs, vec![], vec![]],
        non_preferred_regs_by_class: [vec![], vec![], vec![]],
        scratch_by_class: [None, None, None],
        fixed_stack_slots: vec![],
    }
}

/// Build default regalloc options (Ion algorithm, no verbose log).
pub fn make_regalloc_opts() -> RegallocOptions {
    RegallocOptions {
        verbose_log: false,
        validate_ssa: false,
        algorithm: Algorithm::Ion,
    }
}

// ============================================================
// Rewriter helpers
// ============================================================

/// Create a fresh VReg mapped to a regalloc2 allocation.
pub fn make_vreg_from_alloc(
    alloc: ra2::Allocation,
    v2p: &mut HashMap<VReg, OurPReg>,
    sp: &mut HashSet<VReg>,
    ss: &mut HashMap<VReg, SpillSlot>,
    next_id: &mut u32,
) -> VReg {
    let v = VReg(*next_id);
    *next_id += 1;
    if alloc.is_reg() {
        let preg = alloc.as_reg().unwrap();
        v2p.insert(v, OurPReg(preg.hw_enc() as u8));
    } else if alloc.is_stack() {
        let slot = alloc.as_stack().unwrap();
        sp.insert(v);
        ss.insert(v, SpillSlot(slot.index() as u32));
    }
    v
}

/// Emit Mov instructions for a list of regalloc2 edit moves, updating current_loc.
pub fn emit_edit_moves(
    moves: &[(ra2::Allocation, ra2::Allocation)],
    new_insts: &mut Vec<VRegInst>,
    v2p: &mut HashMap<VReg, OurPReg>,
    sp: &mut HashSet<VReg>,
    ss: &mut HashMap<VReg, SpillSlot>,
    current_loc: &mut HashMap<VReg, ra2::Allocation>,
    next_id: &mut u32,
) {
    for (from, to) in moves {
        let moved: Vec<VReg> = current_loc.iter()
            .filter(|(_, loc)| **loc == *from)
            .map(|(v, _)| *v)
            .collect();
        for v in &moved { current_loc.insert(*v, *to); }
        let src = make_vreg_from_alloc(*from, v2p, sp, ss, next_id);
        let dst = make_vreg_from_alloc(*to, v2p, sp, ss, next_id);
        new_insts.push(VRegInst::Mov { dst, src });
    }
}

// ============================================================
// Multi-block rewriter
// ============================================================

/// Apply regalloc2 allocations to produce a rewritten VRegInst stream.
///
/// Walks all original instructions (including control flow markers),
/// applies per-instruction allocations from regalloc2, inserts edge moves
/// from trampolines, and handles BrIf-to-If conversion for taken-edge trampolines.
///
/// Returns (rewritten_instructions, new_allocation_map).
pub fn rewrite_with_ra2(
    insts: &[VRegInst],
    ra2_func: &Ra2FuncMulti,
    output: &ra2::Output,
    orig_alloc: &RegAllocResult,
) -> (Vec<VRegInst>, RegAllocResult) {
    let mut vreg_to_preg: HashMap<VReg, OurPReg> = HashMap::new();
    let mut spilled: HashSet<VReg> = HashSet::new();
    let mut spill_slots_map: HashMap<VReg, SpillSlot> = HashMap::new();
    let mut new_insts: Vec<VRegInst> = Vec::new();
    let mut next_vreg_id = insts.iter()
        .flat_map(|i| vreg_dst_regs(i).into_iter().chain(vreg_src_regs(i)))
        .map(|v| v.0)
        .max()
        .unwrap_or(0) + 1;

    // Collect edits by regalloc2 instruction index
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

    // Reverse mapping: original instruction index -> regalloc2 flat index
    let mut orig_to_ra2: HashMap<usize, usize> = HashMap::new();
    for (ra2_idx, &orig_idx) in ra2_func.orig_indices.iter().enumerate() {
        if orig_idx != usize::MAX {
            orig_to_ra2.insert(orig_idx, ra2_idx);
        }
    }

    // Collect trampoline info
    struct TrampolineInfo {
        moves: Vec<(ra2::Allocation, ra2::Allocation)>,
    }
    let mut trampoline_infos: Vec<TrampolineInfo> = Vec::new();
    let mut trampoline_at_term: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    let mut trampoline_at_else: HashMap<usize, usize> = HashMap::new();

    for (bi, block) in ra2_func.cfg_blocks.iter().enumerate() {
        if !block.is_trampoline { continue; }

        let src_block = block.preds[0];
        let succ_idx = ra2_func.cfg_blocks[src_block].succs.iter()
            .position(|&s| s == bi)
            .unwrap_or(0);

        let (t_start, t_end) = ra2_func.block_inst_ranges[bi];
        let mut moves = Vec::new();
        for ra2_idx in t_start..t_end {
            if let Some(m) = edits_before.get(&ra2_idx) { moves.extend(m.iter().cloned()); }
            if let Some(m) = edits_after.get(&ra2_idx) { moves.extend(m.iter().cloned()); }
        }

        if moves.is_empty() { continue; }

        let ti_idx = trampoline_infos.len();
        trampoline_infos.push(TrampolineInfo { moves });

        let src = &ra2_func.cfg_blocks[src_block];
        if let Some(&last_ra2_orig) = src.ra2_insts.last() {
            match &insts[last_ra2_orig] {
                VRegInst::If { label, .. } if succ_idx == 1 => {
                    if let Some(&else_pos) = ra2_func.else_positions_map.get(label) {
                        trampoline_at_else.insert(else_pos, ti_idx);
                    }
                }
                _ => {
                    trampoline_at_term.entry(last_ra2_orig).or_default().push((succ_idx, ti_idx));
                }
            }
        }
    }

    // Track where each original vreg currently lives
    let mut current_loc: HashMap<VReg, ra2::Allocation> = HashMap::new();

    // Fresh label counter for synthetic If/End blocks
    let mut next_synth_label = insts.iter()
        .filter_map(|i| match i {
            VRegInst::Block { label } | VRegInst::Loop { label }
            | VRegInst::If { label, .. } | VRegInst::Else { label }
            | VRegInst::End { label } => Some(*label),
            _ => None,
        })
        .max()
        .unwrap_or(0) + 1000;

    // Walk ALL original instructions
    for (orig_idx, inst) in insts.iter().enumerate() {
        if let Some(&ra2_idx) = orig_to_ra2.get(&orig_idx) {
            // This instruction was in regalloc2's scope

            let brif_taken_tramp = if let VRegInst::BrIf { .. } = inst {
                trampoline_at_term.get(&orig_idx)
                    .and_then(|v| v.iter().find(|(si, _)| *si == 0))
                    .map(|(_, ti)| *ti)
            } else {
                None
            };

            // Insert edit moves BEFORE
            if let Some(moves) = edits_before.get(&ra2_idx) {
                emit_edit_moves(moves, &mut new_insts, &mut vreg_to_preg,
                    &mut spilled, &mut spill_slots_map, &mut current_loc, &mut next_vreg_id);
            }

            // Rewrite instruction with per-instruction allocations
            let allocs = output.inst_allocs(ra2::Inst::new(ra2_idx));
            let orig_dsts = vreg_dst_regs(inst);
            let orig_srcs = vreg_src_regs(inst);

            let mut new_dsts = Vec::new();
            let mut new_srcs = Vec::new();
            let mut ai = 0;
            for _ in &orig_dsts {
                if ai < allocs.len() {
                    new_dsts.push(make_vreg_from_alloc(allocs[ai], &mut vreg_to_preg,
                        &mut spilled, &mut spill_slots_map, &mut next_vreg_id));
                }
                ai += 1;
            }
            for _ in &orig_srcs {
                if ai < allocs.len() {
                    new_srcs.push(make_vreg_from_alloc(allocs[ai], &mut vreg_to_preg,
                        &mut spilled, &mut spill_slots_map, &mut next_vreg_id));
                }
                ai += 1;
            }

            if let (Some(ti_idx), VRegInst::BrIf { label, .. }) = (brif_taken_tramp, inst) {
                // BrIf with taken-edge trampoline: convert to If/Br/End
                let synth_label = next_synth_label;
                next_synth_label += 1;

                let cond_vreg = if !new_srcs.is_empty() { new_srcs[0] }
                    else { vreg_src_regs(inst).into_iter().next().unwrap_or(VReg(0)) };

                new_insts.push(VRegInst::If { cond: cond_vreg, label: synth_label });

                // Update current_loc
                let allocs2 = output.inst_allocs(ra2::Inst::new(ra2_idx));
                let ops2 = &ra2_func.operands[ra2_idx];
                for (op_idx, op) in ops2.iter().enumerate() {
                    if op_idx >= allocs2.len() { continue; }
                    let our_vreg = ra2_func.vreg_reverse[op.vreg().vreg()];
                    current_loc.insert(our_vreg, allocs2[op_idx]);
                }

                let tramp = &trampoline_infos[ti_idx];
                emit_edit_moves(&tramp.moves, &mut new_insts, &mut vreg_to_preg,
                    &mut spilled, &mut spill_slots_map, &mut current_loc, &mut next_vreg_id);

                new_insts.push(VRegInst::Br { label: *label });
                new_insts.push(VRegInst::End { label: synth_label });

                if let Some(moves) = edits_after.get(&ra2_idx) {
                    emit_edit_moves(moves, &mut new_insts, &mut vreg_to_preg,
                        &mut spilled, &mut spill_slots_map, &mut current_loc, &mut next_vreg_id);
                }
            } else {
                // Normal instruction rewrite
                if new_dsts.len() == orig_dsts.len() && new_srcs.len() == orig_srcs.len() {
                    new_insts.push(rewrite_inst_vregs(inst, &new_dsts, &new_srcs));
                } else {
                    new_insts.push(inst.clone());
                }

                // Update current_loc
                let allocs2 = output.inst_allocs(ra2::Inst::new(ra2_idx));
                let ops2 = &ra2_func.operands[ra2_idx];
                for (op_idx, op) in ops2.iter().enumerate() {
                    if op_idx >= allocs2.len() { continue; }
                    let our_vreg = ra2_func.vreg_reverse[op.vreg().vreg()];
                    current_loc.insert(our_vreg, allocs2[op_idx]);
                }

                if let Some(moves) = edits_after.get(&ra2_idx) {
                    emit_edit_moves(moves, &mut new_insts, &mut vreg_to_preg,
                        &mut spilled, &mut spill_slots_map, &mut current_loc, &mut next_vreg_id);
                }

                // If then-edge trampoline (succ_idx 0)
                if let VRegInst::If { .. } = inst {
                    if let Some(v) = trampoline_at_term.get(&orig_idx) {
                        for &(si, ti_idx) in v {
                            if si == 0 {
                                let tramp = &trampoline_infos[ti_idx];
                                emit_edit_moves(&tramp.moves, &mut new_insts, &mut vreg_to_preg,
                                    &mut spilled, &mut spill_slots_map, &mut current_loc, &mut next_vreg_id);
                            }
                        }
                    }
                }

                // BrIf fallthrough-edge trampoline (succ_idx 1)
                if let VRegInst::BrIf { .. } = inst {
                    if let Some(v) = trampoline_at_term.get(&orig_idx) {
                        for &(si, ti_idx) in v {
                            if si == 1 {
                                let tramp = &trampoline_infos[ti_idx];
                                emit_edit_moves(&tramp.moves, &mut new_insts, &mut vreg_to_preg,
                                    &mut spilled, &mut spill_slots_map, &mut current_loc, &mut next_vreg_id);
                            }
                        }
                    }
                }
            }
        } else {
            // Control flow marker — not in regalloc2's scope
            for v in vreg_dst_regs(inst).iter().chain(vreg_src_regs(inst).iter()) {
                if let Some(alloc) = current_loc.get(v) {
                    if alloc.is_reg() {
                        let preg = alloc.as_reg().unwrap();
                        vreg_to_preg.insert(*v, OurPReg(preg.hw_enc() as u8));
                    } else if alloc.is_stack() {
                        let slot = alloc.as_stack().unwrap();
                        spilled.insert(*v);
                        spill_slots_map.insert(*v, SpillSlot(slot.index() as u32));
                    }
                } else if let Some(&preg) = orig_alloc.vreg_to_preg.get(v) {
                    vreg_to_preg.entry(*v).or_insert(preg);
                }
            }
            new_insts.push(inst.clone());

            // Else-edge trampoline
            if let VRegInst::Else { .. } = inst {
                if let Some(&ti_idx) = trampoline_at_else.get(&orig_idx) {
                    let tramp = &trampoline_infos[ti_idx];
                    emit_edit_moves(&tramp.moves, &mut new_insts, &mut vreg_to_preg,
                        &mut spilled, &mut spill_slots_map, &mut current_loc, &mut next_vreg_id);
                }
            }
        }
    }

    let new_alloc = RegAllocResult {
        vreg_to_preg,
        spilled,
        spill_slots: spill_slots_map,
        num_spill_slots: output.num_spillslots as u32,
    };

    (new_insts, new_alloc)
}

/// Validate a rewritten VRegInst stream: count vregs with no allocation mapping.
pub fn validate_rewrite(
    insts: &[VRegInst],
    alloc: &RegAllocResult,
) -> u32 {
    let mut unmapped = 0u32;
    for inst in insts {
        for v in vreg_src_regs(inst) {
            if !alloc.vreg_to_preg.contains_key(&v) && !alloc.spill_slots.contains_key(&v) {
                unmapped += 1;
            }
        }
        for v in vreg_dst_regs(inst) {
            if !alloc.vreg_to_preg.contains_key(&v) && !alloc.spill_slots.contains_key(&v) {
                unmapped += 1;
            }
        }
    }
    unmapped
}
