//! ISA-constrained register allocation using regalloc2.
//!
//! Pipeline:
//! 1. Compile WASM → VRegInst (ML mode)
//! 2. Profile with unconstrained allocation → ISA (top-K opcodes)
//! 3. Re-allocate using regalloc2 with FixedReg constraints for ISA compliance
//! 4. Execute and verify 194 nodes
//!
//! Usage: isa_regalloc [num_regs] [isa_budget] [wasm_path]

use anyhow::{Context, Result, anyhow};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs;
use wasmparser::{Parser, Payload};
use reg_analyzer::regvm::{
    WasmToVReg, VRegInst, VReg, PReg as OurPReg, FuncSig, SlotType, SpillSlot,
    GLOBALS_MEM_BASE, FRAME_SP_ADDR, FRAME_STACK_BASE, SLOT_SIZE,
};
use reg_analyzer::regalloc::{compute_live_intervals, linear_scan_alloc, targeted_linear_scan_alloc, RegAllocResult};
use reg_analyzer::preg_vm::{
    PRegVM, vreg_inst_name, vreg_dst_regs, vreg_src_regs, specialized_opcode, rewrite_inst_vregs,
};
use reg_analyzer::interpreter::Value;

// regalloc2 types
use regalloc2::{
    self as ra2, MachineEnv, Operand, OperandConstraint, OperandKind, OperandPos,
    RegallocOptions, Algorithm,
};

fn val_type_to_slot(vt: &wasmparser::ValType) -> SlotType {
    match vt {
        wasmparser::ValType::I64 => SlotType::I64,
        _ => SlotType::I32,
    }
}

fn generate_test_json(target_size: usize) -> String {
    let mut json = String::from(r#"{"data":{"users":["#);
    let user_template = r#"{"name":"user_XXX","email":"user_XXX@example.com","age":25,"active":true,"tags":["a","b","c"]}"#;
    let mut i = 0;
    while json.len() < target_size {
        if i > 0 { json.push(','); }
        json.push_str(&user_template.replace("XXX", &format!("{:04}", i)));
        i += 1;
    }
    json.push_str(r#"]},"meta":{"count":"#);
    json.push_str(&i.to_string());
    json.push_str(r#"}}"#);
    json
}

fn parse_isa_opcode(s: &str) -> (&str, Vec<u8>) {
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

// === Multi-block CFG for regalloc2 ===

/// Basic block in the control flow graph.
#[derive(Debug, Clone)]
struct CfgBlock {
    /// Indices into original VRegInst stream for this block's ra2 instructions.
    /// Includes data instructions and branch terminators (If/BrIf/Br/Return).
    /// Does NOT include pure markers (Block/Loop/Else/End).
    ra2_insts: Vec<usize>,
    /// Successor block indices
    succs: Vec<usize>,
    /// Predecessor block indices
    preds: Vec<usize>,
    /// Range [start, end) of original VRegInst indices covered by this block
    orig_start: usize,
    orig_end: usize,
    /// Is this a synthetic trampoline block?
    is_trampoline: bool,
}

/// Returns true if the instruction is a pure structural marker (no vreg operands).
fn is_pure_marker(inst: &VRegInst) -> bool {
    matches!(inst, VRegInst::Block { .. } | VRegInst::Loop { .. } |
             VRegInst::Else { .. } | VRegInst::End { .. })
}

/// Returns true if the instruction is a branch terminator for a basic block.
fn is_block_terminator(inst: &VRegInst) -> bool {
    matches!(inst, VRegInst::If { .. } | VRegInst::BrIf { .. } | VRegInst::Br { .. } |
             VRegInst::BrTable { .. } | VRegInst::Return { .. } | VRegInst::Unreachable)
}

/// Build a CFG from the VRegInst stream.
fn build_cfg(insts: &[VRegInst]) -> (
    Vec<CfgBlock>,
    HashMap<u32, usize>,  // block_ends: label → End index
    HashMap<u32, usize>,  // loop_starts: label → Loop index
    HashMap<u32, usize>,  // else_positions: label → Else index
) {
    if insts.is_empty() {
        return (vec![], HashMap::new(), HashMap::new(), HashMap::new());
    }

    // Step 1: Build control flow maps (same as execute_vreg)
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
            Some(start + 1) // Loop body start (instruction after Loop marker)
        } else if let Some(&end) = block_ends.get(&label) {
            if end + 1 < insts.len() { Some(end + 1) } else { None }
        } else {
            None
        }
    };

    // Step 2: Collect split points (positions where a new basic block starts)
    let mut splits: BTreeSet<usize> = BTreeSet::new();
    splits.insert(0); // function entry

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

        // Collect ra2 instructions (non-markers)
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
                    // Successor 0: branch taken
                    if let Some(t) = branch_target(*label) {
                        if let Some(&tbi) = split_to_block.get(&t) { s.push(tbi); }
                    }
                    // Successor 1: fallthrough
                    if let Some(&fbi) = split_to_block.get(&orig_end) { s.push(fbi); }
                    s
                }

                VRegInst::If { label, .. } => {
                    let mut s = vec![];
                    // Successor 0: then-body (right after If)
                    if let Some(&tbi) = split_to_block.get(&(last_idx + 1)) { s.push(tbi); }
                    // Successor 1: else-body or after-end
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
                    // Data instruction at end → fallthrough
                    compute_fallthrough(&blocks[bi], &insts, &else_positions, &block_ends, &split_to_block)
                }
            }
        } else {
            // Empty block (only markers) → fallthrough
            compute_fallthrough(&blocks[bi], &insts, &else_positions, &block_ends, &split_to_block)
        };

        blocks[bi].succs = succs;
    }

    // Step 5: Compute predecessors (reverse of successors)
    for bi in 0..blocks.len() {
        let succs = blocks[bi].succs.clone();
        for si in succs {
            if si < blocks.len() { blocks[si].preds.push(bi); }
        }
    }

    (blocks, block_ends, loop_starts, else_positions)
}

/// Compute the fallthrough successor for a block ending with a non-branch instruction.
/// Handles the Else marker case: if Else is between last data inst and block end,
/// the fallthrough goes to after End (the Else acts as a jump).
fn compute_fallthrough(
    block: &CfgBlock,
    insts: &[VRegInst],
    else_positions: &HashMap<u32, usize>,
    block_ends: &HashMap<u32, usize>,
    split_to_block: &HashMap<usize, usize>,
) -> Vec<usize> {
    let last_data = block.ra2_insts.last().copied().unwrap_or(block.orig_start);

    // Check for Else marker between last data instruction and block end
    for j in (last_data + 1)..block.orig_end {
        if let VRegInst::Else { label } = &insts[j] {
            if let Some(&end_pos) = block_ends.get(label) {
                if let Some(&tbi) = split_to_block.get(&(end_pos + 1)) {
                    return vec![tbi];
                }
            }
        }
    }

    // Check if the next block starts with an Else marker
    if block.orig_end < insts.len() {
        if let VRegInst::Else { label } = &insts[block.orig_end] {
            if let Some(&end_pos) = block_ends.get(label) {
                if let Some(&tbi) = split_to_block.get(&(end_pos + 1)) {
                    return vec![tbi];
                }
            }
        }
    }

    // Simple fallthrough to next block
    split_to_block.get(&block.orig_end).copied().into_iter().collect()
}

/// Split critical edges by inserting trampoline blocks.
/// A critical edge is one where source has >1 successors AND target has >1 predecessors.
fn split_critical_edges(blocks: &mut Vec<CfgBlock>) {
    let mut to_split: Vec<(usize, usize, usize)> = Vec::new(); // (src, succ_idx, target)

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
            ra2_insts: vec![],  // just a synthetic branch
            succs: vec![target],
            preds: vec![src],
            orig_start: usize::MAX,
            orig_end: usize::MAX,
            is_trampoline: true,
        });

        // Update source's successor
        blocks[src].succs[succ_idx] = tramp_bi;

        // Update target's predecessor: replace ONE occurrence of src with tramp_bi
        if let Some(pos) = blocks[target].preds.iter().position(|&p| p == src) {
            blocks[target].preds[pos] = tramp_bi;
        }
    }
}

/// Multi-block regalloc2 function.
struct Ra2FuncMulti {
    // Flat ra2 instruction data (contiguous per block)
    orig_indices: Vec<usize>,
    operands: Vec<Vec<Operand>>,
    is_branch_flag: Vec<bool>,
    is_ret_flag: Vec<bool>,

    // Per-block data
    block_inst_ranges: Vec<(usize, usize)>,  // [start, end) in flat instruction array
    block_succs_storage: Vec<Vec<ra2::Block>>,
    block_preds_storage: Vec<Vec<ra2::Block>>,
    entry_params: Vec<ra2::VReg>,

    // VReg mapping
    num_vregs: usize,
    vreg_map: HashMap<VReg, usize>,
    vreg_reverse: Vec<VReg>,

    // CFG info (for rewriter)
    cfg_blocks: Vec<CfgBlock>,
    block_ends_map: HashMap<u32, usize>,
    loop_starts_map: HashMap<u32, usize>,
    else_positions_map: HashMap<u32, usize>,
}

impl Ra2FuncMulti {
    /// Build a multi-block regalloc2 function from a VRegInst stream.
    /// Returns None if the function contains BrTable (fall back to linear scan).
    fn build(
        insts: &[VRegInst],
        num_regs: u32,
        isa_by_base: &HashMap<String, Vec<Vec<u8>>>,
        unconstrained_alloc: &RegAllocResult,
    ) -> Option<Self> {
        // Fall back for BrTable (complex to handle)
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

        // Build flat instruction array: lay out blocks in order
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

                // Build operands — use ISA constraints for data instructions,
                // unconstrained Reg for branches
                let mut ops = Vec::new();
                if is_branch || is_ret {
                    // Branches/returns: unconstrained
                    for v in &dsts { ops.push(Operand::reg_def(to_ra2_vreg(*v))); }
                    for v in &srcs { ops.push(Operand::reg_use(to_ra2_vreg(*v))); }
                } else {
                    // Data instruction: try ISA FixedReg
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

            // If block has no ra2 instructions, or last isn't a branch/return, add synthetic
            let need_synthetic = if block.ra2_insts.is_empty() {
                true
            } else {
                let last = *block.ra2_insts.last().unwrap();
                !is_block_terminator(&insts[last])
            };

            if need_synthetic {
                // Synthetic fallthrough branch
                orig_indices.push(usize::MAX);
                operands_list.push(vec![]);
                is_branch_flag.push(true);
                is_ret_flag.push(false);
            }

            // If this is a trampoline, ensure it has a synthetic branch
            if block.is_trampoline && block.ra2_insts.is_empty() {
                // Already added above (need_synthetic was true)
            }

            let end = orig_indices.len();
            block_inst_ranges.push((start, end));
        }

        // Ensure the very last instruction globally is a branch or return
        // (regalloc2 requires this for the last block)
        if let Some(last_range) = block_inst_ranges.last() {
            if last_range.0 == last_range.1 {
                // Empty range, shouldn't happen after synthetic additions
            }
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
            vreg_map,
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

/// Create a fresh VReg mapped to a regalloc2 allocation.
fn make_vreg_from_alloc(
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
fn emit_edit_moves(
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

fn main() -> Result<()> {
    let num_regs: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let isa_budget: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let json_size: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let wasm_path = std::env::args().nth(4).unwrap_or_else(||
        "../json-crate-wasm/target/wasm32-unknown-unknown/release/json_crate_wasm.wasm".to_string());

    let wasm_bytes = fs::read(&wasm_path).context("Failed to read WASM file")?;

    // === Parse WASM (same boilerplate) ===
    let mut func_types: Vec<wasmparser::FuncType> = Vec::new();
    let mut type_indices: Vec<u32> = Vec::new();
    let mut func_names: HashMap<u32, String> = HashMap::new();
    let mut code_bodies: Vec<wasmparser::FunctionBody> = Vec::new();
    let mut global_inits: Vec<i32> = Vec::new();
    let mut global_types: Vec<SlotType> = Vec::new();
    let mut data_segments: Vec<(u32, Vec<u8>)> = Vec::new();

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
                for func in reader.clone() { type_indices.push(func?); }
            }
            Payload::GlobalSection(reader) => {
                for global in reader.clone() {
                    let global = global?;
                    let init_expr = global.init_expr.get_binary_reader();
                    let mut init_val = 0i32;
                    for op in wasmparser::OperatorsReader::new(init_expr) {
                        if let Ok(wasmparser::Operator::I32Const { value }) = op {
                            init_val = value;
                            break;
                        }
                    }
                    global_inits.push(init_val);
                    global_types.push(val_type_to_slot(&global.ty.content_type));
                }
            }
            Payload::ExportSection(reader) => {
                for export in reader.clone() {
                    let export = export?;
                    if let wasmparser::ExternalKind::Func = export.kind {
                        func_names.insert(export.index, export.name.to_string());
                    }
                }
            }
            Payload::CodeSectionEntry(body) => { code_bodies.push(body.clone()); }
            Payload::DataSection(reader) => {
                for data in reader.clone() {
                    let data = data?;
                    if let wasmparser::DataKind::Active { memory_index: 0, offset_expr } = data.kind {
                        let mut offset = 0u32;
                        for op in wasmparser::OperatorsReader::new(offset_expr.get_binary_reader()) {
                            if let Ok(wasmparser::Operator::I32Const { value }) = op {
                                offset = value as u32;
                                break;
                            }
                        }
                        data_segments.push((offset, data.data.to_vec()));
                    }
                }
            }
            _ => {}
        }
    }

    let func_sigs: Vec<FuncSig> = type_indices.iter().map(|&type_idx| {
        let func_type = func_types.get(type_idx as usize);
        let num_params = func_type.map(|ft| ft.params().len() as u32).unwrap_or(0);
        let num_results = func_type.map(|ft| ft.results().len() as u32).unwrap_or(0);
        (num_params, num_results)
    }).collect();

    // === Compile all functions (unconstrained, for profiling) ===
    let mut vreg_funcs: Vec<(Vec<VRegInst>, RegAllocResult, u32, u32)> = Vec::new();

    for (func_count, body) in code_bodies.iter().enumerate() {
        let type_idx = type_indices.get(func_count).copied().unwrap_or(0);
        let func_type = func_types.get(type_idx as usize);
        let num_params = func_type.map(|ft| ft.params().len() as u32).unwrap_or(0);
        let mut num_locals = 0u32;
        let mut local_types: Vec<SlotType> = Vec::new();
        if let Some(ft) = func_type {
            for p in ft.params() { local_types.push(val_type_to_slot(p)); }
        }
        for local in body.get_locals_reader()? {
            let (count, vt) = local?;
            num_locals += count;
            let st = val_type_to_slot(&vt);
            for _ in 0..count { local_types.push(st); }
        }

        let mut converter = WasmToVReg::new_memory_lowered(
            num_params, num_locals, func_sigs.clone(),
            local_types, global_types.clone(),
        );
        for op in body.get_operators_reader()? {
            converter.convert_op(&op?);
        }

        let intervals = compute_live_intervals(&converter.instructions);
        let alloc = linear_scan_alloc(&intervals, num_regs);
        vreg_funcs.push((converter.instructions, alloc, num_params, num_locals));
    }

    // === Phase 1: Profile run ===
    eprintln!("Phase 1: Profiling with {} regs...", num_regs);

    let mut max_spill_slots = 0u32;
    for (_, alloc, _, _) in &vreg_funcs {
        max_spill_slots = max_spill_slots.max(alloc.num_spill_slots);
    }
    let mut vm = PRegVM::new(num_regs as usize, max_spill_slots as usize + 64, 256);

    for (vreg_insts, alloc, num_params, num_locals) in &vreg_funcs {
        vm.add_vreg_function_ml(
            vreg_insts.clone(),
            RegAllocResult {
                vreg_to_preg: alloc.vreg_to_preg.clone(),
                spilled: alloc.spilled.clone(),
                spill_slots: alloc.spill_slots.clone(),
                num_spill_slots: alloc.num_spill_slots,
            },
            *num_params, *num_locals,
        );
    }

    for (i, val) in global_inits.iter().enumerate() {
        if i < vm.globals.len() { vm.globals[i] = Value::I32(*val); }
    }
    for (offset, data) in &data_segments {
        vm.write_memory(*offset as usize, data);
    }

    let test_json = generate_test_json(json_size);
    let json_data = test_json.as_bytes();
    vm.write_memory(0, json_data);
    vm.globals[0] = Value::I32(global_inits.first().copied().unwrap_or(1048576));

    vm.write_memory(FRAME_SP_ADDR as usize, &FRAME_STACK_BASE.to_le_bytes());
    for (i, val) in global_inits.iter().enumerate() {
        let addr = GLOBALS_MEM_BASE as usize + (i as u32 * SLOT_SIZE) as usize;
        vm.write_memory(addr, &(*val as u32).to_le_bytes());
    }
    let frame_base = FRAME_STACK_BASE as usize;
    vm.write_memory(frame_base, &0u32.to_le_bytes());
    vm.write_memory(frame_base + SLOT_SIZE as usize, &(json_data.len() as u32).to_le_bytes());

    vm.enable_reg_trace();

    let func_idx = func_names.iter()
        .find(|(_, name)| *name == "parse_json_deep")
        .map(|(idx, _)| *idx)
        .ok_or_else(|| anyhow!("No parse_json_deep"))?;

    let func = &vreg_funcs[func_idx as usize];
    let profile_result = vm.execute_vreg(&func.0, &func.1);
    let nodes_profile = profile_result.map(|v| v.as_i32() as u32).unwrap_or(0);
    let reg_trace = vm.reg_trace.take().unwrap_or_default();
    let trace_len = reg_trace.len();

    eprintln!("  Profile: {} nodes, {} trace instructions", nodes_profile, trace_len);
    assert!(nodes_profile > 0, "Profile run must produce >0 nodes");

    // === Phase 2: Build ISA ===
    let mov_budget = num_regs * (num_regs - 1);
    let op_budget = isa_budget - mov_budget as usize;

    let mut freq: HashMap<String, u64> = HashMap::new();
    for (name, dsts, srcs) in &reg_trace {
        let mut parts = vec![name.to_string()];
        for r in dsts { parts.push(format!("r{}", r)); }
        for r in srcs { parts.push(format!("r{}", r)); }
        let spec = parts.join(".");
        *freq.entry(spec).or_insert(0) += 1;
    }

    let mut sorted: Vec<_> = freq.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));

    // Ensure at least 1 variant per base opcode
    let mut base_to_best: HashMap<String, (String, u64)> = HashMap::new();
    for (name, count) in &sorted {
        let (base, _) = parse_isa_opcode(name);
        let e = base_to_best.entry(base.to_string()).or_insert_with(|| (name.to_string(), 0));
        if **count > e.1 { *e = (name.to_string(), **count); }
    }

    let mut isa: HashSet<String> = HashSet::new();
    for src in 0..num_regs {
        for dst in 0..num_regs {
            if src != dst { isa.insert(format!("mov.r{}.r{}", dst, src)); }
        }
    }

    let mut isa_ops_added = 0;
    let mut covered = 0u64;
    let mut mandatory: Vec<(String, u64)> = base_to_best.values().cloned().collect();
    mandatory.sort_by(|a, b| b.1.cmp(&a.1));
    for (name, count) in &mandatory {
        if !isa.contains(name.as_str()) {
            isa.insert(name.clone());
            isa_ops_added += 1;
            covered += *count;
        }
    }

    for (name, count) in &sorted {
        if isa_ops_added >= op_budget { break; }
        if !isa.contains(name.as_str()) {
            isa.insert(name.to_string());
            isa_ops_added += 1;
            covered += **count;
        }
    }

    // Build ISA lookup: base → list of (regs) sorted by frequency
    let mut isa_by_base: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
    for op in &isa {
        let (base, regs) = parse_isa_opcode(op);
        isa_by_base.entry(base.to_string()).or_default().push(regs);
    }
    // Sort each base's variants by frequency (most frequent first)
    for (base, variants) in isa_by_base.iter_mut() {
        variants.sort_by(|a, b| {
            let name_a = {
                let mut p = vec![base.clone()];
                for r in a { p.push(format!("r{}", r)); }
                p.join(".")
            };
            let name_b = {
                let mut p = vec![base.clone()];
                for r in b { p.push(format!("r{}", r)); }
                p.join(".")
            };
            let ca = freq.get(&name_a).copied().unwrap_or(0);
            let cb = freq.get(&name_b).copied().unwrap_or(0);
            cb.cmp(&ca)
        });
    }

    eprintln!("  ISA: {} ops ({} data + {} mov)", isa.len(), isa_ops_added, mov_budget);

    // === Phase 3: Re-allocate with regalloc2 + rewrite ===
    eprintln!("Phase 3: regalloc2 with FixedReg constraints ({} regs)...", num_regs);

    // Build MachineEnv
    let mut preferred_regs = Vec::new();
    for r in 0..num_regs {
        preferred_regs.push(ra2::PReg::new(r as usize, ra2::RegClass::Int));
    }
    let env = MachineEnv {
        preferred_regs_by_class: [preferred_regs, vec![], vec![]],
        non_preferred_regs_by_class: [vec![], vec![], vec![]],
        scratch_by_class: [None, None, None],
        fixed_stack_slots: vec![],
    };
    let opts = RegallocOptions {
        verbose_log: false,
        validate_ssa: false,
        algorithm: Algorithm::Ion,
    };

    let mut hinted_funcs: Vec<(Vec<VRegInst>, RegAllocResult, u32, u32)> = Vec::new();
    let mut ra2_success = 0u32;
    let mut ra2_fallback = 0u32;

    for (func_count, (insts, orig_alloc, num_params, num_locals)) in vreg_funcs.iter().enumerate() {
        // Try multi-block regalloc2
        let ra2_multi = Ra2FuncMulti::build(insts, num_regs, &isa_by_base, orig_alloc);

        if ra2_multi.is_none() {
            // BrTable or empty function — fall back to original allocation
            hinted_funcs.push((insts.clone(), orig_alloc.clone(), *num_params, *num_locals));
            ra2_fallback += 1;
            continue;
        }
        let ra2_func = ra2_multi.unwrap();

        if ra2_func.num_vregs == 0 {
            hinted_funcs.push((insts.clone(), orig_alloc.clone(), *num_params, *num_locals));
            ra2_fallback += 1;
            continue;
        }

        match ra2::run(&ra2_func, &env, &opts) {
            Ok(output) => {
                // === MULTI-BLOCK REWRITER ===
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
                    if let ra2::Edit::Move { from, to } = edit {
                        let idx = pp.inst().index();
                        match pp.pos() {
                            ra2::InstPosition::Before => edits_before.entry(idx).or_default().push((*from, *to)),
                            ra2::InstPosition::After => edits_after.entry(idx).or_default().push((*from, *to)),
                        }
                    }
                }

                // Reverse mapping: original instruction index → regalloc2 flat index
                let mut orig_to_ra2: HashMap<usize, usize> = HashMap::new();
                for (ra2_idx, &orig_idx) in ra2_func.orig_indices.iter().enumerate() {
                    if orig_idx != usize::MAX {
                        orig_to_ra2.insert(orig_idx, ra2_idx);
                    }
                }

                // === Collect trampoline info ===
                // For each trampoline block, gather its edits and determine injection point.
                struct TrampolineInfo {
                    moves: Vec<(ra2::Allocation, ra2::Allocation)>,
                }
                let mut trampoline_infos: Vec<TrampolineInfo> = Vec::new();
                // Map: orig_inst_idx of terminator → [(succ_idx, trampoline_info_idx)]
                let mut trampoline_at_term: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
                // Map: else_marker_orig_idx → trampoline_info_idx
                let mut trampoline_at_else: HashMap<usize, usize> = HashMap::new();

                for (bi, block) in ra2_func.cfg_blocks.iter().enumerate() {
                    if !block.is_trampoline { continue; }

                    let src_block = block.preds[0];
                    // Find which succ_idx in src_block points to this trampoline
                    let succ_idx = ra2_func.cfg_blocks[src_block].succs.iter()
                        .position(|&s| s == bi)
                        .unwrap_or(0);

                    // Collect edits for this trampoline's ra2 instructions
                    let (t_start, t_end) = ra2_func.block_inst_ranges[bi];
                    let mut moves = Vec::new();
                    for ra2_idx in t_start..t_end {
                        if let Some(m) = edits_before.get(&ra2_idx) { moves.extend(m.iter().cloned()); }
                        if let Some(m) = edits_after.get(&ra2_idx) { moves.extend(m.iter().cloned()); }
                    }

                    if moves.is_empty() { continue; }

                    let ti_idx = trampoline_infos.len();
                    trampoline_infos.push(TrampolineInfo { moves });

                    // Find the terminator of src_block
                    let src = &ra2_func.cfg_blocks[src_block];
                    if let Some(&last_ra2_orig) = src.ra2_insts.last() {
                        match &insts[last_ra2_orig] {
                            VRegInst::If { label, .. } if succ_idx == 1 => {
                                // Else-edge trampoline: inject after Else marker
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

                // Walk ALL original instructions (including control flow markers)
                for (orig_idx, inst) in insts.iter().enumerate() {
                    if let Some(&ra2_idx) = orig_to_ra2.get(&orig_idx) {
                        // This instruction was in regalloc2's scope

                        // Check for BrIf with taken-edge trampoline (succ_idx 0)
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
                            // === BrIf with taken-edge trampoline ===
                            // Convert: BrIf { cond, label }
                            // To:      If { cond, SYNTH } [trampoline_moves] Br { label } End { SYNTH }
                            let synth_label = next_synth_label;
                            next_synth_label += 1;

                            let cond_vreg = if !new_srcs.is_empty() { new_srcs[0] }
                                else { vreg_src_regs(inst).into_iter().next().unwrap_or(VReg(0)) };

                            new_insts.push(VRegInst::If { cond: cond_vreg, label: synth_label });

                            // Update current_loc from this instruction's allocations
                            let allocs2 = output.inst_allocs(ra2::Inst::new(ra2_idx));
                            let ops2 = &ra2_func.operands[ra2_idx];
                            for (op_idx, op) in ops2.iter().enumerate() {
                                if op_idx >= allocs2.len() { continue; }
                                let our_vreg = ra2_func.vreg_reverse[op.vreg().vreg()];
                                current_loc.insert(our_vreg, allocs2[op_idx]);
                            }

                            // Emit trampoline moves (only execute when branch taken)
                            let tramp = &trampoline_infos[ti_idx];
                            emit_edit_moves(&tramp.moves, &mut new_insts, &mut vreg_to_preg,
                                &mut spilled, &mut spill_slots_map, &mut current_loc, &mut next_vreg_id);

                            new_insts.push(VRegInst::Br { label: *label });
                            new_insts.push(VRegInst::End { label: synth_label });

                            // After-edits execute on fallthrough path (after End)
                            if let Some(moves) = edits_after.get(&ra2_idx) {
                                emit_edit_moves(moves, &mut new_insts, &mut vreg_to_preg,
                                    &mut spilled, &mut spill_slots_map, &mut current_loc, &mut next_vreg_id);
                            }
                        } else {
                            // === Normal instruction rewrite ===
                            if new_dsts.len() == orig_dsts.len() && new_srcs.len() == orig_srcs.len() {
                                new_insts.push(rewrite_inst_vregs(inst, &new_dsts, &new_srcs));
                            } else {
                                new_insts.push(inst.clone());
                            }

                            // Update current_loc from this instruction's allocations
                            let allocs2 = output.inst_allocs(ra2::Inst::new(ra2_idx));
                            let ops2 = &ra2_func.operands[ra2_idx];
                            for (op_idx, op) in ops2.iter().enumerate() {
                                if op_idx >= allocs2.len() { continue; }
                                let our_vreg = ra2_func.vreg_reverse[op.vreg().vreg()];
                                current_loc.insert(our_vreg, allocs2[op_idx]);
                            }

                            // Insert edit moves AFTER
                            if let Some(moves) = edits_after.get(&ra2_idx) {
                                emit_edit_moves(moves, &mut new_insts, &mut vreg_to_preg,
                                    &mut spilled, &mut spill_slots_map, &mut current_loc, &mut next_vreg_id);
                            }

                            // If this is an If instruction with then-edge trampoline (succ_idx 0)
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

                        // Else-edge trampoline: inject after Else marker
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
                // Validate: check for unmapped vregs in rewritten code
                let mut unmapped = 0u32;
                for (ii, inst) in new_insts.iter().enumerate() {
                    for v in vreg_src_regs(inst) {
                        if !new_alloc.vreg_to_preg.contains_key(&v) && !new_alloc.spill_slots.contains_key(&v) {
                            if unmapped < 5 {
                                eprintln!("  UNMAPPED src vreg {:?} in func {} inst {}: {:?}",
                                    v, func_count, ii, vreg_inst_name(inst));
                            }
                            unmapped += 1;
                        }
                    }
                    for v in vreg_dst_regs(inst) {
                        if !new_alloc.vreg_to_preg.contains_key(&v) && !new_alloc.spill_slots.contains_key(&v) {
                            if unmapped < 5 {
                                eprintln!("  UNMAPPED dst vreg {:?} in func {} inst {}: {:?}",
                                    v, func_count, ii, vreg_inst_name(inst));
                            }
                            unmapped += 1;
                        }
                    }
                }
                if unmapped > 0 {
                    eprintln!("  func {}: {} unmapped vreg references", func_count, unmapped);
                }

                hinted_funcs.push((new_insts, new_alloc, *num_params, *num_locals));
                ra2_success += 1;
            }
            Err(e) => {
                eprintln!("  func {}: regalloc2 failed ({}), using fallback", func_count, e);
                hinted_funcs.push((insts.clone(), orig_alloc.clone(), *num_params, *num_locals));
                ra2_fallback += 1;
            }
        }
    }

    eprintln!("  regalloc2: {} success, {} fallback", ra2_success, ra2_fallback);

    // === Phase 4: Execute ===
    eprintln!("Phase 4: Executing...");

    let mut max_spill_slots2 = 0u32;
    for (_, alloc, _, _) in &hinted_funcs {
        max_spill_slots2 = max_spill_slots2.max(alloc.num_spill_slots);
    }
    let mut vm2 = PRegVM::new(num_regs as usize, max_spill_slots2 as usize + 128, 256);

    for (vreg_insts, alloc, num_params, num_locals) in &hinted_funcs {
        vm2.add_vreg_function_ml(
            vreg_insts.clone(),
            RegAllocResult {
                vreg_to_preg: alloc.vreg_to_preg.clone(),
                spilled: alloc.spilled.clone(),
                spill_slots: alloc.spill_slots.clone(),
                num_spill_slots: alloc.num_spill_slots,
            },
            *num_params, *num_locals,
        );
    }

    for (i, val) in global_inits.iter().enumerate() {
        if i < vm2.globals.len() { vm2.globals[i] = Value::I32(*val); }
    }
    for (offset, data) in &data_segments {
        vm2.write_memory(*offset as usize, data);
    }

    vm2.write_memory(0, json_data);
    vm2.globals[0] = Value::I32(global_inits.first().copied().unwrap_or(1048576));

    vm2.write_memory(FRAME_SP_ADDR as usize, &FRAME_STACK_BASE.to_le_bytes());
    for (i, val) in global_inits.iter().enumerate() {
        let addr = GLOBALS_MEM_BASE as usize + (i as u32 * SLOT_SIZE) as usize;
        vm2.write_memory(addr, &(*val as u32).to_le_bytes());
    }
    vm2.write_memory(frame_base, &0u32.to_le_bytes());
    vm2.write_memory(frame_base + SLOT_SIZE as usize, &(json_data.len() as u32).to_le_bytes());

    vm2.enable_reg_trace();

    let func2 = &hinted_funcs[func_idx as usize];
    let exec_result = vm2.execute_vreg(&func2.0, &func2.1);
    let nodes_exec = exec_result.map(|v| v.as_i32() as u32).unwrap_or(0);
    let reg_trace2 = vm2.reg_trace.take().unwrap_or_default();
    let trace_len2 = reg_trace2.len();

    // === Trace divergence ===
    if nodes_exec != nodes_profile {
        eprintln!("\n--- Trace Divergence (first 20 mismatches) ---");
        let mut mismatches = 0;
        let max_compare = reg_trace.len().min(reg_trace2.len());
        // Filter out mov instructions from the regalloc2 trace to align with original
        // Actually, compare raw traces side by side
        eprintln!("{:<6} {:<40} {:<40}", "Idx", "Original", "Regalloc2");
        for i in 0..max_compare.min(500) {
            let (name1, dsts1, srcs1) = &reg_trace[i];
            let (name2, dsts2, srcs2) = &reg_trace2[i];
            let s1 = format!("{} d{:?} s{:?}", name1, dsts1, srcs1);
            let s2 = format!("{} d{:?} s{:?}", name2, dsts2, srcs2);
            if s1 != s2 {
                eprintln!("{:<6} {:<40} {:<40} <-- DIFF", i, s1, s2);
                mismatches += 1;
                if mismatches >= 20 { break; }
            }
        }
        if mismatches == 0 && reg_trace.len() != reg_trace2.len() {
            eprintln!("Traces match for first {} entries but differ in length: {} vs {}",
                max_compare, reg_trace.len(), reg_trace2.len());
        }
        // Also dump first 50 entries of each
        eprintln!("\n--- Original trace (first 50) ---");
        for (i, (name, dsts, srcs)) in reg_trace.iter().enumerate().take(50) {
            eprintln!("{:>4}: {} d{:?} s{:?}", i, name, dsts, srcs);
        }
        eprintln!("\n--- Regalloc2 trace (first 50) ---");
        for (i, (name, dsts, srcs)) in reg_trace2.iter().enumerate().take(50) {
            eprintln!("{:>4}: {} d{:?} s{:?}", i, name, dsts, srcs);
        }
    }

    // === Phase 5: Report ===
    let mut exec_freq: HashMap<String, u64> = HashMap::new();
    let mut violations = 0u64;
    let mut infra_movs = 0u64;
    for (name, dsts, srcs) in &reg_trace2 {
        let mut parts = vec![name.to_string()];
        for r in dsts { parts.push(format!("r{}", r)); }
        for r in srcs { parts.push(format!("r{}", r)); }
        let spec = parts.join(".");
        *exec_freq.entry(spec.clone()).or_insert(0) += 1;
        if !isa.contains(&spec) {
            if *name == "mov" && (dsts.len() + srcs.len()) < 2 {
                infra_movs += 1;
            } else {
                violations += 1;
            }
        }
    }

    let mov_in_trace = reg_trace2.iter().filter(|(name, _, _)| *name == "mov").count();

    println!("=== regalloc2 + Rewriter Results ({} regs, ISA {}) ===\n", num_regs, isa_budget);
    println!("WASM:    {}", wasm_path);
    println!("Profile: {} nodes", nodes_profile);
    println!("Execute: {} nodes\n", nodes_exec);
    println!("--- Trace ---");
    println!("Original:   {} instructions", trace_len);
    println!("regalloc2:  {} instructions", trace_len2);
    println!("Overhead:   {} ({:.1}%)",
        trace_len2 as i64 - trace_len as i64,
        (trace_len2 as f64 / trace_len as f64 - 1.0) * 100.0);
    println!("Mov total:  {} ({:.1}% of trace)", mov_in_trace, mov_in_trace as f64 / trace_len2 as f64 * 100.0);
    println!("Infra movs: {} (spill/reload)", infra_movs);
    println!();
    println!("--- ISA ---");
    println!("ISA size:     {}", isa.len());
    println!("Unique used:  {}", exec_freq.len());
    println!("Violations:   {}", violations);
    println!("regalloc2:    {} ok, {} fallback", ra2_success, ra2_fallback);
    println!();

    if nodes_exec == nodes_profile && violations == 0 {
        println!("PASS: {} nodes, 0 violations", nodes_exec);
    } else if nodes_exec == nodes_profile {
        println!("PARTIAL: {} nodes, {} violations", nodes_exec, violations);
    } else {
        println!("FAIL: {} nodes (expected {})", nodes_exec, nodes_profile);
    }

    Ok(())
}
