//! Incremental test for the regalloc2 rewriter.
//!
//! LEGACY: Uses TestRa2Func which models the entire function as a single basic block
//! (num_blocks=1, no CFG). The canonical path uses Ra2FuncMulti in isa_regalloc.rs
//! which provides regalloc2 with full CFG visibility. Kept as artifact of gradual
//! complexity build-up.
//!
//! Tests progressively complex VRegInst programs through:
//!   1. Original allocation (linear_scan) → expected result
//!   2. regalloc2 + rewriter → actual result
//! Compares results at each step to isolate bugs.

use std::collections::{HashMap, HashSet};
use reg_analyzer::regvm::{VRegInst, VReg, SpillSlot};
use reg_analyzer::regalloc::{compute_live_intervals, linear_scan_alloc, RegAllocResult};
use reg_analyzer::preg_vm::{
    PRegVM, vreg_dst_regs, vreg_src_regs, specialized_opcode, rewrite_inst_vregs,
};
use reg_analyzer::interpreter::Value;
use reg_analyzer::regvm::PReg as OurPReg;

use regalloc2::{
    self as ra2, MachineEnv, Operand, OperandConstraint, OperandKind, OperandPos,
    RegallocOptions, Algorithm,
};

// === Minimal Ra2Func for testing (no ISA constraints) ===
struct TestRa2Func {
    orig_indices: Vec<usize>,
    operands: Vec<Vec<Operand>>,
    is_branch_flag: Vec<bool>,
    is_ret_flag: Vec<bool>,
    num_vregs: usize,
    vreg_reverse: Vec<VReg>,
    entry_params: Vec<ra2::VReg>,
}

impl TestRa2Func {
    fn build(insts: &[VRegInst]) -> Self {
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
            let idx = vreg_map[&v];
            ra2::VReg::new(idx, ra2::RegClass::Int)
        };

        let mut orig_indices = Vec::new();
        let mut operands = Vec::new();
        let mut is_branch_flag = Vec::new();
        let mut is_ret_flag = Vec::new();

        for (i, inst) in insts.iter().enumerate() {
            match inst {
                VRegInst::Block { .. } | VRegInst::Loop { .. } |
                VRegInst::If { .. } | VRegInst::Else { .. } |
                VRegInst::End { .. } => continue,
                _ => {}
            }

            let dsts = vreg_dst_regs(inst);
            let srcs = vreg_src_regs(inst);

            let mut ops = Vec::new();
            // All unconstrained (Reg)
            for v in &dsts {
                ops.push(Operand::new(
                    to_ra2_vreg(*v),
                    OperandConstraint::Reg,
                    OperandKind::Def,
                    OperandPos::Late,
                ));
            }
            for v in &srcs {
                ops.push(Operand::new(
                    to_ra2_vreg(*v),
                    OperandConstraint::Reg,
                    OperandKind::Use,
                    OperandPos::Early,
                ));
            }

            let is_ret = matches!(inst, VRegInst::Return { .. });
            let is_branch = matches!(inst, VRegInst::Br { .. } | VRegInst::BrIf { .. } | VRegInst::BrTable { .. });

            orig_indices.push(i);
            operands.push(ops);
            is_ret_flag.push(is_ret);
            is_branch_flag.push(is_branch || is_ret);
        }

        // Ensure last is a ret/branch
        if orig_indices.is_empty() || !(*is_branch_flag.last().unwrap_or(&false)) {
            orig_indices.push(usize::MAX);
            operands.push(vec![]);
            is_ret_flag.push(true);
            is_branch_flag.push(true);
        }

        // Entry params
        let mut defined: HashSet<usize> = HashSet::new();
        let mut live_in: Vec<ra2::VReg> = Vec::new();
        let mut live_in_set: HashSet<usize> = HashSet::new();
        for ops in &operands {
            for op in ops {
                if op.kind() == OperandKind::Use {
                    let idx = op.vreg().vreg();
                    if !defined.contains(&idx) && !live_in_set.contains(&idx) {
                        live_in.push(op.vreg());
                        live_in_set.insert(idx);
                    }
                }
            }
            for op in ops {
                if op.kind() == OperandKind::Def {
                    defined.insert(op.vreg().vreg());
                }
            }
        }

        TestRa2Func {
            orig_indices,
            operands,
            is_branch_flag,
            is_ret_flag,
            num_vregs,
            vreg_reverse,
            entry_params: live_in,
        }
    }
}

impl ra2::Function for TestRa2Func {
    fn num_insts(&self) -> usize { self.orig_indices.len() }
    fn num_blocks(&self) -> usize { 1 }
    fn entry_block(&self) -> ra2::Block { ra2::Block::new(0) }
    fn block_insns(&self, _block: ra2::Block) -> ra2::InstRange {
        ra2::InstRange::new(ra2::Inst::new(0), ra2::Inst::new(self.orig_indices.len()))
    }
    fn block_succs(&self, _block: ra2::Block) -> &[ra2::Block] { &[] }
    fn block_preds(&self, _block: ra2::Block) -> &[ra2::Block] { &[] }
    fn block_params(&self, _block: ra2::Block) -> &[ra2::VReg] { &self.entry_params }
    fn is_ret(&self, insn: ra2::Inst) -> bool { self.is_ret_flag[insn.index()] }
    fn is_branch(&self, insn: ra2::Inst) -> bool { self.is_branch_flag[insn.index()] }
    fn branch_blockparams(&self, _b: ra2::Block, _i: ra2::Inst, _s: usize) -> &[ra2::VReg] { &[] }
    fn inst_operands(&self, insn: ra2::Inst) -> &[Operand] { &self.operands[insn.index()] }
    fn inst_clobbers(&self, _insn: ra2::Inst) -> ra2::PRegSet { ra2::PRegSet::empty() }
    fn num_vregs(&self) -> usize { self.num_vregs }
    fn spillslot_size(&self, _regclass: ra2::RegClass) -> usize { 1 }
}

/// Rewrite a VRegInst program using regalloc2's output (same logic as isa_regalloc.rs)
fn rewrite_with_ra2(
    insts: &[VRegInst],
    ra2_func: &TestRa2Func,
    output: &ra2::Output,
    num_regs: u32,
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

    let make_vreg = |alloc: ra2::Allocation,
                      v2p: &mut HashMap<VReg, OurPReg>,
                      sp: &mut HashSet<VReg>,
                      ss: &mut HashMap<VReg, SpillSlot>,
                      next_id: &mut u32| -> VReg {
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
    };

    // Collect edits
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

    // Reverse mapping
    let mut orig_to_ra2: HashMap<usize, usize> = HashMap::new();
    for (ra2_idx, &orig_idx) in ra2_func.orig_indices.iter().enumerate() {
        if orig_idx != usize::MAX {
            orig_to_ra2.insert(orig_idx, ra2_idx);
        }
    }

    // Dynamic current_loc tracking
    let mut current_loc: HashMap<VReg, ra2::Allocation> = HashMap::new();

    for (orig_idx, inst) in insts.iter().enumerate() {
        if let Some(&ra2_idx) = orig_to_ra2.get(&orig_idx) {
            // Edits before (+ update current_loc)
            if let Some(moves) = edits_before.get(&ra2_idx) {
                for (from, to) in moves {
                    let moved: Vec<VReg> = current_loc.iter()
                        .filter(|(_, loc)| **loc == *from)
                        .map(|(v, _)| *v)
                        .collect();
                    for v in &moved { current_loc.insert(*v, *to); }
                    let src = make_vreg(*from, &mut vreg_to_preg, &mut spilled, &mut spill_slots_map, &mut next_vreg_id);
                    let dst = make_vreg(*to, &mut vreg_to_preg, &mut spilled, &mut spill_slots_map, &mut next_vreg_id);
                    new_insts.push(VRegInst::Mov { dst, src });
                }
            }

            // Rewrite instruction
            let allocs = output.inst_allocs(ra2::Inst::new(ra2_idx));
            let orig_dsts = vreg_dst_regs(inst);
            let orig_srcs = vreg_src_regs(inst);

            let mut new_dsts = Vec::new();
            let mut new_srcs = Vec::new();
            let mut ai = 0;
            for _ in &orig_dsts {
                if ai < allocs.len() {
                    new_dsts.push(make_vreg(allocs[ai], &mut vreg_to_preg, &mut spilled, &mut spill_slots_map, &mut next_vreg_id));
                }
                ai += 1;
            }
            for _ in &orig_srcs {
                if ai < allocs.len() {
                    new_srcs.push(make_vreg(allocs[ai], &mut vreg_to_preg, &mut spilled, &mut spill_slots_map, &mut next_vreg_id));
                }
                ai += 1;
            }

            if new_dsts.len() == orig_dsts.len() && new_srcs.len() == orig_srcs.len() {
                new_insts.push(rewrite_inst_vregs(inst, &new_dsts, &new_srcs));
            } else {
                new_insts.push(inst.clone());
            }

            // Update current_loc
            {
                let allocs2 = output.inst_allocs(ra2::Inst::new(ra2_idx));
                let ops2 = &ra2_func.operands[ra2_idx];
                for (op_idx, op) in ops2.iter().enumerate() {
                    if op_idx >= allocs2.len() { continue; }
                    let our_vreg = ra2_func.vreg_reverse[op.vreg().vreg()];
                    current_loc.insert(our_vreg, allocs2[op_idx]);
                }
            }

            // Edits after (+ update current_loc)
            if let Some(moves) = edits_after.get(&ra2_idx) {
                for (from, to) in moves {
                    let moved: Vec<VReg> = current_loc.iter()
                        .filter(|(_, loc)| **loc == *from)
                        .map(|(v, _)| *v)
                        .collect();
                    for v in &moved { current_loc.insert(*v, *to); }
                    let src = make_vreg(*from, &mut vreg_to_preg, &mut spilled, &mut spill_slots_map, &mut next_vreg_id);
                    let dst = make_vreg(*to, &mut vreg_to_preg, &mut spilled, &mut spill_slots_map, &mut next_vreg_id);
                    new_insts.push(VRegInst::Mov { dst, src });
                }
            }
        } else {
            // Control flow marker — use current_loc
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
                }
            }
            new_insts.push(inst.clone());
        }
    }

    let alloc = RegAllocResult {
        vreg_to_preg,
        spilled,
        spill_slots: spill_slots_map,
        num_spill_slots: output.num_spillslots as u32,
    };
    (new_insts, alloc)
}

fn make_env(num_regs: u32) -> MachineEnv {
    let mut preferred = Vec::new();
    for r in 0..num_regs {
        preferred.push(ra2::PReg::new(r as usize, ra2::RegClass::Int));
    }
    MachineEnv {
        preferred_regs_by_class: [preferred, vec![], vec![]],
        non_preferred_regs_by_class: [vec![], vec![], vec![]],
        scratch_by_class: [None, None, None],
        fixed_stack_slots: vec![],
    }
}

/// Run a program through both paths and compare
fn test_program(name: &str, insts: Vec<VRegInst>, num_regs: u32) {
    println!("\n=== Test: {} ({} insts, {} regs) ===", name, insts.len(), num_regs);

    // Path 1: Original (linear_scan)
    let intervals = compute_live_intervals(&insts);
    let orig_alloc = linear_scan_alloc(&intervals, num_regs);

    let mut vm1 = PRegVM::new(num_regs as usize, orig_alloc.num_spill_slots as usize + 16, 1);
    vm1.add_vreg_function(insts.clone(), orig_alloc.clone(), 0, 0);
    let result1 = vm1.execute_vreg(&insts, &orig_alloc);
    let val1 = result1.map(|v| v.as_i32()).unwrap_or(-999);

    // Path 2: regalloc2 + rewriter
    let ra2_func = TestRa2Func::build(&insts);
    let env = make_env(num_regs);
    let opts = RegallocOptions { verbose_log: false, validate_ssa: false, algorithm: Algorithm::Ion };

    match ra2::run(&ra2_func, &env, &opts) {
        Ok(output) => {
            // Print edits
            if !output.edits.is_empty() {
                println!("  regalloc2 edits: {}", output.edits.len());
                for (pp, edit) in &output.edits {
                    println!("    {:?}: {:?}", pp, edit);
                }
            }

            // Print per-instruction allocations
            for (ra2_idx, &orig_idx) in ra2_func.orig_indices.iter().enumerate() {
                if orig_idx == usize::MAX { continue; }
                let allocs = output.inst_allocs(ra2::Inst::new(ra2_idx));
                let inst = &insts[orig_idx];
                let dsts = vreg_dst_regs(inst);
                let srcs = vreg_src_regs(inst);
                let mut parts = Vec::new();
                let mut ai = 0;
                for d in &dsts {
                    if ai < allocs.len() { parts.push(format!("d:{:?}={}", d, allocs[ai])); }
                    ai += 1;
                }
                for s in &srcs {
                    if ai < allocs.len() { parts.push(format!("s:{:?}={}", s, allocs[ai])); }
                    ai += 1;
                }
                println!("  ra2[{}] orig[{}]: {:?}  allocs: {}",
                    ra2_idx, orig_idx, inst, parts.join(", "));
            }

            let (new_insts, new_alloc) = rewrite_with_ra2(&insts, &ra2_func, &output, num_regs);

            println!("  Rewritten: {} insts", new_insts.len());

            // Print rewritten instructions and their alloc
            for (i, inst) in new_insts.iter().enumerate() {
                let dsts = vreg_dst_regs(inst);
                let srcs = vreg_src_regs(inst);
                let dst_regs: Vec<String> = dsts.iter().map(|v| {
                    if let Some(p) = new_alloc.vreg_to_preg.get(v) { format!("r{}", p.0) }
                    else if let Some(s) = new_alloc.spill_slots.get(v) { format!("s{}", s.0) }
                    else { format!("?") }
                }).collect();
                let src_regs: Vec<String> = srcs.iter().map(|v| {
                    if let Some(p) = new_alloc.vreg_to_preg.get(v) { format!("r{}", p.0) }
                    else if let Some(s) = new_alloc.spill_slots.get(v) { format!("s{}", s.0) }
                    else { format!("?") }
                }).collect();
                println!("    [{:>3}] {:?}  dsts={:?} srcs={:?}", i, inst, dst_regs, src_regs);
            }

            let mut vm2 = PRegVM::new(num_regs as usize, new_alloc.num_spill_slots as usize + 16, 1);
            vm2.add_vreg_function(new_insts.clone(), new_alloc.clone(), 0, 0);
            let result2 = vm2.execute_vreg(&new_insts, &new_alloc);
            let val2 = result2.map(|v| v.as_i32()).unwrap_or(-999);

            if val1 == val2 {
                println!("  PASS: both return {}", val1);
            } else {
                println!("  FAIL: original={}, regalloc2={}", val1, val2);
            }
        }
        Err(e) => {
            println!("  regalloc2 FAILED: {}", e);
        }
    }
}

fn main() {
    let v = |n: u32| VReg(n);

    // Test 1: Simplest — const → return
    test_program("const_return", vec![
        VRegInst::I32Const { dst: v(0), val: 42 },
        VRegInst::Return { values: vec![v(0)] },
    ], 4);

    // Test 2: const + const → add → return
    test_program("add_return", vec![
        VRegInst::I32Const { dst: v(0), val: 10 },
        VRegInst::I32Const { dst: v(1), val: 32 },
        VRegInst::I32Add { dst: v(2), src1: v(0), src2: v(1) },
        VRegInst::Return { values: vec![v(2)] },
    ], 4);

    // Test 3: Same but force spills (2 regs)
    test_program("add_return_2regs", vec![
        VRegInst::I32Const { dst: v(0), val: 10 },
        VRegInst::I32Const { dst: v(1), val: 32 },
        VRegInst::I32Add { dst: v(2), src1: v(0), src2: v(1) },
        VRegInst::Return { values: vec![v(2)] },
    ], 2);

    // Test 4: Chain of ops (more register pressure)
    test_program("chain_4regs", vec![
        VRegInst::I32Const { dst: v(0), val: 1 },
        VRegInst::I32Const { dst: v(1), val: 2 },
        VRegInst::I32Const { dst: v(2), val: 3 },
        VRegInst::I32Const { dst: v(3), val: 4 },
        VRegInst::I32Const { dst: v(4), val: 5 },
        VRegInst::I32Add { dst: v(5), src1: v(0), src2: v(1) },  // 1+2=3
        VRegInst::I32Add { dst: v(6), src1: v(2), src2: v(3) },  // 3+4=7
        VRegInst::I32Add { dst: v(7), src1: v(4), src2: v(5) },  // 5+3=8
        VRegInst::I32Add { dst: v(8), src1: v(6), src2: v(7) },  // 7+8=15
        VRegInst::Return { values: vec![v(8)] },
    ], 4);

    // Test 5: Same with 2 regs (heavy spilling)
    test_program("chain_2regs", vec![
        VRegInst::I32Const { dst: v(0), val: 1 },
        VRegInst::I32Const { dst: v(1), val: 2 },
        VRegInst::I32Const { dst: v(2), val: 3 },
        VRegInst::I32Const { dst: v(3), val: 4 },
        VRegInst::I32Const { dst: v(4), val: 5 },
        VRegInst::I32Add { dst: v(5), src1: v(0), src2: v(1) },
        VRegInst::I32Add { dst: v(6), src1: v(2), src2: v(3) },
        VRegInst::I32Add { dst: v(7), src1: v(4), src2: v(5) },
        VRegInst::I32Add { dst: v(8), src1: v(6), src2: v(7) },
        VRegInst::Return { values: vec![v(8)] },
    ], 2);

    // Test 6: Simple block (no branch taken)
    test_program("simple_block", vec![
        VRegInst::I32Const { dst: v(0), val: 42 },
        VRegInst::Block { label: 0 },
        VRegInst::I32Const { dst: v(1), val: 10 },
        VRegInst::I32Add { dst: v(2), src1: v(0), src2: v(1) },
        VRegInst::End { label: 0 },
        VRegInst::Return { values: vec![v(2)] },
    ], 4);

    // Test 7: Block with branch
    test_program("block_branch", vec![
        VRegInst::I32Const { dst: v(0), val: 42 },
        VRegInst::I32Const { dst: v(1), val: 0 },  // false
        VRegInst::Block { label: 0 },
        VRegInst::BrIf { cond: v(1), label: 0 },   // not taken
        VRegInst::I32Const { dst: v(2), val: 10 },
        VRegInst::I32Add { dst: v(3), src1: v(0), src2: v(2) },  // 42+10=52
        VRegInst::End { label: 0 },
        VRegInst::Return { values: vec![v(3)] },
    ], 4);

    // Test 8: Block with branch taken
    test_program("block_branch_taken", vec![
        VRegInst::I32Const { dst: v(0), val: 42 },
        VRegInst::I32Const { dst: v(1), val: 1 },  // true
        VRegInst::Block { label: 0 },
        VRegInst::BrIf { cond: v(1), label: 0 },   // taken: skip to end
        VRegInst::I32Const { dst: v(2), val: 10 },
        VRegInst::I32Add { dst: v(0), src1: v(0), src2: v(2) },  // skipped
        VRegInst::End { label: 0 },
        VRegInst::Return { values: vec![v(0)] },    // returns 42 (untouched)
    ], 4);
}
