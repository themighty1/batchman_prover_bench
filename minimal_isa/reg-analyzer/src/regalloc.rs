//! Linear scan register allocator
//!
//! Takes virtual register IR and allocates N physical registers,
//! inserting spill/reload as needed.

use crate::regvm::{PReg, SpillSlot, VReg, VRegInst};
use std::collections::{HashMap, HashSet, BTreeSet};

/// Live range for a virtual register
#[derive(Debug, Clone)]
pub struct LiveInterval {
    pub vreg: VReg,
    pub start: u32,
    pub end: u32,
}

/// Result of register allocation
#[derive(Clone)]
pub struct RegAllocResult {
    /// Mapping from virtual to physical register (for non-spilled)
    pub vreg_to_preg: HashMap<VReg, PReg>,
    /// Which vregs are spilled
    pub spilled: HashSet<VReg>,
    /// Spill slot assignments
    pub spill_slots: HashMap<VReg, SpillSlot>,
    /// Number of spill slots used
    pub num_spill_slots: u32,
}

/// Compute live intervals for all virtual registers
///
/// LEGACY: This is a flat linear scan with no CFG awareness. Used only as input
/// to `linear_scan_alloc`. The canonical path uses Ra2FuncMulti in isa_regalloc.rs
/// which gives regalloc2 full CFG visibility. Kept as artifact of gradual complexity build-up.
pub fn compute_live_intervals(instructions: &[VRegInst]) -> Vec<LiveInterval> {
    let mut first_use: HashMap<VReg, u32> = HashMap::new();
    let mut last_use: HashMap<VReg, u32> = HashMap::new();

    for (idx, inst) in instructions.iter().enumerate() {
        let idx = idx as u32;

        // Get all vregs used/defined by this instruction
        let (defs, uses) = get_vreg_defs_uses(inst);

        for vreg in defs.iter().chain(uses.iter()) {
            first_use.entry(*vreg).or_insert(idx);
            last_use.insert(*vreg, idx);
        }
    }

    first_use
        .iter()
        .map(|(&vreg, &start)| LiveInterval {
            vreg,
            start,
            end: *last_use.get(&vreg).unwrap_or(&start),
        })
        .collect()
}

/// Get defined and used vregs for an instruction
fn get_vreg_defs_uses(inst: &VRegInst) -> (Vec<VReg>, Vec<VReg>) {
    use VRegInst::*;

    match inst {
        I32Const { dst, .. } | I64Const { dst, .. } => (vec![*dst], vec![]),

        I32Add { dst, src1, src2 } | I32Sub { dst, src1, src2 } | I32Mul { dst, src1, src2 }
        | I32DivS { dst, src1, src2 } | I32DivU { dst, src1, src2 }
        | I32RemS { dst, src1, src2 } | I32RemU { dst, src1, src2 }
        | I32And { dst, src1, src2 } | I32Or { dst, src1, src2 } | I32Xor { dst, src1, src2 }
        | I32Shl { dst, src1, src2 } | I32ShrU { dst, src1, src2 } | I32ShrS { dst, src1, src2 }
        | I32Rotl { dst, src1, src2 } | I32Rotr { dst, src1, src2 }
        | I64Add { dst, src1, src2 } | I64Sub { dst, src1, src2 } | I64Mul { dst, src1, src2 }
        | I64DivS { dst, src1, src2 } | I64DivU { dst, src1, src2 }
        | I64RemS { dst, src1, src2 } | I64RemU { dst, src1, src2 }
        | I64And { dst, src1, src2 } | I64Or { dst, src1, src2 } | I64Xor { dst, src1, src2 }
        | I64Shl { dst, src1, src2 } | I64ShrU { dst, src1, src2 } | I64ShrS { dst, src1, src2 }
        | I32Eq { dst, src1, src2 } | I32Ne { dst, src1, src2 }
        | I32LtS { dst, src1, src2 } | I32LtU { dst, src1, src2 }
        | I32GtS { dst, src1, src2 } | I32GtU { dst, src1, src2 }
        | I32LeS { dst, src1, src2 } | I32LeU { dst, src1, src2 }
        | I32GeS { dst, src1, src2 } | I32GeU { dst, src1, src2 }
        | I64Eq { dst, src1, src2 } | I64Ne { dst, src1, src2 }
        | I64LtS { dst, src1, src2 } | I64LtU { dst, src1, src2 }
        | I64GtS { dst, src1, src2 } | I64GtU { dst, src1, src2 }
        | I64LeS { dst, src1, src2 } | I64LeU { dst, src1, src2 }
        | I64GeS { dst, src1, src2 } | I64GeU { dst, src1, src2 } => {
            (vec![*dst], vec![*src1, *src2])
        }

        I32Eqz { dst, src } | I32Clz { dst, src } | I32Ctz { dst, src } | I32Popcnt { dst, src }
        | I64Eqz { dst, src } | I64Clz { dst, src } | I64Ctz { dst, src }
        | I32WrapI64 { dst, src } | I64ExtendI32S { dst, src } | I64ExtendI32U { dst, src }
        | I32Extend8S { dst, src } | I32Extend16S { dst, src } => {
            (vec![*dst], vec![*src])
        }

        I32Load { dst, addr, .. } | I64Load { dst, addr, .. }
        | I32Load8U { dst, addr, .. } | I32Load8S { dst, addr, .. }
        | I32Load16U { dst, addr, .. } | I32Load16S { dst, addr, .. }
        | I64Load8U { dst, addr, .. } | I64Load8S { dst, addr, .. }
        | I64Load16U { dst, addr, .. } | I64Load16S { dst, addr, .. }
        | I64Load32U { dst, addr, .. } | I64Load32S { dst, addr, .. } => {
            (vec![*dst], vec![*addr])
        }

        I32Store { addr, src, .. } | I64Store { addr, src, .. }
        | I32Store8 { addr, src, .. } | I32Store16 { addr, src, .. }
        | I64Store8 { addr, src, .. } | I64Store16 { addr, src, .. }
        | I64Store32 { addr, src, .. } => {
            (vec![], vec![*addr, *src])
        }

        LocalGet { dst, .. } => (vec![*dst], vec![]),
        LocalSet { src, .. } => (vec![], vec![*src]),
        LocalTee { dst, src, .. } => (vec![*dst], vec![*src]),

        GlobalGet { dst, .. } => (vec![*dst], vec![]),
        GlobalSet { src, .. } => (vec![], vec![*src]),

        Call { results, args, .. } => (results.clone(), args.clone()),
        CallIndirect { results, args, func_ref, .. } => {
            let mut uses = args.clone();
            uses.push(*func_ref);
            (results.clone(), uses)
        }

        Select { dst, cond, src1, src2 } => (vec![*dst], vec![*cond, *src1, *src2]),

        BrIf { cond, .. } => (vec![], vec![*cond]),
        BrTable { idx, .. } => (vec![], vec![*idx]),
        Br { .. } => (vec![], vec![]),

        Block { .. } | Loop { .. } | End { .. } | Else { .. } => (vec![], vec![]),
        If { cond, .. } => (vec![], vec![*cond]),

        Return { values } => (vec![], values.clone()),
        Unreachable | Nop => (vec![], vec![]),
        Drop { src } => (vec![], vec![*src]),

        MemorySize { dst } => (vec![*dst], vec![]),
        MemoryGrow { dst, pages } => (vec![*dst], vec![*pages]),

        MemoryCopy { dst, src, len } => (vec![], vec![*dst, *src, *len]),
        MemoryFill { dst, val, len } => (vec![], vec![*dst, *val, *len]),

        Mov { dst, src } => (vec![*dst], vec![*src]),
    }
}

/// Linear scan register allocation
///
/// LEGACY: Simple allocator with no CFG awareness — treats the entire function as a flat
/// instruction stream. The canonical path uses regalloc2 via Ra2FuncMulti in isa_regalloc.rs
/// which performs proper multi-block allocation. Kept as artifact of gradual complexity build-up.
pub fn linear_scan_alloc(intervals: &[LiveInterval], num_regs: u32) -> RegAllocResult {
    let mut sorted: Vec<_> = intervals.iter().collect();
    sorted.sort_by_key(|i| i.start);

    let mut vreg_to_preg: HashMap<VReg, PReg> = HashMap::new();
    let mut spilled: HashSet<VReg> = HashSet::new();
    let mut spill_slots: HashMap<VReg, SpillSlot> = HashMap::new();

    // Active intervals sorted by end point
    let mut active: BTreeSet<(u32, VReg, PReg)> = BTreeSet::new();
    let mut free_regs: Vec<PReg> = (0..num_regs as u8).map(PReg).rev().collect();
    let mut next_spill_slot = 0u32;

    for interval in sorted {
        // Expire old intervals
        let expired: Vec<_> = active
            .iter()
            .filter(|(end, _, _)| *end < interval.start)
            .copied()
            .collect();

        for (end, vreg, preg) in expired {
            active.remove(&(end, vreg, preg));
            free_regs.push(preg);
        }

        if free_regs.is_empty() {
            // Need to spill
            // Spill the interval that ends last
            if let Some(&(end, vreg, preg)) = active.iter().last() {
                if end > interval.end {
                    // Spill the one ending later, keep current
                    active.remove(&(end, vreg, preg));
                    vreg_to_preg.remove(&vreg);
                    spilled.insert(vreg);
                    spill_slots.insert(vreg, SpillSlot(next_spill_slot));
                    next_spill_slot += 1;

                    vreg_to_preg.insert(interval.vreg, preg);
                    active.insert((interval.end, interval.vreg, preg));
                } else {
                    // Spill current
                    spilled.insert(interval.vreg);
                    spill_slots.insert(interval.vreg, SpillSlot(next_spill_slot));
                    next_spill_slot += 1;
                }
            } else {
                // No active, just spill current
                spilled.insert(interval.vreg);
                spill_slots.insert(interval.vreg, SpillSlot(next_spill_slot));
                next_spill_slot += 1;
            }
        } else {
            let preg = free_regs.pop().unwrap();
            vreg_to_preg.insert(interval.vreg, preg);
            active.insert((interval.end, interval.vreg, preg));
        }
    }

    RegAllocResult {
        vreg_to_preg,
        spilled,
        spill_slots,
        num_spill_slots: next_spill_slot,
    }
}

/// Rewrite instructions with physical registers and insert spill/reload
pub fn rewrite_with_allocation(
    instructions: &[VRegInst],
    alloc: &RegAllocResult,
    _num_regs: u32,
) -> Vec<String> {
    let mut output = Vec::new();

    // Helper to get physical reg or note spill
    let get_preg = |vreg: VReg| -> String {
        if let Some(&preg) = alloc.vreg_to_preg.get(&vreg) {
            format!("r{}", preg.0)
        } else if let Some(&slot) = alloc.spill_slots.get(&vreg) {
            format!("[sp+{}]", slot.0 * 8)
        } else {
            format!("v{}", vreg.0) // Shouldn't happen
        }
    };

    for inst in instructions {
        let line = match inst {
            VRegInst::I32Const { dst, val } => {
                format!("{} = i32.const {}", get_preg(*dst), val)
            }
            VRegInst::I64Const { dst, val } => {
                format!("{} = i64.const {}", get_preg(*dst), val)
            }

            VRegInst::I32Add { dst, src1, src2 } => {
                format!("{} = i32.add {}, {}", get_preg(*dst), get_preg(*src1), get_preg(*src2))
            }
            VRegInst::I32Sub { dst, src1, src2 } => {
                format!("{} = i32.sub {}, {}", get_preg(*dst), get_preg(*src1), get_preg(*src2))
            }
            VRegInst::I32Mul { dst, src1, src2 } => {
                format!("{} = i32.mul {}, {}", get_preg(*dst), get_preg(*src1), get_preg(*src2))
            }
            VRegInst::I32And { dst, src1, src2 } => {
                format!("{} = i32.and {}, {}", get_preg(*dst), get_preg(*src1), get_preg(*src2))
            }
            VRegInst::I32Or { dst, src1, src2 } => {
                format!("{} = i32.or {}, {}", get_preg(*dst), get_preg(*src1), get_preg(*src2))
            }
            VRegInst::I32Xor { dst, src1, src2 } => {
                format!("{} = i32.xor {}, {}", get_preg(*dst), get_preg(*src1), get_preg(*src2))
            }
            VRegInst::I32Shl { dst, src1, src2 } => {
                format!("{} = i32.shl {}, {}", get_preg(*dst), get_preg(*src1), get_preg(*src2))
            }
            VRegInst::I32ShrU { dst, src1, src2 } => {
                format!("{} = i32.shr_u {}, {}", get_preg(*dst), get_preg(*src1), get_preg(*src2))
            }
            VRegInst::I32ShrS { dst, src1, src2 } => {
                format!("{} = i32.shr_s {}, {}", get_preg(*dst), get_preg(*src1), get_preg(*src2))
            }

            VRegInst::I32Load { dst, addr, offset } => {
                format!("{} = i32.load {}[{}]", get_preg(*dst), get_preg(*addr), offset)
            }
            VRegInst::I64Load { dst, addr, offset } => {
                format!("{} = i64.load {}[{}]", get_preg(*dst), get_preg(*addr), offset)
            }
            VRegInst::I32Load8U { dst, addr, offset } => {
                format!("{} = i32.load8_u {}[{}]", get_preg(*dst), get_preg(*addr), offset)
            }

            VRegInst::I32Store { addr, offset, src } => {
                format!("i32.store {}[{}], {}", get_preg(*addr), offset, get_preg(*src))
            }
            VRegInst::I64Store { addr, offset, src } => {
                format!("i64.store {}[{}], {}", get_preg(*addr), offset, get_preg(*src))
            }

            VRegInst::I32Eqz { dst, src } => {
                format!("{} = i32.eqz {}", get_preg(*dst), get_preg(*src))
            }
            VRegInst::I32Eq { dst, src1, src2 } => {
                format!("{} = i32.eq {}, {}", get_preg(*dst), get_preg(*src1), get_preg(*src2))
            }
            VRegInst::I32Ne { dst, src1, src2 } => {
                format!("{} = i32.ne {}, {}", get_preg(*dst), get_preg(*src1), get_preg(*src2))
            }
            VRegInst::I32LtU { dst, src1, src2 } => {
                format!("{} = i32.lt_u {}, {}", get_preg(*dst), get_preg(*src1), get_preg(*src2))
            }
            VRegInst::I32GtU { dst, src1, src2 } => {
                format!("{} = i32.gt_u {}, {}", get_preg(*dst), get_preg(*src1), get_preg(*src2))
            }

            VRegInst::LocalGet { dst, local } => {
                format!("{} = local.get {}", get_preg(*dst), local)
            }
            VRegInst::LocalSet { local, src } => {
                format!("local.set {}, {}", local, get_preg(*src))
            }

            VRegInst::GlobalGet { dst, global } => {
                format!("{} = global.get {}", get_preg(*dst), global)
            }
            VRegInst::GlobalSet { global, src } => {
                format!("global.set {}, {}", global, get_preg(*src))
            }

            VRegInst::BrIf { cond, label } => {
                format!("br_if {}, L{}", get_preg(*cond), label)
            }
            VRegInst::Br { label } => {
                format!("br L{}", label)
            }
            VRegInst::Block { label } => {
                format!("block L{}:", label)
            }
            VRegInst::Loop { label } => {
                format!("loop L{}:", label)
            }
            VRegInst::End { label } => {
                format!("end L{}", label)
            }

            VRegInst::Call { func_idx, args, results } => {
                let args_str: Vec<_> = args.iter().map(|v| get_preg(*v)).collect();
                let res_str: Vec<_> = results.iter().map(|v| get_preg(*v)).collect();
                if results.is_empty() {
                    format!("call func{}({})", func_idx, args_str.join(", "))
                } else {
                    format!("{} = call func{}({})", res_str.join(", "), func_idx, args_str.join(", "))
                }
            }

            VRegInst::Select { dst, cond, src1, src2 } => {
                format!("{} = select {}, {}, {}", get_preg(*dst), get_preg(*cond), get_preg(*src1), get_preg(*src2))
            }

            VRegInst::Return { values } => {
                let vals: Vec<_> = values.iter().map(|v| get_preg(*v)).collect();
                format!("return {}", vals.join(", "))
            }

            VRegInst::Unreachable => "unreachable".to_string(),
            VRegInst::Nop => "nop".to_string(),
            VRegInst::Drop { src } => format!("drop {}", get_preg(*src)),

            _ => format!("{:?}", inst), // Fallback for unhandled
        };

        output.push(line);
    }

    output
}

/// Targeted linear scan register allocation with per-vreg register hints.
///
/// `hints` maps VReg → preferred physical register (0-based).
/// Vregs with a hint try to get that register; vregs without a hint
/// avoid hinted registers (especially r0) to reduce contention.
pub fn targeted_linear_scan_alloc(
    intervals: &[LiveInterval],
    num_regs: u32,
    hints: &HashMap<VReg, u8>,
) -> RegAllocResult {
    let mut sorted: Vec<_> = intervals.iter().collect();
    sorted.sort_by_key(|i| i.start);

    let mut vreg_to_preg: HashMap<VReg, PReg> = HashMap::new();
    let mut spilled: HashSet<VReg> = HashSet::new();
    let mut spill_slots: HashMap<VReg, SpillSlot> = HashMap::new();

    // Track which physical registers are free
    let mut free_regs: HashSet<u8> = (0..num_regs as u8).collect();
    // Active intervals: (end_point, vreg, preg)
    let mut active: BTreeSet<(u32, VReg, PReg)> = BTreeSet::new();
    let mut next_spill_slot = 0u32;

    // Which registers are "hot" (wanted by hints) — we avoid these for non-hinted vregs
    let mut hot_regs: HashSet<u8> = HashSet::new();
    for &h in hints.values() {
        hot_regs.insert(h);
    }

    for interval in sorted {
        // Expire old intervals
        let expired: Vec<_> = active
            .iter()
            .filter(|(end, _, _)| *end < interval.start)
            .copied()
            .collect();
        for (end, vreg, preg) in expired {
            active.remove(&(end, vreg, preg));
            free_regs.insert(preg.0);
        }

        if free_regs.is_empty() {
            // Spill logic: same as standard linear scan
            if let Some(&(end, vreg, preg)) = active.iter().last() {
                if end > interval.end {
                    // Spill the one ending later, keep current
                    active.remove(&(end, vreg, preg));
                    vreg_to_preg.remove(&vreg);
                    spilled.insert(vreg);
                    spill_slots.insert(vreg, SpillSlot(next_spill_slot));
                    next_spill_slot += 1;

                    vreg_to_preg.insert(interval.vreg, preg);
                    active.insert((interval.end, interval.vreg, preg));
                } else {
                    spilled.insert(interval.vreg);
                    spill_slots.insert(interval.vreg, SpillSlot(next_spill_slot));
                    next_spill_slot += 1;
                }
            } else {
                spilled.insert(interval.vreg);
                spill_slots.insert(interval.vreg, SpillSlot(next_spill_slot));
                next_spill_slot += 1;
            }
        } else {
            // HINT-AWARE register selection
            let hint = hints.get(&interval.vreg).copied();

            let preg = if let Some(h) = hint {
                if free_regs.contains(&h) {
                    // Got the hinted register
                    free_regs.remove(&h);
                    PReg(h)
                } else {
                    // Hinted reg busy — try to evict the occupant if it's non-hinted
                    // and ends later than us (standard spill heuristic)
                    let mut evicted = false;
                    let occupant = active.iter()
                        .find(|(_, _, p)| p.0 == h)
                        .copied();
                    if let Some((occ_end, occ_vreg, occ_preg)) = occupant {
                        // Only evict if occupant has no hint for this register
                        // and occupant ends later
                        let occ_hint = hints.get(&occ_vreg).copied();
                        if occ_hint != Some(h) && occ_end > interval.end {
                            // Evict occupant to another free register
                            // First check if there's a free non-hot register for it
                            let alt = free_regs.iter()
                                .filter(|&&r| !hot_regs.contains(&r))
                                .copied()
                                .next()
                                .or_else(|| free_regs.iter().copied().next());
                            if let Some(alt_r) = alt {
                                // Move occupant to alt register
                                active.remove(&(occ_end, occ_vreg, occ_preg));
                                free_regs.remove(&alt_r);
                                vreg_to_preg.insert(occ_vreg, PReg(alt_r));
                                active.insert((occ_end, occ_vreg, PReg(alt_r)));
                                // Give us the hinted register
                                vreg_to_preg.insert(interval.vreg, PReg(h));
                                active.insert((interval.end, interval.vreg, PReg(h)));
                                evicted = true;
                            }
                        }
                    }
                    if !evicted {
                        // Fall back: pick any free register, prefer non-hot
                        let pick = free_regs.iter()
                            .filter(|&&r| !hot_regs.contains(&r))
                            .copied()
                            .next()
                            .or_else(|| free_regs.iter().copied().next())
                            .unwrap();
                        free_regs.remove(&pick);
                        PReg(pick)
                    } else {
                        continue; // already inserted
                    }
                }
            } else {
                // No hint: pick a non-hot register to leave hot regs free
                let pick = free_regs.iter()
                    .filter(|&&r| !hot_regs.contains(&r))
                    .copied()
                    .next()
                    .or_else(|| free_regs.iter().copied().next())
                    .unwrap();
                free_regs.remove(&pick);
                PReg(pick)
            };

            vreg_to_preg.insert(interval.vreg, preg);
            active.insert((interval.end, interval.vreg, preg));
        }
    }

    RegAllocResult {
        vreg_to_preg,
        spilled,
        spill_slots,
        num_spill_slots: next_spill_slot,
    }
}

/// Statistics about the allocation
#[derive(Debug)]
pub struct AllocStats {
    pub num_vregs: u32,
    pub num_pregs: u32,
    pub num_spilled: u32,
    pub num_spill_slots: u32,
    pub total_instructions: u32,
    pub spill_load_count: u32,
    pub spill_store_count: u32,
}

/// Count actual spill loads/stores needed
pub fn count_spill_ops(instructions: &[VRegInst], alloc: &RegAllocResult) -> (u32, u32) {
    let mut loads = 0u32;
    let mut stores = 0u32;

    for inst in instructions {
        let (defs, uses) = get_vreg_defs_uses(inst);

        // Each use of a spilled vreg needs a load
        for vreg in uses {
            if alloc.spilled.contains(&vreg) {
                loads += 1;
            }
        }

        // Each def of a spilled vreg needs a store
        for vreg in defs {
            if alloc.spilled.contains(&vreg) {
                stores += 1;
            }
        }
    }

    (loads, stores)
}
