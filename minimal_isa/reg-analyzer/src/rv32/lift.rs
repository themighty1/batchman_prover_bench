//! Lifting: transforms decoded instructions to virtual-register IR.
//!
//! Steps:
//!   1. `classify_jalr_x0` (cfg.rs) — tag indirect jumps as ret/jr_table/jr_computed
//!   2. `build_cfg` (cfg.rs) — build basic blocks and edges
//!   3. `lift_to_vregs` — assign a fresh virtual register for each definition

use super::decode::DecodedInst;
use super::cfg::{BasicBlock, is_call};

// ---------------------------------------------------------------------------
// Virtual-register lifted instruction
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Rv32VRegInst {
    pub addr: u32,
    pub op: String,
    pub rd: Option<u32>,      // destination vreg
    pub rs1: Option<u32>,     // source vreg 1
    pub rs2: Option<u32>,     // source vreg 2
    pub imm: Option<i32>,
    pub orig_rd_preg: Option<u8>,
    pub orig_rs1_preg: Option<u8>,
    pub orig_rs2_preg: Option<u8>,
}

/// Per-block vreg state after lifting
#[derive(Debug, Clone)]
pub struct BlockVRegInfo {
    pub block_id: usize,
    pub entry_vregs: [u32; 32],  // vreg for each preg at block entry
    pub exit_vregs: [u32; 32],   // vreg for each preg at block exit
    pub insts: Vec<Rv32VRegInst>,
}

/// Caller-saved registers: ra, t0-t2, a0-a7, t3-t6
/// These may be clobbered by a function call.
const CALLER_SAVED: [u8; 16] = [1, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 28, 29, 30, 31];

/// Lift physical-register instructions to virtual registers.
///
/// Within each basic block:
/// - At entry, allocate a fresh "live-in" vreg for each physical register.
/// - Each definition (rd) creates a new vreg.
/// - Each use (rs1, rs2) refers to the current vreg for that physical register.
/// - After a call instruction, invalidate caller-saved registers (fresh vregs).
///
/// Cross-block connections (block params) are deferred to regalloc2 integration.
pub fn lift_to_vregs(
    decoded: &[DecodedInst],
    blocks: &[BasicBlock],
) -> (Vec<BlockVRegInfo>, u32) {
    let mut next_vreg = 0u32;
    let mut block_infos = Vec::new();

    // Vreg 0 is reserved for x0 (always zero)
    let zero_vreg = next_vreg;
    next_vreg += 1;

    for block in blocks {
        // Allocate entry vregs for this block
        let mut current = [0u32; 32];
        current[0] = zero_vreg; // x0 always maps to vreg 0
        for r in 1..32 {
            current[r] = next_vreg;
            next_vreg += 1;
        }
        let entry_vregs = current;

        let mut insts = Vec::new();

        for i in block.start..block.end {
            let d = &decoded[i];

            // Read sources first (before updating dest)
            let rs1_vreg = d.rs1.map(|r| current[r as usize]);
            let rs2_vreg = d.rs2.map(|r| current[r as usize]);

            // Create fresh vreg for destination
            let rd_vreg = d.rd.map(|r| {
                if r == 0 {
                    zero_vreg // writes to x0 are sinks
                } else {
                    let v = next_vreg;
                    next_vreg += 1;
                    current[r as usize] = v;
                    v
                }
            });

            insts.push(Rv32VRegInst {
                addr: d.addr,
                op: d.op.clone(),
                rd: rd_vreg,
                rs1: rs1_vreg,
                rs2: rs2_vreg,
                imm: d.imm,
                orig_rd_preg: d.rd,
                orig_rs1_preg: d.rs1,
                orig_rs2_preg: d.rs2,
            });

            // After a call, create fresh vregs for caller-saved registers
            // (the callee may have modified them)
            if is_call(d) {
                for &r in &CALLER_SAVED {
                    let v = next_vreg;
                    next_vreg += 1;
                    current[r as usize] = v;
                }
            }
        }

        let exit_vregs = current;
        block_infos.push(BlockVRegInfo {
            block_id: block.id,
            entry_vregs,
            exit_vregs,
            insts,
        });
    }

    (block_infos, next_vreg)
}

/// Produce a specialized opcode string with vreg ids baked in.
pub fn specialized_vreg_opcode(inst: &Rv32VRegInst) -> String {
    let mut name = inst.op.clone();
    if let Some(rd)  = inst.rd  { name = format!("{}.v{}", name, rd); }
    if let Some(rs1) = inst.rs1 { name = format!("{}.v{}", name, rs1); }
    if let Some(rs2) = inst.rs2 { name = format!("{}.v{}", name, rs2); }
    name
}
