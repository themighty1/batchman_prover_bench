//! Physical register VM
//!
//! Executes register-allocated IR with a fixed number of physical registers
//! and spill slots in memory.

use crate::regvm::{PReg, SpillSlot, VReg, VRegInst, RegInst, FRAME_SP_ADDR, SLOT_SIZE};
use crate::regalloc::RegAllocResult;
use crate::interpreter::Value;
use std::collections::HashMap;

/// Classify a VRegInst by its PReg instruction name
pub fn vreg_inst_name(inst: &VRegInst) -> &'static str {
    use VRegInst::*;
    match inst {
        I32Const { .. } => "i32.const",
        I64Const { .. } => "i64.const",
        I32Add { .. } => "i32.add",
        I32Sub { .. } => "i32.sub",
        I32Mul { .. } => "i32.mul",
        I32DivS { .. } => "i32.div_s",
        I32DivU { .. } => "i32.div_u",
        I32RemS { .. } => "i32.rem_s",
        I32RemU { .. } => "i32.rem_u",
        I32And { .. } => "i32.and",
        I32Or { .. } => "i32.or",
        I32Xor { .. } => "i32.xor",
        I32Shl { .. } => "i32.shl",
        I32ShrU { .. } => "i32.shr_u",
        I32ShrS { .. } => "i32.shr_s",
        I32Rotl { .. } => "i32.rotl",
        I32Rotr { .. } => "i32.rotr",
        I64Add { .. } => "i64.add",
        I64Sub { .. } => "i64.sub",
        I64Mul { .. } => "i64.mul",
        I64DivS { .. } => "i64.div_s",
        I64DivU { .. } => "i64.div_u",
        I64RemS { .. } => "i64.rem_s",
        I64RemU { .. } => "i64.rem_u",
        I64And { .. } => "i64.and",
        I64Or { .. } => "i64.or",
        I64Xor { .. } => "i64.xor",
        I64Shl { .. } => "i64.shl",
        I64ShrU { .. } => "i64.shr_u",
        I64ShrS { .. } => "i64.shr_s",
        I32Eqz { .. } => "i32.eqz",
        I64Eqz { .. } => "i64.eqz",
        I32Clz { .. } => "i32.clz",
        I32Ctz { .. } => "i32.ctz",
        I32Popcnt { .. } => "i32.popcnt",
        I64Clz { .. } => "i64.clz",
        I64Ctz { .. } => "i64.ctz",
        I32WrapI64 { .. } => "i32.wrap_i64",
        I64ExtendI32S { .. } => "i64.extend_i32_s",
        I64ExtendI32U { .. } => "i64.extend_i32_u",
        I32Extend8S { .. } => "i32.extend8_s",
        I32Extend16S { .. } => "i32.extend16_s",
        I32Eq { .. } => "i32.eq",
        I32Ne { .. } => "i32.ne",
        I32LtS { .. } => "i32.lt_s",
        I32LtU { .. } => "i32.lt_u",
        I32GtS { .. } => "i32.gt_s",
        I32GtU { .. } => "i32.gt_u",
        I32LeS { .. } => "i32.le_s",
        I32LeU { .. } => "i32.le_u",
        I32GeS { .. } => "i32.ge_s",
        I32GeU { .. } => "i32.ge_u",
        I64Eq { .. } => "i64.eq",
        I64Ne { .. } => "i64.ne",
        I64LtS { .. } => "i64.lt_s",
        I64LtU { .. } => "i64.lt_u",
        I64GtS { .. } => "i64.gt_s",
        I64GtU { .. } => "i64.gt_u",
        I64LeS { .. } => "i64.le_s",
        I64LeU { .. } => "i64.le_u",
        I64GeS { .. } => "i64.ge_s",
        I64GeU { .. } => "i64.ge_u",
        I32Load { .. } => "i32.load",
        I64Load { .. } => "i64.load",
        I32Load8U { .. } => "i32.load8_u",
        I32Load8S { .. } => "i32.load8_s",
        I32Load16U { .. } => "i32.load16_u",
        I32Load16S { .. } => "i32.load16_s",
        I64Load8U { .. } => "i64.load8_u",
        I64Load8S { .. } => "i64.load8_s",
        I64Load16U { .. } => "i64.load16_u",
        I64Load16S { .. } => "i64.load16_s",
        I64Load32U { .. } => "i64.load32_u",
        I64Load32S { .. } => "i64.load32_s",
        I32Store { .. } => "i32.store",
        I64Store { .. } => "i64.store",
        I32Store8 { .. } => "i32.store8",
        I32Store16 { .. } => "i32.store16",
        I64Store8 { .. } => "i64.store8",
        I64Store16 { .. } => "i64.store16",
        I64Store32 { .. } => "i64.store32",
        LocalGet { .. } => "local.get",
        LocalSet { .. } => "local.set",
        LocalTee { .. } => "local.tee",
        GlobalGet { .. } => "global.get",
        GlobalSet { .. } => "global.set",
        Call { .. } => "call",
        CallIndirect { .. } => "call_indirect",
        Block { .. } => "block",
        Loop { .. } => "loop",
        If { .. } => "if",
        Else { .. } => "else",
        End { .. } => "end",
        Br { .. } => "br",
        BrIf { .. } => "br_if",
        BrTable { .. } => "br_table",
        Return { .. } => "return",
        Select { .. } => "select",
        Unreachable => "unreachable",
        Nop => "nop",
        Drop { .. } => "drop",
        MemorySize { .. } => "memory.size",
        MemoryGrow { .. } => "memory.grow",
        MemoryCopy { .. } => "memory.copy",
        MemoryFill { .. } => "memory.fill",
        Mov { .. } => "mov",
    }
}

/// Get source VRegs of a VRegInst (for spill/reload tracking)
pub fn vreg_src_regs(inst: &VRegInst) -> Vec<VReg> {
    use VRegInst::*;
    match inst {
        // Binary ops: 2 sources
        I32Add { src1, src2, .. } | I32Sub { src1, src2, .. } | I32Mul { src1, src2, .. } |
        I32DivS { src1, src2, .. } | I32DivU { src1, src2, .. } |
        I32RemS { src1, src2, .. } | I32RemU { src1, src2, .. } |
        I32And { src1, src2, .. } | I32Or { src1, src2, .. } | I32Xor { src1, src2, .. } |
        I32Shl { src1, src2, .. } | I32ShrU { src1, src2, .. } | I32ShrS { src1, src2, .. } |
        I32Rotl { src1, src2, .. } | I32Rotr { src1, src2, .. } |
        I64Add { src1, src2, .. } | I64Sub { src1, src2, .. } | I64Mul { src1, src2, .. } |
        I64DivS { src1, src2, .. } | I64DivU { src1, src2, .. } |
        I64RemS { src1, src2, .. } | I64RemU { src1, src2, .. } |
        I64And { src1, src2, .. } | I64Or { src1, src2, .. } | I64Xor { src1, src2, .. } |
        I64Shl { src1, src2, .. } | I64ShrU { src1, src2, .. } | I64ShrS { src1, src2, .. } |
        I32Eq { src1, src2, .. } | I32Ne { src1, src2, .. } |
        I32LtS { src1, src2, .. } | I32LtU { src1, src2, .. } |
        I32GtS { src1, src2, .. } | I32GtU { src1, src2, .. } |
        I32LeS { src1, src2, .. } | I32LeU { src1, src2, .. } |
        I32GeS { src1, src2, .. } | I32GeU { src1, src2, .. } |
        I64Eq { src1, src2, .. } | I64Ne { src1, src2, .. } |
        I64LtS { src1, src2, .. } | I64LtU { src1, src2, .. } |
        I64GtS { src1, src2, .. } | I64GtU { src1, src2, .. } |
        I64LeS { src1, src2, .. } | I64LeU { src1, src2, .. } |
        I64GeS { src1, src2, .. } | I64GeU { src1, src2, .. }
            => vec![*src1, *src2],

        // Unary ops: 1 source
        I32Eqz { src, .. } | I64Eqz { src, .. } |
        I32Clz { src, .. } | I32Ctz { src, .. } | I32Popcnt { src, .. } |
        I64Clz { src, .. } | I64Ctz { src, .. } |
        I32WrapI64 { src, .. } | I64ExtendI32S { src, .. } | I64ExtendI32U { src, .. } |
        I32Extend8S { src, .. } | I32Extend16S { src, .. }
            => vec![*src],

        // Loads: addr is source
        I32Load { addr, .. } | I64Load { addr, .. } |
        I32Load8U { addr, .. } | I32Load8S { addr, .. } |
        I32Load16U { addr, .. } | I32Load16S { addr, .. } |
        I64Load8U { addr, .. } | I64Load8S { addr, .. } |
        I64Load16U { addr, .. } | I64Load16S { addr, .. } |
        I64Load32U { addr, .. } | I64Load32S { addr, .. }
            => vec![*addr],

        // Stores: addr + src
        I32Store { addr, src, .. } | I64Store { addr, src, .. } |
        I32Store8 { addr, src, .. } | I32Store16 { addr, src, .. } |
        I64Store8 { addr, src, .. } | I64Store16 { addr, src, .. } |
        I64Store32 { addr, src, .. }
            => vec![*addr, *src],

        // Local/Global
        LocalSet { src, .. } => vec![*src],
        LocalTee { src, .. } => vec![*src],
        GlobalSet { src, .. } => vec![*src],

        // Select: cond + two sources
        Select { cond, src1, src2, .. } => vec![*cond, *src1, *src2],

        // Branch: condition
        BrIf { cond, .. } => vec![*cond],
        If { cond, .. } => vec![*cond],
        BrTable { idx, .. } => vec![*idx],

        // Memory ops
        MemoryGrow { pages, .. } => vec![*pages],
        MemoryCopy { dst, src, len } => vec![*dst, *src, *len],
        MemoryFill { dst, val, len } => vec![*dst, *val, *len],

        // Mov
        Mov { src, .. } => vec![*src],

        // Call: args are sources
        Call { args, .. } => args.clone(),
        CallIndirect { args, func_ref, .. } => {
            let mut v = args.clone();
            v.push(*func_ref);
            v
        }

        // Return: values are sources
        Return { values, .. } => values.clone(),

        // No source regs
        _ => vec![],
    }
}

/// Get destination VRegs of a VRegInst (for spill tracking)
pub fn vreg_dst_regs(inst: &VRegInst) -> Vec<VReg> {
    use VRegInst::*;
    match inst {
        I32Const { dst, .. } | I64Const { dst, .. } |
        I32Add { dst, .. } | I32Sub { dst, .. } | I32Mul { dst, .. } |
        I32DivS { dst, .. } | I32DivU { dst, .. } |
        I32RemS { dst, .. } | I32RemU { dst, .. } |
        I32And { dst, .. } | I32Or { dst, .. } | I32Xor { dst, .. } |
        I32Shl { dst, .. } | I32ShrU { dst, .. } | I32ShrS { dst, .. } |
        I32Rotl { dst, .. } | I32Rotr { dst, .. } |
        I64Add { dst, .. } | I64Sub { dst, .. } | I64Mul { dst, .. } |
        I64DivS { dst, .. } | I64DivU { dst, .. } |
        I64RemS { dst, .. } | I64RemU { dst, .. } |
        I64And { dst, .. } | I64Or { dst, .. } | I64Xor { dst, .. } |
        I64Shl { dst, .. } | I64ShrU { dst, .. } | I64ShrS { dst, .. } |
        I32Eqz { dst, .. } | I64Eqz { dst, .. } |
        I32Clz { dst, .. } | I32Ctz { dst, .. } | I32Popcnt { dst, .. } |
        I64Clz { dst, .. } | I64Ctz { dst, .. } |
        I32WrapI64 { dst, .. } | I64ExtendI32S { dst, .. } | I64ExtendI32U { dst, .. } |
        I32Extend8S { dst, .. } | I32Extend16S { dst, .. } |
        I32Eq { dst, .. } | I32Ne { dst, .. } |
        I32LtS { dst, .. } | I32LtU { dst, .. } |
        I32GtS { dst, .. } | I32GtU { dst, .. } |
        I32LeS { dst, .. } | I32LeU { dst, .. } |
        I32GeS { dst, .. } | I32GeU { dst, .. } |
        I64Eq { dst, .. } | I64Ne { dst, .. } |
        I64LtS { dst, .. } | I64LtU { dst, .. } |
        I64GtS { dst, .. } | I64GtU { dst, .. } |
        I64LeS { dst, .. } | I64LeU { dst, .. } |
        I64GeS { dst, .. } | I64GeU { dst, .. } |
        I32Load { dst, .. } | I64Load { dst, .. } |
        I32Load8U { dst, .. } | I32Load8S { dst, .. } |
        I32Load16U { dst, .. } | I32Load16S { dst, .. } |
        I64Load8U { dst, .. } | I64Load8S { dst, .. } |
        I64Load16U { dst, .. } | I64Load16S { dst, .. } |
        I64Load32U { dst, .. } | I64Load32S { dst, .. } |
        LocalGet { dst, .. } | GlobalGet { dst, .. } |
        Select { dst, .. } |
        MemorySize { dst } | MemoryGrow { dst, .. } |
        Mov { dst, .. }
            => vec![*dst],

        LocalTee { dst, .. } => vec![*dst],

        // Call: results are destinations
        Call { results, .. } => results.clone(),
        CallIndirect { results, .. } => results.clone(),

        _ => vec![],
    }
}

/// Build a register-specialized opcode string for a VRegInst.
///
/// Maps each vreg operand to its physical register (or "s<N>" for spilled)
/// and encodes them into the opcode name, e.g. "i32.add.r4.r1.r2".
/// This gives the fully-specialized instruction identity when registers are
/// part of the opcode.
pub fn specialized_opcode(inst: &VRegInst, alloc: &RegAllocResult) -> String {
    let reg_name = |vreg: VReg| -> String {
        if let Some(&preg) = alloc.vreg_to_preg.get(&vreg) {
            format!("r{}", preg.0)
        } else if let Some(&slot) = alloc.spill_slots.get(&vreg) {
            format!("s{}", slot.0)
        } else {
            "?".to_string()
        }
    };

    let name = vreg_inst_name(inst);
    let dsts = vreg_dst_regs(inst);
    let srcs = vreg_src_regs(inst);

    if dsts.is_empty() && srcs.is_empty() {
        return name.to_string();
    }

    let mut parts = vec![name.to_string()];
    for d in &dsts { parts.push(reg_name(*d)); }
    for s in &srcs { parts.push(reg_name(*s)); }
    parts.join(".")
}

/// Replace vreg operands in a VRegInst using a mapping.
/// Vregs not in the map are left unchanged.
pub fn replace_vregs(inst: &VRegInst, map: &std::collections::HashMap<VReg, VReg>) -> VRegInst {
    let m = |v: VReg| -> VReg { map.get(&v).copied().unwrap_or(v) };
    use VRegInst::*;
    macro_rules! bin { ($V:ident, $d:expr, $s1:expr, $s2:expr) => { $V { dst: m(*$d), src1: m(*$s1), src2: m(*$s2) } } }
    macro_rules! una { ($V:ident, $d:expr, $s:expr) => { $V { dst: m(*$d), src: m(*$s) } } }
    macro_rules! ld  { ($V:ident, $d:expr, $a:expr, $o:expr) => { $V { dst: m(*$d), addr: m(*$a), offset: *$o } } }
    macro_rules! st  { ($V:ident, $a:expr, $o:expr, $s:expr) => { $V { addr: m(*$a), offset: *$o, src: m(*$s) } } }
    match inst {
        I32Const { dst, val } => I32Const { dst: m(*dst), val: *val },
        I64Const { dst, val } => I64Const { dst: m(*dst), val: *val },
        I32Add { dst, src1, src2 } => bin!(I32Add, dst, src1, src2),
        I32Sub { dst, src1, src2 } => bin!(I32Sub, dst, src1, src2),
        I32Mul { dst, src1, src2 } => bin!(I32Mul, dst, src1, src2),
        I32DivS { dst, src1, src2 } => bin!(I32DivS, dst, src1, src2),
        I32DivU { dst, src1, src2 } => bin!(I32DivU, dst, src1, src2),
        I32RemS { dst, src1, src2 } => bin!(I32RemS, dst, src1, src2),
        I32RemU { dst, src1, src2 } => bin!(I32RemU, dst, src1, src2),
        I32And { dst, src1, src2 } => bin!(I32And, dst, src1, src2),
        I32Or { dst, src1, src2 } => bin!(I32Or, dst, src1, src2),
        I32Xor { dst, src1, src2 } => bin!(I32Xor, dst, src1, src2),
        I32Shl { dst, src1, src2 } => bin!(I32Shl, dst, src1, src2),
        I32ShrU { dst, src1, src2 } => bin!(I32ShrU, dst, src1, src2),
        I32ShrS { dst, src1, src2 } => bin!(I32ShrS, dst, src1, src2),
        I32Rotl { dst, src1, src2 } => bin!(I32Rotl, dst, src1, src2),
        I32Rotr { dst, src1, src2 } => bin!(I32Rotr, dst, src1, src2),
        I64Add { dst, src1, src2 } => bin!(I64Add, dst, src1, src2),
        I64Sub { dst, src1, src2 } => bin!(I64Sub, dst, src1, src2),
        I64Mul { dst, src1, src2 } => bin!(I64Mul, dst, src1, src2),
        I64DivS { dst, src1, src2 } => bin!(I64DivS, dst, src1, src2),
        I64DivU { dst, src1, src2 } => bin!(I64DivU, dst, src1, src2),
        I64RemS { dst, src1, src2 } => bin!(I64RemS, dst, src1, src2),
        I64RemU { dst, src1, src2 } => bin!(I64RemU, dst, src1, src2),
        I64And { dst, src1, src2 } => bin!(I64And, dst, src1, src2),
        I64Or { dst, src1, src2 } => bin!(I64Or, dst, src1, src2),
        I64Xor { dst, src1, src2 } => bin!(I64Xor, dst, src1, src2),
        I64Shl { dst, src1, src2 } => bin!(I64Shl, dst, src1, src2),
        I64ShrU { dst, src1, src2 } => bin!(I64ShrU, dst, src1, src2),
        I64ShrS { dst, src1, src2 } => bin!(I64ShrS, dst, src1, src2),
        I32Eqz { dst, src } => una!(I32Eqz, dst, src),
        I64Eqz { dst, src } => una!(I64Eqz, dst, src),
        I32Clz { dst, src } => una!(I32Clz, dst, src),
        I32Ctz { dst, src } => una!(I32Ctz, dst, src),
        I32Popcnt { dst, src } => una!(I32Popcnt, dst, src),
        I64Clz { dst, src } => una!(I64Clz, dst, src),
        I64Ctz { dst, src } => una!(I64Ctz, dst, src),
        I32WrapI64 { dst, src } => una!(I32WrapI64, dst, src),
        I64ExtendI32S { dst, src } => una!(I64ExtendI32S, dst, src),
        I64ExtendI32U { dst, src } => una!(I64ExtendI32U, dst, src),
        I32Extend8S { dst, src } => una!(I32Extend8S, dst, src),
        I32Extend16S { dst, src } => una!(I32Extend16S, dst, src),
        I32Eq { dst, src1, src2 } => bin!(I32Eq, dst, src1, src2),
        I32Ne { dst, src1, src2 } => bin!(I32Ne, dst, src1, src2),
        I32LtS { dst, src1, src2 } => bin!(I32LtS, dst, src1, src2),
        I32LtU { dst, src1, src2 } => bin!(I32LtU, dst, src1, src2),
        I32GtS { dst, src1, src2 } => bin!(I32GtS, dst, src1, src2),
        I32GtU { dst, src1, src2 } => bin!(I32GtU, dst, src1, src2),
        I32LeS { dst, src1, src2 } => bin!(I32LeS, dst, src1, src2),
        I32LeU { dst, src1, src2 } => bin!(I32LeU, dst, src1, src2),
        I32GeS { dst, src1, src2 } => bin!(I32GeS, dst, src1, src2),
        I32GeU { dst, src1, src2 } => bin!(I32GeU, dst, src1, src2),
        I64Eq { dst, src1, src2 } => bin!(I64Eq, dst, src1, src2),
        I64Ne { dst, src1, src2 } => bin!(I64Ne, dst, src1, src2),
        I64LtS { dst, src1, src2 } => bin!(I64LtS, dst, src1, src2),
        I64LtU { dst, src1, src2 } => bin!(I64LtU, dst, src1, src2),
        I64GtS { dst, src1, src2 } => bin!(I64GtS, dst, src1, src2),
        I64GtU { dst, src1, src2 } => bin!(I64GtU, dst, src1, src2),
        I64LeS { dst, src1, src2 } => bin!(I64LeS, dst, src1, src2),
        I64LeU { dst, src1, src2 } => bin!(I64LeU, dst, src1, src2),
        I64GeS { dst, src1, src2 } => bin!(I64GeS, dst, src1, src2),
        I64GeU { dst, src1, src2 } => bin!(I64GeU, dst, src1, src2),
        I32Load { dst, addr, offset } => ld!(I32Load, dst, addr, offset),
        I64Load { dst, addr, offset } => ld!(I64Load, dst, addr, offset),
        I32Load8U { dst, addr, offset } => ld!(I32Load8U, dst, addr, offset),
        I32Load8S { dst, addr, offset } => ld!(I32Load8S, dst, addr, offset),
        I32Load16U { dst, addr, offset } => ld!(I32Load16U, dst, addr, offset),
        I32Load16S { dst, addr, offset } => ld!(I32Load16S, dst, addr, offset),
        I64Load8U { dst, addr, offset } => ld!(I64Load8U, dst, addr, offset),
        I64Load8S { dst, addr, offset } => ld!(I64Load8S, dst, addr, offset),
        I64Load16U { dst, addr, offset } => ld!(I64Load16U, dst, addr, offset),
        I64Load16S { dst, addr, offset } => ld!(I64Load16S, dst, addr, offset),
        I64Load32U { dst, addr, offset } => ld!(I64Load32U, dst, addr, offset),
        I64Load32S { dst, addr, offset } => ld!(I64Load32S, dst, addr, offset),
        I32Store { addr, offset, src } => st!(I32Store, addr, offset, src),
        I64Store { addr, offset, src } => st!(I64Store, addr, offset, src),
        I32Store8 { addr, offset, src } => st!(I32Store8, addr, offset, src),
        I32Store16 { addr, offset, src } => st!(I32Store16, addr, offset, src),
        I64Store8 { addr, offset, src } => st!(I64Store8, addr, offset, src),
        I64Store16 { addr, offset, src } => st!(I64Store16, addr, offset, src),
        I64Store32 { addr, offset, src } => st!(I64Store32, addr, offset, src),
        LocalGet { dst, local } => LocalGet { dst: m(*dst), local: *local },
        LocalSet { local, src } => LocalSet { local: *local, src: m(*src) },
        LocalTee { dst, local, src } => LocalTee { dst: m(*dst), local: *local, src: m(*src) },
        GlobalGet { dst, global } => GlobalGet { dst: m(*dst), global: *global },
        GlobalSet { global, src } => GlobalSet { global: *global, src: m(*src) },
        Call { func_idx, args, results } => Call {
            func_idx: *func_idx,
            args: args.iter().map(|v| m(*v)).collect(),
            results: results.iter().map(|v| m(*v)).collect(),
        },
        CallIndirect { table, type_idx, func_ref, args, results } => CallIndirect {
            table: *table, type_idx: *type_idx, func_ref: m(*func_ref),
            args: args.iter().map(|v| m(*v)).collect(),
            results: results.iter().map(|v| m(*v)).collect(),
        },
        Select { dst, cond, src1, src2 } => Select { dst: m(*dst), cond: m(*cond), src1: m(*src1), src2: m(*src2) },
        BrIf { cond, label } => BrIf { cond: m(*cond), label: *label },
        BrTable { idx, labels, default } => BrTable { idx: m(*idx), labels: labels.clone(), default: *default },
        Br { label } => Br { label: *label },
        Block { label } => Block { label: *label },
        Loop { label } => Loop { label: *label },
        If { cond, label } => If { cond: m(*cond), label: *label },
        Else { label } => Else { label: *label },
        End { label } => End { label: *label },
        Return { values } => Return { values: values.iter().map(|v| m(*v)).collect() },
        Unreachable => Unreachable,
        Nop => Nop,
        Drop { src } => Drop { src: m(*src) },
        MemorySize { dst } => MemorySize { dst: m(*dst) },
        MemoryGrow { dst, pages } => MemoryGrow { dst: m(*dst), pages: m(*pages) },
        MemoryCopy { dst, src, len } => MemoryCopy { dst: m(*dst), src: m(*src), len: m(*len) },
        MemoryFill { dst, val, len } => MemoryFill { dst: m(*dst), val: m(*val), len: m(*len) },
        Mov { dst, src } => Mov { dst: m(*dst), src: m(*src) },
    }
}

/// Rewrite a VRegInst with specific dst and src vregs (by position).
///
/// `new_dsts` and `new_srcs` must match the order returned by
/// `vreg_dst_regs()` and `vreg_src_regs()` respectively.
/// This allows different physical register mappings for dst vs src
/// of the same original vreg (needed for regalloc2 per-instruction allocations).
pub fn rewrite_inst_vregs(inst: &VRegInst, new_dsts: &[VReg], new_srcs: &[VReg]) -> VRegInst {
    use VRegInst::*;
    // d(i) gets the i-th new dst, s(i) gets the i-th new src
    macro_rules! d { ($i:expr) => { new_dsts[$i] } }
    macro_rules! s { ($i:expr) => { new_srcs[$i] } }

    match inst {
        // Constants: 1 dst, 0 src
        I32Const { val, .. } => I32Const { dst: d!(0), val: *val },
        I64Const { val, .. } => I64Const { dst: d!(0), val: *val },

        // Binary ops: 1 dst, 2 src
        I32Add { .. } => I32Add { dst: d!(0), src1: s!(0), src2: s!(1) },
        I32Sub { .. } => I32Sub { dst: d!(0), src1: s!(0), src2: s!(1) },
        I32Mul { .. } => I32Mul { dst: d!(0), src1: s!(0), src2: s!(1) },
        I32DivS { .. } => I32DivS { dst: d!(0), src1: s!(0), src2: s!(1) },
        I32DivU { .. } => I32DivU { dst: d!(0), src1: s!(0), src2: s!(1) },
        I32RemS { .. } => I32RemS { dst: d!(0), src1: s!(0), src2: s!(1) },
        I32RemU { .. } => I32RemU { dst: d!(0), src1: s!(0), src2: s!(1) },
        I32And { .. } => I32And { dst: d!(0), src1: s!(0), src2: s!(1) },
        I32Or { .. } => I32Or { dst: d!(0), src1: s!(0), src2: s!(1) },
        I32Xor { .. } => I32Xor { dst: d!(0), src1: s!(0), src2: s!(1) },
        I32Shl { .. } => I32Shl { dst: d!(0), src1: s!(0), src2: s!(1) },
        I32ShrU { .. } => I32ShrU { dst: d!(0), src1: s!(0), src2: s!(1) },
        I32ShrS { .. } => I32ShrS { dst: d!(0), src1: s!(0), src2: s!(1) },
        I32Rotl { .. } => I32Rotl { dst: d!(0), src1: s!(0), src2: s!(1) },
        I32Rotr { .. } => I32Rotr { dst: d!(0), src1: s!(0), src2: s!(1) },
        I64Add { .. } => I64Add { dst: d!(0), src1: s!(0), src2: s!(1) },
        I64Sub { .. } => I64Sub { dst: d!(0), src1: s!(0), src2: s!(1) },
        I64Mul { .. } => I64Mul { dst: d!(0), src1: s!(0), src2: s!(1) },
        I64DivS { .. } => I64DivS { dst: d!(0), src1: s!(0), src2: s!(1) },
        I64DivU { .. } => I64DivU { dst: d!(0), src1: s!(0), src2: s!(1) },
        I64RemS { .. } => I64RemS { dst: d!(0), src1: s!(0), src2: s!(1) },
        I64RemU { .. } => I64RemU { dst: d!(0), src1: s!(0), src2: s!(1) },
        I64And { .. } => I64And { dst: d!(0), src1: s!(0), src2: s!(1) },
        I64Or { .. } => I64Or { dst: d!(0), src1: s!(0), src2: s!(1) },
        I64Xor { .. } => I64Xor { dst: d!(0), src1: s!(0), src2: s!(1) },
        I64Shl { .. } => I64Shl { dst: d!(0), src1: s!(0), src2: s!(1) },
        I64ShrU { .. } => I64ShrU { dst: d!(0), src1: s!(0), src2: s!(1) },
        I64ShrS { .. } => I64ShrS { dst: d!(0), src1: s!(0), src2: s!(1) },
        I32Eq { .. } => I32Eq { dst: d!(0), src1: s!(0), src2: s!(1) },
        I32Ne { .. } => I32Ne { dst: d!(0), src1: s!(0), src2: s!(1) },
        I32LtS { .. } => I32LtS { dst: d!(0), src1: s!(0), src2: s!(1) },
        I32LtU { .. } => I32LtU { dst: d!(0), src1: s!(0), src2: s!(1) },
        I32GtS { .. } => I32GtS { dst: d!(0), src1: s!(0), src2: s!(1) },
        I32GtU { .. } => I32GtU { dst: d!(0), src1: s!(0), src2: s!(1) },
        I32LeS { .. } => I32LeS { dst: d!(0), src1: s!(0), src2: s!(1) },
        I32LeU { .. } => I32LeU { dst: d!(0), src1: s!(0), src2: s!(1) },
        I32GeS { .. } => I32GeS { dst: d!(0), src1: s!(0), src2: s!(1) },
        I32GeU { .. } => I32GeU { dst: d!(0), src1: s!(0), src2: s!(1) },
        I64Eq { .. } => I64Eq { dst: d!(0), src1: s!(0), src2: s!(1) },
        I64Ne { .. } => I64Ne { dst: d!(0), src1: s!(0), src2: s!(1) },
        I64LtS { .. } => I64LtS { dst: d!(0), src1: s!(0), src2: s!(1) },
        I64LtU { .. } => I64LtU { dst: d!(0), src1: s!(0), src2: s!(1) },
        I64GtS { .. } => I64GtS { dst: d!(0), src1: s!(0), src2: s!(1) },
        I64GtU { .. } => I64GtU { dst: d!(0), src1: s!(0), src2: s!(1) },
        I64LeS { .. } => I64LeS { dst: d!(0), src1: s!(0), src2: s!(1) },
        I64LeU { .. } => I64LeU { dst: d!(0), src1: s!(0), src2: s!(1) },
        I64GeS { .. } => I64GeS { dst: d!(0), src1: s!(0), src2: s!(1) },
        I64GeU { .. } => I64GeU { dst: d!(0), src1: s!(0), src2: s!(1) },

        // Unary ops: 1 dst, 1 src
        I32Eqz { .. } => I32Eqz { dst: d!(0), src: s!(0) },
        I64Eqz { .. } => I64Eqz { dst: d!(0), src: s!(0) },
        I32Clz { .. } => I32Clz { dst: d!(0), src: s!(0) },
        I32Ctz { .. } => I32Ctz { dst: d!(0), src: s!(0) },
        I32Popcnt { .. } => I32Popcnt { dst: d!(0), src: s!(0) },
        I64Clz { .. } => I64Clz { dst: d!(0), src: s!(0) },
        I64Ctz { .. } => I64Ctz { dst: d!(0), src: s!(0) },
        I32WrapI64 { .. } => I32WrapI64 { dst: d!(0), src: s!(0) },
        I64ExtendI32S { .. } => I64ExtendI32S { dst: d!(0), src: s!(0) },
        I64ExtendI32U { .. } => I64ExtendI32U { dst: d!(0), src: s!(0) },
        I32Extend8S { .. } => I32Extend8S { dst: d!(0), src: s!(0) },
        I32Extend16S { .. } => I32Extend16S { dst: d!(0), src: s!(0) },

        // Loads: 1 dst, 1 src (addr)
        I32Load { offset, .. } => I32Load { dst: d!(0), addr: s!(0), offset: *offset },
        I64Load { offset, .. } => I64Load { dst: d!(0), addr: s!(0), offset: *offset },
        I32Load8U { offset, .. } => I32Load8U { dst: d!(0), addr: s!(0), offset: *offset },
        I32Load8S { offset, .. } => I32Load8S { dst: d!(0), addr: s!(0), offset: *offset },
        I32Load16U { offset, .. } => I32Load16U { dst: d!(0), addr: s!(0), offset: *offset },
        I32Load16S { offset, .. } => I32Load16S { dst: d!(0), addr: s!(0), offset: *offset },
        I64Load8U { offset, .. } => I64Load8U { dst: d!(0), addr: s!(0), offset: *offset },
        I64Load8S { offset, .. } => I64Load8S { dst: d!(0), addr: s!(0), offset: *offset },
        I64Load16U { offset, .. } => I64Load16U { dst: d!(0), addr: s!(0), offset: *offset },
        I64Load16S { offset, .. } => I64Load16S { dst: d!(0), addr: s!(0), offset: *offset },
        I64Load32U { offset, .. } => I64Load32U { dst: d!(0), addr: s!(0), offset: *offset },
        I64Load32S { offset, .. } => I64Load32S { dst: d!(0), addr: s!(0), offset: *offset },

        // Stores: 0 dst, 2 src (addr, src)
        I32Store { offset, .. } => I32Store { addr: s!(0), offset: *offset, src: s!(1) },
        I64Store { offset, .. } => I64Store { addr: s!(0), offset: *offset, src: s!(1) },
        I32Store8 { offset, .. } => I32Store8 { addr: s!(0), offset: *offset, src: s!(1) },
        I32Store16 { offset, .. } => I32Store16 { addr: s!(0), offset: *offset, src: s!(1) },
        I64Store8 { offset, .. } => I64Store8 { addr: s!(0), offset: *offset, src: s!(1) },
        I64Store16 { offset, .. } => I64Store16 { addr: s!(0), offset: *offset, src: s!(1) },
        I64Store32 { offset, .. } => I64Store32 { addr: s!(0), offset: *offset, src: s!(1) },

        // Local/Global
        LocalGet { local, .. } => LocalGet { dst: d!(0), local: *local },
        LocalSet { local, .. } => LocalSet { local: *local, src: s!(0) },
        LocalTee { local, .. } => LocalTee { dst: d!(0), local: *local, src: s!(0) },
        GlobalGet { global, .. } => GlobalGet { dst: d!(0), global: *global },
        GlobalSet { global, .. } => GlobalSet { global: *global, src: s!(0) },

        // Call: N results (dsts), M args (srcs)
        Call { func_idx, .. } => Call {
            func_idx: *func_idx,
            results: new_dsts.to_vec(),
            args: new_srcs.to_vec(),
        },
        CallIndirect { table, type_idx, func_ref, args, results } => {
            if new_srcs.is_empty() {
                CallIndirect { table: *table, type_idx: *type_idx, func_ref: *func_ref,
                    args: args.clone(), results: results.clone() }
            } else {
                CallIndirect { table: *table, type_idx: *type_idx,
                    func_ref: *new_srcs.last().unwrap(),
                    args: new_srcs[..new_srcs.len()-1].to_vec(),
                    results: new_dsts.to_vec() }
            }
        }

        // Select: 1 dst, 3 src (cond, src1, src2)
        Select { .. } => Select { dst: d!(0), cond: s!(0), src1: s!(1), src2: s!(2) },

        // Branches: 0 dst, cond/idx src
        BrIf { label, .. } => BrIf { cond: s!(0), label: *label },
        BrTable { labels, default, .. } => BrTable { idx: s!(0), labels: labels.clone(), default: *default },
        If { label, .. } => If { cond: s!(0), label: *label },

        // Return: 0 dst, N src
        Return { .. } => Return { values: new_srcs.to_vec() },

        // Drop: 0 dst, 1 src
        Drop { .. } => Drop { src: s!(0) },

        // Mov: 1 dst, 1 src
        Mov { .. } => Mov { dst: d!(0), src: s!(0) },

        // Memory ops
        MemorySize { .. } => MemorySize { dst: d!(0) },
        MemoryGrow { .. } => MemoryGrow { dst: d!(0), pages: s!(0) },
        MemoryCopy { .. } => MemoryCopy { dst: s!(0), src: s!(1), len: s!(2) },
        MemoryFill { .. } => MemoryFill { dst: s!(0), val: s!(1), len: s!(2) },

        // No-op control flow (0 dst, 0 src) — pass through unchanged
        Br { label } => Br { label: *label },
        Block { label } => Block { label: *label },
        Loop { label } => Loop { label: *label },
        Else { label } => Else { label: *label },
        End { label } => End { label: *label },
        Unreachable => Unreachable,
        Nop => Nop,
    }
}

/// Lower VRegInst to RegInst with spill/reload inserted
///
/// LEGACY: Produces RegInst with physical registers baked in, but the execute() method
/// that consumes these has known bugs with control flow (branches, loops). The canonical
/// execution path uses execute_vreg with an allocation map. Kept as artifact of gradual
/// complexity build-up.
pub fn lower_to_preg(
    instructions: &[VRegInst],
    alloc: &RegAllocResult,
) -> Vec<RegInst> {
    let mut output = Vec::new();

    // Helper to get PReg or spill slot for a VReg
    let get_preg = |vreg: VReg| -> Option<PReg> {
        alloc.vreg_to_preg.get(&vreg).copied()
    };

    let get_spill = |vreg: VReg| -> Option<SpillSlot> {
        alloc.spill_slots.get(&vreg).copied()
    };

    // We use r0 as a scratch register for spilled values
    let scratch = PReg(0);

    for inst in instructions {
        match inst {
            VRegInst::I32Const { dst, val } => {
                if let Some(preg) = get_preg(*dst) {
                    output.push(RegInst::I32Const { dst: preg, val: *val });
                } else if let Some(slot) = get_spill(*dst) {
                    // Store constant via scratch
                    output.push(RegInst::I32Const { dst: scratch, val: *val });
                    output.push(RegInst::Spill { src: scratch, slot });
                }
            }

            VRegInst::I64Const { dst, val } => {
                if let Some(preg) = get_preg(*dst) {
                    output.push(RegInst::I64Const { dst: preg, val: *val });
                } else if let Some(slot) = get_spill(*dst) {
                    output.push(RegInst::I64Const { dst: scratch, val: *val });
                    output.push(RegInst::Spill { src: scratch, slot });
                }
            }

            VRegInst::I32Add { dst, src1, src2 } => {
                let s1 = reload_if_spilled(&mut output, *src1, alloc, PReg(1));
                let s2 = reload_if_spilled(&mut output, *src2, alloc, PReg(2));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I32Add { dst: d, src1: s1, src2: s2 });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I32Sub { dst, src1, src2 } => {
                let s1 = reload_if_spilled(&mut output, *src1, alloc, PReg(1));
                let s2 = reload_if_spilled(&mut output, *src2, alloc, PReg(2));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I32Sub { dst: d, src1: s1, src2: s2 });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I32Mul { dst, src1, src2 } => {
                let s1 = reload_if_spilled(&mut output, *src1, alloc, PReg(1));
                let s2 = reload_if_spilled(&mut output, *src2, alloc, PReg(2));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I32Mul { dst: d, src1: s1, src2: s2 });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I32DivU { dst, src1, src2 } => {
                let s1 = reload_if_spilled(&mut output, *src1, alloc, PReg(1));
                let s2 = reload_if_spilled(&mut output, *src2, alloc, PReg(2));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I32DivU { dst: d, src1: s1, src2: s2 });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I32And { dst, src1, src2 } => {
                let s1 = reload_if_spilled(&mut output, *src1, alloc, PReg(1));
                let s2 = reload_if_spilled(&mut output, *src2, alloc, PReg(2));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I32And { dst: d, src1: s1, src2: s2 });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I32Or { dst, src1, src2 } => {
                let s1 = reload_if_spilled(&mut output, *src1, alloc, PReg(1));
                let s2 = reload_if_spilled(&mut output, *src2, alloc, PReg(2));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I32Or { dst: d, src1: s1, src2: s2 });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I32Xor { dst, src1, src2 } => {
                let s1 = reload_if_spilled(&mut output, *src1, alloc, PReg(1));
                let s2 = reload_if_spilled(&mut output, *src2, alloc, PReg(2));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I32Xor { dst: d, src1: s1, src2: s2 });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I32Shl { dst, src1, src2 } => {
                let s1 = reload_if_spilled(&mut output, *src1, alloc, PReg(1));
                let s2 = reload_if_spilled(&mut output, *src2, alloc, PReg(2));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I32Shl { dst: d, src1: s1, src2: s2 });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I32ShrU { dst, src1, src2 } => {
                let s1 = reload_if_spilled(&mut output, *src1, alloc, PReg(1));
                let s2 = reload_if_spilled(&mut output, *src2, alloc, PReg(2));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I32ShrU { dst: d, src1: s1, src2: s2 });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I32ShrS { dst, src1, src2 } => {
                let s1 = reload_if_spilled(&mut output, *src1, alloc, PReg(1));
                let s2 = reload_if_spilled(&mut output, *src2, alloc, PReg(2));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I32ShrS { dst: d, src1: s1, src2: s2 });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I32Eq { dst, src1, src2 } => {
                let s1 = reload_if_spilled(&mut output, *src1, alloc, PReg(1));
                let s2 = reload_if_spilled(&mut output, *src2, alloc, PReg(2));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I32Eq { dst: d, src1: s1, src2: s2 });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I32Ne { dst, src1, src2 } => {
                let s1 = reload_if_spilled(&mut output, *src1, alloc, PReg(1));
                let s2 = reload_if_spilled(&mut output, *src2, alloc, PReg(2));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I32Ne { dst: d, src1: s1, src2: s2 });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I32LtU { dst, src1, src2 } => {
                let s1 = reload_if_spilled(&mut output, *src1, alloc, PReg(1));
                let s2 = reload_if_spilled(&mut output, *src2, alloc, PReg(2));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I32LtU { dst: d, src1: s1, src2: s2 });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I32LtS { dst, src1, src2 } => {
                let s1 = reload_if_spilled(&mut output, *src1, alloc, PReg(1));
                let s2 = reload_if_spilled(&mut output, *src2, alloc, PReg(2));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I32LtS { dst: d, src1: s1, src2: s2 });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I32GtU { dst, src1, src2 } => {
                let s1 = reload_if_spilled(&mut output, *src1, alloc, PReg(1));
                let s2 = reload_if_spilled(&mut output, *src2, alloc, PReg(2));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I32GtU { dst: d, src1: s1, src2: s2 });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I32GtS { dst, src1, src2 } => {
                let s1 = reload_if_spilled(&mut output, *src1, alloc, PReg(1));
                let s2 = reload_if_spilled(&mut output, *src2, alloc, PReg(2));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I32GtS { dst: d, src1: s1, src2: s2 });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I32LeU { dst, src1, src2 } => {
                let s1 = reload_if_spilled(&mut output, *src1, alloc, PReg(1));
                let s2 = reload_if_spilled(&mut output, *src2, alloc, PReg(2));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I32LeU { dst: d, src1: s1, src2: s2 });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I32LeS { dst, src1, src2 } => {
                let s1 = reload_if_spilled(&mut output, *src1, alloc, PReg(1));
                let s2 = reload_if_spilled(&mut output, *src2, alloc, PReg(2));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I32LeS { dst: d, src1: s1, src2: s2 });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I32GeU { dst, src1, src2 } => {
                let s1 = reload_if_spilled(&mut output, *src1, alloc, PReg(1));
                let s2 = reload_if_spilled(&mut output, *src2, alloc, PReg(2));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I32GeU { dst: d, src1: s1, src2: s2 });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I32GeS { dst, src1, src2 } => {
                let s1 = reload_if_spilled(&mut output, *src1, alloc, PReg(1));
                let s2 = reload_if_spilled(&mut output, *src2, alloc, PReg(2));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I32GeS { dst: d, src1: s1, src2: s2 });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I32Eqz { dst, src } => {
                let s = reload_if_spilled(&mut output, *src, alloc, PReg(1));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I32Eqz { dst: d, src: s });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I32Load { dst, addr, offset } => {
                let a = reload_if_spilled(&mut output, *addr, alloc, PReg(1));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I32Load { dst: d, addr: a, offset: *offset });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I32Load8U { dst, addr, offset } => {
                let a = reload_if_spilled(&mut output, *addr, alloc, PReg(1));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I32Load8U { dst: d, addr: a, offset: *offset });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I32Load8S { dst, addr, offset } => {
                let a = reload_if_spilled(&mut output, *addr, alloc, PReg(1));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I32Load8S { dst: d, addr: a, offset: *offset });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I32Load16U { dst, addr, offset } => {
                let a = reload_if_spilled(&mut output, *addr, alloc, PReg(1));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I32Load16U { dst: d, addr: a, offset: *offset });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I32Store { addr, offset, src } => {
                let a = reload_if_spilled(&mut output, *addr, alloc, PReg(1));
                let s = reload_if_spilled(&mut output, *src, alloc, PReg(2));
                output.push(RegInst::I32Store { addr: a, offset: *offset, src: s });
            }

            VRegInst::I32Store8 { addr, offset, src } => {
                let a = reload_if_spilled(&mut output, *addr, alloc, PReg(1));
                let s = reload_if_spilled(&mut output, *src, alloc, PReg(2));
                output.push(RegInst::I32Store8 { addr: a, offset: *offset, src: s });
            }

            VRegInst::I32Store16 { addr, offset, src } => {
                let a = reload_if_spilled(&mut output, *addr, alloc, PReg(1));
                let s = reload_if_spilled(&mut output, *src, alloc, PReg(2));
                output.push(RegInst::I32Store16 { addr: a, offset: *offset, src: s });
            }

            VRegInst::I64Load { dst, addr, offset } => {
                let a = reload_if_spilled(&mut output, *addr, alloc, PReg(1));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I64Load { dst: d, addr: a, offset: *offset });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I64Store { addr, offset, src } => {
                let a = reload_if_spilled(&mut output, *addr, alloc, PReg(1));
                let s = reload_if_spilled(&mut output, *src, alloc, PReg(2));
                output.push(RegInst::I64Store { addr: a, offset: *offset, src: s });
            }

            VRegInst::LocalGet { dst, local } => {
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::LocalGet { dst: d, local: *local });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::LocalSet { local, src } => {
                let s = reload_if_spilled(&mut output, *src, alloc, PReg(1));
                output.push(RegInst::LocalSet { local: *local, src: s });
            }

            VRegInst::LocalTee { dst, local, src } => {
                let s = reload_if_spilled(&mut output, *src, alloc, PReg(1));
                output.push(RegInst::LocalSet { local: *local, src: s });
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::Move { dst: d, src: s });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::GlobalGet { dst, global } => {
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::GlobalGet { dst: d, global: *global });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::GlobalSet { global, src } => {
                let s = reload_if_spilled(&mut output, *src, alloc, PReg(1));
                output.push(RegInst::GlobalSet { global: *global, src: s });
            }

            VRegInst::Block { label } => {
                output.push(RegInst::Block { label: *label });
            }

            VRegInst::Loop { label } => {
                output.push(RegInst::Loop { label: *label });
            }

            VRegInst::End { label } => {
                output.push(RegInst::End { label: *label });
            }

            VRegInst::Br { label } => {
                output.push(RegInst::Br { label: *label });
            }

            VRegInst::BrIf { cond, label } => {
                let c = reload_if_spilled(&mut output, *cond, alloc, PReg(1));
                output.push(RegInst::BrIf { cond: c, label: *label });
            }

            VRegInst::BrTable { idx, labels, default } => {
                let i = reload_if_spilled(&mut output, *idx, alloc, PReg(1));
                // BrTable not in RegInst, emit as series of BrIf
                for (n, label) in labels.iter().enumerate() {
                    output.push(RegInst::I32Const { dst: PReg(2), val: n as i32 });
                    output.push(RegInst::I32Eq { dst: PReg(3), src1: i, src2: PReg(2) });
                    output.push(RegInst::BrIf { cond: PReg(3), label: *label });
                }
                output.push(RegInst::Br { label: *default });
            }

            VRegInst::If { cond, label } => {
                let c = reload_if_spilled(&mut output, *cond, alloc, PReg(1));
                output.push(RegInst::If { cond: c, label: *label });
            }

            VRegInst::Else { label } => {
                output.push(RegInst::Else { label: *label });
            }

            VRegInst::Return { values } => {
                if let Some(v) = values.last() {
                    let s = reload_if_spilled(&mut output, *v, alloc, PReg(1));
                    if s != PReg(0) {
                        output.push(RegInst::Move { dst: PReg(0), src: s });
                    }
                }
                output.push(RegInst::Return);
            }

            VRegInst::Call { func_idx, args, results } => {
                // Move args to fixed registers or spill locations
                let mut arg_pregs = Vec::new();
                for (i, arg) in args.iter().enumerate() {
                    let preg = PReg((i + 3) as u8); // Use r3, r4, r5... for args
                    let s = reload_if_spilled(&mut output, *arg, alloc, preg);
                    if s != preg {
                        output.push(RegInst::Move { dst: preg, src: s });
                    }
                    arg_pregs.push(preg);
                }

                let dst = if results.is_empty() { None } else { Some(PReg(0)) };
                output.push(RegInst::Call { func_idx: *func_idx, args: arg_pregs, dst });

                // Move result if needed
                if let Some(r) = results.first() {
                    if let Some(preg) = get_preg(*r) {
                        if preg != PReg(0) {
                            output.push(RegInst::Move { dst: preg, src: PReg(0) });
                        }
                    }
                    spill_if_needed(&mut output, *r, PReg(0), alloc);
                }
            }

            VRegInst::Select { dst, cond, src1, src2 } => {
                let c = reload_if_spilled(&mut output, *cond, alloc, PReg(1));
                let s1 = reload_if_spilled(&mut output, *src1, alloc, PReg(2));
                let s2 = reload_if_spilled(&mut output, *src2, alloc, PReg(3));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::Select { dst: d, cond: c, src1: s1, src2: s2 });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I64Add { dst, src1, src2 } => {
                let s1 = reload_if_spilled(&mut output, *src1, alloc, PReg(1));
                let s2 = reload_if_spilled(&mut output, *src2, alloc, PReg(2));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I64Add { dst: d, src1: s1, src2: s2 });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I64Sub { dst, src1, src2 } => {
                let s1 = reload_if_spilled(&mut output, *src1, alloc, PReg(1));
                let s2 = reload_if_spilled(&mut output, *src2, alloc, PReg(2));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I64Sub { dst: d, src1: s1, src2: s2 });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I64Shl { dst, src1, src2 } => {
                let s1 = reload_if_spilled(&mut output, *src1, alloc, PReg(1));
                let s2 = reload_if_spilled(&mut output, *src2, alloc, PReg(2));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I64Shl { dst: d, src1: s1, src2: s2 });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I64Or { dst, src1, src2 } => {
                let s1 = reload_if_spilled(&mut output, *src1, alloc, PReg(1));
                let s2 = reload_if_spilled(&mut output, *src2, alloc, PReg(2));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I64Or { dst: d, src1: s1, src2: s2 });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I64And { dst, src1, src2 } => {
                let s1 = reload_if_spilled(&mut output, *src1, alloc, PReg(1));
                let s2 = reload_if_spilled(&mut output, *src2, alloc, PReg(2));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I64And { dst: d, src1: s1, src2: s2 });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I64ExtendI32U { dst, src } => {
                let s = reload_if_spilled(&mut output, *src, alloc, PReg(1));
                let d = get_preg(*dst).unwrap_or(scratch);
                // Emit as move for now (the VM will handle the extension)
                output.push(RegInst::Move { dst: d, src: s });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::I32WrapI64 { dst, src } => {
                let s = reload_if_spilled(&mut output, *src, alloc, PReg(1));
                let d = get_preg(*dst).unwrap_or(scratch);
                output.push(RegInst::I32WrapI64 { dst: d, src: s });
                spill_if_needed(&mut output, *dst, d, alloc);
            }

            VRegInst::Unreachable => {
                output.push(RegInst::Unreachable);
            }

            VRegInst::Nop => {
                output.push(RegInst::Nop);
            }

            VRegInst::Drop { src } => {
                let s = reload_if_spilled(&mut output, *src, alloc, PReg(1));
                output.push(RegInst::Drop { src: s });
            }

            // Skip unhandled for now
            _other => {}
        }
    }

    output
}

fn reload_if_spilled(
    output: &mut Vec<RegInst>,
    vreg: VReg,
    alloc: &RegAllocResult,
    scratch: PReg,
) -> PReg {
    if let Some(preg) = alloc.vreg_to_preg.get(&vreg) {
        *preg
    } else if let Some(slot) = alloc.spill_slots.get(&vreg) {
        output.push(RegInst::Reload { dst: scratch, slot: *slot });
        scratch
    } else {
        scratch // Fallback
    }
}

fn spill_if_needed(
    output: &mut Vec<RegInst>,
    vreg: VReg,
    preg: PReg,
    alloc: &RegAllocResult,
) {
    if let Some(slot) = alloc.spill_slots.get(&vreg) {
        output.push(RegInst::Spill { src: preg, slot: *slot });
    }
}

/// Physical register VM with fixed register count
/// Function metadata for calling
pub struct FuncMeta {
    pub instructions: Vec<RegInst>,
    pub num_params: u32,
    pub num_locals: u32,
}

/// VReg function metadata for execute_vreg
pub struct VRegFuncMeta {
    pub instructions: Vec<VRegInst>,
    pub alloc: RegAllocResult,
    pub num_params: u32,
    pub num_locals: u32,
    /// When true, locals/globals are lowered to memory loads/stores.
    /// call_func_vreg writes params into the frame region instead of self.locals.
    pub memory_lowered: bool,
}

pub struct PRegVM {
    /// Fixed number of physical registers
    pub regs: Vec<Value>,
    /// Spill slots (in memory)
    pub spill_slots: Vec<Value>,
    /// Linear memory
    pub memory: Vec<u8>,
    /// Local variables for current function
    pub locals: Vec<Value>,
    /// Global variables
    pub globals: Vec<Value>,
    /// All compiled functions (RegInst)
    pub functions: Vec<FuncMeta>,
    /// All VReg functions for execute_vreg
    pub vreg_functions: Vec<VRegFuncMeta>,
    /// Call stack for recursion (saves locals)
    call_stack: Vec<Vec<Value>>,
    /// Dynamic instruction trace counts (when tracing is enabled)
    pub trace_counts: Option<HashMap<&'static str, u64>>,
    /// Full ordered trace log (when trace_log is enabled)
    pub trace_log: Option<Vec<&'static str>>,
    /// Register-aware trace: (inst_name, dst_pregs, src_pregs) per executed instruction
    pub reg_trace: Option<Vec<(&'static str, Vec<u8>, Vec<u8>)>>,
}

impl PRegVM {
    pub fn new(num_regs: usize, num_spill_slots: usize, memory_pages: usize) -> Self {
        Self {
            regs: vec![Value::I32(0); num_regs],
            spill_slots: vec![Value::I32(0); num_spill_slots],
            memory: vec![0u8; memory_pages * 65536],
            locals: Vec::new(),
            globals: vec![Value::I32(0); 16],
            functions: Vec::new(),
            vreg_functions: Vec::new(),
            call_stack: Vec::new(),
            trace_counts: None,
            trace_log: None,
            reg_trace: None,
        }
    }

    /// Add a RegInst function to the VM
    pub fn add_function(&mut self, instructions: Vec<RegInst>, num_params: u32, num_locals: u32) {
        self.functions.push(FuncMeta { instructions, num_params, num_locals });
    }

    /// Add a VRegInst function with allocation for execute_vreg
    pub fn add_vreg_function(&mut self, instructions: Vec<VRegInst>, alloc: RegAllocResult, num_params: u32, num_locals: u32) {
        self.vreg_functions.push(VRegFuncMeta { instructions, alloc, num_params, num_locals, memory_lowered: false });
    }

    /// Add a memory-lowered VRegInst function (locals/globals → memory loads/stores)
    pub fn add_vreg_function_ml(&mut self, instructions: Vec<VRegInst>, alloc: RegAllocResult, num_params: u32, num_locals: u32) {
        self.vreg_functions.push(VRegFuncMeta { instructions, alloc, num_params, num_locals, memory_lowered: true });
    }

    /// Enable dynamic instruction tracing
    pub fn enable_tracing(&mut self) {
        self.trace_counts = Some(HashMap::new());
    }

    /// Enable full trace log (every instruction in order)
    pub fn enable_trace_log(&mut self) {
        self.trace_log = Some(Vec::new());
    }

    /// Enable register-aware trace (captures physical register assignments per instruction)
    pub fn enable_reg_trace(&mut self) {
        self.reg_trace = Some(Vec::new());
    }

    /// Count an instruction (if tracing enabled)
    fn trace(&mut self, name: &'static str) {
        if let Some(ref mut counts) = self.trace_counts {
            *counts.entry(name).or_insert(0) += 1;
        }
        if let Some(ref mut log) = self.trace_log {
            log.push(name);
        }
    }

    /// Count spill/reload overhead for a VReg access (if tracing enabled)
    fn trace_vreg_access(&mut self, vreg: VReg, is_write: bool, alloc: &RegAllocResult) {
        if self.trace_counts.is_some() || self.trace_log.is_some() {
            if alloc.spilled.contains(&vreg) {
                if is_write {
                    self.trace("spill");
                } else {
                    self.trace("reload");
                }
            }
        }
    }

    pub fn get_reg(&self, preg: PReg) -> Value {
        self.regs.get(preg.0 as usize).copied().unwrap_or(Value::I32(0))
    }

    pub fn set_reg(&mut self, preg: PReg, val: Value) {
        if (preg.0 as usize) < self.regs.len() {
            self.regs[preg.0 as usize] = val;
        }
    }

    pub fn get_spill(&self, slot: SpillSlot) -> Value {
        self.spill_slots.get(slot.0 as usize).copied().unwrap_or(Value::I32(0))
    }

    pub fn set_spill(&mut self, slot: SpillSlot, val: Value) {
        if (slot.0 as usize) < self.spill_slots.len() {
            self.spill_slots[slot.0 as usize] = val;
        }
    }

    pub fn write_memory(&mut self, offset: usize, data: &[u8]) {
        if offset + data.len() <= self.memory.len() {
            self.memory[offset..offset + data.len()].copy_from_slice(data);
        }
    }

    pub fn read_memory(&self, offset: usize, len: usize) -> &[u8] {
        let start = offset.min(self.memory.len());
        let end = offset.saturating_add(len).min(self.memory.len());
        &self.memory[start..end]
    }

    /// Call a function by index with given arguments
    ///
    /// LEGACY: Executes lowered RegInst via execute(). Has known bugs with control flow.
    /// Use call_func_vreg instead. Kept as artifact of gradual complexity build-up.
    pub fn call_func(&mut self, func_idx: u32, args: &[Value]) -> Option<Value> {
        let func = self.functions.get(func_idx as usize)?;
        let instructions = func.instructions.clone();
        let num_params = func.num_params;
        let num_locals = func.num_locals;

        // Save current state
        let saved_regs = self.regs.clone();
        let saved_spills = self.spill_slots.clone();

        // Save current locals
        self.call_stack.push(std::mem::take(&mut self.locals));

        // Set up new locals: params first, then zeros for local vars
        let total_locals = (num_params + num_locals) as usize;
        self.locals = vec![Value::I32(0); total_locals];
        for (i, arg) in args.iter().enumerate() {
            if i < total_locals {
                self.locals[i] = *arg;
            }
        }

        // Execute the function
        let result = self.execute(&instructions, total_locals);
        let ret_val = self.regs[0];

        // Restore old state
        self.locals = self.call_stack.pop().unwrap_or_default();
        self.regs = saved_regs;
        self.spill_slots = saved_spills;

        // Keep return value in r0
        self.regs[0] = ret_val;

        result
    }

    /// Read a little-endian u32 from linear memory
    fn read_mem_u32(&self, addr: usize) -> u32 {
        if addr + 4 <= self.memory.len() {
            u32::from_le_bytes(self.memory[addr..addr + 4].try_into().unwrap())
        } else {
            0
        }
    }

    /// Write a little-endian u32 to linear memory
    fn write_mem_u32(&mut self, addr: usize, val: u32) {
        if addr + 4 <= self.memory.len() {
            self.memory[addr..addr + 4].copy_from_slice(&val.to_le_bytes());
        }
    }

    /// Write a little-endian i64 to linear memory
    fn write_mem_i64(&mut self, addr: usize, val: i64) {
        if addr + 8 <= self.memory.len() {
            self.memory[addr..addr + 8].copy_from_slice(&val.to_le_bytes());
        }
    }

    /// Call a VReg function by index with given arguments
    pub fn call_func_vreg(&mut self, func_idx: u32, args: &[Value]) -> Option<Value> {
        let func = self.vreg_functions.get(func_idx as usize)?;
        let instructions = func.instructions.clone();
        let alloc = RegAllocResult {
            vreg_to_preg: func.alloc.vreg_to_preg.clone(),
            spilled: func.alloc.spilled.clone(),
            spill_slots: func.alloc.spill_slots.clone(),
            num_spill_slots: func.alloc.num_spill_slots,
        };
        let num_params = func.num_params;
        let num_locals = func.num_locals;
        let memory_lowered = func.memory_lowered;

        // Save caller state: locals, registers, and spill slots
        let saved_locals = std::mem::take(&mut self.locals);
        let saved_regs = self.regs.clone();
        let saved_spills = self.spill_slots.clone();

        if memory_lowered {
            // Memory-lowered path: write params into the frame region in linear memory.
            // The callee's prologue will read FRAME_SP_ADDR to find its frame base,
            // then advance FRAME_SP_ADDR past its own frame.
            let frame_sp = self.read_mem_u32(FRAME_SP_ADDR as usize);
            for (i, arg) in args.iter().enumerate() {
                let slot_addr = frame_sp as usize + (i as u32 * SLOT_SIZE) as usize;
                match arg {
                    Value::I32(v) => self.write_mem_u32(slot_addr, *v as u32),
                    Value::I64(v) => self.write_mem_i64(slot_addr, *v),
                }
            }
            // Also save frame_sp so we can restore after the call
            let saved_frame_sp = frame_sp;

            let result = self.execute_vreg(&instructions, &alloc);

            // Restore frame stack pointer (callee's prologue advanced it)
            self.write_mem_u32(FRAME_SP_ADDR as usize, saved_frame_sp);

            // Restore caller state
            self.locals = saved_locals;
            self.regs = saved_regs;
            self.spill_slots = saved_spills;

            result
        } else {
            // Original path: use self.locals
            let total_locals = (num_params + num_locals) as usize;
            self.locals = vec![Value::I32(0); total_locals];
            for (i, arg) in args.iter().enumerate() {
                if i < total_locals {
                    self.locals[i] = *arg;
                }
            }

            let result = self.execute_vreg(&instructions, &alloc);

            // Restore caller state
            self.locals = saved_locals;
            self.regs = saved_regs;
            self.spill_slots = saved_spills;

            result
        }
    }

    /// Execute a sequence of RegInst instructions
    /// Returns the value in r0 (return register) when done
    /// Note: locals should be pre-initialized by the caller (including function parameters)
    ///
    /// LEGACY: Has known bugs with control flow (branches, If/Else, loops produce wrong results).
    /// Use execute_vreg instead. Kept as artifact of gradual complexity build-up.
    pub fn execute(&mut self, instructions: &[RegInst], _func_locals: usize) -> Option<Value> {
        use RegInst::*;

        // Build control flow maps
        let mut block_ends: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        let mut loop_starts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        let mut else_positions: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        let mut stack: Vec<(u32, bool, usize)> = Vec::new();

        for (i, inst) in instructions.iter().enumerate() {
            match inst {
                Block { label } => { stack.push((*label, false, i)); }
                Loop { label } => { loop_starts.insert(*label, i); stack.push((*label, true, i)); }
                If { label, .. } => { stack.push((*label, false, i)); }
                Else { label } => { else_positions.insert(*label, i); }
                End { label: _ } => { if let Some((l, _, _)) = stack.pop() { block_ends.insert(l, i); } }
                _ => {}
            }
        }

        let max_iterations = 50_000_000u64;
        let mut iterations = 0u64;
        let mut pc = 0usize;

        while pc < instructions.len() && iterations < max_iterations {
            iterations += 1;
            let inst = &instructions[pc];

            match inst {
                I32Const { dst, val } => {
                    self.set_reg(*dst, Value::I32(*val));
                    pc += 1;
                }
                I64Const { dst, val } => {
                    self.set_reg(*dst, Value::I64(*val));
                    pc += 1;
                }
                I32Add { dst, src1, src2 } => {
                    let a = self.get_reg(*src1).as_i32();
                    let b = self.get_reg(*src2).as_i32();
                    self.set_reg(*dst, Value::I32(a.wrapping_add(b)));
                    pc += 1;
                }
                I32Sub { dst, src1, src2 } => {
                    let a = self.get_reg(*src1).as_i32();
                    let b = self.get_reg(*src2).as_i32();
                    self.set_reg(*dst, Value::I32(a.wrapping_sub(b)));
                    pc += 1;
                }
                I32Mul { dst, src1, src2 } => {
                    let a = self.get_reg(*src1).as_i32();
                    let b = self.get_reg(*src2).as_i32();
                    self.set_reg(*dst, Value::I32(a.wrapping_mul(b)));
                    pc += 1;
                }
                I32DivU { dst, src1, src2 } => {
                    let a = self.get_reg(*src1).as_u32();
                    let b = self.get_reg(*src2).as_u32();
                    let result = if b == 0 { 0 } else { a / b };
                    self.set_reg(*dst, Value::I32(result as i32));
                    pc += 1;
                }
                I32And { dst, src1, src2 } => {
                    let a = self.get_reg(*src1).as_i32();
                    let b = self.get_reg(*src2).as_i32();
                    self.set_reg(*dst, Value::I32(a & b));
                    pc += 1;
                }
                I32Or { dst, src1, src2 } => {
                    let a = self.get_reg(*src1).as_i32();
                    let b = self.get_reg(*src2).as_i32();
                    self.set_reg(*dst, Value::I32(a | b));
                    pc += 1;
                }
                I32Xor { dst, src1, src2 } => {
                    let a = self.get_reg(*src1).as_i32();
                    let b = self.get_reg(*src2).as_i32();
                    self.set_reg(*dst, Value::I32(a ^ b));
                    pc += 1;
                }
                I32Shl { dst, src1, src2 } => {
                    let a = self.get_reg(*src1).as_i32();
                    let b = self.get_reg(*src2).as_i32();
                    self.set_reg(*dst, Value::I32(a.wrapping_shl(b as u32)));
                    pc += 1;
                }
                I32ShrU { dst, src1, src2 } => {
                    let a = self.get_reg(*src1).as_u32();
                    let b = self.get_reg(*src2).as_i32();
                    self.set_reg(*dst, Value::I32(a.wrapping_shr(b as u32) as i32));
                    pc += 1;
                }
                I32ShrS { dst, src1, src2 } => {
                    let a = self.get_reg(*src1).as_i32();
                    let b = self.get_reg(*src2).as_i32();
                    self.set_reg(*dst, Value::I32(a.wrapping_shr(b as u32)));
                    pc += 1;
                }
                I32Eqz { dst, src } => {
                    let a = self.get_reg(*src).as_i32();
                    self.set_reg(*dst, Value::I32(if a == 0 { 1 } else { 0 }));
                    pc += 1;
                }
                I32WrapI64 { dst, src } => {
                    let a = self.get_reg(*src).as_i64();
                    self.set_reg(*dst, Value::I32(a as i32));
                    pc += 1;
                }
                I32Eq { dst, src1, src2 } => {
                    let a = self.get_reg(*src1).as_i32();
                    let b = self.get_reg(*src2).as_i32();
                    self.set_reg(*dst, Value::I32(if a == b { 1 } else { 0 }));
                    pc += 1;
                }
                I32Ne { dst, src1, src2 } => {
                    let a = self.get_reg(*src1).as_i32();
                    let b = self.get_reg(*src2).as_i32();
                    self.set_reg(*dst, Value::I32(if a != b { 1 } else { 0 }));
                    pc += 1;
                }
                I32LtU { dst, src1, src2 } => {
                    let a = self.get_reg(*src1).as_u32();
                    let b = self.get_reg(*src2).as_u32();
                    self.set_reg(*dst, Value::I32(if a < b { 1 } else { 0 }));
                    pc += 1;
                }
                I32LtS { dst, src1, src2 } => {
                    let a = self.get_reg(*src1).as_i32();
                    let b = self.get_reg(*src2).as_i32();
                    self.set_reg(*dst, Value::I32(if a < b { 1 } else { 0 }));
                    pc += 1;
                }
                I32GtU { dst, src1, src2 } => {
                    let a = self.get_reg(*src1).as_u32();
                    let b = self.get_reg(*src2).as_u32();
                    self.set_reg(*dst, Value::I32(if a > b { 1 } else { 0 }));
                    pc += 1;
                }
                I32GtS { dst, src1, src2 } => {
                    let a = self.get_reg(*src1).as_i32();
                    let b = self.get_reg(*src2).as_i32();
                    self.set_reg(*dst, Value::I32(if a > b { 1 } else { 0 }));
                    pc += 1;
                }
                I32LeU { dst, src1, src2 } => {
                    let a = self.get_reg(*src1).as_u32();
                    let b = self.get_reg(*src2).as_u32();
                    self.set_reg(*dst, Value::I32(if a <= b { 1 } else { 0 }));
                    pc += 1;
                }
                I32LeS { dst, src1, src2 } => {
                    let a = self.get_reg(*src1).as_i32();
                    let b = self.get_reg(*src2).as_i32();
                    self.set_reg(*dst, Value::I32(if a <= b { 1 } else { 0 }));
                    pc += 1;
                }
                I32GeU { dst, src1, src2 } => {
                    let a = self.get_reg(*src1).as_u32();
                    let b = self.get_reg(*src2).as_u32();
                    self.set_reg(*dst, Value::I32(if a >= b { 1 } else { 0 }));
                    pc += 1;
                }
                I32GeS { dst, src1, src2 } => {
                    let a = self.get_reg(*src1).as_i32();
                    let b = self.get_reg(*src2).as_i32();
                    self.set_reg(*dst, Value::I32(if a >= b { 1 } else { 0 }));
                    pc += 1;
                }
                I32Load { dst, addr, offset } => {
                    let a = self.get_reg(*addr).as_u32();
                    let idx = (a + offset) as usize;
                    let val = if idx + 4 <= self.memory.len() {
                        i32::from_le_bytes(self.memory[idx..idx + 4].try_into().unwrap())
                    } else { 0 };
                    self.set_reg(*dst, Value::I32(val));
                    pc += 1;
                }
                I32Load8U { dst, addr, offset } => {
                    let a = self.get_reg(*addr).as_u32();
                    let idx = (a + offset) as usize;
                    let val = if idx < self.memory.len() { self.memory[idx] as i32 } else { 0 };
                    self.set_reg(*dst, Value::I32(val));
                    pc += 1;
                }
                I32Load8S { dst, addr, offset } => {
                    let a = self.get_reg(*addr).as_u32();
                    let idx = (a + offset) as usize;
                    let val = if idx < self.memory.len() { self.memory[idx] as i8 as i32 } else { 0 };
                    self.set_reg(*dst, Value::I32(val));
                    pc += 1;
                }
                I32Load16U { dst, addr, offset } => {
                    let a = self.get_reg(*addr).as_u32();
                    let idx = (a + offset) as usize;
                    let val = if idx + 2 <= self.memory.len() {
                        u16::from_le_bytes(self.memory[idx..idx + 2].try_into().unwrap()) as i32
                    } else { 0 };
                    self.set_reg(*dst, Value::I32(val));
                    pc += 1;
                }
                I64Load { dst, addr, offset } => {
                    let a = self.get_reg(*addr).as_u32();
                    let idx = (a + offset) as usize;
                    let val = if idx + 8 <= self.memory.len() {
                        i64::from_le_bytes(self.memory[idx..idx + 8].try_into().unwrap())
                    } else { 0 };
                    self.set_reg(*dst, Value::I64(val));
                    pc += 1;
                }
                I32Store { addr, offset, src } => {
                    let a = self.get_reg(*addr).as_u32();
                    let val = self.get_reg(*src).as_i32();
                    let idx = (a + offset) as usize;
                    if idx + 4 <= self.memory.len() {
                        self.memory[idx..idx + 4].copy_from_slice(&val.to_le_bytes());
                    }
                    pc += 1;
                }
                I32Store8 { addr, offset, src } => {
                    let a = self.get_reg(*addr).as_u32();
                    let val = self.get_reg(*src).as_i32() as u8;
                    let idx = (a + offset) as usize;
                    if idx < self.memory.len() {
                        self.memory[idx] = val;
                    }
                    pc += 1;
                }
                I32Store16 { addr, offset, src } => {
                    let a = self.get_reg(*addr).as_u32();
                    let val = self.get_reg(*src).as_i32() as u16;
                    let idx = (a + offset) as usize;
                    if idx + 2 <= self.memory.len() {
                        self.memory[idx..idx + 2].copy_from_slice(&val.to_le_bytes());
                    }
                    pc += 1;
                }
                I64Store { addr, offset, src } => {
                    let a = self.get_reg(*addr).as_u32();
                    let val = self.get_reg(*src).as_i64();
                    let idx = (a + offset) as usize;
                    if idx + 8 <= self.memory.len() {
                        self.memory[idx..idx + 8].copy_from_slice(&val.to_le_bytes());
                    }
                    pc += 1;
                }
                I64Add { dst, src1, src2 } => {
                    let a = self.get_reg(*src1).as_i64();
                    let b = self.get_reg(*src2).as_i64();
                    self.set_reg(*dst, Value::I64(a.wrapping_add(b)));
                    pc += 1;
                }
                I64Sub { dst, src1, src2 } => {
                    let a = self.get_reg(*src1).as_i64();
                    let b = self.get_reg(*src2).as_i64();
                    self.set_reg(*dst, Value::I64(a.wrapping_sub(b)));
                    pc += 1;
                }
                I64Shl { dst, src1, src2 } => {
                    let a = self.get_reg(*src1).as_i64();
                    let b = self.get_reg(*src2).as_i64();
                    self.set_reg(*dst, Value::I64(a.wrapping_shl(b as u32)));
                    pc += 1;
                }
                I64Or { dst, src1, src2 } => {
                    let a = self.get_reg(*src1).as_i64();
                    let b = self.get_reg(*src2).as_i64();
                    self.set_reg(*dst, Value::I64(a | b));
                    pc += 1;
                }
                I64And { dst, src1, src2 } => {
                    let a = self.get_reg(*src1).as_i64();
                    let b = self.get_reg(*src2).as_i64();
                    self.set_reg(*dst, Value::I64(a & b));
                    pc += 1;
                }
                Move { dst, src } => {
                    let val = self.get_reg(*src);
                    self.set_reg(*dst, val);
                    pc += 1;
                }
                Spill { src, slot } => {
                    let val = self.get_reg(*src);
                    self.set_spill(*slot, val);
                    pc += 1;
                }
                Reload { dst, slot } => {
                    let val = self.get_spill(*slot);
                    self.set_reg(*dst, val);
                    pc += 1;
                }
                LocalGet { dst, local } => {
                    let val = self.locals.get(*local as usize).copied().unwrap_or(Value::I32(0));
                    self.set_reg(*dst, val);
                    pc += 1;
                }
                LocalSet { local, src } => {
                    let val = self.get_reg(*src);
                    if (*local as usize) < self.locals.len() {
                        self.locals[*local as usize] = val;
                    }
                    pc += 1;
                }
                Select { dst, cond, src1, src2 } => {
                    let c = self.get_reg(*cond).as_i32();
                    let val = if c != 0 { self.get_reg(*src1) } else { self.get_reg(*src2) };
                    self.set_reg(*dst, val);
                    pc += 1;
                }
                Block { .. } => { pc += 1; }
                Loop { .. } => { pc += 1; }
                If { cond, label } => {
                    let c = self.get_reg(*cond).as_i32();
                    if c == 0 {
                        // Condition false: jump to else or end
                        if let Some(&else_pc) = else_positions.get(label) {
                            pc = else_pc + 1;
                        } else if let Some(&end_pc) = block_ends.get(label) {
                            pc = end_pc + 1;
                        } else {
                            pc += 1;
                        }
                    } else {
                        pc += 1;
                    }
                }
                Else { label } => {
                    // Coming from then branch: skip to end
                    if let Some(&end_pc) = block_ends.get(label) {
                        pc = end_pc + 1;
                    } else {
                        pc += 1;
                    }
                }
                End { .. } => { pc += 1; }
                Return => {
                    return Some(self.get_reg(PReg(0)));
                }
                Br { label } => {
                    if let Some(&start) = loop_starts.get(label) { pc = start + 1; }
                    else if let Some(&end) = block_ends.get(label) { pc = end + 1; }
                    else { pc += 1; }
                }
                BrIf { cond, label } => {
                    let c = self.get_reg(*cond).as_i32();
                    if c != 0 {
                        if let Some(&start) = loop_starts.get(label) { pc = start + 1; }
                        else if let Some(&end) = block_ends.get(label) { pc = end + 1; }
                        else { pc += 1; }
                    } else { pc += 1; }
                }
                Nop => { pc += 1; }
                Unreachable => { return None; }
                Drop { .. } => { pc += 1; }
                Call { func_idx, args, dst } => {
                    // Get argument values from their source registers
                    let arg_vals: Vec<Value> = args.iter()
                        .map(|r| self.get_reg(*r))
                        .collect();

                    // Call the function
                    if let Some(result) = self.call_func(*func_idx, &arg_vals) {
                        // Store result in destination register
                        if let Some(d) = dst {
                            self.set_reg(*d, result);
                        }
                    }
                    pc += 1;
                }
                GlobalGet { dst, global } => {
                    let val = self.globals.get(*global as usize).copied().unwrap_or(Value::I32(0));
                    self.set_reg(*dst, val);
                    pc += 1;
                }
                GlobalSet { global, src } => {
                    let val = self.get_reg(*src);
                    if (*global as usize) < self.globals.len() {
                        self.globals[*global as usize] = val;
                    }
                    pc += 1;
                }
                _ => { pc += 1; }
            }
        }

        // Return value is in r0
        Some(self.get_reg(PReg(0)))
    }

    /// Execute VRegInst directly with proper control flow, using register allocation.
    /// Resolves VRegs through the allocation map (vreg_to_preg / spill_slots) at runtime.
    ///
    /// NOTE: This is not a true "transformed program" execution — the original VReg instructions
    /// are interpreted with a side-table lookup. The canonical transform path uses Ra2FuncMulti
    /// in isa_regalloc.rs to rewrite instructions with concrete register assignments, then
    /// executes the rewritten VRegInst stream through this same method.
    pub fn execute_vreg(
        &mut self,
        instructions: &[VRegInst],
        alloc: &RegAllocResult,
    ) -> Option<Value> {
        // Helper to get/set vreg values using allocation
        let get_vreg = |vm: &Self, vreg: VReg| -> Value {
            if let Some(&preg) = alloc.vreg_to_preg.get(&vreg) {
                vm.get_reg(preg)
            } else if let Some(&slot) = alloc.spill_slots.get(&vreg) {
                vm.get_spill(slot)
            } else {
                Value::I32(0)
            }
        };

        let set_vreg = |vm: &mut Self, vreg: VReg, val: Value| {
            if let Some(&preg) = alloc.vreg_to_preg.get(&vreg) {
                vm.set_reg(preg, val);
            } else if let Some(&slot) = alloc.spill_slots.get(&vreg) {
                vm.set_spill(slot, val);
            }
        };

        // Build control flow maps
        let mut block_ends: HashMap<u32, usize> = HashMap::new();
        let mut loop_starts: HashMap<u32, usize> = HashMap::new();
        let mut else_positions: HashMap<u32, usize> = HashMap::new();
        let mut stack: Vec<(u32, bool, usize)> = Vec::new();

        for (i, inst) in instructions.iter().enumerate() {
            match inst {
                VRegInst::Block { label } => { stack.push((*label, false, i)); }
                VRegInst::Loop { label } => { loop_starts.insert(*label, i); stack.push((*label, true, i)); }
                VRegInst::If { label, .. } => { stack.push((*label, false, i)); }
                VRegInst::Else { label } => { else_positions.insert(*label, i); }
                VRegInst::End { label: _ } => { if let Some((l, _, _)) = stack.pop() { block_ends.insert(l, i); } }
                _ => {}
            }
        }

        let max_iterations = 50_000_000u64;
        let mut iterations = 0u64;
        let mut pc = 0usize;
        let mut last_result: Option<VReg> = None;

        while pc < instructions.len() && iterations < max_iterations {
            iterations += 1;
            let inst = &instructions[pc];

            // Dynamic tracing: count instruction + spill/reload overhead
            if self.trace_counts.is_some() || self.trace_log.is_some() {
                let name = vreg_inst_name(inst);
                self.trace(name);
                // Count spill/reload for each vreg accessed
                for vreg in vreg_src_regs(inst) {
                    self.trace_vreg_access(vreg, false, alloc);
                }
                for vreg in vreg_dst_regs(inst) {
                    self.trace_vreg_access(vreg, true, alloc);
                }
            }

            // Register-aware tracing
            if let Some(ref mut reg_trace) = self.reg_trace {
                let name = vreg_inst_name(inst);
                let dst_pregs: Vec<u8> = vreg_dst_regs(inst).iter()
                    .filter_map(|v| alloc.vreg_to_preg.get(v).map(|p| p.0))
                    .collect();
                let src_pregs: Vec<u8> = vreg_src_regs(inst).iter()
                    .filter_map(|v| alloc.vreg_to_preg.get(v).map(|p| p.0))
                    .collect();
                reg_trace.push((name, dst_pregs, src_pregs));
            }

            match inst {
                VRegInst::I32Const { dst, val } => {
                    set_vreg(self, *dst, Value::I32(*val));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64Const { dst, val } => {
                    set_vreg(self, *dst, Value::I64(*val));
                    last_result = Some(*dst);
                    pc += 1;
                }

                // Binary i32 ops
                VRegInst::I32Add { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i32();
                    let b = get_vreg(self, *src2).as_i32();
                    set_vreg(self, *dst, Value::I32(a.wrapping_add(b)));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Sub { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i32();
                    let b = get_vreg(self, *src2).as_i32();
                    set_vreg(self, *dst, Value::I32(a.wrapping_sub(b)));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Mul { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i32();
                    let b = get_vreg(self, *src2).as_i32();
                    set_vreg(self, *dst, Value::I32(a.wrapping_mul(b)));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32DivS { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i32();
                    let b = get_vreg(self, *src2).as_i32();
                    let result = if b == 0 { 0 } else { a.wrapping_div(b) };
                    set_vreg(self, *dst, Value::I32(result));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32DivU { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_u32();
                    let b = get_vreg(self, *src2).as_u32();
                    let result = if b == 0 { 0 } else { a / b };
                    set_vreg(self, *dst, Value::I32(result as i32));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32RemS { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i32();
                    let b = get_vreg(self, *src2).as_i32();
                    let result = if b == 0 { 0 } else { a.wrapping_rem(b) };
                    set_vreg(self, *dst, Value::I32(result));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32RemU { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_u32();
                    let b = get_vreg(self, *src2).as_u32();
                    let result = if b == 0 { 0 } else { a % b };
                    set_vreg(self, *dst, Value::I32(result as i32));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32And { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i32();
                    let b = get_vreg(self, *src2).as_i32();
                    set_vreg(self, *dst, Value::I32(a & b));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Or { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i32();
                    let b = get_vreg(self, *src2).as_i32();
                    set_vreg(self, *dst, Value::I32(a | b));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Xor { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i32();
                    let b = get_vreg(self, *src2).as_i32();
                    set_vreg(self, *dst, Value::I32(a ^ b));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Shl { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i32();
                    let b = get_vreg(self, *src2).as_i32();
                    set_vreg(self, *dst, Value::I32(a.wrapping_shl(b as u32)));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32ShrU { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_u32();
                    let b = get_vreg(self, *src2).as_i32();
                    set_vreg(self, *dst, Value::I32(a.wrapping_shr(b as u32) as i32));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32ShrS { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i32();
                    let b = get_vreg(self, *src2).as_i32();
                    set_vreg(self, *dst, Value::I32(a.wrapping_shr(b as u32)));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Rotl { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_u32();
                    let b = get_vreg(self, *src2).as_u32();
                    set_vreg(self, *dst, Value::I32(a.rotate_left(b) as i32));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Rotr { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_u32();
                    let b = get_vreg(self, *src2).as_u32();
                    set_vreg(self, *dst, Value::I32(a.rotate_right(b) as i32));
                    last_result = Some(*dst);
                    pc += 1;
                }

                // Binary i64 ops
                VRegInst::I64Add { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i64();
                    let b = get_vreg(self, *src2).as_i64();
                    set_vreg(self, *dst, Value::I64(a.wrapping_add(b)));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64Sub { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i64();
                    let b = get_vreg(self, *src2).as_i64();
                    set_vreg(self, *dst, Value::I64(a.wrapping_sub(b)));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64Mul { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i64();
                    let b = get_vreg(self, *src2).as_i64();
                    set_vreg(self, *dst, Value::I64(a.wrapping_mul(b)));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64DivS { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i64();
                    let b = get_vreg(self, *src2).as_i64();
                    let result = if b == 0 { 0 } else { a.wrapping_div(b) };
                    set_vreg(self, *dst, Value::I64(result));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64DivU { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_u64();
                    let b = get_vreg(self, *src2).as_u64();
                    let result = if b == 0 { 0 } else { a / b };
                    set_vreg(self, *dst, Value::I64(result as i64));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64RemS { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i64();
                    let b = get_vreg(self, *src2).as_i64();
                    let result = if b == 0 { 0 } else { a.wrapping_rem(b) };
                    set_vreg(self, *dst, Value::I64(result));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64RemU { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_u64();
                    let b = get_vreg(self, *src2).as_u64();
                    let result = if b == 0 { 0 } else { a % b };
                    set_vreg(self, *dst, Value::I64(result as i64));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64And { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i64();
                    let b = get_vreg(self, *src2).as_i64();
                    set_vreg(self, *dst, Value::I64(a & b));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64Or { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i64();
                    let b = get_vreg(self, *src2).as_i64();
                    set_vreg(self, *dst, Value::I64(a | b));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64Xor { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i64();
                    let b = get_vreg(self, *src2).as_i64();
                    set_vreg(self, *dst, Value::I64(a ^ b));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64Shl { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i64();
                    let b = get_vreg(self, *src2).as_i64();
                    set_vreg(self, *dst, Value::I64(a.wrapping_shl(b as u32)));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64ShrU { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_u64();
                    let b = get_vreg(self, *src2).as_i64();
                    set_vreg(self, *dst, Value::I64(a.wrapping_shr(b as u32) as i64));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64ShrS { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i64();
                    let b = get_vreg(self, *src2).as_i64();
                    set_vreg(self, *dst, Value::I64(a.wrapping_shr(b as u32)));
                    last_result = Some(*dst);
                    pc += 1;
                }

                // Unary ops
                VRegInst::I32Eqz { dst, src } => {
                    let a = get_vreg(self, *src).as_i32();
                    set_vreg(self, *dst, Value::I32(if a == 0 { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Clz { dst, src } => {
                    let a = get_vreg(self, *src).as_u32();
                    set_vreg(self, *dst, Value::I32(a.leading_zeros() as i32));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Ctz { dst, src } => {
                    let a = get_vreg(self, *src).as_u32();
                    set_vreg(self, *dst, Value::I32(a.trailing_zeros() as i32));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Popcnt { dst, src } => {
                    let a = get_vreg(self, *src).as_u32();
                    set_vreg(self, *dst, Value::I32(a.count_ones() as i32));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64Eqz { dst, src } => {
                    let a = get_vreg(self, *src).as_i64();
                    set_vreg(self, *dst, Value::I32(if a == 0 { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64Clz { dst, src } => {
                    let a = get_vreg(self, *src).as_u64();
                    set_vreg(self, *dst, Value::I64(a.leading_zeros() as i64));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64Ctz { dst, src } => {
                    let a = get_vreg(self, *src).as_u64();
                    set_vreg(self, *dst, Value::I64(a.trailing_zeros() as i64));
                    last_result = Some(*dst);
                    pc += 1;
                }

                // Comparisons i32
                VRegInst::I32Eq { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i32();
                    let b = get_vreg(self, *src2).as_i32();
                    set_vreg(self, *dst, Value::I32(if a == b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Ne { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i32();
                    let b = get_vreg(self, *src2).as_i32();
                    set_vreg(self, *dst, Value::I32(if a != b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32LtS { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i32();
                    let b = get_vreg(self, *src2).as_i32();
                    set_vreg(self, *dst, Value::I32(if a < b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32LtU { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_u32();
                    let b = get_vreg(self, *src2).as_u32();
                    set_vreg(self, *dst, Value::I32(if a < b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32GtS { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i32();
                    let b = get_vreg(self, *src2).as_i32();
                    set_vreg(self, *dst, Value::I32(if a > b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32GtU { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_u32();
                    let b = get_vreg(self, *src2).as_u32();
                    set_vreg(self, *dst, Value::I32(if a > b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32LeS { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i32();
                    let b = get_vreg(self, *src2).as_i32();
                    set_vreg(self, *dst, Value::I32(if a <= b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32LeU { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_u32();
                    let b = get_vreg(self, *src2).as_u32();
                    set_vreg(self, *dst, Value::I32(if a <= b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32GeS { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i32();
                    let b = get_vreg(self, *src2).as_i32();
                    set_vreg(self, *dst, Value::I32(if a >= b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32GeU { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_u32();
                    let b = get_vreg(self, *src2).as_u32();
                    set_vreg(self, *dst, Value::I32(if a >= b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }

                // Comparisons i64
                VRegInst::I64Eq { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i64();
                    let b = get_vreg(self, *src2).as_i64();
                    set_vreg(self, *dst, Value::I32(if a == b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64Ne { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i64();
                    let b = get_vreg(self, *src2).as_i64();
                    set_vreg(self, *dst, Value::I32(if a != b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64LtS { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i64();
                    let b = get_vreg(self, *src2).as_i64();
                    set_vreg(self, *dst, Value::I32(if a < b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64LtU { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_u64();
                    let b = get_vreg(self, *src2).as_u64();
                    set_vreg(self, *dst, Value::I32(if a < b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64GtS { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i64();
                    let b = get_vreg(self, *src2).as_i64();
                    set_vreg(self, *dst, Value::I32(if a > b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64GtU { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_u64();
                    let b = get_vreg(self, *src2).as_u64();
                    set_vreg(self, *dst, Value::I32(if a > b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64LeS { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i64();
                    let b = get_vreg(self, *src2).as_i64();
                    set_vreg(self, *dst, Value::I32(if a <= b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64LeU { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_u64();
                    let b = get_vreg(self, *src2).as_u64();
                    set_vreg(self, *dst, Value::I32(if a <= b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64GeS { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_i64();
                    let b = get_vreg(self, *src2).as_i64();
                    set_vreg(self, *dst, Value::I32(if a >= b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64GeU { dst, src1, src2 } => {
                    let a = get_vreg(self, *src1).as_u64();
                    let b = get_vreg(self, *src2).as_u64();
                    set_vreg(self, *dst, Value::I32(if a >= b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }

                // Type conversions
                VRegInst::I32WrapI64 { dst, src } => {
                    let a = get_vreg(self, *src).as_i64();
                    set_vreg(self, *dst, Value::I32(a as i32));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64ExtendI32S { dst, src } => {
                    let a = get_vreg(self, *src).as_i32();
                    set_vreg(self, *dst, Value::I64(a as i64));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64ExtendI32U { dst, src } => {
                    let a = get_vreg(self, *src).as_u32();
                    set_vreg(self, *dst, Value::I64(a as i64));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Extend8S { dst, src } => {
                    let a = get_vreg(self, *src).as_i32() as i8;
                    set_vreg(self, *dst, Value::I32(a as i32));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Extend16S { dst, src } => {
                    let a = get_vreg(self, *src).as_i32() as i16;
                    set_vreg(self, *dst, Value::I32(a as i32));
                    last_result = Some(*dst);
                    pc += 1;
                }

                // Memory loads
                VRegInst::I32Load { dst, addr, offset } => {
                    let a = get_vreg(self, *addr).as_u32();
                    let idx = (a + offset) as usize;
                    let val = if idx + 4 <= self.memory.len() {
                        i32::from_le_bytes(self.memory[idx..idx + 4].try_into().unwrap())
                    } else { 0 };
                    set_vreg(self, *dst, Value::I32(val));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64Load { dst, addr, offset } => {
                    let a = get_vreg(self, *addr).as_u32();
                    let idx = (a + offset) as usize;
                    let val = if idx + 8 <= self.memory.len() {
                        i64::from_le_bytes(self.memory[idx..idx + 8].try_into().unwrap())
                    } else { 0 };
                    set_vreg(self, *dst, Value::I64(val));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Load8U { dst, addr, offset } => {
                    let a = get_vreg(self, *addr).as_u32();
                    let idx = (a + offset) as usize;
                    let val = if idx < self.memory.len() { self.memory[idx] as i32 } else { 0 };
                    set_vreg(self, *dst, Value::I32(val));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Load8S { dst, addr, offset } => {
                    let a = get_vreg(self, *addr).as_u32();
                    let idx = (a + offset) as usize;
                    let val = if idx < self.memory.len() { self.memory[idx] as i8 as i32 } else { 0 };
                    set_vreg(self, *dst, Value::I32(val));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Load16U { dst, addr, offset } => {
                    let a = get_vreg(self, *addr).as_u32();
                    let idx = (a + offset) as usize;
                    let val = if idx + 2 <= self.memory.len() {
                        u16::from_le_bytes(self.memory[idx..idx + 2].try_into().unwrap()) as i32
                    } else { 0 };
                    set_vreg(self, *dst, Value::I32(val));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Load16S { dst, addr, offset } => {
                    let a = get_vreg(self, *addr).as_u32();
                    let idx = (a + offset) as usize;
                    let val = if idx + 2 <= self.memory.len() {
                        i16::from_le_bytes(self.memory[idx..idx + 2].try_into().unwrap()) as i32
                    } else { 0 };
                    set_vreg(self, *dst, Value::I32(val));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64Load8U { dst, addr, offset } => {
                    let a = get_vreg(self, *addr).as_u32();
                    let idx = (a + offset) as usize;
                    let val = if idx < self.memory.len() { self.memory[idx] as i64 } else { 0 };
                    set_vreg(self, *dst, Value::I64(val));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64Load8S { dst, addr, offset } => {
                    let a = get_vreg(self, *addr).as_u32();
                    let idx = (a + offset) as usize;
                    let val = if idx < self.memory.len() { self.memory[idx] as i8 as i64 } else { 0 };
                    set_vreg(self, *dst, Value::I64(val));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64Load16U { dst, addr, offset } => {
                    let a = get_vreg(self, *addr).as_u32();
                    let idx = (a + offset) as usize;
                    let val = if idx + 2 <= self.memory.len() {
                        u16::from_le_bytes(self.memory[idx..idx + 2].try_into().unwrap()) as i64
                    } else { 0 };
                    set_vreg(self, *dst, Value::I64(val));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64Load16S { dst, addr, offset } => {
                    let a = get_vreg(self, *addr).as_u32();
                    let idx = (a + offset) as usize;
                    let val = if idx + 2 <= self.memory.len() {
                        i16::from_le_bytes(self.memory[idx..idx + 2].try_into().unwrap()) as i64
                    } else { 0 };
                    set_vreg(self, *dst, Value::I64(val));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64Load32U { dst, addr, offset } => {
                    let a = get_vreg(self, *addr).as_u32();
                    let idx = (a + offset) as usize;
                    let val = if idx + 4 <= self.memory.len() {
                        u32::from_le_bytes(self.memory[idx..idx + 4].try_into().unwrap()) as i64
                    } else { 0 };
                    set_vreg(self, *dst, Value::I64(val));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64Load32S { dst, addr, offset } => {
                    let a = get_vreg(self, *addr).as_u32();
                    let idx = (a + offset) as usize;
                    let val = if idx + 4 <= self.memory.len() {
                        i32::from_le_bytes(self.memory[idx..idx + 4].try_into().unwrap()) as i64
                    } else { 0 };
                    set_vreg(self, *dst, Value::I64(val));
                    last_result = Some(*dst);
                    pc += 1;
                }

                // Memory stores
                VRegInst::I32Store { addr, offset, src } => {
                    let a = get_vreg(self, *addr).as_u32();
                    let val = get_vreg(self, *src).as_i32();
                    let idx = (a + offset) as usize;
                    if idx + 4 <= self.memory.len() {
                        self.memory[idx..idx + 4].copy_from_slice(&val.to_le_bytes());
                    }
                    pc += 1;
                }
                VRegInst::I64Store { addr, offset, src } => {
                    let a = get_vreg(self, *addr).as_u32();
                    let val = get_vreg(self, *src).as_i64();
                    let idx = (a + offset) as usize;
                    if idx + 8 <= self.memory.len() {
                        self.memory[idx..idx + 8].copy_from_slice(&val.to_le_bytes());
                    }
                    pc += 1;
                }
                VRegInst::I32Store8 { addr, offset, src } => {
                    let a = get_vreg(self, *addr).as_u32();
                    let val = get_vreg(self, *src).as_i32() as u8;
                    let idx = (a + offset) as usize;
                    if idx < self.memory.len() {
                        self.memory[idx] = val;
                    }
                    pc += 1;
                }
                VRegInst::I32Store16 { addr, offset, src } => {
                    let a = get_vreg(self, *addr).as_u32();
                    let val = get_vreg(self, *src).as_i32() as u16;
                    let idx = (a + offset) as usize;
                    if idx + 2 <= self.memory.len() {
                        self.memory[idx..idx + 2].copy_from_slice(&val.to_le_bytes());
                    }
                    pc += 1;
                }
                VRegInst::I64Store8 { addr, offset, src } => {
                    let a = get_vreg(self, *addr).as_u32();
                    let val = get_vreg(self, *src).as_i64() as u8;
                    let idx = (a + offset) as usize;
                    if idx < self.memory.len() {
                        self.memory[idx] = val;
                    }
                    pc += 1;
                }
                VRegInst::I64Store16 { addr, offset, src } => {
                    let a = get_vreg(self, *addr).as_u32();
                    let val = get_vreg(self, *src).as_i64() as u16;
                    let idx = (a + offset) as usize;
                    if idx + 2 <= self.memory.len() {
                        self.memory[idx..idx + 2].copy_from_slice(&val.to_le_bytes());
                    }
                    pc += 1;
                }
                VRegInst::I64Store32 { addr, offset, src } => {
                    let a = get_vreg(self, *addr).as_u32();
                    let val = get_vreg(self, *src).as_i64() as u32;
                    let idx = (a + offset) as usize;
                    if idx + 4 <= self.memory.len() {
                        self.memory[idx..idx + 4].copy_from_slice(&val.to_le_bytes());
                    }
                    pc += 1;
                }

                // Local variables
                VRegInst::LocalGet { dst, local } => {
                    let val = self.locals.get(*local as usize).copied().unwrap_or(Value::I32(0));
                    set_vreg(self, *dst, val);
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::LocalSet { local, src } => {
                    let val = get_vreg(self, *src);
                    if (*local as usize) < self.locals.len() {
                        self.locals[*local as usize] = val;
                    }
                    pc += 1;
                }
                VRegInst::LocalTee { dst, local, src } => {
                    let val = get_vreg(self, *src);
                    if (*local as usize) < self.locals.len() {
                        self.locals[*local as usize] = val;
                    }
                    set_vreg(self, *dst, val);
                    last_result = Some(*dst);
                    pc += 1;
                }

                // Globals
                VRegInst::GlobalGet { dst, global } => {
                    let val = self.globals.get(*global as usize).copied().unwrap_or(Value::I32(0));
                    set_vreg(self, *dst, val);
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::GlobalSet { global, src } => {
                    let val = get_vreg(self, *src);
                    if (*global as usize) < self.globals.len() {
                        self.globals[*global as usize] = val;
                    }
                    pc += 1;
                }

                // Control flow with proper If/Else handling
                VRegInst::Block { .. } => { pc += 1; }
                VRegInst::Loop { .. } => { pc += 1; }
                VRegInst::If { cond, label } => {
                    let c = get_vreg(self, *cond).as_i32();
                    if c == 0 {
                        // Condition false: jump to else or end
                        if let Some(&else_pc) = else_positions.get(label) {
                            pc = else_pc + 1;
                        } else if let Some(&end_pc) = block_ends.get(label) {
                            pc = end_pc + 1;
                        } else {
                            pc += 1;
                        }
                    } else {
                        pc += 1;
                    }
                }
                VRegInst::Else { label } => {
                    // Coming from then branch: skip to end
                    if let Some(&end_pc) = block_ends.get(label) {
                        pc = end_pc + 1;
                    } else {
                        pc += 1;
                    }
                }
                VRegInst::End { .. } => { pc += 1; }
                VRegInst::Br { label } => {
                    if let Some(&start) = loop_starts.get(label) {
                        pc = start + 1;
                    } else if let Some(&end) = block_ends.get(label) {
                        pc = end + 1;
                    } else {
                        pc += 1;
                    }
                }
                VRegInst::BrIf { cond, label } => {
                    let c = get_vreg(self, *cond).as_i32();
                    if c != 0 {
                        if let Some(&start) = loop_starts.get(label) {
                            pc = start + 1;
                        } else if let Some(&end) = block_ends.get(label) {
                            pc = end + 1;
                        } else {
                            pc += 1;
                        }
                    } else {
                        pc += 1;
                    }
                }
                VRegInst::BrTable { idx, labels, default } => {
                    let i = get_vreg(self, *idx).as_u32() as usize;
                    let target = if i < labels.len() { labels[i] } else { *default };
                    if let Some(&start) = loop_starts.get(&target) {
                        pc = start + 1;
                    } else if let Some(&end) = block_ends.get(&target) {
                        pc = end + 1;
                    } else {
                        pc += 1;
                    }
                }

                // Return - actually returns!
                VRegInst::Return { values } => {
                    if let Some(vreg) = values.last() {
                        return Some(get_vreg(self, *vreg));
                    }
                    return Some(Value::I32(0));
                }

                VRegInst::Select { dst, cond, src1, src2 } => {
                    let c = get_vreg(self, *cond).as_i32();
                    let val = if c != 0 { get_vreg(self, *src1) } else { get_vreg(self, *src2) };
                    set_vreg(self, *dst, val);
                    last_result = Some(*dst);
                    pc += 1;
                }

                VRegInst::Call { func_idx, args, results } => {
                    // Get argument values
                    let arg_vals: Vec<Value> = args.iter().map(|v| get_vreg(self, *v)).collect();

                    // Call the function
                    if let Some(ret) = self.call_func_vreg(*func_idx, &arg_vals) {
                        if let Some(r) = results.first() {
                            set_vreg(self, *r, ret);
                        }
                    }
                    pc += 1;
                }

                VRegInst::CallIndirect { results, .. } => {
                    if let Some(r) = results.first() {
                        set_vreg(self, *r, Value::I32(0));
                    }
                    pc += 1;
                }

                VRegInst::Drop { .. } => { pc += 1; }
                VRegInst::Nop => { pc += 1; }
                VRegInst::Unreachable => { return None; }

                VRegInst::MemorySize { dst } => {
                    set_vreg(self, *dst, Value::I32((self.memory.len() / 65536) as i32));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::MemoryGrow { dst, pages } => {
                    let pages_to_grow = get_vreg(self, *pages).as_i32() as usize;
                    let old_pages = self.memory.len() / 65536;
                    self.memory.resize((old_pages + pages_to_grow) * 65536, 0);
                    set_vreg(self, *dst, Value::I32(old_pages as i32));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::MemoryCopy { dst, src, len } => {
                    let d = get_vreg(self, *dst).as_u32() as usize;
                    let s = get_vreg(self, *src).as_u32() as usize;
                    let l = get_vreg(self, *len).as_u32() as usize;
                    if d + l <= self.memory.len() && s + l <= self.memory.len() {
                        self.memory.copy_within(s..s + l, d);
                    }
                    pc += 1;
                }
                VRegInst::MemoryFill { dst, val, len } => {
                    let d = get_vreg(self, *dst).as_u32() as usize;
                    let v = get_vreg(self, *val).as_i32() as u8;
                    let l = get_vreg(self, *len).as_u32() as usize;
                    if d + l <= self.memory.len() {
                        self.memory[d..d + l].fill(v);
                    }
                    pc += 1;
                }
                VRegInst::Mov { dst, src } => {
                    let val = get_vreg(self, *src);
                    set_vreg(self, *dst, val);
                    pc += 1;
                }
            }
        }

        // Fall through - return last result
        if let Some(vreg) = last_result {
            Some(get_vreg(self, vreg))
        } else {
            Some(Value::I32(0))
        }
    }
}
