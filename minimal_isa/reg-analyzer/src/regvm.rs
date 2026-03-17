//! Convert WASM stack machine to register-based IR with N registers
//!
//! Each instruction becomes something like:
//!   add r1, r2, r3    // r1 = r2 + r3
//!   load r4, [r5+8]   // r4 = mem[r5+8]
//!   spill r2, slot3   // mem[sp+slot3] = r2
//!   reload r2, slot3  // r2 = mem[sp+slot3]

use std::collections::HashMap;
use std::fmt;

// Memory layout for the "all-memory" lowering of locals and globals.
//
// In the target architecture, there are no special "local variable" or "global
// variable" instructions — everything is addressable memory. We lower WASM
// locals and globals to ordinary load/store instructions in the WasmToVReg
// converter (the earliest IR stage) so that:
//
//  1. The entire downstream pipeline (regalloc, PReg lowering, tracing) sees a
//     homogeneous instruction set — no special local/global ops, just loads and
//     stores with different addresses.
//
//  2. Superinstruction analysis on the execution trace can treat ALL memory
//     accesses uniformly, concentrating pair frequencies instead of splitting
//     them across local.get/i32.load/global.get.
//
// We do this HERE (in the WASM→VReg conversion) rather than in a later pass
// because this is where WASM semantics are translated into our target IR.
// Local/global access is a *semantic choice about the target architecture*, not
// a post-hoc optimisation. Placing it at the translation boundary means every
// subsequent pass works with the final instruction set — no surprises, no
// special-casing downstream.
//
// Memory regions (within the VM's 16 MB linear memory):
//   0x000000 .. ~0x200000  — program data (JSON input, .rodata, WASM heap)
//   GLOBALS_MEM_BASE       — globals slots (8 bytes each)
//   FRAME_SP_ADDR          — single u32: current frame base pointer
//   FRAME_STACK_BASE ..    — per-call-level frames for lowered locals

/// Base address in linear memory where globals[0..N] are stored (8 bytes each).
pub const GLOBALS_MEM_BASE: u32 = 0x800000;
/// Fixed address that holds the current frame-stack pointer (u32).
pub const FRAME_SP_ADDR: u32 = 0x800100;
/// Start of the frame stack — each function call allocates a frame here.
pub const FRAME_STACK_BASE: u32 = 0x800200;
/// Bytes per slot (fits both i32 and i64 values).
pub const SLOT_SIZE: u32 = 8;

/// Physical register (0..NUM_REGS-1)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PReg(pub u8);

impl fmt::Display for PReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "r{}", self.0)
    }
}

/// Spill slot on stack
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpillSlot(pub u32);

impl fmt::Display for SpillSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "slot{}", self.0)
    }
}

/// Register-based instruction
#[derive(Debug, Clone)]
pub enum RegInst {
    // Constants
    I32Const { dst: PReg, val: i32 },
    I64Const { dst: PReg, val: i64 },

    // Arithmetic (dst = src1 op src2)
    I32Add { dst: PReg, src1: PReg, src2: PReg },
    I32Sub { dst: PReg, src1: PReg, src2: PReg },
    I32Mul { dst: PReg, src1: PReg, src2: PReg },
    I32And { dst: PReg, src1: PReg, src2: PReg },
    I32Or { dst: PReg, src1: PReg, src2: PReg },
    I32Xor { dst: PReg, src1: PReg, src2: PReg },
    I32Shl { dst: PReg, src1: PReg, src2: PReg },
    I32ShrU { dst: PReg, src1: PReg, src2: PReg },
    I32ShrS { dst: PReg, src1: PReg, src2: PReg },

    I64Add { dst: PReg, src1: PReg, src2: PReg },
    I64Sub { dst: PReg, src1: PReg, src2: PReg },
    I64Mul { dst: PReg, src1: PReg, src2: PReg },
    I64And { dst: PReg, src1: PReg, src2: PReg },
    I64Or { dst: PReg, src1: PReg, src2: PReg },
    I64Xor { dst: PReg, src1: PReg, src2: PReg },
    I64Shl { dst: PReg, src1: PReg, src2: PReg },

    // Division
    I32DivU { dst: PReg, src1: PReg, src2: PReg },

    // Unary
    I32Eqz { dst: PReg, src: PReg },
    I32WrapI64 { dst: PReg, src: PReg },
    I32Clz { dst: PReg, src: PReg },
    I64Eqz { dst: PReg, src: PReg },

    // Comparisons
    I32Eq { dst: PReg, src1: PReg, src2: PReg },
    I32Ne { dst: PReg, src1: PReg, src2: PReg },
    I32LtS { dst: PReg, src1: PReg, src2: PReg },
    I32LtU { dst: PReg, src1: PReg, src2: PReg },
    I32GtS { dst: PReg, src1: PReg, src2: PReg },
    I32GtU { dst: PReg, src1: PReg, src2: PReg },
    I32LeS { dst: PReg, src1: PReg, src2: PReg },
    I32LeU { dst: PReg, src1: PReg, src2: PReg },
    I32GeS { dst: PReg, src1: PReg, src2: PReg },
    I32GeU { dst: PReg, src1: PReg, src2: PReg },

    // Memory (dst = mem[addr + offset])
    I32Load { dst: PReg, addr: PReg, offset: u32 },
    I64Load { dst: PReg, addr: PReg, offset: u32 },
    I32Load8U { dst: PReg, addr: PReg, offset: u32 },
    I32Load8S { dst: PReg, addr: PReg, offset: u32 },
    I32Load16U { dst: PReg, addr: PReg, offset: u32 },
    I32Load16S { dst: PReg, addr: PReg, offset: u32 },

    // Memory stores (mem[addr + offset] = src)
    I32Store { addr: PReg, offset: u32, src: PReg },
    I64Store { addr: PReg, offset: u32, src: PReg },
    I32Store8 { addr: PReg, offset: u32, src: PReg },
    I32Store16 { addr: PReg, offset: u32, src: PReg },

    // Control flow
    BrIf { cond: PReg, label: u32 },
    Br { label: u32 },
    Label { id: u32 },

    // Function calls
    Call { func_idx: u32, args: Vec<PReg>, dst: Option<PReg> },

    // Register moves
    Move { dst: PReg, src: PReg },

    // Spill/reload for register pressure
    Spill { src: PReg, slot: SpillSlot },
    Reload { dst: PReg, slot: SpillSlot },

    // Select (dst = cond ? src1 : src2)
    Select { dst: PReg, cond: PReg, src1: PReg, src2: PReg },

    // Misc
    Unreachable,
    Nop,
    Drop { src: PReg },

    // Block markers (for structure)
    Block { label: u32 },
    Loop { label: u32 },
    If { cond: PReg, label: u32 },
    Else { label: u32 },
    End { label: u32 },
    Return,

    // Local variable access (before register allocation maps these away)
    LocalGet { dst: PReg, local: u32 },
    LocalSet { local: u32, src: PReg },

    // Global variable access
    GlobalGet { dst: PReg, global: u32 },
    GlobalSet { global: u32, src: PReg },
}

impl fmt::Display for RegInst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegInst::I32Const { dst, val } => write!(f, "{} = i32.const {}", dst, val),
            RegInst::I64Const { dst, val } => write!(f, "{} = i64.const {}", dst, val),

            RegInst::I32Add { dst, src1, src2 } => write!(f, "{} = i32.add {}, {}", dst, src1, src2),
            RegInst::I32Sub { dst, src1, src2 } => write!(f, "{} = i32.sub {}, {}", dst, src1, src2),
            RegInst::I32Mul { dst, src1, src2 } => write!(f, "{} = i32.mul {}, {}", dst, src1, src2),
            RegInst::I32And { dst, src1, src2 } => write!(f, "{} = i32.and {}, {}", dst, src1, src2),
            RegInst::I32Or { dst, src1, src2 } => write!(f, "{} = i32.or {}, {}", dst, src1, src2),
            RegInst::I32Xor { dst, src1, src2 } => write!(f, "{} = i32.xor {}, {}", dst, src1, src2),
            RegInst::I32Shl { dst, src1, src2 } => write!(f, "{} = i32.shl {}, {}", dst, src1, src2),
            RegInst::I32ShrU { dst, src1, src2 } => write!(f, "{} = i32.shr_u {}, {}", dst, src1, src2),
            RegInst::I32ShrS { dst, src1, src2 } => write!(f, "{} = i32.shr_s {}, {}", dst, src1, src2),

            RegInst::I64Add { dst, src1, src2 } => write!(f, "{} = i64.add {}, {}", dst, src1, src2),
            RegInst::I64Sub { dst, src1, src2 } => write!(f, "{} = i64.sub {}, {}", dst, src1, src2),
            RegInst::I64Mul { dst, src1, src2 } => write!(f, "{} = i64.mul {}, {}", dst, src1, src2),
            RegInst::I64And { dst, src1, src2 } => write!(f, "{} = i64.and {}, {}", dst, src1, src2),
            RegInst::I64Or { dst, src1, src2 } => write!(f, "{} = i64.or {}, {}", dst, src1, src2),
            RegInst::I64Xor { dst, src1, src2 } => write!(f, "{} = i64.xor {}, {}", dst, src1, src2),
            RegInst::I64Shl { dst, src1, src2 } => write!(f, "{} = i64.shl {}, {}", dst, src1, src2),

            RegInst::I32DivU { dst, src1, src2 } => write!(f, "{} = i32.div_u {}, {}", dst, src1, src2),

            RegInst::I32Eqz { dst, src } => write!(f, "{} = i32.eqz {}", dst, src),
            RegInst::I32WrapI64 { dst, src } => write!(f, "{} = i32.wrap_i64 {}", dst, src),
            RegInst::I32Clz { dst, src } => write!(f, "{} = i32.clz {}", dst, src),
            RegInst::I64Eqz { dst, src } => write!(f, "{} = i64.eqz {}", dst, src),

            RegInst::I32Eq { dst, src1, src2 } => write!(f, "{} = i32.eq {}, {}", dst, src1, src2),
            RegInst::I32Ne { dst, src1, src2 } => write!(f, "{} = i32.ne {}, {}", dst, src1, src2),
            RegInst::I32LtS { dst, src1, src2 } => write!(f, "{} = i32.lt_s {}, {}", dst, src1, src2),
            RegInst::I32LtU { dst, src1, src2 } => write!(f, "{} = i32.lt_u {}, {}", dst, src1, src2),
            RegInst::I32GtS { dst, src1, src2 } => write!(f, "{} = i32.gt_s {}, {}", dst, src1, src2),
            RegInst::I32GtU { dst, src1, src2 } => write!(f, "{} = i32.gt_u {}, {}", dst, src1, src2),
            RegInst::I32LeS { dst, src1, src2 } => write!(f, "{} = i32.le_s {}, {}", dst, src1, src2),
            RegInst::I32LeU { dst, src1, src2 } => write!(f, "{} = i32.le_u {}, {}", dst, src1, src2),
            RegInst::I32GeS { dst, src1, src2 } => write!(f, "{} = i32.ge_s {}, {}", dst, src1, src2),
            RegInst::I32GeU { dst, src1, src2 } => write!(f, "{} = i32.ge_u {}, {}", dst, src1, src2),

            RegInst::I32Load { dst, addr, offset } => write!(f, "{} = i32.load {}[{}]", dst, addr, offset),
            RegInst::I64Load { dst, addr, offset } => write!(f, "{} = i64.load {}[{}]", dst, addr, offset),
            RegInst::I32Load8U { dst, addr, offset } => write!(f, "{} = i32.load8_u {}[{}]", dst, addr, offset),
            RegInst::I32Load8S { dst, addr, offset } => write!(f, "{} = i32.load8_s {}[{}]", dst, addr, offset),
            RegInst::I32Load16U { dst, addr, offset } => write!(f, "{} = i32.load16_u {}[{}]", dst, addr, offset),
            RegInst::I32Load16S { dst, addr, offset } => write!(f, "{} = i32.load16_s {}[{}]", dst, addr, offset),

            RegInst::I32Store { addr, offset, src } => write!(f, "i32.store {}[{}], {}", addr, offset, src),
            RegInst::I64Store { addr, offset, src } => write!(f, "i64.store {}[{}], {}", addr, offset, src),
            RegInst::I32Store8 { addr, offset, src } => write!(f, "i32.store8 {}[{}], {}", addr, offset, src),
            RegInst::I32Store16 { addr, offset, src } => write!(f, "i32.store16 {}[{}], {}", addr, offset, src),

            RegInst::BrIf { cond, label } => write!(f, "br_if {}, L{}", cond, label),
            RegInst::Br { label } => write!(f, "br L{}", label),
            RegInst::Label { id } => write!(f, "L{}:", id),

            RegInst::Call { func_idx, args, dst } => {
                if let Some(d) = dst {
                    write!(f, "{} = call func{} ({:?})", d, func_idx, args)
                } else {
                    write!(f, "call func{} ({:?})", func_idx, args)
                }
            }

            RegInst::Move { dst, src } => write!(f, "{} = move {}", dst, src),
            RegInst::Spill { src, slot } => write!(f, "spill {}, {}", src, slot),
            RegInst::Reload { dst, slot } => write!(f, "{} = reload {}", dst, slot),

            RegInst::Select { dst, cond, src1, src2 } => write!(f, "{} = select {}, {}, {}", dst, cond, src1, src2),

            RegInst::Unreachable => write!(f, "unreachable"),
            RegInst::Nop => write!(f, "nop"),
            RegInst::Drop { src } => write!(f, "drop {}", src),

            RegInst::Block { label } => write!(f, "block L{}:", label),
            RegInst::Loop { label } => write!(f, "loop L{}:", label),
            RegInst::If { cond, label } => write!(f, "if {}, L{}", cond, label),
            RegInst::Else { label } => write!(f, "else L{}", label),
            RegInst::End { label } => write!(f, "end L{}", label),
            RegInst::Return => write!(f, "return"),

            RegInst::LocalGet { dst, local } => write!(f, "{} = local.get {}", dst, local),
            RegInst::LocalSet { local, src } => write!(f, "local.set {}, {}", local, src),

            RegInst::GlobalGet { dst, global } => write!(f, "{} = global.get {}", dst, global),
            RegInst::GlobalSet { global, src } => write!(f, "global.set {}, {}", global, src),
        }
    }
}

/// Virtual register (before allocation)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VReg(pub u32);

/// Instruction with virtual registers
#[derive(Debug, Clone)]
pub enum VRegInst {
    I32Const { dst: VReg, val: i32 },
    I64Const { dst: VReg, val: i64 },

    I32Add { dst: VReg, src1: VReg, src2: VReg },
    I32Sub { dst: VReg, src1: VReg, src2: VReg },
    I32Mul { dst: VReg, src1: VReg, src2: VReg },
    I32DivS { dst: VReg, src1: VReg, src2: VReg },
    I32DivU { dst: VReg, src1: VReg, src2: VReg },
    I32RemS { dst: VReg, src1: VReg, src2: VReg },
    I32RemU { dst: VReg, src1: VReg, src2: VReg },
    I32And { dst: VReg, src1: VReg, src2: VReg },
    I32Or { dst: VReg, src1: VReg, src2: VReg },
    I32Xor { dst: VReg, src1: VReg, src2: VReg },
    I32Shl { dst: VReg, src1: VReg, src2: VReg },
    I32ShrU { dst: VReg, src1: VReg, src2: VReg },
    I32ShrS { dst: VReg, src1: VReg, src2: VReg },
    I32Rotl { dst: VReg, src1: VReg, src2: VReg },
    I32Rotr { dst: VReg, src1: VReg, src2: VReg },

    I64Add { dst: VReg, src1: VReg, src2: VReg },
    I64Sub { dst: VReg, src1: VReg, src2: VReg },
    I64Mul { dst: VReg, src1: VReg, src2: VReg },
    I64DivS { dst: VReg, src1: VReg, src2: VReg },
    I64DivU { dst: VReg, src1: VReg, src2: VReg },
    I64RemS { dst: VReg, src1: VReg, src2: VReg },
    I64RemU { dst: VReg, src1: VReg, src2: VReg },
    I64And { dst: VReg, src1: VReg, src2: VReg },
    I64Or { dst: VReg, src1: VReg, src2: VReg },
    I64Xor { dst: VReg, src1: VReg, src2: VReg },
    I64Shl { dst: VReg, src1: VReg, src2: VReg },
    I64ShrU { dst: VReg, src1: VReg, src2: VReg },
    I64ShrS { dst: VReg, src1: VReg, src2: VReg },

    I32Eqz { dst: VReg, src: VReg },
    I32Clz { dst: VReg, src: VReg },
    I32Ctz { dst: VReg, src: VReg },
    I32Popcnt { dst: VReg, src: VReg },
    I64Eqz { dst: VReg, src: VReg },
    I64Clz { dst: VReg, src: VReg },
    I64Ctz { dst: VReg, src: VReg },

    I32Eq { dst: VReg, src1: VReg, src2: VReg },
    I32Ne { dst: VReg, src1: VReg, src2: VReg },
    I32LtS { dst: VReg, src1: VReg, src2: VReg },
    I32LtU { dst: VReg, src1: VReg, src2: VReg },
    I32GtS { dst: VReg, src1: VReg, src2: VReg },
    I32GtU { dst: VReg, src1: VReg, src2: VReg },
    I32LeS { dst: VReg, src1: VReg, src2: VReg },
    I32LeU { dst: VReg, src1: VReg, src2: VReg },
    I32GeS { dst: VReg, src1: VReg, src2: VReg },
    I32GeU { dst: VReg, src1: VReg, src2: VReg },

    I64Eq { dst: VReg, src1: VReg, src2: VReg },
    I64Ne { dst: VReg, src1: VReg, src2: VReg },
    I64LtS { dst: VReg, src1: VReg, src2: VReg },
    I64LtU { dst: VReg, src1: VReg, src2: VReg },
    I64GtS { dst: VReg, src1: VReg, src2: VReg },
    I64GtU { dst: VReg, src1: VReg, src2: VReg },
    I64LeS { dst: VReg, src1: VReg, src2: VReg },
    I64LeU { dst: VReg, src1: VReg, src2: VReg },
    I64GeS { dst: VReg, src1: VReg, src2: VReg },
    I64GeU { dst: VReg, src1: VReg, src2: VReg },

    I32WrapI64 { dst: VReg, src: VReg },
    I64ExtendI32S { dst: VReg, src: VReg },
    I64ExtendI32U { dst: VReg, src: VReg },
    I32Extend8S { dst: VReg, src: VReg },
    I32Extend16S { dst: VReg, src: VReg },

    I32Load { dst: VReg, addr: VReg, offset: u32 },
    I64Load { dst: VReg, addr: VReg, offset: u32 },
    I32Load8U { dst: VReg, addr: VReg, offset: u32 },
    I32Load8S { dst: VReg, addr: VReg, offset: u32 },
    I32Load16U { dst: VReg, addr: VReg, offset: u32 },
    I32Load16S { dst: VReg, addr: VReg, offset: u32 },
    I64Load8U { dst: VReg, addr: VReg, offset: u32 },
    I64Load8S { dst: VReg, addr: VReg, offset: u32 },
    I64Load16U { dst: VReg, addr: VReg, offset: u32 },
    I64Load16S { dst: VReg, addr: VReg, offset: u32 },
    I64Load32U { dst: VReg, addr: VReg, offset: u32 },
    I64Load32S { dst: VReg, addr: VReg, offset: u32 },

    I32Store { addr: VReg, offset: u32, src: VReg },
    I64Store { addr: VReg, offset: u32, src: VReg },
    I32Store8 { addr: VReg, offset: u32, src: VReg },
    I32Store16 { addr: VReg, offset: u32, src: VReg },
    I64Store8 { addr: VReg, offset: u32, src: VReg },
    I64Store16 { addr: VReg, offset: u32, src: VReg },
    I64Store32 { addr: VReg, offset: u32, src: VReg },

    LocalGet { dst: VReg, local: u32 },
    LocalSet { local: u32, src: VReg },
    LocalTee { dst: VReg, local: u32, src: VReg },

    GlobalGet { dst: VReg, global: u32 },
    GlobalSet { global: u32, src: VReg },

    Call { func_idx: u32, args: Vec<VReg>, results: Vec<VReg> },
    CallIndirect { table: u32, type_idx: u32, func_ref: VReg, args: Vec<VReg>, results: Vec<VReg> },

    Select { dst: VReg, cond: VReg, src1: VReg, src2: VReg },

    BrIf { cond: VReg, label: u32 },
    BrTable { idx: VReg, labels: Vec<u32>, default: u32 },
    Br { label: u32 },

    Block { label: u32 },
    Loop { label: u32 },
    If { cond: VReg, label: u32 },
    Else { label: u32 },
    End { label: u32 },

    Return { values: Vec<VReg> },
    Unreachable,
    Nop,
    Drop { src: VReg },

    MemorySize { dst: VReg },
    MemoryGrow { dst: VReg, pages: VReg },

    // Bulk memory operations
    MemoryCopy { dst: VReg, src: VReg, len: VReg },
    MemoryFill { dst: VReg, val: VReg, len: VReg },

    /// Register-to-register move (inserted by legalization pass)
    Mov { dst: VReg, src: VReg },
}

/// Convert WASM function to virtual register IR
/// Function signature: (num_params, num_results)
pub type FuncSig = (u32, u32);

/// Whether a local/global slot holds a 32-bit or 64-bit value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SlotType { I32, I64 }

pub struct WasmToVReg {
    /// Next virtual register to allocate
    next_vreg: u32,
    /// Operand stack (virtual registers)
    stack: Vec<VReg>,
    /// Local variables mapped to virtual registers
    locals: HashMap<u32, VReg>,
    /// Output instructions
    pub instructions: Vec<VRegInst>,
    /// Label counter
    next_label: u32,
    /// Control flow stack (label ids)
    control_stack: Vec<u32>,
    /// Function signatures for all functions (indexed by function index)
    func_sigs: Vec<FuncSig>,
    /// If set, locals/globals are lowered to memory loads/stores.
    /// frame_ptr VReg holds the base address of the current frame.
    frame_ptr: Option<VReg>,
    /// VReg holding GLOBALS_MEM_BASE constant (when memory_locals is active).
    globals_ptr: Option<VReg>,
    /// Type of each local slot (params first, then declared locals).
    local_types: Vec<SlotType>,
    /// Type of each global slot.
    global_types: Vec<SlotType>,
}

impl WasmToVReg {
    pub fn new(num_params: u32, num_locals: u32) -> Self {
        Self::new_with_sigs(num_params, num_locals, Vec::new())
    }

    pub fn new_with_sigs(num_params: u32, num_locals: u32, func_sigs: Vec<FuncSig>) -> Self {
        let mut converter = Self {
            next_vreg: 0,
            stack: Vec::new(),
            locals: HashMap::new(),
            instructions: Vec::new(),
            next_label: 0,
            control_stack: Vec::new(),
            func_sigs,
            frame_ptr: None,
            globals_ptr: None,
            local_types: Vec::new(),
            global_types: Vec::new(),
        };

        // Allocate virtual registers for all locals (params + locals)
        for i in 0..(num_params + num_locals) {
            let vreg = converter.alloc_vreg();
            converter.locals.insert(i, vreg);
        }

        converter
    }

    /// Create a converter that lowers locals and globals to memory loads/stores.
    ///
    /// Instead of emitting LocalGet/Set/Tee and GlobalGet/Set, emits I32Load/
    /// I64Load and I32Store/I64Store through a frame pointer (for locals) and a
    /// globals base pointer.  This produces a homogeneous instruction stream
    /// where every value access is an explicit memory operation — matching the
    /// target architecture's "everything is addressable memory" model.
    pub fn new_memory_lowered(
        num_params: u32,
        num_locals: u32,
        func_sigs: Vec<FuncSig>,
        local_types: Vec<SlotType>,
        global_types: Vec<SlotType>,
    ) -> Self {
        let mut converter = Self {
            next_vreg: 0,
            stack: Vec::new(),
            locals: HashMap::new(),
            instructions: Vec::new(),
            next_label: 0,
            control_stack: Vec::new(),
            func_sigs,
            frame_ptr: None,
            globals_ptr: None,
            local_types,
            global_types,
        };

        // Still allocate VRegs for locals (needed for stack simulation bookkeeping)
        for i in 0..(num_params + num_locals) {
            let vreg = converter.alloc_vreg();
            converter.locals.insert(i, vreg);
        }

        // Prologue: load frame pointer, advance frame stack, set globals base.
        //
        // The frame protocol:
        //   1. Read our frame base from memory[FRAME_SP_ADDR]  (caller wrote params here)
        //   2. Advance FRAME_SP_ADDR past our frame so nested calls get fresh space
        //   3. Load GLOBALS_MEM_BASE constant for global access
        let zero = converter.alloc_vreg();
        converter.instructions.push(VRegInst::I32Const { dst: zero, val: 0 });

        // fp = memory[FRAME_SP_ADDR]  — our frame base
        let fp = converter.alloc_vreg();
        converter.instructions.push(VRegInst::I32Load {
            dst: fp, addr: zero, offset: FRAME_SP_ADDR,
        });
        converter.frame_ptr = Some(fp);

        // Advance: memory[FRAME_SP_ADDR] = fp + (num_params + num_locals) * SLOT_SIZE
        let frame_size = (num_params + num_locals) * SLOT_SIZE;
        let frame_size_vreg = converter.alloc_vreg();
        converter.instructions.push(VRegInst::I32Const { dst: frame_size_vreg, val: frame_size as i32 });
        let new_sp = converter.alloc_vreg();
        converter.instructions.push(VRegInst::I32Add { dst: new_sp, src1: fp, src2: frame_size_vreg });
        converter.instructions.push(VRegInst::I32Store { addr: zero, offset: FRAME_SP_ADDR, src: new_sp });

        // globals base
        let gp = converter.alloc_vreg();
        converter.instructions.push(VRegInst::I32Const {
            dst: gp, val: GLOBALS_MEM_BASE as i32,
        });
        converter.globals_ptr = Some(gp);

        converter
    }

    fn alloc_vreg(&mut self) -> VReg {
        let vreg = VReg(self.next_vreg);
        self.next_vreg += 1;
        vreg
    }

    fn alloc_label(&mut self) -> u32 {
        let label = self.next_label;
        self.next_label += 1;
        label
    }

    fn push(&mut self, vreg: VReg) {
        self.stack.push(vreg);
    }

    fn pop(&mut self) -> VReg {
        self.stack.pop().unwrap_or_else(|| {
            // Create a dummy vreg for underflow cases (complex control flow)
            self.alloc_vreg()
        })
    }

    fn peek(&self) -> VReg {
        self.stack.last().copied().unwrap_or_else(|| VReg(0))
    }

    pub fn convert_op(&mut self, op: &wasmparser::Operator) {
        use wasmparser::Operator::*;

        match op {
            // Constants
            I32Const { value } => {
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32Const { dst, val: *value });
                self.push(dst);
            }
            I64Const { value } => {
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64Const { dst, val: *value });
                self.push(dst);
            }

            // Binary i32 ops
            I32Add => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32Add { dst, src1, src2 });
                self.push(dst);
            }
            I32Sub => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32Sub { dst, src1, src2 });
                self.push(dst);
            }
            I32Mul => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32Mul { dst, src1, src2 });
                self.push(dst);
            }
            I32DivS => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32DivS { dst, src1, src2 });
                self.push(dst);
            }
            I32DivU => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32DivU { dst, src1, src2 });
                self.push(dst);
            }
            I32RemS => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32RemS { dst, src1, src2 });
                self.push(dst);
            }
            I32RemU => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32RemU { dst, src1, src2 });
                self.push(dst);
            }
            I32And => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32And { dst, src1, src2 });
                self.push(dst);
            }
            I32Or => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32Or { dst, src1, src2 });
                self.push(dst);
            }
            I32Xor => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32Xor { dst, src1, src2 });
                self.push(dst);
            }
            I32Shl => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32Shl { dst, src1, src2 });
                self.push(dst);
            }
            I32ShrU => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32ShrU { dst, src1, src2 });
                self.push(dst);
            }
            I32ShrS => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32ShrS { dst, src1, src2 });
                self.push(dst);
            }
            I32Rotl => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32Rotl { dst, src1, src2 });
                self.push(dst);
            }
            I32Rotr => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32Rotr { dst, src1, src2 });
                self.push(dst);
            }

            // Binary i64 ops
            I64Add => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64Add { dst, src1, src2 });
                self.push(dst);
            }
            I64Sub => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64Sub { dst, src1, src2 });
                self.push(dst);
            }
            I64Mul => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64Mul { dst, src1, src2 });
                self.push(dst);
            }
            I64DivS => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64DivS { dst, src1, src2 });
                self.push(dst);
            }
            I64DivU => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64DivU { dst, src1, src2 });
                self.push(dst);
            }
            I64RemS => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64RemS { dst, src1, src2 });
                self.push(dst);
            }
            I64RemU => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64RemU { dst, src1, src2 });
                self.push(dst);
            }
            I64And => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64And { dst, src1, src2 });
                self.push(dst);
            }
            I64Or => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64Or { dst, src1, src2 });
                self.push(dst);
            }
            I64Xor => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64Xor { dst, src1, src2 });
                self.push(dst);
            }
            I64Shl => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64Shl { dst, src1, src2 });
                self.push(dst);
            }
            I64ShrU => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64ShrU { dst, src1, src2 });
                self.push(dst);
            }
            I64ShrS => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64ShrS { dst, src1, src2 });
                self.push(dst);
            }

            // Unary ops
            I32Eqz => {
                let src = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32Eqz { dst, src });
                self.push(dst);
            }
            I32Clz => {
                let src = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32Clz { dst, src });
                self.push(dst);
            }
            I32Ctz => {
                let src = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32Ctz { dst, src });
                self.push(dst);
            }
            I32Popcnt => {
                let src = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32Popcnt { dst, src });
                self.push(dst);
            }
            I64Eqz => {
                let src = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64Eqz { dst, src });
                self.push(dst);
            }
            I64Clz => {
                let src = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64Clz { dst, src });
                self.push(dst);
            }
            I64Ctz => {
                let src = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64Ctz { dst, src });
                self.push(dst);
            }

            // Comparisons i32
            I32Eq => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32Eq { dst, src1, src2 });
                self.push(dst);
            }
            I32Ne => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32Ne { dst, src1, src2 });
                self.push(dst);
            }
            I32LtS => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32LtS { dst, src1, src2 });
                self.push(dst);
            }
            I32LtU => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32LtU { dst, src1, src2 });
                self.push(dst);
            }
            I32GtS => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32GtS { dst, src1, src2 });
                self.push(dst);
            }
            I32GtU => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32GtU { dst, src1, src2 });
                self.push(dst);
            }
            I32LeS => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32LeS { dst, src1, src2 });
                self.push(dst);
            }
            I32LeU => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32LeU { dst, src1, src2 });
                self.push(dst);
            }
            I32GeS => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32GeS { dst, src1, src2 });
                self.push(dst);
            }
            I32GeU => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32GeU { dst, src1, src2 });
                self.push(dst);
            }

            // Comparisons i64
            I64Eq => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64Eq { dst, src1, src2 });
                self.push(dst);
            }
            I64Ne => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64Ne { dst, src1, src2 });
                self.push(dst);
            }
            I64LtS => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64LtS { dst, src1, src2 });
                self.push(dst);
            }
            I64LtU => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64LtU { dst, src1, src2 });
                self.push(dst);
            }
            I64GtS => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64GtS { dst, src1, src2 });
                self.push(dst);
            }
            I64GtU => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64GtU { dst, src1, src2 });
                self.push(dst);
            }
            I64LeS => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64LeS { dst, src1, src2 });
                self.push(dst);
            }
            I64LeU => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64LeU { dst, src1, src2 });
                self.push(dst);
            }
            I64GeS => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64GeS { dst, src1, src2 });
                self.push(dst);
            }
            I64GeU => {
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64GeU { dst, src1, src2 });
                self.push(dst);
            }

            // Type conversions
            I32WrapI64 => {
                let src = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32WrapI64 { dst, src });
                self.push(dst);
            }
            I64ExtendI32S => {
                let src = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64ExtendI32S { dst, src });
                self.push(dst);
            }
            I64ExtendI32U => {
                let src = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64ExtendI32U { dst, src });
                self.push(dst);
            }
            I32Extend8S => {
                let src = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32Extend8S { dst, src });
                self.push(dst);
            }
            I32Extend16S => {
                let src = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32Extend16S { dst, src });
                self.push(dst);
            }

            // Memory loads
            I32Load { memarg } => {
                let addr = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32Load { dst, addr, offset: memarg.offset as u32 });
                self.push(dst);
            }
            I64Load { memarg } => {
                let addr = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64Load { dst, addr, offset: memarg.offset as u32 });
                self.push(dst);
            }
            I32Load8U { memarg } => {
                let addr = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32Load8U { dst, addr, offset: memarg.offset as u32 });
                self.push(dst);
            }
            I32Load8S { memarg } => {
                let addr = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32Load8S { dst, addr, offset: memarg.offset as u32 });
                self.push(dst);
            }
            I32Load16U { memarg } => {
                let addr = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32Load16U { dst, addr, offset: memarg.offset as u32 });
                self.push(dst);
            }
            I32Load16S { memarg } => {
                let addr = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I32Load16S { dst, addr, offset: memarg.offset as u32 });
                self.push(dst);
            }
            I64Load8U { memarg } => {
                let addr = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64Load8U { dst, addr, offset: memarg.offset as u32 });
                self.push(dst);
            }
            I64Load8S { memarg } => {
                let addr = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64Load8S { dst, addr, offset: memarg.offset as u32 });
                self.push(dst);
            }
            I64Load16U { memarg } => {
                let addr = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64Load16U { dst, addr, offset: memarg.offset as u32 });
                self.push(dst);
            }
            I64Load16S { memarg } => {
                let addr = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64Load16S { dst, addr, offset: memarg.offset as u32 });
                self.push(dst);
            }
            I64Load32U { memarg } => {
                let addr = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64Load32U { dst, addr, offset: memarg.offset as u32 });
                self.push(dst);
            }
            I64Load32S { memarg } => {
                let addr = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::I64Load32S { dst, addr, offset: memarg.offset as u32 });
                self.push(dst);
            }

            // Memory stores
            I32Store { memarg } => {
                let src = self.pop();
                let addr = self.pop();
                self.instructions.push(VRegInst::I32Store { addr, offset: memarg.offset as u32, src });
            }
            I64Store { memarg } => {
                let src = self.pop();
                let addr = self.pop();
                self.instructions.push(VRegInst::I64Store { addr, offset: memarg.offset as u32, src });
            }
            I32Store8 { memarg } => {
                let src = self.pop();
                let addr = self.pop();
                self.instructions.push(VRegInst::I32Store8 { addr, offset: memarg.offset as u32, src });
            }
            I32Store16 { memarg } => {
                let src = self.pop();
                let addr = self.pop();
                self.instructions.push(VRegInst::I32Store16 { addr, offset: memarg.offset as u32, src });
            }
            I64Store8 { memarg } => {
                let src = self.pop();
                let addr = self.pop();
                self.instructions.push(VRegInst::I64Store8 { addr, offset: memarg.offset as u32, src });
            }
            I64Store16 { memarg } => {
                let src = self.pop();
                let addr = self.pop();
                self.instructions.push(VRegInst::I64Store16 { addr, offset: memarg.offset as u32, src });
            }
            I64Store32 { memarg } => {
                let src = self.pop();
                let addr = self.pop();
                self.instructions.push(VRegInst::I64Store32 { addr, offset: memarg.offset as u32, src });
            }

            // Local variables — lowered to memory loads/stores when frame_ptr is set
            LocalGet { local_index } => {
                if let Some(fp) = self.frame_ptr {
                    let offset = *local_index * SLOT_SIZE;
                    let dst = self.alloc_vreg();
                    let ty = self.local_types.get(*local_index as usize)
                        .copied().unwrap_or(SlotType::I32);
                    match ty {
                        SlotType::I32 => self.instructions.push(VRegInst::I32Load { dst, addr: fp, offset }),
                        SlotType::I64 => self.instructions.push(VRegInst::I64Load { dst, addr: fp, offset }),
                    }
                    self.push(dst);
                } else {
                    let _local_vreg = self.locals[local_index];
                    let dst = self.alloc_vreg();
                    self.instructions.push(VRegInst::LocalGet { dst, local: *local_index });
                    self.push(dst);
                }
            }
            LocalSet { local_index } => {
                let src = self.pop();
                if let Some(fp) = self.frame_ptr {
                    let offset = *local_index * SLOT_SIZE;
                    let ty = self.local_types.get(*local_index as usize)
                        .copied().unwrap_or(SlotType::I32);
                    match ty {
                        SlotType::I32 => self.instructions.push(VRegInst::I32Store { addr: fp, offset, src }),
                        SlotType::I64 => self.instructions.push(VRegInst::I64Store { addr: fp, offset, src }),
                    }
                } else {
                    self.instructions.push(VRegInst::LocalSet { local: *local_index, src });
                }
            }
            LocalTee { local_index } => {
                let src = self.peek();
                if let Some(fp) = self.frame_ptr {
                    let offset = *local_index * SLOT_SIZE;
                    let ty = self.local_types.get(*local_index as usize)
                        .copied().unwrap_or(SlotType::I32);
                    match ty {
                        SlotType::I32 => self.instructions.push(VRegInst::I32Store { addr: fp, offset, src }),
                        SlotType::I64 => self.instructions.push(VRegInst::I64Store { addr: fp, offset, src }),
                    }
                    // LocalTee leaves value on stack — it's already there (peek didn't remove it)
                } else {
                    let dst = self.alloc_vreg();
                    self.instructions.push(VRegInst::LocalTee { dst, local: *local_index, src });
                    // LocalTee leaves value on stack
                }
            }

            // Globals — lowered to memory loads/stores when globals_ptr is set
            GlobalGet { global_index } => {
                if let Some(gp) = self.globals_ptr {
                    let offset = *global_index * SLOT_SIZE;
                    let dst = self.alloc_vreg();
                    let ty = self.global_types.get(*global_index as usize)
                        .copied().unwrap_or(SlotType::I32);
                    match ty {
                        SlotType::I32 => self.instructions.push(VRegInst::I32Load { dst, addr: gp, offset }),
                        SlotType::I64 => self.instructions.push(VRegInst::I64Load { dst, addr: gp, offset }),
                    }
                    self.push(dst);
                } else {
                    let dst = self.alloc_vreg();
                    self.instructions.push(VRegInst::GlobalGet { dst, global: *global_index });
                    self.push(dst);
                }
            }
            GlobalSet { global_index } => {
                let src = self.pop();
                if let Some(gp) = self.globals_ptr {
                    let offset = *global_index * SLOT_SIZE;
                    let ty = self.global_types.get(*global_index as usize)
                        .copied().unwrap_or(SlotType::I32);
                    match ty {
                        SlotType::I32 => self.instructions.push(VRegInst::I32Store { addr: gp, offset, src }),
                        SlotType::I64 => self.instructions.push(VRegInst::I64Store { addr: gp, offset, src }),
                    }
                } else {
                    self.instructions.push(VRegInst::GlobalSet { global: *global_index, src });
                }
            }

            // Control flow
            Block { blockty: _ } => {
                let label = self.alloc_label();
                self.control_stack.push(label);
                self.instructions.push(VRegInst::Block { label });
            }
            Loop { blockty: _ } => {
                let label = self.alloc_label();
                self.control_stack.push(label);
                self.instructions.push(VRegInst::Loop { label });
            }
            If { blockty: _ } => {
                let cond = self.pop();
                let label = self.alloc_label();
                self.control_stack.push(label);
                self.instructions.push(VRegInst::If { cond, label });
            }
            Else => {
                let label = *self.control_stack.last().unwrap();
                self.instructions.push(VRegInst::Else { label });
            }
            End => {
                if let Some(label) = self.control_stack.pop() {
                    self.instructions.push(VRegInst::End { label });
                } else {
                    // Function body end: emit Return with whatever's on the stack
                    if !self.stack.is_empty() {
                        let values = self.stack.clone();
                        self.instructions.push(VRegInst::Return { values });
                    }
                }
            }
            Br { relative_depth } => {
                let idx = self.control_stack.len() - 1 - (*relative_depth as usize);
                let label = self.control_stack[idx];
                self.instructions.push(VRegInst::Br { label });
            }
            BrIf { relative_depth } => {
                let cond = self.pop();
                let idx = self.control_stack.len() - 1 - (*relative_depth as usize);
                let label = self.control_stack[idx];
                self.instructions.push(VRegInst::BrIf { cond, label });
            }
            BrTable { targets } => {
                let idx = self.pop();
                let labels: Vec<u32> = targets.targets().map(|t| {
                    let t = t.unwrap();
                    let stack_idx = self.control_stack.len() - 1 - (t as usize);
                    self.control_stack[stack_idx]
                }).collect();
                let default_idx = self.control_stack.len() - 1 - (targets.default() as usize);
                let default = self.control_stack[default_idx];
                self.instructions.push(VRegInst::BrTable { idx, labels, default });
            }

            Return => {
                // Collect return values from stack
                let values = self.stack.clone();
                self.instructions.push(VRegInst::Return { values });
            }

            Call { function_index } => {
                // Look up function signature to determine args/results
                let (num_params, num_results) = self.func_sigs
                    .get(*function_index as usize)
                    .copied()
                    .unwrap_or((0, 1)); // Default: no args, one result

                // Pop arguments from stack (in reverse order)
                let mut args = Vec::new();
                for _ in 0..num_params {
                    args.push(self.pop());
                }
                args.reverse(); // Arguments are pushed in order, so reverse after popping

                // Create result vregs
                let mut results = Vec::new();
                for _ in 0..num_results {
                    results.push(self.alloc_vreg());
                }

                self.instructions.push(VRegInst::Call {
                    func_idx: *function_index,
                    args,
                    results: results.clone(),
                });

                // Push results onto stack
                for r in results {
                    self.push(r);
                }
            }

            CallIndirect { type_index, table_index } => {
                let func_ref = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::CallIndirect {
                    table: *table_index,
                    type_idx: *type_index,
                    func_ref,
                    args: vec![],
                    results: vec![dst],
                });
                self.push(dst);
            }

            Select => {
                let cond = self.pop();
                let src2 = self.pop();
                let src1 = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::Select { dst, cond, src1, src2 });
                self.push(dst);
            }

            Drop => {
                let src = self.pop();
                self.instructions.push(VRegInst::Drop { src });
            }

            Unreachable => {
                self.instructions.push(VRegInst::Unreachable);
            }

            Nop => {
                self.instructions.push(VRegInst::Nop);
            }

            MemorySize { .. } => {
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::MemorySize { dst });
                self.push(dst);
            }

            MemoryGrow { .. } => {
                let pages = self.pop();
                let dst = self.alloc_vreg();
                self.instructions.push(VRegInst::MemoryGrow { dst, pages });
                self.push(dst);
            }

            MemoryCopy { .. } => {
                let len = self.pop();
                let src = self.pop();
                let dst = self.pop();
                self.instructions.push(VRegInst::MemoryCopy { dst, src, len });
            }

            MemoryFill { .. } => {
                let len = self.pop();
                let val = self.pop();
                let dst = self.pop();
                self.instructions.push(VRegInst::MemoryFill { dst, val, len });
            }

            // Ignore other ops for now
            _ => {
                // eprintln!("Unhandled op: {:?}", op);
            }
        }
    }

    pub fn num_vregs(&self) -> u32 {
        self.next_vreg
    }
}
