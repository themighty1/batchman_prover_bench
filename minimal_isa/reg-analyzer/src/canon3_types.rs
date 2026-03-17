//! Public types for the Canon3 ISA, usable by external crates.
//!
//! These types carry no dependencies on reg-analyzer internals.

/// A single Canon3 instruction: opcode byte + immediate.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Canon3Inst {
    pub op: u8,
    pub imm: i32,
}

/// A compiled Canon3 program: everything needed to execute.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Canon3Program {
    /// One opcode byte per instruction, indexed by PC.
    pub code: Vec<u8>,
    /// One immediate per instruction, indexed by PC.
    pub imm_table: Vec<i32>,
    /// Entry PC (index into code/imm_table).
    pub entry_pc: u32,
    /// Number of unique opcodes used.
    pub num_opcodes: usize,
}

/// A contiguous region of memory to be loaded before execution.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MemoryRegion {
    pub addr: u32,
    pub data: Vec<u8>,
}

/// Complete initial memory state: a list of non-overlapping regions.
/// Everything outside these regions is zero.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MemorySnapshot {
    pub regions: Vec<MemoryRegion>,
}

/// VM register state at a single point in time.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegisterState {
    pub r0: u32,
    pub r1: u32,
    pub r2: u32,
}

/// One step of the execution trace.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TraceStep {
    /// PC before this step executed.
    pub pc: u32,
    /// Opcode byte executed.
    pub op: u8,
    /// Immediate value for this instruction.
    pub imm: i32,
    /// Register state *before* this step executed.
    pub regs_before: RegisterState,
    /// Register state *after* this step executed.
    pub regs_after: RegisterState,
    /// If this step read from memory: (addr, value).
    pub mem_read: Option<(u32, u32)>,
    /// If this step wrote to memory: (addr, value).
    pub mem_write: Option<(u32, u32)>,
}

/// Complete execution trace.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ExecutionTrace {
    pub steps: Vec<TraceStep>,
}

/// Everything a witness generator needs: program, initial state, and trace.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Canon3Witness {
    pub program: Canon3Program,
    pub initial_memory: MemorySnapshot,
    pub trace: ExecutionTrace,
    pub final_memory: MemorySnapshot,
}
