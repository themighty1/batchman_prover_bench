//! Lean canonical VM for 3 registers: r0 (result), r1 (source/base), r2 (source2).
//!
//! Canonical forms:
//!   R-type:  r0 = r1 op r2
//!   I-type:  r0 = r1 op imm
//!   Load:    r0 = mem[r1 + imm]
//!   Store:   mem[r1 + imm] = r0
//!   Branch:  branch r0, r1, target
//!   LUI:     r0 = imm << 12
//!
//! Cache management via absolute load/store (address baked into immediate):
//!   lw_abs0/1/2 addr   — rN = mem[addr]
//!   sw_abs0/1/2 addr   — mem[addr] = rN

use crate::rv32_vm::Memory;
use crate::rv32_isa_vm::MAILBOX_BASE;
use crate::canon3_types::*;

/// Mailbox slot for x1/ra (return address). Hardcoded: MAILBOX_BASE + 4.
const MAILBOX_RA: u32 = MAILBOX_BASE + 4;
/// Scratch slot A (used by sw_aligned decomposition to stash the byte value).
const SCRATCH_A: u32 = MAILBOX_BASE + 33 * 4;
use anyhow::{Result, bail};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Canon3Op {
    // R-type: r0 = r1 op r2
    Add = 0, Sub, Mul, Mulh, Mulhsu, Mulhu, Div, Divu, Rem, Remu,
    Sll, Srl, Sra, Slt, Sltu, Xor, Or, And,
    // I-type: r0 = r1 op imm
    Addi, Slti, Sltiu, Xori, Ori, Andi, Slli, Srli, Srai,
    // Loads: r0 = mem[r1 + imm]
    Lw,
    // Stores: mem[r1 + imm] = r0
    Sw,
    // Byte store via read-modify-write: write r0[7:0] to byte at r1+imm
    SwAligned,
    // Aligned word load + byte offset: r0=mem32[(r1+imm)&~3], r2=(r1+imm)&3
    LwAligned,
    // Byte extraction from r0 using r2 as byte index
    ByteSelR2,
    // Static byte extraction from r0
    ByteSel0, ByteSel1, ByteSel2, ByteSel3,
    // Sign-extend byte in r0
    Sext8,
    // Branches: branch r0, r1, target
    Beq, Bne, Blt, Bge, Bltu, Bgeu,
    // Upper immediate
    Lui,
    // Control flow
    Jal, JalCall, Jalr, JalrCall, Ret,
    JrTableIdx, JrComputed, Ecall, Halt,
    // Cache management (absolute address in immediate)
    LwAbs0, LwAbs1, LwAbs2,
    SwAbs0, SwAbs1, SwAbs2,
    // Fixed shifts — no immediate needed.
    // Shifting by a dynamic (immediate-encoded) amount is expensive in a boolean
    // circuit: it requires a barrel shifter (chain of muxes gated by each bit of
    // the shift amount).  Replacing slli/srli/srai with fixed-shift ops turns each
    // shift into a simple wire re-routing (constant propagation in the circuit),
    // eliminating the mux chain entirely.  Only 9 distinct shift amounts appear in
    // practice; we cover them with power-of-two building blocks (1, 4, 8, 16) plus
    // dedicated shift-by-31, composing the rare amounts from ≤6 ops at +0.3% steps.
    Sll1, Sll4, Sll8, Sll16, Sll31,
    Srl1, Srl4, Srl8, Srl16, Srl31,
    Sra1, Sra4, Sra8, Sra16, Sra31,
    // sw_aligned decomposition — break the largest-gate-count op into simpler
    // sub-ops.  sw_aligned is a read-modify-write (load word, insert byte, store
    // word) with dynamic byte-position indexing.  Decomposing it lets each sub-op
    // have a smaller gate footprint in a boolean circuit:
    //   sw_abs0 SCRATCH_A   — save the byte value
    //   lw_aligned imm      — load existing word, get byte offset in r2
    //   byte_ins_r2         — insert saved byte into word at position r2
    //   sw_waligned imm     — write modified word back to aligned address
    ByteInsR2,   // r0 = insert SCRATCH_A[7:0] at byte position r2 into r0
    SwWaligned,  // mem[(r1+imm) & ~3] = r0 (word-aligned store)
    // sentinel
    NumOps,
}

fn name_to_op(name: &str) -> Option<Canon3Op> {
    use Canon3Op::*;
    Some(match name {
        "add" => Add, "sub" => Sub, "mul" => Mul,
        "mulh" => Mulh, "mulhsu" => Mulhsu, "mulhu" => Mulhu,
        "div" => Div, "divu" => Divu, "rem" => Rem, "remu" => Remu,
        "sll" => Sll, "srl" => Srl, "sra" => Sra,
        "slt" => Slt, "sltu" => Sltu,
        "xor" => Xor, "or" => Or, "and" => And,
        "addi" => Addi, "slti" => Slti, "sltiu" => Sltiu,
        "xori" => Xori, "ori" => Ori, "andi" => Andi,
        "slli" => Slli, "srli" => Srli, "srai" => Srai,
        "lw" => Lw,
        "sw" => Sw,
        "sw_aligned" => SwAligned,
        "lw_aligned" => LwAligned,
        "byte_sel_r2" => ByteSelR2,
        "byte_sel0" => ByteSel0, "byte_sel1" => ByteSel1,
        "byte_sel2" => ByteSel2, "byte_sel3" => ByteSel3,
        "sext8" => Sext8,
        "beq" => Beq, "bne" => Bne, "blt" => Blt, "bge" => Bge,
        "bltu" => Bltu, "bgeu" => Bgeu,
        "lui" => Lui,
        "jal" => Jal, "jal_call" => JalCall,
        "jalr" => Jalr, "jalr_call" => JalrCall,
        "ret" => Ret,
        "jr_table_idx" => JrTableIdx, "jr_computed" => JrComputed,
        "ecall" => Ecall, "halt" => Halt,
        "lw_abs0" => LwAbs0, "lw_abs1" => LwAbs1, "lw_abs2" => LwAbs2,
        "sw_abs0" => SwAbs0, "sw_abs1" => SwAbs1, "sw_abs2" => SwAbs2,
        "sll1" => Sll1, "sll4" => Sll4, "sll8" => Sll8, "sll16" => Sll16, "sll31" => Sll31,
        "srl1" => Srl1, "srl4" => Srl4, "srl8" => Srl8, "srl16" => Srl16, "srl31" => Srl31,
        "sra1" => Sra1, "sra4" => Sra4, "sra8" => Sra8, "sra16" => Sra16, "sra31" => Sra31,
        "byte_ins_r2" => ByteInsR2,
        "sw_waligned" => SwWaligned,
        _ => return None,
    })
}

/// Canonical 3-register VM: r0, r1, r2, pc, memory.
pub struct Canon3Vm {
    pub r0: u32,
    pub r1: u32,
    pub r2: u32,
    pub memory: Memory,
    pub code: Vec<u8>,
    pub imm_table: Vec<i32>,
    pub pc: u32,
    pub steps: u64,
    pub halted: bool,
}

impl Canon3Vm {
    pub fn new(code: Vec<u8>, imm_table: Vec<i32>) -> Self {
        Canon3Vm {
            r0: 0, r1: 0, r2: 0,
            memory: Memory::new(),
            code, imm_table,
            pc: 0, steps: 0, halted: false,
        }
    }

    pub fn remap_code(opcode_names: &[String], code_u8: &[u8]) -> Result<Vec<u8>> {
        let mut remap = vec![Canon3Op::NumOps; 256];
        for (id, name) in opcode_names.iter().enumerate() {
            match name_to_op(name) {
                Some(op) => remap[id] = op,
                None => bail!("Unknown canon3 opcode: '{}' (id={})", name, id),
            }
        }
        Ok(code_u8.iter().map(|&id| remap[id as usize] as u8).collect())
    }

    pub fn execute(&mut self) -> Result<()> {
        use Canon3Op::*;
        let max_steps: u64 = 100_000_000;

        let trace_file = std::env::var("TRACE_FILE").ok();
        let mut trace_writer: Option<std::io::BufWriter<std::fs::File>> = trace_file.map(|path| {
            std::io::BufWriter::new(std::fs::File::create(&path).expect("cannot create TRACE_FILE"))
        });

        while !self.halted && self.steps < max_steps {
            let cur_pc = self.pc;
            let op = self.code[cur_pc as usize];
            let imm = self.imm_table[cur_pc as usize];
            self.steps += 1;

            if let Some(ref mut w) = trace_writer {
                use std::io::Write;
                writeln!(w, "[{}] op={} imm={} | r0={} r1={} r2={}",
                    cur_pc, op, imm, self.r0, self.r1, self.r2).unwrap();
            }

            match op {
                // --- Cache management (absolute address in immediate) ---
                x if x == LwAbs0 as u8 => { self.r0 = self.memory.read_u32(imm as u32); }
                x if x == LwAbs1 as u8 => { self.r1 = self.memory.read_u32(imm as u32); }
                x if x == LwAbs2 as u8 => { self.r2 = self.memory.read_u32(imm as u32); }
                x if x == SwAbs0 as u8 => { self.memory.write_u32(imm as u32, self.r0); }
                x if x == SwAbs1 as u8 => { self.memory.write_u32(imm as u32, self.r1); }
                x if x == SwAbs2 as u8 => { self.memory.write_u32(imm as u32, self.r2); }

                // --- R-type: r0 = r1 op r2 ---
                x if x == Add as u8  => { self.r0 = self.r1.wrapping_add(self.r2); }
                x if x == Sub as u8  => { self.r0 = self.r1.wrapping_sub(self.r2); }
                x if x == Mul as u8  => { self.r0 = self.r1.wrapping_mul(self.r2); }
                x if x == Mulh as u8 => {
                    self.r0 = ((self.r1 as i32 as i64).wrapping_mul(self.r2 as i32 as i64) >> 32) as u32;
                }
                x if x == Mulhsu as u8 => {
                    self.r0 = ((self.r1 as i32 as i64).wrapping_mul(self.r2 as u64 as i64) >> 32) as u32;
                }
                x if x == Mulhu as u8 => {
                    self.r0 = ((self.r1 as u64).wrapping_mul(self.r2 as u64) >> 32) as u32;
                }
                x if x == Div as u8 => {
                    let (a, b) = (self.r1, self.r2);
                    self.r0 = if b == 0 { u32::MAX }
                        else if a == 0x80000000 && b == 0xFFFFFFFF { a }
                        else { ((a as i32).wrapping_div(b as i32)) as u32 };
                }
                x if x == Divu as u8 => {
                    self.r0 = if self.r2 == 0 { u32::MAX } else { self.r1 / self.r2 };
                }
                x if x == Rem as u8 => {
                    let (a, b) = (self.r1, self.r2);
                    self.r0 = if b == 0 { a }
                        else if a == 0x80000000 && b == 0xFFFFFFFF { 0 }
                        else { ((a as i32).wrapping_rem(b as i32)) as u32 };
                }
                x if x == Remu as u8 => {
                    self.r0 = if self.r2 == 0 { self.r1 } else { self.r1 % self.r2 };
                }
                x if x == Sll as u8  => { self.r0 = self.r1 << (self.r2 & 0x1F); }
                x if x == Srl as u8  => { self.r0 = self.r1 >> (self.r2 & 0x1F); }
                x if x == Sra as u8  => { self.r0 = ((self.r1 as i32) >> (self.r2 & 0x1F)) as u32; }
                x if x == Slt as u8  => { self.r0 = if (self.r1 as i32) < (self.r2 as i32) { 1 } else { 0 }; }
                x if x == Sltu as u8 => { self.r0 = if self.r1 < self.r2 { 1 } else { 0 }; }
                x if x == Xor as u8  => { self.r0 = self.r1 ^ self.r2; }
                x if x == Or as u8   => { self.r0 = self.r1 | self.r2; }
                x if x == And as u8  => { self.r0 = self.r1 & self.r2; }

                // --- I-type: r0 = r1 op imm ---
                x if x == Addi as u8  => { self.r0 = self.r1.wrapping_add(imm as u32); }
                x if x == Slti as u8  => { self.r0 = if (self.r1 as i32) < imm { 1 } else { 0 }; }
                x if x == Sltiu as u8 => { self.r0 = if self.r1 < (imm as u32) { 1 } else { 0 }; }
                x if x == Xori as u8  => { self.r0 = self.r1 ^ (imm as u32); }
                x if x == Ori as u8   => { self.r0 = self.r1 | (imm as u32); }
                x if x == Andi as u8  => { self.r0 = self.r1 & (imm as u32); }
                x if x == Slli as u8  => { self.r0 = self.r1 << (imm as u32 & 0x1F); }
                x if x == Srli as u8  => { self.r0 = self.r1 >> (imm as u32 & 0x1F); }
                x if x == Srai as u8  => { self.r0 = ((self.r1 as i32) >> (imm as u32 & 0x1F)) as u32; }

                // Fixed shifts: first in a chain reads r1, rest chain through r0.
                // All write r0; the compiler emits the first shift normally (r0 = r1 << N)
                // and subsequent shifts as (r0 = r0 << N) by sharing the same semantic:
                // single-op shifts behave like I-type (r0 = r1 op N), multi-op chains
                // use r0 as both source and destination after the first step.
                // We define them as r0 = r0 op N since the compiler inserts a
                // copy from r1→r0 before a chain via the first shift reading r1.
                x if x == Sll1  as u8 => { self.r0 = self.r0 << 1; }
                x if x == Sll4  as u8 => { self.r0 = self.r0 << 4; }
                x if x == Sll8  as u8 => { self.r0 = self.r0 << 8; }
                x if x == Sll16 as u8 => { self.r0 = self.r0 << 16; }
                x if x == Sll31 as u8 => { self.r0 = self.r0 << 31; }
                x if x == Srl1  as u8 => { self.r0 = self.r0 >> 1; }
                x if x == Srl4  as u8 => { self.r0 = self.r0 >> 4; }
                x if x == Srl8  as u8 => { self.r0 = self.r0 >> 8; }
                x if x == Srl16 as u8 => { self.r0 = self.r0 >> 16; }
                x if x == Srl31 as u8 => { self.r0 = self.r0 >> 31; }
                x if x == Sra1  as u8 => { self.r0 = ((self.r0 as i32) >> 1) as u32; }
                x if x == Sra4  as u8 => { self.r0 = ((self.r0 as i32) >> 4) as u32; }
                x if x == Sra8  as u8 => { self.r0 = ((self.r0 as i32) >> 8) as u32; }
                x if x == Sra16 as u8 => { self.r0 = ((self.r0 as i32) >> 16) as u32; }
                x if x == Sra31 as u8 => { self.r0 = ((self.r0 as i32) >> 31) as u32; }

                // --- Loads: r0 = mem[r1 + imm] ---
                x if x == Lw as u8  => { let a = self.r1.wrapping_add(imm as u32); self.r0 = self.memory.read_u32(a); }
                x if x == LwAligned as u8 => {
                    let a = self.r1.wrapping_add(imm as u32);
                    self.r0 = self.memory.read_u32(a & !3);
                    self.r2 = a & 3;
                }
                x if x == ByteSelR2 as u8 => { self.r0 = (self.r0 >> (self.r2 * 8)) & 0xFF; }
                x if x == ByteInsR2 as u8 => {
                    let byte_val = self.memory.read_u32(SCRATCH_A) & 0xFF;
                    let shift = self.r2 * 8;
                    self.r0 = (self.r0 & !(0xFF << shift)) | (byte_val << shift);
                }
                x if x == ByteSel0 as u8  => { self.r0 = self.r0 & 0xFF; }
                x if x == ByteSel1 as u8  => { self.r0 = (self.r0 >> 8) & 0xFF; }
                x if x == ByteSel2 as u8  => { self.r0 = (self.r0 >> 16) & 0xFF; }
                x if x == ByteSel3 as u8  => { self.r0 = (self.r0 >> 24) & 0xFF; }
                x if x == Sext8 as u8     => { self.r0 = self.r0 as u8 as i8 as i32 as u32; }

                // --- Stores: mem[r1 + imm] = r0 ---
                x if x == Sw as u8 => { let a = self.r1.wrapping_add(imm as u32); self.memory.write_u32(a, self.r0); }
                x if x == SwAligned as u8 => {
                    let a = self.r1.wrapping_add(imm as u32);
                    let shift = (a & 3) * 8;
                    let cell_addr = a & !3;
                    let cell = self.memory.read_u32(cell_addr);
                    let cell = (cell & !(0xFF << shift)) | ((self.r0 & 0xFF) << shift);
                    self.memory.write_u32(cell_addr, cell);
                }
                x if x == SwWaligned as u8 => {
                    let a = self.r1.wrapping_add(imm as u32);
                    self.memory.write_u32(a & !3, self.r0);
                }

                // --- Branches: compare r0, r1 ---
                x if x == Beq as u8  => { if self.r0 == self.r1 { self.pc = imm as u32; } }
                x if x == Bne as u8  => { if self.r0 != self.r1 { self.pc = imm as u32; } }
                x if x == Blt as u8  => { if (self.r0 as i32) < (self.r1 as i32) { self.pc = imm as u32; } }
                x if x == Bge as u8  => { if (self.r0 as i32) >= (self.r1 as i32) { self.pc = imm as u32; } }
                x if x == Bltu as u8 => { if self.r0 < self.r1 { self.pc = imm as u32; } }
                x if x == Bgeu as u8 => { if self.r0 >= self.r1 { self.pc = imm as u32; } }

                // --- Upper immediate ---
                x if x == Lui as u8 => { self.r0 = (imm as u32) << 12; }

                // --- Control flow ---
                x if x == Jal as u8 => {
                    if imm as u32 == cur_pc { self.halted = true; }
                    else { self.pc = imm as u32; }
                }
                x if x == JalCall as u8 => {
                    self.memory.write_u32(MAILBOX_RA, cur_pc + 1);
                    self.pc = imm as u32;
                }
                x if x == Jalr as u8 => { self.pc = self.r0; }
                x if x == JalrCall as u8 => {
                    let target = self.r0;
                    self.memory.write_u32(MAILBOX_RA, cur_pc + 1);
                    self.pc = target;
                }
                x if x == Ret as u8 => {
                    if self.r0 == 0 { self.halted = true; }
                    else { self.pc = self.r0; }
                }
                x if x == JrTableIdx as u8 => { self.pc = self.r0; }
                x if x == JrComputed as u8 => { self.pc = imm as u32; }
                x if x == Ecall as u8 => { self.halted = true; }
                x if x == Halt as u8 => { self.halted = true; }

                _ => { bail!("Unknown canon3 opcode {} at pc={}", op, cur_pc); }
            }

            // Post-increment: advance PC if the op didn't change it.
            if self.pc == cur_pc {
                self.pc += 1;
            }

            if self.steps % 1_000_000 == 0 {
                eprintln!("  [step {:>8}] pc={}", self.steps, cur_pc);
            }
        }

        if self.steps >= max_steps && !self.halted {
            bail!("Execution limit reached ({} steps)", max_steps);
        }
        Ok(())
    }

    /// Snapshot the current memory state as a list of non-overlapping regions.
    pub fn snapshot_memory(&self) -> MemorySnapshot {
        self.memory.snapshot()
    }

    /// Build a Canon3Program from this VM's code and imm_table.
    pub fn program(&self, entry_pc: u32) -> Canon3Program {
        let num_opcodes = {
            let mut seen = [false; 256];
            for &op in &self.code { seen[op as usize] = true; }
            seen.iter().filter(|&&s| s).count()
        };
        Canon3Program {
            code: self.code.clone(),
            imm_table: self.imm_table.clone(),
            entry_pc,
            num_opcodes,
        }
    }

    /// Execute and collect a full trace. Returns the trace on success.
    pub fn execute_with_trace(&mut self) -> Result<ExecutionTrace> {
        use Canon3Op::*;
        let max_steps: u64 = 100_000_000;
        let mut trace = Vec::new();

        while !self.halted && self.steps < max_steps {
            let cur_pc = self.pc;
            let op = self.code[cur_pc as usize];
            let imm = self.imm_table[cur_pc as usize];
            let regs_before = RegisterState { r0: self.r0, r1: self.r1, r2: self.r2 };

            self.steps += 1;

            let mut mem_read: Option<(u32, u32)> = None;
            let mut mem_write: Option<(u32, u32)> = None;

            match op {
                // --- Cache management (absolute address in immediate) ---
                x if x == LwAbs0 as u8 => { self.r0 = self.memory.read_u32(imm as u32); mem_read = Some((imm as u32, self.r0)); }
                x if x == LwAbs1 as u8 => { self.r1 = self.memory.read_u32(imm as u32); mem_read = Some((imm as u32, self.r1)); }
                x if x == LwAbs2 as u8 => { self.r2 = self.memory.read_u32(imm as u32); mem_read = Some((imm as u32, self.r2)); }
                x if x == SwAbs0 as u8 => { self.memory.write_u32(imm as u32, self.r0); mem_write = Some((imm as u32, self.r0)); }
                x if x == SwAbs1 as u8 => { self.memory.write_u32(imm as u32, self.r1); mem_write = Some((imm as u32, self.r1)); }
                x if x == SwAbs2 as u8 => { self.memory.write_u32(imm as u32, self.r2); mem_write = Some((imm as u32, self.r2)); }

                // --- R-type: r0 = r1 op r2 ---
                x if x == Add as u8  => { self.r0 = self.r1.wrapping_add(self.r2); }
                x if x == Sub as u8  => { self.r0 = self.r1.wrapping_sub(self.r2); }
                x if x == Mul as u8  => { self.r0 = self.r1.wrapping_mul(self.r2); }
                x if x == Mulh as u8 => {
                    self.r0 = ((self.r1 as i32 as i64).wrapping_mul(self.r2 as i32 as i64) >> 32) as u32;
                }
                x if x == Mulhsu as u8 => {
                    self.r0 = ((self.r1 as i32 as i64).wrapping_mul(self.r2 as u64 as i64) >> 32) as u32;
                }
                x if x == Mulhu as u8 => {
                    self.r0 = ((self.r1 as u64).wrapping_mul(self.r2 as u64) >> 32) as u32;
                }
                x if x == Div as u8 => {
                    let (a, b) = (self.r1, self.r2);
                    self.r0 = if b == 0 { u32::MAX }
                        else if a == 0x80000000 && b == 0xFFFFFFFF { a }
                        else { ((a as i32).wrapping_div(b as i32)) as u32 };
                }
                x if x == Divu as u8 => {
                    self.r0 = if self.r2 == 0 { u32::MAX } else { self.r1 / self.r2 };
                }
                x if x == Rem as u8 => {
                    let (a, b) = (self.r1, self.r2);
                    self.r0 = if b == 0 { a }
                        else if a == 0x80000000 && b == 0xFFFFFFFF { 0 }
                        else { ((a as i32).wrapping_rem(b as i32)) as u32 };
                }
                x if x == Remu as u8 => {
                    self.r0 = if self.r2 == 0 { self.r1 } else { self.r1 % self.r2 };
                }
                x if x == Sll as u8  => { self.r0 = self.r1 << (self.r2 & 0x1F); }
                x if x == Srl as u8  => { self.r0 = self.r1 >> (self.r2 & 0x1F); }
                x if x == Sra as u8  => { self.r0 = ((self.r1 as i32) >> (self.r2 & 0x1F)) as u32; }
                x if x == Slt as u8  => { self.r0 = if (self.r1 as i32) < (self.r2 as i32) { 1 } else { 0 }; }
                x if x == Sltu as u8 => { self.r0 = if self.r1 < self.r2 { 1 } else { 0 }; }
                x if x == Xor as u8  => { self.r0 = self.r1 ^ self.r2; }
                x if x == Or as u8   => { self.r0 = self.r1 | self.r2; }
                x if x == And as u8  => { self.r0 = self.r1 & self.r2; }

                // --- I-type: r0 = r1 op imm ---
                x if x == Addi as u8  => { self.r0 = self.r1.wrapping_add(imm as u32); }
                x if x == Slti as u8  => { self.r0 = if (self.r1 as i32) < imm { 1 } else { 0 }; }
                x if x == Sltiu as u8 => { self.r0 = if self.r1 < (imm as u32) { 1 } else { 0 }; }
                x if x == Xori as u8  => { self.r0 = self.r1 ^ (imm as u32); }
                x if x == Ori as u8   => { self.r0 = self.r1 | (imm as u32); }
                x if x == Andi as u8  => { self.r0 = self.r1 & (imm as u32); }
                x if x == Slli as u8  => { self.r0 = self.r1 << (imm as u32 & 0x1F); }
                x if x == Srli as u8  => { self.r0 = self.r1 >> (imm as u32 & 0x1F); }
                x if x == Srai as u8  => { self.r0 = ((self.r1 as i32) >> (imm as u32 & 0x1F)) as u32; }

                // Fixed shifts
                x if x == Sll1  as u8 => { self.r0 = self.r0 << 1; }
                x if x == Sll4  as u8 => { self.r0 = self.r0 << 4; }
                x if x == Sll8  as u8 => { self.r0 = self.r0 << 8; }
                x if x == Sll16 as u8 => { self.r0 = self.r0 << 16; }
                x if x == Sll31 as u8 => { self.r0 = self.r0 << 31; }
                x if x == Srl1  as u8 => { self.r0 = self.r0 >> 1; }
                x if x == Srl4  as u8 => { self.r0 = self.r0 >> 4; }
                x if x == Srl8  as u8 => { self.r0 = self.r0 >> 8; }
                x if x == Srl16 as u8 => { self.r0 = self.r0 >> 16; }
                x if x == Srl31 as u8 => { self.r0 = self.r0 >> 31; }
                x if x == Sra1  as u8 => { self.r0 = ((self.r0 as i32) >> 1) as u32; }
                x if x == Sra4  as u8 => { self.r0 = ((self.r0 as i32) >> 4) as u32; }
                x if x == Sra8  as u8 => { self.r0 = ((self.r0 as i32) >> 8) as u32; }
                x if x == Sra16 as u8 => { self.r0 = ((self.r0 as i32) >> 16) as u32; }
                x if x == Sra31 as u8 => { self.r0 = ((self.r0 as i32) >> 31) as u32; }

                // --- Loads: r0 = mem[r1 + imm] ---
                x if x == Lw as u8  => {
                    let a = self.r1.wrapping_add(imm as u32);
                    self.r0 = self.memory.read_u32(a);
                    mem_read = Some((a, self.r0));
                }
                x if x == LwAligned as u8 => {
                    let a = self.r1.wrapping_add(imm as u32);
                    let aligned = a & !3;
                    self.r0 = self.memory.read_u32(aligned);
                    self.r2 = a & 3;
                    mem_read = Some((aligned, self.r0));
                }
                x if x == ByteSelR2 as u8 => { self.r0 = (self.r0 >> (self.r2 * 8)) & 0xFF; }
                x if x == ByteInsR2 as u8 => {
                    let raw = self.memory.read_u32(SCRATCH_A);
                    let byte_val = raw & 0xFF;
                    let shift = self.r2 * 8;
                    self.r0 = (self.r0 & !(0xFF << shift)) | (byte_val << shift);
                    mem_read = Some((SCRATCH_A, raw));
                }
                x if x == ByteSel0 as u8  => { self.r0 = self.r0 & 0xFF; }
                x if x == ByteSel1 as u8  => { self.r0 = (self.r0 >> 8) & 0xFF; }
                x if x == ByteSel2 as u8  => { self.r0 = (self.r0 >> 16) & 0xFF; }
                x if x == ByteSel3 as u8  => { self.r0 = (self.r0 >> 24) & 0xFF; }
                x if x == Sext8 as u8     => { self.r0 = self.r0 as u8 as i8 as i32 as u32; }

                // --- Stores: mem[r1 + imm] = r0 ---
                x if x == Sw as u8 => {
                    let a = self.r1.wrapping_add(imm as u32);
                    self.memory.write_u32(a, self.r0);
                    mem_write = Some((a, self.r0));
                }
                x if x == SwAligned as u8 => {
                    let a = self.r1.wrapping_add(imm as u32);
                    let shift = (a & 3) * 8;
                    let cell_addr = a & !3;
                    let cell = self.memory.read_u32(cell_addr);
                    let cell = (cell & !(0xFF << shift)) | ((self.r0 & 0xFF) << shift);
                    self.memory.write_u32(cell_addr, cell);
                    mem_read = Some((cell_addr, self.memory.read_u32(cell_addr)));
                    mem_write = Some((cell_addr, cell));
                }
                x if x == SwWaligned as u8 => {
                    let a = self.r1.wrapping_add(imm as u32);
                    let aligned = a & !3;
                    self.memory.write_u32(aligned, self.r0);
                    mem_write = Some((aligned, self.r0));
                }

                // --- Branches: compare r0, r1 ---
                x if x == Beq as u8  => { if self.r0 == self.r1 { self.pc = imm as u32; } }
                x if x == Bne as u8  => { if self.r0 != self.r1 { self.pc = imm as u32; } }
                x if x == Blt as u8  => { if (self.r0 as i32) < (self.r1 as i32) { self.pc = imm as u32; } }
                x if x == Bge as u8  => { if (self.r0 as i32) >= (self.r1 as i32) { self.pc = imm as u32; } }
                x if x == Bltu as u8 => { if self.r0 < self.r1 { self.pc = imm as u32; } }
                x if x == Bgeu as u8 => { if self.r0 >= self.r1 { self.pc = imm as u32; } }

                // --- Upper immediate ---
                x if x == Lui as u8 => { self.r0 = (imm as u32) << 12; }

                // --- Control flow ---
                x if x == Jal as u8 => {
                    if imm as u32 == cur_pc { self.halted = true; }
                    else { self.pc = imm as u32; }
                }
                x if x == JalCall as u8 => {
                    self.memory.write_u32(MAILBOX_RA, cur_pc + 1);
                    mem_write = Some((MAILBOX_RA, cur_pc + 1));
                    self.pc = imm as u32;
                }
                x if x == Jalr as u8 => { self.pc = self.r0; }
                x if x == JalrCall as u8 => {
                    let target = self.r0;
                    self.memory.write_u32(MAILBOX_RA, cur_pc + 1);
                    mem_write = Some((MAILBOX_RA, cur_pc + 1));
                    self.pc = target;
                }
                x if x == Ret as u8 => {
                    if self.r0 == 0 { self.halted = true; }
                    else { self.pc = self.r0; }
                }
                x if x == JrTableIdx as u8 => { self.pc = self.r0; }
                x if x == JrComputed as u8 => { self.pc = imm as u32; }
                x if x == Ecall as u8 => { self.halted = true; }
                x if x == Halt as u8 => { self.halted = true; }

                _ => { bail!("Unknown canon3 opcode {} at pc={}", op, cur_pc); }
            }

            // Post-increment: advance PC if the op didn't change it.
            if self.pc == cur_pc {
                self.pc += 1;
            }

            let regs_after = RegisterState { r0: self.r0, r1: self.r1, r2: self.r2 };

            trace.push(TraceStep {
                pc: cur_pc,
                op,
                imm,
                regs_before,
                regs_after,
                mem_read,
                mem_write,
            });

            if self.steps % 1_000_000 == 0 {
                eprintln!("  [step {:>8}] pc={}", self.steps, cur_pc);
            }
        }

        if self.steps >= max_steps && !self.halted {
            bail!("Execution limit reached ({} steps)", max_steps);
        }
        Ok(ExecutionTrace { steps: trace })
    }
}
