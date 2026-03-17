//! Lean canonical VM: 2 registers, no opcode table indirection.
//!
//! Every canonical opcode has fixed register positions (r0, r1), so
//! the dispatch reads the opcode byte directly — no OpcodeInfo lookup needed.
//! The only runtime tables are code[pc] → u8 opcode and imm[pc] → i32 immediate.

use crate::rv32_vm::Memory;
use crate::rv32_isa_vm::MAILBOX_BASE;
use anyhow::{Result, bail};

/// Opcode IDs for the canonical ISA.
/// Assigned sequentially — must match the compiler's opcode_map order.
/// Instead of relying on order, we resolve by name at load time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CanonOp {
    Add = 0, Sub, Mul, Mulh, Mulhsu, Mulhu, Div, Divu, Rem, Remu,
    Sll, Srl, Sra, Slt, Sltu, Xor, Or, And,
    Addi, Slti, Sltiu, Xori, Ori, Andi, Slli, Srli, Srai,
    Lw, Lb, Lh, Lbu, Lhu,
    Sw, Sb, Sh,
    Beq, Bne, Blt, Bge, Bltu, Bgeu,
    Lui,
    Jal, JalCall, Jalr, JalrCall, Ret,
    Swap, LoadReg, StoreReg,
    JrTableIdx, JrComputed, Ecall, Halt,
    // sentinel
    NumOps,
}

/// Map from compiler opcode name → CanonOp.
fn name_to_canon_op(name: &str) -> Option<CanonOp> {
    use CanonOp::*;
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
        "lw" => Lw, "lb" => Lb, "lh" => Lh, "lbu" => Lbu, "lhu" => Lhu,
        "sw" => Sw, "sb" => Sb, "sh" => Sh,
        "beq" => Beq, "bne" => Bne, "blt" => Blt, "bge" => Bge,
        "bltu" => Bltu, "bgeu" => Bgeu,
        "lui" => Lui,
        "jal" => Jal, "jal_call" => JalCall,
        "jalr" => Jalr, "jalr_call" => JalrCall,
        "ret" => Ret,
        "swap" => Swap, "load_reg" => LoadReg, "store_reg" => StoreReg,
        "jr_table_idx" => JrTableIdx, "jr_computed" => JrComputed,
        "ecall" => Ecall, "halt" => Halt,
        _ => return None,
    })
}

/// Canonical VM state: r0, r1, pc, memory. That's it.
pub struct CanonVm {
    pub r0: u32,
    pub r1: u32,
    pub memory: Memory,
    pub code: Vec<u8>,        // code[pc] = remapped CanonOp as u8
    pub imm_table: Vec<i32>,  // imm_table[pc] = immediate
    pub pc: u32,
    pub steps: u64,
    pub halted: bool,
}

impl CanonVm {
    pub fn new(code: Vec<u8>, imm_table: Vec<i32>) -> Self {
        CanonVm {
            r0: 0,
            r1: 0,
            memory: Memory::new(),
            code,
            imm_table,
            pc: 0,
            steps: 0,
            halted: false,
        }
    }

    /// Build a remapping table from compiler opcode IDs to CanonOp values.
    /// Returns remapped code where each byte is a CanonOp discriminant.
    pub fn remap_code(
        opcode_names: &[String],
        code_u8: &[u8],
    ) -> Result<Vec<u8>> {
        // Build compiler_id → CanonOp
        let mut remap = vec![CanonOp::NumOps; 256];
        for (id, name) in opcode_names.iter().enumerate() {
            match name_to_canon_op(name) {
                Some(op) => remap[id] = op,
                None => bail!("Unknown canonical opcode: '{}' (id={})", name, id),
            }
        }
        // Remap code
        let remapped: Vec<u8> = code_u8.iter().map(|&id| remap[id as usize] as u8).collect();
        Ok(remapped)
    }

    /// Execute until halt.
    pub fn execute(&mut self) -> Result<()> {
        use CanonOp::*;
        let max_steps: u64 = 100_000_000;

        let trace_file = std::env::var("TRACE_FILE").ok();
        let mut trace_writer: Option<std::io::BufWriter<std::fs::File>> = trace_file.map(|path| {
            std::io::BufWriter::new(std::fs::File::create(&path).expect("cannot create TRACE_FILE"))
        });

        while !self.halted && self.steps < max_steps {
            let cur_pc = self.pc;
            let op = self.code[cur_pc as usize];
            let imm = self.imm_table[cur_pc as usize];
            self.pc += 1;
            self.steps += 1;

            if let Some(ref mut w) = trace_writer {
                use std::io::Write;
                writeln!(w, "[{}] op={} imm={} | r0={} r1={}", cur_pc, op, imm, self.r0, self.r1).unwrap();
            }

            // SAFETY: op is a u8, and we handle all CanonOp variants + default.
            // This is the hot loop — avoid string matching entirely.
            match op {
                // --- Cache management ---
                x if x == LoadReg as u8 => {
                    self.r0 = self.memory.read_u32(MAILBOX_BASE + (imm as u32) * 4);
                }
                x if x == StoreReg as u8 => {
                    self.memory.write_u32(MAILBOX_BASE + (imm as u32) * 4, self.r0);
                }
                x if x == Swap as u8 => {
                    let tmp = self.r0;
                    self.r0 = self.r1;
                    self.r1 = tmp;
                }

                // --- R-type: r0 = r0 op r1 ---
                x if x == Add as u8 => { self.r0 = self.r0.wrapping_add(self.r1); }
                x if x == Sub as u8 => { self.r0 = self.r0.wrapping_sub(self.r1); }
                x if x == Mul as u8 => { self.r0 = self.r0.wrapping_mul(self.r1); }
                x if x == Mulh as u8 => {
                    self.r0 = ((self.r0 as i32 as i64).wrapping_mul(self.r1 as i32 as i64) >> 32) as u32;
                }
                x if x == Mulhsu as u8 => {
                    self.r0 = ((self.r0 as i32 as i64).wrapping_mul(self.r1 as u64 as i64) >> 32) as u32;
                }
                x if x == Mulhu as u8 => {
                    self.r0 = ((self.r0 as u64).wrapping_mul(self.r1 as u64) >> 32) as u32;
                }
                x if x == Div as u8 => {
                    let a = self.r0; let b = self.r1;
                    self.r0 = if b == 0 { u32::MAX }
                        else if a == 0x80000000 && b == 0xFFFFFFFF { a }
                        else { ((a as i32).wrapping_div(b as i32)) as u32 };
                }
                x if x == Divu as u8 => {
                    self.r0 = if self.r1 == 0 { u32::MAX } else { self.r0 / self.r1 };
                }
                x if x == Rem as u8 => {
                    let a = self.r0; let b = self.r1;
                    self.r0 = if b == 0 { a }
                        else if a == 0x80000000 && b == 0xFFFFFFFF { 0 }
                        else { ((a as i32).wrapping_rem(b as i32)) as u32 };
                }
                x if x == Remu as u8 => {
                    self.r0 = if self.r1 == 0 { self.r0 } else { self.r0 % self.r1 };
                }
                x if x == Sll as u8 => { self.r0 = self.r0 << (self.r1 & 0x1F); }
                x if x == Srl as u8 => { self.r0 = self.r0 >> (self.r1 & 0x1F); }
                x if x == Sra as u8 => { self.r0 = ((self.r0 as i32) >> (self.r1 & 0x1F)) as u32; }
                x if x == Slt as u8 => { self.r0 = if (self.r0 as i32) < (self.r1 as i32) { 1 } else { 0 }; }
                x if x == Sltu as u8 => { self.r0 = if self.r0 < self.r1 { 1 } else { 0 }; }
                x if x == Xor as u8 => { self.r0 = self.r0 ^ self.r1; }
                x if x == Or as u8  => { self.r0 = self.r0 | self.r1; }
                x if x == And as u8 => { self.r0 = self.r0 & self.r1; }

                // --- I-type: r0 = r0 op imm ---
                x if x == Addi as u8 => { self.r0 = self.r0.wrapping_add(imm as u32); }
                x if x == Slti as u8 => { self.r0 = if (self.r0 as i32) < imm { 1 } else { 0 }; }
                x if x == Sltiu as u8 => { self.r0 = if self.r0 < (imm as u32) { 1 } else { 0 }; }
                x if x == Xori as u8 => { self.r0 = self.r0 ^ (imm as u32); }
                x if x == Ori as u8  => { self.r0 = self.r0 | (imm as u32); }
                x if x == Andi as u8 => { self.r0 = self.r0 & (imm as u32); }
                x if x == Slli as u8 => { self.r0 = self.r0 << (imm as u32 & 0x1F); }
                x if x == Srli as u8 => { self.r0 = self.r0 >> (imm as u32 & 0x1F); }
                x if x == Srai as u8 => { self.r0 = ((self.r0 as i32) >> (imm as u32 & 0x1F)) as u32; }

                // --- Loads: r0 = mem[r1 + imm] ---
                x if x == Lw as u8  => { let a = self.r1.wrapping_add(imm as u32); self.r0 = self.memory.read_u32(a); }
                x if x == Lb as u8  => { let a = self.r1.wrapping_add(imm as u32); self.r0 = self.memory.read_u8(a) as i8 as i32 as u32; }
                x if x == Lh as u8  => { let a = self.r1.wrapping_add(imm as u32); self.r0 = self.memory.read_u16(a) as i16 as i32 as u32; }
                x if x == Lbu as u8 => { let a = self.r1.wrapping_add(imm as u32); self.r0 = self.memory.read_u8(a) as u32; }
                x if x == Lhu as u8 => { let a = self.r1.wrapping_add(imm as u32); self.r0 = self.memory.read_u16(a) as u32; }

                // --- Stores: mem[r1 + imm] = r0 ---
                x if x == Sw as u8 => { let a = self.r1.wrapping_add(imm as u32); self.memory.write_u32(a, self.r0); }
                x if x == Sb as u8 => { let a = self.r1.wrapping_add(imm as u32); self.memory.write_u8(a, self.r0 as u8); }
                x if x == Sh as u8 => { let a = self.r1.wrapping_add(imm as u32); self.memory.write_u16(a, self.r0 as u16); }

                // --- Branches: compare r0, r1, jump to imm ---
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
                    let target = imm as u32;
                    if target == cur_pc {
                        self.halted = true;
                    } else {
                        self.pc = target;
                    }
                }
                x if x == JalCall as u8 => {
                    let return_addr = self.pc;
                    // Write return addr to regfile[1] (x1/ra)
                    self.memory.write_u32(MAILBOX_BASE + 1 * 4, return_addr);
                    self.pc = imm as u32;
                }
                x if x == Jalr as u8 => {
                    self.pc = self.r0;
                }
                x if x == JalrCall as u8 => {
                    let return_addr = self.pc;
                    let target = self.r0;
                    self.memory.write_u32(MAILBOX_BASE + 1 * 4, return_addr);
                    self.pc = target;
                }
                x if x == Ret as u8 => {
                    if self.r0 == 0 {
                        self.halted = true;
                    } else {
                        self.pc = self.r0;
                    }
                }
                x if x == JrTableIdx as u8 => {
                    self.pc = self.r0;
                }
                x if x == JrComputed as u8 => {
                    self.pc = imm as u32;
                }
                x if x == Ecall as u8 => {
                    self.halted = true;
                }
                x if x == Halt as u8 => {
                    self.halted = true;
                }
                _ => {
                    bail!("Unknown canonical opcode {} at pc={}", op, cur_pc);
                }
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
}
