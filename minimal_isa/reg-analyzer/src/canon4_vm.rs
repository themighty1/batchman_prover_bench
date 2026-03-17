//! Lean canonical VM for 4 registers: r0 (result), r1 (source/base), r2 (source2), r3 (extra).
//!
//! Canonical forms:
//!   R-type:  r0 = r1 op r2
//!   I-type:  r0 = r1 op imm
//!   Load:    r0 = mem[r1 + imm]
//!   Store:   mem[r1 + imm] = r0
//!   Branch:  branch r0, r1, target
//!   LUI:     r0 = imm << 12
//!
//! Cache management via per-slot load/store and register-to-register moves:
//!   load_reg0/1/2/3 imm  — load regfile[imm] into r0/r1/r2/r3
//!   store_reg0/1/2/3 imm — store r0/r1/r2/r3 to regfile[imm]
//!   movXY                — rX = rY (register copy, no memory access)

use crate::rv32_vm::Memory;
use crate::rv32_isa_vm::MAILBOX_BASE;
use anyhow::{Result, bail};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Canon4Op {
    // R-type: r0 = r1 op r2
    Add = 0, Sub, Mul, Mulh, Mulhsu, Mulhu, Div, Divu, Rem, Remu,
    Sll, Srl, Sra, Slt, Sltu, Xor, Or, And,
    // I-type: r0 = r1 op imm
    Addi, Slti, Sltiu, Xori, Ori, Andi, Slli, Srli, Srai,
    // Loads: r0 = mem[r1 + imm]
    Lw, Lb, Lh, Lbu, Lhu,
    // Stores: mem[r1 + imm] = r0
    Sw, Sb, Sh,
    // Branches: branch r0, r1, target
    Beq, Bne, Blt, Bge, Bltu, Bgeu,
    // Upper immediate
    Lui,
    // Control flow
    Jal, JalCall, Jalr, JalrCall, Ret,
    JrTableIdx, JrComputed, Ecall, Halt,
    // Cache management (per-slot)
    LoadReg0, LoadReg1, LoadReg2, LoadReg3,
    StoreReg0, StoreReg1, StoreReg2, StoreReg3,
    // Register-to-register moves: movXY = rX = rY
    Mov01, Mov02, Mov03,
    Mov10, Mov12, Mov13,
    Mov20, Mov21, Mov23,
    Mov30, Mov31, Mov32,
    // sentinel
    NumOps,
}

fn name_to_op(name: &str) -> Option<Canon4Op> {
    use Canon4Op::*;
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
        "jr_table_idx" => JrTableIdx, "jr_computed" => JrComputed,
        "ecall" => Ecall, "halt" => Halt,
        "load_reg0" => LoadReg0, "load_reg1" => LoadReg1,
        "load_reg2" => LoadReg2, "load_reg3" => LoadReg3,
        "store_reg0" => StoreReg0, "store_reg1" => StoreReg1,
        "store_reg2" => StoreReg2, "store_reg3" => StoreReg3,
        "mov01" => Mov01, "mov02" => Mov02, "mov03" => Mov03,
        "mov10" => Mov10, "mov12" => Mov12, "mov13" => Mov13,
        "mov20" => Mov20, "mov21" => Mov21, "mov23" => Mov23,
        "mov30" => Mov30, "mov31" => Mov31, "mov32" => Mov32,
        _ => return None,
    })
}

/// Canonical 4-register VM: r0, r1, r2, r3, pc, memory.
pub struct Canon4Vm {
    pub r0: u32,
    pub r1: u32,
    pub r2: u32,
    pub r3: u32,
    pub memory: Memory,
    pub code: Vec<u8>,
    pub imm_table: Vec<i32>,
    pub pc: u32,
    pub steps: u64,
    pub halted: bool,
}

impl Canon4Vm {
    pub fn new(code: Vec<u8>, imm_table: Vec<i32>) -> Self {
        Canon4Vm {
            r0: 0, r1: 0, r2: 0, r3: 0,
            memory: Memory::new(),
            code, imm_table,
            pc: 0, steps: 0, halted: false,
        }
    }

    pub fn remap_code(opcode_names: &[String], code_u8: &[u8]) -> Result<Vec<u8>> {
        let mut remap = vec![Canon4Op::NumOps; 256];
        for (id, name) in opcode_names.iter().enumerate() {
            match name_to_op(name) {
                Some(op) => remap[id] = op,
                None => bail!("Unknown canon4 opcode: '{}' (id={})", name, id),
            }
        }
        Ok(code_u8.iter().map(|&id| remap[id as usize] as u8).collect())
    }

    pub fn execute(&mut self) -> Result<()> {
        use Canon4Op::*;
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
                writeln!(w, "[{}] op={} imm={} | r0={} r1={} r2={} r3={}",
                    cur_pc, op, imm, self.r0, self.r1, self.r2, self.r3).unwrap();
            }

            match op {
                // --- Cache management (per-slot) ---
                x if x == LoadReg0 as u8 => { self.r0 = self.memory.read_u32(MAILBOX_BASE + (imm as u32) * 4); }
                x if x == LoadReg1 as u8 => { self.r1 = self.memory.read_u32(MAILBOX_BASE + (imm as u32) * 4); }
                x if x == LoadReg2 as u8 => { self.r2 = self.memory.read_u32(MAILBOX_BASE + (imm as u32) * 4); }
                x if x == LoadReg3 as u8 => { self.r3 = self.memory.read_u32(MAILBOX_BASE + (imm as u32) * 4); }
                x if x == StoreReg0 as u8 => { self.memory.write_u32(MAILBOX_BASE + (imm as u32) * 4, self.r0); }
                x if x == StoreReg1 as u8 => { self.memory.write_u32(MAILBOX_BASE + (imm as u32) * 4, self.r1); }
                x if x == StoreReg2 as u8 => { self.memory.write_u32(MAILBOX_BASE + (imm as u32) * 4, self.r2); }
                x if x == StoreReg3 as u8 => { self.memory.write_u32(MAILBOX_BASE + (imm as u32) * 4, self.r3); }

                // --- Register-to-register moves ---
                x if x == Mov01 as u8 => { self.r0 = self.r1; }
                x if x == Mov02 as u8 => { self.r0 = self.r2; }
                x if x == Mov03 as u8 => { self.r0 = self.r3; }
                x if x == Mov10 as u8 => { self.r1 = self.r0; }
                x if x == Mov12 as u8 => { self.r1 = self.r2; }
                x if x == Mov13 as u8 => { self.r1 = self.r3; }
                x if x == Mov20 as u8 => { self.r2 = self.r0; }
                x if x == Mov21 as u8 => { self.r2 = self.r1; }
                x if x == Mov23 as u8 => { self.r2 = self.r3; }
                x if x == Mov30 as u8 => { self.r3 = self.r0; }
                x if x == Mov31 as u8 => { self.r3 = self.r1; }
                x if x == Mov32 as u8 => { self.r3 = self.r2; }

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
                    self.memory.write_u32(MAILBOX_BASE + 1 * 4, self.pc);
                    self.pc = imm as u32;
                }
                x if x == Jalr as u8 => { self.pc = self.r0; }
                x if x == JalrCall as u8 => {
                    let target = self.r0;
                    self.memory.write_u32(MAILBOX_BASE + 1 * 4, self.pc);
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

                _ => { bail!("Unknown canon4 opcode {} at pc={}", op, cur_pc); }
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
