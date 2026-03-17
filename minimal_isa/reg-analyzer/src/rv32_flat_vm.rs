//! Flat VM: instructions as lookup tables.
//!
//! Code is a `Vec<u16>` of opcode IDs, indexed by PC (0-based).
//! Immediates are stored in a separate `imm_table[pc]`.
//! PC advances by 1 each step.

use crate::rv32_vm::Memory;
use crate::rv32_isa_vm::MAILBOX_BASE;
use anyhow::{Result, bail};

/// Metadata for a specialized opcode: which handler + which registers.
#[derive(Clone, Debug)]
pub struct OpcodeInfo {
    pub name: String,       // e.g. "addi.r0.r1" or "addi.r0.r1.42"
    pub base_op: String,    // e.g. "addi"
    pub rd: Option<u8>,
    pub rs1: Option<u8>,
    pub rs2: Option<u8>,
    pub orig_rd: Option<u8>,
    pub orig_rs1: Option<u8>,
    pub orig_rs2: Option<u8>,
}

/// Compiled flat program: everything needed to execute.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct FlatProgram {
    pub num_regs: u32,
    pub entry_pc: u32,           // 0-based instruction index
    pub segments: Vec<crate::rv32_isa_vm::MemSegment>,
    pub code_segment: Vec<u16>,  // code_segment[pc] = opcode_id (flat compiler, >255 opcodes)
    pub opcode_table: Vec<SerializedOpcodeInfo>,  // opcode_id → info
    #[serde(default)]
    pub imm_table: Vec<i32>,     // imm_table[inst_idx] = immediate value
    #[serde(default)]
    pub code_segment_u8: Vec<u8>, // code_segment[pc] = opcode_id (canonical, ≤255 opcodes)
}

/// Serializable opcode info.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SerializedOpcodeInfo {
    pub name: String,
    pub base_op: String,
    pub rd: Option<u8>,
    pub rs1: Option<u8>,
    pub rs2: Option<u8>,
    pub orig_rd: Option<u8>,
}

/// Maximum size of the spill frame area in bytes.
pub const FRAME_MAX: u32 = 1024;

/// Read u32 little-endian from a byte buffer.
#[inline(always)]
fn buf_read_u32(buf: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([buf[offset], buf[offset+1], buf[offset+2], buf[offset+3]])
}

/// Write u32 little-endian to a byte buffer.
#[inline(always)]
fn buf_write_u32(buf: &mut [u8], offset: usize, val: u32) {
    buf[offset..offset+4].copy_from_slice(&val.to_le_bytes());
}

/// Flat VM state: just registers, memory, PC.
///
/// Memory map:
///   save_stack[0..256]   — call save/restore stack (private)
///   spill_area[0..1024]  — register spill frames (private)
///   memory               — program data (ELF segments, stack, heap, mailbox)
///   code                 — instructions (private, indexed by PC)
pub struct FlatVm {
    pub regs: Vec<u32>,
    pub frame_reg: u32,       // offset into spill_area (0..1024)
    pub save_reg: u32,        // offset into save_stack (dedicated register)
    pub save_stack: Vec<u8>,  // call save/restore stack (private)
    pub spill_area: Vec<u8>,  // 1024 bytes, register spill frames
    pub memory: Memory,       // program data only
    pub code: Vec<u16>,       // code[pc] = opcode_id
    pub imm_table: Vec<i32>,  // imm_table[pc] = immediate
    pub pc: u32,              // 0-based instruction index
    pub steps: u64,
    pub halted: bool,
    pub inst_hits: Option<Vec<u64>>,
}

impl FlatVm {
    pub fn new(num_regs: usize, code: Vec<u16>, imm_table: Vec<i32>) -> Self {
        FlatVm {
            regs: vec![0; num_regs],
            frame_reg: 0,
            save_reg: 0,
            save_stack: vec![0; 4096],
            spill_area: vec![0; FRAME_MAX as usize],
            memory: Memory::new(),
            code,
            imm_table,
            pc: 0,
            steps: 0,
            halted: false,
            inst_hits: None,
        }
    }


    #[inline(always)]
    fn _read_reg(&self, r: u8) -> u32 {
        self.regs[r as usize]
    }

    /// Read register, or 0 if None (x0 = always 0 in RV32).
    #[inline(always)]
    fn read_reg_or_zero(&self, r: Option<u8>) -> u32 {
        r.map(|r| self.regs[r as usize]).unwrap_or(0)
    }

    /// Execute until halt.
    pub fn execute(&mut self, opcode_table: &[OpcodeInfo]) -> Result<()> {
        let max_steps: u64 = 100_000_000;

        // Circular trace buffer for debugging self-loops
        let trace_size = 200usize;
        let mut trace_buf: Vec<String> = Vec::with_capacity(trace_size);
        let mut trace_idx = 0usize;
        let trace_enabled = std::env::var("TRACE_LAST").is_ok();
        let trace_calls = std::env::var("TRACE_CALLS").is_ok();
        let trace_file = std::env::var("TRACE_FILE").ok();
        let mut trace_writer: Option<std::io::BufWriter<std::fs::File>> = trace_file.map(|path| {
            std::io::BufWriter::new(std::fs::File::create(&path).expect("cannot create TRACE_FILE"))
        });
        let mut save_reg_max: u32 = self.save_reg;
        let mut frame_reg_min: u32 = self.frame_reg;
        let mut frame_reg_max: u32 = self.frame_reg;

        while !self.halted && self.steps < max_steps {
            let cur_pc = self.pc;
            let opcode_id = self.code[cur_pc as usize];
            let info = &opcode_table[opcode_id as usize];
            let imm = self.imm_table[cur_pc as usize];
            self.pc += 1;
            self.steps += 1;

            if let Some(ref mut w) = trace_writer {
                use std::io::Write;
                let regs_str: String = self.regs.iter().enumerate()
                    .map(|(i, v)| format!("r{}={}", i, v))
                    .collect::<Vec<_>>().join(" ");
                writeln!(w, "[{}] {} {} | {}", cur_pc, info.name, imm, regs_str).unwrap();
            }

            if let Some(ref mut hits) = self.inst_hits {
                hits[cur_pc as usize] += 1;
            }

            if trace_enabled {
                let inst_idx = cur_pc;
                let regs_str: String = self.regs.iter().enumerate()
                    .map(|(i, v)| format!("r{}={}", i, v))
                    .collect::<Vec<_>>().join(" ");
                let entry = format!("[step {:>8}] idx={:>5} {} imm={} frame=0x{:x} {}",
                    self.steps, inst_idx, info.name, imm, self.frame_reg, regs_str);
                if trace_buf.len() < trace_size {
                    trace_buf.push(entry);
                } else {
                    trace_buf[trace_idx % trace_size] = entry;
                }
                trace_idx += 1;
            }

            match info.base_op.as_str() {
                "conv_load" => {
                    let dst = info.rd.unwrap() as usize;
                    let orig = info.orig_rd.unwrap() as u32;
                    self.regs[dst] = self.memory.read_u32(MAILBOX_BASE + orig * 4);
                }
                "conv_store" => {
                    let orig = info.orig_rd.unwrap() as u32;
                    self.memory.write_u32(MAILBOX_BASE + orig * 4, self.read_reg_or_zero(info.rs1));
                }
                "halt" => {
                    self.halted = true;
                }
                "ret" => {
                    let ra = info.rs1.map(|r| self.regs[r as usize]).unwrap_or(0);
                    if trace_calls {
                        let a0 = self.memory.read_u32(MAILBOX_BASE + 10 * 4);
                        eprintln!("  EXIT  from_pc={} to_pc={} step={} a0={}",
                            cur_pc, ra, self.steps, a0);
                    }
                    if ra == 0 {
                        self.halted = true;
                    } else {
                        self.pc = ra;
                    }
                }
                "jr_computed" => {
                    self.pc = imm as u32;
                }
                "jr_table_idx" => {
                    // Register holds a byte offset directly
                    let target = self.read_reg_or_zero(info.rs1);
                    self.pc = target;
                }
                "mov" => {
                    let dst = info.rd.unwrap() as usize;
                    let src = self.read_reg_or_zero(info.rs1);
                    self.regs[dst] = src;
                }
                "swap" => {
                    let a = info.rd.unwrap() as usize;
                    let b = info.rs1.unwrap() as usize;
                    let tmp = self.regs[a];
                    self.regs[a] = self.regs[b];
                    self.regs[b] = tmp;
                }
                "load_reg" => {
                    // imm is precomputed absolute address (MAILBOX_BASE + vreg * 4)
                    self.regs[0] = self.memory.read_u32(imm as u32);
                }
                "store_reg" => {
                    // imm is precomputed absolute address (MAILBOX_BASE + vreg * 4)
                    self.memory.write_u32(imm as u32, self.regs[0]);
                }
                "addi" => {
                    let dst = info.rd.unwrap() as usize;
                    let src = info.rs1.map(|r| self.regs[r as usize]).unwrap_or(0);
                    self.regs[dst] = src.wrapping_add(imm as u32);
                }
                "addi_frame" => {
                    self.frame_reg = self.frame_reg.wrapping_add(imm as u32);
                    assert!(self.frame_reg <= FRAME_MAX,
                        "frame_reg overflow: {} > {}", self.frame_reg, FRAME_MAX);
                    if self.frame_reg > frame_reg_max { frame_reg_max = self.frame_reg; }
                    if self.frame_reg < frame_reg_min { frame_reg_min = self.frame_reg; }
                }
                "add" => {
                    let dst = info.rd.unwrap() as usize;
                    let a = self.read_reg_or_zero(info.rs1);
                    let b = self.read_reg_or_zero(info.rs2);
                    self.regs[dst] = a.wrapping_add(b);
                }
                "sub" => {
                    let dst = info.rd.unwrap() as usize;
                    let a = self.read_reg_or_zero(info.rs1);
                    let b = self.read_reg_or_zero(info.rs2);
                    self.regs[dst] = a.wrapping_sub(b);
                }
                "lw" => {
                    let dst = info.rd.unwrap() as usize;
                    let base = self.read_reg_or_zero(info.rs1);
                    let addr = base.wrapping_add(imm as u32);
                    self.regs[dst] = self.memory.read_u32(addr);
                }
                "lw_frame" => {
                    let dst = info.rd.unwrap() as usize;
                    let off = self.frame_reg.wrapping_add(imm as u32) as usize;
                    self.regs[dst] = buf_read_u32(&self.spill_area, off);
                }
                "sw" => {
                    let base = self.read_reg_or_zero(info.rs1);
                    let addr = base.wrapping_add(imm as u32);
                    let val = self.read_reg_or_zero(info.rs2);
                    self.memory.write_u32(addr, val);
                }
                "sw_frame" => {
                    let off = self.frame_reg.wrapping_add(imm as u32) as usize;
                    let val = self.read_reg_or_zero(info.rs2);
                    buf_write_u32(&mut self.spill_area, off, val);
                }
                "beq" => {
                    let a = self.read_reg_or_zero(info.rs1);
                    let b = self.read_reg_or_zero(info.rs2);
                    if a == b { self.pc = imm as u32; }
                }
                "bne" => {
                    let a = self.read_reg_or_zero(info.rs1);
                    let b = self.read_reg_or_zero(info.rs2);
                    if a != b { self.pc = imm as u32; }
                }
                "blt" => {
                    let a = self.read_reg_or_zero(info.rs1) as i32;
                    let b = self.read_reg_or_zero(info.rs2) as i32;
                    if a < b { self.pc = imm as u32; }
                }
                "bge" => {
                    let a = self.read_reg_or_zero(info.rs1) as i32;
                    let b = self.read_reg_or_zero(info.rs2) as i32;
                    if a >= b { self.pc = imm as u32; }
                }
                "bltu" => {
                    let a = self.read_reg_or_zero(info.rs1);
                    let b = self.read_reg_or_zero(info.rs2);
                    if a < b { self.pc = imm as u32; }
                }
                "bgeu" => {
                    let a = self.read_reg_or_zero(info.rs1);
                    let b = self.read_reg_or_zero(info.rs2);
                    if a >= b { self.pc = imm as u32; }
                }
                "jal" => {
                    let target = imm as u32;
                    if target == cur_pc {
                        // Self-loop detected — dump trace buffer
                        if trace_enabled {
                            eprintln!("=== SELF-LOOP at step {} pc={} ===",
                                self.steps, cur_pc);
                            let n = trace_buf.len().min(trace_size);
                            let start = if trace_idx >= n { trace_idx - n } else { 0 };
                            for i in start..trace_idx {
                                eprintln!("  {}", trace_buf[i % trace_size]);
                            }
                        }
                        self.halted = true;
                    } else {
                        self.pc = target;
                    }
                }
                "jal_call" => {
                    // Return address = next instruction (PC already advanced past this inst)
                    let return_addr = self.pc;
                    let orig = info.orig_rd.unwrap_or(1) as u32;
                    self.memory.write_u32(MAILBOX_BASE + orig * 4, return_addr);
                    if trace_calls {
                        let a0 = self.memory.read_u32(MAILBOX_BASE + 10 * 4);
                        let a1 = self.memory.read_u32(MAILBOX_BASE + 11 * 4);
                        eprintln!("  ENTER target={} step={} a0={} a1={}",
                            imm, self.steps, a0, a1);
                    }
                    if let Some(rd) = info.rd {
                        let frame_size = (self.regs.len() + 1) * 4;
                        let frame_off = self.save_reg as usize - frame_size;
                        buf_write_u32(&mut self.save_stack, frame_off + (rd as usize) * 4, return_addr);
                    }
                    self.pc = imm as u32;
                }
                "jalr" => {
                    // Register holds a byte offset directly
                    let target = self.read_reg_or_zero(info.rs1);
                    if let Some(rd) = info.rd {
                        let return_addr = self.pc;
                        self.regs[rd as usize] = return_addr;
                        let frame_size = (self.regs.len() + 1) * 4;
                        let frame_off = self.save_reg as usize - frame_size;
                        buf_write_u32(&mut self.save_stack, frame_off + (rd as usize) * 4, return_addr);
                    }
                    self.pc = target;
                }
                "jalr_call" => {
                    // Indirect call: target from rs1, return addr to regfile[orig_rd]
                    let target = self.read_reg_or_zero(info.rs1);
                    let return_addr = self.pc;
                    let orig = info.orig_rd.unwrap_or(1) as u32;
                    self.memory.write_u32(MAILBOX_BASE + orig * 4, return_addr);
                    self.pc = target;
                }
                "lui" => {
                    let dst = info.rd.unwrap() as usize;
                    // imm holds upper 20 bits >> 12 (set by compiler)
                    self.regs[dst] = (imm as u32) << 12;
                }
                "auipc" => {
                    let dst = info.rd.unwrap() as usize;
                    // auipc uses the address of THIS instruction, not the next
                    self.regs[dst] = cur_pc.wrapping_add((imm as u32) << 12);
                }
                "lb" => {
                    let dst = info.rd.unwrap() as usize;
                    let addr = self.read_reg_or_zero(info.rs1).wrapping_add(imm as u32);
                    self.regs[dst] = self.memory.read_u8(addr) as i8 as i32 as u32;
                }
                "lh" => {
                    let dst = info.rd.unwrap() as usize;
                    let addr = self.read_reg_or_zero(info.rs1).wrapping_add(imm as u32);
                    self.regs[dst] = self.memory.read_u16(addr) as i16 as i32 as u32;
                }
                "lbu" => {
                    let dst = info.rd.unwrap() as usize;
                    let addr = self.read_reg_or_zero(info.rs1).wrapping_add(imm as u32);
                    self.regs[dst] = self.memory.read_u8(addr) as u32;
                }
                "lhu" => {
                    let dst = info.rd.unwrap() as usize;
                    let addr = self.read_reg_or_zero(info.rs1).wrapping_add(imm as u32);
                    self.regs[dst] = self.memory.read_u16(addr) as u32;
                }
                "sb" => {
                    let addr = self.read_reg_or_zero(info.rs1).wrapping_add(imm as u32);
                    let val = self.read_reg_or_zero(info.rs2);
                    self.memory.write_u8(addr, val as u8);
                }
                "sh" => {
                    let addr = self.read_reg_or_zero(info.rs1).wrapping_add(imm as u32);
                    let val = self.read_reg_or_zero(info.rs2);
                    self.memory.write_u16(addr, val as u16);
                }
                "sll" => {
                    let dst = info.rd.unwrap() as usize;
                    let a = self.read_reg_or_zero(info.rs1);
                    let b = self.read_reg_or_zero(info.rs2);
                    self.regs[dst] = a << (b & 0x1F);
                }
                "srl" => {
                    let dst = info.rd.unwrap() as usize;
                    let a = self.read_reg_or_zero(info.rs1);
                    let b = self.read_reg_or_zero(info.rs2);
                    self.regs[dst] = a >> (b & 0x1F);
                }
                "sra" => {
                    let dst = info.rd.unwrap() as usize;
                    let a = self.read_reg_or_zero(info.rs1);
                    let b = self.read_reg_or_zero(info.rs2);
                    self.regs[dst] = ((a as i32) >> (b & 0x1F)) as u32;
                }
                "slt" => {
                    let dst = info.rd.unwrap() as usize;
                    let a = self.read_reg_or_zero(info.rs1) as i32;
                    let b = self.read_reg_or_zero(info.rs2) as i32;
                    self.regs[dst] = if a < b { 1 } else { 0 };
                }
                "sltu" => {
                    let dst = info.rd.unwrap() as usize;
                    let a = self.read_reg_or_zero(info.rs1);
                    let b = self.read_reg_or_zero(info.rs2);
                    self.regs[dst] = if a < b { 1 } else { 0 };
                }
                "xor" | "or" | "and" => {
                    let dst = info.rd.unwrap() as usize;
                    let a = self.read_reg_or_zero(info.rs1);
                    let b = self.read_reg_or_zero(info.rs2);
                    let result = match info.base_op.as_str() {
                        "xor" => a ^ b,
                        "or" => a | b,
                        "and" => a & b,
                        _ => unreachable!(),
                    };
                    self.regs[dst] = result;
                }
                "slli" | "srli" | "srai" => {
                    let dst = info.rd.unwrap() as usize;
                    let a = self.read_reg_or_zero(info.rs1);
                    let result = match info.base_op.as_str() {
                        "slli" => a << (imm as u32 & 0x1F),
                        "srli" => a >> (imm as u32 & 0x1F),
                        "srai" => ((a as i32) >> (imm as u32 & 0x1F)) as u32,
                        _ => unreachable!(),
                    };
                    self.regs[dst] = result;
                }
                "slti" => {
                    let dst = info.rd.unwrap() as usize;
                    let a = self.read_reg_or_zero(info.rs1) as i32;
                    self.regs[dst] = if a < imm { 1 } else { 0 };
                }
                "sltiu" => {
                    let dst = info.rd.unwrap() as usize;
                    let a = self.read_reg_or_zero(info.rs1);
                    self.regs[dst] = if a < (imm as u32) { 1 } else { 0 };
                }
                "xori" | "ori" | "andi" => {
                    let dst = info.rd.unwrap() as usize;
                    let a = self.read_reg_or_zero(info.rs1);
                    let result = match info.base_op.as_str() {
                        "xori" => a ^ (imm as u32),
                        "ori" => a | (imm as u32),
                        "andi" => a & (imm as u32),
                        _ => unreachable!(),
                    };
                    self.regs[dst] = result;
                }
                "mul" => {
                    let dst = info.rd.unwrap() as usize;
                    let a = self.read_reg_or_zero(info.rs1);
                    let b = self.read_reg_or_zero(info.rs2);
                    self.regs[dst] = a.wrapping_mul(b);
                }
                "mulh" => {
                    let dst = info.rd.unwrap() as usize;
                    let a = self.read_reg_or_zero(info.rs1);
                    let b = self.read_reg_or_zero(info.rs2);
                    self.regs[dst] = ((a as i32 as i64).wrapping_mul(b as i32 as i64) >> 32) as u32;
                }
                "mulhsu" => {
                    let dst = info.rd.unwrap() as usize;
                    let a = self.read_reg_or_zero(info.rs1);
                    let b = self.read_reg_or_zero(info.rs2);
                    self.regs[dst] = ((a as i32 as i64).wrapping_mul(b as u64 as i64) >> 32) as u32;
                }
                "mulhu" => {
                    let dst = info.rd.unwrap() as usize;
                    let a = self.read_reg_or_zero(info.rs1);
                    let b = self.read_reg_or_zero(info.rs2);
                    self.regs[dst] = ((a as u64).wrapping_mul(b as u64) >> 32) as u32;
                }
                "div" => {
                    let dst = info.rd.unwrap() as usize;
                    let a = self.read_reg_or_zero(info.rs1);
                    let b = self.read_reg_or_zero(info.rs2);
                    self.regs[dst] = if b == 0 { u32::MAX }
                        else if a == 0x80000000 && b == 0xFFFFFFFF { a }
                        else { ((a as i32).wrapping_div(b as i32)) as u32 };
                }
                "divu" => {
                    let dst = info.rd.unwrap() as usize;
                    let a = self.read_reg_or_zero(info.rs1);
                    let b = self.read_reg_or_zero(info.rs2);
                    self.regs[dst] = if b == 0 { u32::MAX } else { a / b };
                }
                "rem" => {
                    let dst = info.rd.unwrap() as usize;
                    let a = self.read_reg_or_zero(info.rs1);
                    let b = self.read_reg_or_zero(info.rs2);
                    self.regs[dst] = if b == 0 { a }
                        else if a == 0x80000000 && b == 0xFFFFFFFF { 0 }
                        else { ((a as i32).wrapping_rem(b as i32)) as u32 };
                }
                "remu" => {
                    let dst = info.rd.unwrap() as usize;
                    let a = self.read_reg_or_zero(info.rs1);
                    let b = self.read_reg_or_zero(info.rs2);
                    self.regs[dst] = if b == 0 { a } else { a % b };
                }
                "ecall" => {
                    self.halted = true;
                }
                "sw_save" => {
                    let off = self.save_reg as usize + imm as usize;
                    buf_write_u32(&mut self.save_stack, off, self.regs[info.rs2.unwrap() as usize]);
                }
                "lw_save" => {
                    let off = self.save_reg as usize + imm as usize;
                    self.regs[info.rd.unwrap() as usize] = buf_read_u32(&self.save_stack, off);
                }
                "sw_save_frame" => {
                    let off = self.save_reg as usize + imm as usize;
                    buf_write_u32(&mut self.save_stack, off, self.frame_reg);
                }
                "lw_save_frame" => {
                    let off = self.save_reg as usize + imm as usize;
                    self.frame_reg = buf_read_u32(&self.save_stack, off);
                }
                "addi_save" => {
                    self.save_reg = (self.save_reg as i32 + imm) as u32;
                    if self.save_reg > save_reg_max { save_reg_max = self.save_reg; }
                }
                _ => {
                    bail!("Unimplemented flat opcode: {} (id={}) at pc=0x{:x}", info.base_op, opcode_id, cur_pc);
                }
            }

            if self.steps % 1_000_000 == 0 {
                eprintln!("  [step {:>8}] pc={} op={}", self.steps, cur_pc, info.name);
            }
        }

        if self.steps >= max_steps && !self.halted {
            bail!("Execution limit reached ({} steps)", max_steps);
        }

        let peak = save_reg_max as usize;
        let frame = (self.regs.len() + 1) * 4;
        eprintln!("  Save stack:   {} bytes peak ({} call depth, {} bytes/frame)",
            peak, if frame > 0 { peak / frame } else { 0 }, frame);
        eprintln!("  Frame reg:    0x{:x}..0x{:x} (range {} bytes)",
            frame_reg_min, frame_reg_max, frame_reg_max - frame_reg_min);


        Ok(())
    }
}
