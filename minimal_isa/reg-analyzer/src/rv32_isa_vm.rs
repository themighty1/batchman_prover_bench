//! Interpreter for the rewritten RV32 ISA with reduced register file.
//!
//! Uses an ABI memory mailbox at MAILBOX_BASE to bridge calling conventions
//! across functions with independent register allocations. Each original RV32
//! register has a dedicated 4-byte slot; `conv_store` writes to it after every
//! computation, and `conv_load` reads from it at function entry / after calls.

use crate::rv32_regalloc::{RewrittenInst, frame_reg_id};
use crate::rv32_vm::Memory;
use anyhow::{Result, bail};
use std::collections::HashMap;

/// Compact I/O address map (guest ↔ host convention).
pub const IO_OUTPUT_LEN: u32  = 0x1000; // u32 output byte length  (guest→host)
pub const IO_INPUT_LEN: u32   = 0x1004; // u32 input JSON byte len (host→guest)
pub const IO_PATH_LEN: u32    = 0x1008; // u32 path string byte len(host→guest)
pub const IO_INPUT_DATA: u32  = 0x2000; // raw JSON bytes           (host→guest, 4KB max)
pub const IO_PATH_DATA: u32   = 0x3000; // path string bytes        (host→guest, 1KB max)
pub const IO_OUTPUT_DATA: u32 = 0x4000; // output value bytes       (guest→host, 256B max)

/// Base address of the ABI mailbox region in VM memory.
/// 35 slots × 4 bytes = 140 bytes (regs 0-31 + scratch A,B + spare).
pub const MAILBOX_BASE: u32 = 0x4100;

/// A memory segment to be loaded into VM memory.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct MemSegment {
    pub vaddr: u32,
    pub data: Vec<u8>,
}

/// Serializable compiled program: everything needed to execute without the original ELF.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct CompiledProgram {
    pub num_regs: u32,
    pub entry_addr: u32,
    pub segments: Vec<MemSegment>,
    pub functions: Vec<Rv32FuncInfo>,
    /// Maps any instruction address → function entry address.
    pub addr_to_func: HashMap<u32, u32>,
}

/// VM state for executing the rewritten ISA.
pub struct Rv32IsaVm {
    /// GP registers (r0..r(N-1) for N-register ISA)
    pub regs: Vec<u32>,
    /// Frame pointer register value (for spill area addressing)
    pub frame_reg: u32,
    /// Which physical register ID is the frame pointer (= num_regs, above allocatable range)
    _frame_reg_id: u8,
    /// Shared memory
    pub memory: Memory,
    /// Execution steps
    pub steps: u64,
    /// Halted flag
    pub halted: bool,
    /// Instruction type counters
    pub op_counts: HashMap<String, u64>,
    pub spec_counts: HashMap<String, u64>,
    /// Debug-only shadow state: mirrors the mailbox for validation.
    /// Only maintained when DEBUG_CONV env var is set.
    pub conv_regs: [u32; 32],
    /// Current call depth (incremented on call, decremented on return).
    call_depth: u32,
    /// Shadow ownership tags for frame-relative memory writes.
    /// Maps word-aligned address → (func_entry_addr, call_depth, write_step).
    /// On frame-relative load, if the tag doesn't match the current function+depth,
    /// the spill slot was overwritten by another function (frame aliasing).
    spill_tags: HashMap<u32, (u32, u32, u64)>,
    /// Count of frame aliasing events (reads from slots written by different func/depth).
    pub frame_alias_count: u64,
}

/// Per-function metadata for execution.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Rv32FuncInfo {
    pub rewritten: Vec<RewrittenInst>,
    pub num_spill_slots: usize,
    /// Mapping of (original_preg, physical_reg) at function entry.
    /// Computed from regalloc2's allocation of the entry block's synthetic defs.
    pub entry_reg_map: Vec<(u8, u8)>,
    /// Redirect map for jr_table targets: (jr_table_inst_addr, original_target) → synth_addr.
    /// When a jr_table target has a critical-edge trampoline, the ISA VM must
    /// jump to the trampoline (for edge moves) instead of directly to the target.
    /// Keyed per-instruction because different jr_tables may need different
    /// trampolines for the same target address.
    pub jr_table_redirects: HashMap<(u32, u32), u32>,
}

impl Rv32IsaVm {
    pub fn new(num_regs: usize) -> Self {
        Rv32IsaVm {
            regs: vec![0; num_regs],
            frame_reg: 0,
            _frame_reg_id: frame_reg_id(num_regs as u32),
            memory: Memory::new(),
            steps: 0,
            halted: false,
            op_counts: HashMap::new(),
            spec_counts: HashMap::new(),
            conv_regs: [0u32; 32],
            call_depth: 0,
            spill_tags: HashMap::new(),
            frame_alias_count: 0,
        }
    }

    /// Write a guest input fixture — delegates to Memory::write_input.
    pub fn write_input(&mut self, data: &[u8]) {
        self.memory.write_input(data);
    }

    /// Load ELF segments into memory, return entry point.
    pub fn load_elf(&mut self, data: &[u8]) -> Result<u32> {
        use object::elf::*;
        use object::read::elf::FileHeader as _;
        use object::Endianness;

        let elf = FileHeader32::<Endianness>::parse(data)?;
        let endian = elf.endian()?;
        let entry = elf.e_entry.get(endian);

        let segments = elf.program_headers(endian, data)?;
        for seg in segments {
            if seg.p_type.get(endian) != PT_LOAD { continue; }
            let vaddr = seg.p_vaddr.get(endian);
            let filesz = seg.p_filesz.get(endian) as usize;
            let offset = seg.p_offset.get(endian) as usize;

            if filesz > 0 && offset + filesz <= data.len() {
                self.memory.write_bytes(vaddr, &data[offset..offset + filesz]);
            }
        }

        Ok(entry)
    }

    /// Read a register value, treating None as x0 (zero).
    #[inline]
    fn read_reg(&self, r: Option<u8>) -> u32 {
        r.map(|r| self.regs[r as usize]).unwrap_or(0)
    }

    /// Write to a physical register.
    /// The orig_rd parameter is kept for signature compat but no longer
    /// triggers a conv_regs shadow write — that's handled by explicit
    /// conv_store instructions in the rewritten stream.
    #[inline]
    fn write_reg(&mut self, phys: u8, _orig_rd: Option<u8>, value: u32) {
        self.regs[phys as usize] = value;
    }

    /// Load physical registers from mailbox using the function's entry_reg_map.
    #[inline]
    fn _load_regs_from_mailbox(&mut self, entry_reg_map: &[(u8, u8)]) {
        for &(orig, phys) in entry_reg_map {
            self.regs[phys as usize] = self.memory.read_u32(MAILBOX_BASE + orig as u32 * 4);
        }
    }

    /// Execute a function in the rewritten ISA.
    /// If target_addr is Some, starts execution at that address within the function.
    /// Otherwise starts at the beginning (pc = 0).
    pub fn execute_function(
        &mut self,
        func: &Rv32FuncInfo,
        func_table: &HashMap<u32, Rv32FuncInfo>,
        addr_to_func: &HashMap<u32, u32>,
        target_addr: Option<u32>,
    ) -> Result<()> {
        let trace_calls = std::env::var("TRACE_CALLS").is_ok();
        let func_entry_addr = func.rewritten.iter().find(|i| i.addr != 0).map(|i| i.addr).unwrap_or(0);
        if trace_calls {
            let mb = |r: u32| self.memory.read_u32(MAILBOX_BASE + r * 4);
            eprintln!("  ENTER 0x{:x} a0={} a1={} a2={} step={}",
                func_entry_addr, mb(10), mb(11), mb(12), self.steps);
        }

        let mut pc = if let Some(addr) = target_addr {
            // Mid-function entry: not used in the compiled binary (all branch/jump
            // targets are resolved to instruction indices at compile time).
            // Kept for debugging with non-compiled paths.
            panic!("Mid-function entry at 0x{:x} requires addr_to_idx (not available in compiled binary)", addr);
        } else {
            0
        };
        let max_steps = 10_000_000;

        // Note: mailbox slots are loaded by conv_load instructions in the rewritten stream,
        // interleaved with regalloc2's entry block moves. No bulk loading needed here.

        let trace_func = std::env::var("TRACE_FUNC").ok().and_then(|s| u32::from_str_radix(&s, 16).ok());
        let do_trace_func = trace_func == Some(func_entry_addr);
        let _trace_addr_range = std::env::var("TRACE_ADDR").is_ok();

        while pc < func.rewritten.len() && !self.halted && self.steps < max_steps {
            let inst = &func.rewritten[pc];
            self.steps += 1;

            // Count instruction types
            *self.op_counts.entry(inst.op.clone()).or_insert(0) += 1;
            *self.spec_counts.entry(inst.specialized.clone()).or_insert(0) += 1;

            let trace_before_regs = if do_trace_func { Some(self.regs.clone()) } else { None };

            // Periodic trace for debugging infinite loops
            if self.steps % 1_000_000 == 0 {
                eprintln!("  [step {:>8}] in func 0x{:x} pc={} op={} addr=0x{:x}",
                    self.steps, func_entry_addr, pc, inst.op, inst.addr);
            }

            match inst.op.as_str() {
                "conv_load" => {
                    // Load a register from the ABI mailbox
                    let dst = inst.rd.unwrap() as usize;
                    let orig = inst.orig_rd.unwrap() as u32;
                    self.regs[dst] = self.memory.read_u32(MAILBOX_BASE + orig * 4);
                    pc += 1;
                }
                "conv_store" => {
                    // Store a register value to the ABI mailbox
                    let src = inst.rs1.unwrap() as usize;
                    let orig = inst.orig_rd.unwrap() as u32;
                    self.memory.write_u32(MAILBOX_BASE + orig * 4, self.regs[src]);
                    pc += 1;
                }
                "ret" => {
                    if trace_calls {
                        let retval = self.memory.read_u32(MAILBOX_BASE + 10 * 4);
                        eprintln!("  EXIT  0x{:x} a0={} step={} (ret)",
                            func_entry_addr, retval, self.steps);
                    }
                    return Ok(());
                }
                "jr_computed" => {
                    // Intra-function jump — imm is pre-resolved instruction index
                    pc = inst.imm.unwrap() as usize;
                }
                "jr_table" => {
                    // Legacy path: not used in compiled binaries (pass_rewrite_jump_tables
                    // converts all jr_table → jr_table_idx at compile time).
                    panic!("jr_table requires addr_to_idx (not available in compiled binary; use jr_table_idx)");
                }
                "jr_table_idx" => {
                    // Jump table dispatch: register already holds instruction index
                    pc = self.read_reg(inst.rs1) as usize;
                }
                "mov" => {
                    let dst = inst.rd.unwrap() as usize;
                    let src_val = self.read_reg(inst.rs1);
                    self.regs[dst] = src_val;
                    pc += 1;
                }
                "addi" => {
                    let imm = inst.imm.unwrap() as u32;
                    let dst = inst.rd.unwrap();
                    let src_val = inst.rs1.map(|r| self.regs[r as usize]).unwrap_or(0);
                    self.write_reg(dst, inst.orig_rd, src_val.wrapping_add(imm));
                    pc += 1;
                }
                "addi_frame" => {
                    let imm = inst.imm.unwrap() as u32;
                    self.frame_reg = self.frame_reg.wrapping_add(imm);
                    pc += 1;
                }
                "add" => {
                    let dst = inst.rd.unwrap();
                    let a = self.read_reg(inst.rs1);
                    let b = self.read_reg(inst.rs2);
                    self.write_reg(dst, inst.orig_rd, a.wrapping_add(b));
                    pc += 1;
                }
                "sub" => {
                    let dst = inst.rd.unwrap();
                    let a = self.read_reg(inst.rs1);
                    let b = self.read_reg(inst.rs2);
                    self.write_reg(dst, inst.orig_rd, a.wrapping_sub(b));
                    pc += 1;
                }
                "lw" => {
                    let dst = inst.rd.unwrap();
                    let offset = inst.imm.unwrap();
                    let addr = self.read_reg(inst.rs1).wrapping_add(offset as u32);
                    let val = self.memory.read_u32(addr);
                    self.write_reg(dst, inst.orig_rd, val);
                    pc += 1;
                }
                "lw_frame" => {
                    let dst = inst.rd.unwrap();
                    let offset = inst.imm.unwrap();
                    let addr = self.frame_reg.wrapping_add(offset as u32);
                    // Check spill ownership
                    if let Some(&(tag_func, tag_depth, _tag_step)) = self.spill_tags.get(&(addr & !3)) {
                        if tag_func != func_entry_addr || tag_depth != self.call_depth {
                            self.frame_alias_count += 1;
                            if std::env::var("TRACE_ALIAS").is_ok() && self.frame_alias_count <= 20 {
                                eprintln!("  ALIAS READ #{}: func=0x{:x} depth={} pc={} addr=0x{:x} step={} (writer=0x{:x} d={})",
                                    self.frame_alias_count, func_entry_addr, self.call_depth, pc, addr, self.steps,
                                    tag_func, tag_depth);
                            }
                        }
                    } else if std::env::var("CHECK_UNINIT").is_ok() {
                        eprintln!("  UNINIT SPILL READ: func=0x{:x} depth={} pc={} addr=0x{:x} step={}",
                            func_entry_addr, self.call_depth, pc, addr, self.steps);
                    }
                    let val = self.memory.read_u32(addr);
                    self.write_reg(dst, inst.orig_rd, val);
                    pc += 1;
                }
                "sw" => {
                    let offset = inst.imm.unwrap();
                    let addr = self.read_reg(inst.rs1).wrapping_add(offset as u32);
                    let val = self.read_reg(inst.rs2);
                    self.memory.write_u32(addr, val);
                    pc += 1;
                }
                "sw_frame" => {
                    let offset = inst.imm.unwrap();
                    let addr = self.frame_reg.wrapping_add(offset as u32);
                    let val = self.read_reg(inst.rs2);
                    // Tag frame-relative writes with ownership
                    self.spill_tags.insert(addr & !3, (func_entry_addr, self.call_depth, self.steps));
                    self.memory.write_u32(addr, val);
                    pc += 1;
                }
                "beq" => {
                    let a = self.read_reg(inst.rs1);
                    let b = self.read_reg(inst.rs2);
                    if a == b {
                        pc = inst.imm.unwrap() as usize;
                    } else {
                        pc += 1;
                    }
                }
                "jal" => {
                    let rd = inst.rd;
                    let offset = inst.imm.unwrap_or(0);
                    let orig_rd = inst.orig_rd.unwrap_or(0);

                    if orig_rd == 0 {
                        // Unconditional jump — imm is pre-resolved instruction index
                        let target_idx = inst.imm.unwrap() as usize;
                        if target_idx == pc {
                            self.halted = true;
                            return Ok(());
                        }
                        pc = target_idx;
                    } else {
                        // Call (original rd is non-zero, e.g., x1 or x5)
                        let return_addr = inst.addr.wrapping_add(4);
                        if let Some(phys) = rd {
                            self.write_reg(phys, inst.orig_rd, return_addr);
                        }

                        let target_addr = (inst.addr as i64 + offset as i64) as u32;

                        if let Some(&func_entry) = addr_to_func.get(&target_addr) {
                            if let Some(callee) = func_table.get(&func_entry) {
                                let saved_regs = self.regs.clone();
                                let saved_frame = self.frame_reg;
                                self.call_depth += 1;
                                self.execute_function(callee, func_table, addr_to_func, None)?;
                                self.call_depth -= 1;
                                self.regs = saved_regs;
                                self.frame_reg = saved_frame;
                                // Return value is in mailbox[10], caller's conv_load reads it
                            }
                        }
                        pc += 1;
                    }
                }
                "jalr" => {
                    let rd = inst.rd;
                    let offset = inst.imm.unwrap_or(0);
                    let orig_rd = inst.orig_rd.unwrap_or(0);
                    let target_addr = self.read_reg(inst.rs1).wrapping_add(offset as u32) & !1;
                    if orig_rd == 0 && std::env::var("TRACE_JALR").is_ok() {
                        let entry_addr = func.rewritten.iter().find(|i| i.addr != 0).map(|i| i.addr).unwrap_or(0);
                        eprintln!("  [step {:6}] jalr x0 in 0x{:x}: rs1=r{:?}={} offset={} -> target=0x{:x}",
                            self.steps, entry_addr, inst.rs1, self.read_reg(inst.rs1), offset, target_addr);
                    }

                    // Note: intra-function jalr jumps are all resolved at decode time
                    // (classify_jalr_x0 converts them to jr_computed/jr_table). No
                    // addr_to_idx lookup needed here — only inter-function calls and
                    // returns remain.
                    if orig_rd == 0 {
                        // orig_rd=x0: either a return or a tail-jump to another function.
                        // Tail-call heuristic: if target is the ENTRY of a known function,
                        // it's a jump to an outlined epilogue or tail call (e.g., `jr offset(t1)`
                        // → OUTLINED_FUNCTION). If target is in the MIDDLE of a function,
                        // it's a return (e.g., `ret` targets the caller's return address).
                        if let Some(&func_entry) = addr_to_func.get(&target_addr) {
                            if target_addr == func_entry {
                                assert!(
                                    func_entry != func_entry_addr,
                                    "tail-call heuristic: jalr x0 at 0x{:x} targets own entry \
                                     0x{:x} — infinite tail-call loop or heuristic misfire",
                                    inst.addr, func_entry
                                );
                                if let Some(callee) = func_table.get(&func_entry) {
                                    // Tail call / outlined helper jump
                                    let saved_regs = self.regs.clone();
                                    let saved_frame = self.frame_reg;
                                    self.call_depth += 1;
                                    self.execute_function(callee, func_table, addr_to_func, None)?;
                                    self.call_depth -= 1;
                                    self.regs = saved_regs;
                                    self.frame_reg = saved_frame;
                                    // Return value is in mailbox[10]
                                    if trace_calls {
                                        let retval = self.memory.read_u32(MAILBOX_BASE + 10 * 4);
                                        eprintln!("  EXIT  0x{:x} a0={} step={} (tail->0x{:x})",
                                            func_entry_addr, retval, self.steps, func_entry);
                                    }
                                    return Ok(());
                                }
                            }
                        }
                        // Plain return
                        if trace_calls {
                            let retval = self.memory.read_u32(MAILBOX_BASE + 10 * 4);
                            eprintln!("  EXIT  0x{:x} a0={} step={} (plain-ret jalr)",
                                func_entry_addr, retval, self.steps);
                        }
                        return Ok(());
                    } else {
                        // Inter-function call (orig_rd non-zero)
                        let return_addr = inst.addr.wrapping_add(4);
                        self.write_reg(rd.unwrap(), inst.orig_rd, return_addr);

                        if let Some(&func_entry) = addr_to_func.get(&target_addr) {
                            if let Some(callee) = func_table.get(&func_entry) {
                                let saved_regs = self.regs.clone();
                                let saved_frame = self.frame_reg;
                                self.call_depth += 1;
                                if target_addr == func_entry {
                                    self.execute_function(callee, func_table, addr_to_func, None)?;
                                } else {
                                    self.execute_function(callee, func_table, addr_to_func, Some(target_addr))?;
                                }
                                self.call_depth -= 1;
                                self.regs = saved_regs;
                                self.frame_reg = saved_frame;
                                // Return value is in mailbox[10], caller's conv_load reads it
                            }
                        }
                        pc += 1;
                    }
                }
                "lui" => {
                    let dst = inst.rd.unwrap();
                    let imm = inst.imm.unwrap() as u32;
                    self.write_reg(dst, inst.orig_rd, imm);
                    pc += 1;
                }
                "auipc" => {
                    let dst = inst.rd.unwrap();
                    let imm = inst.imm.unwrap() as u32;
                    assert!(
                        inst.addr != 0,
                        "auipc with inst.addr=0 — synthetic instruction would compute 0+imm \
                         instead of a real PC-relative address"
                    );
                    self.write_reg(dst, inst.orig_rd, inst.addr.wrapping_add(imm));
                    pc += 1;
                }
                "bne" | "blt" | "bge" | "bltu" | "bgeu" => {
                    let a = self.read_reg(inst.rs1);
                    let b = self.read_reg(inst.rs2);
                    let taken = match inst.op.as_str() {
                        "bne" => a != b,
                        "blt" => (a as i32) < (b as i32),
                        "bge" => (a as i32) >= (b as i32),
                        "bltu" => a < b,
                        "bgeu" => a >= b,
                        _ => unreachable!(),
                    };
                    if taken {
                        pc = inst.imm.unwrap() as usize;
                    } else {
                        pc += 1;
                    }
                }
                "lb" | "lh" | "lbu" | "lhu" => {
                    let dst = inst.rd.unwrap();
                    let offset = inst.imm.unwrap();
                    let addr = self.read_reg(inst.rs1).wrapping_add(offset as u32);
                    let val = match inst.op.as_str() {
                        "lb" => self.memory.read_u8(addr) as i8 as i32 as u32,
                        "lh" => self.memory.read_u16(addr) as i16 as i32 as u32,
                        "lbu" => self.memory.read_u8(addr) as u32,
                        "lhu" => self.memory.read_u16(addr) as u32,
                        _ => unreachable!(),
                    };
                    self.write_reg(dst, inst.orig_rd, val);
                    pc += 1;
                }
                "sb" | "sh" => {
                    let offset = inst.imm.unwrap();
                    let addr = self.read_reg(inst.rs1).wrapping_add(offset as u32);
                    let val = self.read_reg(inst.rs2);
                    match inst.op.as_str() {
                        "sb" => self.memory.write_u8(addr, val as u8),
                        "sh" => self.memory.write_u16(addr, val as u16),
                        _ => unreachable!(),
                    }
                    pc += 1;
                }
                "sll" | "srl" | "sra" | "slt" | "sltu" | "xor" | "or" | "and" => {
                    let dst = inst.rd.unwrap();
                    let a = self.read_reg(inst.rs1);
                    let b = self.read_reg(inst.rs2);
                    let result = match inst.op.as_str() {
                        "sll" => a << (b & 0x1F),
                        "srl" => a >> (b & 0x1F),
                        "sra" => ((a as i32) >> (b & 0x1F)) as u32,
                        "slt" => if (a as i32) < (b as i32) { 1 } else { 0 },
                        "sltu" => if a < b { 1 } else { 0 },
                        "xor" => a ^ b,
                        "or" => a | b,
                        "and" => a & b,
                        _ => unreachable!(),
                    };
                    self.write_reg(dst, inst.orig_rd, result);
                    pc += 1;
                }
                "slli" | "srli" | "srai" | "slti" | "sltiu" | "xori" | "ori" | "andi" => {
                    let dst = inst.rd.unwrap();
                    let imm = inst.imm.unwrap() as u32;
                    let a = self.read_reg(inst.rs1);
                    let result = match inst.op.as_str() {
                        "slli" => a << (imm & 0x1F),
                        "srli" => a >> (imm & 0x1F),
                        "srai" => ((a as i32) >> (imm & 0x1F)) as u32,
                        "slti" => if (a as i32) < (imm as i32) { 1 } else { 0 },
                        "sltiu" => if a < imm { 1 } else { 0 },
                        "xori" => a ^ imm,
                        "ori" => a | imm,
                        "andi" => a & imm,
                        _ => unreachable!(),
                    };
                    self.write_reg(dst, inst.orig_rd, result);
                    pc += 1;
                }
                "mul" | "mulh" | "mulhsu" | "mulhu" | "div" | "divu" | "rem" | "remu" => {
                    let dst = inst.rd.unwrap();
                    let a = self.read_reg(inst.rs1);
                    let b = self.read_reg(inst.rs2);
                    let result = match inst.op.as_str() {
                        "mul" => a.wrapping_mul(b),
                        "mulh" => ((a as i32 as i64).wrapping_mul(b as i32 as i64) >> 32) as u32,
                        "mulhsu" => ((a as i32 as i64).wrapping_mul(b as u64 as i64) >> 32) as u32,
                        "mulhu" => ((a as u64).wrapping_mul(b as u64) >> 32) as u32,
                        "div" => {
                            if b == 0 { u32::MAX }
                            else if a == 0x80000000 && b == 0xFFFFFFFF { a }
                            else { ((a as i32).wrapping_div(b as i32)) as u32 }
                        }
                        "divu" => if b == 0 { u32::MAX } else { a / b },
                        "rem" => {
                            if b == 0 { a }
                            else if a == 0x80000000 && b == 0xFFFFFFFF { 0 }
                            else { ((a as i32).wrapping_rem(b as i32)) as u32 }
                        }
                        "remu" => if b == 0 { a } else { a % b },
                        _ => unreachable!(),
                    };
                    self.write_reg(dst, inst.orig_rd, result);
                    pc += 1;
                }
                "ecall" => {
                    self.halted = true;
                    return Ok(());
                }
                _ => {
                    bail!("Unimplemented opcode: {} at pc={}", inst.op, pc);
                }
            }

            if let Some(before) = trace_before_regs {
                eprintln!("  [{:4}] pc={:3} 0x{:06x} {:8} {:35} imm={:?} {:?} -> {:?} f={}",
                    self.steps, pc, inst.addr, inst.op, inst.specialized,
                    inst.imm, before, self.regs, self.frame_reg);
            }
        }

        Ok(())
    }
}
