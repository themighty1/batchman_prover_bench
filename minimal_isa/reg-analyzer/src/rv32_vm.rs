//! RV32IM interpreter with sparse paged memory.

use crate::canon3_types::{MemorySnapshot, MemoryRegion};
use crate::rv32_isa_vm::{IO_OUTPUT_LEN, IO_INPUT_LEN, IO_PATH_LEN,
                          IO_INPUT_DATA, IO_PATH_DATA, IO_OUTPUT_DATA};
use anyhow::{Result, bail};
use std::collections::HashMap;

const PAGE_SIZE: usize = 4096;
const PAGE_MASK: u32 = !(PAGE_SIZE as u32 - 1);

pub struct Memory {
    pages: HashMap<u32, Box<[u8; PAGE_SIZE]>>,
}

impl Memory {
    pub fn new() -> Self {
        Memory { pages: HashMap::new() }
    }

    pub fn num_pages(&self) -> usize {
        self.pages.len()
    }

    pub fn read_u8(&self, addr: u32) -> u8 {
        match self.pages.get(&(addr & PAGE_MASK)) {
            Some(p) => p[(addr & (PAGE_SIZE as u32 - 1)) as usize],
            None => 0,
        }
    }

    pub fn write_u8(&mut self, addr: u32, val: u8) {
        let page_addr = addr & PAGE_MASK;
        let off = (addr & (PAGE_SIZE as u32 - 1)) as usize;
        self.pages.entry(page_addr)
            .or_insert_with(|| Box::new([0u8; PAGE_SIZE]))[off] = val;
    }

    pub fn read_u16(&self, addr: u32) -> u16 {
        self.read_u8(addr) as u16 | ((self.read_u8(addr.wrapping_add(1)) as u16) << 8)
    }

    pub fn read_u32(&self, addr: u32) -> u32 {
        self.read_u8(addr) as u32
            | ((self.read_u8(addr.wrapping_add(1)) as u32) << 8)
            | ((self.read_u8(addr.wrapping_add(2)) as u32) << 16)
            | ((self.read_u8(addr.wrapping_add(3)) as u32) << 24)
    }

    pub fn write_u16(&mut self, addr: u32, val: u16) {
        self.write_u8(addr, val as u8);
        self.write_u8(addr.wrapping_add(1), (val >> 8) as u8);
    }

    pub fn write_u32(&mut self, addr: u32, val: u32) {
        self.write_u8(addr, val as u8);
        self.write_u8(addr.wrapping_add(1), (val >> 8) as u8);
        self.write_u8(addr.wrapping_add(2), (val >> 16) as u8);
        self.write_u8(addr.wrapping_add(3), (val >> 24) as u8);
    }

    pub fn write_bytes(&mut self, addr: u32, data: &[u8]) {
        for (i, &b) in data.iter().enumerate() {
            self.write_u8(addr.wrapping_add(i as u32), b);
        }
    }

    /// Write a byte-array guest input (json-query convention).
    pub fn write_input(&mut self, data: &[u8]) {
        self.write_u32(IO_INPUT_LEN, data.len() as u32);
        self.write_bytes(IO_INPUT_DATA, data);
    }

    /// Write a single u32 input (toy-rv32 convention).
    pub fn write_input_u32(&mut self, n: u32) {
        self.write_u32(IO_INPUT_LEN, n);
    }

    /// Write a path string for json-query convention.
    pub fn write_path(&mut self, path: &[u8]) {
        self.write_u32(IO_PATH_LEN, path.len() as u32);
        self.write_bytes(IO_PATH_DATA, path);
    }

    /// Read the output string written by json-query guest.
    pub fn read_output_string(&self) -> String {
        let len = self.read_u32(IO_OUTPUT_LEN) as usize;
        let mut bytes = vec![0u8; len];
        for i in 0..len {
            bytes[i] = self.read_u8(IO_OUTPUT_DATA + i as u32);
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Snapshot all allocated pages as contiguous memory regions.
    /// Adjacent pages are merged into single regions.
    pub fn snapshot(&self) -> MemorySnapshot {
        let mut addrs: Vec<u32> = self.pages.keys().copied().collect();
        addrs.sort();

        let mut regions = Vec::new();
        let mut i = 0;
        while i < addrs.len() {
            let start = addrs[i];
            let mut end = start + PAGE_SIZE as u32;
            // merge adjacent pages
            while i + 1 < addrs.len() && addrs[i + 1] == end {
                i += 1;
                end += PAGE_SIZE as u32;
            }
            let mut data = Vec::with_capacity((end - start) as usize);
            let mut addr = start;
            while addr < end {
                let page = &self.pages[&addr];
                data.extend_from_slice(&page[..]);
                addr += PAGE_SIZE as u32;
            }
            regions.push(MemoryRegion { addr: start, data });
            i += 1;
        }
        MemorySnapshot { regions }
    }
}

pub struct Rv32Vm {
    pub regs: [u32; 32],
    pub pc: u32,
    pub memory: Memory,
    pub halted: bool,
    pub steps: u64,
}

impl Rv32Vm {
    pub fn new() -> Self {
        Rv32Vm {
            regs: [0; 32],
            pc: 0,
            memory: Memory::new(),
            halted: false,
            steps: 0,
        }
    }

    /// Write guest input fixture — delegates to Memory::write_input.
    pub fn write_input(&mut self, data: &[u8]) {
        self.memory.write_input(data);
    }

    /// Load an RV32 ELF, return the entry point address.
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
            // BSS (memsz > filesz) is already zero since pages initialize to 0
        }

        Ok(entry)
    }

    #[inline(always)]
    fn reg(&self, r: u32) -> u32 {
        if r == 0 { 0 } else { self.regs[r as usize] }
    }

    #[inline(always)]
    fn set_reg(&mut self, r: u32, val: u32) {
        if r != 0 { self.regs[r as usize] = val; }
    }

    /// Execute one instruction. Returns Ok(()) on success.
    pub fn step(&mut self) -> Result<()> {
        let inst = self.memory.read_u32(self.pc);
        let opcode = inst & 0x7F;
        let rd = (inst >> 7) & 0x1F;
        let funct3 = (inst >> 12) & 0x7;
        let rs1 = (inst >> 15) & 0x1F;
        let rs2 = (inst >> 20) & 0x1F;
        let funct7 = (inst >> 25) & 0x7F;

        // I-type immediate (sign-extended from bit 11)
        let imm_i = ((inst as i32) >> 20) as u32;
        // S-type immediate
        let imm_s = (((inst & 0xFE000000) as i32) >> 20) as u32 | ((inst >> 7) & 0x1F);
        // B-type immediate
        let imm_b = {
            let b12 = (inst >> 31) & 1;
            let b11 = (inst >> 7) & 1;
            let b10_5 = (inst >> 25) & 0x3F;
            let b4_1 = (inst >> 8) & 0xF;
            let imm = (b12 << 12) | (b11 << 11) | (b10_5 << 5) | (b4_1 << 1);
            if b12 != 0 { imm | 0xFFFFE000 } else { imm }
        };
        // U-type immediate
        let imm_u = inst & 0xFFFFF000;
        // J-type immediate
        let imm_j = {
            let b20 = (inst >> 31) & 1;
            let b19_12 = (inst >> 12) & 0xFF;
            let b11 = (inst >> 20) & 1;
            let b10_1 = (inst >> 21) & 0x3FF;
            let imm = (b20 << 20) | (b19_12 << 12) | (b11 << 11) | (b10_1 << 1);
            if b20 != 0 { imm | 0xFFE00000 } else { imm }
        };

        let mut next_pc = self.pc.wrapping_add(4);

        match opcode {
            0x37 => { // LUI
                self.set_reg(rd, imm_u);
            }
            0x17 => { // AUIPC
                self.set_reg(rd, self.pc.wrapping_add(imm_u));
            }
            0x6F => { // JAL
                self.set_reg(rd, next_pc);
                next_pc = self.pc.wrapping_add(imm_j);
            }
            0x67 => { // JALR
                let target = self.reg(rs1).wrapping_add(imm_i) & !1;
                self.set_reg(rd, next_pc);
                next_pc = target;
            }
            0x63 => { // Branches
                let a = self.reg(rs1);
                let b = self.reg(rs2);
                let taken = match funct3 {
                    0 => a == b,                              // BEQ
                    1 => a != b,                              // BNE
                    4 => (a as i32) < (b as i32),             // BLT
                    5 => (a as i32) >= (b as i32),            // BGE
                    6 => a < b,                               // BLTU
                    7 => a >= b,                               // BGEU
                    _ => bail!("unknown branch funct3={} at pc=0x{:x}", funct3, self.pc),
                };
                if taken {
                    next_pc = self.pc.wrapping_add(imm_b);
                }
            }
            0x03 => { // Loads
                let addr = self.reg(rs1).wrapping_add(imm_i);
                let val = match funct3 {
                    0 => self.memory.read_u8(addr) as i8 as i32 as u32,     // LB
                    1 => self.memory.read_u16(addr) as i16 as i32 as u32,   // LH
                    2 => self.memory.read_u32(addr),                         // LW
                    4 => self.memory.read_u8(addr) as u32,                   // LBU
                    5 => self.memory.read_u16(addr) as u32,                  // LHU
                    _ => bail!("unknown load funct3={} at pc=0x{:x}", funct3, self.pc),
                };
                self.set_reg(rd, val);
            }
            0x23 => { // Stores
                let addr = self.reg(rs1).wrapping_add(imm_s);
                let val = self.reg(rs2);
                match funct3 {
                    0 => self.memory.write_u8(addr, val as u8),      // SB
                    1 => self.memory.write_u16(addr, val as u16),    // SH
                    2 => self.memory.write_u32(addr, val),            // SW
                    _ => bail!("unknown store funct3={} at pc=0x{:x}", funct3, self.pc),
                }
            }
            0x13 => { // ALU-immediate
                let a = self.reg(rs1);
                let imm = imm_i;
                let result = match funct3 {
                    0 => a.wrapping_add(imm),                                    // ADDI
                    1 => a << (imm & 0x1F),                                      // SLLI
                    2 => if (a as i32) < (imm as i32) { 1 } else { 0 },         // SLTI
                    3 => if a < imm { 1 } else { 0 },                           // SLTIU
                    4 => a ^ imm,                                                 // XORI
                    5 => {
                        if funct7 == 0x20 {
                            ((a as i32) >> (imm & 0x1F)) as u32                  // SRAI
                        } else {
                            a >> (imm & 0x1F)                                     // SRLI
                        }
                    }
                    6 => a | imm,                                                 // ORI
                    7 => a & imm,                                                 // ANDI
                    _ => unreachable!(),
                };
                self.set_reg(rd, result);
            }
            0x33 => { // ALU-register + M extension
                let a = self.reg(rs1);
                let b = self.reg(rs2);
                let result = if funct7 == 1 {
                    // M extension
                    match funct3 {
                        0 => a.wrapping_mul(b),                                  // MUL
                        1 => {                                                    // MULH
                            ((a as i32 as i64).wrapping_mul(b as i32 as i64) >> 32) as u32
                        }
                        2 => {                                                    // MULHSU
                            ((a as i32 as i64).wrapping_mul(b as u64 as i64) >> 32) as u32
                        }
                        3 => {                                                    // MULHU
                            ((a as u64).wrapping_mul(b as u64) >> 32) as u32
                        }
                        4 => {                                                    // DIV
                            if b == 0 { u32::MAX }
                            else if a == 0x80000000 && b == 0xFFFFFFFF { a }
                            else { ((a as i32).wrapping_div(b as i32)) as u32 }
                        }
                        5 => {                                                    // DIVU
                            if b == 0 { u32::MAX } else { a / b }
                        }
                        6 => {                                                    // REM
                            if b == 0 { a }
                            else if a == 0x80000000 && b == 0xFFFFFFFF { 0 }
                            else { ((a as i32).wrapping_rem(b as i32)) as u32 }
                        }
                        7 => {                                                    // REMU
                            if b == 0 { a } else { a % b }
                        }
                        _ => unreachable!(),
                    }
                } else {
                    match funct3 {
                        0 => if funct7 == 0x20 { a.wrapping_sub(b) } else { a.wrapping_add(b) },
                        1 => a << (b & 0x1F),                                    // SLL
                        2 => if (a as i32) < (b as i32) { 1 } else { 0 },       // SLT
                        3 => if a < b { 1 } else { 0 },                          // SLTU
                        4 => a ^ b,                                               // XOR
                        5 => {
                            if funct7 == 0x20 {
                                ((a as i32) >> (b & 0x1F)) as u32                // SRA
                            } else {
                                a >> (b & 0x1F)                                   // SRL
                            }
                        }
                        6 => a | b,                                               // OR
                        7 => a & b,                                               // AND
                        _ => unreachable!(),
                    }
                };
                self.set_reg(rd, result);
            }
            0x0F => { // FENCE — no-op
            }
            0x73 => { // SYSTEM
                if funct3 == 0 {
                    if inst == 0x00000073 {
                        // ECALL
                        self.handle_ecall()?;
                    }
                    // EBREAK (0x00100073) — ignore
                }
                // CSR instructions — no-op (return 0 for reads)
                // funct3 != 0 means CSR: CSRRW/CSRRS/CSRRC/CSRRWI/CSRRSI/CSRRCI
                // Just write 0 to rd
                if funct3 != 0 {
                    self.set_reg(rd, 0);
                }
            }
            _ => {
                bail!("unknown opcode 0x{:02x} at pc=0x{:x} (inst=0x{:08x})", opcode, self.pc, inst);
            }
        }

        self.pc = next_pc;
        self.steps += 1;
        Ok(())
    }

    fn handle_ecall(&mut self) -> Result<()> {
        // Both target programs are bare-metal no_std with no ecalls.
        // If we hit an ecall, treat it as halt.
        self.halted = true;
        Ok(())
    }

    /// Run until halted, infinite loop detected, or max_steps reached.
    pub fn run(&mut self, max_steps: u64) -> Result<u64> {
        while !self.halted && self.steps < max_steps {
            let prev_pc = self.pc;
            self.step()?;
            // Detect tight infinite loop: jal x0, 0 (instruction = 0x0000006F)
            if self.pc == prev_pc {
                self.halted = true;
            }
        }
        Ok(self.steps)
    }
}
