//! Register-based VM interpreter
//! Executes VRegInst IR with actual values

use crate::regvm::{VReg, VRegInst};
use std::collections::HashMap;

/// Runtime value (i32 or i64)
#[derive(Debug, Clone, Copy)]
pub enum Value {
    I32(i32),
    I64(i64),
}

impl Value {
    pub fn as_i32(&self) -> i32 {
        match self {
            Value::I32(v) => *v,
            Value::I64(v) => *v as i32,
        }
    }

    pub fn as_i64(&self) -> i64 {
        match self {
            Value::I32(v) => *v as i64,
            Value::I64(v) => *v,
        }
    }

    pub fn as_u32(&self) -> u32 {
        self.as_i32() as u32
    }

    pub fn as_u64(&self) -> u64 {
        self.as_i64() as u64
    }
}

/// Interpreter state for a single function
pub struct FuncInterpreter {
    /// Virtual register values
    regs: HashMap<VReg, Value>,
    /// Local variable values
    locals: HashMap<u32, Value>,
    /// Global variable values (shared)
    globals: HashMap<u32, Value>,
    /// Linear memory
    memory: Vec<u8>,
    /// Call stack for block labels
    _label_stack: Vec<u32>,
    /// Program counter
    _pc: usize,
    /// Return value
    return_value: Option<Value>,
}

impl FuncInterpreter {
    pub fn new(memory_pages: u32) -> Self {
        Self {
            regs: HashMap::new(),
            locals: HashMap::new(),
            globals: HashMap::new(),
            memory: vec![0u8; (memory_pages as usize) * 65536],
            _label_stack: Vec::new(),
            _pc: 0,
            return_value: None,
        }
    }

    pub fn set_local(&mut self, idx: u32, val: Value) {
        self.locals.insert(idx, val);
    }

    pub fn get_local(&self, idx: u32) -> Value {
        self.locals.get(&idx).copied().unwrap_or(Value::I32(0))
    }

    pub fn set_global(&mut self, idx: u32, val: Value) {
        self.globals.insert(idx, val);
    }

    pub fn get_global(&self, idx: u32) -> Value {
        self.globals.get(&idx).copied().unwrap_or(Value::I32(0))
    }

    pub fn get_reg(&self, vreg: VReg) -> Value {
        self.regs.get(&vreg).copied().unwrap_or(Value::I32(0))
    }

    pub fn set_reg(&mut self, vreg: VReg, val: Value) {
        self.regs.insert(vreg, val);
    }

    pub fn memory(&self) -> &[u8] {
        &self.memory
    }

    pub fn memory_mut(&mut self) -> &mut [u8] {
        &mut self.memory
    }

    /// Write bytes to memory
    pub fn write_memory(&mut self, offset: usize, data: &[u8]) {
        if offset + data.len() <= self.memory.len() {
            self.memory[offset..offset + data.len()].copy_from_slice(data);
        }
    }

    /// Read i32 from memory
    fn mem_load_i32(&self, addr: u32, offset: u32) -> i32 {
        let idx = (addr + offset) as usize;
        if idx + 4 <= self.memory.len() {
            i32::from_le_bytes(self.memory[idx..idx + 4].try_into().unwrap())
        } else {
            0
        }
    }

    /// Read i64 from memory
    fn mem_load_i64(&self, addr: u32, offset: u32) -> i64 {
        let idx = (addr + offset) as usize;
        if idx + 8 <= self.memory.len() {
            i64::from_le_bytes(self.memory[idx..idx + 8].try_into().unwrap())
        } else {
            0
        }
    }

    /// Read u8 from memory
    fn mem_load_u8(&self, addr: u32, offset: u32) -> u8 {
        let idx = (addr + offset) as usize;
        if idx < self.memory.len() {
            self.memory[idx]
        } else {
            0
        }
    }

    /// Read i8 from memory
    fn mem_load_i8(&self, addr: u32, offset: u32) -> i8 {
        self.mem_load_u8(addr, offset) as i8
    }

    /// Read u16 from memory
    fn mem_load_u16(&self, addr: u32, offset: u32) -> u16 {
        let idx = (addr + offset) as usize;
        if idx + 2 <= self.memory.len() {
            u16::from_le_bytes(self.memory[idx..idx + 2].try_into().unwrap())
        } else {
            0
        }
    }

    /// Read i16 from memory
    fn mem_load_i16(&self, addr: u32, offset: u32) -> i16 {
        self.mem_load_u16(addr, offset) as i16
    }

    /// Store i32 to memory
    fn mem_store_i32(&mut self, addr: u32, offset: u32, val: i32) {
        let idx = (addr + offset) as usize;
        if idx + 4 <= self.memory.len() {
            self.memory[idx..idx + 4].copy_from_slice(&val.to_le_bytes());
        }
    }

    /// Store i64 to memory
    fn mem_store_i64(&mut self, addr: u32, offset: u32, val: i64) {
        let idx = (addr + offset) as usize;
        if idx + 8 <= self.memory.len() {
            self.memory[idx..idx + 8].copy_from_slice(&val.to_le_bytes());
        }
    }

    /// Store u8 to memory
    fn mem_store_u8(&mut self, addr: u32, offset: u32, val: u8) {
        let idx = (addr + offset) as usize;
        if idx < self.memory.len() {
            self.memory[idx] = val;
        }
    }

    /// Store u16 to memory
    fn mem_store_u16(&mut self, addr: u32, offset: u32, val: u16) {
        let idx = (addr + offset) as usize;
        if idx + 2 <= self.memory.len() {
            self.memory[idx..idx + 2].copy_from_slice(&val.to_le_bytes());
        }
    }

    /// Execute a single instruction, returns true if should continue
    pub fn step(&mut self, inst: &VRegInst) -> bool {
        use VRegInst::*;

        match inst {
            // Constants
            I32Const { dst, val } => {
                self.set_reg(*dst, Value::I32(*val));
            }
            I64Const { dst, val } => {
                self.set_reg(*dst, Value::I64(*val));
            }

            // Binary i32 ops
            I32Add { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i32();
                let b = self.get_reg(*src2).as_i32();
                self.set_reg(*dst, Value::I32(a.wrapping_add(b)));
            }
            I32Sub { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i32();
                let b = self.get_reg(*src2).as_i32();
                self.set_reg(*dst, Value::I32(a.wrapping_sub(b)));
            }
            I32Mul { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i32();
                let b = self.get_reg(*src2).as_i32();
                self.set_reg(*dst, Value::I32(a.wrapping_mul(b)));
            }
            I32DivS { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i32();
                let b = self.get_reg(*src2).as_i32();
                self.set_reg(*dst, Value::I32(if b != 0 { a / b } else { 0 }));
            }
            I32DivU { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_u32();
                let b = self.get_reg(*src2).as_u32();
                self.set_reg(*dst, Value::I32(if b != 0 { (a / b) as i32 } else { 0 }));
            }
            I32RemS { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i32();
                let b = self.get_reg(*src2).as_i32();
                self.set_reg(*dst, Value::I32(if b != 0 { a % b } else { 0 }));
            }
            I32RemU { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_u32();
                let b = self.get_reg(*src2).as_u32();
                self.set_reg(*dst, Value::I32(if b != 0 { (a % b) as i32 } else { 0 }));
            }
            I32And { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i32();
                let b = self.get_reg(*src2).as_i32();
                self.set_reg(*dst, Value::I32(a & b));
            }
            I32Or { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i32();
                let b = self.get_reg(*src2).as_i32();
                self.set_reg(*dst, Value::I32(a | b));
            }
            I32Xor { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i32();
                let b = self.get_reg(*src2).as_i32();
                self.set_reg(*dst, Value::I32(a ^ b));
            }
            I32Shl { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i32();
                let b = self.get_reg(*src2).as_i32();
                self.set_reg(*dst, Value::I32(a.wrapping_shl(b as u32)));
            }
            I32ShrU { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_u32();
                let b = self.get_reg(*src2).as_i32();
                self.set_reg(*dst, Value::I32(a.wrapping_shr(b as u32) as i32));
            }
            I32ShrS { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i32();
                let b = self.get_reg(*src2).as_i32();
                self.set_reg(*dst, Value::I32(a.wrapping_shr(b as u32)));
            }
            I32Rotl { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_u32();
                let b = self.get_reg(*src2).as_u32();
                self.set_reg(*dst, Value::I32(a.rotate_left(b) as i32));
            }
            I32Rotr { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_u32();
                let b = self.get_reg(*src2).as_u32();
                self.set_reg(*dst, Value::I32(a.rotate_right(b) as i32));
            }

            // Binary i64 ops
            I64Add { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i64();
                let b = self.get_reg(*src2).as_i64();
                self.set_reg(*dst, Value::I64(a.wrapping_add(b)));
            }
            I64Sub { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i64();
                let b = self.get_reg(*src2).as_i64();
                self.set_reg(*dst, Value::I64(a.wrapping_sub(b)));
            }
            I64Mul { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i64();
                let b = self.get_reg(*src2).as_i64();
                self.set_reg(*dst, Value::I64(a.wrapping_mul(b)));
            }
            I64DivS { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i64();
                let b = self.get_reg(*src2).as_i64();
                self.set_reg(*dst, Value::I64(if b != 0 { a / b } else { 0 }));
            }
            I64DivU { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_u64();
                let b = self.get_reg(*src2).as_u64();
                self.set_reg(*dst, Value::I64(if b != 0 { (a / b) as i64 } else { 0 }));
            }
            I64RemS { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i64();
                let b = self.get_reg(*src2).as_i64();
                self.set_reg(*dst, Value::I64(if b != 0 { a % b } else { 0 }));
            }
            I64RemU { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_u64();
                let b = self.get_reg(*src2).as_u64();
                self.set_reg(*dst, Value::I64(if b != 0 { (a % b) as i64 } else { 0 }));
            }
            I64And { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i64();
                let b = self.get_reg(*src2).as_i64();
                self.set_reg(*dst, Value::I64(a & b));
            }
            I64Or { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i64();
                let b = self.get_reg(*src2).as_i64();
                self.set_reg(*dst, Value::I64(a | b));
            }
            I64Xor { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i64();
                let b = self.get_reg(*src2).as_i64();
                self.set_reg(*dst, Value::I64(a ^ b));
            }
            I64Shl { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i64();
                let b = self.get_reg(*src2).as_i64();
                self.set_reg(*dst, Value::I64(a.wrapping_shl(b as u32)));
            }
            I64ShrU { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_u64();
                let b = self.get_reg(*src2).as_i64();
                self.set_reg(*dst, Value::I64(a.wrapping_shr(b as u32) as i64));
            }
            I64ShrS { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i64();
                let b = self.get_reg(*src2).as_i64();
                self.set_reg(*dst, Value::I64(a.wrapping_shr(b as u32)));
            }

            // Unary ops
            I32Eqz { dst, src } => {
                let a = self.get_reg(*src).as_i32();
                self.set_reg(*dst, Value::I32(if a == 0 { 1 } else { 0 }));
            }
            I32Clz { dst, src } => {
                let a = self.get_reg(*src).as_u32();
                self.set_reg(*dst, Value::I32(a.leading_zeros() as i32));
            }
            I32Ctz { dst, src } => {
                let a = self.get_reg(*src).as_u32();
                self.set_reg(*dst, Value::I32(a.trailing_zeros() as i32));
            }
            I32Popcnt { dst, src } => {
                let a = self.get_reg(*src).as_u32();
                self.set_reg(*dst, Value::I32(a.count_ones() as i32));
            }
            I64Eqz { dst, src } => {
                let a = self.get_reg(*src).as_i64();
                self.set_reg(*dst, Value::I32(if a == 0 { 1 } else { 0 }));
            }
            I64Clz { dst, src } => {
                let a = self.get_reg(*src).as_u64();
                self.set_reg(*dst, Value::I64(a.leading_zeros() as i64));
            }
            I64Ctz { dst, src } => {
                let a = self.get_reg(*src).as_u64();
                self.set_reg(*dst, Value::I64(a.trailing_zeros() as i64));
            }

            // Comparisons i32
            I32Eq { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i32();
                let b = self.get_reg(*src2).as_i32();
                self.set_reg(*dst, Value::I32(if a == b { 1 } else { 0 }));
            }
            I32Ne { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i32();
                let b = self.get_reg(*src2).as_i32();
                self.set_reg(*dst, Value::I32(if a != b { 1 } else { 0 }));
            }
            I32LtS { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i32();
                let b = self.get_reg(*src2).as_i32();
                self.set_reg(*dst, Value::I32(if a < b { 1 } else { 0 }));
            }
            I32LtU { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_u32();
                let b = self.get_reg(*src2).as_u32();
                self.set_reg(*dst, Value::I32(if a < b { 1 } else { 0 }));
            }
            I32GtS { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i32();
                let b = self.get_reg(*src2).as_i32();
                self.set_reg(*dst, Value::I32(if a > b { 1 } else { 0 }));
            }
            I32GtU { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_u32();
                let b = self.get_reg(*src2).as_u32();
                self.set_reg(*dst, Value::I32(if a > b { 1 } else { 0 }));
            }
            I32LeS { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i32();
                let b = self.get_reg(*src2).as_i32();
                self.set_reg(*dst, Value::I32(if a <= b { 1 } else { 0 }));
            }
            I32LeU { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_u32();
                let b = self.get_reg(*src2).as_u32();
                self.set_reg(*dst, Value::I32(if a <= b { 1 } else { 0 }));
            }
            I32GeS { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i32();
                let b = self.get_reg(*src2).as_i32();
                self.set_reg(*dst, Value::I32(if a >= b { 1 } else { 0 }));
            }
            I32GeU { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_u32();
                let b = self.get_reg(*src2).as_u32();
                self.set_reg(*dst, Value::I32(if a >= b { 1 } else { 0 }));
            }

            // Comparisons i64
            I64Eq { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i64();
                let b = self.get_reg(*src2).as_i64();
                self.set_reg(*dst, Value::I32(if a == b { 1 } else { 0 }));
            }
            I64Ne { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i64();
                let b = self.get_reg(*src2).as_i64();
                self.set_reg(*dst, Value::I32(if a != b { 1 } else { 0 }));
            }
            I64LtS { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i64();
                let b = self.get_reg(*src2).as_i64();
                self.set_reg(*dst, Value::I32(if a < b { 1 } else { 0 }));
            }
            I64LtU { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_u64();
                let b = self.get_reg(*src2).as_u64();
                self.set_reg(*dst, Value::I32(if a < b { 1 } else { 0 }));
            }
            I64GtS { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i64();
                let b = self.get_reg(*src2).as_i64();
                self.set_reg(*dst, Value::I32(if a > b { 1 } else { 0 }));
            }
            I64GtU { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_u64();
                let b = self.get_reg(*src2).as_u64();
                self.set_reg(*dst, Value::I32(if a > b { 1 } else { 0 }));
            }
            I64LeS { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i64();
                let b = self.get_reg(*src2).as_i64();
                self.set_reg(*dst, Value::I32(if a <= b { 1 } else { 0 }));
            }
            I64LeU { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_u64();
                let b = self.get_reg(*src2).as_u64();
                self.set_reg(*dst, Value::I32(if a <= b { 1 } else { 0 }));
            }
            I64GeS { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_i64();
                let b = self.get_reg(*src2).as_i64();
                self.set_reg(*dst, Value::I32(if a >= b { 1 } else { 0 }));
            }
            I64GeU { dst, src1, src2 } => {
                let a = self.get_reg(*src1).as_u64();
                let b = self.get_reg(*src2).as_u64();
                self.set_reg(*dst, Value::I32(if a >= b { 1 } else { 0 }));
            }

            // Type conversions
            I32WrapI64 { dst, src } => {
                let a = self.get_reg(*src).as_i64();
                self.set_reg(*dst, Value::I32(a as i32));
            }
            I64ExtendI32S { dst, src } => {
                let a = self.get_reg(*src).as_i32();
                self.set_reg(*dst, Value::I64(a as i64));
            }
            I64ExtendI32U { dst, src } => {
                let a = self.get_reg(*src).as_u32();
                self.set_reg(*dst, Value::I64(a as i64));
            }
            I32Extend8S { dst, src } => {
                let a = self.get_reg(*src).as_i32() as i8;
                self.set_reg(*dst, Value::I32(a as i32));
            }
            I32Extend16S { dst, src } => {
                let a = self.get_reg(*src).as_i32() as i16;
                self.set_reg(*dst, Value::I32(a as i32));
            }

            // Memory loads
            I32Load { dst, addr, offset } => {
                let a = self.get_reg(*addr).as_u32();
                let val = self.mem_load_i32(a, *offset);
                self.set_reg(*dst, Value::I32(val));
            }
            I64Load { dst, addr, offset } => {
                let a = self.get_reg(*addr).as_u32();
                let val = self.mem_load_i64(a, *offset);
                self.set_reg(*dst, Value::I64(val));
            }
            I32Load8U { dst, addr, offset } => {
                let a = self.get_reg(*addr).as_u32();
                let val = self.mem_load_u8(a, *offset);
                self.set_reg(*dst, Value::I32(val as i32));
            }
            I32Load8S { dst, addr, offset } => {
                let a = self.get_reg(*addr).as_u32();
                let val = self.mem_load_i8(a, *offset);
                self.set_reg(*dst, Value::I32(val as i32));
            }
            I32Load16U { dst, addr, offset } => {
                let a = self.get_reg(*addr).as_u32();
                let val = self.mem_load_u16(a, *offset);
                self.set_reg(*dst, Value::I32(val as i32));
            }
            I32Load16S { dst, addr, offset } => {
                let a = self.get_reg(*addr).as_u32();
                let val = self.mem_load_i16(a, *offset);
                self.set_reg(*dst, Value::I32(val as i32));
            }
            I64Load8U { dst, addr, offset } => {
                let a = self.get_reg(*addr).as_u32();
                let val = self.mem_load_u8(a, *offset);
                self.set_reg(*dst, Value::I64(val as i64));
            }
            I64Load8S { dst, addr, offset } => {
                let a = self.get_reg(*addr).as_u32();
                let val = self.mem_load_i8(a, *offset);
                self.set_reg(*dst, Value::I64(val as i64));
            }
            I64Load16U { dst, addr, offset } => {
                let a = self.get_reg(*addr).as_u32();
                let val = self.mem_load_u16(a, *offset);
                self.set_reg(*dst, Value::I64(val as i64));
            }
            I64Load16S { dst, addr, offset } => {
                let a = self.get_reg(*addr).as_u32();
                let val = self.mem_load_i16(a, *offset);
                self.set_reg(*dst, Value::I64(val as i64));
            }
            I64Load32U { dst, addr, offset } => {
                let a = self.get_reg(*addr).as_u32();
                let val = self.mem_load_i32(a, *offset) as u32;
                self.set_reg(*dst, Value::I64(val as i64));
            }
            I64Load32S { dst, addr, offset } => {
                let a = self.get_reg(*addr).as_u32();
                let val = self.mem_load_i32(a, *offset);
                self.set_reg(*dst, Value::I64(val as i64));
            }

            // Memory stores
            I32Store { addr, offset, src } => {
                let a = self.get_reg(*addr).as_u32();
                let val = self.get_reg(*src).as_i32();
                self.mem_store_i32(a, *offset, val);
            }
            I64Store { addr, offset, src } => {
                let a = self.get_reg(*addr).as_u32();
                let val = self.get_reg(*src).as_i64();
                self.mem_store_i64(a, *offset, val);
            }
            I32Store8 { addr, offset, src } => {
                let a = self.get_reg(*addr).as_u32();
                let val = self.get_reg(*src).as_i32() as u8;
                self.mem_store_u8(a, *offset, val);
            }
            I32Store16 { addr, offset, src } => {
                let a = self.get_reg(*addr).as_u32();
                let val = self.get_reg(*src).as_i32() as u16;
                self.mem_store_u16(a, *offset, val);
            }
            I64Store8 { addr, offset, src } => {
                let a = self.get_reg(*addr).as_u32();
                let val = self.get_reg(*src).as_i64() as u8;
                self.mem_store_u8(a, *offset, val);
            }
            I64Store16 { addr, offset, src } => {
                let a = self.get_reg(*addr).as_u32();
                let val = self.get_reg(*src).as_i64() as u16;
                self.mem_store_u16(a, *offset, val);
            }
            I64Store32 { addr, offset, src } => {
                let a = self.get_reg(*addr).as_u32();
                let val = self.get_reg(*src).as_i64() as i32;
                self.mem_store_i32(a, *offset, val);
            }

            // Local variables
            LocalGet { dst, local } => {
                let val = self.get_local(*local);
                self.set_reg(*dst, val);
            }
            LocalSet { local, src } => {
                let val = self.get_reg(*src);
                self.set_local(*local, val);
            }
            LocalTee { dst, local, src } => {
                let val = self.get_reg(*src);
                self.set_local(*local, val);
                self.set_reg(*dst, val);
            }

            // Global variables
            GlobalGet { dst, global } => {
                let val = self.get_global(*global);
                self.set_reg(*dst, val);
            }
            GlobalSet { global, src } => {
                let val = self.get_reg(*src);
                self.set_global(*global, val);
            }

            // Select
            Select { dst, cond, src1, src2 } => {
                let c = self.get_reg(*cond).as_i32();
                let val = if c != 0 {
                    self.get_reg(*src1)
                } else {
                    self.get_reg(*src2)
                };
                self.set_reg(*dst, val);
            }

            // Control flow - these need special handling in the VM
            Block { .. } | Loop { .. } | If { .. } | Else { .. } | End { .. } => {
                // Handled by VM control flow
            }
            Br { .. } | BrIf { .. } | BrTable { .. } => {
                // Handled by VM control flow
            }

            Return { values } => {
                if let Some(vreg) = values.first() {
                    self.return_value = Some(self.get_reg(*vreg));
                }
                return false;
            }

            // Function calls - need VM-level handling
            Call { results, .. } | CallIndirect { results, .. } => {
                // Placeholder - actual calls handled by VM
                for r in results {
                    self.set_reg(*r, Value::I32(0));
                }
            }

            Unreachable => {
                return false;
            }
            Nop => {}
            Drop { .. } => {}

            MemorySize { dst } => {
                let pages = (self.memory.len() / 65536) as i32;
                self.set_reg(*dst, Value::I32(pages));
            }
            MemoryGrow { dst, pages } => {
                let req = self.get_reg(*pages).as_i32() as usize;
                let old_pages = self.memory.len() / 65536;
                self.memory.resize((old_pages + req) * 65536, 0);
                self.set_reg(*dst, Value::I32(old_pages as i32));
            }
            MemoryCopy { dst, src, len } => {
                let dst_addr = self.get_reg(*dst).as_u32() as usize;
                let src_addr = self.get_reg(*src).as_u32() as usize;
                let copy_len = self.get_reg(*len).as_u32() as usize;
                // Handle overlapping regions
                if src_addr < dst_addr && src_addr + copy_len > dst_addr {
                    for i in (0..copy_len).rev() {
                        if src_addr + i < self.memory.len() && dst_addr + i < self.memory.len() {
                            self.memory[dst_addr + i] = self.memory[src_addr + i];
                        }
                    }
                } else {
                    for i in 0..copy_len {
                        if src_addr + i < self.memory.len() && dst_addr + i < self.memory.len() {
                            self.memory[dst_addr + i] = self.memory[src_addr + i];
                        }
                    }
                }
            }
            MemoryFill { dst, val, len } => {
                let dst_addr = self.get_reg(*dst).as_u32() as usize;
                let fill_val = self.get_reg(*val).as_i32() as u8;
                let fill_len = self.get_reg(*len).as_u32() as usize;
                for i in 0..fill_len {
                    if dst_addr + i < self.memory.len() {
                        self.memory[dst_addr + i] = fill_val;
                    }
                }
            }
            Mov { dst, src } => {
                let val = self.get_reg(*src);
                self.set_reg(*dst, val);
            }
        }

        true
    }

    pub fn return_value(&self) -> Option<Value> {
        self.return_value
    }
}
