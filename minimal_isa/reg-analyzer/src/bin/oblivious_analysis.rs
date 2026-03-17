//! Analyze execution trace for oblivious computation potential
//!
//! Categorizes operations by input-dependence:
//! - Input-independent: Always same regardless of input data
//! - Input-dependent: Varies based on input values
//!
//! For zkVM: revealing input-dependent operations leaks information

use anyhow::{Context, Result, anyhow};
use std::collections::HashMap;
use std::fs;
use wasmparser::{Parser, Payload};
use reg_analyzer::regvm::{WasmToVReg, VRegInst, VReg, FuncSig};
use reg_analyzer::interpreter::Value;

/// Taint tracking: which values depend on input
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Taint {
    Clean,      // Input-independent (constants, static addresses)
    Tainted,    // Input-dependent
}

/// Execution trace entry
#[derive(Debug)]
struct TraceEntry {
    pc: usize,
    op_type: &'static str,
    is_branch: bool,
    branch_taken: Option<bool>,
    memory_addr: Option<u32>,
    memory_is_read: bool,
    taint: Taint,
}

/// Trace collector that runs VRegInst and tracks taint
struct TaintTracer {
    /// Register taint status
    vreg_taint: HashMap<VReg, Taint>,
    /// Local variable taint
    local_taint: HashMap<u32, Taint>,
    /// Global variable taint
    global_taint: HashMap<u32, Taint>,
    /// Memory taint by address (simplified - tracks ranges)
    memory_taint: HashMap<u32, Taint>,
    /// Execution trace
    trace: Vec<TraceEntry>,
    /// Register values for execution
    vreg_values: HashMap<VReg, Value>,
    /// Local values
    locals: Vec<Value>,
    /// Global values
    globals: Vec<Value>,
    /// Memory
    memory: Vec<u8>,
    /// Input memory range (tainted)
    input_start: u32,
    input_end: u32,
}

impl TaintTracer {
    fn new(memory_pages: usize, input_start: u32, input_end: u32) -> Self {
        Self {
            vreg_taint: HashMap::new(),
            local_taint: HashMap::new(),
            global_taint: HashMap::new(),
            memory_taint: HashMap::new(),
            trace: Vec::new(),
            vreg_values: HashMap::new(),
            locals: Vec::new(),
            globals: vec![Value::I32(0); 16],
            memory: vec![0u8; memory_pages * 65536],
            input_start,
            input_end,
        }
    }

    fn get_vreg(&self, vreg: VReg) -> Value {
        self.vreg_values.get(&vreg).copied().unwrap_or(Value::I32(0))
    }

    fn set_vreg(&mut self, vreg: VReg, val: Value) {
        self.vreg_values.insert(vreg, val);
    }

    fn get_taint(&self, vreg: VReg) -> Taint {
        self.vreg_taint.get(&vreg).copied().unwrap_or(Taint::Clean)
    }

    fn set_taint(&mut self, vreg: VReg, taint: Taint) {
        self.vreg_taint.insert(vreg, taint);
    }

    fn combine_taint(&self, t1: Taint, t2: Taint) -> Taint {
        if t1 == Taint::Tainted || t2 == Taint::Tainted {
            Taint::Tainted
        } else {
            Taint::Clean
        }
    }

    fn is_input_addr(&self, addr: u32) -> bool {
        addr >= self.input_start && addr < self.input_end
    }

    fn get_memory_taint(&self, addr: u32) -> Taint {
        if self.is_input_addr(addr) {
            Taint::Tainted
        } else {
            self.memory_taint.get(&addr).copied().unwrap_or(Taint::Clean)
        }
    }

    fn classify_op(inst: &VRegInst) -> &'static str {
        use VRegInst::*;
        match inst {
            I32Const { .. } | I64Const { .. } => "const",
            I32Add { .. } | I32Sub { .. } | I64Add { .. } | I64Sub { .. } => "arith",
            I32Mul { .. } | I64Mul { .. } | I32DivS { .. } | I32DivU { .. } |
            I32RemS { .. } | I32RemU { .. } | I64DivS { .. } | I64DivU { .. } |
            I64RemS { .. } | I64RemU { .. } => "arith",
            I32And { .. } | I32Or { .. } | I32Xor { .. } |
            I64And { .. } | I64Or { .. } | I64Xor { .. } => "bitwise",
            I32Shl { .. } | I32ShrU { .. } | I32ShrS { .. } |
            I32Rotl { .. } | I32Rotr { .. } |
            I64Shl { .. } | I64ShrU { .. } | I64ShrS { .. } => "shift",
            I32Eq { .. } | I32Ne { .. } | I32LtS { .. } | I32LtU { .. } |
            I32GtS { .. } | I32GtU { .. } | I32LeS { .. } | I32LeU { .. } |
            I32GeS { .. } | I32GeU { .. } |
            I64Eq { .. } | I64Ne { .. } | I64LtS { .. } | I64LtU { .. } |
            I64GtS { .. } | I64GtU { .. } | I64LeS { .. } | I64LeU { .. } |
            I64GeS { .. } | I64GeU { .. } => "compare",
            I32Eqz { .. } | I64Eqz { .. } | I32Clz { .. } | I32Ctz { .. } |
            I32Popcnt { .. } | I64Clz { .. } | I64Ctz { .. } => "unary",
            I32WrapI64 { .. } | I64ExtendI32S { .. } | I64ExtendI32U { .. } |
            I32Extend8S { .. } | I32Extend16S { .. } => "convert",
            I32Load { .. } | I64Load { .. } | I32Load8U { .. } | I32Load8S { .. } |
            I32Load16U { .. } | I32Load16S { .. } | I64Load8U { .. } | I64Load8S { .. } |
            I64Load16U { .. } | I64Load16S { .. } | I64Load32U { .. } | I64Load32S { .. } => "load",
            I32Store { .. } | I64Store { .. } | I32Store8 { .. } | I32Store16 { .. } |
            I64Store8 { .. } | I64Store16 { .. } | I64Store32 { .. } => "store",
            LocalGet { .. } => "local.get",
            LocalSet { .. } => "local.set",
            LocalTee { .. } => "local.tee",
            GlobalGet { .. } | GlobalSet { .. } => "global",
            Call { .. } | CallIndirect { .. } => "call",
            Block { .. } | Loop { .. } | End { .. } => "block",
            If { .. } => "if",
            Else { .. } => "else",
            Br { .. } => "br",
            BrIf { .. } => "br_if",
            BrTable { .. } => "br_table",
            Return { .. } => "return",
            Select { .. } => "select",
            Unreachable | Nop => "nop",
            Drop { .. } => "drop",
            MemorySize { .. } | MemoryGrow { .. } | MemoryCopy { .. } | MemoryFill { .. } => "memory",
            Mov { .. } => "mov",
        }
    }

    /// Execute and trace, returning the result
    fn execute_traced(&mut self, instructions: &[VRegInst]) -> Option<Value> {
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

        let max_iterations = 10_000_000u64;
        let mut iterations = 0u64;
        let mut pc = 0usize;
        let mut last_result: Option<VReg> = None;

        while pc < instructions.len() && iterations < max_iterations {
            iterations += 1;
            let inst = &instructions[pc];
            let op_type = Self::classify_op(inst);

            let mut entry = TraceEntry {
                pc,
                op_type,
                is_branch: false,
                branch_taken: None,
                memory_addr: None,
                memory_is_read: false,
                taint: Taint::Clean,
            };

            match inst {
                VRegInst::I32Const { dst, val } => {
                    self.set_vreg(*dst, Value::I32(*val));
                    self.set_taint(*dst, Taint::Clean);
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64Const { dst, val } => {
                    self.set_vreg(*dst, Value::I64(*val));
                    self.set_taint(*dst, Taint::Clean);
                    last_result = Some(*dst);
                    pc += 1;
                }

                VRegInst::I32Add { dst, src1, src2 } => {
                    let a = self.get_vreg(*src1).as_i32();
                    let b = self.get_vreg(*src2).as_i32();
                    self.set_vreg(*dst, Value::I32(a.wrapping_add(b)));
                    let taint = self.combine_taint(self.get_taint(*src1), self.get_taint(*src2));
                    self.set_taint(*dst, taint);
                    entry.taint = taint;
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Sub { dst, src1, src2 } => {
                    let a = self.get_vreg(*src1).as_i32();
                    let b = self.get_vreg(*src2).as_i32();
                    self.set_vreg(*dst, Value::I32(a.wrapping_sub(b)));
                    let taint = self.combine_taint(self.get_taint(*src1), self.get_taint(*src2));
                    self.set_taint(*dst, taint);
                    entry.taint = taint;
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Mul { dst, src1, src2 } => {
                    let a = self.get_vreg(*src1).as_i32();
                    let b = self.get_vreg(*src2).as_i32();
                    self.set_vreg(*dst, Value::I32(a.wrapping_mul(b)));
                    let taint = self.combine_taint(self.get_taint(*src1), self.get_taint(*src2));
                    self.set_taint(*dst, taint);
                    entry.taint = taint;
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32And { dst, src1, src2 } => {
                    let a = self.get_vreg(*src1).as_i32();
                    let b = self.get_vreg(*src2).as_i32();
                    self.set_vreg(*dst, Value::I32(a & b));
                    let taint = self.combine_taint(self.get_taint(*src1), self.get_taint(*src2));
                    self.set_taint(*dst, taint);
                    entry.taint = taint;
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Or { dst, src1, src2 } => {
                    let a = self.get_vreg(*src1).as_i32();
                    let b = self.get_vreg(*src2).as_i32();
                    self.set_vreg(*dst, Value::I32(a | b));
                    let taint = self.combine_taint(self.get_taint(*src1), self.get_taint(*src2));
                    self.set_taint(*dst, taint);
                    entry.taint = taint;
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Shl { dst, src1, src2 } => {
                    let a = self.get_vreg(*src1).as_i32();
                    let b = self.get_vreg(*src2).as_i32();
                    self.set_vreg(*dst, Value::I32(a.wrapping_shl(b as u32)));
                    let taint = self.combine_taint(self.get_taint(*src1), self.get_taint(*src2));
                    self.set_taint(*dst, taint);
                    entry.taint = taint;
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32ShrU { dst, src1, src2 } => {
                    let a = self.get_vreg(*src1).as_u32();
                    let b = self.get_vreg(*src2).as_i32();
                    self.set_vreg(*dst, Value::I32(a.wrapping_shr(b as u32) as i32));
                    let taint = self.combine_taint(self.get_taint(*src1), self.get_taint(*src2));
                    self.set_taint(*dst, taint);
                    entry.taint = taint;
                    last_result = Some(*dst);
                    pc += 1;
                }

                VRegInst::I32Eq { dst, src1, src2 } => {
                    let a = self.get_vreg(*src1).as_i32();
                    let b = self.get_vreg(*src2).as_i32();
                    self.set_vreg(*dst, Value::I32(if a == b { 1 } else { 0 }));
                    let taint = self.combine_taint(self.get_taint(*src1), self.get_taint(*src2));
                    self.set_taint(*dst, taint);
                    entry.taint = taint;
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Ne { dst, src1, src2 } => {
                    let a = self.get_vreg(*src1).as_i32();
                    let b = self.get_vreg(*src2).as_i32();
                    self.set_vreg(*dst, Value::I32(if a != b { 1 } else { 0 }));
                    let taint = self.combine_taint(self.get_taint(*src1), self.get_taint(*src2));
                    self.set_taint(*dst, taint);
                    entry.taint = taint;
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32LtU { dst, src1, src2 } => {
                    let a = self.get_vreg(*src1).as_u32();
                    let b = self.get_vreg(*src2).as_u32();
                    self.set_vreg(*dst, Value::I32(if a < b { 1 } else { 0 }));
                    let taint = self.combine_taint(self.get_taint(*src1), self.get_taint(*src2));
                    self.set_taint(*dst, taint);
                    entry.taint = taint;
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32LtS { dst, src1, src2 } => {
                    let a = self.get_vreg(*src1).as_i32();
                    let b = self.get_vreg(*src2).as_i32();
                    self.set_vreg(*dst, Value::I32(if a < b { 1 } else { 0 }));
                    let taint = self.combine_taint(self.get_taint(*src1), self.get_taint(*src2));
                    self.set_taint(*dst, taint);
                    entry.taint = taint;
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32GtU { dst, src1, src2 } => {
                    let a = self.get_vreg(*src1).as_u32();
                    let b = self.get_vreg(*src2).as_u32();
                    self.set_vreg(*dst, Value::I32(if a > b { 1 } else { 0 }));
                    let taint = self.combine_taint(self.get_taint(*src1), self.get_taint(*src2));
                    self.set_taint(*dst, taint);
                    entry.taint = taint;
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32GtS { dst, src1, src2 } => {
                    let a = self.get_vreg(*src1).as_i32();
                    let b = self.get_vreg(*src2).as_i32();
                    self.set_vreg(*dst, Value::I32(if a > b { 1 } else { 0 }));
                    let taint = self.combine_taint(self.get_taint(*src1), self.get_taint(*src2));
                    self.set_taint(*dst, taint);
                    entry.taint = taint;
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32LeU { dst, src1, src2 } => {
                    let a = self.get_vreg(*src1).as_u32();
                    let b = self.get_vreg(*src2).as_u32();
                    self.set_vreg(*dst, Value::I32(if a <= b { 1 } else { 0 }));
                    let taint = self.combine_taint(self.get_taint(*src1), self.get_taint(*src2));
                    self.set_taint(*dst, taint);
                    entry.taint = taint;
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32GeU { dst, src1, src2 } => {
                    let a = self.get_vreg(*src1).as_u32();
                    let b = self.get_vreg(*src2).as_u32();
                    self.set_vreg(*dst, Value::I32(if a >= b { 1 } else { 0 }));
                    let taint = self.combine_taint(self.get_taint(*src1), self.get_taint(*src2));
                    self.set_taint(*dst, taint);
                    entry.taint = taint;
                    last_result = Some(*dst);
                    pc += 1;
                }

                VRegInst::I32Eqz { dst, src } => {
                    let a = self.get_vreg(*src).as_i32();
                    self.set_vreg(*dst, Value::I32(if a == 0 { 1 } else { 0 }));
                    self.set_taint(*dst, self.get_taint(*src));
                    entry.taint = self.get_taint(*src);
                    last_result = Some(*dst);
                    pc += 1;
                }

                VRegInst::I32Load { dst, addr, offset } => {
                    let a = self.get_vreg(*addr).as_u32();
                    let mem_addr = a + offset;
                    let idx = mem_addr as usize;
                    let val = if idx + 4 <= self.memory.len() {
                        i32::from_le_bytes(self.memory[idx..idx + 4].try_into().unwrap())
                    } else { 0 };
                    self.set_vreg(*dst, Value::I32(val));

                    // Taint: address taint + memory content taint
                    let addr_taint = self.get_taint(*addr);
                    let mem_taint = self.get_memory_taint(mem_addr);
                    let taint = self.combine_taint(addr_taint, mem_taint);
                    self.set_taint(*dst, taint);

                    entry.memory_addr = Some(mem_addr);
                    entry.memory_is_read = true;
                    entry.taint = addr_taint; // Address determines if access pattern leaks
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Load8U { dst, addr, offset } => {
                    let a = self.get_vreg(*addr).as_u32();
                    let mem_addr = a + offset;
                    let idx = mem_addr as usize;
                    let val = if idx < self.memory.len() { self.memory[idx] as i32 } else { 0 };
                    self.set_vreg(*dst, Value::I32(val));

                    let addr_taint = self.get_taint(*addr);
                    let mem_taint = self.get_memory_taint(mem_addr);
                    let taint = self.combine_taint(addr_taint, mem_taint);
                    self.set_taint(*dst, taint);

                    entry.memory_addr = Some(mem_addr);
                    entry.memory_is_read = true;
                    entry.taint = addr_taint;
                    last_result = Some(*dst);
                    pc += 1;
                }

                VRegInst::I32Store { addr, offset, src } => {
                    let a = self.get_vreg(*addr).as_u32();
                    let val = self.get_vreg(*src).as_i32();
                    let mem_addr = a + offset;
                    let idx = mem_addr as usize;
                    if idx + 4 <= self.memory.len() {
                        self.memory[idx..idx + 4].copy_from_slice(&val.to_le_bytes());
                    }

                    // Store taint in memory
                    let addr_taint = self.get_taint(*addr);
                    let val_taint = self.get_taint(*src);
                    self.memory_taint.insert(mem_addr, val_taint);

                    entry.memory_addr = Some(mem_addr);
                    entry.memory_is_read = false;
                    entry.taint = addr_taint;
                    pc += 1;
                }
                VRegInst::I32Store8 { addr, offset, src } => {
                    let a = self.get_vreg(*addr).as_u32();
                    let val = self.get_vreg(*src).as_i32() as u8;
                    let mem_addr = a + offset;
                    let idx = mem_addr as usize;
                    if idx < self.memory.len() {
                        self.memory[idx] = val;
                    }

                    let addr_taint = self.get_taint(*addr);
                    let val_taint = self.get_taint(*src);
                    self.memory_taint.insert(mem_addr, val_taint);

                    entry.memory_addr = Some(mem_addr);
                    entry.memory_is_read = false;
                    entry.taint = addr_taint;
                    pc += 1;
                }

                VRegInst::LocalGet { dst, local } => {
                    let val = self.locals.get(*local as usize).copied().unwrap_or(Value::I32(0));
                    self.set_vreg(*dst, val);
                    let taint = self.local_taint.get(local).copied().unwrap_or(Taint::Clean);
                    self.set_taint(*dst, taint);
                    entry.taint = taint;
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::LocalSet { local, src } => {
                    let val = self.get_vreg(*src);
                    if (*local as usize) < self.locals.len() {
                        self.locals[*local as usize] = val;
                    }
                    self.local_taint.insert(*local, self.get_taint(*src));
                    entry.taint = self.get_taint(*src);
                    pc += 1;
                }
                VRegInst::LocalTee { dst, local, src } => {
                    let val = self.get_vreg(*src);
                    if (*local as usize) < self.locals.len() {
                        self.locals[*local as usize] = val;
                    }
                    self.set_vreg(*dst, val);
                    let taint = self.get_taint(*src);
                    self.local_taint.insert(*local, taint);
                    self.set_taint(*dst, taint);
                    entry.taint = taint;
                    last_result = Some(*dst);
                    pc += 1;
                }

                VRegInst::GlobalGet { dst, global } => {
                    let val = self.globals.get(*global as usize).copied().unwrap_or(Value::I32(0));
                    self.set_vreg(*dst, val);
                    let taint = self.global_taint.get(global).copied().unwrap_or(Taint::Clean);
                    self.set_taint(*dst, taint);
                    entry.taint = taint;
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::GlobalSet { global, src } => {
                    let val = self.get_vreg(*src);
                    if (*global as usize) < self.globals.len() {
                        self.globals[*global as usize] = val;
                    }
                    self.global_taint.insert(*global, self.get_taint(*src));
                    entry.taint = self.get_taint(*src);
                    pc += 1;
                }

                VRegInst::Block { .. } | VRegInst::Loop { .. } | VRegInst::End { .. } => {
                    pc += 1;
                }

                VRegInst::If { cond, label } => {
                    let c = self.get_vreg(*cond).as_i32();
                    let cond_taint = self.get_taint(*cond);

                    entry.is_branch = true;
                    entry.branch_taken = Some(c != 0);
                    entry.taint = cond_taint;

                    if c == 0 {
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
                    if let Some(&end_pc) = block_ends.get(label) {
                        pc = end_pc + 1;
                    } else {
                        pc += 1;
                    }
                }
                VRegInst::Br { label } => {
                    entry.is_branch = true;
                    entry.branch_taken = Some(true);

                    if let Some(&start) = loop_starts.get(label) {
                        pc = start + 1;
                    } else if let Some(&end) = block_ends.get(label) {
                        pc = end + 1;
                    } else {
                        pc += 1;
                    }
                }
                VRegInst::BrIf { cond, label } => {
                    let c = self.get_vreg(*cond).as_i32();
                    let cond_taint = self.get_taint(*cond);

                    entry.is_branch = true;
                    entry.branch_taken = Some(c != 0);
                    entry.taint = cond_taint;

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
                    let i = self.get_vreg(*idx).as_u32() as usize;
                    let idx_taint = self.get_taint(*idx);
                    let target = if i < labels.len() { labels[i] } else { *default };

                    entry.is_branch = true;
                    entry.branch_taken = Some(true);
                    entry.taint = idx_taint;

                    if let Some(&start) = loop_starts.get(&target) {
                        pc = start + 1;
                    } else if let Some(&end) = block_ends.get(&target) {
                        pc = end + 1;
                    } else {
                        pc += 1;
                    }
                }

                VRegInst::Return { values } => {
                    self.trace.push(entry);
                    if let Some(vreg) = values.last() {
                        return Some(self.get_vreg(*vreg));
                    }
                    return Some(Value::I32(0));
                }

                VRegInst::Select { dst, cond, src1, src2 } => {
                    let c = self.get_vreg(*cond).as_i32();
                    let val = if c != 0 { self.get_vreg(*src1) } else { self.get_vreg(*src2) };
                    self.set_vreg(*dst, val);
                    let taint = self.combine_taint(
                        self.get_taint(*cond),
                        self.combine_taint(self.get_taint(*src1), self.get_taint(*src2))
                    );
                    self.set_taint(*dst, taint);
                    entry.taint = self.get_taint(*cond); // Select based on tainted condition
                    last_result = Some(*dst);
                    pc += 1;
                }

                VRegInst::Call { .. } => {
                    // Simplified: skip function calls for now
                    pc += 1;
                }

                _ => {
                    pc += 1;
                }
            }

            self.trace.push(entry);
        }

        // Fall through
        if let Some(vreg) = last_result {
            Some(self.get_vreg(vreg))
        } else {
            Some(Value::I32(0))
        }
    }
}

/// Converted function with VReg IR
struct ConvertedFunc {
    instructions: Vec<VRegInst>,
    num_params: u32,
    num_locals: u32,
}

fn main() -> Result<()> {
    let wasm_path = "../pure-wasm/target/wasm32-unknown-unknown/release/pure_json_wasm.wasm";
    let wasm_bytes = fs::read(wasm_path).context("Failed to read WASM file")?;

    // Parse WASM - convert to VReg during initial pass
    let mut func_types: Vec<wasmparser::FuncType> = Vec::new();
    let mut type_indices: Vec<u32> = Vec::new();
    let mut func_names: HashMap<String, u32> = HashMap::new();
    let mut converted_funcs: Vec<ConvertedFunc> = Vec::new();
    let mut func_sigs: Vec<FuncSig> = Vec::new();

    // First pass: collect types and function indices
    for payload in Parser::new(0).parse_all(&wasm_bytes) {
        let payload = payload?;
        match &payload {
            Payload::TypeSection(reader) => {
                for rec_group in reader.clone() {
                    let rec_group = rec_group?;
                    for sub_type in rec_group.types() {
                        if let wasmparser::CompositeInnerType::Func(ft) = &sub_type.composite_type.inner {
                            func_types.push(ft.clone());
                        }
                    }
                }
            }
            Payload::FunctionSection(reader) => {
                for func in reader.clone() { type_indices.push(func?); }
            }
            Payload::ExportSection(reader) => {
                for export in reader.clone() {
                    let export = export?;
                    if let wasmparser::ExternalKind::Func = export.kind {
                        func_names.insert(export.name.to_string(), export.index);
                    }
                }
            }
            _ => {}
        }
    }

    // Build function signatures
    func_sigs = type_indices.iter().map(|&type_idx| {
        let func_type = func_types.get(type_idx as usize);
        let num_params = func_type.map(|ft| ft.params().len() as u32).unwrap_or(0);
        let num_results = func_type.map(|ft| ft.results().len() as u32).unwrap_or(0);
        (num_params, num_results)
    }).collect();

    // Second pass: convert function bodies
    for payload in Parser::new(0).parse_all(&wasm_bytes) {
        let payload = payload?;
        if let Payload::CodeSectionEntry(body) = payload {
            let func_count = converted_funcs.len();
            let type_idx = type_indices.get(func_count).copied().unwrap_or(0);
            let func_type = func_types.get(type_idx as usize);
            let num_params = func_type.map(|ft| ft.params().len() as u32).unwrap_or(0);

            let mut num_locals = 0u32;
            for local in body.get_locals_reader()? {
                let (count, _) = local?;
                num_locals += count;
            }

            // Convert to VReg IR
            let mut converter = WasmToVReg::new_with_sigs(num_params, num_locals, func_sigs.clone());
            for op in body.get_operators_reader()? {
                converter.convert_op(&op?);
            }

            converted_funcs.push(ConvertedFunc {
                instructions: converter.instructions,
                num_params,
                num_locals,
            });
        }
    }

    // Find json_query function
    let func_idx = func_names.get("json_query")
        .ok_or_else(|| anyhow!("json_query not found"))?;

    let func = &converted_funcs[*func_idx as usize];

    // Set up test data
    let json_data = br#"{"company":{"departments":{"engineering":{"teams":{"frontend":{"lead":"Bob Smith"}}}}}}"#;
    let query = b"company.departments.engineering.teams.frontend.lead";

    let json_ptr = 0u32;
    let query_ptr = json_data.len() as u32;
    let output_ptr = query_ptr + query.len() as u32 + 100;

    // Input range: JSON data and query are both input
    let input_start = 0u32;
    let input_end = query_ptr + query.len() as u32;

    // Create tracer
    let mut tracer = TaintTracer::new(256, input_start, input_end);

    // Initialize memory with test data
    tracer.memory[json_ptr as usize..json_ptr as usize + json_data.len()].copy_from_slice(json_data);
    tracer.memory[query_ptr as usize..query_ptr as usize + query.len()].copy_from_slice(query);

    // Initialize locals (function parameters)
    let total_locals = (func.num_params + func.num_locals) as usize;
    tracer.locals = vec![Value::I32(0); total_locals];
    tracer.locals[0] = Value::I32(json_ptr as i32);      // json_ptr - TAINTED
    tracer.locals[1] = Value::I32(json_data.len() as i32); // json_len - TAINTED (reveals size)
    tracer.locals[2] = Value::I32(query_ptr as i32);     // query_ptr - could be clean if fixed
    tracer.locals[3] = Value::I32(query.len() as i32);   // query_len - TAINTED
    tracer.locals[4] = Value::I32(output_ptr as i32);    // output_ptr - clean

    // Taint the parameters that depend on input
    tracer.local_taint.insert(0, Taint::Tainted);  // json_ptr points to input
    tracer.local_taint.insert(1, Taint::Tainted);  // json_len is input-dependent
    tracer.local_taint.insert(2, Taint::Tainted);  // query_ptr points to input
    tracer.local_taint.insert(3, Taint::Tainted);  // query_len is input-dependent
    tracer.local_taint.insert(4, Taint::Clean);    // output_ptr is fixed

    // Initialize global 0 (stack pointer) - clean
    tracer.globals[0] = Value::I32(1048576);
    tracer.global_taint.insert(0, Taint::Clean);

    // Execute with tracing
    println!("=== Oblivious Computation Analysis for zkVM ===\n");
    println!("Analyzing: json_query function");
    println!("Input: {} byte JSON, {} byte query\n", json_data.len(), query.len());

    let result = tracer.execute_traced(&func.instructions);

    // Analyze trace
    let total_ops = tracer.trace.len();
    let mut tainted_ops = 0usize;
    let mut clean_ops = 0usize;
    let mut tainted_branches = 0usize;
    let mut clean_branches = 0usize;
    let mut tainted_mem_reads = 0usize;
    let mut clean_mem_reads = 0usize;
    let mut tainted_mem_writes = 0usize;
    let mut clean_mem_writes = 0usize;

    let mut op_stats: HashMap<&'static str, (usize, usize)> = HashMap::new(); // (clean, tainted)

    for entry in &tracer.trace {
        let stat = op_stats.entry(entry.op_type).or_insert((0, 0));
        if entry.taint == Taint::Tainted {
            tainted_ops += 1;
            stat.1 += 1;
        } else {
            clean_ops += 1;
            stat.0 += 1;
        }

        if entry.is_branch {
            if entry.taint == Taint::Tainted {
                tainted_branches += 1;
            } else {
                clean_branches += 1;
            }
        }

        if let Some(_addr) = entry.memory_addr {
            if entry.memory_is_read {
                if entry.taint == Taint::Tainted {
                    tainted_mem_reads += 1;
                } else {
                    clean_mem_reads += 1;
                }
            } else {
                if entry.taint == Taint::Tainted {
                    tainted_mem_writes += 1;
                } else {
                    clean_mem_writes += 1;
                }
            }
        }
    }

    println!("=== Execution Trace Summary ===\n");
    println!("Total operations executed: {}", total_ops);
    println!();
    println!("{:<15} {:>10} {:>10} {:>10}", "Category", "Clean", "Tainted", "% Tainted");
    println!("{}", "=".repeat(50));
    println!("{:<15} {:>10} {:>10} {:>9.1}%", "All ops", clean_ops, tainted_ops,
             (tainted_ops as f64 / total_ops as f64) * 100.0);
    println!("{:<15} {:>10} {:>10} {:>9.1}%", "Branches", clean_branches, tainted_branches,
             if clean_branches + tainted_branches > 0 {
                 (tainted_branches as f64 / (clean_branches + tainted_branches) as f64) * 100.0
             } else { 0.0 });
    println!("{:<15} {:>10} {:>10} {:>9.1}%", "Mem reads", clean_mem_reads, tainted_mem_reads,
             if clean_mem_reads + tainted_mem_reads > 0 {
                 (tainted_mem_reads as f64 / (clean_mem_reads + tainted_mem_reads) as f64) * 100.0
             } else { 0.0 });
    println!("{:<15} {:>10} {:>10} {:>9.1}%", "Mem writes", clean_mem_writes, tainted_mem_writes,
             if clean_mem_writes + tainted_mem_writes > 0 {
                 (tainted_mem_writes as f64 / (clean_mem_writes + tainted_mem_writes) as f64) * 100.0
             } else { 0.0 });

    println!("\n=== Per-Operation Type Breakdown ===\n");
    println!("{:<15} {:>10} {:>10} {:>10}", "Op Type", "Clean", "Tainted", "% Tainted");
    println!("{}", "-".repeat(50));

    let mut sorted_stats: Vec<_> = op_stats.into_iter().collect();
    sorted_stats.sort_by_key(|(_, (c, t))| std::cmp::Reverse(c + t));

    for (op, (clean, tainted)) in &sorted_stats {
        let total = clean + tainted;
        let pct = if total > 0 { (*tainted as f64 / total as f64) * 100.0 } else { 0.0 };
        println!("{:<15} {:>10} {:>10} {:>9.1}%", op, clean, tainted, pct);
    }

    println!("\n=== zkVM Implications ===\n");

    println!("1. CONTROL FLOW LEAKAGE:");
    if tainted_branches > 0 {
        println!("   ⚠️  {} branches depend on input data", tainted_branches);
        println!("   Revealing branch decisions leaks information about:");
        println!("   - JSON structure (which characters are seen)");
        println!("   - Query path (which keys are matched)");
        println!("   - String lengths and array sizes");
    } else {
        println!("   ✓ All branches are input-independent");
    }

    println!("\n2. MEMORY ACCESS PATTERN LEAKAGE:");
    let total_mem = tainted_mem_reads + tainted_mem_writes + clean_mem_reads + clean_mem_writes;
    if tainted_mem_reads + tainted_mem_writes > 0 {
        println!("   ⚠️  {} of {} memory accesses have input-dependent addresses",
                 tainted_mem_reads + tainted_mem_writes, total_mem);
        println!("   Revealing memory indices leaks:");
        println!("   - Position in JSON being read");
        println!("   - Output location (reveals result length)");
    } else {
        println!("   ✓ All memory accesses have fixed addresses");
    }

    println!("\n3. OBLIVIOUS EXECUTION FEASIBILITY:");
    let tainted_pct = (tainted_ops as f64 / total_ops as f64) * 100.0;
    if tainted_pct > 50.0 {
        println!("   ❌ {:.1}% of execution is input-dependent", tainted_pct);
        println!("   This program is NOT suitable for non-oblivious segments");
        println!("   Recommendation: Use fully oblivious execution or");
        println!("   restructure to have input-independent control flow");
    } else if tainted_pct > 10.0 {
        println!("   ⚠️  {:.1}% of execution is input-dependent", tainted_pct);
        println!("   Partial oblivious execution may be possible");
        println!("   Consider: ORAM for memory, dummy branches for control flow");
    } else {
        println!("   ✓ Only {:.1}% of execution is input-dependent", tainted_pct);
        println!("   Good candidate for mixed oblivious/non-oblivious execution");
    }

    if let Some(val) = result {
        println!("\n4. RESULT:");
        println!("   Return value: {} (result string length)", val.as_i32());
    }

    Ok(())
}
