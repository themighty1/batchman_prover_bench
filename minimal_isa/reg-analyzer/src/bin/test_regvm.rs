//! Test register VM against wasmi reference implementation
//!
//! Runs the pure-wasm JSON parser with both:
//! 1. wasmi (reference implementation)
//! 2. Register-based VM with 8 registers
//!
//! Tests json_query function with path query from fixtures.

use anyhow::{Context, Result, anyhow};
use std::collections::HashMap;
use std::fs;
use wasmi::{Engine, Linker, Module, Store, Memory, MemoryType};
use wasmparser::{Parser, Payload};
use reg_analyzer::regvm::{WasmToVReg, VRegInst, VReg, FuncSig};
use reg_analyzer::interpreter::Value;
use reg_analyzer::regalloc::{compute_live_intervals, linear_scan_alloc};

/// Converted function with metadata
struct ConvertedFunc {
    instructions: Vec<VRegInst>,
    num_params: u32,
    num_locals: u32,
}

/// Full module VM state
struct RegVM {
    functions: Vec<ConvertedFunc>,
    func_names: HashMap<String, u32>,
    memory: Vec<u8>,
    globals: Vec<Value>,
}

impl RegVM {
    fn new(memory_pages: u32) -> Self {
        Self {
            functions: Vec::new(),
            func_names: HashMap::new(),
            memory: vec![0u8; (memory_pages as usize) * 65536],
            globals: vec![Value::I32(0); 16],
        }
    }

    fn add_function(&mut self, name: Option<&str>, func: ConvertedFunc) {
        let idx = self.functions.len() as u32;
        if let Some(n) = name {
            self.func_names.insert(n.to_string(), idx);
        }
        self.functions.push(func);
    }

    fn get_func_idx(&self, name: &str) -> Option<u32> {
        self.func_names.get(name).copied()
    }

    fn write_memory(&mut self, offset: usize, data: &[u8]) {
        if offset + data.len() <= self.memory.len() {
            self.memory[offset..offset + data.len()].copy_from_slice(data);
        }
    }

    fn read_memory(&self, offset: usize, len: usize) -> &[u8] {
        &self.memory[offset..offset + len]
    }

    /// Execute a function by index with given arguments
    fn call(&mut self, func_idx: u32, args: &[Value], depth: u32) -> Result<Option<Value>> {
        let debug = false;
        if depth > 1000 {
            return Err(anyhow!("Call stack overflow"));
        }

        let func = &self.functions[func_idx as usize];
        let instructions = func.instructions.clone();

        if debug && depth == 0 {
            eprintln!("DEBUG: Calling func {} with {} args", func_idx, args.len());
            eprintln!("DEBUG: Func has {} params, {} locals, {} instructions",
                func.num_params, func.num_locals, instructions.len());
        }

        let mut regs: HashMap<VReg, Value> = HashMap::new();
        let mut locals: Vec<Value> = vec![Value::I32(0); (func.num_params + func.num_locals) as usize];

        for (i, arg) in args.iter().enumerate() {
            if i < locals.len() {
                locals[i] = *arg;
                if debug && depth == 0 {
                    eprintln!("DEBUG: Initial local[{}] = {:?}", i, arg);
                }
            }
        }

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

        let max_iterations = 50_000_000u64;
        let mut iterations = 0u64;
        let mut pc = 0usize;
        let mut control_stack: Vec<(u32, bool)> = Vec::new();
        let mut last_result: Option<VReg> = None;

        while pc < instructions.len() && iterations < max_iterations {
            iterations += 1;
            let inst = &instructions[pc];

            // Trace key points - disabled for clean output
            let trace_iter = false;

            match inst {
                VRegInst::I32Const { dst, val } => {
                    regs.insert(*dst, Value::I32(*val));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64Const { dst, val } => {
                    regs.insert(*dst, Value::I64(*val));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Add { dst, src1, src2 } => {
                    let a = regs.get(src1).map(|v| v.as_i32()).unwrap_or(0);
                    let b = regs.get(src2).map(|v| v.as_i32()).unwrap_or(0);
                    regs.insert(*dst, Value::I32(a.wrapping_add(b)));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Sub { dst, src1, src2 } => {
                    let a = regs.get(src1).map(|v| v.as_i32()).unwrap_or(0);
                    let b = regs.get(src2).map(|v| v.as_i32()).unwrap_or(0);
                    regs.insert(*dst, Value::I32(a.wrapping_sub(b)));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Mul { dst, src1, src2 } => {
                    let a = regs.get(src1).map(|v| v.as_i32()).unwrap_or(0);
                    let b = regs.get(src2).map(|v| v.as_i32()).unwrap_or(0);
                    regs.insert(*dst, Value::I32(a.wrapping_mul(b)));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32And { dst, src1, src2 } => {
                    let a = regs.get(src1).map(|v| v.as_i32()).unwrap_or(0);
                    let b = regs.get(src2).map(|v| v.as_i32()).unwrap_or(0);
                    regs.insert(*dst, Value::I32(a & b));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Or { dst, src1, src2 } => {
                    let a = regs.get(src1).map(|v| v.as_i32()).unwrap_or(0);
                    let b = regs.get(src2).map(|v| v.as_i32()).unwrap_or(0);
                    regs.insert(*dst, Value::I32(a | b));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Xor { dst, src1, src2 } => {
                    let a = regs.get(src1).map(|v| v.as_i32()).unwrap_or(0);
                    let b = regs.get(src2).map(|v| v.as_i32()).unwrap_or(0);
                    regs.insert(*dst, Value::I32(a ^ b));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Shl { dst, src1, src2 } => {
                    let a = regs.get(src1).map(|v| v.as_i32()).unwrap_or(0);
                    let b = regs.get(src2).map(|v| v.as_i32()).unwrap_or(0);
                    regs.insert(*dst, Value::I32(a.wrapping_shl(b as u32)));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32ShrU { dst, src1, src2 } => {
                    let a = regs.get(src1).map(|v| v.as_u32()).unwrap_or(0);
                    let b = regs.get(src2).map(|v| v.as_i32()).unwrap_or(0);
                    regs.insert(*dst, Value::I32(a.wrapping_shr(b as u32) as i32));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32ShrS { dst, src1, src2 } => {
                    let a = regs.get(src1).map(|v| v.as_i32()).unwrap_or(0);
                    let b = regs.get(src2).map(|v| v.as_i32()).unwrap_or(0);
                    regs.insert(*dst, Value::I32(a.wrapping_shr(b as u32)));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Eqz { dst, src } => {
                    let a = regs.get(src).map(|v| v.as_i32()).unwrap_or(0);
                    regs.insert(*dst, Value::I32(if a == 0 { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Eq { dst, src1, src2 } => {
                    let a = regs.get(src1).map(|v| v.as_i32()).unwrap_or(0);
                    let b = regs.get(src2).map(|v| v.as_i32()).unwrap_or(0);
                    if debug && depth == 0 && trace_iter {
                        eprintln!("  I32Eq: {} == {} -> {}", a, b, if a == b { 1 } else { 0 });
                    }
                    regs.insert(*dst, Value::I32(if a == b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Ne { dst, src1, src2 } => {
                    let a = regs.get(src1).map(|v| v.as_i32()).unwrap_or(0);
                    let b = regs.get(src2).map(|v| v.as_i32()).unwrap_or(0);
                    regs.insert(*dst, Value::I32(if a != b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32LtU { dst, src1, src2 } => {
                    let a = regs.get(src1).map(|v| v.as_u32()).unwrap_or(0);
                    let b = regs.get(src2).map(|v| v.as_u32()).unwrap_or(0);
                    regs.insert(*dst, Value::I32(if a < b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32LtS { dst, src1, src2 } => {
                    let a = regs.get(src1).map(|v| v.as_i32()).unwrap_or(0);
                    let b = regs.get(src2).map(|v| v.as_i32()).unwrap_or(0);
                    regs.insert(*dst, Value::I32(if a < b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32GtU { dst, src1, src2 } => {
                    let a = regs.get(src1).map(|v| v.as_u32()).unwrap_or(0);
                    let b = regs.get(src2).map(|v| v.as_u32()).unwrap_or(0);
                    regs.insert(*dst, Value::I32(if a > b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32GtS { dst, src1, src2 } => {
                    let a = regs.get(src1).map(|v| v.as_i32()).unwrap_or(0);
                    let b = regs.get(src2).map(|v| v.as_i32()).unwrap_or(0);
                    regs.insert(*dst, Value::I32(if a > b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32LeU { dst, src1, src2 } => {
                    let a = regs.get(src1).map(|v| v.as_u32()).unwrap_or(0);
                    let b = regs.get(src2).map(|v| v.as_u32()).unwrap_or(0);
                    regs.insert(*dst, Value::I32(if a <= b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32LeS { dst, src1, src2 } => {
                    let a = regs.get(src1).map(|v| v.as_i32()).unwrap_or(0);
                    let b = regs.get(src2).map(|v| v.as_i32()).unwrap_or(0);
                    regs.insert(*dst, Value::I32(if a <= b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32GeU { dst, src1, src2 } => {
                    let a = regs.get(src1).map(|v| v.as_u32()).unwrap_or(0);
                    let b = regs.get(src2).map(|v| v.as_u32()).unwrap_or(0);
                    regs.insert(*dst, Value::I32(if a >= b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32GeS { dst, src1, src2 } => {
                    let a = regs.get(src1).map(|v| v.as_i32()).unwrap_or(0);
                    let b = regs.get(src2).map(|v| v.as_i32()).unwrap_or(0);
                    regs.insert(*dst, Value::I32(if a >= b { 1 } else { 0 }));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Load { dst, addr, offset } => {
                    let a = regs.get(addr).map(|v| v.as_u32()).unwrap_or(0);
                    let idx = (a + offset) as usize;
                    let val = if idx + 4 <= self.memory.len() {
                        i32::from_le_bytes(self.memory[idx..idx + 4].try_into().unwrap())
                    } else { 0 };
                    regs.insert(*dst, Value::I32(val));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Load8U { dst, addr, offset } => {
                    let a = regs.get(addr).map(|v| v.as_u32()).unwrap_or(0);
                    let idx = (a + offset) as usize;
                    let val = if idx < self.memory.len() { self.memory[idx] as i32 } else { 0 };
                    if debug && depth == 0 && iterations >= 1430 && iterations <= 1950 {
                        eprintln!("  iter={} mem[{}] = {} ('{}')", iterations, idx, val,
                            if val >= 32 && val < 127 { val as u8 as char } else { '?' });
                    }
                    regs.insert(*dst, Value::I32(val));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Load8S { dst, addr, offset } => {
                    let a = regs.get(addr).map(|v| v.as_u32()).unwrap_or(0);
                    let idx = (a + offset) as usize;
                    let val = if idx < self.memory.len() { self.memory[idx] as i8 as i32 } else { 0 };
                    regs.insert(*dst, Value::I32(val));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Load16U { dst, addr, offset } => {
                    let a = regs.get(addr).map(|v| v.as_u32()).unwrap_or(0);
                    let idx = (a + offset) as usize;
                    let val = if idx + 2 <= self.memory.len() {
                        u16::from_le_bytes(self.memory[idx..idx + 2].try_into().unwrap()) as i32
                    } else { 0 };
                    regs.insert(*dst, Value::I32(val));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I32Store { addr, offset, src } => {
                    let a = regs.get(addr).map(|v| v.as_u32()).unwrap_or(0);
                    let val = regs.get(src).map(|v| v.as_i32()).unwrap_or(0);
                    let idx = (a + offset) as usize;
                    if debug && depth == 0 && iterations > 9000 {
                        eprintln!("  iter={} I32Store: mem[{}] = {}", iterations, idx, val);
                    }
                    if idx + 4 <= self.memory.len() {
                        self.memory[idx..idx + 4].copy_from_slice(&val.to_le_bytes());
                    }
                    pc += 1;
                }
                VRegInst::I32Store8 { addr, offset, src } => {
                    let a = regs.get(addr).map(|v| v.as_u32()).unwrap_or(0);
                    let val = regs.get(src).map(|v| v.as_i32()).unwrap_or(0) as u8;
                    let idx = (a + offset) as usize;
                    if debug && depth == 0 && val >= 65 && val <= 122 {
                        eprintln!("  iter={} Store8: mem[{}] = {} ('{}')", iterations, idx, val,
                            val as char);
                    }
                    if idx < self.memory.len() {
                        self.memory[idx] = val;
                    }
                    pc += 1;
                }
                VRegInst::LocalGet { dst, local } => {
                    let val = locals.get(*local as usize).copied().unwrap_or(Value::I32(0));
                    regs.insert(*dst, val);
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::LocalSet { local, src } => {
                    let val = regs.get(src).copied().unwrap_or(Value::I32(0));
                    if (*local as usize) < locals.len() {
                        locals[*local as usize] = val;
                    }
                    pc += 1;
                }
                VRegInst::LocalTee { dst, local, src } => {
                    let val = regs.get(src).copied().unwrap_or(Value::I32(0));
                    if (*local as usize) < locals.len() {
                        locals[*local as usize] = val;
                    }
                    regs.insert(*dst, val);
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::GlobalGet { dst, global } => {
                    let val = self.globals.get(*global as usize).copied().unwrap_or(Value::I32(0));
                    regs.insert(*dst, val);
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::GlobalSet { global, src } => {
                    let val = regs.get(src).copied().unwrap_or(Value::I32(0));
                    if (*global as usize) < self.globals.len() {
                        self.globals[*global as usize] = val;
                    }
                    pc += 1;
                }
                VRegInst::Block { label } => { control_stack.push((*label, false)); pc += 1; }
                VRegInst::Loop { label } => { control_stack.push((*label, true)); pc += 1; }
                VRegInst::If { cond, label } => {
                    let c = regs.get(cond).map(|v| v.as_i32()).unwrap_or(0);
                    control_stack.push((*label, false));
                    if c == 0 {
                        if let Some(&else_pc) = else_positions.get(label) { pc = else_pc + 1; }
                        else if let Some(&end_pc) = block_ends.get(label) { pc = end_pc + 1; }
                        else { pc += 1; }
                    } else { pc += 1; }
                }
                VRegInst::Else { label } => {
                    if let Some(&end_pc) = block_ends.get(label) { pc = end_pc + 1; }
                    else { pc += 1; }
                }
                VRegInst::End { .. } => { control_stack.pop(); pc += 1; }
                VRegInst::Br { label } => {
                    if let Some(&start) = loop_starts.get(label) { pc = start + 1; }
                    else if let Some(&end) = block_ends.get(label) { pc = end + 1; }
                    else { pc += 1; }
                }
                VRegInst::BrIf { cond, label } => {
                    let c = regs.get(cond).map(|v| v.as_i32()).unwrap_or(0);
                    if c != 0 {
                        if let Some(&start) = loop_starts.get(label) { pc = start + 1; }
                        else if let Some(&end) = block_ends.get(label) { pc = end + 1; }
                        else { pc += 1; }
                    } else { pc += 1; }
                }
                VRegInst::BrTable { idx, labels, default } => {
                    let i = regs.get(idx).map(|v| v.as_u32()).unwrap_or(0) as usize;
                    let target = if i < labels.len() { labels[i] } else { *default };
                    if trace_iter {
                        eprintln!("  BrTable idx={} -> target={}", i, target);
                    }
                    if let Some(&start) = loop_starts.get(&target) { pc = start + 1; }
                    else if let Some(&end) = block_ends.get(&target) { pc = end + 1; }
                    else { pc += 1; }
                }
                VRegInst::Return { values } => {
                    if debug && depth == 0 {
                        eprintln!("DEBUG: Return with {} values at pc={}", values.len(), pc);
                        for (i, vreg) in values.iter().enumerate() {
                            eprintln!("  value[{}]: vreg={:?} = {:?}", i, vreg, regs.get(vreg));
                        }
                    }
                    if let Some(vreg) = values.last() { return Ok(regs.get(vreg).copied()); }
                    return Ok(None);
                }
                VRegInst::Call { func_idx: callee, args: call_args, results } => {
                    let arg_vals: Vec<Value> = call_args.iter()
                        .map(|r| regs.get(r).copied().unwrap_or(Value::I32(0)))
                        .collect();
                    if debug && depth == 0 && trace_iter {
                        eprintln!("  Calling func {} with args: {:?}", callee, arg_vals);
                    }
                    if let Ok(ret) = self.call(*callee, &arg_vals, depth + 1) {
                        if let (Some(r), Some(v)) = (results.first(), ret) {
                            if debug && depth == 0 && trace_iter {
                                eprintln!("  Call returned {:?}, storing in {:?}", v, r);
                            }
                            regs.insert(*r, v);
                        }
                    }
                    pc += 1;
                }
                VRegInst::Select { dst, cond, src1, src2 } => {
                    let c = regs.get(cond).map(|v| v.as_i32()).unwrap_or(0);
                    let val = if c != 0 { regs.get(src1).copied().unwrap_or(Value::I32(0)) }
                              else { regs.get(src2).copied().unwrap_or(Value::I32(0)) };
                    if debug && depth == 0 && trace_iter {
                        let v1 = regs.get(src1).map(|v| v.as_i32()).unwrap_or(0);
                        let v2 = regs.get(src2).map(|v| v.as_i32()).unwrap_or(0);
                        eprintln!("  Select: cond={}, src1={}, src2={}, result={}", c, v1, v2, val.as_i32());
                    }
                    regs.insert(*dst, val);
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::Unreachable => { return Err(anyhow!("Unreachable executed")); }
                VRegInst::Nop => { pc += 1; }
                VRegInst::Drop { .. } => { pc += 1; }
                VRegInst::I64Add { dst, src1, src2 } => {
                    let a = regs.get(src1).map(|v| v.as_i64()).unwrap_or(0);
                    let b = regs.get(src2).map(|v| v.as_i64()).unwrap_or(0);
                    regs.insert(*dst, Value::I64(a.wrapping_add(b)));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64Sub { dst, src1, src2 } => {
                    let a = regs.get(src1).map(|v| v.as_i64()).unwrap_or(0);
                    let b = regs.get(src2).map(|v| v.as_i64()).unwrap_or(0);
                    regs.insert(*dst, Value::I64(a.wrapping_sub(b)));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64Shl { dst, src1, src2 } => {
                    let a = regs.get(src1).map(|v| v.as_i64()).unwrap_or(0);
                    let b = regs.get(src2).map(|v| v.as_i64()).unwrap_or(0);
                    regs.insert(*dst, Value::I64(a.wrapping_shl(b as u32)));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64Or { dst, src1, src2 } => {
                    let a = regs.get(src1).map(|v| v.as_i64()).unwrap_or(0);
                    let b = regs.get(src2).map(|v| v.as_i64()).unwrap_or(0);
                    regs.insert(*dst, Value::I64(a | b));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64And { dst, src1, src2 } => {
                    let a = regs.get(src1).map(|v| v.as_i64()).unwrap_or(0);
                    let b = regs.get(src2).map(|v| v.as_i64()).unwrap_or(0);
                    regs.insert(*dst, Value::I64(a & b));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::I64ExtendI32U { dst, src } => {
                    let a = regs.get(src).map(|v| v.as_u32()).unwrap_or(0);
                    regs.insert(*dst, Value::I64(a as i64));
                    last_result = Some(*dst);
                    pc += 1;
                }
                VRegInst::MemoryCopy { dst, src, len } => {
                    let dst_addr = regs.get(dst).map(|v| v.as_u32()).unwrap_or(0) as usize;
                    let src_addr = regs.get(src).map(|v| v.as_u32()).unwrap_or(0) as usize;
                    let copy_len = regs.get(len).map(|v| v.as_u32()).unwrap_or(0) as usize;
                    if debug && depth == 0 {
                        eprintln!("  MemoryCopy: src={} dst={} len={}", src_addr, dst_addr, copy_len);
                    }
                    // Copy bytes (handle overlapping regions)
                    if src_addr < dst_addr && src_addr + copy_len > dst_addr {
                        // Overlapping, copy backwards
                        for i in (0..copy_len).rev() {
                            if src_addr + i < self.memory.len() && dst_addr + i < self.memory.len() {
                                self.memory[dst_addr + i] = self.memory[src_addr + i];
                            }
                        }
                    } else {
                        // Non-overlapping or forward copy
                        for i in 0..copy_len {
                            if src_addr + i < self.memory.len() && dst_addr + i < self.memory.len() {
                                self.memory[dst_addr + i] = self.memory[src_addr + i];
                            }
                        }
                    }
                    pc += 1;
                }
                VRegInst::MemoryFill { dst, val, len } => {
                    let dst_addr = regs.get(dst).map(|v| v.as_u32()).unwrap_or(0) as usize;
                    let fill_val = regs.get(val).map(|v| v.as_i32()).unwrap_or(0) as u8;
                    let fill_len = regs.get(len).map(|v| v.as_u32()).unwrap_or(0) as usize;
                    for i in 0..fill_len {
                        if dst_addr + i < self.memory.len() {
                            self.memory[dst_addr + i] = fill_val;
                        }
                    }
                    pc += 1;
                }
                _ => { pc += 1; }
            }
        }

        if iterations >= max_iterations {
            return Err(anyhow!("Exceeded max iterations ({} at pc={})", iterations, pc));
        }

        if debug && depth == 0 {
            eprintln!("DEBUG: Fell through at pc={} after {} iterations", pc, iterations);
            eprintln!("DEBUG: last_result={:?}", last_result);
            if let Some(vreg) = last_result {
                eprintln!("DEBUG: last_result value={:?}", regs.get(&vreg));
            }
        }

        if let Some(vreg) = last_result { return Ok(regs.get(&vreg).copied()); }
        Ok(None)
    }
}

fn load_module(wasm_bytes: &[u8]) -> Result<RegVM> {
    let mut vm = RegVM::new(256);
    let mut func_types: Vec<wasmparser::FuncType> = Vec::new();
    let mut type_indices: Vec<u32> = Vec::new();
    let mut func_names: HashMap<u32, String> = HashMap::new();
    let mut code_bodies: Vec<wasmparser::FunctionBody> = Vec::new();
    let mut global_inits: Vec<i32> = Vec::new();

    // First pass: collect types, function indices, exports, globals, and code bodies
    for payload in Parser::new(0).parse_all(wasm_bytes) {
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
            Payload::FunctionSection(reader) => { for func in reader.clone() { type_indices.push(func?); } }
            Payload::GlobalSection(reader) => {
                for global in reader.clone() {
                    let global = global?;
                    // Parse the init expression to get the initial value
                    let init_expr = global.init_expr.get_binary_reader();
                    let mut init_val = 0i32;
                    for op in wasmparser::OperatorsReader::new(init_expr) {
                        if let Ok(wasmparser::Operator::I32Const { value }) = op {
                            init_val = value;
                            break;
                        }
                    }
                    global_inits.push(init_val);
                }
            }
            Payload::ExportSection(reader) => {
                for export in reader.clone() {
                    let export = export?;
                    if let wasmparser::ExternalKind::Func = export.kind {
                        func_names.insert(export.index, export.name.to_string());
                    }
                }
            }
            Payload::CodeSectionEntry(body) => {
                code_bodies.push(body.clone());
            }
            _ => {}
        }
    }

    // Initialize globals with parsed values
    for (i, val) in global_inits.iter().enumerate() {
        if i < vm.globals.len() {
            vm.globals[i] = Value::I32(*val);
        }
    }

    // Build function signature table for all functions
    let func_sigs: Vec<FuncSig> = type_indices.iter().map(|&type_idx| {
        let func_type = func_types.get(type_idx as usize);
        let num_params = func_type.map(|ft| ft.params().len() as u32).unwrap_or(0);
        let num_results = func_type.map(|ft| ft.results().len() as u32).unwrap_or(0);
        (num_params, num_results)
    }).collect();

    // Second pass: convert functions with access to all signatures
    for (func_count, body) in code_bodies.iter().enumerate() {
        let type_idx = type_indices.get(func_count).copied().unwrap_or(0);
        let func_type = func_types.get(type_idx as usize);
        let num_params = func_type.map(|ft| ft.params().len() as u32).unwrap_or(0);
        let mut num_locals = 0u32;
        for local in body.get_locals_reader()? { let (count, _) = local?; num_locals += count; }
        let mut converter = WasmToVReg::new_with_sigs(num_params, num_locals, func_sigs.clone());
        for op in body.get_operators_reader()? { converter.convert_op(&op?); }
        let name = func_names.get(&(func_count as u32)).map(|s| s.as_str());
        vm.add_function(name, ConvertedFunc { instructions: converter.instructions, num_params, num_locals });
    }

    Ok(vm)
}

fn run_wasmi_query(wasm_bytes: &[u8], json_data: &[u8], query: &[u8]) -> Result<String> {
    let config = wasmi::Config::default();
    let engine = Engine::new(&config);
    let module = Module::new(&engine, wasm_bytes).context("Failed to parse WASM module")?;
    let mut store = Store::new(&engine, ());
    let mut linker = Linker::new(&engine);
    let memory_type = MemoryType::new(1, Some(256)).unwrap();
    let memory = Memory::new(&mut store, memory_type).unwrap();
    linker.define("env", "memory", memory.clone())?;
    let instance = linker.instantiate(&mut store, &module)?.start(&mut store)?;
    let memory = instance.get_memory(&store, "memory").unwrap_or(memory);

    // Layout: [0..json_len] = json, [json_len..json_len+query_len] = query, [output_offset..] = output
    let json_ptr = 0u32;
    let query_ptr = json_data.len() as u32;
    let output_ptr = query_ptr + query.len() as u32 + 100; // some padding

    memory.write(&mut store, json_ptr as usize, json_data)?;
    memory.write(&mut store, query_ptr as usize, query)?;

    let json_query_fn = instance.get_func(&store, "json_query")
        .ok_or_else(|| anyhow!("No json_query function"))?;

    let result_len = json_query_fn.typed::<(i32, i32, i32, i32, i32), i32>(&store)?
        .call(&mut store, (json_ptr as i32, json_data.len() as i32, query_ptr as i32, query.len() as i32, output_ptr as i32))?;

    if result_len == 0 {
        return Ok("(not found)".to_string());
    }

    let mut output = vec![0u8; result_len as usize];
    memory.read(&store, output_ptr as usize, &mut output)?;
    Ok(String::from_utf8_lossy(&output).to_string())
}

fn main() -> Result<()> {
    let json_data = include_bytes!("../../../guest-programs/json-query/fixtures/test_input.json");
    let query = include_bytes!("../../../guest-programs/json-query/fixtures/query.txt");
    let expected_result = "Bob Smith";

    println!("=== JSON Query Test (8 registers) ===\n");
    println!("JSON fixture: {} bytes", json_data.len());
    println!("Query: {}", String::from_utf8_lossy(query));
    println!("Expected: \"{}\"\n", expected_result);

    let wasm_path = "../pure-wasm/target/wasm32-unknown-unknown/release/pure_json_wasm.wasm";
    let wasm_bytes = fs::read(wasm_path).context("Failed to read WASM file")?;
    println!("WASM size: {} bytes\n", wasm_bytes.len());

    // Trim query
    let query_trimmed: Vec<u8> = query.iter().copied().take_while(|&c| c != b'\n' && c != b'\r').collect();

    // Test with wasmi
    println!("--- wasmi (reference) ---");
    let wasmi_result = run_wasmi_query(&wasm_bytes, json_data, &query_trimmed)?;
    println!("Result: \"{}\"", wasmi_result);
    let wasmi_pass = wasmi_result == expected_result;
    println!("Check: {} (expected \"{}\")\n", if wasmi_pass { "PASS" } else { "FAIL" }, expected_result);

    // Convert to register IR
    println!("--- Register VM (8 regs) ---");
    let mut vm = load_module(&wasm_bytes)?;
    println!("Converted {} functions", vm.functions.len());

    let mut total_insts = 0usize;
    let mut total_spills = 0u32;
    for func in vm.functions.iter() {
        total_insts += func.instructions.len();
        let intervals = compute_live_intervals(&func.instructions);
        let alloc = linear_scan_alloc(&intervals, 8);
        total_spills += alloc.spilled.len() as u32;
    }
    println!("Total instructions: {}", total_insts);
    println!("Spilled vregs (8 regs): {}", total_spills);

    // Set up memory layout
    let json_ptr = 0u32;
    let query_ptr = json_data.len() as u32;
    let output_ptr = query_ptr + query.len() as u32 + 100;

    vm.write_memory(json_ptr as usize, json_data);
    vm.write_memory(query_ptr as usize, query);

    if let Some(func_idx) = vm.get_func_idx("json_query") {
        // Trim any trailing whitespace from query
        let query_trimmed: Vec<u8> = query.iter().copied().take_while(|&c| c != b'\n' && c != b'\r').collect();
        println!("Query trimmed len: {}", query_trimmed.len());

        // Re-write memory with trimmed query
        vm.write_memory(query_ptr as usize, &query_trimmed);

        let args = vec![
            Value::I32(json_ptr as i32),
            Value::I32(json_data.len() as i32),
            Value::I32(query_ptr as i32),
            Value::I32(query_trimmed.len() as i32),
            Value::I32(output_ptr as i32),
        ];

        match vm.call(func_idx, &args, 0) {
            Ok(Some(result)) => {
                let result_len = result.as_i32() as usize;
                if result_len == 0 {
                    println!("Result: (not found)");
                } else {
                    let output = vm.read_memory(output_ptr as usize, result_len);
                    let regvm_result = String::from_utf8_lossy(output).to_string();
                    println!("Result: \"{}\"", regvm_result);
                    let regvm_pass = regvm_result == expected_result;
                    println!("Check: {} (expected \"{}\")\n", if regvm_pass { "PASS" } else { "FAIL" }, expected_result);

                    println!("=== Final Comparison ===");
                    if wasmi_result == regvm_result && regvm_result == expected_result {
                        println!("PASS: Both wasmi and RegVM return correct result \"{}\"", expected_result);
                    } else {
                        println!("FAIL: Results differ");
                        println!("  wasmi:    \"{}\"", wasmi_result);
                        println!("  regvm:    \"{}\"", regvm_result);
                        println!("  expected: \"{}\"", expected_result);
                    }
                }
            }
            Ok(None) => println!("RegVM returned no value"),
            Err(e) => println!("RegVM error: {}", e),
        }
    } else {
        println!("json_query function not found");
    }

    Ok(())
}
