//! Compare test-wasm functions: wasmi (reference) vs canonical Ra2FuncMulti pipeline.
//!
//! Pipeline: WASM → VRegInst → unconstrained alloc → Ra2FuncMulti + regalloc2
//!           → rewrite → execute_vreg
//!
//! Tests simple functions (no memory) and memory-based functions
//! (matmul, bubble sort, sieve, gcd).

use anyhow::{Result, anyhow};
use std::collections::{HashMap, HashSet};
use std::fs;
use wasmi::{Engine, Linker, Module, Store, Memory, MemoryType};
use wasmparser::{Parser, Payload};
use reg_analyzer::regvm::{
    WasmToVReg, VRegInst, FuncSig, SlotType,
    GLOBALS_MEM_BASE, FRAME_SP_ADDR, FRAME_STACK_BASE, SLOT_SIZE,
};
use reg_analyzer::regalloc::{compute_live_intervals, linear_scan_alloc, RegAllocResult};
use reg_analyzer::preg_vm::{PRegVM, specialized_opcode, vreg_inst_name};
use reg_analyzer::interpreter::Value;
use reg_analyzer::ra2_cfg::{
    Ra2FuncMulti, rewrite_with_ra2, validate_rewrite,
    make_machine_env, make_regalloc_opts,
};
use regalloc2 as ra2;

fn val_type_to_slot(vt: &wasmparser::ValType) -> SlotType {
    match vt {
        wasmparser::ValType::I64 => SlotType::I64,
        _ => SlotType::I32,
    }
}

/// Parsed WASM module ready for pipeline execution (memory-lowered mode)
struct PipelineModule {
    /// Per-function: (VRegInst stream, unconstrained alloc, num_params, num_locals)
    vreg_funcs: Vec<(Vec<VRegInst>, RegAllocResult, u32, u32)>,
    export_to_func: HashMap<String, u32>,
    global_inits: Vec<i32>,
    data_segments: Vec<(u32, Vec<u8>)>,
    num_regs: u32,
}

impl PipelineModule {
    fn load(wasm_bytes: &[u8], num_regs: u32) -> Result<Self> {
        let mut func_types: Vec<wasmparser::FuncType> = Vec::new();
        let mut type_indices: Vec<u32> = Vec::new();
        let mut export_to_func: HashMap<String, u32> = HashMap::new();
        let mut code_bodies: Vec<wasmparser::FunctionBody> = Vec::new();
        let mut global_inits: Vec<i32> = Vec::new();
        let mut global_types: Vec<SlotType> = Vec::new();
        let mut data_segments: Vec<(u32, Vec<u8>)> = Vec::new();

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
                Payload::FunctionSection(reader) => {
                    for func in reader.clone() { type_indices.push(func?); }
                }
                Payload::GlobalSection(reader) => {
                    for global in reader.clone() {
                        let global = global?;
                        let init_expr = global.init_expr.get_binary_reader();
                        let mut init_val = 0i32;
                        for op in wasmparser::OperatorsReader::new(init_expr) {
                            if let Ok(wasmparser::Operator::I32Const { value }) = op {
                                init_val = value;
                                break;
                            }
                        }
                        global_inits.push(init_val);
                        global_types.push(val_type_to_slot(&global.ty.content_type));
                    }
                }
                Payload::ExportSection(reader) => {
                    for export in reader.clone() {
                        let export = export?;
                        if let wasmparser::ExternalKind::Func = export.kind {
                            export_to_func.insert(export.name.to_string(), export.index);
                        }
                    }
                }
                Payload::CodeSectionEntry(body) => {
                    code_bodies.push(body.clone());
                }
                Payload::DataSection(reader) => {
                    for data in reader.clone() {
                        let data = data?;
                        if let wasmparser::DataKind::Active { memory_index: 0, offset_expr } = data.kind {
                            let mut offset = 0u32;
                            for op in wasmparser::OperatorsReader::new(offset_expr.get_binary_reader()) {
                                if let Ok(wasmparser::Operator::I32Const { value }) = op {
                                    offset = value as u32;
                                    break;
                                }
                            }
                            data_segments.push((offset, data.data.to_vec()));
                        }
                    }
                }
                _ => {}
            }
        }

        let func_sigs: Vec<FuncSig> = type_indices.iter().map(|&type_idx| {
            let func_type = func_types.get(type_idx as usize);
            let num_params = func_type.map(|ft| ft.params().len() as u32).unwrap_or(0);
            let num_results = func_type.map(|ft| ft.results().len() as u32).unwrap_or(0);
            (num_params, num_results)
        }).collect();

        let mut vreg_funcs = Vec::new();
        for (func_count, body) in code_bodies.iter().enumerate() {
            let type_idx = type_indices.get(func_count).copied().unwrap_or(0);
            let func_type = func_types.get(type_idx as usize);
            let num_params = func_type.map(|ft| ft.params().len() as u32).unwrap_or(0);
            let mut num_locals = 0u32;
            let mut local_types: Vec<SlotType> = Vec::new();
            if let Some(ft) = func_type {
                for p in ft.params() { local_types.push(val_type_to_slot(p)); }
            }
            for local in body.get_locals_reader()? {
                let (count, vt) = local?;
                num_locals += count;
                let st = val_type_to_slot(&vt);
                for _ in 0..count { local_types.push(st); }
            }
            let mut converter = WasmToVReg::new_memory_lowered(
                num_params, num_locals, func_sigs.clone(),
                local_types, global_types.clone(),
            );
            for op in body.get_operators_reader()? {
                converter.convert_op(&op?);
            }
            let intervals = compute_live_intervals(&converter.instructions);
            let alloc = linear_scan_alloc(&intervals, num_regs);
            vreg_funcs.push((converter.instructions, alloc, num_params, num_locals));
        }

        Ok(Self { vreg_funcs, export_to_func, global_inits, data_segments, num_regs })
    }

    /// Initialize a PRegVM with memory-lowered globals/frame regions and data segments
    fn init_vm(&self, vm: &mut PRegVM) {
        for (i, val) in self.global_inits.iter().enumerate() {
            if i < vm.globals.len() {
                vm.globals[i] = Value::I32(*val);
            }
        }
        for (offset, data) in &self.data_segments {
            vm.write_memory(*offset as usize, data);
        }
        // Memory-lowered mode: set up frame stack and globals memory region
        vm.write_memory(FRAME_SP_ADDR as usize, &FRAME_STACK_BASE.to_le_bytes());
        for (i, val) in self.global_inits.iter().enumerate() {
            let addr = GLOBALS_MEM_BASE as usize + (i as u32 * SLOT_SIZE) as usize;
            vm.write_memory(addr, &(*val as u32).to_le_bytes());
        }
    }

    /// Run Ra2FuncMulti + regalloc2 on each function. Returns per-function results:
    /// Ok((rewritten_insts, new_alloc)) or Err(reason) for fallback.
    fn run_ra2(&self) -> Vec<Result<(Vec<VRegInst>, RegAllocResult), String>> {
        let empty_isa: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        let env = make_machine_env(self.num_regs);
        let opts = make_regalloc_opts();
        let mut results = Vec::new();

        for (insts, alloc, _, _) in &self.vreg_funcs {
            match Ra2FuncMulti::build(insts, self.num_regs, &empty_isa, alloc) {
                None => {
                    results.push(Err("BrTable or empty".to_string()));
                }
                Some(ra2_func) => {
                    if ra2_func.num_vregs == 0 {
                        results.push(Err("no vregs".to_string()));
                        continue;
                    }
                    match ra2::run(&ra2_func, &env, &opts) {
                        Ok(output) => {
                            let (new_insts, new_alloc) = rewrite_with_ra2(insts, &ra2_func, &output, alloc);
                            results.push(Ok((new_insts, new_alloc)));
                        }
                        Err(e) => {
                            results.push(Err(format!("regalloc2: {}", e)));
                        }
                    }
                }
            }
        }
        results
    }

    /// Create a fresh PRegVM with Ra2-rewritten functions loaded (memory-lowered).
    /// Falls back to unconstrained alloc for functions that failed Ra2.
    fn make_ra2_vm(&self, memory_pages: usize, ra2_results: &[Result<(Vec<VRegInst>, RegAllocResult), String>]) -> PRegVM {
        let mut max_spills = 0u32;
        for (_, alloc, _, _) in &self.vreg_funcs {
            max_spills = max_spills.max(alloc.num_spill_slots);
        }
        for r in ra2_results {
            if let Ok((_, alloc)) = r {
                max_spills = max_spills.max(alloc.num_spill_slots);
            }
        }
        // ML mode needs memory up to FRAME_STACK_BASE (0x800200+), so at least 256 pages
        let ml_pages = 256usize;
        let pages = memory_pages.max(ml_pages);
        let mut vm = PRegVM::new(self.num_regs as usize, max_spills as usize + 128, pages);
        for (fi, (orig_insts, orig_alloc, np, nl)) in self.vreg_funcs.iter().enumerate() {
            if let Some(Ok((new_insts, new_alloc))) = ra2_results.get(fi) {
                vm.add_vreg_function_ml(new_insts.clone(), new_alloc.clone(), *np, *nl);
            } else {
                vm.add_vreg_function_ml(orig_insts.clone(), orig_alloc.clone(), *np, *nl);
            }
        }
        self.init_vm(&mut vm);
        vm
    }

    fn func_idx(&self, name: &str) -> Result<u32> {
        self.export_to_func.get(name).copied()
            .ok_or_else(|| anyhow!("function {} not found", name))
    }
}

fn main() -> Result<()> {
    let wasm_path = "../test-wasm/target/wasm32-unknown-unknown/release/test_wasm.wasm";
    let wasm_bytes = fs::read(wasm_path)?;
    let num_regs: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3);

    println!("=== test-wasm: wasmi vs Ra2FuncMulti pipeline ({} regs) ===\n", num_regs);

    let module = PipelineModule::load(&wasm_bytes, num_regs)?;
    let mut pass = 0u32;
    let mut fail = 0u32;

    let check = |name: &str, args_str: &str, wasmi: i32, pipeline: i32, expected: i32,
                 pass: &mut u32, fail: &mut u32| {
        let ok = wasmi == expected && pipeline == expected && wasmi == pipeline;
        let status = if ok { "PASS" } else { "FAIL" };
        if ok { *pass += 1; } else { *fail += 1; }
        println!("{} {}({}): wasmi={}, pipeline={}, expected={}",
            status, name, args_str, wasmi, pipeline, expected);
    };

    // ===== Step 1-4: Ra2FuncMulti + regalloc2 allocation =====
    println!("--- Ra2FuncMulti: regalloc2 allocation (no ISA constraints) ---");
    let ra2_results = module.run_ra2();
    for (name, idx) in &module.export_to_func {
        let fi = *idx as usize;
        match &ra2_results[fi] {
            Ok((new_insts, new_alloc)) => {
                let unmapped = validate_rewrite(new_insts, new_alloc);
                let (orig_insts, _, _, _) = &module.vreg_funcs[fi];
                println!("  OK  {}: {} blocks, {} → {} insts, {} unmapped",
                    name,
                    Ra2FuncMulti::build(orig_insts, num_regs, &HashMap::new(),
                        &module.vreg_funcs[fi].1).map(|f| f.cfg_blocks.len()).unwrap_or(0),
                    orig_insts.len(), new_insts.len(), unmapped);
            }
            Err(reason) => {
                println!("  SKIP {}: {}", name, reason);
            }
        }
    }

    // ===== ISA specialization: bake register assignments into opcodes =====
    println!("\n--- ISA specialization (registers baked into opcodes) ---");
    {
        let mut global_isa: HashSet<String> = HashSet::new();
        let mut sorted_names: Vec<&String> = module.export_to_func.keys().collect();
        sorted_names.sort();
        for name in &sorted_names {
            let fi = module.export_to_func[*name] as usize;
            if let Some(Ok((new_insts, new_alloc))) = ra2_results.get(fi) {
                let mut func_isa: HashSet<String> = HashSet::new();
                let mut total_data = 0u32;
                for inst in new_insts {
                    let base = vreg_inst_name(inst);
                    // Skip control flow markers — they don't become ISA opcodes
                    match base {
                        "block" | "loop" | "end" | "br" | "br_if" | "if" | "else"
                        | "return" | "unreachable" | "nop" => continue,
                        _ => {}
                    }
                    let spec = specialized_opcode(inst, new_alloc);
                    func_isa.insert(spec.clone());
                    global_isa.insert(spec);
                    total_data += 1;
                }
                println!("  {}: {} data insts, {} unique opcodes",
                    name, total_data, func_isa.len());
            }
        }
        let mut isa_sorted: Vec<&String> = global_isa.iter().collect();
        isa_sorted.sort();
        println!("\n  Total unique ISA opcodes: {}", global_isa.len());
        println!("  ----");
        for op in &isa_sorted {
            println!("  {}", op);
        }
    }

    // ===== Step 5-6: Execute rewritten code =====
    let simple_tests: Vec<(&str, Vec<i32>, i32)> = vec![
        ("test_const", vec![], 42),
        ("test_add", vec![], 42),
        ("test_branch", vec![15], 16),
        ("test_branch", vec![5], 10),
        ("test_loop", vec![5], 15),
        ("test_loop", vec![10], 55),
        ("test_loop", vec![100], 5050),
        ("test_nested", vec![50], 150),
        ("test_nested", vec![150], 50),
        ("test_nested", vec![250], 50),
        ("test_call", vec![], 57),
    ];

    println!("\n--- Ra2FuncMulti: execute rewritten (simple) ---");
    for (func_name, args, expected) in &simple_tests {
        let wasmi_result = run_wasmi_simple(&wasm_bytes, func_name, args)?;

        let mut vm = module.make_ra2_vm(1, &ra2_results);
        let call_args: Vec<Value> = args.iter().map(|a| Value::I32(*a)).collect();
        let func_idx = module.func_idx(func_name)?;
        let ra2_result = vm.call_func_vreg(func_idx, &call_args)
            .map(|v| v.as_i32()).unwrap_or(-999);

        let args_str = format!("{:?}", args);
        check(func_name, &args_str, wasmi_result, ra2_result, *expected,
              &mut pass, &mut fail);
    }

    // ===== Matrix multiply =====
    println!("\n--- Matrix multiply (3×3) ---");
    {
        let n = 3u32;
        let a_off = 0u32;
        let b_off = 9u32;
        let c_off = 18u32;
        let a_data: [u32; 9] = [1,0,0, 0,2,0, 0,0,3];
        let b_data: [u32; 9] = [1,2,3, 4,5,6, 7,8,9];
        let expected = 108i32;

        let wasmi_result = run_wasmi_matmul(&wasm_bytes, n, a_off, b_off, c_off, &a_data, &b_data)?;

        let mut vm = module.make_ra2_vm(16, &ra2_results);
        for (i, v) in a_data.iter().enumerate() {
            vm.write_memory((a_off as usize + i) * 4, &v.to_le_bytes());
        }
        for (i, v) in b_data.iter().enumerate() {
            vm.write_memory((b_off as usize + i) * 4, &v.to_le_bytes());
        }
        let func_idx = module.func_idx("test_matmul")?;
        let ra2_result = vm.call_func_vreg(func_idx, &[
            Value::I32(0), Value::I32(n as i32),
            Value::I32(a_off as i32), Value::I32(b_off as i32), Value::I32(c_off as i32),
        ]).map(|v| v.as_i32()).unwrap_or(-999);

        check("test_matmul", "3×3 A*B", wasmi_result, ra2_result, expected,
              &mut pass, &mut fail);
    }

    {
        let n = 4u32;
        let a_off = 0u32;
        let b_off = 16u32;
        let c_off = 32u32;
        let a_data: Vec<u32> = vec![1; 16];
        let b_data: Vec<u32> = vec![1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
        let expected = 16i32;

        let wasmi_result = run_wasmi_matmul(&wasm_bytes, n, a_off, b_off, c_off, &a_data, &b_data)?;

        let mut vm = module.make_ra2_vm(16, &ra2_results);
        for (i, v) in a_data.iter().enumerate() {
            vm.write_memory((a_off as usize + i) * 4, &v.to_le_bytes());
        }
        for (i, v) in b_data.iter().enumerate() {
            vm.write_memory((b_off as usize + i) * 4, &v.to_le_bytes());
        }
        let func_idx = module.func_idx("test_matmul")?;
        let ra2_result = vm.call_func_vreg(func_idx, &[
            Value::I32(0), Value::I32(n as i32),
            Value::I32(a_off as i32), Value::I32(b_off as i32), Value::I32(c_off as i32),
        ]).map(|v| v.as_i32()).unwrap_or(-999);

        check("test_matmul", "4×4 ones*identity", wasmi_result, ra2_result, expected,
              &mut pass, &mut fail);
    }

    // ===== Bubble sort =====
    println!("\n--- Bubble sort ---");
    {
        let data: Vec<u32> = vec![5, 3, 8, 1, 9, 2, 7, 4, 6, 10];
        let expected = 385i32;
        let wasmi_result = run_wasmi_array(&wasm_bytes, "test_bubble_sort", &data)?;

        let mut vm = module.make_ra2_vm(16, &ra2_results);
        for (i, v) in data.iter().enumerate() {
            vm.write_memory(i * 4, &v.to_le_bytes());
        }
        let func_idx = module.func_idx("test_bubble_sort")?;
        let ra2_result = vm.call_func_vreg(func_idx, &[
            Value::I32(0), Value::I32(data.len() as i32),
        ]).map(|v| v.as_i32()).unwrap_or(-999);

        check("test_bubble_sort", "10 elements", wasmi_result, ra2_result, expected,
              &mut pass, &mut fail);
    }

    {
        let data: Vec<u32> = vec![100, 50, 75, 25, 1, 99, 42, 88, 3, 67, 55, 12, 91, 33, 77, 8];
        let wasmi_result = run_wasmi_array(&wasm_bytes, "test_bubble_sort", &data)?;

        let mut vm = module.make_ra2_vm(16, &ra2_results);
        for (i, v) in data.iter().enumerate() {
            vm.write_memory(i * 4, &v.to_le_bytes());
        }
        let func_idx = module.func_idx("test_bubble_sort")?;
        let ra2_result = vm.call_func_vreg(func_idx, &[
            Value::I32(0), Value::I32(data.len() as i32),
        ]).map(|v| v.as_i32()).unwrap_or(-999);

        check("test_bubble_sort", "16 elements", wasmi_result, ra2_result, wasmi_result,
              &mut pass, &mut fail);
    }

    // ===== Sieve =====
    println!("\n--- Sieve of Eratosthenes ---");
    for &n in &[10u32, 100, 1000] {
        let expected = match n { 10 => 4, 100 => 25, 1000 => 168, _ => 0 };
        let wasmi_result = run_wasmi_sieve(&wasm_bytes, n)?;

        let mut vm = module.make_ra2_vm(16, &ra2_results);
        let func_idx = module.func_idx("test_sieve")?;
        let ra2_result = vm.call_func_vreg(func_idx, &[
            Value::I32(0), Value::I32(n as i32),
        ]).map(|v| v.as_i32()).unwrap_or(-999);

        check("test_sieve", &format!("n={}", n), wasmi_result, ra2_result, expected,
              &mut pass, &mut fail);
    }

    // ===== GCD sum =====
    println!("\n--- GCD sum ---");
    {
        let data: Vec<u32> = vec![12, 8, 6, 15, 10];
        let expected = 14i32;
        let wasmi_result = run_wasmi_array(&wasm_bytes, "test_gcd_sum", &data)?;

        let mut vm = module.make_ra2_vm(16, &ra2_results);
        for (i, v) in data.iter().enumerate() {
            vm.write_memory(i * 4, &v.to_le_bytes());
        }
        let func_idx = module.func_idx("test_gcd_sum")?;
        let ra2_result = vm.call_func_vreg(func_idx, &[
            Value::I32(0), Value::I32(data.len() as i32),
        ]).map(|v| v.as_i32()).unwrap_or(-999);

        check("test_gcd_sum", "[12,8,6,15,10]", wasmi_result, ra2_result, expected,
              &mut pass, &mut fail);
    }

    {
        let data: Vec<u32> = vec![100, 75, 50, 125, 1000, 625, 48, 36, 24, 60];
        let wasmi_result = run_wasmi_array(&wasm_bytes, "test_gcd_sum", &data)?;

        let mut vm = module.make_ra2_vm(16, &ra2_results);
        for (i, v) in data.iter().enumerate() {
            vm.write_memory(i * 4, &v.to_le_bytes());
        }
        let func_idx = module.func_idx("test_gcd_sum")?;
        let ra2_result = vm.call_func_vreg(func_idx, &[
            Value::I32(0), Value::I32(data.len() as i32),
        ]).map(|v| v.as_i32()).unwrap_or(-999);

        check("test_gcd_sum", "10 large values", wasmi_result, ra2_result, wasmi_result,
              &mut pass, &mut fail);
    }

    // ===== parse_json_deep =====
    println!("\n--- parse_json_deep ---");
    {
        let test_cases: Vec<(&str, &[u8], u32)> = vec![
            ("empty", b"", 0),
            ("no strings", b"{}", 0),
            ("one key-value", br#"{"a":1}"#, 1),           // 1 string: "a"
            ("two pairs", br#"{"a":1,"b":2}"#, 2),         // 2 strings: "a","b"
            ("nested", br#"{"a":{"b":"c"}}"#, 3),          // 3 strings: "a","b","c"
            ("array of strings", br#"["x","y","z"]"#, 3),  // 3 strings: "x","y","z"
            ("realistic", br#"{"name":"Alice","age":30,"items":["pen","book"]}"#, 6),
        ];

        for (label, json_bytes, expected) in &test_cases {
            // wasmi
            let wasmi_result = run_wasmi_array_u8(&wasm_bytes, "parse_json_deep", json_bytes)?;

            // Ra2FuncMulti pipeline
            let mut vm = module.make_ra2_vm(16, &ra2_results);
            for (i, &byte) in json_bytes.iter().enumerate() {
                vm.write_memory(i, &[byte]);
            }
            let func_idx = module.func_idx("parse_json_deep")?;
            let ra2_result = vm.call_func_vreg(func_idx, &[
                Value::I32(0), Value::I32(json_bytes.len() as i32),
            ]).map(|v| v.as_i32()).unwrap_or(-999);

            check("parse_json_deep", label, wasmi_result, ra2_result, *expected as i32,
                  &mut pass, &mut fail);
        }
    }

    println!("\n=== test-wasm: {} passed, {} failed ===", pass, fail);
    let test_wasm_fail = fail;

    // ========================================================================
    // json-crate-wasm: real JSON parser (json = "0.12" compiled to WASM)
    // ========================================================================
    let json_wasm_path = "../json-crate-wasm/target/wasm32-unknown-unknown/release/json_crate_wasm.wasm";
    if let Ok(json_wasm_bytes) = fs::read(json_wasm_path) {
        println!("\n\n=== json-crate-wasm: wasmi vs Ra2FuncMulti pipeline ({} regs) ===\n", num_regs);

        let json_module = PipelineModule::load(&json_wasm_bytes, num_regs)?;
        let mut jpass = 0u32;
        let mut jfail = 0u32;

        println!("  {} functions, {} exports, {} globals, {} data segments",
            json_module.vreg_funcs.len(),
            json_module.export_to_func.len(),
            json_module.global_inits.len(),
            json_module.data_segments.len());

        // Ra2FuncMulti allocation
        println!("\n--- Ra2FuncMulti: regalloc2 allocation ---");
        let json_ra2 = json_module.run_ra2();
        let mut ra2_ok = 0u32;
        let mut ra2_skip = 0u32;
        for (fi, r) in json_ra2.iter().enumerate() {
            match r {
                Ok((new_insts, new_alloc)) => {
                    let unmapped = validate_rewrite(new_insts, new_alloc);
                    if unmapped > 0 {
                        let name = json_module.export_to_func.iter()
                            .find(|(_, &idx)| idx as usize == fi)
                            .map(|(n, _)| n.as_str()).unwrap_or("?");
                        println!("  WARN func {} ({}): {} unmapped", fi, name, unmapped);
                    }
                    ra2_ok += 1;
                }
                Err(reason) => {
                    ra2_skip += 1;
                    if ra2_skip <= 5 {
                        println!("  SKIP func {}: {}", fi, reason);
                    }
                }
            }
        }
        println!("  {} ok, {} skipped", ra2_ok, ra2_skip);

        // ISA specialization
        println!("\n--- ISA specialization ---");
        {
            let mut global_isa: HashSet<String> = HashSet::new();
            let mut total_data = 0u32;
            for (fi, r) in json_ra2.iter().enumerate() {
                if let Ok((new_insts, new_alloc)) = r {
                    for inst in new_insts {
                        let base = vreg_inst_name(inst);
                        match base {
                            "block" | "loop" | "end" | "br" | "br_if" | "if" | "else"
                            | "return" | "unreachable" | "nop" => continue,
                            _ => {}
                        }
                        let spec = specialized_opcode(inst, new_alloc);
                        global_isa.insert(spec);
                        total_data += 1;
                    }
                }
            }
            println!("  {} data instructions, {} unique ISA opcodes (program size vs ISA size)",
                total_data, global_isa.len());
            let mut isa_sorted: Vec<&String> = global_isa.iter().collect();
            isa_sorted.sort();
            for op in &isa_sorted {
                println!("    {}", op);
            }
        }

        // Execute: compare wasmi vs pipeline
        println!("\n--- Execute: wasmi vs Ra2FuncMulti ---");

        let test_jsons: Vec<(&str, String)> = vec![
            ("simple", r#"{"a":1,"b":"hello"}"#.to_string()),
            ("nested", r#"{"a":{"b":{"c":1}}}"#.to_string()),
            ("array", r#"[1,2,3,"x","y"]"#.to_string()),
            ("2KB generated", generate_test_json(2048)),
        ];

        for (label, json_str) in &test_jsons {
            let json_bytes = json_str.as_bytes();

            // wasmi reference
            let wasmi_result = run_wasmi_json_crate(&json_wasm_bytes, json_bytes)?;

            // Ra2FuncMulti pipeline
            let mut vm = json_module.make_ra2_vm(256, &json_ra2);
            vm.write_memory(0, json_bytes);
            if let Some(&sp_init) = json_module.global_inits.first() {
                let addr = GLOBALS_MEM_BASE as usize;
                vm.write_memory(addr, &(sp_init as u32).to_le_bytes());
            }
            vm.enable_trace_log();
            let func_idx = json_module.func_idx("parse_json_deep")?;
            let ra2_result = vm.call_func_vreg(func_idx, &[
                Value::I32(0), Value::I32(json_bytes.len() as i32),
            ]).map(|v| v.as_i32()).unwrap_or(-999);
            let trace_len = vm.trace_log.as_ref().map(|t| t.len()).unwrap_or(0);

            let ok = wasmi_result == ra2_result;
            let status = if ok { "PASS" } else { "FAIL" };
            if ok { jpass += 1; } else { jfail += 1; }
            println!("{} parse_json_deep({}, {} bytes): wasmi={}, pipeline={}, trace={}",
                status, label, json_bytes.len(), wasmi_result, ra2_result, trace_len);
        }

        println!("\n=== json-crate-wasm: {} passed, {} failed ===", jpass, jfail);
        fail += jfail;
    } else {
        println!("\n(skipping json-crate-wasm: {} not found)", json_wasm_path);
    }

    if fail + test_wasm_fail > 0 { std::process::exit(1); }
    Ok(())
}

// ===== wasmi helpers =====

fn run_wasmi_simple(wasm_bytes: &[u8], func_name: &str, args: &[i32]) -> Result<i32> {
    let engine = Engine::new(&wasmi::Config::default());
    let module = Module::new(&engine, wasm_bytes)?;
    let mut store = Store::new(&engine, ());
    let linker = Linker::new(&engine);
    let instance = linker.instantiate(&mut store, &module)?.start(&mut store)?;
    let func = instance.get_func(&store, func_name)
        .ok_or_else(|| anyhow!("No {} function", func_name))?;
    let result = match args.len() {
        0 => func.typed::<(), i32>(&store)?.call(&mut store, ())?,
        1 => func.typed::<i32, i32>(&store)?.call(&mut store, args[0])?,
        _ => return Err(anyhow!("unsupported arg count")),
    };
    Ok(result)
}

fn run_wasmi_matmul(wasm_bytes: &[u8], n: u32, a_off: u32, b_off: u32, c_off: u32,
                     a_data: &[u32], b_data: &[u32]) -> Result<i32> {
    let engine = Engine::new(&wasmi::Config::default());
    let module = Module::new(&engine, wasm_bytes)?;
    let mut store = Store::new(&engine, ());
    let mut linker = Linker::new(&engine);
    let memory = Memory::new(&mut store, MemoryType::new(16, None).unwrap()).unwrap();
    linker.define("env", "memory", memory.clone())?;
    let instance = linker.instantiate(&mut store, &module)?.start(&mut store)?;
    let memory = instance.get_memory(&store, "memory").unwrap_or(memory);

    for (i, v) in a_data.iter().enumerate() {
        memory.write(&mut store, (a_off as usize + i) * 4, &v.to_le_bytes())?;
    }
    for (i, v) in b_data.iter().enumerate() {
        memory.write(&mut store, (b_off as usize + i) * 4, &v.to_le_bytes())?;
    }

    let func = instance.get_func(&store, "test_matmul")
        .ok_or_else(|| anyhow!("No test_matmul"))?;
    let result = func.typed::<(i32,i32,i32,i32,i32), i32>(&store)?
        .call(&mut store, (0, n as i32, a_off as i32, b_off as i32, c_off as i32))?;
    Ok(result)
}

fn run_wasmi_array(wasm_bytes: &[u8], func_name: &str, data: &[u32]) -> Result<i32> {
    let engine = Engine::new(&wasmi::Config::default());
    let module = Module::new(&engine, wasm_bytes)?;
    let mut store = Store::new(&engine, ());
    let mut linker = Linker::new(&engine);
    let memory = Memory::new(&mut store, MemoryType::new(16, None).unwrap()).unwrap();
    linker.define("env", "memory", memory.clone())?;
    let instance = linker.instantiate(&mut store, &module)?.start(&mut store)?;
    let memory = instance.get_memory(&store, "memory").unwrap_or(memory);

    for (i, v) in data.iter().enumerate() {
        memory.write(&mut store, i * 4, &v.to_le_bytes())?;
    }

    let func = instance.get_func(&store, func_name)
        .ok_or_else(|| anyhow!("No {}", func_name))?;
    let result = func.typed::<(i32,i32), i32>(&store)?
        .call(&mut store, (0, data.len() as i32))?;
    Ok(result)
}

fn run_wasmi_array_u8(wasm_bytes: &[u8], func_name: &str, data: &[u8]) -> Result<i32> {
    let engine = Engine::new(&wasmi::Config::default());
    let module = Module::new(&engine, wasm_bytes)?;
    let mut store = Store::new(&engine, ());
    let mut linker = Linker::new(&engine);
    let memory = Memory::new(&mut store, MemoryType::new(16, None).unwrap()).unwrap();
    linker.define("env", "memory", memory.clone())?;
    let instance = linker.instantiate(&mut store, &module)?.start(&mut store)?;
    let memory = instance.get_memory(&store, "memory").unwrap_or(memory);

    memory.write(&mut store, 0, data)?;

    let func = instance.get_func(&store, func_name)
        .ok_or_else(|| anyhow!("No {}", func_name))?;
    let result = func.typed::<(i32,i32), i32>(&store)?
        .call(&mut store, (0, data.len() as i32))?;
    Ok(result)
}

fn run_wasmi_sieve(wasm_bytes: &[u8], n: u32) -> Result<i32> {
    let engine = Engine::new(&wasmi::Config::default());
    let module = Module::new(&engine, wasm_bytes)?;
    let mut store = Store::new(&engine, ());
    let mut linker = Linker::new(&engine);
    let memory = Memory::new(&mut store, MemoryType::new(16, None).unwrap()).unwrap();
    linker.define("env", "memory", memory.clone())?;
    let instance = linker.instantiate(&mut store, &module)?.start(&mut store)?;

    let func = instance.get_func(&store, "test_sieve")
        .ok_or_else(|| anyhow!("No test_sieve"))?;
    let result = func.typed::<(i32,i32), i32>(&store)?
        .call(&mut store, (0, n as i32))?;
    Ok(result)
}

fn run_wasmi_json_crate(wasm_bytes: &[u8], json_data: &[u8]) -> Result<i32> {
    let engine = Engine::new(&wasmi::Config::default());
    let module = Module::new(&engine, wasm_bytes)?;
    let mut store = Store::new(&engine, ());
    let linker = Linker::new(&engine);
    let instance = linker.instantiate(&mut store, &module)?.start(&mut store)?;
    let memory = instance.get_memory(&store, "memory")
        .ok_or_else(|| anyhow!("No memory export"))?;

    memory.write(&mut store, 0, json_data)?;

    let func = instance.get_typed_func::<(i32, i32), i32>(&store, "parse_json_deep")?;
    let result = func.call(&mut store, (0, json_data.len() as i32))?;
    Ok(result)
}

fn generate_test_json(target_size: usize) -> String {
    let mut json = String::from(r#"{"data":{"users":["#);
    let tpl = r#"{"name":"user_XXX","email":"user_XXX@example.com","age":25,"active":true,"tags":["a","b","c"]}"#;
    let mut i = 0;
    while json.len() < target_size {
        if i > 0 { json.push(','); }
        json.push_str(&tpl.replace("XXX", &format!("{:04}", i)));
        i += 1;
    }
    json.push_str(r#"]},"meta":{"count":"#);
    json.push_str(&i.to_string());
    json.push_str(r#"}}"#);
    json
}
