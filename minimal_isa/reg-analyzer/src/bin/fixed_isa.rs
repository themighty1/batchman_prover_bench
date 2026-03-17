//! Fixed-ISA execution pipeline.
//!
//! 1. Compile WASM → VReg IR → regalloc with N regs (ML mode)
//! 2. Profile run of parse_json_deep to capture dynamic specialized opcodes
//! 3. Select top-K as the ISA (+ mov variants)
//! 4. Legalize all functions: insert Mov instructions for non-ISA register combos
//! 5. Execute legalized code and verify correctness
//!
//! Usage: fixed_isa [num_regs] [isa_budget] [wasm_path]

use anyhow::{Context, Result, anyhow};
use std::collections::{HashMap, HashSet};
use std::fs;
use wasmparser::{Parser, Payload};
use reg_analyzer::regvm::{
    WasmToVReg, VRegInst, VReg, PReg, FuncSig, SlotType,
    GLOBALS_MEM_BASE, FRAME_SP_ADDR, FRAME_STACK_BASE, SLOT_SIZE,
};
use reg_analyzer::regalloc::{compute_live_intervals, linear_scan_alloc, RegAllocResult};
use reg_analyzer::preg_vm::{
    PRegVM, vreg_dst_regs, vreg_src_regs, specialized_opcode, replace_vregs,
};
use reg_analyzer::interpreter::Value;

fn val_type_to_slot(vt: &wasmparser::ValType) -> SlotType {
    match vt {
        wasmparser::ValType::I64 => SlotType::I64,
        _ => SlotType::I32,
    }
}

fn generate_test_json(target_size: usize) -> String {
    let mut json = String::from(r#"{"data":{"users":["#);
    let user_template = r#"{"name":"user_XXX","email":"user_XXX@example.com","age":25,"active":true,"tags":["a","b","c"]}"#;
    let mut i = 0;
    while json.len() < target_size {
        if i > 0 { json.push(','); }
        json.push_str(&user_template.replace("XXX", &format!("{:04}", i)));
        i += 1;
    }
    json.push_str(r#"]},"meta":{"count":"#);
    json.push_str(&i.to_string());
    json.push_str(r#"}}"#);
    json
}

/// Parse ISA opcode string into (base_name, register_fields)
/// e.g. "i32.add.r0.r1.r3" → ("i32.add", [0, 1, 3])
fn parse_isa_opcode(s: &str) -> (&str, Vec<u8>) {
    let parts: Vec<&str> = s.split('.').collect();
    // Find where register fields start
    let mut base_end = 0;
    let mut regs = Vec::new();
    for (i, p) in parts.iter().enumerate() {
        if p.starts_with('r') && p.len() > 1 && p[1..].chars().all(|c| c.is_ascii_digit()) {
            if base_end == 0 { base_end = i; }
            regs.push(p[1..].parse::<u8>().unwrap());
        } else if p.starts_with('s') && p.len() > 1 && p[1..].chars().all(|c| c.is_ascii_digit()) {
            // spilled operand — treat as a special "register" 255
            if base_end == 0 { base_end = i; }
            regs.push(255);
        } else if *p == "?" {
            if base_end == 0 { base_end = i; }
            regs.push(254);
        }
    }
    if base_end == 0 { base_end = parts.len(); }
    // Find the byte position after the base parts
    let mut byte_pos = 0;
    for i in 0..base_end {
        byte_pos += parts[i].len();
        if i < base_end - 1 { byte_pos += 1; } // '.'
    }
    (&s[..byte_pos], regs)
}

/// Count register differences between two register lists
fn reg_distance(a: &[u8], b: &[u8]) -> usize {
    if a.len() != b.len() { return usize::MAX; }
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

/// Legalize a function's instruction stream.
/// For each instruction whose specialized opcode is not in the ISA,
/// find the nearest ISA variant and insert Mov instructions.
///
/// Strategy: save-all-restore-all. For each non-ISA instruction:
/// 1. Save ALL registers to spill slots (avoids any clobber issues)
/// 2. Load sources from saved spill slots to ISA target registers
/// 3. Execute instruction with ISA-compatible register assignments
/// 4. Save result to spill slot
/// 5. Restore ALL registers from saved spill slots
/// 6. Load result from spill to original destination
fn legalize_function(
    instructions: &[VRegInst],
    alloc: &mut RegAllocResult,
    isa: &HashSet<String>,
    isa_by_base: &HashMap<String, Vec<Vec<u8>>>,
    next_vreg: &mut u32,
    num_regs: u32,
) -> (Vec<VRegInst>, u32, u32) {
    let mut result = Vec::new();
    let mut mov_count = 0u32;
    let mut non_isa_count = 0u32;

    // Helper to allocate a new vreg mapped to a preg
    let mut new_preg_vreg = |next: &mut u32, alloc: &mut RegAllocResult, preg: u8| -> VReg {
        let v = VReg(*next);
        *next += 1;
        alloc.vreg_to_preg.insert(v, PReg(preg));
        v
    };

    // Helper to allocate a new vreg mapped to a spill slot
    let mut new_spill_vreg = |next: &mut u32, alloc: &mut RegAllocResult| -> VReg {
        let v = VReg(*next);
        *next += 1;
        let slot = alloc.num_spill_slots;
        alloc.num_spill_slots += 1;
        alloc.spilled.insert(v);
        alloc.spill_slots.insert(v, reg_analyzer::regvm::SpillSlot(slot));
        v
    };

    for inst in instructions {
        let spec = specialized_opcode(inst, alloc);
        if isa.contains(&spec) {
            result.push(inst.clone());
            continue;
        }

        let dsts = vreg_dst_regs(inst);
        let srcs = vreg_src_regs(inst);
        if dsts.is_empty() && srcs.is_empty() {
            result.push(inst.clone());
            continue;
        }

        non_isa_count += 1;

        let (base, cur_regs) = parse_isa_opcode(&spec);
        let base_str = base.to_string();

        let best_regs = if let Some(variants) = isa_by_base.get(&base_str) {
            variants.iter()
                .min_by_key(|v| reg_distance(v, &cur_regs))
                .cloned()
        } else {
            None
        };

        if let Some(target_regs) = best_regs {
            if target_regs.len() == cur_regs.len() {
                let n_dsts = dsts.len();
                let all_orig_vregs: Vec<VReg> = dsts.iter().chain(srcs.iter()).copied().collect();
                let src_orig_vregs = &all_orig_vregs[n_dsts..];
                let src_cur_pregs = &cur_regs[n_dsts..];
                let src_target_pregs = &target_regs[n_dsts..];
                let dst_orig_vregs = &all_orig_vregs[..n_dsts];
                let dst_target_pregs = &target_regs[..n_dsts];

                // Compute the set of registers clobbered by the legalization.
                // Clobbered = ISA target registers (sources + dsts) that differ from current.
                let mut clobbered: HashSet<u8> = HashSet::new();
                for i in 0..src_orig_vregs.len() {
                    if src_cur_pregs[i] != src_target_pregs[i] && src_target_pregs[i] < num_regs as u8 {
                        clobbered.insert(src_target_pregs[i]);
                    }
                }
                for i in 0..dst_orig_vregs.len() {
                    if dst_target_pregs[i] < num_regs as u8 {
                        clobbered.insert(dst_target_pregs[i]);
                    }
                }

                // Step 1: Save only clobbered registers to spill slots
                let mut save_vregs: HashMap<u8, VReg> = HashMap::new();
                for &r in &clobbered {
                    let read_v = new_preg_vreg(next_vreg, alloc, r);
                    let save_v = new_spill_vreg(next_vreg, alloc);
                    result.push(VRegInst::Mov { dst: save_v, src: read_v });
                    save_vregs.insert(r, save_v);
                    mov_count += 1;
                }

                // Step 2: Load sources from saved slots (or original) to ISA target regs
                let mut vreg_map: HashMap<VReg, VReg> = HashMap::new();
                for i in 0..src_orig_vregs.len() {
                    if src_cur_pregs[i] != src_target_pregs[i] || src_cur_pregs[i] >= num_regs as u8 {
                        let new_src = new_preg_vreg(next_vreg, alloc, src_target_pregs[i]);
                        if src_cur_pregs[i] < num_regs as u8 {
                            if let Some(&save_v) = save_vregs.get(&src_cur_pregs[i]) {
                                // Source register was clobbered — load from saved spill
                                result.push(VRegInst::Mov { dst: new_src, src: save_v });
                            } else {
                                // Source register not clobbered — load directly from original
                                result.push(VRegInst::Mov { dst: new_src, src: src_orig_vregs[i] });
                            }
                        } else {
                            // Source was already spilled — load from original vreg
                            result.push(VRegInst::Mov { dst: new_src, src: src_orig_vregs[i] });
                        }
                        vreg_map.insert(src_orig_vregs[i], new_src);
                        mov_count += 1;
                    }
                }

                // Step 3: Remap dst to ISA target register
                let mut dst_fixups: Vec<(VReg, VReg)> = Vec::new();
                for i in 0..dst_orig_vregs.len() {
                    let new_dst = new_preg_vreg(next_vreg, alloc, dst_target_pregs[i]);
                    vreg_map.insert(dst_orig_vregs[i], new_dst);
                    dst_fixups.push((dst_orig_vregs[i], new_dst));
                }

                // Emit the remapped instruction
                let remapped = replace_vregs(inst, &vreg_map);
                result.push(remapped);

                // Step 4: Save result(s) to spill
                let mut result_saves: Vec<(VReg, VReg)> = Vec::new();
                for (orig_dst, new_dst) in &dst_fixups {
                    let result_spill = new_spill_vreg(next_vreg, alloc);
                    result.push(VRegInst::Mov { dst: result_spill, src: *new_dst });
                    result_saves.push((*orig_dst, result_spill));
                    mov_count += 1;
                }

                // Step 5: Restore clobbered registers from saved spills
                for &r in &clobbered {
                    let restore_v = new_preg_vreg(next_vreg, alloc, r);
                    result.push(VRegInst::Mov { dst: restore_v, src: save_vregs[&r] });
                    mov_count += 1;
                }

                // Step 6: Load results from spill to original destinations
                for (orig_dst, result_spill) in &result_saves {
                    result.push(VRegInst::Mov { dst: *orig_dst, src: *result_spill });
                    mov_count += 1;
                }

                continue;
            }
        }

        // Fallback: can't legalize, keep as-is
        result.push(inst.clone());
    }

    (result, mov_count, non_isa_count)
}

fn main() -> Result<()> {
    let num_regs: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let isa_budget: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let wasm_path = std::env::args().nth(3).unwrap_or_else(||
        "../json-crate-wasm/target/wasm32-unknown-unknown/release/json_crate_wasm.wasm".to_string());

    let wasm_bytes = fs::read(&wasm_path).context("Failed to read WASM file")?;

    // === Parse WASM ===
    let mut func_types: Vec<wasmparser::FuncType> = Vec::new();
    let mut type_indices: Vec<u32> = Vec::new();
    let mut func_names: HashMap<u32, String> = HashMap::new();
    let mut code_bodies: Vec<wasmparser::FunctionBody> = Vec::new();
    let mut global_inits: Vec<i32> = Vec::new();
    let mut global_types: Vec<SlotType> = Vec::new();
    let mut data_segments: Vec<(u32, Vec<u8>)> = Vec::new();

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
                        func_names.insert(export.index, export.name.to_string());
                    }
                }
            }
            Payload::CodeSectionEntry(body) => { code_bodies.push(body.clone()); }
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

    // === Compile all functions ===
    let mut vreg_funcs: Vec<(Vec<VRegInst>, RegAllocResult, u32, u32)> = Vec::new();
    let mut max_vreg_per_func: Vec<u32> = Vec::new();

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

        // Track max vreg used
        let max_vreg = converter.instructions.iter()
            .flat_map(|i| vreg_dst_regs(i).into_iter().chain(vreg_src_regs(i).into_iter()))
            .map(|v| v.0)
            .max()
            .unwrap_or(0) + 1;
        max_vreg_per_func.push(max_vreg);

        let intervals = compute_live_intervals(&converter.instructions);
        let alloc = linear_scan_alloc(&intervals, num_regs);
        vreg_funcs.push((converter.instructions, alloc, num_params, num_locals));
    }

    // === Phase 1: Profile run ===
    eprintln!("Phase 1: Profiling with {} regs...", num_regs);

    let mut max_spill_slots = 0u32;
    for (_, alloc, _, _) in &vreg_funcs {
        max_spill_slots = max_spill_slots.max(alloc.num_spill_slots);
    }
    let mut vm = PRegVM::new(num_regs as usize, max_spill_slots as usize + 64, 256);

    for (vreg_insts, alloc, num_params, num_locals) in &vreg_funcs {
        vm.add_vreg_function_ml(
            vreg_insts.clone(),
            RegAllocResult {
                vreg_to_preg: alloc.vreg_to_preg.clone(),
                spilled: alloc.spilled.clone(),
                spill_slots: alloc.spill_slots.clone(),
                num_spill_slots: alloc.num_spill_slots,
            },
            *num_params, *num_locals,
        );
    }

    for (i, val) in global_inits.iter().enumerate() {
        if i < vm.globals.len() { vm.globals[i] = Value::I32(*val); }
    }
    for (offset, data) in &data_segments {
        vm.write_memory(*offset as usize, data);
    }

    let test_json = generate_test_json(2048);
    let json_data = test_json.as_bytes();
    vm.write_memory(0, json_data);
    vm.globals[0] = Value::I32(global_inits.first().copied().unwrap_or(1048576));

    vm.write_memory(FRAME_SP_ADDR as usize, &FRAME_STACK_BASE.to_le_bytes());
    for (i, val) in global_inits.iter().enumerate() {
        let addr = GLOBALS_MEM_BASE as usize + (i as u32 * SLOT_SIZE) as usize;
        vm.write_memory(addr, &(*val as u32).to_le_bytes());
    }
    let frame_base = FRAME_STACK_BASE as usize;
    vm.write_memory(frame_base, &0u32.to_le_bytes());
    vm.write_memory(frame_base + SLOT_SIZE as usize, &(json_data.len() as u32).to_le_bytes());

    vm.enable_reg_trace();

    let func_idx = func_names.iter()
        .find(|(_, name)| *name == "parse_json_deep")
        .map(|(idx, _)| *idx)
        .ok_or_else(|| anyhow!("No parse_json_deep"))?;

    let func = &vreg_funcs[func_idx as usize];
    let profile_result = vm.execute_vreg(&func.0, &func.1);
    let nodes_profile = profile_result.map(|v| v.as_i32() as u32).unwrap_or(0);
    let reg_trace = vm.reg_trace.take().unwrap_or_default();
    let trace_len = reg_trace.len();

    eprintln!("  Profile: {} nodes, {} trace instructions", nodes_profile, trace_len);
    assert_eq!(nodes_profile, 194, "Profile run must produce 194 nodes");

    // === Phase 2: Build ISA ===
    let mov_budget = num_regs * (num_regs - 1); // all reg-to-reg mov variants
    let op_budget = isa_budget - mov_budget as usize;

    // Count dynamic frequencies
    let mut freq: HashMap<String, u64> = HashMap::new();
    for (name, dsts, srcs) in &reg_trace {
        let mut parts = vec![name.to_string()];
        for r in dsts { parts.push(format!("r{}", r)); }
        for r in srcs { parts.push(format!("r{}", r)); }
        let spec = parts.join(".");
        *freq.entry(spec).or_insert(0) += 1;
    }

    let mut sorted: Vec<_> = freq.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));

    // Select top-K opcodes for the ISA
    let mut isa: HashSet<String> = HashSet::new();

    // Add all mov variants (r0→r1, r0→r2, r1→r0, etc.)
    for src in 0..num_regs {
        for dst in 0..num_regs {
            if src != dst {
                isa.insert(format!("mov.r{}.r{}", dst, src));
            }
        }
    }

    // First: ensure at least 1 variant per executed base opcode is in the ISA.
    // This prevents "no ISA variant found" for rare opcodes.
    let mut base_to_best: HashMap<String, (String, u64)> = HashMap::new();
    for (name, count) in &sorted {
        let (base, _regs) = parse_isa_opcode(name);
        let base_s = base.to_string();
        let e = base_to_best.entry(base_s).or_insert_with(|| (name.to_string(), 0));
        if **count > e.1 {
            *e = (name.to_string(), **count);
        }
    }

    let mut isa_ops_added = 0;
    let mut covered = 0u64;

    // Add best variant per base opcode
    let mut mandatory: Vec<(String, u64)> = base_to_best.values().cloned().collect();
    mandatory.sort_by(|a, b| b.1.cmp(&a.1));
    for (name, count) in &mandatory {
        if !isa.contains(name.as_str()) {
            isa.insert(name.clone());
            isa_ops_added += 1;
            covered += *count;
        }
    }
    let mandatory_count = isa_ops_added;

    // Fill remaining budget with hottest opcodes
    for (name, count) in &sorted {
        if isa_ops_added >= op_budget { break; }
        if !isa.contains(name.as_str()) {
            isa.insert(name.to_string());
            isa_ops_added += 1;
            covered += **count;
        }
    }
    let uncovered = trace_len as u64 - covered;

    eprintln!("  ISA: {} ops ({} data [{} mandatory + {} hot] + {} mov), covers {:.2}% of trace ({} uncovered)",
        isa.len(), isa_ops_added, mandatory_count, isa_ops_added - mandatory_count,
        mov_budget, covered as f64 / trace_len as f64 * 100.0, uncovered);

    // Build lookup: base_name → list of register combos in ISA
    let mut isa_by_base: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
    for op in &isa {
        let (base, regs) = parse_isa_opcode(op);
        isa_by_base.entry(base.to_string()).or_default().push(regs);
    }

    // === Phase 3: Legalize all functions ===
    eprintln!("Phase 3: Legalizing {} functions...", vreg_funcs.len());

    let mut legalized_funcs: Vec<(Vec<VRegInst>, RegAllocResult, u32, u32)> = Vec::new();
    let mut total_movs = 0u32;
    let mut total_non_isa = 0u32;

    for (i, (insts, alloc, num_params, num_locals)) in vreg_funcs.iter().enumerate() {
        let mut alloc_clone = RegAllocResult {
            vreg_to_preg: alloc.vreg_to_preg.clone(),
            spilled: alloc.spilled.clone(),
            spill_slots: alloc.spill_slots.clone(),
            num_spill_slots: alloc.num_spill_slots,
        };
        let mut next_vreg = max_vreg_per_func[i];
        let (legalized, movs, non_isa) = legalize_function(
            insts, &mut alloc_clone, &isa, &isa_by_base, &mut next_vreg, num_regs,
        );
        total_movs += movs;
        total_non_isa += non_isa;
        legalized_funcs.push((legalized, alloc_clone, *num_params, *num_locals));
    }

    eprintln!("  Legalized: {} non-ISA instructions, {} movs inserted", total_non_isa, total_movs);

    // === Phase 4: Execute legalized code ===
    eprintln!("Phase 4: Executing legalized code...");

    let mut max_spill_slots2 = 0u32;
    for (_, alloc, _, _) in &legalized_funcs {
        max_spill_slots2 = max_spill_slots2.max(alloc.num_spill_slots);
    }
    let mut vm2 = PRegVM::new(num_regs as usize, max_spill_slots2 as usize + 64, 256);

    for (vreg_insts, alloc, num_params, num_locals) in &legalized_funcs {
        vm2.add_vreg_function_ml(
            vreg_insts.clone(),
            RegAllocResult {
                vreg_to_preg: alloc.vreg_to_preg.clone(),
                spilled: alloc.spilled.clone(),
                spill_slots: alloc.spill_slots.clone(),
                num_spill_slots: alloc.num_spill_slots,
            },
            *num_params, *num_locals,
        );
    }

    for (i, val) in global_inits.iter().enumerate() {
        if i < vm2.globals.len() { vm2.globals[i] = Value::I32(*val); }
    }
    for (offset, data) in &data_segments {
        vm2.write_memory(*offset as usize, data);
    }

    vm2.write_memory(0, json_data);
    vm2.globals[0] = Value::I32(global_inits.first().copied().unwrap_or(1048576));

    vm2.write_memory(FRAME_SP_ADDR as usize, &FRAME_STACK_BASE.to_le_bytes());
    for (i, val) in global_inits.iter().enumerate() {
        let addr = GLOBALS_MEM_BASE as usize + (i as u32 * SLOT_SIZE) as usize;
        vm2.write_memory(addr, &(*val as u32).to_le_bytes());
    }
    vm2.write_memory(frame_base, &0u32.to_le_bytes());
    vm2.write_memory(frame_base + SLOT_SIZE as usize, &(json_data.len() as u32).to_le_bytes());

    vm2.enable_reg_trace();

    // Debug: compare function sizes
    let orig_size = vreg_funcs[func_idx as usize].0.len();
    let legal_size = legalized_funcs[func_idx as usize].0.len();
    eprintln!("  parse_json_deep: {} orig insts → {} legalized insts", orig_size, legal_size);
    eprintln!("  spill slots: orig={}, legalized={}",
        vreg_funcs[func_idx as usize].1.num_spill_slots,
        legalized_funcs[func_idx as usize].1.num_spill_slots);

    let func2 = &legalized_funcs[func_idx as usize];
    let exec_result = vm2.execute_vreg(&func2.0, &func2.1);
    let nodes_exec = exec_result.map(|v| v.as_i32() as u32).unwrap_or(0);
    let reg_trace2 = vm2.reg_trace.take().unwrap_or_default();
    let trace_len2 = reg_trace2.len();

    // === Phase 5: Verify ===
    println!("=== Fixed-ISA Execution Results ({} regs, ISA budget {}) ===\n", num_regs, isa_budget);
    println!("WASM:    {}", wasm_path);
    println!("Profile: {} nodes (expected 194)", nodes_profile);
    println!("Execute: {} nodes (expected 194)", nodes_exec);
    println!();

    // Verify all executed instructions are in the ISA
    // Note: Mov instructions involving spill slots are infrastructure (save/restore),
    // not data ISA opcodes. Only count violations for non-infrastructure instructions.
    let mut violations = 0u64;
    let mut infra_movs = 0u64;
    let mut violation_examples: Vec<String> = Vec::new();
    let mut exec_freq: HashMap<String, u64> = HashMap::new();
    for (name, dsts, srcs) in &reg_trace2 {
        let mut parts = vec![name.to_string()];
        for r in dsts { parts.push(format!("r{}", r)); }
        for r in srcs { parts.push(format!("r{}", r)); }
        let spec = parts.join(".");
        *exec_freq.entry(spec.clone()).or_insert(0) += 1;
        if !isa.contains(&spec) {
            // Check if this is an infrastructure mov (spill/reload: has 0 or 1 register operands)
            if *name == "mov" && (dsts.len() + srcs.len()) < 2 {
                infra_movs += 1;
            } else {
                violations += 1;
                if violation_examples.len() < 10 {
                    violation_examples.push(spec);
                }
            }
        }
    }

    let exec_unique = exec_freq.len();
    let mov_in_trace = reg_trace2.iter().filter(|(name, _, _)| *name == "mov").count();

    println!("--- Execution Stats ---");
    println!("Profile trace:    {} instructions", trace_len);
    println!("Legalized trace:  {} instructions", trace_len2);
    println!("Overhead:         {} instructions ({:.2}%)",
        trace_len2 as i64 - trace_len as i64,
        (trace_len2 as f64 / trace_len as f64 - 1.0) * 100.0);
    println!("Mov instructions: {} ({:.2}% of legalized trace)",
        mov_in_trace, mov_in_trace as f64 / trace_len2 as f64 * 100.0);
    println!();
    println!("--- ISA Compliance ---");
    println!("ISA size:          {}", isa.len());
    println!("Unique ops used:   {}", exec_unique);
    println!("Infra movs:        {} (spill/reload, not ISA opcodes)", infra_movs);
    println!("ISA violations:    {}", violations);

    if !violation_examples.is_empty() {
        println!("\nViolation examples:");
        for v in &violation_examples {
            println!("  {}", v);
        }
    }

    println!();
    if nodes_exec == 194 && violations == 0 {
        println!("PASS: Correct result (194 nodes), 0 ISA violations, {} infra movs", infra_movs);
    } else if nodes_exec == 194 {
        println!("PARTIAL: Correct result (194 nodes) but {} ISA violations", violations);
    } else {
        println!("FAIL: Expected 194 nodes, got {}", nodes_exec);
    }

    Ok(())
}
