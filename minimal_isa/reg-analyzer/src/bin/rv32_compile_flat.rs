//! Compile an RV32 ELF into a flat binary where instructions are 4-byte words in memory.
//!
//! Usage:
//!   rv32_compile_flat [num_regs] [elf_path] [output_path]
//!
//! Defaults: 4 regs, json_query.elf, flat_program.bin

use anyhow::Result;
use reg_analyzer::rv32::{decode_elf, get_elf_functions_named, build_cfg, classify_jalr_x0};
use reg_analyzer::rv32_regalloc::{RewrittenInst, run_regalloc_with_symbols};
use reg_analyzer::rv32_isa_vm::MemSegment;
use reg_analyzer::rv32_flat_vm::*;
use reg_analyzer::rv32_passes::is_call;
use std::collections::{HashMap, HashSet};
use std::fs;

struct FuncMapping {
    global_base: usize,
    local_to_global: Vec<usize>, // local_idx → global_idx
}

/// Shuffle instructions to emit at a call site.
struct CallShuffle {
    ops: Vec<ShuffleOp>,
}

#[derive(Debug)]
enum ShuffleOp {
    Mov { dst: u8, src: u8 },
    Swap { a: u8, b: u8 },
}

/// Decompose a set of required moves (src_phys → dst_phys) into movs and swaps.
/// Handles cycles by using swap instead of a temp register.
fn compute_shuffle(needed: &[(u8, u8)]) -> Vec<ShuffleOp> {
    // needed: Vec<(src_phys, dst_phys)> — each says "value in src needs to go to dst"
    if needed.is_empty() { return Vec::new(); }

    let mut remaining: Vec<(u8, u8)> = needed.to_vec();
    let mut ops = Vec::new();

    // Repeatedly find and resolve chains/cycles
    loop {
        if remaining.is_empty() { break; }

        // Find a dst that is NOT a src of any other move — safe to emit as mov
        let safe_idx = remaining.iter().position(|&(_s, d)| {
            !remaining.iter().any(|&(s2, _d2)| s2 == d)
        });

        if let Some(idx) = safe_idx {
            let (src, dst) = remaining.remove(idx);
            ops.push(ShuffleOp::Mov { dst, src });
        } else {
            // All dsts are also srcs — we have a cycle. Break it with a swap.
            let (src, dst) = remaining.remove(0);
            ops.push(ShuffleOp::Swap { a: src, b: dst });
            // After swap: value that was in src is now in dst (done),
            // value that was in dst is now in src.
            // Update remaining: anything that had src as its source now has dst,
            // since the value moved from dst→src.
            // Wait: swap(a,b) means regs[a]↔regs[b].
            // Before: regs[src]=V1, regs[dst]=V2
            // After:  regs[src]=V2, regs[dst]=V1
            // We wanted V1 → dst, which is done.
            // Any remaining move that needs V2 (was sourced from dst) now finds V2 in src.
            for item in remaining.iter_mut() {
                if item.0 == dst {
                    item.0 = src;
                }
            }
        }
    }

    ops
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let num_regs: u32 = args.get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let elf_path = args.get(2)
        .cloned()
        .unwrap_or_else(|| "../rv32-build/target/riscv32i-unknown-none-elf/release/json_query".to_string());

    let output_path = args.get(3)
        .cloned()
        .unwrap_or_else(|| "flat_program.bin".to_string());

    println!("=== rv32_compile_flat ({} regs) ===", num_regs);
    println!("  ELF:    {}", elf_path);
    println!("  Output: {}", output_path);

    // Step 1: decode + regalloc
    let data = fs::read(&elf_path)?;
    let (decoded_raw, _text_addr, _text_len) = decode_elf(&data)?;
    let elf_funcs_named = get_elf_functions_named(&data)?;
    let mut decoded = decoded_raw;
    let elf_funcs: Vec<(u32, u32)> = elf_funcs_named.iter().map(|(a, s, _)| (*a, *s)).collect();
    let (_jump_table_targets, jump_table_bases) = classify_jalr_x0(&mut decoded, &data, &elf_funcs_named);
    let blocks = build_cfg(&decoded, &_jump_table_targets);
    let alloc_result = run_regalloc_with_symbols(&decoded, &blocks, num_regs, &elf_funcs);

    // Step 1.5: Resolve auipc instructions.
    // auipc computes rd = original_pc + imm, but in the flat model PC is in a
    // different address space. We pre-compute the absolute result for each auipc
    // and resolve auipc+consumer pairs at compile time.
    //
    // Map: auipc_original_addr → (auipc_result, consumer_addr, consumer_op)
    // Map: consumer_original_addr → absolute_target
    let mut auipc_pair_target: HashMap<u32, u32> = HashMap::new(); // consumer_addr → abs target
    let mut auipc_dead: HashMap<u32, u32> = HashMap::new(); // auipc_addr → abs auipc_result
    let mut auipc_data_load: HashMap<u32, (u32, u32)> = HashMap::new(); // auipc_addr → (abs_upper, consumer_addr)
    let mut auipc_data_consumer: HashMap<u32, i32> = HashMap::new(); // consumer_addr → adjusted_lower
    let mut auipc_unresolved = 0;

    for i in 0..decoded.len() {
        if decoded[i].op != "auipc" { continue; }
        let auipc_rd = decoded[i].rd.unwrap();
        let auipc_result = decoded[i].addr.wrapping_add(decoded[i].imm.unwrap_or(0) as u32);

        // Find the consumer instruction that reads auipc_rd
        let mut found = false;
        for j in (i+1)..decoded.len().min(i+10) {
            let reads = decoded[j].rs1 == Some(auipc_rd);
            if reads {
                let consumer_offset = decoded[j].imm.unwrap_or(0);
                let target = auipc_result.wrapping_add(consumer_offset as u32);

                match decoded[j].op.as_str() {
                    "jalr" => {
                        // auipc+jalr (tail call or call): resolve to direct jump
                        auipc_pair_target.insert(decoded[j].addr, target);
                        auipc_dead.insert(decoded[i].addr, auipc_result);
                    }
                    "addi" | "lw" | "sw" | "lb" | "lh" | "lbu" | "lhu" | "sb" | "sh" => {
                        // auipc+load/store/addi: convert to lui+adjusted_offset
                        // RISC-V lui+addi convention:
                        //   upper = (target + 0x800) >> 12
                        //   lower = target - (upper << 12)  [fits in -2048..2047]
                        let upper = target.wrapping_add(0x800) >> 12;
                        let lower = target.wrapping_sub(upper << 12) as i32;
                        auipc_data_load.insert(decoded[i].addr, (upper, decoded[j].addr));
                        auipc_data_consumer.insert(decoded[j].addr, lower);
                    }
                    _ => {
                        eprintln!("  WARNING: auipc at 0x{:x} consumed by {} at 0x{:x} (unhandled)",
                            decoded[i].addr, decoded[j].op, decoded[j].addr);
                        auipc_unresolved += 1;
                    }
                }
                found = true;
                break;
            }
            // If instruction overwrites auipc_rd, stop searching
            if decoded[j].rd == Some(auipc_rd) { break; }
        }
        if !found {
            eprintln!("  WARNING: auipc at 0x{:x} has no consumer found", decoded[i].addr);
            auipc_unresolved += 1;
        }
    }

    let auipc_jalr_count = auipc_pair_target.len();
    let auipc_data_count = auipc_data_load.len();
    eprintln!("  auipc resolved: {} jalr pairs, {} data loads, {} unresolved",
        auipc_jalr_count, auipc_data_count, auipc_unresolved);

    let ok_funcs = alloc_result.func_results.iter().filter(|r| r.ok).count();
    let total_funcs = alloc_result.func_results.len();
    println!("  Functions: {}/{} OK", ok_funcs, total_funcs);

    for r in &alloc_result.func_results {
        if !r.ok {
            eprintln!("  FAIL 0x{:x}: {:?}", r.entry_addr, r.error);
        }
    }

    // Step 1.9: Dead conv_store elimination (per-function backward dataflow).
    let mut total_conv_stores = 0u64;
    let mut stripped_conv_stores = 0u64;
    let mut keep_flags: Vec<Vec<bool>> = Vec::new();

    for r in &alloc_result.func_results {
        if !r.ok {
            keep_flags.push(Vec::new());
            continue;
        }
        let keep = if std::env::var("NO_DSE").is_ok() {
            vec![true; r.rewritten.len()]
        } else {
            reg_analyzer::rv32_passes::pass_dead_conv_store(&r.rewritten)
        };
        for (i, inst) in r.rewritten.iter().enumerate() {
            if inst.op == "conv_store" && inst.orig_rd.is_some() {
                total_conv_stores += 1;
                if !keep[i] { stripped_conv_stores += 1; }
            }
        }
        keep_flags.push(keep);
    }

    println!("  Conv stores: {} total, {} stripped ({:.1}%)",
        total_conv_stores, stripped_conv_stores,
        if total_conv_stores > 0 { stripped_conv_stores as f64 / total_conv_stores as f64 * 100.0 } else { 0.0 });

    // Step 1.95: Compute per-call-site register shuffles.
    //
    // For each direct call, compare caller's orig→phys map with callee's entry_reg_map.
    // Produce mov/swap sequences to put registers in the right place before the call.
    // Also track which callee entry conv_loads are covered by ALL callers.

    // Build func entry_addr → entry_reg_map lookup
    let entry_maps: HashMap<u32, Vec<(u8, u8)>> = alloc_result.func_results.iter()
        .filter(|r| r.ok)
        .map(|r| (r.entry_addr, r.entry_reg_map.clone()))
        .collect();

    // call_shuffles: (func_index, inst_index) → CallShuffle
    let mut call_shuffles: HashMap<(usize, usize), CallShuffle> = HashMap::new();

    // Track per-callee which orig regs are covered at each call site.
    // covered_at_call: callee_entry_addr → Vec<HashSet<u8>> (one set per call site)
    let mut covered_at_call: HashMap<u32, Vec<HashSet<u8>>> = HashMap::new();
    // Count total callers per function (including indirect)
    let mut total_callers: HashMap<u32, usize> = HashMap::new();

    let mut total_calls = 0u64;
    let mut total_shuffles = 0u64;
    let mut total_movs = 0u64;
    let mut total_swaps = 0u64;
    let mut total_uncovered = 0u64;

    // Resolve callee address for a call instruction
    let resolve_callee = |inst: &RewrittenInst| -> Option<u32> {
        if inst.op == "jal" && inst.addr != 0 {
            inst.imm.map(|offset| (inst.addr as i64 + offset as i64) as u32)
        } else if inst.op == "jalr" {
            auipc_pair_target.get(&inst.addr).copied()
        } else {
            None
        }
    };

    for (func_i, r) in alloc_result.func_results.iter().enumerate() {
        if !r.ok { continue; }

        let mut caller_map: HashMap<u8, u8> = HashMap::new(); // orig → phys
        let mut phys_to_orig: HashMap<u8, u8> = HashMap::new(); // phys → orig (valid mapping)
        for &(orig, phys) in &r.entry_reg_map {
            caller_map.insert(orig, phys);
            phys_to_orig.insert(phys, orig);
        }

        for (inst_i, inst) in r.rewritten.iter().enumerate() {
            // Invalidate mapping when a phys reg is overwritten by non-conv instructions
            if inst.op != "conv_store" && inst.op != "conv_load" {
                if let Some(rd) = inst.rd {
                    if let Some(old_orig) = phys_to_orig.remove(&rd) {
                        caller_map.remove(&old_orig);
                    }
                }
            }

            if inst.op == "conv_store" {
                if let (Some(orig), Some(phys)) = (inst.orig_rd, inst.rs1) {
                    // Clear any stale orig that was in this phys reg
                    if let Some(old_orig) = phys_to_orig.get(&phys) {
                        if *old_orig != orig {
                            caller_map.remove(old_orig);
                        }
                    }
                    caller_map.insert(orig, phys);
                    phys_to_orig.insert(phys, orig);
                }
            }

            if inst.op == "conv_load" {
                if let (Some(rd), Some(orig)) = (inst.rd, inst.orig_rd) {
                    if let Some(old_orig) = phys_to_orig.get(&rd) {
                        if *old_orig != orig {
                            caller_map.remove(old_orig);
                        }
                    }
                    caller_map.insert(orig, rd);
                    phys_to_orig.insert(rd, orig);
                }
            }

            if is_call(inst) {
                let callee_addr = resolve_callee(inst);
                if let Some(addr) = callee_addr {
                    *total_callers.entry(addr).or_default() += 1;
                    if let Some(callee_map) = entry_maps.get(&addr) {
                        total_calls += 1;
                        let mut needed_moves: Vec<(u8, u8)> = Vec::new(); // (src_phys, dst_phys)
                        let mut covered: HashSet<u8> = HashSet::new();

                        // The return address register (x1) is set by jal_call writing
                        // to MAILBOX, not by the caller's register state. Never shuffle it.
                        let call_orig_rd = inst.orig_rd.unwrap_or(1);

                        for &(orig, callee_phys) in callee_map {
                            if orig == call_orig_rd {
                                // Return address reg — set by jal_call, not shuffleable
                                continue;
                            }
                            if let Some(&caller_phys) = caller_map.get(&orig) {
                                covered.insert(orig);
                                if caller_phys != callee_phys {
                                    needed_moves.push((caller_phys, callee_phys));
                                }
                            } else {
                                total_uncovered += 1;
                            }
                        }

                        let no_shuffle = std::env::var("NO_SHUFFLE").is_ok();
                        let shuffle_ops = if no_shuffle { Vec::new() } else { compute_shuffle(&needed_moves) };
                        for op in &shuffle_ops {
                            match op {
                                ShuffleOp::Mov { .. } => total_movs += 1,
                                ShuffleOp::Swap { .. } => total_swaps += 1,
                            }
                        }
                        if !shuffle_ops.is_empty() { total_shuffles += 1; }

                        {
                            let mut cov: Vec<u8> = covered.iter().copied().collect();
                            cov.sort();
                            let mut cm: Vec<(u8, u8)> = caller_map.iter().map(|(&k,&v)| (k,v)).collect();
                            cm.sort();
                            let callee_sorted: Vec<(u8,u8)> = {
                                let mut v: Vec<(u8,u8)> = callee_map.iter().copied().collect();
                                v.sort();
                                v
                            };
                            println!("    CALL 0x{:x}→0x{:x}: covered={:?} caller_map={:?} callee_map={:?} moves={:?} shuffle={:?}",
                                inst.addr, addr, cov, cm, callee_sorted, needed_moves, shuffle_ops);
                        }
                        call_shuffles.insert((func_i, inst_i), CallShuffle { ops: shuffle_ops });
                        covered_at_call.entry(addr).or_default().push(covered);
                    }
                }

                // After a call, the callee may have updated mailbox slots.
                // The caller's registers are restored from the save area (pre-call values),
                // but the mailbox may now hold different values. If we don't invalidate,
                // stale caller_map entries could cause incorrect conv_load stripping
                // for subsequent calls (the shuffle would provide the restored pre-call
                // value, but the callee expects the mailbox value which the previous
                // callee may have updated).
                caller_map.clear();
                phys_to_orig.clear();
            }
        }
    }

    // Identify functions that must NOT have entry conv_loads stripped:
    // 1. Functions whose address appears in ELF data segments (indirect call targets)
    // 2. Functions that are targets of tail calls (jal/jalr with orig_rd==0 from other functions)
    let func_entry_addrs: HashSet<u32> = entry_maps.keys().copied().collect();
    let mut no_strip_funcs: HashSet<u32> = HashSet::new();

    // Check ELF data segments for function pointer references
    {
        let segments_tmp = extract_elf_segments(&data)?;
        for seg in &segments_tmp {
            let mut off = 0;
            while off + 4 <= seg.data.len() {
                let val = u32::from_le_bytes([seg.data[off], seg.data[off+1], seg.data[off+2], seg.data[off+3]]);
                if func_entry_addrs.contains(&val) {
                    no_strip_funcs.insert(val);
                }
                off += 4;
            }
        }
    }

    // Check for tail calls: auipc+jalr with rd==0 in original decoded stream
    {
        let decoded_by_addr: HashMap<u32, &reg_analyzer::rv32::DecodedInst> =
            decoded.iter().map(|d| (d.addr, d)).collect();
        for (&consumer_addr, &target_addr) in &auipc_pair_target {
            if let Some(d) = decoded_by_addr.get(&consumer_addr) {
                let rd = d.rd.unwrap_or(0);
                if rd == 0 && func_entry_addrs.contains(&target_addr) {
                    no_strip_funcs.insert(target_addr);
                }
            }
        }
        // Check for direct jal tail calls crossing function boundaries
        for d in &decoded {
            if d.op == "jal" && d.rd == Some(0) {
                let target = (d.addr as i64 + d.imm.unwrap_or(0) as i64) as u32;
                if func_entry_addrs.contains(&target) {
                    // Check if cross-function: find which function d.addr belongs to
                    let d_func = elf_funcs.iter().find(|&&(a, s)| d.addr >= a && d.addr < a + s);
                    if d_func.map(|f| f.0) != Some(target) {
                        no_strip_funcs.insert(target);
                    }
                }
            }
        }
    }
    if !no_strip_funcs.is_empty() {
        println!("  No-strip functions: {} (indirect/tail-call targets)", no_strip_funcs.len());
    }

    // Determine which callee entry conv_loads can be stripped:
    // An orig reg is "covered" if ALL callers of the function provide it via shuffle.
    let mut covered_conv_loads: HashMap<u32, HashSet<u8>> = HashMap::new();
    for (addr, call_sets) in &covered_at_call {
        if no_strip_funcs.contains(addr) {
            continue; // May be entered without shuffle — can't strip
        }
        let num_direct = call_sets.len();
        let num_total = total_callers.get(addr).copied().unwrap_or(0);
        if num_direct != num_total || num_direct == 0 {
            // Function has unknown callers — can't strip any
            continue;
        }
        // Intersect all coverage sets
        let mut common: HashSet<u8> = call_sets[0].clone();
        for s in &call_sets[1..] {
            common = common.intersection(s).copied().collect();
        }
        if !common.is_empty() {
            // ONLY_STRIP=0x112bc restricts stripping to a single function for debugging
            if let Ok(only) = std::env::var("ONLY_STRIP") {
                let only_addr = u32::from_str_radix(only.trim_start_matches("0x"), 16).unwrap_or(0);
                if *addr != only_addr { continue; }
            }
            let mut sorted: Vec<u8> = common.iter().copied().collect();
            sorted.sort();
            println!("    STRIP func 0x{:x}: orig regs {:?} ({} callers)", addr, sorted, num_direct);
            covered_conv_loads.insert(*addr, common);
        }
    }

    let stripped_conv_loads: usize = covered_conv_loads.values().map(|s| s.len()).sum();
    println!("  Call-site shuffles ({} direct calls):", total_calls);
    println!("    Shuffled calls:    {} ({} movs, {} swaps)", total_shuffles, total_movs, total_swaps);
    println!("    Uncovered regs:    {} (still use mailbox)", total_uncovered);
    println!("    Stripped conv_loads: {} (across {} functions)", stripped_conv_loads,
        covered_conv_loads.len());


    // Step 2: Flatten all functions into one instruction stream.
    // For each function, record its base index in the flat array.
    // Also build entry_addr → flat_base_index map for resolving call targets.

    let mut flat_insts: Vec<RewrittenInst> = Vec::new();
    let mut func_entry_to_flat_idx: HashMap<u32, usize> = HashMap::new();

    // First pass: collect functions and their flat offsets.
    // We need to emit save/restore instructions around calls, which changes
    // the instruction count. So we do a two-pass approach:
    // Pass 1: expand each function (add save/restore), concatenate
    // Pass 2: resolve all targets

    let frame_reg_id = reg_analyzer::rv32_regalloc::frame_reg_id(num_regs);

    for r in &alloc_result.func_results {
        if !r.ok { continue; }

        let flat_base = flat_insts.len();
        func_entry_to_flat_idx.insert(r.entry_addr, flat_base);

        // Expand function: insert save/restore around call sites
        for (i, inst) in r.rewritten.iter().enumerate() {
            if is_call(inst) {
                // Before call: save all physical regs + frame_reg to frame
                // Use negative offsets from frame_reg for the call-save area.
                // Reserve slots: num_regs + 1 (for frame_reg itself).
                // Offset layout: -(num_regs+1)*4 for frame_reg, then -(num_regs)*4 .. -4 for regs
                let save_area_size = (num_regs as i32 + 1) * 4;
                // Advance frame pointer to make room for save area
                flat_insts.push(make_synth("addi_frame", &format!("addi_frame"),
                    None, None, None, Some(save_area_size), None));

                // Save frame_reg (the value BEFORE advancing, which is frame_reg - save_area_size)
                // Actually we just advanced, so current frame_reg = old + save_area_size.
                // We want to save old frame_reg = current - save_area_size.
                // Store at frame_reg + offset where offset = -save_area_size
                flat_insts.push(make_synth("save_frame", "save_frame",
                    None, None, None, Some(-save_area_size), None));

                // Save each physical register
                for reg_idx in 0..num_regs as u8 {
                    let offset = -save_area_size + 4 + (reg_idx as i32) * 4;
                    flat_insts.push(make_synth("sw_frame", &format!("sw_frame.r{}", reg_idx),
                        None, None, Some(reg_idx), Some(offset), None));
                }

                // The actual call instruction (will be patched to jal_call later)
                flat_insts.push(inst.clone());

                // After call returns: restore all physical regs + frame_reg
                // Restore each physical register
                for reg_idx in 0..num_regs as u8 {
                    let offset = -save_area_size + 4 + (reg_idx as i32) * 4;
                    flat_insts.push(make_synth("lw_frame", &format!("lw_frame.r{}", reg_idx),
                        Some(reg_idx), None, None, Some(offset), None));
                }

                // Restore frame_reg
                flat_insts.push(make_synth("restore_frame", "restore_frame",
                    None, None, None, Some(-save_area_size), None));

                // Retract frame pointer
                flat_insts.push(make_synth("addi_frame", "addi_frame",
                    None, None, None, Some(-save_area_size), None));
            } else {
                flat_insts.push(inst.clone());
            }
        }
    }

    println!("  Flat instructions: {} (before target resolution)", flat_insts.len());

    // Also need addr_to_func mapping for handling jalr (tail calls)
    let mut addr_to_func_entry: HashMap<u32, u32> = HashMap::new();
    for r in &alloc_result.func_results {
        if !r.ok { continue; }
        for inst in &r.rewritten {
            if inst.addr != 0 && inst.addr < 0xF000_0000 {
                addr_to_func_entry.insert(inst.addr, r.entry_addr);
            }
        }
    }

    // Step 3: Resolve all targets to byte addresses.
    // - Branch targets (beq/bne/etc): currently local instruction indices → convert to global byte addr
    // - jal (unconditional jump): currently local index → global byte addr
    // - jal (call): currently code addr offset → look up func_entry_to_flat_idx → global byte addr
    // - jr_computed: currently local index → global byte addr
    // - jr_table_idx: register holds local index → need to patch jump table memory to global indices
    // - jalr (tail call / indirect call): register holds code addr → patch to global byte addr

    // We need per-function base offset to convert local indices to global.
    // But after expanding with save/restore, local indices are gone.
    // Instead we track: for each original function, the mapping of old local index → new global index.

    // Actually, let's take a different approach. We already have pass_resolve_branches
    // running in the regalloc pipeline which converts branch targets to per-function indices.
    // After flattening, those per-function indices need to become global indices.
    //
    // Strategy: Track the global index where each function starts. Then for each instruction
    // that has a per-function index in imm, add the function's global base offset.
    // BUT: the save/restore expansion changed the instruction count per function.
    // So we need to build a mapping: (func_entry, old_local_idx) → new_global_idx.

    // Let's rebuild the mapping by re-scanning the flat stream.
    // During expansion, we can track the original instruction's position.

    // Actually, let me redo the flattening more carefully, tracking positions.

    flat_insts.clear();
    func_entry_to_flat_idx.clear();

    let mut func_mappings: Vec<(u32, FuncMapping)> = Vec::new(); // (entry_addr, mapping)

    for (func_i, r) in alloc_result.func_results.iter().enumerate() {
        if !r.ok { continue; }

        let global_base = flat_insts.len();
        func_entry_to_flat_idx.insert(r.entry_addr, global_base);
        let flags = &keep_flags[func_i];

        // Pre-compute entry conv_loads to strip (callers provide via shuffle)
        let mut strip_entry_conv_load: HashSet<usize> = HashSet::new();
        let no_strip_conv = std::env::var("NO_STRIP_CONV").is_ok();
        if !no_strip_conv {
            if let Some(covered) = covered_conv_loads.get(&r.entry_addr) {
                for (i, inst) in r.rewritten.iter().enumerate() {
                    if inst.op == "conv_load" {
                        if let Some(orig) = inst.orig_rd {
                            if covered.contains(&orig) {
                                strip_entry_conv_load.insert(i);
                            }
                        }
                    } else {
                        break; // Past entry conv_loads
                    }
                }
            }
        }

        // Two-pass approach:
        // Pass 1: compute local_to_global (original_local_idx → global_flat_idx)
        //   For stripped instructions, map to the same position as the next kept instruction.
        // Pass 2: emit flat instructions.

        // Pass 1: compute positions
        let mut local_to_global = vec![0usize; r.rewritten.len()];
        let mut next_global = global_base;
        for (orig_i, inst) in r.rewritten.iter().enumerate() {
            local_to_global[orig_i] = next_global;
            if strip_entry_conv_load.contains(&orig_i) {
                continue; // Stripped entry conv_load
            }
            if flags[orig_i] {
                if is_call(inst) {
                    let shuffle_len = call_shuffles.get(&(func_i, orig_i))
                        .map(|s| s.ops.len())
                        .unwrap_or(0);
                    next_global += 3 + shuffle_len; // save_context + shuffles + call + restore_context
                } else {
                    next_global += 1;
                }
            }
        }

        // Pass 2: emit
        for (orig_i, inst) in r.rewritten.iter().enumerate() {
            if strip_entry_conv_load.contains(&orig_i) { continue; }
            if !flags[orig_i] { continue; }

            if is_call(inst) {
                flat_insts.push(make_synth("save_context", "save_context",
                    None, None, None, None, None));
                // Emit shuffle ops (mov/swap to arrange regs for callee)
                if let Some(shuffle) = call_shuffles.get(&(func_i, orig_i)) {
                    for op in &shuffle.ops {
                        match op {
                            ShuffleOp::Mov { dst, src } => {
                                flat_insts.push(make_synth("mov",
                                    &format!("mov.r{}.r{}", dst, src),
                                    Some(*dst), Some(*src), None, None, None));
                            }
                            ShuffleOp::Swap { a, b } => {
                                flat_insts.push(make_synth("swap",
                                    &format!("swap.r{}.r{}", a, b),
                                    Some(*a), Some(*b), None, None, None));
                            }
                        }
                    }
                }
                flat_insts.push(inst.clone());
                flat_insts.push(make_synth("restore_context", "restore_context",
                    None, None, None, None, None));
            } else {
                flat_insts.push(inst.clone());
            }
        }

        func_mappings.push((r.entry_addr, FuncMapping {
            global_base,
            local_to_global,
        }));
    }

    println!("  Flat instructions: {}", flat_insts.len());

    // Step 4: Resolve targets. For each instruction, convert per-function local indices
    // to global byte addresses.
    // We need to find which function each instruction belongs to.
    let mut inst_to_func: Vec<usize> = Vec::with_capacity(flat_insts.len()); // global_idx → func_mapping_idx
    for (func_idx, (_, mapping)) in func_mappings.iter().enumerate() {
        let func_end = if func_idx + 1 < func_mappings.len() {
            func_mappings[func_idx + 1].1.global_base
        } else {
            flat_insts.len()
        };
        let count = func_end - mapping.global_base;
        for _ in 0..count {
            inst_to_func.push(func_idx);
        }
    }

    // Get ELF entry point
    let elf_entry_addr = {
        use object::elf::*;
        use object::read::elf::FileHeader as _;
        use object::Endianness;
        let elf = FileHeader32::<Endianness>::parse(data.as_slice())?;
        let endian = elf.endian()?;
        elf.e_entry.get(endian)
    };

    let entry_flat_idx = *func_entry_to_flat_idx.get(&elf_entry_addr)
        .ok_or_else(|| anyhow::anyhow!("entry 0x{:x} not in func table", elf_entry_addr))?;
    // entry_pc stored as global instruction index; VM converts to byte addr
    let entry_pc = entry_flat_idx as u32;

    // Now patch all instructions: convert per-function local indices to GLOBAL instruction indices.
    // The VM converts: target_pc = global_index * 4 (code in separate memory space).
    // This keeps immediates small (fits in 20 bits for up to ~500K instructions).
    for gi in 0..flat_insts.len() {
        let func_idx = inst_to_func[gi];
        let mapping = &func_mappings[func_idx].1;
        let op = flat_insts[gi].op.clone();
        let orig_rd = flat_insts[gi].orig_rd.unwrap_or(0);
        let imm = flat_insts[gi].imm.unwrap_or(0);
        let addr = flat_insts[gi].addr;
        let rd = flat_insts[gi].rd;

        match op.as_str() {
            "beq" | "bne" | "blt" | "bge" | "bltu" | "bgeu" => {
                let local_idx = imm as usize;
                let global_idx = mapping.local_to_global[local_idx];
                flat_insts[gi].imm = Some(global_idx as i32);
            }
            "jal" => {
                if orig_rd == 0 {
                    let local_idx = imm as usize;
                    let global_idx = mapping.local_to_global[local_idx];
                    flat_insts[gi].imm = Some(global_idx as i32);
                } else {
                    // Call: resolve code addr → global index
                    let target_code_addr = (addr as i64 + imm as i64) as u32;
                    if let Some(&flat_idx) = func_entry_to_flat_idx.get(&target_code_addr) {
                        flat_insts[gi].imm = Some(flat_idx as i32);
                        flat_insts[gi].op = "jal_call".to_string();
                        flat_insts[gi].specialized = format!("jal_call.r{}",
                            rd.map(|r| r.to_string()).unwrap_or_default());
                        // jal_call writes return addr to MAILBOX, not to a phys reg.
                        // Clear rd so DCE doesn't think this defines a phys register.
                        flat_insts[gi].rd = None;
                    } else {
                        eprintln!("  WARNING: jal call target 0x{:x} not found (gi={})", target_code_addr, gi);
                    }
                }
            }
            "jr_computed" => {
                let local_idx = imm as usize;
                let global_idx = mapping.local_to_global[local_idx];
                flat_insts[gi].imm = Some(global_idx as i32);
            }
            "jr_table" => {
                // Jump table dispatch: register holds instruction index loaded from
                // patched memory. Rename to jr_table_idx for the flat VM handler.
                flat_insts[gi].op = "jr_table_idx".to_string();
                flat_insts[gi].specialized = flat_insts[gi].specialized.replace("jr_table", "jr_table_idx");
            }
            "jalr" => {
                // Check if this jalr was an auipc+jalr pair with known target
                if addr != 0 {
                    if let Some(&target_code_addr) = auipc_pair_target.get(&addr) {
                        // Resolve target code address to flat index
                        if let Some(flat_idx) = resolve_code_addr_to_flat_idx(
                            target_code_addr, &func_entry_to_flat_idx,
                            &addr_to_func_entry, &func_mappings, &alloc_result.func_results)
                        {
                            if orig_rd == 0 {
                                // Tail call → direct jump
                                flat_insts[gi].op = "jal".to_string();
                                flat_insts[gi].imm = Some(flat_idx as i32);
                                flat_insts[gi].specialized = "jal".to_string();
                            } else {
                                // Call → jal_call
                                flat_insts[gi].op = "jal_call".to_string();
                                flat_insts[gi].imm = Some(flat_idx as i32);
                                flat_insts[gi].specialized = format!("jal_call.r{}",
                                    rd.map(|r| r.to_string()).unwrap_or_default());
                                // jal_call writes return addr to MAILBOX, not to a phys reg.
                                flat_insts[gi].rd = None;
                            }
                        } else {
                            eprintln!("  WARNING: auipc+jalr target 0x{:x} not found (gi={})", target_code_addr, gi);
                        }
                    }
                }
                // Remaining jalr: targets come from registers at runtime
                // (function pointers patched in memory → global instruction indices)
            }
            "auipc" => {
                // Check if this auipc feeds a resolved jalr → mark as dead (nop)
                if addr != 0 {
                    if auipc_dead.contains_key(&addr) {
                        // Result is no longer needed (jalr was converted to direct jal)
                        // Convert to lui rd, 0 (harmless write)
                        flat_insts[gi].op = "lui".to_string();
                        flat_insts[gi].imm = Some(0);
                        flat_insts[gi].specialized = format!("lui.r{}", rd.unwrap_or(0));
                    } else if let Some(&(upper, _consumer_addr)) = auipc_data_load.get(&addr) {
                        // Data load: convert to lui with absolute upper bits
                        flat_insts[gi].op = "lui".to_string();
                        flat_insts[gi].imm = Some((upper << 12) as i32);
                        flat_insts[gi].specialized = format!("lui.r{}", rd.unwrap_or(0));
                    }
                }
            }
            _ => {
                // Check if this instruction is a data-load consumer of auipc
                if addr != 0 {
                    if let Some(&lower) = auipc_data_consumer.get(&addr) {
                        flat_insts[gi].imm = Some(lower);
                    }
                }
            }
        }
    }

    // Step 4.25: Expand save_context/restore_context into explicit ops
    let (flat_insts, expand_old_to_new) = reg_analyzer::rv32_passes::pass_expand_save_restore(&flat_insts, num_regs as usize);
    let entry_flat_idx = expand_old_to_new[entry_flat_idx];

    // Step 4.5: Dead code elimination (can now eliminate dead lw_save loads)
    let (flat_insts, dce_raw_old_to_new) = reg_analyzer::rv32_passes::pass_dce(&flat_insts, num_regs as usize);
    let entry_flat_idx = dce_raw_old_to_new[entry_flat_idx];

    // Compose expand + DCE mappings: original pre-expansion index → final index
    let dce_old_to_new: Vec<usize> = expand_old_to_new.iter()
        .map(|&expanded_idx| dce_raw_old_to_new[expanded_idx])
        .collect();

    // Step 5: Extract ELF segments and patch function pointers in memory.
    let mut segments = extract_elf_segments(&data)?;

    // Patch jump table entries: they currently hold per-function local indices
    // (from pass_rewrite_jump_tables). Convert to global byte addresses.
    // Actually, pass_rewrite_jump_tables was NOT run yet in our pipeline
    // (it was in the old rv32_compile). We need to handle jump tables here.
    // The jump_table_bases tell us where tables are in memory and how many entries.
    // The entries currently hold ORIGINAL code addresses (since we decoded from ELF).
    // We need to convert them to global byte addresses.

    // Track which memory addresses hold instruction indices (need idx→byte conversion later)
    let mut patched_addrs: Vec<u32> = Vec::new();

    for (&_jr_addr, &(table_base, num_entries)) in &jump_table_bases {
        for entry_i in 0..num_entries {
            let entry_addr = table_base.wrapping_add((entry_i * 4) as u32);
            let code_addr = read_seg_u32(&segments, entry_addr)
                .unwrap_or_else(|| panic!("Can't read jump table entry at 0x{:x}", entry_addr));

            if let Some(&func_entry) = addr_to_func_entry.get(&code_addr) {
                if let Some(&_flat_base) = func_entry_to_flat_idx.get(&func_entry) {
                    let func_mapping = func_mappings.iter().find(|(e, _)| *e == func_entry)
                        .map(|(_, m)| m).unwrap();
                    let orig_func = alloc_result.func_results.iter()
                        .find(|r| r.ok && r.entry_addr == func_entry).unwrap();
                    let local_idx = orig_func.rewritten.iter().position(|inst| inst.addr == code_addr);
                    if let Some(li) = local_idx {
                        let redirect_key = (_jr_addr, code_addr);
                        let final_local_idx = if let Some(&redirect_addr) = orig_func.jr_table_redirects.get(&redirect_key) {
                            orig_func.rewritten.iter().position(|inst| inst.addr == redirect_addr)
                                .unwrap_or(li)
                        } else {
                            li
                        };
                        let global_idx = dce_old_to_new[func_mapping.local_to_global[final_local_idx]];
                        eprintln!("    JT entry[{}]: code_addr=0x{:x} → local={} → global={}",
                            entry_i, code_addr, final_local_idx, global_idx);
                        write_seg_u32(&mut segments, entry_addr, global_idx as u32);
                        patched_addrs.push(entry_addr);
                    }
                }
            }
        }
    }

    // Patch function entry addresses in memory (for function pointers).
    let mut func_ptr_patches = 0;
    for seg in &mut segments {
        let mut off = 0;
        while off + 4 <= seg.data.len() {
            let val = u32::from_le_bytes([seg.data[off], seg.data[off+1], seg.data[off+2], seg.data[off+3]]);
            if let Some(&flat_idx) = func_entry_to_flat_idx.get(&val) {
                let remapped = dce_old_to_new[flat_idx] as u32;
                seg.data[off..off+4].copy_from_slice(&remapped.to_le_bytes());
                patched_addrs.push(seg.vaddr + off as u32);
                func_ptr_patches += 1;
            }
            off += 4;
        }
    }
    if func_ptr_patches > 0 {
        eprintln!("  Patched {} function pointer entries in memory", func_ptr_patches);
    }

    // Step 6: Build opcode table and encode instructions.
    let mut opcode_map: HashMap<String, u16> = HashMap::new();
    let mut opcode_table: Vec<OpcodeInfo> = Vec::new();

    // Assign opcode IDs from specialized strings
    for inst in &flat_insts {
        if !opcode_map.contains_key(&inst.specialized) {
            let id = opcode_table.len() as u16;
            opcode_map.insert(inst.specialized.clone(), id);
            opcode_table.push(OpcodeInfo {
                name: inst.specialized.clone(),
                base_op: inst.op.clone(),
                rd: inst.rd,
                rs1: inst.rs1,
                rs2: inst.rs2,
                orig_rd: inst.orig_rd,
                orig_rs1: inst.orig_rs1,
                orig_rs2: inst.orig_rs2,
            });
        }
    }

    println!("  ISA size: {} unique opcodes", opcode_table.len());
    assert!(opcode_table.len() <= 4096, "Too many opcodes ({}) for 12-bit encoding", opcode_table.len());

    // Patch memory segments: instruction indices stay as-is (PC is an index now).
    // No conversion needed — patched_addrs hold instruction indices directly.

    // Encode: Vec<u16> opcodes + separate imm_table
    let mut code_table: Vec<u16> = Vec::with_capacity(flat_insts.len());
    let mut imm_table: Vec<i32> = Vec::with_capacity(flat_insts.len());

    for inst in &flat_insts {
        let opcode_id = opcode_map[&inst.specialized];
        code_table.push(opcode_id);

        let mut imm = inst.imm.unwrap_or(0);

        // Special handling for lui/auipc: store upper 20 bits >> 12
        if inst.op == "lui" || inst.op == "auipc" {
            imm = ((imm as u32) >> 12) as i32;
        }

        // Branch/jump targets: imm already holds instruction index, which IS the PC now
        imm_table.push(imm);
    }

    println!("  Code table: {} entries ({} bytes)", code_table.len(), code_table.len() * 2);
    println!("  Imm table:  {} entries ({} bytes)", imm_table.len(), imm_table.len() * 4);

    let entry_pc = entry_flat_idx as u32;
    println!("  Entry PC:  0x{:x} (flat idx {})", entry_pc, entry_flat_idx);

    // Step 7: Serialize
    let serial_opcode_table: Vec<SerializedOpcodeInfo> = opcode_table.iter().map(|o| {
        SerializedOpcodeInfo {
            name: o.name.clone(),
            base_op: o.base_op.clone(),
            rd: o.rd,
            rs1: o.rs1,
            rs2: o.rs2,
            orig_rd: o.orig_rd,
        }
    }).collect();

    let program = FlatProgram {
        num_regs,
        entry_pc,
        segments,
        code_segment: code_table,
        opcode_table: serial_opcode_table,
        imm_table,
        code_segment_u8: Vec::new(),
    };

    let encoded = bincode::serialize(&program)?;
    fs::write(&output_path, &encoded)?;
    println!("  Written: {} bytes", encoded.len());

    Ok(())
}

/// Resolve a code address to a flat instruction index.
/// First checks if it's a direct function entry, then searches within functions.
fn resolve_code_addr_to_flat_idx(
    code_addr: u32,
    func_entry_to_flat_idx: &HashMap<u32, usize>,
    addr_to_func_entry: &HashMap<u32, u32>,
    func_mappings: &[(u32, FuncMapping)],
    func_results: &[reg_analyzer::rv32_regalloc::FuncAllocResult],
) -> Option<usize> {
    // Direct function entry?
    if let Some(&flat_idx) = func_entry_to_flat_idx.get(&code_addr) {
        return Some(flat_idx);
    }
    // Find containing function, then locate instruction within it
    if let Some(&func_entry) = addr_to_func_entry.get(&code_addr) {
        let func_data = func_results.iter()
            .find(|r| r.ok && r.entry_addr == func_entry)?;
        let local_idx = func_data.rewritten.iter()
            .position(|inst| inst.addr == code_addr)?;
        let mapping = func_mappings.iter()
            .find(|(e, _)| *e == func_entry)?;
        return Some(mapping.1.local_to_global[local_idx]);
    }
    None
}

fn make_synth(op: &str, specialized: &str, rd: Option<u8>, rs1: Option<u8>, rs2: Option<u8>, imm: Option<i32>, orig_rd: Option<u8>) -> RewrittenInst {
    RewrittenInst {
        addr: 0,
        op: op.to_string(),
        rd,
        rs1,
        rs2,
        imm,
        is_move: false,
        specialized: specialized.to_string(),
        orig_rd,
        orig_rs1: None,
        orig_rs2: None,
    }
}

fn extract_elf_segments(data: &[u8]) -> Result<Vec<MemSegment>> {
    use object::elf::*;
    use object::read::elf::FileHeader as _;
    use object::Endianness;

    let elf = FileHeader32::<Endianness>::parse(data)?;
    let endian = elf.endian()?;
    let segments_hdr = elf.program_headers(endian, data)?;

    let mut segments = Vec::new();
    for seg in segments_hdr {
        if seg.p_type.get(endian) != PT_LOAD { continue; }
        let vaddr = seg.p_vaddr.get(endian);
        let filesz = seg.p_filesz.get(endian) as usize;
        let offset = seg.p_offset.get(endian) as usize;

        if filesz > 0 && offset + filesz <= data.len() {
            segments.push(MemSegment {
                vaddr,
                data: data[offset..offset + filesz].to_vec(),
            });
        }
    }
    Ok(segments)
}

fn read_seg_u32(segments: &[MemSegment], addr: u32) -> Option<u32> {
    for seg in segments {
        let end = seg.vaddr.wrapping_add(seg.data.len() as u32);
        if addr >= seg.vaddr && addr.wrapping_add(4) <= end {
            let off = (addr - seg.vaddr) as usize;
            return Some(u32::from_le_bytes([seg.data[off], seg.data[off+1], seg.data[off+2], seg.data[off+3]]));
        }
    }
    None
}

fn write_seg_u32(segments: &mut [MemSegment], addr: u32, val: u32) {
    for seg in segments.iter_mut() {
        let end = seg.vaddr.wrapping_add(seg.data.len() as u32);
        if addr >= seg.vaddr && addr.wrapping_add(4) <= end {
            let off = (addr - seg.vaddr) as usize;
            seg.data[off..off+4].copy_from_slice(&val.to_le_bytes());
            return;
        }
    }
    panic!("write_seg_u32: addr 0x{:x} not in any segment", addr);
}
