//! Compile an RV32 ELF into a canonical ISA binary with 2-register LRU cache.
//!
//! Each base operation has exactly ONE form with fixed register positions.
//! A compile-time 2-slot cache manages which original RV32 registers are in r0/r1.
//! No regalloc needed — cache management inserts load_reg/store_reg/swap as needed.
//!
//! Usage:
//!   rv32_compile_canonical [elf_path] [output_path]
//!
//! Defaults: json_query.elf, canonical.bin

use anyhow::{Result, bail};
use reg_analyzer::rv32::decode::{DecodedInst, decode_elf, get_elf_functions_named};
use reg_analyzer::rv32::cfg::{build_cfg, classify_jalr_x0};
use reg_analyzer::rv32_isa_vm::{MemSegment, MAILBOX_BASE};
use reg_analyzer::rv32_flat_vm::*;
use std::collections::HashMap;
use std::fs;

/// Compile-time cache state: which original RV32 registers are in r0/r1.
#[derive(Clone, Debug)]
struct CacheState {
    slots: [Option<u8>; 2], // slots[i] = Some(vreg) if ri holds vreg
    dirty: [bool; 2],       // dirty[i] = true if ri was written since last load/store
}

impl CacheState {
    fn new() -> Self {
        CacheState { slots: [None, None], dirty: [false, false] }
    }

    fn find(&self, vreg: u8) -> Option<usize> {
        if self.slots[0] == Some(vreg) { Some(0) }
        else if self.slots[1] == Some(vreg) { Some(1) }
        else { None }
    }
}

/// A canonical instruction to emit.
struct CanonInst {
    op: String,
    imm: Option<i32>,
    addr: u32, // original code address (for branch/jump resolution)
}

fn emit_store(out: &mut Vec<CanonInst>, slot: usize, vreg: u8) {
    assert_eq!(slot, 0, "store_reg always stores from r0");
    let addr = MAILBOX_BASE as i32 + (vreg as i32) * 4;
    out.push(CanonInst { op: "store_reg".into(), imm: Some(addr), addr: 0 });
}

fn emit_load(out: &mut Vec<CanonInst>, slot: usize, vreg: u8) {
    assert_eq!(slot, 0, "load_reg always loads into r0");
    let addr = MAILBOX_BASE as i32 + (vreg as i32) * 4;
    out.push(CanonInst { op: "load_reg".into(), imm: Some(addr), addr: 0 });
}

fn emit_swap(out: &mut Vec<CanonInst>) {
    out.push(CanonInst { op: "swap".into(), imm: None, addr: 0 });
}

/// Flush all dirty cache slots to register file.
fn flush_cache(cache: &mut CacheState, out: &mut Vec<CanonInst>) {
    // Must store from r0, so if r1 is dirty, swap first then store, then swap back.
    // Actually: store r0 if dirty, swap, store r0 (which is now old r1) if dirty, swap back.
    if cache.dirty[0] && cache.slots[0].is_some() {
        emit_store(out, 0, cache.slots[0].unwrap());
        cache.dirty[0] = false;
    }
    if cache.dirty[1] && cache.slots[1].is_some() {
        emit_swap(out);
        emit_store(out, 0, cache.slots[1].unwrap());
        emit_swap(out);
        cache.dirty[1] = false;
    }
}

/// After writing rd to slot 0, invalidate slot 1 if it held the same vreg
/// (the old value in slot 1 is now stale).
fn invalidate_stale(cache: &mut CacheState, rd: u8) {
    if cache.slots[1] == Some(rd) {
        cache.slots[1] = None;
        cache.dirty[1] = false;
    }
}

/// Invalidate cache (after a call returns, cache contents are unknown).
fn invalidate_cache(cache: &mut CacheState) {
    cache.slots = [None, None];
    cache.dirty = [false, false];
}

/// Ensure vreg is in the specified slot (0 or 1).
/// Returns whether a swap was needed (for tracking).
fn ensure_in_slot(cache: &mut CacheState, out: &mut Vec<CanonInst>, vreg: u8, target_slot: usize) {
    if vreg == 0 {
        // x0 is always 0 — we need to synthesize it
        // Load x0 from regfile (which should always be 0)
        // Just treat it like any other vreg
    }

    if cache.slots[target_slot] == Some(vreg) {
        return; // Already in correct slot
    }

    let other_slot = 1 - target_slot;

    if cache.slots[other_slot] == Some(vreg) {
        // vreg is in the OTHER slot — need swap
        emit_swap(out);
        // Swap cache state
        cache.slots.swap(0, 1);
        cache.dirty.swap(0, 1);
        return;
    }

    // vreg is not cached at all — need to evict target_slot and load
    if target_slot == 0 {
        // Evict r0 if dirty
        if cache.dirty[0] {
            if let Some(old) = cache.slots[0] {
                emit_store(out, 0, old);
            }
            cache.dirty[0] = false;
        }
        emit_load(out, 0, vreg);
        cache.slots[0] = Some(vreg);
        cache.dirty[0] = false;
    } else {
        // target_slot == 1: need vreg in r1
        // Strategy: swap r0↔r1, evict r0 (now old r1), load vreg into r0, swap back
        // But simpler: evict r1 via swap+store+swap, then load into r1 via swap+load+swap
        // Actually, even simpler approach:
        // 1. swap (now old_r1 is in r0, old_r0 is in r1)
        // 2. store old_r1 from r0 if dirty
        // 3. load vreg into r0
        // 4. swap (now vreg is in r1, old_r0 is back in r0)
        emit_swap(out);
        cache.slots.swap(0, 1);
        cache.dirty.swap(0, 1);
        // Now target vreg slot is r0
        if cache.dirty[0] {
            if let Some(old) = cache.slots[0] {
                emit_store(out, 0, old);
            }
            cache.dirty[0] = false;
        }
        emit_load(out, 0, vreg);
        cache.slots[0] = Some(vreg);
        cache.dirty[0] = false;
        // Swap back
        emit_swap(out);
        cache.slots.swap(0, 1);
        cache.dirty.swap(0, 1);
    }
}

/// Ensure two vregs are in r0 and r1 respectively.
fn ensure_two(cache: &mut CacheState, out: &mut Vec<CanonInst>, vreg0: u8, vreg1: u8) {
    if vreg0 == vreg1 {
        // Same vreg in both slots — load into r0, then copy to r1
        ensure_in_slot(cache, out, vreg0, 0);
        // For r1: we need the same value. Just swap+load would work but wasteful.
        // Actually, since vreg0==vreg1, after ensure_in_slot(0), r0 has it.
        // We need it in r1 too. Emit swap, then load again into r0, or just
        // accept that both r0 and r1 should have it.
        // Simplest: put in r0 first, then handle r1.
        ensure_in_slot(cache, out, vreg1, 1);
        return;
    }

    // Check if they're already in swapped positions
    if cache.slots[0] == Some(vreg1) && cache.slots[1] == Some(vreg0) {
        emit_swap(out);
        cache.slots.swap(0, 1);
        cache.dirty.swap(0, 1);
        return;
    }

    // If vreg0 is already in r0, just ensure vreg1 in r1
    if cache.slots[0] == Some(vreg0) {
        ensure_in_slot(cache, out, vreg1, 1);
        return;
    }

    // If vreg1 is already in r1, just ensure vreg0 in r0
    if cache.slots[1] == Some(vreg1) {
        ensure_in_slot(cache, out, vreg0, 0);
        return;
    }

    // Neither is in the right place. Load r0 first (may evict), then r1.
    ensure_in_slot(cache, out, vreg0, 0);
    ensure_in_slot(cache, out, vreg1, 1);
}

/// Determine the register needs of an instruction.
/// Returns (need_r0, need_r1, writes_r0) where need_r0/r1 are Option<vreg>.
fn classify_inst(inst: &DecodedInst) -> (Option<u8>, Option<u8>, Option<u8>) {
    match inst.op.as_str() {
        // R-type: rd = rs1 op rs2 → r0 = r0 op r1
        "add" | "sub" | "sll" | "srl" | "sra" | "slt" | "sltu" |
        "xor" | "or" | "and" | "mul" | "mulh" | "mulhsu" | "mulhu" |
        "div" | "divu" | "rem" | "remu" => {
            (inst.rs1, inst.rs2, inst.rd)
        }
        // I-type: rd = rs1 op imm → r0 = r0 op imm
        "addi" | "slti" | "sltiu" | "xori" | "ori" | "andi" |
        "slli" | "srli" | "srai" => {
            (inst.rs1, None, inst.rd)
        }
        // Load: rd = mem[rs1 + imm] → r0 = mem[r1 + imm]
        "lw" | "lb" | "lh" | "lbu" | "lhu" => {
            (None, inst.rs1, inst.rd)
        }
        // Store: mem[rs1 + imm] = rs2 → mem[r1 + imm] = r0
        "sw" | "sb" | "sh" => {
            (inst.rs2, inst.rs1, None)
        }
        // Branch: branch rs1, rs2, target → branch r0, r1, target
        "beq" | "bne" | "blt" | "bge" | "bltu" | "bgeu" => {
            (inst.rs1, inst.rs2, None)
        }
        // LUI: rd = imm << 12 → r0 = imm << 12
        "lui" => {
            (None, None, inst.rd)
        }
        // AUIPC: rd = pc + imm → r0 = pc + imm (will be resolved)
        "auipc" => {
            (None, None, inst.rd)
        }
        // JAL: jump (rd=0) or call (rd!=0)
        "jal" => {
            (None, None, None) // Handled specially
        }
        // JALR: jump/call via register
        "jalr" => {
            (inst.rs1, None, None) // rs1 in r0, handled specially
        }
        // ECALL
        "ecall" => {
            (None, None, None)
        }
        _ => {
            (None, None, None)
        }
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let elf_path = args.get(1)
        .cloned()
        .unwrap_or_else(|| "../rv32-build/target/riscv32i-unknown-none-elf/release/json_query".to_string());

    let output_path = args.get(2)
        .cloned()
        .unwrap_or_else(|| "canonical.bin".to_string());

    println!("=== rv32_compile_canonical (2 regs) ===");
    println!("  ELF:    {}", elf_path);
    println!("  Output: {}", output_path);

    // Step 1: Decode ELF
    let data = fs::read(&elf_path)?;
    let (decoded_raw, _text_addr, _text_len) = decode_elf(&data)?;
    let elf_funcs_named = get_elf_functions_named(&data)?;
    let mut decoded = decoded_raw;
    let elf_funcs: Vec<(u32, u32)> = elf_funcs_named.iter().map(|(a, s, _)| (*a, *s)).collect();
    let (jump_table_targets, jump_table_bases) = classify_jalr_x0(&mut decoded, &data, &elf_funcs_named);
    let _blocks = build_cfg(&decoded, &jump_table_targets);

    println!("  Decoded: {} instructions, {} functions", decoded.len(), elf_funcs.len());

    // Step 2: Resolve auipc pairs (same as rv32_compile_flat)
    let mut auipc_pair_target: HashMap<u32, u32> = HashMap::new();
    let mut auipc_dead: HashMap<u32, u32> = HashMap::new();
    let mut auipc_data_load: HashMap<u32, (u32, u32)> = HashMap::new();
    let mut auipc_data_consumer: HashMap<u32, i32> = HashMap::new();

    for i in 0..decoded.len() {
        if decoded[i].op != "auipc" { continue; }
        let auipc_rd = decoded[i].rd.unwrap();
        let auipc_result = decoded[i].addr.wrapping_add(decoded[i].imm.unwrap_or(0) as u32);

        let mut found = false;
        for j in (i+1)..decoded.len().min(i+10) {
            let reads = decoded[j].rs1 == Some(auipc_rd);
            if reads {
                let consumer_offset = decoded[j].imm.unwrap_or(0);
                let target = auipc_result.wrapping_add(consumer_offset as u32);

                match decoded[j].op.as_str() {
                    "jalr" => {
                        auipc_pair_target.insert(decoded[j].addr, target);
                        auipc_dead.insert(decoded[i].addr, auipc_result);
                    }
                    "addi" | "lw" | "sw" | "lb" | "lh" | "lbu" | "lhu" | "sb" | "sh" => {
                        let upper = target.wrapping_add(0x800) >> 12;
                        let lower = target.wrapping_sub(upper << 12) as i32;
                        auipc_data_load.insert(decoded[i].addr, (upper, decoded[j].addr));
                        auipc_data_consumer.insert(decoded[j].addr, lower);
                    }
                    _ => {}
                }
                found = true;
                break;
            }
            if decoded[j].rd == Some(auipc_rd) { break; }
        }
        if !found {
            eprintln!("  WARNING: auipc at 0x{:x} has no consumer", decoded[i].addr);
        }
    }

    // Step 3: Split decoded instructions into per-function streams
    // Sort functions by address
    let mut sorted_funcs: Vec<(u32, u32, String)> = elf_funcs_named.clone();
    sorted_funcs.sort_by_key(|(a, _, _)| *a);

    // Build function → decoded instructions mapping
    let mut func_insts: Vec<(u32, Vec<&DecodedInst>)> = Vec::new();
    for &(addr, size, _) in &sorted_funcs {
        let end = addr + size;
        let insts: Vec<&DecodedInst> = decoded.iter()
            .filter(|d| d.addr >= addr && d.addr < end)
            .collect();
        if !insts.is_empty() {
            func_insts.push((addr, insts));
        }
    }
    println!("  Functions with code: {}", func_insts.len());

    // Step 4: Canonicalize each function
    // For each function, walk instructions and emit canonical ops with cache management.
    // Track: addr → local instruction index for branch resolution.

    struct FuncResult {
        entry_addr: u32,
        canon_insts: Vec<CanonInst>,
        // Map from original code address → local canon instruction index
        addr_to_local: HashMap<u32, usize>,
    }

    let mut func_results: Vec<FuncResult> = Vec::new();
    let mut total_loads = 0u64;
    let mut total_stores = 0u64;
    let mut total_swaps = 0u64;
    let mut total_canon = 0u64;

    for &(func_entry, ref insts) in &func_insts {
        let mut out: Vec<CanonInst> = Vec::new();
        let mut cache = CacheState::new();
        let mut addr_to_local: HashMap<u32, usize> = HashMap::new();

        // Pre-scan: find all branch targets within this function so we can flush before them
        let func_addrs: std::collections::HashSet<u32> = insts.iter().map(|d| d.addr).collect();
        let mut branch_targets: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for inst in insts.iter() {
            match inst.op.as_str() {
                "beq" | "bne" | "blt" | "bge" | "bltu" | "bgeu" | "jal" => {
                    if let Some(imm) = inst.imm {
                        let target = (inst.addr as i64 + imm as i64) as u32;
                        if func_addrs.contains(&target) {
                            branch_targets.insert(target);
                        }
                    }
                }
                _ => {}
            }
        }

        let trace_canon = std::env::var("TRACE_CANON").is_ok();
        for inst in insts.iter() {
            // If this address is a branch target, flush cache first for consistent state
            if branch_targets.contains(&inst.addr) {
                flush_cache(&mut cache, &mut out);
                invalidate_cache(&mut cache);
            }

            if trace_canon {
                eprintln!("  [0x{:x}] {} rs1={:?} rs2={:?} rd={:?} imm={:?} | cache=[{:?},{:?}] dirty=[{},{}] out_idx={}",
                    inst.addr, inst.op, inst.rs1, inst.rs2, inst.rd, inst.imm,
                    cache.slots[0], cache.slots[1], cache.dirty[0], cache.dirty[1], out.len());
            }

            // Record the local index for this original address
            if let Some(old) = addr_to_local.insert(inst.addr, out.len()) {
                panic!("addr_to_local: duplicate addr 0x{:x} in func 0x{:x} (old={}, new={})",
                    inst.addr, func_entry, old, out.len());
            }

            // Handle auipc — convert to lui or skip if dead
            if inst.op == "auipc" {
                if auipc_dead.contains_key(&inst.addr) {
                    // Dead auipc (jalr pair resolved) — skip entirely
                    continue;
                }
                if let Some(&(upper, _consumer_addr)) = auipc_data_load.get(&inst.addr) {
                    // Data load: convert to lui with absolute upper bits
                    let rd = inst.rd.unwrap();
                    // Evict r0 if needed, then emit lui
                    if cache.dirty[0] {
                        if let Some(old) = cache.slots[0] {
                            emit_store(&mut out, 0, old);
                        }
                        cache.dirty[0] = false;
                    }
                    out.push(CanonInst { op: "lui".into(), imm: Some((upper << 12) as i32), addr: inst.addr });
                    cache.slots[0] = Some(rd);
                    cache.dirty[0] = true;
                    invalidate_stale(&mut cache, rd);
                    continue;
                }
                // Unresolved auipc — emit as lui (shouldn't happen in practice)
                let rd = inst.rd.unwrap();
                if cache.dirty[0] {
                    if let Some(old) = cache.slots[0] {
                        emit_store(&mut out, 0, old);
                    }
                    cache.dirty[0] = false;
                }
                let auipc_result = inst.addr.wrapping_add(inst.imm.unwrap_or(0) as u32);
                let upper = auipc_result.wrapping_add(0x800) >> 12;
                out.push(CanonInst { op: "lui".into(), imm: Some((upper << 12) as i32), addr: inst.addr });
                cache.slots[0] = Some(rd);
                cache.dirty[0] = true;
                invalidate_stale(&mut cache, rd);
                continue;
            }

            // Handle jal
            if inst.op == "jal" {
                let rd = inst.rd.unwrap_or(0);
                if rd == 0 {
                    // Unconditional jump — flush cache, emit jal
                    flush_cache(&mut cache, &mut out);
                    invalidate_cache(&mut cache);
                    out.push(CanonInst { op: "jal".into(), imm: inst.imm, addr: inst.addr });
                } else {
                    // Function call — flush cache, emit jal_call
                    flush_cache(&mut cache, &mut out);
                    out.push(CanonInst { op: "jal_call".into(), imm: inst.imm, addr: inst.addr });
                    invalidate_cache(&mut cache);
                }
                continue;
            }

            // Handle jalr
            if inst.op == "jalr" {
                let rd = inst.rd.unwrap_or(0);
                let rs1 = inst.rs1.unwrap_or(0);

                if auipc_pair_target.contains_key(&inst.addr) {
                    // Resolved auipc+jalr pair
                    let target = auipc_pair_target[&inst.addr];
                    if rd == 0 {
                        // Tail call / direct jump
                        flush_cache(&mut cache, &mut out);
                        invalidate_cache(&mut cache);
                        out.push(CanonInst { op: "jal".into(), imm: Some(target as i32), addr: inst.addr });
                    } else {
                        // Call
                        flush_cache(&mut cache, &mut out);
                        out.push(CanonInst { op: "jal_call".into(), imm: Some(target as i32), addr: inst.addr });
                        invalidate_cache(&mut cache);
                    }
                    continue;
                }

                if rd == 0 && rs1 == 1 {
                    // ret (jalr x0, ra, 0)
                    flush_cache(&mut cache, &mut out);
                    // Need ra (x1) in r0
                    ensure_in_slot(&mut cache, &mut out, 1, 0);
                    out.push(CanonInst { op: "ret".into(), imm: None, addr: inst.addr });
                    invalidate_cache(&mut cache);
                    continue;
                }

                if rd == 0 {
                    // Indirect jump (jr rs1) — for jump tables
                    flush_cache(&mut cache, &mut out);
                    ensure_in_slot(&mut cache, &mut out, rs1, 0);
                    out.push(CanonInst { op: "jalr".into(), imm: inst.imm, addr: inst.addr });
                    invalidate_cache(&mut cache);
                    continue;
                }

                // Indirect call
                flush_cache(&mut cache, &mut out);
                ensure_in_slot(&mut cache, &mut out, rs1, 0);
                out.push(CanonInst { op: "jalr_call".into(), imm: inst.imm, addr: inst.addr });
                invalidate_cache(&mut cache);
                continue;
            }

            // Handle ret (classified by classify_jalr_x0 from jalr x0, x1, 0)
            if inst.op == "ret" {
                flush_cache(&mut cache, &mut out);
                // Load ra (x1) into r0 for ret
                ensure_in_slot(&mut cache, &mut out, 1, 0);
                out.push(CanonInst { op: "ret".into(), imm: None, addr: inst.addr });
                invalidate_cache(&mut cache);
                continue;
            }

            // Handle jr_table (indirect jump via register, for switch statements)
            if inst.op == "jr_table" {
                let rs1 = inst.rs1.unwrap_or(0);
                flush_cache(&mut cache, &mut out);
                ensure_in_slot(&mut cache, &mut out, rs1, 0);
                out.push(CanonInst { op: "jr_table_idx".into(), imm: None, addr: inst.addr });
                invalidate_cache(&mut cache);
                continue;
            }

            // Handle jr_computed
            if inst.op == "jr_computed" {
                flush_cache(&mut cache, &mut out);
                out.push(CanonInst { op: "jr_computed".into(), imm: inst.imm, addr: inst.addr });
                invalidate_cache(&mut cache);
                continue;
            }

            // Handle ecall
            if inst.op == "ecall" {
                flush_cache(&mut cache, &mut out);
                out.push(CanonInst { op: "ecall".into(), imm: None, addr: inst.addr });
                continue;
            }

            // Handle branches — need to flush before branch (conservative)
            if matches!(inst.op.as_str(), "beq" | "bne" | "blt" | "bge" | "bltu" | "bgeu") {
                let rs1 = inst.rs1.unwrap_or(0);
                let rs2 = inst.rs2.unwrap_or(0);
                // Ensure operands, then flush, then branch
                ensure_two(&mut cache, &mut out, rs1, rs2);
                flush_cache(&mut cache, &mut out);

                out.push(CanonInst { op: inst.op.clone(), imm: inst.imm, addr: inst.addr });
                // After branch (fall-through): cache is unknown since branch target also invalidates
                invalidate_cache(&mut cache);
                continue;
            }

            // Regular instructions
            let (need_r0, need_r1, writes_rd) = classify_inst(inst);

            // Get the actual immediate (may be adjusted by auipc resolution)
            let imm = if let Some(&lower) = auipc_data_consumer.get(&inst.addr) {
                Some(lower)
            } else {
                inst.imm
            };

            match inst.op.as_str() {
                // R-type: r0 = r0 op r1
                "add" | "sub" | "sll" | "srl" | "sra" | "slt" | "sltu" |
                "xor" | "or" | "and" | "mul" | "mulh" | "mulhsu" | "mulhu" |
                "div" | "divu" | "rem" | "remu" => {
                    let rs1 = need_r0.unwrap_or(0);
                    let rs2 = need_r1.unwrap_or(0);
                    let rd = writes_rd.unwrap_or(0);
                    ensure_two(&mut cache, &mut out, rs1, rs2);
                    // If rd differs from current r0 vreg and r0 is dirty, store BEFORE the op
                    // (the op will overwrite r0, losing the old value)
                    if cache.slots[0] != Some(rd) && cache.dirty[0] {
                        if let Some(old) = cache.slots[0] {
                            emit_store(&mut out, 0, old);
                            cache.dirty[0] = false;
                        }
                    }
                    out.push(CanonInst { op: inst.op.clone(), imm: None, addr: inst.addr });
                    cache.slots[0] = Some(rd);
                    cache.dirty[0] = true;
                    invalidate_stale(&mut cache, rd);
                }

                // I-type: r0 = r0 op imm
                "addi" | "slti" | "sltiu" | "xori" | "ori" | "andi" |
                "slli" | "srli" | "srai" => {
                    let rs1 = need_r0.unwrap_or(0);
                    let rd = writes_rd.unwrap_or(0);
                    ensure_in_slot(&mut cache, &mut out, rs1, 0);
                    // If rd differs from current r0 vreg and r0 is dirty, store BEFORE the op
                    if cache.slots[0] != Some(rd) && cache.dirty[0] {
                        if let Some(old) = cache.slots[0] {
                            emit_store(&mut out, 0, old);
                            cache.dirty[0] = false;
                        }
                    }
                    out.push(CanonInst { op: inst.op.clone(), imm, addr: inst.addr });
                    cache.slots[0] = Some(rd);
                    cache.dirty[0] = true;
                    invalidate_stale(&mut cache, rd);
                }

                // Loads: r0 = mem[r1 + imm]
                "lw" | "lb" | "lh" | "lbu" | "lhu" => {
                    let base = need_r1.unwrap_or(0);
                    let rd = writes_rd.unwrap_or(0);
                    ensure_in_slot(&mut cache, &mut out, base, 1);
                    // Evict r0 if dirty (load will overwrite it)
                    if cache.dirty[0] {
                        if let Some(old) = cache.slots[0] {
                            emit_store(&mut out, 0, old);
                        }
                        cache.dirty[0] = false;
                    }
                    out.push(CanonInst { op: inst.op.clone(), imm, addr: inst.addr });
                    cache.slots[0] = Some(rd);
                    cache.dirty[0] = true;
                    invalidate_stale(&mut cache, rd);
                }

                // Stores: mem[r1 + imm] = r0
                "sw" | "sb" | "sh" => {
                    let val = need_r0.unwrap_or(0);
                    let base = need_r1.unwrap_or(0);
                    ensure_two(&mut cache, &mut out, val, base);
                    out.push(CanonInst { op: inst.op.clone(), imm, addr: inst.addr });
                    // No register write
                }

                // LUI: r0 = imm << 12
                "lui" => {
                    let rd = writes_rd.unwrap_or(0);
                    if cache.dirty[0] {
                        if let Some(old) = cache.slots[0] {
                            emit_store(&mut out, 0, old);
                        }
                        cache.dirty[0] = false;
                    }
                    out.push(CanonInst { op: "lui".into(), imm: inst.imm, addr: inst.addr });
                    cache.slots[0] = Some(rd);
                    cache.dirty[0] = true;
                    invalidate_stale(&mut cache, rd);
                }

                other => {
                    eprintln!("  WARNING: unhandled op '{}' at 0x{:x}", other, inst.addr);
                }
            }
        }

        // Count synthetic ops
        for ci in &out {
            match ci.op.as_str() {
                "load_reg" => total_loads += 1,
                "store_reg" => total_stores += 1,
                "swap" => total_swaps += 1,
                _ => total_canon += 1,
            }
        }

        func_results.push(FuncResult { entry_addr: func_entry, canon_insts: out, addr_to_local });
    }

    println!("  Canonical instructions: {} total", func_results.iter().map(|f| f.canon_insts.len()).sum::<usize>());
    println!("    load_reg:  {}", total_loads);
    println!("    store_reg: {}", total_stores);
    println!("    swap:      {}", total_swaps);
    println!("    canonical: {}", total_canon);

    // Step 5: Flatten all functions into one stream
    let mut flat_insts: Vec<CanonInst> = Vec::new();
    let mut func_entry_to_flat_idx: HashMap<u32, usize> = HashMap::new();

    struct FuncMapping {
        global_base: usize,
        addr_to_local: HashMap<u32, usize>,
    }
    let mut func_mappings: Vec<(u32, FuncMapping)> = Vec::new();

    for fr in &func_results {
        let global_base = flat_insts.len();
        // eprintln!("  func 0x{:x} → flat_base={} ({} canon insts)", fr.entry_addr, global_base, fr.canon_insts.len());
        if let Some(old) = func_entry_to_flat_idx.insert(fr.entry_addr, global_base) {
            panic!("func_entry_to_flat_idx: duplicate entry 0x{:x} (old={}, new={})",
                fr.entry_addr, old, global_base);
        }
        func_mappings.push((fr.entry_addr, FuncMapping {
            global_base,
            addr_to_local: fr.addr_to_local.clone(),
        }));
        for ci in &fr.canon_insts {
            flat_insts.push(CanonInst { op: ci.op.clone(), imm: ci.imm, addr: ci.addr });
        }
    }

    println!("  Flat instructions: {}", flat_insts.len());

    // Build addr_to_func_entry for target resolution
    let mut addr_to_func_entry: HashMap<u32, u32> = HashMap::new();
    for &(func_entry, ref insts) in &func_insts {
        for inst in insts {
            if let Some(old) = addr_to_func_entry.insert(inst.addr, func_entry) {
                if old != func_entry {
                    panic!("addr_to_func_entry: addr 0x{:x} claimed by both func 0x{:x} and 0x{:x}",
                        inst.addr, old, func_entry);
                }
            }
        }
    }

    // Step 6: Resolve targets
    // For branches: imm holds code offset → compute absolute code addr → find local index → global index
    // For jal: same
    // For jal_call: imm holds code offset or absolute addr → find func entry → global index

    // Helper: resolve code addr to global flat index
    let resolve_to_global = |code_addr: u32| -> Option<usize> {
        // Direct function entry?
        if let Some(&idx) = func_entry_to_flat_idx.get(&code_addr) {
            return Some(idx);
        }
        // Find containing function, then local index
        if let Some(&func_entry) = addr_to_func_entry.get(&code_addr) {
            if let Some(&func_flat_base) = func_entry_to_flat_idx.get(&func_entry) {
                let fm = func_mappings.iter().find(|(e, _)| *e == func_entry)?;
                let local_idx = fm.1.addr_to_local.get(&code_addr)?;
                return Some(func_flat_base + local_idx);
            }
        }
        None
    };

    for gi in 0..flat_insts.len() {
        let op = flat_insts[gi].op.clone();
        let addr = flat_insts[gi].addr;
        let imm = flat_insts[gi].imm.unwrap_or(0);

        match op.as_str() {
            "beq" | "bne" | "blt" | "bge" | "bltu" | "bgeu" => {
                // imm is code offset from addr
                let target_code_addr = (addr as i64 + imm as i64) as u32;
                if let Some(global_idx) = resolve_to_global(target_code_addr) {
                    flat_insts[gi].imm = Some(global_idx as i32);
                } else {
                    eprintln!("  WARNING: branch target 0x{:x} not found at gi={}", target_code_addr, gi);
                }
            }
            "jal" => {
                if addr != 0 {
                    // Check if this jal came from auipc+jalr resolution (addr is the jalr addr)
                    let is_auipc_resolved = auipc_pair_target.contains_key(&addr);
                    if is_auipc_resolved {
                        // imm is already an absolute code address
                        if let Some(global_idx) = resolve_to_global(imm as u32) {
                            flat_insts[gi].imm = Some(global_idx as i32);
                        } else {
                            eprintln!("  WARNING: jal abs target 0x{:x} not found at gi={}", imm, gi);
                        }
                    } else {
                        // Original jal: imm is code-relative offset
                        let target_code_addr = (addr as i64 + imm as i64) as u32;
                        if let Some(global_idx) = resolve_to_global(target_code_addr) {
                            flat_insts[gi].imm = Some(global_idx as i32);
                        } else {
                            eprintln!("  WARNING: jal target 0x{:x} not found at gi={} (addr=0x{:x}, imm={})", target_code_addr, gi, addr, imm);
                        }
                    }
                }
            }
            "jal_call" => {
                // imm is either code offset or absolute address
                if addr != 0 {
                    let target = if auipc_pair_target.get(&addr).is_some() {
                        // From auipc+jalr resolution — imm is already the absolute target
                        imm as u32
                    } else {
                        // Code-relative jal
                        (addr as i64 + imm as i64) as u32
                    };
                    if let Some(global_idx) = resolve_to_global(target) {
                        flat_insts[gi].imm = Some(global_idx as i32);
                    } else {
                        eprintln!("  WARNING: call target 0x{:x} not found at gi={}", target, gi);
                    }
                }
            }
            "jalr" | "jalr_call" => {
                // Indirect jump/call — register holds target at runtime
                // For jump tables: the table entries need to be patched (done below)
            }
            _ => {}
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

    let entry_pc = *func_entry_to_flat_idx.get(&elf_entry_addr)
        .ok_or_else(|| anyhow::anyhow!("entry 0x{:x} not in func table", elf_entry_addr))? as u32;
    eprintln!("  ELF entry: 0x{:x} → flat idx {}", elf_entry_addr, entry_pc);

    // Step 7: Build opcode table and encode
    let mut opcode_map: HashMap<String, u8> = HashMap::new();
    let mut opcode_table: Vec<OpcodeInfo> = Vec::new();

    // Build canonical specialized names
    // Each op has exactly one form, so the specialized name IS the op name
    // (except load_reg/store_reg which have no register suffixes)

    let mut code_table: Vec<u8> = Vec::with_capacity(flat_insts.len());
    let mut imm_table: Vec<i32> = Vec::with_capacity(flat_insts.len());

    for inst in &flat_insts {
        let (base_op, specialized, rd, rs1, rs2, orig_rd): (String, String, Option<u8>, Option<u8>, Option<u8>, Option<u8>) = match inst.op.as_str() {
            // Canonical forms — fixed register positions
            "add" | "sub" | "sll" | "srl" | "sra" | "slt" | "sltu" |
            "xor" | "or" | "and" | "mul" | "mulh" | "mulhsu" | "mulhu" |
            "div" | "divu" | "rem" | "remu" => {
                (inst.op.clone(), inst.op.clone(), Some(0u8), Some(0u8), Some(1u8), None)
            }
            "addi" | "slti" | "sltiu" | "xori" | "ori" | "andi" |
            "slli" | "srli" | "srai" => {
                (inst.op.clone(), inst.op.clone(), Some(0u8), Some(0u8), None, None)
            }
            "lw" | "lb" | "lh" | "lbu" | "lhu" => {
                (inst.op.clone(), inst.op.clone(), Some(0u8), Some(1u8), None, None)
            }
            "sw" | "sb" | "sh" => {
                (inst.op.clone(), inst.op.clone(), None, Some(1u8), Some(0u8), None)
            }
            "beq" | "bne" | "blt" | "bge" | "bltu" | "bgeu" => {
                (inst.op.clone(), inst.op.clone(), None, Some(0u8), Some(1u8), None)
            }
            "lui" => {
                ("lui".into(), "lui".into(), Some(0u8), None, None, None)
            }
            "jal" => {
                ("jal".into(), "jal".into(), None, None, None, None)
            }
            "jal_call" => {
                ("jal_call".into(), "jal_call".into(), None, None, None, Some(1u8))
            }
            "jalr" => {
                ("jalr".into(), "jalr".into(), None, Some(0u8), None, None)
            }
            "jalr_call" => {
                ("jalr_call".into(), "jalr_call".into(), Some(0u8), Some(0u8), None, Some(1u8))
            }
            "ret" => {
                ("ret".into(), "ret".into(), None, Some(0u8), None, None)
            }
            "swap" => {
                ("swap".into(), "swap".into(), Some(0u8), Some(1u8), None, None)
            }
            "load_reg" => {
                ("load_reg".into(), "load_reg".into(), Some(0u8), None, None, None)
            }
            "store_reg" => {
                ("store_reg".into(), "store_reg".into(), None, Some(0u8), None, None)
            }
            "jr_table_idx" => {
                ("jr_table_idx".into(), "jr_table_idx".into(), None, Some(0u8), None, None)
            }
            "jr_computed" => {
                ("jr_computed".into(), "jr_computed".into(), None, None, None, None)
            }
            "ecall" => {
                ("ecall".into(), "ecall".into(), None, None, None, None)
            }
            "halt" => {
                ("halt".into(), "halt".into(), None, None, None, None)
            }
            other => {
                bail!("Unknown canonical op: {}", other);
            }
        };

        if !opcode_map.contains_key(&specialized) {
            assert!(opcode_table.len() < 256, "canonical ISA exceeds 255 opcodes");
            let id = opcode_table.len() as u8;
            opcode_map.insert(specialized.clone(), id);
            opcode_table.push(OpcodeInfo {
                name: specialized.clone(),
                base_op: base_op.clone(),
                rd,
                rs1,
                rs2,
                orig_rd,
                orig_rs1: None,
                orig_rs2: None,
            });
        }

        let opcode_id = opcode_map[&specialized];
        code_table.push(opcode_id);

        let mut imm_val = inst.imm.unwrap_or(0);
        // Special handling for lui: store upper 20 bits >> 12
        if inst.op == "lui" {
            imm_val = ((imm_val as u32) >> 12) as i32;
        }
        imm_table.push(imm_val);
    }

    println!("  ISA size: {} unique opcodes", opcode_table.len());
    println!("  Entry PC: {} (0x{:x})", entry_pc, entry_pc);

    // Step 8: Extract ELF segments and patch pointers
    let mut segments = extract_elf_segments(&data)?;

    // Patch jump table entries
    for (&_jr_addr, &(table_base, num_entries)) in &jump_table_bases {
        for entry_i in 0..num_entries {
            let entry_addr = table_base.wrapping_add((entry_i * 4) as u32);
            let code_addr = read_seg_u32(&segments, entry_addr)
                .unwrap_or_else(|| panic!("Can't read jump table at 0x{:x}", entry_addr));
            if let Some(global_idx) = resolve_to_global(code_addr) {
                write_seg_u32(&mut segments, entry_addr, global_idx as u32);
            }
        }
    }

    // Patch function pointers in data segments
    let func_entry_addrs: std::collections::HashSet<u32> = func_entry_to_flat_idx.keys().copied().collect();
    let mut func_ptr_patches = 0;
    for seg in &mut segments {
        let mut off = 0;
        while off + 4 <= seg.data.len() {
            let val = u32::from_le_bytes([seg.data[off], seg.data[off+1], seg.data[off+2], seg.data[off+3]]);
            if let Some(&flat_idx) = func_entry_to_flat_idx.get(&val) {
                seg.data[off..off+4].copy_from_slice(&(flat_idx as u32).to_le_bytes());
                func_ptr_patches += 1;
            }
            off += 4;
        }
    }
    if func_ptr_patches > 0 {
        eprintln!("  Patched {} function pointer entries", func_ptr_patches);
    }

    // Step 9: Serialize
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
        num_regs: 2,
        entry_pc,
        segments,
        code_segment: Vec::new(),
        opcode_table: serial_opcode_table,
        imm_table,
        code_segment_u8: code_table,
    };

    let encoded = bincode::serialize(&program)?;
    fs::write(&output_path, &encoded)?;
    println!("  Written: {} bytes", encoded.len());

    Ok(())
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
        let end = seg.vaddr + seg.data.len() as u32;
        if addr >= seg.vaddr && addr + 4 <= end {
            let off = (addr - seg.vaddr) as usize;
            return Some(u32::from_le_bytes([
                seg.data[off], seg.data[off+1], seg.data[off+2], seg.data[off+3]
            ]));
        }
    }
    None
}

fn write_seg_u32(segments: &mut [MemSegment], addr: u32, val: u32) {
    for seg in segments.iter_mut() {
        let end = seg.vaddr + seg.data.len() as u32;
        if addr >= seg.vaddr && addr + 4 <= end {
            let off = (addr - seg.vaddr) as usize;
            seg.data[off..off+4].copy_from_slice(&val.to_le_bytes());
            return;
        }
    }
}
