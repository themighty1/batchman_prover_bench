//! Compile an RV32 ELF into a canonical 3-register ISA binary.
//!
//! Canonical forms:
//!   R-type:  r0 = r1 op r2
//!   I-type:  r0 = r1 op imm
//!   Load:    r0 = mem[r1 + imm]
//!   Store:   mem[r1 + imm] = r0
//!   Branch:  branch r0, r1, target
//!
//! 3-slot LRU cache, per-slot load/store (no swap needed).
//!
//! Usage:
//!   rv32_compile_canon3 [elf_path] [output_path]

use anyhow::{Result, bail};
use reg_analyzer::rv32::decode::{DecodedInst, decode_elf, get_elf_functions_named};
use reg_analyzer::rv32::cfg::{build_cfg, classify_jalr_x0};
use reg_analyzer::rv32_isa_vm::{MemSegment, MAILBOX_BASE};
use reg_analyzer::rv32_flat_vm::*;
use std::collections::HashMap;
use std::fs;

#[derive(Clone, Debug)]
struct CacheState {
    slots: [Option<u8>; 3],
    dirty: [bool; 3],
}

impl CacheState {
    fn new() -> Self {
        CacheState { slots: [None, None, None], dirty: [false, false, false] }
    }
}

#[derive(Clone)]
struct CanonInst {
    op: String,
    imm: Option<i32>,
    addr: u32,
}

fn regfile_addr(vreg: u8) -> i32 {
    (MAILBOX_BASE + (vreg as u32) * 4) as i32
}

// Scratch slots above the 32 RV32 registers, used for lh/lhu/sh expansion.
const SCRATCH_A: i32 = (MAILBOX_BASE + 33 * 4) as i32;
const SCRATCH_B: i32 = (MAILBOX_BASE + 34 * 4) as i32;

fn emit_store_slot(out: &mut Vec<CanonInst>, slot: usize, vreg: u8) {
    let op = match slot {
        0 => "sw_abs0", 1 => "sw_abs1", 2 => "sw_abs2",
        _ => panic!("bad slot"),
    };
    out.push(CanonInst { op: op.into(), imm: Some(regfile_addr(vreg)), addr: 0 });
}

fn emit_load_slot(out: &mut Vec<CanonInst>, slot: usize, vreg: u8) {
    let op = match slot {
        0 => "lw_abs0", 1 => "lw_abs1", 2 => "lw_abs2",
        _ => panic!("bad slot"),
    };
    out.push(CanonInst { op: op.into(), imm: Some(regfile_addr(vreg)), addr: 0 });
}

fn flush_cache(cache: &mut CacheState, out: &mut Vec<CanonInst>) {
    for i in 0..3 {
        if cache.dirty[i] {
            if let Some(vreg) = cache.slots[i] {
                emit_store_slot(out, i, vreg);
            }
            cache.dirty[i] = false;
        }
    }
}

fn invalidate_cache(cache: &mut CacheState) {
    cache.slots = [None, None, None];
    cache.dirty = [false, false, false];
}

/// After writing rd to a slot, invalidate other slots that claim the same vreg.
fn invalidate_stale(cache: &mut CacheState, rd: u8, written_slot: usize) {
    for i in 0..3 {
        if i == written_slot { continue; }
        if cache.slots[i] == Some(rd) {
            cache.slots[i] = None;
            cache.dirty[i] = false;
        }
    }
}

/// Ensure vreg is in the specified slot (0, 1, or 2).
fn ensure_in_slot(cache: &mut CacheState, out: &mut Vec<CanonInst>, vreg: u8, target_slot: usize) {
    if cache.slots[target_slot] == Some(vreg) {
        return;
    }

    // If vreg is in another slot and dirty, store it so regfile has the latest
    for other in 0..3 {
        if other == target_slot { continue; }
        if cache.slots[other] == Some(vreg) {
            if cache.dirty[other] {
                emit_store_slot(out, other, vreg);
                cache.dirty[other] = false;
            }
            cache.slots[other] = None;
            break;
        }
    }

    // Evict target slot if dirty
    if cache.dirty[target_slot] {
        if let Some(old) = cache.slots[target_slot] {
            emit_store_slot(out, target_slot, old);
        }
        cache.dirty[target_slot] = false;
    }

    // Load vreg into target slot
    emit_load_slot(out, target_slot, vreg);
    cache.slots[target_slot] = Some(vreg);
    cache.dirty[target_slot] = false;
}

/// Before writing rd to r0, store r0 if it holds a different dirty vreg.
fn save_r0_if_needed(cache: &mut CacheState, out: &mut Vec<CanonInst>, rd: u8) {
    if cache.slots[0] != Some(rd) && cache.dirty[0] {
        if let Some(old) = cache.slots[0] {
            emit_store_slot(out, 0, old);
        }
        cache.dirty[0] = false;
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let elf_path = args.get(1).cloned()
        .unwrap_or_else(|| "../rv32-build/target/riscv32i-unknown-none-elf/release/json_query".to_string());
    let output_path = args.get(2).cloned()
        .unwrap_or_else(|| "canonical3.bin".to_string());

    println!("=== rv32_compile_canon3 (3 regs) ===");
    println!("  ELF:    {}", elf_path);
    println!("  Output: {}", output_path);

    // Step 1: Decode ELF
    let data = fs::read(&elf_path)?;
    let (decoded_raw, _text_addr, _text_len) = decode_elf(&data)?;
    let elf_funcs_named = get_elf_functions_named(&data)?;
    let mut decoded = decoded_raw;
    let _elf_funcs: Vec<(u32, u32)> = elf_funcs_named.iter().map(|(a, s, _)| (*a, *s)).collect();
    let (jump_table_targets, jump_table_bases) = classify_jalr_x0(&mut decoded, &data, &elf_funcs_named);
    let _blocks = build_cfg(&decoded, &jump_table_targets);

    // println!("  Decoded: {} instructions, {} functions", decoded.len(), elf_funcs.len());

    // Step 2: Resolve auipc pairs
    let mut auipc_pair_target: HashMap<u32, u32> = HashMap::new();
    let mut auipc_dead: HashMap<u32, u32> = HashMap::new();
    let mut auipc_data_load: HashMap<u32, (u32, u32)> = HashMap::new();
    let mut auipc_data_consumer: HashMap<u32, i32> = HashMap::new();

    for i in 0..decoded.len() {
        if decoded[i].op != "auipc" { continue; }
        let auipc_rd = decoded[i].rd.unwrap();
        let auipc_result = decoded[i].addr.wrapping_add(decoded[i].imm.unwrap_or(0) as u32);

        for j in (i+1)..decoded.len().min(i+10) {
            if decoded[j].rs1 == Some(auipc_rd) {
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
                break;
            }
            if decoded[j].rd == Some(auipc_rd) { break; }
        }
    }

    // Step 3: Split into per-function streams
    let mut sorted_funcs: Vec<(u32, u32, String)> = elf_funcs_named.clone();
    sorted_funcs.sort_by_key(|(a, _, _)| *a);

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
    // println!("  Functions with code: {}", func_insts.len());

    // Step 4: Canonicalize each function (3-slot cache)
    struct FuncResult {
        entry_addr: u32,
        canon_insts: Vec<CanonInst>,
        addr_to_local: HashMap<u32, usize>,
    }

    let mut func_results: Vec<FuncResult> = Vec::new();
    let mut _total_loads = 0u64;
    let mut _total_stores = 0u64;
    let mut _total_canon = 0u64;

    for &(func_entry, ref insts) in &func_insts {
        let mut out: Vec<CanonInst> = Vec::new();
        let mut cache = CacheState::new();
        let mut addr_to_local: HashMap<u32, usize> = HashMap::new();

        // Pre-scan: branch targets
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

        for inst in insts.iter() {
            // Flush at branch targets
            if branch_targets.contains(&inst.addr) {
                flush_cache(&mut cache, &mut out);
                invalidate_cache(&mut cache);
            }

            addr_to_local.insert(inst.addr, out.len());

            // --- auipc ---
            if inst.op == "auipc" {
                if auipc_dead.contains_key(&inst.addr) {
                    continue;
                }
                if let Some(&(upper, _)) = auipc_data_load.get(&inst.addr) {
                    let rd = inst.rd.unwrap();
                    save_r0_if_needed(&mut cache, &mut out, rd);
                    out.push(CanonInst { op: "lui".into(), imm: Some((upper << 12) as i32), addr: inst.addr });
                    cache.slots[0] = Some(rd);
                    cache.dirty[0] = true;
                    invalidate_stale(&mut cache, rd, 0);
                    continue;
                }
                let rd = inst.rd.unwrap();
                save_r0_if_needed(&mut cache, &mut out, rd);
                let auipc_result = inst.addr.wrapping_add(inst.imm.unwrap_or(0) as u32);
                let upper = auipc_result.wrapping_add(0x800) >> 12;
                out.push(CanonInst { op: "lui".into(), imm: Some((upper << 12) as i32), addr: inst.addr });
                cache.slots[0] = Some(rd);
                cache.dirty[0] = true;
                invalidate_stale(&mut cache, rd, 0);
                continue;
            }

            // --- jal ---
            if inst.op == "jal" {
                let rd = inst.rd.unwrap_or(0);
                if rd == 0 {
                    flush_cache(&mut cache, &mut out);
                    invalidate_cache(&mut cache);
                    out.push(CanonInst { op: "jal".into(), imm: inst.imm, addr: inst.addr });
                } else {
                    flush_cache(&mut cache, &mut out);
                    out.push(CanonInst { op: "jal_call".into(), imm: inst.imm, addr: inst.addr });
                    invalidate_cache(&mut cache);
                }
                continue;
            }

            // --- jalr ---
            if inst.op == "jalr" {
                let rd = inst.rd.unwrap_or(0);
                let rs1 = inst.rs1.unwrap_or(0);

                if auipc_pair_target.contains_key(&inst.addr) {
                    let target = auipc_pair_target[&inst.addr];
                    if rd == 0 {
                        flush_cache(&mut cache, &mut out);
                        invalidate_cache(&mut cache);
                        out.push(CanonInst { op: "jal".into(), imm: Some(target as i32), addr: inst.addr });
                    } else {
                        flush_cache(&mut cache, &mut out);
                        out.push(CanonInst { op: "jal_call".into(), imm: Some(target as i32), addr: inst.addr });
                        invalidate_cache(&mut cache);
                    }
                    continue;
                }

                if rd == 0 && rs1 == 1 {
                    flush_cache(&mut cache, &mut out);
                    ensure_in_slot(&mut cache, &mut out, 1, 0);
                    out.push(CanonInst { op: "ret".into(), imm: None, addr: inst.addr });
                    invalidate_cache(&mut cache);
                    continue;
                }

                if rd == 0 {
                    flush_cache(&mut cache, &mut out);
                    ensure_in_slot(&mut cache, &mut out, rs1, 0);
                    out.push(CanonInst { op: "jalr".into(), imm: inst.imm, addr: inst.addr });
                    invalidate_cache(&mut cache);
                    continue;
                }

                flush_cache(&mut cache, &mut out);
                ensure_in_slot(&mut cache, &mut out, rs1, 0);
                out.push(CanonInst { op: "jalr_call".into(), imm: inst.imm, addr: inst.addr });
                invalidate_cache(&mut cache);
                continue;
            }

            // --- ret ---
            if inst.op == "ret" {
                flush_cache(&mut cache, &mut out);
                ensure_in_slot(&mut cache, &mut out, 1, 0);
                out.push(CanonInst { op: "ret".into(), imm: None, addr: inst.addr });
                invalidate_cache(&mut cache);
                continue;
            }

            // --- jr_table ---
            if inst.op == "jr_table" {
                let rs1 = inst.rs1.unwrap_or(0);
                flush_cache(&mut cache, &mut out);
                ensure_in_slot(&mut cache, &mut out, rs1, 0);
                out.push(CanonInst { op: "jr_table_idx".into(), imm: None, addr: inst.addr });
                invalidate_cache(&mut cache);
                continue;
            }

            // --- jr_computed ---
            if inst.op == "jr_computed" {
                flush_cache(&mut cache, &mut out);
                out.push(CanonInst { op: "jr_computed".into(), imm: inst.imm, addr: inst.addr });
                invalidate_cache(&mut cache);
                continue;
            }

            // --- ecall ---
            if inst.op == "ecall" {
                flush_cache(&mut cache, &mut out);
                out.push(CanonInst { op: "ecall".into(), imm: None, addr: inst.addr });
                continue;
            }

            // --- branches: r0=rs1, r1=rs2, flush, branch ---
            if matches!(inst.op.as_str(), "beq" | "bne" | "blt" | "bge" | "bltu" | "bgeu") {
                let rs1 = inst.rs1.unwrap_or(0);
                let rs2 = inst.rs2.unwrap_or(0);
                ensure_in_slot(&mut cache, &mut out, rs1, 0);
                ensure_in_slot(&mut cache, &mut out, rs2, 1);
                flush_cache(&mut cache, &mut out);
                out.push(CanonInst { op: inst.op.clone(), imm: inst.imm, addr: inst.addr });
                invalidate_cache(&mut cache);
                continue;
            }

            // --- Regular instructions ---
            let imm = if let Some(&lower) = auipc_data_consumer.get(&inst.addr) {
                Some(lower)
            } else {
                inst.imm
            };

            match inst.op.as_str() {
                // R-type: r0 = r1 op r2
                "add" | "sub" | "sll" | "srl" | "sra" | "slt" | "sltu" |
                "xor" | "or" | "and" | "mul" | "mulh" | "mulhsu" | "mulhu" |
                "div" | "divu" | "rem" | "remu" => {
                    let rs1 = inst.rs1.unwrap_or(0);
                    let rs2 = inst.rs2.unwrap_or(0);
                    let rd = inst.rd.unwrap_or(0);
                    ensure_in_slot(&mut cache, &mut out, rs1, 1);
                    ensure_in_slot(&mut cache, &mut out, rs2, 2);
                    save_r0_if_needed(&mut cache, &mut out, rd);
                    out.push(CanonInst { op: inst.op.clone(), imm: None, addr: inst.addr });
                    cache.slots[0] = Some(rd);
                    cache.dirty[0] = true;
                    invalidate_stale(&mut cache, rd, 0);
                }

                // I-type: r0 = r1 op imm
                "addi" | "slti" | "sltiu" | "xori" | "ori" | "andi" |
                "slli" | "srli" | "srai" => {
                    let rs1 = inst.rs1.unwrap_or(0);
                    let rd = inst.rd.unwrap_or(0);
                    ensure_in_slot(&mut cache, &mut out, rs1, 1);
                    save_r0_if_needed(&mut cache, &mut out, rd);
                    out.push(CanonInst { op: inst.op.clone(), imm, addr: inst.addr });
                    cache.slots[0] = Some(rd);
                    cache.dirty[0] = true;
                    invalidate_stale(&mut cache, rd, 0);
                }

                // Loads: r0 = mem32[r1 + imm]
                "lw" => {
                    let base = inst.rs1.unwrap_or(0);
                    let rd = inst.rd.unwrap_or(0);
                    ensure_in_slot(&mut cache, &mut out, base, 1);
                    save_r0_if_needed(&mut cache, &mut out, rd);
                    out.push(CanonInst { op: inst.op.clone(), imm, addr: inst.addr });
                    cache.slots[0] = Some(rd);
                    cache.dirty[0] = true;
                    invalidate_stale(&mut cache, rd, 0);
                }

                // lb → lw_aligned + byte_sel_r2 + sext8
                "lb" => {
                    let base = inst.rs1.unwrap_or(0);
                    let rd = inst.rd.unwrap_or(0);
                    if cache.dirty[2] {
                        if let Some(v) = cache.slots[2] { emit_store_slot(&mut out, 2, v); }
                        cache.dirty[2] = false;
                    }
                    cache.slots[2] = None;
                    ensure_in_slot(&mut cache, &mut out, base, 1);
                    save_r0_if_needed(&mut cache, &mut out, rd);
                    out.push(CanonInst { op: "lw_aligned".into(), imm, addr: inst.addr });
                    out.push(CanonInst { op: "byte_sel_r2".into(), imm: None, addr: inst.addr });
                    out.push(CanonInst { op: "sext8".into(), imm: None, addr: inst.addr });
                    cache.slots[0] = Some(rd);
                    cache.dirty[0] = true;
                    cache.slots[2] = None;
                    cache.dirty[2] = false;
                    invalidate_stale(&mut cache, rd, 0);
                }

                // lbu → lw_aligned (r0=cell, r2=byte_offset) + byte_sel_r2 (r0=byte)
                "lbu" => {
                    let base = inst.rs1.unwrap_or(0);
                    let rd = inst.rd.unwrap_or(0);
                    // lw_aligned clobbers r2: flush slot 2 if dirty
                    if cache.dirty[2] {
                        if let Some(v) = cache.slots[2] { emit_store_slot(&mut out, 2, v); }
                        cache.dirty[2] = false;
                    }
                    cache.slots[2] = None;
                    ensure_in_slot(&mut cache, &mut out, base, 1);
                    save_r0_if_needed(&mut cache, &mut out, rd);
                    out.push(CanonInst { op: "lw_aligned".into(), imm, addr: inst.addr });
                    out.push(CanonInst { op: "byte_sel_r2".into(), imm: None, addr: inst.addr });
                    cache.slots[0] = Some(rd);
                    cache.dirty[0] = true;
                    cache.slots[2] = None;
                    cache.dirty[2] = false;
                    invalidate_stale(&mut cache, rd, 0);
                }

                // lhu → lw_aligned(low)+byte_sel + save_low + lw_aligned(high)+byte_sel
                //       + save_high + r1=high + slli8 + r1=low + or
                // lh  → same + r1=result + slli16 + r1=result<<16 + srai16
                "lhu" | "lh" => {
                    let base = inst.rs1.unwrap_or(0);
                    let rd = inst.rd.unwrap_or(0);
                    let offset = imm.unwrap_or(0);
                    // flush slot 2: lw_aligned clobbers r2
                    if cache.dirty[2] {
                        if let Some(v) = cache.slots[2] { emit_store_slot(&mut out, 2, v); }
                        cache.dirty[2] = false;
                    }
                    cache.slots[2] = None;
                    ensure_in_slot(&mut cache, &mut out, base, 1);
                    save_r0_if_needed(&mut cache, &mut out, rd);
                    // load low byte into r0
                    out.push(CanonInst { op: "lw_aligned".into(), imm: Some(offset), addr: inst.addr });
                    out.push(CanonInst { op: "byte_sel_r2".into(), imm: None, addr: inst.addr });
                    out.push(CanonInst { op: "sw_abs0".into(), imm: Some(SCRATCH_A), addr: inst.addr });
                    // load high byte into r0 (r1 still = base)
                    out.push(CanonInst { op: "lw_aligned".into(), imm: Some(offset + 1), addr: inst.addr });
                    out.push(CanonInst { op: "byte_sel_r2".into(), imm: None, addr: inst.addr });
                    // save high byte, move into r1 for shift
                    out.push(CanonInst { op: "sw_abs0".into(), imm: Some(SCRATCH_B), addr: inst.addr });
                    out.push(CanonInst { op: "lw_abs1".into(), imm: Some(SCRATCH_B), addr: inst.addr });
                    out.push(CanonInst { op: "slli".into(), imm: Some(8), addr: inst.addr });
                    // bring low byte into r1, OR together
                    out.push(CanonInst { op: "lw_abs1".into(), imm: Some(SCRATCH_A), addr: inst.addr });
                    out.push(CanonInst { op: "or".into(), imm: None, addr: inst.addr });
                    // for lh: sign-extend via r1=result, slli16, r1=result<<16, srai16
                    if inst.op == "lh" {
                        out.push(CanonInst { op: "sw_abs0".into(), imm: Some(SCRATCH_A), addr: inst.addr });
                        out.push(CanonInst { op: "lw_abs1".into(), imm: Some(SCRATCH_A), addr: inst.addr });
                        out.push(CanonInst { op: "slli".into(), imm: Some(16), addr: inst.addr });
                        out.push(CanonInst { op: "sw_abs0".into(), imm: Some(SCRATCH_A), addr: inst.addr });
                        out.push(CanonInst { op: "lw_abs1".into(), imm: Some(SCRATCH_A), addr: inst.addr });
                        out.push(CanonInst { op: "srai".into(), imm: Some(16), addr: inst.addr });
                    }
                    cache.slots[0] = Some(rd);
                    cache.dirty[0] = true;
                    cache.slots[1] = None;
                    cache.dirty[1] = false;
                    cache.slots[2] = None;
                    cache.dirty[2] = false;
                    invalidate_stale(&mut cache, rd, 0);
                }

                // Stores: mem[r1 + imm] = r0
                "sw" => {
                    let val = inst.rs2.unwrap_or(0);
                    let base = inst.rs1.unwrap_or(0);
                    ensure_in_slot(&mut cache, &mut out, val, 0);
                    ensure_in_slot(&mut cache, &mut out, base, 1);
                    out.push(CanonInst { op: "sw".into(), imm, addr: inst.addr });
                }

                // sb → sw_aligned (read-modify-write, no branching)
                // After decomposition, sw_aligned clobbers r0 and r2, so
                // flush dirty slots first, then invalidate.
                "sb" => {
                    let val = inst.rs2.unwrap_or(0);
                    let base = inst.rs1.unwrap_or(0);
                    // Flush r2 if dirty BEFORE clobbering (sw_aligned decomp uses r2)
                    if cache.dirty[2] {
                        if let Some(v) = cache.slots[2] { emit_store_slot(&mut out, 2, v); }
                    }
                    ensure_in_slot(&mut cache, &mut out, val, 0);
                    ensure_in_slot(&mut cache, &mut out, base, 1);
                    out.push(CanonInst { op: "sw_aligned".into(), imm, addr: inst.addr });
                    // r0 was clean (just loaded by ensure_in_slot), no flush needed
                    cache.slots[0] = None;
                    cache.dirty[0] = false;
                    cache.slots[2] = None;
                    cache.dirty[2] = false;
                }

                // sh → save_val + sw_aligned(low) + load_val + srli8 + load_base + sw_aligned(high)
                // Val is saved to SCRATCH_B (not SCRATCH_A) because sw_aligned
                // decomposition uses SCRATCH_A internally.
                "sh" => {
                    let val = inst.rs2.unwrap_or(0);
                    let base = inst.rs1.unwrap_or(0);
                    let offset = imm.unwrap_or(0);
                    // Flush r2 if dirty BEFORE clobbering (sw_aligned decomp uses r2)
                    if cache.dirty[2] {
                        if let Some(v) = cache.slots[2] { emit_store_slot(&mut out, 2, v); }
                    }
                    ensure_in_slot(&mut cache, &mut out, val, 0);
                    ensure_in_slot(&mut cache, &mut out, base, 1);
                    // save val to SCRATCH_B before sw_aligned clobbers r0
                    out.push(CanonInst { op: "sw_abs0".into(), imm: Some(SCRATCH_B), addr: inst.addr });
                    // store low byte via read-modify-write
                    out.push(CanonInst { op: "sw_aligned".into(), imm: Some(offset), addr: inst.addr });
                    // recover val from SCRATCH_B into r1 for shift
                    out.push(CanonInst { op: "lw_abs1".into(), imm: Some(SCRATCH_B), addr: inst.addr });
                    out.push(CanonInst { op: "srli".into(), imm: Some(8), addr: inst.addr });
                    // reload base from regfile (ensure_in_slot left it clean there)
                    out.push(CanonInst { op: "lw_abs1".into(), imm: Some(regfile_addr(base)), addr: inst.addr });
                    // store high byte via read-modify-write
                    out.push(CanonInst { op: "sw_aligned".into(), imm: Some(offset + 1), addr: inst.addr });
                    cache.slots[0] = None;
                    cache.dirty[0] = false;
                    cache.slots[1] = None;
                    cache.dirty[1] = false;
                    cache.slots[2] = None;
                    cache.dirty[2] = false;
                }

                // LUI: r0 = imm << 12
                "lui" => {
                    let rd = inst.rd.unwrap_or(0);
                    save_r0_if_needed(&mut cache, &mut out, rd);
                    out.push(CanonInst { op: "lui".into(), imm: inst.imm, addr: inst.addr });
                    cache.slots[0] = Some(rd);
                    cache.dirty[0] = true;
                    invalidate_stale(&mut cache, rd, 0);
                }

                other => {
                    eprintln!("  WARNING: unhandled op '{}' at 0x{:x}", other, inst.addr);
                }
            }
        }

        for ci in &out {
            match ci.op.as_str() {
                "lw_abs0" | "lw_abs1" | "lw_abs2" => _total_loads += 1,
                "sw_abs0" | "sw_abs1" | "sw_abs2" => _total_stores += 1,
                _ => _total_canon += 1,
            }
        }

        func_results.push(FuncResult { entry_addr: func_entry, canon_insts: out, addr_to_local });
    }

    // println!("  Canonical instructions: {} total", func_results.iter().map(|f| f.canon_insts.len()).sum::<usize>());
    // println!("    lw_abs:    {}", total_loads);
    // println!("    sw_abs:    {}", total_stores);
    // println!("    canonical: {}", total_canon);

    // Step 5: Flatten
    let mut flat_insts: Vec<CanonInst> = Vec::new();
    let mut func_entry_to_flat_idx: HashMap<u32, usize> = HashMap::new();

    struct FuncMapping {
        _global_base: usize,
        addr_to_local: HashMap<u32, usize>,
    }
    let mut func_mappings: Vec<(u32, FuncMapping)> = Vec::new();

    for fr in &func_results {
        let global_base = flat_insts.len();
        func_entry_to_flat_idx.insert(fr.entry_addr, global_base);
        func_mappings.push((fr.entry_addr, FuncMapping {
            _global_base: global_base,
            addr_to_local: fr.addr_to_local.clone(),
        }));
        for ci in &fr.canon_insts {
            flat_insts.push(CanonInst { op: ci.op.clone(), imm: ci.imm, addr: ci.addr });
        }
    }

    // println!("  Flat instructions: {}", flat_insts.len());

    // Build addr_to_func_entry
    let mut addr_to_func_entry: HashMap<u32, u32> = HashMap::new();
    for &(func_entry, ref insts) in &func_insts {
        for inst in insts {
            addr_to_func_entry.insert(inst.addr, func_entry);
        }
    }

    // Step 6: Resolve targets
    let resolve_to_global = |code_addr: u32| -> Option<usize> {
        if let Some(&idx) = func_entry_to_flat_idx.get(&code_addr) {
            return Some(idx);
        }
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
                let target_code_addr = (addr as i64 + imm as i64) as u32;
                if let Some(global_idx) = resolve_to_global(target_code_addr) {
                    flat_insts[gi].imm = Some(global_idx as i32);
                }
            }
            "jal" => {
                if addr != 0 {
                    let is_auipc = auipc_pair_target.contains_key(&addr);
                    let target_code_addr = if is_auipc { imm as u32 }
                        else { (addr as i64 + imm as i64) as u32 };
                    if let Some(global_idx) = resolve_to_global(target_code_addr) {
                        flat_insts[gi].imm = Some(global_idx as i32);
                    }
                }
            }
            "jal_call" => {
                if addr != 0 {
                    let target = if auipc_pair_target.contains_key(&addr) { imm as u32 }
                        else { (addr as i64 + imm as i64) as u32 };
                    if let Some(global_idx) = resolve_to_global(target) {
                        flat_insts[gi].imm = Some(global_idx as i32);
                    }
                }
            }
            _ => {}
        }
    }

    // ELF entry
    let elf_entry_addr = {
        use object::elf::*;
        use object::read::elf::FileHeader as _;
        use object::Endianness;
        let elf = FileHeader32::<Endianness>::parse(data.as_slice())?;
        let endian = elf.endian()?;
        elf.e_entry.get(endian)
    };
    let mut entry_pc = *func_entry_to_flat_idx.get(&elf_entry_addr)
        .ok_or_else(|| anyhow::anyhow!("entry 0x{:x} not in func table", elf_entry_addr))? as u32;

    // Step 6b: Decompose slli/srli/srai into fixed-shift ops.
    //
    // Shifting by a dynamic (immediate-encoded) amount is expensive in a boolean
    // circuit: it requires a barrel shifter — a chain of muxes gated by each bit
    // of the shift amount.  Fixed-shift ops turn each shift into simple wire
    // re-routing (constant propagation), eliminating the mux chain entirely.
    //
    // Only 9 distinct shift amounts appear in practice.  We cover them with
    // power-of-two building blocks {1, 4, 8, 16} plus dedicated shift-by-31,
    // growing the ISA by 8 ops but removing all shift immediates.  Rare amounts
    // are composed from multiple fixed-shifts at +0.3% total steps.
    let shift_old_to_new: Vec<usize>;
    let skip_shift_decomp = std::env::var("NO_SHIFT_DECOMP").is_ok();
    if skip_shift_decomp {
        shift_old_to_new = (0..flat_insts.len()).collect();
        // println!("  Shift decomposition: SKIPPED");
    } else {
        let blocks: &[i32] = &[31, 16, 8, 4, 1];

        // Build old→new index mapping (each old instruction may expand to 1..N new ones)
        let mut new_insts: Vec<CanonInst> = Vec::with_capacity(flat_insts.len());
        let mut old_to_new: Vec<usize> = Vec::with_capacity(flat_insts.len());

        let mut _slli_decomposed = 0u64;
        let mut _srli_decomposed = 0u64;
        let mut _srai_decomposed = 0u64;

        for inst in &flat_insts {
            old_to_new.push(new_insts.len());

            match inst.op.as_str() {
                "slli" | "srli" | "srai" => {
                    let shamt = inst.imm.unwrap_or(0) & 0x1F;
                    let prefix = match inst.op.as_str() {
                        "slli" => { _slli_decomposed += 1; "sll" }
                        "srli" => { _srli_decomposed += 1; "srl" }
                        "srai" => { _srai_decomposed += 1; "sra" }
                        _ => unreachable!(),
                    };
                    // Fixed shifts operate as r0 = r0 << N (chaining through r0).
                    // The original slli/srli/srai reads r1, so we always prepend
                    // "addi 0" (r0 = r1 + 0) to copy r1 → r0.
                    new_insts.push(CanonInst {
                        op: "addi".into(), imm: Some(0), addr: inst.addr,
                    });
                    let mut remaining = shamt;
                    for &b in blocks {
                        while remaining >= b {
                            remaining -= b;
                            let op_name = format!("{}{}", prefix, b);
                            new_insts.push(CanonInst {
                                op: op_name,
                                imm: None,
                                addr: 0,
                            });
                        }
                    }
                    assert_eq!(remaining, 0, "shift amount {} not fully decomposed", shamt);
                }
                _ => {
                    new_insts.push(inst.clone());
                }
            }
        }

        // Remap branch/jump targets from old indices to new indices
        for inst in &mut new_insts {
            match inst.op.as_str() {
                "beq" | "bne" | "blt" | "bge" | "bltu" | "bgeu"
                | "jal" | "jal_call" | "jr_computed" => {
                    if let Some(ref mut imm) = inst.imm {
                        let old_idx = *imm as usize;
                        if old_idx < old_to_new.len() {
                            *imm = old_to_new[old_idx] as i32;
                        }
                    }
                }
                _ => {}
            }
        }

        // Remap entry_pc
        entry_pc = old_to_new[entry_pc as usize] as u32;

        let _expanded = new_insts.len() - flat_insts.len();
        // println!("  Shift decomposition: slli={} srli={} srai={} → +{} instructions",
        //     _slli_decomposed, _srli_decomposed, _srai_decomposed, _expanded);

        flat_insts = new_insts;
        shift_old_to_new = old_to_new;
    }

    // Step 6c: Decompose sw_aligned into sub-ops.
    //
    // sw_aligned is the largest single op gate-count-wise: it performs a full
    // read-modify-write with dynamic byte-position indexing (address calc,
    // alignment mask, word load, byte clear/insert, word store — all in one op).
    // Decomposing it into 4 simpler ops reduces peak gate count per op:
    //   sw_abs0 SCRATCH_A   — save the byte value to scratch
    //   lw_aligned imm      — load existing word, get byte offset in r2
    //   byte_ins_r2         — insert scratch byte into word at position r2
    //   sw_waligned imm     — write modified word to aligned address
    // Each sub-op has a much smaller gate footprint in a boolean circuit.
    let sw_old_to_new: Vec<usize>;
    let skip_sw_decomp = std::env::var("NO_SW_DECOMP").is_ok();
    if skip_sw_decomp {
        sw_old_to_new = (0..flat_insts.len()).collect();
        // println!("  sw_aligned decomposition: SKIPPED");
    } else {
        let mut new_insts: Vec<CanonInst> = Vec::with_capacity(flat_insts.len());
        let mut old_to_new: Vec<usize> = Vec::with_capacity(flat_insts.len());
        let mut _decomposed = 0u64;

        for inst in &flat_insts {
            old_to_new.push(new_insts.len());
            if inst.op == "sw_aligned" {
                _decomposed += 1;
                // 1. Save r0 (byte to store) to SCRATCH_A
                new_insts.push(CanonInst { op: "sw_abs0".into(), imm: Some(SCRATCH_A), addr: inst.addr });
                // 2. Load existing word + get byte offset
                new_insts.push(CanonInst { op: "lw_aligned".into(), imm: inst.imm, addr: inst.addr });
                // 3. Insert SCRATCH_A[7:0] at byte position r2
                new_insts.push(CanonInst { op: "byte_ins_r2".into(), imm: None, addr: inst.addr });
                // 4. Write modified word to aligned address
                new_insts.push(CanonInst { op: "sw_waligned".into(), imm: inst.imm, addr: inst.addr });
            } else {
                new_insts.push(inst.clone());
            }
        }

        // Remap branch/jump targets from old indices to new indices
        for inst in &mut new_insts {
            match inst.op.as_str() {
                "beq" | "bne" | "blt" | "bge" | "bltu" | "bgeu"
                | "jal" | "jal_call" | "jr_computed" => {
                    if let Some(ref mut imm) = inst.imm {
                        let old_idx = *imm as usize;
                        if old_idx < old_to_new.len() {
                            *imm = old_to_new[old_idx] as i32;
                        }
                    }
                }
                _ => {}
            }
        }

        // Remap entry_pc
        entry_pc = old_to_new[entry_pc as usize] as u32;

        let _expanded = new_insts.len() - flat_insts.len();
        // println!("  sw_aligned decomposition: {} → +{} instructions", _decomposed, _expanded);

        flat_insts = new_insts;
        sw_old_to_new = old_to_new;
    }

    // Compose pass mappings: original → post-shift → post-sw = final indices.
    // Jump table and func_ptr patches need original-to-final mapping.
    let final_old_to_new: Vec<usize> = shift_old_to_new.iter()
        .map(|&shift_idx| sw_old_to_new[shift_idx])
        .collect();

    // Step 7: Build opcode table
    let mut opcode_map: HashMap<String, u8> = HashMap::new();
    let mut opcode_table: Vec<OpcodeInfo> = Vec::new();
    let mut code_table: Vec<u8> = Vec::with_capacity(flat_insts.len());
    let mut imm_table: Vec<i32> = Vec::with_capacity(flat_insts.len());

    for inst in &flat_insts {
        let (base_op, specialized, rd, rs1, rs2, orig_rd): (String, String, Option<u8>, Option<u8>, Option<u8>, Option<u8>) = match inst.op.as_str() {
            "add" | "sub" | "sll" | "srl" | "sra" | "slt" | "sltu" |
            "xor" | "or" | "and" | "mul" | "mulh" | "mulhsu" | "mulhu" |
            "div" | "divu" | "rem" | "remu" =>
                (inst.op.clone(), inst.op.clone(), Some(0), Some(1), Some(2), None),
            "addi" | "slti" | "sltiu" | "xori" | "ori" | "andi" |
            "slli" | "srli" | "srai" =>
                (inst.op.clone(), inst.op.clone(), Some(0), Some(1), None, None),
            "sll1" | "sll4" | "sll8" | "sll16" | "sll31" |
            "srl1" | "srl4" | "srl8" | "srl16" | "srl31" |
            "sra1" | "sra4" | "sra8" | "sra16" | "sra31" =>
                (inst.op.clone(), inst.op.clone(), Some(0), Some(1), None, None),
            "lw" =>
                (inst.op.clone(), inst.op.clone(), Some(0), Some(1), None, None),
            "sext8" =>
                ("sext8".into(), "sext8".into(), Some(0), None, None, None),
            "lw_aligned" =>
                ("lw_aligned".into(), "lw_aligned".into(), Some(0), Some(1), None, None),
            "byte_sel_r2" =>
                ("byte_sel_r2".into(), "byte_sel_r2".into(), Some(0), None, Some(2), None),
            "byte_sel0" | "byte_sel1" | "byte_sel2" | "byte_sel3" =>
                (inst.op.clone(), inst.op.clone(), Some(0), None, None, None),
            "sw" =>
                (inst.op.clone(), inst.op.clone(), None, Some(1), Some(0), None),
            "sw_aligned" =>
                ("sw_aligned".into(), "sw_aligned".into(), None, Some(1), Some(0), None),
            "byte_ins_r2" =>
                ("byte_ins_r2".into(), "byte_ins_r2".into(), Some(0), None, Some(2), None),
            "sw_waligned" =>
                ("sw_waligned".into(), "sw_waligned".into(), None, Some(1), Some(0), None),
            "beq" | "bne" | "blt" | "bge" | "bltu" | "bgeu" =>
                (inst.op.clone(), inst.op.clone(), None, Some(0), Some(1), None),
            "lui" =>
                ("lui".into(), "lui".into(), Some(0), None, None, None),
            "jal" =>
                ("jal".into(), "jal".into(), None, None, None, None),
            "jal_call" =>
                ("jal_call".into(), "jal_call".into(), None, None, None, Some(1)),
            "jalr" =>
                ("jalr".into(), "jalr".into(), None, Some(0), None, None),
            "jalr_call" =>
                ("jalr_call".into(), "jalr_call".into(), Some(0), Some(0), None, Some(1)),
            "ret" =>
                ("ret".into(), "ret".into(), None, Some(0), None, None),
            "lw_abs0" =>
                ("lw_abs0".into(), "lw_abs0".into(), Some(0), None, None, None),
            "lw_abs1" =>
                ("lw_abs1".into(), "lw_abs1".into(), Some(1), None, None, None),
            "lw_abs2" =>
                ("lw_abs2".into(), "lw_abs2".into(), Some(2), None, None, None),
            "sw_abs0" =>
                ("sw_abs0".into(), "sw_abs0".into(), None, Some(0), None, None),
            "sw_abs1" =>
                ("sw_abs1".into(), "sw_abs1".into(), None, Some(1), None, None),
            "sw_abs2" =>
                ("sw_abs2".into(), "sw_abs2".into(), None, Some(2), None, None),
            "jr_table_idx" =>
                ("jr_table_idx".into(), "jr_table_idx".into(), None, Some(0), None, None),
            "jr_computed" =>
                ("jr_computed".into(), "jr_computed".into(), None, None, None, None),
            "ecall" =>
                ("ecall".into(), "ecall".into(), None, None, None, None),
            "halt" =>
                ("halt".into(), "halt".into(), None, None, None, None),
            other => bail!("Unknown op: {}", other),
        };

        if !opcode_map.contains_key(&specialized) {
            assert!(opcode_table.len() < 256, "canon3 ISA exceeds 255 opcodes");
            let id = opcode_table.len() as u8;
            opcode_map.insert(specialized.clone(), id);
            opcode_table.push(OpcodeInfo {
                name: specialized.clone(),
                base_op: base_op.clone(),
                rd, rs1, rs2, orig_rd,
                orig_rs1: None, orig_rs2: None,
            });
        }

        code_table.push(opcode_map[&specialized]);

        let mut imm_val = inst.imm.unwrap_or(0);
        if inst.op == "lui" {
            imm_val = ((imm_val as u32) >> 12) as i32;
        }
        imm_table.push(imm_val);
    }

    println!("  ISA size: {} unique opcodes", opcode_table.len());
    // println!("  Entry PC: {} (0x{:x})", entry_pc, entry_pc);

    // Step 8: ELF segments + patch
    let mut segments = extract_elf_segments(&data)?;

    for (&_jr_addr, &(table_base, num_entries)) in &jump_table_bases {
        for entry_i in 0..num_entries {
            let entry_addr = table_base.wrapping_add((entry_i * 4) as u32);
            let code_addr = read_seg_u32(&segments, entry_addr)
                .unwrap_or_else(|| panic!("Can't read jump table at 0x{:x}", entry_addr));
            if let Some(global_idx) = resolve_to_global(code_addr) {
                let new_idx = final_old_to_new[global_idx];
                write_seg_u32(&mut segments, entry_addr, new_idx as u32);
            }
        }
    }

    let mut func_ptr_patches = 0;
    for seg in &mut segments {
        let mut off = 0;
        while off + 4 <= seg.data.len() {
            let val = u32::from_le_bytes([seg.data[off], seg.data[off+1], seg.data[off+2], seg.data[off+3]]);
            if let Some(&flat_idx) = func_entry_to_flat_idx.get(&val) {
                let new_idx = final_old_to_new[flat_idx];
                seg.data[off..off+4].copy_from_slice(&(new_idx as u32).to_le_bytes());
                func_ptr_patches += 1;
            }
            off += 4;
        }
    }
    if func_ptr_patches > 0 {
        eprintln!("  Patched {} function pointer entries", func_ptr_patches);
    }

    // Serialize
    let serial_opcode_table: Vec<SerializedOpcodeInfo> = opcode_table.iter().map(|o| {
        SerializedOpcodeInfo {
            name: o.name.clone(), base_op: o.base_op.clone(),
            rd: o.rd, rs1: o.rs1, rs2: o.rs2, orig_rd: o.orig_rd,
        }
    }).collect();

    let program = FlatProgram {
        num_regs: 3,
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

    // Collect .eh_frame address range from section headers so we can exclude it.
    let sections = elf.sections(endian, data)?;
    let mut eh_frame_range: Option<(u32, u32)> = None;
    for section in sections.iter() {
        if let Ok(name) = sections.section_name(endian, section) {
            if name == b".eh_frame" {
                let addr = section.sh_addr.get(endian);
                let size = section.sh_size.get(endian);
                eh_frame_range = Some((addr, addr + size));
            }
        }
    }

    let segments_hdr = elf.program_headers(endian, data)?;
    let mut segments = Vec::new();
    for seg in segments_hdr {
        if seg.p_type.get(endian) != PT_LOAD { continue; }
        let flags = seg.p_flags.get(endian);
        // Skip executable segments (.text) — we execute bytecode, not native code.
        if flags & PF_X != 0 { continue; }
        let vaddr = seg.p_vaddr.get(endian);
        let filesz = seg.p_filesz.get(endian) as usize;
        let offset = seg.p_offset.get(endian) as usize;
        if filesz == 0 || offset + filesz > data.len() { continue; }
        // Truncate to exclude .eh_frame if it sits at the end of this segment.
        let mut end = vaddr + filesz as u32;
        if let Some((eh_start, _)) = eh_frame_range {
            if eh_start >= vaddr && eh_start < end {
                end = eh_start;
            }
        }
        let keep = (end - vaddr) as usize;
        if keep > 0 {
            segments.push(MemSegment { vaddr, data: data[offset..offset+keep].to_vec() });
        }
    }
    Ok(segments)
}

fn read_seg_u32(segments: &[MemSegment], addr: u32) -> Option<u32> {
    for seg in segments {
        let end = seg.vaddr + seg.data.len() as u32;
        if addr >= seg.vaddr && addr + 4 <= end {
            let off = (addr - seg.vaddr) as usize;
            return Some(u32::from_le_bytes([seg.data[off], seg.data[off+1], seg.data[off+2], seg.data[off+3]]));
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
