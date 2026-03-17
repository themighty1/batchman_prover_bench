//! Compile an RV32 ELF into a canonical 4-register ISA binary.
//!
//! Canonical forms:
//!   R-type:  r0 = r1 op r2
//!   I-type:  r0 = r1 op imm
//!   Load:    r0 = mem[r1 + imm]
//!   Store:   mem[r1 + imm] = r0
//!   Branch:  branch r0, r1, target
//!
//! 4-slot cache, per-slot load/store (no swap needed).
//!
//! Usage:
//!   rv32_compile_canon4 [elf_path] [output_path]

use anyhow::{Result, bail};
use reg_analyzer::rv32::decode::{DecodedInst, decode_elf, get_elf_functions_named};
use reg_analyzer::rv32::cfg::{build_cfg, classify_jalr_x0};
use reg_analyzer::rv32_isa_vm::MemSegment;
use reg_analyzer::rv32_flat_vm::*;
use std::collections::HashMap;
use std::fs;

const NUM_SLOTS: usize = 4;

#[derive(Clone, Debug)]
struct CacheState {
    slots: [Option<u8>; NUM_SLOTS],
    dirty: [bool; NUM_SLOTS],
}

impl CacheState {
    fn new() -> Self {
        CacheState { slots: [None; NUM_SLOTS], dirty: [false; NUM_SLOTS] }
    }
}

struct CanonInst {
    op: String,
    imm: Option<i32>,
    addr: u32,
}

fn emit_store_slot(out: &mut Vec<CanonInst>, slot: usize, vreg: u8) {
    let op = match slot {
        0 => "store_reg0", 1 => "store_reg1", 2 => "store_reg2", 3 => "store_reg3",
        _ => panic!("bad slot"),
    };
    out.push(CanonInst { op: op.into(), imm: Some(vreg as i32), addr: 0 });
}

fn emit_load_slot(out: &mut Vec<CanonInst>, slot: usize, vreg: u8) {
    let op = match slot {
        0 => "load_reg0", 1 => "load_reg1", 2 => "load_reg2", 3 => "load_reg3",
        _ => panic!("bad slot"),
    };
    out.push(CanonInst { op: op.into(), imm: Some(vreg as i32), addr: 0 });
}

fn emit_mov(out: &mut Vec<CanonInst>, dst: usize, src: usize) {
    let op = format!("mov{}{}", dst, src);
    out.push(CanonInst { op, imm: None, addr: 0 });
}

fn flush_cache(cache: &mut CacheState, out: &mut Vec<CanonInst>) {
    for i in 0..NUM_SLOTS {
        if cache.dirty[i] {
            if let Some(vreg) = cache.slots[i] {
                emit_store_slot(out, i, vreg);
            }
            cache.dirty[i] = false;
        }
    }
}

fn invalidate_cache(cache: &mut CacheState) {
    cache.slots = [None; NUM_SLOTS];
    cache.dirty = [false; NUM_SLOTS];
}

/// After writing rd to a slot, invalidate other slots that claim the same vreg.
fn invalidate_stale(cache: &mut CacheState, rd: u8, written_slot: usize) {
    for i in 0..NUM_SLOTS {
        if i == written_slot { continue; }
        if cache.slots[i] == Some(rd) {
            cache.slots[i] = None;
            cache.dirty[i] = false;
        }
    }
}

/// Ensure vreg is in the specified slot.
/// Uses mov when vreg is already in another slot (avoids store+load for dirty values).
fn ensure_in_slot(cache: &mut CacheState, out: &mut Vec<CanonInst>, vreg: u8, target_slot: usize) {
    if cache.slots[target_slot] == Some(vreg) {
        return;
    }

    // Check if vreg is in another slot
    let mut src_slot = None;
    for other in 0..NUM_SLOTS {
        if other == target_slot { continue; }
        if cache.slots[other] == Some(vreg) {
            src_slot = Some(other);
            break;
        }
    }

    if let Some(src) = src_slot {
        // Vreg is in another slot — use mov to transfer it.
        // First evict target slot if dirty.
        if cache.dirty[target_slot] {
            if let Some(old) = cache.slots[target_slot] {
                emit_store_slot(out, target_slot, old);
            }
            cache.dirty[target_slot] = false;
            cache.slots[target_slot] = None;
        }
        // Move vreg from src to target (non-destructive copy).
        emit_mov(out, target_slot, src);
        let was_dirty = cache.dirty[src];
        cache.slots[target_slot] = Some(vreg);
        cache.dirty[target_slot] = was_dirty;
        // Source becomes clean copy (target is the authoritative dirty one).
        cache.dirty[src] = false;
    } else {
        // Vreg not in any slot — evict target and load from regfile.
        if cache.dirty[target_slot] {
            if let Some(old) = cache.slots[target_slot] {
                emit_store_slot(out, target_slot, old);
            }
            cache.dirty[target_slot] = false;
        }
        emit_load_slot(out, target_slot, vreg);
        cache.slots[target_slot] = Some(vreg);
        cache.dirty[target_slot] = false;
    }
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
        .unwrap_or_else(|| "canonical4.bin".to_string());

    println!("=== rv32_compile_canon4 (4 regs) ===");
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
    println!("  Functions with code: {}", func_insts.len());

    // Step 4: Canonicalize each function (4-slot cache)
    struct FuncResult {
        entry_addr: u32,
        canon_insts: Vec<CanonInst>,
        addr_to_local: HashMap<u32, usize>,
    }

    let mut func_results: Vec<FuncResult> = Vec::new();
    let mut total_loads = 0u64;
    let mut total_stores = 0u64;
    let mut total_movs = 0u64;
    let mut total_canon = 0u64;

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

                // Loads: r0 = mem[r1 + imm]
                "lw" | "lb" | "lh" | "lbu" | "lhu" => {
                    let base = inst.rs1.unwrap_or(0);
                    let rd = inst.rd.unwrap_or(0);
                    ensure_in_slot(&mut cache, &mut out, base, 1);
                    save_r0_if_needed(&mut cache, &mut out, rd);
                    out.push(CanonInst { op: inst.op.clone(), imm, addr: inst.addr });
                    cache.slots[0] = Some(rd);
                    cache.dirty[0] = true;
                    invalidate_stale(&mut cache, rd, 0);
                }

                // Stores: mem[r1 + imm] = r0
                "sw" | "sb" | "sh" => {
                    let val = inst.rs2.unwrap_or(0);
                    let base = inst.rs1.unwrap_or(0);
                    ensure_in_slot(&mut cache, &mut out, val, 0);
                    ensure_in_slot(&mut cache, &mut out, base, 1);
                    out.push(CanonInst { op: inst.op.clone(), imm, addr: inst.addr });
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
            if ci.op.starts_with("load_reg") { total_loads += 1; }
            else if ci.op.starts_with("store_reg") { total_stores += 1; }
            else if ci.op.starts_with("mov") { total_movs += 1; }
            else { total_canon += 1; }
        }

        func_results.push(FuncResult { entry_addr: func_entry, canon_insts: out, addr_to_local });
    }

    println!("  Canonical instructions: {} total", func_results.iter().map(|f| f.canon_insts.len()).sum::<usize>());
    println!("    load_reg:  {}", total_loads);
    println!("    store_reg: {}", total_stores);
    println!("    mov:       {}", total_movs);
    println!("    canonical: {}", total_canon);

    // Step 5: Flatten
    let mut flat_insts: Vec<CanonInst> = Vec::new();
    let mut func_entry_to_flat_idx: HashMap<u32, usize> = HashMap::new();

    struct FuncMapping {
        global_base: usize,
        addr_to_local: HashMap<u32, usize>,
    }
    let mut func_mappings: Vec<(u32, FuncMapping)> = Vec::new();

    for fr in &func_results {
        let global_base = flat_insts.len();
        func_entry_to_flat_idx.insert(fr.entry_addr, global_base);
        func_mappings.push((fr.entry_addr, FuncMapping {
            global_base,
            addr_to_local: fr.addr_to_local.clone(),
        }));
        for ci in &fr.canon_insts {
            flat_insts.push(CanonInst { op: ci.op.clone(), imm: ci.imm, addr: ci.addr });
        }
    }

    println!("  Flat instructions: {}", flat_insts.len());

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
    let entry_pc = *func_entry_to_flat_idx.get(&elf_entry_addr)
        .ok_or_else(|| anyhow::anyhow!("entry 0x{:x} not in func table", elf_entry_addr))? as u32;

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
            "lw" | "lb" | "lh" | "lbu" | "lhu" =>
                (inst.op.clone(), inst.op.clone(), Some(0), Some(1), None, None),
            "sw" | "sb" | "sh" =>
                (inst.op.clone(), inst.op.clone(), None, Some(1), Some(0), None),
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
            "load_reg0" =>
                ("load_reg0".into(), "load_reg0".into(), Some(0), None, None, None),
            "load_reg1" =>
                ("load_reg1".into(), "load_reg1".into(), Some(1), None, None, None),
            "load_reg2" =>
                ("load_reg2".into(), "load_reg2".into(), Some(2), None, None, None),
            "load_reg3" =>
                ("load_reg3".into(), "load_reg3".into(), Some(3), None, None, None),
            "store_reg0" =>
                ("store_reg0".into(), "store_reg0".into(), None, Some(0), None, None),
            "store_reg1" =>
                ("store_reg1".into(), "store_reg1".into(), None, Some(1), None, None),
            "store_reg2" =>
                ("store_reg2".into(), "store_reg2".into(), None, Some(2), None, None),
            "store_reg3" =>
                ("store_reg3".into(), "store_reg3".into(), None, Some(3), None, None),
            "mov01" => ("mov01".into(), "mov01".into(), Some(0), Some(1), None, None),
            "mov02" => ("mov02".into(), "mov02".into(), Some(0), Some(2), None, None),
            "mov03" => ("mov03".into(), "mov03".into(), Some(0), Some(3), None, None),
            "mov10" => ("mov10".into(), "mov10".into(), Some(1), Some(0), None, None),
            "mov12" => ("mov12".into(), "mov12".into(), Some(1), Some(2), None, None),
            "mov13" => ("mov13".into(), "mov13".into(), Some(1), Some(3), None, None),
            "mov20" => ("mov20".into(), "mov20".into(), Some(2), Some(0), None, None),
            "mov21" => ("mov21".into(), "mov21".into(), Some(2), Some(1), None, None),
            "mov23" => ("mov23".into(), "mov23".into(), Some(2), Some(3), None, None),
            "mov30" => ("mov30".into(), "mov30".into(), Some(3), Some(0), None, None),
            "mov31" => ("mov31".into(), "mov31".into(), Some(3), Some(1), None, None),
            "mov32" => ("mov32".into(), "mov32".into(), Some(3), Some(2), None, None),
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
            assert!(opcode_table.len() < 256, "canon4 ISA exceeds 255 opcodes");
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
    println!("  Entry PC: {} (0x{:x})", entry_pc, entry_pc);

    // Step 8: ELF segments + patch
    let mut segments = extract_elf_segments(&data)?;

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

    // Serialize
    let serial_opcode_table: Vec<SerializedOpcodeInfo> = opcode_table.iter().map(|o| {
        SerializedOpcodeInfo {
            name: o.name.clone(), base_op: o.base_op.clone(),
            rd: o.rd, rs1: o.rs1, rs2: o.rs2, orig_rd: o.orig_rd,
        }
    }).collect();

    let program = FlatProgram {
        num_regs: 4,
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
            segments.push(MemSegment { vaddr, data: data[offset..offset+filesz].to_vec() });
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
