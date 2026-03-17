//! LEGACY — workaround for ELFs built without `-enable-machine-outliner=never`.
//!
//! When the LLVM Machine Outliner is active it extracts repeated
//! prologue/epilogue sequences into tiny shared helper functions called via
//! t0/t1 (not ra).  That non-standard ABI breaks per-function register
//! allocation.  `inline_outlined_functions` re-merges those helpers back into
//! their callers so regalloc2 sees a self-contained function body.
//!
//! **Preferred solution**: build with `-enable-machine-outliner=never` (set in
//! `.cargo/config.toml`) and never call this module.

use std::collections::{HashMap, HashSet};
use super::decode::DecodedInst;

/// Inline OUTLINED_FUNCTION calls back into their callers.
///
/// Returns `(modified_decoded, modified_elf_funcs)`.
/// Caller should follow with `classify_jalr_x0` then `build_cfg`.
pub fn inline_outlined_functions(
    decoded: &[DecodedInst],
    elf_funcs: &[(u32, u32, String)],
) -> (Vec<DecodedInst>, Vec<(u32, u32)>) {
    // Step 1: Identify outlined function addresses and their instruction ranges
    let mut outlined_addrs: HashSet<u32> = HashSet::new();
    let mut outlined_insts: HashMap<u32, Vec<DecodedInst>> = HashMap::new();

    let mut addr_to_idx: HashMap<u32, usize> = HashMap::new();
    for (i, inst) in decoded.iter().enumerate() {
        addr_to_idx.insert(inst.addr, i);
    }

    for (addr, size, name) in elf_funcs {
        if name.contains("OUTLINED_FUNCTION") {
            outlined_addrs.insert(*addr);
            let end_addr = addr + size;
            let mut insts = Vec::new();
            if let Some(&start_idx) = addr_to_idx.get(addr) {
                for i in start_idx.. {
                    if i >= decoded.len() || decoded[i].addr >= end_addr { break; }
                    insts.push(decoded[i].clone());
                }
            }
            outlined_insts.insert(*addr, insts);
        }
    }

    if outlined_addrs.is_empty() {
        let result = decoded.to_vec();
        let funcs = elf_funcs.iter().map(|(a, s, _)| (*a, *s)).collect();
        return (result, funcs);
    }

    // Step 2: Scan decoded instructions, inline outlined calls
    let mut result: Vec<DecodedInst> = Vec::new();
    let mut i = 0;
    while i < decoded.len() {
        let inst = &decoded[i];

        // Check for auipc + jalr/jr pattern targeting an outlined function
        if inst.op == "auipc" && i + 1 < decoded.len() {
            let next = &decoded[i + 1];
            if next.op == "jalr" && inst.rd == next.rs1 {
                let auipc_addr = inst.addr;
                let auipc_imm = inst.imm.unwrap_or(0) as u32;
                let jalr_imm = next.imm.unwrap_or(0) as u32;
                let target = auipc_addr.wrapping_add(auipc_imm).wrapping_add(jalr_imm) & !1;

                if outlined_addrs.contains(&target) {
                    let orig_rd = next.rd.unwrap_or(0);
                    let base_addr = inst.addr;

                    if let Some(body) = outlined_insts.get(&target) {
                        let mut offset = 0u32;
                        for oinst in body {
                            let is_final_return = oinst.op == "jalr" && oinst.rd == Some(0);
                            if is_final_return {
                                if orig_rd == 0 {
                                    // Tail-call: keep the ret
                                    let mut remapped = oinst.clone();
                                    remapped.addr = base_addr.wrapping_add(offset);
                                    result.push(remapped);
                                }
                                // Call style: skip the jr t0
                                continue;
                            }
                            let mut remapped = oinst.clone();
                            remapped.addr = base_addr.wrapping_add(offset);
                            offset += 1;
                            result.push(remapped);
                        }
                        i += 2; // skip auipc + jalr
                        continue;
                    }
                }
            }
        }

        // Skip instructions that belong to outlined functions (now inlined above)
        if outlined_addrs.iter().any(|&oaddr| {
            if let Some(body) = outlined_insts.get(&oaddr) {
                if let Some(last) = body.last() {
                    return inst.addr >= oaddr && inst.addr <= last.addr;
                }
            }
            false
        }) {
            i += 1;
            continue;
        }

        result.push(inst.clone());
        i += 1;
    }

    // Step 3: Rebuild elf_funcs without the outlined stubs
    let new_funcs = elf_funcs.iter()
        .filter(|(_, _, name)| !name.contains("OUTLINED_FUNCTION"))
        .map(|(a, s, _)| (*a, *s))
        .collect();

    let total_outlined_insts: usize = outlined_insts.values().map(|v| v.len()).sum();
    eprintln!("  Inlined {} outlined functions ({} outlined insts, {} → {} decoded insts)",
        outlined_addrs.len(), total_outlined_insts, decoded.len(), result.len());

    (result, new_funcs)
}
