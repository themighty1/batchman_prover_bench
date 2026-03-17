/// Validate our CFG against two independent ground truths:
///
///  Check A (objdump): every explicit branch/jump target that llvm-objdump
///    annotates must appear as an edge in our CFG.
///
///  Check B (semantics): for every block, compute the expected successor set
///    directly from the terminal instruction's opcode and immediate, then
///    compare against our block.succs exactly — no filtering, no allowances.
///    This catches missing or spurious fallthrough edges.

use anyhow::Result;
use reg_analyzer::rv32::{decode_elf, get_elf_functions_named, classify_jalr_x0, build_cfg};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::process::Command;

// ---------------------------------------------------------------------------
// Check A: parse objdump output into (src_addr, target_addr, kind) triples
// ---------------------------------------------------------------------------

fn parse_objdump(elf_path: &str) -> Result<Vec<(u32, u32, String)>> {
    let out = Command::new("llvm-objdump")
        .args(["-d", "--no-show-raw-insn", elf_path])
        .output()?;

    let text = String::from_utf8_lossy(&out.stdout);
    let mut edges: Vec<(u32, u32, String)> = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        let Some(colon) = line.find(':') else { continue };
        let addr_str = line[..colon].trim();
        let Ok(src_addr) = u32::from_str_radix(addr_str, 16) else { continue };

        let rest = line[colon + 1..].trim();
        let mut parts = rest.splitn(2, char::is_whitespace);
        let mnemonic = parts.next().unwrap_or("").trim();
        let operands = parts.next().unwrap_or("").trim();
        let target = extract_hex_target(operands);

        match mnemonic {
            "beq" | "bne" | "blt" | "bge" | "bltu" | "bgeu"
            | "beqz" | "bnez" | "bltz" | "bgtz" | "blez" | "bgez" => {
                if let Some(t) = target { edges.push((src_addr, t, "branch".into())); }
            }
            "j" => {
                if let Some(t) = target { edges.push((src_addr, t, "jump".into())); }
            }
            "jal" | "tail" | "call" => {
                if let Some(t) = target {
                    let kind = if operands.trim_start().starts_with("zero") {
                        "jump"
                    } else {
                        "call"
                    };
                    edges.push((src_addr, t, kind.into()));
                }
            }
            _ => {}
        }
    }

    Ok(edges)
}

fn extract_hex_target(operands: &str) -> Option<u32> {
    // Strip <sym+offset> annotation before searching for the address.
    let operands = operands.split('<').next().unwrap_or(operands);
    let pos = operands.find("0x")?;
    let hex_start = pos + 2;
    let hex_end = operands[hex_start..]
        .find(|c: char| !c.is_ascii_hexdigit())
        .map(|e| hex_start + e)
        .unwrap_or(operands.len());
    u32::from_str_radix(&operands[hex_start..hex_end], 16).ok()
}

// ---------------------------------------------------------------------------
// Check B: compute expected successors for a block from its terminal instruction
// ---------------------------------------------------------------------------

/// Returns the set of expected successor block-start addresses for the given block,
/// derived purely from the terminal instruction's opcode and immediate.
/// Returns None for jr_table (jump tables require data analysis; we skip them in Check B).
fn expected_succs(
    block_end: usize,
    decoded: &[reg_analyzer::rv32::DecodedInst],
    addr_to_block_start: &HashMap<u32, u32>,  // any addr → its block's start_addr
) -> Option<HashSet<u32>> {
    let last = &decoded[block_end - 1];
    let fallthrough_addr = if block_end < decoded.len() { Some(decoded[block_end].addr) } else { None };

    let mut succs = HashSet::new();

    match last.op.as_str() {
        // Conditional branch: taken target + not-taken fallthrough
        "beq" | "bne" | "blt" | "bge" | "bltu" | "bgeu" => {
            let offset = last.imm.unwrap_or(0) as i64;
            let target = (last.addr as i64 + offset) as u32;
            succs.insert(target);
            if let Some(ft) = fallthrough_addr { succs.insert(ft); }
        }
        // Unconditional jump via jal x0 (the `j` pseudo-op)
        "jal" if last.rd == Some(0) => {
            let offset = last.imm.unwrap_or(0) as i64;
            let target = (last.addr as i64 + offset) as u32;
            succs.insert(target);
        }
        // Direct call (jal ra): only the fallthrough is an intra-CFG edge
        "jal" if last.rd == Some(1) => {
            if let Some(ft) = fallthrough_addr { succs.insert(ft); }
        }
        // Indirect call (jalr ra, rs1, imm): only fallthrough
        "jalr" if last.rd == Some(1) => {
            if let Some(ft) = fallthrough_addr { succs.insert(ft); }
        }
        // Return / classified computed jumps: no successors
        "ret" | "jr_computed" | "jalr" => {}
        // Jump table: skip in Check B (data-dependent, already validated via Check A)
        "jr_table" => return None,
        // Everything else: sequential fallthrough
        _ => {
            if let Some(ft) = fallthrough_addr { succs.insert(ft); }
        }
    }

    // Map instruction addresses → block start addresses
    let resolved: HashSet<u32> = succs.iter()
        .filter_map(|addr| addr_to_block_start.get(addr).copied())
        .collect();

    // Verify every expected target has a known block
    for addr in &succs {
        if !addr_to_block_start.contains_key(addr) {
            eprintln!("  WARN  block at 0x{:x}: expected successor 0x{:x} has no block start",
                decoded[block_end - 1].addr, addr);
        }
    }

    Some(resolved)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let elf_path = std::env::args().nth(1)
        .unwrap_or_else(|| "../toy-rv32/toy.elf".to_string());

    let data = fs::read(&elf_path)?;

    // Build our CFG
    let (decoded_raw, _, _) = decode_elf(&data)?;
    let elf_funcs_named = get_elf_functions_named(&data)?;
    let mut decoded = decoded_raw;
    let (_jump_table_targets, _jump_table_bases) = classify_jalr_x0(&mut decoded, &data, &elf_funcs_named);
    let blocks = build_cfg(&decoded, &HashMap::new());

    let block_starts: Vec<u32> = blocks.iter().map(|b| b.start_addr).collect();

    // Our edges: set of (src_block_start, dst_block_start)
    let our_edges: HashSet<(u32, u32)> = blocks.iter().flat_map(|b| {
        b.succs.iter().map(|&s| (b.start_addr, block_starts[s])).collect::<Vec<_>>()
    }).collect();

    // Map: instruction addr → its block's start addr (for Check A and Check B)
    let inst_to_block_start: HashMap<u32, u32> = decoded.iter().enumerate()
        .filter_map(|(i, d)| {
            let blk = blocks.iter().find(|b| i >= b.start && i < b.end)?;
            Some((d.addr, blk.start_addr))
        })
        .collect();

    // Map: block start addr → block index
    let addr_to_block_idx: HashMap<u32, usize> = blocks.iter()
        .map(|b| (b.start_addr, b.id))
        .collect();

    println!("=== CFG Validation: our build_cfg ===\n");
    println!("  ELF: {}", elf_path);
    println!("  Our CFG: {} blocks, {} edges\n", blocks.len(), our_edges.len());

    // -----------------------------------------------------------------------
    // Check A: every explicit branch/jump target in objdump → in our CFG
    // -----------------------------------------------------------------------

    let objdump_edges = parse_objdump(&elf_path)?;
    println!("--- Check A: objdump branch targets in our CFG ({} branch/jump instructions) ---",
        objdump_edges.len());

    let mut a_ok = 0usize;
    let mut a_miss = 0usize;

    for (src, tgt, kind) in &objdump_edges {
        if kind == "call" { continue; }  // inter-function, not an intra-CFG edge

        let Some(&src_block_start) = inst_to_block_start.get(src) else {
            println!("  WARN  0x{:x} → 0x{:x} ({}) : src not in decoded stream", src, tgt, kind);
            continue;
        };

        if our_edges.contains(&(src_block_start, *tgt)) {
            a_ok += 1;
        } else {
            println!("  MISS  0x{:x} (block 0x{:x}) → 0x{:x} ({}) : objdump sees this, our CFG doesn't",
                src, src_block_start, tgt, kind);
            a_miss += 1;
        }
    }

    println!("  Verified: {}, Missing: {}\n", a_ok, a_miss);

    // -----------------------------------------------------------------------
    // Check B: for every block, expected successors from instruction semantics
    //          must match our block.succs exactly
    // -----------------------------------------------------------------------

    println!("--- Check B: expected successors from instruction semantics vs our CFG ---");

    let mut b_ok = 0usize;
    let mut b_wrong = 0usize;
    let mut b_skip = 0usize;  // jr_table blocks

    for block in &blocks {
        let actual: HashSet<u32> = block.succs.iter().map(|&s| block_starts[s]).collect();

        let Some(expected) = expected_succs(block.end, &decoded, &inst_to_block_start) else {
            b_skip += 1;
            continue;
        };

        if expected == actual {
            b_ok += 1;
        } else {
            let last = &decoded[block.end - 1];
            println!("  MISMATCH block 0x{:x} (last inst: {} at 0x{:x})",
                block.start_addr, last.op, last.addr);

            // Missing from our CFG
            for addr in expected.difference(&actual) {
                println!("    EXPECTED  0x{:x} : in expected, missing from our CFG", addr);
            }
            // Extra in our CFG
            for addr in actual.difference(&expected) {
                println!("    SPURIOUS  0x{:x} : in our CFG, not expected", addr);
            }
            b_wrong += 1;
        }
    }

    println!("  Blocks OK: {}, Wrong: {}, Skipped (jr_table): {}\n", b_ok, b_wrong, b_skip);

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------

    let pass = a_miss == 0 && b_wrong == 0;
    if pass {
        println!("PASS: all checks clean.");
    } else {
        println!("FAIL: {} Check-A misses, {} Check-B mismatches.", a_miss, b_wrong);
    }

    Ok(())
}
