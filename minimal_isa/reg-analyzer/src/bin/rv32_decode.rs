use anyhow::Result;
use reg_analyzer::rv32::*;
use std::collections::HashMap;
use std::fs;

fn main() -> Result<()> {
    let elf_path = std::env::args().nth(1)
        .unwrap_or_else(|| "../rv32-build/target/riscv32i-unknown-none-elf/release/json_query".to_string());

    let data = fs::read(&elf_path)?;
    let (decoded, text_addr, text_len) = decode_elf(&data)?;

    println!("ELF: {}", elf_path);
    println!(".text: {} bytes at 0x{:x} ({} instructions)\n",
        text_len, text_addr, decoded.len());

    // ---------------------------------------------------------------
    // Step 1: Decode summary
    // ---------------------------------------------------------------
    let mut op_counts: HashMap<String, usize> = HashMap::new();
    for d in &decoded {
        *op_counts.entry(d.op.clone()).or_default() += 1;
    }
    println!("=== Step 1: Decoded ({} base opcodes) ===\n", op_counts.len());

    // Specialized opcodes with physical registers baked in
    let mut phys_specialized: HashMap<String, usize> = HashMap::new();
    for d in &decoded {
        let mut name = d.op.clone();
        if let Some(rd) = d.rd { name = format!("{}.{}", name, REG_NAMES[rd as usize]); }
        if let Some(rs1) = d.rs1 { name = format!("{}.{}", name, REG_NAMES[rs1 as usize]); }
        if let Some(rs2) = d.rs2 { name = format!("{}.{}", name, REG_NAMES[rs2 as usize]); }
        *phys_specialized.entry(name).or_default() += 1;
    }
    println!("  Physical-reg specialized opcodes: {}", phys_specialized.len());

    // ---------------------------------------------------------------
    // Step 2: Build CFG + lift to virtual registers
    // ---------------------------------------------------------------
    let blocks = build_cfg(&decoded, &std::collections::HashMap::new());
    let (block_infos, total_vregs) = lift_to_vregs(&decoded, &blocks);

    println!("\n=== Step 2: CFG + VReg lifting ===\n");
    println!("  Basic blocks:   {}", blocks.len());
    println!("  Total vregs:    {}", total_vregs);

    // Block size distribution
    let mut sizes: Vec<usize> = blocks.iter().map(|b| b.end - b.start).collect();
    sizes.sort();
    let total_insts: usize = sizes.iter().sum();
    println!("  Total instructions: {}", total_insts);
    println!("  Block sizes: min={}, median={}, max={}, avg={:.1}",
        sizes.first().unwrap_or(&0),
        sizes.get(sizes.len() / 2).unwrap_or(&0),
        sizes.last().unwrap_or(&0),
        total_insts as f64 / blocks.len().max(1) as f64,
    );

    // Edge stats
    let total_succs: usize = blocks.iter().map(|b| b.succs.len()).sum();
    let no_succ = blocks.iter().filter(|b| b.succs.is_empty()).count();
    let multi_pred = blocks.iter().filter(|b| b.preds.len() > 1).count();
    println!("  Edges: {} total, {} blocks with no successor (returns/exits)",
        total_succs, no_succ);
    println!("  Merge points (>1 predecessor): {}", multi_pred);

    // ---------------------------------------------------------------
    // Show first few blocks as sample
    // ---------------------------------------------------------------
    println!("\n=== Sample blocks (first 5) ===\n");
    for info in block_infos.iter().take(5) {
        let block = &blocks[info.block_id];
        println!("--- Block {} (addr 0x{:x}, {} insts, succs={:?}, preds={:?}) ---",
            block.id, block.start_addr, info.insts.len(), block.succs, block.preds);

        // Show entry vreg mapping (non-zero only, skip x0)
        print!("  entry: ");
        for r in 1..32 {
            if r == 1 { print!("ra=v{}", info.entry_vregs[r]); }
            else if r == 2 { print!(" sp=v{}", info.entry_vregs[r]); }
            else if r <= 5 { continue; } // skip for brevity
            else { break; }
        }
        println!(" ...");

        for inst in &info.insts {
            let rd_s = inst.rd.map(|v| format!("v{}", v)).unwrap_or_else(|| "-".into());
            let rs1_s = inst.rs1.map(|v| format!("v{}", v)).unwrap_or_else(|| "-".into());
            let rs2_s = inst.rs2.map(|v| format!("v{}", v)).unwrap_or_else(|| "-".into());
            let imm_s = inst.imm.map(|i| format!("{}", i)).unwrap_or_else(|| "-".into());
            let orig = format!("{}{}{}",
                inst.orig_rd_preg.map(|r| format!(" {}", REG_NAMES[r as usize])).unwrap_or_default(),
                inst.orig_rs1_preg.map(|r| format!(" {}", REG_NAMES[r as usize])).unwrap_or_default(),
                inst.orig_rs2_preg.map(|r| format!(" {}", REG_NAMES[r as usize])).unwrap_or_default(),
            );
            println!("  0x{:06x}  {:6} rd={:5} rs1={:5} rs2={:5} imm={:8}  (was:{})",
                inst.addr, inst.op, rd_s, rs1_s, rs2_s, imm_s, orig);
        }
        println!();
    }

    // ---------------------------------------------------------------
    // VReg lifetime stats
    // ---------------------------------------------------------------
    // Count how many vregs are actually used (appear as source)
    let mut vreg_def_count = HashMap::<u32, usize>::new();
    let mut vreg_use_count = HashMap::<u32, usize>::new();
    for info in &block_infos {
        for inst in &info.insts {
            if let Some(rd) = inst.rd { *vreg_def_count.entry(rd).or_default() += 1; }
            if let Some(rs1) = inst.rs1 { *vreg_use_count.entry(rs1).or_default() += 1; }
            if let Some(rs2) = inst.rs2 { *vreg_use_count.entry(rs2).or_default() += 1; }
        }
    }

    let defined = vreg_def_count.len();
    let used = vreg_use_count.len();
    let dead_defs = vreg_def_count.keys()
        .filter(|v| !vreg_use_count.contains_key(v))
        .count();
    let entry_only = (0..total_vregs)
        .filter(|v| !vreg_def_count.contains_key(v) && !vreg_use_count.contains_key(v))
        .count();

    println!("=== VReg statistics ===\n");
    println!("  Total vregs allocated:  {}", total_vregs);
    println!("  Vregs with definitions: {}", defined);
    println!("  Vregs with uses:        {}", used);
    println!("  Dead definitions:       {} (defined but never read)", dead_defs);
    println!("  Entry-only (unused):    {} (live-in vregs never referenced)", entry_only);

    // ---------------------------------------------------------------
    // Round-trip correctness check
    // ---------------------------------------------------------------
    // Map each vreg instruction back to physical regs using orig_*_preg,
    // produce specialized opcodes, and verify they match the original.
    println!("\n=== Round-trip correctness check ===\n");

    let mut roundtrip_specialized: HashMap<String, usize> = HashMap::new();
    let mut orig_idx = 0usize;
    let mut mismatches = 0usize;

    for info in &block_infos {
        for inst in &info.insts {
            // Reconstruct the physical-register specialized opcode from orig_preg fields
            let mut rt_name = inst.op.clone();
            if let Some(rd) = inst.orig_rd_preg {
                rt_name = format!("{}.{}", rt_name, REG_NAMES[rd as usize]);
            }
            if let Some(rs1) = inst.orig_rs1_preg {
                rt_name = format!("{}.{}", rt_name, REG_NAMES[rs1 as usize]);
            }
            if let Some(rs2) = inst.orig_rs2_preg {
                rt_name = format!("{}.{}", rt_name, REG_NAMES[rs2 as usize]);
            }
            *roundtrip_specialized.entry(rt_name.clone()).or_default() += 1;

            // Compare against original decoded instruction
            if orig_idx < decoded.len() {
                let d = &decoded[orig_idx];
                let mut orig_name = d.op.clone();
                if let Some(rd) = d.rd {
                    orig_name = format!("{}.{}", orig_name, REG_NAMES[rd as usize]);
                }
                if let Some(rs1) = d.rs1 {
                    orig_name = format!("{}.{}", orig_name, REG_NAMES[rs1 as usize]);
                }
                if let Some(rs2) = d.rs2 {
                    orig_name = format!("{}.{}", orig_name, REG_NAMES[rs2 as usize]);
                }

                if rt_name != orig_name {
                    if mismatches < 10 {
                        println!("  MISMATCH at idx {}: roundtrip='{}' vs original='{}'",
                            orig_idx, rt_name, orig_name);
                    }
                    mismatches += 1;
                }

                // Also verify opcode, imm, address
                if inst.op != d.op || inst.imm != d.imm || inst.addr != d.addr {
                    if mismatches < 10 {
                        println!("  DATA MISMATCH at idx {}: addr 0x{:x} vs 0x{:x}, op '{}' vs '{}', imm {:?} vs {:?}",
                            orig_idx, inst.addr, d.addr, inst.op, d.op, inst.imm, d.imm);
                    }
                    mismatches += 1;
                }
            }
            orig_idx += 1;
        }
    }

    let rt_count = roundtrip_specialized.len();
    let orig_count = phys_specialized.len();

    println!("  Instructions checked:  {}", orig_idx);
    println!("  Mismatches:            {}", mismatches);
    println!("  Roundtrip ISA opcodes: {} (original: {})", rt_count, orig_count);

    if mismatches == 0 && rt_count == orig_count && orig_idx == decoded.len() {
        println!("  PASS: round-trip is lossless");
    } else {
        println!("  FAIL: round-trip has discrepancies");
        if orig_idx != decoded.len() {
            println!("    instruction count: lifted {} vs original {}", orig_idx, decoded.len());
        }
        if rt_count != orig_count {
            // Show which opcodes differ
            let mut only_orig: Vec<_> = phys_specialized.keys()
                .filter(|k| !roundtrip_specialized.contains_key(*k))
                .take(10).collect();
            let mut only_rt: Vec<_> = roundtrip_specialized.keys()
                .filter(|k| !phys_specialized.contains_key(*k))
                .take(10).collect();
            only_orig.sort();
            only_rt.sort();
            if !only_orig.is_empty() {
                println!("    Only in original: {:?}", only_orig);
            }
            if !only_rt.is_empty() {
                println!("    Only in roundtrip: {:?}", only_rt);
            }
        }
    }

    Ok(())
}
