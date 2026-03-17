use anyhow::Result;
use reg_analyzer::rv32::*;
use reg_analyzer::rv32_regalloc::*;
use std::collections::HashMap;
use std::fs;

fn main() -> Result<()> {
    let num_regs: u32 = std::env::args().nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    let elf_path = std::env::args().nth(2)
        .unwrap_or_else(|| "../rv32-build/target/riscv32i-unknown-none-elf/release/json_query".to_string());

    let data = fs::read(&elf_path)?;
    let (decoded, text_addr, text_len) = decode_elf(&data)?;

    println!("ELF: {}", elf_path);
    println!(".text: {} bytes at 0x{:x} ({} instructions)\n",
        text_len, text_addr, decoded.len());

    // Original ISA (LLVM's 31 regs)
    let mut orig_isa: HashMap<String, usize> = HashMap::new();
    for d in &decoded {
        let mut name = d.op.clone();
        if let Some(rd)  = d.rd  { name = format!("{}.{}", name, REG_NAMES[rd as usize]); }
        if let Some(rs1) = d.rs1 { name = format!("{}.{}", name, REG_NAMES[rs1 as usize]); }
        if let Some(rs2) = d.rs2 { name = format!("{}.{}", name, REG_NAMES[rs2 as usize]); }
        *orig_isa.entry(name).or_default() += 1;
    }
    println!("Original ISA (LLVM, 31 regs): {} specialized opcodes", orig_isa.len());

    // Build CFG
    let blocks = build_cfg(&decoded, &std::collections::HashMap::new());
    println!("CFG: {} blocks\n", blocks.len());

    // Run regalloc2 + rewrite
    println!("Running regalloc2 with {} registers...\n", num_regs);
    let result = run_regalloc(&decoded, &blocks, num_regs);

    // Aggregate stats
    let total_funcs = result.func_results.len();
    let ok_funcs = result.func_results.iter().filter(|r| r.ok).count();
    let fail_funcs = total_funcs - ok_funcs;
    let total_spills: usize = result.func_results.iter().map(|r| r.num_spills).sum();

    // Merge all rewritten instructions
    let mut all_rewritten: Vec<&RewrittenInst> = Vec::new();
    for r in &result.func_results {
        if !r.ok { continue; }
        for inst in &r.rewritten {
            all_rewritten.push(inst);
        }
    }

    // Count original (non-move) and move instructions
    let total_orig_rewritten = all_rewritten.iter().filter(|i| !i.is_move).count();
    let total_moves = all_rewritten.iter().filter(|i| i.is_move).count();
    let total_rewritten = all_rewritten.len();

    // Move breakdown
    let mov_count = all_rewritten.iter().filter(|i| i.op == "mov").count();
    let spill_count = all_rewritten.iter().filter(|i| i.op == "spill").count();
    let reload_count = all_rewritten.iter().filter(|i| i.op == "reload").count();

    // Merged ISA from rewritten stream
    let mut merged_isa: HashMap<String, usize> = HashMap::new();
    for inst in &all_rewritten {
        *merged_isa.entry(inst.specialized.clone()).or_default() += 1;
    }

    println!("=== Regalloc2 Results ({} regs) ===\n", num_regs);
    println!("  Functions:       {} total, {} OK, {} failed", total_funcs, ok_funcs, fail_funcs);
    println!("  Spill slots:     {}", total_spills);

    println!("\n=== Rewritten Instruction Stream ===\n");
    println!("  Original instructions:   {:6}", total_orig_rewritten);
    println!("  Inserted moves:          {:6} ({} mov, {} spill, {} reload)",
        total_moves, mov_count, spill_count, reload_count);
    println!("  Total rewritten:         {:6} ({:.1}% overhead)",
        total_rewritten,
        if total_orig_rewritten > 0 { total_moves as f64 / total_orig_rewritten as f64 * 100.0 } else { 0.0 });
    println!("  Original ELF insts:      {:6}", decoded.len());
    println!("  Coverage:                {:.1}%",
        total_orig_rewritten as f64 / decoded.len() as f64 * 100.0);

    // ISA stats
    let isa_orig_only: usize = merged_isa.iter()
        .filter(|(name, _)| !name.starts_with("mov.") && !name.starts_with("spill.") && !name.starts_with("reload.") && *name != "nop_move" && !name.starts_with("restack."))
        .count();
    let isa_moves: usize = merged_isa.len() - isa_orig_only;
    println!("\n  ISA size:  {} total specialized opcodes", merged_isa.len());
    println!("    original ops:  {}", isa_orig_only);
    println!("    move ops:      {} (mov/spill/reload)", isa_moves);
    println!("    LLVM original: {}", orig_isa.len());
    println!("    reduction:     {:.1}x ({} → {})",
        orig_isa.len() as f64 / merged_isa.len() as f64,
        orig_isa.len(), merged_isa.len());

    // Show failures
    if fail_funcs > 0 {
        println!("\n--- Failed functions (first 10) ---");
        for r in result.func_results.iter().filter(|r| !r.ok).take(10) {
            println!("  func at 0x{:x}: {} blocks, error: {}",
                r.entry_addr, r.num_blocks,
                r.error.as_deref().unwrap_or("unknown"));
        }
    }

    // Top opcodes by frequency
    let mut sorted: Vec<_> = merged_isa.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    let total_specialized_insts: usize = sorted.iter().map(|(_, c)| **c).sum();
    println!("\n--- Top 40 specialized opcodes ({} regs) ---\n", num_regs);
    for (i, (name, count)) in sorted.iter().take(40).enumerate() {
        let pct = **count as f64 / total_specialized_insts as f64 * 100.0;
        let kind = if name.starts_with("mov.") || name.starts_with("spill.") || name.starts_with("reload.") { " [move]" } else { "" };
        println!("  {:3}. {:40} {:6} ({:.1}%){}", i + 1, name, count, pct, kind);
    }

    // Coverage: how many opcodes to cover 90%, 95%, 99%
    let mut cumulative = 0usize;
    let mut t90 = 0; let mut t95 = 0; let mut t99 = 0;
    for (i, (_, count)) in sorted.iter().enumerate() {
        cumulative += **count;
        let pct = cumulative as f64 / total_specialized_insts as f64 * 100.0;
        if t90 == 0 && pct >= 90.0 { t90 = i + 1; }
        if t95 == 0 && pct >= 95.0 { t95 = i + 1; }
        if t99 == 0 && pct >= 99.0 { t99 = i + 1; }
    }
    println!("\n  Coverage: 90%={} opcodes, 95%={} opcodes, 99%={} opcodes (of {} total)",
        t90, t95, t99, merged_isa.len());

    // Show a sample of the rewritten stream (first function, first 20 insts)
    if let Some(first_ok) = result.func_results.iter().find(|r| r.ok && !r.rewritten.is_empty()) {
        println!("\n--- Sample rewritten stream (func at 0x{:x}, first 20 insts) ---\n",
            first_ok.entry_addr);
        for (i, inst) in first_ok.rewritten.iter().take(20).enumerate() {
            let move_tag = if inst.is_move { " [move]" } else { "" };
            println!("  {:3}. 0x{:06x}  {:40}{}", i, inst.addr, inst.specialized, move_tag);
        }
        if first_ok.rewritten.len() > 20 {
            println!("  ... ({} more)", first_ok.rewritten.len() - 20);
        }
    }

    Ok(())
}
