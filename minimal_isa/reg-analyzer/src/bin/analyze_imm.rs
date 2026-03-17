//! Analyze immediate values in a compiled FlatProgram binary.

use std::collections::HashMap;
use reg_analyzer::rv32_flat_vm::FlatProgram;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/flat3.bin".to_string());
    eprintln!("Loading flat binary: {}", path);

    let data = std::fs::read(&path)?;
    let prog: FlatProgram = bincode::deserialize(&data)?;

    eprintln!("  num_regs: {}", prog.num_regs);
    eprintln!("  entry_pc: {}", prog.entry_pc);
    eprintln!("  opcode_table entries: {}", prog.opcode_table.len());
    eprintln!("  code_segment: {} instructions", prog.code_segment.len());

    let num_insts = prog.code_segment.len();

    // Categorize base_ops
    let imm_using_ops: &[&str] = &[
        "addi", "lw", "sw", "lb", "sb", "lh", "sh", "lbu", "lhu",
        "lui", "auipc",
        "slli", "srli", "srai",
        "slti", "sltiu",
        "xori", "ori", "andi",
        "beq", "bne", "blt", "bge", "bltu", "bgeu",
        "jal", "jal_call", "jr_computed",
        "lw_frame", "sw_frame", "addi_frame",
    ];

    let branch_jump_ops: &[&str] = &[
        "beq", "bne", "blt", "bge", "bltu", "bgeu",
        "jal", "jal_call", "jr_computed",
    ];

    let mut total = 0u64;
    let mut imm_count = 0u64;
    let mut no_imm_count = 0u64;

    let mut all_imm_values: HashMap<i32, u64> = HashMap::new();
    let mut non_branch_imm_values: HashMap<i32, u64> = HashMap::new();
    let mut per_op_imm_values: HashMap<String, HashMap<i32, u64>> = HashMap::new();

    let mut min_imm: i32 = i32::MAX;
    let mut max_imm: i32 = i32::MIN;

    // Per base_op counts
    let mut base_op_imm_counts: HashMap<String, u64> = HashMap::new();

    for i in 0..num_insts {
        let opcode_id = prog.code_segment[i];
        let imm = prog.imm_table[i];
        let info = &prog.opcode_table[opcode_id as usize];
        let base_op = info.base_op.as_str();

        total += 1;

        let uses_imm = imm_using_ops.contains(&base_op);
        if uses_imm {
            imm_count += 1;
            *all_imm_values.entry(imm).or_insert(0) += 1;
            *base_op_imm_counts.entry(base_op.to_string()).or_insert(0) += 1;
            per_op_imm_values.entry(base_op.to_string()).or_default()
                .entry(imm).and_modify(|c| *c += 1).or_insert(1);

            if imm < min_imm { min_imm = imm; }
            if imm > max_imm { max_imm = imm; }

            let is_branch = branch_jump_ops.contains(&base_op);
            if !is_branch {
                *non_branch_imm_values.entry(imm).or_insert(0) += 1;
            }
        } else {
            no_imm_count += 1;
        }
    }

    println!("=== Immediate Value Analysis ===");
    println!();
    println!("Total instructions:           {:>8}", total);
    println!("  Use immediate:              {:>8} ({:.1}%)", imm_count, imm_count as f64 / total as f64 * 100.0);
    println!("  Don't use immediate:        {:>8} ({:.1}%)", no_imm_count, no_imm_count as f64 / total as f64 * 100.0);
    println!();

    println!("Unique immediate values (ALL imm-using):          {:>6}", all_imm_values.len());
    println!("Unique immediate values (non-branch/jump only):   {:>6}", non_branch_imm_values.len());
    println!();

    if min_imm <= max_imm {
        println!("Min immediate:  {:>10} (0x{:08x})", min_imm, min_imm as u32);
        println!("Max immediate:  {:>10} (0x{:08x})", max_imm, max_imm as u32);
        println!();
    }

    // Bits needed analysis
    println!("=== Bits Needed (non-branch/jump immediates) ===");
    let mut bits_hist: Vec<u64> = vec![0; 21]; // 0..20 bits
    for (&val, &count) in &non_branch_imm_values {
        let bits = if val == 0 {
            0
        } else if val > 0 {
            32 - (val as u32).leading_zeros() as usize + 1 // +1 for sign bit
        } else {
            32 - ((!val) as u32).leading_zeros() as usize + 1 // +1 for sign bit
        };
        let bits = bits.min(20);
        bits_hist[bits] += count;
    }
    let non_branch_total: u64 = non_branch_imm_values.values().sum();
    let mut cumul = 0u64;
    for bits in 0..=20 {
        if bits_hist[bits] > 0 {
            cumul += bits_hist[bits];
            println!("  {:>2} bits: {:>8} instructions ({:>5.1}%, cumul {:>5.1}%)",
                bits, bits_hist[bits],
                bits_hist[bits] as f64 / non_branch_total as f64 * 100.0,
                cumul as f64 / non_branch_total as f64 * 100.0);
        }
    }
    println!();

    // Top 20 most common immediate values (ALL)
    println!("=== Top 20 Most Common Immediate Values (ALL imm-using) ===");
    let mut sorted_all: Vec<(i32, u64)> = all_imm_values.iter().map(|(&v, &c)| (v, c)).collect();
    sorted_all.sort_by(|a, b| b.1.cmp(&a.1));
    for (rank, (val, count)) in sorted_all.iter().take(20).enumerate() {
        println!("  #{:>2}: imm={:>10} (0x{:08x})  count={:>8} ({:.1}%)",
            rank + 1, val, *val as u32, count, *count as f64 / imm_count as f64 * 100.0);
    }
    println!();

    // Top 20 most common NON-branch immediate values
    println!("=== Top 20 Most Common Immediate Values (non-branch/jump) ===");
    let mut sorted_nb: Vec<(i32, u64)> = non_branch_imm_values.iter().map(|(&v, &c)| (v, c)).collect();
    sorted_nb.sort_by(|a, b| b.1.cmp(&a.1));
    for (rank, (val, count)) in sorted_nb.iter().take(20).enumerate() {
        println!("  #{:>2}: imm={:>10} (0x{:08x})  count={:>8} ({:.1}%)",
            rank + 1, val, *val as u32, count, *count as f64 / non_branch_total as f64 * 100.0);
    }
    println!();

    // Per base_op breakdown
    println!("=== Per Base-Op Breakdown (imm-using ops) ===");
    let mut ops_sorted: Vec<(String, u64)> = base_op_imm_counts.iter().map(|(k, &v)| (k.clone(), v)).collect();
    ops_sorted.sort_by(|a, b| b.1.cmp(&a.1));
    for (op, count) in &ops_sorted {
        let unique = per_op_imm_values.get(op).map(|m| m.len()).unwrap_or(0);
        println!("  {:<16} {:>8} instructions, {:>6} unique imm values", op, count, unique);
    }
    println!();

    // For each imm-using op, show top 5 values
    println!("=== Top 5 Immediates Per Op ===");
    for (op, _) in &ops_sorted {
        if let Some(vals) = per_op_imm_values.get(op) {
            let mut sorted: Vec<(i32, u64)> = vals.iter().map(|(&v, &c)| (v, c)).collect();
            sorted.sort_by(|a, b| b.1.cmp(&a.1));
            let op_total: u64 = sorted.iter().map(|(_, c)| c).sum();
            println!("  {} ({} total, {} unique):", op, op_total, sorted.len());
            for (val, count) in sorted.iter().take(5) {
                println!("    imm={:>10} (0x{:08x})  count={:>8} ({:.1}%)",
                    val, *val as u32, count, *count as f64 / op_total as f64 * 100.0);
            }
        }
    }
    println!();

    // Summary: how many unique non-branch imm values fit in N bits
    println!("=== Unique Non-Branch/Jump Imm Values by Bit Width ===");
    let mut unique_by_bits: Vec<usize> = vec![0; 21];
    for &val in non_branch_imm_values.keys() {
        let bits = if val == 0 {
            0
        } else if val > 0 {
            32 - (val as u32).leading_zeros() as usize + 1
        } else {
            32 - ((!val) as u32).leading_zeros() as usize + 1
        };
        let bits = bits.min(20);
        unique_by_bits[bits] += 1;
    }
    let mut ucumul = 0usize;
    let total_unique = non_branch_imm_values.len();
    for bits in 0..=20 {
        if unique_by_bits[bits] > 0 {
            ucumul += unique_by_bits[bits];
            println!("  {:>2} bits: {:>6} unique values (cumul {:>6}, {:.1}% of {})",
                bits, unique_by_bits[bits], ucumul, ucumul as f64 / total_unique as f64 * 100.0, total_unique);
        }
    }

    Ok(())
}
