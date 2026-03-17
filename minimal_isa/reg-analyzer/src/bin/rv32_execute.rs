use anyhow::Result;
use reg_analyzer::rv32::*;
use reg_analyzer::rv32_regalloc::*;
use reg_analyzer::rv32_vm::Rv32Vm;
use std::collections::HashMap;
use std::fs;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let num_regs: u32 = args.get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    let elf_path = args.get(2)
        .cloned()
        .unwrap_or_else(|| "../rv32-build/target/riscv32i-unknown-none-elf/release/json_query".to_string());

    let fixture_path = args.get(3).cloned();
    let query_path = args.get(4).cloned();
    let query_mode = query_path.is_some();

    let data = fs::read(&elf_path)?;

    // ---------------------------------------------------------------
    // Step 1: Static rewrite (regalloc2)
    // ---------------------------------------------------------------
    println!("=== Step 1: Static rewrite ({} regs) ===\n", num_regs);
    let (decoded, text_addr, text_len) = decode_elf(&data)?;
    let blocks = build_cfg(&decoded, &std::collections::HashMap::new());
    let alloc_result = run_regalloc(&decoded, &blocks, num_regs);

    let ok_funcs = alloc_result.func_results.iter().filter(|r| r.ok).count();
    let total_funcs = alloc_result.func_results.len();
    println!("  Functions: {}/{} OK", ok_funcs, total_funcs);

    // Print all failed functions
    eprintln!("\nFailed functions:");
    for r in alloc_result.func_results.iter() {
        if !r.ok {
            eprintln!("  0x{:08x}: {} blocks, {} insts, {} vregs - {:?}",
                r.entry_addr, r.num_blocks, r.num_insts, r.num_vregs, r.error);
        }
    }

    // Build addr → specialized opcode index mapping
    let mut opcode_names: Vec<String> = Vec::new();
    let mut opcode_to_idx: HashMap<String, usize> = HashMap::new();
    let mut addr_to_opcode_idx: HashMap<u32, usize> = HashMap::new();

    let mut move_opcode_names: Vec<String> = Vec::new();
    let mut move_opcode_to_idx: HashMap<String, usize> = HashMap::new();

    for r in &alloc_result.func_results {
        if !r.ok { continue; }
        for inst in &r.rewritten {
            if !inst.is_move {
                let idx = *opcode_to_idx.entry(inst.specialized.clone()).or_insert_with(|| {
                    opcode_names.push(inst.specialized.clone());
                    opcode_names.len() - 1
                });
                addr_to_opcode_idx.insert(inst.addr, idx);
            } else {
                move_opcode_to_idx.entry(inst.specialized.clone()).or_insert_with(|| {
                    move_opcode_names.push(inst.specialized.clone());
                    move_opcode_names.len() - 1
                });
            }
        }
    }

    let static_isa_size = opcode_names.len() + move_opcode_names.len();
    println!("  Static ISA: {} opcodes ({} original + {} move)",
        static_isa_size, opcode_names.len(), move_opcode_names.len());
    println!("  addr→opcode map: {} entries", addr_to_opcode_idx.len());

    // ---------------------------------------------------------------
    // Step 2: Execute with standard RV32IM interpreter
    // ---------------------------------------------------------------
    println!("\n=== Step 2: Execute RV32IM ===\n");

    let mut vm = Rv32Vm::new();
    let entry = vm.load_elf(&data)?;
    vm.pc = entry;

    // Write input: fixture file (JSON) + optional path for query mode
    if let Some(ref fp) = fixture_path {
        let fixture = fs::read(fp)?;
        vm.write_input(&fixture);
    }
    if let Some(ref qp) = query_path {
        vm.memory.write_path(qp.as_bytes());
    }

    println!("  Entry: 0x{:x}", entry);
    println!("  .text: {} bytes at 0x{:x} ({} insts)", text_len, text_addr, decoded.len());

    // Track dynamic ISA
    let mut dynamic_counts = vec![0u64; opcode_names.len()];
    let mut unmapped_count = 0u64;
    let mut total_dynamic = 0u64;

    let max_steps: u64 = 200_000_000;
    let report_interval: u64 = 10_000_000;

    // Ring buffer for last N instructions before crash
    let ring_size = 40;
    let mut ring: Vec<(u64, u32, u32, [u32; 32])> = Vec::with_capacity(ring_size);
    let mut ring_idx = 0usize;

    while !vm.halted && vm.steps < max_steps {
        let pc = vm.pc;
        let inst_word = vm.memory.read_u32(pc);
        let entry = (vm.steps, pc, inst_word, vm.regs);
        if ring.len() < ring_size {
            ring.push(entry);
        } else {
            ring[ring_idx] = entry;
        }
        ring_idx = (ring_idx + 1) % ring_size;

        match vm.step() {
            Ok(()) => {}
            Err(e) => {
                eprintln!("\n--- Last {} instructions before error ---", ring.len().min(ring_size));
                for i in 0..ring.len().min(ring_size) {
                    let idx = (ring_idx + i) % ring.len();
                    let (step, pc, inst, regs) = &ring[idx];
                    eprintln!("  [{:6}] pc=0x{:06x} inst=0x{:08x} sp=0x{:x} ra=0x{:x} a0=0x{:x} a1=0x{:x} t0=0x{:x}",
                        step, pc, inst, regs[2], regs[1], regs[10], regs[11], regs[5]);
                }
                return Err(e);
            }
        }

        // Detect tight infinite loop (jal x0, 0 = halt)
        if vm.pc == pc {
            vm.halted = true;
        }

        if let Some(&idx) = addr_to_opcode_idx.get(&pc) {
            dynamic_counts[idx] += 1;
        } else {
            unmapped_count += 1;
        }
        total_dynamic += 1;

        if vm.steps % report_interval == 0 {
            eprintln!("  ... {} M steps, pc=0x{:x}", vm.steps / 1_000_000, vm.pc);
        }
    }

    println!("  Steps: {}", vm.steps);
    println!("  Halted: {}", vm.halted);

    // Read result
    if query_mode {
        let result = vm.memory.read_output_string();
        println!("\n  Query result: {:?}", result);
    } else {
        let result = vm.memory.read_u32(0x80000000);
        let expected = 281;  // 2 + 3*93 items from generate_test_json(2048)
        println!("\n  Result at 0x80000000: {}", result);
        if result == expected {
            println!("  PASS (expected {})", expected);
        } else {
            println!("  FAIL (expected {}, got {})", expected, result);
        }
    }

    // ---------------------------------------------------------------
    // Step 3: Dynamic ISA statistics
    // ---------------------------------------------------------------
    println!("\n=== Step 3: Dynamic ISA ({} regs) ===\n", num_regs);

    let mapped_count = total_dynamic - unmapped_count;
    println!("  Total dynamic insts: {}", total_dynamic);
    println!("  Mapped to ISA:       {} ({:.1}%)", mapped_count,
        mapped_count as f64 / total_dynamic as f64 * 100.0);
    println!("  Unmapped:            {} ({:.1}%)", unmapped_count,
        unmapped_count as f64 / total_dynamic as f64 * 100.0);

    // Count unique dynamic opcodes
    let mut dynamic_isa: Vec<(usize, u64)> = dynamic_counts.iter().enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(i, &c)| (i, c))
        .collect();
    dynamic_isa.sort_by(|a, b| b.1.cmp(&a.1));

    let dynamic_isa_size = dynamic_isa.len();
    println!("  Dynamic ISA: {} unique opcodes exercised (of {} static)",
        dynamic_isa_size, opcode_names.len());

    // Top opcodes
    let dynamic_total: u64 = dynamic_isa.iter().map(|(_, c)| c).sum();
    println!("\n--- Top 30 dynamic specialized opcodes ---\n");
    for (rank, (idx, count)) in dynamic_isa.iter().take(30).enumerate() {
        let pct = *count as f64 / dynamic_total as f64 * 100.0;
        println!("  {:3}. {:40} {:8} ({:.1}%)", rank + 1, opcode_names[*idx], count, pct);
    }

    // Coverage
    let mut cumulative = 0u64;
    let mut t90 = 0; let mut t95 = 0; let mut t99 = 0;
    for (i, (_, count)) in dynamic_isa.iter().enumerate() {
        cumulative += count;
        let pct = cumulative as f64 / dynamic_total as f64 * 100.0;
        if t90 == 0 && pct >= 90.0 { t90 = i + 1; }
        if t95 == 0 && pct >= 95.0 { t95 = i + 1; }
        if t99 == 0 && pct >= 99.0 { t99 = i + 1; }
    }
    println!("\n  Coverage: 90%={} opcodes, 95%={} opcodes, 99%={} opcodes (of {} dynamic)",
        t90, t95, t99, dynamic_isa_size);

    // Unused static opcodes
    let unused_static = opcode_names.len() - dynamic_isa_size;
    println!("  Unused static opcodes: {} ({:.1}% of static ISA never exercised)",
        unused_static, unused_static as f64 / opcode_names.len() as f64 * 100.0);

    Ok(())
}
