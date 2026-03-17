//! Compile an RV32 ELF into a serialized ISA-VM program file.
//!
//! Usage:
//!   rv32_compile [num_regs] [elf_path] [output_path]
//!
//! Defaults: 4 regs, json_query.elf, program.bin

use anyhow::Result;
use reg_analyzer::rv32::{decode_elf, get_elf_functions_named, build_cfg, classify_jalr_x0};
use reg_analyzer::rv32_regalloc::run_regalloc_with_symbols;
use reg_analyzer::rv32_isa_vm::{Rv32FuncInfo, CompiledProgram, MemSegment};
use std::collections::HashMap;
use std::fs;

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
        .unwrap_or_else(|| "program.bin".to_string());

    println!("=== rv32_compile ({} regs) ===", num_regs);
    println!("  ELF:    {}", elf_path);
    println!("  Output: {}", output_path);

    // Step 1: decode + regalloc
    let data = fs::read(&elf_path)?;
    let (decoded_raw, _text_addr, _text_len) = decode_elf(&data)?;
    let elf_funcs_named = get_elf_functions_named(&data)?;
    let mut decoded = decoded_raw;
    let elf_funcs: Vec<(u32, u32)> = elf_funcs_named.iter().map(|(a, s, _)| (*a, *s)).collect();
    let (jump_table_targets, jump_table_bases) = classify_jalr_x0(&mut decoded, &data, &elf_funcs_named);
    let blocks = build_cfg(&decoded, &jump_table_targets);
    let alloc_result = run_regalloc_with_symbols(&decoded, &blocks, num_regs, &elf_funcs);

    let ok_funcs = alloc_result.func_results.iter().filter(|r| r.ok).count();
    let total_funcs = alloc_result.func_results.len();
    println!("  Functions: {}/{} OK", ok_funcs, total_funcs);

    for r in &alloc_result.func_results {
        if !r.ok {
            eprintln!("  FAIL 0x{:x}: {:?}", r.entry_addr, r.error);
        }
    }

    // Count unique opcodes
    let mut all_opcodes = std::collections::HashSet::new();
    for r in &alloc_result.func_results {
        if r.ok {
            for inst in &r.rewritten {
                all_opcodes.insert(inst.specialized.clone());
            }
        }
    }
    println!("  ISA size: {} unique opcodes", all_opcodes.len());

    // Step 2: build function table + addr_to_func
    let mut functions = Vec::new();
    let mut addr_to_func: HashMap<u32, u32> = HashMap::new();

    for r in &alloc_result.func_results {
        if !r.ok { continue; }
        for inst in &r.rewritten {
            if inst.addr != 0 && inst.addr < 0xF000_0000 {
                addr_to_func.insert(inst.addr, r.entry_addr);
            }
        }
        // Note: addr_to_idx is not included in the compiled binary — all branch/jump
        // targets are resolved to instruction indices by post-rewrite passes
        // (pass_resolve_branches, pass_rewrite_jump_tables).
        functions.push(Rv32FuncInfo {
            rewritten: r.rewritten.clone(),
            num_spill_slots: r.num_spill_slots,
            entry_reg_map: r.entry_reg_map.clone(),
            jr_table_redirects: r.jr_table_redirects.clone(),
        });
    }

    // Step 3: extract ELF loadable segments
    let mut segments = extract_elf_segments(&data)?;

    // Get entry point
    let entry_addr = {
        use object::elf::*;
        use object::read::elf::FileHeader as _;
        use object::Endianness;
        let elf = FileHeader32::<Endianness>::parse(data.as_slice())?;
        let endian = elf.endian()?;
        elf.e_entry.get(endian)
    };

    // Step 3b: rewrite jump tables in memory to instruction indices
    reg_analyzer::rv32_passes::pass_rewrite_jump_tables(
        &mut functions, &mut segments, &jump_table_bases,
    );

    let total_insts: usize = functions.iter().map(|f| f.rewritten.len()).sum();
    println!("  Total instructions: {}", total_insts);
    println!("  Entry: 0x{:x}", entry_addr);

    // Step 4: serialize
    let program = CompiledProgram {
        num_regs,
        entry_addr,
        segments,
        functions,
        addr_to_func,
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
