//! Execute a flat compiled program from file.
//!
//! Usage:
//!   rv32_run_flat [program.bin] [json_file] [path] [expected_string]
//!
//! Defaults: flat_program.bin, fixture.json, data.5.v, val5

use anyhow::Result;
use reg_analyzer::rv32_flat_vm::*;
use reg_analyzer::rv32_isa_vm::{MAILBOX_BASE, IO_OUTPUT_LEN};
use std::fs;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let program_path = args.get(1)
        .cloned()
        .unwrap_or_else(|| "flat_program.bin".to_string());

    // Load compiled program
    let encoded = fs::read(&program_path)?;
    let program: FlatProgram = bincode::deserialize(&encoded)?;

    // Use u8 code segment if present, otherwise widen from u16
    let code_u16: Vec<u16> = if !program.code_segment_u8.is_empty() {
        program.code_segment_u8.iter().map(|&b| b as u16).collect()
    } else {
        program.code_segment.clone()
    };
    let code_len = code_u16.len();

    println!("=== rv32_run_flat ({} regs) ===", program.num_regs);
    println!("  Program:      {} ({} bytes)", program_path, encoded.len());
    println!("  Code size:    {} entries", code_len);
    println!("  ISA size:     {} opcodes", program.opcode_table.len());
    println!("  Entry PC:     0x{:x}", program.entry_pc);

    // Build opcode table
    let opcode_table: Vec<OpcodeInfo> = program.opcode_table.iter().map(|s| {
        OpcodeInfo {
            name: s.name.clone(),
            base_op: s.base_op.clone(),
            rd: s.rd,
            rs1: s.rs1,
            rs2: s.rs2,
            orig_rd: s.orig_rd,
            orig_rs1: None,
            orig_rs2: None,
        }
    }).collect();

    // Detect input mode
    let input_arg = args.get(2).cloned();
    let arg3 = args.get(3).cloned();
    let arg4 = args.get(4).cloned();
    let query_mode = args.len() >= 5;
    let input_is_int = !query_mode && input_arg.as_ref().map_or(false, |s| s.parse::<u32>().is_ok());

    // Create VM with code and imm_table in separate memory space
    let mut vm = FlatVm::new(program.num_regs as usize, code_u16, program.imm_table.clone());

    // Load data segments (into data memory only, code is separate)
    for seg in &program.segments {
        vm.memory.write_bytes(seg.vaddr, &seg.data);
    }

    // Write inputs
    if query_mode {
        let json_path = input_arg.as_deref().unwrap_or("../guest-programs/json-query/fixtures/test_input.json");
        let path = arg3.as_deref().unwrap_or("data.5.v");
        let expected = arg4.as_deref().unwrap_or("val5");
        println!("  JSON:         {}", json_path);
        println!("  Path:         {}", path);
        println!("  Expected:     {:?}", expected);

        let fixture = fs::read(json_path)?;
        vm.memory.write_input(&fixture);
        vm.memory.write_path(path.as_bytes());
    } else if input_is_int {
        let n: u32 = input_arg.as_ref().unwrap().parse().unwrap();
        println!("  Input:        n={}", n);
        vm.memory.write_input_u32(n);
    } else if let Some(ref file_path) = input_arg {
        println!("  Input:        {}", file_path);
        let fixture = fs::read(file_path)?;
        vm.memory.write_input(&fixture);
    }

    // Initialize SP in mailbox; frame_reg starts at 0 (offset into spill_area)
    vm.memory.write_u32(MAILBOX_BASE + 2 * 4, 0x7FFF_0000);
    // entry_pc is a 0-based instruction index
    vm.pc = program.entry_pc;

    println!();

    // Execute
    vm.execute(&opcode_table)?;

    println!("  Steps:        {}", vm.steps);
    let pages = vm.memory.num_pages();
    println!("  Memory:       {} pages = {} KB", pages, pages * 4);

    // Check result
    if query_mode {
        let result = vm.memory.read_output_string();
        let expected = arg4.as_deref().unwrap_or("val5");
        println!("  Result:       {:?} (expected {:?})", result, expected);
        if result == expected {
            println!("\n  PASS");
        } else {
            println!("\n  FAIL");
            std::process::exit(1);
        }
    } else {
        let expected: u32 = if input_is_int {
            arg3.as_ref().and_then(|s| s.parse().ok()).unwrap_or(0)
        } else {
            arg3.as_ref().and_then(|s| s.parse().ok()).unwrap_or(0)
        };
        let result = vm.memory.read_u32(IO_OUTPUT_LEN);
        println!("  Result:       {} (expected {})", result, expected);
        if result == expected {
            println!("\n  PASS");
        } else {
            println!("\n  FAIL");
            std::process::exit(1);
        }
    }

    Ok(())
}
