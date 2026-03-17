//! Execute a canonical 3-register compiled program using Canon3Vm.
//!
//! Usage:
//!   rv32_run_canon3 [program.bin] [json_file] [path] [expected_string]
//!   rv32_run_canon3 [program.bin] [n]

use anyhow::Result;
use reg_analyzer::rv32_flat_vm::FlatProgram;
use reg_analyzer::rv32_isa_vm::{MAILBOX_BASE, IO_OUTPUT_LEN};
use reg_analyzer::canon3_vm::Canon3Vm;
use std::fs;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let program_path = args.get(1).cloned().unwrap_or_else(|| "canonical3.bin".to_string());

    let encoded = fs::read(&program_path)?;
    let program: FlatProgram = bincode::deserialize(&encoded)?;

    let opcode_names: Vec<String> = program.opcode_table.iter().map(|o| o.base_op.clone()).collect();
    let remapped_code = Canon3Vm::remap_code(&opcode_names, &program.code_segment_u8)?;

    println!("=== rv32_run_canon3 (3 regs) ===");
    println!("  Program:      {} ({} bytes)", program_path, encoded.len());
    println!("  Code size:    {} entries", remapped_code.len());
    println!("  ISA size:     {} opcodes", program.opcode_table.len());
    println!("  Entry PC:     0x{:x}", program.entry_pc);

    let input_arg = args.get(2).cloned();
    let arg3 = args.get(3).cloned();
    let arg4 = args.get(4).cloned();
    let query_mode = args.len() >= 5;
    let input_is_int = !query_mode && input_arg.as_ref().map_or(false, |s| s.parse::<u32>().is_ok());

    let mut vm = Canon3Vm::new(remapped_code, program.imm_table.clone());

    for seg in &program.segments {
        vm.memory.write_bytes(seg.vaddr, &seg.data);
    }

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

    // Stack: 4KB should be enough for typical JSON inputs, but needs program-specific
    // analysis on various input sizes and nesting depths when we compact memory later.
    const STACK_SIZE: u32 = 4 * 1024;
    const STACK_TOP: u32 = STACK_SIZE;
    vm.memory.write_u32(MAILBOX_BASE + 2 * 4, STACK_TOP);
    vm.pc = program.entry_pc;
    println!();

    vm.execute()?;

    println!("  Steps:        {}", vm.steps);
    let pages = vm.memory.num_pages();
    println!("  Memory:       {} pages = {} KB", pages, pages * 4);

    if query_mode {
        let result = vm.memory.read_output_string();
        let expected = arg4.as_deref().unwrap_or("val5");
        println!("  Result:       {:?} (expected {:?})", result, expected);
        if result == expected { println!("\n  PASS"); }
        else { println!("\n  FAIL"); std::process::exit(1); }
    } else {
        let expected: u32 = arg3.as_ref().and_then(|s| s.parse().ok()).unwrap_or(0);
        let result = vm.memory.read_u32(IO_OUTPUT_LEN);
        println!("  Result:       {} (expected {})", result, expected);
        if result == expected { println!("\n  PASS"); }
        else { println!("\n  FAIL"); std::process::exit(1); }
    }

    Ok(())
}
