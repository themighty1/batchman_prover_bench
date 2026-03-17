use anyhow::Result;
use reg_analyzer::rv32_isa_vm::IO_OUTPUT_LEN;
use reg_analyzer::rv32_vm::Rv32Vm;
use std::fs;

fn main() -> Result<()> {
    let elf_path = std::env::args().nth(1)
        .unwrap_or_else(|| "../ecdsa-rv32/target/riscv32i-unknown-none-elf/release/ecdsa-rv32".to_string());

    let data = fs::read(&elf_path)?;

    println!("=== RV32 ECDSA Execution ===\n");

    let mut vm = Rv32Vm::new();
    let entry = vm.load_elf(&data)?;
    vm.pc = entry;

    // 16-byte message: "Hello ECDSA test"
    let msg: [u8; 16] = [
        0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x20, 0x45, 0x43,
        0x44, 0x53, 0x41, 0x20, 0x74, 0x65, 0x73, 0x74,
    ];
    vm.write_input(&msg);

    // 64-byte signature (r=1, s=1 in big-endian P-256 scalar format)
    // These are valid non-zero scalars < n, so Signature::from_slice succeeds
    // and the full EC verification math runs (result will be 0 = invalid).
    let mut sig = [0u8; 64];
    sig[31] = 1;  // r = 1
    sig[63] = 1;  // s = 1
    vm.memory.write_path(&sig);

    // Initialize SP
    vm.regs[2] = 0x7FFF0000;

    println!("  Entry: 0x{:x}", entry);
    println!("  ELF size: {} bytes", data.len());

    let max_steps: u64 = 500_000_000;
    let report_interval: u64 = 10_000_000;

    while !vm.halted && vm.steps < max_steps {
        let prev_pc = vm.pc;
        vm.step()?;
        if vm.pc == prev_pc {
            vm.halted = true;
        }
        if vm.steps % report_interval == 0 {
            eprintln!("  ... {} M steps, pc=0x{:x}", vm.steps / 1_000_000, vm.pc);
        }
    }

    let result = vm.memory.read_u32(IO_OUTPUT_LEN);
    println!("\n  Steps: {}", vm.steps);
    println!("  Halted: {}", vm.halted);
    println!("  Result (IO_OUTPUT_LEN): {}", result);
    if result == 1 {
        println!("  Signature VALID");
    } else {
        println!("  Signature INVALID (expected for dummy sig)");
    }

    Ok(())
}
