/// End-to-end test: run an ELF through both standard RV32 VM and ISA VM,
/// verify both produce the same result for various inputs.
///
/// MMIO convention:
///   IO_OUTPUT_LEN  OUTPUT_ADDR      u32 result
///   IO_INPUT_LEN   INPUT_LEN_ADDR   u32 n (iteration count)
///
/// Usage:
///   rv32_toy_test <num_regs> <elf_path> [n ...]
///   rv32_toy_test --stress <duration_secs> <elf_path>

use anyhow::Result;
use reg_analyzer::rv32::{decode_elf, get_elf_functions_named, build_cfg, classify_jalr_x0};
use reg_analyzer::rv32_regalloc::run_regalloc_with_symbols;
use reg_analyzer::rv32_isa_vm::{Rv32FuncInfo, Rv32IsaVm};
use reg_analyzer::rv32_vm::Rv32Vm;
use std::collections::HashMap;
use std::fs;

use reg_analyzer::rv32_isa_vm::{IO_INPUT_LEN, IO_OUTPUT_LEN};

const INPUT_ADDR:  u32 = IO_INPUT_LEN;
const OUTPUT_ADDR: u32 = IO_OUTPUT_LEN;

fn run_standard_vm(data: &[u8], input: u32) -> Result<(u32, u64)> {
    let mut vm = Rv32Vm::new();
    let entry = vm.load_elf(data)?;
    vm.pc = entry;
    vm.memory.write_u32(INPUT_ADDR, input);
    vm.run(10_000_000)?;
    Ok((vm.memory.read_u32(OUTPUT_ADDR), vm.steps))
}

fn run_isa_vm(
    data: &[u8],
    num_regs: u32,
    func_table: &HashMap<u32, Rv32FuncInfo>,
    addr_to_func: &HashMap<u32, u32>,
    entry_addr: u32,
    input: u32,
) -> Result<(u32, u64)> {
    let mut vm = Rv32IsaVm::new(num_regs as usize);
    vm.load_elf(data)?;
    vm.conv_regs[2] = 0x7FFF0000;
    vm.frame_reg = 0x8000_1000;
    vm.memory.write_u32(INPUT_ADDR, input);
    let entry_func = func_table.get(&entry_addr)
        .ok_or_else(|| anyhow::anyhow!("entry function not found"))?;
    vm.execute_function(entry_func, func_table, addr_to_func, None)?;
    Ok((vm.memory.read_u32(OUTPUT_ADDR), vm.steps))
}

fn setup_pipeline(data: &[u8], num_regs: u32) -> Result<(HashMap<u32, Rv32FuncInfo>, HashMap<u32, u32>, u32)> {
    let (decoded_raw, _, _) = decode_elf(data)?;
    let elf_funcs_named = get_elf_functions_named(data)?;
    let mut decoded = decoded_raw;
    let elf_funcs: Vec<(u32, u32)> = elf_funcs_named.iter().map(|(a, s, _)| (*a, *s)).collect();
    let (jump_table_targets, _jump_table_bases) = classify_jalr_x0(&mut decoded, data, &elf_funcs_named);
    let blocks = build_cfg(&decoded, &jump_table_targets);
    let alloc_result = run_regalloc_with_symbols(&decoded, &blocks, num_regs, &elf_funcs);

    let ok = alloc_result.func_results.iter().filter(|r| r.ok).count();
    let total = alloc_result.func_results.len();
    println!("  Functions: {}/{} OK", ok, total);
    for r in &alloc_result.func_results {
        if !r.ok {
            println!("    FAILED 0x{:x}: {}", r.entry_addr, r.error.as_deref().unwrap_or("?"));
        }
    }

    let mut func_table: HashMap<u32, Rv32FuncInfo> = HashMap::new();
    let mut addr_to_func: HashMap<u32, u32> = HashMap::new();
    for r in &alloc_result.func_results {
        if !r.ok { continue; }
        for inst in &r.rewritten {
            if inst.addr != 0 && inst.addr < 0xF000_0000 {
                addr_to_func.insert(inst.addr, r.entry_addr);
            }
        }
        func_table.insert(r.entry_addr, Rv32FuncInfo {
            rewritten: r.rewritten.clone(),
            num_spill_slots: r.num_spill_slots,
            entry_reg_map: r.entry_reg_map.clone(),
            jr_table_redirects: r.jr_table_redirects.clone(),
        });
    }

    let entry_addr = {
        use object::read::elf::FileHeader as _;
        use object::Endianness;
        let elf = object::elf::FileHeader32::<Endianness>::parse(data)?;
        elf.e_entry.get(elf.endian()?)
    };

    Ok((func_table, addr_to_func, entry_addr))
}

fn stress_test(data: &[u8], elf_path: &str, duration_secs: u64, num_regs: u32) -> Result<()> {
    use std::time::Instant;

    let reg_counts: Vec<u32> = vec![num_regs];
    let deadline = Instant::now() + std::time::Duration::from_secs(duration_secs);

    println!("=== Stress Test ({} sec, reg counts: {:?}) ===", duration_secs, reg_counts);
    println!("  ELF: {}", elf_path);

    let mut pipelines = Vec::new();
    for &nr in &reg_counts {
        print!("  Building pipeline for {} regs... ", nr);
        let (func_table, addr_to_func, entry_addr) = setup_pipeline(data, nr)?;
        println!("done");
        pipelines.push((nr, func_table, addr_to_func, entry_addr));
    }

    let mut rng_state: u64 = 0xdeadbeef12345678;
    let mut total_tests: u64 = 0;
    let mut total_pass: u64 = 0;
    let mut total_timeout: u64 = 0;
    let mut first_fail: Option<String> = None;

    println!("\n  Running...");
    let start = Instant::now();
    let mut last_report = start;

    while Instant::now() < deadline {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let n: u32 = if (rng_state & 0xF) < 11 {
            ((rng_state >> 16) % 51) as u32
        } else {
            ((rng_state >> 16) % 501) as u32
        };

        let std_res = run_standard_vm(data, n);

        for (nr, func_table, addr_to_func, entry_addr) in &pipelines {
            let isa_res = run_isa_vm(data, *nr, func_table, addr_to_func, *entry_addr, n);
            total_tests += 1;

            match (&std_res, &isa_res) {
                (Ok((std_out, _)), Ok((isa_out, _))) => {
                    if *std_out == *isa_out {
                        total_pass += 1;
                    } else if first_fail.is_none() {
                        let msg = format!("regs={} n={} std={} isa={}", nr, n, std_out, isa_out);
                        println!("  FAIL: {}", msg);
                        first_fail = Some(msg);
                    }
                }
                _ => {
                    let is_timeout = isa_res.as_ref().err().map_or(false, |e| format!("{}", e).contains("step limit"))
                        || std_res.as_ref().err().map_or(false, |e| format!("{}", e).contains("step limit"));
                    if is_timeout {
                        total_timeout += 1;
                    } else if first_fail.is_none() {
                        let std_str = std_res.as_ref().map(|(v,_)| v.to_string()).unwrap_or_else(|e| format!("ERR({})", e));
                        let isa_str = isa_res.as_ref().map(|(v,_)| v.to_string()).unwrap_or_else(|e| format!("ERR({})", e));
                        let msg = format!("regs={} n={} std={} isa={}", nr, n, std_str, isa_str);
                        println!("  FAIL: {}", msg);
                        first_fail = Some(msg);
                    }
                }
            }
        }

        let now = Instant::now();
        if now.duration_since(last_report).as_secs() >= 5 {
            let elapsed = now.duration_since(start).as_secs_f64();
            let real_fails = total_tests - total_pass - total_timeout;
            println!("  [{:.0}s] {} tests, {} pass, {} fail, {} timeout ({:.1} tests/sec)",
                elapsed, total_tests, total_pass, real_fails, total_timeout, total_tests as f64 / elapsed);
            last_report = now;
        }
    }

    let elapsed = Instant::now().duration_since(start).as_secs_f64();
    let real_fails = total_tests - total_pass - total_timeout;
    println!("\n  === Results ({:.1}s) ===", elapsed);
    println!("  Total tests: {}", total_tests);
    println!("  Passed:      {}", total_pass);
    println!("  Failed:      {}", real_fails);
    println!("  Timeouts:    {}", total_timeout);
    println!("  Rate:        {:.1} tests/sec", total_tests as f64 / elapsed);

    if let Some(msg) = &first_fail {
        println!("  First failure: {}", msg);
    }

    if real_fails > 0 {
        anyhow::bail!("{} tests failed", real_fails);
    }
    println!("  ALL PASS");
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // --stress <duration> <elf_path> [--regs N]
    if let Some(si) = args.iter().position(|a| a == "--stress") {
        let mut duration = 30u64;
        let mut elf_path = String::new();
        let mut num_regs = 3u32;
        let rest = &args[si+1..];
        let mut i = 0;
        while i < rest.len() {
            if rest[i] == "--regs" && i + 1 < rest.len() {
                num_regs = rest[i+1].parse().unwrap_or(3);
                i += 2;
            } else if let Ok(d) = rest[i].parse::<u64>() {
                duration = d;
                i += 1;
            } else {
                elf_path = rest[i].to_string();
                i += 1;
            }
        }
        if elf_path.is_empty() {
            anyhow::bail!("--stress requires an ELF path");
        }
        let data = fs::read(&elf_path)?;
        return stress_test(&data, &elf_path, duration, num_regs);
    }

    // Normal mode: rv32_toy_test <num_regs> <elf_path> [n ...]
    let num_regs: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let elf_path = args.get(2).expect("usage: rv32_toy_test <num_regs> <elf_path> [n ...]");
    let data = fs::read(elf_path)?;

    println!("=== RV32 Test ({} regs, {}) ===", num_regs, elf_path);
    let (func_table, addr_to_func, entry_addr) = setup_pipeline(&data, num_regs)?;

    let inputs: Vec<u32> = if args.len() > 3 {
        args[3..].iter().filter_map(|s| s.parse().ok()).collect()
    } else {
        vec![0, 1, 2, 3, 4, 5, 10, 20]
    };

    println!("  {:>6}  {:>12}  {:>12}  {:>10}  {:>10}  {}",
        "n", "std_out", "isa_out", "std_steps", "isa_steps", "status");
    println!("  {:->6}  {:->12}  {:->12}  {:->10}  {:->10}  {:->6}",
        "", "", "", "", "", "");

    let std_only = std::env::var("STD_ONLY").is_ok();
    let mut all_pass = true;
    for &n in &inputs {
        let std_res = run_standard_vm(&data, n);
        if std_only {
            match std_res {
                Ok((out, steps)) => println!("  {:>6}  {:>12}  {:>10}", n, out, steps),
                Err(e) => println!("  {:>6}  ERR({})", n, e),
            }
            continue;
        }
        let isa_res = run_isa_vm(&data, num_regs, &func_table, &addr_to_func, entry_addr, n);

        match (std_res, isa_res) {
            (Ok((std_out, std_steps)), Ok((isa_out, isa_steps))) => {
                let pass = std_out == isa_out;
                if !pass { all_pass = false; }
                println!("  {:>6}  {:>12}  {:>12}  {:>10}  {:>10}  {}",
                    n, std_out, isa_out, std_steps, isa_steps,
                    if pass { "PASS" } else { "FAIL" });
            }
            (std_res, isa_res) => {
                all_pass = false;
                let std_str = std_res.map(|(v,_)| v.to_string()).unwrap_or_else(|e| format!("ERR({})", e));
                let isa_str = isa_res.map(|(v,_)| v.to_string()).unwrap_or_else(|e| format!("ERR({})", e));
                println!("  {:>6}  {:>12}  {:>12}  {:>10}  {:>10}  FAIL",
                    n, std_str, isa_str, "-", "-");
            }
        }
    }

    println!();
    if all_pass { println!("  All tests passed."); } else { println!("  SOME TESTS FAILED."); }
    Ok(())
}
