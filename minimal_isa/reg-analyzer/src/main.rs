use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use wasmparser::{Parser, Payload};

#[derive(Debug, Clone)]
struct LiveRange {
    local_id: u32,
    first_use: u32,
    last_use: u32,
}

#[derive(Debug)]
struct FunctionAnalysis {
    func_idx: u32,
    num_params: u32,
    num_locals: u32,
    total_locals: u32,
    live_ranges: Vec<LiveRange>,
    access_counts: HashMap<u32, u32>,  // local_id -> number of accesses
    max_pressure: u32,
    num_instructions: u32,
}

fn analyze_function_body(
    func_idx: u32,
    func_body: wasmparser::FunctionBody,
    num_params: u32,
) -> Result<FunctionAnalysis> {
    let mut num_locals = 0u32;
    let locals_reader = func_body.get_locals_reader()?;
    for local in locals_reader {
        let (count, _ty) = local?;
        num_locals += count;
    }

    let total_locals = num_params + num_locals;

    // Track first and last use of each local, plus access counts
    let mut first_use: HashMap<u32, u32> = HashMap::new();
    let mut last_use: HashMap<u32, u32> = HashMap::new();
    let mut access_counts: HashMap<u32, u32> = HashMap::new();

    let mut instruction_idx = 0u32;
    let ops_reader = func_body.get_operators_reader()?;

    for op in ops_reader {
        let op = op?;

        let local_id = match op {
            wasmparser::Operator::LocalGet { local_index } => Some(local_index),
            wasmparser::Operator::LocalSet { local_index } => Some(local_index),
            wasmparser::Operator::LocalTee { local_index } => Some(local_index),
            _ => None,
        };

        if let Some(id) = local_id {
            first_use.entry(id).or_insert(instruction_idx);
            last_use.insert(id, instruction_idx);
            *access_counts.entry(id).or_insert(0) += 1;
        }

        instruction_idx += 1;
    }

    // Build live ranges
    let mut live_ranges: Vec<LiveRange> = first_use
        .iter()
        .map(|(&local_id, &first)| LiveRange {
            local_id,
            first_use: first,
            last_use: *last_use.get(&local_id).unwrap_or(&first),
        })
        .collect();

    live_ranges.sort_by_key(|lr| lr.first_use);

    // Compute max pressure using interval counting
    let num_instructions = instruction_idx;
    let max_pressure = compute_max_pressure(&live_ranges, num_instructions);

    Ok(FunctionAnalysis {
        func_idx,
        num_params,
        num_locals,
        total_locals,
        live_ranges,
        access_counts,
        max_pressure,
        num_instructions,
    })
}

fn compute_max_pressure(live_ranges: &[LiveRange], _num_instructions: u32) -> u32 {
    if live_ranges.is_empty() {
        return 0;
    }

    // Event-based sweep: +1 at start, -1 after end
    let mut events: Vec<(u32, i32)> = Vec::with_capacity(live_ranges.len() * 2);

    for lr in live_ranges {
        events.push((lr.first_use, 1));
        events.push((lr.last_use + 1, -1));
    }

    events.sort_by_key(|(pos, delta)| (*pos, -*delta)); // Process +1 before -1 at same position

    let mut current_pressure = 0i32;
    let mut max_pressure = 0i32;

    for (_pos, delta) in events {
        current_pressure += delta;
        max_pressure = max_pressure.max(current_pressure);
    }

    max_pressure as u32
}

/// Returns (num_spilled_vars, total_memory_ops)
/// where total_memory_ops = all accesses to spilled variables (each becomes a load/store)
fn estimate_spills_linear_scan(
    live_ranges: &[LiveRange],
    num_registers: u32,
    access_counts: &HashMap<u32, u32>,  // local_id -> number of accesses
) -> (u32, u32) {
    if live_ranges.is_empty() || num_registers == 0 {
        return (0, 0);
    }

    let mut sorted_ranges: Vec<&LiveRange> = live_ranges.iter().collect();
    sorted_ranges.sort_by_key(|lr| lr.first_use);

    // Active set: (end_pos, local_id)
    let mut active: Vec<(u32, u32)> = Vec::new();
    let mut spilled: HashSet<u32> = HashSet::new();

    for lr in sorted_ranges {
        // Expire old ranges
        active.retain(|(end, _)| *end >= lr.first_use);

        if active.len() >= num_registers as usize {
            // Need to spill - pick the one with furthest end point
            if let Some((idx, _)) = active
                .iter()
                .enumerate()
                .max_by_key(|(_, (end, _))| *end)
            {
                let (candidate_end, candidate_id) = active[idx];
                if candidate_end > lr.last_use {
                    // Spill the candidate
                    active.remove(idx);
                    spilled.insert(candidate_id);
                    active.push((lr.last_use, lr.local_id));
                } else {
                    // Spill current
                    spilled.insert(lr.local_id);
                }
            }
        } else {
            active.push((lr.last_use, lr.local_id));
        }
    }

    // Count total memory ops: every access to a spilled variable becomes a memory op
    let num_spilled_vars = spilled.len() as u32;
    let total_memory_ops: u32 = spilled
        .iter()
        .map(|id| access_counts.get(id).copied().unwrap_or(0))
        .sum();

    (num_spilled_vars, total_memory_ops)
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let wasm_path = args.get(1).map(|s| s.as_str()).unwrap_or("../target/wasm32-unknown-unknown/release/json_wasm_bench.wasm");

    let wasm_bytes = fs::read(wasm_path).context("Failed to read WASM file")?;

    println!("Analyzing {}...", wasm_path);
    println!("File size: {} bytes", wasm_bytes.len());

    let mut func_types: Vec<wasmparser::FuncType> = Vec::new();
    let mut type_indices: Vec<u32> = Vec::new();
    let mut functions: Vec<FunctionAnalysis> = Vec::new();

    for payload in Parser::new(0).parse_all(&wasm_bytes) {
        let payload = payload?;

        match &payload {
            Payload::TypeSection(reader) => {
                for rec_group in reader.clone() {
                    let rec_group = rec_group?;
                    for sub_type in rec_group.types() {
                        if let wasmparser::CompositeInnerType::Func(ft) = &sub_type.composite_type.inner {
                            func_types.push(ft.clone());
                        }
                    }
                }
            }
            Payload::FunctionSection(reader) => {
                for func in reader.clone() {
                    type_indices.push(func?);
                }
            }
            Payload::CodeSectionEntry(body) => {
                let func_idx = functions.len() as u32;
                let type_idx = type_indices.get(func_idx as usize).copied().unwrap_or(0);
                let num_params = func_types
                    .get(type_idx as usize)
                    .map(|ft| ft.params().len() as u32)
                    .unwrap_or(0);

                match analyze_function_body(func_idx, body.clone(), num_params) {
                    Ok(analysis) => functions.push(analysis),
                    Err(e) => eprintln!("Warning: Failed to analyze function {}: {}", func_idx, e),
                }
            }
            _ => {}
        }
    }

    // Overall stats
    let total_locals: u32 = functions.iter().map(|f| f.total_locals).sum();
    let max_pressure_overall = functions.iter().map(|f| f.max_pressure).max().unwrap_or(0);
    let total_instructions: u32 = functions.iter().map(|f| f.num_instructions).sum();

    println!("\n=== Overall Stats ===");
    println!("Functions: {}", functions.len());
    println!("Total instructions: {}", total_instructions);
    println!("Total locals across all functions: {}", total_locals);
    println!("Max register pressure (any function): {}", max_pressure_overall);

    // Top functions by pressure
    println!("\n=== Top 10 Functions by Register Pressure ===");
    let mut sorted_funcs: Vec<&FunctionAnalysis> = functions.iter().collect();
    sorted_funcs.sort_by_key(|f| std::cmp::Reverse(f.max_pressure));

    for f in sorted_funcs.iter().take(10) {
        println!(
            "  func[{:3}] pressure={:3} locals={:3} instructions={:5}",
            f.func_idx, f.max_pressure, f.total_locals, f.num_instructions
        );
    }

    // Spill analysis
    println!("\n=== Spill Estimate by Register Count ===");
    println!("{:<10} {:>15} {:>15} {:>18}", "Registers", "Spilled Vars", "Memory Ops", "% of Local Ops");
    println!("{}", "-".repeat(65));

    // Count total local ops across all functions
    let total_local_ops: u32 = functions.iter().map(|f| f.access_counts.values().sum::<u32>()).sum();

    for num_regs in [8, 9, 10, 11, 12, 13, 14, 15, 16] {
        let mut total_spilled_vars = 0u32;
        let mut total_memory_ops = 0u32;

        for f in &functions {
            let (spilled_vars, memory_ops) = estimate_spills_linear_scan(&f.live_ranges, num_regs, &f.access_counts);
            total_spilled_vars += spilled_vars;
            total_memory_ops += memory_ops;
        }

        let pct = if total_local_ops > 0 {
            (total_memory_ops as f64 / total_local_ops as f64) * 100.0
        } else {
            0.0
        };

        println!(
            "{:<10} {:>15} {:>15} {:>17.1}%",
            num_regs, total_spilled_vars, total_memory_ops, pct
        );
    }

    println!("\nTotal local.get/set/tee ops: {}", total_local_ops);

    // Find exact minimum
    println!("\n=== Minimum Registers for Zero Spills ===");
    for num_regs in 1..=100 {
        let total_spills: u32 = functions
            .iter()
            .map(|f| estimate_spills_linear_scan(&f.live_ranges, num_regs, &f.access_counts).0)
            .sum();

        if total_spills == 0 {
            println!("Need {} registers for zero spills", num_regs);
            break;
        }
    }

    // Distribution of pressure
    println!("\n=== Register Pressure Distribution ===");
    let mut pressure_hist: HashMap<u32, u32> = HashMap::new();
    for f in &functions {
        *pressure_hist.entry(f.max_pressure).or_insert(0) += 1;
    }

    let mut pressures: Vec<_> = pressure_hist.into_iter().collect();
    pressures.sort_by_key(|(p, _)| *p);

    println!("{:<10} {:>10} {:>10}", "Pressure", "# Funcs", "Cumulative");
    let mut cumulative = 0u32;
    for (pressure, count) in pressures.iter().rev() {
        cumulative += count;
        println!("{:<10} {:>10} {:>10}", pressure, count, cumulative);
    }

    Ok(())
}
