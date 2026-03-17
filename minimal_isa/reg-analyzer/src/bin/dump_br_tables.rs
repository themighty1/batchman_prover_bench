//! Dump all BrTable instructions from the WASM JSON parser

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use wasmparser::{Parser, Payload};
use reg_analyzer::regvm::{WasmToVReg, VRegInst, FuncSig};

fn main() -> Result<()> {
    let wasm_path = "../json-crate-wasm/target/wasm32-unknown-unknown/release/json_crate_wasm.wasm";
    let wasm_bytes = fs::read(wasm_path).context("Failed to read WASM file")?;

    let mut func_types: Vec<wasmparser::FuncType> = Vec::new();
    let mut type_indices: Vec<u32> = Vec::new();
    let mut func_names: HashMap<u32, String> = HashMap::new();

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
                for func in reader.clone() { type_indices.push(func?); }
            }
            Payload::ExportSection(reader) => {
                for export in reader.clone() {
                    let export = export?;
                    if let wasmparser::ExternalKind::Func = export.kind {
                        func_names.insert(export.index, export.name.to_string());
                    }
                }
            }
            _ => {}
        }
    }

    let func_sigs: Vec<FuncSig> = type_indices.iter().map(|&type_idx| {
        let func_type = func_types.get(type_idx as usize);
        let num_params = func_type.map(|ft| ft.params().len() as u32).unwrap_or(0);
        let num_results = func_type.map(|ft| ft.results().len() as u32).unwrap_or(0);
        (num_params, num_results)
    }).collect();

    let mut func_count = 0u32;
    for payload in Parser::new(0).parse_all(&wasm_bytes) {
        let payload = payload?;
        if let Payload::CodeSectionEntry(body) = &payload {
            let type_idx = type_indices.get(func_count as usize).copied().unwrap_or(0);
            let func_type = func_types.get(type_idx as usize);
            let num_params = func_type.map(|ft| ft.params().len() as u32).unwrap_or(0);
            let mut num_locals = 0u32;
            for local in body.get_locals_reader()? {
                let (count, _) = local?;
                num_locals += count;
            }

            let mut converter = WasmToVReg::new_with_sigs(num_params, num_locals, func_sigs.clone());
            for op in body.get_operators_reader()? {
                converter.convert_op(&op?);
            }

            let fname = func_names.get(&func_count)
                .cloned()
                .unwrap_or_else(|| format!("func_{}", func_count));

            for (i, inst) in converter.instructions.iter().enumerate() {
                if let VRegInst::BrTable { idx, labels, default } = inst {
                    println!("{}  [pc={}]", fname, i);
                    println!("  br_table {:?}, labels={:?}, default={}", idx, labels, default);
                    println!("  {} entries + 1 default\n", labels.len());
                }
            }

            func_count += 1;
        }
    }

    Ok(())
}
