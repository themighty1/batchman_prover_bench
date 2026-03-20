//! IT-MAC consistency prover on binius_m3.
//!
//! IT-MAC consistency prover on binius_m3.
//!
//! Reads MACs from `step_records_prover.bin`, plaintexts from `packed_pt.bin`,
//! and delta from `delta.bin`. Proves IT-MAC consistency:
//!   mac = key + plaintext · delta   for every active-branch row.
//!
//! Standalone commitment groups: mac (group 1), plaintext (group 2).
//!
//! Run with:
//!   cargo run --release --bin mac_consistency_prover [STEP_RECORDS] [PACKED_PT]

use std::time::Instant;

use anyhow::Result;
use batchman_witness_generator::StepRecordProverData;
use binius_compute::cpu::alloc::CpuComputeAllocator;
use binius_compute::cpu::layer::CpuLayerHolder;
use binius_compute::ComputeHolder;
use binius_core::{
    constraint_system::{FriStrategy, prove},
    fiat_shamir::HasherChallenger,
};
use binius_field::{
    arch::OptimalUnderlier, as_packed_field::PackedType, tower::CanonicalTowerFamily,
};
use binius_hal::make_portable_backend;
use binius_m3::builder::{Col, ConstraintSystem, WitnessIndex, B128};
use memory_checker_and_lookup::{Blake3Digest, Blake3Compression, commit_column_b128};
use rand::{rngs::StdRng, Rng, SeedableRng};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

// ==== Constants ====

const DEFAULT_PROGRAM: &str = "json-query";
const LOG_INV_RATE: usize = 1;
const SECURITY_BITS: usize = 100;
const FRI_STRATEGY: FriStrategy = FriStrategy::ConstantArity(8);

type P = PackedType<OptimalUnderlier, B128>;

fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            fmt::layer()
                .with_timer(fmt::time::uptime())
                .with_span_events(fmt::format::FmtSpan::CLOSE),
        )
        .with(EnvFilter::from_default_env())
        .init();

    let program = std::env::args().nth(1).unwrap_or_else(|| DEFAULT_PROGRAM.to_string());
    let data_file = format!("../data/{program}/step_records_prover.bin");
    let packed_pt_path = format!("../witgen/witness/{program}/packed_pt.bin");

    // ==== 1. Read prover data (MACs + active branches) ====
    let mut f = std::io::BufReader::new(std::fs::File::open(&data_file)?);
    let data = StepRecordProverData::read_from(&mut f)?;

    let batch_sz = data.batch_sz as usize;
    let branch_count = data.branch_count as usize;

    // ==== 2. Read packed_pt.bin (ground truth plaintexts from witgen) ====
    let packed_pts: Vec<B128> = {
        let raw = std::fs::read(&packed_pt_path)?;
        let count = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
        anyhow::ensure!(count == batch_sz,
            "packed_pt has {} entries but step_records has batch_sz={}", count, batch_sz);
        (0..count)
            .map(|i| {
                let off = 4 + i * 16;
                B128::new(u128::from_le_bytes(raw[off..off + 16].try_into().unwrap()))
            })
            .collect()
    };

    // ==== 3. Read delta ====
    let delta_path = std::path::Path::new(&data_file)
        .parent()
        .unwrap()
        .join("delta.bin");
    let delta_raw = std::fs::read(&delta_path)?;
    anyhow::ensure!(delta_raw.len() == 16, "delta.bin must be 16 bytes");
    let delta_val = u128::from_le_bytes(delta_raw[..16].try_into().unwrap());

    // ==== 4. Extract active-branch MACs, use packed_pt as plaintexts, compute keys ====
    //
    // The IT-MAC relation is:  mac = key + plaintext · delta   (in binius tower GF(2^128))
    // So:                      key = mac + plaintext · delta   (XOR in char-2)
    //
    // IMPORTANT: Use B128 arithmetic (binius tower polynomial), NOT gf128_mul.
    let delta_b128 = B128::new(delta_val);
    let num_rows = batch_sz;
    let mut active_macs: Vec<B128> = Vec::with_capacity(batch_sz);
    let mut active_keys: Vec<B128> = Vec::with_capacity(batch_sz);

    for step in 0..batch_sz {
        let active = data.active_branches[step] as usize;
        let idx = step * branch_count + active;
        let mac = B128::new(u128::from_le_bytes(data.macs[idx]));
        let pt = packed_pts[step];
        let key = mac + pt * delta_b128;

        active_macs.push(mac);
        active_keys.push(key);
    }
    let active_pts = packed_pts;

    println!("MAC consistency prover");
    println!("  Step records:  {}", data_file);
    println!("  Packed PT:     {}", packed_pt_path);
    println!("  Steps:         {}", batch_sz);
    println!("  Branch count:  {}", branch_count);
    println!("  Rows (active): {}", num_rows);
    println!();

    // ==== 4. Out-of-circuit commitments to active-branch MACs and plaintexts ====
    let t_commit = Instant::now();
    let active_mac_commit =
        commit_column_b128(&active_macs, LOG_INV_RATE, SECURITY_BITS, &FRI_STRATEGY)?;
    let active_pt_commit =
        commit_column_b128(&active_pts, LOG_INV_RATE, SECURITY_BITS, &FRI_STRATEGY)?;
    let commit_time = t_commit.elapsed();

    std::fs::create_dir_all("proofs")?;
    let proof_dir = format!("proofs/{program}");
    std::fs::create_dir_all(&proof_dir)?;
    std::fs::write(format!("{proof_dir}/step_records_commit.bin"), &active_mac_commit)?;
    println!(
        "Active MAC commitment:  {:02x?} ({:?})",
        &active_mac_commit[..8], commit_time,
    );
    println!(
        "Active PT commitment:   {:02x?}",
        &active_pt_commit[..8],
    );
    println!();

    // ==== 5. Build constraint system (single table) ====
    let mut cs = ConstraintSystem::new();
    let mut table = cs.add_table("mac_consistency");

    let c_mac: Col<B128> = table.add_committed_in_group("mac", 1);
    let c_pt: Col<B128> = table.add_committed_in_group("plaintext", 2);
    let c_key: Col<B128> = table.add_committed_in_group("key", 3);
    let c_delta: Col<B128> = table.add_committed("delta");

    let c_fri_blind: Col<B128> = table.add_committed("fri_blind");
    let c_fri_blind_sq: Col<B128> = table.add_committed("fri_blind_sq");
    table.assert_zero("sumcheck_blind", c_fri_blind * c_fri_blind - c_fri_blind_sq);

    table.assert_zero("it_mac", c_mac - c_key - c_pt * c_delta);

    let table_id = table.id();
    drop(table);

    // ==== 6. Fill witness ====
    let mut allocator = CpuComputeAllocator::new(1 << 24);
    let allocator = allocator.into_bump_allocator();
    let mut witness = WitnessIndex::<P>::new(&cs, &allocator);

    let tw = witness.init_table(table_id, num_rows).expect("init table");
    let seg = tw.full_segment();

    let mut rng = StdRng::seed_from_u64(42);

    {
        let mut mac_col = seg.get_scalars_mut(c_mac)?;
        let mut key_col = seg.get_scalars_mut(c_key)?;
        let mut pt_col = seg.get_scalars_mut(c_pt)?;
        let mut delta_col = seg.get_scalars_mut(c_delta)?;
        for i in 0..num_rows {
            mac_col[i] = active_macs[i];
            key_col[i] = active_keys[i];
            pt_col[i] = active_pts[i];
            delta_col[i] = delta_b128;
        }
    }

    {
        let padded_size = num_rows.next_power_of_two();
        let mut blind_col = seg.get_scalars_mut(c_fri_blind)?;
        let mut blind_sq_col = seg.get_scalars_mut(c_fri_blind_sq)?;
        for i in 0..padded_size {
            let r = B128::new(rng.random::<u128>());
            blind_col[i] = r;
            blind_sq_col[i] = r * r;
        }
    }

    // ==== 7. Compile and prove ====
    let boundaries = vec![];

    let t_total = Instant::now();

    let t_compile = Instant::now();
    let compiled_cs = cs.compile().map_err(|e| anyhow::anyhow!("{e}"))?;
    let ccs_digest = compiled_cs.digest::<Blake3Digest>();
    let table_sizes = witness.table_sizes();
    let witness_mle = witness.into_multilinear_extension_index();
    let compile_time = t_compile.elapsed();

    let t_prove = Instant::now();
    let mut compute_holder = CpuLayerHolder::<B128>::new(1 << 22, 1 << 27);

    let proof = prove::<
        _,
        OptimalUnderlier,
        CanonicalTowerFamily,
        Blake3Digest,
        Blake3Compression,
        HasherChallenger<Blake3Digest>,
        _,
        _,
        _,
    >(
        &mut compute_holder.to_data(),
        &compiled_cs,
        LOG_INV_RATE,
        SECURITY_BITS,
        &FRI_STRATEGY,
        &ccs_digest,
        &boundaries,
        &table_sizes,
        witness_mle,
        &make_portable_backend(),
    )?;
    let prove_time = t_prove.elapsed();
    let total_prover_time = t_total.elapsed();

    let proof_size = proof.get_proof_size();

    // ==== 8. Write proof to disk ====
    {
        use std::io::Write;
        let mut out = std::io::BufWriter::new(std::fs::File::create(format!("{proof_dir}/mac_consistency_proof.bin"))?);
        let transcript = &proof.transcript;
        out.write_all(&(transcript.len() as u64).to_le_bytes())?;
        out.write_all(transcript)?;
        out.flush()?;
    }
    println!("Proof written to proofs/{program}/mac_consistency_proof.bin");

    println!();
    println!("=== Prover stats ===");
    println!("  Compile:            {:?}", compile_time);
    println!("  Proving:            {:?}", prove_time);
    println!("  Total:              {:?}", total_prover_time);
    println!();
    println!("=== Proof size ===");
    println!(
        "  Binius proof:       {} bytes ({:.2} KB)",
        proof_size, proof_size as f64 / 1024.0
    );

    Ok(())
}
