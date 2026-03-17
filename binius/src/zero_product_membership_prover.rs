//! Zero-product prover.
//!
//! Reads topology MACs from a binary file and proves that the active branch
//! MACs are zero via a binius channel membership proof.
//!
//! Run with:
//!   cargo run --release --bin zero_product_membership_prover [PROGRAM]

use std::time::Instant;

use anyhow::Result;
use batchman_witness_generator::ZeroProductProverData;
use memory_checker_and_lookup::commit_column_b64;
use binius_compute::cpu::alloc::CpuComputeAllocator;
use binius_compute::cpu::layer::CpuLayerHolder;
use binius_compute::ComputeHolder;
use binius_core::{
    constraint_system::{FriStrategy, prove},
    fiat_shamir::HasherChallenger,
};
use binius_field::{
    arch::OptimalUnderlier, as_packed_field::PackedType, tower::CanonicalTowerFamily,
    PackedExtension, PackedField,
};
use binius_hal::make_portable_backend;
use binius_hash::compression::{CompressionFunction, PseudoCompressionFunction};
use binius_m3::builder::{
    upcast_col, Col, ConstraintSystem, FlushOpts, WitnessIndex, B1, B32, B64, B128,
};
use digest::{
    core_api::BlockSizeUser,
    consts::{U32, U64},
    FixedOutput, FixedOutputReset, HashMarker, OutputSizeUser, Reset, Update,
};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

// ==== Blake3 wrapper for binius digest traits ====

#[derive(Clone)]
struct Blake3Digest {
    hasher: blake3::Hasher,
}

impl Default for Blake3Digest {
    fn default() -> Self {
        Self {
            hasher: blake3::Hasher::new(),
        }
    }
}

impl HashMarker for Blake3Digest {}

impl Update for Blake3Digest {
    fn update(&mut self, data: &[u8]) {
        self.hasher.update(data);
    }
}

impl Reset for Blake3Digest {
    fn reset(&mut self) {
        self.hasher = blake3::Hasher::new();
    }
}

impl OutputSizeUser for Blake3Digest {
    type OutputSize = U32;
}

impl BlockSizeUser for Blake3Digest {
    type BlockSize = U64;
}

impl FixedOutput for Blake3Digest {
    fn finalize_into(self, out: &mut digest::Output<Self>) {
        let hash = self.hasher.finalize();
        out.copy_from_slice(hash.as_bytes());
    }
}

impl FixedOutputReset for Blake3Digest {
    fn finalize_into_reset(&mut self, out: &mut digest::Output<Self>) {
        let hash = self.hasher.finalize();
        out.copy_from_slice(hash.as_bytes());
        self.hasher = blake3::Hasher::new();
    }
}

#[derive(Debug, Default, Clone)]
struct Blake3Compression;

impl PseudoCompressionFunction<digest::Output<Blake3Digest>, 2> for Blake3Compression {
    fn compress(&self, input: [digest::Output<Blake3Digest>; 2]) -> digest::Output<Blake3Digest> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(input[0].as_slice());
        hasher.update(input[1].as_slice());
        let hash = hasher.finalize();
        let mut out = digest::Output::<Blake3Digest>::default();
        out.copy_from_slice(hash.as_bytes());
        out
    }
}

impl CompressionFunction<digest::Output<Blake3Digest>, 2> for Blake3Compression {}

// ==== GF(2^64) with x^64 + x^4 + x^3 + x + 1 ====

/// Multiply in GF(2^64) using irreducible polynomial x^64 + x^4 + x^3 + x + 1.
/// Matches EMP-ZK's CLMUL + reduce64 implementation exactly:
/// 1. Full 128-bit carry-less multiply
/// 2. Barrett reduction via 64-bit shifts (overflow bits truncated, matching
///    _mm_slli_epi64 semantics in emp-zk/emp-zk-bool/batched_disjunction.h).
fn gf64_mul(a: u64, b: u64) -> u64 {
    // Step 1: full 128-bit carry-less multiply
    let mut product: u128 = 0;
    for i in 0..64 {
        if (b >> i) & 1 != 0 {
            product ^= (a as u128) << i;
        }
    }
    // Step 2: reduce64 — matches stacking-vole-zk exactly
    let lo = product as u64;
    let hi = (product >> 64) as u64;
    // _mm_slli_epi64 truncates overflow within the 64-bit lane
    let r1 = hi << 4; // hi * x^4, overflow lost
    let r2 = hi << 3; // hi * x^3, overflow lost
    let r3 = hi << 1; // hi * x^1
    let reduced = r1 ^ r2 ^ r3 ^ hi;
    lo ^ reduced
}

// ==== Constants ====

const DEFAULT_PROGRAM: &str = "json-query";
const LOG_INV_RATE: usize = 1;
const SECURITY_BITS: usize = 100;
const FRI_STRATEGY: FriStrategy = FriStrategy::ConstantArity(8);

type P = PackedType<OptimalUnderlier, B128>;
type PackedB1 = <P as PackedExtension<B1>>::PackedSubfield;

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
    let data_file = format!("../data/{program}/zero_product_prover.bin");

    // ==== Read prover data ====
    let mut f = std::io::BufReader::new(std::fs::File::open(&data_file)?);
    let data = ZeroProductProverData::read_from(&mut f)?;

    let num_selected = data.batch_sz as usize;
    let branch_count = data.branch_count as usize;
    let num_rows = num_selected * branch_count;

    // Build active MACs (B64) for out-of-circuit commitment.
    let mut selected_macs: Vec<B64> = Vec::with_capacity(num_selected);
    let mut selected_set: Vec<bool> = vec![false; num_rows];

    for step in 0..num_selected {
        let active = data.active_branches[step] as usize;
        for bid in 0..branch_count {
            if bid == active {
                selected_set[step * branch_count + bid] = true;
                let mac_bytes = data.topology_macs[step * branch_count + bid];
                selected_macs.push(B64::new(u64::from_le_bytes(mac_bytes)));
            }
        }
    }

    println!("Zero-product prover");
    println!("  Data file:     {}", data_file);
    println!("  Steps:         {}", num_selected);
    println!("  Branch count:  {}", branch_count);
    println!("  Source rows:   {}", num_rows);
    println!("  Selected:      {}", num_selected);
    println!();

    // ==== Out-of-circuit commitment to active MACs column ====
    let t_commit = Instant::now();
    let active_mac_commitment =
        commit_column_b64(&selected_macs, LOG_INV_RATE, SECURITY_BITS, &FRI_STRATEGY)?;
    let commit_time = t_commit.elapsed();

    std::fs::create_dir_all("proofs")?;
    let proof_dir = format!("proofs/{program}");
    std::fs::create_dir_all(&proof_dir)?;
    std::fs::write(format!("{proof_dir}/zero_product_commit.bin"), &active_mac_commitment)?;
    println!(
        "Active MACs commitment written to proofs/{program}/zero_product_commit.bin ({} bytes, {:?})",
        active_mac_commitment.len(),
        commit_time,
    );
    println!();

    // ==== Read delta (lower 64 bits) and compute keys: key = mac + delta * plaintext ====
    let delta_path = std::path::Path::new(&data_file)
        .parent()
        .unwrap()
        .join("delta.bin");
    let delta_raw = std::fs::read(&delta_path)?;
    anyhow::ensure!(delta_raw.len() == 16, "delta.bin must be 16 bytes");
    let delta_val = u64::from_le_bytes(delta_raw[..8].try_into().unwrap());

    let mut all_keys: Vec<B64> = Vec::with_capacity(num_rows);

    // key = mac XOR gf64_mul(delta, plaintext) in GF(2^64) with x^64+x^4+x^3+x+1.
    // Active branches have plaintext=0, so key = mac for those rows.
    for step in 0..num_selected {
        for bid in 0..branch_count {
            let idx = step * branch_count + bid;
            let mac_val = u64::from_le_bytes(data.topology_macs[idx]);
            let pt_val = u64::from_le_bytes(data.topology_plaintexts[idx]);
            let key_val = mac_val ^ gf64_mul(delta_val, pt_val);
            all_keys.push(B64::new(key_val));
        }
    }

    // ==== Build constraint system ====
    let mut cs = ConstraintSystem::new();
    let chan = cs.add_channel("selected");

    // ---- Source table: key + chunk_id + selector, push when selected ----
    let mut src_tab = cs.add_table("source");
    let src_val: Col<B64> = src_tab.add_committed_in_group("key", 1);
    let src_chunk_id: Col<B32> = src_tab.add_committed_in_group("chunk_id", 3);
    let src_sel: Col<B1> = src_tab.add_committed("sel");
    src_tab.push_with_opts(
        chan,
        [upcast_col::<B128, _, 1>(src_val), upcast_col::<B128, _, 1>(src_chunk_id)],
        FlushOpts { selectors: vec![src_sel], ..Default::default() },
    );
    let src_id = src_tab.id();
    drop(src_tab);

    // ---- Receiver table: pulls the active MACs with chunk_id ----
    // Active branches have plaintext=0, so key = mac + delta*0 = mac.
    let mut rcv_tab = cs.add_table("receiver");
    let rcv_val: Col<B64> = rcv_tab.add_committed_in_group("mac", 2);
    let rcv_chunk_id: Col<B32> = rcv_tab.add_committed_in_group("chunk_id", 4);
    rcv_tab.pull(chan, [upcast_col::<B128, _, 1>(rcv_val), upcast_col::<B128, _, 1>(rcv_chunk_id)]);
    let rcv_id = rcv_tab.id();
    drop(rcv_tab);

    // ==== Fill witnesses ====
    let mut allocator = CpuComputeAllocator::new(1 << 26);
    let allocator = allocator.into_bump_allocator();
    let mut witness = WitnessIndex::<P>::new(&cs, &allocator);

    // Fill source table with all keys (B64).
    {
        let tw = witness.init_table(src_id, num_rows).expect("init source");
        let seg = tw.full_segment();
        {
            let mut val_col = seg.get_scalars_mut(src_val)?;
            let mut cid_col = seg.get_scalars_mut(src_chunk_id)?;
            for (i, &k) in all_keys.iter().enumerate() {
                val_col[i] = k;
                cid_col[i] = B32::new((i / branch_count) as u32);
            }
        }
        {
            let mut acc = seg.get_mut(src_sel)?;
            for (idx, packed) in acc.iter_mut().enumerate() {
                for k in 0..PackedB1::WIDTH {
                    let row = idx * PackedB1::WIDTH + k;
                    if row < num_rows {
                        packed.set(k, B1::from(selected_set[row]));
                    }
                }
            }
        }
    }

    // Fill receiver table with active MACs (B64).
    {
        let tw = witness.init_table(rcv_id, num_selected).expect("init receiver");
        let seg = tw.full_segment();
        {
            let mut val_col = seg.get_scalars_mut(rcv_val)?;
            let mut cid_col = seg.get_scalars_mut(rcv_chunk_id)?;
            for (i, &m) in selected_macs.iter().enumerate() {
                val_col[i] = m;
                cid_col[i] = B32::new(i as u32);
            }
        }
    }

    // ==== Compile and prove ====
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

    // ==== Write proof to disk (sent to verifier) ====
    std::fs::write(format!("{proof_dir}/zero_product_membership_proof.bin"), &proof.transcript)?;
    println!("Proof written to proofs/{program}/zero_product_membership_proof.bin ({} bytes) — sent to verifier", proof_size);
    println!();

    println!("=== Prover stats ===");
    println!("  Compile:            {:?}", compile_time);
    println!("  Proving:            {:?}", prove_time);
    println!("  Total:              {:?}", total_prover_time);
    println!();
    println!("=== Proof size ===");
    println!(
        "  Total:              {} bytes ({:.2} KB)",
        proof_size, proof_size as f64 / 1024.0
    );

    Ok(())
}
