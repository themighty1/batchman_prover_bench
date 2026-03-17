//! Zero-product verifier.
//!
//! Reads verifier keys, commits to them, then verifies the prover's binius
//! proof and checks that both the keys commitment and MAC commitment appear
//! in the proof's standalone commitments.
//!
//! Run with:
//!   cargo run --release --bin zero_product_membership_verifier

use std::time::Instant;

use anyhow::Result;
use batchman_witness_generator::ZeroProductVerifierData;
use memory_checker_and_lookup::{commit_column_b32, commit_column_b64};
use binius_core::{
    constraint_system::{FriStrategy, verify, Proof},
    fiat_shamir::HasherChallenger,
};
use binius_field::{arch::OptimalUnderlier, tower::CanonicalTowerFamily};
use binius_hash::compression::{CompressionFunction, PseudoCompressionFunction};
use binius_m3::builder::{
    upcast_col, Col, ConstraintSystem, FlushOpts, B1, B32, B64, B128,
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

// ==== Constants (must match prover) ====

const DEFAULT_PROGRAM: &str = "json-query";
const LOG_INV_RATE: usize = 1;
const SECURITY_BITS: usize = 100;
const FRI_STRATEGY: FriStrategy = FriStrategy::ConstantArity(8);

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
    let data_dir_str = format!("../data/{program}");
    let data_dir = std::path::Path::new(&data_dir_str);

    // ==== 1. Read verifier keys ====
    let keys_path = data_dir.join("zero_product_verifier.bin");
    let mut f = std::io::BufReader::new(std::fs::File::open(&keys_path)?);
    let vdata = ZeroProductVerifierData::read_from(&mut f)?;

    let num_steps = vdata.batch_sz as usize;
    let branch_count = vdata.branch_count as usize;
    let num_rows = num_steps * branch_count;

    println!("Zero-product verifier");
    println!("  Keys file:     {}", keys_path.display());
    println!("  Steps:         {}", num_steps);
    println!("  Branch count:  {}", branch_count);
    println!("  Total keys:    {}", num_rows);
    println!();

    let t_total = Instant::now();

    // ==== 2. Commit to all keys out-of-circuit ====
    let all_keys: Vec<B64> = vdata
        .topology_keys
        .iter()
        .map(|k| B64::new(u64::from_le_bytes(*k)))
        .collect();

    let t0 = Instant::now();
    let keys_commitment = commit_column_b64(&all_keys, LOG_INV_RATE, SECURITY_BITS, &FRI_STRATEGY)?;
    let keys_commit_time = t0.elapsed();

    // ==== 2b. Commit to chunk_id columns out-of-circuit ====
    // Source chunk_id (group 3): step index for each row = row / branch_count.
    let src_chunk_ids: Vec<B32> = (0..num_rows)
        .map(|i| B32::new((i / branch_count) as u32))
        .collect();
    let t0 = Instant::now();
    let src_chunk_id_commitment =
        commit_column_b32(&src_chunk_ids, LOG_INV_RATE, SECURITY_BITS, &FRI_STRATEGY)?;
    let src_cid_commit_time = t0.elapsed();

    // Receiver chunk_id (group 4): sequential 0, 1, 2, ..., num_steps-1.
    let rcv_chunk_ids: Vec<B32> = (0..num_steps)
        .map(|i| B32::new(i as u32))
        .collect();
    let t0 = Instant::now();
    let rcv_chunk_id_commitment =
        commit_column_b32(&rcv_chunk_ids, LOG_INV_RATE, SECURITY_BITS, &FRI_STRATEGY)?;
    let rcv_cid_commit_time = t0.elapsed();

    // ==== 3. Read prover's out-of-circuit MAC commitment ====
    let proof_dir = format!("proofs/{program}");
    let mac_commitment = std::fs::read(format!("{proof_dir}/zero_product_commit.bin"))?;

    // ==== 4. Read proof and verify ====
    let transcript = std::fs::read(format!("{proof_dir}/zero_product_membership_proof.bin"))?;

    // Rebuild constraint system (must match prover exactly).
    let mut cs = ConstraintSystem::new();
    let chan = cs.add_channel("selected");

    let mut src_tab = cs.add_table("source");
    let src_val: Col<B64> = src_tab.add_committed_in_group("key", 1);
    let src_chunk_id: Col<B32> = src_tab.add_committed_in_group("chunk_id", 3);
    let src_sel: Col<B1> = src_tab.add_committed("sel");
    src_tab.push_with_opts(
        chan,
        [upcast_col::<B128, _, 1>(src_val), upcast_col::<B128, _, 1>(src_chunk_id)],
        FlushOpts { selectors: vec![src_sel], ..Default::default() },
    );
    drop(src_tab);

    // Active branches have plaintext=0, so key = mac for those rows.
    let mut rcv_tab = cs.add_table("receiver");
    let rcv_val: Col<B64> = rcv_tab.add_committed_in_group("mac", 2);
    let rcv_chunk_id: Col<B32> = rcv_tab.add_committed_in_group("chunk_id", 4);
    rcv_tab.pull(chan, [upcast_col::<B128, _, 1>(rcv_val), upcast_col::<B128, _, 1>(rcv_chunk_id)]);
    drop(rcv_tab);

    let t0 = Instant::now();
    let compiled_cs = cs.compile().map_err(|e| anyhow::anyhow!("{e}"))?;
    let compile_time = t0.elapsed();

    let ccs_digest = compiled_cs.digest::<Blake3Digest>();
    let boundaries = vec![];
    let proof = Proof { transcript };

    let t0 = Instant::now();
    let standalone_commitments = verify::<
        OptimalUnderlier,
        CanonicalTowerFamily,
        Blake3Digest,
        Blake3Compression,
        HasherChallenger<Blake3Digest>,
    >(
        &compiled_cs,
        LOG_INV_RATE,
        SECURITY_BITS,
        &FRI_STRATEGY,
        &ccs_digest,
        &boundaries,
        proof,
    )?;
    let proof_verify_time = t0.elapsed();

    // ==== 5. Check commitments match ====
    let proof_keys_commit = standalone_commitments
        .get(&1)
        .ok_or_else(|| anyhow::anyhow!("group 1 (source keys) missing from proof"))?;
    let proof_macs_commit = standalone_commitments
        .get(&2)
        .ok_or_else(|| anyhow::anyhow!("group 2 (receiver MACs) missing from proof"))?;
    let proof_src_cid_commit = standalone_commitments
        .get(&3)
        .ok_or_else(|| anyhow::anyhow!("group 3 (source chunk_id) missing from proof"))?;
    let proof_rcv_cid_commit = standalone_commitments
        .get(&4)
        .ok_or_else(|| anyhow::anyhow!("group 4 (receiver chunk_id) missing from proof"))?;

    let keys_match = proof_keys_commit.as_slice() == keys_commitment.as_slice();
    let macs_match = proof_macs_commit.as_slice() == mac_commitment.as_slice();
    let src_cid_match = proof_src_cid_commit.as_slice() == src_chunk_id_commitment.as_slice();
    let rcv_cid_match = proof_rcv_cid_commit.as_slice() == rcv_chunk_id_commitment.as_slice();

    let total_time = t_total.elapsed();

    println!("=== Verifier stats ===");
    println!("  Keys commit:        {:?}", keys_commit_time);
    println!("  Src chunk_id commit:{:?}", src_cid_commit_time);
    println!("  Rcv chunk_id commit:{:?}", rcv_cid_commit_time);
    println!("  CS compile:         {:?}", compile_time);
    println!("  Proof verify:       {:?}", proof_verify_time);
    println!("  Total:              {:?}", total_time);
    println!();
    println!("Commitment check:");
    println!("  Keys  (group 1):       {}", keys_match);
    println!("  MACs  (group 2):       {}", macs_match);
    println!("  Src chunk_id (group 3):{}", src_cid_match);
    println!("  Rcv chunk_id (group 4):{}", rcv_cid_match);
    println!();

    anyhow::ensure!(keys_match, "Keys commitment mismatch!");
    anyhow::ensure!(macs_match, "MACs commitment mismatch!");
    anyhow::ensure!(src_cid_match, "Source chunk_id commitment mismatch!");
    anyhow::ensure!(rcv_cid_match, "Receiver chunk_id commitment mismatch!");

    println!("All checks passed!");

    Ok(())
}
