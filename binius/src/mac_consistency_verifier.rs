//! IT-MAC consistency verifier on binius_m3.
//!
//! Reads the proof written by mac_consistency_prover, rebuilds the constraint
//! system (single table: mac, key, plaintext, delta), and verifies.
//!
//! Run with:
//!   cargo run --release --bin mac_consistency_verifier [PROOF_FILE]

use std::time::Instant;

use anyhow::Result;
use binius_core::{
    constraint_system::{FriStrategy, verify, Proof},
    fiat_shamir::HasherChallenger,
};
use binius_field::{arch::OptimalUnderlier, tower::CanonicalTowerFamily};
use binius_m3::builder::{Col, ConstraintSystem, B128};
use memory_checker_and_lookup::{Blake3Digest, Blake3Compression};

// ==== Constants (must match prover) ====

const LOG_INV_RATE: usize = 1;
const SECURITY_BITS: usize = 100;
const FRI_STRATEGY: FriStrategy = FriStrategy::ConstantArity(8);

fn main() -> Result<()> {
    let program = std::env::args().nth(1).unwrap_or_else(|| "json-query".to_string());
    let proof_path = format!("proofs/{program}/mac_consistency_proof.bin");

    // ==== 1. Read proof ====
    let t_total = Instant::now();

    let raw = std::fs::read(&proof_path)?;
    let mut cursor = &raw[..];

    let transcript_len = {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&cursor[..8]);
        cursor = &cursor[8..];
        u64::from_le_bytes(buf) as usize
    };
    let transcript = cursor[..transcript_len].to_vec();

    println!("MAC consistency verifier");
    println!("  Proof file:    {}", proof_path);
    println!("  Transcript:    {} bytes", transcript_len);
    println!();

    // ==== 2. Rebuild constraint system (must match prover exactly) ====
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

    drop(table);

    // ==== 3. Compile and verify ====
    let boundaries = vec![];

    let t0 = Instant::now();
    let compiled_cs = cs.compile().map_err(|e| anyhow::anyhow!("{e}"))?;
    let compile_time = t0.elapsed();

    let ccs_digest = compiled_cs.digest::<Blake3Digest>();
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
    let verify_time = t0.elapsed();

    let total_time = t_total.elapsed();

    // ==== 4. Print standalone commitments ====
    println!("Standalone commitments:");
    for &(group, label) in &[(1u32, "MAC"), (2u32, "Plaintext"), (3u32, "Key")] {
        if let Some(commit) = standalone_commitments.get(&group) {
            println!("  {} (group {}): {:02x?}", label, group, commit.as_slice());
        }
    }
    println!();

    println!("=== Verifier stats ===");
    println!("  CS compile:     {:?}", compile_time);
    println!("  Proof verify:   {:?}", verify_time);
    println!("  Total:          {:?}", total_time);
    println!();

    println!("Proof verified successfully!");

    Ok(())
}
