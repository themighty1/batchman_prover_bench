// PCS subset proof benchmark using Binius FRI over binary tower fields.
//
// This implements the same split-commit subset proof protocol as main.rs but
// uses Binius's FRI over GF(2^128) (with GF(2^32) as the RS encoding subfield)
// instead of Plonky3's FRI over Goldilocks.
//
// Key differences from the Plonky3 version:
//   - Field: BinaryField128b (binary tower) vs Goldilocks (64-bit prime)
//   - NTT: Additive NTT over binary subspaces vs multiplicative FFT
//   - FRI: Variable fold arity, interleaved codes
//
// NOTE: This benchmark uses random messages as a proxy for actual polynomial
// coefficients. FRI commit/prove/verify timing is identical regardless of
// message content, so this accurately measures PCS performance.
//
// NOTE: Binius FRI proves degree bounds but does not natively provide evaluation
// proofs at arbitrary points (unlike Plonky3's TwoAdicFriPcs). A full protocol
// would additionally need sumcheck-based evaluation proofs. This benchmark
// measures the dominant FRI commit/prove/verify cost.

use std::iter::repeat_with;
use std::time::Instant;

use binius_compute::{ComputeHolder, cpu::layer::CpuLayerHolder};
use binius_core::{
    fiat_shamir::{CanSample, HasherChallenger},
    merkle_tree::{BinaryMerkleTreeProver, MerkleTreeProver, MerkleTreeScheme},
    protocols::fri::{
        CommitOutput, FRIFolder, FRIParams, FRIVerifier, FoldRoundOutput,
        calculate_n_test_queries, commit_interleaved,
    },
    reed_solomon::ReedSolomonCode,
    transcript::{ProverTranscript, VerifierTranscript},
};
use binius_field::{
    BinaryField32b, BinaryField128b, PackedField,
    arch::OptimalUnderlier128b,
    as_packed_field::PackedType,
};
use binius_hash::compression::PseudoCompressionFunction;
use digest::{Digest, Output, OutputSizeUser, typenum::U32};

// Blake3 wrapper that implements the digest traits binius needs
#[derive(Clone)]
struct Blake3Digest(blake3::Hasher);

impl Default for Blake3Digest {
    fn default() -> Self {
        Self(blake3::Hasher::new())
    }
}

impl digest::HashMarker for Blake3Digest {}

impl digest::OutputSizeUser for Blake3Digest {
    type OutputSize = U32;
}

impl digest::Update for Blake3Digest {
    fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }
}

impl digest::FixedOutput for Blake3Digest {
    fn finalize_into(self, out: &mut Output<Self>) {
        out.copy_from_slice(self.0.finalize().as_bytes());
    }
}

impl digest::FixedOutputReset for Blake3Digest {
    fn finalize_into_reset(&mut self, out: &mut Output<Self>) {
        out.copy_from_slice(self.0.finalize().as_bytes());
        self.0.reset();
    }
}

impl digest::Reset for Blake3Digest {
    fn reset(&mut self) {
        self.0.reset();
    }
}

impl digest::core_api::BlockSizeUser for Blake3Digest {
    type BlockSize = digest::typenum::U64; // Blake3 internally uses 64-byte blocks
}

impl digest::core_api::BufferKindUser for Blake3Digest {
    type BufferKind = digest::block_buffer::Eager;
}

// Blake3 compression function for Merkle tree
#[derive(Clone, Copy)]
struct Blake3Compression;

impl PseudoCompressionFunction<Output<Blake3Digest>, 2> for Blake3Compression {
    fn compress(&self, input: [Output<Blake3Digest>; 2]) -> Output<Blake3Digest> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&input[0]);
        hasher.update(&input[1]);
        *digest::generic_array::GenericArray::from_slice(hasher.finalize().as_bytes())
    }
}
use binius_ntt::SingleThreadedNTT;
use rand::prelude::*;

// Type aliases
type U = OptimalUnderlier128b;
type F = BinaryField128b;    // FRI field (128-bit tower top, required by binius compute layer)
type FA = BinaryField32b;    // RS encoding subfield
type PF = PackedType<U, F>;

type Challenger = HasherChallenger<Blake3Digest>;
type MerkleProverType = BinaryMerkleTreeProver<F, Blake3Digest, Blake3Compression>;
type VCSScheme = <MerkleProverType as MerkleTreeProver<F>>::Scheme;
type VCSDigest = <VCSScheme as MerkleTreeScheme<F>>::Digest;

// ── Helpers ─────────────────────────────────────────────────────────────

fn fmt_dur(d: std::time::Duration) -> String {
    if d.as_secs_f64() >= 1.0 {
        format!("{:.3}s", d.as_secs_f64())
    } else {
        format!("{:.2}ms", d.as_secs_f64() * 1000.0)
    }
}

fn fmt_bytes(n: usize) -> String {
    if n >= 1_048_576 {
        format!("{:.2} MB", n as f64 / 1_048_576.0)
    } else if n >= 1024 {
        format!("{:.2} KB", n as f64 / 1024.0)
    } else {
        format!("{} B", n)
    }
}

/// Find the closest sweet spot to a target Z root count for a given branch count
/// and number of committed Q splits. Same logic as the Plonky3 version.
fn find_sweet_spot(
    target_z: usize,
    branch_count: usize,
    num_splits: usize,
) -> (usize, usize, usize, usize) {
    let ratio = branch_count - 1;
    let mut best_z = 0usize;
    let mut best_j = 0usize;
    let mut best_dist = usize::MAX;

    for j in 1..=30 {
        let deg_qi = (1usize << j) - 1;
        let total_capacity = num_splits * deg_qi;
        let sweet_z = total_capacity / ratio;
        if sweet_z == 0 {
            continue;
        }
        let dist = if sweet_z >= target_z {
            sweet_z - target_z
        } else {
            target_z - sweet_z
        };
        if dist < best_dist {
            best_dist = dist;
            best_z = sweet_z;
            best_j = j;
        }
    }

    let deg_qi = (1usize << best_j) - 1;
    let total_capacity = num_splits * deg_qi;
    let dummy_count = total_capacity - ratio * best_z;
    (best_z, best_j, deg_qi, dummy_count)
}

fn main() {
    let num_sets: usize = 1;
    let target_z: usize = 256000; // 256K roots
    let branch_count: usize = 256;
    let num_splits: usize = 16;

    let (num_roots, j, deg_qi, dummy_count) = find_sweet_spot(target_z, branch_count, num_splits);
    let num_q_roots_total = num_splits * deg_qi;
    let deg_p = num_roots + num_q_roots_total;

    println!("=== Split-Commit {}x Prover [Binius GF(2^128)/GF(2^32)] ===", num_sets);
    println!(
        "branch_count={}, target_z={} -> sweet_z={}",
        branch_count, target_z, num_roots
    );
    println!(
        "num_splits(K)={}, deg(Qi)={} (2^{}-1), q_domain=2^{}",
        num_splits, deg_qi, j, j
    );
    println!(
        "deg(Z)={}, deg(P)={}, total Q roots={}, dummies={}",
        num_roots, deg_p, num_q_roots_total, dummy_count
    );
    println!(
        "Q columns: {}x{} = {}",
        num_splits,
        num_sets,
        num_splits * num_sets
    );
    println!();

    // ── FRI setup ──────────────────────────────────────────────────────

    // For Q polynomials: K * num_sets splits interleaved as batch
    let total_columns = num_splits * num_sets;
    let log_batch_size_q = (total_columns as f64).log2().ceil() as usize;
    let log_inv_rate = 2; // 4x blowup
    let security_bits = 100; // 100-bit FRI security
    let grinding_bits = 24; // 24-bit grinding for 124-bit total security
    let fold_arity = 4;

    // Manual FRI parameters
    // NOTE: Auto-chooser is LEGACY - it's limited by log_batch <= arity constraint,
    // resulting in poor performance. Always use manual params for production.
    let manual_params = true;
    let manual_q_log_dim = 14;  // Optimal for 100-bit security
    let manual_q_log_batch = 4; // 2^4 = 16 columns for smaller queries

    // NTT must be large enough for the RS code domain
    let log_ntt_size = if manual_params {
        (manual_q_log_dim + manual_q_log_batch + log_inv_rate + 4).max(20)
    } else {
        (j + log_batch_size_q + log_inv_rate + 4).max(20)
    };
    let ntt = SingleThreadedNTT::<FA>::new(log_ntt_size).unwrap();

    // Q commitment FRI params
    let q_log_msg_len = j + log_batch_size_q;
    let q_fri_params = if manual_params {
        // Manual: use provided log_dim and log_batch with shared NTT
        let q_rs_code = ReedSolomonCode::<FA>::with_ntt_subspace(&ntt, manual_q_log_dim, log_inv_rate).unwrap();
        let q_n_test_queries = calculate_n_test_queries::<F, FA>(security_bits, &q_rs_code).unwrap();
        let manual_q_log_msg_len = manual_q_log_dim + manual_q_log_batch;
        let cap_height = (q_n_test_queries as f64).log2().ceil() as usize;

        // CRITICAL: fold_arities[0] must be >= log_batch_size for first fold to work
        // Use the same formula as auto-chooser (see common.rs line 83-86)
        let n_fold_rounds = manual_q_log_msg_len.saturating_sub(
            cap_height.saturating_sub(log_inv_rate)
        ) / fold_arity;

        let q_fold_arities: Vec<usize> = if n_fold_rounds == 0 {
            // No folding needed - terminal codeword is the original message
            vec![]
        } else {
            // First arity must be at least log_batch_size
            let first_arity = manual_q_log_batch.max(fold_arity);
            let total_needed = manual_q_log_msg_len.saturating_sub(
                cap_height.saturating_sub(log_inv_rate)
            );
            let remaining = total_needed.saturating_sub(first_arity);
            let mut arities = vec![first_arity];
            arities.extend(vec![fold_arity; remaining / fold_arity]);
            arities
        };

        println!("Manual: log_msg={}, log_batch={}, cap_height={}, n_fold_rounds={}, fold_arities={:?}, sum={}",
            manual_q_log_msg_len, manual_q_log_batch, cap_height, n_fold_rounds,
            q_fold_arities, q_fold_arities.iter().sum::<usize>());
        FRIParams::<F, FA>::new(q_rs_code, manual_q_log_batch, q_fold_arities, q_n_test_queries).unwrap()
    } else {
        // Auto: let binius choose the split
        FRIParams::<F, FA>::choose_with_constant_fold_arity(
            &ntt,
            q_log_msg_len,
            security_bits,
            log_inv_rate,
            fold_arity,
        )
        .unwrap()
    };

    // Z commitment FRI params
    let z_log_msg_len = (num_roots + 1).next_power_of_two().trailing_zeros() as usize;
    let z_fri_params = FRIParams::<F, FA>::choose_with_constant_fold_arity(
        &ntt,
        z_log_msg_len,
        security_bits,
        log_inv_rate,
        fold_arity,
    )
    .unwrap();

    let merkle_prover =
        BinaryMerkleTreeProver::<_, Blake3Digest, _>::new(Blake3Compression);

    println!(
        "Q FRI: log_msg={}, log_dim={}, log_batch={}, log_inv_rate={}, arity={}, n_queries={}, n_fold_rounds={}",
        q_log_msg_len,
        q_fri_params.rs_code().log_dim(),
        q_fri_params.log_batch_size(),
        log_inv_rate,
        fold_arity,
        q_fri_params.n_test_queries(),
        q_fri_params.n_fold_rounds(),
    );
    println!(
        "Z FRI: log_msg={}, log_dim={}, n_queries={}, n_fold_rounds={}",
        z_log_msg_len,
        z_fri_params.rs_code().log_dim(),
        z_fri_params.n_test_queries(),
        z_fri_params.n_fold_rounds(),
    );
    if grinding_bits > 0 {
        println!(
            "Security: {}-bit FRI + {}-bit grinding = {}-bit total",
            security_bits, grinding_bits, security_bits + grinding_bits
        );
    }
    println!();

    let mut rng = StdRng::seed_from_u64(42);

    // ── Generate random messages (proxy for polynomial coefficients) ──
    print!("Setup: generating random messages... ");

    // Q message: padded batch of Qi polynomials interleaved
    let q_msg_elems = q_fri_params.rs_code().dim() << q_fri_params.log_batch_size();
    let q_msg: Vec<PF> = repeat_with(|| PF::random(&mut rng))
        .take(q_msg_elems >> PF::LOG_WIDTH)
        .collect();

    // Z message
    let z_msg_elems = z_fri_params.rs_code().dim() << z_fri_params.log_batch_size();
    let z_msg: Vec<PF> = repeat_with(|| PF::random(&mut rng))
        .take(z_msg_elems >> PF::LOG_WIDTH)
        .collect();

    println!("done (Q: {} elems, Z: {} elems)", q_msg_elems, z_msg_elems);
    println!();

    // ── Z commit (untimed, happens during Batchman) ──
    let CommitOutput {
        commitment: z_commitment,
        committed: z_committed,
        codeword: z_codeword,
    } = commit_interleaved(
        z_fri_params.rs_code(),
        &z_fri_params,
        &ntt,
        &merkle_prover,
        &z_msg,
    )
    .unwrap();

    // ── PROVER WORK (all timed) ─────────────────────────────────────────
    let total_start = Instant::now();

    // 1. Commit Q (RS encode + Merkle tree)
    let t = Instant::now();
    let CommitOutput {
        commitment: q_commitment,
        committed: q_committed,
        codeword: q_codeword,
    } = commit_interleaved(
        q_fri_params.rs_code(),
        &q_fri_params,
        &ntt,
        &merkle_prover,
        &q_msg,
    )
    .unwrap();
    let commit_q_time = t.elapsed();

    // 2. FRI prove for Z (fold rounds + query proofs)
    let t = Instant::now();
    let mut compute_holder_z = CpuLayerHolder::<F>::new(1 << 16, 1 << 22);
    let compute_data_z = compute_holder_z.to_data();

    let mut z_prover_transcript = ProverTranscript::<Challenger>::new();
    z_prover_transcript.message().write(&z_commitment);

    let mut z_folder = FRIFolder::new(
        compute_data_z.hal,
        &z_fri_params,
        &ntt,
        &merkle_prover,
        &z_codeword,
        &z_committed,
    )
    .unwrap();

    let mut z_round_commitments = Vec::new();
    for _ in 0..z_folder.n_rounds() {
        let challenge: F = z_prover_transcript.sample();
        match z_folder
            .execute_fold_round(&compute_data_z.dev_alloc, challenge)
            .unwrap()
        {
            FoldRoundOutput::NoCommitment => {}
            FoldRoundOutput::Commitment(comm) => {
                z_prover_transcript.message().write(&comm);
                z_round_commitments.push(comm);
            }
        }
    }
    z_folder.finish_proof(&mut z_prover_transcript).unwrap();
    let z_prove_time = t.elapsed();

    // 3. FRI prove for Q (fold rounds + query proofs)
    let t = Instant::now();
    let mut compute_holder_q = CpuLayerHolder::<F>::new(1 << 16, 1 << 28);
    let compute_data_q = compute_holder_q.to_data();

    let mut q_prover_transcript = ProverTranscript::<Challenger>::new();
    q_prover_transcript.message().write(&q_commitment);

    let mut q_folder = FRIFolder::new(
        compute_data_q.hal,
        &q_fri_params,
        &ntt,
        &merkle_prover,
        &q_codeword,
        &q_committed,
    )
    .unwrap();

    let mut q_round_commitments = Vec::new();
    for _ in 0..q_folder.n_rounds() {
        let challenge: F = q_prover_transcript.sample();
        match q_folder
            .execute_fold_round(&compute_data_q.dev_alloc, challenge)
            .unwrap()
        {
            FoldRoundOutput::NoCommitment => {}
            FoldRoundOutput::Commitment(comm) => {
                q_prover_transcript.message().write(&comm);
                q_round_commitments.push(comm);
            }
        }
    }
    q_folder.finish_proof(&mut q_prover_transcript).unwrap();
    let q_prove_time = t.elapsed();

    let total_time = total_start.elapsed();

    // ── Get proof bytes and create verifier transcripts ─────────────────
    let z_proof_bytes_vec = z_prover_transcript.finalize();
    let z_proof_bytes = z_proof_bytes_vec.len();
    let q_proof_bytes_vec = q_prover_transcript.finalize();
    let q_proof_bytes = q_proof_bytes_vec.len();
    let total_proof_bytes = z_proof_bytes + q_proof_bytes;

    // ── Grinding (proof-of-work for additional security) ────────────────
    let grinding_time = if grinding_bits > 0 {
        let t = std::time::Instant::now();

        // Hash proof transcripts once to get a commitment (32 bytes)
        let mut proof_hasher = blake3::Hasher::new();
        proof_hasher.update(&z_proof_bytes_vec);
        proof_hasher.update(&q_proof_bytes_vec);
        let proof_commitment = proof_hasher.finalize();

        let mut nonce: u64 = 0;
        let target_zeros = grinding_bits;

        loop {
            // Hash commitment + nonce (only 40 bytes, very fast!)
            let mut hasher = blake3::Hasher::new();
            hasher.update(proof_commitment.as_bytes());
            hasher.update(&nonce.to_le_bytes());
            let hash = hasher.finalize();

            // Check if hash meets difficulty target (leading zero bits)
            let leading_zeros = hash.as_bytes().iter()
                .take_while(|&&b| b == 0)
                .count() * 8
                + hash.as_bytes().iter()
                    .find(|&&b| b != 0)
                    .map(|&b| b.leading_zeros() as usize)
                    .unwrap_or(0);

            if leading_zeros >= target_zeros {
                println!("Grinding: found nonce {} after {} attempts ({} bits)",
                    nonce, nonce + 1, leading_zeros);
                break;
            }

            nonce += 1;
            if nonce % 1_000_000 == 0 && nonce > 0 {
                print!("\rGrinding: {} M attempts...", nonce / 1_000_000);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }
        }
        if nonce >= 1_000_000 {
            println!(); // Newline after progress
        }
        t.elapsed()
    } else {
        std::time::Duration::ZERO
    };

    // ── Results table ───────────────────────────────────────────────────
    println!("================================================");
    println!(" Step                       | Time");
    println!("------------------------------------------------");
    println!(" {:<27}| {:>10}", "Commit Q (RS+Merkle)", fmt_dur(commit_q_time));
    println!(" {:<27}| {:>10}", "FRI prove Z", fmt_dur(z_prove_time));
    println!(" {:<27}| {:>10}", "FRI prove Q", fmt_dur(q_prove_time));
    if grinding_bits > 0 {
        println!(" {:<27}| {:>10}", format!("Grinding ({}-bit)", grinding_bits), fmt_dur(grinding_time));
    }
    println!("------------------------------------------------");
    println!(
        " {:<27}| {:>10}",
        "TOTAL PROVER",
        fmt_dur(total_time + grinding_time)
    );
    println!("================================================");
    println!();
    println!("Proof size:");
    println!("  Z FRI proof:     {:>10}", fmt_bytes(z_proof_bytes));
    println!("  Q FRI proof:     {:>10}", fmt_bytes(q_proof_bytes));
    println!("  ─────────────────────────────");
    println!("  TOTAL:           {:>10}", fmt_bytes(total_proof_bytes));
    println!();

    // ── VERIFIER ──────────────────────────────────────────────────────────
    let verify_start = Instant::now();

    // Verify Z FRI proof
    let t = Instant::now();
    {
        let mut vt = VerifierTranscript::<Challenger>::new(z_proof_bytes_vec);
        let z_commitment_v = vt.message().read().unwrap();

        let mut z_challenges = Vec::new();
        for (i, _) in z_round_commitments.iter().enumerate() {
            z_challenges.append(&mut vt.sample_vec(z_fri_params.fold_arities()[i]));
            let _comm: VCSDigest = vt.message().read().unwrap();
        }
        z_challenges.append(&mut vt.sample_vec(z_fri_params.n_final_challenges()));

        let z_verifier = FRIVerifier::new(
            &z_fri_params,
            merkle_prover.scheme(),
            &z_commitment_v,
            &z_round_commitments,
            &z_challenges,
        )
        .unwrap();
        let _z_final = z_verifier.verify(&mut vt).unwrap();
    }
    let z_verify_time = t.elapsed();

    // Verify Q FRI proof
    let t = Instant::now();
    {
        let mut vt = VerifierTranscript::<Challenger>::new(q_proof_bytes_vec);
        let q_commitment_v = vt.message().read().unwrap();

        let mut q_challenges = Vec::new();
        for (i, _) in q_round_commitments.iter().enumerate() {
            q_challenges.append(&mut vt.sample_vec(q_fri_params.fold_arities()[i]));
            let _comm: VCSDigest = vt.message().read().unwrap();
        }
        q_challenges.append(&mut vt.sample_vec(q_fri_params.n_final_challenges()));

        let q_verifier = FRIVerifier::new(
            &q_fri_params,
            merkle_prover.scheme(),
            &q_commitment_v,
            &q_round_commitments,
            &q_challenges,
        )
        .unwrap();
        let _q_final = q_verifier.verify(&mut vt).unwrap();
    }
    let q_verify_time = t.elapsed();

    let verify_time = verify_start.elapsed();

    println!("================================================");
    println!(" VERIFIER                   | Time");
    println!("------------------------------------------------");
    println!(
        " {:<27}| {:>10}",
        "Verify Z FRI",
        fmt_dur(z_verify_time)
    );
    println!(
        " {:<27}| {:>10}",
        "Verify Q FRI",
        fmt_dur(q_verify_time)
    );
    println!("------------------------------------------------");
    println!(
        " {:<27}| {:>10}",
        "TOTAL VERIFIER",
        fmt_dur(verify_time)
    );
    println!("================================================");
}
