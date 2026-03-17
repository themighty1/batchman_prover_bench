// Full split-commit subset membership proof using Binius PIOP.
//
// Proves P(α) = Z(α) · ∏ Qi(α) where Z and Qi are committed as multilinear
// polynomials, and evaluations at a Fiat-Shamir challenge α are proven via
// binius's PIOP (interleaved sumcheck + FRI).
//
// The key insight: univariate evaluation Z(α) = Σ c_i · α^i is exactly the
// inner product of the coefficient multilinear and the Powers(α) transparent
// polynomial over the boolean hypercube. Binius already provides this as
// `transparent::powers::Powers<F>`.

use std::iter::repeat_with;
use std::time::Instant;

use binius_compute::{
    ComputeHolder, ComputeLayer, ComputeMemory,
    alloc::ComputeAllocator, cpu::CpuMemory, cpu::layer::CpuLayerHolder,
};
use binius_core::{
    fiat_shamir::{CanSample, HasherChallenger},
    merkle_tree::{BinaryMerkleTreeProver, MerkleTreeScheme},
    piop::{
        CommitMeta, PIOPSumcheckClaim, commit, make_commit_params_with_constant_arity, prove,
        verify,
    },
    polynomial::MultivariatePoly,
    protocols::fri::{CommitOutput, FRIParams, calculate_n_test_queries},
    reed_solomon::ReedSolomonCode,
    transcript::{ProverTranscript, VerifierTranscript},
    transparent::powers::Powers,
};
use binius_field::{
    BinaryField128b, BinaryField32b, PackedField, TowerField,
    arch::OptimalUnderlier128b,
    as_packed_field::PackedType,
};
use binius_hash::compression::PseudoCompressionFunction;
use binius_field::Field;
use binius_math::{MLEDirectAdapter, MultilinearExtension, MultilinearPoly};
use binius_ntt::{MultithreadedNTT, SingleThreadedNTT};
use digest::{Output, OutputSizeUser, typenum::U32};
use rand::prelude::*;

// ── Blake3 wrappers (same as bench_binius.rs) ────────────────────────────

#[derive(Clone)]
struct Blake3Digest(blake3::Hasher);

impl Default for Blake3Digest {
    fn default() -> Self {
        Self(blake3::Hasher::new())
    }
}

impl digest::HashMarker for Blake3Digest {}
impl OutputSizeUser for Blake3Digest {
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
    type BlockSize = digest::typenum::U64;
}

impl digest::core_api::BufferKindUser for Blake3Digest {
    type BufferKind = digest::block_buffer::Eager;
}

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

// ── Type aliases ─────────────────────────────────────────────────────────

type U = OptimalUnderlier128b;
type F = BinaryField128b;
type FA = BinaryField32b;
type PF = PackedType<U, F>;

type Challenger = HasherChallenger<Blake3Digest>;
type MerkleProverType = BinaryMerkleTreeProver<F, Blake3Digest, Blake3Compression>;
type VCSScheme = <MerkleProverType as binius_core::merkle_tree::MerkleTreeProver<F>>::Scheme;
type VCSDigest = <VCSScheme as MerkleTreeScheme<F>>::Digest;

// ── Helpers ──────────────────────────────────────────────────────────────

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
        let dist = sweet_z.abs_diff(target_z);
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

/// Evaluate univariate polynomial (given by coefficients) at a point.
/// poly(x) = coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ...
fn eval_univariate(coeffs: &[F], x: F) -> F {
    let mut result = F::ZERO;
    let mut x_pow = F::ONE;
    for &c in coeffs {
        result += c * x_pow;
        x_pow *= x;
    }
    result
}

fn main() {
    // ── Protocol parameters ─────────────────────────────────────────────
    //
    // Tuning knobs and their best observed values (prover speed only):
    //
    //   num_splits: 16 is optimal across sizes. Fewer = fewer sumcheck claims,
    //     but below 16 the gain is negligible (FRI fold dominates).
    //     8k:  8→7.3s, 16→6.5s, 32→6.5s, 64→7.4s
    //     32k: 8→27.7s, 16→27.8s, 32→28.6s, 64→30.1s, 128→29.7s
    //
    //   manual_log_batch: optimal = total_vars - 15, fold_arity = 4.
    //     Higher log_batch = more interleaved smaller RS codes = faster commit,
    //     but first fold arity must be >= log_batch (FRI constraint).
    //     8k (tv=20): log_batch=5 →  7.3s
    //     16k (tv=21): log_batch=6 → 13.1s  (auto log_batch=4: 17.1s)
    //     32k (tv=22): log_batch=7 → 27.8s  (auto log_batch=4: 37.6s)
    //
    //   log_inv_rate: 1 = fastest prover (smallest codeword), 2 = smaller proof.
    //
    let num_sets: usize = 1;
    let target_z: usize = 16000;
    let branch_count: usize = 51;
    let num_splits: usize = 16;

    let (num_roots, j, deg_qi, dummy_count) =
        find_sweet_spot(target_z, branch_count, num_splits);
    let num_q_roots_total = num_splits * deg_qi;
    let deg_p = num_roots + num_q_roots_total;

    println!(
        "=== Split-Commit {}x Prover [Binius PIOP GF(2^128)/GF(2^32)] ===",
        num_sets
    );
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
    println!();

    // ── Polynomial setup ─────────────────────────────────────────────────
    // Z has num_roots+1 coefficients, padded to next power of 2
    let z_num_coeffs = num_roots + 1;
    let z_padded_size = z_num_coeffs.next_power_of_two();
    let log_z = z_padded_size.trailing_zeros() as usize;

    // Each Qi has deg_qi+1 = 2^j coefficients
    let qi_num_coeffs = deg_qi + 1; // = 2^j
    assert_eq!(qi_num_coeffs, 1 << j);
    let log_qi = j;

    let total_committed = num_splits * num_sets + num_sets; // K*sets Qi's + sets Z's
    println!(
        "Committed multilinears: {} Qi (log_size={}) + {} Z (log_size={})",
        num_splits * num_sets, log_qi, num_sets, log_z
    );

    let mut rng = StdRng::seed_from_u64(42);

    // Generate random coefficients for Z and Qi polynomials
    print!("Setup: generating polynomial coefficients... ");
    let setup_start = Instant::now();

    // Z coefficients per set (padded to power of 2 with zeros)
    let z_coeffs_all: Vec<Vec<F>> = (0..num_sets)
        .map(|_| {
            let mut coeffs: Vec<F> = repeat_with(|| <F as Field>::random(&mut rng))
                .take(z_num_coeffs)
                .collect();
            coeffs.resize(z_padded_size, F::ZERO);
            coeffs
        })
        .collect();

    // Qi coefficients (already power-of-2 sized)
    let qi_polys: Vec<Vec<F>> = (0..num_splits * num_sets)
        .map(|_| {
            repeat_with(|| <F as Field>::random(&mut rng))
                .take(qi_num_coeffs)
                .collect()
        })
        .collect();

    println!("done ({})", fmt_dur(setup_start.elapsed()));

    // ── Create multilinear extensions ────────────────────────────────────
    // Committed multilinears must be sorted ascending by n_vars.
    // Qi have log_qi vars, Z has log_z vars. Since log_qi <= log_z (typically),
    // Qi come first.

    let mut committed_multilins: Vec<MLEDirectAdapter<PF, _>> = Vec::with_capacity(total_committed);

    // Pack scalar coefficients into PackedField elements
    let pack_coeffs = |coeffs: &[F]| -> Vec<PF> {
        let n_packed = coeffs.len().div_ceil(PF::WIDTH);
        let mut packed = vec![PF::zero(); n_packed];
        for (i, &c) in coeffs.iter().enumerate() {
            packed[i / PF::WIDTH].set(i % PF::WIDTH, c);
        }
        packed
    };

    if log_qi <= log_z {
        // Qi first (smaller), then Z's
        for qi_coeffs in &qi_polys {
            let packed = pack_coeffs(qi_coeffs);
            let mle = MultilinearExtension::new(log_qi, packed).unwrap();
            committed_multilins.push(MLEDirectAdapter::from(mle));
        }
        for z_coeffs in &z_coeffs_all {
            let packed_z = pack_coeffs(z_coeffs);
            let mle_z = MultilinearExtension::new(log_z, packed_z).unwrap();
            committed_multilins.push(MLEDirectAdapter::from(mle_z));
        }
    } else {
        // Z's first (smaller), then Qi
        for z_coeffs in &z_coeffs_all {
            let packed_z = pack_coeffs(z_coeffs);
            let mle_z = MultilinearExtension::new(log_z, packed_z).unwrap();
            committed_multilins.push(MLEDirectAdapter::from(mle_z));
        }
        for qi_coeffs in &qi_polys {
            let packed = pack_coeffs(qi_coeffs);
            let mle = MultilinearExtension::new(log_qi, packed).unwrap();
            committed_multilins.push(MLEDirectAdapter::from(mle));
        }
    }

    // Build CommitMeta
    let n_vars_iter = committed_multilins.iter().map(|m| m.n_vars());
    let commit_meta = CommitMeta::with_vars(n_vars_iter);

    println!(
        "CommitMeta: total_vars={}, total_multilins={}",
        commit_meta.total_vars(),
        commit_meta.total_multilins()
    );

    // ── FRI parameters ───────────────────────────────────────────────────
    let log_inv_rate = 1;
    let security_bits = 100;
    let fold_arity = 4;

    let manual_params = true;
    let manual_log_batch = commit_meta.total_vars().saturating_sub(15);
    let manual_log_dim = commit_meta.total_vars() - manual_log_batch;

    // Use MultithreadedNTT for parallel RS encoding (the main commit bottleneck).
    let log_ntt_size = (commit_meta.total_vars() + log_inv_rate).max(20);
    let ntt = SingleThreadedNTT::<FA>::new(log_ntt_size)
        .unwrap()
        .multithreaded();

    let fri_params = if manual_params {
        let rs_code =
            ReedSolomonCode::<FA>::with_ntt_subspace(&ntt, manual_log_dim, log_inv_rate).unwrap();
        let n_test_queries =
            calculate_n_test_queries::<F, FA>(security_bits, &rs_code).unwrap();
        let log_msg = manual_log_dim + manual_log_batch;
        let cap_height = (n_test_queries as f64).log2().ceil() as usize;

        // First fold arity must be >= log_batch (FRI fold constraint).
        // Then pack as many uniform arity folds as possible, with a partial
        // final fold if there's a remainder >= 1.
        let fold_arities: Vec<usize> = {
            let first_arity = manual_log_batch.max(fold_arity);
            let total_needed =
                log_msg.saturating_sub(cap_height.saturating_sub(log_inv_rate));
            let remaining = total_needed.saturating_sub(first_arity);
            let mut arities = vec![first_arity];
            let n_full = remaining / fold_arity;
            arities.extend(vec![fold_arity; n_full]);
            let leftover = remaining - n_full * fold_arity;
            if leftover > 0 {
                arities.push(leftover);
            }
            arities
        };

        println!(
            "Manual FRI: log_msg={}, log_dim={}, log_batch={}, cap_height={}, fold_arities={:?}",
            log_msg, manual_log_dim, manual_log_batch, cap_height, fold_arities,
        );
        FRIParams::<F, FA>::new(rs_code, manual_log_batch, fold_arities, n_test_queries).unwrap()
    } else {
        make_commit_params_with_constant_arity(
            &ntt,
            &commit_meta,
            security_bits,
            log_inv_rate,
            fold_arity,
        )
        .unwrap()
    };

    println!(
        "FRI: log_msg={}, log_dim={}, log_batch={}, n_queries={}, n_fold_rounds={}",
        commit_meta.total_vars(),
        fri_params.rs_code().log_dim(),
        fri_params.log_batch_size(),
        fri_params.n_test_queries(),
        fri_params.n_fold_rounds(),
    );
    println!();

    let merkle_prover = BinaryMerkleTreeProver::<_, Blake3Digest, _>::new(Blake3Compression);

    // ── PIOP Commit ──────────────────────────────────────────────────────
    let t = Instant::now();
    let CommitOutput {
        commitment,
        committed,
        codeword,
    } = commit(&fri_params, &ntt, &merkle_prover, &committed_multilins).unwrap();
    let commit_time = t.elapsed();
    println!("Commit: {} (all polynomials)", fmt_dur(commit_time));

    // ── Fiat-Shamir challenge α ──────────────────────────────────────────
    let mut prover_transcript = ProverTranscript::<Challenger>::new();
    prover_transcript.message().write(&commitment);
    let alpha: F = prover_transcript.sample();
    println!("Challenge α derived from commitment");

    // ── Compute polynomial evaluations at α ──────────────────────────────
    let t = Instant::now();
    let z_evals: Vec<F> = z_coeffs_all
        .iter()
        .map(|coeffs| eval_univariate(coeffs, alpha))
        .collect();

    let qi_evals: Vec<F> = qi_polys
        .iter()
        .map(|coeffs| eval_univariate(coeffs, alpha))
        .collect();

    // P_s(α) = Z_s(α) · ∏ Q_{s,i}(α) for each set s
    let p_alphas: Vec<F> = (0..num_sets)
        .map(|s| {
            let qi_start = s * num_splits;
            let product_qi: F = qi_evals[qi_start..qi_start + num_splits]
                .iter()
                .copied()
                .fold(F::ONE, |acc, v| acc * v);
            z_evals[s] * product_qi
        })
        .collect();
    let eval_time = t.elapsed();
    println!("Evaluations at α: {} ({} Z + {} Qi)", fmt_dur(eval_time), num_sets, qi_evals.len());

    // ── Build transparent polynomials and sumcheck claims ────────────────
    let t = Instant::now();

    // We need transparent Powers polynomials for each distinct n_vars.
    // If log_qi != log_z, we have two distinct sizes; otherwise one.
    // Transparents must be sorted ascending by n_vars.
    let (z_committed_base, qi_committed_start, z_transparent_idx, qi_transparent_idx);

    let transparent_powers: Vec<Powers<F>>;

    if log_qi < log_z {
        // Qi (indices 0..K*sets) then Z's (indices K*sets..K*sets+sets) in committed list
        qi_committed_start = 0;
        z_committed_base = num_splits * num_sets;
        transparent_powers = vec![
            Powers::new(log_qi, alpha),
            Powers::new(log_z, alpha),
        ];
        qi_transparent_idx = 0;
        z_transparent_idx = 1;
    } else if log_qi > log_z {
        // Z's first (indices 0..sets) then Qi (indices sets..sets+K*sets)
        z_committed_base = 0;
        qi_committed_start = num_sets;
        transparent_powers = vec![
            Powers::new(log_z, alpha),
            Powers::new(log_qi, alpha),
        ];
        z_transparent_idx = 0;
        qi_transparent_idx = 1;
    } else {
        // Same size: Qi then Z's, single transparent
        qi_committed_start = 0;
        z_committed_base = num_splits * num_sets;
        transparent_powers = vec![Powers::new(log_qi, alpha)];
        qi_transparent_idx = 0;
        z_transparent_idx = 0;
    }

    // Build sumcheck claims: for each set, one Z claim + K Qi claims
    let mut claims: Vec<PIOPSumcheckClaim<F>> = Vec::with_capacity(total_committed);

    for s in 0..num_sets {
        // Z claim for set s
        claims.push(PIOPSumcheckClaim {
            n_vars: log_z,
            committed: z_committed_base + s,
            transparent: z_transparent_idx,
            sum: z_evals[s],
        });

        // Qi claims for set s
        for k in 0..num_splits {
            claims.push(PIOPSumcheckClaim {
                n_vars: log_qi,
                committed: qi_committed_start + s * num_splits + k,
                transparent: qi_transparent_idx,
                sum: qi_evals[s * num_splits + k],
            });
        }
    }

    // Compute hypercube evaluations of transparent polynomials for the prover
    let transparent_mles: Vec<MultilinearExtension<PF>> = transparent_powers
        .iter()
        .map(|p| p.multilinear_extension().unwrap())
        .collect();

    let claim_setup_time = t.elapsed();
    println!(
        "Claim setup (transparent evals): {}",
        fmt_dur(claim_setup_time)
    );

    // ── PIOP Prove ───────────────────────────────────────────────────────
    let t = Instant::now();

    // Allocate compute holder for the PIOP prover
    let dev_mem_size = 1usize << 28; // 256M elements
    let host_mem_size = 1usize << 24;
    let mut compute_holder = CpuLayerHolder::<F>::new(host_mem_size, dev_mem_size);
    let mut compute_data = compute_holder.to_data();
    let compute_data_ref = &mut compute_data;

    let hal = compute_data_ref.hal;
    let dev_alloc = &compute_data_ref.dev_alloc;

    // Upload transparent evaluations to device memory
    let transparent_fslices: Vec<_> = transparent_mles
        .iter()
        .map(|mle| {
            let n_elems = 1 << mle.n_vars();
            let mut buffer = dev_alloc.alloc(n_elems).unwrap();
            let evals: Vec<F> = PackedField::iter_slice(mle.evals())
                .take(n_elems)
                .collect();
            hal.copy_h2d(&evals[..n_elems], &mut buffer).unwrap();
            CpuMemory::to_const(buffer)
        })
        .collect();

    prove(
        compute_data_ref,
        &fri_params,
        &ntt,
        &merkle_prover,
        &commit_meta,
        committed,
        &codeword,
        &committed_multilins,
        transparent_fslices,
        &claims,
        &mut prover_transcript,
    )
    .unwrap();

    let prove_time = t.elapsed();
    println!("PIOP prove: {}", fmt_dur(prove_time));

    // ── Get proof bytes ──────────────────────────────────────────────────
    let proof_bytes_vec = prover_transcript.finalize();
    let proof_size = proof_bytes_vec.len();

    // ── PIOP Verify ──────────────────────────────────────────────────────
    let t = Instant::now();

    let mut verifier_transcript = VerifierTranscript::<Challenger>::new(proof_bytes_vec);
    let commitment_v: VCSDigest = verifier_transcript.message().read().unwrap();
    // Verifier derives the same α
    let alpha_v: F = verifier_transcript.sample();
    assert_eq!(alpha_v, alpha, "Fiat-Shamir consistency");

    let transparent_polys: Vec<&dyn MultivariatePoly<F>> = transparent_powers
        .iter()
        .map(|p| p as &dyn MultivariatePoly<F>)
        .collect();

    verify(
        &commit_meta,
        merkle_prover.scheme(),
        &fri_params,
        &commitment_v,
        &transparent_polys,
        &claims,
        &mut verifier_transcript,
    )
    .unwrap();

    // Verifier algebraic check: P_s(α) = Z_s(α) · ∏ Q_{s,i}(α) for each set
    let claims_per_set = 1 + num_splits; // 1 Z + K Qi per set
    for s in 0..num_sets {
        let set_claims = &claims[s * claims_per_set..(s + 1) * claims_per_set];
        let z_sum = set_claims[0].sum;
        let product_qi = set_claims[1..]
            .iter()
            .map(|c| c.sum)
            .fold(F::ONE, |acc, v| acc * v);
        assert_eq!(
            z_sum * product_qi, p_alphas[s],
            "Divisibility check failed for set {}", s
        );
    }

    let verify_time = t.elapsed();
    println!("PIOP verify: {}", fmt_dur(verify_time));

    // ── Results ──────────────────────────────────────────────────────────
    println!();
    println!("================================================");
    println!(" Step                       | Time");
    println!("------------------------------------------------");
    println!(
        " {:<27}| {:>10}",
        "Commit (RS+Merkle)", fmt_dur(commit_time)
    );
    println!(
        " {:<27}| {:>10}",
        "Eval at α", fmt_dur(eval_time)
    );
    println!(
        " {:<27}| {:>10}",
        "Claim setup", fmt_dur(claim_setup_time)
    );
    println!(
        " {:<27}| {:>10}",
        "PIOP prove (sumcheck+FRI)", fmt_dur(prove_time)
    );
    println!("------------------------------------------------");
    let total_prover = commit_time + eval_time + claim_setup_time + prove_time;
    println!(
        " {:<27}| {:>10}",
        "TOTAL PROVER", fmt_dur(total_prover)
    );
    println!("================================================");
    println!();
    println!(
        " {:<27}| {:>10}",
        "PIOP verify", fmt_dur(verify_time)
    );
    println!("================================================");
    println!();
    println!("Proof size: {}", fmt_bytes(proof_size));
}
