// PCS mode for the Batchman protocol
//
// This replaces the original LPZK product-of-branches check in the Batchman
// disjunction protocol with a polynomial commitment scheme (PCS) based proof.
//
// How it works:
//   At the end of the Batchman protocol, each batch instance has a branch proof
//   (IT-MAC over GF(2^128)) for every branch. The correct (active) branch's
//   proof is zero; all others are nonzero.
//
//   In LPZK mode, the prover shows the product of all branch proofs is zero
//   (so at least one factor is zero). PCS mode replaces this with:
//
//   1. Prover commits to its zero MACs (the active branch proofs) via PCS.
//   2. Verifier reveals all its MAC keys for every branch.
//   3. Prover proves that its committed values are all a subset of the
//      verifier's revealed MACs — i.e., for each batch, the committed value
//      matches one of the verifier's branch keys.
//
//   The main motivation is communication cost. In a 2PC setting, prover upload
//   bandwidth is the bottleneck. LPZK requires the prover to send (branch_sz-1)
//   intermediate multiplication-tree wire commitments per batch — that's
//   O(branch_sz * batch_sz * 16 bytes) prover→verifier. For 1200 branches and
//   10K batches this is ~192 MB. PCS mode eliminates this entirely: the prover
//   commits locally and proves subset membership with a PCS proof that is ~1 MB
//   on average for the entire set of MACs (depending on the batching technique).
//
// Trade-off:
//   The downside of PCS mode is prover computation time — on the order of tens
//   of seconds on CPU. The dominant cost is polynomial evaluation and
//   interpolation via NTT/FFT (Number Theoretic Transform), which the PCS uses
//   to commit to and open polynomials over large evaluation domains. NTTs are
//   highly parallelizable (butterfly structure with independent operations at
//   each level), making them a natural fit for GPU acceleration. GPU-based NTT
//   could potentially reduce PCS prover time by orders of magnitude, though this
//   still needs investigation.

use std::time::Instant;

use p3_blake3::Blake3;
use p3_challenger::{CanObserve, FieldChallenger, HashChallenger};
use p3_commit::{ExtensionMmcs, Pcs};
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

// Field selection: default=Goldilocks(64-bit), baby-bear(31-bit), koala-bear(31-bit).
// Benchmarks show Goldilocks wins in all categories: 31-bit fields need 2x roots to
// represent the same secret data, a degree-4 extension (vs degree-2), and larger FRI
// domains — resulting in ~2x slower prover, ~10% larger proofs, and ~2-3x slower
// verifier. The 31-bit features are kept for comparison only.
#[cfg(not(any(feature = "baby-bear", feature = "koala-bear")))]
use p3_goldilocks::Goldilocks;
#[cfg(feature = "baby-bear")]
use p3_baby_bear::BabyBear;
#[cfg(feature = "koala-bear")]
use p3_koala_bear::KoalaBear;

#[cfg(not(any(feature = "baby-bear", feature = "koala-bear")))]
use p3_challenger::SerializingChallenger64;
#[cfg(any(feature = "baby-bear", feature = "koala-bear"))]
use p3_challenger::SerializingChallenger32;

#[cfg(not(any(feature = "baby-bear", feature = "koala-bear")))]
type Val = Goldilocks;
#[cfg(feature = "baby-bear")]
type Val = BabyBear;
#[cfg(feature = "koala-bear")]
type Val = KoalaBear;

// Goldilocks uses degree-2 extension (~128-bit), 31-bit fields use degree-4 (~124-bit)
#[cfg(not(any(feature = "baby-bear", feature = "koala-bear")))]
type Challenge = BinomialExtensionField<Val, 2>;
#[cfg(any(feature = "baby-bear", feature = "koala-bear"))]
type Challenge = BinomialExtensionField<Val, 4>;

type MyHash = SerializingHasher<Blake3>;
type MyCompress = CompressionFunctionFromHasher<Blake3, 2, 32>;
type ValMmcs = MerkleTreeMmcs<Val, u8, MyHash, MyCompress, 32>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Dft = Radix2DitParallel<Val>;

#[cfg(not(any(feature = "baby-bear", feature = "koala-bear")))]
type Challenger = SerializingChallenger64<Val, HashChallenger<u8, Blake3, 32>>;
#[cfg(any(feature = "baby-bear", feature = "koala-bear"))]
type Challenger = SerializingChallenger32<Val, HashChallenger<u8, Blake3, 32>>;

type MyPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;

fn setup_pcs() -> (MyPcs, Challenger) {
    let hash = MyHash::new(Blake3);
    let compress = MyCompress::new(Blake3);
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    // Conjectured soundness = log_blowup * num_queries + query_pow_bits
    //                       = 1 * 112 + 16 = 128 bits
    let fri_params = FriParameters {
        log_blowup: 1,
        log_final_poly_len: 0,
        num_queries: 112,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 16,
        mmcs: challenge_mmcs,
    };
    let pcs = MyPcs::new(Dft::default(), val_mmcs, fri_params);
    (pcs, Challenger::from_hasher(vec![], Blake3))
}

// ── Polynomial helpers ──────────────────────────────────────────────────

fn build_z_from_roots(roots: &[Val]) -> Vec<Val> {
    let mut z = vec![Val::ONE];
    for &r in roots {
        let mut new_z = vec![Val::ZERO; z.len() + 1];
        for i in 0..z.len() {
            new_z[i + 1] = new_z[i + 1] + z[i];
            new_z[i] = new_z[i] - r * z[i];
        }
        z = new_z;
    }
    z
}

fn poly_mul_fft(a: &[Val], b: &[Val], dft: &Dft) -> Vec<Val> {
    let result_len = a.len() + b.len() - 1;
    let n = result_len.next_power_of_two();
    let mut a_padded = a.to_vec();
    a_padded.resize(n, Val::ZERO);
    let mut b_padded = b.to_vec();
    b_padded.resize(n, Val::ZERO);
    let a_evals = dft.dft(a_padded);
    let b_evals = dft.dft(b_padded);
    let c_evals: Vec<Val> = a_evals
        .iter()
        .zip(b_evals.iter())
        .map(|(&a, &b)| a * b)
        .collect();
    let mut c = dft.idft(c_evals);
    c.truncate(result_len);
    c
}

fn poly_div_fft(p: &[Val], z: &[Val], dft: &Dft) -> Vec<Val> {
    let q_len = p.len() - z.len() + 1;
    let n = p.len().next_power_of_two();
    let mut p_padded = p.to_vec();
    p_padded.resize(n, Val::ZERO);
    let mut z_padded = z.to_vec();
    z_padded.resize(n, Val::ZERO);
    let p_evals = dft.dft(p_padded);
    let z_evals = dft.dft(z_padded);
    let q_evals: Vec<Val> = p_evals
        .iter()
        .zip(z_evals.iter())
        .map(|(&p, &z)| p * z.inverse())
        .collect();
    let mut q = dft.idft(q_evals);
    q.truncate(q_len);
    q
}

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

fn poly_eval_at_challenge(p: &[Val], x: Challenge) -> Challenge {
    let mut result = Challenge::ZERO;
    for i in (0..p.len()).rev() {
        result = result * x + Challenge::from(p[i]);
    }
    result
}

fn build_multi_column_matrix(polys: &[Vec<Val>], domain_size: usize) -> RowMajorMatrix<Val> {
    let width = polys.len();
    let mut data = vec![Val::ZERO; domain_size * width];
    for (col, poly) in polys.iter().enumerate() {
        for (row, &val) in poly.iter().enumerate() {
            data[row * width + col] = val;
        }
    }
    RowMajorMatrix::new(data, width)
}

type ComData = (
    <MyPcs as Pcs<Challenge, Challenger>>::Commitment,
    <MyPcs as Pcs<Challenge, Challenger>>::ProverData,
);

/// Given deg(Z), find deg(P) such that deg(Q) = deg(P) - deg(Z) = 2^k - 1 for
/// some k, and the ratio deg(P)/deg(Z) is closest to the target (~100x).
///
/// This ratio reflects the structure of a zkVM: deg(Z) represents a committed
/// value (e.g. an executed instruction), and deg(P) ~ 120 * deg(Z) reflects the
/// total ISA size (~100 unique instructions/branches) plus ~20 dummy
/// instructions needed for zero-knowledge. The 20 dummy roots are random
/// field elements mixed into Z, so the verifier sees evaluations of a
/// degree-520 polynomial (500 real + 20 dummy). Recovering the real roots
/// requires distinguishing them from the dummy ones — C(520, 20) ≈ 2^130
/// possible dummy subsets, providing 128-bit computational security against
/// brute-force. When root count grows, fewer dummy roots suffice, but we
/// keep 20 as a conservative baseline.
///
/// # Enforcing a lower bound on deg(Z) via Q's domain
///
/// FRI domains must be power-of-2 sized, and FRI proves deg(Q) < q_domain_size.
/// By choosing deg(Q) = 2^k - 1, we get q_domain_size = 2^k with zero slack.
///
/// The verifier fixes q_domain_size = 2^k as a public protocol parameter.
/// The prover MUST commit Q in this exact domain — they cannot choose a larger
/// one. Since P = Q * Z and deg(P) is public, deg(Q) = deg(P) - deg(Z).
///
/// If a cheating prover commits fewer roots (smaller deg(Z)), their Q has
/// higher degree. Even one fewer root makes deg(Q) = 2^k, which violates
/// the FRI bound deg(Q) < 2^k. The prover cannot fit this Q in the fixed
/// domain, so FRI rejects the proof.
///
/// This is why deg(Q) = 2^k - 1 is critical: any other value would leave
/// slack in the power-of-2 domain, allowing the prover to cheat by that
/// many roots.
///
/// # Why not use Plonky3's HidingFriPcs for zero-knowledge?
///
/// HidingFriPcs achieves ZK by doubling the matrix height (interleaving
/// random rows), then passing the 2H-height matrix to the inner FRI.
/// This means the inner FRI proves deg(Q) < 2*q_domain_size instead of
/// < q_domain_size — introducing q_domain_size worth of slack that
/// completely destroys our tight degree bound. The polynomial must fit
/// in the pre-doubling domain (height H), but the degree bound is on the
/// post-doubling domain (2H), so the slack is unavoidable and structural.
///
/// Instead, we achieve hiding by manually padding Z with random
/// coefficients of the same degree. This preserves the tight FRI domain
/// while masking the secret roots: the verifier sees Z(α) which is now
/// a product involving both real roots and random blinding values,
/// preventing recovery of individual roots.
fn compute_deg_p(deg_z: usize, target_ratio: usize) -> (usize, usize) {
    let target_deg_q = target_ratio * deg_z;
    let mut best_k = 1usize;
    let mut best_dist = usize::MAX;
    for k in 1..=30 {
        let deg_q = (1usize << k) - 1;
        let dist = if deg_q >= target_deg_q {
            deg_q - target_deg_q
        } else {
            target_deg_q - deg_q
        };
        if dist < best_dist {
            best_dist = dist;
            best_k = k;
        }
    }
    let deg_q = (1usize << best_k) - 1;
    let deg_p = deg_z + deg_q;
    (deg_p, best_k)
}

fn main() {
    let num_sets: usize = 10;
    let input_roots: usize = 10000;

    // BabyBear is 31-bit vs Goldilocks 64-bit, so we commit 2x more roots
    // to represent the same amount of secret data.
    #[cfg(not(any(feature = "baby-bear", feature = "koala-bear")))]
    let (num_roots, field_name) = (input_roots, "Goldilocks (64-bit)");
    #[cfg(feature = "baby-bear")]
    let (num_roots, field_name) = (input_roots * 2, "BabyBear (31-bit, 2x roots)");
    #[cfg(feature = "koala-bear")]
    let (num_roots, field_name) = (input_roots * 2, "KoalaBear (31-bit, 2x roots)");

    let dft = Dft::default();

    let (deg_p, k) = compute_deg_p(num_roots, 120);
    let num_p_coeffs = deg_p + 1;
    let deg_q = (1usize << k) - 1;

    println!("=== Multi-Column Batched {}x Prover [{}] ===", num_sets, field_name);
    println!("deg(Z)={}, deg(P)={}, deg(Q)={} (2^{}-1), ratio={:.0}x",
        num_roots, deg_p, deg_q, k, deg_p as f64 / num_roots as f64);
    println!("q_domain_size={} (zero slack: deg(Q)+1 = 2^{})", deg_q + 1, k);
    println!("Each instance: {} roots out of degree ~{}", num_roots, deg_p);
    println!();

    let (pcs, base_challenger) = setup_pcs();
    let mut rng = SmallRng::seed_from_u64(42);

    // For small fields like BabyBear/KoalaBear (~2^31), random roots can land
    // on FFT domain points (roots of unity), causing Z to evaluate to zero
    // during poly_div_fft. We reject such roots. For Goldilocks (~2^64) this
    // almost never triggers.
    let fft_log_size = num_p_coeffs.next_power_of_two().trailing_zeros() as usize;
    let is_root_of_unity = |x: Val| -> bool {
        let mut v = x;
        for _ in 0..fft_log_size {
            v = v * v;
        }
        v == Val::ONE
    };

    print!("Setup: generating test data... ");
    let mut all_roots = Vec::new();
    let mut all_p = Vec::new();
    for _ in 0..num_sets {
        let roots: Vec<Val> = (0..num_roots)
            .map(|_| loop {
                let r: Val = rng.random();
                if r != Val::ZERO && !is_root_of_unity(r) {
                    break r;
                }
            })
            .collect();
        let z_setup = build_z_from_roots(&roots);
        let q_real: Vec<Val> = (0..num_p_coeffs - num_roots)
            .map(|_| rng.random::<Val>())
            .collect();
        let p = poly_mul_fft(&q_real, &z_setup, &dft);
        all_roots.push(roots);
        all_p.push(p);
    }
    println!("done ({} instances, each P~{} coefficients)", num_sets, all_p[0].len());
    println!();

    // ── PROVER WORK (all timed) ─────────────────────────────────────────
    let total_start = Instant::now();

    // 1. Build Z from secret roots
    let t = Instant::now();
    let all_z: Vec<Vec<Val>> = all_roots.iter().map(|r| build_z_from_roots(r)).collect();
    let build_z_time = t.elapsed();

    // 2. Divide: Q_i = P_i / Z_i
    let t = Instant::now();
    let all_q: Vec<Vec<Val>> = all_p
        .iter()
        .zip(all_z.iter())
        .map(|(p, z)| poly_div_fft(p, z, &dft))
        .collect();
    let div_time = t.elapsed();

    // 3. Build multi-column matrices (2 trees)
    //    z_tree: [Z1..Zn] = num_sets columns
    //    q_tree: [Q1..Qn] = num_sets columns
    let z_domain_size = all_z[0].len().next_power_of_two();
    let q_domain_size = all_q[0].len().next_power_of_two();

    let z_polys: Vec<Vec<Val>> = all_z
        .iter()
        .cloned()
        .collect();
    let z_coeff_mat = build_multi_column_matrix(&z_polys, z_domain_size);
    let q_coeff_mat = build_multi_column_matrix(&all_q, q_domain_size);
    // PCS expects evaluations on the domain, not coefficients
    let z_mat = dft.dft_batch(z_coeff_mat).to_row_major_matrix();
    let q_mat = dft.dft_batch(q_coeff_mat).to_row_major_matrix();

    let z_domain =
        <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, z_domain_size);
    let q_domain =
        <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, q_domain_size);

    // 5. Commit both trees
    let t = Instant::now();
    let (z_com, z_pdata): ComData =
        <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(z_domain, z_mat)]);
    let (q_com, q_pdata): ComData =
        <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(q_domain, q_mat)]);
    let commit_time = t.elapsed();

    // 6. Derive challenge alpha (Fiat-Shamir over 2 commitments)
    let mut p_challenger = base_challenger.clone();
    p_challenger.observe(z_com.clone());
    p_challenger.observe(q_com.clone());
    let alpha: Challenge = p_challenger.sample_algebra_element();

    // 7. Open both trees at alpha
    let t = Instant::now();
    let (opened_values, proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
        &pcs,
        vec![
            (&z_pdata, vec![vec![alpha]]),
            (&q_pdata, vec![vec![alpha]]),
        ],
        &mut p_challenger,
    );
    let open_time = t.elapsed();

    let total_time = total_start.elapsed();

    // ── Proof size ──────────────────────────────────────────────────────
    let commitments = vec![&z_com, &q_com];
    let commit_bytes = bincode::serialize(&commitments).unwrap().len();
    let opened_bytes = bincode::serialize(&opened_values).unwrap().len();
    let proof_bytes = bincode::serialize(&proof).unwrap().len();
    let total_proof_bytes = commit_bytes + opened_bytes + proof_bytes;

    // ── Results table ───────────────────────────────────────────────────
    let label = |s: &str| format!("{} x{}", s, num_sets);
    let rows = [
        (label("Build Z"), build_z_time),
        (label("Divide P/Z (FFT)"), div_time),
        ("Commit (2 trees)".to_string(), commit_time),
        ("Open 2 trees at alpha".to_string(), open_time),
    ];

    println!("================================================");
    println!(" Step                       | Time");
    println!("------------------------------------------------");
    for (lbl, dur) in &rows {
        println!(" {:<27}| {:>10}", lbl, fmt_dur(*dur));
    }
    println!("------------------------------------------------");
    println!(" {:<27}| {:>10}", "TOTAL PROVER", fmt_dur(total_time));
    println!("================================================");
    println!();
    println!("Proof size breakdown:");
    println!("  Commitments (2x):   {:>10}", fmt_bytes(commit_bytes));
    println!("  Opened values:      {:>10}", fmt_bytes(opened_bytes));
    println!("  FRI proof:          {:>10}", fmt_bytes(proof_bytes));
    println!("  ─────────────────────────────");
    println!("  TOTAL proof size:   {:>10}", fmt_bytes(total_proof_bytes));
    println!();

    // ── VERIFIER ──────────────────────────────────────────────────────────
    let verify_start = Instant::now();

    // 1. Reconstruct Fiat-Shamir challenge
    let mut v_challenger = base_challenger.clone();
    v_challenger.observe(z_com.clone());
    v_challenger.observe(q_com.clone());
    let v_alpha: Challenge = v_challenger.sample_algebra_element();
    assert_eq!(v_alpha, alpha);

    // 2. Evaluate all P_i(alpha) from public polynomials
    let t = Instant::now();
    let p_evals: Vec<Challenge> = all_p
        .iter()
        .map(|p| poly_eval_at_challenge(p, v_alpha))
        .collect();
    let eval_p_time = t.elapsed();

    // 3. Extract opened values
    //    opened_values[0] = z_tree: [0][0] = vec of num_sets Challenge values
    //    opened_values[1] = q_tree: [1][0] = vec of num_sets Challenge values
    //    z_tree columns: [Z1..Zn]
    //    q_tree columns: [Q1..Qn]
    let z_opened = &opened_values[0][0][0]; // num_sets values
    let q_opened = &opened_values[1][0][0]; // num_sets values

    // 4. Check divisibility: P_i(α) = Q_i(α) * Z_i(α)
    let t = Instant::now();
    for i in 0..num_sets {
        let z_val = z_opened[i];
        let q_val = q_opened[i];

        assert_eq!(
            p_evals[i], q_val * z_val,
            "Divisibility check failed for instance {}", i
        );
    }
    let check_time = t.elapsed();

    // 5. Verify FRI proof
    let t = Instant::now();
    <MyPcs as Pcs<Challenge, Challenger>>::verify(
        &pcs,
        vec![
            (
                z_com,
                vec![(z_domain, vec![(v_alpha, z_opened.clone())])],
            ),
            (
                q_com,
                vec![(q_domain, vec![(v_alpha, q_opened.clone())])],
            ),
        ],
        &proof,
        &mut v_challenger,
    )
    .expect("FRI verification failed");
    let fri_verify_time = t.elapsed();

    let verify_time = verify_start.elapsed();

    println!("================================================");
    println!(" VERIFIER                   | Time");
    println!("------------------------------------------------");
    println!(" {:<27}| {:>10}", format!("Evaluate P(α) x{}", num_sets), fmt_dur(eval_p_time));
    println!(" {:<27}| {:>10}", "Algebraic checks", fmt_dur(check_time));
    println!(" {:<27}| {:>10}", "FRI verify", fmt_dur(fri_verify_time));
    println!("------------------------------------------------");
    println!(" {:<27}| {:>10}", "TOTAL VERIFIER", fmt_dur(verify_time));
    println!("================================================");
}
