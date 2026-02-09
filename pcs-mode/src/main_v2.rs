use std::time::Instant;

use p3_blake3::Blake3;
use p3_challenger::{CanObserve, FieldChallenger, HashChallenger, SerializingChallenger64};
use p3_commit::{ExtensionMmcs, Pcs};
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

type Val = Goldilocks;
type Challenge = BinomialExtensionField<Val, 2>;

type MyHash = SerializingHasher<Blake3>;
type MyCompress = CompressionFunctionFromHasher<Blake3, 2, 32>;
type ValMmcs = MerkleTreeMmcs<Val, u8, MyHash, MyCompress, 32>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Dft = Radix2DitParallel<Val>;
type Challenger = SerializingChallenger64<Val, HashChallenger<u8, Blake3, 32>>;
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

fn poly_trim(p: &mut Vec<Val>) {
    while p.len() > 1 && *p.last().unwrap() == Val::ZERO {
        p.pop();
    }
}

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

fn poly_derivative(p: &[Val]) -> Vec<Val> {
    if p.len() <= 1 {
        return vec![Val::ZERO];
    }
    let mut d = vec![Val::ZERO; p.len() - 1];
    for i in 1..p.len() {
        d[i - 1] = p[i] * Goldilocks::new(i as u64);
    }
    d
}

fn poly_mul_naive(a: &[Val], b: &[Val]) -> Vec<Val> {
    if (a.len() == 1 && a[0] == Val::ZERO) || (b.len() == 1 && b[0] == Val::ZERO) {
        return vec![Val::ZERO];
    }
    let mut result = vec![Val::ZERO; a.len() + b.len() - 1];
    for i in 0..a.len() {
        if a[i] == Val::ZERO {
            continue;
        }
        for j in 0..b.len() {
            result[i + j] = result[i + j] + a[i] * b[j];
        }
    }
    result
}

fn poly_sub(a: &[Val], b: &[Val]) -> Vec<Val> {
    let len = a.len().max(b.len());
    let mut result = vec![Val::ZERO; len];
    for i in 0..a.len() {
        result[i] = result[i] + a[i];
    }
    for i in 0..b.len() {
        result[i] = result[i] - b[i];
    }
    poly_trim(&mut result);
    result
}

fn poly_divmod(a: &[Val], b: &[Val]) -> (Vec<Val>, Vec<Val>) {
    let n = a.len();
    let m = b.len();
    if n < m {
        return (vec![Val::ZERO], a.to_vec());
    }
    let inv_lead = b[m - 1].inverse();
    let mut rem = a.to_vec();
    let q_len = n - m + 1;
    let mut q = vec![Val::ZERO; q_len];
    for k in (0..q_len).rev() {
        let c = rem[k + m - 1] * inv_lead;
        q[k] = c;
        for j in 0..m {
            rem[k + j] = rem[k + j] - c * b[j];
        }
    }
    if m > 1 {
        rem.truncate(m - 1);
        poly_trim(&mut rem);
    } else {
        rem = vec![Val::ZERO];
    }
    (q, rem)
}

fn poly_extended_gcd(a: &[Val], b: &[Val]) -> (Vec<Val>, Vec<Val>, Vec<Val>) {
    let mut old_r = a.to_vec();
    let mut r = b.to_vec();
    let mut old_s: Vec<Val> = vec![Val::ONE];
    let mut s: Vec<Val> = vec![Val::ZERO];
    let mut old_t: Vec<Val> = vec![Val::ZERO];
    let mut t: Vec<Val> = vec![Val::ONE];

    while !(r.len() == 1 && r[0] == Val::ZERO) {
        let (q, rem) = poly_divmod(&old_r, &r);
        old_r = r;
        r = rem;

        let qs = poly_mul_naive(&q, &s);
        let new_s = poly_sub(&old_s, &qs);
        old_s = s;
        s = new_s;

        let qt = poly_mul_naive(&q, &t);
        let new_t = poly_sub(&old_t, &qt);
        old_t = t;
        t = new_t;
    }

    let lead = *old_r.last().unwrap();
    let inv_lead = lead.inverse();
    for c in &mut old_r {
        *c = *c * inv_lead;
    }
    for c in &mut old_s {
        *c = *c * inv_lead;
    }
    for c in &mut old_t {
        *c = *c * inv_lead;
    }

    (old_r, old_s, old_t)
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
/// value (e.g. an executed instruction), and deg(P) ~ 100 * deg(Z) reflects the
/// total ISA size (~100 unique instructions/branches). The constraint
/// deg(Q) = 2^k - 1 ensures q_domain_size = 2^k has zero slack, which lets FRI
/// enforce a lower bound on deg(Z) — a cheating prover who uses fewer roots
/// would produce a Q that exceeds the degree bound.
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
    let num_sets: usize = 10_000;
    let num_roots: usize = 1;
    let dft = Dft::default();

    let (deg_p, k) = compute_deg_p(num_roots, 100);
    let num_p_coeffs = deg_p + 1;
    let deg_q = (1usize << k) - 1;

    println!("=== Multi-Column Batched {}x Prover ===", num_sets);
    println!("deg(Z)={}, deg(P)={}, deg(Q)={} (2^{}-1), ratio={:.0}x",
        num_roots, deg_p, deg_q, k, deg_p as f64 / num_roots as f64);
    println!("q_domain_size={} (zero slack: deg(Q)+1 = 2^{})", deg_q + 1, k);
    println!("Each instance: {} roots out of degree ~{}", num_roots, deg_p);
    println!();

    let (pcs, base_challenger) = setup_pcs();
    let mut rng = SmallRng::seed_from_u64(42);

    // ── Setup: N independent instances ───────────────────────────────────
    print!("Setup: generating test data... ");
    let mut all_roots = Vec::new();
    let mut all_p = Vec::new();
    for _ in 0..num_sets {
        let roots: Vec<Val> = (0..num_roots).map(|_| rng.random::<Val>()).collect();
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

    // 3. Bezout for all: A_i*Z_i + B_i*Z_i' = 1
    let t = Instant::now();
    let mut all_zp = Vec::new();
    let mut all_a = Vec::new();
    let mut all_b = Vec::new();
    for z in &all_z {
        let zp = poly_derivative(z);
        let (_gcd, a_poly, b_poly) = poly_extended_gcd(z, &zp);
        all_zp.push(zp);
        all_a.push(a_poly);
        all_b.push(b_poly);
    }
    let bezout_time = t.elapsed();

    // 4. Build multi-column matrices (2 trees instead of 15)
    //    z_tree: [Z1,Z2,Z3, Z1',Z2',Z3', A1,A2,A3, B1,B2,B3] = 12 columns
    //    q_tree: [Q1,Q2,Q3] = 3 columns
    let z_domain_size = all_z[0].len().next_power_of_two();
    let q_domain_size = all_q[0].len().next_power_of_two();

    let z_polys: Vec<Vec<Val>> = all_z
        .iter()
        .chain(all_zp.iter())
        .chain(all_a.iter())
        .chain(all_b.iter())
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
        (label("Bezout (xgcd)"), bezout_time),
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
    //    opened_values[0] = z_tree: [0][0] = vec of 4*num_sets Challenge values
    //    opened_values[1] = q_tree: [1][0] = vec of num_sets Challenge values
    //    z_tree columns: [Z1..Zn, Z1'..Zn', A1..An, B1..Bn]
    //    q_tree columns: [Q1..Qn]
    let z_opened = &opened_values[0][0][0]; // 4*num_sets values
    let q_opened = &opened_values[1][0][0]; // num_sets values

    // 4. Check algebraic relations for each instance
    let t = Instant::now();
    for i in 0..num_sets {
        let z_val = z_opened[i];
        let zp_val = z_opened[num_sets + i];
        let a_val = z_opened[2 * num_sets + i];
        let b_val = z_opened[3 * num_sets + i];
        let q_val = q_opened[i];

        // Check P_i(α) = Q_i(α) * Z_i(α)
        assert_eq!(
            p_evals[i], q_val * z_val,
            "Divisibility check failed for instance {}", i
        );

        // Check A_i(α)*Z_i(α) + B_i(α)*Z_i'(α) = 1
        assert_eq!(
            a_val * z_val + b_val * zp_val,
            Challenge::ONE,
            "Bezout check failed for instance {}", i
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
