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
    //                       = 4 * 28 + 16 = 128 bits
    let fri_params = FriParameters {
        log_blowup: 4,
        log_final_poly_len: 0,
        num_queries: 28,
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

/// O(n log²n) polynomial-from-roots via divide-and-conquer + FFT multiply.
/// Falls back to naive O(n²) for small inputs where FFT overhead dominates.
fn build_z_from_roots_fast(roots: &[Val], dft: &Dft) -> Vec<Val> {
    if roots.len() <= 64 {
        return build_z_from_roots(roots);
    }
    let mid = roots.len() / 2;
    let left = build_z_from_roots_fast(&roots[..mid], dft);
    let right = build_z_from_roots_fast(&roots[mid..], dft);
    poly_mul_fft(&left, &right, dft)
}

/// O(n log²n) polynomial-from-roots using batched FFT at each tree level.
/// At each level, all pair multiplications have the same FFT size, so we pack
/// them into a single matrix and use dft_batch/idft_batch instead of individual
/// dft/idft calls. This improves cache utilization and SIMD throughput.
fn build_z_from_roots_batched(roots: &[Val], dft: &Dft) -> Vec<Val> {
    if roots.is_empty() {
        return vec![Val::ONE];
    }

    let base_size = 64;

    // Level 0: build base-case polynomials from chunks of roots
    let mut polys: Vec<Vec<Val>> = roots.chunks(base_size)
        .map(|chunk| build_z_from_roots(chunk))
        .collect();

    // Iteratively combine pairs level-by-level
    while polys.len() > 1 {
        let num_pairs = polys.len() / 2;
        let odd = polys.len() % 2 == 1;

        if num_pairs == 0 {
            break;
        }

        // All left polys are at even indices, rights at odd indices.
        // Find max lengths to determine FFT size for this level.
        let mut max_left_len = 0usize;
        let mut max_right_len = 0usize;
        for i in 0..num_pairs {
            max_left_len = max_left_len.max(polys[2 * i].len());
            max_right_len = max_right_len.max(polys[2 * i + 1].len());
        }
        let result_len = max_left_len + max_right_len - 1;
        let fft_size = result_len.next_power_of_two();

        // Pack lefts and rights into matrices (fft_size rows × num_pairs columns)
        let mut left_data = vec![Val::ZERO; fft_size * num_pairs];
        let mut right_data = vec![Val::ZERO; fft_size * num_pairs];
        for i in 0..num_pairs {
            for (row, &val) in polys[2 * i].iter().enumerate() {
                left_data[row * num_pairs + i] = val;
            }
            for (row, &val) in polys[2 * i + 1].iter().enumerate() {
                right_data[row * num_pairs + i] = val;
            }
        }

        let left_mat = RowMajorMatrix::new(left_data, num_pairs);
        let right_mat = RowMajorMatrix::new(right_data, num_pairs);

        // Batch DFT both sides
        let left_evals = dft.dft_batch(left_mat).to_row_major_matrix();
        let right_evals = dft.dft_batch(right_mat).to_row_major_matrix();

        // Pointwise multiply
        let product_data: Vec<Val> = left_evals.values.iter()
            .zip(right_evals.values.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        let product_mat = RowMajorMatrix::new(product_data, num_pairs);

        // Batch IDFT
        let result_mat = dft.idft_batch(product_mat).to_row_major_matrix();

        // Extract result polynomials (truncate to result_len)
        let mut next_level = Vec::with_capacity(num_pairs + if odd { 1 } else { 0 });
        for col in 0..num_pairs {
            let mut poly = Vec::with_capacity(result_len);
            for row in 0..result_len {
                poly.push(result_mat.values[row * num_pairs + col]);
            }
            next_level.push(poly);
        }

        // Carry unpaired polynomial to next level
        if odd {
            next_level.push(polys.last().unwrap().clone());
        }

        polys = next_level;
    }

    polys.into_iter().next().unwrap()
}

/// Split roots into K segments, build K subproduct trees with interleaved
/// batching across all K trees at each level, then merge the K results.
/// This ensures every dft_batch call has at least K columns, giving better
/// utilization at the upper tree levels where a single tree has only 1-4 pairs.
fn build_z_from_roots_split(roots: &[Val], num_segments: usize, dft: &Dft) -> Vec<Val> {
    if roots.is_empty() {
        return vec![Val::ONE];
    }
    let k = num_segments.min(roots.len()).max(1);

    let base_size = 64;

    // Split roots into K roughly-equal segments
    let segment_size = (roots.len() + k - 1) / k;
    let segments: Vec<&[Val]> = roots.chunks(segment_size).collect();
    let actual_k = segments.len();

    // Build base-case polynomials for each segment, tagged with segment index
    // polys[i] = (segment_id, polynomial)
    let mut polys: Vec<(usize, Vec<Val>)> = Vec::new();
    for (seg_id, segment) in segments.iter().enumerate() {
        for chunk in segment.chunks(base_size) {
            polys.push((seg_id, build_z_from_roots(chunk)));
        }
    }

    // Group by segment and combine level-by-level.
    // At each level, we batch all pairs from ALL segments into one dft_batch call.
    loop {
        // Group polys by segment
        let mut by_segment: Vec<Vec<Vec<Val>>> = vec![Vec::new(); actual_k];
        for (seg_id, poly) in polys {
            by_segment[seg_id].push(poly);
        }

        // Check if all segments are done (each has exactly 1 poly)
        let all_done = by_segment.iter().all(|s| s.len() <= 1);
        if all_done {
            // Collect the K result polynomials and merge them
            let results: Vec<Vec<Val>> = by_segment.into_iter()
                .filter_map(|mut s| if s.is_empty() { None } else { Some(s.remove(0)) })
                .collect();
            return merge_polys(&results, dft);
        }

        // Collect all pairs across all segments for this level
        let mut lefts: Vec<(usize, &[Val])> = Vec::new();
        let mut rights: Vec<(usize, &[Val])> = Vec::new();
        let mut carried: Vec<(usize, Vec<Val>)> = Vec::new();

        for (seg_id, seg_polys) in by_segment.iter().enumerate() {
            let mut i = 0;
            while i + 1 < seg_polys.len() {
                lefts.push((seg_id, &seg_polys[i]));
                rights.push((seg_id, &seg_polys[i + 1]));
                i += 2;
            }
            if seg_polys.len() % 2 == 1 {
                carried.push((seg_id, seg_polys.last().unwrap().clone()));
            }
        }

        if lefts.is_empty() {
            // No pairs to combine, all carried — shouldn't happen if !all_done
            polys = carried;
            continue;
        }

        let num_pairs = lefts.len();

        // Find max lengths for FFT size
        let max_left_len = lefts.iter().map(|(_, p)| p.len()).max().unwrap();
        let max_right_len = rights.iter().map(|(_, p)| p.len()).max().unwrap();
        let result_len = max_left_len + max_right_len - 1;
        let fft_size = result_len.next_power_of_two();

        // Pack into matrices
        let mut left_data = vec![Val::ZERO; fft_size * num_pairs];
        let mut right_data = vec![Val::ZERO; fft_size * num_pairs];
        for (col, ((_, l), (_, r))) in lefts.iter().zip(rights.iter()).enumerate() {
            for (row, &val) in l.iter().enumerate() {
                left_data[row * num_pairs + col] = val;
            }
            for (row, &val) in r.iter().enumerate() {
                right_data[row * num_pairs + col] = val;
            }
        }

        let left_mat = RowMajorMatrix::new(left_data, num_pairs);
        let right_mat = RowMajorMatrix::new(right_data, num_pairs);

        let left_evals = dft.dft_batch(left_mat).to_row_major_matrix();
        let right_evals = dft.dft_batch(right_mat).to_row_major_matrix();

        let product_data: Vec<Val> = left_evals.values.iter()
            .zip(right_evals.values.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        let product_mat = RowMajorMatrix::new(product_data, num_pairs);
        let result_mat = dft.idft_batch(product_mat).to_row_major_matrix();

        // Extract results with segment tags
        polys = Vec::with_capacity(num_pairs + carried.len());
        for (col, (seg_id, _)) in lefts.iter().enumerate() {
            let mut poly = Vec::with_capacity(result_len);
            for row in 0..result_len {
                poly.push(result_mat.values[row * num_pairs + col]);
            }
            polys.push((*seg_id, poly));
        }
        polys.extend(carried);
    }
}

/// Merge K polynomials into one via batched binary-tree multiplication.
fn merge_polys(polys: &[Vec<Val>], dft: &Dft) -> Vec<Val> {
    if polys.is_empty() {
        return vec![Val::ONE];
    }
    if polys.len() == 1 {
        return polys[0].clone();
    }

    let mut current: Vec<Vec<Val>> = polys.to_vec();
    while current.len() > 1 {
        let num_pairs = current.len() / 2;
        let odd = current.len() % 2 == 1;

        if num_pairs == 0 {
            break;
        }

        let mut max_left_len = 0usize;
        let mut max_right_len = 0usize;
        for i in 0..num_pairs {
            max_left_len = max_left_len.max(current[2 * i].len());
            max_right_len = max_right_len.max(current[2 * i + 1].len());
        }
        let result_len = max_left_len + max_right_len - 1;
        let fft_size = result_len.next_power_of_two();

        let mut left_data = vec![Val::ZERO; fft_size * num_pairs];
        let mut right_data = vec![Val::ZERO; fft_size * num_pairs];
        for i in 0..num_pairs {
            for (row, &val) in current[2 * i].iter().enumerate() {
                left_data[row * num_pairs + i] = val;
            }
            for (row, &val) in current[2 * i + 1].iter().enumerate() {
                right_data[row * num_pairs + i] = val;
            }
        }

        let left_mat = RowMajorMatrix::new(left_data, num_pairs);
        let right_mat = RowMajorMatrix::new(right_data, num_pairs);

        let left_evals = dft.dft_batch(left_mat).to_row_major_matrix();
        let right_evals = dft.dft_batch(right_mat).to_row_major_matrix();

        let product_data: Vec<Val> = left_evals.values.iter()
            .zip(right_evals.values.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        let product_mat = RowMajorMatrix::new(product_data, num_pairs);
        let result_mat = dft.idft_batch(product_mat).to_row_major_matrix();

        let mut next = Vec::with_capacity(num_pairs + if odd { 1 } else { 0 });
        for col in 0..num_pairs {
            let mut poly = Vec::with_capacity(result_len);
            for row in 0..result_len {
                poly.push(result_mat.values[row * num_pairs + col]);
            }
            next.push(poly);
        }
        if odd {
            next.push(current.last().unwrap().clone());
        }
        current = next;
    }
    current.into_iter().next().unwrap()
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

/// Find the closest sweet spot to a target Z root count for a given branch count
/// and number of committed Q splits.
///
/// With num_splits=K, the prover commits K polynomials Q1..QK, each of degree
/// 2^j - 1, in a single FRI commitment over domain 2^j. Total Q capacity =
/// K * (2^j - 1). The sweet spot maximizes Z such that ratio * Z <= K * (2^j - 1).
///
/// Returns (sweet_z, j, deg_qi, dummy_count):
///   - sweet_z: the optimal Z root count
///   - j: FRI domain exponent per split (domain = 2^j)
///   - deg_qi: degree per Qi = 2^j - 1
///   - dummy_count: total extra roots across all K splits for the degree bound
fn find_sweet_spot(target_z: usize, branch_count: usize, num_splits: usize) -> (usize, usize, usize, usize) {
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
    let num_sets: usize = 7;
    let target_z: usize = 16000;
    let branch_count: usize = 132;
    // K: number of committed Q polynomials (split-commit).
    // At 100 KHz proving rate with branch_count=132, K=32 with 8K roots (13 sets)
    // gives the best KHz/proof-bytes ratio: 370 KB, 20s (log_blowup=3).
    // Further gains may be possible with more fine-tuning of K, log_blowup, pow_bits.
    let num_splits: usize = 256;

    // Find optimal Z root count (sweet spot) for this branch count and split count
    let (num_roots, j, deg_qi, dummy_count) = find_sweet_spot(target_z, branch_count, num_splits);
    let num_q_roots_total = num_splits * deg_qi; // total Q roots across all K splits
    let deg_p = num_roots + num_q_roots_total;

    #[cfg(not(any(feature = "baby-bear", feature = "koala-bear")))]
    let field_name = "Goldilocks (64-bit)";
    #[cfg(feature = "baby-bear")]
    let field_name = "BabyBear (31-bit)";
    #[cfg(feature = "koala-bear")]
    let field_name = "KoalaBear (31-bit)";

    println!("=== Split-Commit {}x Prover [{}] ===", num_sets, field_name);
    println!("branch_count={}, target_z={} -> sweet_z={}", branch_count, target_z, num_roots);
    println!("num_splits(K)={}, deg(Qi)={} (2^{}-1), q_domain=2^{}", num_splits, deg_qi, j, j);
    println!("deg(Z)={}, deg(P)={}, total Q roots={}, dummies={}",
        num_roots, deg_p, num_q_roots_total, dummy_count);
    println!("Q columns: {}x{} = {}", num_splits, num_sets, num_splits * num_sets);
    println!();

    let dft = Dft::default();
    let (pcs, base_challenger) = setup_pcs();
    let mut rng = SmallRng::seed_from_u64(42);

    let fft_log_size = (deg_p + 1).next_power_of_two().trailing_zeros() as usize;
    let is_root_of_unity = |x: Val| -> bool {
        let mut v = x;
        for _ in 0..fft_log_size {
            v = v * v;
        }
        v == Val::ONE
    };

    print!("Setup: generating test data... ");
    let mut all_z_roots = Vec::new();
    let mut all_q_root_segments: Vec<Vec<Vec<Val>>> = Vec::new(); // [instance][split][root]
    let mut all_p_roots = Vec::new();
    for _ in 0..num_sets {
        // P's roots = branch MAC keys + dummy roots for degree bound
        let p_roots: Vec<Val> = (0..deg_p)
            .map(|_| loop {
                let r: Val = rng.random();
                if r != Val::ZERO && !is_root_of_unity(r) {
                    break r;
                }
            })
            .collect();
        // Z's roots = prover's secret subset (first num_roots of P's roots)
        let z_roots: Vec<Val> = p_roots[..num_roots].to_vec();
        // Q's roots split into K segments of deg_qi each
        let q_segments: Vec<Vec<Val>> = (0..num_splits)
            .map(|s| p_roots[num_roots + s * deg_qi .. num_roots + (s + 1) * deg_qi].to_vec())
            .collect();
        all_p_roots.push(p_roots);
        all_z_roots.push(z_roots);
        all_q_root_segments.push(q_segments);
    }
    println!("done ({} instances, {} P roots, {} Z roots, {}x{} Q roots/split)",
        num_sets, deg_p, num_roots, num_splits, deg_qi);
    println!();

    // Z is built during the Batchman protocol run, before the verifier reveals MAC keys.
    let all_z: Vec<Vec<Val>> = all_z_roots.iter()
        .map(|r| build_z_from_roots_batched(r, &dft))
        .collect();

    // ── PROVER WORK (all timed) ─────────────────────────────────────────
    let total_start = Instant::now();

    // 1. Build K Qi polynomials per instance (no merge across splits!)
    //    Each Qi built from deg_qi roots using √n segmented construction.
    //    Truncate to deg_qi+1 coeffs (merge_polys may add a trailing zero).
    let t = Instant::now();
    let all_q_polys: Vec<Vec<Vec<Val>>> = all_q_root_segments.iter()
        .map(|segments| {
            segments.iter().map(|roots| {
                let num_seg = (roots.len() as f64).sqrt().max(1.0) as usize;
                let mut poly = build_z_from_roots_split(roots, num_seg, &dft);
                poly.truncate(deg_qi + 1);
                poly
            }).collect()
        })
        .collect();
    let build_q_time = t.elapsed();

    // 2. Build multi-column matrices (2 trees)
    //    z_tree: num_sets columns in z_domain
    //    q_tree: K * num_sets columns in q_domain (smaller!)
    let z_domain_size = all_z[0].len().next_power_of_two();
    let q_domain_size = 1usize << j; // = deg_qi + 1, exact zero-slack

    // Z matrix: num_sets columns
    let z_coeff_mat = build_multi_column_matrix(&all_z, z_domain_size);

    // Q matrix: K * num_sets columns, ordered [Q_0_0, Q_0_1, .., Q_0_{K-1}, Q_1_0, ..]
    let mut q_polys_flat: Vec<Vec<Val>> = Vec::with_capacity(num_splits * num_sets);
    for inst in 0..num_sets {
        for split in 0..num_splits {
            q_polys_flat.push(all_q_polys[inst][split].clone());
        }
    }
    let q_coeff_mat = build_multi_column_matrix(&q_polys_flat, q_domain_size);

    // PCS expects evaluations on the domain, not coefficients
    let z_mat = dft.dft_batch(z_coeff_mat).to_row_major_matrix();
    let q_mat = dft.dft_batch(q_coeff_mat).to_row_major_matrix();

    let z_domain =
        <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, z_domain_size);
    let q_domain =
        <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, q_domain_size);

    // 3. Commit both trees (Q tree has K * num_sets columns in domain 2^j)
    let t = Instant::now();
    let (z_com, z_pdata): ComData =
        <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(z_domain, z_mat)]);
    let (q_com, q_pdata): ComData =
        <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(q_domain, q_mat)]);
    let commit_time = t.elapsed();

    // 4. Derive challenge alpha (Fiat-Shamir over 2 commitments)
    let mut p_challenger = base_challenger.clone();
    p_challenger.observe(z_com.clone());
    p_challenger.observe(q_com.clone());
    let alpha: Challenge = p_challenger.sample_algebra_element();

    // 5. Open both trees at alpha
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
        (label("Build Q splits"), build_q_time),
        (format!("Commit (Z + Q[{}col])", num_splits * num_sets), commit_time),
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

    // 2. Evaluate all P_i(alpha) from roots: P(α) = ∏(α - r_i)
    let t = Instant::now();
    let p_evals: Vec<Challenge> = all_p_roots
        .iter()
        .map(|roots| {
            roots.iter().fold(Challenge::ONE, |acc, &r| {
                acc * (v_alpha - Challenge::from(r))
            })
        })
        .collect();
    let eval_p_time = t.elapsed();

    // 3. Extract opened values
    //    z_opened: num_sets values
    //    q_opened: K * num_sets values [Q_0_0, Q_0_1, .., Q_0_{K-1}, Q_1_0, ..]
    let z_opened = &opened_values[0][0][0];
    let q_opened = &opened_values[1][0][0];

    // 4. Check divisibility: P_i(α) = Z_i(α) * ∏_s Qi_s(α)
    let t = Instant::now();
    for i in 0..num_sets {
        let z_val = z_opened[i];
        // Reconstruct Q_i(α) = Q_i_0(α) * Q_i_1(α) * ... * Q_i_{K-1}(α)
        let mut q_val = Challenge::ONE;
        for s in 0..num_splits {
            q_val = q_val * q_opened[i * num_splits + s];
        }

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
