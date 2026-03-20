//! Integration test: cross-field subset membership proof.
//!
//! 1. Generate 10000 random B64 values as set Z (active branch MACs)
//! 2. Create set P = Z embedded in 50x larger set (all branch keys)
//! 3. Convert both sets to Goldilocks field elements
//! 4. Build polynomials: P(x) from all roots, Z(x) from active roots
//! 5. Compute Q(x) = P(x) / Z(x) — must divide cleanly
//! 6. Verify P(α) = Z(α) · Q(α) at random challenge
//!
//! This validates that binius B64 values can be used as Goldilocks roots
//! for a WHIR-based subset membership proof.

use p3_goldilocks::Goldilocks;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_field::extension::BinomialExtensionField;
use rand::{SeedableRng, rngs::SmallRng, Rng};

type F = Goldilocks;
type EF = BinomialExtensionField<F, 2>;

fn fmt_dur(d: std::time::Duration) -> String {
    if d.as_secs_f64() >= 1.0 {
        format!("{:.2}s", d.as_secs_f64())
    } else {
        format!("{:.1}ms", d.as_secs_f64() * 1000.0)
    }
}

const NUM_ACTIVE: usize = 1_000;
const BRANCH_COUNT: usize = 50;
const GOLDILOCKS_P: u64 = Goldilocks::ORDER_U64;

/// Convert a B64 value (u64) to Goldilocks, applying overflow fixup if needed.
fn b64_to_goldilocks(val: u64) -> F {
    if val < GOLDILOCKS_P {
        Goldilocks::new(val)
    } else {
        // Deterministic fixup: hash until < p
        let mut v = val;
        loop {
            let hash = blake3::hash(&v.to_le_bytes());
            v = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
            if v < GOLDILOCKS_P {
                return Goldilocks::new(v);
            }
        }
    }
}

/// Build vanishing polynomial from roots: P(x) = ∏(x - r_i)
fn build_vanishing_poly(roots: &[F]) -> Vec<F> {
    let mut coeffs = vec![F::ONE];
    for &r in roots {
        let mut new = vec![F::ZERO; coeffs.len() + 1];
        for i in 0..coeffs.len() {
            new[i + 1] = new[i + 1] + coeffs[i];
            new[i] = new[i] - r * coeffs[i];
        }
        coeffs = new;
    }
    coeffs
}

/// Evaluate polynomial at a point in extension field
fn eval_poly(coeffs: &[F], x: EF) -> EF {
    let mut result = EF::ZERO;
    for &c in coeffs.iter().rev() {
        result = result * x + EF::from(c);
    }
    result
}

/// Polynomial division: compute Q = P / Z, returns (quotient, remainder)
fn poly_div(p: &[F], z: &[F]) -> (Vec<F>, Vec<F>) {
    if z.is_empty() || p.len() < z.len() {
        return (vec![], p.to_vec());
    }
    let mut remainder = p.to_vec();
    let mut quotient = vec![F::ZERO; p.len() - z.len() + 1];
    let leading = *z.last().unwrap();
    let leading_inv = leading.inverse();

    for i in (0..quotient.len()).rev() {
        let coeff = remainder[i + z.len() - 1] * leading_inv;
        quotient[i] = coeff;
        for j in 0..z.len() {
            remainder[i + j] = remainder[i + j] - coeff * z[j];
        }
    }

    // Trim trailing zeros from remainder
    while remainder.last() == Some(&F::ZERO) {
        remainder.pop();
    }

    (quotient, remainder)
}

fn main() {
    println!("=== Cross-field subset membership test ===");
    println!("  Active set (Z):  {} elements", NUM_ACTIVE);
    println!("  Branch count:    {}", BRANCH_COUNT);
    println!("  Full set (P):    {} elements", NUM_ACTIVE * BRANCH_COUNT);
    println!();

    let mut rng = SmallRng::seed_from_u64(42);

    // Step 1: Generate random B64 values for Z (active MACs)
    let z_raw: Vec<u64> = (0..NUM_ACTIVE).map(|_| rng.gen::<u64>()).collect();

    // Step 2: Build P — for each Z value, add (BRANCH_COUNT - 1) random "inactive" values
    let mut p_raw: Vec<u64> = Vec::with_capacity(NUM_ACTIVE * BRANCH_COUNT);
    for &z_val in &z_raw {
        p_raw.push(z_val); // active branch
        for _ in 1..BRANCH_COUNT {
            p_raw.push(rng.gen::<u64>()); // inactive branches
        }
    }

    // Step 3: Convert to Goldilocks with overflow fixup
    let t = std::time::Instant::now();
    let mut overflow_count = 0u64;
    let z_gl: Vec<F> = z_raw.iter().map(|&v| {
        if v >= GOLDILOCKS_P { overflow_count += 1; }
        b64_to_goldilocks(v)
    }).collect();
    let p_gl: Vec<F> = p_raw.iter().map(|&v| {
        if v >= GOLDILOCKS_P { overflow_count += 1; }
        b64_to_goldilocks(v)
    }).collect();
    let convert_time = t.elapsed();

    // Verify Z ⊂ P at the value level
    for (i, z) in z_gl.iter().enumerate() {
        let p_idx = i * BRANCH_COUNT;
        assert_eq!(*z, p_gl[p_idx], "Z[{}] not in P at expected position", i);
    }

    // Polynomial test at full size
    {
        let test_size = NUM_ACTIVE;
        let total_size = test_size * BRANCH_COUNT;
        println!("─── {} active / {} total roots ───", test_size, total_size);

        let test_z: Vec<F> = z_gl[..test_size].to_vec();
        let test_p: Vec<F> = p_gl[..total_size].to_vec();

        let t = std::time::Instant::now();
        let z_poly = build_vanishing_poly(&test_z);
        let z_build = t.elapsed();

        let t = std::time::Instant::now();
        let p_poly = build_vanishing_poly(&test_p);
        let p_build = t.elapsed();

        let t = std::time::Instant::now();
        let (q_poly, remainder) = poly_div(&p_poly, &z_poly);
        let div_time = t.elapsed();

        assert!(remainder.is_empty(), "P / Z has non-zero remainder!");

        let t = std::time::Instant::now();
        let alpha = EF::from(Goldilocks::new(rng.gen::<u64>() % GOLDILOCKS_P));
        let p_alpha = eval_poly(&p_poly, alpha);
        let z_alpha = eval_poly(&z_poly, alpha);
        let q_alpha = eval_poly(&q_poly, alpha);
        let eval_time = t.elapsed();

        assert_eq!(p_alpha, z_alpha * q_alpha, "P(α) ≠ Z(α) · Q(α)");

        println!("  B64→GL convert:  {:>10}", fmt_dur(convert_time));
        println!("  Build Z poly:    {:>10}  ({} coeffs)", fmt_dur(z_build), z_poly.len());
        println!("  Build P poly:    {:>10}  ({} coeffs)", fmt_dur(p_build), p_poly.len());
        println!("  Q = P / Z:       {:>10}  ({} coeffs)", fmt_dur(div_time), q_poly.len());
        println!("  Eval at α:       {:>10}", fmt_dur(eval_time));
        println!("  Overflows:       {}", overflow_count);
        let total = z_build + p_build + div_time + eval_time;
        println!("  Total:           {:>10}", fmt_dur(total));
        println!("  P(α) = Z(α)·Q(α): PASSED");
        println!();
    }

    println!("=== All checks passed ===");
}
