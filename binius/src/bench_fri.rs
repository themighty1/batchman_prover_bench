//! Benchmark FRI parameter tuning for commit_column_b128.
//!
//! Sweeps target_log_dim to find optimal commit speed for 128K B128 values.
//!
//! Run: cargo run --release --bin bench_fri

use std::time::Instant;

use anyhow::Result;
use binius_core::{
    merkle_tree::BinaryMerkleTreeProver,
    piop,
    protocols::fri::{FRIParams, calculate_n_test_queries},
    reed_solomon::ReedSolomonCode,
};
use binius_field::{
    arch::OptimalUnderlier, as_packed_field::PackedType, packed::set_packed_slice,
    BinaryField32b as FEncode, PackedField,
};
use binius_math::MultilinearExtension;
use binius_ntt::SingleThreadedNTT;
use binius_utils::checked_arithmetics::log2_ceil_usize;
use memory_checker_and_lookup::{Blake3Compression, Blake3Digest, B128};
use rand::{rngs::StdRng, Rng, SeedableRng};

type P = PackedType<OptimalUnderlier, B128>;

const LOG_INV_RATE: usize = 1;
const SECURITY_BITS: usize = 100;
const NUM_VALUES: usize = 131072; // 128K

fn main() -> Result<()> {
    let mut rng = StdRng::seed_from_u64(42);
    let values: Vec<B128> = (0..NUM_VALUES)
        .map(|_| B128::new(rng.random::<u128>()))
        .collect();

    let len = values.len().next_power_of_two();
    let n_vars = log2_ceil_usize(len);
    let n_packed_vars = n_vars.saturating_sub(1);

    let mut packed = vec![P::default(); len.div_ceil(P::WIDTH)];
    for (i, &v) in values.iter().enumerate() {
        set_packed_slice(&mut packed, i, v);
    }

    let mle = MultilinearExtension::new(n_vars, packed)?;
    let mle_witness: binius_core::witness::MultilinearWitness<'_, P> =
        mle.specialize_arc_dyn();

    let mut n_multilins_by_vars = vec![0usize; n_packed_vars + 1];
    n_multilins_by_vars[n_packed_vars] = 1;
    let commit_meta = piop::CommitMeta::new(n_multilins_by_vars);
    let total_vars = commit_meta.total_vars();

    println!("Benchmarking commit_column_b128 for {} B128 values", NUM_VALUES);
    println!("  n_vars={}, n_packed_vars={}, total_vars={}", n_vars, n_packed_vars, total_vars);
    println!();
    println!("{:<12} {:<10} {:<10} {:<8} {:<14} {:<12} {:<10}",
        "target_dim", "log_batch", "log_dim", "queries", "fold_arities", "commit_ms", "proof_KB");
    println!("{}", "-".repeat(80));

    let merkle_prover = BinaryMerkleTreeProver::<_, Blake3Digest, _>::new(Blake3Compression);

    // log_batch must be >= 1 so target_log_dim <= total_vars - 1
    for target_log_dim in (4..total_vars).rev() {
        let fold_arity = 4usize;
        let manual_log_batch = total_vars.saturating_sub(target_log_dim);
        let manual_log_dim = total_vars - manual_log_batch;

        let log_ntt_size = (total_vars + LOG_INV_RATE).max(20);
        let ntt = SingleThreadedNTT::<FEncode>::new(log_ntt_size)?;
        let rs_code = ReedSolomonCode::<FEncode>::with_ntt_subspace(
            &ntt, manual_log_dim, LOG_INV_RATE,
        )?;
        let n_test_queries = calculate_n_test_queries::<B128, FEncode>(SECURITY_BITS, &rs_code)?;

        let log_msg = manual_log_dim + manual_log_batch;
        let cap_height = log2_ceil_usize(n_test_queries);

        let fold_arities: Vec<usize> = {
            let first_arity = manual_log_batch.max(fold_arity);
            let total_needed = log_msg.saturating_sub(cap_height.saturating_sub(LOG_INV_RATE));
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

        // Validate fold_arities sum < rs_code.log_dim() + log_batch
        let sum: usize = fold_arities.iter().sum();
        if sum >= rs_code.log_dim() + manual_log_batch {
            println!("{:<12} {:<10} {:<10} {:<8} SKIP (fold sum {} >= {})",
                target_log_dim, manual_log_batch, manual_log_dim,
                n_test_queries, sum, rs_code.log_dim() + manual_log_batch);
            continue;
        }

        let fri_params = match FRIParams::<B128, FEncode>::new(
            rs_code, manual_log_batch, fold_arities.clone(), n_test_queries,
        ) {
            Ok(p) => p,
            Err(e) => {
                println!("{:<12} {:<10} {:<10} {:<8} ERROR: {}",
                    target_log_dim, manual_log_batch, manual_log_dim, n_test_queries, e);
                continue;
            }
        };

        let commit_ntt = SingleThreadedNTT::with_subspace(fri_params.rs_code().subspace())?
            .precompute_twiddles()
            .multithreaded();

        // Warm up — catch panics from invalid configs
        let warmup = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            piop::commit(&fri_params, &commit_ntt, &merkle_prover, &[mle_witness.clone()])
        }));
        if warmup.is_err() {
            println!("{:<12} {:<10} {:<10} {:<8} PANIC",
                target_log_dim, manual_log_batch, manual_log_dim, n_test_queries);
            continue;
        }

        // Benchmark (3 runs)
        let mut times = Vec::new();
        for _ in 0..3 {
            let t = Instant::now();
            let output = piop::commit(&fri_params, &commit_ntt, &merkle_prover, &[mle_witness.clone()])?;
            times.push(t.elapsed());
            std::hint::black_box(output);
        }
        let best = times.iter().min().unwrap();

        // Estimate proof contribution: fold_arities determine FRI proof size
        // Each round sends 2^arity field elements + Merkle proof
        let proof_est_kb = (n_test_queries * fold_arities.len() * 32 +
            fold_arities.iter().map(|&a| (1 << a) * 16).sum::<usize>()) as f64 / 1024.0;

        let arities_str = format!("{:?}", fold_arities);
        println!("{:<12} {:<10} {:<10} {:<8} {:<14} {:<12.2} {:<10.1}",
            target_log_dim, manual_log_batch, manual_log_dim,
            n_test_queries, arities_str, best.as_secs_f64() * 1000.0, proof_est_kb);
    }

    Ok(())
}
