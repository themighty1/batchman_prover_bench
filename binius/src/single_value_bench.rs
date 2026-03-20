//! Benchmark: single row, 40 standalone columns, a==b constraint.

use std::time::Instant;
use anyhow::Result;

use binius_compute::cpu::alloc::CpuComputeAllocator;
use binius_compute::cpu::layer::CpuLayerHolder;
use binius_compute::ComputeHolder;
use binius_core::{
    constraint_system::{self, FriStrategy, Proof},
    fiat_shamir::HasherChallenger,
};
use binius_field::{
    arch::OptimalUnderlier, as_packed_field::PackedType, tower::CanonicalTowerFamily,
};
use binius_hal::make_portable_backend;
use binius_m3::builder::{Col, ConstraintSystem, WitnessIndex, B128};
use memory_checker_and_lookup::{Blake3Digest, Blake3Compression};
use rand::{rngs::StdRng, Rng, SeedableRng};

type P = PackedType<OptimalUnderlier, B128>;

const NUM_COLS: usize = 4000;
const LOG_INV_RATE: usize = 1;
const SECURITY_BITS: usize = 100;
const FRI_STRATEGY: FriStrategy = FriStrategy::ConstantArity(1);

fn main() -> Result<()> {
    println!("=== Single-row benchmark ===");
    println!("  Rows: 1");
    println!("  Columns: {} standalone + a, b, delta, fri_blind, fri_blind_sq", NUM_COLS);

    let mut cs = ConstraintSystem::new();
    let mut table = cs.add_table("test");

    let col_a: Col<B128> = table.add_committed_in_group("a", 1);
    let col_b: Col<B128> = table.add_committed_in_group("b", 2);
    let c_delta: Col<B128> = table.add_committed("delta");
    let c_fri_blind: Col<B128> = table.add_committed("fri_blind");
    let c_fri_blind_sq: Col<B128> = table.add_committed("fri_blind_sq");
    table.assert_zero("sumcheck_blind", c_fri_blind * c_fri_blind - c_fri_blind_sq);
    table.assert_zero("a_eq_b", col_a - col_b);

    let mut standalone_cols: Vec<Col<B128>> = Vec::new();
    for i in 0..NUM_COLS {
        standalone_cols.push(table.add_committed_in_group(
            &format!("s_{i}"), (100 + i) as u32));
    }

    let table_id = table.id();
    drop(table);

    let mut allocator = CpuComputeAllocator::new(1 << 24);
    let allocator = allocator.into_bump_allocator();
    let mut witness = WitnessIndex::<P>::new(&cs, &allocator);

    let tw = witness.init_table(table_id, 1).expect("init table");
    let seg = tw.full_segment();
    let mut rng = StdRng::seed_from_u64(42);

    let num_rows = 1;
    {
        let mut a_col = seg.get_scalars_mut(col_a)?;
        let mut b_col = seg.get_scalars_mut(col_b)?;
        let mut delta_col = seg.get_scalars_mut(c_delta)?;
        for i in 0..num_rows {
            let val = B128::new(rng.gen::<u128>());
            a_col[i] = val;
            b_col[i] = val;
            delta_col[i] = B128::new(0x12345678);
        }
    }
    {
        let mut blind = seg.get_scalars_mut(c_fri_blind)?;
        let mut blind_sq = seg.get_scalars_mut(c_fri_blind_sq)?;
        for i in 0..num_rows {
            let r = B128::new(rng.gen::<u128>());
            blind[i] = r;
            blind_sq[i] = r * r;
        }
    }
    for i in 0..NUM_COLS {
        let mut col = seg.get_scalars_mut(standalone_cols[i])?;
        let id = B128::new(i as u128);
        for j in 0..num_rows {
            col[j] = id;
        }
    }

    let t = Instant::now();
    let compiled_cs = cs.compile().map_err(|e| anyhow::anyhow!("{e}"))?;
    let ccs_digest = compiled_cs.digest::<Blake3Digest>();
    let table_sizes = witness.table_sizes();
    let witness_mle = witness.into_multilinear_extension_index();
    println!("  Compile:  {:?}", t.elapsed());

    let t = Instant::now();
    let mut compute_holder = CpuLayerHolder::<B128>::new(1 << 22, 1 << 27);
    let proof = constraint_system::prove::<
        _, OptimalUnderlier, CanonicalTowerFamily,
        Blake3Digest, Blake3Compression,
        HasherChallenger<Blake3Digest>, _, _, _,
    >(
        &mut compute_holder.to_data(),
        &compiled_cs, LOG_INV_RATE, SECURITY_BITS, &FRI_STRATEGY,
        &ccs_digest, &[], &table_sizes, witness_mle,
        &make_portable_backend(),
    )?;
    let prove_time = t.elapsed();
    let proof_size = proof.get_proof_size();
    println!("  Prove:    {:?}", prove_time);
    println!("  Proof:    {} bytes ({:.1} KB)", proof_size, proof_size as f64 / 1024.0);

    let t = Instant::now();
    constraint_system::verify::<
        OptimalUnderlier, CanonicalTowerFamily,
        Blake3Digest, Blake3Compression,
        HasherChallenger<Blake3Digest>,
    >(
        &compiled_cs, LOG_INV_RATE, SECURITY_BITS, &FRI_STRATEGY,
        &ccs_digest, &[],
        Proof { transcript: proof.transcript },
    )?;
    println!("  Verify:   {:?}", t.elapsed());

    println!("\n=== PASSED ===");
    Ok(())
}
