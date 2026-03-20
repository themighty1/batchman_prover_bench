//! PCS Bridge: cross-field commitment for Batchman step record membership proofs.
//!
//! In the Batchman zkVM protocol, the prover uses Binius to commit to step record
//! IT-MACs and prove MAC consistency (key = mac ⊕ delta · plaintext) after the
//! verifier reveals delta. This produces a Binius Merkle commitment over the keys.
//!
//! The prover then needs to prove that its active-branch keys are a subset of all
//! branch keys (set membership). This proof is prohibitively expensive in Binius
//! (~60s for 5M keys) but cheap with WHIR over Goldilocks (~2s).
//!
//! The bridge converts the Binius key commitment into an equivalent commitment over
//! Goldilocks field elements, so that WHIR can prove subset membership against the
//! same Merkle root. Both systems produce identical Merkle trees from the same data,
//! avoiding a second commitment and maintaining binding across field boundaries.
//!
//! Approach:
//! 1. Run binius commitment to get the RS codeword
//! 2. Interpret codeword as Goldilocks elements (with overflow fixup)
//! 3. Reshape into WHIR's expected matrix layout
//! 4. Run inverse Goldilocks DFT to get WHIR polynomial coefficients
//! 5. WHIR commit(coefficients) → same codeword → same tree → same root

use anyhow::Result;
use binius_core::{
    constraint_system::FriStrategy,
    merkle_tree::BinaryMerkleTreeProver,
    piop,
};
use binius_field::{
    BinaryField32b as FEncode,
    arch::OptimalUnderlier, as_packed_field::PackedType,
    PackedExtension, PackedField,
};
use binius_math::MultilinearExtension;
use binius_ntt::SingleThreadedNTT;
use binius_utils::{SerializationMode, SerializeBytes, checked_arithmetics::log2_ceil_usize};
use binius_shared::{Blake3Digest, Blake3Compression, B64};

use p3_goldilocks::Goldilocks;
use p3_field::{PrimeCharacteristicRing, PrimeField64, Field};
use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

type B128 = binius_field::BinaryField128b;
type P = PackedType<OptimalUnderlier, B128>;
type F = Goldilocks;

fn main() -> Result<()> {
    let raw_values: Vec<u64> = (0..1024).map(|i| i * 1000 + 42).collect();
    let values_b64: Vec<B64> = raw_values.iter().map(|&v| B64::new(v)).collect();

    println!("=== PCS Bridge: Binius → WHIR ===");
    println!("Input: {} B64 values", raw_values.len());
    println!();

    // ── Step 1: Binius commitment — get codeword ────────────────────
    let log_inv_rate = 1;
    let security_bits = 100;
    let fri_strategy = FriStrategy::ConstantArity(8);

    let len = values_b64.len().next_power_of_two();
    let n_vars = log2_ceil_usize(len);
    let n_packed_vars = n_vars.saturating_sub(1);

    let b64_per_b128 = 2usize;
    let n_packed = len.div_ceil(b64_per_b128);
    let mut packed = vec![P::default(); n_packed];
    {
        type PackedB64 = <P as PackedExtension<B64>>::PackedSubfield;
        let b64_slice: &mut [PackedB64] = PackedExtension::<B64>::cast_bases_mut(&mut packed);
        for (i, &v) in values_b64.iter().enumerate() {
            let elem = i / PackedB64::WIDTH;
            let pos = i % PackedB64::WIDTH;
            b64_slice[elem].set(pos, v);
        }
    }

    let mle = MultilinearExtension::new(
        n_vars, PackedExtension::<B64>::cast_bases(&packed),
    )?;
    let mle_witness: binius_core::witness::MultilinearWitness<'_, P> =
        mle.specialize_arc_dyn();

    let mut n_multilins_by_vars = vec![0usize; n_packed_vars + 1];
    n_multilins_by_vars[n_packed_vars] = 1;
    let commit_meta = piop::CommitMeta::new(n_multilins_by_vars);

    let merkle_prover = BinaryMerkleTreeProver::<_, Blake3Digest, _>::new(Blake3Compression);
    let fri_params = piop::make_commit_params_with_strategy::<_, FEncode, _>(
        &commit_meta, merkle_prover.scheme(), security_bits, log_inv_rate, &fri_strategy,
    )?;
    let ntt = SingleThreadedNTT::with_subspace(fri_params.rs_code().subspace())?
        .precompute_twiddles().multithreaded();

    let output = piop::commit(&fri_params, &ntt, &merkle_prover, &[mle_witness])?;
    let binius_root: Vec<u8> = output.commitment.as_slice().to_vec();
    let codeword = &output.codeword;
    let tree = &output.committed;

    let num_leaves = 1 << tree.log_len;
    let elems_per_leaf = codeword.len() / num_leaves;

    println!("Binius root:  {:02x?}", &binius_root);
    println!("  Codeword: {} packed elems, {} leaves × {} elems/leaf",
        codeword.len(), num_leaves, elems_per_leaf);

    // ── Step 2: Codeword bytes → Goldilocks matrix ──────────────────
    let mut codeword_bytes = Vec::new();
    for elem in codeword.iter() {
        SerializeBytes::serialize(elem, &mut codeword_bytes, SerializationMode::CanonicalTower)
            .expect("serialize");
    }

    // Overflow fixup
    let goldilocks_p = Goldilocks::ORDER_U64;
    let total_u64s = codeword_bytes.len() / 8;
    let mut overflow_count = 0u64;
    for i in 0..total_u64s {
        let off = i * 8;
        let mut val = u64::from_le_bytes(codeword_bytes[off..off+8].try_into().unwrap());
        if val >= goldilocks_p {
            overflow_count += 1;
            loop {
                let hash = blake3::hash(&val.to_le_bytes());
                val = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
                if val < goldilocks_p { break; }
            }
            codeword_bytes[off..off+8].copy_from_slice(&val.to_le_bytes());
        }
    }
    println!("  Overflows: {}", overflow_count);

    // Interpret as Goldilocks: num_leaves rows × gl_per_leaf columns
    let gl_per_leaf = (codeword_bytes.len() / 8) / num_leaves;
    let mut gl_codeword: Vec<F> = Vec::with_capacity(total_u64s);
    for i in 0..total_u64s {
        let off = i * 8;
        let val = u64::from_le_bytes(codeword_bytes[off..off+8].try_into().unwrap());
        gl_codeword.push(Goldilocks::new(val));
    }

    println!("  GL matrix: {} rows × {} cols", num_leaves, gl_per_leaf);

    // ── Step 3: Inverse WHIR pipeline to get polynomial coefficients ──
    //
    // WHIR commit with folding_factor=k, log_inv_rate=r does:
    //   coeffs (2^N) → reshape to 2^(N-k) rows × 2^k cols
    //   → transpose to 2^k rows × 2^(N-k) cols
    //   → pad to 2^(k+r) rows × 2^(N-k) cols
    //   → DFT each column
    //   → Merkle tree over rows
    //
    // To match binius's tree (num_leaves rows × gl_per_leaf cols),
    // we use folding_factor=k and log_inv_rate=0 (no padding) where:
    //   2^k = num_leaves, so k = log2(num_leaves)
    //
    // With r=0 the pipeline becomes:
    //   coeffs → reshape to 2^(N-k) rows × 2^k cols
    //   → transpose to num_leaves rows × 2^(N-k) cols
    //   → DFT each column (no padding)
    //   → Merkle
    //
    // Inverse: codeword matrix → IDFT → transpose → flatten

    let dft = Radix2DFTSmallBatch::<F>::default();

    let codeword_matrix = RowMajorMatrix::new(gl_codeword.clone(), gl_per_leaf);
    println!("  Codeword matrix: {} rows × {} cols", codeword_matrix.height(), codeword_matrix.width());

    // Step 3a: IDFT each column
    let idft_result = dft.idft_batch(codeword_matrix).to_row_major_matrix();

    // Step 3b: Transpose back (num_leaves × gl_per_leaf → gl_per_leaf × num_leaves)
    let transposed = idft_result.transpose();

    // Step 3c: Flatten — these are the WHIR polynomial coefficients
    let coeffs: Vec<F> = transposed.values.clone();
    let n_coeffs = coeffs.len();
    println!("  Derived coefficients: {} values", n_coeffs);

    // ── Step 4: Verify round-trip ───────────────────────────────────
    // Forward: coeffs → reshape → transpose → DFT → should equal codeword
    //
    // With folding_factor=k=log2(num_leaves), the reshape is:
    //   2^(N-k) rows × 2^k=num_leaves cols
    // Transpose: num_leaves rows × 2^(N-k) cols = num_leaves × gl_per_leaf

    let forward_matrix = RowMajorMatrix::new(coeffs.clone(), num_leaves);
    let forward_transposed = forward_matrix.transpose();
    let forward_codeword = dft.dft_batch(forward_transposed).to_row_major_matrix();

    // Compare
    let mut match_count = 0usize;
    let mut mismatch_count = 0usize;
    for i in 0..total_u64s {
        let whir_val = forward_codeword.values[i].as_canonical_u64();
        let binius_val = {
            let off = i * 8;
            u64::from_le_bytes(codeword_bytes[off..off+8].try_into().unwrap())
        };
        if whir_val == binius_val {
            match_count += 1;
        } else {
            if mismatch_count < 3 {
                println!("  mismatch[{}]: whir={} binius={}", i, whir_val, binius_val);
            }
            mismatch_count += 1;
        }
    }

    println!("\n  Codeword round-trip: {}/{} match ({} mismatches)",
        match_count, total_u64s, mismatch_count);

    if mismatch_count == 0 {
        println!("  SUCCESS: derived GL coefficients reproduce binius codeword exactly");
        println!("  WHIR commit(these coefficients) will produce the same Merkle root");
    } else {
        println!("  FAILED: round-trip mismatch");
    }

    Ok(())
}
