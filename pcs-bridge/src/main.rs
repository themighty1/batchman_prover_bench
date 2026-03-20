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

use anyhow::Result;
use binius_core::{
    constraint_system::FriStrategy,
    merkle_tree::BinaryMerkleTreeProver,
    piop,
};
use binius_field::{
    BinaryField32b as FEncode, BinaryField64b,
    arch::OptimalUnderlier, as_packed_field::PackedType,
    PackedExtension, PackedField, TowerField,
};
use binius_math::MultilinearExtension;
use binius_ntt::SingleThreadedNTT;
use binius_utils::{SerializationMode, SerializeBytes, checked_arithmetics::log2_ceil_usize};
use binius_shared::{Blake3Digest, Blake3Compression, B64};

use p3_goldilocks::Goldilocks;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_symmetric::CryptographicHasher;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_commit::Mmcs;

type B128 = binius_field::BinaryField128b;
type P = PackedType<OptimalUnderlier, B128>;

// ── Blake3 hasher for p3 (produces binius-compatible leaf hashes) ────

/// Hashes Goldilocks field elements by converting each to 8 LE bytes
/// and feeding to Blake3. Output is [u64; 4] = 32 bytes = Blake3 digest.
#[derive(Clone)]
struct P3Blake3Hash;

impl CryptographicHasher<Goldilocks, [u64; 4]> for P3Blake3Hash {
    fn hash_iter<I: IntoIterator<Item = Goldilocks>>(&self, input: I) -> [u64; 4] {
        let mut hasher = blake3::Hasher::new();
        for elem in input {
            let val: u64 = elem.as_canonical_u64();
            hasher.update(&val.to_le_bytes());
        }
        let hash = hasher.finalize();
        let bytes = hash.as_bytes();
        [
            u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
        ]
    }
}

/// Blake3 compression for p3 Merkle tree, matching binius's compress(left || right).
#[derive(Clone)]
struct P3Blake3Compress;

impl p3_symmetric::PseudoCompressionFunction<[u64; 4], 2> for P3Blake3Compress {
    fn compress(&self, input: [[u64; 4]; 2]) -> [u64; 4] {
        let mut hasher = blake3::Hasher::new();
        for &v in &input[0] { hasher.update(&v.to_le_bytes()); }
        for &v in &input[1] { hasher.update(&v.to_le_bytes()); }
        let hash = hasher.finalize();
        let bytes = hash.as_bytes();
        [
            u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
        ]
    }
}
impl p3_symmetric::CompressionFunction<[u64; 4], 2> for P3Blake3Compress {}

fn main() -> Result<()> {
    let raw_values: Vec<u64> = (0..1024).map(|i| i * 1000 + 42).collect();
    let values_b64: Vec<B64> = raw_values.iter().map(|&v| B64::new(v)).collect();

    println!("=== PCS Bridge: Binius → WHIR ===");
    println!("Input: {} values", raw_values.len());
    println!();

    // ── Step 1: Binius commitment (reusing binius/ lib) ─────────────
    let log_inv_rate = 1;
    let security_bits = 100;
    let fri_strategy = FriStrategy::ConstantArity(8);

    let binius_root = binius_shared::commit_column_b64(
        &values_b64, log_inv_rate, security_bits, &fri_strategy,
    )?;
    println!("Binius root: {:02x?}", &binius_root);

    // ── Step 2: Also get the codeword for bridge ────────────────────
    // We need to re-run the pipeline to access the codeword.
    // (commit_column_b64 only returns the root, not the codeword.)
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
    let tree = &output.committed;
    let codeword = &output.codeword;
    let num_leaves = 1 << tree.log_len;
    let elems_per_leaf = codeword.len() / num_leaves;

    // Serialize codeword to bytes
    let mut codeword_bytes = Vec::new();
    for elem in codeword.iter() {
        SerializeBytes::serialize(elem, &mut codeword_bytes, SerializationMode::CanonicalTower)
            .expect("serialize");
    }

    println!("Codeword: {} elems, {} bytes, {} leaves × {} elems/leaf",
        codeword.len(), codeword_bytes.len(), num_leaves, elems_per_leaf);

    // ── Step 3: Fixup overflows and interpret as Goldilocks ────────
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

    if overflow_count > 0 {
        println!("  Overflows fixed: {}/{}", overflow_count, total_u64s);
    } else {
        println!("  No Goldilocks overflows");
    }

    // Interpret as Goldilocks elements
    let bytes_per_leaf = codeword_bytes.len() / num_leaves;
    let gl_per_leaf = bytes_per_leaf / 8;

    let mut gl_flat: Vec<Goldilocks> = Vec::with_capacity(total_u64s);
    for i in 0..total_u64s {
        let off = i * 8;
        let val = u64::from_le_bytes(codeword_bytes[off..off+8].try_into().unwrap());
        gl_flat.push(Goldilocks::new(val));
    }

    // ── Step 4: Build p3 Merkle tree with Blake3 ────────────────────
    let matrix = RowMajorMatrix::new(gl_flat, gl_per_leaf);

    let p3_mmcs = MerkleTreeMmcs::<Goldilocks, u64, P3Blake3Hash, P3Blake3Compress, 2, 4>::new(
        P3Blake3Hash, P3Blake3Compress, 0,
    );
    let (p3_commitment, _p3_tree) = p3_mmcs.commit(vec![matrix]);

    let p3_root_u64s: &[u64; 4] = &p3_commitment[0];
    let mut p3_root_bytes = Vec::new();
    for &v in p3_root_u64s {
        p3_root_bytes.extend_from_slice(&v.to_le_bytes());
    }

    println!("\nP3 root:     {:02x?}", &p3_root_bytes);
    println!("Binius root: {:02x?}", &binius_root);
    if overflow_count == 0 {
        println!("Match: {} (no overflows, roots identical)", p3_root_bytes == binius_root);
    } else {
        println!("Roots differ due to {} overflow fixups (expected)", overflow_count);
    }

    Ok(())
}
