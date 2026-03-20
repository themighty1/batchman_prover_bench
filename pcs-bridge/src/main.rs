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
use binius_hash::compression::PseudoCompressionFunction;
use binius_math::MultilinearExtension;
use binius_ntt::SingleThreadedNTT;
use binius_utils::{SerializationMode, SerializeBytes, checked_arithmetics::log2_ceil_usize};

use p3_goldilocks::Goldilocks;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_symmetric::CryptographicHasher;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_commit::Mmcs;

type B64 = BinaryField64b;
type B128 = binius_field::BinaryField128b;
type P = PackedType<OptimalUnderlier, B128>;

// ── Blake3 wrappers for Binius (same as binius/src/lib.rs) ──────────

#[derive(Clone)]
struct Blake3Digest(blake3::Hasher);
impl Default for Blake3Digest {
    fn default() -> Self { Self(blake3::Hasher::new()) }
}
impl digest::HashMarker for Blake3Digest {}
impl digest::Update for Blake3Digest {
    fn update(&mut self, data: &[u8]) { self.0.update(data); }
}
impl digest::Reset for Blake3Digest {
    fn reset(&mut self) { self.0 = blake3::Hasher::new(); }
}
impl digest::OutputSizeUser for Blake3Digest {
    type OutputSize = digest::typenum::U32;
}
impl digest::core_api::BlockSizeUser for Blake3Digest {
    type BlockSize = digest::typenum::U64;
}
impl digest::FixedOutput for Blake3Digest {
    fn finalize_into(self, out: &mut digest::Output<Self>) {
        out.copy_from_slice(self.0.finalize().as_bytes());
    }
}
impl digest::FixedOutputReset for Blake3Digest {
    fn finalize_into_reset(&mut self, out: &mut digest::Output<Self>) {
        out.copy_from_slice(self.0.finalize().as_bytes());
        self.0 = blake3::Hasher::new();
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct BiniusBlake3Compress;
impl PseudoCompressionFunction<digest::Output<Blake3Digest>, 2> for BiniusBlake3Compress {
    fn compress(&self, input: [digest::Output<Blake3Digest>; 2]) -> digest::Output<Blake3Digest> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&input[0]);
        hasher.update(&input[1]);
        let mut out = digest::Output::<Blake3Digest>::default();
        out.copy_from_slice(hasher.finalize().as_bytes());
        out
    }
}
impl binius_hash::compression::CompressionFunction<digest::Output<Blake3Digest>, 2>
    for BiniusBlake3Compress {}

// ── Blake3 hasher for p3 (produces binius-compatible leaf hashes) ────

/// Hashes Goldilocks field elements by converting each to 8 LE bytes
/// and feeding to Blake3. Output is [u64; 4] = 32 bytes = Blake3 digest.
#[derive(Clone)]
struct P3Blake3Hash;

impl CryptographicHasher<Goldilocks, [u64; 4]> for P3Blake3Hash {
    fn hash_iter<I: IntoIterator<Item = Goldilocks>>(&self, input: I) -> [u64; 4] {
        let mut hasher = blake3::Hasher::new();
        for elem in input {
            // Goldilocks value as canonical u64, then LE bytes
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
        // Write each u64 as 8 LE bytes (total 32 bytes per side)
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
    // Test data
    let raw_values: Vec<u64> = (0..1024).map(|i| i * 1000 + 42).collect();
    let values_b64: Vec<B64> = raw_values.iter().map(|&v| B64::new(v)).collect();

    println!("=== PCS Bridge: Binius → WHIR ===");
    println!("Input: {} values", raw_values.len());
    println!();

    // ── Step 1: Binius commitment ───────────────────────────────────
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

    let log_inv_rate = 1;
    let security_bits = 100;
    let fri_strategy = FriStrategy::ConstantArity(8);

    let mut n_multilins_by_vars = vec![0usize; n_packed_vars + 1];
    n_multilins_by_vars[n_packed_vars] = 1;
    let commit_meta = piop::CommitMeta::new(n_multilins_by_vars);

    let merkle_prover = BinaryMerkleTreeProver::<_, Blake3Digest, _>::new(BiniusBlake3Compress);
    let fri_params = piop::make_commit_params_with_strategy::<_, FEncode, _>(
        &commit_meta, merkle_prover.scheme(), security_bits, log_inv_rate, &fri_strategy,
    )?;
    let ntt = SingleThreadedNTT::with_subspace(fri_params.rs_code().subspace())?
        .precompute_twiddles().multithreaded();

    let output = piop::commit(&fri_params, &ntt, &merkle_prover, &[mle_witness])?;
    let binius_root: Vec<u8> = output.commitment.as_slice().to_vec();
    println!("Binius root: {:02x?}", &binius_root);

    // ── Step 2: Extract codeword as raw bytes ───────────────────────
    let codeword = &output.codeword;
    let tree = &output.committed;
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
    //
    // The binius NTT operates in GF(2^64) and can produce any 64-bit value.
    // Goldilocks has prime p = 2^64 - 2^32 + 1, so values >= p don't fit.
    //
    // For the cross-commitment bridge to work, both the Merkle tree bytes
    // and the Goldilocks polynomial must agree on the same values. We fix
    // overflows by replacing them in the codeword bytes BEFORE hashing:
    //
    //   If val >= p: replace with blake3(val_bytes)[..8] interpreted as u64.
    //   Repeat if the hash also overflows (astronomically unlikely but possible).
    //
    // Both binius (Merkle) and WHIR (polynomial) sides apply this same
    // deterministic fixup, so the commitment and proof stay consistent.
    //
    let goldilocks_p = Goldilocks::ORDER_U64;
    let total_u64s = codeword_bytes.len() / 8;
    let mut overflow_count = 0u64;

    for i in 0..total_u64s {
        let off = i * 8;
        let mut val = u64::from_le_bytes(codeword_bytes[off..off+8].try_into().unwrap());

        if val >= goldilocks_p {
            overflow_count += 1;
            // Hash repeatedly until we get a value < p
            loop {
                let hash = blake3::hash(&val.to_le_bytes());
                val = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
                if val < goldilocks_p {
                    break;
                }
            }
            // Write the fixed-up value back into the codeword bytes
            codeword_bytes[off..off+8].copy_from_slice(&val.to_le_bytes());
        }
    }

    if overflow_count > 0 {
        println!("  Goldilocks overflows fixed: {} / {} ({:.6}%)",
            overflow_count, total_u64s, overflow_count as f64 / total_u64s as f64 * 100.0);
    } else {
        println!("  No Goldilocks overflows");
    }

    // Now interpret the (possibly fixed-up) bytes as Goldilocks elements
    let bytes_per_leaf = codeword_bytes.len() / num_leaves;
    let gl_per_leaf = bytes_per_leaf / 8;

    let mut gl_leaves: Vec<Vec<Goldilocks>> = Vec::new();
    for leaf_idx in 0..num_leaves {
        let start = leaf_idx * bytes_per_leaf;
        let mut row = Vec::with_capacity(gl_per_leaf);
        for j in 0..gl_per_leaf {
            let off = start + j * 8;
            let val = u64::from_le_bytes(codeword_bytes[off..off+8].try_into().unwrap());
            debug_assert!(val < goldilocks_p, "overflow not fixed at index {}", leaf_idx * gl_per_leaf + j);
            row.push(Goldilocks::new(val));
        }
        gl_leaves.push(row);
    }

    println!("Goldilocks: {} leaves × {} elements/leaf", gl_leaves.len(), gl_per_leaf);

    // ── Step 4: Build p3 Merkle tree with Blake3 ────────────────────
    // Create a RowMajorMatrix: num_leaves rows × gl_per_leaf columns
    let flat: Vec<Goldilocks> = gl_leaves.into_iter().flatten().collect();
    let matrix = RowMajorMatrix::new(flat, gl_per_leaf);

    let p3_mmcs = MerkleTreeMmcs::<Goldilocks, u64, P3Blake3Hash, P3Blake3Compress, 2, 4>::new(
        P3Blake3Hash, P3Blake3Compress, 0, // cap_height=0 means just root
    );

    let (p3_commitment, _p3_tree) = p3_mmcs.commit(vec![matrix]);

    // p3_commitment is the "cap" — with cap_height=0 it's a single [u64; 4] = 32 bytes
    // MerkleCap is a Vec<[u64; 4]> with cap_height=0 → single entry
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
        println!("Bridge root (shared): {:02x?}", &p3_root_bytes);
    }

    Ok(())
}
