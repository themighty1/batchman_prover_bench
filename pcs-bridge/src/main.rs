//! PCS Bridge: derive WHIR (Goldilocks) inputs that produce the same
//! Merkle commitment as Binius for a column of B64 values.
//!
//! Approach:
//! 1. Run Binius RS encoding (additive NTT) to get the codeword
//! 2. Extract the raw Merkle leaves (byte representation)
//! 3. Interpret leaf bytes as Goldilocks elements
//! 4. Compute inverse Goldilocks FFT to get WHIR coefficients
//! 5. Verify: WHIR commit(coeffs) produces the same Merkle root
//!
//! For now, this is an exploration binary that:
//! - Takes a vector of u64 values
//! - Computes the Binius B64 column commitment
//! - Extracts the RS codeword after NTT
//! - Prints the codeword for analysis

use anyhow::Result;
use binius_core::{
    constraint_system::FriStrategy,
    merkle_tree::BinaryMerkleTreeProver,
    piop,
    reed_solomon::ReedSolomonCode,
};
use binius_field::{
    BinaryField32b as FEncode, BinaryField64b,
    arch::OptimalUnderlier, as_packed_field::PackedType,
    PackedExtension, PackedField, TowerField,
    packed::set_packed_slice,
};
use binius_hash::compression::PseudoCompressionFunction;
use binius_math::MultilinearExtension;
use binius_ntt::SingleThreadedNTT;
use binius_utils::checked_arithmetics::log2_ceil_usize;

type B64 = BinaryField64b;
type B128 = binius_field::BinaryField128b;
type P = PackedType<OptimalUnderlier, B128>;

// Blake3 wrapper (same as binius/src/lib.rs)
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
struct Blake3Compression;
impl PseudoCompressionFunction<digest::Output<Blake3Digest>, 2> for Blake3Compression {
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
    for Blake3Compression {}

fn main() -> Result<()> {
    // Test data: small column of B64 values
    let raw_values: Vec<u64> = (0..1024).map(|i| i * 1000 + 42).collect();
    let values: Vec<B64> = raw_values.iter().map(|&v| B64::new(v)).collect();

    println!("=== PCS Bridge: Binius → WHIR ===");
    println!("Input: {} B64 values", values.len());
    println!();

    // Step 1: Reproduce Binius commitment pipeline
    let len = values.len().next_power_of_two();
    let n_vars = log2_ceil_usize(len);
    let n_packed_vars = n_vars.saturating_sub(1);

    let b64_per_b128 = 2usize;
    let n_packed = len.div_ceil(b64_per_b128);
    let mut packed = vec![P::default(); n_packed];
    {
        type PackedB64 = <P as PackedExtension<B64>>::PackedSubfield;
        let b64_slice: &mut [PackedB64] = PackedExtension::<B64>::cast_bases_mut(&mut packed);
        for (i, &v) in values.iter().enumerate() {
            let elem = i / PackedB64::WIDTH;
            let pos = i % PackedB64::WIDTH;
            b64_slice[elem].set(pos, v);
        }
    }

    let mle = MultilinearExtension::new(
        n_vars,
        PackedExtension::<B64>::cast_bases(&packed),
    )?;
    let mle_witness: binius_core::witness::MultilinearWitness<'_, P> =
        mle.specialize_arc_dyn();

    // Build commit params
    let log_inv_rate = 1;
    let security_bits = 100;
    let fri_strategy = FriStrategy::ConstantArity(8);

    let mut n_multilins_by_vars = vec![0usize; n_packed_vars + 1];
    n_multilins_by_vars[n_packed_vars] = 1;
    let commit_meta = piop::CommitMeta::new(n_multilins_by_vars);

    let merkle_prover = BinaryMerkleTreeProver::<_, Blake3Digest, _>::new(Blake3Compression);
    let fri_params = piop::make_commit_params_with_strategy::<_, FEncode, _>(
        &commit_meta,
        merkle_prover.scheme(),
        security_bits,
        log_inv_rate,
        &fri_strategy,
    )?;

    println!("FRI params:");
    println!("  n_vars: {}", n_vars);
    println!("  n_packed_vars: {}", n_packed_vars);
    println!("  log_inv_rate: {}", log_inv_rate);
    println!("  rs_code log_dim: {}", fri_params.rs_code().log_dim());
    println!("  rs_code log_len: {}", fri_params.rs_code().log_len());
    println!();

    let ntt = SingleThreadedNTT::with_subspace(fri_params.rs_code().subspace())?
        .precompute_twiddles()
        .multithreaded();

    // Step 2: Commit and get the root
    let output = piop::commit(&fri_params, &ntt, &merkle_prover, &[mle_witness])?;
    let binius_root = output.commitment.as_slice().to_vec();

    println!("Binius commitment (Merkle root):");
    println!("  {:02x?}", &binius_root[..]);
    println!();

    // Step 3: Extract the codeword and understand leaf structure
    let codeword = &output.codeword;
    println!("Codeword: {} packed elements", codeword.len());

    // The codeword is a flat array of packed B128 elements.
    // The Merkle tree has 2^log_len leaves.
    // Each leaf hashes a contiguous chunk of the serialized codeword.
    //
    // For our test case: serialize each packed element to bytes,
    // then manually build Blake3 leaf hashes and compare to binius tree.

    // Serialize codeword to raw bytes (CanonicalTower mode = LE bytes)
    let mut codeword_bytes = Vec::new();
    for elem in codeword.iter() {
        let bytes: [u8; 16] = bytemuck_cast(*elem);
        codeword_bytes.extend_from_slice(&bytes);
    }
    println!("  Serialized codeword: {} bytes", codeword_bytes.len());

    // The Merkle tree has 2^(rs_code.log_len - log_batch) leaves.
    // Each leaf covers batch_size * 16 bytes of codeword data.
    let log_len = fri_params.rs_code().log_len();
    let log_batch = fri_params.log_batch_size();
    let tree_log_len = tree.log_len;
    let num_leaves = 1 << tree_log_len;
    let bytes_per_leaf = codeword_bytes.len() / num_leaves;
    println!("  rs log_len: {}, log_batch: {}, tree log_len: {}", log_len, log_batch, tree_log_len);
    println!("  Leaves: {}, bytes per leaf: {}", num_leaves, bytes_per_leaf);

    // Manually hash each leaf chunk with Blake3
    println!("\n  Manual leaf hashes (first 4):");
    for i in 0..std::cmp::min(4, num_leaves) {
        let chunk = &codeword_bytes[i * bytes_per_leaf .. (i+1) * bytes_per_leaf];
        let hash = blake3::hash(chunk);
        println!("    leaf[{}]: {:02x?}  (data: {:02x?})", i, &hash.as_bytes()[..8], &chunk[..std::cmp::min(16, chunk.len())]);
    }

    // Access the committed tree to compare
    let tree = &output.committed;
    println!("\n  Tree log_len: {}", tree.log_len);
    for depth in 0..=tree.log_len {
        match tree.layer(depth) {
            Ok(layer) => println!("    layer[{}]: {} nodes, first: {:02x?}", depth, layer.len(), &layer[0][..8]),
            Err(e) => println!("    layer[{}]: error: {}", depth, e),
        }
    }

    // Check if our manual hashes match the deepest layer
    if let Ok(leaves) = tree.layer(tree.log_len) {
        let mut match_count = 0;
        for i in 0..std::cmp::min(num_leaves, leaves.len()) {
            let chunk = &codeword_bytes[i * bytes_per_leaf .. (i+1) * bytes_per_leaf];
            let hash = blake3::hash(chunk);
            if hash.as_bytes() == leaves[i].as_slice() {
                match_count += 1;
            }
        }
        println!("\n  Leaf hash match: {}/{}", match_count, leaves.len());
    }

    Ok(())
}

fn bytemuck_cast<T: Copy>(val: T) -> [u8; 16] {
    let mut out = [0u8; 16];
    unsafe {
        let src = &val as *const T as *const u8;
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), std::mem::size_of::<T>().min(16));
    }
    out
}
