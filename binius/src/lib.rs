//! Shared utilities: Blake3 digest wrapper and binius column commitment.

use anyhow::Result;
use binius_core::{
    constraint_system::FriStrategy,
    merkle_tree::BinaryMerkleTreeProver,
    piop,
    protocols::fri::{FRIParams, calculate_n_test_queries},
    reed_solomon::ReedSolomonCode,
};
use binius_field::{
    arch::OptimalUnderlier, as_packed_field::PackedType, packed::set_packed_slice,
    BinaryField32b as FEncode, PackedExtension, PackedField,
};
use binius_hash::compression::{CompressionFunction, PseudoCompressionFunction};
use binius_math::MultilinearExtension;
use binius_ntt::SingleThreadedNTT;
use binius_utils::checked_arithmetics::log2_ceil_usize;
use digest::{
    core_api::BlockSizeUser,
    consts::{U32, U64},
    FixedOutput, FixedOutputReset, HashMarker, OutputSizeUser, Reset, Update,
};

pub use binius_m3::builder::{B32, B64, B128};

// ==== Blake3 wrapper for binius digest traits ====

#[derive(Clone)]
pub struct Blake3Digest {
    hasher: blake3::Hasher,
}

impl Default for Blake3Digest {
    fn default() -> Self {
        Self {
            hasher: blake3::Hasher::new(),
        }
    }
}

impl HashMarker for Blake3Digest {}

impl Update for Blake3Digest {
    fn update(&mut self, data: &[u8]) {
        self.hasher.update(data);
    }
}

impl Reset for Blake3Digest {
    fn reset(&mut self) {
        self.hasher = blake3::Hasher::new();
    }
}

impl OutputSizeUser for Blake3Digest {
    type OutputSize = U32;
}

impl BlockSizeUser for Blake3Digest {
    type BlockSize = U64;
}

impl FixedOutput for Blake3Digest {
    fn finalize_into(self, out: &mut digest::Output<Self>) {
        let hash = self.hasher.finalize();
        out.copy_from_slice(hash.as_bytes());
    }
}

impl FixedOutputReset for Blake3Digest {
    fn finalize_into_reset(&mut self, out: &mut digest::Output<Self>) {
        let hash = self.hasher.finalize();
        out.copy_from_slice(hash.as_bytes());
        self.hasher = blake3::Hasher::new();
    }
}

#[derive(Debug, Default, Clone)]
pub struct Blake3Compression;

impl PseudoCompressionFunction<digest::Output<Blake3Digest>, 2> for Blake3Compression {
    fn compress(&self, input: [digest::Output<Blake3Digest>; 2]) -> digest::Output<Blake3Digest> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(input[0].as_slice());
        hasher.update(input[1].as_slice());
        let hash = hasher.finalize();
        let mut out = digest::Output::<Blake3Digest>::default();
        out.copy_from_slice(hash.as_bytes());
        out
    }
}

impl CompressionFunction<digest::Output<Blake3Digest>, 2> for Blake3Compression {}

// ==== Column commitment ====

type P = PackedType<OptimalUnderlier, B128>;

/// Commit to a single B128 column, reproducing the exact same commitment as
/// binius's `prove`/`verify` pipeline for a standalone group.
///
/// `fri_strategy` must match the prover's `FRI_STRATEGY`.
///
/// Returns the 32-byte Merkle root.
pub fn commit_column_b128(
    values: &[B128],
    log_inv_rate: usize,
    security_bits: usize,
    fri_strategy: &FriStrategy,
) -> Result<Vec<u8>> {
    let len = values.len().next_power_of_two();
    let n_vars = log2_ceil_usize(len);
    // B128 tower_level=7, F::TOWER_LEVEL=7 → n_packed_vars = n_vars
    let n_packed_vars = n_vars;

    let mut packed = vec![P::default(); len.div_ceil(P::WIDTH)];
    for (i, &v) in values.iter().enumerate() {
        set_packed_slice(&mut packed, i, v);
    }

    let mle = MultilinearExtension::new(n_vars, packed)
        .expect("correct number of coefficients");
    let mle_witness: binius_core::witness::MultilinearWitness<'_, P> =
        mle.specialize_arc_dyn();

    commit_with_meta(n_packed_vars, &[mle_witness], log_inv_rate, security_bits, fri_strategy)
}

/// Commit to a single B64 column, reproducing the exact same commitment as
/// binius's `prove`/`verify` pipeline for a standalone group.
///
/// B64 values are packed into B128 storage (2 per element), matching binius's
/// internal `PackedExtension::<B64>::cast_bases` layout.
///
/// Returns the 32-byte Merkle root.
pub fn commit_column_b64(
    values: &[B64],
    log_inv_rate: usize,
    security_bits: usize,
    fri_strategy: &FriStrategy,
) -> Result<Vec<u8>> {
    let len = values.len().next_power_of_two();
    let n_vars = log2_ceil_usize(len);
    // B64 tower_level=6, F::TOWER_LEVEL=7 → n_packed_vars = n_vars - 1
    let n_packed_vars = n_vars.saturating_sub(1);

    // Allocate B128 storage and fill via the B64 subfield view.
    // Each B128 holds 2 B64 values, so we need len/2 B128 elements.
    let b64_per_b128 = 2usize; // 2^(TOWER_LEVEL_B128 - TOWER_LEVEL_B64) = 2^(7-6)
    let n_packed = len.div_ceil(b64_per_b128);
    let mut packed = vec![P::default(); n_packed];
    {
        type PackedB64 = <P as PackedExtension<B64>>::PackedSubfield;
        let b64_slice: &mut [PackedB64] =
            PackedExtension::<B64>::cast_bases_mut(&mut packed);
        for (i, &v) in values.iter().enumerate() {
            let elem = i / PackedB64::WIDTH;
            let pos = i % PackedB64::WIDTH;
            b64_slice[elem].set(pos, v);
        }
    }

    // packed must outlive the MLE (cast_bases borrows it), so we commit inline.
    let mle = MultilinearExtension::new(
        n_vars,
        PackedExtension::<B64>::cast_bases(&packed),
    )
    .expect("correct number of coefficients");
    let mle_witness: binius_core::witness::MultilinearWitness<'_, P> =
        mle.specialize_arc_dyn();

    let mut n_multilins_by_vars = vec![0usize; n_packed_vars + 1];
    n_multilins_by_vars[n_packed_vars] = 1;
    let commit_meta = piop::CommitMeta::new(n_multilins_by_vars);

    let merkle_prover = BinaryMerkleTreeProver::<_, Blake3Digest, _>::new(Blake3Compression);
    let fri_params = piop::make_commit_params_with_strategy::<_, FEncode, _>(
        &commit_meta,
        merkle_prover.scheme(),
        security_bits,
        log_inv_rate,
        fri_strategy,
    )?;
    let ntt = SingleThreadedNTT::with_subspace(fri_params.rs_code().subspace())?
        .precompute_twiddles()
        .multithreaded();

    let output = piop::commit(&fri_params, &ntt, &merkle_prover, &[mle_witness])?;
    Ok(output.commitment.as_slice().to_vec())
}

/// Commit to a single B32 column, reproducing the exact same commitment as
/// binius's `prove`/`verify` pipeline for a standalone group.
///
/// B32 values are packed into B128 storage (4 per element), matching binius's
/// internal `PackedExtension::<B32>::cast_bases` layout.
///
/// Returns the 32-byte Merkle root.
pub fn commit_column_b32(
    values: &[B32],
    log_inv_rate: usize,
    security_bits: usize,
    fri_strategy: &FriStrategy,
) -> Result<Vec<u8>> {
    let len = values.len().next_power_of_two();
    let n_vars = log2_ceil_usize(len);
    // B32 tower_level=5, F::TOWER_LEVEL=7 → n_packed_vars = n_vars - 2
    let n_packed_vars = n_vars.saturating_sub(2);

    // Each B128 holds 4 B32 values: 2^(7-5) = 4.
    let b32_per_b128 = 4usize;
    let n_packed = len.div_ceil(b32_per_b128);
    let mut packed = vec![P::default(); n_packed];
    {
        type PackedB32 = <P as PackedExtension<B32>>::PackedSubfield;
        let b32_slice: &mut [PackedB32] =
            PackedExtension::<B32>::cast_bases_mut(&mut packed);
        for (i, &v) in values.iter().enumerate() {
            let elem = i / PackedB32::WIDTH;
            let pos = i % PackedB32::WIDTH;
            b32_slice[elem].set(pos, v);
        }
    }

    let mle = MultilinearExtension::new(
        n_vars,
        PackedExtension::<B32>::cast_bases(&packed),
    )
    .expect("correct number of coefficients");
    let mle_witness: binius_core::witness::MultilinearWitness<'_, P> =
        mle.specialize_arc_dyn();

    let mut n_multilins_by_vars = vec![0usize; n_packed_vars + 1];
    n_multilins_by_vars[n_packed_vars] = 1;
    let commit_meta = piop::CommitMeta::new(n_multilins_by_vars);

    let merkle_prover = BinaryMerkleTreeProver::<_, Blake3Digest, _>::new(Blake3Compression);
    let fri_params = piop::make_commit_params_with_strategy::<_, FEncode, _>(
        &commit_meta,
        merkle_prover.scheme(),
        security_bits,
        log_inv_rate,
        fri_strategy,
    )?;
    let ntt = SingleThreadedNTT::with_subspace(fri_params.rs_code().subspace())?
        .precompute_twiddles()
        .multithreaded();

    let output = piop::commit(&fri_params, &ntt, &merkle_prover, &[mle_witness])?;
    Ok(output.commitment.as_slice().to_vec())
}

/// Shared commitment logic: build CommitMeta, FRI params, and call piop::commit.
fn commit_with_meta(
    n_packed_vars: usize,
    witnesses: &[binius_core::witness::MultilinearWitness<'_, P>],
    log_inv_rate: usize,
    security_bits: usize,
    fri_strategy: &FriStrategy,
) -> Result<Vec<u8>> {
    let mut n_multilins_by_vars = vec![0usize; n_packed_vars + 1];
    n_multilins_by_vars[n_packed_vars] = witnesses.len();
    let commit_meta = piop::CommitMeta::new(n_multilins_by_vars);

    let merkle_prover = BinaryMerkleTreeProver::<_, Blake3Digest, _>::new(Blake3Compression);
    let fri_params = piop::make_commit_params_with_strategy::<_, FEncode, _>(
        &commit_meta,
        merkle_prover.scheme(),
        security_bits,
        log_inv_rate,
        fri_strategy,
    )?;
    let ntt = SingleThreadedNTT::with_subspace(fri_params.rs_code().subspace())?
        .precompute_twiddles()
        .multithreaded();

    let output = piop::commit(&fri_params, &ntt, &merkle_prover, witnesses)?;
    Ok(output.commitment.as_slice().to_vec())
}
