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

    // WHIR commit with folding_factor=10, log_inv_rate=0, num_vars=11:
    //   poly (2048 flat) → view as width=2^(11-10)=2 → 1024 rows × 2 cols
    //   → transpose → 2 rows × 1024 cols
    //   → pad to 2^(10+0)=1024... no, pad_to_height = 2^(11+0-10) = 2 (no change)
    //   → DFT each column → 2 rows × 1024 cols
    //   → Merkle (2 leaves × 1024 GL elements)  ← matches binius!
    //
    // Inverse: binius codeword (2 rows × 1024 cols)
    //   = WHIR DFT output directly (same layout!)
    //   → IDFT each column → 2 rows × 1024 cols (pre-DFT)
    //   → un-transpose → 1024 rows × 2 cols
    //   → flatten (width=2) → 2048 coefficients

    // Codeword is already in WHIR's DFT output layout (2 rows × 1024 cols)
    // IDFT each column
    let idft_result = dft.idft_batch(codeword_matrix).to_row_major_matrix();

    // Un-transpose: 2 rows × 1024 cols → 1024 rows × 2 cols
    let pre_transpose = idft_result.transpose();

    // Flatten with width=2 (matching WHIR's RowMajorMatrixView::new(poly, 2))
    let coeffs: Vec<F> = pre_transpose.values.clone();
    let n_coeffs = coeffs.len();
    println!("  Derived coefficients: {} values", n_coeffs);

    // ── Step 4: Verify round-trip ───────────────────────────────────
    // Forward: coeffs → reshape → transpose → DFT → should equal codeword
    //
    // With folding_factor=k=log2(num_leaves), the reshape is:
    //   2^(N-k) rows × 2^k=num_leaves cols
    // Transpose: num_leaves rows × 2^(N-k) cols = num_leaves × gl_per_leaf

    // Forward: mimic WHIR's commit pipeline with folding_factor=10
    // view as width=2 → 1024 rows × 2 cols → transpose → 2 rows × 1024 cols → DFT
    let forward_matrix = RowMajorMatrix::new(coeffs.clone(), num_leaves); // width=2, 1024×2
    let forward_transposed = forward_matrix.transpose(); // 2 rows × 1024 cols
    let forward_codeword = dft.dft_batch(forward_transposed).to_row_major_matrix(); // 2×1024

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

    if mismatch_count > 0 {
        println!("  FAILED: round-trip mismatch");
        return Ok(());
    }
    println!("  Round-trip: PASSED");

    // ── Step 5: Call WHIR commit and verify root matches ────────────
    //
    // WHIR params: folding_factor=1 (first round), starting_log_inv_rate=0
    // This gives: 2^1=2 rows after transpose, no padding → 2 leaves × 1024 GL
    //
    // num_vars = log2(n_coeffs) = log2(2048) = 11
    use p3_symmetric::CryptographicHasher;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_commit::Mmcs;
    use p3_multilinear_util::evals::EvaluationsList;
    use p3_challenger::{HashChallenger, SerializingChallenger64};
    use p3_keccak::Keccak256Hash;

    use whir_p3::{
        fiat_shamir::domain_separator::DomainSeparator,
        parameters::{
            FoldingFactor, ProtocolParameters, SecurityAssumption, SumcheckStrategy, WhirConfig,
        },
        whir::{
            committer::writer::CommitmentWriter,
            proof::WhirProof,
        },
    };

    // P3 Blake3 hasher and compressor matching binius leaf hashing
    #[derive(Clone)]
    struct P3Blake3Hash;
    impl CryptographicHasher<Goldilocks, [u64; 4]> for P3Blake3Hash {
        fn hash_iter<I: IntoIterator<Item = Goldilocks>>(&self, input: I) -> [u64; 4] {
            let mut hasher = blake3::Hasher::new();
            for elem in input {
                hasher.update(&elem.as_canonical_u64().to_le_bytes());
            }
            let hash = hasher.finalize();
            let b = hash.as_bytes();
            [
                u64::from_le_bytes(b[0..8].try_into().unwrap()),
                u64::from_le_bytes(b[8..16].try_into().unwrap()),
                u64::from_le_bytes(b[16..24].try_into().unwrap()),
                u64::from_le_bytes(b[24..32].try_into().unwrap()),
            ]
        }
    }

    #[derive(Clone)]
    struct P3Blake3Compress;
    impl p3_symmetric::PseudoCompressionFunction<[u64; 4], 2> for P3Blake3Compress {
        fn compress(&self, input: [[u64; 4]; 2]) -> [u64; 4] {
            let mut hasher = blake3::Hasher::new();
            for &v in &input[0] { hasher.update(&v.to_le_bytes()); }
            for &v in &input[1] { hasher.update(&v.to_le_bytes()); }
            let hash = hasher.finalize();
            let b = hash.as_bytes();
            [
                u64::from_le_bytes(b[0..8].try_into().unwrap()),
                u64::from_le_bytes(b[8..16].try_into().unwrap()),
                u64::from_le_bytes(b[16..24].try_into().unwrap()),
                u64::from_le_bytes(b[24..32].try_into().unwrap()),
            ]
        }
    }
    impl p3_symmetric::CompressionFunction<[u64; 4], 2> for P3Blake3Compress {}

    type WhirMmcs = MerkleTreeMmcs<Goldilocks, u64, P3Blake3Hash, P3Blake3Compress, 2, 4>;
    type EF = p3_field::extension::BinomialExtensionField<Goldilocks, 2>;
    type WhirChallenger = SerializingChallenger64<Goldilocks, HashChallenger<u8, Keccak256Hash, 32>>;

    let whir_num_vars = log2_ceil_usize(n_coeffs);
    let whir_folding = whir_num_vars - 1; // = 10 for 2048 coeffs
    println!("\n  WHIR commit: num_vars={}, folding_factor=ConstantFromSecondRound({},4), log_inv_rate=0",
        whir_num_vars, whir_folding);

    let mmcs = WhirMmcs::new(P3Blake3Hash, P3Blake3Compress, 0);

    let whir_params = ProtocolParameters {
        security_level: 100,
        pow_bits: 0,
        rs_domain_initial_reduction_factor: 1,
        folding_factor: FoldingFactor::ConstantFromSecondRound(whir_folding, 4),
        mmcs,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 0,
    };

    let params = WhirConfig::<EF, Goldilocks, WhirMmcs, WhirChallenger>::new(
        whir_num_vars, whir_params.clone());

    let polynomial = EvaluationsList::<Goldilocks>::new(coeffs);

    let mut statement = params.initial_statement(polynomial, SumcheckStrategy::default());

    let mut domainsep: DomainSeparator<EF, Goldilocks> = DomainSeparator::new(vec![]);
    domainsep.commit_statement::<_, _, 4>(&params);
    domainsep.add_whir_proof::<_, _, 4>(&params);

    let inner = HashChallenger::<u8, Keccak256Hash, 32>::new(vec![], Keccak256Hash {});
    let mut challenger = WhirChallenger::new(inner);
    domainsep.observe_domain_separator(&mut challenger);

    let committer = CommitmentWriter::new(&params);
    let mut proof = WhirProof::<Goldilocks, EF, WhirMmcs>::from_protocol_parameters(
        &whir_params, whir_num_vars);

    let _prover_data = committer.commit(&dft, &mut proof, &mut challenger, &mut statement)
        .expect("WHIR commit failed");

    // Extract WHIR root from proof
    let whir_commitment = proof.initial_commitment.expect("no commitment in proof");
    let mut whir_root_bytes = Vec::new();
    for &v in &whir_commitment[0] {
        whir_root_bytes.extend_from_slice(&v.to_le_bytes());
    }

    println!("  WHIR root:   {:02x?}", &whir_root_bytes);
    println!("  Binius root: {:02x?}", &binius_root);
    println!("  WHIR==Binius: {}", whir_root_bytes == binius_root);

    // Sanity: direct p3 Merkle commit of the same codeword (bypass WHIR)
    let direct_mmcs = WhirMmcs::new(P3Blake3Hash, P3Blake3Compress, 0);
    let direct_matrix = RowMajorMatrix::new(gl_codeword, gl_per_leaf);
    let (direct_cap, _) = direct_mmcs.commit(vec![direct_matrix]);
    let mut direct_root_bytes = Vec::new();
    for &v in &direct_cap[0] {
        direct_root_bytes.extend_from_slice(&v.to_le_bytes());
    }
    println!("  Direct p3:   {:02x?}", &direct_root_bytes);
    println!("  Direct==Binius: {}", direct_root_bytes == binius_root);

    Ok(())
}
