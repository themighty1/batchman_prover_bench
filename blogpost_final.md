# Replacing the Zero-Product Protocol in Batchman

## Motivation

We investigated VOLE-based ZK proving systems for short-statement, low-latency proving — the kind needed in zkTLS applications where a user proves properties of TLS session data in real time.

Existing zkVM solutions like RISC Zero are optimized for large statements and add significant latency for shorter computations. They also do not support running in the browser.

Since the zkTLS protocol already operates in a two-party computation setting (the user and a notary), it is natural to use a VOLE-based ZK proving system — which is itself an interactive two-party protocol. The interactive model fits the existing architecture: no extra round trips, no heavyweight client-side proving, just an extension of the two-party session that's already in progress.

We set out to determine whether a VOLE-based zkVM can prove general computations (like JSON parsing) fast enough to be practical in this setting.

## State of the Art VOLE-based zkVM

The state of the art in VOLE-based zkVMs is [Batchman](https://eprint.iacr.org/2023/1257). Its key contribution is that the prover only commits to and evaluates the **active branch** — not all B branches. For a zkVM with 50+ ISA operations, this is the difference between evaluating one circuit per step versus fifty.

The trade-off for this reduced computation is that the prover must run a **zero-product check** — a bandwidth-heavy ZK sub-protocol that proves the active branch is valid (i.e., corresponds to one of the B known circuit topologies). Essentially, Batchman trades less compute for more bandwidth.

## The Zero-Product Check

Each branch produces a "topology token" — zero for the active branch, random for all others. The prover proves that the product of all B tokens is zero. If at least one is zero, the product is zero — proving a valid branch was selected without revealing which one.

## The Bandwidth Problem

This check dominates communication cost. For B = 51 branches, each step sends ~100 extension field elements (64 bits each) — **800 bytes per step**.

At 10 Mbps upload, that caps proving at ~1,500 steps/second. A 100K-step JSON parser would need ~67 seconds of upload just for zero-product checks.

## The Alternative: Set Membership via Polynomial Commitments

The natural workaround is to replace the zero-product protocol with a **set membership proof**. Instead of proving a product is zero, the prover proves that one of its committed tokens belongs to the set of valid branch tokens.

This can be done with a polynomial commitment scheme (PCS). Encode the full valid set as roots of P(x) and the prover's active-branch subset as roots of Z(x). The prover commits to a quotient Q(x) such that P(x) = Z(x) · Q(x), proving that Z's roots are contained in P's roots — i.e., every active-branch token is a valid branch token. The quotient hides which specific roots were selected.

### The Catch

To construct the quotient polynomial, the prover needs the **entire set** — all B × R tokens across every branch and every step. This essentially forces the prover to evaluate every single branch, undoing Batchman's main benefit.

### The Constraint

The set membership proof cannot run during the main Batchman protocol. The verifier's secret delta (the IT-MAC correlation) must remain hidden from the prover throughout — otherwise the prover could forge MACs. Only *after* the prover has committed to all its values does the verifier reveal delta, at which point the prover can reconstruct all branch tokens and build the full set P(x).

This means the PCS proof is strictly sequential: the main protocol runs first, then delta is revealed, then the prover constructs and proves the quotient polynomial. It adds end-to-end latency that cannot be parallelized with the main protocol.

## Experimental Results

We use Binius as the PCS backend because it works natively with binary fields (GF(2^128)) — exactly the field our IT-MACs live in.

Our benchmarks on a real workload — a simple JSON query circuit with approximately 100,000 execution steps and 50 branches in our custom ISA — reveal that the set membership proof is the bottleneck.

The core problem: we are proving that a subset of 100,000 active branches exists within a set of 5 million total branches (100K steps × 50 branches). With the Binius proving system, this single membership proof takes roughly **60 seconds**, which is prohibitively expensive.

This cost is dominated by the source table size: 5 million rows of GF(2^128) keys that must be committed and constrained. While the proof itself is compact (~1.5 MB) and verification is fast, the prover's work scales linearly with the full set size — precisely the cost that Batchman's active-branch optimization was designed to avoid.

## Possible Optimizations

The main bottleneck is that the set membership proof runs strictly after the entire Batchman protocol. One potential optimization is to break the execution trace into **segments** and prove each segment independently.

After Batchman finishes proving a segment, the verifier reveals that segment's delta, and the prover begins the set membership proof for that segment — interleaved with the Batchman protocol still running on subsequent segments. This would pipeline the PCS work with the interactive protocol instead of serializing them.

However, this introduces a constraint: each segment must use a **different delta**, which means each segment requires its own independent correlated OT setup. The COT correlations can no longer be shared across segments, increasing the setup cost proportionally to the number of segments.

## Code

The full implementation is available at [github.com/themighty1/batchman_prover_bench](https://github.com/themighty1/batchman_prover_bench).
