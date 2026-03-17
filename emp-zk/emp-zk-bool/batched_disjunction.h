#ifndef EMP_ZK_BOOL_BATCHED_DISJUNCTION_H__
#define EMP_ZK_BOOL_BATCHED_DISJUNCTION_H__

#include "emp-zk/emp-zk-bool/emp-zk-bool.h"
#include <algorithm>
#include <cstdio>

namespace emp {

// NOTE: The original protocol lifts single-bit MACs into GF(2^128) via random
// linear combination (RLC). This implementation uses GF(2^64) instead, which
// we assume provides sufficient statistical security parameter (SSP) for MAC
// forgery resistance. This assumption needs thorough investigation to confirm
// it does not break the security claims in the original paper.

// Raw carry-less multiply of lower 64-bit halves → 128-bit unreduced product.
// Use with reduce64() for delayed reduction: accumulate unreduced products via
// XOR, reduce once at the end. Saves ~9 instructions per multiply in tight loops.
inline __m128i clmul64(block a, block b) {
    return _mm_clmulepi64_si128(a, b, 0x00);
}

// Reduce a 128-bit unreduced product to GF(2^64) using x^64 + x^4 + x^3 + x + 1.
inline block reduce64(__m128i product) {
    __m128i hi = _mm_srli_si128(product, 8);
    __m128i r1 = _mm_slli_epi64(hi, 4);
    __m128i r2 = _mm_slli_epi64(hi, 3);
    __m128i r3 = _mm_slli_epi64(hi, 1);
    __m128i reduced = _mm_xor_si128(r1, r2);
    reduced = _mm_xor_si128(reduced, r3);
    reduced = _mm_xor_si128(reduced, hi);
    block res = _mm_xor_si128(product, reduced);
    return _mm_and_si128(res, _mm_set_epi64x(0, 0xFFFFFFFFFFFFFFFFULL));
}

// GF(2^64) multiplication: clmul + immediate reduction.
// Use clmul64() + reduce64() instead when accumulating multiple products.
inline void gfmul64(block a, block b, block *res) {
    *res = reduce64(clmul64(a, b));
}

// Mask a block to its lower 64 bits (for extracting the GF(2^64) component of a MAC)
inline block lo64(block x) {
    return _mm_and_si128(x, _mm_set_epi64x(0, 0xFFFFFFFFFFFFFFFFULL));
}

// GF(2^128) multiply by x (= α), using irreducible polynomial x^128 + x^7 + x^2 + x + 1.
// Used to pack bit-level IT-MACs into word-level GF(2^128) IT-MACs via Horner evaluation.
inline block gf128_mul_x(block a) {
    // Carry from bit 63 of low lane into bit 0 of high lane
    __m128i carry = _mm_slli_si128(_mm_srli_epi64(a, 63), 8);
    __m128i shifted = _mm_or_si128(_mm_slli_epi64(a, 1), carry);
    // If bit 127 was set, reduce: XOR with 0x87 (x^7 + x^2 + x + 1)
    __m128i top = _mm_srli_epi64(_mm_srli_si128(a, 8), 63); // bit 127 → low bit
    __m128i mask = _mm_sub_epi64(_mm_setzero_si128(), top);  // 0 or all-ones
    return _mm_xor_si128(shifted, _mm_and_si128(mask, _mm_set_epi64x(0, 0x87)));
}

// Gate type constants
static const int GATE_AND = 0;
static const int GATE_XOR = 1;
static const int GATE_INV = 2;

/**
 * BoolBatchedDisjunction - Batched disjunction protocol for boolean circuits
 *
 * Proves: I know a witness satisfying ONE of B branch circuits.
 *
 * Each branch is an independent circuit with its own gate sequence and wiring.
 * The only shared dimensions across branches are:
 *   - input_count (n_in): number of input/output wires
 *   - mul_count (n_×): number of AND (multiplication) gates
 * Branches with fewer AND gates pad with dummy AND gates (0·0=0).
 * XOR/INV gate counts and topology are completely independent per branch.
 *
 * Wire index convention (per branch):
 *   - 0 <= w < input_count:                           input wire w
 *   - input_count <= w < input_count+gate_count[b]:   gate output (w - input_count)
 *   - input_count + gate_count[b]:                    virtual one_wire (constant 1)
 *   Gate g must only reference inputs or outputs of gates 0..g-1.
 *
 * Protocol:
 *   authenticate_and_multiply(): commit n_in inputs + n_× AND triples.
 *     Branch-independent — verifier needs no circuit knowledge.
 *     COT cost: n_in + 3*n_× per step.
 *   generate_proofs(): per-branch topology check.
 *     Each branch walks its own circuit and checks that the committed AND
 *     triples are correctly wired. Active branch produces zero proof token.
 *   final_proof(): LPZK proves product of branch tokens is zero.
 *
 * Setup: call set_branch_circuit(b, ...) and set_output_source(b, ...) for each branch.
 */

template<typename IO, typename COTType = FerretCOT<IO>>
class BoolBatchedDisjunction {
public:
    int party;
    int threads;
    int input_count;
    int connect_count; // first connect_count I/O positions connected across steps
    int mul_count;    // n_× — same for all branches
    int branch_sz;
    int batch_sz;

    // Per-branch circuit definitions (set via set_branch_circuit)
    int *gate_count_per_branch = nullptr;  // total gates (AND+XOR+INV) per branch
    int **gate_type = nullptr;             // gate_type[b][g]
    int **wire_left = nullptr;             // wire_left[b][g]
    int **wire_right = nullptr;            // wire_right[b][g]

    // Per-branch output source mapping
    // output_source[branch][i] < 0: pass-through (output[i] = input[i])
    // output_source[branch][i] >= 0: gate output index g
    int **output_source = nullptr;

    // Authenticated values
    Bit *inputs = nullptr;
    Bit *mul_le = nullptr;
    Bit *mul_ri = nullptr;
    Bit *mul_ou = nullptr;
    Bit one;
    block delta;

    // Connection challenge coefficients
    block *conn_chi = nullptr;

    // Prover-only: which branch is active for each batch
    int *active_branch_map = nullptr;
    bool owns_branch_map = true;  // false if set externally
    // Prover-only: final output state after last step
    bool *prover_final_state = nullptr;

    // Prover-only: initial input state (if not null, seeds next_input_vals at step 0).
    bool *prover_initial_state = nullptr;

    // Prover-only: per-step hint injection.
    // If hint_bits != nullptr, at each step batch_id the hint region
    // [hint_offset .. hint_offset + hint_count) in next_input_vals is
    // overwritten from hint_bits[batch_id * hint_count ..].
    bool *hint_bits = nullptr;
    int hint_offset = 0;
    int hint_count = 0;
    // Prover-only: cleartext values per step (for all-branch evaluation)
    bool *input_vals = nullptr;
    bool *mul_le_vals = nullptr;
    bool *mul_ri_vals = nullptr;

    // Protocol buffers
    block *chis = nullptr;
    block *proofs = nullptr;
    block *values = nullptr;
    block *aut = nullptr;
    block *val = nullptr;

    // Per-step outputs from the optimized generate_proofs():
    //   v_tokens[j]     = topology proof token for the active branch at step j (should be 0)
    //   alpha_tokens[j] = ct[active_branch_map[j]], the topology fingerprint of the active branch
    block *v_tokens = nullptr;
    block *alpha_tokens = nullptr;

    // ── Authenticated step records ──────────────────────────────────────
    //
    // Per-step, per-branch 128-bit IT-MACs that commit to the execution
    // metadata needed by the external proving system (binius).
    //
    // Each step record packs 124 bits into a GF(2^128) element using the
    // polynomial basis {1, α, α², ...}.
    //
    // Tower-friendly packing: fields are aligned to binary tower boundaries
    // (B32, B16, B8, B1) for efficient decomposition inside binius.
    //
    // The bits are (LSB → MSB):
    //
    //   Bits 0–31:    value          (input wires 160–191)
    //   Bits 32–63:   imm            (input wires 112–143)
    //   Bits 64–79:   pc             (input wires 96–111)
    //   Bits 80–95:   next_pc        (input wires 192–207)  always pc+1
    //   Bits 96–111:  addr           (input wires 144–159)
    //   Bits 112–119: op/id          (const output bits 0–7)
    //   Bit 120:      is_mem_write   (const output bit 9)
    //   Bit 121:      is_mem_read    (const output bit 8)
    //   Bit 122:      has_immediate  (const output bit 10)
    //   Bit 123:      is_byte_sel_r2 (const output bit 11)
    //
    // For the active branch, this IT-MAC authenticates the true execution
    // values. The prover first commits these MACs, then later opens them
    // inside binius to prove memory consistency and instruction lookups.
    //
    // step_records[batch_id * branch_sz + bid]:
    //   ALICE holds the MAC, BOB holds the corresponding key.
    //   Relation: Key = MAC + f(α) · Δ  where f(α) ∈ GF(2^128) is the
    //   packed step record value.
    block *step_records = nullptr;

    // Prover-only: plaintext step records for all branches per step.
    // step_record_plaintexts[batch_id * branch_sz + bid] = 124-bit packed value (high 4 bits zero).
    block *step_record_plaintexts = nullptr;

    // Per-branch const gate indices: const_gate_idx[bid][i] = gate index
    // for the i-th const output wire of branch bid.
    int **const_gate_idx = nullptr;
    int *const_count_per_branch = nullptr;

    // Topology fingerprints: ct[bid] = topology token for branch bid
    // evaluated on the t-challenge instead of actual wire MACs.
    // ct[bid] = sum_k chi_k * t_left_k(bid) + chi_{n_x+k} * t_right_k(bid)
    // where t_left_k(bid) is the t-value propagated through branch bid's
    // XOR gates to the left input of AND gate k.
    // Computed once in generate_proofs() after the chi and t challenges.
    block *ct = nullptr;

    IO **ios;

    // External PCS mode: skip LPZK, export data instead
    bool use_external_pcs = false;
    // When true, both parties evaluate all branches per step (fills proofs[]/values[]).
    // When false, prover walks only the active branch (v_tokens/alpha_tokens).
    bool evaluate_all_branches = true;

    BoolBatchedDisjunction(IO **ios, int party, int input_count, int mul_count,
                           int branch_sz, int batch_sz, int connect_count = -1)
        : party(party), threads(1), input_count(input_count),
          connect_count(connect_count < 0 ? input_count : connect_count),
          mul_count(mul_count),
          branch_sz(branch_sz), batch_sz(batch_sz), ios(ios) {
        owns_zk_exec = false;

        // Allocate per-branch pointer arrays (entries filled by set_branch_circuit)
        gate_count_per_branch = new int[branch_sz]();
        gate_type  = new int*[branch_sz]();
        wire_left  = new int*[branch_sz]();
        wire_right = new int*[branch_sz]();
        const_gate_idx = new int*[branch_sz]();
        const_count_per_branch = new int[branch_sz]();

        setup_default_output_source();
        preallocate();
    }

    ~BoolBatchedDisjunction() {
        for (int b = 0; b < branch_sz; b++) {
            delete[] gate_type[b];
            delete[] wire_left[b];
            delete[] wire_right[b];
        }
        delete[] gate_type;
        delete[] wire_left;
        delete[] wire_right;
        delete[] gate_count_per_branch;

        if (output_source != nullptr) {
            for (int b = 0; b < branch_sz; b++) delete[] output_source[b];
            delete[] output_source;
        }
        delete[] inputs;
        delete[] mul_le;
        delete[] mul_ri;
        delete[] mul_ou;
        delete[] chis;
        delete[] proofs;
        delete[] values;
        delete[] aut;
        delete[] val;
        delete[] conn_chi;
        if (owns_branch_map) delete[] active_branch_map;
        delete[] prover_final_state;
        delete[] ct;
        delete[] v_tokens;
        delete[] alpha_tokens;
        delete[] step_records;
        delete[] step_record_plaintexts;
        if (const_gate_idx) {
            for (int b = 0; b < branch_sz; b++) delete[] const_gate_idx[b];
            delete[] const_gate_idx;
        }
        delete[] const_count_per_branch;
        delete[] input_vals;
        delete[] mul_le_vals;
        delete[] mul_ri_vals;
        if (owns_zk_exec) {
            finalize_zk_bool<IO, COTType>();
        }
    }

private:
    bool owns_zk_exec = false;

    void preallocate() {
        long long input_total = (long long)input_count * batch_sz;
        long long mul_total   = (long long)mul_count * batch_sz;
        long long proof_total = (long long)branch_sz * batch_sz;
        long long conn_total  = (long long)(batch_sz - 1) * connect_count;
        long long chi_total   = (long long)mul_count * 2;

        inputs = new Bit[input_total];
        mul_le = new Bit[mul_total];
        mul_ri = new Bit[mul_total];
        mul_ou = new Bit[mul_total];
        chis   = new block[chi_total];
        proofs = new block[proof_total];
        values = new block[proof_total];
        step_records = new block[proof_total];
        if (conn_total > 0) conn_chi = new block[conn_total];

        volatile char *p;
        long long page_sz = 4096;
        p = (volatile char*)inputs; for (long long i = 0; i < input_total*(long long)sizeof(Bit); i += page_sz) p[i] = 0;
        p = (volatile char*)mul_le; for (long long i = 0; i < mul_total*(long long)sizeof(Bit); i += page_sz) p[i] = 0;
        p = (volatile char*)mul_ri; for (long long i = 0; i < mul_total*(long long)sizeof(Bit); i += page_sz) p[i] = 0;
        p = (volatile char*)mul_ou; for (long long i = 0; i < mul_total*(long long)sizeof(Bit); i += page_sz) p[i] = 0;
        p = (volatile char*)chis;   for (long long i = 0; i < chi_total*(long long)sizeof(block); i += page_sz) p[i] = 0;
        p = (volatile char*)proofs; for (long long i = 0; i < proof_total*(long long)sizeof(block); i += page_sz) p[i] = 0;
        p = (volatile char*)values; for (long long i = 0; i < proof_total*(long long)sizeof(block); i += page_sz) p[i] = 0;
        if (conn_total > 0) {
            p = (volatile char*)conn_chi; for (long long i = 0; i < conn_total*(long long)sizeof(block); i += page_sz) p[i] = 0;
        }
    }

    void setup_default_output_source() {
        output_source = new int*[branch_sz];
        for (int b = 0; b < branch_sz; b++) {
            output_source[b] = new int[input_count];
            for (int i = 0; i < input_count; i++) output_source[b][i] = -1;
        }
    }

public:

    OSTriple<IO, COTType>* get_ostriple() {
        if (party == ALICE)
            return ((ZKProver<IO, COTType>*)(ProtocolExecution::prot_exec))->ostriple;
        else
            return ((ZKVerifier<IO, COTType>*)(ProtocolExecution::prot_exec))->ostriple;
    }

    void set_external_pcs_mode(bool enabled) { use_external_pcs = enabled; }
    void set_evaluate_all_branches(bool v) { evaluate_all_branches = v; }

    void setup_active_branch_map() {
        if (active_branch_map && owns_branch_map) delete[] active_branch_map;
        active_branch_map = new int[batch_sz];
        owns_branch_map = true;
        PRG prg;
        for (int b = 0; b < batch_sz; b++) {
            uint32_t r;
            prg.random_data(&r, sizeof(uint32_t));
            active_branch_map[b] = r % branch_sz;
        }
    }

    // Use an externally-provided branch map (not owned — caller must keep alive).
    void set_active_branch_map(int *map) {
        if (active_branch_map && owns_branch_map) delete[] active_branch_map;
        active_branch_map = map;
        owns_branch_map = false;
    }

    // Set per-step hint data to inject into next_input_vals.
    // hints: flat array of batch_sz * count bools (LSB-first bit expansion).
    // offset: starting bit position in input_vals to write hints.
    // count: number of hint bits per step.
    void set_hints(bool *hints, int offset, int count) {
        hint_bits = hints;
        hint_offset = offset;
        hint_count = count;
    }

    block* get_branch_proofs() { return proofs; }
    block* get_branch_values() { return values; }
    int get_branch_count() { return branch_sz; }
    int get_batch_count()  { return batch_sz; }

    // Set the full circuit for a branch: gate count, gate types, and wire routing.
    // gc = total gate count (AND + XOR + INV) for this branch.
    // If the circuit has fewer than mul_count AND gates, dummy AND(0,0) gates
    // are appended automatically so the chi indexing stays consistent.
    void set_branch_circuit(int branch, int gc, int *gt, int *left, int *right) {
        delete[] gate_type[branch];
        delete[] wire_left[branch];
        delete[] wire_right[branch];

        // Count AND gates in the provided circuit
        int and_count = 0;
        for (int g = 0; g < gc; g++)
            if (gt[g] == GATE_AND) and_count++;

        int pad = mul_count - and_count;
        int total_gc = gc + pad;

        gate_count_per_branch[branch] = total_gc;
        gate_type[branch]  = new int[total_gc];
        wire_left[branch]  = new int[total_gc];
        wire_right[branch] = new int[total_gc];

        memcpy(gate_type[branch],  gt,    gc * sizeof(int));
        memcpy(wire_left[branch],  left,  gc * sizeof(int));
        memcpy(wire_right[branch], right, gc * sizeof(int));

        // Pad with dummy AND(0,0) gates
        for (int g = gc; g < total_gc; g++) {
            gate_type[branch][g]  = GATE_AND;
            wire_left[branch][g]  = 0;
            wire_right[branch][g] = 0;
        }
    }

    // Set output source for a branch.
    // source[i] < 0: pass-through. source[i] >= 0: gate output index g.
    void set_output_source(int branch, int *source) {
        for (int i = 0; i < input_count; i++)
            output_source[branch][i] = source[i];
    }

    // Set const gate indices for a branch (for step record computation).
    // gate_indices[i] = gate index of the i-th const output wire.
    void set_branch_const_gates(int branch, int count, int *gate_indices) {
        delete[] const_gate_idx[branch];
        const_count_per_branch[branch] = count;
        const_gate_idx[branch] = new int[count];
        memcpy(const_gate_idx[branch], gate_indices, count * sizeof(int));
    }

    // Compute the authenticated step record from wire MACs/keys after a
    // branch circuit walk. pw[] holds the per-wire MAC (prover) or key
    // (verifier) values. Uses Horner evaluation in GF(2^128) to pack
    // 124 bit-level IT-MACs into a single GF(2^128) IT-MAC.
    //
    // Const output bit indices (within const_gate_idx[bid]):
    //   0–7: id (op), 8: is_mem_read, 9: is_mem_write,
    //   10: has_immediate, 11: is_byte_sel_r2
    //
    // Returns: MAC (prover) or Key (verifier) for the packed step record.
    block compute_step_record(block *pw, int bid) {
        // Horner from highest bit (123) down to bit 0.
        block acc = _mm_setzero_si128();

        // Bit 123: is_byte_sel_r2 (const bit 11)
        acc = gf128_mul_x(acc);
        acc = _mm_xor_si128(acc, pw[input_count + const_gate_idx[bid][11]]);
        // Bit 122: has_immediate (const bit 10)
        acc = gf128_mul_x(acc);
        acc = _mm_xor_si128(acc, pw[input_count + const_gate_idx[bid][10]]);
        // Bit 121: is_mem_read (const bit 8)
        acc = gf128_mul_x(acc);
        acc = _mm_xor_si128(acc, pw[input_count + const_gate_idx[bid][8]]);
        // Bit 120: is_mem_write (const bit 9)
        acc = gf128_mul_x(acc);
        acc = _mm_xor_si128(acc, pw[input_count + const_gate_idx[bid][9]]);
        // Bits 119..112: op/id[7..0] (const bits 7..0)
        for (int i = 7; i >= 0; i--) {
            acc = gf128_mul_x(acc);
            acc = _mm_xor_si128(acc, pw[input_count + const_gate_idx[bid][i]]);
        }
        // Bits 111..96: addr[15..0] (gate output if computed, else input wire)
        for (int i = 15; i >= 0; i--) {
            acc = gf128_mul_x(acc);
            int src = output_source[bid][144 + i];
            acc = _mm_xor_si128(acc, (src >= 0) ? pw[input_count + src] : pw[144 + i]);
        }
        // Bits 95..80: next_pc[15..0] (input wires 207..192)
        for (int i = 15; i >= 0; i--) {
            acc = gf128_mul_x(acc);
            acc = _mm_xor_si128(acc, pw[192 + i]);
        }
        // Bits 79..64: pc[15..0] (input wires 111..96)
        for (int i = 15; i >= 0; i--) {
            acc = gf128_mul_x(acc);
            acc = _mm_xor_si128(acc, pw[96 + i]);
        }
        // Bits 63..32: imm[31..0] (input wires 143..112)
        for (int i = 31; i >= 0; i--) {
            acc = gf128_mul_x(acc);
            acc = _mm_xor_si128(acc, pw[112 + i]);
        }
        // Bits 31..0: value[31..0] (gate output if computed, else input wire)
        for (int i = 31; i >= 0; i--) {
            acc = gf128_mul_x(acc);
            int src = output_source[bid][160 + i];
            acc = _mm_xor_si128(acc, (src >= 0) ? pw[input_count + src] : pw[160 + i]);
        }

        return acc;
    }

    // Compute the plaintext step record from cleartext wire values.
    // Same bit layout as compute_step_record(), but packs bool values
    // into a 128-bit block via direct bit-setting (polynomial basis).
    // cw[] must have been fully evaluated (all gates computed).
    block compute_step_record_plaintext(bool *cw, int bid) {
        uint64_t lo = 0, hi = 0;
        // Bits 0–31: value (gate output if computed, else input wire)
        for (int i = 0; i < 32; i++) {
            int src = output_source[bid][160 + i];
            bool bit = (src >= 0) ? cw[input_count + src] : cw[160 + i];
            if (bit) lo |= (1ULL << i);
        }
        // Bits 32–63: imm (input wires 112–143)
        for (int i = 0; i < 32; i++)
            if (cw[112 + i]) lo |= (1ULL << (32 + i));
        // Bits 64–79: pc (input wires 96–111)
        for (int i = 0; i < 16; i++)
            if (cw[96 + i]) hi |= (1ULL << i);
        // Bits 80–95: next_pc (input wires 192–207)
        for (int i = 0; i < 16; i++)
            if (cw[192 + i]) hi |= (1ULL << (16 + i));
        // Bits 96–111: addr (gate output if computed, else input wire)
        for (int i = 0; i < 16; i++) {
            int src = output_source[bid][144 + i];
            bool bit = (src >= 0) ? cw[input_count + src] : cw[144 + i];
            if (bit) hi |= (1ULL << (32 + i));
        }
        // Bits 112–119: op/id (const bits 0–7)
        for (int i = 0; i < 8; i++)
            if (cw[input_count + const_gate_idx[bid][i]]) hi |= (1ULL << (48 + i));
        // Bit 120: is_mem_write (const bit 9)
        if (cw[input_count + const_gate_idx[bid][9]]) hi |= (1ULL << 56);
        // Bit 121: is_mem_read (const bit 8)
        if (cw[input_count + const_gate_idx[bid][8]]) hi |= (1ULL << 57);
        // Bit 122: has_immediate (const bit 10)
        if (cw[input_count + const_gate_idx[bid][10]]) hi |= (1ULL << 58);
        // Bit 123: is_byte_sel_r2 (const bit 11)
        if (cw[input_count + const_gate_idx[bid][11]]) hi |= (1ULL << 59);
        return makeBlock(hi, lo);
    }

    // Step 1: Authenticate inputs and AND gate triples.
    //
    // Branch-independent: the verifier participates in COT without knowing
    // which branch is active or any circuit topology.
    //
    // Sequence per step:
    //   - n_in COTs: authenticate input wires
    //   - n_× * 3 COTs: for each AND triple, authenticate left, right, then AND
    //
    // The prover evaluates the active branch in cleartext to determine AND gate
    // input values, then provides them as fresh authenticated values.
    void authenticate_and_multiply() {
        if (party == ALICE && active_branch_map == nullptr)
            setup_active_branch_map();

        // Authenticated constant 1 (for XOR(a, one_wire) = NOT(a))
        one = Bit(true, PUBLIC);

        // Prover: AND gate values from cleartext evaluation (lv[k], rv[k])
        bool *lv_vals = new bool[mul_count]();
        bool *rv_vals = new bool[mul_count]();
        bool *next_input_vals = nullptr;
        if (party == ALICE) {
            next_input_vals = new bool[input_count]();
            if (prover_initial_state) {
                memcpy(next_input_vals, prover_initial_state, input_count);
            }
            if (evaluate_all_branches) {
                delete[] input_vals; delete[] mul_le_vals; delete[] mul_ri_vals;
                input_vals   = new bool[(long long)input_count * batch_sz]();
                mul_le_vals  = new bool[(long long)mul_count * batch_sz]();
                mul_ri_vals  = new bool[(long long)mul_count * batch_sz]();
            }
        }

        for (int batch_id = 0; batch_id < batch_sz; batch_id++) {
            long long in_off  = (long long)batch_id * input_count;
            long long mul_off = (long long)batch_id * mul_count;

            // Inject per-step hints (prover only, overwrites hint region)
            if (party == ALICE && hint_bits != nullptr) {
                long long h_off = (long long)batch_id * hint_count;
                for (int i = 0; i < hint_count; i++)
                    next_input_vals[hint_offset + i] = hint_bits[h_off + i];
            }

            // Save cleartext input values before authentication
            if (evaluate_all_branches && party == ALICE) {
                for (int i = 0; i < input_count; i++)
                    input_vals[in_off + i] = next_input_vals[i];
            }

            // Authenticate inputs (n_in COTs)
            for (int i = 0; i < input_count; i++) {
                bool v = (party == ALICE) ? next_input_vals[i] : false;
                inputs[in_off + i] = Bit(v, ALICE);
            }

            // Prover: evaluate active branch in cleartext to get AND gate values
            if (party == ALICE) {
                int br = active_branch_map[batch_id];
                int gc = gate_count_per_branch[br];
                int one_wire_idx = input_count + gc;

                bool *cw = new bool[one_wire_idx + 1];
                for (int i = 0; i < input_count; i++) cw[i] = next_input_vals[i];
                cw[one_wire_idx] = true;

                int and_idx = 0;
                for (int g = 0; g < gc; g++) {
                    int l = wire_left[br][g], r = wire_right[br][g];
                    switch (gate_type[br][g]) {
                        case GATE_XOR:
                            cw[input_count + g] = cw[l] ^ cw[r];
                            break;
                        case GATE_INV:
                            cw[input_count + g] = !cw[l];
                            break;
                        case GATE_AND:
                            lv_vals[and_idx] = cw[l];
                            rv_vals[and_idx] = cw[r];
                            cw[input_count + g] = cw[l] & cw[r];
                            and_idx++;
                            break;
                    }
                }

                // Chain outputs
                bool *dst;
                if (batch_id < batch_sz - 1) {
                    dst = next_input_vals;
                } else {
                    prover_final_state = new bool[input_count];
                    dst = prover_final_state;
                }
                for (int i = 0; i < input_count; i++) {
                    if (output_source[br][i] < 0)
                        dst[i] = cw[i];
                    else
                        dst[i] = cw[input_count + output_source[br][i]];
                }
                // Sanity check: for hint wires where the circuit computes
                // an output (output_source >= 0), verify the computed value
                // matches the injected hint.
                for (int i = hint_offset; i < input_count; i++) {
                    if (output_source[br][i] >= 0) {
                        bool computed = cw[input_count + output_source[br][i]];
                        bool hint_val = cw[i];
                        if (computed != hint_val) {
                            fprintf(stderr, "HINT MISMATCH step %d wire %d: "
                                "hint=%d computed=%d (branch %d)\n",
                                batch_id, i, (int)hint_val, (int)computed, br);
                        }
                    }
                }

                delete[] cw;
            }

            // Save cleartext AND triple values
            if (evaluate_all_branches && party == ALICE) {
                for (int k = 0; k < mul_count; k++) {
                    mul_le_vals[mul_off + k] = lv_vals[k];
                    mul_ri_vals[mul_off + k] = rv_vals[k];
                }
            }

            // Authenticate n_× AND triples (n_× * 3 COTs, both parties)
            for (int k = 0; k < mul_count; k++) {
                mul_le[mul_off + k] = Bit(lv_vals[k], ALICE);
                mul_ri[mul_off + k] = Bit(rv_vals[k], ALICE);
                mul_ou[mul_off + k] = mul_le[mul_off + k] & mul_ri[mul_off + k];
            }
        }

        delete[] lv_vals;
        delete[] rv_vals;
        if (next_input_vals) delete[] next_input_vals;

        delta = get_bool_delta<IO, COTType>(party);
    }

    void generate_connection_challenge() {
        long long conn_count = (long long)(batch_sz - 1) * connect_count;
        block seed;
        if (party == ALICE) {
            ios[0]->recv_data(&seed, sizeof(block));
        } else {
            PRG().random_block(&seed, 1);
            ios[0]->send_data(&seed, sizeof(block));
            ios[0]->flush();
        }
        PRG prg(&seed);
        for (long long i = 0; i < conn_count; i++) {
            prg.random_block(&conn_chi[i], 1);
            conn_chi[i] = lo64(conn_chi[i]);
        }
    }

    // Step 2: Per-branch topology check.
    // Each branch walks its own circuit (with its own gate types and wiring),
    // checking that the committed AND triples are correctly placed.
    // The active branch's topology is consistent → zero proof token.
    void generate_proofs() {
        if (batch_sz > 1) generate_connection_challenge();

        long long chi_len = (long long)mul_count * 2;
        block s_seed;
        if (party == ALICE) {
            ios[0]->recv_data(&s_seed, sizeof(block));
        } else {
            PRG().random_block(&s_seed, 1);
            ios[0]->send_data(&s_seed, sizeof(block));
            ios[0]->flush();
        }
        PRG prg_s(&s_seed);
        for (long long i = 0; i < chi_len; i++) {
            prg_s.random_block(&chis[i], 1);
            chis[i] = lo64(chis[i]);
        }

        // Prover: allocate plaintext array for all branches' step records
        if (party == ALICE) {
            delete[] step_record_plaintexts;
            step_record_plaintexts = new block[(long long)batch_sz * branch_sz]();
        }

        if (evaluate_all_branches) {
            // All-branch mode: both parties walk every branch per step.
            // proofs[batch_id * branch_sz + bid] = topology token (MAC for prover, Key for verifier)
            // values[batch_id * branch_sz + bid] = cleartext value token (prover only)
            for (int batch_id = 0; batch_id < batch_sz; batch_id++) {
                long long in_off  = (long long)batch_id * input_count;
                long long mul_off = (long long)batch_id * mul_count;

                // MAC/Key walk: both parties walk ALL branches
                for (int bid = 0; bid < branch_sz; bid++) {
                    int gc = gate_count_per_branch[bid];
                    int one_wire_idx = input_count + gc;

                    block *pw = new block[one_wire_idx + 1];
                    pw[one_wire_idx] = one.bit;
                    for (int i = 0; i < input_count; i++)
                        pw[i] = inputs[in_off + i].bit;

                    __m128i proof_acc = _mm_setzero_si128(); // unreduced accumulator
                    int and_idx = 0;
                    for (int g = 0; g < gc; g++) {
                        int l = wire_left[bid][g], r = wire_right[bid][g];
                        switch (gate_type[bid][g]) {
                            case GATE_XOR:
                                pw[input_count + g] = pw[l] ^ pw[r];
                                break;
                            case GATE_INV:
                                pw[input_count + g] = pw[l] ^ pw[one_wire_idx];
                                break;
                            case GATE_AND: {
                                block diff_l = lo64(mul_le[mul_off + and_idx].bit ^ pw[l]);
                                proof_acc = _mm_xor_si128(proof_acc, clmul64(chis[and_idx], diff_l));

                                block diff_r = lo64(mul_ri[mul_off + and_idx].bit ^ pw[r]);
                                proof_acc = _mm_xor_si128(proof_acc, clmul64(chis[mul_count + and_idx], diff_r));

                                pw[input_count + g] = mul_ou[mul_off + and_idx].bit;
                                and_idx++;
                                break;
                            }
                        }
                    }

                    // Connection proof (only first connect_count positions)
                    if (batch_sz > 1 && batch_id < batch_sz - 1) {
                        long long in_off_next = (long long)(batch_id + 1) * input_count;
                        for (int i = 0; i < connect_count; i++) {
                            long long chi_idx = (long long)batch_id * connect_count + i;
                            block prev_output = (output_source[bid][i] < 0)
                                ? pw[i]
                                : pw[input_count + output_source[bid][i]];
                            block diff = lo64(inputs[in_off_next + i].bit ^ prev_output);
                            proof_acc = _mm_xor_si128(proof_acc, clmul64(conn_chi[chi_idx], diff));
                        }
                    }

                    proofs[batch_id * branch_sz + bid] = reduce64(proof_acc);

                    // Authenticated step record: pack 126 bit-MACs into one
                    // GF(2^128) IT-MAC. Committed here, opened later in binius
                    // to prove memory accesses and instruction lookups.
                    if (const_gate_idx[bid] != nullptr)
                        step_records[batch_id * branch_sz + bid] = compute_step_record(pw, bid);

                    delete[] pw;
                }

                // Prover only: cleartext value walk for values[]
                if (party == ALICE) {
                    for (int bid = 0; bid < branch_sz; bid++) {
                        int gc = gate_count_per_branch[bid];
                        int one_wire_idx = input_count + gc;

                        bool *cw = new bool[one_wire_idx + 1];
                        cw[one_wire_idx] = true;
                        for (int i = 0; i < input_count; i++)
                            cw[i] = input_vals[in_off + i];

                        block value = makeBlock(0, 0);
                        int and_idx = 0;
                        for (int g = 0; g < gc; g++) {
                            int l = wire_left[bid][g], r = wire_right[bid][g];
                            switch (gate_type[bid][g]) {
                                case GATE_XOR:
                                    cw[input_count + g] = cw[l] ^ cw[r];
                                    break;
                                case GATE_INV:
                                    cw[input_count + g] = !cw[l];
                                    break;
                                case GATE_AND: {
                                    bool val_l = mul_le_vals[mul_off + and_idx];
                                    bool val_r = mul_ri_vals[mul_off + and_idx];
                                    if (val_l != cw[l])
                                        value = value ^ chis[and_idx];
                                    if (val_r != cw[r])
                                        value = value ^ chis[mul_count + and_idx];
                                    cw[input_count + g] = val_l & val_r;
                                    and_idx++;
                                    break;
                                }
                            }
                        }

                        // Connection value (only first connect_count positions)
                        if (batch_sz > 1 && batch_id < batch_sz - 1) {
                            for (int i = 0; i < connect_count; i++) {
                                long long chi_idx = (long long)batch_id * connect_count + i;
                                bool prev_out = (output_source[bid][i] < 0)
                                    ? cw[i]
                                    : cw[input_count + output_source[bid][i]];
                                bool next_in = input_vals[(long long)(batch_id + 1) * input_count + i];
                                if (next_in != prev_out)
                                    value = value ^ conn_chi[chi_idx];
                            }
                        }

                        values[batch_id * branch_sz + bid] = value;

                        // Compute plaintext step record for all branches
                        if (const_gate_idx[bid] != nullptr)
                            step_record_plaintexts[(long long)batch_id * branch_sz + bid] = compute_step_record_plaintext(cw, bid);

                        delete[] cw;
                    }
                }
            }
        } else {
        // Active-branch-only mode: prover walks only the active branch.
        // Uses t-challenge fingerprints; outputs v_tokens/alpha_tokens.

        // Second challenge t: topology fingerprint basis.
        // t_in[i]  = t-value for input wire i
        // t_ou[k]  = t-value for AND gate k's output wire
        // t_one    = t-value for the constant-1 wire
        // These are random GF(2^64) elements, shared between both parties.
        block t_seed;
        if (party == ALICE) {
            ios[0]->recv_data(&t_seed, sizeof(block));
        } else {
            PRG().random_block(&t_seed, 1);
            ios[0]->send_data(&t_seed, sizeof(block));
            ios[0]->flush();
        }
        PRG prg_t(&t_seed);

        block *t_in = new block[input_count];
        block *t_ou = new block[mul_count];
        block t_one;
        for (int i = 0; i < input_count; i++) { prg_t.random_block(&t_in[i], 1); t_in[i] = lo64(t_in[i]); }
        for (int k = 0; k < mul_count;   k++) { prg_t.random_block(&t_ou[k], 1); t_ou[k] = lo64(t_ou[k]); }
        prg_t.random_block(&t_one, 1); t_one = lo64(t_one);

        // Precompute ct[bid] for each branch: O(B * |C|), done once.
        // Walk branch bid's circuit with t-values in place of wire MACs.
        // XOR gates propagate t-values linearly.
        // AND gate k: accumulate chi_k * t_left + chi_{n_x+k} * t_right,
        //             then set the output wire to t_ou[k] (the t-challenge
        //             for that AND gate's extended-witness output entry).
        delete[] ct;
        ct = new block[branch_sz];
        for (int bid = 0; bid < branch_sz; bid++) {
            int gc = gate_count_per_branch[bid];
            int one_wire_idx = input_count + gc;
            block *tw = new block[one_wire_idx + 1];
            for (int i = 0; i < input_count; i++) tw[i] = t_in[i];
            tw[one_wire_idx] = t_one;

            __m128i ct_acc = _mm_setzero_si128();
            int and_idx = 0;
            for (int g = 0; g < gc; g++) {
                int l = wire_left[bid][g], r = wire_right[bid][g];
                switch (gate_type[bid][g]) {
                    case GATE_XOR:
                        tw[input_count + g] = tw[l] ^ tw[r];
                        break;
                    case GATE_INV:
                        tw[input_count + g] = tw[l] ^ t_one;
                        break;
                    case GATE_AND: {
                        ct_acc = _mm_xor_si128(ct_acc, clmul64(chis[and_idx], tw[l]));
                        ct_acc = _mm_xor_si128(ct_acc, clmul64(chis[mul_count + and_idx], tw[r]));
                        tw[input_count + g] = t_ou[and_idx];
                        and_idx++;
                        break;
                    }
                }
            }
            ct[bid] = reduce64(ct_acc);
            delete[] tw;
        }

        delete[] t_in;
        delete[] t_ou;

        // Allocate per-step proof token arrays.
        // v_tokens[j]     = topology proof token for active branch at step j (should be 0)
        // alpha_tokens[j] = ct[active_branch_map[j]], the topology fingerprint of the active branch
        // NOTE: final_proof() will be updated in a future step to use these instead of LPZK.
        delete[] v_tokens;
        delete[] alpha_tokens;
        v_tokens     = new block[batch_sz]();
        alpha_tokens = new block[batch_sz]();

        for (int batch_id = 0; batch_id < batch_sz; batch_id++) {
            long long in_off  = (long long)batch_id * input_count;
            long long mul_off = (long long)batch_id * mul_count;

            if (party == ALICE) {
                // Prover: walk only the active branch. O(|C|) per step.
                int bid = active_branch_map[batch_id];
                int gc  = gate_count_per_branch[bid];
                int one_wire_idx = input_count + gc;

                block *pw = new block[one_wire_idx + 1];
                pw[one_wire_idx] = one.bit;
                for (int i = 0; i < input_count; i++)
                    pw[i] = inputs[in_off + i].bit;

                __m128i proof_acc = _mm_setzero_si128();
                int and_idx = 0;
                for (int g = 0; g < gc; g++) {
                    int l = wire_left[bid][g], r = wire_right[bid][g];
                    block left_wire  = pw[l];
                    block right_wire = pw[r];

                    switch (gate_type[bid][g]) {
                        case GATE_XOR:
                            pw[input_count + g] = left_wire ^ right_wire;
                            break;
                        case GATE_INV:
                            pw[input_count + g] = left_wire ^ pw[one_wire_idx];
                            break;
                        case GATE_AND: {
                            block diff_l = lo64(mul_le[mul_off + and_idx].bit ^ left_wire);
                            proof_acc = _mm_xor_si128(proof_acc, clmul64(chis[and_idx], diff_l));

                            block diff_r = lo64(mul_ri[mul_off + and_idx].bit ^ right_wire);
                            proof_acc = _mm_xor_si128(proof_acc, clmul64(chis[mul_count + and_idx], diff_r));

                            pw[input_count + g] = mul_ou[mul_off + and_idx].bit;
                            and_idx++;
                            break;
                        }
                    }
                }

                // Connection proof: active branch output must match next step's input
                // (only first connect_count positions).
                if (batch_sz > 1 && batch_id < batch_sz - 1) {
                    long long in_off_next = (long long)(batch_id + 1) * input_count;
                    for (int i = 0; i < connect_count; i++) {
                        long long chi_idx = (long long)batch_id * connect_count + i;
                        block prev_output = (output_source[bid][i] < 0)
                            ? pw[i]
                            : pw[input_count + output_source[bid][i]];

                        block diff = lo64(inputs[in_off_next + i].bit ^ prev_output);
                        proof_acc = _mm_xor_si128(proof_acc, clmul64(conn_chi[chi_idx], diff));
                    }
                }

                v_tokens[batch_id]     = reduce64(proof_acc);
                alpha_tokens[batch_id] = ct[bid];
                delete[] pw;
            }
            // BOB: no circuit walk here — deferred to updated final_proof() (step 3).
        }
        } // end else (!evaluate_all_branches)
    }

    // Step 3: Final proof - LPZK proves product of branch tokens is zero
    void final_proof() {
        if (use_external_pcs) {
            if (party == ALICE) {
                block sum_proof = makeBlock(0, 0);
                block sum_value = makeBlock(0, 0);
                for (int b = 0; b < batch_sz; b++) {
                    int ab = active_branch_map[b];
                    sum_proof = sum_proof ^ proofs[b * branch_sz + ab];
                    sum_value = sum_value ^ values[b * branch_sz + ab];
                }
                ios[0]->send_data(&sum_proof, sizeof(block));
                ios[0]->send_data(&sum_value, sizeof(block));
                ios[0]->flush();
            } else {
                block recv_mac, recv_val;
                ios[0]->recv_data(&recv_mac, sizeof(block));
                ios[0]->recv_data(&recv_val, sizeof(block));
                block zero = makeBlock(0, 0);
                if (!cmpBlock(&recv_val, &zero, 1))
                    error("PCS sanity check: value sum is not zero\n");
            }
            return;
        }

        aut = new block[branch_sz * batch_sz];
        val = new block[branch_sz * batch_sz];

        for (int batch_id = 0; batch_id < batch_sz; batch_id++) {
            for (int i = 0; i < branch_sz; i++)
                zkp_get_ope<IO, COTType>(aut[batch_id * branch_sz + i], val[batch_id * branch_sz + i]);

            int base = batch_id * branch_sz;

            if (party == ALICE) {
                block inter_mul[branch_sz];
                gfmul(values[base + 0], values[base + 1], &inter_mul[0]);
                for (int i = 1; i < branch_sz - 1; i++)
                    gfmul(inter_mul[i-1], values[base + i + 1], &inter_mul[i]);
                for (int i = 0; i < branch_sz - 1; i++) {
                    val[base + i] = val[base + i] ^ inter_mul[i];
                    ios[0]->send_data(&val[base + i], sizeof(block));
                    val[base + i] = inter_mul[i];
                }
                ios[0]->flush();

                uint64_t tmp1, tmp2;
                ios[0]->recv_data(&tmp1, sizeof(uint64_t));
                ios[0]->recv_data(&tmp2, sizeof(uint64_t));
                block kai = makeBlock(tmp1, tmp2);

                block coeff = makeBlock(0, 1);
                block A0 = makeBlock(0, 0);
                block A1 = makeBlock(0, 0);

                block tmp, tmpmul;
                gfmul(proofs[base + 0], proofs[base + 1], &tmp);
                gfmul(coeff, tmp, &tmpmul);    A0 = A0 ^ tmpmul;
                gfmul(values[base + 0], proofs[base + 1], &tmp);
                gfmul(coeff, tmp, &tmpmul);    A1 = A1 ^ tmpmul;
                gfmul(values[base + 1], proofs[base + 0], &tmp);
                gfmul(coeff, tmp, &tmpmul);    A1 = A1 ^ tmpmul;
                gfmul(coeff, aut[base + 0], &tmpmul); A1 = A1 ^ tmpmul;
                gfmul(kai, coeff, &tmpmul);    coeff = tmpmul;

                for (int i = 1; i < branch_sz - 1; i++) {
                    gfmul(aut[base + i - 1], proofs[base + i + 1], &tmp);
                    gfmul(coeff, tmp, &tmpmul);           A0 = A0 ^ tmpmul;
                    gfmul(val[base + i - 1], proofs[base + i + 1], &tmp);
                    gfmul(coeff, tmp, &tmpmul);           A1 = A1 ^ tmpmul;
                    gfmul(aut[base + i - 1], values[base + i + 1], &tmp);
                    gfmul(coeff, tmp, &tmpmul);           A1 = A1 ^ tmpmul;
                    gfmul(coeff, aut[base + i], &tmpmul); A1 = A1 ^ tmpmul;
                    gfmul(kai, coeff, &tmpmul);           coeff = tmpmul;
                }

                A0 = A0 ^ aut[base + branch_sz - 1];
                A1 = A1 ^ val[base + branch_sz - 1];
                ios[0]->send_data(&A0, sizeof(block));
                ios[0]->send_data(&A1, sizeof(block));
                ios[0]->send_data(&aut[base + branch_sz - 2], sizeof(block));
                ios[0]->flush();

            } else {
                for (int i = 0; i < branch_sz - 1; i++) {
                    block tmp, tmpmul;
                    ios[0]->recv_data(&tmp, sizeof(block));
                    gfmul(delta, tmp, &tmpmul);
                    aut[base + i] = aut[base + i] ^ tmpmul;
                }

                PRG tmpprg;
                uint64_t tmp1, tmp2;
                tmpprg.random_data(&tmp1, sizeof(uint64_t));
                tmpprg.random_data(&tmp2, sizeof(uint64_t));
                ios[0]->send_data(&tmp1, sizeof(uint64_t));
                ios[0]->send_data(&tmp2, sizeof(uint64_t));
                ios[0]->flush();
                block kai = makeBlock(tmp1, tmp2);

                block coeff = makeBlock(0, 1);
                block accum = makeBlock(0, 0);

                block tmp, tmpmul;
                gfmul(proofs[base + 0], proofs[base + 1], &tmp);
                gfmul(coeff, tmp, &tmpmul);      accum = accum ^ tmpmul;
                gfmul(aut[base + 0], delta, &tmp);
                gfmul(coeff, tmp, &tmpmul);      accum = accum ^ tmpmul;
                gfmul(kai, coeff, &tmpmul);      coeff = tmpmul;

                for (int i = 1; i < branch_sz - 1; i++) {
                    gfmul(aut[base + i - 1], proofs[base + i + 1], &tmp);
                    gfmul(coeff, tmp, &tmpmul);      accum = accum ^ tmpmul;
                    gfmul(aut[base + i], delta, &tmp);
                    gfmul(coeff, tmp, &tmpmul);      accum = accum ^ tmpmul;
                    gfmul(kai, coeff, &tmpmul);      coeff = tmpmul;
                }
                accum = accum ^ aut[base + branch_sz - 1];

                block A0, A1;
                ios[0]->recv_data(&A0, sizeof(block));
                ios[0]->recv_data(&A1, sizeof(block));
                gfmul(A1, delta, &tmp);
                tmp = tmp ^ A0;
                if (!cmpBlock(&tmp, &accum, 1))
                    error("LPZK check failed\n");

                block final_check;
                ios[0]->recv_data(&final_check, sizeof(block));
                if (!cmpBlock(&final_check, &aut[base + branch_sz - 2], 1))
                    error("Final 0 check failed\n");
            }
        }
    }

    // Write step records to a binary file.
    //
    // Prover format:
    //   batch_sz: u32, branch_count: u32
    //   per step: active_branch: u16,
    //             plaintexts: [u8;16] * branch_count,
    //             step_record_macs: [u8;16] * branch_count
    //
    // Verifier format:
    //   batch_sz: u32, branch_count: u32
    //   per step: step_record_keys: [u8;16] * branch_count
    void write_step_records(const char *path) {
        FILE *f = fopen(path, "wb");
        if (!f) { error("Failed to open step records output file\n"); return; }

        uint32_t bs = (uint32_t)batch_sz;
        uint32_t bc = (uint32_t)branch_sz;
        fwrite(&bs, 4, 1, f);
        fwrite(&bc, 4, 1, f);

        for (int b = 0; b < batch_sz; b++) {
            if (party == ALICE) {
                uint16_t ab = (uint16_t)active_branch_map[b];
                fwrite(&ab, 2, 1, f);
                fwrite(&step_record_plaintexts[(long long)b * branch_sz], 16 * branch_sz, 1, f);
            }
            fwrite(&step_records[(long long)b * branch_sz], 16 * branch_sz, 1, f);
        }

        fclose(f);
    }

    // Write delta to a separate file (verifier only).
    // Format: delta: [u8;16]
    // Write zero-product proof data to a binary file (verifier only).
    //
    // Format:
    //   batch_sz: u32, branch_count: u32
    //   per step: topology_keys: [u8;8] * branch_count   (GF(2^64))
    void write_zero_product_verifier(const char *path) {
        if (party == ALICE) return;
        FILE *f = fopen(path, "wb");
        if (!f) { error("Failed to open zero product verifier output file\n"); return; }

        uint32_t bs = (uint32_t)batch_sz;
        uint32_t bc = (uint32_t)branch_sz;
        fwrite(&bs, 4, 1, f);
        fwrite(&bc, 4, 1, f);

        for (int b = 0; b < batch_sz; b++) {
            for (int bid = 0; bid < branch_sz; bid++)
                fwrite(&proofs[(long long)b * branch_sz + bid], 8, 1, f);
        }

        fclose(f);
    }

    void write_delta(const char *path) {
        if (party == ALICE) return;
        FILE *f = fopen(path, "wb");
        if (!f) { error("Failed to open delta output file\n"); return; }
        fwrite(&delta, 16, 1, f);
        fclose(f);
    }

    // Write zero-product proof data to a binary file (prover only).
    //
    // Format:
    //   batch_sz: u32, branch_count: u32
    //   per step: active_branch: u16,
    //             for each branch: mac: [u8;8], plaintext: [u8;8]   (GF(2^64))
    //
    // Active branch: mac=0, plaintext=0. Inactive: mac≠0, plaintext≠0.
    void write_zero_product(const char *path) {
        if (party != ALICE) return;
        FILE *f = fopen(path, "wb");
        if (!f) { error("Failed to open zero product output file\n"); return; }

        uint32_t bs = (uint32_t)batch_sz;
        uint32_t bc = (uint32_t)branch_sz;
        fwrite(&bs, 4, 1, f);
        fwrite(&bc, 4, 1, f);

        for (int b = 0; b < batch_sz; b++) {
            uint16_t ab = (uint16_t)active_branch_map[b];
            fwrite(&ab, 2, 1, f);
            for (int bid = 0; bid < branch_sz; bid++) {
                fwrite(&proofs[(long long)b * branch_sz + bid], 8, 1, f);
                fwrite(&values[(long long)b * branch_sz + bid], 8, 1, f);
            }
        }

        fclose(f);
    }
};

} // namespace emp

#endif // EMP_ZK_BOOL_BATCHED_DISJUNCTION_H__
