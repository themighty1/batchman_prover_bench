#ifndef EMP_ZK_BOOL_BATCHED_DISJUNCTION_H__
#define EMP_ZK_BOOL_BATCHED_DISJUNCTION_H__

#include "emp-zk/emp-zk-bool/emp-zk-bool.h"
#include <algorithm>

namespace emp {

// NOTE: The original protocol lifts single-bit MACs into GF(2^128) via random
// linear combination (RLC). This implementation uses GF(2^64) instead, which
// we assume provides sufficient statistical security parameter (SSP) for MAC
// forgery resistance. This assumption needs thorough investigation to confirm
// it does not break the security claims in the original paper.

// GF(2^64) multiplication using irreducible polynomial x^64 + x^4 + x^3 + x + 1
// Single PCLMULQDQ + reduction, ~3-4x faster than GF(2^128) gfmul.
// Operates on lower 64 bits of each block; upper 64 bits are ignored/zeroed.
inline void gfmul64(block a, block b, block *res) {
    // Carry-less multiply of lower 64-bit halves -> 128-bit result
    __m128i product = _mm_clmulepi64_si128(a, b, 0x00);

    // Reduce modulo x^64 + x^4 + x^3 + x + 1
    // High 64 bits of product need to be folded back
    __m128i hi = _mm_srli_si128(product, 8);  // high 64 bits -> low position

    // Reduction: hi * (x^4 + x^3 + x + 1) in the low 64 bits
    __m128i r1 = _mm_slli_epi64(hi, 4);   // hi * x^4
    __m128i r2 = _mm_slli_epi64(hi, 3);   // hi * x^3
    __m128i r3 = _mm_slli_epi64(hi, 1);   // hi * x
    // hi * 1 = hi itself

    __m128i reduced = _mm_xor_si128(r1, r2);
    reduced = _mm_xor_si128(reduced, r3);
    reduced = _mm_xor_si128(reduced, hi);

    // XOR with low 64 bits of product
    *res = _mm_xor_si128(product, reduced);

    // Clear upper 64 bits
    *res = _mm_and_si128(*res, _mm_set_epi64x(0, 0xFFFFFFFFFFFFFFFFULL));
}

// Mask a block to its lower 64 bits (for extracting the GF(2^64) component of a MAC)
inline block lo64(block x) {
    return _mm_and_si128(x, _mm_set_epi64x(0, 0xFFFFFFFFFFFFFFFFULL));
}

/**
 * BoolBatchedDisjunction - Batched disjunction protocol for boolean circuits
 *
 * zkVM Emulation:
 *   This benchmark emulates zkVM execution where each step has:
 *   - 96 computed output bits from AND gates:
 *     * 32 bits: instruction (what operation was executed)
 *     * 32 bits: PC (program counter - next instruction address)
 *     * 32 bits: register (the modified register value)
 *   - Remaining outputs are pass-through (unchanged state/registers)
 *
 *   The AND gates only operate on the first 96 input bits, and only
 *   the first 96 outputs come from AND gates. All other inputs pass through
 *   directly to outputs, representing unchanged VM state between steps.
 *
 * Parameters:
 *   - input_count: Number of input wires (also output count for connected reps)
 *   - mul_count: Number of AND gates
 *   - branch_sz: Number of branches in disjunction
 *   - batch_sz: Number of batch instances
 *
 * Output source mapping (per-branch):
 *   - output_source[branch][i] < 0: output[i] = input[i] (pass-through)
 *   - output_source[branch][i] >= 0: output[i] = mul_ou[output_source[branch][i]]
 */

// Number of inputs/outputs that are computed (go through AND gates)
// Represents zkVM: 32-bit instruction + 32-bit PC + 32-bit register = 96 bits
static const int ZKVM_BOOL_COMPUTED_COUNT = 96;

template<typename IO, typename COTType = FerretCOT<IO>>
class BoolBatchedDisjunction {
public:
    int party;
    int threads;
    int input_count;
    int mul_count;
    int branch_sz;
    int batch_sz;

    // Wire routing tables
    int *wire_left = nullptr;   // wire_left[m] = which input feeds AND gate m's left
    int *wire_right = nullptr;  // wire_right[m] = which input feeds AND gate m's right

    // Per-branch output source mapping
    // output_source[branch][i] < 0: pass-through (output[i] = input[i])
    // output_source[branch][i] >= 0: computed (output[i] = mul_ou[value])
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

    // Per-branch AND gate counts (95% small, 5% full size)
    int *mul_count_per_branch = nullptr;
    // Prover-only: which branch is active for each batch
    int *active_branch_map = nullptr;

    // Protocol buffers
    block *chis = nullptr;
    block *proofs = nullptr;
    block *values = nullptr;
    block *aut = nullptr;
    block *val = nullptr;

    // Per-batch prefix sum arrays (reused across batches)
    block *wire_proof_prefix = nullptr;
    block *wire_value_prefix = nullptr;
    block *aut_prefix = nullptr;
    bool *val_prefix = nullptr;

    IO **ios;

    // External PCS mode: skip LPZK, export data instead
    bool use_external_pcs = false;

    // Constructor that uses already-setup ZK framework (use for MockCOT)
    BoolBatchedDisjunction(IO **ios, int party, int input_count, int mul_count, int branch_sz, int batch_sz)
        : party(party), threads(1), input_count(input_count), mul_count(mul_count),
          branch_sz(branch_sz), batch_sz(batch_sz), ios(ios) {
        owns_zk_exec = false;
        setup_wire_routing();
        setup_mul_counts_per_branch();
        setup_default_output_source();
        preallocate();
    }

    ~BoolBatchedDisjunction() {
        delete[] wire_left;
        delete[] wire_right;
        if (output_source != nullptr) {
            for (int b = 0; b < branch_sz; b++) {
                delete[] output_source[b];
            }
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
        delete[] mul_count_per_branch;
        delete[] active_branch_map;
        delete[] wire_proof_prefix;
        delete[] wire_value_prefix;
        delete[] aut_prefix;
        delete[] val_prefix;
        if (owns_zk_exec) {
            finalize_zk_bool<IO, COTType>();
        }
    }

private:
    bool owns_zk_exec = false;

    // Pre-allocate all large arrays and touch every page to avoid page faults during timed section
    void preallocate() {
        long long input_total = (long long)input_count * batch_sz;
        long long mul_total = (long long)mul_count * batch_sz;
        long long proof_total = (long long)branch_sz * batch_sz;
        long long conn_total = (long long)(batch_sz - 1) * input_count;
        long long chi_total = (long long)mul_count * 2 + input_count;

        inputs = new Bit[input_total];
        mul_le = new Bit[mul_total];
        mul_ri = new Bit[mul_total];
        mul_ou = new Bit[mul_total];
        chis = new block[chi_total];
        proofs = new block[proof_total];
        values = new block[proof_total];
        if (conn_total > 0) conn_chi = new block[conn_total];
        wire_proof_prefix = new block[mul_count];
        wire_value_prefix = new block[mul_count];
        aut_prefix = new block[mul_count];
        val_prefix = new bool[mul_count];

        // Touch every page to force page faults now (before timing).
        // Use volatile write to prevent compiler from optimizing away.
        volatile char *p;
        long long page_sz = 4096;

        p = (volatile char*)inputs;
        for (long long i = 0; i < input_total * (long long)sizeof(Bit); i += page_sz) p[i] = 0;

        p = (volatile char*)mul_le;
        for (long long i = 0; i < mul_total * (long long)sizeof(Bit); i += page_sz) p[i] = 0;

        p = (volatile char*)mul_ri;
        for (long long i = 0; i < mul_total * (long long)sizeof(Bit); i += page_sz) p[i] = 0;

        p = (volatile char*)mul_ou;
        for (long long i = 0; i < mul_total * (long long)sizeof(Bit); i += page_sz) p[i] = 0;

        p = (volatile char*)chis;
        for (long long i = 0; i < chi_total * (long long)sizeof(block); i += page_sz) p[i] = 0;

        p = (volatile char*)proofs;
        for (long long i = 0; i < proof_total * (long long)sizeof(block); i += page_sz) p[i] = 0;

        p = (volatile char*)values;
        for (long long i = 0; i < proof_total * (long long)sizeof(block); i += page_sz) p[i] = 0;

        if (conn_total > 0) {
            p = (volatile char*)conn_chi;
            for (long long i = 0; i < conn_total * (long long)sizeof(block); i += page_sz) p[i] = 0;
        }
    }

    // Setup per-branch AND gate counts: 95% get 100 gates, 5% get mul_count (max)
    void setup_mul_counts_per_branch() {
        mul_count_per_branch = new int[branch_sz];
        int small_count = std::min(100, mul_count);
        int threshold = (int)(0.95 * branch_sz);
        for (int b = 0; b < branch_sz; b++) {
            mul_count_per_branch[b] = (b < threshold) ? small_count : mul_count;
        }
    }

    // Setup wire routing for zkVM emulation
    // Only the first ZKVM_BOOL_COMPUTED_COUNT inputs are used in AND gates
    // This represents: instruction, PC, modified register
    void setup_wire_routing() {
        wire_left = new int[mul_count];
        wire_right = new int[mul_count];

        // Only use first ZKVM_BOOL_COMPUTED_COUNT inputs in AND gates
        int computed = std::min(ZKVM_BOOL_COMPUTED_COUNT, input_count);
        for (int m = 0; m < mul_count; m++) {
            wire_left[m] = m % computed;
            wire_right[m] = (m + 1) % computed;
        }
    }

    // Setup default output source for zkVM emulation:
    // - bits 0-31 (instruction): same for all branches, from AND gates
    // - bits 32-63 (PC): same for all branches, from AND gates
    // - bits 64-95 (dest register): DIFFERS per branch, from AND gates
    // - bits 96+ : pass-through (unchanged VM state)
    //
    // This models realistic zkVM where only the destination register varies
    // between instructions (ADD writes to different reg than SUB, etc.)
    void setup_default_output_source() {
        output_source = new int*[branch_sz];

        // Ensure we have enough AND gates for branch-specific outputs
        // Need: 64 (shared) + 32 * num_unique_branches
        int shared_bits = 64;  // instruction + PC
        int reg_bits = 32;     // destination register

        for (int b = 0; b < branch_sz; b++) {
            output_source[b] = new int[input_count];
            int ng = mul_count_per_branch[b];
            for (int i = 0; i < input_count; i++) {
                if (i < shared_bits && i < ng) {
                    // instruction (32 bits) + PC (32 bits): same for all branches
                    output_source[b][i] = i;
                } else if (i < shared_bits + reg_bits && ng > shared_bits) {
                    // destination register (32 bits): differs per branch
                    // Each branch uses AND outputs from a different offset
                    int reg_bit = i - shared_bits;  // 0-31 within register
                    int available = ng - shared_bits;
                    int base = shared_bits + (b * reg_bits) % available;
                    output_source[b][i] = base + (reg_bit % available);
                } else {
                    // Pass-through: output = input (unchanged state)
                    output_source[b][i] = -1;
                }
            }
        }
    }

public:

    // Get ostriple for COT stats access
    OSTriple<IO, COTType>* get_ostriple() {
        if (party == ALICE) {
            return ((ZKProver<IO, COTType>*)(ProtocolExecution::prot_exec))->ostriple;
        } else {
            return ((ZKVerifier<IO, COTType>*)(ProtocolExecution::prot_exec))->ostriple;
        }
    }

    // Enable external PCS mode (call before final_proof)
    void set_external_pcs_mode(bool enabled) {
        use_external_pcs = enabled;
    }

    // Build active branch map (prover-only, random assignment)
    // Picks from even-indexed branches so mock output check produces zero proofs
    void setup_active_branch_map() {
        if (active_branch_map) delete[] active_branch_map;
        active_branch_map = new int[batch_sz];
        PRG prg;
        int num_even = (branch_sz + 1) / 2;  // number of even-indexed branches
        for (int b = 0; b < batch_sz; b++) {
            uint32_t r;
            prg.random_data(&r, sizeof(uint32_t));
            active_branch_map[b] = (r % num_even) * 2;
        }
    }

    // Getters for external PCS - data needed to prove branch membership
    // External PCS proves: ∃ br: values[b * branch_sz + br] == 0

    // Branch proofs: proofs[b * branch_sz + br] = IT-MAC (GF(2^64), lower 64 bits) for batch b, branch br
    block* get_branch_proofs() { return proofs; }
    // Branch values: values[b * branch_sz + br] = plaintext RLC (prover only)
    block* get_branch_values() { return values; }
    int get_branch_count() { return branch_sz; }
    int get_batch_count() { return batch_sz; }

    // Set output source for a specific branch
    // source[i] < 0 means pass-through (output[i] = input[i])
    // source[i] >= 0 means computed (output[i] = mul_ou[source[i]])
    void set_output_source(int branch, int *source) {
        for (int i = 0; i < input_count; i++) {
            output_source[branch][i] = source[i];
        }
    }

    // Set all branches to use last input_count mul outputs (for compatibility)
    void set_output_source_from_mul_tail() {
        if (mul_count < input_count) {
            error("set_output_source_from_mul_tail requires mul_count >= input_count\n");
        }
        for (int b = 0; b < branch_sz; b++) {
            for (int i = 0; i < input_count; i++) {
                output_source[b][i] = mul_count - input_count + i;
            }
        }
    }

    // Step 1: Input authentication and AND gates (arrays pre-allocated in constructor)
    void authenticate_and_multiply() {
        long long input_total = (long long)input_count * batch_sz;

        // Authenticate inputs
        for (long long i = 0; i < input_total; i++) {
            inputs[i] = Bit(true, ALICE);
        }

        for (int batch_id = 0; batch_id < batch_sz; batch_id++) {
            long long in_off = (long long)batch_id * input_count;
            long long mul_off = (long long)batch_id * mul_count;

            for (int m = 0; m < mul_count; m++) {
                // Use wire routing tables (zkVM: only first 3 inputs)
                mul_le[mul_off + m] = inputs[in_off + wire_left[m]];
                mul_ri[mul_off + m] = inputs[in_off + wire_right[m]];
                mul_ou[mul_off + m] = mul_le[mul_off + m] & mul_ri[mul_off + m];
            }
        }

        one = Bit(true, PUBLIC);
        delta = get_bool_delta<IO, COTType>(party);
    }

    // Generate random challenge for connection proofs (conn_chi pre-allocated)
    void generate_connection_challenge() {
        long long conn_count = (long long)(batch_sz - 1) * input_count;

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

    // Step 2: Generate challenge and compute proofs for all branches
    // Per-batch precomputation reused across branches:
    //   - Wire check prefix sums: 2*mul_count gfmuls, each branch picks prefix at ng-1
    //   - Output XOR prefix: mul_count XORs, each branch picks prefix at ng-1
    //   - Connection base (branch 0): input_count gfmuls, per-branch adjustment ~32 gfmuls
    void generate_proofs() {
        // Generate connection challenge if needed
        if (batch_sz > 1) {
            generate_connection_challenge();
        }

        long long chi_len = (long long)mul_count * 2 + input_count;

        // Exchange seed for random oracle
        block s_seed;
        if (party == ALICE) {
            ios[0]->recv_data(&s_seed, sizeof(block));
        } else {
            PRG().random_block(&s_seed, 1);
            ios[0]->send_data(&s_seed, sizeof(block));
            ios[0]->flush();
        }
        PRG prg_s(&s_seed);

        // Generate challenge coefficients (64-bit)
        for (long long i = 0; i < chi_len; i++) {
            prg_s.random_block(&chis[i], 1);
            chis[i] = lo64(chis[i]);
        }

        // Output chi sum (constant across all branches and batches)
        block chi_output_sum = makeBlock(0, 0);
        for (int i = 0; i < input_count; i++) {
            chi_output_sum = chi_output_sum ^ chis[mul_count * 2 + i];
        }

        for (int batch_id = 0; batch_id < batch_sz; batch_id++) {
            long long in_off = (long long)batch_id * input_count;
            long long mul_off = (long long)batch_id * mul_count;

            // === Wire check prefix sums (branch-independent) ===
            {
                block running_proof = makeBlock(0, 0);
                block running_value = makeBlock(0, 0);
                for (int m = 0; m < mul_count; m++) {
                    block diff_l = lo64(mul_le[mul_off + m].bit ^ inputs[in_off + wire_left[m]].bit);
                    block tmp;
                    gfmul64(chis[m], diff_l, &tmp);
                    running_proof = running_proof ^ tmp;

                    block diff_r = lo64(mul_ri[mul_off + m].bit ^ inputs[in_off + wire_right[m]].bit);
                    gfmul64(chis[mul_count + m], diff_r, &tmp);
                    running_proof = running_proof ^ tmp;

                    if (party == ALICE) {
                        if (getLSB(diff_l)) running_value = running_value ^ chis[m];
                        if (getLSB(diff_r)) running_value = running_value ^ chis[mul_count + m];
                    }

                    wire_proof_prefix[m] = running_proof;
                    wire_value_prefix[m] = running_value;
                }
            }

            // === Output XOR prefix sums (branch-independent) ===
            {
                block running_aut = makeBlock(0, 0);
                bool running_val = false;
                for (int m = 0; m < mul_count; m++) {
                    running_aut = running_aut ^ lo64(mul_ou[mul_off + m].bit);
                    if (party == ALICE) {
                        running_val = running_val ^ getLSB(mul_ou[mul_off + m].bit);
                    }
                    aut_prefix[m] = running_aut;
                    val_prefix[m] = running_val;
                }
            }

            // === Connection base using branch 0's output_source (branch-independent) ===
            block base_conn_auth = makeBlock(0, 0);
            block base_conn_val = makeBlock(0, 0);
            if (batch_sz > 1 && batch_id < batch_sz - 1) {
                long long in_off_next = (long long)(batch_id + 1) * input_count;
                for (int i = 0; i < input_count; i++) {
                    long long chi_idx = (long long)batch_id * input_count + i;

                    block prev_output;
                    if (output_source[0][i] < 0) {
                        prev_output = inputs[in_off + i].bit;
                    } else {
                        prev_output = mul_ou[mul_off + output_source[0][i]].bit;
                    }

                    block diff = lo64(inputs[in_off_next + i].bit ^ prev_output);

                    block tmp;
                    gfmul64(conn_chi[chi_idx], diff, &tmp);
                    base_conn_auth = base_conn_auth ^ tmp;

                    if (party == ALICE) {
                        if (getLSB(diff)) {
                            base_conn_val = base_conn_val ^ conn_chi[chi_idx];
                        }
                    }
                }
            }

            // === Per-branch proofs ===
            for (int bid = 0; bid < branch_sz; bid++) {
                int proof_idx = batch_id * branch_sz + bid;
                int ng = mul_count_per_branch[bid];

                // Wire check: pick prefix at ng-1
                proofs[proof_idx] = wire_proof_prefix[ng - 1];
                if (party == ALICE) values[proof_idx] = wire_value_prefix[ng - 1];

                // Output check: pick prefix at ng-1
                block aut_tmp = aut_prefix[ng - 1];
                bool val_tmp = val_prefix[ng - 1];

                if (bid % 2 != 0) {
                    aut_tmp = aut_tmp ^ lo64(one.bit);
                    if (party == ALICE) val_tmp = !val_tmp;
                }

                block output_proof;
                gfmul64(chi_output_sum, aut_tmp, &output_proof);
                proofs[proof_idx] = proofs[proof_idx] ^ output_proof;

                if (party == ALICE && val_tmp) {
                    values[proof_idx] = values[proof_idx] ^ chi_output_sum;
                }

                // Connection: base + per-branch adjustment
                if (batch_sz > 1 && batch_id < batch_sz - 1) {
                    proofs[proof_idx] = proofs[proof_idx] ^ base_conn_auth;
                    if (party == ALICE) {
                        values[proof_idx] = values[proof_idx] ^ base_conn_val;
                    }

                    // Adjust only positions where this branch differs from branch 0
                    if (bid != 0) {
                        for (int i = 0; i < input_count; i++) {
                            if (output_source[bid][i] == output_source[0][i]) continue;

                            long long chi_idx = (long long)batch_id * input_count + i;

                            block old_output, new_output;
                            if (output_source[0][i] < 0) {
                                old_output = inputs[in_off + i].bit;
                            } else {
                                old_output = mul_ou[mul_off + output_source[0][i]].bit;
                            }
                            if (output_source[bid][i] < 0) {
                                new_output = inputs[in_off + i].bit;
                            } else {
                                new_output = mul_ou[mul_off + output_source[bid][i]].bit;
                            }

                            block adjustment = lo64(old_output ^ new_output);

                            block tmp;
                            gfmul64(conn_chi[chi_idx], adjustment, &tmp);
                            proofs[proof_idx] = proofs[proof_idx] ^ tmp;

                            if (party == ALICE) {
                                if (getLSB(adjustment)) {
                                    values[proof_idx] = values[proof_idx] ^ conn_chi[chi_idx];
                                }
                            }
                        }
                    }
                }
            }
        }

    }

    // Step 3: Final proof - multiply branch proofs and verify one is zero
    void final_proof() {
        // External PCS mode: skip LPZK, do simple sanity check instead
        // External PCS will prove: ∃ br: values[b * branch_sz + br] == 0
        if (use_external_pcs) {
            if (party == ALICE) {
                // Build randomized active branch map (prover-only)
                setup_active_branch_map();

                // Sum active branch IT-MACs across all batches
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
                // Verifier: doesn't know which branches are active.
                // In real PCS, verifier verifies the polynomial commitment.
                // Here we just check the prover claims value sum is zero.
                block recv_mac, recv_val;
                ios[0]->recv_data(&recv_mac, sizeof(block));
                ios[0]->recv_data(&recv_val, sizeof(block));

                block zero = makeBlock(0, 0);
                if (!cmpBlock(&recv_val, &zero, 1)) {
                    error("PCS sanity check: value sum is not zero\n");
                }
            }
            return;
        }

        aut = new block[branch_sz * batch_sz];
        val = new block[branch_sz * batch_sz];

        for (int batch_id = 0; batch_id < batch_sz; batch_id++) {
            // Get OPE blocks for this batch
            for (int i = 0; i < branch_sz; i++) {
                zkp_get_ope<IO, COTType>(aut[batch_id * branch_sz + i], val[batch_id * branch_sz + i]);
            }

            int base = batch_id * branch_sz;

            if (party == ALICE) {
                // Commit intermediate wires of final MUL gates
                block inter_mul[branch_sz];
                gfmul(values[base + 0], values[base + 1], &inter_mul[0]);
                for (int i = 1; i < branch_sz - 1; i++) {
                    gfmul(inter_mul[i-1], values[base + i + 1], &inter_mul[i]);
                }
                for (int i = 0; i < branch_sz - 1; i++) {
                    val[base + i] = val[base + i] ^ inter_mul[i];
                    ios[0]->send_data(&val[base + i], sizeof(block));
                    val[base + i] = inter_mul[i];
                }
                ios[0]->flush();

                // LPZK
                uint64_t tmp1, tmp2;
                ios[0]->recv_data(&tmp1, sizeof(uint64_t));
                ios[0]->recv_data(&tmp2, sizeof(uint64_t));
                block kai = makeBlock(tmp1, tmp2);

                block coeff = makeBlock(0, 1);
                block A0 = makeBlock(0, 0);
                block A1 = makeBlock(0, 0);

                // First mul gate
                block tmp, tmpmul;
                gfmul(proofs[base + 0], proofs[base + 1], &tmp);
                gfmul(coeff, tmp, &tmpmul);
                A0 = A0 ^ tmpmul;

                gfmul(values[base + 0], proofs[base + 1], &tmp);
                gfmul(coeff, tmp, &tmpmul);
                A1 = A1 ^ tmpmul;
                gfmul(values[base + 1], proofs[base + 0], &tmp);
                gfmul(coeff, tmp, &tmpmul);
                A1 = A1 ^ tmpmul;
                gfmul(coeff, aut[base + 0], &tmpmul);
                A1 = A1 ^ tmpmul;

                gfmul(kai, coeff, &tmpmul);
                coeff = tmpmul;

                for (int i = 1; i < branch_sz - 1; i++) {
                    gfmul(aut[base + i - 1], proofs[base + i + 1], &tmp);
                    gfmul(coeff, tmp, &tmpmul);
                    A0 = A0 ^ tmpmul;

                    gfmul(val[base + i - 1], proofs[base + i + 1], &tmp);
                    gfmul(coeff, tmp, &tmpmul);
                    A1 = A1 ^ tmpmul;
                    gfmul(aut[base + i - 1], values[base + i + 1], &tmp);
                    gfmul(coeff, tmp, &tmpmul);
                    A1 = A1 ^ tmpmul;
                    gfmul(coeff, aut[base + i], &tmpmul);
                    A1 = A1 ^ tmpmul;

                    gfmul(kai, coeff, &tmpmul);
                    coeff = tmpmul;
                }

                // Add OTP for LPZK
                A0 = A0 ^ aut[base + branch_sz - 1];
                A1 = A1 ^ val[base + branch_sz - 1];
                ios[0]->send_data(&A0, sizeof(block));
                ios[0]->send_data(&A1, sizeof(block));
                ios[0]->send_data(&aut[base + branch_sz - 2], sizeof(block));
                ios[0]->flush();

            } else {
                // Verifier
                for (int i = 0; i < branch_sz - 1; i++) {
                    block tmp, tmpmul;
                    ios[0]->recv_data(&tmp, sizeof(block));
                    gfmul(delta, tmp, &tmpmul);
                    aut[base + i] = aut[base + i] ^ tmpmul;
                }

                // LPZK
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

                // First mul gate
                block tmp, tmpmul;
                gfmul(proofs[base + 0], proofs[base + 1], &tmp);
                gfmul(coeff, tmp, &tmpmul);
                accum = accum ^ tmpmul;
                gfmul(aut[base + 0], delta, &tmp);
                gfmul(coeff, tmp, &tmpmul);
                accum = accum ^ tmpmul;
                gfmul(kai, coeff, &tmpmul);
                coeff = tmpmul;

                for (int i = 1; i < branch_sz - 1; i++) {
                    gfmul(aut[base + i - 1], proofs[base + i + 1], &tmp);
                    gfmul(coeff, tmp, &tmpmul);
                    accum = accum ^ tmpmul;
                    gfmul(aut[base + i], delta, &tmp);
                    gfmul(coeff, tmp, &tmpmul);
                    accum = accum ^ tmpmul;
                    gfmul(kai, coeff, &tmpmul);
                    coeff = tmpmul;
                }
                accum = accum ^ aut[base + branch_sz - 1];

                block A0, A1;
                ios[0]->recv_data(&A0, sizeof(block));
                ios[0]->recv_data(&A1, sizeof(block));
                gfmul(A1, delta, &tmp);
                tmp = tmp ^ A0;

                if (!cmpBlock(&tmp, &accum, 1)) {
                    error("LPZK check failed\n");
                }

                block final_check;
                ios[0]->recv_data(&final_check, sizeof(block));
                if (!cmpBlock(&final_check, &aut[base + branch_sz - 2], 1)) {
                    error("Final 0 check failed\n");
                }
            }
        }
    }
};

} // namespace emp

#endif // EMP_ZK_BOOL_BATCHED_DISJUNCTION_H__
