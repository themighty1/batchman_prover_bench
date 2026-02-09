#ifndef EMP_ZK_BATCHED_DISJUNCTION_H__
#define EMP_ZK_BATCHED_DISJUNCTION_H__

#include "emp-zk/emp-zk-arith/emp-zk-arith.h"
#include <algorithm>

namespace emp {

/**
 * BatchedDisjunction - Batched disjunction protocol for arithmetic circuits
 *
 * zkVM Emulation:
 *   This benchmark emulates zkVM execution where each step has:
 *   - 3 computed outputs: instruction, program counter (PC), modified register
 *   - Remaining outputs are pass-through (unchanged state/registers)
 *
 *   The multiplication gates only operate on the first 3 inputs, and only
 *   the first 3 outputs come from mul gates. All other inputs pass through
 *   directly to outputs, representing unchanged VM state between steps.
 *
 * Parameters:
 *   - input_count: Number of input wires (also output count for connected reps)
 *   - mul_count: Number of multiplication gates
 *   - branch_sz: Number of branches in disjunction
 *   - batch_sz: Number of batch instances
 *
 * Output source mapping (per-branch):
 *   - output_source[branch][i] < 0: output[i] = input[i] (pass-through)
 *   - output_source[branch][i] >= 0: output[i] = mul_ou[output_source[branch][i]]
 */

// Number of inputs/outputs that are computed (go through mul gates)
// Represents zkVM: instruction, program counter (PC), modified register
static const int ZKVM_COMPUTED_COUNT = 3;
template<typename IO, typename VoleType = VoleTriple<IO>>
class BatchedDisjunction {
public:
    int party;
    int threads;
    int input_count;
    int mul_count;
    int branch_sz;
    int batch_sz;

    long long w_length;      // Witness length: inputs + 3*muls + 1
    long long check_length;  // Constraint count: 2*muls + 1

    // Wire routing tables
    int *wire_left = nullptr;   // wire_left[m] = which input feeds mul m's left
    int *wire_right = nullptr;  // wire_right[m] = which input feeds mul m's right

    // Per-branch output source mapping
    // output_source[branch][i] < 0: pass-through (output[i] = input[i])
    // output_source[branch][i] >= 0: computed (output[i] = mul_ou[value])
    int **output_source = nullptr;

    // Authenticated values
    IntFp *inputs = nullptr;
    IntFp *mul_le = nullptr;
    IntFp *mul_ri = nullptr;
    IntFp *mul_ou = nullptr;
    IntFp one;
    uint64_t delta;

    // Protocol buffers
    uint64_t *left_r = nullptr;
    uint64_t *left_vec_a = nullptr;
    IntFp *left_v = nullptr;
    uint64_t *right_s = nullptr;
    uint64_t *mac = nullptr;
    IntFp *MAC = nullptr;

    // Connection challenge (shared randomness for connection proofs)
    uint64_t *conn_chi = nullptr;

    // External PCS mode: skip product-of-branches check, export data instead
    bool use_external_pcs = false;
    IntFp *exported_conn_proofs = nullptr;  // Connection proofs for external PCS

    // Constructor with ZK setup (for real VOLE)
    BatchedDisjunction(IO **ios, int threads, int party,
                              int input_count, int mul_count,
                              int branch_sz, int batch_sz)
        : party(party), threads(threads),
          input_count(input_count), mul_count(mul_count),
          branch_sz(branch_sz), batch_sz(batch_sz) {

        w_length = input_count + mul_count * 3 + 1;
        check_length = mul_count * 2 + 1;

        setup_wire_routing();
        setup_default_output_source();
        setup_zk_arith<IO, VoleType>(ios, threads, party);
        owns_zk_exec = true;
    }

    // Constructor without ZK setup (for MockVole)
    BatchedDisjunction(int party, int input_count, int mul_count,
                              int branch_sz, int batch_sz)
        : party(party), threads(1),
          input_count(input_count), mul_count(mul_count),
          branch_sz(branch_sz), batch_sz(batch_sz) {

        w_length = input_count + mul_count * 3 + 1;
        check_length = mul_count * 2 + 1;

        setup_wire_routing();
        setup_default_output_source();
        owns_zk_exec = false;
    }

    ~BatchedDisjunction() {
        delete[] wire_left;
        delete[] wire_right;
        if (output_source) {
            for (int b = 0; b < branch_sz; b++) {
                delete[] output_source[b];
            }
            delete[] output_source;
        }
        delete[] inputs;
        delete[] mul_le;
        delete[] mul_ri;
        delete[] mul_ou;
        delete[] left_r;
        delete[] left_vec_a;
        delete[] left_v;
        delete[] right_s;
        delete[] mac;
        delete[] MAC;
        delete[] conn_chi;
        delete[] exported_conn_proofs;
        if (owns_zk_exec) {
            finalize_zk_arith<IO>();
        }
    }

private:
    bool owns_zk_exec = false;

    // Setup wire routing for zkVM emulation
    // Only the first ZKVM_COMPUTED_COUNT inputs are used in multiplication gates
    // This represents: instruction, PC, modified register
    void setup_wire_routing() {
        wire_left = new int[mul_count];
        wire_right = new int[mul_count];

        // Only use first ZKVM_COMPUTED_COUNT inputs in mul gates
        int computed = std::min(ZKVM_COMPUTED_COUNT, input_count);
        for (int m = 0; m < mul_count; m++) {
            wire_left[m] = m % computed;
            wire_right[m] = (m + 1) % computed;
        }
    }

    // Setup default output source for zkVM emulation:
    // - First ZKVM_COMPUTED_COUNT outputs come from mul gates (instruction, PC, register)
    // - Remaining outputs are pass-through (unchanged VM state)
    void setup_default_output_source() {
        output_source = new int*[branch_sz];
        int computed = std::min(ZKVM_COMPUTED_COUNT, input_count);

        for (int b = 0; b < branch_sz; b++) {
            output_source[b] = new int[input_count];
            for (int i = 0; i < input_count; i++) {
                if (i < computed && i < mul_count) {
                    // Computed output: comes from mul gate i
                    output_source[b][i] = i;
                } else {
                    // Pass-through: output = input (unchanged state)
                    output_source[b][i] = -1;
                }
            }
        }
    }

public:

    // Set output source for a specific branch
    // source[i] < 0: pass-through (output[i] = input[i])
    // source[i] >= 0: computed (output[i] = mul_ou[source[i]])
    void set_output_source(int branch, int *source) {
        for (int i = 0; i < input_count; i++) {
            output_source[branch][i] = source[i];
        }
    }

    // Set output source for all branches (same mapping)
    void set_output_source_all(int *source) {
        for (int b = 0; b < branch_sz; b++) {
            set_output_source(b, source);
        }
    }

    // Get ostriple for VOLE stats access
    FpOSTriple<IO, VoleType>* get_ostriple() {
        if (party == ALICE) {
            return ((ZKFpExecPrv<IO, VoleType>*)(ZKFpExec::zk_exec))->ostriple;
        } else {
            return ((ZKFpExecVer<IO, VoleType>*)(ZKFpExec::zk_exec))->ostriple;
        }
    }

    // Enable external PCS mode (call before prove_mac_in_branches)
    void set_external_pcs_mode(bool enabled) {
        use_external_pcs = enabled;
    }

    // Getters for external PCS - data needed to prove branch membership
    // External PCS proves: ∃ br: MAC[b] - mac[br] + conn_proofs[br] == 0

    // Branch MACs (80 values) - public constants
    uint64_t* get_branch_macs() { return mac; }
    int get_branch_count() { return branch_sz; }

    // Batch MACs (1000 IT-MACs) - authenticated values
    IntFp* get_batch_macs() { return MAC; }
    int get_batch_count() { return batch_sz; }

    // Connection proofs (80 IT-MACs) - only available after prove_mac_in_branches() in external PCS mode
    IntFp* get_connection_proofs() { return exported_conn_proofs; }

    // Step 1: Input authentication and multiplication
    void authenticate_and_multiply() {
        long long input_batch = (long long)input_count * batch_sz;
        long long mul_batch = (long long)mul_count * batch_sz;

        inputs = new IntFp[input_batch];
        mul_le = new IntFp[mul_batch];
        mul_ri = new IntFp[mul_batch];
        mul_ou = new IntFp[mul_batch];

        for (int rep = 0; rep < batch_sz; rep++) {
            long long in_off = (long long)rep * input_count;
            long long mul_off = (long long)rep * mul_count;

            // Authenticate inputs (all 1s for benchmark)
            for (int i = 0; i < input_count; i++) {
                inputs[in_off + i] = IntFp(1, ALICE);
            }

            // Create multiplication gates
            for (int m = 0; m < mul_count; m++) {
                mul_le[mul_off + m] = IntFp(1, ALICE);
                mul_ri[mul_off + m] = IntFp(1, ALICE);
                mul_ou[mul_off + m] = mul_le[mul_off + m] * mul_ri[mul_off + m];
            }
        }
        ZKFpExec::zk_exec->flush_and_proofs();

        one = IntFp(1, PUBLIC);
        delta = ZKFpExec::zk_exec->get_delta();
    }

    // Generate connection challenge (shared randomness)
    void generate_connection_challenge() {
        // Number of connection constraints per rep-pair: input_count
        // Total: (batch_sz - 1) * input_count
        long long conn_count = (long long)(batch_sz - 1) * input_count;
        conn_chi = new uint64_t[conn_count];

        block conn_seed;
        if (party == ALICE) {
            ZKFpExec::zk_exec->recv_data(&conn_seed, sizeof(block));
        } else {
            PRG().random_block(&conn_seed, 1);
            ZKFpExec::zk_exec->send_data(&conn_seed, sizeof(block));
        }

        PRG prg(&conn_seed);
        for (long long i = 0; i < conn_count; i++) {
            block tmp;
            prg.random_block(&tmp, 1);
            conn_chi[i] = LOW64(tmp) % PR;
        }
    }

    // Compute connection proof for a specific branch
    // Returns the random linear combination of connection constraints
    IntFp compute_branch_connection_proof(int branch) {
        IntFp result(0, PUBLIC);
        long long chi_idx = 0;

        for (int rep = 0; rep < batch_sz - 1; rep++) {
            long long in_off_next = (long long)(rep + 1) * input_count;
            long long in_off_prev = (long long)rep * input_count;
            long long mul_off_prev = (long long)rep * mul_count;

            for (int i = 0; i < input_count; i++) {
                // Get output source for this branch
                IntFp prev_output;
                if (output_source[branch][i] < 0) {
                    // Pass-through: output = input
                    prev_output = inputs[in_off_prev + i];
                } else {
                    // Computed: output = mul_ou[source]
                    prev_output = mul_ou[mul_off_prev + output_source[branch][i]];
                }

                // Constraint: input[next][i] - prev_output = 0
                IntFp diff = inputs[in_off_next + i] + prev_output.negate();

                // Add to random linear combination
                result = result + diff * conn_chi[chi_idx];
                chi_idx++;
            }
        }

        return result;
    }

    // Generate left challenge
    void generate_left_challenge() {
        left_r = new uint64_t[check_length];
        block left_r_seed;

        if (party == ALICE) {
            ZKFpExec::zk_exec->recv_data(&left_r_seed, sizeof(block));
        } else {
            PRG().random_block(&left_r_seed, 1);
            ZKFpExec::zk_exec->send_data(&left_r_seed, sizeof(block));
        }

        PRG prg_left_r(&left_r_seed);
        for (long long i = 0; i < check_length; i++) {
            block tmp;
            prg_left_r.random_block(&tmp, 1);
            left_r[i] = LOW64(tmp) % PR;
        }
    }

    // Compute left vectors for each branch
    void compute_left_vectors() {
        long long w_length_branch = w_length * branch_sz;
        left_vec_a = new uint64_t[w_length_branch];
        memset(left_vec_a, 0, w_length_branch * sizeof(uint64_t));

        for (int bid = 0; bid < branch_sz; bid++) {
            uint64_t *vec = left_vec_a + (long long)bid * w_length;

            // Wire routing: mul_le[m] - inputs[wire_left[m]] = 0
            for (int m = 0; m < mul_count; m++) {
                int src = wire_left[m];
                vec[src] = add_mod(vec[src], left_r[m]);
                vec[input_count + m] = PR - left_r[m];
            }

            // Wire routing: mul_ri[m] - inputs[wire_right[m]] = 0
            for (int m = 0; m < mul_count; m++) {
                int src = wire_right[m];
                vec[src] = add_mod(vec[src], left_r[mul_count + m]);
                vec[input_count + mul_count + m] = PR - left_r[mul_count + m];
            }

            // Output constraint: sum(mul_ou) - expected[bid] = 0
            uint64_t out_coeff = left_r[2 * mul_count];
            for (int m = 0; m < mul_count; m++) {
                vec[input_count + 2 * mul_count + m] = PR - out_coeff;
            }

            uint64_t expected = mult_mod((uint64_t)mul_count, (uint64_t)(bid + 1));
            vec[w_length - 1] = mult_mod(expected, out_coeff);
        }
    }

    // Commit left vectors
    void commit_left_vectors() {
        long long w_length_batch = w_length * batch_sz;
        left_v = new IntFp[w_length_batch];

        for (int b = 0; b < batch_sz; b++) {
            for (long long i = 0; i < w_length; i++) {
                left_v[(long long)b * w_length + i] = IntFp(left_vec_a[i], ALICE);
            }
        }
    }

    // Inner product proof (prover side)
    void inner_product_prove(uint64_t chi) {
        uint64_t coeff = 1;
        uint64_t C0 = 0, C1 = 0;

        for (int b = 0; b < batch_sz; b++) {
            uint64_t tmp0 = 0, tmp1 = 0;
            long long v_off = (long long)b * w_length;
            long long in_off = (long long)b * input_count;
            long long mul_off = (long long)b * mul_count;

            for (int i = 0; i < input_count; i++) {
                tmp0 = add_mod(tmp0, mult_mod(inputs[in_off + i].value, left_v[v_off + i].value));
                tmp1 = add_mod(tmp1, add_mod(
                    mult_mod(HIGH64(inputs[in_off + i].value), left_v[v_off + i].value),
                    mult_mod(inputs[in_off + i].value, HIGH64(left_v[v_off + i].value))));
            }

            for (int m = 0; m < mul_count; m++) {
                long long idx = v_off + input_count + m;
                tmp0 = add_mod(tmp0, mult_mod(mul_le[mul_off + m].value, left_v[idx].value));
                tmp1 = add_mod(tmp1, add_mod(
                    mult_mod(HIGH64(mul_le[mul_off + m].value), left_v[idx].value),
                    mult_mod(mul_le[mul_off + m].value, HIGH64(left_v[idx].value))));
            }

            for (int m = 0; m < mul_count; m++) {
                long long idx = v_off + input_count + mul_count + m;
                tmp0 = add_mod(tmp0, mult_mod(mul_ri[mul_off + m].value, left_v[idx].value));
                tmp1 = add_mod(tmp1, add_mod(
                    mult_mod(HIGH64(mul_ri[mul_off + m].value), left_v[idx].value),
                    mult_mod(mul_ri[mul_off + m].value, HIGH64(left_v[idx].value))));
            }

            for (int m = 0; m < mul_count; m++) {
                long long idx = v_off + input_count + 2 * mul_count + m;
                tmp0 = add_mod(tmp0, mult_mod(mul_ou[mul_off + m].value, left_v[idx].value));
                tmp1 = add_mod(tmp1, add_mod(
                    mult_mod(HIGH64(mul_ou[mul_off + m].value), left_v[idx].value),
                    mult_mod(mul_ou[mul_off + m].value, HIGH64(left_v[idx].value))));
            }

            long long const_idx = v_off + w_length - 1;
            tmp0 = add_mod(tmp0, mult_mod(one.value, left_v[const_idx].value));
            tmp1 = add_mod(tmp1, add_mod(
                mult_mod(HIGH64(one.value), left_v[const_idx].value),
                mult_mod(one.value, HIGH64(left_v[const_idx].value))));

            C0 = add_mod(C0, mult_mod(tmp0, coeff));
            C1 = add_mod(C1, mult_mod(tmp1, coeff));
            coeff = mult_mod(coeff, chi);
        }

        __uint128_t random_mask = ZKFpExec::zk_exec->get_one_role();
        C1 = add_mod(C1, HIGH64(random_mask));
        C1 = PR - C1;
        C0 = add_mod(C0, LOW64(random_mask));
        ZKFpExec::zk_exec->send_data(&C0, sizeof(uint64_t));
        ZKFpExec::zk_exec->send_data(&C1, sizeof(uint64_t));
    }

    // Inner product proof (verifier side)
    bool inner_product_verify(uint64_t chi) {
        uint64_t coeff = 1;
        uint64_t expect_value = 0;

        for (int b = 0; b < batch_sz; b++) {
            uint64_t tmp = 0;
            long long v_off = (long long)b * w_length;
            long long in_off = (long long)b * input_count;
            long long mul_off = (long long)b * mul_count;

            for (int i = 0; i < input_count; i++)
                tmp = add_mod(tmp, mult_mod(inputs[in_off + i].value, left_v[v_off + i].value));
            for (int m = 0; m < mul_count; m++)
                tmp = add_mod(tmp, mult_mod(mul_le[mul_off + m].value, left_v[v_off + input_count + m].value));
            for (int m = 0; m < mul_count; m++)
                tmp = add_mod(tmp, mult_mod(mul_ri[mul_off + m].value, left_v[v_off + input_count + mul_count + m].value));
            for (int m = 0; m < mul_count; m++)
                tmp = add_mod(tmp, mult_mod(mul_ou[mul_off + m].value, left_v[v_off + input_count + 2 * mul_count + m].value));
            tmp = add_mod(tmp, mult_mod(one.value, left_v[v_off + w_length - 1].value));

            expect_value = add_mod(expect_value, mult_mod(tmp, coeff));
            coeff = mult_mod(coeff, chi);
        }

        __uint128_t random_mask = ZKFpExec::zk_exec->get_one_role();
        expect_value = add_mod(expect_value, LOW64(random_mask));

        uint64_t C0, C1;
        ZKFpExec::zk_exec->recv_data(&C0, sizeof(uint64_t));
        ZKFpExec::zk_exec->recv_data(&C1, sizeof(uint64_t));

        uint64_t proof_value = add_mod(C0, mult_mod(C1, delta));
        return proof_value == expect_value;
    }

    // Inner product proof dispatch
    bool inner_product_proof() {
        if (party == ALICE) {
            uint64_t chi;
            ZKFpExec::zk_exec->recv_data(&chi, sizeof(uint64_t));
            inner_product_prove(chi);
            return true;
        } else {
            uint64_t chi;
            PRG().random_data(&chi, sizeof(uint64_t));
            chi = chi % PR;
            ZKFpExec::zk_exec->send_data(&chi, sizeof(uint64_t));
            return inner_product_verify(chi);
        }
    }

    // Generate MAC challenge
    void generate_mac_challenge() {
        right_s = new uint64_t[w_length];
        block right_s_seed;

        if (party == ALICE) {
            ZKFpExec::zk_exec->recv_data(&right_s_seed, sizeof(block));
        } else {
            PRG().random_block(&right_s_seed, 1);
            ZKFpExec::zk_exec->send_data(&right_s_seed, sizeof(block));
        }

        PRG prg_right_s(&right_s_seed);
        for (long long i = 0; i < w_length; i++) {
            block tmp;
            prg_right_s.random_block(&tmp, 1);
            right_s[i] = LOW64(tmp) % PR;
        }
    }

    // Generate MACs
    void generate_macs() {
        mac = new uint64_t[branch_sz];
        MAC = new IntFp[batch_sz];

        for (int bid = 0; bid < branch_sz; bid++) {
            mac[bid] = 0;
            uint64_t *vec = left_vec_a + (long long)bid * w_length;
            for (long long i = 0; i < w_length; i++) {
                mac[bid] = add_mod(mac[bid], mult_mod(right_s[i], vec[i]));
            }
        }

        for (int b = 0; b < batch_sz; b++) {
            MAC[b] = IntFp(0, PUBLIC);
            long long v_off = (long long)b * w_length;
            for (long long i = 0; i < w_length; i++) {
                MAC[b] = MAC[b] + left_v[v_off + i] * right_s[i];
            }
        }
    }

    // Prove MAC matches one branch (includes connection proofs)
    // If use_external_pcs is true, skips product-of-branches check and exports data instead
    void prove_mac_in_branches(uint64_t *res) {
        // Generate connection challenge if we have connections
        if (batch_sz > 1) {
            generate_connection_challenge();
        }

        // Precompute connection proofs once per branch (same for all batches)
        IntFp *conn_proofs = nullptr;
        if (batch_sz > 1) {
            conn_proofs = new IntFp[branch_sz];
            for (int br = 0; br < branch_sz; br++) {
                conn_proofs[br] = compute_branch_connection_proof(br);
            }
        }

        // External PCS mode: export data and skip product-of-branches check
        if (use_external_pcs) {
            // Store connection proofs for external PCS
            if (batch_sz > 1) {
                exported_conn_proofs = conn_proofs;  // Transfer ownership
            }
            // mac[] and MAC[] already available via getters
            // External PCS will prove: ∃ br: MAC[b] - mac[br] + conn_proofs[br] == 0
            return;
        }

        // Standard mode: product-of-branches check
        IntFp *f_mac = new IntFp[batch_sz];

        for (int b = 0; b < batch_sz; b++) {
            f_mac[b] = IntFp(1, PUBLIC);

            for (int br = 0; br < branch_sz; br++) {
                // MAC term: should be 0 for active branch
                IntFp mac_term = MAC[b] + IntFp(mac[br], PUBLIC).negate();

                // Connection term: should be 0 for active branch
                IntFp conn_term(0, PUBLIC);
                if (batch_sz > 1) {
                    conn_term = conn_proofs[br];
                }

                // Combine: both must be 0 for active branch
                // Use additive combination (if either is non-zero, result is non-zero)
                IntFp combined = mac_term + conn_term;

                f_mac[b] = f_mac[b] * combined;
            }
        }

        batch_reveal_check(f_mac, res, batch_sz);
        delete[] f_mac;
        delete[] conn_proofs;
    }
};

} // namespace emp

#endif // EMP_ZK_BATCHED_DISJUNCTION_H__
