#ifndef EMP_ZK_BATCHED_DISJUNCTION_H__
#define EMP_ZK_BATCHED_DISJUNCTION_H__

#include "emp-zk/emp-zk-arith/emp-zk-arith.h"

namespace emp {

// Encapsulated batched disjunction protocol
template<typename IO, typename VoleType = VoleTriple<IO>>
class BatchedDisjunction {
public:
    int party;
    int threads;
    int matrix_sz;
    int branch_sz;
    int batch_sz;

    long long test_n;
    long long mul_sz;
    long long w_length;
    long long check_length;

    // Authenticated values
    IntFp *mat_a = nullptr;
    IntFp *mat_b = nullptr;
    IntFp *mul_le = nullptr;
    IntFp *mul_ri = nullptr;
    IntFp *mul_ou = nullptr;
    IntFp one;
    uint64_t delta;

    // Protocol buffers (managed internally)
    uint64_t *left_r = nullptr;
    uint64_t *left_vec_a = nullptr;
    IntFp *left_v = nullptr;
    uint64_t *right_s = nullptr;
    uint64_t *mac = nullptr;
    IntFp *MAC = nullptr;

    // Constructor that sets up ZK framework (use for real VOLE)
    BatchedDisjunction(IO **ios, int threads, int party, int matrix_sz, int branch_sz, int batch_sz)
        : party(party), threads(threads), matrix_sz(matrix_sz), branch_sz(branch_sz), batch_sz(batch_sz) {

        test_n = matrix_sz * matrix_sz;
        mul_sz = matrix_sz * matrix_sz * matrix_sz;
        w_length = test_n * 2 + mul_sz * 3 + 1;
        check_length = mul_sz * 2 + test_n;

        setup_zk_arith<IO, VoleType>(ios, threads, party);
        owns_zk_exec = true;
    }

    // Constructor that uses already-setup ZK framework (use for MockVole)
    BatchedDisjunction(int party, int matrix_sz, int branch_sz, int batch_sz)
        : party(party), threads(1), matrix_sz(matrix_sz), branch_sz(branch_sz), batch_sz(batch_sz) {

        test_n = matrix_sz * matrix_sz;
        mul_sz = matrix_sz * matrix_sz * matrix_sz;
        w_length = test_n * 2 + mul_sz * 3 + 1;
        check_length = mul_sz * 2 + test_n;

        owns_zk_exec = false;
    }

    ~BatchedDisjunction() {
        delete[] mat_a;
        delete[] mat_b;
        delete[] mul_le;
        delete[] mul_ri;
        delete[] mul_ou;
        delete[] left_r;
        delete[] left_vec_a;
        delete[] left_v;
        delete[] right_s;
        delete[] mac;
        delete[] MAC;
        if (owns_zk_exec) {
            finalize_zk_arith<IO>();
        }
    }

private:
    bool owns_zk_exec = false;

public:

    // Get ostriple for VOLE stats access
    FpOSTriple<IO, VoleType>* get_ostriple() {
        if (party == ALICE) {  // Prover
            return ((ZKFpExecPrv<IO, VoleType>*)(ZKFpExec::zk_exec))->ostriple;
        } else {  // Verifier
            return ((ZKFpExecVer<IO, VoleType>*)(ZKFpExec::zk_exec))->ostriple;
        }
    }

    // Step 1: Input authentication and multiplication
    void authenticate_and_multiply() {
        long long test_n_batch_sz = test_n * batch_sz;
        long long mul_sz_batch_sz = mul_sz * batch_sz;

        mat_a = new IntFp[test_n_batch_sz];
        mat_b = new IntFp[test_n_batch_sz];
        for (int i = 0; i < test_n_batch_sz; i++) {
            mat_a[i] = IntFp(1, ALICE);
            mat_b[i] = IntFp(1, ALICE);
        }

        mul_le = new IntFp[mul_sz_batch_sz];
        mul_ri = new IntFp[mul_sz_batch_sz];
        mul_ou = new IntFp[mul_sz_batch_sz];
        for (int i = 0; i < mul_sz_batch_sz; i++) {
            mul_le[i] = IntFp(1, ALICE);
            mul_ri[i] = IntFp(1, ALICE);
            mul_ou[i] = mul_le[i] * mul_ri[i];
        }
        ZKFpExec::zk_exec->flush_and_proofs();

        one = IntFp(1, PUBLIC);
        delta = ZKFpExec::zk_exec->get_delta();
    }

    // Inner product proof (Prover side)
    void inner_product_prove(uint64_t chi) {
        uint64_t coeff = 1;
        uint64_t C0 = 0, C1 = 0;

        for (int bid = 0; bid < batch_sz; bid++) {
            uint64_t tmp0 = 0, tmp1 = 0;
            for (int i = 0; i < test_n; i++) {
                tmp0 = add_mod(tmp0, mult_mod(mat_a[i].value, left_v[bid*w_length + i].value));
                tmp1 = add_mod(tmp1, add_mod(mult_mod(HIGH64(mat_a[i].value), left_v[bid*w_length + i].value), mult_mod(mat_a[i].value, HIGH64(left_v[bid*w_length + i].value))));
            }
            for (int i = 0; i < test_n; i++) {
                tmp0 = add_mod(tmp0, mult_mod(mat_b[i].value, left_v[bid*w_length + test_n + i].value));
                tmp1 = add_mod(tmp1, add_mod(mult_mod(HIGH64(mat_b[i].value), left_v[bid*w_length + test_n + i].value), mult_mod(mat_b[i].value, HIGH64(left_v[bid*w_length + test_n + i].value))));
            }
            for (int i = 0; i < mul_sz; i++) {
                tmp0 = add_mod(tmp0, mult_mod(mul_le[i].value, left_v[bid*w_length + 2*test_n + i].value));
                tmp1 = add_mod(tmp1, add_mod(mult_mod(HIGH64(mul_le[i].value), left_v[bid*w_length + 2*test_n + i].value), mult_mod(mul_le[i].value, HIGH64(left_v[bid*w_length + 2*test_n + i].value))));
            }
            for (int i = 0; i < mul_sz; i++) {
                tmp0 = add_mod(tmp0, mult_mod(mul_ri[i].value, left_v[bid*w_length + 2*test_n + mul_sz + i].value));
                tmp1 = add_mod(tmp1, add_mod(mult_mod(HIGH64(mul_ri[i].value), left_v[bid*w_length + 2*test_n + mul_sz + i].value), mult_mod(mul_ri[i].value, HIGH64(left_v[bid*w_length + 2*test_n + mul_sz + i].value))));
            }
            for (int i = 0; i < mul_sz; i++) {
                tmp0 = add_mod(tmp0, mult_mod(mul_ou[i].value, left_v[bid*w_length + 2*test_n + 2*mul_sz + i].value));
                tmp1 = add_mod(tmp1, add_mod(mult_mod(HIGH64(mul_ou[i].value), left_v[bid*w_length + 2*test_n + 2*mul_sz + i].value), mult_mod(mul_ou[i].value, HIGH64(left_v[bid*w_length + 2*test_n + 2*mul_sz + i].value))));
            }
            tmp0 = add_mod(tmp0, mult_mod(one.value, left_v[bid*w_length + 2*test_n + 3*mul_sz].value));
            tmp1 = add_mod(tmp1, add_mod(mult_mod(HIGH64(one.value), left_v[bid*w_length + 2*test_n + 3*mul_sz].value), mult_mod(one.value, HIGH64(left_v[bid*w_length + 2*test_n + 3*mul_sz].value))));
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

    // Inner product proof (Verifier side)
    bool inner_product_verify(uint64_t chi) {
        uint64_t coeff = 1;
        uint64_t expect_value = 0;

        for (int bid = 0; bid < batch_sz; bid++) {
            uint64_t tmp = 0;
            for (int i = 0; i < test_n; i++) tmp = add_mod(tmp, mult_mod(mat_a[i].value, left_v[bid*w_length + i].value));
            for (int i = 0; i < test_n; i++) tmp = add_mod(tmp, mult_mod(mat_b[i].value, left_v[bid*w_length + test_n + i].value));
            for (int i = 0; i < mul_sz; i++) tmp = add_mod(tmp, mult_mod(mul_le[i].value, left_v[bid*w_length + 2*test_n + i].value));
            for (int i = 0; i < mul_sz; i++) tmp = add_mod(tmp, mult_mod(mul_ri[i].value, left_v[bid*w_length + 2*test_n + mul_sz + i].value));
            for (int i = 0; i < mul_sz; i++) tmp = add_mod(tmp, mult_mod(mul_ou[i].value, left_v[bid*w_length + 2*test_n + 2*mul_sz + i].value));
            tmp = add_mod(tmp, mult_mod(one.value, left_v[bid*w_length + 2*test_n + 3*mul_sz].value));
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

    // Generate left challenge vector from exchanged seed (Verifier sends to Prover)
    void generate_left_challenge() {
        left_r = new uint64_t[check_length];
        block left_r_seed;
        if (party == ALICE) {  // Prover receives
            ZKFpExec::zk_exec->recv_data(&left_r_seed, sizeof(block));
        } else {  // Verifier generates and sends
            PRG().random_block(&left_r_seed, 1);
            ZKFpExec::zk_exec->send_data(&left_r_seed, sizeof(block));
        }
        PRG prg_left_r(&left_r_seed);
        for (int i = 0; i < check_length; i++) {
            block tmp;
            prg_left_r.random_block(&tmp, 1);
            left_r[i] = LOW64(tmp) % PR;
        }
    }

    // Step 2: Compute left vectors for each branch
    void compute_left_vectors() {
        long long w_length_branch_sz = w_length * branch_sz;
        left_vec_a = new uint64_t[w_length_branch_sz];

        for (int bid = 0; bid < branch_sz; bid++) {
            uint64_t head = w_length * bid;
            for (int i = 0; i < matrix_sz; i++)
                for (int j = 0; j < matrix_sz; j++) {
                    left_vec_a[head + i*matrix_sz + j] = 0;
                    for (int k = 0; k < matrix_sz; k++)
                        left_vec_a[head + i*matrix_sz + j] = add_mod(left_vec_a[head + i*matrix_sz + j], left_r[i*test_n + j*matrix_sz + k]);
                }
            head = head + test_n;
            for (int i = 0; i < matrix_sz; i++)
                for (int j = 0; j < matrix_sz; j++) {
                    left_vec_a[head + i*matrix_sz + j] = 0;
                    for (int k = 0; k < matrix_sz; k++)
                        left_vec_a[head + i*matrix_sz + j] = add_mod(left_vec_a[head + i*matrix_sz + j], left_r[mul_sz + k*test_n + i*matrix_sz + j]);
                }
            head = head + test_n;
            for (int i = 0; i < mul_sz; i++) left_vec_a[head + i] = PR - left_r[i];
            head = head + mul_sz;
            for (int i = 0; i < mul_sz; i++) left_vec_a[head + i] = PR - left_r[mul_sz + i];
            head = head + mul_sz;
            for (int i = 0; i < matrix_sz; i++)
                for (int j = 0; j < matrix_sz; j++)
                    for (int k = 0; k < matrix_sz; k++)
                        left_vec_a[head + i*test_n + j*matrix_sz + k] = PR - left_r[mul_sz*2 + i*matrix_sz + k];
            head = head + mul_sz;
            left_vec_a[head] = 0;
            for (int i = 0; i < test_n; i++)
                left_vec_a[head] = add_mod(left_vec_a[head], mult_mod((bid+1)*matrix_sz, left_r[mul_sz*2 + i]));
        }
    }

    // Commit left vectors as authenticated values
    void commit_left_vectors() {
        long long w_length_batch_sz = w_length * batch_sz;
        left_v = new IntFp[w_length_batch_sz];
        for (int bid = 0; bid < batch_sz; bid++)
            for (int i = 0; i < w_length; i++)
                left_v[bid*w_length + i] = IntFp(left_vec_a[i], ALICE);
    }

    // Inner product proof with challenge exchange (Verifier sends chi to Prover)
    bool inner_product_proof() {
        if (party == ALICE) {  // Prover
            uint64_t chi;
            ZKFpExec::zk_exec->recv_data(&chi, sizeof(uint64_t));
            inner_product_prove(chi);
            return true;
        } else {  // Verifier
            uint64_t chi;
            PRG().random_data(&chi, sizeof(uint64_t));
            chi = chi % PR;
            ZKFpExec::zk_exec->send_data(&chi, sizeof(uint64_t));
            return inner_product_verify(chi);
        }
    }

    // Generate MAC challenge vector from exchanged seed (Verifier sends to Prover)
    void generate_mac_challenge() {
        right_s = new uint64_t[w_length];
        block right_s_seed;
        if (party == ALICE) {  // Prover receives
            ZKFpExec::zk_exec->recv_data(&right_s_seed, sizeof(block));
        } else {  // Verifier generates and sends
            PRG().random_block(&right_s_seed, 1);
            ZKFpExec::zk_exec->send_data(&right_s_seed, sizeof(block));
        }
        PRG prg_right_s(&right_s_seed);
        for (int i = 0; i < w_length; i++) {
            block tmp;
            prg_right_s.random_block(&tmp, 1);
            right_s[i] = LOW64(tmp) % PR;
        }
    }

    // Step 4: Generate MACs for branches
    void generate_macs() {
        mac = new uint64_t[branch_sz];
        MAC = new IntFp[batch_sz];

        for (int bid = 0; bid < branch_sz; bid++) {
            mac[bid] = 0;
            for (int i = 0; i < w_length; i++)
                mac[bid] = add_mod(mac[bid], mult_mod(right_s[i], left_vec_a[bid*w_length + i]));
        }

        for (int bid = 0; bid < batch_sz; bid++) {
            MAC[bid] = IntFp(0, PUBLIC);
            for (int i = 0; i < w_length; i++)
                MAC[bid] = MAC[bid] + left_v[bid*w_length + i] * right_s[i];
        }
    }

    // Step 5: f([MAC])=0 proof - proves MAC matches one of the branches
    void prove_mac_in_branches(uint64_t *res) {
        IntFp *f_mac = new IntFp[batch_sz];
        for (int bid = 0; bid < batch_sz; bid++) {
            f_mac[bid] = IntFp(1, PUBLIC);
            for (int br = 0; br < branch_sz; br++) {
                IntFp term = MAC[bid] + IntFp(mac[br], PUBLIC).negate();
                f_mac[bid] = f_mac[bid] * term;
            }
        }
        batch_reveal_check(f_mac, res, batch_sz);
        delete[] f_mac;
    }
};

} // namespace emp

#endif // EMP_ZK_BATCHED_DISJUNCTION_H__
