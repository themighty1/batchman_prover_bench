// Arithmetic circuit VERIFIER benchmark
//
// zkVM Emulation:
//   This benchmark emulates zkVM execution where each step has:
//   - 3 computed outputs from mul gates:
//     * instruction  (what operation was executed)
//     * PC           (program counter - next instruction address)
//     * register     (the modified register value)
//   - Remaining outputs are pass-through (unchanged VM state/registers)
//
//   Connected repetitions prove: output[step] == input[step+1]
//   This chains VM steps together, proving execution trace validity.
#include "emp-zk/emp-zk.h"
#include "emp-zk/emp-vole/mock_vole.h"
#include "emp-zk/emp-zk-arith/batched_disjunction.h"
#include <iostream>

using namespace emp;
using namespace std;

int port;
const int threads = 1;

void run_verifier(BoolIO<NetIO> *ios[threads], int input_count, int mul_count, int branch_sz, int batch_sz) {
    uint64_t *res = new uint64_t[batch_sz];
    for (int i = 0; i < batch_sz; i++) res[i] = 0;

    // Set up ZK as Verifier with MockVole
    using MockVoleType = MockVole<BoolIO<NetIO>>;
    auto exec = new ZKFpExecVer<BoolIO<NetIO>, MockVoleType>(ios, threads);
    ZKFpExec::zk_exec = exec;
    auto ostriple = exec->ostriple;
    // Override delta to match MockVole's MOCK_DELTA
    ostriple->delta = MOCK_DELTA;
    exec->delta = MOCK_DELTA;

    // Create protocol instance
    BatchedDisjunction<BoolIO<NetIO>, MockVoleType> protocol(
        BOB, input_count, mul_count, branch_sz, batch_sz);

    // Run protocol
    protocol.authenticate_and_multiply();

    // Connection proofs are now handled inside prove_mac_in_branches()

    protocol.generate_left_challenge();
    protocol.compute_left_vectors();
    protocol.commit_left_vectors();
    if (!protocol.inner_product_proof())
        error("Inner product check fails!");
    protocol.generate_mac_challenge();
    protocol.generate_macs();

    // Enable external PCS mode: skip product-of-branches check
    // External PCS will prove: ∃ br: MAC[b] - mac[br] + conn_proofs[br] == 0
    protocol.set_external_pcs_mode(true);
    protocol.prove_mac_in_branches(res);

    delete[] res;
    delete ZKFpExec::zk_exec;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        cout << "usage: " << argv[0] << " PORT INPUT MUL BRANCHES BATCHES" << endl;
        return -1;
    }

    port = atoi(argv[1]);
    int input_count = atoi(argv[2]);
    int mul_count = atoi(argv[3]);
    int branches = atoi(argv[4]);
    int batches = atoi(argv[5]);

    BoolIO<NetIO>* ios[threads];
    for (int i = 0; i < threads; ++i)
        ios[i] = new BoolIO<NetIO>(new NetIO("127.0.0.1", port+i, true), false);

    run_verifier(ios, input_count, mul_count, branches, batches);

    for (int i = 0; i < threads; ++i) {
        delete ios[i]->io;
        delete ios[i];
    }
    return 0;
}
