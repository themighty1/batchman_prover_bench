// Arithmetic circuit PROVER benchmark
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

void run_prover(BoolIO<NetIO> *ios[threads], int input_count, int mul_count, int branch_sz, int batch_sz) {
    uint64_t *res = new uint64_t[batch_sz];
    for (int i = 0; i < batch_sz; i++) res[i] = 0;

    // Reset BoolIO counters before benchmark
    ios[0]->counter = 0;
    ios[0]->recv_counter = 0;

    auto start = clock_start();

    // Set up ZK as Prover with MockVole
    using MockVoleType = MockVole<BoolIO<NetIO>>;
    ZKFpExec::zk_exec = new ZKFpExecPrv<BoolIO<NetIO>, MockVoleType>(ios, threads);
    auto ostriple = ((ZKFpExecPrv<BoolIO<NetIO>, MockVoleType>*)(ZKFpExec::zk_exec))->ostriple;
    ostriple->vole->reset_counter();

    // Create protocol instance
    BatchedDisjunction<BoolIO<NetIO>, MockVoleType> protocol(
        ALICE, input_count, mul_count, branch_sz, batch_sz);

    // Run protocol
    protocol.authenticate_and_multiply();

    // Connection proofs are now handled inside prove_mac_in_branches()
    // with per-branch output_source mapping (default: all pass-through)

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

    auto total_us = time_from(start);
    double total_ms = total_us / 1000.0;

    // Get IO stats from BoolIO
    size_t bytes_sent = ios[0]->counter;
    size_t bytes_recv = ios[0]->recv_counter;

    cout << "Total: " << total_ms << " ms" << endl;
    cout << "VOLEs consumed: " << ostriple->vole->voles_consumed << endl;
    cout << "  (VOLEs are pre-generated; not included in runtime)" << endl;
    cout << "Bytes sent: " << bytes_sent << " (" << (bytes_sent / 1024.0) << " KB)" << endl;
    cout << "Bytes recv: " << bytes_recv << " (" << (bytes_recv / 1024.0) << " KB)" << endl;

    delete[] res;
    delete ZKFpExec::zk_exec;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        cout << "usage: " << argv[0] << " PORT INPUT MUL BRANCHES BATCHES" << endl;
        cout << endl;
        cout << "Parameters:" << endl;
        cout << "  INPUT    Number of input wires" << endl;
        cout << "  MUL      Number of multiplication gates" << endl;
        cout << "  BRANCHES Number of branches in disjunction" << endl;
        cout << "  BATCHES  Number of batch instances" << endl;
        return -1;
    }

    port = atoi(argv[1]);
    int input_count = atoi(argv[2]);
    int mul_count = atoi(argv[3]);
    int branches = atoi(argv[4]);
    int batches = atoi(argv[5]);

    cout << "=== Batchman Prover ===" << endl;
    cout << "Inputs: " << input_count << ", Muls: " << mul_count << endl;
    cout << "Branches: " << branches << ", Batches: " << batches << endl;

    BoolIO<NetIO>* ios[threads];
    for (int i = 0; i < threads; ++i)
        ios[i] = new BoolIO<NetIO>(new NetIO(nullptr, port+i, true), true);

    run_prover(ios, input_count, mul_count, branches, batches);

    for (int i = 0; i < threads; ++i) {
        delete ios[i]->io;
        delete ios[i];
    }
    return 0;
}
