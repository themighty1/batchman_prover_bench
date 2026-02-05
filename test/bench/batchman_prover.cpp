// Batchman PROVER benchmark
// Uses encapsulated BatchedDisjunction class with MockVole
#include "emp-zk/emp-zk.h"
#include "emp-zk/emp-vole/mock_vole.h"
#include "emp-zk/emp-zk-arith/batched_disjunction.h"
#include <iostream>

using namespace emp;
using namespace std;

int port;
const int threads = 1;

void run_prover(BoolIO<NetIO> *ios[threads], int matrix_sz, int branch_sz, int batch_sz) {
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
    BatchedDisjunction<BoolIO<NetIO>, MockVoleType> protocol(ALICE, matrix_sz, branch_sz, batch_sz);

    // Run protocol
    protocol.authenticate_and_multiply();
    protocol.generate_left_challenge();
    protocol.compute_left_vectors();
    protocol.commit_left_vectors();
    if (!protocol.inner_product_proof())
        error("Inner product check fails!");
    protocol.generate_mac_challenge();
    protocol.generate_macs();
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
    if (argc < 5) {
        cout << "usage: " << argv[0] << " PORT DIMENSION BRANCHES BATCHES" << endl;
        return -1;
    }

    port = atoi(argv[1]);
    int dim = atoi(argv[2]);
    int branches = atoi(argv[3]);
    int batches = atoi(argv[4]);

    cout << "=== Batchman Prover ===" << endl;
    cout << "Dim: " << dim << " (" << dim*dim*dim << " muls/branch)" << endl;
    cout << "Branches: " << branches << ", Batches: " << batches << endl;

    BoolIO<NetIO>* ios[threads];
    for (int i = 0; i < threads; ++i)
        ios[i] = new BoolIO<NetIO>(new NetIO(nullptr, port+i, true), true);

    run_prover(ios, dim, branches, batches);

    for (int i = 0; i < threads; ++i) {
        delete ios[i]->io;
        delete ios[i];
    }
    return 0;
}
