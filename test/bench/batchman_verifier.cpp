// Batchman benchmark - Verifier
// Uses encapsulated BatchedDisjunction class with MockVole
#include "emp-zk/emp-zk.h"
#include "emp-zk/emp-vole/mock_vole.h"
#include "emp-zk/emp-zk-arith/batched_disjunction.h"
#include <iostream>

using namespace emp;
using namespace std;

int port;
const int threads = 1;

void run_verifier(BoolIO<NetIO> *ios[threads], int matrix_sz, int branch_sz, int batch_sz) {
    uint64_t *res = new uint64_t[batch_sz];
    for (int i = 0; i < batch_sz; i++) res[i] = 0;

    auto start = clock_start();

    // Set up ZK as Verifier with MockVole
    using MockVoleType = MockVole<BoolIO<NetIO>>;
    auto exec = new ZKFpExecVer<BoolIO<NetIO>, MockVoleType>(ios, threads);
    ZKFpExec::zk_exec = exec;
    auto ostriple = exec->ostriple;
    // Override delta to match MockVole's MOCK_DELTA
    ostriple->delta = MOCK_DELTA;
    exec->delta = MOCK_DELTA;
    ostriple->vole->reset_counter();

    // Create protocol instance
    BatchedDisjunction<BoolIO<NetIO>, MockVoleType> protocol(BOB, matrix_sz, branch_sz, batch_sz);

    // Step 1: Authenticate and multiply
    protocol.authenticate_and_multiply();

    cout << "Multiplication Proofs: " << time_from(start) << " us" << endl;
    auto t0 = time_from(start);

    // Step 2: Left challenge and vectors
    protocol.generate_left_challenge();
    protocol.compute_left_vectors();

    cout << "Left Vectors: " << time_from(start) - t0 << " us" << endl;
    auto t1 = time_from(start);

    // Step 3: Commit and inner product proof
    protocol.commit_left_vectors();

    cout << "Commit Left: " << time_from(start) - t1 << " us" << endl;
    auto t2 = time_from(start);

    if (!protocol.inner_product_proof())
        error("Inner product check fails!");

    cout << "Inner Product: " << time_from(start) - t2 << " us" << endl;
    auto t3 = time_from(start);

    // Step 4-5: MAC generation and proof
    protocol.generate_mac_challenge();
    protocol.generate_macs();
    protocol.prove_mac_in_branches(res);

    cout << "MAC Proofs: " << time_from(start) - t3 << " us" << endl;

    auto total = time_from(start);
    cout << endl << "=== VERIFIER RESULTS ===" << endl;
    cout << "Total: " << total << " us" << endl;
    cout << "Protocol only: " << total - t0 << " us" << endl;
    cout << "VOLEs consumed: " << ostriple->vole->voles_consumed << endl;

    delete[] res;
    delete ZKFpExec::zk_exec;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        cout << "usage: " << argv[0] << " PORT IP DIMENSION BRANCHES BATCHES" << endl;
        return -1;
    }

    port = atoi(argv[1]);
    const char* ip = argv[2];
    int dim = atoi(argv[3]);
    int branches = atoi(argv[4]);
    int batches = atoi(argv[5]);

    cout << "=== Verifier ===" << endl;
    cout << "Dim: " << dim << " (" << dim*dim*dim << " muls/branch)" << endl;
    cout << "Branches: " << branches << ", Batches: " << batches << endl << endl;

    BoolIO<NetIO>* ios[threads];
    for (int i = 0; i < threads; ++i)
        ios[i] = new BoolIO<NetIO>(new NetIO(ip, port+i), false);

    run_verifier(ios, dim, branches, batches);

    for (int i = 0; i < threads; ++i) {
        delete ios[i]->io;
        delete ios[i];
    }
    return 0;
}
