// Batchman benchmark with mock VOLE (no OT/LPN overhead)
// Uses encapsulated BatchedDisjunction class with MockVole
#include "emp-zk/emp-zk.h"
#include "emp-zk/emp-vole/mock_vole.h"
#include "emp-zk/emp-zk-arith/batched_disjunction.h"
#include <iostream>

using namespace emp;
using namespace std;

int port, party;
const int threads = 1;

void test_circuit_zk(BoolIO<NetIO> *ios[threads], int party, int matrix_sz, int branch_sz, int batch_sz) {
    uint64_t *res = new uint64_t[batch_sz];
    for (int i = 0; i < batch_sz; i++) res[i] = 0;

    auto start = clock_start();

    // Manually set up ZK with MockVole (bypasses FpPolyProof which isn't templated on VoleType)
    using MockVoleType = MockVole<BoolIO<NetIO>>;
    FpOSTriple<BoolIO<NetIO>, MockVoleType> *ostriple;
    if (party == ALICE) {  // Prover
        ZKFpExec::zk_exec = new ZKFpExecPrv<BoolIO<NetIO>, MockVoleType>(ios, threads);
        ostriple = ((ZKFpExecPrv<BoolIO<NetIO>, MockVoleType>*)(ZKFpExec::zk_exec))->ostriple;
    } else {  // Verifier
        auto exec = new ZKFpExecVer<BoolIO<NetIO>, MockVoleType>(ios, threads);
        ZKFpExec::zk_exec = exec;
        ostriple = exec->ostriple;
        // Override delta to match MockVole's MOCK_DELTA (delta_gen() generated random)
        ostriple->delta = MOCK_DELTA;
        exec->delta = MOCK_DELTA;
    }
    ostriple->vole->reset_counter();

    // Create protocol instance (uses pre-setup ZK framework)
    BatchedDisjunction<BoolIO<NetIO>, MockVoleType> protocol(party, matrix_sz, branch_sz, batch_sz);

    // Step 1: Authenticate and multiply
    protocol.authenticate_and_multiply();

    cout << "ACCEPT Multiplication Proofs!" << endl;
    auto ttt0 = time_from(start);
    cout << ttt0 << " us\t" << party << endl;

    // Generate challenge
    protocol.generate_left_challenge();

    // Step 2: Compute left vectors
    protocol.compute_left_vectors();

    cout << "Calculated Left Vectors!" << endl;
    auto ttt1 = time_from(start)-ttt0;
    cout << ttt1 << " us\t" << party << endl;

    // Commit left compressed a
    protocol.commit_left_vectors();

    cout << "Committed left compressed a!" << endl;
    auto ttt2 = time_from(start)-ttt1-ttt0;
    cout << ttt2 << " us\t" << party << endl;

    // Step 3: Inner product proofs
    if (!protocol.inner_product_proof())
        error("Inner product check fails!");

    cout << "ACCEPT Inner-Product Proofs" << endl;
    auto ttt3 = time_from(start)-ttt2-ttt1-ttt0;
    cout << ttt3 << " us\t" << party << endl;

    // Step 4: MAC generation
    protocol.generate_mac_challenge();
    protocol.generate_macs();

    // Step 5: f([MAC])=0 proofs
    protocol.prove_mac_in_branches(res);

    cout << "ACCEPT Mac Proofs" << endl;
    auto ttt4 = time_from(start)-ttt3-ttt2-ttt1-ttt0;
    cout << ttt4 << " us\t" << party << endl;

    auto timeuse = time_from(start);
    cout << "Total:" << endl;
    cout << matrix_sz << "\t" << timeuse << " us\t" << party << endl;
    cout << "VOLEs consumed: " << ostriple->vole->voles_consumed << "\t" << party << endl;

    delete[] res;
    delete ZKFpExec::zk_exec;
}

int main(int argc, char** argv) {
    if (argc < 7) {
        cout << "usage: " << argv[0] << " PARTY(1/2) PORT IP DIMENSION #BRANCH #BATCH" << endl;
        return -1;
    }

    parse_party_and_port(argv, &party, &port);
    BoolIO<NetIO>* ios[threads];
    for (int i = 0; i < threads; ++i)
        ios[i] = new BoolIO<NetIO>(new NetIO(party == ALICE ? nullptr : argv[3], port+i), party == ALICE);

    cout << endl << "------------ mock VOLE benchmark (encapsulated) ------------" << endl << endl;

    int num = atoi(argv[4]);
    int branch = atoi(argv[5]);
    int batch = atoi(argv[6]);

    test_circuit_zk(ios, party, num, branch, batch);

    for (int i = 0; i < threads; ++i) {
        delete ios[i]->io;
        delete ios[i];
    }
    return 0;
}
