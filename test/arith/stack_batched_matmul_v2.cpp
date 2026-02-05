// ENCAPSULATED: Protocol logic properly encapsulated in BatchedDisjunction class
#include "emp-zk/emp-zk.h"
#include "emp-zk/emp-zk-arith/batched_disjunction.h"
#include <iostream>
#if defined(__linux__)
	#include <sys/time.h>
	#include <sys/resource.h>
#elif defined(__APPLE__)
	#include <unistd.h>
	#include <sys/resource.h>
	#include <mach/mach.h>
#endif

using namespace emp;
using namespace std;

int port, party;
const int threads = 1;

void test_circuit_zk(BoolIO<NetIO> *ios[threads], int party, int matrix_sz, int branch_sz, int batch_sz) {
    uint64_t *res = new uint64_t[batch_sz];
    for (int i = 0; i < batch_sz; i++) res[i] = 0;

    auto start = clock_start();

    // Create protocol instance
    BatchedDisjunction<BoolIO<NetIO>> protocol(ios, threads, party, matrix_sz, branch_sz, batch_sz);

    // Step 1: Authenticate and multiply
    protocol.authenticate_and_multiply();

    cout << "ACCEPT Multiplication Proofs!" << endl;
    auto ttt0 = time_from(start);
    cout << ttt0 << " us\t" << party << endl;

    // Generate challenge (allocates internally)
    protocol.generate_left_challenge();

    // Step 2: Compute left vectors (allocates internally)
    protocol.compute_left_vectors();

    cout << "Calculated Left Vectors!" << endl;
    auto ttt1 = time_from(start)-ttt0;
    cout << ttt1 << " us\t" << party << endl;

    // Commit left compressed a (allocates internally)
    protocol.commit_left_vectors();

    cout << "Committed left compressed a!" << endl;
    auto ttt2 = time_from(start)-ttt1-ttt0;
    cout << ttt2 << " us\t" << party << endl;

    // Step 3: Inner product proofs (uses internal buffers)
    if (!protocol.inner_product_proof())
        error("Inner product check fails!");

    cout << "ACCEPT Inner-Product Proofs" << endl;
    auto ttt3 = time_from(start)-ttt2-ttt1-ttt0;
    cout << ttt3 << " us\t" << party << endl;

    // Step 4: MAC generation (allocates internally)
    protocol.generate_mac_challenge();
    protocol.generate_macs();

    // Step 5: f([MAC])=0 proofs (uses internal buffers)
    protocol.prove_mac_in_branches(res);

    cout << "ACCEPT Mac Proofs" << endl;
    auto ttt4 = time_from(start)-ttt3-ttt2-ttt1-ttt0;
    cout << ttt4 << " us\t" << party << endl;

    auto timeuse = time_from(start);
    cout << "Total:" << endl;
    cout << matrix_sz << "\t" << timeuse << " us\t" << party << endl;

    delete[] res;

#if defined(__linux__)
    struct rusage rusage;
    if (!getrusage(RUSAGE_SELF, &rusage))
        cout << "[Linux]Peak resident set size: " << (size_t)rusage.ru_maxrss << endl;
#elif defined(__APPLE__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS)
        cout << "[Mac]Peak resident set size: " << (size_t)info.resident_size_max << endl;
#endif
}

int main(int argc, char** argv) {
    if(argc < 7) {
        cout << "usage: a.out PARTY(1/2) PORT IP DIMENSION #BRANCH #REPETITION" << endl;
        return -1;
    }

    parse_party_and_port(argv, &party, &port);
    BoolIO<NetIO>* ios[threads];
    for(int i = 0; i < threads; ++i)
        ios[i] = new BoolIO<NetIO>(new NetIO(party == ALICE?nullptr:argv[3],port+i), party==ALICE);

    cout << endl << "------------ circuit zero-knowledge proof test (v2 encapsulated) ------------" << endl << endl;

    int num = atoi(argv[4]);
    int branch = atoi(argv[5]);
    int batch = atoi(argv[6]);

    test_circuit_zk(ios, party, num, branch, batch);

    for(int i = 0; i < threads; ++i) {
        delete ios[i]->io;
        delete ios[i];
    }
    return 0;
}
