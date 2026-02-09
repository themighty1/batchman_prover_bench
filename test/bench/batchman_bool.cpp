// Boolean circuit benchmark with MockCOT
//
// zkVM Emulation:
//   This benchmark emulates zkVM execution where each step has:
//   - 96 computed output bits from AND gates:
//     * 32 bits: instruction (what operation was executed)
//     * 32 bits: PC (program counter - next instruction address)
//     * 32 bits: register (the modified register value)
//   - Remaining outputs are pass-through (unchanged VM state/registers)
//
//   Connected repetitions prove: output[step] == input[step+1]
//   This chains VM steps together, proving execution trace validity.
#include "emp-zk/emp-zk.h"
#include "emp-zk/emp-vole/mock_cot.h"
#include "emp-zk/emp-zk-bool/batched_disjunction.h"
#include <iostream>

using namespace emp;
using namespace std;

int port, party;
const int threads = 1;

void test_circuit_zk(BoolIO<NetIO> *ios[threads], int party, int input_count, int mul_count, int branch_sz, int batch_sz) {
    // Set up ZK with MockCOT
    using MockCOTType = MockCOT<BoolIO<NetIO>>;
    OSTriple<BoolIO<NetIO>, MockCOTType> *ostriple;
    if (party == ALICE) {  // Prover
        ZKBoolCircExecPrv<BoolIO<NetIO>, MockCOTType> *t = new ZKBoolCircExecPrv<BoolIO<NetIO>, MockCOTType>();
        CircuitExecution::circ_exec = t;
        ProtocolExecution::prot_exec = new ZKProver<BoolIO<NetIO>, MockCOTType>(ios, threads, t, nullptr);
        ostriple = ((ZKProver<BoolIO<NetIO>, MockCOTType>*)(ProtocolExecution::prot_exec))->ostriple;
    } else {  // Verifier
        ZKBoolCircExecVer<BoolIO<NetIO>, MockCOTType> *t = new ZKBoolCircExecVer<BoolIO<NetIO>, MockCOTType>();
        CircuitExecution::circ_exec = t;
        ProtocolExecution::prot_exec = new ZKVerifier<BoolIO<NetIO>, MockCOTType>(ios, threads, t, nullptr);
        ostriple = ((ZKVerifier<BoolIO<NetIO>, MockCOTType>*)(ProtocolExecution::prot_exec))->ostriple;
        // Override delta to match MockCOT's delta
        ostriple->delta = MOCK_COT_DELTA;
        ostriple->choice[1] = MOCK_COT_DELTA;
    }
    ostriple->ferret->reset_counter();

    // Create protocol instance
    BoolBatchedDisjunction<BoolIO<NetIO>, MockCOTType> protocol(ios, party, input_count, mul_count, branch_sz, batch_sz);

    // Reset counters and start timing after setup
    ios[0]->counter = 0;
    ios[0]->recv_counter = 0;
    auto start = clock_start();

    // Run protocol
    protocol.authenticate_and_multiply();

    // Connection proofs are now handled inside generate_proofs()
    // with per-branch output_source mapping (default: all pass-through)

    protocol.generate_proofs();

    // Enable external PCS mode: skip LPZK product-of-branches check
    // External PCS will prove: ∃ br: values[b * branch_sz + br] == 0
    protocol.set_external_pcs_mode(true);
    protocol.final_proof();

    auto total_us = time_from(start);
    double total_ms = total_us / 1000.0;

    // Finalize
    bool cheated = finalize_zk_bool<BoolIO<NetIO>, MockCOTType>();
    if (cheated) error("cheated\n");

    // Get IO stats
    size_t bytes_sent = ios[0]->counter;
    size_t bytes_recv = ios[0]->recv_counter;

    // Only prover prints stats
    if (party == ALICE) {
        // Calculate actual COTs: 1 per input bit + 1 per AND gate
        long long cots_for_inputs = (long long)input_count * batch_sz;
        long long cots_for_ands = (long long)mul_count * batch_sz;
        long long total_cots = cots_for_inputs + cots_for_ands;

        cout << "Total: " << total_ms << " ms" << endl;
        cout << "COTs: " << total_cots << " (" << (total_cots / 1000000.0) << "M)" << endl;
        cout << "  (COTs are pre-generated; not included in runtime)" << endl;
        cout << "Bytes sent: " << bytes_sent << " (" << (bytes_sent / 1024.0) << " KB)" << endl;
        cout << "Bytes recv: " << bytes_recv << " (" << (bytes_recv / 1024.0) << " KB)" << endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 7) {
        cout << "usage: " << argv[0] << " PARTY(1/2) PORT IP INPUT MUL BRANCHES BATCHES" << endl;
        cout << endl;
        cout << "Parameters:" << endl;
        cout << "  INPUT    Number of input wires" << endl;
        cout << "  MUL      Number of AND gates" << endl;
        cout << "  BRANCHES Number of branches in disjunction" << endl;
        cout << "  BATCHES  Number of batch instances" << endl;
        return -1;
    }

    parse_party_and_port(argv, &party, &port);
    int input_count = atoi(argv[4]);
    int mul_count = atoi(argv[5]);
    int branches = atoi(argv[6]);
    int batches = atoi(argv[7]);

    // Only prover prints header
    if (party == ALICE) {
        cout << "=== Batchman Prover (Boolean) ===" << endl;
        cout << "Inputs: " << input_count << ", ANDs: " << mul_count << endl;
        cout << "Branches: " << branches << ", Batches: " << batches << endl;
    }

    BoolIO<NetIO>* ios[threads];
    for (int i = 0; i < threads; ++i)
        ios[i] = new BoolIO<NetIO>(new NetIO(party == ALICE ? nullptr : argv[3], port+i, true), party == ALICE);

    test_circuit_zk(ios, party, input_count, mul_count, branches, batches);

    for (int i = 0; i < threads; ++i) {
        delete ios[i]->io;
        delete ios[i];
    }
    return 0;
}
