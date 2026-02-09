#ifndef MOCK_COT_H__
#define MOCK_COT_H__

#include "emp-tool/emp-tool.h"

namespace emp {

// Fixed delta for mock - both parties use this
// Upper 64 bits are zero: delta lives in GF(2^64) subfield (see batched_disjunction.h)
const block MOCK_COT_DELTA = makeBlock(0x0ULL, 0xCAFEBABE87654321ULL);

// Fixed PRG seed - both parties must use same seed for consistent correlations
const block MOCK_COT_PRG_SEED = makeBlock(0xDEADBEEF12345678ULL, 0xFEEDFACE11111111ULL);

// Mock FerretCOT: pre-allocates buffer, all rcot calls draw from it
// Mimics FerretCOT interface used by OSTriple in emp-zk-bool
template<typename IO>
class MockCOT {
public:
    IO* io;
    int party;
    block Delta;

    static const int64_t BUFFER_SIZE = 100000000;  // 100M COTs
    block* buffer = nullptr;
    int64_t buffer_pos = 0;
    int64_t cots_consumed = 0;

    // Mimic FerretCOT's param structure
    struct {
        int64_t n = 10168320;
        int64_t buf_sz() { return 10005354; }
    } param;

    // Constructor matches FerretCOT signature used in OSTriple
    // Note: OSTriple calls with (3-party), so:
    //   ZK ALICE (Prover) -> MockCOT party = BOB
    //   ZK BOB (Verifier) -> MockCOT party = ALICE
    MockCOT(int party, int threads, IO** ios, bool malicious = false, bool run_setup = true)
        : io(ios[0]), party(party), Delta(MOCK_COT_DELTA) {

        buffer = new block[BUFFER_SIZE];
        PRG prg(&MOCK_COT_PRG_SEED);

        // COT correlation:
        // - Sender (ALICE, has Delta): gets K (key)
        // - Receiver (BOB): gets M = K ⊕ (b ? Delta : 0), with choice bit b in LSB
        //
        // Generate random K and b, then compute appropriate output per party
        // IMPORTANT: Clear LSB of K first, then set LSB to b for receiver
        block K;
        bool b;
        const block lsb_mask = makeBlock(0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFEULL);
        const block one_block = makeBlock(0, 1);

        for (int64_t i = 0; i < BUFFER_SIZE; i++) {
            prg.random_block(&K, 1);
            prg.random_data(&b, 1);
            b = b & 1;  // ensure single bit

            // Clear LSB of K for clean correlation
            K = K & lsb_mask;

            if (party == ALICE) {
                // Sender: K with LSB=0
                buffer[i] = K;
            } else {
                // Receiver (BOB): K ⊕ (b ? Delta : 0), with b in LSB
                // Note: Delta's LSB should be 1, so XORing with Delta when b=1 sets LSB=1
                block mask = b ? Delta : zero_block;
                buffer[i] = K ^ mask;
                // Ensure LSB encodes the choice bit b
                // After XOR with Delta (if b=1), LSB should be Delta's LSB (which is 1)
                // After XOR with zero (if b=0), LSB should be K's LSB (which is 0)
                // So if Delta's LSB is 1, we're good. But let's be explicit:
                buffer[i] = buffer[i] & lsb_mask;
                if (b) buffer[i] = buffer[i] | one_block;
            }
        }
    }

    ~MockCOT() {
        if (buffer) delete[] buffer;
    }

    void reset_counter() {
        cots_consumed = 0;
    }

    // Random COT - fills data with num random COTs
    void rcot(block* data, int64_t num) {
        if (buffer_pos + num > BUFFER_SIZE) {
            error("MockCOT: out of COTs!");
        }
        memcpy(data, buffer + buffer_pos, num * sizeof(block));
        buffer_pos += num;
        cots_consumed += num;
    }

    // Random COT in-place - matches FerretCOT interface
    int64_t rcot_inplace(block* ot_buffer, int64_t length, block seed = zero_block) {
        rcot(ot_buffer, length);
        return param.buf_sz();
    }

    // State management stubs (not needed for mock)
    void assemble_state(void* data, int64_t size) {}
    int disassemble_state(const void* data, int64_t size) { return 0; }
    int64_t state_size() { return 0; }
};

} // namespace emp

#endif // MOCK_COT_H__
