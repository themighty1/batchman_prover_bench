#ifndef MOCK_VOLE_H__
#define MOCK_VOLE_H__

#include "emp-zk/emp-vole/utility.h"

// Fixed delta for mock - both parties use this
const __uint128_t MOCK_DELTA = 0x123456789ABCDEF0ULL;

// Fixed PRG seed - both parties must use same seed for consistent correlations
const block MOCK_PRG_SEED = makeBlock(0xDEADBEEF12345678ULL, 0xCAFEBABE87654321ULL);

// Mock VOLE: pre-allocates buffer, all extend() calls draw from it
template<typename IO>
class MockVole {
public:
    IO* io;
    int party;
    __uint128_t Delta = MOCK_DELTA;

    static const int64_t BUFFER_SIZE = 50000000;  // 50M VOLEs
    __uint128_t* buffer = nullptr;
    int64_t buffer_pos = 0;
    int64_t voles_consumed = 0;

    struct {
        int64_t n = 10168320;
        int64_t buf_sz() { return 10005354; }
    } param;

    MockVole(int party, int threads, IO** ios)
        : io(ios[0]), party(party) {
        // Pre-allocate and fill buffer
        buffer = new __uint128_t[BUFFER_SIZE];
        PRG prg{&MOCK_PRG_SEED};
        // VOLE correlation: mac = val * Delta + key
        // MockVole party is OPPOSITE of ZK party (due to 3-party in ostriple)
        // When MockVole.party==BOB, ZK party is ALICE (Prover) -> needs (val, mac)
        // When MockVole.party==ALICE, ZK party is BOB (Verifier) -> needs key
        for (int64_t i = 0; i < BUFFER_SIZE; i++) {
            uint64_t v, k;
            prg.random_data(&v, 8);
            prg.random_data(&k, 8);
            v %= PR; k %= PR;
            uint64_t mac = add_mod(mult_mod(v, (uint64_t)Delta), k);
            if (party == BOB)
                buffer[i] = ((__uint128_t)v << 64) | mac;
            else
                buffer[i] = k;
        }
    }

    ~MockVole() { delete[] buffer; }

    void setup() {}
    void setup(__uint128_t delta) {}
    void reset_counter() { voles_consumed = 0; }
    __uint128_t delta() { return Delta; }

    void extend(__uint128_t* out, int n) {
        if (buffer_pos + n > BUFFER_SIZE) {
            error("MockVole: out of VOLEs!");
        }
        memcpy(out, buffer + buffer_pos, n * sizeof(__uint128_t));
        buffer_pos += n;
        voles_consumed += n;
    }

    uint64_t extend_inplace(__uint128_t* data, int64_t sz) {
        extend(data, sz);
        return param.buf_sz();
    }
};

#endif
