// Boolean circuit benchmark with MockCOT
//
// Usage: PARTY PORT IP [BATCHES] [CIRCUIT_DIR] [CPU_TRACE_FILE]
//
// Loads all .bin circuit files from CIRCUIT_DIR (default: "circuits").
// If CPU_TRACE_FILE is provided (cpu_trace.bin from the witness generator),
// loads witness rows and derives the branch sequence + per-step hints.
// Otherwise uses random branches.
//
// Binary format (all little-endian):
//   [u32] input_count
//   [u32] and_count
//   [u32] xor_count
//   [u32] inv_count
//   [u32] gate_count (and_count + xor_count + inv_count)
//   For each gate (gate_count entries):
//     [u8]  gate_type (0=AND, 1=XOR, 2=INV)
//     [u32] left_wire
//     [u32] right_wire  (unused for INV)
//   For each output (input_count entries):
//     [i32] output_source (-1=pass-through, >=0 = gate index)
//   Const table (topology constants, not in I/O layout):
//     [u32] num_consts
//     For each const (num_consts entries):
//       [u32] gate_index
//       [u8]  value (0 or 1)
//   [u32] connect_count — first connect_count I/O positions are connected across steps
#include "emp-zk/emp-zk.h"
#include "emp-zk/emp-vole/mock_cot.h"
#include "emp-zk/emp-zk-bool/batched_disjunction.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <dirent.h>

using namespace emp;
using namespace std;

int port, party;
const int threads = 1;

// GF(2^64) multiply using x^64 + x^4 + x^3 + x + 1.
static uint64_t gf64_mul(uint64_t a, uint64_t b) {
    __m128i va = _mm_set_epi64x(0, a);
    __m128i vb = _mm_set_epi64x(0, b);
    __m128i product = _mm_clmulepi64_si128(va, vb, 0x00);
    // reduce: hi * (x^4 + x^3 + x + 1)
    __m128i hi = _mm_srli_si128(product, 8);
    __m128i r1 = _mm_slli_epi64(hi, 4);
    __m128i r2 = _mm_slli_epi64(hi, 3);
    __m128i r3 = _mm_slli_epi64(hi, 1);
    __m128i reduced = _mm_xor_si128(r1, r2);
    reduced = _mm_xor_si128(reduced, r3);
    reduced = _mm_xor_si128(reduced, hi);
    __m128i res = _mm_xor_si128(product, reduced);
    return (uint64_t)_mm_extract_epi64(res, 0);
}

// Verify zero product bin files: Key == MAC + plaintext * delta (GF(2^64)).
static bool verify_zero_product_files(const char *prover_path,
                                      const char *verifier_path,
                                      const char *delta_path) {
    FILE *fp = fopen(prover_path, "rb");
    FILE *fv = fopen(verifier_path, "rb");
    FILE *fd = fopen(delta_path, "rb");
    if (!fp || !fv || !fd) {
        cerr << "Zero-product verify: cannot open files" << endl;
        if (fp) fclose(fp); if (fv) fclose(fv); if (fd) fclose(fd);
        return false;
    }

    uint8_t delta_buf[16];
    fread(delta_buf, 16, 1, fd);
    fclose(fd);
    uint64_t delta;
    memcpy(&delta, delta_buf, 8);  // lower 64 bits

    uint32_t p_bs, p_bc, v_bs, v_bc;
    fread(&p_bs, 4, 1, fp); fread(&p_bc, 4, 1, fp);
    fread(&v_bs, 4, 1, fv); fread(&v_bc, 4, 1, fv);

    if (p_bs != v_bs || p_bc != v_bc) {
        cerr << "Zero-product verify: header mismatch" << endl;
        fclose(fp); fclose(fv);
        return false;
    }

    uint32_t steps = p_bs, bc = p_bc;
    int errors = 0;
    for (uint32_t s = 0; s < steps; s++) {
        uint16_t active_branch;
        fread(&active_branch, 2, 1, fp);

        for (uint32_t bid = 0; bid < bc; bid++) {
            uint64_t mac, pt, key;
            fread(&mac, 8, 1, fp);
            fread(&pt, 8, 1, fp);
            fread(&key, 8, 1, fv);

            uint64_t expected_key = mac ^ gf64_mul(pt, delta);
            if (key != expected_key) {
                if (errors < 5)
                    cerr << "Zero-product verify FAIL: IT-MAC mismatch step=" << s
                         << " branch=" << bid << endl;
                errors++;
            }

            if (bid == active_branch && pt != 0) {
                if (errors < 5)
                    cerr << "Zero-product verify FAIL: active branch plaintext nonzero"
                         << " step=" << s << " branch=" << bid
                         << " pt=" << pt << endl;
                errors++;
            }
        }
    }

    fclose(fp);
    fclose(fv);

    if (errors == 0) {
        cout << "Zero-product check: PASSED (" << steps << " steps, "
             << bc << " branches)" << endl;
        return true;
    } else {
        cerr << "Zero-product check: FAILED (" << errors << " errors)" << endl;
        return false;
    }
}

struct CircuitDef {
    int input_count;
    int and_count;
    int xor_count;
    int inv_count;
    int gate_count;
    int *gate_types;
    int *left;
    int *right;
    int *output_source;
    // Topology constants (offset gates via unit wire, not in I/O layout).
    // TODO: authenticate const wires with proper IT-MACs:
    //   1. Verifier sends random seed (128 bits, once per protocol)
    //   2. Both parties expand PRG(seed, rep, wire_pos) → MAC M
    //   3. Prover injects M into wire array at const wire slot
    //   4. Verifier derives key K_b = M + c_b·Δ per branch b
    //   5. Topology check RLC picks up const wires as regular AND inputs
    int const_count;
    int *const_gate_indices;  // gate index for each const wire
    bool *const_values;       // value (0 or 1) for each const wire
    int connect_count;        // first connect_count positions connected across steps
    string name;

    CircuitDef() : input_count(0), and_count(0), xor_count(0), inv_count(0),
                   gate_count(0), gate_types(nullptr), left(nullptr),
                   right(nullptr), output_source(nullptr),
                   const_count(0), const_gate_indices(nullptr), const_values(nullptr),
                   connect_count(0) {}

    ~CircuitDef() {
        delete[] gate_types;
        delete[] left;
        delete[] right;
        delete[] output_source;
        delete[] const_gate_indices;
        delete[] const_values;
    }

    // Non-copyable
    CircuitDef(const CircuitDef&) = delete;
    CircuitDef& operator=(const CircuitDef&) = delete;
};

CircuitDef* load_circuit(const char *path) {
    ifstream f(path, ios::binary);
    if (!f) {
        cerr << "Failed to open circuit file: " << path << endl;
        exit(1);
    }

    CircuitDef *c = new CircuitDef();
    uint32_t u;

    f.read((char*)&u, 4); c->input_count = u;
    f.read((char*)&u, 4); c->and_count = u;
    f.read((char*)&u, 4); c->xor_count = u;
    f.read((char*)&u, 4); c->inv_count = u;
    f.read((char*)&u, 4); c->gate_count = u;

    if (c->gate_count != c->and_count + c->xor_count + c->inv_count) {
        cerr << "Circuit file corrupt: " << path << endl;
        exit(1);
    }

    c->gate_types = new int[c->gate_count];
    c->left = new int[c->gate_count];
    c->right = new int[c->gate_count];

    for (int g = 0; g < c->gate_count; g++) {
        uint8_t type_byte;
        uint32_t lw, rw;
        f.read((char*)&type_byte, 1);
        f.read((char*)&lw, 4);
        f.read((char*)&rw, 4);
        c->gate_types[g] = (int)type_byte;
        c->left[g] = (int)lw;
        c->right[g] = (int)rw;
    }

    c->output_source = new int[c->input_count];
    for (int i = 0; i < c->input_count; i++) {
        int32_t src;
        f.read((char*)&src, 4);
        c->output_source[i] = src;
    }

    if (!f) {
        cerr << "Circuit file truncated: " << path << endl;
        exit(1);
    }

    // Read const table (topology constants)
    uint32_t nc = 0;
    f.read((char*)&nc, 4);
    if (f && nc > 0) {
        c->const_count = (int)nc;
        c->const_gate_indices = new int[nc];
        c->const_values = new bool[nc];
        for (uint32_t i = 0; i < nc; i++) {
            uint32_t gi;
            uint8_t val;
            f.read((char*)&gi, 4);
            f.read((char*)&val, 1);
            c->const_gate_indices[i] = (int)gi;
            c->const_values[i] = (val != 0);
        }
    }

    // Read connect_count
    uint32_t cc = 0;
    f.read((char*)&cc, 4);
    c->connect_count = f ? (int)cc : c->input_count; // fallback: all connected

    // Extract filename for display
    string p = path;
    size_t slash = p.rfind('/');
    c->name = (slash != string::npos) ? p.substr(slash + 1) : p;

    return c;
}

vector<string> find_bin_files(const string &dir) {
    vector<string> files;
    DIR *d = opendir(dir.c_str());
    if (!d) {
        cerr << "Failed to open directory: " << dir << endl;
        exit(1);
    }
    struct dirent *entry;
    while ((entry = readdir(d)) != NULL) {
        string name = entry->d_name;
        if (name.size() > 4 && name.substr(name.size() - 4) == ".bin") {
            files.push_back(dir + "/" + name);
        }
    }
    closedir(d);
    sort(files.begin(), files.end());
    return files;
}

// Evaluate a circuit in cleartext, updating state in-place.
// Supports normalized circuits: wire index input_count+gate_count = virtual one_wire (constant 1).
void evaluate_cleartext(const CircuitDef *circ, bool *state) {
    int one_wire_idx = circ->input_count + circ->gate_count;
    int wire_count = one_wire_idx + 1;
    bool *wires = new bool[wire_count];

    for (int i = 0; i < circ->input_count; i++)
        wires[i] = state[i];
    wires[one_wire_idx] = true; // virtual constant-1 wire

    for (int g = 0; g < circ->gate_count; g++) {
        int l = circ->left[g];
        int r = circ->right[g];
        switch (circ->gate_types[g]) {
            case 0: wires[circ->input_count + g] = wires[l] & wires[r]; break; // AND
            case 1: wires[circ->input_count + g] = wires[l] ^ wires[r]; break; // XOR
            case 2: wires[circ->input_count + g] = !wires[l]; break;           // INV
        }
    }

    for (int i = 0; i < circ->input_count; i++) {
        if (circ->output_source[i] >= 0) {
            state[i] = wires[circ->input_count + circ->output_source[i]];
        }
    }

    delete[] wires;
}

uint32_t bits_to_u32(const bool *bits, int offset) {
    uint32_t v = 0;
    for (int i = 0; i < 32; i++)
        if (bits[offset + i]) v |= (1u << i);
    return v;
}

uint16_t bits_to_u16(const bool *bits, int offset) {
    uint16_t v = 0;
    for (int i = 0; i < 16; i++)
        if (bits[offset + i]) v |= (1u << i);
    return v;
}

void u32_to_bits(uint32_t v, bool *dst, int n) {
    for (int i = 0; i < n; i++) dst[i] = (v >> i) & 1;
}

// Witness row: matches the Rust WitnessRow binary layout (28 bytes, LE packed).
#pragma pack(push, 1)
struct WitnessRow {
    uint32_t r0, r1, r2;
    uint16_t pc;
    uint16_t next_pc;
    int32_t  imm;
    uint16_t addr;
    uint32_t value;
    uint8_t  op;
    uint8_t  flags;  // bit 0 = has_imm, bit 1 = has_mem
};
#pragma pack(pop)
static_assert(sizeof(WitnessRow) == 28, "WitnessRow must be 28 bytes");

int load_witness(const char *path, WitnessRow **rows_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { cerr << "Failed to open witness: " << path << endl; exit(1); }
    uint32_t count;
    if (fread(&count, 4, 1, f) != 1) { cerr << "Bad witness header" << endl; exit(1); }
    WitnessRow *rows = new WitnessRow[count];
    if (fread(rows, sizeof(WitnessRow), count, f) != count) {
        cerr << "Witness truncated" << endl; exit(1);
    }
    fclose(f);
    *rows_out = rows;
    return (int)count;
}

// Extract 8-bit op ID from circuit const bits (first 8 const values = id bits, LSB first).
int circuit_op_id(const CircuitDef *c) {
    if (c->const_count < 8) return -1;
    int id = 0;
    for (int i = 0; i < 8; i++)
        if (c->const_values[i]) id |= (1 << i);
    return id;
}

// Expand a witness row's hint fields into LSB-first bool bits.
// IMM(32) + ADDR(16) + VALUE(32) + NEXT_PC(16) = 96 hint bits.
static const int HINT_BITS_COUNT = 96;
void witness_row_to_hint_bits(const WitnessRow &row, bool *dst) {
    u32_to_bits((uint32_t)row.imm,     dst,       32);
    u32_to_bits((uint32_t)row.addr,    dst + 32,  16);
    u32_to_bits(row.value,             dst + 48,  32);
    u32_to_bits((uint32_t)row.next_pc, dst + 80,  16);
}

// Expand a witness row's CPU state into LSB-first bool bits (112 bits).
// r0(32) + r1(32) + r2(32) + pc(16)
void witness_row_to_state_bits(const WitnessRow &row, bool *dst) {
    u32_to_bits(row.r0, dst, 32);
    u32_to_bits(row.r1, dst + 32, 32);
    u32_to_bits(row.r2, dst + 64, 32);
    u32_to_bits((uint32_t)row.pc, dst + 96, 16);
}

// Each branch is a distinct ISA circuit.
// If witness is provided (non-null), the branch sequence and per-step hints
// are derived from it; otherwise random branches are used.
void test_zk(BoolIO<NetIO> *ios[threads], int party,
             int batch_sz, vector<CircuitDef*> &isa_circuits,
             WitnessRow *witness, int witness_count,
             const char *cpu_trace_path) {
    int isa_count = (int)isa_circuits.size();
    int branch_sz = isa_count;

    int input_count = isa_circuits[0]->input_count;
    int connect_count = isa_circuits[0]->connect_count;
    int and_count = isa_circuits[0]->and_count;

    // input_count and connect_count must be uniform. and_count = max across branches.
    for (int i = 1; i < isa_count; i++) {
        if (isa_circuits[i]->input_count != input_count) {
            cerr << "Input count mismatch: " << isa_circuits[i]->name
                 << " (got " << isa_circuits[i]->input_count
                 << ", expected " << input_count << ")" << endl;
            exit(1);
        }
        if (isa_circuits[i]->connect_count != connect_count) {
            cerr << "Connect count mismatch: " << isa_circuits[i]->name
                 << " (got " << isa_circuits[i]->connect_count
                 << ", expected " << connect_count << ")" << endl;
            exit(1);
        }
        if (isa_circuits[i]->and_count > and_count)
            and_count = isa_circuits[i]->and_count;
    }

    // Build op_id → branch index lookup
    int op_to_branch[256];
    memset(op_to_branch, -1, sizeof(op_to_branch));
    for (int i = 0; i < isa_count; i++) {
        int op_id = circuit_op_id(isa_circuits[i]);
        if (op_id >= 0) op_to_branch[op_id] = i;
    }

    // Build active_branch_map and hint bits
    int *branch_map = new int[batch_sz];
    bool *all_hint_bits = nullptr;
    bool *initial_state = nullptr;
    int hint_offset = connect_count;  // hints start after CPU state
    int hint_count_per_step = input_count - connect_count;

    if (witness != nullptr) {
        if (witness_count < batch_sz) {
            cerr << "Witness has " << witness_count << " rows but need " << batch_sz << endl;
            exit(1);
        }
        // Derive branch map from witness op IDs
        for (int b = 0; b < batch_sz; b++) {
            int br = op_to_branch[witness[b].op];
            if (br < 0) {
                cerr << "No circuit for witness op " << (int)witness[b].op
                     << " at step " << b << endl;
                exit(1);
            }
            branch_map[b] = br;
        }
        // Build hint bits: flat array of batch_sz * hint_count_per_step bools
        all_hint_bits = new bool[(long long)batch_sz * hint_count_per_step]();
        for (int b = 0; b < batch_sz; b++) {
            witness_row_to_hint_bits(witness[b], all_hint_bits + (long long)b * hint_count_per_step);
        }
        // Build initial state from first witness row
        initial_state = new bool[input_count]();
        witness_row_to_state_bits(witness[0], initial_state);
        witness_row_to_hint_bits(witness[0], initial_state + hint_offset);
    } else {
        // Random branch assignment (both parties use same PRG)
        PRG prg;
        for (int b = 0; b < batch_sz; b++) {
            uint32_t r;
            prg.random_data(&r, sizeof(uint32_t));
            branch_map[b] = r % branch_sz;
        }
    }

    using MockCOTType = MockCOT<BoolIO<NetIO>>;
    OSTriple<BoolIO<NetIO>, MockCOTType> *ostriple;
    if (party == ALICE) {
        ZKBoolCircExecPrv<BoolIO<NetIO>, MockCOTType> *t = new ZKBoolCircExecPrv<BoolIO<NetIO>, MockCOTType>();
        CircuitExecution::circ_exec = t;
        ProtocolExecution::prot_exec = new ZKProver<BoolIO<NetIO>, MockCOTType>(ios, threads, t, nullptr);
        ostriple = ((ZKProver<BoolIO<NetIO>, MockCOTType>*)(ProtocolExecution::prot_exec))->ostriple;
    } else {
        ZKBoolCircExecVer<BoolIO<NetIO>, MockCOTType> *t = new ZKBoolCircExecVer<BoolIO<NetIO>, MockCOTType>();
        CircuitExecution::circ_exec = t;
        ProtocolExecution::prot_exec = new ZKVerifier<BoolIO<NetIO>, MockCOTType>(ios, threads, t, nullptr);
        ostriple = ((ZKVerifier<BoolIO<NetIO>, MockCOTType>*)(ProtocolExecution::prot_exec))->ostriple;
        ostriple->delta = MOCK_COT_DELTA;
        ostriple->choice[1] = MOCK_COT_DELTA;
    }
    ostriple->ferret->reset_counter();

    BoolBatchedDisjunction<BoolIO<NetIO>, MockCOTType> protocol(
        ios, party, input_count, and_count, branch_sz, batch_sz, connect_count);

    // Per-branch: full circuit definition, output source, and const gate indices
    for (int b = 0; b < branch_sz; b++) {
        protocol.set_branch_circuit(b, isa_circuits[b]->gate_count, isa_circuits[b]->gate_types, isa_circuits[b]->left, isa_circuits[b]->right);
        protocol.set_output_source(b, isa_circuits[b]->output_source);
        if (isa_circuits[b]->const_count > 0)
            protocol.set_branch_const_gates(b, isa_circuits[b]->const_count, isa_circuits[b]->const_gate_indices);
    }

    // Set branch map and hints
    protocol.set_active_branch_map(branch_map);
    if (witness != nullptr && party == ALICE) {
        protocol.set_hints(all_hint_bits, hint_offset, hint_count_per_step);
        // Set initial state
        protocol.prover_initial_state = initial_state;
    }

    // DEBUG: cleartext simulation to find first state divergence
    bool debug_mode = (getenv("DEBUG") != nullptr);
    if (debug_mode && witness != nullptr && party == ALICE) {
        bool *sim_state = new bool[input_count]();
        // Init from witness row 0
        witness_row_to_state_bits(witness[0], sim_state);
        witness_row_to_hint_bits(witness[0], sim_state + hint_offset);

        for (int step = 0; step < batch_sz; step++) {
            // Check CPU state matches witness BEFORE evaluation
            uint32_t sim_r0 = bits_to_u32(sim_state, 0);
            uint32_t sim_r1 = bits_to_u32(sim_state, 32);
            uint32_t sim_r2 = bits_to_u32(sim_state, 64);
            uint16_t sim_pc  = bits_to_u16(sim_state, 96);

            bool mismatch = (sim_r0 != witness[step].r0 ||
                             sim_r1 != witness[step].r1 ||
                             sim_r2 != witness[step].r2 ||
                             sim_pc  != witness[step].pc);
            if (mismatch) {
                cerr << dec; // reset to decimal
                cerr << "=== STATE DIVERGENCE at step " << step
                     << " (op=" << (int)witness[step].op
                     << " branch=" << branch_map[step] << ") ===" << endl;
                cerr << "  sim: r0=0x" << hex << sim_r0
                     << " r1=0x" << sim_r1
                     << " r2=0x" << sim_r2
                     << " pc=0x" << hex << sim_pc << dec << endl;
                cerr << "  wit: r0=0x" << hex << witness[step].r0
                     << " r1=0x" << witness[step].r1
                     << " r2=0x" << witness[step].r2
                     << " pc=0x" << hex << witness[step].pc << dec << endl;
                cerr << "  r0 XOR: 0x" << hex << (sim_r0 ^ witness[step].r0) << dec << endl;
                if (step > 0) {
                    cerr << "  prev step " << dec << (step-1)
                         << " op=" << (int)witness[step-1].op
                         << " branch=" << branch_map[step-1]
                         << " circuit=" << isa_circuits[branch_map[step-1]]->name
                         << " value=" << witness[step-1].value
                         << " r2=" << witness[step-1].r2 << endl;
                    // Show 3 steps before divergence
                    for (int k = max(0, step-4); k < step; k++) {
                        cerr << "  trace[" << k << "] op=" << (int)witness[k].op
                             << " r0=0x" << hex << witness[k].r0
                             << " r1=0x" << witness[k].r1
                             << " r2=0x" << witness[k].r2
                             << " pc=0x" << witness[k].pc
                             << " val=" << witness[k].value
                             << " imm=" << witness[k].imm << dec << endl;
                    }
                }
                // Stop at first mismatch
                cerr << "(stopping at first divergence)" << endl;
                break;
            }

            // Inject hints for this step
            bool hint_bits_tmp[HINT_BITS_COUNT];
            witness_row_to_hint_bits(witness[step], hint_bits_tmp);
            for (int i = 0; i < hint_count_per_step; i++)
                sim_state[hint_offset + i] = hint_bits_tmp[i];

            // Evaluate active branch circuit
            evaluate_cleartext(isa_circuits[branch_map[step]], sim_state);
        }
        delete[] sim_state;
        cerr << "DEBUG: cleartext simulation done (" << batch_sz << " steps)" << endl;
    }

    ios[0]->counter = 0;
    ios[0]->recv_counter = 0;
    auto start = clock_start();

    // Configure segment-based parallel proving.
    // SEGMENT_SIZE env var controls steps per segment (default: 0 = no segmentation).
    // CONCURRENCY env var controls max parallel segments (default: 2).
    {
        const char *seg_env = getenv("SEGMENT_SIZE");
        const char *conc_env = getenv("CONCURRENCY");
        if (seg_env) protocol.segment_size = atoi(seg_env);
        if (conc_env) protocol.max_concurrent_segments = atoi(conc_env);
    }

    protocol.authenticate_and_multiply();
    protocol.generate_proofs();

    // Sanity check: compare step record plaintexts against witness
    if (debug_mode && witness != nullptr && party == ALICE) {
        int sr_errors = 0;
        for (int step = 0; step < batch_sz && sr_errors < 5; step++) {
            int br = branch_map[step];
            block sr = protocol.step_record_plaintexts[(long long)step * branch_sz + br];
            uint64_t lo = _mm_extract_epi64(sr, 0);
            uint64_t hi = _mm_extract_epi64(sr, 1);

            // Decode step record fields
            uint32_t sr_value = (uint32_t)(lo & 0xFFFFFFFF);
            int32_t  sr_imm   = (int32_t)((lo >> 32) & 0xFFFFFFFF);
            uint16_t sr_pc    = (uint16_t)(hi & 0xFFFF);
            uint16_t sr_npc   = (uint16_t)((hi >> 16) & 0xFFFF);
            uint16_t sr_addr  = (uint16_t)((hi >> 32) & 0xFFFF);
            // uint8_t  sr_op    = (uint8_t)((hi >> 48) & 0xFF);

            const WitnessRow &w = witness[step];
            bool bad = false;
            string msg;

            if (sr_pc != w.pc) {
                msg += " pc=" + to_string(sr_pc) + "!=" + to_string(w.pc);
                bad = true;
            }
            if (sr_npc != w.next_pc) {
                msg += " npc=" + to_string(sr_npc) + "!=" + to_string(w.next_pc);
                bad = true;
            }
            if (sr_imm != w.imm) {
                msg += " imm=" + to_string(sr_imm) + "!=" + to_string(w.imm);
                bad = true;
            }
            if (w.addr != 0 && sr_addr != w.addr) {
                msg += " addr=0x" + to_string(sr_addr) + "!=0x" + to_string(w.addr);
                bad = true;
            }
            if (w.value != 0 && sr_value != w.value) {
                msg += " val=" + to_string(sr_value) + "!=" + to_string(w.value);
                bad = true;
            }

            if (bad) {
                cerr << "STEP_RECORD DIVERGE step " << step
                     << " op=" << (int)w.op
                     << " (" << isa_circuits[br]->name << "):"
                     << msg << endl;
                sr_errors++;
            }
        }
        if (sr_errors == 0)
            cerr << "Step record check: PASSED (" << batch_sz << " steps)" << endl;
        else
            cerr << "Step record check: " << sr_errors << " divergences" << endl;
    }

    protocol.set_external_pcs_mode(true);
    protocol.final_proof();

    // Derive data output directory from cpu_trace_path:
    //   witgen/witness/<program>/cpu_trace.bin → data/<program>/
    string data_dir = "data";
    if (cpu_trace_path) {
        string tp(cpu_trace_path);
        // Strip trailing filename
        size_t s1 = tp.find_last_of('/');
        if (s1 != string::npos) {
            string parent = tp.substr(0, s1);
            size_t s2 = parent.find_last_of('/');
            string program = (s2 != string::npos) ? parent.substr(s2 + 1) : parent;
            data_dir = "data/" + program;
        }
    }
    {
        string mkdir_cmd = "mkdir -p " + data_dir;
        system(mkdir_cmd.c_str());
    }

    // Write step records to disk
    if (party == ALICE) {
        protocol.write_step_records((data_dir + "/step_records_prover.bin").c_str());
        protocol.write_zero_product((data_dir + "/zero_product_prover.bin").c_str());

        // Cross-check Batchman's plaintext step records against witgen's packed_pt.bin.
        if (cpu_trace_path) {
            string trace_dir(cpu_trace_path);
            size_t slash = trace_dir.find_last_of('/');
            string packed_pt_path = (slash != string::npos)
                ? trace_dir.substr(0, slash) + "/packed_pt.bin"
                : "packed_pt.bin";
            FILE *pf = fopen(packed_pt_path.c_str(), "rb");
            if (pf) {
                uint32_t pt_count;
                fread(&pt_count, 4, 1, pf);
                int mismatches = 0;
                for (uint32_t s = 0; s < pt_count && s < (uint32_t)batch_sz; s++) {
                    uint8_t expected[16];
                    fread(expected, 16, 1, pf);
                    int active = protocol.active_branch_map[s];
                    block actual = protocol.step_record_plaintexts[(long long)s * protocol.branch_sz + active];
                    uint8_t actual_bytes[16];
                    memcpy(actual_bytes, &actual, 16);
                    if (memcmp(expected, actual_bytes, 16) != 0) {
                        if (mismatches < 5)
                            cerr << "packed_pt mismatch at step " << s << endl;
                        mismatches++;
                    }
                }
                fclose(pf);
                if (mismatches == 0)
                    cerr << "packed_pt cross-check: PASSED (" << pt_count << " steps)" << endl;
                else
                    cerr << "packed_pt cross-check: FAILED (" << mismatches << " mismatches)" << endl;
            }
        }
    } else {
        protocol.write_step_records((data_dir + "/step_records_verifier.bin").c_str());
        protocol.write_zero_product_verifier((data_dir + "/zero_product_verifier.bin").c_str());
        protocol.write_delta((data_dir + "/delta.bin").c_str());
    }

    auto total_us = time_from(start);
    double total_ms = total_us / 1000.0;

    bool cheated = finalize_zk_bool<BoolIO<NetIO>, MockCOTType>();
    if (cheated) error("cheated\n");

    if (debug_mode && party == ALICE) {
        // Cleartext verification: replay the same branch sequence
        bool *state = new bool[input_count]();
        if (initial_state) {
            memcpy(state, initial_state, input_count);
        }

        for (int b = 0; b < batch_sz; b++) {
            if (witness != nullptr) {
                witness_row_to_hint_bits(witness[b], state + hint_offset);
            }
            int br = branch_map[b];
            evaluate_cleartext(isa_circuits[br], state);
        }

        bool match = true;
        for (int i = 0; i < input_count; i++) {
            if (state[i] != protocol.prover_final_state[i]) {
                match = false;
                break;
            }
        }

        if (!match) {
            uint32_t ct_r0 = bits_to_u32(state, 0);
            uint32_t ct_r1 = bits_to_u32(state, 32);
            uint32_t ct_r2 = bits_to_u32(state, 64);
            uint16_t ct_pc = bits_to_u16(state, 96);
            uint32_t pv_r0 = bits_to_u32(protocol.prover_final_state, 0);
            uint32_t pv_r1 = bits_to_u32(protocol.prover_final_state, 32);
            uint32_t pv_r2 = bits_to_u32(protocol.prover_final_state, 64);
            uint16_t pv_pc = bits_to_u16(protocol.prover_final_state, 96);
            cerr << "Cleartext check: FAILED" << endl;
            cerr << "  Expected: r0=" << ct_r0 << " r1=" << ct_r1
                 << " r2=" << ct_r2 << " pc=" << ct_pc << endl;
            cerr << "  Got:      r0=" << pv_r0 << " r1=" << pv_r1
                 << " r2=" << pv_r2 << " pc=" << pv_pc << endl;
            exit(1);
        } else {
            cerr << "Cleartext check: PASSED" << endl;
        }

        delete[] state;
    }

    if (party == ALICE) {
        size_t bytes_sent = ios[0]->counter;
        size_t bytes_recv = ios[0]->recv_counter;
        long long cots_per_step = (long long)input_count + 3LL * and_count;
        long long total_cots = cots_per_step * batch_sz;
        cout << "COTs: " << total_cots << " (" << (total_cots / 1000000.0) << "M)" << endl;
        cout << "Bytes sent: " << bytes_sent << " (" << (bytes_sent / 1024.0) << " KB)" << endl;
        cout << "Bytes recv: " << bytes_recv << " (" << (bytes_recv / 1024.0) << " KB)" << endl;
        cout << "Total: " << total_ms << " ms" << endl;
    }

    delete[] branch_map;
    delete[] all_hint_bits;
    delete[] initial_state;
}

int main(int argc, char** argv) {
    // Verify subcommand: cross-check zero product bin files
    if (argc >= 2 && string(argv[1]) == "verify") {
        bool ok = verify_zero_product_files(
            "data/zero_product_prover.bin",
            "data/zero_product_verifier.bin",
            "data/delta.bin");
        return ok ? 0 : 1;
    }

    if (argc < 4) {
        cout << "usage: " << argv[0] << " PARTY PORT IP [BATCHES] [CIRCUIT_DIR] [CPU_TRACE_FILE]" << endl;
        return -1;
    }

    parse_party_and_port(argv, &party, &port);

    int batches        = (argc > 4) ? atoi(argv[4]) : 100;
    string circuit_dir = (argc > 5) ? argv[5] : "circuits";
    const char *cpu_trace_path = (argc > 6) ? argv[6] : nullptr;

    BoolIO<NetIO>* ios[threads];
    for (int i = 0; i < threads; ++i)
        ios[i] = new BoolIO<NetIO>(new NetIO(party == ALICE ? nullptr : argv[3], port+i, true), party == ALICE);

    vector<string> paths = find_bin_files(circuit_dir);
    if (paths.empty()) {
        cerr << "No .bin files found in " << circuit_dir << endl;
        return 1;
    }

    vector<CircuitDef*> circuits;
    for (size_t i = 0; i < paths.size(); i++)
        circuits.push_back(load_circuit(paths[i].c_str()));

    // Find max AND across all branches
    int max_and = 0;
    for (size_t i = 0; i < circuits.size(); i++)
        if (circuits[i]->and_count > max_and) max_and = circuits[i]->and_count;

    // Load witness if provided
    WitnessRow *witness = nullptr;
    int witness_count = 0;
    if (cpu_trace_path) {
        witness_count = load_witness(cpu_trace_path, &witness);
        for (int i = 0; i < witness_count; i++) {
            if (witness[i].next_pc != (uint16_t)(witness[i].pc + 1)) {
                cerr << "Witness inconsistency at step " << i
                     << ": pc=" << witness[i].pc
                     << " next_pc=" << witness[i].next_pc
                     << " (expected " << (uint16_t)(witness[i].pc + 1) << ")" << endl;
                exit(1);
            }
        }
        if (batches > witness_count) batches = witness_count;
        if (party == ALICE)
            cout << "Loaded witness: " << witness_count << " rows from " << cpu_trace_path << endl;
    }

    if (party == ALICE) {
        cout << circuits.size() << " circuits, " << batches << " steps" << endl;
    }

    test_zk(ios, party, batches, circuits, witness, witness_count, cpu_trace_path);

    delete[] witness;
    for (size_t i = 0; i < circuits.size(); i++)
        delete circuits[i];

    for (int i = 0; i < threads; ++i) {
        delete ios[i]->io;
        delete ios[i];
    }
    return 0;
}
