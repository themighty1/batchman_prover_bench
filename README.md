# Batchman Prover Benchmark

Benchmark for the Batchman batched disjunction protocol prover.

Fork of [gconeice/stacking-vole-zk](https://github.com/gconeice/stacking-vole-zk).

## Build

```bash
./run_arithm build   # arithmetic circuit benchmark
./run_bool build     # boolean circuit benchmark
```

## Run

### Arithmetic Circuit Benchmark

```bash
./run_arithm <INPUT> <MUL> <BRANCHES> <BATCHES>
```

Parameters:
- `INPUT` - Number of input wires per circuit
- `MUL` - Number of multiplication gates per circuit
- `BRANCHES` - Number of branches in disjunction (B)
- `BATCHES` - Number of batch instances to prove (R)

Example:
```bash
./run_arithm 10 100 20 1000
```

Output:
```
=== Batchman Prover ===
Inputs: 10, Muls: 100
Branches: 20, Batches: 1000
Total: 1045.54 ms
VOLEs consumed: 521002
  (VOLEs are pre-generated; not included in runtime)
Bytes sent: 4968032 (4851.59 KB)
Bytes recv: 72 (0.0703125 KB)

Verifier: PASSED
```


### Boolean Circuit Benchmark

```bash
./run_bool <INPUT> <MUL> <BRANCHES> <BATCHES>
```

Parameters:
- `INPUT` - Number of input wires per circuit
- `MUL` - Number of AND gates per circuit
- `BRANCHES` - Number of branches in disjunction (B)
- `BATCHES` - Number of batch instances to prove (R)

Example:
```bash
./run_bool 10 100 20 1000
```

Output:
```
=== Batchman Prover (Boolean) ===
Inputs: 10, ANDs: 100
Branches: 20, Batches: 1000
Total: 30.517 ms
COTs: 110000 (0.11M)
  (COTs are pre-generated; not included in runtime)
Bytes sent: 16480 (16.0938 KB)
Bytes recv: 32 (0.03125 KB)

Verifier: PASSED
```



## Extensions Beyond Original Repository

This fork implements several features not present in the original repository:

### 1. Connected Repetitions

**Files:**
- `emp-zk/emp-zk-arith/batched_disjunction.h` (arithmetic)
- `emp-zk/emp-zk-bool/batched_disjunction.h` (boolean)

The original repository implements independent batching where each repetition has
separate witnesses. This fork adds support for **connected repetitions** as
described in the Batchman paper:

> "A more powerful constraint requires P to prove some consistency between the
> repetition witnesses. For instance, some wires of the first repetition should
> be used as particular input wires to the second repetition."

The technique uses IT-MAC linear homomorphism: subtract two supposedly-equal
values and prove the result is an IT-MAC of zero. Many such zero checks are
compressed into O(1) communication via random linear combination.

**Use cases:**
- Hash chains (output of hash i is input to hash i+1)
- State machine transitions
- Iterated function evaluation
- Any computation where batches are sequentially dependent

### 2. Arbitrary Circuit Parameters

**Files:**
- `emp-zk/emp-zk-arith/batched_disjunction.h` (arithmetic)
- `emp-zk/emp-zk-bool/batched_disjunction.h` (boolean)

The original repository hardcodes matrix multiplication as the benchmark circuit, with
coupled INPUT and MUL counts (INPUT = 2xDIM^2, MUL = DIM^3). This fork accepts
arbitrary INPUT and MUL parameters independently.

**Wire routing:** Round-robin assignment:
- `mul_left[i] = inputs[i % input_count]`
- `mul_right[i] = inputs[(i * 7 + 1) % input_count]`

### 3. Boolean Circuit Support

**Files:**
- `emp-zk/emp-zk-bool/batched_disjunction.h` - Boolean protocol
- `emp-zk/emp-vole/mock_cot.h` - Mock COT for benchmarking

The original repository focused on arithmetic circuits (over Fp). This fork adds
equivalent support for boolean circuits using COT (Correlated Oblivious Transfer)
instead of VOLE, with GF(2^128) operations for proof compression.

### 4. Mock VOLE/COT for Benchmarking

**Files:**
- `emp-zk/emp-vole/mock_vole.h` - Pre-generated VOLE correlations
- `emp-zk/emp-vole/mock_cot.h` - Pre-generated COT correlations

These mock implementations pre-generate correlations to benchmark the protocol
logic separately from the OT extension / LPN overhead.

### 5. Binius FRI PCS Mode

**Directory:** `pcs-mode/`

This fork adds a Binius-based FRI polynomial commitment scheme implementation as an
alternative to the Plonky3-based approach. Uses Binius FRI over GF(2^128) with manual
parameter control to bypass auto-chooser limitations.

**Key Features:**
- **Manual FRI parameters** for optimal performance (auto-chooser is legacy)
- **Blake3 hashing** for better performance than Groestl256
- **Grinding support** for computational security boost (proof-of-work)
- **Flat scaling** - nearly constant proof size and time across workload sizes

**Build and Run:**
```bash
cd pcs-mode
cargo build --release --bin bench_binius
./target/release/bench_binius
```

**Benchmark Results (256K roots, K=16, 256× ratio):**

| Metric | Value |
|--------|-------|
| Workload | 256K roots, 16 splits, 67M Q roots |
| Polynomial degrees | deg(Qi)=2^22-1, deg(Z)=263K, deg(P)=67M |
| **Prover time** | **3.43s** (1.56s commit + 1.60s FRI + 0.27s grinding) |
| **Proof size** | **407 KB** (213 KB Z + 195 KB Q) |
| **Verifier time** | **8.85 ms** |
| **Security** | **124-bit total** (100-bit FRI + 24-bit grinding) |

**Parameters:**
- Manual FRI: log_dim=14, log_batch=4, log_inv_rate=2, arity=4
- 16 columns batched (2^14 = 16,384 elements per column)
- 148 test queries, 18-19 fold rounds
- Grinding: 24-bit (2.7M hash attempts, ~10M hashes/sec)

**Scaling (K=16, 100-bit + 20-bit grinding):**
- 16K roots: 1.95s, 307 KB
- 32K roots: 2.10s, 320 KB
- 64K roots: 1.98s, 338 KB
- 128K roots: 2.10s, 363 KB
- 256K roots: 2.52s, 389 KB

Performance remains nearly flat across workload sizes due to fixed manual FRI parameters.
