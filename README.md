# Batchman Prover Benchmark

Benchmark for the Batchman batched disjunction protocol prover.

Fork of [gconeice/stacking-vole-zk](https://github.com/gconeice/stacking-vole-zk).

## Build

```bash
./run build
```

## Run

```bash
./run <DIM> <BRANCHES> <BATCHES>
```

Example:
```bash
./run 3 20 1000
```

## Output

```
=== Batchman Prover ===
Dim: 3 (27 muls/branch)
Branches: 20, Batches: 1000
Total: 992.53 ms
VOLEs consumed: 172002
  (VOLEs are pre-generated; not included in runtime)
Bytes sent: 1760032 (1718.78 KB)
Bytes recv: 56 (0.0546875 KB)
```
