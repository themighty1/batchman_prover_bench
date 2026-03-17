# zkVM

Zero-knowledge proof system for general-purpose computation. Compiles Rust programs to a minimal RISC-V ISA, generates execution traces, and proves correctness using Batchman (batched disjunction) + Binius (polynomial commitments).

## Setup

One-time setup — installs emp-toolkit dependencies and system packages:

```bash
./setup.sh
```

You also need the Rust toolchain with the RISC-V target:

```bash
rustup target add riscv32i-unknown-none-elf
```

## Usage

```bash
./prove_zkvm json-query
```

```bash
./prove_zkvm simple-predicate
```

Run without arguments to see available programs:

```bash
./prove_zkvm
```

The script handles everything automatically: building toolchains, generating witnesses, compiling circuits, and running all proof components in parallel.

## Example Output

```
=== prove_zkvm: simple-predicate ===

Building binius binaries...
  CPU step proof              0.1s        72 KB  (1618 steps, 0.6M COTs)
  Memory check                2.0s      1000 KB
  MAC consistency             0.2s       452 KB
  Step records                2.0s      1496 KB
  Zero product                0.6s      2057 KB
  ─────────────────────────────────────────
                              time   proof size
  Total                       5.0s      5079 KB

  Note: COT generation cost not included. COTs are provided by an external protocol.
```

## Architecture

See `CLAUDE.md` for full architecture overview.
