# Project Rules

## Step-by-Step Confirmation

Before modifying or adding actual code logic, explain what you are about to do and only proceed once the user confirms.

No confirmation needed for:
- Config files (CMakeLists.txt, etc.)
- Running compilation/build commands
- Running benchmarks/tests
- Deleting generated/temporary files

# Architecture

ZK proof system for general-purpose computation. Pipeline:

1. **Guest programs** (Rust, no_std) → compiled to **RV32I** (no multiply)
2. **Register allocation** down to 3 physical registers → each op bakes in its register assignments (e.g., `add.r0.r1.r2`)
3. **Canon3 ISA** — 76 ops mapped to ~54 circuit branches, optimized for minimal circuit size
4. **Witness generation** (witgen) — executes Canon3 bytecode, records CPU/memory/lookup traces
5. **Batched disjunction** (VOLE-based) — proves correct execution across all steps
6. **Polynomial commitment** (Binius FRI) — produces final ZK proof

Key directories:
- `minimal_isa/` — compilation, register allocation, ISA transforms
- `witgen/` — witness trace generation
- `circuits/` — per-operation circuit definitions (208-bit I/O per step)
- `pcs-mode/` — polynomial commitment / proof generation
