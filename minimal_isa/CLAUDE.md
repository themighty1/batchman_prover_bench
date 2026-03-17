# minimal_isa

RV32I compilation and ISA minimization pipeline.

## Subdirectories

- `guest-programs/` — Rust no_std source programs (json-query, simple-predicate)
- `rv32-build/` — cross-compilation to RV32I ELF binaries
- `reg-analyzer/` — ELF decode → CFG → VReg lifting → regalloc2 (3 regs) → canonical bytecode

## Compilation passes (rv32_compile_canon3)

1. **Decode ELF** — parse .text section into decoded instructions, identify functions from symbol table
2. **Resolve auipc pairs** — fuse auipc+jalr/addi/lw/sw into direct references (eliminates PC-relative addressing)
3. **Split into per-function streams** — each function gets its own instruction sequence
4. **Canonicalize (3-slot LRU cache)** — rewrite RV32I register ops into 3-register form using a cache of r0/r1/r2 backed by absolute loads/stores to a mailbox region; insert spill/reload as needed
5. **Flatten** — concatenate per-function streams into a single flat instruction array, remap branch/jump targets to global indices
6. **Decompose shifts** — replace `slli`/`srli`/`srai` with fixed-shift ops (sll1/sll4/sll8/sll16/sll31 etc.) to avoid barrel-shifter gates in circuits. Skippable via `NO_SHIFT_DECOMP=1`
7. **Decompose sw_aligned** — split read-modify-write byte stores into 4 simpler sub-ops (sw_abs0 → lw_aligned → byte_ins_r2 → sw_waligned). Skippable via `NO_SW_DECOMP=1`
8. **Build opcode table** — assign each unique op string a u8 ID, produce code + immediate tables
9. **Patch ELF segments** — rewrite jump table entries and function pointers to use final flat indices
10. **Serialize** — output `FlatProgram` (opcode table, code, immediates, segments, entry PC) as bincode

Output: `canonical.bin` in the guest program directory.

## Build

From `rv32-build/`:
```
cargo build --target riscv32i-unknown-none-elf --release
```

Compile canonical bytecode (from `reg-analyzer/`):
```
cargo run --release --bin rv32_compile_canon3 -- ../rv32-build/target/riscv32i-unknown-none-elf/release/json_query ../guest-programs/json-query/canonical.bin
```

## Key design choices

- 3 physical registers (empirically optimal)
- Register assignments baked into opcodes (`add.r0.r1.r2`)
- ~76 ops, no multiply/divide
- Spills lowered to explicit `sw`/`lw` via dedicated frame register (r4)
- All shifts decomposed into fixed-amount ops to eliminate barrel-shifter circuits
