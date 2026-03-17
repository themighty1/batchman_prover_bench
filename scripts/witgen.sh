#!/bin/bash
#
# End-to-end witness generation pipeline.
# Builds the guest program, compiles to canonical bytecode, generates witness.
#
# Usage: ./witgen.sh <program-name>
# Example: ./witgen.sh json-query

set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ $# -eq 0 ]; then
    echo "Usage: ./witgen.sh <program-name>"
    echo "  Builds guest program, compiles canonical bytecode, generates witness."
    echo ""
    echo "Examples:"
    echo "  ./witgen.sh json-query"
    echo "  ./witgen.sh simple-predicate"
    exit 0
fi

PROGRAM="$1"
GUEST_DIR="$DIR/minimal_isa/guest-programs/$PROGRAM"
ELF_NAME=$(echo "$PROGRAM" | tr '-' '_')
ELF="$DIR/minimal_isa/rv32-build/target/riscv32i-unknown-none-elf/release/$ELF_NAME"
CANONICAL="$GUEST_DIR/canonical.bin"

if [ ! -d "$GUEST_DIR" ]; then
    echo "Error: guest program not found: $GUEST_DIR"
    exit 1
fi

# Step 1: Build RV32I ELF
echo "=== Building RV32I ELF ==="
rustup target add riscv32i-unknown-none-elf 2>/dev/null || true
cd "$DIR/minimal_isa/rv32-build"
cargo build --target riscv32i-unknown-none-elf --release --bin "$ELF_NAME"

# Step 2: Compile canonical bytecode
echo ""
echo "=== Compiling canonical bytecode ==="
cd "$DIR/minimal_isa/reg-analyzer"
cargo run --release --bin rv32_compile_canon3 -- "$ELF" "$CANONICAL"

# Step 3: Generate witness
echo ""
echo "=== Generating witness ==="
WITNESS_DIR="$DIR/witgen/witness/$PROGRAM"
cd "$DIR/witgen"
cargo run --release -- "$CANONICAL" "$WITNESS_DIR"

echo ""
echo "=== Done ==="
echo "ELF:                $ELF"
echo "Canonical bytecode: $CANONICAL"
echo "Witness traces:     $WITNESS_DIR/"
