use mpz_circuits_core::{CircuitBuilder, Circuit, Gate, Node, Feed, ops};
use std::collections::HashMap;
use std::io::Write;

/// I/O layout (208 bits):
///
///   CPU state (112 bits) — connected across steps via connection check:
///     r0(32) + r1(32) + r2(32) + pc(16)
///
///   Hints (96 bits) — prover-supplied each step, no connection check:
///     immediate(32)    — immediate operand value (checked by instruction decoding)
///     addr(16)         — memory address (64KB address space)
///     value(32)        — memory value
///     pc_inc(16)      — always pc+1 (incremented PC, supplied by witness)
///
///   Not all ops use all hints. Unused slots are don't-care (pass-through).
///   Ops may repurpose hint slots for op-specific data (documented per builder).
///
///     Op            | imm  | addr | value              | pc_inc
///     --------------|------|------|--------------------|--------
///     add,slt,sll,  |      |      |                    | pc+1 → out_pc
///       srl,sra     |      |      |                    |
///     addi,xori,lui | imm  |      |                    | pc+1 → out_pc
///     lw_aligned    | imm  |      | value              | pc+1 → out_pc
///     lw, sw        | imm  |      | value              | pc+1 → out_pc
///     lw_abs0/1/2   | imm  |      | value              | pc+1 → out_pc
///     sw_abs0/1/2   | imm  |      |                    | pc+1 → out_pc
///     byte_sel_r2   |      |      | value[0..7]=byte   | pc+1 → out_pc
///     sext8         |      |      |                    | pc+1 → out_pc
///     blt,bge,bne,  | imm  |      |                    | MUX(cond, pc_inc, imm)
///       bltu,beq    |      |      |                    |
///     jal,jal_call  | imm  |      |                    | (out_pc = imm)
///     jalr_call     |      |      |                    | (out_pc = r0)
///     ret           |      |      |                    | (out_pc = r0)
///
/// Per-branch constants are NOT in the I/O layout.
/// They are internal offset gates (XOR with one_wire) baked into the topology.
/// The binary format includes a const table so batchman can read them.
/// Current const layout (12 bits): id(8) + is_mem_read(1) + is_mem_write(1) + has_immediate(1) + is_byte_sel_r2(1)
///
const R0_BITS: usize = 32;
const R1_BITS: usize = 32;
const R2_BITS: usize = 32;
const PC_BITS: usize = 16;
const IMM_BITS: usize = 32;
const ADDR_BITS: usize = 16;
const VALUE_BITS: usize = 32;
const PC_INC_BITS: usize = 16;
const CPU_STATE_BITS: usize = R0_BITS + R1_BITS + R2_BITS + PC_BITS; // connected across steps
const HINT_BITS: usize = IMM_BITS + ADDR_BITS + VALUE_BITS + PC_INC_BITS;
const INPUT_COUNT: usize = CPU_STATE_BITS + HINT_BITS;

// CPU state (connected)
const R0_OFF: usize = 0;
const R1_OFF: usize = 32;
const R2_OFF: usize = 64;
const PC_OFF: usize = 96;
// Hints (fresh)
const IMM_OFF: usize = 112;
const ADDR_OFF: usize = 144;
const VALUE_OFF: usize = ADDR_OFF + ADDR_BITS;  // 163
const PC_INC_OFF: usize = VALUE_OFF + VALUE_BITS;  // 195

type W = Node<Feed>;

struct Inputs {
    r0: Vec<W>, r1: Vec<W>, r2: Vec<W>,
    pc: Vec<W>, imm: Vec<W>, addr: Vec<W>, value: Vec<W>,
    pc_inc: Vec<W>,
}

fn add_inputs(b: &mut CircuitBuilder) -> Inputs {
    Inputs {
        r0: (0..R0_BITS).map(|_| b.add_input()).collect(),
        r1: (0..R1_BITS).map(|_| b.add_input()).collect(),
        r2: (0..R2_BITS).map(|_| b.add_input()).collect(),
        pc: (0..PC_BITS).map(|_| b.add_input()).collect(),
        imm: (0..IMM_BITS).map(|_| b.add_input()).collect(),
        addr: (0..ADDR_BITS).map(|_| b.add_input()).collect(),
        value: (0..VALUE_BITS).map(|_| b.add_input()).collect(),
        pc_inc: (0..PC_INC_BITS).map(|_| b.add_input()).collect(),
    }
}

/// Standard output wiring: result → r0, everything else passes through.
fn add_outputs(b: &mut CircuitBuilder, inp: &Inputs, result: &[W]) {
    // r0 = result
    for &s in result { b.add_output(s); }
    // r1 pass-through
    for &w in &inp.r1 { let c = b.add_id_gate(w); b.add_output(c); }
    // r2 pass-through
    for &w in &inp.r2 { let c = b.add_id_gate(w); b.add_output(c); }
    // pc = pc_inc (XOR with zero to avoid cross-wire check: out[96..] != in[208..])
    let z = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    for &w in &inp.pc_inc { let c = b.add_xor_gate(w, z); b.add_output(c); }
    // hints pass-through
    for &w in &inp.imm { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.addr { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.value { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.pc_inc { let c = b.add_id_gate(w); b.add_output(c); }
}

/// Build const bit vector: id(8) + is_mem_read(1) + is_mem_write(1) + has_immediate(1) + is_byte_sel_r2(1)
fn const_bits(id: u8, is_mem_read: bool, is_mem_write: bool, has_immediate: bool, is_byte_sel_r2: bool) -> Vec<bool> {
    let mut bits: Vec<bool> = (0..8).map(|i| (id >> i) & 1 == 1).collect();
    bits.push(is_mem_read);
    bits.push(is_mem_write);
    bits.push(has_immediate);
    bits.push(is_byte_sel_r2);
    bits
}

/// Build a constant wire (0 or 1) as an XOR gate.
fn make_const_wire(b: &mut CircuitBuilder, zero: W, one: W, val: bool) -> W {
    if val { b.add_xor_gate(one, zero) } else { b.add_xor_gate(zero, zero) }
}

/// Append const-wire outputs beyond the I/O layout.
/// These are offset gates (XOR with one_wire) — pure topology, not in I/O.
fn add_const_outputs(b: &mut CircuitBuilder, zero: W, one: W, values: &[bool]) {
    for &val in values {
        let c = make_const_wire(b, zero, one, val);
        b.add_output(c);
    }
}

// ── Shared sub-circuits ──────────────────────────────────────────────

/// 32-bit addition: a + b (overflow discarded). 31 AND gates.
fn ripple_carry_add(b: &mut CircuitBuilder, a: &[W], operand_b: &[W]) -> Vec<W> {
    ops::wrapping_add(b, a, operand_b)
}

/// 32-bit subtraction: a - b (underflow discarded). 32 AND gates.
fn ripple_carry_sub(b: &mut CircuitBuilder, a: &[W], operand_b: &[W]) -> Vec<W> {
    let (diff, _underflow) = ops::wrapping_sub(b, a, operand_b);
    diff
}

// Variable-amount barrel shifters removed — custom ISA uses constant-amount shifts.
/// Signed less-than: result = (a < b) ? 1 : 0 (signed). 33 AND gates.
/// Uses wrapping_sub (32 AND) + 1 AND for sign-bit MUX.
fn signed_less_than(b: &mut CircuitBuilder, a: &[W], operand_b: &[W]) -> Vec<W> {
    let (_diff, underflow) = ops::wrapping_sub(b, a, operand_b);
    // underflow = 1 when a < b unsigned (borrow)

    // Signed compare: if signs differ, result = sign_a; else result = underflow
    let sign_a = a[31];
    let signs_differ = b.add_xor_gate(sign_a, operand_b[31]);
    let diff = b.add_xor_gate(underflow, sign_a);
    let sel = b.add_and_gate(signs_differ, diff);
    let result_bit = b.add_xor_gate(underflow, sel);

    let zero = b.add_xor_gate(a[0], a[0]);
    let mut result = Vec::with_capacity(32);
    result.push(result_bit);
    for _ in 1..32 {
        result.push(b.add_id_gate(zero));
    }
    result
}

/// Unsigned less-than: result = (a < b) ? 1 : 0 (unsigned). 32 AND gates.
/// Uses wrapping_sub underflow bit directly.
fn unsigned_less_than(b: &mut CircuitBuilder, a: &[W], operand_b: &[W]) -> Vec<W> {
    let (_diff, underflow) = ops::wrapping_sub(b, a, operand_b);
    // underflow = 1 when a < b unsigned

    let zero = b.add_xor_gate(a[0], a[0]);
    let mut result = Vec::with_capacity(32);
    result.push(underflow);
    for _ in 1..32 {
        result.push(b.add_id_gate(zero));
    }
    result
}

// ── Instruction builders ─────────────────────────────────────────────

/// r0 = r1 + r2
fn build_add(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);
    let sum = ripple_carry_add(&mut b, &inp.r1, &inp.r2);
    add_outputs(&mut b, &inp, &sum);
    let cbits = const_bits(id, false, false, false, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// r0 = r1 - r2
fn build_sub(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);
    let diff = ripple_carry_sub(&mut b, &inp.r1, &inp.r2);
    add_outputs(&mut b, &inp, &diff);
    let cbits = const_bits(id, false, false, false, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// r0 = r1 & r2 (32 AND gates)
fn build_and(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);
    let result: Vec<W> = (0..32).map(|i| b.add_and_gate(inp.r1[i], inp.r2[i])).collect();
    add_outputs(&mut b, &inp, &result);
    let cbits = const_bits(id, false, false, false, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// r0 = r1 & imm (32 AND gates)
fn build_andi(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);
    let result: Vec<W> = (0..32).map(|i| b.add_and_gate(inp.r1[i], inp.imm[i])).collect();
    add_outputs(&mut b, &inp, &result);
    let cbits = const_bits(id, false, false, true, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// r0 = r1 | r2 (a|b = (a^b) ^ (a&b), 32 AND gates)
fn build_or(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);
    let result: Vec<W> = (0..32).map(|i| {
        let x = b.add_xor_gate(inp.r1[i], inp.r2[i]);
        let a = b.add_and_gate(inp.r1[i], inp.r2[i]);
        b.add_xor_gate(x, a)
    }).collect();
    add_outputs(&mut b, &inp, &result);
    let cbits = const_bits(id, false, false, false, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

// Variable-amount barrel shifters removed — our custom ISA replaced them
// with constant-amount shifts (sll1/4/8/16/31, srl1/4/8/16/31, sra1/4/8/16/31).
// fn build_sll(id: u8) -> (Circuit, Vec<bool>) { ... }
// fn build_srl(id: u8) -> (Circuit, Vec<bool>) { ... }
// fn build_sra(id: u8) -> (Circuit, Vec<bool>) { ... }

/// r0 = r1 ^ r2 (0 AND gates)
fn build_xor(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);
    let result: Vec<W> = (0..32).map(|i| b.add_xor_gate(inp.r1[i], inp.r2[i])).collect();
    add_outputs(&mut b, &inp, &result);
    let cbits = const_bits(id, false, false, false, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// r0 = (r1 < r2) ? 1 : 0  (signed)
fn build_slt(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);
    let result = signed_less_than(&mut b, &inp.r1, &inp.r2);
    add_outputs(&mut b, &inp, &result);
    let cbits = const_bits(id, false, false, false, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// r0 = (r1 < r2) ? 1 : 0  (unsigned)
fn build_sltu(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);
    let result = unsigned_less_than(&mut b, &inp.r1, &inp.r2);
    add_outputs(&mut b, &inp, &result);
    let cbits = const_bits(id, false, false, false, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// r0 = (r1 < imm) ? 1 : 0  (signed)
fn build_slti(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);
    let result = signed_less_than(&mut b, &inp.r1, &inp.imm);
    add_outputs(&mut b, &inp, &result);
    let cbits = const_bits(id, false, false, true, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// lw_aligned — word-aligned load.
///   u16addr = (r1 + imm)[0..ADDR_BITS]
///   addr    = u16addr & ~3                (word-aligned address)
///   r2      = u16addr & 3                 (byte offset within the word: 0, 1, 2, or 3)
///   value   = mem[addr]                   (hint, verified externally)
///   r0      = value
///   r1 unchanged, pc = pc_inc
fn build_lw_aligned(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let computed_addr = ripple_carry_add(&mut b, &inp.r1[..ADDR_BITS], &inp.imm[..ADDR_BITS]);

    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);

    // r0 = value (XOR with zero to avoid translator's cross-wire check)
    for &w in &inp.value {
        let c = b.add_xor_gate(w, zero);
        b.add_output(c);
    }
    // r1 pass-through
    for &w in &inp.r1 { let c = b.add_id_gate(w); b.add_output(c); }
    // r2 = computed_addr & 3 (2 LSBs, zero-padded)
    { let c = b.add_id_gate(computed_addr[0]); b.add_output(c); }
    { let c = b.add_id_gate(computed_addr[1]); b.add_output(c); }
    for _ in 2..R2_BITS {
        let c = b.add_id_gate(zero);
        b.add_output(c);
    }
    // pc = pc_inc (XOR with zero to avoid cross-wire check)
    for &w in &inp.pc_inc { let c = b.add_xor_gate(w, zero); b.add_output(c); }
    // immediate pass-through
    for &w in &inp.imm { let c = b.add_id_gate(w); b.add_output(c); }
    // addr = (r1+imm) & ~3 (zero the 2 LSBs for word alignment)
    { let c = b.add_xor_gate(zero, zero); b.add_output(c); }
    { let c = b.add_xor_gate(zero, zero); b.add_output(c); }
    for i in 2..ADDR_BITS {
        let c = b.add_id_gate(computed_addr[i]);
        b.add_output(c);
    }
    // value pass-through
    for &w in &inp.value { let c = b.add_id_gate(w); b.add_output(c); }
    // pc_inc hint pass-through
    for &w in &inp.pc_inc { let c = b.add_id_gate(w); b.add_output(c); }

    // Const outputs (appended beyond I/O layout)
    let cbits = const_bits(id, true, false, true, false);
    add_const_outputs(&mut b, zero, one, &cbits);

    (b.build().expect("circuit build failed"), cbits)
}

/// sw_waligned — word-aligned store.
///   u16addr = (r1 + imm)[0..ADDR_BITS]
///   addr    = u16addr & ~3                (word-aligned address)
///   value   = r0
///   r0, r1, r2 unchanged, pc = pc_inc
fn build_sw_waligned(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let computed_addr = ripple_carry_add(&mut b, &inp.r1[..ADDR_BITS], &inp.imm[..ADDR_BITS]);

    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);

    // r0 pass-through
    for &w in &inp.r0 { let c = b.add_id_gate(w); b.add_output(c); }
    // r1 pass-through
    for &w in &inp.r1 { let c = b.add_id_gate(w); b.add_output(c); }
    // r2 pass-through
    for &w in &inp.r2 { let c = b.add_id_gate(w); b.add_output(c); }
    // pc = pc_inc (XOR with zero to avoid cross-wire check)
    for &w in &inp.pc_inc { let c = b.add_xor_gate(w, zero); b.add_output(c); }
    // imm pass-through
    for &w in &inp.imm { let c = b.add_id_gate(w); b.add_output(c); }
    // addr = (r1+imm) & ~3 (zero the 2 LSBs)
    { let c = b.add_xor_gate(zero, zero); b.add_output(c); }
    { let c = b.add_xor_gate(zero, zero); b.add_output(c); }
    for i in 2..ADDR_BITS {
        let c = b.add_id_gate(computed_addr[i]);
        b.add_output(c);
    }
    // value = r0 (XOR with zero to avoid cross-wire)
    for i in 0..R0_BITS {
        let c = b.add_xor_gate(inp.r0[i], zero);
        b.add_output(c);
    }
    // pc_inc hint pass-through
    for &w in &inp.pc_inc { let c = b.add_id_gate(w); b.add_output(c); }

    let cbits = const_bits(id, false, true, true, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// lw_abs: r{reg} = mem[imm], addr = imm
fn build_lw_abs(id: u8, reg: usize) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);
    let regs = [&inp.r0, &inp.r1, &inp.r2];

    // r0, r1, r2: target reg = value, others pass-through
    for r in 0..3 {
        if r == reg {
            for i in 0..VALUE_BITS {
                let c = b.add_xor_gate(inp.value[i], zero);
                b.add_output(c);
            }
        } else {
            for &w in regs[r] { let c = b.add_id_gate(w); b.add_output(c); }
        }
    }
    // pc = pc_inc (XOR with zero to avoid cross-wire check)
    for &w in &inp.pc_inc { let c = b.add_xor_gate(w, zero); b.add_output(c); }
    for &w in &inp.imm { let c = b.add_id_gate(w); b.add_output(c); }
    // addr = imm[0..ADDR_BITS]
    for i in 0..ADDR_BITS {
        let c = b.add_xor_gate(inp.imm[i], zero);
        b.add_output(c);
    }
    for &w in &inp.value { let c = b.add_id_gate(w); b.add_output(c); }
    // pc_inc hint pass-through
    for &w in &inp.pc_inc { let c = b.add_id_gate(w); b.add_output(c); }

    let cbits = const_bits(id, true, false, true, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// sw_abs: mem[imm] = r{reg}, addr = imm, value = r{reg}
fn build_sw_abs(id: u8, reg: usize) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);
    let regs = [&inp.r0, &inp.r1, &inp.r2];

    // r0, r1, r2 all pass-through
    for &w in &inp.r0 { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.r1 { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.r2 { let c = b.add_id_gate(w); b.add_output(c); }
    // pc = pc_inc (XOR with zero to avoid cross-wire check)
    for &w in &inp.pc_inc { let c = b.add_xor_gate(w, zero); b.add_output(c); }
    for &w in &inp.imm { let c = b.add_id_gate(w); b.add_output(c); }
    // addr = imm[0..ADDR_BITS]
    for i in 0..ADDR_BITS {
        let c = b.add_xor_gate(inp.imm[i], zero);
        b.add_output(c);
    }
    // value = r{reg}
    for i in 0..32 {
        let c = b.add_xor_gate(regs[reg][i], zero);
        b.add_output(c);
    }
    // pc_inc hint pass-through
    for &w in &inp.pc_inc { let c = b.add_id_gate(w); b.add_output(c); }

    let cbits = const_bits(id, false, true, true, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// lw: r0 = mem[r1 + imm], addr = (r1 + imm)[0..ADDR_BITS]
fn build_lw(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let computed_addr = ripple_carry_add(&mut b, &inp.r1[..ADDR_BITS], &inp.imm[..ADDR_BITS]);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);

    // r0 = value (XOR with zero to avoid cross-wire)
    for i in 0..VALUE_BITS {
        let c = b.add_xor_gate(inp.value[i], zero);
        b.add_output(c);
    }
    // r1 pass-through
    for &w in &inp.r1 { let c = b.add_id_gate(w); b.add_output(c); }
    // r2 pass-through
    for &w in &inp.r2 { let c = b.add_id_gate(w); b.add_output(c); }
    // pc = pc_inc (XOR with zero to avoid cross-wire check)
    for &w in &inp.pc_inc { let c = b.add_xor_gate(w, zero); b.add_output(c); }
    // imm pass-through
    for &w in &inp.imm { let c = b.add_id_gate(w); b.add_output(c); }
    // addr = r1 + imm (gate outputs, no cross-wire issue)
    for &w in &computed_addr { let c = b.add_id_gate(w); b.add_output(c); }
    // value pass-through
    for &w in &inp.value { let c = b.add_id_gate(w); b.add_output(c); }
    // pc_inc hint pass-through
    for &w in &inp.pc_inc { let c = b.add_id_gate(w); b.add_output(c); }

    let cbits = const_bits(id, true, false, true, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// sw: mem[r1 + imm] = r0, addr = (r1 + imm)[0..ADDR_BITS], value = r0
fn build_sw(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let computed_addr = ripple_carry_add(&mut b, &inp.r1[..ADDR_BITS], &inp.imm[..ADDR_BITS]);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);

    // r0 pass-through
    for &w in &inp.r0 { let c = b.add_id_gate(w); b.add_output(c); }
    // r1 pass-through
    for &w in &inp.r1 { let c = b.add_id_gate(w); b.add_output(c); }
    // r2 pass-through
    for &w in &inp.r2 { let c = b.add_id_gate(w); b.add_output(c); }
    // pc = pc_inc (XOR with zero to avoid cross-wire check)
    for &w in &inp.pc_inc { let c = b.add_xor_gate(w, zero); b.add_output(c); }
    // imm pass-through
    for &w in &inp.imm { let c = b.add_id_gate(w); b.add_output(c); }
    // addr = r1 + imm (gate outputs, no cross-wire issue)
    for &w in &computed_addr { let c = b.add_id_gate(w); b.add_output(c); }
    // value = r0 (XOR with zero to avoid cross-wire)
    for i in 0..R0_BITS {
        let c = b.add_xor_gate(inp.r0[i], zero);
        b.add_output(c);
    }
    // pc_inc hint pass-through
    for &w in &inp.pc_inc { let c = b.add_id_gate(w); b.add_output(c); }

    let cbits = const_bits(id, false, true, true, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// r0 = r0 << n (constant shift left, 0 AND gates)
fn build_sll_const(id: u8, n: usize) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);
    let mut result = Vec::with_capacity(32);
    for _ in 0..n { result.push(make_const_wire(&mut b, zero, one, false)); }
    for i in 0..(32 - n) { result.push(b.add_xor_gate(inp.r0[i], zero)); }
    add_outputs(&mut b, &inp, &result);
    let cbits = const_bits(id, false, false, false, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// r0 = r0 >> n logical (constant shift right, 0 AND gates)
fn build_srl_const(id: u8, n: usize) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);
    let mut result = Vec::with_capacity(32);
    for i in n..32 { result.push(b.add_xor_gate(inp.r0[i], zero)); }
    for _ in 0..n { result.push(make_const_wire(&mut b, zero, one, false)); }
    add_outputs(&mut b, &inp, &result);
    let cbits = const_bits(id, false, false, false, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// r0 = r0 >>> n arithmetic (constant shift right, 0 AND gates)
fn build_sra_const(id: u8, n: usize) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);
    let mut result = Vec::with_capacity(32);
    for i in n..32 { result.push(b.add_xor_gate(inp.r0[i], zero)); }
    for _ in 0..n { result.push(b.add_xor_gate(inp.r0[31], zero)); } // sign fill
    add_outputs(&mut b, &inp, &result);
    let cbits = const_bits(id, false, false, false, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// sext8: r0 = sign_extend(r0[7:0])
fn build_sext8(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);

    // r0[0..7] = pass-through (same positions, id_gate OK)
    for i in 0..8 {
        let c = b.add_id_gate(inp.r0[i]);
        b.add_output(c);
    }
    // r0[8..31] = sign bit (r0[7]), 24 copies via XOR with zero
    for _ in 8..R0_BITS {
        let c = b.add_xor_gate(inp.r0[7], zero);
        b.add_output(c);
    }
    // r1 pass-through
    for &w in &inp.r1 { let c = b.add_id_gate(w); b.add_output(c); }
    // r2 pass-through
    for &w in &inp.r2 { let c = b.add_id_gate(w); b.add_output(c); }
    // pc = pc_inc (XOR with zero to avoid cross-wire check)
    for &w in &inp.pc_inc { let c = b.add_xor_gate(w, zero); b.add_output(c); }
    // hints pass-through
    for &w in &inp.imm { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.addr { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.value { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.pc_inc { let c = b.add_id_gate(w); b.add_output(c); }

    let cbits = const_bits(id, false, false, false, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// byte_sel_r2: r0 = (r0 >> (r2 * 8)) & 0xFF
/// Hint: value[0..7] repurposed as byte_val (the selected byte).
/// Outer binius context constrains byte_val == (prev_r0 >> (r2 * 8)) & 0xFF.
fn build_byte_sel_r2(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);

    // r0[0..7] = byte_val (value[0..7], XOR with zero for cross-wire)
    for i in 0..8 {
        let c = b.add_xor_gate(inp.value[i], zero);
        b.add_output(c);
    }
    // r0[8..31] = hardcoded zero
    for _ in 8..R0_BITS {
        let c = make_const_wire(&mut b, zero, one, false);
        b.add_output(c);
    }
    // r1 pass-through
    for &w in &inp.r1 { let c = b.add_id_gate(w); b.add_output(c); }
    // r2[0..1] = pass-through (byte index)
    { let c = b.add_id_gate(inp.r2[0]); b.add_output(c); }
    { let c = b.add_id_gate(inp.r2[1]); b.add_output(c); }
    // r2[2..31] = hardcoded zero
    for _ in 2..R2_BITS {
        let c = make_const_wire(&mut b, zero, one, false);
        b.add_output(c);
    }
    // pc = pc_inc (XOR with zero to avoid cross-wire check)
    for &w in &inp.pc_inc { let c = b.add_xor_gate(w, zero); b.add_output(c); }
    // hints pass-through
    for &w in &inp.imm { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.addr { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.value { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.pc_inc { let c = b.add_id_gate(w); b.add_output(c); }

    let cbits = const_bits(id, false, false, false, true);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// byte_ins_r2: read byte from mem[SCRATCH_A], insert into r0 at byte position r2.
///   byte = value[0..7] (from mem[0x4184])
///   r0[byte_pos*8 .. byte_pos*8+7] = byte, rest of r0 unchanged.
/// 36 AND gates: 4 (decode r2[0:1]) + 32 (byte MUX).
fn build_byte_ins_r2(id: u8) -> (Circuit, Vec<bool>) {
    const SCRATCH_A: u32 = 0x4184;

    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);

    // Decode r2[0:1] into 4 one-hot selects (4 AND)
    let a0 = inp.r2[0];
    let a1 = inp.r2[1];
    let na0 = b.add_inv_gate(a0);
    let na1 = b.add_inv_gate(a1);
    let sel0 = b.add_and_gate(na1, na0); // r2 == 0
    let sel1 = b.add_and_gate(na1, a0);  // r2 == 1
    let sel2 = b.add_and_gate(a1, na0);  // r2 == 2
    let sel3 = b.add_and_gate(a1, a0);   // r2 == 3
    let sels = [sel0, sel1, sel2, sel3];

    // For each byte of r0, MUX(sel_k, r0[i], byte[i%8])
    // MUX(sel, orig, repl) = orig XOR (sel AND (orig XOR repl))
    for byte_idx in 0..4 {
        let sel = sels[byte_idx];
        for bit in 0..8 {
            let i = byte_idx * 8 + bit;
            let diff = b.add_xor_gate(inp.r0[i], inp.value[bit]);
            let masked = b.add_and_gate(sel, diff);
            let out = b.add_xor_gate(inp.r0[i], masked);
            b.add_output(out);
        }
    }
    // r1 pass-through
    for &w in &inp.r1 { let c = b.add_id_gate(w); b.add_output(c); }
    // r2 pass-through
    for &w in &inp.r2 { let c = b.add_id_gate(w); b.add_output(c); }
    // pc = pc_inc (XOR with zero to avoid cross-wire check)
    for &w in &inp.pc_inc { let c = b.add_xor_gate(w, zero); b.add_output(c); }
    // imm pass-through
    for &w in &inp.imm { let c = b.add_id_gate(w); b.add_output(c); }
    // addr = SCRATCH_A (hardcoded)
    for bit in 0..ADDR_BITS {
        let val = (SCRATCH_A >> bit) & 1 == 1;
        let w = make_const_wire(&mut b, zero, one, val);
        b.add_output(w);
    }
    // value pass-through
    for &w in &inp.value { let c = b.add_id_gate(w); b.add_output(c); }
    // pc_inc hint pass-through
    for &w in &inp.pc_inc { let c = b.add_id_gate(w); b.add_output(c); }

    // is_mem_read=true, is_mem_write=false, has_immediate=false, is_byte_sel_r2=false
    let cbits = const_bits(id, true, false, false, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// addi: r0 = r1 + imm
fn build_addi(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);
    let sum = ripple_carry_add(&mut b, &inp.r1, &inp.imm);
    add_outputs(&mut b, &inp, &sum);
    let cbits = const_bits(id, false, false, true, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// xori: r0 = r1 ^ imm (0 AND gates)
fn build_xori(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);
    let result: Vec<W> = (0..32).map(|i| b.add_xor_gate(inp.r1[i], inp.imm[i])).collect();
    add_outputs(&mut b, &inp, &result);
    let cbits = const_bits(id, false, false, true, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// lui: r0 = imm << 12 (0 AND gates)
/// Bottom 12 bits = 0, top 20 bits = imm[0..19].
fn build_lui(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);
    // r0[0..11] = hardcoded zero
    let mut result: Vec<W> = (0..12).map(|_| make_const_wire(&mut b, zero, one, false)).collect();
    // r0[12..31] = imm[0..19] (XOR with zero for cross-wire avoidance)
    for i in 0..20 {
        result.push(b.add_xor_gate(inp.imm[i], zero));
    }
    add_outputs(&mut b, &inp, &result);
    let cbits = const_bits(id, false, false, true, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// blt: if (r0 < r1) pc = imm[0..15], else pc = pc (signed comparison)
/// Hints: imm = branch target address
/// 80 AND gates: 64 (signed comparison) + 16 (pc MUX)
fn build_blt(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);

    // Signed comparison: cmp = (r0 < r1) ? 1 : 0
    let (_diff, underflow) = ops::wrapping_sub(&mut b, &inp.r0, &inp.r1);
    let sign_r0 = inp.r0[31];
    let signs_differ = b.add_xor_gate(sign_r0, inp.r1[31]);
    let diff_bit = b.add_xor_gate(underflow, sign_r0);
    let sel = b.add_and_gate(signs_differ, diff_bit);
    let cmp = b.add_xor_gate(underflow, sel); // cmp = 1 if r0 < r1

    // r0 pass-through
    for &w in &inp.r0 { let c = b.add_id_gate(w); b.add_output(c); }
    // r1 pass-through
    for &w in &inp.r1 { let c = b.add_id_gate(w); b.add_output(c); }
    // r2 pass-through
    for &w in &inp.r2 { let c = b.add_id_gate(w); b.add_output(c); }
    // pc = MUX(cmp, pc_inc, imm[0..15]): cmp=0 → pc_inc, cmp=1 → imm
    for i in 0..PC_BITS {
        let d = b.add_xor_gate(inp.pc_inc[i], inp.imm[i]);
        let m = b.add_and_gate(cmp, d);
        let out = b.add_xor_gate(inp.pc_inc[i], m);
        b.add_output(out);
    }
    // hints pass-through
    for &w in &inp.imm { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.addr { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.value { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.pc_inc { let c = b.add_id_gate(w); b.add_output(c); }

    let cbits = const_bits(id, false, false, true, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// bge: if (r0 >= r1) pc = imm[0..15], else pc = pc (signed comparison)
/// Mirror of blt: sel = NOT(r0 < r1)
/// 80 AND gates: 64 (signed comparison) + 16 (pc MUX)
fn build_bge(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);

    // Signed comparison: lt = (r0 < r1) ? 1 : 0
    let (_diff, underflow) = ops::wrapping_sub(&mut b, &inp.r0, &inp.r1);
    let sign_r0 = inp.r0[31];
    let signs_differ = b.add_xor_gate(sign_r0, inp.r1[31]);
    let diff_bit = b.add_xor_gate(underflow, sign_r0);
    let sel = b.add_and_gate(signs_differ, diff_bit);
    let lt = b.add_xor_gate(underflow, sel); // lt = 1 if r0 < r1
    let ge = b.add_inv_gate(lt);             // ge = 1 if r0 >= r1

    // r0, r1, r2 pass-through
    for &w in &inp.r0 { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.r1 { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.r2 { let c = b.add_id_gate(w); b.add_output(c); }
    // pc = MUX(ge, pc_inc, imm[0..15]): ge=0 → pc_inc, ge=1 → imm
    for i in 0..PC_BITS {
        let d = b.add_xor_gate(inp.pc_inc[i], inp.imm[i]);
        let m = b.add_and_gate(ge, d);
        let out = b.add_xor_gate(inp.pc_inc[i], m);
        b.add_output(out);
    }
    // hints pass-through
    for &w in &inp.imm { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.addr { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.value { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.pc_inc { let c = b.add_id_gate(w); b.add_output(c); }

    let cbits = const_bits(id, false, false, true, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// bltu: if (r0 < r1) unsigned, pc = imm[0..15], else pc = pc
/// 79 AND gates: 63 (unsigned comparison) + 16 (pc MUX)
fn build_bltu(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);

    // Unsigned comparison: underflow = 1 if r0 < r1
    let (_diff, underflow) = ops::wrapping_sub(&mut b, &inp.r0, &inp.r1);
    let cmp = underflow;

    // r0, r1, r2 pass-through
    for &w in &inp.r0 { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.r1 { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.r2 { let c = b.add_id_gate(w); b.add_output(c); }
    // pc = MUX(cmp, pc_inc, imm[0..15])
    for i in 0..PC_BITS {
        let d = b.add_xor_gate(inp.pc_inc[i], inp.imm[i]);
        let m = b.add_and_gate(cmp, d);
        let out = b.add_xor_gate(inp.pc_inc[i], m);
        b.add_output(out);
    }
    // hints pass-through
    for &w in &inp.imm { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.addr { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.value { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.pc_inc { let c = b.add_id_gate(w); b.add_output(c); }

    let cbits = const_bits(id, false, false, true, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// bgeu: if (r0 >= r1) unsigned, pc = imm[0..15], else pc = pc
/// Mirror of bltu: ge = NOT(lt). 79 AND gates.
fn build_bgeu(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);

    // Unsigned comparison: underflow = 1 if r0 < r1
    let (_diff, underflow) = ops::wrapping_sub(&mut b, &inp.r0, &inp.r1);
    let lt = underflow;
    let ge = b.add_inv_gate(lt); // ge = 1 if r0 >= r1 unsigned

    // r0, r1, r2 pass-through
    for &w in &inp.r0 { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.r1 { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.r2 { let c = b.add_id_gate(w); b.add_output(c); }
    // pc = MUX(ge, pc_inc, imm[0..15])
    for i in 0..PC_BITS {
        let d = b.add_xor_gate(inp.pc_inc[i], inp.imm[i]);
        let m = b.add_and_gate(ge, d);
        let out = b.add_xor_gate(inp.pc_inc[i], m);
        b.add_output(out);
    }
    // hints pass-through
    for &w in &inp.imm { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.addr { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.value { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.pc_inc { let c = b.add_id_gate(w); b.add_output(c); }

    let cbits = const_bits(id, false, false, true, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// bne: if (r0 != r1) pc = imm[0..15], else pc = pc
/// 47 AND gates: 31 (OR-tree equality check) + 16 (pc MUX)
fn build_bne(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);

    // XOR each bit pair: diff[i] = 1 if bits differ
    let diff: Vec<W> = (0..32).map(|i| b.add_xor_gate(inp.r0[i], inp.r1[i])).collect();

    // OR-tree reduce 32 bits → 1 bit (31 AND gates)
    // OR(a,b) = (a ^ b) ^ (a & b)
    let mut level = diff;
    while level.len() > 1 {
        let mut next = Vec::with_capacity((level.len() + 1) / 2);
        for pair in level.chunks(2) {
            if pair.len() == 2 {
                let x = b.add_xor_gate(pair[0], pair[1]);
                let a = b.add_and_gate(pair[0], pair[1]);
                next.push(b.add_xor_gate(x, a));
            } else {
                next.push(pair[0]);
            }
        }
        level = next;
    }
    let ne = level[0]; // 1 if r0 != r1

    // r0, r1, r2 pass-through
    for &w in &inp.r0 { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.r1 { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.r2 { let c = b.add_id_gate(w); b.add_output(c); }
    // pc = MUX(ne, pc_inc, imm[0..15]): ne=0 → pc_inc, ne=1 → imm
    for i in 0..PC_BITS {
        let d = b.add_xor_gate(inp.pc_inc[i], inp.imm[i]);
        let m = b.add_and_gate(ne, d);
        let out = b.add_xor_gate(inp.pc_inc[i], m);
        b.add_output(out);
    }
    // hints pass-through
    for &w in &inp.imm { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.addr { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.value { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.pc_inc { let c = b.add_id_gate(w); b.add_output(c); }

    let cbits = const_bits(id, false, false, true, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// beq: if (r0 == r1) pc = imm[0..15], else pc = pc
/// Mirror of bne: eq = INV(ne). 47 AND gates.
fn build_beq(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);

    // XOR each bit pair
    let diff: Vec<W> = (0..32).map(|i| b.add_xor_gate(inp.r0[i], inp.r1[i])).collect();

    // OR-tree reduce 32 bits → 1 bit (31 AND)
    let mut level = diff;
    while level.len() > 1 {
        let mut next = Vec::with_capacity((level.len() + 1) / 2);
        for pair in level.chunks(2) {
            if pair.len() == 2 {
                let x = b.add_xor_gate(pair[0], pair[1]);
                let a = b.add_and_gate(pair[0], pair[1]);
                next.push(b.add_xor_gate(x, a));
            } else {
                next.push(pair[0]);
            }
        }
        level = next;
    }
    let ne = level[0];
    let eq = b.add_inv_gate(ne); // 1 if r0 == r1

    // r0, r1, r2 pass-through
    for &w in &inp.r0 { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.r1 { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.r2 { let c = b.add_id_gate(w); b.add_output(c); }
    // pc = MUX(eq, pc_inc, imm[0..15])
    for i in 0..PC_BITS {
        let d = b.add_xor_gate(inp.pc_inc[i], inp.imm[i]);
        let m = b.add_and_gate(eq, d);
        let out = b.add_xor_gate(inp.pc_inc[i], m);
        b.add_output(out);
    }
    // hints pass-through
    for &w in &inp.imm { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.addr { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.value { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.pc_inc { let c = b.add_id_gate(w); b.add_output(c); }

    let cbits = const_bits(id, false, false, true, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// jal: if imm[0..15] == pc then halt (pc=0xFFFF), else pc = imm[0..15]
/// 31 AND gates: 15 (16-bit equality OR-tree) + 16 (pc MUX)
fn build_jal(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);

    // 16-bit equality: imm[0..15] == pc?
    let diff: Vec<W> = (0..PC_BITS).map(|i| b.add_xor_gate(inp.imm[i], inp.pc[i])).collect();

    // OR-tree reduce 16 bits → 1 bit (15 AND)
    let mut level = diff;
    while level.len() > 1 {
        let mut next = Vec::with_capacity((level.len() + 1) / 2);
        for pair in level.chunks(2) {
            if pair.len() == 2 {
                let x = b.add_xor_gate(pair[0], pair[1]);
                let a = b.add_and_gate(pair[0], pair[1]);
                next.push(b.add_xor_gate(x, a));
            } else {
                next.push(pair[0]);
            }
        }
        level = next;
    }
    let ne = level[0];
    let eq = b.add_inv_gate(ne); // eq = 1 → halt

    // r0, r1, r2 pass-through
    for &w in &inp.r0 { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.r1 { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.r2 { let c = b.add_id_gate(w); b.add_output(c); }
    // pc = MUX(eq, imm[0..15], 0xFFFF): eq=0 → imm (jump), eq=1 → all 1s (halt)
    // out[i] = imm[i] XOR (eq AND INV(imm[i]))  — when eq=1, forces output to 1
    for i in 0..PC_BITS {
        let inv_imm = b.add_inv_gate(inp.imm[i]);
        let masked = b.add_and_gate(eq, inv_imm);
        let out = b.add_xor_gate(inp.imm[i], masked);
        b.add_output(out);
    }
    // hints pass-through
    for &w in &inp.imm { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.addr { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.value { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.pc_inc { let c = b.add_id_gate(w); b.add_output(c); }

    let cbits = const_bits(id, false, false, true, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// ret: if r0 == 0 then halt (pc=0xFFFF), else pc = r0[0..15]
/// 47 AND gates: 31 (32-bit OR-tree r0==0) + 16 (pc MUX)
fn build_ret(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);

    // OR-tree of all 32 r0 bits → ne (r0 != 0)
    let mut level: Vec<W> = inp.r0.clone();
    while level.len() > 1 {
        let mut next = Vec::with_capacity((level.len() + 1) / 2);
        for pair in level.chunks(2) {
            if pair.len() == 2 {
                let x = b.add_xor_gate(pair[0], pair[1]);
                let a = b.add_and_gate(pair[0], pair[1]);
                next.push(b.add_xor_gate(x, a));
            } else {
                next.push(pair[0]);
            }
        }
        level = next;
    }
    let ne = level[0];          // 1 if r0 != 0 (return)
    let eq = b.add_inv_gate(ne); // 1 if r0 == 0 (halt)

    // r0, r1, r2 pass-through
    for &w in &inp.r0 { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.r1 { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.r2 { let c = b.add_id_gate(w); b.add_output(c); }
    // pc = MUX(eq, r0[0..15], 0xFFFF): eq=0 → r0 (return), eq=1 → all 1s (halt)
    for i in 0..PC_BITS {
        let inv_r0 = b.add_inv_gate(inp.r0[i]);
        let masked = b.add_and_gate(eq, inv_r0);
        let out = b.add_xor_gate(inp.r0[i], masked);
        b.add_output(out);
    }
    // hints pass-through
    for &w in &inp.imm { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.addr { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.value { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.pc_inc { let c = b.add_id_gate(w); b.add_output(c); }

    let cbits = const_bits(id, false, false, false, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// jal_call — direct function call.
///   addr  = MAILBOX_ADDR (0x4104, hardcoded)
///   value = pc_inc (return address = pc+1, zero-extended to 32b)
///   pc    = imm[0..15] (jump target)
///   r0, r1, r2 unchanged
/// 0 AND gates — pure wiring.
fn build_jal_call(id: u8) -> (Circuit, Vec<bool>) {
    const MAILBOX_ADDR: u32 = 0x4104;

    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);

    // r0, r1, r2 pass-through
    for &w in &inp.r0 { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.r1 { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.r2 { let c = b.add_id_gate(w); b.add_output(c); }

    // pc = imm[0..15] (XOR with zero to avoid cross-wire)
    for i in 0..PC_BITS {
        let c = b.add_xor_gate(inp.imm[i], zero);
        b.add_output(c);
    }

    // imm pass-through
    for &w in &inp.imm { let c = b.add_id_gate(w); b.add_output(c); }

    // addr = MAILBOX_ADDR (hardcoded constant)
    for bit in 0..ADDR_BITS {
        let val = (MAILBOX_ADDR >> bit) & 1 == 1;
        let w = make_const_wire(&mut b, zero, one, val);
        b.add_output(w);
    }

    // value = pc_inc (return address = pc+1) zero-extended to 32 bits
    for i in 0..PC_BITS {
        let c = b.add_xor_gate(inp.pc_inc[i], zero);
        b.add_output(c);
    }
    for _ in PC_BITS..VALUE_BITS {
        let z = b.add_id_gate(zero);
        b.add_output(z);
    }
    // pc_inc hint pass-through
    for &w in &inp.pc_inc { let c = b.add_id_gate(w); b.add_output(c); }

    // is_mem_read=false, is_mem_write=true, has_immediate=true, is_byte_sel_r2=false
    let cbits = const_bits(id, false, true, true, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// jalr_call: indirect function call — save return address to mailbox, jump to r0
///   addr = MAILBOX_ADDR (hardcoded), value = pc (zero-extended to 32b), pc = r0[0..15]
/// 0 AND gates — pure wiring.
fn build_jalr_call(id: u8) -> (Circuit, Vec<bool>) {
    const MAILBOX_ADDR: u32 = 0x4104;

    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);

    // r0, r1, r2 pass-through
    for &w in &inp.r0 { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.r1 { let c = b.add_id_gate(w); b.add_output(c); }
    for &w in &inp.r2 { let c = b.add_id_gate(w); b.add_output(c); }

    // pc = r0[0..15] (XOR with zero to avoid cross-wire)
    for i in 0..PC_BITS {
        let c = b.add_xor_gate(inp.r0[i], zero);
        b.add_output(c);
    }

    // imm pass-through
    for &w in &inp.imm { let c = b.add_id_gate(w); b.add_output(c); }

    // addr = MAILBOX_ADDR (hardcoded constant)
    for bit in 0..ADDR_BITS {
        let val = (MAILBOX_ADDR >> bit) & 1 == 1;
        let w = make_const_wire(&mut b, zero, one, val);
        b.add_output(w);
    }

    // value = pc zero-extended to 32 bits (XOR with zero to avoid cross-wire)
    for i in 0..PC_BITS {
        let c = b.add_xor_gate(inp.pc[i], zero);
        b.add_output(c);
    }
    for _ in PC_BITS..VALUE_BITS {
        let z = b.add_id_gate(zero);
        b.add_output(z);
    }
    // pc_inc hint pass-through
    for &w in &inp.pc_inc { let c = b.add_id_gate(w); b.add_output(c); }

    // is_mem_read=false, is_mem_write=true, has_immediate=false, is_byte_sel_r2=false
    let cbits = const_bits(id, false, true, false, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

/// padding: all pass-through (0 AND gates)
/// Used to fill remaining steps after halt. Outer context checks pc=0xFFFF.
fn build_padding(id: u8) -> (Circuit, Vec<bool>) {
    let mut b = CircuitBuilder::new();
    let inp = add_inputs(&mut b);
    let zero = b.add_xor_gate(inp.r1[0], inp.r1[0]);
    let one = b.add_inv_gate(zero);
    // r0 pass-through (use as "result" for add_outputs)
    let r0_copy: Vec<W> = inp.r0.iter().map(|&w| b.add_id_gate(w)).collect();
    add_outputs(&mut b, &inp, &r0_copy);
    let cbits = const_bits(id, false, false, false, false);
    add_const_outputs(&mut b, zero, one, &cbits);
    (b.build().expect("circuit build failed"), cbits)
}

// ── Translation + export (unchanged logic) ───────────────────────────

const GATE_AND: u8 = 0;
const GATE_XOR: u8 = 1;
const GATE_INV: u8 = 2;

struct TranslatedCircuit {
    input_count: usize,
    connect_count: usize, // first connect_count positions are connected across steps
    gate_count: usize,
    and_count: usize,
    xor_count: usize,
    inv_count: usize,
    gates: Vec<(u8, u32, u32)>,
    output_source: Vec<i32>,
    const_wires: Vec<(u32, bool)>, // (gate_index, value) — topology constants
}

fn translate(circuit: &Circuit, const_values: &[bool]) -> TranslatedCircuit {
    let input_count = circuit.inputs().len();
    let input_range = circuit.inputs();

    let mut wire_map: HashMap<usize, u32> = HashMap::new();
    for (i, w) in input_range.clone().enumerate() {
        wire_map.insert(w, i as u32);
    }

    let mut id_trace: HashMap<usize, usize> = HashMap::new();
    for gate in circuit.gates() {
        if let Gate::Id { x, z } = gate {
            id_trace.insert(z.id(), x.id());
        }
    }

    let mut gate_idx: u32 = 0;
    let mut and_count = 0usize;
    let mut xor_count = 0usize;
    let mut inv_count = 0usize;
    let mut gates = Vec::new();

    for gate in circuit.gates() {
        match gate {
            Gate::And { x, y, z } => {
                wire_map.insert(z.id(), input_count as u32 + gate_idx);
                gates.push((GATE_AND, x.id(), y.id()));
                gate_idx += 1;
                and_count += 1;
            }
            Gate::Xor { x, y, z } => {
                wire_map.insert(z.id(), input_count as u32 + gate_idx);
                gates.push((GATE_XOR, x.id(), y.id()));
                gate_idx += 1;
                xor_count += 1;
            }
            Gate::Inv { x, z } => {
                wire_map.insert(z.id(), input_count as u32 + gate_idx);
                gates.push((GATE_INV, x.id(), 0));
                gate_idx += 1;
                inv_count += 1;
            }
            Gate::Id { .. } => {}
        }
    }

    let translated_gates: Vec<(u8, u32, u32)> = gates.iter().map(|&(gtype, lx, ly)| {
        let left = *wire_map.get(&lx).unwrap_or_else(|| panic!("unmapped wire {}", lx));
        let right = if gtype == GATE_INV { 0 } else {
            *wire_map.get(&ly).unwrap_or_else(|| panic!("unmapped wire {}", ly))
        };
        (gtype, left, right)
    }).collect();

    let mut output_source = vec![-1i32; input_count];
    let mut const_wires = Vec::new();

    for (i, out_wire) in circuit.outputs().enumerate() {
        let mut w = out_wire;
        while let Some(&src) = id_trace.get(&w) { w = src; }

        if i < input_count {
            // Normal I/O output
            if let Some(&mapped) = wire_map.get(&w) {
                if mapped < input_count as u32 {
                    if mapped == i as u32 {
                        output_source[i] = -1;
                    } else {
                        panic!("cross-wired output: out[{}] = input[{}]", i, mapped);
                    }
                } else {
                    output_source[i] = (mapped - input_count as u32) as i32;
                }
            } else {
                panic!("unmapped output wire {}", w);
            }
        } else {
            // Const wire output (beyond I/O layout)
            let mapped = *wire_map.get(&w).unwrap_or_else(|| panic!("unmapped const wire {}", w));
            assert!(mapped >= input_count as u32, "const wire mapped to input");
            let bit_idx = i - input_count;
            const_wires.push((mapped - input_count as u32, const_values[bit_idx]));
        }
    }

    TranslatedCircuit {
        input_count, connect_count: CPU_STATE_BITS,
        gate_count: gate_idx as usize,
        and_count, xor_count, inv_count,
        gates: translated_gates, output_source, const_wires,
    }
}

fn export_binary(tc: &TranslatedCircuit, path: &str) {
    let mut f = std::fs::File::create(path).expect("create file");
    f.write_all(&(tc.input_count as u32).to_le_bytes()).unwrap();
    f.write_all(&(tc.and_count as u32).to_le_bytes()).unwrap();
    f.write_all(&(tc.xor_count as u32).to_le_bytes()).unwrap();
    f.write_all(&(tc.inv_count as u32).to_le_bytes()).unwrap();
    f.write_all(&(tc.gate_count as u32).to_le_bytes()).unwrap();
    for &(gtype, left, right) in &tc.gates {
        f.write_all(&[gtype]).unwrap();
        f.write_all(&left.to_le_bytes()).unwrap();
        f.write_all(&right.to_le_bytes()).unwrap();
    }
    for &src in &tc.output_source {
        f.write_all(&src.to_le_bytes()).unwrap();
    }
    // Const table: num_consts, then (gate_index: u32, value: u8) per entry
    f.write_all(&(tc.const_wires.len() as u32).to_le_bytes()).unwrap();
    for &(gate_idx, val) in &tc.const_wires {
        f.write_all(&gate_idx.to_le_bytes()).unwrap();
        f.write_all(&[val as u8]).unwrap();
    }
    // Connect count: first connect_count I/O positions are connected across steps
    f.write_all(&(tc.connect_count as u32).to_le_bytes()).unwrap();
}

// ── Normalization ────────────────────────────────────────────────────

/// Evaluate a translated circuit in cleartext.
fn evaluate_translated(tc: &TranslatedCircuit, inputs: &[bool]) -> Vec<bool> {
    let one_idx = tc.input_count + tc.gate_count;
    let mut w = vec![false; one_idx + 1];
    for i in 0..tc.input_count { w[i] = inputs[i]; }
    w[one_idx] = true; // virtual constant-1 wire

    for (g, &(gt, left, right)) in tc.gates.iter().enumerate() {
        let l = w[left as usize];
        let r = w[right as usize];
        w[tc.input_count + g] = match gt {
            GATE_AND => l & r,
            GATE_XOR => l ^ r,
            GATE_INV => !l,
            _ => panic!("unknown gate type {}", gt),
        };
    }

    (0..tc.input_count).map(|i| {
        if tc.output_source[i] < 0 { w[i] }
        else { w[tc.input_count + tc.output_source[i] as usize] }
    }).collect()
}

/// Normalize translated circuits to a universal gate-type layout.
///
// ── Test + export harness ────────────────────────────────────────────

fn bits_to_u32(output: &[bool], offset: usize, count: usize) -> u32 {
    (0..count).map(|i| if output[offset + i] { 1u32 << i } else { 0 }).sum()
}

fn bits_to_u16(output: &[bool], offset: usize, count: usize) -> u16 {
    (0..count).map(|i| if output[offset + i] { 1u16 << i } else { 0 }).sum()
}

fn set_u32(bits: &mut [bool], offset: usize, val: u32) {
    for i in 0..32 { bits[offset + i] = (val >> i) & 1 == 1; }
}

fn set_u16(bits: &mut [bool], offset: usize, val: u16) {
    for i in 0..16 { bits[offset + i] = (val >> i) & 1 == 1; }
}

#[derive(Clone)]
struct TestCase {
    r0: u32, r1: u32, r2: u32, pc: u16,
    immediate: u32, addr: u32, value: u32, pc_inc: u16,
    expect_r0: u32, expect_r1: u32, expect_r2: u32, expect_pc: u16,
    expect_addr: u32, expect_value: u32,
}

/// Verify const wires produce correct values in a translated circuit.
fn verify_const_wires(name: &str, tc: &TranslatedCircuit, inputs: &[bool]) {
    let one_idx = tc.input_count + tc.gate_count;
    let mut w = vec![false; one_idx + 1];
    for i in 0..tc.input_count { w[i] = inputs[i]; }
    w[one_idx] = true;

    for (g, &(gt, left, right)) in tc.gates.iter().enumerate() {
        let l = w[left as usize];
        let r = w[right as usize];
        w[tc.input_count + g] = match gt {
            GATE_AND => l & r,
            GATE_XOR => l ^ r,
            GATE_INV => !l,
            _ => panic!("unknown gate type {}", gt),
        };
    }

    for &(gi, expected) in &tc.const_wires {
        let actual = w[tc.input_count + gi as usize];
        assert_eq!(actual, expected,
            "{}: const wire at gate {}: got {} expected {}", name, gi, actual, expected);
    }
}

/// Build, evaluate with mpz, and translate a circuit.
fn test_and_translate(
    name: &str, circuit: &Circuit, const_values: &[bool], test: &TestCase,
) -> (TranslatedCircuit, Vec<bool>) {
    println!("=== {} ===", name);

    let mut input_bits = vec![false; INPUT_COUNT];
    set_u32(&mut input_bits, R0_OFF, test.r0);
    set_u32(&mut input_bits, R1_OFF, test.r1);
    set_u32(&mut input_bits, R2_OFF, test.r2);
    set_u16(&mut input_bits, PC_OFF, test.pc);
    set_u32(&mut input_bits, IMM_OFF, test.immediate);
    for i in 0..ADDR_BITS { input_bits[ADDR_OFF + i] = (test.addr >> i) & 1 == 1; }
    set_u32(&mut input_bits, VALUE_OFF, test.value);
    set_u16(&mut input_bits, PC_INC_OFF, test.pc_inc);

    let output = circuit.evaluate(input_bits.clone()).expect("evaluate failed");
    // First INPUT_COUNT outputs are I/O layout
    let r0_out = bits_to_u32(&output, R0_OFF, R0_BITS);
    let r1_out = bits_to_u32(&output, R1_OFF, R1_BITS);
    let r2_out = bits_to_u32(&output, R2_OFF, R2_BITS);
    let pc_out = bits_to_u16(&output, PC_OFF, PC_BITS);
    let addr_out = bits_to_u32(&output, ADDR_OFF, ADDR_BITS);
    let value_out = bits_to_u32(&output, VALUE_OFF, VALUE_BITS);

    assert_eq!(r0_out, test.expect_r0, "r0: got {} expected {}", r0_out, test.expect_r0);
    assert_eq!(r1_out, test.expect_r1, "r1: got {} expected {}", r1_out, test.expect_r1);
    assert_eq!(r2_out, test.expect_r2, "r2: got {} expected {}", r2_out, test.expect_r2);
    assert_eq!(pc_out, test.expect_pc, "pc: got {} expected {}", pc_out, test.expect_pc);
    assert_eq!(addr_out, test.expect_addr, "addr: got {} expected {}", addr_out, test.expect_addr);
    assert_eq!(value_out, test.expect_value, "value: got {} expected {}", value_out, test.expect_value);

    // Outputs beyond INPUT_COUNT are const wire values
    for (i, &expected) in const_values.iter().enumerate() {
        let actual = output[INPUT_COUNT + i];
        assert_eq!(actual, expected, "const bit {}: got {} expected {}", i, actual, expected);
    }

    println!("  mpz eval: r0={} r1={} r2={} pc={} addr={} value={} consts={} OK",
             r0_out, r1_out, r2_out, pc_out, addr_out, value_out, const_values.len());

    let tc = translate(circuit, const_values);
    println!("  translated: {} gates ({} AND + {} XOR + {} INV), {} const wires",
             tc.gate_count, tc.and_count, tc.xor_count, tc.inv_count, tc.const_wires.len());

    (tc, input_bits)
}

fn build_and_register(
    name: &str, circuit: &Circuit, const_values: &[bool], test: &TestCase,
    names: &mut Vec<String>, translations: &mut Vec<TranslatedCircuit>,
    test_inputs: &mut Vec<Vec<bool>>, test_expects: &mut Vec<TestCase>,
) {
    let (tc, inp) = test_and_translate(name, circuit, const_values, test);
    names.push(name.to_string());
    translations.push(tc);
    test_inputs.push(inp);
    test_expects.push(test.clone());
}

fn main() {
    let mut names: Vec<String> = Vec::new();
    let mut translations: Vec<TranslatedCircuit> = Vec::new();
    let mut test_inputs: Vec<Vec<bool>> = Vec::new();
    let mut test_expects: Vec<TestCase> = Vec::new();

    // ADD: r0 = r1 + r2 (id=0)
    let (circ, cbits) = build_add(0);
    build_and_register("add", &circ, &cbits, &TestCase {
        r0: 0, r1: 3, r2: 5, pc: 10,
        immediate: 0, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 8, expect_r1: 3, expect_r2: 5, expect_pc: 11,
        expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // SUB: r0 = r1 - r2 (id=43)
    let (circ, cbits) = build_sub(43);
    build_and_register("sub", &circ, &cbits, &TestCase {
        r0: 0, r1: 10, r2: 3, pc: 10,
        immediate: 0, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 7, expect_r1: 10, expect_r2: 3, expect_pc: 11,
        expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // AND: r0 = r1 & r2 (id=44)
    let (circ, cbits) = build_and(44);
    build_and_register("and", &circ, &cbits, &TestCase {
        r0: 0, r1: 0xFF00FF00, r2: 0x0F0F0F0F, pc: 10,
        immediate: 0, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 0x0F000F00, expect_r1: 0xFF00FF00, expect_r2: 0x0F0F0F0F, expect_pc: 11,
        expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // ANDI: r0 = r1 & imm (id=46)
    let (circ, cbits) = build_andi(46);
    build_and_register("andi", &circ, &cbits, &TestCase {
        r0: 0, r1: 0xDEADBEEF, r2: 0, pc: 10,
        immediate: 0xFF, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 0xEF, expect_r1: 0xDEADBEEF, expect_r2: 0, expect_pc: 11,
        expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // OR: r0 = r1 | r2 (id=21)
    let (circ, cbits) = build_or(21);
    build_and_register("or", &circ, &cbits, &TestCase {
        r0: 0, r1: 0xF0F0F0F0, r2: 0x0F0F0F0F, pc: 10,
        immediate: 0, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 0xFFFFFFFF, expect_r1: 0xF0F0F0F0, expect_r2: 0x0F0F0F0F, expect_pc: 11,
        expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // XOR: r0 = r1 ^ r2 (id=23)
    let (circ, cbits) = build_xor(23);
    build_and_register("xor", &circ, &cbits, &TestCase {
        r0: 0, r1: 0xFF00FF00, r2: 0x0F0F0F0F, pc: 10,
        immediate: 0, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 0xF00FF00F, expect_r1: 0xFF00FF00, expect_r2: 0x0F0F0F0F, expect_pc: 11,
        expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // SLT: r0 = (r1 < r2) ? 1 : 0 (id=1)
    let (circ, cbits) = build_slt(1);
    build_and_register("slt", &circ, &cbits, &TestCase {
        r0: 0, r1: 0xFFFFFFFA, r2: 3, pc: 10,
        immediate: 0, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 1, expect_r1: 0xFFFFFFFA, expect_r2: 3, expect_pc: 11,
        expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // SLTU: r0 = (r1 < r2) ? 1 : 0, unsigned (id=22)
    // 0xFFFFFFFA = 4294967290 unsigned, which is > 3 unsigned → result = 0
    let (circ, cbits) = build_sltu(22);
    build_and_register("sltu", &circ, &cbits, &TestCase {
        r0: 0, r1: 0xFFFFFFFA, r2: 3, pc: 10,
        immediate: 0, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 0, expect_r1: 0xFFFFFFFA, expect_r2: 3, expect_pc: 11,
        expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // SLTI: r0 = (r1 < imm) ? 1 : 0, signed (id=24)
    // -6 (signed) < 10 → result = 1
    let (circ, cbits) = build_slti(24);
    build_and_register("slti", &circ, &cbits, &TestCase {
        r0: 0, r1: 0xFFFFFFFA, r2: 0, pc: 10,
        immediate: 10, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 1, expect_r1: 0xFFFFFFFA, expect_r2: 0, expect_pc: 11,
        expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // Variable-amount barrel shifters (sll, srl, sra) removed — our custom ISA
    // replaced them with constant-amount shifts (sll1/4/8/16/31, etc.).

    // LW_ALIGNED: addr = r1 + imm, r0 = value & ~3, r2 = addr & 3 (id=5)
    let (circ, cbits) = build_lw_aligned(5);
    build_and_register("lw_aligned", &circ, &cbits, &TestCase {
        r0: 0, r1: 100, r2: 0, pc: 10,
        immediate: 7, addr: 0, value: 0xDEADBEEF, pc_inc: 11,
        expect_r0: 0xDEADBEEF, expect_r1: 100, expect_r2: 3, expect_pc: 11,
        expect_addr: 104, expect_value: 0xDEADBEEF,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);
    // Extra lw_aligned tests: cover r2 = 0, 1, 2 (existing test has r2 = 3)
    test_and_translate("lw_aligned (r2=0)", &circ, &cbits, &TestCase {
        r0: 0, r1: 100, r2: 0, pc: 10,
        immediate: 4, addr: 0, value: 0xDEADBEEF, pc_inc: 11,
        expect_r0: 0xDEADBEEF, expect_r1: 100, expect_r2: 0, expect_pc: 11,
        expect_addr: 104, expect_value: 0xDEADBEEF,
    });
    test_and_translate("lw_aligned (r2=1)", &circ, &cbits, &TestCase {
        r0: 0, r1: 100, r2: 0, pc: 10,
        immediate: 5, addr: 0, value: 0xDEADBEEF, pc_inc: 11,
        expect_r0: 0xDEADBEEF, expect_r1: 100, expect_r2: 1, expect_pc: 11,
        expect_addr: 104, expect_value: 0xDEADBEEF,
    });
    test_and_translate("lw_aligned (r2=2)", &circ, &cbits, &TestCase {
        r0: 0, r1: 100, r2: 0, pc: 10,
        immediate: 6, addr: 0, value: 0xDEADBEEF, pc_inc: 11,
        expect_r0: 0xDEADBEEF, expect_r1: 100, expect_r2: 2, expect_pc: 11,
        expect_addr: 104, expect_value: 0xDEADBEEF,
    });

    // LW_ABS0: r0 = mem[imm] (id=6)
    let (circ, cbits) = build_lw_abs(6, 0);
    build_and_register("lw_abs0", &circ, &cbits, &TestCase {
        r0: 0, r1: 1, r2: 2, pc: 42,
        immediate: 0x1000, addr: 0, value: 0xCAFEBABE, pc_inc: 43,
        expect_r0: 0xCAFEBABE, expect_r1: 1, expect_r2: 2, expect_pc: 43,
        expect_addr: 0x1000, expect_value: 0xCAFEBABE,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // LW_ABS1: r1 = mem[imm] (id=7)
    let (circ, cbits) = build_lw_abs(7, 1);
    build_and_register("lw_abs1", &circ, &cbits, &TestCase {
        r0: 99, r1: 0, r2: 7, pc: 42,
        immediate: 0x1000, addr: 0, value: 0xCAFEBABE, pc_inc: 43,
        expect_r0: 99, expect_r1: 0xCAFEBABE, expect_r2: 7, expect_pc: 43,
        expect_addr: 0x1000, expect_value: 0xCAFEBABE,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // LW_ABS2: r2 = mem[imm] (id=8)
    let (circ, cbits) = build_lw_abs(8, 2);
    build_and_register("lw_abs2", &circ, &cbits, &TestCase {
        r0: 10, r1: 20, r2: 0, pc: 42,
        immediate: 0x1000, addr: 0, value: 0xBEEFCAFE, pc_inc: 43,
        expect_r0: 10, expect_r1: 20, expect_r2: 0xBEEFCAFE, expect_pc: 43,
        expect_addr: 0x1000, expect_value: 0xBEEFCAFE,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // SW_ABS0: mem[imm] = r0 (id=9)
    let (circ, cbits) = build_sw_abs(9, 0);
    build_and_register("sw_abs0", &circ, &cbits, &TestCase {
        r0: 0xDEADBEEF, r1: 1, r2: 2, pc: 50,
        immediate: 0x2000, addr: 0, value: 0, pc_inc: 51,
        expect_r0: 0xDEADBEEF, expect_r1: 1, expect_r2: 2, expect_pc: 51,
        expect_addr: 0x2000, expect_value: 0xDEADBEEF,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // SW_ABS1: mem[imm] = r1 (id=10)
    let (circ, cbits) = build_sw_abs(10, 1);
    build_and_register("sw_abs1", &circ, &cbits, &TestCase {
        r0: 1, r1: 0xAABBCCDD, r2: 2, pc: 50,
        immediate: 0x3000, addr: 0, value: 0, pc_inc: 51,
        expect_r0: 1, expect_r1: 0xAABBCCDD, expect_r2: 2, expect_pc: 51,
        expect_addr: 0x3000, expect_value: 0xAABBCCDD,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // SW_ABS2: mem[imm] = r2 (id=11)
    let (circ, cbits) = build_sw_abs(11, 2);
    build_and_register("sw_abs2", &circ, &cbits, &TestCase {
        r0: 1, r1: 2, r2: 0x11223344, pc: 50,
        immediate: 0x4000, addr: 0, value: 0, pc_inc: 51,
        expect_r0: 1, expect_r1: 2, expect_r2: 0x11223344, expect_pc: 51,
        expect_addr: 0x4000, expect_value: 0x11223344,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // LW: r0 = mem[r1 + imm] (id=13)
    let (circ, cbits) = build_lw(13);
    build_and_register("lw", &circ, &cbits, &TestCase {
        r0: 0, r1: 100, r2: 5, pc: 10,
        immediate: 7, addr: 0, value: 0xDEADBEEF, pc_inc: 11,
        expect_r0: 0xDEADBEEF, expect_r1: 100, expect_r2: 5, expect_pc: 11,
        expect_addr: 107, expect_value: 0xDEADBEEF,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // SW: mem[r1 + imm] = r0 (id=14)
    let (circ, cbits) = build_sw(14);
    build_and_register("sw", &circ, &cbits, &TestCase {
        r0: 0xCAFEBABE, r1: 200, r2: 5, pc: 10,
        immediate: 12, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 0xCAFEBABE, expect_r1: 200, expect_r2: 5, expect_pc: 11,
        expect_addr: 212, expect_value: 0xCAFEBABE,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // SW_WALIGNED: word store to aligned address (id=52)
    // Test: r0=0xDEADBEEF, r1=100, imm=7 → addr=(107)&~3=104, value=0xDEADBEEF
    let (circ, cbits) = build_sw_waligned(52);
    build_and_register("sw_waligned", &circ, &cbits, &TestCase {
        r0: 0xDEADBEEF, r1: 100, r2: 3, pc: 10,
        immediate: 7, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 0xDEADBEEF, expect_r1: 100, expect_r2: 3, expect_pc: 11,
        expect_addr: 104, expect_value: 0xDEADBEEF,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // SEXT8: r0 = sign_extend(r0[7:0]) (id=16)
    // Test 1: 0xFF → 0xFFFFFFFF (-1)
    let (circ, cbits) = build_sext8(16);
    build_and_register("sext8", &circ, &cbits, &TestCase {
        r0: 0x000000FF, r1: 1, r2: 2, pc: 10,
        immediate: 0, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 0xFFFFFFFF, expect_r1: 1, expect_r2: 2, expect_pc: 11,
        expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // Test sext8 with positive byte: 0x7F → 0x0000007F
    let (circ2, _) = build_sext8(16);
    {
        let mut input_bits = vec![false; INPUT_COUNT];
        set_u32(&mut input_bits, R0_OFF, 0x0000007F);
        let output = circ2.evaluate(input_bits).expect("evaluate failed");
        let r0_out = bits_to_u32(&output, R0_OFF, R0_BITS);
        assert_eq!(r0_out, 0x0000007F, "sext8 positive: got 0x{:08X} expected 0x0000007F", r0_out);
        println!("  sext8 positive byte (0x7F → 0x{:08X}): OK", r0_out);
    }

    // BYTE_SEL_R2: r0 = (r0 >> (r2 * 8)) & 0xFF (id=15)
    // Test: r0=0xDEADBEEF, r2=2 → byte 2 = 0xAD, value[0..7] = 0xAD hint
    let (circ, cbits) = build_byte_sel_r2(15);
    build_and_register("byte_sel_r2", &circ, &cbits, &TestCase {
        r0: 0xDEADBEEF, r1: 0, r2: 2, pc: 10,
        immediate: 0, addr: 0, value: 0xAD, pc_inc: 11,
        expect_r0: 0xAD, expect_r1: 0, expect_r2: 2, expect_pc: 11,
        expect_addr: 0, expect_value: 0xAD,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // BYTE_INS_R2: insert byte from mem[0x4184] into r0 at byte position r2 (id=53)
    // Test: r0=0xDEADBEEF, r2=1, value[0..7]=0xAB → replace byte 1 → 0xDEADABEF
    let (circ, cbits) = build_byte_ins_r2(53);
    build_and_register("byte_ins_r2", &circ, &cbits, &TestCase {
        r0: 0xDEADBEEF, r1: 5, r2: 1, pc: 10,
        immediate: 0, addr: 0, value: 0xAB, pc_inc: 11,
        expect_r0: 0xDEADABEF, expect_r1: 5, expect_r2: 1, expect_pc: 11,
        expect_addr: 0x4184, expect_value: 0xAB,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // ADDI: r0 = r1 + imm (id=12)
    let (circ, cbits) = build_addi(12);
    build_and_register("addi", &circ, &cbits, &TestCase {
        r0: 0, r1: 100, r2: 7, pc: 10,
        immediate: 50, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 150, expect_r1: 100, expect_r2: 7, expect_pc: 11,
        expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // XORI: r0 = r1 ^ imm (id=19)
    let (circ, cbits) = build_xori(19);
    build_and_register("xori", &circ, &cbits, &TestCase {
        r0: 0, r1: 0xFF00FF00, r2: 7, pc: 10,
        immediate: 0x0F0F0F0F, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 0xF00FF00F, expect_r1: 0xFF00FF00, expect_r2: 7, expect_pc: 11,
        expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // LUI: r0 = imm << 12 (id=20)
    // Test: imm = 0xABCDE → r0 = 0xABCDE000
    let (circ, cbits) = build_lui(20);
    build_and_register("lui", &circ, &cbits, &TestCase {
        r0: 0, r1: 1, r2: 2, pc: 10,
        immediate: 0xABCDE, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 0xABCDE000, expect_r1: 1, expect_r2: 2, expect_pc: 11,
        expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // BLT: if (r0 < r1) pc = imm[0..15] (id=17)
    // Test 1: branch taken (-6 < 3 → pc = imm = 80)
    let (circ, cbits) = build_blt(17);
    build_and_register("blt", &circ, &cbits, &TestCase {
        r0: 0xFFFFFFFA, r1: 3, r2: 7, pc: 10,
        immediate: 80, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 0xFFFFFFFA, expect_r1: 3, expect_r2: 7, expect_pc: 80,
        expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // Test blt not taken: 5 < 3 is false → pc = pc_inc (11)
    let (circ2, _) = build_blt(17);
    {
        let mut input_bits = vec![false; INPUT_COUNT];
        set_u32(&mut input_bits, R0_OFF, 5);
        set_u32(&mut input_bits, R1_OFF, 3);
        set_u32(&mut input_bits, R2_OFF, 7);
        set_u16(&mut input_bits, PC_OFF, 10);
        set_u16(&mut input_bits, PC_INC_OFF, 11);
        set_u32(&mut input_bits, IMM_OFF, 80);
        let output = circ2.evaluate(input_bits).expect("evaluate failed");
        let pc_out = bits_to_u16(&output, PC_OFF, PC_BITS);
        let r0_out = bits_to_u32(&output, R0_OFF, R0_BITS);
        assert_eq!(pc_out, 11, "blt not-taken: pc got {} expected 11", pc_out);
        assert_eq!(r0_out, 5, "blt not-taken: r0 got {} expected 5", r0_out);
        println!("  blt not-taken (5 < 3 = false → pc=11): OK");
    }

    // BGE: if (r0 >= r1) pc = imm[0..15] (id=18)
    // Test 1: branch taken (5 >= 3 → pc = imm = 80)
    let (circ, cbits) = build_bge(18);
    build_and_register("bge", &circ, &cbits, &TestCase {
        r0: 5, r1: 3, r2: 7, pc: 10,
        immediate: 80, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 5, expect_r1: 3, expect_r2: 7, expect_pc: 80,
        expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // Test bge not taken: -6 >= 3 is false → pc = pc_inc (11)
    let (circ2, _) = build_bge(18);
    {
        let mut input_bits = vec![false; INPUT_COUNT];
        set_u32(&mut input_bits, R0_OFF, 0xFFFFFFFA);
        set_u32(&mut input_bits, R1_OFF, 3);
        set_u32(&mut input_bits, R2_OFF, 7);
        set_u16(&mut input_bits, PC_OFF, 10);
        set_u16(&mut input_bits, PC_INC_OFF, 11);
        set_u32(&mut input_bits, IMM_OFF, 80);
        let output = circ2.evaluate(input_bits).expect("evaluate failed");
        let pc_out = bits_to_u16(&output, PC_OFF, PC_BITS);
        assert_eq!(pc_out, 11, "bge not-taken: pc got {} expected 11", pc_out);
        println!("  bge not-taken (-6 >= 3 = false → pc=11): OK");
    }
    // Test bge taken with equal values: 3 >= 3 → taken
    {
        let mut input_bits = vec![false; INPUT_COUNT];
        set_u32(&mut input_bits, R0_OFF, 3);
        set_u32(&mut input_bits, R1_OFF, 3);
        set_u16(&mut input_bits, PC_OFF, 10);
        set_u16(&mut input_bits, PC_INC_OFF, 11);
        set_u32(&mut input_bits, IMM_OFF, 80);
        let output = circ2.evaluate(input_bits).expect("evaluate failed");
        let pc_out = bits_to_u16(&output, PC_OFF, PC_BITS);
        assert_eq!(pc_out, 80, "bge equal: pc got {} expected 80", pc_out);
        println!("  bge equal (3 >= 3 = true → pc=80): OK");
    }

    // BLTU: if (r0 < r1) unsigned, pc = imm[0..15] (id=26)
    // Test 1: branch taken (3 < 0xFFFFFFFA unsigned → taken, pc = 80)
    let (circ, cbits) = build_bltu(26);
    build_and_register("bltu", &circ, &cbits, &TestCase {
        r0: 3, r1: 0xFFFFFFFA, r2: 7, pc: 10,
        immediate: 80, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 3, expect_r1: 0xFFFFFFFA, expect_r2: 7, expect_pc: 80,
        expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // Test bltu not taken: 0xFFFFFFFA < 3 unsigned → false, pc = pc_inc
    let (circ2, _) = build_bltu(26);
    {
        let mut input_bits = vec![false; INPUT_COUNT];
        set_u32(&mut input_bits, R0_OFF, 0xFFFFFFFA);
        set_u32(&mut input_bits, R1_OFF, 3);
        set_u16(&mut input_bits, PC_OFF, 10);
        set_u16(&mut input_bits, PC_INC_OFF, 11);
        set_u32(&mut input_bits, IMM_OFF, 80);
        let output = circ2.evaluate(input_bits).expect("evaluate failed");
        let pc_out = bits_to_u16(&output, PC_OFF, PC_BITS);
        assert_eq!(pc_out, 11, "bltu not-taken: pc got {} expected 11", pc_out);

        println!("  bltu not-taken (0xFFFFFFFA < 3 unsigned = false → pc=11): OK");
    }

    // BGEU: if (r0 >= r1) unsigned, pc = imm[0..15] (id=45)
    // Test 1: branch taken (0xFFFFFFFA >= 3 unsigned → pc = 80)
    let (circ, cbits) = build_bgeu(45);
    build_and_register("bgeu", &circ, &cbits, &TestCase {
        r0: 0xFFFFFFFA, r1: 3, r2: 7, pc: 10,
        immediate: 80, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 0xFFFFFFFA, expect_r1: 3, expect_r2: 7, expect_pc: 80,
        expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // Test bgeu not taken: 3 < 0xFFFFFFFA unsigned → not taken, pc = pc_inc
    let (circ2, _) = build_bgeu(45);
    {
        let mut input_bits = vec![false; INPUT_COUNT];
        set_u32(&mut input_bits, R0_OFF, 3);
        set_u32(&mut input_bits, R1_OFF, 0xFFFFFFFA);
        set_u16(&mut input_bits, PC_OFF, 10);
        set_u16(&mut input_bits, PC_INC_OFF, 11);
        set_u32(&mut input_bits, IMM_OFF, 80);
        let output = circ2.evaluate(input_bits).expect("evaluate failed");
        let pc_out = bits_to_u16(&output, PC_OFF, PC_BITS);
        assert_eq!(pc_out, 11, "bgeu not-taken: pc got {} expected 11", pc_out);

        println!("  bgeu not-taken (3 >= 0xFFFFFFFA unsigned = false → pc=11): OK");
    }
    // Test bgeu taken with equal: 5 >= 5 → taken
    {
        let mut input_bits = vec![false; INPUT_COUNT];
        set_u32(&mut input_bits, R0_OFF, 5);
        set_u32(&mut input_bits, R1_OFF, 5);
        set_u16(&mut input_bits, PC_OFF, 10);
        set_u16(&mut input_bits, PC_INC_OFF, 11);
        set_u32(&mut input_bits, IMM_OFF, 80);
        let output = circ2.evaluate(input_bits).expect("evaluate failed");
        let pc_out = bits_to_u16(&output, PC_OFF, PC_BITS);
        assert_eq!(pc_out, 80, "bgeu equal: pc got {} expected 80", pc_out);

        println!("  bgeu equal (5 >= 5 unsigned = true → pc=80): OK");
    }

    // BNE: if (r0 != r1) pc = imm[0..15] (id=25)
    // Test 1: branch taken (5 != 3 → pc = imm = 80)
    let (circ, cbits) = build_bne(25);
    build_and_register("bne", &circ, &cbits, &TestCase {
        r0: 5, r1: 3, r2: 7, pc: 10,
        immediate: 80, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 5, expect_r1: 3, expect_r2: 7, expect_pc: 80,
        expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // Test bne not taken: 3 == 3 → pc = pc_inc (11)
    let (circ2, _) = build_bne(25);
    {
        let mut input_bits = vec![false; INPUT_COUNT];
        set_u32(&mut input_bits, R0_OFF, 3);
        set_u32(&mut input_bits, R1_OFF, 3);
        set_u16(&mut input_bits, PC_OFF, 10);
        set_u16(&mut input_bits, PC_INC_OFF, 11);
        set_u32(&mut input_bits, IMM_OFF, 80);
        let output = circ2.evaluate(input_bits).expect("evaluate failed");
        let pc_out = bits_to_u16(&output, PC_OFF, PC_BITS);
        assert_eq!(pc_out, 11, "bne not-taken: pc got {} expected 11", pc_out);

        println!("  bne not-taken (3 == 3 → pc=11): OK");
    }

    // BEQ: if (r0 == r1) pc = imm[0..15] (id=27)
    // Test 1: branch taken (3 == 3 → pc = imm = 80)
    let (circ, cbits) = build_beq(27);
    build_and_register("beq", &circ, &cbits, &TestCase {
        r0: 3, r1: 3, r2: 7, pc: 10,
        immediate: 80, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 3, expect_r1: 3, expect_r2: 7, expect_pc: 80,
        expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // Test beq not taken: 5 != 3 → pc = pc_inc (11)
    let (circ2, _) = build_beq(27);
    {
        let mut input_bits = vec![false; INPUT_COUNT];
        set_u32(&mut input_bits, R0_OFF, 5);
        set_u32(&mut input_bits, R1_OFF, 3);
        set_u16(&mut input_bits, PC_OFF, 10);
        set_u16(&mut input_bits, PC_INC_OFF, 11);
        set_u32(&mut input_bits, IMM_OFF, 80);
        let output = circ2.evaluate(input_bits).expect("evaluate failed");
        let pc_out = bits_to_u16(&output, PC_OFF, PC_BITS);
        assert_eq!(pc_out, 11, "beq not-taken: pc got {} expected 11", pc_out);

        println!("  beq not-taken (5 != 3 → pc=11): OK");
    }

    // JAL: if imm == pc then halt, else pc = imm (id=47)
    // Test 1: jump (imm=80 != pc=10 → pc = 80)
    let (circ, cbits) = build_jal(47);
    build_and_register("jal", &circ, &cbits, &TestCase {
        r0: 1, r1: 2, r2: 3, pc: 10,
        immediate: 80, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 1, expect_r1: 2, expect_r2: 3, expect_pc: 80,
        expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // Test 2: halt (imm == pc → pc = 0xFFFF)
    let (circ2, _) = build_jal(47);
    {
        let mut input_bits = vec![false; INPUT_COUNT];
        set_u32(&mut input_bits, R0_OFF, 1);
        set_u16(&mut input_bits, PC_OFF, 42);
        set_u16(&mut input_bits, PC_INC_OFF, 43);
        set_u32(&mut input_bits, IMM_OFF, 42); // imm[0..15] == pc
        let output = circ2.evaluate(input_bits).expect("evaluate failed");
        let pc_out = bits_to_u16(&output, PC_OFF, PC_BITS);
        assert_eq!(pc_out, 0xFFFF, "jal halt: pc got 0x{:04X} expected 0xFFFF", pc_out);
        println!("  jal halt (imm==pc=42 → pc=0xFFFF): OK");
    }

    // JAL_CALL: save pc to mailbox, jump to imm (id=49)
    // Test: pc=42, imm=100 → pc=100, addr=0x4104, value=42
    let (circ, cbits) = build_jal_call(49);
    build_and_register("jal_call", &circ, &cbits, &TestCase {
        r0: 1, r1: 2, r2: 3, pc: 42,
        immediate: 100, addr: 0, value: 0, pc_inc: 43,
        expect_r0: 1, expect_r1: 2, expect_r2: 3, expect_pc: 100,
        expect_addr: 0x4104, expect_value: 43,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // JALR_CALL: save pc to mailbox, jump to r0 (id=50)
    // Test: r0=200, pc=42 → pc=200, addr=0x4104, value=42
    let (circ, cbits) = build_jalr_call(50);
    build_and_register("jalr_call", &circ, &cbits, &TestCase {
        r0: 200, r1: 2, r2: 3, pc: 42,
        immediate: 0, addr: 0, value: 0, pc_inc: 43,
        expect_r0: 200, expect_r1: 2, expect_r2: 3, expect_pc: 200,
        expect_addr: 0x4104, expect_value: 42,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // RET: if r0 == 0 halt, else pc = r0[0..15] (id=48)
    // Test 1: return (r0=80 → pc=80)
    let (circ, cbits) = build_ret(48);
    build_and_register("ret", &circ, &cbits, &TestCase {
        r0: 80, r1: 2, r2: 3, pc: 10,
        immediate: 0, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 80, expect_r1: 2, expect_r2: 3, expect_pc: 80,
        expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // Test 2: halt (r0=0 → pc=0xFFFF)
    let (circ2, _) = build_ret(48);
    {
        let mut input_bits = vec![false; INPUT_COUNT];
        set_u32(&mut input_bits, R0_OFF, 0);
        set_u16(&mut input_bits, PC_OFF, 10);
        set_u16(&mut input_bits, PC_INC_OFF, 11);
        let output = circ2.evaluate(input_bits).expect("evaluate failed");
        let pc_out = bits_to_u16(&output, PC_OFF, PC_BITS);
        assert_eq!(pc_out, 0xFFFF, "ret halt: pc got 0x{:04X} expected 0xFFFF", pc_out);
        println!("  ret halt (r0=0 → pc=0xFFFF): OK");
    }

    // PADDING: all pass-through (id=255)
    let (circ, cbits) = build_padding(255);
    build_and_register("padding", &circ, &cbits, &TestCase {
        r0: 0xDEAD, r1: 0xBEEF, r2: 0xCAFE, pc: 0xFFFF,
        immediate: 0, addr: 0, value: 0, pc_inc: 0xFFFF,
        expect_r0: 0xDEAD, expect_r1: 0xBEEF, expect_r2: 0xCAFE, expect_pc: 0xFFFF,
        expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // Constant-amount shifts (all 0 AND, pure wiring)
    // SLL: shift left
    let (circ, cbits) = build_sll_const(28, 1);
    build_and_register("sll1", &circ, &cbits, &TestCase {
        r0: 15, r1: 0, r2: 0, pc: 10, immediate: 0, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 30, expect_r1: 0, expect_r2: 0, expect_pc: 11, expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    let (circ, cbits) = build_sll_const(29, 4);
    build_and_register("sll4", &circ, &cbits, &TestCase {
        r0: 0x0F, r1: 0, r2: 0, pc: 10, immediate: 0, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 0xF0, expect_r1: 0, expect_r2: 0, expect_pc: 11, expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    let (circ, cbits) = build_sll_const(30, 8);
    build_and_register("sll8", &circ, &cbits, &TestCase {
        r0: 0x0F, r1: 0, r2: 0, pc: 10, immediate: 0, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 0x0F00, expect_r1: 0, expect_r2: 0, expect_pc: 11, expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    let (circ, cbits) = build_sll_const(31, 16);
    build_and_register("sll16", &circ, &cbits, &TestCase {
        r0: 0x0F, r1: 0, r2: 0, pc: 10, immediate: 0, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 0x000F0000, expect_r1: 0, expect_r2: 0, expect_pc: 11, expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    let (circ, cbits) = build_sll_const(32, 31);
    build_and_register("sll31", &circ, &cbits, &TestCase {
        r0: 1, r1: 0, r2: 0, pc: 10, immediate: 0, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 0x80000000, expect_r1: 0, expect_r2: 0, expect_pc: 11, expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // SRL: shift right logical
    let (circ, cbits) = build_srl_const(33, 1);
    build_and_register("srl1", &circ, &cbits, &TestCase {
        r0: 0x80000002, r1: 0, r2: 0, pc: 10, immediate: 0, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 0x40000001, expect_r1: 0, expect_r2: 0, expect_pc: 11, expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    let (circ, cbits) = build_srl_const(34, 4);
    build_and_register("srl4", &circ, &cbits, &TestCase {
        r0: 0xF0, r1: 0, r2: 0, pc: 10, immediate: 0, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 0x0F, expect_r1: 0, expect_r2: 0, expect_pc: 11, expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    let (circ, cbits) = build_srl_const(35, 8);
    build_and_register("srl8", &circ, &cbits, &TestCase {
        r0: 0xFF00, r1: 0, r2: 0, pc: 10, immediate: 0, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 0xFF, expect_r1: 0, expect_r2: 0, expect_pc: 11, expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    let (circ, cbits) = build_srl_const(36, 16);
    build_and_register("srl16", &circ, &cbits, &TestCase {
        r0: 0xFFFF0000, r1: 0, r2: 0, pc: 10, immediate: 0, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 0xFFFF, expect_r1: 0, expect_r2: 0, expect_pc: 11, expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    let (circ, cbits) = build_srl_const(37, 31);
    build_and_register("srl31", &circ, &cbits, &TestCase {
        r0: 0x80000000, r1: 0, r2: 0, pc: 10, immediate: 0, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 1, expect_r1: 0, expect_r2: 0, expect_pc: 11, expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // SRA: shift right arithmetic
    let (circ, cbits) = build_sra_const(38, 1);
    build_and_register("sra1", &circ, &cbits, &TestCase {
        r0: 0x80000002, r1: 0, r2: 0, pc: 10, immediate: 0, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 0xC0000001, expect_r1: 0, expect_r2: 0, expect_pc: 11, expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    let (circ, cbits) = build_sra_const(39, 4);
    build_and_register("sra4", &circ, &cbits, &TestCase {
        r0: 0x80000000, r1: 0, r2: 0, pc: 10, immediate: 0, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 0xF8000000, expect_r1: 0, expect_r2: 0, expect_pc: 11, expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    let (circ, cbits) = build_sra_const(40, 8);
    build_and_register("sra8", &circ, &cbits, &TestCase {
        r0: 0x80000000, r1: 0, r2: 0, pc: 10, immediate: 0, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 0xFF800000, expect_r1: 0, expect_r2: 0, expect_pc: 11, expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    let (circ, cbits) = build_sra_const(41, 16);
    build_and_register("sra16", &circ, &cbits, &TestCase {
        r0: 0x80000000, r1: 0, r2: 0, pc: 10, immediate: 0, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 0xFFFF8000, expect_r1: 0, expect_r2: 0, expect_pc: 11, expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    let (circ, cbits) = build_sra_const(42, 31);
    build_and_register("sra31", &circ, &cbits, &TestCase {
        r0: 0x80000000, r1: 0, r2: 0, pc: 10, immediate: 0, addr: 0, value: 0, pc_inc: 11,
        expect_r0: 0xFFFFFFFF, expect_r1: 0, expect_r2: 0, expect_pc: 11, expect_addr: 0, expect_value: 0,
    }, &mut names, &mut translations, &mut test_inputs, &mut test_expects);

    // Verify translated circuits produce correct I/O outputs + const wires
    println!("\n=== Verifying {} circuits ===", translations.len());
    for i in 0..translations.len() {
        let out = evaluate_translated(&translations[i], &test_inputs[i]);
        let r0 = bits_to_u32(&out, R0_OFF, R0_BITS);
        let r1 = bits_to_u32(&out, R1_OFF, R1_BITS);
        let r2 = bits_to_u32(&out, R2_OFF, R2_BITS);
        let pc = bits_to_u16(&out, PC_OFF, PC_BITS);
        let addr = bits_to_u32(&out, ADDR_OFF, ADDR_BITS);
        let value = bits_to_u32(&out, VALUE_OFF, VALUE_BITS);
        let e = &test_expects[i];
        assert_eq!(r0, e.expect_r0, "{}: r0 got {} expected {}", names[i], r0, e.expect_r0);
        assert_eq!(r1, e.expect_r1, "{}: r1 got {} expected {}", names[i], r1, e.expect_r1);
        assert_eq!(r2, e.expect_r2, "{}: r2 got {} expected {}", names[i], r2, e.expect_r2);
        assert_eq!(pc, e.expect_pc, "{}: pc got {} expected {}", names[i], pc, e.expect_pc);
        assert_eq!(addr, e.expect_addr, "{}: addr got {} expected {}", names[i], addr, e.expect_addr);
        assert_eq!(value, e.expect_value, "{}: value got {} expected {}", names[i], value, e.expect_value);
        // Verify const wires
        verify_const_wires(&names[i], &translations[i], &test_inputs[i]);
        println!("  {}: {} gates ({} AND + {} XOR + {} INV), {} consts OK",
                 names[i], translations[i].gate_count, translations[i].and_count,
                 translations[i].xor_count, translations[i].inv_count,
                 translations[i].const_wires.len());
    }

    // Export raw translated circuits
    println!("\n=== Exporting {} circuits ===", translations.len());
    for i in 0..translations.len() {
        let out_dir = format!("{}/generated", env!("CARGO_MANIFEST_DIR"));
        std::fs::create_dir_all(&out_dir).expect("create generated dir");
        let filename = format!("{}/{}.bin", out_dir, names[i]);
        export_binary(&translations[i], &filename);
        println!("  {}: exported", names[i]);
    }
}
