//! Memory check verifier.
//!
//! Reads the proof written by memory_check_prover, rebuilds the constraint
//! system, and verifies the proof.
//!
//! Run with:
//!   cargo run --release --bin memory_check_verifier [PROOF_FILE]

use std::time::Instant;

use std::collections::HashSet;

use anyhow::Result;
use batchman_witness_generator::{LookupTrace, MemTrace};
use binius_core::{
    constraint_system::{FriStrategy, verify, Proof},
    fiat_shamir::HasherChallenger,
};
use binius_field::{arch::OptimalUnderlier, ext_basis, tower::CanonicalTowerFamily, Field};
use binius_m3::builder::{
    upcast_col, Boundary, Col, ConstraintSystem, FlushDirection, FlushOpts,
    B1, B8, B16, B32, B128,
};
use binius_m3::gadgets::lookup::LookupProducer;
use memory_checker_and_lookup::{Blake3Digest, Blake3Compression, commit_column_b128, commit_column_b32};

// ==== Constants (must match prover) ====

const LOG_INV_RATE: usize = 2;
const SECURITY_BITS: usize = 100;
const FRI_STRATEGY: FriStrategy = FriStrategy::ConstantArity(8);

const PC_IMM_GROUP: u32 = 2;
const PC_OP_GROUP: u32 = 3;

/// Build a main table with all columns, constraints, and channel flushes.
/// Must match memory_check_prover::build_main_table exactly.
fn build_main_table(
    cs: &mut ConstraintSystem,
    name: &str,
    mem_chan: usize,
    step_cmp_chan: usize,
    pc_imm_chan: usize,
    pc_op_chan: usize,
    pc_inc_chan: usize,
) {
    let mut table = cs.add_table(name);

    let c_addr: Col<B16> = table.add_committed("addr");
    let c_value: Col<B32> = table.add_committed("value");
    let c_imm: Col<B32> = table.add_committed("imm");
    let c_pc: Col<B16> = table.add_committed("pc");
    let c_next_pc: Col<B16> = table.add_committed("next_pc");
    let c_op: Col<B8> = table.add_committed("op");
    let is_write: Col<B1> = table.add_committed("is_write");
    let is_read: Col<B1> = table.add_committed("is_read");
    let c_has_imm: Col<B1> = table.add_committed("has_imm");
    let c_is_byte_sel_r2: Col<B1> = table.add_committed("is_byte_sel_r2");
    let c_prev_value: Col<B32> = table.add_committed("prev_value");
    let c_step: Col<B32> = table.add_committed_in_group("step", 4);
    let c_r0_b0: Col<B8> = table.add_committed("r0_b0");
    let c_r0_b1: Col<B8> = table.add_committed("r0_b1");
    let c_r0_b2: Col<B8> = table.add_committed("r0_b2");
    let c_r0_b3: Col<B8> = table.add_committed("r0_b3");
    let c_r2_idx0: Col<B1> = table.add_committed("r2_idx0");
    let c_r2_idx1: Col<B1> = table.add_committed("r2_idx1");
    let c_new_r0: Col<B32> = table.add_committed("new_r0");

    // Byte select constraint: new_r0 = r0.bytes[r2_idx] when is_byte_sel_r2=1
    {
        let one = B32::ONE;
        let idx0 = upcast_col::<B32, _, 1>(c_r2_idx0);
        let idx1 = upcast_col::<B32, _, 1>(c_r2_idx1);
        let b0 = upcast_col::<B32, _, 1>(c_r0_b0);
        let b1 = upcast_col::<B32, _, 1>(c_r0_b1);
        let b2 = upcast_col::<B32, _, 1>(c_r0_b2);
        let b3 = upcast_col::<B32, _, 1>(c_r0_b3);
        let sel = upcast_col::<B32, _, 1>(c_is_byte_sel_r2);
        let mux = (idx0 + one) * (idx1 + one) * b0
                + idx0 * (idx1 + one) * b1
                + (idx0 + one) * idx1 * b2
                + idx0 * idx1 * b3;
        table.assert_zero("byte_select_mux", sel * (c_new_r0 - mux));
    }

    let c_fri_blind: Col<B128> = table.add_committed("fri_blind");
    let c_fri_blind_sq: Col<B128> = table.add_committed("fri_blind_sq");
    table.assert_zero("sumcheck_blind", c_fri_blind * c_fri_blind - c_fri_blind_sq);

    // Packed plaintext: single B128 in standalone group 1.
    // Tower-optimized layout (must match prover):
    //   [0:31]    value    (B32) × 1
    //   [32:63]   imm      (B32) × X^32
    //   [64:79]   pc       (B16) × X^64
    //   [80:95]   next_pc  (B16) × X^80
    //   [96:111]  addr     (B16) × X^96
    //   [112:119] op       (B8)  × X^112
    //   [120]     is_w     (B1)  × X^120
    //   [121]     is_r     (B1)  × X^121
    //   [122]     has_imm  (B1)  × X^122
    //   [123]     bsel     (B1)  × X^123
    //   [124:127] zero
    let c_packed_pt: Col<B128> = table.add_committed_in_group("packed_pt", 1);
    {
        let x32  = B128::new(1u128 << 32);
        let x64  = B128::new(1u128 << 64);
        let x80  = B128::new(1u128 << 80);
        let x96  = B128::new(1u128 << 96);
        let x112 = B128::new(1u128 << 112);
        let x120 = B128::new(1u128 << 120);
        let x121 = B128::new(1u128 << 121);
        let x122 = B128::new(1u128 << 122);
        let x123 = B128::new(1u128 << 123);
        table.assert_zero("decompose_packed_pt",
            c_packed_pt
                - upcast_col::<B128, _, 1>(c_value)
                - upcast_col::<B128, _, 1>(c_imm) * x32
                - upcast_col::<B128, _, 1>(c_pc) * x64
                - upcast_col::<B128, _, 1>(c_next_pc) * x80
                - upcast_col::<B128, _, 1>(c_addr) * x96
                - upcast_col::<B128, _, 1>(c_op) * x112
                - upcast_col::<B128, _, 1>(is_write) * x120
                - upcast_col::<B128, _, 1>(is_read) * x121
                - upcast_col::<B128, _, 1>(c_has_imm) * x122
                - upcast_col::<B128, _, 1>(c_is_byte_sel_r2) * x123);
    }

    let c_prev_access_step: Col<B32> = table.add_committed("prev_access_step");

    let c_mem_blind: Col<B128> = table.add_committed("mem_blind");
    let c_addr_b128: Col<B128> = upcast_col(c_addr);
    let c_value_b128: Col<B128> = upcast_col(c_value);
    let c_prev_value_b128: Col<B128> = upcast_col(c_prev_value);
    let step_128: Col<B128> = upcast_col(c_step);
    let prev_access_step_128: Col<B128> = upcast_col(c_prev_access_step);

    // Write: pull old (addr, prev_value, prev_access_step, blind), push new (addr, value, step, blind).
    table.pull_with_opts(
        mem_chan,
        [c_addr_b128, c_prev_value_b128, prev_access_step_128, c_mem_blind],
        FlushOpts { selectors: vec![is_write], ..Default::default() },
    );
    table.push_with_opts(
        mem_chan,
        [c_addr_b128, c_value_b128, step_128, c_mem_blind],
        FlushOpts { selectors: vec![is_write], ..Default::default() },
    );
    // Read: pull old (addr, prev_value, prev_access_step, blind), push new (addr, value, step, blind).
    table.pull_with_opts(
        mem_chan,
        [c_addr_b128, c_prev_value_b128, prev_access_step_128, c_mem_blind],
        FlushOpts { selectors: vec![is_read], ..Default::default() },
    );
    table.push_with_opts(
        mem_chan,
        [c_addr_b128, c_value_b128, step_128, c_mem_blind],
        FlushOpts { selectors: vec![is_read], ..Default::default() },
    );

    // Step comparison: push (step, prev_access_step) for active rows.
    table.push_with_opts(
        step_cmp_chan,
        [step_128, prev_access_step_128],
        FlushOpts { selectors: vec![is_write], ..Default::default() },
    );
    table.push_with_opts(
        step_cmp_chan,
        [step_128, prev_access_step_128],
        FlushOpts { selectors: vec![is_read], ..Default::default() },
    );

    // PC-immediate lookup: read (pc, imm) when has_imm=1.
    let pc_128: Col<B128> = upcast_col(c_pc);
    let imm_128: Col<B128> = upcast_col(c_imm);
    table.pull_with_opts(
        pc_imm_chan,
        [pc_128, imm_128],
        FlushOpts { selectors: vec![c_has_imm], ..Default::default() },
    );

    // PC-op lookup: every row reads (pc, op) from the lookup table.
    let op_128: Col<B128> = upcast_col(c_op);
    table.read(pc_op_chan, [pc_128, op_128]);

    // PC-increment lookup: every row reads (pc, next_pc) from the increment table.
    let npc_128: Col<B128> = upcast_col(c_next_pc);
    table.read(pc_inc_chan, [pc_128, npc_128]);

    drop(table);
}

fn main() -> Result<()> {
    let program = std::env::args().nth(1).unwrap_or_else(|| "json-query".to_string());
    let witness_dir = format!("../witgen/witness/{program}");
    let proof_path = format!("proofs/{program}/memory_check_proof.bin");
    let mem_path = format!("{witness_dir}/mem_trace.bin");
    let lookup_path = format!("{witness_dir}/lookup_trace.bin");

    // ==== 1. Read proof + boundary data ====
    let t_total = Instant::now();

    let raw = std::fs::read(&proof_path)?;
    let mut cursor = &raw[..];

    let transcript_len = {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&cursor[..8]);
        cursor = &cursor[8..];
        u64::from_le_bytes(buf) as usize
    };
    let transcript = cursor[..transcript_len].to_vec();
    cursor = &cursor[transcript_len..];

    let num_boundaries = {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&cursor[..4]);
        cursor = &cursor[4..];
        u32::from_le_bytes(buf) as usize
    };
    let mut boundary_pairs: Vec<(u32, u32)> = Vec::with_capacity(num_boundaries);
    for _ in 0..num_boundaries {
        let mut buf4 = [0u8; 4];
        buf4.copy_from_slice(&cursor[..4]);
        let slot = u32::from_le_bytes(buf4);
        cursor = &cursor[4..];
        buf4.copy_from_slice(&cursor[..4]);
        let value = u32::from_le_bytes(buf4);
        cursor = &cursor[4..];
        boundary_pairs.push((slot, value));
    }

    println!("Memory check verifier");
    println!("  Proof file:    {}", proof_path);
    println!("  Transcript:    {} bytes", transcript_len);
    println!("  Boundaries:    {}", num_boundaries);
    println!();

    // ==== 2. Rebuild constraint system (must match prover exactly) ====
    let mut cs = ConstraintSystem::new();
    let mem_chan = cs.add_channel("memory");
    let step_cmp_chan = cs.add_channel("step_cmp");
    let pc_imm_chan = cs.add_channel("pc_immediate");
    let pc_op_chan = cs.add_channel("pc_op");
    let pc_inc_chan = cs.add_channel("pc_inc");
    // Main table.
    build_main_table(&mut cs, "main", mem_chan, step_cmp_chan, pc_imm_chan, pc_op_chan, pc_inc_chan);

    // Page-in table.
    let mut pi_table = cs.add_table("page_in");
    let pi_fri_blind: Col<B128> = pi_table.add_committed("fri_blind");
    let pi_fri_blind_sq: Col<B128> = pi_table.add_committed("fri_blind_sq");
    pi_table.assert_zero("sumcheck_blind", pi_fri_blind * pi_fri_blind - pi_fri_blind_sq);
    let pi_addr: Col<B16> = pi_table.add_committed("addr");
    let pi_value: Col<B32> = pi_table.add_committed("value");
    let pi_step: Col<B32> = pi_table.add_committed("access_step");
    let pi_blind: Col<B128> = pi_table.add_committed("blind");
    pi_table.push(mem_chan, [
        upcast_col::<B128, _, 1>(pi_addr), upcast_col::<B128, _, 1>(pi_value),
        upcast_col::<B128, _, 1>(pi_step), pi_blind,
    ]);
    drop(pi_table);

    // Page-out table.
    let pub_io_chan = cs.add_channel("pub_io");
    let mut po_table = cs.add_table("page_out");
    let po_fri_blind: Col<B128> = po_table.add_committed("fri_blind");
    let po_fri_blind_sq: Col<B128> = po_table.add_committed("fri_blind_sq");
    po_table.assert_zero("sumcheck_blind", po_fri_blind * po_fri_blind - po_fri_blind_sq);
    let po_addr: Col<B16> = po_table.add_committed("addr");
    let po_value: Col<B32> = po_table.add_committed("value");
    let po_step: Col<B32> = po_table.add_committed("access_step");
    let po_blind: Col<B128> = po_table.add_committed("blind");
    let po_is_output: Col<B1> = po_table.add_committed("is_output");
    po_table.pull(mem_chan, [
        upcast_col::<B128, _, 1>(po_addr), upcast_col::<B128, _, 1>(po_value),
        upcast_col::<B128, _, 1>(po_step), po_blind,
    ]);
    po_table.push_with_opts(
        pub_io_chan,
        [upcast_col::<B128, _, 1>(po_addr), upcast_col::<B128, _, 1>(po_value)],
        FlushOpts { selectors: vec![po_is_output], ..Default::default() },
    );
    drop(po_table);

    // Step comparison table (byte-limb decomposition + 8-bit GT borrow chain).
    // Must match prover's step_cmp table exactly.
    {
        let mut sct = cs.add_table("step_cmp");
        let sc_fri_blind: Col<B128> = sct.add_committed("fri_blind");
        let sc_fri_blind_sq: Col<B128> = sct.add_committed("fri_blind_sq");
        sct.assert_zero("sumcheck_blind", sc_fri_blind * sc_fri_blind - sc_fri_blind_sq);

        let sc_step: Col<B32> = sct.add_committed("step");
        let sc_prev: Col<B32> = sct.add_committed("prev_access_step");
        let sc_is_active: Col<B1> = sct.add_committed("is_active");
        sct.pull_with_opts(step_cmp_chan, [upcast_col::<B128, _, 1>(sc_step), upcast_col::<B128, _, 1>(sc_prev)],
            FlushOpts { selectors: vec![sc_is_active], ..Default::default() });

        let s0: Col<B8> = sct.add_committed("s0");
        let s1: Col<B8> = sct.add_committed("s1");
        let s2: Col<B8> = sct.add_committed("s2");
        let ps0: Col<B8> = sct.add_committed("ps0");
        let ps1: Col<B8> = sct.add_committed("ps1");
        let ps2: Col<B8> = sct.add_committed("ps2");

        let ext1: B32 = ext_basis::<B32, B8>(1);
        let ext2: B32 = ext_basis::<B32, B8>(2);

        sct.assert_zero("step_pack",
            sc_step + upcast_col::<B32, _, 1>(s0)
                    + upcast_col::<B32, _, 1>(s1) * ext1
                    + upcast_col::<B32, _, 1>(s2) * ext2);
        sct.assert_zero("prev_pack",
            sc_prev + upcast_col::<B32, _, 1>(ps0)
                    + upcast_col::<B32, _, 1>(ps1) * ext1
                    + upcast_col::<B32, _, 1>(ps2) * ext2);

        let eq2: Col<B1> = sct.add_committed("eq2");
        let eq1: Col<B1> = sct.add_committed("eq1");
        let diff2_inv: Col<B8> = sct.add_committed("diff2_inv");
        let diff1_inv: Col<B8> = sct.add_committed("diff1_inv");

        let eq2_b8: Col<B8> = upcast_col(eq2);
        let eq1_b8: Col<B8> = upcast_col(eq1);

        sct.assert_zero("eq2_det", (s2 + ps2) * diff2_inv + eq2_b8 + B8::ONE);
        sct.assert_zero("eq2_guard", eq2_b8 * (s2 + ps2));
        sct.assert_zero("eq1_det", (s1 + ps1) * diff1_inv + eq1_b8 + B8::ONE);
        sct.assert_zero("eq1_guard", eq1_b8 * (s1 + ps1));

        let cmp_a: Col<B8> = sct.add_committed("cmp_a");
        let cmp_b: Col<B8> = sct.add_committed("cmp_b");

        sct.assert_zero("cmp_a_sel",
            cmp_a + (eq2_b8 + B8::ONE) * s2
                  + eq2_b8 * (eq1_b8 + B8::ONE) * s1
                  + eq2_b8 * eq1_b8 * s0);
        sct.assert_zero("cmp_b_sel",
            cmp_b + (eq2_b8 + B8::ONE) * ps2
                  + eq2_b8 * (eq1_b8 + B8::ONE) * ps1
                  + eq2_b8 * eq1_b8 * ps0);

        let a_bits: [Col<B1>; 8] = std::array::from_fn(|_| sct.add_committed("a_bit"));
        let b_bits: [Col<B1>; 8] = std::array::from_fn(|_| sct.add_committed("b_bit"));

        {
            let b0: B8 = ext_basis::<B8, B1>(0);
            let b1: B8 = ext_basis::<B8, B1>(1);
            let b2: B8 = ext_basis::<B8, B1>(2);
            let b3: B8 = ext_basis::<B8, B1>(3);
            let b4: B8 = ext_basis::<B8, B1>(4);
            let b5: B8 = ext_basis::<B8, B1>(5);
            let b6: B8 = ext_basis::<B8, B1>(6);
            let b7: B8 = ext_basis::<B8, B1>(7);
            sct.assert_zero("a_bits_pack", cmp_a
                + upcast_col::<B8, _, 1>(a_bits[0]) * b0
                + upcast_col::<B8, _, 1>(a_bits[1]) * b1
                + upcast_col::<B8, _, 1>(a_bits[2]) * b2
                + upcast_col::<B8, _, 1>(a_bits[3]) * b3
                + upcast_col::<B8, _, 1>(a_bits[4]) * b4
                + upcast_col::<B8, _, 1>(a_bits[5]) * b5
                + upcast_col::<B8, _, 1>(a_bits[6]) * b6
                + upcast_col::<B8, _, 1>(a_bits[7]) * b7);
            sct.assert_zero("b_bits_pack", cmp_b
                + upcast_col::<B8, _, 1>(b_bits[0]) * b0
                + upcast_col::<B8, _, 1>(b_bits[1]) * b1
                + upcast_col::<B8, _, 1>(b_bits[2]) * b2
                + upcast_col::<B8, _, 1>(b_bits[3]) * b3
                + upcast_col::<B8, _, 1>(b_bits[4]) * b4
                + upcast_col::<B8, _, 1>(b_bits[5]) * b5
                + upcast_col::<B8, _, 1>(b_bits[6]) * b6
                + upcast_col::<B8, _, 1>(b_bits[7]) * b7);
        }

        let borrows: [Col<B1>; 8] = std::array::from_fn(|_| sct.add_committed("borrow"));

        sct.assert_zero("borrow_0",
            upcast_col::<B8, _, 1>(borrows[0])
            + B8::ONE
            + upcast_col::<B8, _, 1>(a_bits[0])
            + upcast_col::<B8, _, 1>(a_bits[0]) * upcast_col::<B8, _, 1>(b_bits[0]));

        for i in 1..8 {
            let ai = upcast_col::<B8, _, 1>(a_bits[i]);
            let bi = upcast_col::<B8, _, 1>(b_bits[i]);
            let ci = upcast_col::<B8, _, 1>(borrows[i - 1]);
            sct.assert_zero(&format!("borrow_{i}"),
                upcast_col::<B8, _, 1>(borrows[i])
                + bi + ai * bi
                + ci + ai * ci
                + bi * ci);
        }

        sct.assert_zero("no_underflow", upcast_col::<B8, _, 1>(borrows[7]) + B8::ZERO);
        drop(sct);
    }

    // PC-immediate lookup page-in.
    let mut pcimm_pi_table = cs.add_table("pc_imm_page_in");
    let pcimm_pi_pc: Col<B16> = pcimm_pi_table.add_committed("pc");
    let pcimm_pi_imm: Col<B32> = pcimm_pi_table.add_committed("imm");
    let pcimm_pi_packed: Col<B128> = pcimm_pi_table.add_committed_in_group("packed_pc_imm", PC_IMM_GROUP);
    let x32 = B128::new(1u128 << 32);
    pcimm_pi_table.assert_zero("decompose_pc_imm",
        pcimm_pi_packed
            - upcast_col::<B128, _, 1>(pcimm_pi_pc)
            - upcast_col::<B128, _, 1>(pcimm_pi_imm) * x32);
    const PC_IMM_MULT_BITS: usize = 11;
    LookupProducer::new(
        &mut pcimm_pi_table,
        pc_imm_chan,
        &[upcast_col::<B128, _, 1>(pcimm_pi_pc), upcast_col::<B128, _, 1>(pcimm_pi_imm)],
        PC_IMM_MULT_BITS,
    );
    drop(pcimm_pi_table);

    // PC-op lookup page-in.
    let mut pcop_pi_table = cs.add_table("pc_op_page_in");
    let pcop_pi_pc: Col<B16> = pcop_pi_table.add_committed("pc");
    let pcop_pi_op: Col<B8> = pcop_pi_table.add_committed("op");
    let pcop_pi_packed: Col<B128> = pcop_pi_table.add_committed_in_group("packed_pc_op", PC_OP_GROUP);
    let x16 = B128::new(1u128 << 16);
    pcop_pi_table.assert_zero("decompose_pc_op",
        pcop_pi_packed
            - upcast_col::<B128, _, 1>(pcop_pi_pc)
            - upcast_col::<B128, _, 1>(pcop_pi_op) * x16);
    const PC_OP_MULT_BITS: usize = 15;
    LookupProducer::new(
        &mut pcop_pi_table,
        pc_op_chan,
        &[upcast_col::<B128, _, 1>(pcop_pi_pc), upcast_col::<B128, _, 1>(pcop_pi_op)],
        PC_OP_MULT_BITS,
    );
    drop(pcop_pi_table);

    // PC-increment lookup table: all (pc, pc+1) pairs + (0,0) for padding.
    let mut pcinc_pi_table = cs.add_table("pc_inc_page_in");
    let pcinc_pi_pc: Col<B16> = pcinc_pi_table.add_committed("pc");
    let pcinc_pi_npc: Col<B16> = pcinc_pi_table.add_committed("next_pc");
    const PC_INC_MULT_BITS: usize = 15;
    LookupProducer::new(
        &mut pcinc_pi_table,
        pc_inc_chan,
        &[upcast_col::<B128, _, 1>(pcinc_pi_pc), upcast_col::<B128, _, 1>(pcinc_pi_npc)],
        PC_INC_MULT_BITS,
    );
    drop(pcinc_pi_table);

    // ==== 3. Rebuild boundaries ====
    let mut boundaries = Vec::new();
    for &(slot, value) in &boundary_pairs {
        boundaries.push(Boundary {
            values: vec![B128::from(B16::new(slot as u16)), B128::from(B32::new(value))],
            channel_id: pub_io_chan,
            direction: FlushDirection::Pull,
            multiplicity: 1,
        });
    }

    // ==== 4. Compile and verify ====
    let t0 = Instant::now();
    let compiled_cs = cs.compile().map_err(|e| anyhow::anyhow!("{e}"))?;
    let compile_time = t0.elapsed();

    let ccs_digest = compiled_cs.digest::<Blake3Digest>();
    let proof = Proof { transcript };

    let t0 = Instant::now();
    let standalone_commitments = verify::<
        OptimalUnderlier,
        CanonicalTowerFamily,
        Blake3Digest,
        Blake3Compression,
        HasherChallenger<Blake3Digest>,
    >(
        &compiled_cs,
        LOG_INV_RATE,
        SECURITY_BITS,
        &FRI_STRATEGY,
        &ccs_digest,
        &boundaries,
        proof,
    )?;
    let verify_time = t0.elapsed();

    // ==== 5. Independently verify standalone commitments for groups 2 and 3 ====
    // Read lookup trace to rebuild the deterministic page-in columns.
    let t0 = Instant::now();

    let mf = {
        let f = std::fs::File::open(&mem_path)?;
        MemTrace::read_from(&mut std::io::BufReader::new(f))?
    };
    let lt = {
        let f = std::fs::File::open(&lookup_path)?;
        LookupTrace::read_from(&mut std::io::BufReader::new(f))?
    };
    assert_eq!(mf.access.len(), lt.rows.len(),
        "mem_trace has {} rows but lookup_trace has {}", mf.access.len(), lt.rows.len());

    // Build unique (pc, imm) set — same order as prover.
    let mut pc_imm_set: Vec<(u16, u32)> = Vec::new();
    {
        let mut seen = HashSet::new();
        for (row_idx, lr) in lt.rows.iter().enumerate() {
            let has_imm = lr.has_imm != 0;
            let is_active = mf.access[row_idx].is_read || mf.access[row_idx].is_write;
            // Prover iterates all ops (including no-ops), checking has_imm.
            if has_imm && seen.insert((lr.pc, lr.imm as u32)) {
                pc_imm_set.push((lr.pc, lr.imm as u32));
            }
            let _ = is_active; // unused here but documents the logic
        }
    }

    // Build unique (pc, op) set — all ops plus (0,0) for padding, same order as prover.
    let mut pc_op_set: Vec<(u16, u8)> = Vec::new();
    {
        let mut seen = HashSet::new();
        for lr in &lt.rows {
            if seen.insert((lr.pc, lr.op)) {
                pc_op_set.push((lr.pc, lr.op));
            }
        }
        // Padding entry (0, 0) if not already present.
        if seen.insert((0u16, 0u8)) {
            pc_op_set.push((0, 0));
        }
    }

    // Pack into B128 columns and commit independently.
    let packed_pc_imm: Vec<B128> = pc_imm_set.iter()
        .map(|&(pc, imm)| B128::new((pc as u128) | ((imm as u128) << 32)))
        .collect();
    let packed_pc_op: Vec<B128> = pc_op_set.iter()
        .map(|&(pc, op)| B128::new((pc as u128) | ((op as u128) << 16)))
        .collect();

    let local_pc_imm_commit = commit_column_b128(
        &packed_pc_imm, LOG_INV_RATE, SECURITY_BITS, &FRI_STRATEGY,
    )?;
    let local_pc_op_commit = commit_column_b128(
        &packed_pc_op, LOG_INV_RATE, SECURITY_BITS, &FRI_STRATEGY,
    )?;
    let commit_time = t0.elapsed();

    // Compare against standalone commitments from proof.
    let proof_pc_imm = standalone_commitments
        .get(&PC_IMM_GROUP)
        .expect("pc_imm group should exist");
    let proof_pc_op = standalone_commitments
        .get(&PC_OP_GROUP)
        .expect("pc_op group should exist");

    println!("Standalone commitment checks:");
    println!("  group {} (pc_imm):  {}", PC_IMM_GROUP,
        if local_pc_imm_commit == proof_pc_imm.as_slice().to_vec() { "PASS" } else { "FAIL" });
    println!("  group {} (pc_op):   {}", PC_OP_GROUP,
        if local_pc_op_commit == proof_pc_op.as_slice().to_vec() { "PASS" } else { "FAIL" });

    // Group 1 (packed_pt) contains witness data with random dummy/noop rows —
    // the verifier cannot reconstruct it independently, but writes it
    // for cross-check by mac_consistency_verifier.
    let packed_pt_commit = standalone_commitments.get(&1)
        .expect("group 1 should exist").as_slice();
    let proof_dir = format!("proofs/{program}");
    std::fs::create_dir_all(&proof_dir)?;
    std::fs::write(format!("{proof_dir}/packed_pt_commit.bin"), packed_pt_commit)?;
    println!("  group 1 (packed_pt): {:02x?}  (written to proofs/{program}/packed_pt_commit.bin)", packed_pt_commit);

    // Group 4 (step): deterministic column [0, 1, 2, ..., total_ops-1, 0-pad].
    let total_ops = mf.access.len();
    let step_values: Vec<B32> = (0..total_ops)
        .map(|i| B32::new((i + 1) as u32))
        .collect();
    let local_step_commit = commit_column_b32(
        &step_values, LOG_INV_RATE, SECURITY_BITS, &FRI_STRATEGY,
    )?;
    let proof_step = standalone_commitments
        .get(&4)
        .expect("step group should exist");
    println!("  group 4 (step):      {}",
        if local_step_commit == proof_step.as_slice().to_vec() { "PASS" } else { "FAIL" });
    println!();

    assert_eq!(local_pc_imm_commit, proof_pc_imm.as_slice().to_vec(),
        "pc_imm standalone commitment mismatch!");
    assert_eq!(local_pc_op_commit, proof_pc_op.as_slice().to_vec(),
        "pc_op standalone commitment mismatch!");
    assert_eq!(local_step_commit, proof_step.as_slice().to_vec(),
        "step standalone commitment mismatch!");

    let total_time = t_total.elapsed();

    println!("=== Verifier stats ===");
    println!("  CS compile:     {:?}", compile_time);
    println!("  Proof verify:   {:?}", verify_time);
    println!("  Commit check:   {:?}", commit_time);
    println!("  Total:          {:?}", total_time);
    println!();

    println!("Proof verified successfully!");

    Ok(())
}
