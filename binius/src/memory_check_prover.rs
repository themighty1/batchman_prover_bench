//! Memory checker on binius_m3 (no IT-MACs).
//!
//! This checker can be run before the verifier reveals its delta. The prover
//! commits the important columns (addr, value, step, etc.) inside binius —
//! these are the values whose MACs the prover commits to during the Batchman
//! protocol before learning delta. Later, when delta is revealed, an
//! additional MAC consistency proof is created to bind the committed columns
//! to their IT-MAC authenticators.
//!
//! Single-table architecture:
//!   - One main table with all operations.
//!   - Memory tuples: (addr, value, access_step, blind).
//!     access_step = the execution step (B32) of this access; page-in starts at 0.
//!   - prev_access_step is witnessed per-row for pulls.
//!
//! Zero-knowledge considerations:
//!   No-op rows are zero-filled (packed_pt matches mac_consistency zero-padding).
//!   fri_blind (B128) masks FRI codewords.
//!   mem_blind (per-address) masks channel grand products.
//!
//! Run with:
//!   cargo run --release --bin checker_no_macs [MEMORY_BIN_PATH]

use std::time::Instant;

use anyhow::Result;
use batchman_witness_generator::{LookupTrace, MemTrace};
use binius_compute::{cpu::alloc::CpuComputeAllocator, ComputeHolder};
use binius_core::{
    constraint_system::{FriStrategy, prove},
    fiat_shamir::HasherChallenger,
};
use binius_fast_compute::layer::FastCpuLayerHolder;
use binius_field::{
    arch::OptimalUnderlier, as_packed_field::PackedType,
    ext_basis, tower::CanonicalTowerFamily,
    Field, PackedExtension, PackedField,
};
use binius_hal::make_portable_backend;
use binius_m3::builder::{
    upcast_col, Boundary, Col, ConstraintSystem, FlushDirection, FlushOpts, TableId,
    WitnessIndex, B1, B8, B16, B128, B32,
};
use binius_m3::gadgets::lookup::LookupProducer;
use memory_checker_and_lookup::{Blake3Digest, Blake3Compression};
use rand::{rngs::StdRng, Rng, SeedableRng};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

// ==== Constants ====

const MEMORY_SIZE: usize = 64 * 1024;         // 64KB (all byte addrs fit in u16)
const NUM_ADDRESSES: usize = MEMORY_SIZE / 4;  // 16384 u32 slots
const LOG_INV_RATE: usize = 2;
const SECURITY_BITS: usize = 100;
const FRI_STRATEGY: FriStrategy = FriStrategy::ConstantArity(8);
const IO_OUTPUT_LEN_SLOT: usize = 0x1000 / 4;   // byte 0x1000 -> u32 slot 1024
const IO_OUTPUT_DATA_SLOT: usize = 0x4000 / 4;   // byte 0x4000 -> u32 slot 4096

type P = PackedType<OptimalUnderlier, B128>;
type PackedB1 = <P as PackedExtension<B1>>::PackedSubfield;

/// Column handles for a main table.
struct MainTableCols {
    addr: Col<B16>,
    value: Col<B32>,
    prev_value: Col<B32>,
    is_write: Col<B1>,
    is_read: Col<B1>,
    pc: Col<B16>,
    next_pc: Col<B16>,
    op: Col<B8>,
    imm: Col<B32>,
    has_imm: Col<B1>,
    is_byte_sel_r2: Col<B1>,
    step: Col<B32>,
    r0_b0: Col<B8>,
    r0_b1: Col<B8>,
    r0_b2: Col<B8>,
    r0_b3: Col<B8>,
    r2_idx0: Col<B1>,
    r2_idx1: Col<B1>,
    new_r0: Col<B32>,
    prev_access_step: Col<B32>,
    packed_pt: Col<B128>,
    mem_blind: Col<B128>,
    fri_blind: Col<B128>,
    fri_blind_sq: Col<B128>,
}

/// A resolved memory operation with all fields needed for the witness.
struct ResolvedOp {
    addr: u16,
    value: u32,
    prev_value: u32,
    is_write: bool,
    is_read: bool,
    pc: u16,
    next_pc: u16,
    op: u8,
    imm: u32,
    has_imm: bool,
    is_byte_sel_r2: bool,
    step: u32,
}

/// Trace: resolved ops plus initial and final memory state.
struct Trace {
    ops: Vec<ResolvedOp>,
    init_mem: Vec<u32>,
    final_mem: Vec<u32>,
}

/// Load a MemoryFile + LookupTrace and replay accesses to derive prev_value for each op.
fn load_trace(mem_path: &str, lookup_path: &str) -> Result<Trace> {
    let file = std::fs::File::open(mem_path)?;
    let mut reader = std::io::BufReader::new(file);
    let mf = MemTrace::read_from(&mut reader)?;

    let lf = std::fs::File::open(lookup_path)?;
    let mut lreader = std::io::BufReader::new(lf);
    let lt = LookupTrace::read_from(&mut lreader)?;
    assert_eq!(mf.access.len(), lt.rows.len(),
        "mem_trace has {} rows but lookup_trace has {}", mf.access.len(), lt.rows.len());

    // Build initial memory from regions.
    let mut init_mem = vec![0u32; NUM_ADDRESSES];
    for region in &mf.initial {
        let base_byte = region.addr as usize;
        assert!(base_byte % 4 == 0, "region addr {base_byte} not u32-aligned");
        let base_slot = base_byte / 4;
        for (i, chunk) in region.data.chunks(4).enumerate() {
            let slot = base_slot + i;
            assert!(slot < NUM_ADDRESSES, "region overflows memory at slot {slot}");
            let mut word = 0u32;
            for (j, &b) in chunk.iter().enumerate() {
                word |= (b as u32) << (j * 8);
            }
            init_mem[slot] = word;
        }
    }

    println!("Loaded memory trace from {mem_path}:");
    for (i, r) in mf.initial.iter().enumerate() {
        println!("  region [{i}] byte {:#x}..{:#x} ({} bytes)",
            r.addr, r.addr as usize + r.data.len(), r.data.len());
    }
    println!("  access rows: {}", mf.access.len());
    println!("Loaded lookup trace from {lookup_path}: {} steps", lt.rows.len());

    // Replay accesses to derive prev_value.
    let mut mem = init_mem.clone();
    let mut ops = Vec::with_capacity(mf.access.len());
    let mut step_counter: u32 = 1;  // start at 1; page-in uses step=0
    for (row_idx, row) in mf.access.iter().enumerate() {
        let lr = &lt.rows[row_idx];
        let pc = lr.pc;
        let op = lr.op;
        let imm = lr.imm as u32;
        let has_imm = lr.has_imm != 0;
        let is_byte_sel_r2 = lr.is_byte_sel_r2 != 0;
        if row.is_read {
            let addr = row.read_addr as u16;
            let slot = (addr / 4) as usize;
            let value = mem[slot];
            assert_eq!(value, row.read_value,
                "read mismatch at row {row_idx} addr {addr:#x}: mem has {value}, trace says {}",
                row.read_value);
            ops.push(ResolvedOp {
                addr, value, prev_value: value,
                is_write: false, is_read: true,
                pc, next_pc: pc.wrapping_add(1),
                op, imm, has_imm, is_byte_sel_r2,
                step: step_counter,
            });
        } else if row.is_write {
            let addr = row.write_addr as u16;
            let slot = (addr / 4) as usize;
            let prev_value = mem[slot];
            let value = row.write_value;
            mem[slot] = value;
            ops.push(ResolvedOp {
                addr, value, prev_value,
                is_write: true, is_read: false,
                pc, next_pc: pc.wrapping_add(1),
                op, imm, has_imm, is_byte_sel_r2,
                step: step_counter,
            });
        } else {
            // No memory access — still an execution step for pc_inc.
            ops.push(ResolvedOp {
                addr: 0u16, value: 0, prev_value: 0,
                is_write: false, is_read: false,
                pc, next_pc: pc.wrapping_add(1),
                op, imm, has_imm, is_byte_sel_r2,
                step: step_counter,
            });
        }
        step_counter += 1;
    }

    let final_mem = mem;
    Ok(Trace { ops, init_mem, final_mem })
}

/// Build a main table with all columns, constraints, and channel flushes.
fn build_main_table(
    cs: &mut ConstraintSystem,
    name: &str,
    mem_chan: usize,
    step_cmp_chan: usize,
    pc_imm_chan: usize,
    pc_op_chan: usize,
    pc_inc_chan: usize,
) -> (TableId, MainTableCols) {
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
    // Tower-optimized layout (all sub-fields at tower-aligned positions):
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

    let id = table.id();
    drop(table);

    (id, MainTableCols {
        addr: c_addr, value: c_value, prev_value: c_prev_value,
        is_write, is_read,
        pc: c_pc, next_pc: c_next_pc, op: c_op, imm: c_imm, has_imm: c_has_imm,
        is_byte_sel_r2: c_is_byte_sel_r2,
        step: c_step,
        r0_b0: c_r0_b0, r0_b1: c_r0_b1, r0_b2: c_r0_b2, r0_b3: c_r0_b3,
        r2_idx0: c_r2_idx0, r2_idx1: c_r2_idx1, new_r0: c_new_r0,
        packed_pt: c_packed_pt,
        prev_access_step: c_prev_access_step,
        mem_blind: c_mem_blind, fri_blind: c_fri_blind, fri_blind_sq: c_fri_blind_sq,
    })
}

fn main() -> Result<()> {
    // The 65537-entry pc_inc table causes deep recursion in binius compile/prove.
    // Spawn on a thread with 16 MB stack to avoid overflow.
    let builder = std::thread::Builder::new().stack_size(16 * 1024 * 1024);
    let handler = builder.spawn(run_prover).unwrap();
    handler.join().unwrap()
}

fn run_prover() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            fmt::layer()
                .with_timer(fmt::time::uptime())
                .with_span_events(fmt::format::FmtSpan::CLOSE),
        )
        .with(EnvFilter::from_default_env())
        .init();

    // Load memory trace and lookup trace files.
    let program = std::env::args().nth(1).unwrap_or_else(|| "json-query".to_string());
    let witness_dir = format!("../witgen/witness/{program}");
    let mem_path = format!("{witness_dir}/mem_trace.bin");
    let lookup_path = format!("{witness_dir}/lookup_trace.bin");
    let packed_pt_path = format!("{witness_dir}/packed_pt.bin");
    let mut trace = load_trace(&mem_path, &lookup_path)?;

    // Read packed_pt.bin (ground truth from witgen).
    // Format: u32 LE count, then count × 16-byte LE u128 entries.
    let packed_pts: Vec<B128> = {
        let data = std::fs::read(&packed_pt_path)?;
        let count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        (0..count)
            .map(|i| {
                let off = 4 + i * 16;
                let val = u128::from_le_bytes(data[off..off + 16].try_into().unwrap());
                B128::new(val)
            })
            .collect()
    };

    let total_ops = trace.ops.len();
    assert_eq!(total_ops, packed_pts.len(),
        "mem_trace has {} ops but packed_pt has {}", total_ops, packed_pts.len());
    let num_writes = trace.ops.iter().filter(|op| op.is_write).count();
    let num_reads = trace.ops.iter().filter(|op| op.is_read).count();

    // Size table: next power of 2 >= total_ops. No dummy rows.
    // Padding rows are zero-filled so packed_pt commitment matches mac_consistency.
    let num_rows = total_ops.next_power_of_two();

    let mut rng = StdRng::seed_from_u64(42);

    // Backfill value for byte_sel_r2 no-op rows from packed_pt.
    // Witgen packs the selected byte into the value field (low 32 bits of packed_pt).
    for i in 0..total_ops {
        if !trace.ops[i].is_read && !trace.ops[i].is_write && trace.ops[i].is_byte_sel_r2 {
            trace.ops[i].value = packed_pts[i].val() as u32;
        }
    }

    // Build table_ops: real ops then None padding (zero-filled).
    let table_ops: Vec<Option<&ResolvedOp>> = {
        let mut v: Vec<Option<&ResolvedOp>> = trace.ops.iter().map(Some).collect();
        v.resize(num_rows, None);
        v
    };

    // Byte-select witness data: r0 (4 bytes), 2-bit index, new_r0.
    let bsel_r0: Vec<u32> = (0..num_rows).map(|_| rng.random()).collect();
    let bsel_idx: Vec<u8> = (0..num_rows).map(|_| rng.random_range(0u8..4)).collect();
    let bsel_new_r0: Vec<u32> = (0..num_rows)
        .map(|i| (bsel_r0[i] >> (bsel_idx[i] as u32 * 8)) & 0xFF)
        .collect();

    let mem_blinds: Vec<B128> = (0..NUM_ADDRESSES)
        .map(|_| B128::new(rng.random::<u128>()))
        .collect();

    // Per-address last-access-step tracking.  Page-in starts at step 0;
    // each access records the current execution step.
    let mut addr_last_step = vec![0u32; NUM_ADDRESSES];
    let mut row_prev_access_steps = vec![0u32; num_rows];

    for (i, row) in table_ops.iter().enumerate() {
        if let Some(op) = row {
            if op.is_read || op.is_write {
                let slot = (op.addr / 4) as usize;
                row_prev_access_steps[i] = addr_last_step[slot];
                addr_last_step[slot] = op.step;
            }
        }
    }
    let final_access_steps = addr_last_step;

    // Stats.
    println!(
        "Main table:       {} rows (ops: {}, padding: {})",
        num_rows, total_ops, num_rows - total_ops,
    );
    println!("  total ops:      {total_ops} (writes: {num_writes}, reads: {num_reads})");
    println!("Page-in table:    {NUM_ADDRESSES} rows");
    println!("Page-out table:   {NUM_ADDRESSES} rows");
    println!();

    // Build unique (pc, imm) lookup map from all ops with has_imm.
    let mut pc_imm_set: Vec<(u16, u32)> = Vec::new();
    let mut pc_imm_counts: Vec<u32> = Vec::new();
    {
        use std::collections::HashMap;
        let mut index_map: HashMap<(u16, u32), usize> = HashMap::new();
        for row in &table_ops {
            if let Some(op) = row {
                if op.has_imm {
                    if let Some(&idx) = index_map.get(&(op.pc, op.imm)) {
                        pc_imm_counts[idx] += 1;
                    } else {
                        let idx = pc_imm_set.len();
                        index_map.insert((op.pc, op.imm), idx);
                        pc_imm_set.push((op.pc, op.imm));
                        pc_imm_counts.push(1);
                    }
                }
            }
        }
    }
    println!("PC-imm lookup:    {} unique (pc, imm) pairs (max count: {})",
        pc_imm_set.len(), pc_imm_counts.iter().max().unwrap_or(&0));

    // Build unique (pc, op) lookup map with access counts (every row, including padding).
    let mut pc_op_set: Vec<(u16, u8)> = Vec::new();
    let mut pc_op_counts: Vec<u32> = Vec::new();
    {
        use std::collections::HashMap;
        let mut index_map: HashMap<(u16, u8), usize> = HashMap::new();
        // Count all active ops.
        for row in &table_ops {
            if let Some(op) = row {
                if let Some(&idx) = index_map.get(&(op.pc, op.op)) {
                    pc_op_counts[idx] += 1;
                } else {
                    let idx = pc_op_set.len();
                    index_map.insert((op.pc, op.op), idx);
                    pc_op_set.push((op.pc, op.op));
                    pc_op_counts.push(1);
                }
            }
        }
        // Padding rows have (pc=0, op=0).
        let padding_count = (num_rows - total_ops) as u32;
        if let Some(&idx) = index_map.get(&(0u16, 0u8)) {
            pc_op_counts[idx] += padding_count;
        } else {
            pc_op_set.push((0, 0));
            pc_op_counts.push(padding_count);
        }
    }
    println!("PC-op lookup:     {} unique (pc, op) pairs (max count: {})",
        pc_op_set.len(), pc_op_counts.iter().max().unwrap_or(&0));

    // Build (pc, next_pc) increment lookup with counts (every row including padding).
    // The table contains all 65536 pairs (i, (i+1) % 65536) plus (0, 0) for padding.
    let pc_inc_entries: Vec<(u16, u16)> = (0..65536u32)
        .map(|i| (i as u16, ((i + 1) % 65536) as u16))
        .collect();
    // (0, 0) is needed for padding rows where pc=0, next_pc=0.
    // It's NOT already in pc_inc_entries (entry 0 is (0, 1)), so append it.
    let mut pc_inc_set: Vec<(u16, u16)> = pc_inc_entries.clone();
    pc_inc_set.push((0, 0));
    let mut pc_inc_counts: Vec<u32> = vec![0u32; pc_inc_set.len()]; // 65537 entries
    {
        use std::collections::HashMap;
        let mut index_map: HashMap<(u16, u16), usize> = HashMap::new();
        for (i, &pair) in pc_inc_set.iter().enumerate() {
            index_map.insert(pair, i);
        }
        for row in &table_ops {
            if let Some(op) = row {
                let key = (op.pc, op.next_pc);
                if let Some(&idx) = index_map.get(&key) {
                    pc_inc_counts[idx] += 1;
                } else {
                    panic!("(pc={}, next_pc={}) not in increment table", op.pc, op.next_pc);
                }
            }
        }
        // Padding rows: (0, 0).
        let padding_count = (num_rows - total_ops) as u32;
        let padding_idx = *index_map.get(&(0u16, 0u16)).unwrap();
        pc_inc_counts[padding_idx] += padding_count;
    }
    println!("PC-inc lookup:    {} entries (max count: {})",
        pc_inc_set.len(), pc_inc_counts.iter().max().unwrap_or(&0));

    // ==== Build constraint system ====
    let mut cs = ConstraintSystem::new();
    let mem_chan = cs.add_channel("memory");
    let step_cmp_chan = cs.add_channel("step_cmp");
    let pc_imm_chan = cs.add_channel("pc_immediate");
    let pc_op_chan = cs.add_channel("pc_op");
    let pc_inc_chan = cs.add_channel("pc_inc");
    // ---- Main table ----
    let (main_id, cols) = build_main_table(&mut cs, "main", mem_chan, step_cmp_chan, pc_imm_chan, pc_op_chan, pc_inc_chan);

    // ---- Page-in table ----
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
    let pi_table_id = pi_table.id();
    drop(pi_table);

    // ---- Page-out table ----
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
    // Push (addr, value) to pub_io for output rows.
    po_table.push_with_opts(
        pub_io_chan,
        [upcast_col::<B128, _, 1>(po_addr), upcast_col::<B128, _, 1>(po_value)],
        FlushOpts { selectors: vec![po_is_output], ..Default::default() },
    );
    let po_table_id = po_table.id();
    drop(po_table);

    // ---- Step comparison table (byte-limb decomposition + 8-bit GT) ----
    // Proves step > prev_access_step for every active (read/write) row.
    // Main table pushes (step, prev_access_step); this table pulls and verifies.
    //
    // Approach: decompose both B32 values into 3 B8 limbs (17 bits, byte 3 = 0).
    // Find the most-significant differing byte via eq2/eq1 flags.
    // Then prove cmp_a > cmp_b (the deciding byte pair) via an 8-bit borrow chain.
    let step_cmp_table_id;
    #[allow(non_snake_case)]
    let SC;
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

        // Byte decomposition: step = s0 + s1*ext1 + s2*ext2 (byte 3 implicitly 0)
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

        // Equality flags: eq2=1 iff s2==ps2, eq1=1 iff s1==ps1
        let eq2: Col<B1> = sct.add_committed("eq2");
        let eq1: Col<B1> = sct.add_committed("eq1");
        let diff2_inv: Col<B8> = sct.add_committed("diff2_inv");
        let diff1_inv: Col<B8> = sct.add_committed("diff1_inv");

        let eq2_b8: Col<B8> = upcast_col(eq2);
        let eq1_b8: Col<B8> = upcast_col(eq1);

        // (s2+ps2)*diff2_inv + eq2 = 1  and  eq2*(s2+ps2) = 0
        sct.assert_zero("eq2_det", (s2 + ps2) * diff2_inv + eq2_b8 + B8::ONE);
        sct.assert_zero("eq2_guard", eq2_b8 * (s2 + ps2));
        sct.assert_zero("eq1_det", (s1 + ps1) * diff1_inv + eq1_b8 + B8::ONE);
        sct.assert_zero("eq1_guard", eq1_b8 * (s1 + ps1));

        // Deciding pair: most significant differing byte
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

        // 8-bit unsigned GT: prove cmp_a > cmp_b via borrow chain on (cmp_a - cmp_b - 1).
        let a_bits: [Col<B1>; 8] = std::array::from_fn(|_| sct.add_committed("a_bit"));
        let b_bits: [Col<B1>; 8] = std::array::from_fn(|_| sct.add_committed("b_bit"));

        // Constrain bit decompositions
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

        // Borrow chain
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

        SC = (
            sc_step, sc_prev, sc_is_active, sc_fri_blind, sc_fri_blind_sq,
            s0, s1, s2, ps0, ps1, ps2,
            eq2, eq1, diff2_inv, diff1_inv,
            cmp_a, cmp_b, a_bits, b_bits, borrows,
        );
        step_cmp_table_id = sct.id();
        drop(sct);
    }

    // ---- PC-immediate lookup page-in: push all unique (pc, imm) pairs ----
    const PC_IMM_GROUP: u32 = 2;
    let mut pcimm_pi_table = cs.add_table("pc_imm_page_in");
    let pcimm_pi_pc: Col<B16> = pcimm_pi_table.add_committed("pc");
    let pcimm_pi_imm: Col<B32> = pcimm_pi_table.add_committed("imm");
    let pcimm_pi_packed: Col<B128> = pcimm_pi_table.add_committed_in_group("packed_pc_imm", PC_IMM_GROUP);
    // Decomposition: packed_pc_imm = pc + imm·X32
    // (B32 must be shifted by ≥ level-5 basis; X16 = α₄ is internal to B32)
    let x32 = B128::new(1u128 << 32);
    pcimm_pi_table.assert_zero("decompose_pc_imm",
        pcimm_pi_packed
            - upcast_col::<B128, _, 1>(pcimm_pi_pc)
            - upcast_col::<B128, _, 1>(pcimm_pi_imm) * x32);
    const PC_IMM_MULT_BITS: usize = 11; // max 2047 accesses per (pc, imm) pair
    let pcimm_lookup = LookupProducer::new(
        &mut pcimm_pi_table,
        pc_imm_chan,
        &[upcast_col::<B128, _, 1>(pcimm_pi_pc), upcast_col::<B128, _, 1>(pcimm_pi_imm)],
        PC_IMM_MULT_BITS,
    );
    let pcimm_pi_id = pcimm_pi_table.id();
    drop(pcimm_pi_table);

    // ---- PC-op lookup page-in: push all unique (pc, op) pairs ----
    const PC_OP_GROUP: u32 = 3;
    let mut pcop_pi_table = cs.add_table("pc_op_page_in");
    let pcop_pi_pc: Col<B16> = pcop_pi_table.add_committed("pc");
    let pcop_pi_op: Col<B8> = pcop_pi_table.add_committed("op");
    let pcop_pi_packed: Col<B128> = pcop_pi_table.add_committed_in_group("packed_pc_op", PC_OP_GROUP);
    // Decomposition: packed_pc_op = pc + op·X16
    // (B8 at level 3, X16 = α₄ at level 4 ≥ 3, so this is a clean shift)
    let x16 = B128::new(1u128 << 16);
    pcop_pi_table.assert_zero("decompose_pc_op",
        pcop_pi_packed
            - upcast_col::<B128, _, 1>(pcop_pi_pc)
            - upcast_col::<B128, _, 1>(pcop_pi_op) * x16);
    const PC_OP_MULT_BITS: usize = 15; // max 32767 accesses per (pc, op) pair
    let pcop_lookup = LookupProducer::new(
        &mut pcop_pi_table,
        pc_op_chan,
        &[upcast_col::<B128, _, 1>(pcop_pi_pc), upcast_col::<B128, _, 1>(pcop_pi_op)],
        PC_OP_MULT_BITS,
    );
    let pcop_pi_id = pcop_pi_table.id();
    drop(pcop_pi_table);

    // ---- PC-increment lookup table: all (pc, pc+1) pairs + (0,0) for padding ----
    let mut pcinc_pi_table = cs.add_table("pc_inc_page_in");
    let pcinc_pi_pc: Col<B16> = pcinc_pi_table.add_committed("pc");
    let pcinc_pi_npc: Col<B16> = pcinc_pi_table.add_committed("next_pc");
    const PC_INC_MULT_BITS: usize = 15; // max 32767 accesses per (pc, next_pc) pair
    let pcinc_lookup = LookupProducer::new(
        &mut pcinc_pi_table,
        pc_inc_chan,
        &[upcast_col::<B128, _, 1>(pcinc_pi_pc), upcast_col::<B128, _, 1>(pcinc_pi_npc)],
        PC_INC_MULT_BITS,
    );
    let pcinc_pi_id = pcinc_pi_table.id();
    drop(pcinc_pi_table);

    // ==== Fill witnesses ====
    let mut allocator = CpuComputeAllocator::new(1 << 24);
    let allocator = allocator.into_bump_allocator();
    let mut witness = WitnessIndex::<P>::new(&cs, &allocator);

    // ---- Fill main table ----
    {
        let c = &cols;
        let tw = witness
            .init_table(main_id, num_rows)
            .expect("init main table");
        let seg = tw.full_segment();

        {
            let mut a_col = seg.get_scalars_mut(c.addr)?;
            let mut v_col = seg.get_scalars_mut(c.value)?;
            let mut pv_col = seg.get_scalars_mut(c.prev_value)?;
            let mut pc_col = seg.get_scalars_mut(c.pc)?;
            let mut npc_col = seg.get_scalars_mut(c.next_pc)?;
            let mut op_col = seg.get_scalars_mut(c.op)?;
            let mut imm_col = seg.get_scalars_mut(c.imm)?;
            let mut step_col = seg.get_scalars_mut(c.step)?;
            let mut b0_col = seg.get_scalars_mut(c.r0_b0)?;
            let mut b1_col = seg.get_scalars_mut(c.r0_b1)?;
            let mut b2_col = seg.get_scalars_mut(c.r0_b2)?;
            let mut b3_col = seg.get_scalars_mut(c.r0_b3)?;
            let mut nr0_col = seg.get_scalars_mut(c.new_r0)?;
            // Scalar columns: real ops get actual values, padding stays zero.
            for (i, row) in table_ops.iter().enumerate() {
                if let Some(op) = row {
                    a_col[i] = B16::new(op.addr);
                    v_col[i] = B32::new(op.value);
                    pv_col[i] = B32::new(op.prev_value);
                    pc_col[i] = B16::new(op.pc);
                    npc_col[i] = B16::new(op.next_pc);
                    op_col[i] = B8::new(op.op);
                    imm_col[i] = B32::new(op.imm);
                    step_col[i] = B32::new(op.step);
                }
                // Byte-select data: consistent for all rows.
                let r0 = bsel_r0[i];
                b0_col[i] = B8::new((r0 & 0xFF) as u8);
                b1_col[i] = B8::new(((r0 >> 8) & 0xFF) as u8);
                b2_col[i] = B8::new(((r0 >> 16) & 0xFF) as u8);
                b3_col[i] = B8::new(((r0 >> 24) & 0xFF) as u8);
                nr0_col[i] = B32::new(bsel_new_r0[i]);
            }
        }

        {
            let mut wr = seg.get_mut(c.is_write)?;
            let mut rd = seg.get_mut(c.is_read)?;
            let mut hi = seg.get_mut(c.has_imm)?;
            let mut bsr = seg.get_mut(c.is_byte_sel_r2)?;
            let mut ri0 = seg.get_mut(c.r2_idx0)?;
            let mut ri1 = seg.get_mut(c.r2_idx1)?;
            for idx in 0..wr.len() {
                for k in 0..PackedB1::WIDTH {
                    let row = idx * PackedB1::WIDTH + k;
                    if row < num_rows {
                        // r2_idx bits for all rows (must match bsel_idx).
                        let idx_val = bsel_idx[row];
                        ri0[idx].set(k, B1::from(idx_val & 1 != 0));
                        ri1[idx].set(k, B1::from(idx_val & 2 != 0));

                        if let Some(op) = table_ops[row] {
                            wr[idx].set(k, B1::from(op.is_write));
                            rd[idx].set(k, B1::from(op.is_read));
                            hi[idx].set(k, B1::from(op.has_imm));
                            bsr[idx].set(k, B1::from(op.is_byte_sel_r2));
                        }
                    }
                }
            }
        }

        {
            let mut blind_col = seg.get_scalars_mut(c.fri_blind)?;
            let mut blind_sq_col = seg.get_scalars_mut(c.fri_blind_sq)?;
            for i in 0..num_rows {
                let r = B128::new(rng.random::<u128>());
                blind_col[i] = r;
                blind_sq_col[i] = r * r;
            }
        }

        {
            let mut mb_col = seg.get_scalars_mut(c.mem_blind)?;
            for (i, row) in table_ops.iter().enumerate() {
                if let Some(op) = row {
                    mb_col[i] = mem_blinds[(op.addr / 4) as usize];
                }
                // Padding rows: mem_blind stays zero.
            }
        }

        {
            let mut pas_col = seg.get_scalars_mut(c.prev_access_step)?;
            for (i, row) in table_ops.iter().enumerate() {
                if let Some(op) = row {
                    if op.is_read || op.is_write {
                        pas_col[i] = B32::new(row_prev_access_steps[i]);
                    } else {
                        pas_col[i] = B32::new(rng.random::<u32>());
                    }
                } else {
                    pas_col[i] = B32::new(rng.random::<u32>());
                }
            }
        }

        // Fill packed_pt: read back from already-filled scalar + B1 columns.
        {
            let v_col = seg.get_scalars_mut(c.value)?;
            let imm_col = seg.get_scalars_mut(c.imm)?;
            let pc_col = seg.get_scalars_mut(c.pc)?;
            let npc_col = seg.get_scalars_mut(c.next_pc)?;
            let a_col = seg.get_scalars_mut(c.addr)?;
            let op_col = seg.get_scalars_mut(c.op)?;
            let wr = seg.get_mut(c.is_write)?;
            let rd = seg.get_mut(c.is_read)?;
            let hi = seg.get_mut(c.has_imm)?;
            let bsr = seg.get_mut(c.is_byte_sel_r2)?;
            let mut pt_col = seg.get_scalars_mut(c.packed_pt)?;
            for i in 0..total_ops {
                let value_val = v_col[i].val() as u128;
                let imm_val = imm_col[i].val() as u128;
                let pc_val = pc_col[i].val() as u128;
                let npc_val = npc_col[i].val() as u128;
                let addr_val = a_col[i].val() as u128;
                let op_val = op_col[i].val() as u128;
                let packed_idx = i / PackedB1::WIDTH;
                let packed_k = i % PackedB1::WIDTH;
                let is_w = u8::from(wr[packed_idx].get(packed_k).val()) as u128;
                let is_r = u8::from(rd[packed_idx].get(packed_k).val()) as u128;
                let h_imm = u8::from(hi[packed_idx].get(packed_k).val()) as u128;
                let b_sel = u8::from(bsr[packed_idx].get(packed_k).val()) as u128;
                let pt = B128::new(
                    value_val
                    | (imm_val << 32)
                    | (pc_val << 64)
                    | (npc_val << 80)
                    | (addr_val << 96)
                    | (op_val << 112)
                    | (is_w << 120)
                    | (is_r << 121)
                    | (h_imm << 122)
                    | (b_sel << 123)
                );
                assert_eq!(pt, packed_pts[i],
                    "packed_pt mismatch at row {i}: memory_check={:?} step_record={:?}",
                    pt, packed_pts[i]);
                pt_col[i] = pt;
            }
            // Padding rows: packed_pt stays zero (matches mac_consistency's binius zero-padding).
            println!("Packed PT cross-check: all {} rows match packed_pt.bin", total_ops);
        }

    }

    // ---- Fill page-in table ----
    let pi_table_size = NUM_ADDRESSES.next_power_of_two();
    {
        let tw = witness.init_table(pi_table_id, pi_table_size).expect("init page-in table");
        let seg = tw.full_segment();
        {
            let mut a_col = seg.get_scalars_mut(pi_addr)?;
            let mut v_col = seg.get_scalars_mut(pi_value)?;
            let mut s_col = seg.get_scalars_mut(pi_step)?;
            let mut b_col = seg.get_scalars_mut(pi_blind)?;
            for slot in 0..NUM_ADDRESSES {
                a_col[slot] = B16::new((slot * 4) as u16);
                v_col[slot] = B32::new(trace.init_mem[slot]);
                s_col[slot] = B32::ZERO;
                b_col[slot] = mem_blinds[slot];
            }
        }
        {
            let mut blind_col = seg.get_scalars_mut(pi_fri_blind)?;
            let mut blind_sq_col = seg.get_scalars_mut(pi_fri_blind_sq)?;
            for i in 0..pi_table_size {
                let r = B128::new(rng.random::<u128>());
                blind_col[i] = r;
                blind_sq_col[i] = r * r;
            }
        }
    }

    // ---- Fill page-out table ----
    // Determine output slots from final memory.
    let output_len = trace.final_mem[IO_OUTPUT_LEN_SLOT] as usize;
    let output_num_words = (output_len + 3) / 4;
    let mut output_slots: Vec<usize> = Vec::new();
    output_slots.push(IO_OUTPUT_LEN_SLOT);
    for i in 0..output_num_words {
        output_slots.push(IO_OUTPUT_DATA_SLOT + i);
    }
    {
        let output_bytes: Vec<u8> = (0..output_num_words)
            .flat_map(|i| trace.final_mem[IO_OUTPUT_DATA_SLOT + i].to_le_bytes())
            .take(output_len)
            .collect();
        println!("Public output:    {} bytes {:02x?}", output_len, output_bytes);
        if let Ok(s) = std::str::from_utf8(&output_bytes) {
            println!("  as string:      {:?}", s);
        }
    }

    let po_table_size = NUM_ADDRESSES.next_power_of_two();
    {
        let tw = witness.init_table(po_table_id, po_table_size).expect("init page-out table");
        let seg = tw.full_segment();
        {
            let mut a_col = seg.get_scalars_mut(po_addr)?;
            let mut v_col = seg.get_scalars_mut(po_value)?;
            let mut s_col = seg.get_scalars_mut(po_step)?;
            let mut b_col = seg.get_scalars_mut(po_blind)?;
            for slot in 0..NUM_ADDRESSES {
                a_col[slot] = B16::new((slot * 4) as u16);
                v_col[slot] = B32::new(trace.final_mem[slot]);
                s_col[slot] = B32::new(final_access_steps[slot]);
                b_col[slot] = mem_blinds[slot];
            }
        }
        {
            let mut is_out = seg.get_mut(po_is_output)?;
            for (idx, packed) in is_out.iter_mut().enumerate() {
                for k in 0..PackedB1::WIDTH {
                    let row = idx * PackedB1::WIDTH + k;
                    if output_slots.contains(&row) {
                        packed.set(k, B1::from(true));
                    }
                }
            }
        }
        {
            let mut blind_col = seg.get_scalars_mut(po_fri_blind)?;
            let mut blind_sq_col = seg.get_scalars_mut(po_fri_blind_sq)?;
            for i in 0..po_table_size {
                let r = B128::new(rng.random::<u128>());
                blind_col[i] = r;
                blind_sq_col[i] = r * r;
            }
        }
    }

    // ---- Fill pc_imm lookup table ----
    let pcimm_table_size = pc_imm_set.len();
    {
        let tw = witness.init_table(pcimm_pi_id, pcimm_table_size).expect("init pc_imm page-in");
        let mut seg = tw.full_segment();
        {
            let mut pc_col = seg.get_scalars_mut(pcimm_pi_pc)?;
            let mut imm_col = seg.get_scalars_mut(pcimm_pi_imm)?;
            let mut packed_col = seg.get_scalars_mut(pcimm_pi_packed)?;
            for (i, &(pc, imm)) in pc_imm_set.iter().enumerate() {
                pc_col[i] = B16::new(pc);
                imm_col[i] = B32::new(imm);
                packed_col[i] = B128::new((pc as u128) | ((imm as u128) << 32));
            }
        }
        pcimm_lookup.populate(&mut seg, pc_imm_counts.iter().copied())?;
    }

    // ---- Fill pc_op lookup table ----
    let pcop_table_size = pc_op_set.len();
    {
        let tw = witness.init_table(pcop_pi_id, pcop_table_size).expect("init pc_op page-in");
        let mut seg = tw.full_segment();
        {
            let mut pc_col = seg.get_scalars_mut(pcop_pi_pc)?;
            let mut op_col = seg.get_scalars_mut(pcop_pi_op)?;
            let mut packed_col = seg.get_scalars_mut(pcop_pi_packed)?;
            for (i, &(pc, op)) in pc_op_set.iter().enumerate() {
                pc_col[i] = B16::new(pc);
                op_col[i] = B8::new(op);
                packed_col[i] = B128::new((pc as u128) | ((op as u128) << 16));
            }
        }
        pcop_lookup.populate(&mut seg, pc_op_counts.iter().copied())?;
    }

    // ---- Fill pc_inc lookup table ----
    let pcinc_table_size = pc_inc_set.len();
    {
        let tw = witness.init_table(pcinc_pi_id, pcinc_table_size).expect("init pc_inc page-in");
        let mut seg = tw.full_segment();
        {
            let mut pc_col = seg.get_scalars_mut(pcinc_pi_pc)?;
            let mut npc_col = seg.get_scalars_mut(pcinc_pi_npc)?;
            for (i, &(pc, npc)) in pc_inc_set.iter().enumerate() {
                pc_col[i] = B16::new(pc);
                npc_col[i] = B16::new(npc);
            }
        }
        pcinc_lookup.populate(&mut seg, pc_inc_counts.iter().copied())?;
    }

    // ---- Fill step_cmp table ----
    {
        let (sc_step, sc_prev, sc_is_active, sc_fri_blind, sc_fri_blind_sq,
             s0, s1, s2, ps0, ps1, ps2,
             eq2, eq1, diff2_inv, diff1_inv,
             cmp_a, cmp_b, a_bits, b_bits, borrows) = SC;

        // Collect (step, prev_access_step) for all active rows.
        let mut cmp_pairs: Vec<(u32, u32)> = Vec::new();
        for (i, row) in table_ops.iter().enumerate() {
            if let Some(op) = row {
                if op.is_read || op.is_write {
                    cmp_pairs.push((op.step, row_prev_access_steps[i]));
                }
            }
        }
        let sct_size = cmp_pairs.len();
        let padded_size = sct_size.next_power_of_two();
        // Padding: (1, 0) satisfies 1 > 0.
        cmp_pairs.resize(padded_size, (1, 0));

        println!("Step cmp table:   {sct_size} rows (padded to {padded_size})");

        let tw = witness.init_table(step_cmp_table_id, sct_size).expect("init step_cmp table");
        let seg = tw.full_segment();

        {
            let mut sc_s = seg.get_scalars_mut(sc_step)?;
            let mut sc_ps = seg.get_scalars_mut(sc_prev)?;
            let mut s0_col = seg.get_scalars_mut(s0)?;
            let mut s1_col = seg.get_scalars_mut(s1)?;
            let mut s2_col = seg.get_scalars_mut(s2)?;
            let mut ps0_col = seg.get_scalars_mut(ps0)?;
            let mut ps1_col = seg.get_scalars_mut(ps1)?;
            let mut ps2_col = seg.get_scalars_mut(ps2)?;
            let mut d2inv_col = seg.get_scalars_mut(diff2_inv)?;
            let mut d1inv_col = seg.get_scalars_mut(diff1_inv)?;
            let mut ca_col = seg.get_scalars_mut(cmp_a)?;
            let mut cb_col = seg.get_scalars_mut(cmp_b)?;

            for (i, &(step, prev)) in cmp_pairs.iter().enumerate() {
                sc_s[i] = B32::new(step);
                sc_ps[i] = B32::new(prev);

                let sb0 = (step & 0xFF) as u8;
                let sb1 = ((step >> 8) & 0xFF) as u8;
                let sb2 = ((step >> 16) & 0xFF) as u8;
                let pb0 = (prev & 0xFF) as u8;
                let pb1 = ((prev >> 8) & 0xFF) as u8;
                let pb2 = ((prev >> 16) & 0xFF) as u8;

                s0_col[i] = B8::new(sb0);
                s1_col[i] = B8::new(sb1);
                s2_col[i] = B8::new(sb2);
                ps0_col[i] = B8::new(pb0);
                ps1_col[i] = B8::new(pb1);
                ps2_col[i] = B8::new(pb2);

                let d2 = B8::new(sb2 ^ pb2);
                let d1 = B8::new(sb1 ^ pb1);
                d2inv_col[i] = if d2 != B8::ZERO { d2.invert().unwrap() } else { B8::ZERO };
                d1inv_col[i] = if d1 != B8::ZERO { d1.invert().unwrap() } else { B8::ZERO };

                let (ca, cb) = if sb2 != pb2 {
                    (sb2, pb2)
                } else if sb1 != pb1 {
                    (sb1, pb1)
                } else {
                    (sb0, pb0)
                };
                ca_col[i] = B8::new(ca);
                cb_col[i] = B8::new(cb);
            }
        }

        // B1 columns: is_active, eq2, eq1, a_bits, b_bits, borrows
        {
            let mut active_col = seg.get_mut(sc_is_active)?;
            let mut eq2_col = seg.get_mut(eq2)?;
            let mut eq1_col = seg.get_mut(eq1)?;
            let mut ab: [_; 8] = std::array::from_fn(|j| seg.get_mut(a_bits[j]).unwrap());
            let mut bb: [_; 8] = std::array::from_fn(|j| seg.get_mut(b_bits[j]).unwrap());
            let mut br: [_; 8] = std::array::from_fn(|j| seg.get_mut(borrows[j]).unwrap());

            for idx in 0..eq2_col.len() {
                for k in 0..PackedB1::WIDTH {
                    let row = idx * PackedB1::WIDTH + k;
                    if row < padded_size {
                        active_col[idx].set(k, B1::from(row < sct_size));
                        let (step, prev) = cmp_pairs[row];
                        let sb2 = ((step >> 16) & 0xFF) as u8;
                        let pb2 = ((prev >> 16) & 0xFF) as u8;
                        let sb1 = ((step >> 8) & 0xFF) as u8;
                        let pb1 = ((prev >> 8) & 0xFF) as u8;
                        eq2_col[idx].set(k, B1::from(sb2 == pb2));
                        eq1_col[idx].set(k, B1::from(sb1 == pb1));

                        // Deciding pair
                        let (ca, cb) = if sb2 != pb2 {
                            (sb2, pb2)
                        } else if sb1 != pb1 {
                            (sb1, pb1)
                        } else {
                            let sb0 = (step & 0xFF) as u8;
                            let pb0 = (prev & 0xFF) as u8;
                            (sb0, pb0)
                        };

                        // Bit decomposition + borrow chain
                        for bit in 0..8 {
                            ab[bit][idx].set(k, B1::from((ca >> bit) & 1 != 0));
                            bb[bit][idx].set(k, B1::from((cb >> bit) & 1 != 0));
                        }

                        // Borrow chain: compute a - b - 1
                        let mut borrow = true; // initial borrow = 1 (the -1)
                        for bit in 0..8 {
                            let a_bit = (ca >> bit) & 1;
                            let b_bit = (cb >> bit) & 1;
                            let borrow_in = borrow as u8;
                            // borrow_out = maj(NOT a, b, borrow_in)
                            let not_a = 1 - a_bit;
                            borrow = (not_a & b_bit) | (not_a & borrow_in) | (b_bit & borrow_in) != 0;
                            br[bit][idx].set(k, B1::from(borrow));
                        }
                        assert!(!borrow, "step_cmp: step={step} not > prev={prev} at row {row}");
                    }
                }
            }
        }

        // fri_blind
        {
            let mut blind_col = seg.get_scalars_mut(sc_fri_blind)?;
            let mut blind_sq_col = seg.get_scalars_mut(sc_fri_blind_sq)?;
            for i in 0..padded_size {
                let r = B128::new(rng.random::<u128>());
                blind_col[i] = r;
                blind_sq_col[i] = r * r;
            }
        }
    }

    // ==== Boundary conditions: public IO output ====
    let mut boundaries = Vec::new();
    for &slot in &output_slots {
        boundaries.push(Boundary {
            values: vec![B128::from(B16::new((slot * 4) as u16)), B128::from(B32::new(trace.final_mem[slot]))],
            channel_id: pub_io_chan,
            direction: FlushDirection::Pull,
            multiplicity: 1,
        });
    }
    println!("Boundaries:       {} (pub_io: {} output length + {} data words)",
        boundaries.len(), 1, output_num_words);

    // ==== Prover: compile + prove ====
    let t_total = Instant::now();

    let t_compile = Instant::now();
    let compiled_cs = cs.compile().map_err(|e| anyhow::anyhow!("{e}"))?;
    let ccs_digest = compiled_cs.digest::<Blake3Digest>();
    let table_sizes = witness.table_sizes();
    let witness_mle = witness.into_multilinear_extension_index();
    let compile_time = t_compile.elapsed();

    let t_prove = Instant::now();
    let mut compute_holder =
        FastCpuLayerHolder::<CanonicalTowerFamily, P>::new(1 << 20, 1 << 25);

    let proof = prove::<
        _,
        OptimalUnderlier,
        CanonicalTowerFamily,
        Blake3Digest,
        Blake3Compression,
        HasherChallenger<Blake3Digest>,
        _,
        _,
        _,
    >(
        &mut compute_holder.to_data(),
        &compiled_cs,
        LOG_INV_RATE,
        SECURITY_BITS,
        &FRI_STRATEGY,
        &ccs_digest,
        &boundaries,
        &table_sizes,
        witness_mle,
        &make_portable_backend(),
    )?;
    let prove_time = t_prove.elapsed();
    let total_prover_time = t_total.elapsed();

    let proof_size = proof.get_proof_size();

    // ==== Write proof + boundary data to disk ====
    std::fs::create_dir_all("proofs")?;
    {
        use std::io::Write;
        let proof_dir = format!("proofs/{program}");
        std::fs::create_dir_all(&proof_dir)?;
        let mut out = std::io::BufWriter::new(std::fs::File::create(format!("{proof_dir}/memory_check_proof.bin"))?);
        let transcript = &proof.transcript;
        out.write_all(&(transcript.len() as u64).to_le_bytes())?;
        out.write_all(transcript)?;
        out.write_all(&(boundaries.len() as u32).to_le_bytes())?;
        for b in &boundaries {
            // Each boundary has values: [B128::from(B32::new(slot)), B128::from(B32::new(value))]
            let slot = b.values[0].val() as u32;
            let value = b.values[1].val() as u32;
            out.write_all(&slot.to_le_bytes())?;
            out.write_all(&value.to_le_bytes())?;
        }
        out.flush()?;
    }
    println!("Proof written to proofs/{program}/memory_check_proof.bin");
    println!();

    println!("=== Prover stats ===");
    println!("  Compile:            {:?}", compile_time);
    println!("  Proving:            {:?}", prove_time);
    println!("  Total:              {:?}", total_prover_time);
    println!();
    println!("=== Proof size ===");
    println!(
        "  Binius proof:       {} bytes ({:.2} KB)",
        proof_size, proof_size as f64 / 1024.0
    );

    Ok(())
}
