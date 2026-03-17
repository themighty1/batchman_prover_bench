use anyhow::{Result, bail};
use batchman_witness_generator::{MemAccessRow, MemRegion, MemTrace, LookupRow, LookupTrace};
use reg_analyzer::canon3_types::*;
use reg_analyzer::canon3_vm::Canon3Vm;
use reg_analyzer::rv32_flat_vm::FlatProgram;
use reg_analyzer::rv32_isa_vm::MAILBOX_BASE;
use std::fs;
use std::io::Write;


/// Hardcoded witness op IDs.
/// Maps Canon3Op enum value (u8) → witness op ID.
/// None = unmapped (panic if encountered at runtime).
const OP_REMAP: [Option<u8>; 256] = {
    let mut table = [None; 256];
    // Canon3Op enum value      → witness ID
    table[0]  = Some(0);   // Add         → 0
    table[1]  = Some(43);  // Sub         → 43
    // Mul(2), Mulh(3), Mulhsu(4), Mulhu(5), Div(6), Divu(7), Rem(8), Remu(9) — unused
    // Sll(10), Srl(11), Sra(12) — replaced by fixed shifts
    table[13] = Some(1);   // Slt         → 1
    table[14] = Some(22);  // Sltu        → 22
    table[15] = Some(23);  // Xor         → 23
    table[16] = Some(21);  // Or          → 21
    table[17] = Some(44);  // And         → 44
    table[18] = Some(12);  // Addi        → 12
    table[19] = Some(24);  // Slti        → 24
    // Sltiu(20) — unused in canon3 binary
    table[21] = Some(19);  // Xori        → 19
    // Ori(22) — unused
    table[23] = Some(46);  // Andi        → 46
    // Slli(24), Srli(25), Srai(26) — replaced by fixed shifts
    table[27] = Some(13);  // Lw          → 13
    table[28] = Some(14);  // Sw          → 14
    // SwAligned(29) — decomposed into sub-ops
    table[30] = Some(5);   // LwAligned   → 5
    table[31] = Some(15);  // ByteSelR2   → 15
    // ByteSel0(32), ByteSel1(33), ByteSel2(34), ByteSel3(35) — unused in canon3 binary
    table[36] = Some(16);  // Sext8       → 16
    table[37] = Some(27);  // Beq         → 27
    table[38] = Some(25);  // Bne         → 25
    table[39] = Some(17);  // Blt         → 17
    table[40] = Some(18);  // Bge         → 18
    table[41] = Some(26);  // Bltu        → 26
    table[42] = Some(45);  // Bgeu        → 45
    table[43] = Some(20);  // Lui         → 20
    table[44] = Some(47);  // Jal         → 47
    table[45] = Some(49);  // JalCall     → 49
    // Jalr(46) — unused in canon3 binary
    table[47] = Some(50);  // JalrCall    → 50
    table[48] = Some(48);  // Ret         → 48
    // JrTableIdx(49), JrComputed(50), Ecall(51), Halt(52) — unused in canon3 binary
    table[53] = Some(6);   // LwAbs0      → 6
    table[54] = Some(7);   // LwAbs1      → 7
    table[55] = Some(8);   // LwAbs2      → 8
    table[56] = Some(9);   // SwAbs0      → 9
    table[57] = Some(10);  // SwAbs1      → 10
    table[58] = Some(11);  // SwAbs2      → 11
    table[59] = Some(28);  // Sll1        → 28  sll_const1
    table[60] = Some(29);  // Sll4        → 29  sll_const4
    table[61] = Some(30);  // Sll8        → 30  sll_const8
    table[62] = Some(31);  // Sll16       → 31  sll_const16
    table[63] = Some(32);  // Sll31       → 32  sll_const31
    table[64] = Some(33);  // Srl1        → 33  srl_const1
    table[65] = Some(34);  // Srl4        → 34  srl_const4
    table[66] = Some(35);  // Srl8        → 35  srl_const8
    table[67] = Some(36);  // Srl16       → 36  srl_const16
    table[68] = Some(37);  // Srl31       → 37  srl_const31
    table[69] = Some(38);  // Sra1        → 38  sra_const1
    table[70] = Some(39);  // Sra4        → 39  sra_const4
    table[71] = Some(40);  // Sra8        → 40  sra_const8
    table[72] = Some(41);  // Sra16       → 41  sra_const16
    table[73] = Some(42);  // Sra31       → 42  sra_const31
    table[74] = Some(53);  // ByteInsR2   → 53  byte_ins_r2
    table[75] = Some(52);  // SwWaligned  → 52  sw_waligned
    // Rare ops not in primary ISA table — assigned from 254 downward
    table[10] = Some(254);  // Sll         → 254
    table[11] = Some(253);  // Srl         → 253
    table[20] = Some(252);  // Sltiu       → 252
    table[22] = Some(251);  // Ori         → 251
    table[49] = Some(250);  // JrTableIdx  → 250
    table
};

const OP_RET: u8 = 48;
const OP_JALR_CALL: u8 = 50;
const OP_JR_TBL_IDX: u8 = 250;

// Witness op IDs for ops that don't use an immediate
const OP_ADD: u8 = 0;
const OP_SLT: u8 = 1;
const OP_BYTE_SEL_R2: u8 = 15;
const OP_SEXT8: u8 = 16;
const OP_OR: u8 = 21;
const OP_SLTU: u8 = 22;
const OP_XOR: u8 = 23;
const OP_SLL_CONST1: u8 = 28;
const OP_SLL_CONST4: u8 = 29;
const OP_SLL_CONST8: u8 = 30;
const OP_SLL_CONST16: u8 = 31;
const OP_SLL_CONST31: u8 = 32;
const OP_SRL_CONST1: u8 = 33;
const OP_SRL_CONST4: u8 = 34;
const OP_SRL_CONST8: u8 = 35;
const OP_SRL_CONST16: u8 = 36;
const OP_SRL_CONST31: u8 = 37;
const OP_SRA_CONST1: u8 = 38;
const OP_SRA_CONST4: u8 = 39;
const OP_SRA_CONST8: u8 = 40;
const OP_SRA_CONST16: u8 = 41;
const OP_SRA_CONST31: u8 = 42;
const OP_SUB: u8 = 43;
const OP_AND: u8 = 44;
const OP_BYTE_INS_R2: u8 = 53;
const OP_SLL: u8 = 254;
const OP_SRL: u8 = 253;

/// Whether this witness op uses an immediate.
fn op_has_imm(op: u8) -> bool {
    !matches!(op,
        OP_ADD | OP_SLT | OP_SLTU | OP_XOR | OP_OR | OP_AND | OP_SUB
        | OP_SLL | OP_SRL
        | OP_SLL_CONST1 | OP_SLL_CONST4 | OP_SLL_CONST8 | OP_SLL_CONST16 | OP_SLL_CONST31
        | OP_SRL_CONST1 | OP_SRL_CONST4 | OP_SRL_CONST8 | OP_SRL_CONST16 | OP_SRL_CONST31
        | OP_SRA_CONST1 | OP_SRA_CONST4 | OP_SRA_CONST8 | OP_SRA_CONST16 | OP_SRA_CONST31
        | OP_BYTE_SEL_R2 | OP_BYTE_INS_R2 | OP_SEXT8
        | OP_RET | OP_JALR_CALL | OP_JR_TBL_IDX
    )
}


fn remap_op(canon3_op: u8) -> Result<u8> {
    match OP_REMAP[canon3_op as usize] {
        Some(id) => Ok(id),
        None => bail!("No witness op ID for Canon3Op enum value {}", canon3_op),
    }
}

/// Witness row: one per execution step.
///
/// Layout (28 bytes packed):
///   r0           u32 LE    offset 0
///   r1           u32 LE    offset 4
///   r2           u32 LE    offset 8
///   pc           u16 LE    offset 12
///   next_pc      u16 LE    offset 14
///   imm          i32 LE    offset 16
///   addr         u16 LE    offset 20
///   value        u32 LE    offset 22
///   op           u8        offset 26
///   flags        u8        offset 27  (bit 0 = has_imm, bit 1 = has_mem)
#[derive(Clone, Debug)]
#[repr(C)]
pub struct WitnessRow {
    pub r0: u32,
    pub r1: u32,
    pub r2: u32,
    pub pc: u16,
    pub next_pc: u16,
    pub imm: Option<i32>,
    pub addr: Option<u16>,
    pub value: Option<u32>,
    pub op: u8,
}

const ROW_BYTES: usize = 28;

impl WitnessRow {
    pub fn to_bytes(&self) -> [u8; ROW_BYTES] {
        let mut buf = [0u8; ROW_BYTES];
        buf[0..4].copy_from_slice(&self.r0.to_le_bytes());
        buf[4..8].copy_from_slice(&self.r1.to_le_bytes());
        buf[8..12].copy_from_slice(&self.r2.to_le_bytes());
        buf[12..14].copy_from_slice(&self.pc.to_le_bytes());
        buf[14..16].copy_from_slice(&self.next_pc.to_le_bytes());
        buf[16..20].copy_from_slice(&self.imm.unwrap_or(0).to_le_bytes());
        buf[20..22].copy_from_slice(&self.addr.unwrap_or(0).to_le_bytes());
        buf[22..26].copy_from_slice(&self.value.unwrap_or(0).to_le_bytes());
        buf[26] = self.op;
        let mut flags: u8 = 0;
        if self.imm.is_some() { flags |= 1; }
        if self.addr.is_some() { flags |= 2; }
        buf[27] = flags;
        buf
    }
}

fn generate_witness_rows(trace: &ExecutionTrace) -> Result<Vec<WitnessRow>> {
    let mut rows = Vec::with_capacity(trace.steps.len());

    for (i, step) in trace.steps.iter().enumerate() {
        let op = remap_op(step.op)?;

        let (addr, value) = if let Some((a, v)) = step.mem_write {
            (Some(a), Some(v))
        } else if let Some((a, v)) = step.mem_read {
            (Some(a), Some(v))
        } else if op == OP_BYTE_SEL_R2 {
            let byte = (step.regs_before.r0 >> (step.regs_before.r2 * 8)) & 0xFF;
            (None, Some(byte))
        } else {
            (None, None)
        };
        let addr = addr.map(|a| {
            assert!(a <= u16::MAX as u32, "step {}: addr {} exceeds u16", i, a);
            a as u16
        });
        let pc = step.pc as u16;
        let next_pc = pc + 1;

        rows.push(WitnessRow {
            r0: step.regs_before.r0,
            r1: step.regs_before.r1,
            r2: step.regs_before.r2,
            pc,
            next_pc,
            imm: Some(step.imm),
            addr,
            value,
            op,
        });
    }

    Ok(rows)
}

fn main() -> Result<()> {
    // 1. Load compiled program
    let args: Vec<String> = std::env::args().collect();
    let program_path = args.get(1).unwrap_or_else(|| {
        eprintln!("Usage: witgen <program.bin> <output-dir>");
        eprintln!("  program.bin  — compiled canonical bytecode (e.g. canonical.bin)");
        eprintln!("  output-dir   — witness output directory (e.g. witness/json-query)");
        std::process::exit(1);
    });
    let output_dir = args.get(2).unwrap_or_else(|| {
        eprintln!("Usage: witgen <program.bin> <output-dir>");
        std::process::exit(1);
    });
    fs::create_dir_all(output_dir)?;
    let encoded = fs::read(program_path)?;
    let program: FlatProgram = bincode::deserialize(&encoded)?;

    let opcode_names: Vec<String> = program.opcode_table.iter().map(|o| o.base_op.clone()).collect();
    let remapped_code = Canon3Vm::remap_code(&opcode_names, &program.code_segment_u8)?;

    // 2. Set up VM and populate memory
    let mut vm = Canon3Vm::new(remapped_code, program.imm_table.clone());

    for seg in &program.segments {
        vm.memory.write_bytes(seg.vaddr, &seg.data);
    }

    // Load input from fixtures/ next to canonical.bin
    let program_dir = std::path::Path::new(program_path.as_str()).parent()
        .unwrap_or(std::path::Path::new("."));
    let fixtures_dir = program_dir.join("fixtures");
    let input_path = fixtures_dir.join("input.bin");
    let query_path = fixtures_dir.join("query.txt");

    let fixture = fs::read(&input_path)?;
    vm.memory.write_input(&fixture);
    if query_path.exists() {
        let query = fs::read_to_string(&query_path)?;
        vm.memory.write_path(query.trim().as_bytes());
    }

    const STACK_TOP: u32 = 4 * 1024;
    vm.memory.write_u32(MAILBOX_BASE + 2 * 4, STACK_TOP);
    vm.pc = program.entry_pc;

    // 2b. Snapshot initial memory (written to disk after trace)
    let init_snap = vm.memory.snapshot();

    // 3. Execute with trace
    let trace = vm.execute_with_trace()?;

    // 4. Build remapped code/imm table (witness op IDs)
    let witness_code: Vec<u8> = vm.code.iter()
        .map(|&op| remap_op(op))
        .collect::<Result<Vec<u8>>>()?;
    // witness_code[pc] = witness op ID, vm.imm_table[pc] = immediate

    // 5. Generate witness rows (remaps op IDs)
    let rows = generate_witness_rows(&trace)?;

    // 6. Generate memory access trace
    let mem_access: Vec<MemAccessRow> = trace.steps.iter().map(|step| {
        let (is_read, read_addr, read_value) = match step.mem_read {
            Some((a, v)) => (true, a, v),
            None => (false, 0, 0),
        };
        let (is_write, write_addr, write_value) = match step.mem_write {
            Some((a, v)) => (true, a, v),
            None => (false, 0, 0),
        };
        MemAccessRow { is_read, is_write, read_addr, read_value, write_addr, write_value }
    }).collect();

    // Build lookup trace rows
    let lookup_rows: Vec<LookupRow> = trace.steps.iter().enumerate().map(|(_i, step)| {
        let op = remap_op(step.op).unwrap();
        let hi = op_has_imm(op);
        LookupRow {
            pc: step.pc as u16,
            op,
            has_imm: if hi { 1 } else { 0 },
            imm: if hi { step.imm } else { 0 },
            is_byte_sel_r2: if op == OP_BYTE_SEL_R2 { 1 } else { 0 },
        }
    }).collect();

    // Build memory trace
    let initial: Vec<MemRegion> = init_snap.regions.iter().map(|r| {
        MemRegion { addr: r.addr, data: r.data.clone() }
    }).collect();
    let mt = MemTrace { initial, access: mem_access.clone() };

    // === Cross-trace consistency checks (before writing files) ===
    {
        let n = rows.len();
        assert_eq!(n, lookup_rows.len(), "cpu_trace and lookup_trace row count mismatch");
        assert_eq!(n, mem_access.len(), "cpu_trace and mem_trace row count mismatch");

        let mut errors = 0u64;

        // 1. PC consistency: cpu_trace.pc == lookup_trace.pc for every step
        for i in 0..n {
            if rows[i].pc != lookup_rows[i].pc {
                if errors < 10 {
                    eprintln!("  PC mismatch step {}: cpu={} lookup={}", i, rows[i].pc, lookup_rows[i].pc);
                }
                errors += 1;
            }
        }

        // 2. op consistency: cpu_trace.op == lookup_trace.op
        for i in 0..n {
            if rows[i].op != lookup_rows[i].op {
                if errors < 10 {
                    eprintln!("  OP mismatch step {}: cpu={} lookup={}", i, rows[i].op, lookup_rows[i].op);
                }
                errors += 1;
            }
        }

        // 3. Memory consistency: every read returns last written value
        let mut mem: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for region in &mt.initial {
            let mut off = 0u32;
            while (off + 4) as usize <= region.data.len() {
                let addr = region.addr + off;
                let val = u32::from_le_bytes(region.data[off as usize..off as usize + 4].try_into().unwrap());
                if val != 0 {
                    mem.insert(addr, val);
                }
                off += 4;
            }
        }
        for (i, row) in mem_access.iter().enumerate() {
            if row.is_read {
                let expected = mem.get(&row.read_addr).copied().unwrap_or(0);
                if row.read_value != expected {
                    if errors < 10 {
                        eprintln!("  MEM mismatch step {}: read addr={} expected={} got={}",
                            i, row.read_addr, expected, row.read_value);
                    }
                    errors += 1;
                }
            }
            if row.is_write {
                mem.insert(row.write_addr, row.write_value);
            }
        }

        if errors == 0 {
            println!("=== Consistency checks: PASS ===");
        } else {
            bail!("Consistency checks FAILED: {} errors", errors);
        }
    }

    // === Write trace files ===
    {
        let mt_path = format!("{}/mem_trace.bin", output_dir);
        let mut f = std::io::BufWriter::new(fs::File::create(&mt_path)?);
        mt.write_to(&mut f)?;
        f.flush()?;

        let reads = mem_access.iter().filter(|r| r.is_read).count();
        let writes = mem_access.iter().filter(|r| r.is_write).count();
        let _both = mem_access.iter().filter(|r| r.is_read && r.is_write).count();
        println!("=== mem_trace ===");
        println!("  Initial regions: {}", mt.initial.len());
        for (i, r) in mt.initial.iter().enumerate() {
            println!("    [{}] {} .. {} ({} bytes)", i, r.addr, r.addr + r.data.len() as u32 - 1, r.data.len());
        }
        println!("  Reads:           {}", reads);
        println!("  Writes:          {}", writes);
        let file_size = fs::metadata(&mt_path)?.len();
        println!("  Output:          {} ({} bytes)", mt_path, file_size);
    }

    {
        let lt = LookupTrace { rows: lookup_rows.clone() };
        let lt_path = format!("{}/lookup_trace.bin", output_dir);
        let mut f = std::io::BufWriter::new(fs::File::create(&lt_path)?);
        lt.write_to(&mut f)?;
        f.flush()?;

        let file_size = fs::metadata(&lt_path)?.len();
        println!("=== lookup_trace: {} ({} bytes) ===", lt_path, file_size);
    }

    // Count rows with memory access
    let mem_rows = rows.iter().filter(|r| r.addr.is_some()).count();
    let max_addr = rows.iter().filter_map(|r| r.addr).max().unwrap_or(0);
    println!("  Max mem addr:   {} (0x{:x}, {} bits)", max_addr, max_addr, 32 - max_addr.leading_zeros());

    // 7. Write CPU state transition trace
    let output_path = format!("{}/cpu_trace.bin", output_dir);
    let mut f = std::io::BufWriter::new(fs::File::create(&output_path)?);

    // Header: 4-byte LE row count
    f.write_all(&(rows.len() as u32).to_le_bytes())?;

    for row in &rows {
        f.write_all(&row.to_bytes())?;
    }
    f.flush()?;

    // Summary
    let unique_witness_ops: std::collections::BTreeSet<u8> = witness_code.iter().copied().collect();
    println!("=== Witness generated ===");
    println!("  Code table:     {} entries, {} unique witness op IDs",
        witness_code.len(), unique_witness_ops.len());
    println!("  Trace steps:    {}", trace.steps.len());
    println!("  Rows with mem:  {} ({:.1}%)", mem_rows, mem_rows as f64 / rows.len() as f64 * 100.0);
    println!("  Output:         {} ({} bytes)", output_path,
        4 + rows.len() * ROW_BYTES);

    // 8. Write packed_pt.bin — 128-bit packed plaintext per step.
    //
    // Bit layout (matches Batchman and binius memory_check/mac_consistency):
    //   bits 0-31:    value (u32)
    //   bits 32-63:   imm (i32 as u32)
    //   bits 64-79:   pc (u16)
    //   bits 80-95:   next_pc (u16)
    //   bits 96-111:  addr (u16)
    //   bits 112-119: op (u8)
    //   bit 120:      is_mem_write
    //   bit 121:      is_mem_read
    //   bit 122:      has_immediate
    //   bit 123:      is_byte_sel_r2
    {
        let pt_path = format!("{}/packed_pt.bin", output_dir);
        let mut f = std::io::BufWriter::new(fs::File::create(&pt_path)?);
        f.write_all(&(rows.len() as u32).to_le_bytes())?;

        for i in 0..rows.len() {
            let row = &rows[i];
            let ma = &mem_access[i];
            let lr = &lookup_rows[i];

            let value = row.value.unwrap_or(0) as u128;
            let imm = row.imm.unwrap_or(0) as u32 as u128;
            let pc = row.pc as u128;
            let next_pc = row.next_pc as u128;
            let addr = row.addr.unwrap_or(0) as u128;
            let op = row.op as u128;
            let is_write = ma.is_write as u128;
            let is_read = ma.is_read as u128;
            let has_imm = lr.has_imm as u128;
            let is_bsr = lr.is_byte_sel_r2 as u128;

            let pt: u128 = value
                | (imm << 32)
                | (pc << 64)
                | (next_pc << 80)
                | (addr << 96)
                | (op << 112)
                | (is_write << 120)
                | (is_read << 121)
                | (has_imm << 122)
                | (is_bsr << 123);

            f.write_all(&pt.to_le_bytes())?;
        }
        f.flush()?;
        let file_size = fs::metadata(&pt_path)?.len();
        println!("  packed_pt:      {} ({} bytes)", pt_path, file_size);
    }

    // Verify result
    let result = vm.memory.read_output_string();
    println!("  Result:         {:?}", result);

    Ok(())
}
