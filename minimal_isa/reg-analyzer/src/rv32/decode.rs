//! ELF decoding: raw bytes → physical-register instruction stream.

use object::{Object, ObjectSection, ObjectSymbol};
use rvdc::{Inst, Reg};

// ---------------------------------------------------------------------------
// Physical-register decoded instruction
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DecodedInst {
    pub addr: u32,
    pub op: String,
    pub rd: Option<u8>,   // destination physical register 0-31
    pub rs1: Option<u8>,  // source physical register 1
    pub rs2: Option<u8>,  // source physical register 2
    pub imm: Option<i32>,
}

pub const REG_NAMES: [&str; 32] = [
    "x0", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
    "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
    "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7",
    "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6",
];

pub(super) fn reg_num(r: &Reg) -> u8 {
    let s = format!("{}", r);
    match s.as_str() {
        "zero" => 0, "ra" => 1, "sp" => 2, "gp" => 3, "tp" => 4,
        "t0" => 5, "t1" => 6, "t2" => 7,
        "s0" => 8, "s1" => 9,
        "a0" => 10, "a1" => 11, "a2" => 12, "a3" => 13,
        "a4" => 14, "a5" => 15, "a6" => 16, "a7" => 17,
        "s2" => 18, "s3" => 19, "s4" => 20, "s5" => 21,
        "s6" => 22, "s7" => 23, "s8" => 24, "s9" => 25,
        "s10" => 26, "s11" => 27,
        "t3" => 28, "t4" => 29, "t5" => 30, "t6" => 31,
        _ => 0,
    }
}

pub(super) fn decode_one(addr: u32, inst: &Inst) -> DecodedInst {
    match inst {
        Inst::Lui { uimm, dest } => DecodedInst {
            addr, op: "lui".into(), rd: Some(reg_num(dest)), rs1: None, rs2: None, imm: Some(*uimm as i32),
        },
        Inst::Auipc { uimm, dest } => DecodedInst {
            addr, op: "auipc".into(), rd: Some(reg_num(dest)), rs1: None, rs2: None, imm: Some(*uimm as i32),
        },
        Inst::Jal { offset, dest } => DecodedInst {
            addr, op: "jal".into(), rd: Some(reg_num(dest)), rs1: None, rs2: None, imm: Some(*offset as i32),
        },
        Inst::Jalr { offset, base, dest } => DecodedInst {
            addr, op: "jalr".into(), rd: Some(reg_num(dest)), rs1: Some(reg_num(base)), rs2: None, imm: Some(*offset as i32),
        },
        Inst::Beq  { offset, src1, src2 } => DecodedInst { addr, op: "beq".into(),  rd: None, rs1: Some(reg_num(src1)), rs2: Some(reg_num(src2)), imm: Some(*offset as i32) },
        Inst::Bne  { offset, src1, src2 } => DecodedInst { addr, op: "bne".into(),  rd: None, rs1: Some(reg_num(src1)), rs2: Some(reg_num(src2)), imm: Some(*offset as i32) },
        Inst::Blt  { offset, src1, src2 } => DecodedInst { addr, op: "blt".into(),  rd: None, rs1: Some(reg_num(src1)), rs2: Some(reg_num(src2)), imm: Some(*offset as i32) },
        Inst::Bge  { offset, src1, src2 } => DecodedInst { addr, op: "bge".into(),  rd: None, rs1: Some(reg_num(src1)), rs2: Some(reg_num(src2)), imm: Some(*offset as i32) },
        Inst::Bltu { offset, src1, src2 } => DecodedInst { addr, op: "bltu".into(), rd: None, rs1: Some(reg_num(src1)), rs2: Some(reg_num(src2)), imm: Some(*offset as i32) },
        Inst::Bgeu { offset, src1, src2 } => DecodedInst { addr, op: "bgeu".into(), rd: None, rs1: Some(reg_num(src1)), rs2: Some(reg_num(src2)), imm: Some(*offset as i32) },
        Inst::Lb  { offset, dest, base } => DecodedInst { addr, op: "lb".into(),  rd: Some(reg_num(dest)), rs1: Some(reg_num(base)), rs2: None, imm: Some(*offset as i32) },
        Inst::Lbu { offset, dest, base } => DecodedInst { addr, op: "lbu".into(), rd: Some(reg_num(dest)), rs1: Some(reg_num(base)), rs2: None, imm: Some(*offset as i32) },
        Inst::Lh  { offset, dest, base } => DecodedInst { addr, op: "lh".into(),  rd: Some(reg_num(dest)), rs1: Some(reg_num(base)), rs2: None, imm: Some(*offset as i32) },
        Inst::Lhu { offset, dest, base } => DecodedInst { addr, op: "lhu".into(), rd: Some(reg_num(dest)), rs1: Some(reg_num(base)), rs2: None, imm: Some(*offset as i32) },
        Inst::Lw  { offset, dest, base } => DecodedInst { addr, op: "lw".into(),  rd: Some(reg_num(dest)), rs1: Some(reg_num(base)), rs2: None, imm: Some(*offset as i32) },
        Inst::Sb { offset, src, base } => DecodedInst { addr, op: "sb".into(), rd: None, rs1: Some(reg_num(base)), rs2: Some(reg_num(src)), imm: Some(*offset as i32) },
        Inst::Sh { offset, src, base } => DecodedInst { addr, op: "sh".into(), rd: None, rs1: Some(reg_num(base)), rs2: Some(reg_num(src)), imm: Some(*offset as i32) },
        Inst::Sw { offset, src, base } => DecodedInst { addr, op: "sw".into(), rd: None, rs1: Some(reg_num(base)), rs2: Some(reg_num(src)), imm: Some(*offset as i32) },
        Inst::Addi  { imm, dest, src1 } => DecodedInst { addr, op: "addi".into(),  rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: None, imm: Some(*imm as i32) },
        Inst::Slti  { imm, dest, src1 } => DecodedInst { addr, op: "slti".into(),  rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: None, imm: Some(*imm as i32) },
        Inst::Sltiu { imm, dest, src1 } => DecodedInst { addr, op: "sltiu".into(), rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: None, imm: Some(*imm as i32) },
        Inst::Xori  { imm, dest, src1 } => DecodedInst { addr, op: "xori".into(),  rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: None, imm: Some(*imm as i32) },
        Inst::Ori   { imm, dest, src1 } => DecodedInst { addr, op: "ori".into(),   rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: None, imm: Some(*imm as i32) },
        Inst::Andi  { imm, dest, src1 } => DecodedInst { addr, op: "andi".into(),  rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: None, imm: Some(*imm as i32) },
        Inst::Slli  { imm, dest, src1 } => DecodedInst { addr, op: "slli".into(),  rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: None, imm: Some(*imm as i32) },
        Inst::Srli  { imm, dest, src1 } => DecodedInst { addr, op: "srli".into(),  rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: None, imm: Some(*imm as i32) },
        Inst::Srai  { imm, dest, src1 } => DecodedInst { addr, op: "srai".into(),  rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: None, imm: Some(*imm as i32) },
        Inst::Add  { dest, src1, src2 } => DecodedInst { addr, op: "add".into(),  rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: Some(reg_num(src2)), imm: None },
        Inst::Sub  { dest, src1, src2 } => DecodedInst { addr, op: "sub".into(),  rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: Some(reg_num(src2)), imm: None },
        Inst::Sll  { dest, src1, src2 } => DecodedInst { addr, op: "sll".into(),  rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: Some(reg_num(src2)), imm: None },
        Inst::Slt  { dest, src1, src2 } => DecodedInst { addr, op: "slt".into(),  rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: Some(reg_num(src2)), imm: None },
        Inst::Sltu { dest, src1, src2 } => DecodedInst { addr, op: "sltu".into(), rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: Some(reg_num(src2)), imm: None },
        Inst::Xor  { dest, src1, src2 } => DecodedInst { addr, op: "xor".into(),  rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: Some(reg_num(src2)), imm: None },
        Inst::Srl  { dest, src1, src2 } => DecodedInst { addr, op: "srl".into(),  rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: Some(reg_num(src2)), imm: None },
        Inst::Sra  { dest, src1, src2 } => DecodedInst { addr, op: "sra".into(),  rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: Some(reg_num(src2)), imm: None },
        Inst::Or   { dest, src1, src2 } => DecodedInst { addr, op: "or".into(),   rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: Some(reg_num(src2)), imm: None },
        Inst::And  { dest, src1, src2 } => DecodedInst { addr, op: "and".into(),  rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: Some(reg_num(src2)), imm: None },
        Inst::Mul    { dest, src1, src2 } => DecodedInst { addr, op: "mul".into(),    rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: Some(reg_num(src2)), imm: None },
        Inst::Mulh   { dest, src1, src2 } => DecodedInst { addr, op: "mulh".into(),   rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: Some(reg_num(src2)), imm: None },
        Inst::Mulhsu { dest, src1, src2 } => DecodedInst { addr, op: "mulhsu".into(), rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: Some(reg_num(src2)), imm: None },
        Inst::Mulhu  { dest, src1, src2 } => DecodedInst { addr, op: "mulhu".into(),  rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: Some(reg_num(src2)), imm: None },
        Inst::Div    { dest, src1, src2 } => DecodedInst { addr, op: "div".into(),    rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: Some(reg_num(src2)), imm: None },
        Inst::Divu   { dest, src1, src2 } => DecodedInst { addr, op: "divu".into(),   rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: Some(reg_num(src2)), imm: None },
        Inst::Rem    { dest, src1, src2 } => DecodedInst { addr, op: "rem".into(),    rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: Some(reg_num(src2)), imm: None },
        Inst::Remu   { dest, src1, src2 } => DecodedInst { addr, op: "remu".into(),   rd: Some(reg_num(dest)), rs1: Some(reg_num(src1)), rs2: Some(reg_num(src2)), imm: None },
        Inst::Ecall    => DecodedInst { addr, op: "ecall".into(),  rd: None, rs1: None, rs2: None, imm: None },
        Inst::Ebreak   => DecodedInst { addr, op: "ebreak".into(), rd: None, rs1: None, rs2: None, imm: None },
        Inst::Fence { .. } => DecodedInst { addr, op: "fence".into(), rd: None, rs1: None, rs2: None, imm: None },
        _ => DecodedInst { addr, op: format!("unknown:{:?}", inst), rd: None, rs1: None, rs2: None, imm: None },
    }
}

// ---------------------------------------------------------------------------
// ELF decoding
// ---------------------------------------------------------------------------

pub fn decode_elf(data: &[u8]) -> anyhow::Result<(Vec<DecodedInst>, u32, u32)> {
    let obj = object::File::parse(data)?;
    let text = obj.section_by_name(".text")
        .ok_or_else(|| anyhow::anyhow!("no .text section"))?;
    let text_data = text.data()?;
    let text_addr = text.address() as u32;

    let mut decoded = Vec::new();
    for i in (0..text_data.len()).step_by(4) {
        let word = u32::from_le_bytes([
            text_data[i], text_data[i+1], text_data[i+2], text_data[i+3]
        ]);
        let addr = text_addr + i as u32;
        match Inst::decode(word) {
            Ok((inst, _)) => decoded.push(decode_one(addr, &inst)),
            Err(_) => decoded.push(DecodedInst {
                addr, op: format!("???_{:08x}", word),
                rd: None, rs1: None, rs2: None, imm: None,
            }),
        }
    }
    Ok((decoded, text_addr, text_data.len() as u32))
}

/// Extract function symbols from ELF: (address, size).
/// Sorted by address.
pub fn get_elf_functions(data: &[u8]) -> anyhow::Result<Vec<(u32, u32)>> {
    let obj = object::File::parse(data)?;
    let mut funcs: Vec<(u32, u32)> = Vec::new();
    for sym in obj.symbols() {
        if sym.kind() == object::SymbolKind::Text && sym.size() > 0 {
            funcs.push((sym.address() as u32, sym.size() as u32));
        }
    }
    funcs.sort_by_key(|&(addr, _)| addr);
    funcs.dedup_by_key(|f| f.0);
    Ok(funcs)
}

/// Extract function symbols with names: (address, size, name).
pub fn get_elf_functions_named(data: &[u8]) -> anyhow::Result<Vec<(u32, u32, String)>> {
    let obj = object::File::parse(data)?;
    let mut funcs: Vec<(u32, u32, String)> = Vec::new();
    for sym in obj.symbols() {
        if sym.kind() == object::SymbolKind::Text && sym.size() > 0 {
            let name = sym.name().unwrap_or("").to_string();
            funcs.push((sym.address() as u32, sym.size() as u32, name));
        }
    }
    funcs.sort_by_key(|f| f.0);
    funcs.dedup_by_key(|f| f.0);
    Ok(funcs)
}
