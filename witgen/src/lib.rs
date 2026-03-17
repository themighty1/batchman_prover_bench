//! Shared witness types for external crate consumption.

use serde::{Serialize, Deserialize};

/// A contiguous region of pre-loaded memory.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemRegion {
    pub addr: u32,
    pub data: Vec<u8>,
}

/// Memory access record for a single execution step.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemAccessRow {
    pub is_read: bool,
    pub is_write: bool,
    pub read_addr: u32,
    pub read_value: u32,
    pub write_addr: u32,
    pub write_value: u32,
}

/// Per-step lookup trace row: what the code/imm tables yield for this step's pc.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LookupRow {
    pub pc: u16,
    pub op: u8,
    pub has_imm: u8,
    pub imm: i32,
    /// 1 if this step is a byte_sel_r2 instruction (op == 15).
    pub is_byte_sel_r2: u8,
}

/// Memory trace file: initial memory + per-step memory access trace.
///
/// Binary layout:
///   num_regions         u32 LE
///   for each region:
///     addr              u32 LE
///     len               u32 LE
///     data              [u8; len]
///   num_steps           u32 LE
///   for each step:
///     MemAccessRow      18 bytes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemTrace {
    pub initial: Vec<MemRegion>,
    pub access: Vec<MemAccessRow>,
}

/// Lookup trace file: per-step lookup trace.
///
/// Binary layout:
///   num_steps           u32 LE
///   for each step:
///     LookupRow         9 bytes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LookupTrace {
    pub rows: Vec<LookupRow>,
}

pub const MEM_ACCESS_ROW_BYTES: usize = 18;
pub const LOOKUP_ROW_BYTES: usize = 9;

/// Zero-product proof data from the prover (ALICE).
///
/// Per-branch topology MACs from `generate_proofs()`. The active branch's
/// MAC is zero (correct wiring); inactive branches have nonzero MACs.
///
/// Binary layout:
///   batch_sz:       u32 LE
///   branch_count:   u32 LE
///   for each step (batch_sz entries):
///     active_branch:    u16 LE
///     for each branch:
///       mac:            [u8; 8]   (GF(2^64))
///       plaintext:      [u8; 8]   (GF(2^64))
#[derive(Clone, Debug)]
pub struct ZeroProductProverData {
    pub batch_sz: u32,
    pub branch_count: u32,
    pub active_branches: Vec<u16>,
    /// `topology_macs[step * branch_count + bid]`. Active branch entry is zero.
    pub topology_macs: Vec<[u8; 8]>,
    /// `topology_plaintexts[step * branch_count + bid]`. Active branch entry is zero.
    pub topology_plaintexts: Vec<[u8; 8]>,
}

/// Zero-product proof data from the verifier (BOB).
///
/// Per-branch topology keys from `generate_proofs()`.
///
/// Binary layout:
///   batch_sz:       u32 LE
///   branch_count:   u32 LE
///   for each step (batch_sz entries):
///     topology_keys:    [u8; 8] * branch_count   (GF(2^64))
///
/// Delta is in a separate file (16 bytes raw).
#[derive(Clone, Debug)]
pub struct ZeroProductVerifierData {
    pub batch_sz: u32,
    pub branch_count: u32,
    /// `topology_keys[step * branch_count + bid]`.
    pub topology_keys: Vec<[u8; 8]>,
}

/// Step record data from the prover (ALICE).
///
/// Binary layout:
///   batch_sz:       u32 LE
///   branch_count:   u32 LE
///   for each step (batch_sz entries):
///     active_branch:  u16 LE
///     plaintexts:     [u8; 16] * branch_count  (126-bit packed, high 2 bits zero)
///     step_record_macs:  [u8; 16] * branch_count
#[derive(Clone, Debug)]
pub struct StepRecordProverData {
    pub batch_sz: u32,
    pub branch_count: u32,
    pub active_branches: Vec<u16>,
    /// Plaintexts for all branches: `plaintexts[step * branch_count + bid]`.
    pub plaintexts: Vec<[u8; 16]>,
    pub macs: Vec<[u8; 16]>,
}

/// Step record data from the verifier (BOB).
///
/// Binary layout:
///   batch_sz:       u32 LE
///   branch_count:   u32 LE
///   for each step (batch_sz entries):
///     step_record_keys:  [u8; 16] * branch_count
///
/// Delta is in a separate file (16 bytes raw).
#[derive(Clone, Debug)]
pub struct StepRecordVerifierData {
    pub batch_sz: u32,
    pub branch_count: u32,
    pub keys: Vec<[u8; 16]>,
}

impl MemAccessRow {
    pub fn to_bytes(&self) -> [u8; MEM_ACCESS_ROW_BYTES] {
        let mut buf = [0u8; MEM_ACCESS_ROW_BYTES];
        buf[0] = self.is_read as u8;
        buf[1] = self.is_write as u8;
        buf[2..6].copy_from_slice(&self.read_addr.to_le_bytes());
        buf[6..10].copy_from_slice(&self.read_value.to_le_bytes());
        buf[10..14].copy_from_slice(&self.write_addr.to_le_bytes());
        buf[14..18].copy_from_slice(&self.write_value.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8; MEM_ACCESS_ROW_BYTES]) -> Self {
        MemAccessRow {
            is_read: buf[0] != 0,
            is_write: buf[1] != 0,
            read_addr: u32::from_le_bytes(buf[2..6].try_into().unwrap()),
            read_value: u32::from_le_bytes(buf[6..10].try_into().unwrap()),
            write_addr: u32::from_le_bytes(buf[10..14].try_into().unwrap()),
            write_value: u32::from_le_bytes(buf[14..18].try_into().unwrap()),
        }
    }
}

impl LookupRow {
    pub fn to_bytes(&self) -> [u8; LOOKUP_ROW_BYTES] {
        let mut buf = [0u8; LOOKUP_ROW_BYTES];
        buf[0..2].copy_from_slice(&self.pc.to_le_bytes());
        buf[2] = self.op;
        buf[3] = self.has_imm;
        buf[4..8].copy_from_slice(&self.imm.to_le_bytes());
        buf[8] = self.is_byte_sel_r2;
        buf
    }

    pub fn from_bytes(buf: &[u8; LOOKUP_ROW_BYTES]) -> Self {
        LookupRow {
            pc: u16::from_le_bytes(buf[0..2].try_into().unwrap()),
            op: buf[2],
            has_imm: buf[3],
            imm: i32::from_le_bytes(buf[4..8].try_into().unwrap()),
            is_byte_sel_r2: buf[8],
        }
    }
}

impl MemTrace {
    pub fn write_to(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_all(&(self.initial.len() as u32).to_le_bytes())?;
        for region in &self.initial {
            w.write_all(&region.addr.to_le_bytes())?;
            w.write_all(&(region.data.len() as u32).to_le_bytes())?;
            w.write_all(&region.data)?;
        }
        w.write_all(&(self.access.len() as u32).to_le_bytes())?;
        for row in &self.access {
            w.write_all(&row.to_bytes())?;
        }
        Ok(())
    }

    pub fn read_from(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let mut buf4 = [0u8; 4];

        r.read_exact(&mut buf4)?;
        let num_regions = u32::from_le_bytes(buf4) as usize;
        let mut initial = Vec::with_capacity(num_regions);
        for _ in 0..num_regions {
            r.read_exact(&mut buf4)?;
            let addr = u32::from_le_bytes(buf4);
            r.read_exact(&mut buf4)?;
            let len = u32::from_le_bytes(buf4) as usize;
            let mut data = vec![0u8; len];
            r.read_exact(&mut data)?;
            initial.push(MemRegion { addr, data });
        }

        r.read_exact(&mut buf4)?;
        let num_steps = u32::from_le_bytes(buf4) as usize;
        let mut access = Vec::with_capacity(num_steps);
        let mut row_buf = [0u8; MEM_ACCESS_ROW_BYTES];
        for _ in 0..num_steps {
            r.read_exact(&mut row_buf)?;
            access.push(MemAccessRow::from_bytes(&row_buf));
        }

        Ok(MemTrace { initial, access })
    }
}

impl LookupTrace {
    pub fn write_to(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_all(&(self.rows.len() as u32).to_le_bytes())?;
        for row in &self.rows {
            w.write_all(&row.to_bytes())?;
        }
        Ok(())
    }

    pub fn read_from(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let mut buf4 = [0u8; 4];

        r.read_exact(&mut buf4)?;
        let num_steps = u32::from_le_bytes(buf4) as usize;
        let mut rows = Vec::with_capacity(num_steps);
        let mut row_buf = [0u8; LOOKUP_ROW_BYTES];
        for _ in 0..num_steps {
            r.read_exact(&mut row_buf)?;
            rows.push(LookupRow::from_bytes(&row_buf));
        }

        Ok(LookupTrace { rows })
    }
}

impl ZeroProductVerifierData {
    pub fn read_from(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        r.read_exact(&mut buf4)?;
        let batch_sz = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf4)?;
        let branch_count = u32::from_le_bytes(buf4);

        let steps = batch_sz as usize;
        let bc = branch_count as usize;
        let mut topology_keys = Vec::with_capacity(steps * bc);

        for _ in 0..steps {
            for _ in 0..bc {
                r.read_exact(&mut buf8)?;
                topology_keys.push(buf8);
            }
        }

        Ok(ZeroProductVerifierData { batch_sz, branch_count, topology_keys })
    }
}

impl ZeroProductProverData {
    pub fn read_from(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        r.read_exact(&mut buf4)?;
        let batch_sz = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf4)?;
        let branch_count = u32::from_le_bytes(buf4);

        let steps = batch_sz as usize;
        let bc = branch_count as usize;
        let mut active_branches = Vec::with_capacity(steps);
        let mut topology_macs = Vec::with_capacity(steps * bc);
        let mut topology_plaintexts = Vec::with_capacity(steps * bc);

        let mut buf2 = [0u8; 2];
        for _ in 0..steps {
            r.read_exact(&mut buf2)?;
            active_branches.push(u16::from_le_bytes(buf2));
            for _ in 0..bc {
                r.read_exact(&mut buf8)?;
                topology_macs.push(buf8);
                r.read_exact(&mut buf8)?;
                topology_plaintexts.push(buf8);
            }
        }

        Ok(ZeroProductProverData { batch_sz, branch_count, active_branches, topology_macs, topology_plaintexts })
    }
}

impl StepRecordProverData {
    pub fn read_from(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let mut buf4 = [0u8; 4];
        let mut buf16 = [0u8; 16];

        r.read_exact(&mut buf4)?;
        let batch_sz = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf4)?;
        let branch_count = u32::from_le_bytes(buf4);

        let steps = batch_sz as usize;
        let bc = branch_count as usize;
        let mut active_branches = Vec::with_capacity(steps);
        let mut plaintexts = Vec::with_capacity(steps * bc);
        let mut macs = Vec::with_capacity(steps * bc);

        let mut buf2 = [0u8; 2];
        for _ in 0..steps {
            r.read_exact(&mut buf2)?;
            active_branches.push(u16::from_le_bytes(buf2));
            for _ in 0..bc {
                r.read_exact(&mut buf16)?;
                plaintexts.push(buf16);
            }
            for _ in 0..bc {
                r.read_exact(&mut buf16)?;
                macs.push(buf16);
            }
        }

        Ok(StepRecordProverData { batch_sz, branch_count, active_branches, plaintexts, macs })
    }
}

impl StepRecordVerifierData {
    pub fn read_from(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let mut buf4 = [0u8; 4];
        let mut buf16 = [0u8; 16];

        r.read_exact(&mut buf4)?;
        let batch_sz = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf4)?;
        let branch_count = u32::from_le_bytes(buf4);

        let steps = batch_sz as usize;
        let bc = branch_count as usize;
        let mut keys = Vec::with_capacity(steps * bc);

        for _ in 0..steps {
            for _ in 0..bc {
                r.read_exact(&mut buf16)?;
                keys.push(buf16);
            }
        }

        Ok(StepRecordVerifierData { batch_sz, branch_count, keys })
    }
}
