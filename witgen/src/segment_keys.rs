//! Read a segment's zero_product_prover.bin + delta.bin, compute keys,
//! output active MACs and all keys to stdout as binary.
//!
//! Output format (all LE):
//!   num_active: u32
//!   num_all_keys: u32
//!   active_macs: [u64; num_active]
//!   all_keys: [u64; num_all_keys]
//!
//! Usage: segment_keys <segment_dir>

use std::io::{Write, BufReader};
use std::fs::File;
use anyhow::Result;
use batchman_witness_generator::ZeroProductProverData;

fn gf64_mul(a: u64, b: u64) -> u64 {
    let mut product: u128 = 0;
    for i in 0..64 {
        if (b >> i) & 1 != 0 {
            product ^= (a as u128) << i;
        }
    }
    let lo = product as u64;
    let hi = (product >> 64) as u64;
    let r1 = hi << 4;
    let r2 = hi << 3;
    let r3 = hi << 1;
    let reduced = r1 ^ r2 ^ r3 ^ hi;
    lo ^ reduced
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let seg_dir = args.get(1).unwrap_or_else(|| {
        eprintln!("Usage: segment_keys <segment_dir>");
        std::process::exit(1);
    });

    // Read zero_product_prover.bin
    let zp_path = format!("{}/zero_product_prover.bin", seg_dir);
    let mut f = BufReader::new(File::open(&zp_path)?);
    let data = ZeroProductProverData::read_from(&mut f)?;

    let num_steps = data.batch_sz as usize;
    let branch_count = data.branch_count as usize;

    // Read delta
    let delta_path = format!("{}/delta.bin", seg_dir);
    let delta_raw = std::fs::read(&delta_path)?;
    anyhow::ensure!(delta_raw.len() == 16, "delta.bin must be 16 bytes");
    let delta_val = u64::from_le_bytes(delta_raw[..8].try_into().unwrap());

    // Compute active MACs and all keys
    let mut active_macs: Vec<u64> = Vec::with_capacity(num_steps);
    let mut all_keys: Vec<u64> = Vec::with_capacity(num_steps * branch_count);

    for step in 0..num_steps {
        let active = data.active_branches[step] as usize;
        for bid in 0..branch_count {
            let idx = step * branch_count + bid;
            let mac_val = u64::from_le_bytes(data.topology_macs[idx]);
            let pt_val = u64::from_le_bytes(data.topology_plaintexts[idx]);
            let key_val = mac_val ^ gf64_mul(delta_val, pt_val);
            all_keys.push(key_val);

            if bid == active {
                active_macs.push(mac_val);
            }
        }
    }

    // Write to stdout
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    out.write_all(&(active_macs.len() as u32).to_le_bytes())?;
    out.write_all(&(all_keys.len() as u32).to_le_bytes())?;
    for v in &active_macs {
        out.write_all(&v.to_le_bytes())?;
    }
    for v in &all_keys {
        out.write_all(&v.to_le_bytes())?;
    }
    out.flush()?;

    Ok(())
}
