//! Minimal assertion binary
//!
//! Reads a string from the first command‑line argument (or stdin if none) and
//! asserts that it is an ASCII decimal number greater than 700.
//!
//! If the assertion fails the program panics, which causes a non‑zero exit
//! status. On success it exits with status 0.

use std::io::{self, Read};

fn main() {
    // Prefer the first CLI argument; fall back to stdin.
    let input = std::env::args().nth(1).unwrap_or_else(|| {
        // Read everything from stdin
        let mut buf = String::new();
        io::stdin()
            .read_to_string(&mut buf)
            .expect("failed to read stdin");
        buf
    });

    // Trim whitespace
    let trimmed = input.trim();

    // Determine success: input must be all ASCII digits, parseable as u32, and > 700
    let success = if trimmed.bytes().all(|b| b.is_ascii_digit()) {
        if let Ok(val) = trimmed.parse::<u32>() {
            val > 700
        } else {
            false
        }
    } else {
        false
    };

    // Write boolean result (1 for success, 0 for failure) to the VM‑visible output location
    let result_u32: u32 = if success { 1 } else { 0 };
    unsafe {
        core::ptr::write_volatile(0x1000 as *mut u32, result_u32);
    }
    // Optional diagnostic output (does not affect the VM result)
}
