#![no_std]

extern crate alloc;

use alloc::string::String;

/// Entry point: parse decimal u32 from input, check predicate (> 700).
/// Returns "1" if the predicate holds, "0" otherwise.
pub fn run(input_bytes: &[u8]) -> String {
    let ok = (|| {
        let input_str = core::str::from_utf8(input_bytes).ok()?.trim();
        if !input_str.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let value = parse_u32(input_str)?;
        if value > 700 { Some(()) } else { None }
    })();

    String::from(if ok.is_some() { "1" } else { "0" })
}

fn parse_u32(s: &str) -> Option<u32> {
    let mut acc: u32 = 0;
    for b in s.bytes() {
        if b < b'0' || b > b'9' {
            return None;
        }
        acc = acc.checked_mul(10)?.checked_add((b - b'0') as u32)?;
    }
    Some(acc)
}

