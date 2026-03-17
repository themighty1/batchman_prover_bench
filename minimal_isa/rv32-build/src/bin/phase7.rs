#![no_std]
#![no_main]

use core::panic::PanicInfo;

const OUTPUT_ADDR: u32 = 0x1000;
const INPUT_LEN_ADDR: u32 = 0x1004;

#[inline(always)]
fn peek_byte(pos: u32) -> u8 {
    (pos.wrapping_mul(2654435761) >> 24) as u8
}

#[inline(never)]
fn skip_whitespace(pos: u32, len: u32) -> u32 {
    let mut i: u32 = 0;
    while i < len {
        let b = peek_byte(pos.wrapping_add(i));
        if b >= 33 { break; }
        i += 1;
    }
    i
}

#[inline(never)]
fn scan_string(pos: u32, len: u32) -> u32 {
    let mut h: u32 = 5381;
    let mut i: u32 = 0;
    while i < len {
        let b = peek_byte(pos.wrapping_add(i));
        if b == b'\\' {
            i += 1;
            if i < len {
                let esc = peek_byte(pos.wrapping_add(i));
                h = h.wrapping_mul(37).wrapping_add(esc as u32);
            }
        } else if b == b'"' {
            break;
        } else {
            h = h.wrapping_mul(33).wrapping_add(b as u32);
        }
        i += 1;
    }
    h
}

#[inline(never)]
fn scan_number(pos: u32, len: u32) -> u32 {
    let mut acc: u32 = 0;
    let mut i: u32 = 0;
    while i < len {
        let b = peek_byte(pos.wrapping_add(i));
        let digit = b % 10;
        acc = acc.wrapping_mul(10).wrapping_add(digit as u32);
        i += 1;
    }
    let next = peek_byte(pos.wrapping_add(len));
    if next & 3 == 0 {
        acc = acc.wrapping_mul(100).wrapping_add(next as u32);
    }
    acc
}

#[inline(never)]
fn parse_value(pos: u32, remaining: u32, depth: u32) -> u32 {
    if remaining == 0 || depth > 8 {
        return 0;
    }

    let ws = skip_whitespace(pos, remaining.min(16));
    let p = pos.wrapping_add(ws);
    let rem = remaining.saturating_sub(ws);
    if rem == 0 { return 0; }

    let b = peek_byte(p);

    match b & 7 {
        0 => parse_object(p.wrapping_add(1), rem.saturating_sub(1), depth + 1),
        1 => parse_array(p.wrapping_add(1), rem.saturating_sub(1), depth + 1),
        2 | 3 => {
            let slen = (rem - 1).min(20);
            scan_string(p.wrapping_add(1), slen)
        }
        4 | 5 => {
            let nlen = rem.min(10);
            scan_number(p, nlen)
        }
        6 => {
            if b & 8 != 0 { 1 } else { 0 }
        }
        _ => {
            0
        }
    }
}

#[inline(never)]
fn parse_object(pos: u32, remaining: u32, depth: u32) -> u32 {
    let mut result: u32 = 200;
    let mut p = pos;
    let mut rem = remaining;
    let mut first = true;
    let mut count: u32 = 0;
    let max_pairs = 6u32;

    while rem > 2 && count < max_pairs {
        let ws = skip_whitespace(p, rem.min(8));
        p = p.wrapping_add(ws);
        rem = rem.saturating_sub(ws);
        if rem == 0 { break; }

        let b = peek_byte(p);
        if b & 0xF == 0xF { break; }

        if !first {
            p = p.wrapping_add(1);
            rem = rem.saturating_sub(1);
            if rem == 0 { break; }
        }
        first = false;

        let key_len = rem.min(8);
        let key_hash = scan_string(p, key_len);
        result = result.wrapping_add(key_hash);
        p = p.wrapping_add(key_len + 1);
        rem = rem.saturating_sub(key_len + 1);
        if rem == 0 { break; }

        let consumed = rem.min(16);
        let val = parse_value(p, consumed, depth);
        result = result.wrapping_add(val);
        p = p.wrapping_add(consumed);
        rem = rem.saturating_sub(consumed);

        count += 1;
    }

    result
}

#[inline(never)]
fn parse_array(pos: u32, remaining: u32, depth: u32) -> u32 {
    let mut result: u32 = 100;
    let mut p = pos;
    let mut rem = remaining;
    let mut first = true;
    let mut count: u32 = 0;
    let max_elems = 8u32;

    while rem > 1 && count < max_elems {
        let ws = skip_whitespace(p, rem.min(8));
        p = p.wrapping_add(ws);
        rem = rem.saturating_sub(ws);
        if rem == 0 { break; }

        let b = peek_byte(p);
        if b & 0xF == 0xE { break; }

        if !first {
            p = p.wrapping_add(1);
            rem = rem.saturating_sub(1);
            if rem == 0 { break; }
        }
        first = false;

        let consumed = rem.min(12);
        let val = parse_value(p, consumed, depth);
        result = result.wrapping_add(val);
        p = p.wrapping_add(consumed);
        rem = rem.saturating_sub(consumed);

        count += 1;
    }

    result
}

#[no_mangle]
pub extern "C" fn _start() -> ! {
    let n = unsafe { core::ptr::read_volatile(INPUT_LEN_ADDR as *const u32) };
    let mut result: u32 = 0;
    let mut i: u32 = 0;
    while i < n {
        let parse_len = 32 + (i & 31);
        result = result.wrapping_add(parse_value(i.wrapping_mul(97), parse_len, 0));
        i += 1;
    }
    unsafe { core::ptr::write_volatile(OUTPUT_ADDR as *mut u32, result); }
    loop {}
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! { loop {} }
