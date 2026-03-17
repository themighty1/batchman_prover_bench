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
fn match_key(pos: u32, key_hash: u32, len: u32) -> u32 {
    let mut h: u32 = 5381;
    let mut i: u32 = 0;
    while i < len {
        let b = peek_byte(pos.wrapping_add(i));
        h = h.wrapping_mul(33).wrapping_add(b as u32);
        i += 1;
    }
    if h == key_hash { 1 } else { 0 }
}

#[inline(never)]
fn checksum(start: u32, len: u32) -> u32 {
    let mut cs: u32 = 0;
    let mut i: u32 = 0;
    while i < len {
        let b = peek_byte(start.wrapping_add(i));
        cs = cs.wrapping_shl(1) ^ (b as u32);
        i += 1;
    }
    cs
}

/// Mid-level: two calls with local state held across
#[inline(never)]
fn validate_token(pos: u32, len: u32) -> u32 {
    let h = scan_string(pos, len);
    let cs = checksum(pos.wrapping_add(h & 0xF), len);
    h ^ cs
}

#[inline(never)]
fn match_pattern(pos: u32, len: u32) -> u32 {
    let m1 = match_key(pos, 12345, len);
    let m2 = match_key(pos.wrapping_add(4), 67890, len);
    m1.wrapping_mul(3).wrapping_add(m2)
}

#[inline(never)]
fn normalize_value(pos: u32, len: u32) -> u32 {
    let ws = skip_whitespace(pos, len);
    let num = scan_number(pos.wrapping_add(ws), len);
    ws.wrapping_shl(8).wrapping_add(num)
}

#[inline(never)]
fn extract_field(pos: u32, len: u32) -> u32 {
    let num = scan_number(pos, len);
    let h = scan_string(num & 0xFF, len);
    num.wrapping_add(h)
}

#[no_mangle]
pub extern "C" fn _start() -> ! {
    let n = unsafe { core::ptr::read_volatile(INPUT_LEN_ADDR as *const u32) };
    let mut result: u32 = 0;
    let mut i: u32 = 0;
    while i < n {
        let aux_len = 4 + (i & 7);
        let phase = (i.wrapping_mul(17)) & 3;
        if phase == 0 {
            result = result.wrapping_add(validate_token(i, aux_len));
        } else if phase == 1 {
            result = result.wrapping_add(match_pattern(i, aux_len));
        } else if phase == 2 {
            result = result.wrapping_add(normalize_value(i, aux_len));
        } else {
            result = result.wrapping_add(extract_field(i, aux_len));
        }
        i += 1;
    }
    unsafe { core::ptr::write_volatile(OUTPUT_ADDR as *mut u32, result); }
    loop {}
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! { loop {} }
