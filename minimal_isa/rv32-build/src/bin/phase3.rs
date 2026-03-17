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
fn combined_scan(pos: u32, len: u32) -> u32 {
    let ws  = skip_whitespace(pos, len.min(16));
    let h   = scan_string(pos.wrapping_add(ws), len.min(8));
    let num = scan_number(pos.wrapping_add(h & 0xFF), len.min(6));
    h.wrapping_add(num).wrapping_add(ws)
}

#[no_mangle]
pub extern "C" fn _start() -> ! {
    let n = unsafe { core::ptr::read_volatile(INPUT_LEN_ADDR as *const u32) };
    let mut result: u32 = 0;
    let mut i: u32 = 0;
    while i < n {
        let len = 4 + (i & 7);
        result = result.wrapping_add(combined_scan(i.wrapping_mul(53), len));
        i += 1;
    }
    unsafe { core::ptr::write_volatile(OUTPUT_ADDR as *mut u32, result); }
    loop {}
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! { loop {} }
