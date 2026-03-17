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

#[inline(never)]
fn recurse_simple(pos: u32, depth: u32) -> u32 {
    if depth == 0 { return checksum(pos, 4); }
    let h = checksum(pos, 4);
    h.wrapping_add(recurse_simple(pos.wrapping_add(h & 0xF), depth - 1))
}

#[no_mangle]
pub extern "C" fn _start() -> ! {
    let n = unsafe { core::ptr::read_volatile(INPUT_LEN_ADDR as *const u32) };
    let mut result: u32 = 0;
    let mut i: u32 = 0;
    while i < n {
        result = result.wrapping_add(recurse_simple(i.wrapping_mul(113), 3));
        i += 1;
    }
    unsafe { core::ptr::write_volatile(OUTPUT_ADDR as *mut u32, result); }
    loop {}
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! { loop {} }
