//! Minimal test: _start with loop calling a leaf function that has a loop (= phase1)
#![no_std]
#![no_main]

use core::panic::PanicInfo;

const OUTPUT_ADDR: u32 = 0x1000;
const INPUT_LEN_ADDR: u32 = 0x1004;

#[inline(never)]
fn checksum(start: u32, len: u32) -> u32 {
    let mut cs: u32 = 0;
    let mut i: u32 = 0;
    while i < len {
        let b = (start.wrapping_add(i)).wrapping_mul(2654435761) >> 24;
        cs = cs.wrapping_shl(1) ^ b;
        i += 1;
    }
    cs
}

#[no_mangle]
pub extern "C" fn _start() -> ! {
    let n = unsafe { core::ptr::read_volatile(INPUT_LEN_ADDR as *const u32) };
    let mut result: u32 = 0;
    let mut i: u32 = 0;
    while i < n {
        let cs_len = 4 + (i & 7);
        result = result.wrapping_add(checksum(i, cs_len));
        i += 1;
    }
    unsafe { core::ptr::write_volatile(OUTPUT_ADDR as *mut u32, result); }
    loop {}
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! { loop {} }
