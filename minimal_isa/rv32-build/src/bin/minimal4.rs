//! Minimal test: _start with loop calling a trivial leaf function (no loop in leaf)
#![no_std]
#![no_main]

use core::panic::PanicInfo;

const OUTPUT_ADDR: u32 = 0x1000;
const INPUT_LEN_ADDR: u32 = 0x1004;

#[inline(never)]
fn add42(x: u32) -> u32 {
    x.wrapping_add(42)
}

#[no_mangle]
pub extern "C" fn _start() -> ! {
    let n = unsafe { core::ptr::read_volatile(INPUT_LEN_ADDR as *const u32) };
    let mut acc: u32 = 0;
    let mut i: u32 = 0;
    while i < n {
        acc = acc.wrapping_add(add42(i));
        i += 1;
    }
    unsafe { core::ptr::write_volatile(OUTPUT_ADDR as *mut u32, acc); }
    loop {}
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! { loop {} }
