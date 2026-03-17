//! Minimal test: loop with more computation (shift+xor like checksum, but no call)
#![no_std]
#![no_main]

use core::panic::PanicInfo;

const OUTPUT_ADDR: u32 = 0x1000;
const INPUT_LEN_ADDR: u32 = 0x1004;

#[no_mangle]
pub extern "C" fn _start() -> ! {
    let n = unsafe { core::ptr::read_volatile(INPUT_LEN_ADDR as *const u32) };
    let mut cs: u32 = 0;
    let mut i: u32 = 0;
    while i < n {
        let b = i.wrapping_mul(2654435761) >> 24;
        cs = cs.wrapping_shl(1) ^ b;
        i += 1;
    }
    unsafe { core::ptr::write_volatile(OUTPUT_ADDR as *mut u32, cs); }
    loop {}
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! { loop {} }
