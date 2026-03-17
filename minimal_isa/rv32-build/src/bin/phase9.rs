#![no_std]
#![no_main]

use core::panic::PanicInfo;

const OUTPUT_ADDR: u32 = 0x1000;
const INPUT_LEN_ADDR: u32 = 0x1004;

/// Leaf function — forces caller to spill live registers across this call.
#[inline(never)]
fn leaf(x: u32) -> u32 {
    x.wrapping_mul(2654435761)
}

/// "Grandchild" function: called by wrapper_a, writes to spill slots
/// in the frame region that will later be reused by heavy_func.
/// Has many live values across calls → many spills.
#[inline(never)]
fn grandchild(a: u32, b: u32, c: u32) -> u32 {
    let r1 = leaf(a);
    // r1, b, c are all live across the next call → spills
    let r2 = leaf(b);
    // r1, r2, c live
    let r3 = leaf(c);
    r1.wrapping_add(r2).wrapping_add(r3)
}

/// Small wrapper: very few spill slots of its own.
/// Calls grandchild which uses more frame space.
#[inline(never)]
fn wrapper_a(x: u32) -> u32 {
    grandchild(x, x.wrapping_add(1), x.wrapping_add(2))
}

/// Heavy function: many parameters and live values across calls.
/// With low register counts, its spill frame extends past wrapper_a's
/// and overlaps with grandchild's old spill region.
#[inline(never)]
fn heavy_func(a: u32, b: u32, c: u32, d: u32) -> u32 {
    // 4 live values + return value across each call = lots of spills
    let r1 = leaf(a);
    // r1, b, c, d all live
    let r2 = leaf(b);
    // r1, r2, c, d live
    let r3 = leaf(c);
    // r1, r2, r3, d live
    let r4 = leaf(d);
    r1.wrapping_add(r2).wrapping_add(r3).wrapping_add(r4)
}

/// Parent function: calls wrapper_a, then heavy_func as sibling.
/// This creates the frame aliasing pattern.
#[inline(never)]
fn parent(n: u32) -> u32 {
    let a_result = wrapper_a(n);
    let h_result = heavy_func(
        a_result,
        n.wrapping_add(10),
        n.wrapping_add(20),
        n.wrapping_add(30),
    );
    a_result.wrapping_add(h_result)
}

#[no_mangle]
pub extern "C" fn _start() -> ! {
    let n = unsafe { core::ptr::read_volatile(INPUT_LEN_ADDR as *const u32) };
    let mut total: u32 = 0;
    let mut i: u32 = 0;
    while i < n {
        total = total.wrapping_add(parent(i.wrapping_mul(97)));
        i += 1;
    }
    unsafe { core::ptr::write_volatile(OUTPUT_ADDR as *mut u32, total); }
    loop {}
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! { loop {} }
