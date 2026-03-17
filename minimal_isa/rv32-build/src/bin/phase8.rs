#![no_std]
#![no_main]

use core::panic::PanicInfo;

const OUTPUT_ADDR: u32 = 0x1000;
const INPUT_LEN_ADDR: u32 = 0x1004;

const HEAP_BASE: u32 = 0x0002_0000;
const HEAP_END: u32 = 0x000A_0000;
const HEAP_PTR_ADDR: u32 = 0x0009_FF00;

#[inline(never)]
fn heap_init() {
    unsafe { core::ptr::write_volatile(HEAP_PTR_ADDR as *mut u32, HEAP_BASE); }
}

#[inline(never)]
fn heap_alloc(size: u32) -> u32 {
    unsafe {
        let cur = core::ptr::read_volatile(HEAP_PTR_ADDR as *const u32);
        let aligned = (cur + 3) & !3;
        let next = aligned + size;
        if next > HEAP_END { return 0; }
        core::ptr::write_volatile(HEAP_PTR_ADDR as *mut u32, next);
        aligned
    }
}

#[inline(never)]
fn node_new(tag: u32, value: u32, child: u32) -> u32 {
    let ptr = heap_alloc(12);
    if ptr == 0 { return 0; }
    unsafe {
        core::ptr::write_volatile(ptr as *mut u32, tag);
        core::ptr::write_volatile((ptr + 4) as *mut u32, value);
        core::ptr::write_volatile((ptr + 8) as *mut u32, child);
    }
    ptr
}

#[inline(never)]
fn node_value(ptr: u32) -> u32 {
    if ptr == 0 { return 0; }
    unsafe { core::ptr::read_volatile((ptr + 4) as *const u32) }
}

#[inline(never)]
fn node_child(ptr: u32) -> u32 {
    if ptr == 0 { return 0; }
    unsafe { core::ptr::read_volatile((ptr + 8) as *const u32) }
}

#[inline(never)]
fn build_list(pos: u32, count: u32, depth: u32) -> u32 {
    if count == 0 || depth > 4 { return 0; }
    let first_val = build_value(pos, depth + 1);
    let first = node_new(3, first_val, 0);
    let mut prev = first;
    let mut i: u32 = 1;
    while i < count {
        let val = build_value(pos.wrapping_add(i.wrapping_mul(17)), depth + 1);
        let node = node_new(3, val, 0);
        if prev != 0 && node != 0 {
            unsafe { core::ptr::write_volatile((prev + 8) as *mut u32, node); }
        }
        prev = node;
        i += 1;
    }
    first
}

#[inline(never)]
fn build_value(pos: u32, depth: u32) -> u32 {
    if depth > 4 { return 0; }
    if (pos.wrapping_mul(2654435761) >> 31) == 0 {
        node_new(1, pos.wrapping_mul(2654435761), 0)
    } else {
        let child = build_list(pos.wrapping_add(1), 2, depth);
        node_new(3, 0, child)
    }
}

/// Recursive walk with sibling traversal — this triggers the bug.
#[inline(never)]
fn walk_tree(ptr: u32) -> u32 {
    if ptr == 0 { return 0; }
    let val = node_value(ptr);
    let child = node_child(ptr);
    let mut result = val;
    let mut next = child;
    while next != 0 {
        result = result.wrapping_add(walk_tree(next));
        next = node_child(next);
    }
    result
}

#[no_mangle]
pub extern "C" fn _start() -> ! {
    let n = unsafe { core::ptr::read_volatile(INPUT_LEN_ADDR as *const u32) };
    heap_init();
    let mut total: u32 = 0;
    let mut i: u32 = 0;
    while i < n {
        let root = build_value(i.wrapping_mul(97), 0);
        total = total.wrapping_add(walk_tree(root));
        i += 1;
    }
    unsafe { core::ptr::write_volatile(OUTPUT_ADDR as *mut u32, total); }
    loop {}
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! { loop {} }
