#![no_std]

use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;
use core::panic::PanicInfo;

// --- MMIO addresses (shared across all guest programs) ---
pub const OUTPUT_LEN_ADDR: u32 = 0x1000;
pub const OUTPUT_DATA_ADDR: u32 = 0x4000;

// --- Bump allocator ---

struct BumpAlloc {
    arena: UnsafeCell<*mut u8>,
    end: UnsafeCell<*mut u8>,
}

unsafe impl Sync for BumpAlloc {}

unsafe impl GlobalAlloc for BumpAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let cur = *self.arena.get();
        let align = layout.align();
        let aligned = ((cur as usize + align - 1) & !(align - 1)) as *mut u8;
        let next = aligned.add(layout.size());
        if next > *self.end.get() {
            core::ptr::null_mut()
        } else {
            *self.arena.get() = next;
            aligned
        }
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
}

const HEAP_SIZE: usize = 8 * 1024;
static mut HEAP: [u8; HEAP_SIZE] = [0u8; HEAP_SIZE];

#[global_allocator]
static ALLOCATOR: BumpAlloc = BumpAlloc {
    arena: UnsafeCell::new(core::ptr::null_mut()),
    end: UnsafeCell::new(core::ptr::null_mut()),
};

/// Initialize the bump allocator. Must be called at the top of _start().
pub unsafe fn init_heap() {
    let heap_start = core::ptr::addr_of_mut!(HEAP) as *mut u8;
    *ALLOCATOR.arena.get() = heap_start;
    *ALLOCATOR.end.get() = heap_start.add(HEAP_SIZE);
}

// --- MMIO helpers ---

/// Read a u32 from a volatile MMIO address.
pub unsafe fn mmio_read_u32(addr: u32) -> u32 {
    core::ptr::read_volatile(addr as *const u32)
}

/// Get a byte slice from a memory-mapped region.
pub unsafe fn mmio_read_bytes(addr: u32, len: usize) -> &'static [u8] {
    core::slice::from_raw_parts(addr as *const u8, len)
}

/// Write a string to the standard output MMIO region (0x1000 len, 0x4000 data).
pub unsafe fn write_output(s: &str) {
    let bytes = s.as_bytes();
    core::ptr::write_volatile(OUTPUT_LEN_ADDR as *mut u32, bytes.len() as u32);
    let dst = OUTPUT_DATA_ADDR as *mut u8;
    for (i, &b) in bytes.iter().enumerate() {
        core::ptr::write_volatile(dst.add(i), b);
    }
}

// --- Panic handler ---

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
