#![no_std]
#![no_main]

extern crate rv32_rt; // pull in allocator + panic handler

const INPUT_LEN_ADDR: u32 = 0x1004;
const INPUT_DATA_ADDR: u32 = 0x2000;

#[no_mangle]
pub extern "C" fn _start() -> ! {
    unsafe { rv32_rt::init_heap() };

    unsafe {
        let input_len = rv32_rt::mmio_read_u32(INPUT_LEN_ADDR) as usize;
        let input_bytes = rv32_rt::mmio_read_bytes(INPUT_DATA_ADDR, input_len);

        let result = simple_predicate::run(input_bytes);
        rv32_rt::write_output(&result);
    }

    loop {}
}
