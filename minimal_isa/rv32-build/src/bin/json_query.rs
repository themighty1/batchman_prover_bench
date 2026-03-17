#![no_std]
#![no_main]

extern crate rv32_rt; // pull in allocator + panic handler

const INPUT_JSON_LEN_ADDR: u32 = 0x1004;
const INPUT_PATH_LEN_ADDR: u32 = 0x1008;
const INPUT_JSON_DATA_ADDR: u32 = 0x2000;
const INPUT_PATH_DATA_ADDR: u32 = 0x3000;

#[no_mangle]
pub extern "C" fn _start() -> ! {
    unsafe { rv32_rt::init_heap() };

    unsafe {
        let json_len = rv32_rt::mmio_read_u32(INPUT_JSON_LEN_ADDR) as usize;
        let json_bytes = rv32_rt::mmio_read_bytes(INPUT_JSON_DATA_ADDR, json_len);

        let path_len = rv32_rt::mmio_read_u32(INPUT_PATH_LEN_ADDR) as usize;
        let path_bytes = rv32_rt::mmio_read_bytes(INPUT_PATH_DATA_ADDR, path_len);

        let result = json_query::run(json_bytes, path_bytes);
        rv32_rt::write_output(&result);
    }

    loop {}
}
