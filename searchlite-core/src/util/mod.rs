pub mod bitpack;
pub mod checksum;
pub mod fst;
pub mod varint;

#[cfg(not(target_arch = "wasm32"))]
pub mod mmap;
