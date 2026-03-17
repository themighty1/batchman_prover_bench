//! RV32IM instruction decoding, CFG construction, and virtual register lifting.
//!
//! Submodules:
//! - [`decode`] — ELF parsing → physical-register instruction stream
//! - [`cfg`]    — CFG construction: basic blocks and control-flow edges
//! - [`lift`]   — virtual-register lifting
//! - [`legacy`] — LLVM Machine Outliner workaround (only needed for old ELFs)

pub mod decode;
pub mod cfg;
pub mod lift;
pub mod legacy;

// Re-export everything so existing `rv32::Foo` paths continue to work.
pub use decode::*;
pub use cfg::*;
pub use lift::*;
