#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]

#[cfg(target_arch = "wasm32")]
mod wasm;

#[cfg(target_arch = "wasm32")]
pub use wasm::*;

#[cfg(not(target_arch = "wasm32"))]
mod not_wasm {
  /// Placeholder to keep host-target builds working; real exports only exist for wasm32.
  pub fn wasm_only() {
    panic!("searchlite-wasm is only available for wasm32 targets");
  }
}
