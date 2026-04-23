//! WebAssembly bindings for [searchlite](https://crates.io/crates/searchlite-core).
//!
//! `searchlite-wasm` compiles Searchlite to a `wasm32-unknown-unknown` target
//! so full-text search can run entirely in the browser. Indexes are persisted
//! to IndexedDB, ingest and search happen in a Web Worker by default, and
//! every operation surfaces typed `{ type, reason }` error payloads to JS.
//!
//! This crate only produces meaningful exports for the `wasm32` target; the
//! host-target build is a placeholder so `cargo check` keeps working in the
//! workspace.
//!
//! # Getting started
//!
//! Build with [`wasm-pack`](https://rustwasm.github.io/wasm-pack/):
//!
//! ```bash
//! wasm-pack build searchlite-wasm --target web --release
//! ```
//!
//! Then consume the generated `pkg/` directory from JavaScript / TypeScript.
//! See:
//!
//! - The [package README](https://github.com/davidkelley/searchlite/blob/main/searchlite-wasm/README.md)
//!   for a vanilla-ESM quickstart and worker-first example.
//! - [`docs/wasm.md`](https://github.com/davidkelley/searchlite/blob/main/docs/wasm.md)
//!   for the full API surface, build targets, threaded builds, and the runtime
//!   fallback matrix.
//! - [`docs/wasm-errors.md`](https://github.com/davidkelley/searchlite/blob/main/docs/wasm-errors.md)
//!   for every typed error code and the recommended recovery.

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
