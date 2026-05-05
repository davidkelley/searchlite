//! `searchlite-s3` — S3-compatible [`BlobStore`] implementation.
//!
//! Stage 10b of the searchlite cloud-storage migration. This crate
//! provides a single concrete [`S3BlobStore`] type that implements the
//! [`BlobStore`] trait defined in `searchlite-core`, mapped onto the
//! `aws-sdk-s3` client. Targets:
//!
//! * **AWS S3** — the canonical implementation. `endpoint_url: None`,
//!   `force_path_style: false`, `conditional_put: true`.
//! * **Cloudflare R2** — S3-compatible. Set `endpoint_url` to your
//!   account's R2 URL. Conditional PUTs (`If-Match` / `If-None-Match`)
//!   rolled out on R2 in late 2024; default `conditional_put: false`
//!   and opt in once you've confirmed your bucket supports it.
//! * **MinIO / LocalStack / wiremock** — set `endpoint_url` and
//!   `force_path_style: true`.
//!
//! ## Stage 10b scope
//!
//! Per Stage 10b's plan: this crate ships the `BlobStore` trait impl
//! and protocol-level unit tests via wiremock. End-to-end Index open
//! over S3 (the `BlobStoreAdapter<CachedBlobStore<S3BlobStore>>`
//! shape) and any CLI/HTTP wiring land in Stage 10c.
//!
//! ## Runtime requirements
//!
//! `aws-sdk-s3` futures depend on the Tokio reactor — `hyper`,
//! `tokio-rustls`, and `tokio-util` are all baked into the SDK. Sync
//! callers must drive these futures via
//! [`searchlite_core::runtime::block_on_blob`] with the
//! `tokio-runtime` feature enabled (this crate does that
//! transitively). The bridge supports every Tokio runtime flavor:
//! multi-thread runtimes use `block_in_place` to park the worker, and
//! current-thread runtimes route the future to a `std::thread::scope`-spawned
//! worker on a global multi-thread bridge runtime. See the runtime
//! module's docs for the full contract.

#![cfg(not(target_arch = "wasm32"))]

mod config;
mod errors;
mod object;
mod open;
mod store;
mod sync;

pub use config::{S3Config, S3Credentials};
pub use errors::S3StoreError;
pub use open::open_index_read_only;
pub use store::S3BlobStore;
pub use sync::{sync_to_s3, SyncReport};
