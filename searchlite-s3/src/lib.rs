//! `searchlite-s3` — S3-compatible [`BlobStore`] implementation and
//! helpers for serving Searchlite indexes out of object storage.
//!
//! This crate provides:
//!
//! * [`S3BlobStore`] — a concrete `BlobStore` backed by the
//!   `aws-sdk-s3` client. Targets AWS S3, Cloudflare R2, and
//!   MinIO / LocalStack / wiremock. Provider defaults are exposed via
//!   [`S3Config::aws_default`] and [`S3Config::r2_default`].
//! * [`sync_to_s3`] — bake-locally-then-upload helper. Walks a local
//!   index directory, uploads every segment artifact, then publishes
//!   `MANIFEST.json` last as the visibility fence. Returns a
//!   [`SyncReport`] with file/byte counts. Refuses to run against a
//!   partially-baked index (pending manifest, non-empty WAL, staging
//!   files, legacy v1 manifests, non-canonical keys).
//! * [`open_index_read_only`] — top-level `Index` constructor for
//!   S3-backed read-only deployments. Wraps `S3BlobStore` in
//!   `CachedBlobStore` (64 MiB byte-weighted RAM cache), threads it
//!   through a `BlobStoreAdapter`, and returns an `Index` with every
//!   mutator wired off.
//! * [`open_index_read_only_with_options`] — same shape as
//!   [`open_index_read_only`] but lets the caller customize
//!   [`searchlite_core::api::types::IndexOptions`] (notably
//!   `checksum_policy`). `path`, `create_if_missing`, and `read_only`
//!   are still forced regardless of caller input.
//!
//! ## Provider compatibility
//!
//! * **AWS S3** — `endpoint_url: None`, `force_path_style: false`,
//!   `conditional_put: true`.
//! * **Cloudflare R2** — set `endpoint_url` to your account's R2 URL.
//!   Conditional PUTs (`If-Match` / `If-None-Match`) rolled out on R2
//!   in late 2024; the default is `conditional_put: false`. Opt in
//!   once you've confirmed your bucket supports them.
//! * **MinIO / LocalStack / wiremock** — set `endpoint_url` and
//!   `force_path_style: true`.
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
pub use open::{open_index_read_only, open_index_read_only_with_options};
pub use store::S3BlobStore;
pub use sync::{sync_to_s3, SyncReport};
