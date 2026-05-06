//! S3 error mapping.
//!
//! Existing adapter code (notably `BlobStoreAdapter::open_append`)
//! walks the `anyhow::Error` chain looking for a `std::io::Error`
//! whose `kind() == NotFound`. To preserve that contract for
//! S3-backed callers, every method that could return an
//! "object/bucket not found" error must put a real `io::Error` of
//! kind `NotFound` into the chain. We emit that via
//! [`not_found_anyhow`] (so the chain root is an `io::Error` that
//! `chain().downcast_ref::<io::Error>()` will surface).
//!
//! Other S3-specific error kinds — conditional precondition failure
//! (412/409), missing conditional capability — are surfaced via the
//! strongly-typed [`S3StoreError`] enum so callers can downcast to
//! discriminate.

use std::io;

/// Strongly-typed S3-specific error variants. Surfaced as the inner
/// cause of an `anyhow::Error`; downcast via `error.downcast_ref()`.
#[derive(thiserror::Error, Debug)]
pub enum S3StoreError {
  /// Conditional PUT (`put_if_match`) was attempted but the backend
  /// does not advertise `Capabilities::conditional_put`. We refuse to
  /// issue a request rather than silently sending headers the endpoint
  /// would ignore.
  #[error(
    "S3 backend does not advertise conditional_put; \
     refusing to issue put_if_match. Check `S3Config::conditional_put`."
  )]
  ConditionalPutNotSupported,

  /// A precondition (`If-Match` or `If-None-Match`) failed. Returned
  /// from `put_if_match` and from conditional `read_range` when the
  /// pinned object's ETag has changed under the reader.
  ///
  /// S3 returns 412 PreconditionFailed; R2 may return 409
  /// ConditionalRequestConflict for racing conditional ops. Both map
  /// here.
  #[error("S3 conditional precondition failed: {context}")]
  PreconditionFailed { context: String },
}

/// Build an `anyhow::Error` whose chain root is an `io::Error` of
/// kind `NotFound`. Existing chain-walking logic in
/// `searchlite-core/src/storage/blob_adapter.rs` (`error_is_not_found`)
/// looks for exactly this shape, so S3-backed callers must produce
/// it for the "object missing" case.
pub fn not_found_anyhow(key: &str) -> anyhow::Error {
  anyhow::Error::new(io::Error::new(
    io::ErrorKind::NotFound,
    format!("S3 object not found: {key}"),
  ))
  .context(format!("S3 object {key} not found"))
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Sanity check: the chain walk used by
  /// `BlobStoreAdapter::error_is_not_found` succeeds against errors
  /// produced by [`not_found_anyhow`].
  #[test]
  fn not_found_anyhow_is_visible_to_chain_walk() {
    let err = not_found_anyhow("seg_X.terms");
    let saw_io_not_found = err.chain().any(|cause| {
      cause
        .downcast_ref::<io::Error>()
        .map(|e| e.kind() == io::ErrorKind::NotFound)
        .unwrap_or(false)
    });
    assert!(
      saw_io_not_found,
      "the io::Error must be visible to the BlobStoreAdapter chain walk"
    );
  }
}
