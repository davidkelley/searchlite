//! `S3Object` — a pinned-at-open `BlobStore::Object` over S3.

use std::ops::Range;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use searchlite_core::storage::blob::{Object, ObjectStat};

use crate::errors::S3StoreError;

/// Per-open S3 object handle. Captures the resolved S3 key and the
/// `ObjectStat` observed at open time, so [`Object::read_range`]
/// doesn't re-HEAD per call.
///
/// Stage 10b [Codex review #1]: `read_range` sends `If-Match: <etag>`
/// when the open-time ETag is known, so a concurrent overwrite by
/// another writer can't return bytes from a different generation than
/// the stat we cached. Without `If-Match`, an `open()` followed by
/// `read_range` would silently mix old-stat metadata with new-content
/// bytes — exactly the kind of stat/get race the BlobStore identity
/// model was designed to eliminate.
pub(crate) struct S3Object {
  pub(crate) client: Arc<aws_sdk_s3::Client>,
  pub(crate) bucket: String,
  pub(crate) key: String,
  pub(crate) stat: ObjectStat,
}

#[async_trait]
impl Object for S3Object {
  fn stat(&self) -> &ObjectStat {
    &self.stat
  }

  async fn read_range(&self, range: Range<u64>) -> Result<Bytes> {
    if range.start > range.end {
      anyhow::bail!(
        "S3Object::read_range: inverted range {}..{}",
        range.start,
        range.end
      );
    }
    if range.end > self.stat.len {
      anyhow::bail!(
        "S3Object::read_range: range {}..{} exceeds object length {} for {}",
        range.start,
        range.end,
        self.stat.len,
        self.key
      );
    }
    if range.start == range.end {
      return Ok(Bytes::new());
    }
    // S3 ranges are inclusive on both ends; `0..N` in Rust = `bytes=0-(N-1)` in HTTP.
    let header = format!("bytes={}-{}", range.start, range.end - 1);
    let mut req = self
      .client
      .get_object()
      .bucket(&self.bucket)
      .key(&self.key)
      .range(header);
    // Stage 10b [Codex review #1]: pin to the open-time ETag so a
    // concurrent overwrite can't surface mismatched bytes. ETag is
    // preserved verbatim — quotes included if the provider sent
    // them — because `If-Match` requires byte-for-byte equality.
    let sent_conditional = self.stat.provider_version.is_some();
    if let Some(etag) = self.stat.provider_version.as_deref() {
      req = req.if_match(etag);
    }
    // Stage 10b v4 [P3] (Codex review): only use the conditional-
    // aware mapper when the request actually sent `If-Match`.
    // Otherwise a server-returned 412/409 wouldn't carry the
    // "pinned ETag stale" semantics that `PreconditionFailed`
    // implies, and typing it as such would mislead callers.
    let resp = req.send().await.map_err(|sdk_err| {
      let ctx = format!("read_range {}", self.key);
      if sent_conditional {
        crate::store::map_conditional_sdk_error(sdk_err, &ctx)
      } else {
        crate::store::map_sdk_error(sdk_err, &ctx)
      }
    })?;
    let bytes = resp
      .body
      .collect()
      .await
      .map_err(anyhow::Error::new)?
      .into_bytes();
    Ok(bytes)
  }
}

/// Stage 10b: helper used by [`S3Object`] and the top-level
/// `BlobStore::get_range` to detect a 412/409 conditional failure
/// returned when the pinned object has been overwritten under the
/// reader. Surface as a typed [`S3StoreError::PreconditionFailed`]
/// so callers can discriminate from generic transport errors.
pub(crate) fn classify_conditional_status(status: u16, key: &str) -> Option<S3StoreError> {
  match status {
    412 | 409 => Some(S3StoreError::PreconditionFailed {
      context: format!("status {status} on {key}"),
    }),
    _ => None,
  }
}
