//! BlobStore: object-storage-shaped abstraction for searchlite segments.
//!
//! Stage 5 of the BlobStore migration plan defines the substrate types and
//! the [`BlobStore`] trait without any implementations or call-site
//! migrations. Stage 6 will add `LocalBlobStore` (and a `Storage`-over-
//! `BlobStore` adapter); Stage 7 layers a bounded-LRU cache; Stage 8
//! migrates the segment readers' hot paths to bounded `read_range` calls.
//!
//! ## Type model
//!
//! Two distinct identity concepts live on this surface:
//!
//! - [`ObjectStat`] — the *observed* state from a `HEAD` (or `stat`) at a
//!   particular moment. Carries `len`, `provider_version` (e.g. an S3 ETag
//!   or R2 version token — opaque to us; not always a content hash, see
//!   multipart-upload semantics), and an optional `provider_checksum` from
//!   the provider itself.
//! - [`ArtifactIdentity`] — the *expected* identity recorded in the
//!   manifest. Carries the manifest-recorded `content_hash`, computed by
//!   the writer at segment-write time. Stage 9 (portable manifests) is what
//!   actually populates these; Stage 5 just defines the type.
//!
//! Keeping these separate lets cache layers key on the trustworthy
//! `content_hash` (we computed it) without conflating it with the
//! provider's `etag` (we didn't).
//!
//! ## Conditional PUT
//!
//! [`BlobStore::put_if_match`] is the CAS primitive. It is only callable on
//! backends whose [`Capabilities::conditional_put`] is `true`; other
//! backends MUST NOT fake CAS via stat-then-put (that's a TOCTOU race, not
//! atomic CAS). The index-level commit code chooses an alternative
//! strategy on backends without conditional support — single-writer
//! lockfile, or refusing to commit. The conflict surface is typed
//! ([`PutIfMatchError::Conflict`]) and carries the current `ObjectStat` so
//! callers can retry with up-to-date state without string-matching error
//! messages.

use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;

/// SHA-256-shaped content hash. Computed by the writer at segment-write
/// time and recorded in the manifest as part of [`ArtifactIdentity`].
/// Stage 5 only defines the type; Stage 9 (portable manifests) is what
/// actually populates these into existing segment metadata.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentHash([u8; 32]);

impl ContentHash {
  pub const fn new(bytes: [u8; 32]) -> Self {
    Self(bytes)
  }

  pub const fn as_bytes(&self) -> &[u8; 32] {
    &self.0
  }
}

impl std::fmt::Debug for ContentHash {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "ContentHash({})", HexBytes(&self.0))
  }
}

impl std::fmt::Display for ContentHash {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", HexBytes(&self.0))
  }
}

/// Provider-emitted integrity checksum returned by `stat`. The variants
/// cover the algorithms S3 / R2 surface today via `x-amz-checksum-*`
/// headers; we do not interpret them in Stage 5, only carry them. Callers
/// that need to validate against an expected value can match on the
/// algorithm and compare bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderChecksum {
  Crc32(u32),
  Crc32C(u32),
  Sha1([u8; 20]),
  Sha256([u8; 32]),
}

/// Observed state of an object via `HEAD` / `stat`. Distinct from
/// [`ArtifactIdentity`]: this is what the *backend* reports right now;
/// the manifest's identity is what the writer *recorded*. They may diverge
/// transiently (e.g. provider returning eventual-consistency state) or
/// permanently (corruption); cache layers use the content hash from the
/// manifest, not the `provider_version` here.
#[derive(Debug, Clone)]
pub struct ObjectStat {
  pub len: u64,
  /// Opaque provider version token: S3 ETag, R2 version, etc. Not always
  /// a content hash — multipart-upload ETags are MD5-of-MD5s, and some
  /// providers use generation counters.
  pub provider_version: Option<String>,
  /// Optional provider-emitted integrity checksum from `x-amz-checksum-*`
  /// or equivalent. `None` for backends that don't expose one.
  pub provider_checksum: Option<ProviderChecksum>,
}

/// Manifest-recorded expected identity for a segment artifact. Carries the
/// content hash the writer computed at segment-write time, plus the
/// resolved object key and length. Stage 9 populates these into manifests
/// in place of (or alongside) the existing CRC32 checksums; Stage 5 just
/// defines the type so trait callers have a stable shape to consume.
#[derive(Debug, Clone)]
pub struct ArtifactIdentity {
  pub key: PathBuf,
  pub len: u64,
  pub content_hash: ContentHash,
}

/// Capability flags an implementation declares about its backing provider.
/// Stable for the lifetime of the impl — capability discovery is a
/// constructor-time concern, not a per-call concern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capabilities {
  /// True iff the backend supports atomic conditional PUT (e.g. S3
  /// `If-Match` / `If-None-Match`). When false, [`BlobStore::put_if_match`]
  /// MUST NOT be called; index-level commit code selects a different
  /// strategy or refuses to commit. There is no fallback to stat-then-put
  /// CAS; that pattern is a TOCTOU race, not atomic CAS.
  pub conditional_put: bool,
  /// True iff the backend supports true *multipart* / parallel-part
  /// uploads (e.g. S3 `CreateMultipartUpload` + `UploadPart` x N +
  /// `CompleteMultipartUpload`). This is informational: it tells callers
  /// they can let an [`ObjectWriter`] internally fan out concurrent part
  /// uploads instead of streaming serially.
  ///
  /// **It does not gate availability of [`BlobStore::put_stream`]** —
  /// `put_stream` is a mandatory trait method and always available.
  /// Backends without multipart (e.g. local filesystem) implement
  /// `put_stream` as a serial streaming write to the underlying file.
  /// Callers should always prefer `put_stream` for large bodies regardless
  /// of this flag's value.
  pub multipart_upload: bool,
  /// True iff in-process `mmap` of the backing storage is sound. Local FS
  /// backends set this to `true`; object-storage backends set it to
  /// `false` (network-backed range reads aren't mmap-able).
  pub mmap_friendly: bool,
}

/// Conflict surface for [`BlobStore::put_if_match`]. The `current` field
/// carries the backend-observed state at conflict time so the caller can
/// retry with fresh expectations (or merge their state and try again)
/// without string-matching error messages or issuing a separate `stat`.
#[derive(Debug)]
pub enum PutIfMatchError {
  /// The provider's atomic precondition (`If-Match` / `If-None-Match`)
  /// failed. `current` is the backend-observed `ObjectStat` at conflict
  /// time, when the provider exposes it; some providers omit it.
  Conflict { current: Option<ObjectStat> },
  /// Any other error: I/O, transport, malformed response, etc.
  Other(anyhow::Error),
}

impl std::fmt::Display for PutIfMatchError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Conflict { current } => {
        write!(f, "put_if_match precondition failed; observed {current:?}")
      }
      Self::Other(e) => write!(f, "{e}"),
    }
  }
}

impl std::error::Error for PutIfMatchError {
  fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
    match self {
      Self::Conflict { .. } => None,
      Self::Other(e) => Some(e.as_ref()),
    }
  }
}

impl From<anyhow::Error> for PutIfMatchError {
  fn from(e: anyhow::Error) -> Self {
    Self::Other(e)
  }
}

/// Cheap handle over an opened object. Pinned to the [`ObjectStat`]
/// observed at open time, so subsequent [`Object::read_range`] calls have
/// a stable upper bound (`stat().len`) without a second `stat` round-trip.
///
/// A handle does not guarantee the underlying object is unchanged for the
/// handle's lifetime — Stage 5 doesn't impose that; provider-specific
/// impls may use `If-Match` on each range read or accept divergence.
#[async_trait]
pub trait Object: Send + Sync {
  /// The state observed at open time. Immutable for the lifetime of this
  /// handle.
  fn stat(&self) -> &ObjectStat;

  /// Convenience accessor: the object's length at open time.
  fn len(&self) -> u64 {
    self.stat().len
  }

  /// Returns `true` iff the object was empty at open time.
  fn is_empty(&self) -> bool {
    self.len() == 0
  }

  /// Read the byte range `[range.start, range.end)` from the object.
  ///
  /// **Range contract**: `range.start <= range.end <= self.len()`.
  /// Implementations MUST return an error for inverted ranges (`start >
  /// end`) — a `Range<u64>` can express that, but it has no meaningful
  /// translation to backend protocols (`bytes=start-(end-1)` would
  /// underflow on S3). Implementations MUST also return an error when
  /// `range.end > self.len()` so a truncated object doesn't silently
  /// short-read.
  ///
  /// Empty ranges (`range.start == range.end`) return `Bytes::new()`
  /// without issuing a backend read.
  async fn read_range(&self, range: Range<u64>) -> Result<Bytes>;
}

/// Streaming write handle returned by [`BlobStore::put_stream`].
/// Implementations are expected to use multipart uploads on capable
/// backends so large segment writes don't buffer in memory.
///
/// Callers MUST call either [`ObjectWriter::complete`] or
/// [`ObjectWriter::abort`] before the writer drops. On multipart-capable
/// backends, dropping a writer without calling either of those leaks
/// in-progress parts that cost money — implementations should defend
/// against this in their `Drop` impl, but the contract is on the caller.
#[async_trait]
pub trait ObjectWriter: Send {
  /// Append `chunk` to the in-progress upload. Repeated calls accumulate.
  async fn write(&mut self, chunk: Bytes) -> Result<()>;

  /// Finalize the upload and return the resulting [`ObjectStat`]. On
  /// multipart backends this completes the multipart upload (`Complete
  /// MultipartUpload` on S3); on simpler backends it flushes any buffered
  /// state and returns the stat of the finished object.
  async fn complete(self: Box<Self>) -> Result<ObjectStat>;

  /// Abort the upload. On multipart backends this MUST issue an `Abort
  /// MultipartUpload` so in-progress parts are reclaimed; on simpler
  /// backends it removes any partially-written object.
  async fn abort(self: Box<Self>) -> Result<()>;
}

/// Object-storage-shaped storage abstraction. Stage 6 will provide a
/// `LocalBlobStore` impl over `std::fs` and a `Storage`-over-`BlobStore`
/// adapter so existing index code can run through this surface
/// transparently. Stage 7 layers a bounded-LRU cache on top. Stage 8
/// migrates the segment-internal hot paths (postings / docstore range
/// reads) to consume `BlobStore` directly.
///
/// All read methods return [`Bytes`], so caches and request-fan-out layers
/// can clone results cheaply. All write methods take [`Bytes`] for the
/// same reason — callers building bodies from owned `Vec<u8>` can convert
/// via `Bytes::from(vec)` at no allocation cost.
///
/// The trait is `Send + Sync` and intended to be held as `Arc<dyn BlobStore>`
/// inside `InnerIndex` (Stage 6+). `#[async_trait]` is used so the trait
/// is object-safe; native `async fn` in traits doesn't satisfy the
/// `dyn`-with-`Send` bounds that holding `Arc<dyn BlobStore>` requires.
#[async_trait]
pub trait BlobStore: Send + Sync {
  /// Returns the observed state of `key` (size, provider version,
  /// optional checksum). Cheap on object stores (a `HEAD` request); on
  /// local FS impls, it's a single `metadata` call.
  async fn stat(&self, key: &Path) -> Result<ObjectStat>;

  /// Open a handle pinned to `key`'s observed [`ObjectStat`]. Subsequent
  /// reads through the handle don't re-stat. Cheap on every backend.
  async fn open(&self, key: &Path) -> Result<Arc<dyn Object>>;

  /// Range-shaped GET — the hot read primitive for Stage 8's segment
  /// reader migration. Implementations should issue a single bounded
  /// range read against the backend (`bytes=start-(end-1)` on S3) and
  /// avoid pulling more bytes than `range.len()`.
  ///
  /// **Range contract**: `range.start <= range.end <= object_len`, where
  /// `object_len` is the current size of the object at `key`.
  /// Implementations MUST return an error for inverted ranges (`start >
  /// end`) — a `Range<u64>` can express that, but it has no meaningful
  /// translation to backend protocols (`bytes=start-(end-1)` would
  /// underflow on S3). Implementations MUST also return an error when
  /// `range.end` exceeds the object length so a truncated object doesn't
  /// silently short-read.
  ///
  /// Empty ranges (`range.start == range.end`) return `Bytes::new()`
  /// without issuing a backend read.
  async fn get_range(&self, key: &Path, range: Range<u64>) -> Result<Bytes>;

  /// Whole-object GET. Sugar over `get_range(key, 0..stat(key).len)` for
  /// known-small objects. Implementations may optimize the small-object
  /// case (e.g. avoiding the separate `stat`).
  async fn get(&self, key: &Path) -> Result<Bytes>;

  /// Whole-object PUT with the body buffered in memory. Returns the
  /// resulting [`ObjectStat`]. For large bodies, prefer [`Self::put_stream`].
  async fn put(&self, key: &Path, body: Bytes) -> Result<ObjectStat>;

  /// Open a streaming/multipart writer. Callers must finalize via
  /// [`ObjectWriter::complete`] or release via [`ObjectWriter::abort`]
  /// before drop; see the [`ObjectWriter`] docs for the leakage contract.
  async fn put_stream(&self, key: &Path) -> Result<Box<dyn ObjectWriter>>;

  /// Conditional PUT using the provider's atomic precondition (`If-Match`
  /// / `If-None-Match`). The `expected` argument is a provider-version
  /// token from a prior [`ObjectStat::provider_version`]; `None` requests
  /// "must not exist" semantics where the provider supports it
  /// (`If-None-Match: *`).
  ///
  /// **`expected` is a provider version, not a content hash.** Backends
  /// without conditional-PUT support MUST NOT fake this primitive via
  /// stat-then-put on a content hash — that pattern is a TOCTOU race, not
  /// atomic CAS. Such backends should declare
  /// `Capabilities::conditional_put = false`, and callers must check
  /// capabilities before invoking this method.
  ///
  /// Returns [`PutIfMatchError::Conflict`] with the current
  /// [`ObjectStat`] (when the provider exposes it) on precondition
  /// failure, so callers can retry with up-to-date expectations without a
  /// separate `stat` round-trip.
  async fn put_if_match(
    &self,
    key: &Path,
    body: Bytes,
    expected: Option<&str>,
  ) -> std::result::Result<ObjectStat, PutIfMatchError>;

  /// Delete an object by key. Idempotent on most backends (the absence of
  /// the key is not an error).
  async fn delete(&self, key: &Path) -> Result<()>;

  /// Capability discovery. Stable for the lifetime of this impl —
  /// callers may cache the result.
  fn capabilities(&self) -> Capabilities;
}

/// Internal helper for hex-formatting a byte slice without pulling in a
/// `hex` crate dependency. Used by `ContentHash`'s Debug/Display impls.
struct HexBytes<'a>(&'a [u8]);

impl<'a> std::fmt::Display for HexBytes<'a> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    for b in self.0 {
      write!(f, "{b:02x}")?;
    }
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn content_hash_hex_round_trip() {
    let bytes = [
      0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
      0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
      0x1e, 0xff,
    ];
    let h = ContentHash::new(bytes);
    let display = format!("{h}");
    assert_eq!(
      display, "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1eff",
      "Display must produce lowercase hex with no separators"
    );
    let debug = format!("{h:?}");
    assert!(
      debug.starts_with("ContentHash(") && debug.ends_with(')'),
      "Debug must wrap the hex in `ContentHash(...)`, got: {debug}"
    );
    assert_eq!(h.as_bytes(), &bytes);
  }

  #[test]
  fn put_if_match_error_conflict_carries_current_stat() {
    let stat = ObjectStat {
      len: 42,
      provider_version: Some("\"abc\"".into()),
      provider_checksum: Some(ProviderChecksum::Crc32(0xdeadbeef)),
    };
    let err = PutIfMatchError::Conflict {
      current: Some(stat.clone()),
    };
    let formatted = format!("{err}");
    assert!(
      formatted.contains("precondition failed") && formatted.contains("len: 42"),
      "Display must surface the conflict reason and the current stat: {formatted}"
    );
    // Conflict carries no `source` — it's a structured signal, not a
    // wrapped error chain. Other(...) is where wrapped errors live.
    use std::error::Error;
    assert!(err.source().is_none());
  }

  #[test]
  fn put_if_match_error_from_anyhow_wraps_via_other() {
    let err: PutIfMatchError = anyhow::anyhow!("disk full").into();
    match err {
      PutIfMatchError::Other(e) => assert!(e.to_string().contains("disk full")),
      PutIfMatchError::Conflict { .. } => panic!("From<anyhow::Error> must produce Other"),
    }
  }

  /// Object-safety check: holding the trait as `Arc<dyn BlobStore>` must
  /// compile, which is the shape Stage 6+ uses inside `InnerIndex`.
  /// `async_trait` is the load-bearing dep here; native `async fn` in
  /// traits doesn't satisfy `dyn`-with-`Send` bounds for this case.
  #[test]
  fn dyn_blob_store_is_object_safe() {
    fn _accepts_dyn(_store: &Arc<dyn BlobStore>) {}
    fn _accepts_dyn_object(_obj: &Arc<dyn Object>) {}
    fn _accepts_dyn_writer(_w: &mut dyn ObjectWriter) {}
    // The functions above will fail to compile if any trait is not
    // object-safe. Construction is unnecessary; the type-level check is
    // what we want.
  }

  #[test]
  fn capabilities_is_copy_and_eq() {
    let cap = Capabilities {
      conditional_put: true,
      multipart_upload: false,
      mmap_friendly: false,
    };
    let copy = cap;
    assert_eq!(cap, copy);
  }
}
