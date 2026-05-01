//! `StorageAsBlobStore` — Stage 8 transitional adapter.
//!
//! Wraps an `Arc<dyn Storage>` and exposes it as a [`BlobStore`],
//! serving raw file bytes WITHOUT the 37-byte version header that
//! [`LocalBlobStore`](super::LocalBlobStore) prepends. This adapter is
//! the bridge that lets Stage 8a route segment-internal reads through
//! the BlobStore surface without first migrating segment writes to the
//! header-bearing format.
//!
//! ## What it doesn't do
//!
//! - **No CAS**: `provider_version` is always `None` (the `Storage`
//!   trait has no per-write version primitive). Codex flagged the
//!   downstream consequence in Stage 7: observed-mode `CachedBlobStore`
//!   bypasses caching when version is `None`. That's the correct
//!   degradation here — the cache has nothing to safely key on, and
//!   trusted-mode (via the manifest's `ContentHash`) is the right
//!   caching path for these files once Stage 9 populates manifests.
//! - **`put_if_match` returns `Other`**: capabilities advertise
//!   `conditional_put: false`, and the trait contract is that callers
//!   check capabilities first.
//! - **`put_stream` buffers**: there's no streaming primitive in
//!   `Storage`; we accumulate in memory and flush via `atomic_write`
//!   on `complete`. Acceptable for the small-file write paths
//!   `Storage` callers use.
//!
//! ## When it's used
//!
//! Stage 8a wires this adapter as the default `blob_store` on
//! `InnerIndex` so segment readers can open `Object` handles for
//! postings (and, in 8b, docstore) without changing how segments are
//! written. Stage 9 and beyond replace this with a real `BlobStore`
//! (`LocalBlobStore` for local FS, `S3BlobStore` for cloud) and
//! rebuild segments using the native format.

use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use bytes::Bytes;

use super::blob::{BlobStore, Capabilities, Object, ObjectStat, ObjectWriter, PutIfMatchError};
use super::Storage;

/// Adapter wrapping `Arc<dyn Storage>` and exposing it as a `BlobStore`
/// with raw file bytes (no header). See module docs for the contract
/// and the limitations relative to a real BlobStore.
pub struct StorageAsBlobStore {
  storage: Arc<dyn Storage>,
}

impl StorageAsBlobStore {
  pub fn new(storage: Arc<dyn Storage>) -> Self {
    Self { storage }
  }
}

#[async_trait]
impl BlobStore for StorageAsBlobStore {
  async fn stat(&self, key: &Path) -> Result<ObjectStat> {
    // Stage 8a [P1] fix (Codex review): use `open_read + seek(End)` to
    // get the file length without slurping the entire file via
    // `read_to_end`. Previously `stat` paid a full-file read on every
    // segment open — the dominant cost the BlobStore migration is
    // supposed to eliminate. `open_read` returns a stateful handle;
    // `seek(SeekFrom::End(0))` is metadata-only on every supported
    // backend and returns the file size as `u64`.
    let mut file = self
      .storage
      .open_read(key)
      .with_context(|| format!("StorageAsBlobStore::stat open({})", key.display()))?;
    let len = file
      .seek(SeekFrom::End(0))
      .with_context(|| format!("StorageAsBlobStore::stat seek({})", key.display()))?;
    Ok(ObjectStat {
      len,
      provider_version: None,
      provider_checksum: None,
    })
  }

  async fn open(&self, key: &Path) -> Result<Arc<dyn Object>> {
    let stat = self.stat(key).await?;
    Ok(Arc::new(StorageObject {
      storage: self.storage.clone(),
      key: key.to_path_buf(),
      stat,
    }))
  }

  async fn get_range(&self, key: &Path, range: Range<u64>) -> Result<Bytes> {
    if range.start > range.end {
      bail!(
        "StorageAsBlobStore::get_range: inverted range {}..{}",
        range.start,
        range.end
      );
    }
    let mut file = self
      .storage
      .open_read(key)
      .with_context(|| format!("StorageAsBlobStore::get_range open({})", key.display()))?;
    // Use a real seek+read so we don't slurp the whole file just to
    // serve a small range. `Storage::open_read` returns a stateful
    // handle; the `seek` + `read_exact` shape mirrors what
    // `LocalBlobStore::get_range` does on the inner file.
    let total_len = file.seek(SeekFrom::End(0))?;
    if range.end > total_len {
      bail!(
        "StorageAsBlobStore::get_range: range {}..{} exceeds object length {} for {}",
        range.start,
        range.end,
        total_len,
        key.display()
      );
    }
    if range.start == range.end {
      return Ok(Bytes::new());
    }
    file.seek(SeekFrom::Start(range.start))?;
    let want = (range.end - range.start) as usize;
    let mut buf = vec![0u8; want];
    file.read_exact(&mut buf)?;
    Ok(Bytes::from(buf))
  }

  async fn get(&self, key: &Path) -> Result<Bytes> {
    let bytes = self
      .storage
      .read_to_end(key)
      .with_context(|| format!("StorageAsBlobStore::get({})", key.display()))?;
    Ok(Bytes::from(bytes))
  }

  async fn put(&self, key: &Path, body: Bytes) -> Result<ObjectStat> {
    self
      .storage
      .atomic_write(key, &body)
      .with_context(|| format!("StorageAsBlobStore::put({})", key.display()))?;
    self.stat(key).await
  }

  async fn put_stream(&self, key: &Path) -> Result<Box<dyn ObjectWriter>> {
    Ok(Box::new(StorageObjectWriter {
      storage: self.storage.clone(),
      key: key.to_path_buf(),
      buffer: Mutex::new(Vec::new()),
      finalized: false,
    }))
  }

  async fn put_if_match(
    &self,
    _key: &Path,
    _body: Bytes,
    _expected: Option<&str>,
  ) -> std::result::Result<ObjectStat, PutIfMatchError> {
    // `Storage` has no atomic conditional primitive. Capabilities
    // advertise `conditional_put: false`; callers MUST check before
    // calling per Stage 5's trait contract. Returning `Other` instead
    // of `Conflict` so it's clearly a "not supported" surface, not a
    // "your version was stale" one.
    Err(PutIfMatchError::Other(anyhow!(
      "StorageAsBlobStore does not support conditional PUT; check Capabilities first"
    )))
  }

  async fn delete(&self, key: &Path) -> Result<()> {
    self
      .storage
      .remove(key)
      .with_context(|| format!("StorageAsBlobStore::delete({})", key.display()))
  }

  fn capabilities(&self) -> Capabilities {
    Capabilities {
      conditional_put: false,
      multipart_upload: false,
      // Local FS via `Storage`. For an `InMemoryStorage` this would be
      // false, but no callers currently inspect this for
      // `StorageAsBlobStore`-wrapped flows; the field is informational.
      mmap_friendly: true,
    }
  }
}

struct StorageObject {
  storage: Arc<dyn Storage>,
  key: PathBuf,
  stat: ObjectStat,
}

#[async_trait]
impl Object for StorageObject {
  fn stat(&self) -> &ObjectStat {
    &self.stat
  }

  async fn read_range(&self, range: Range<u64>) -> Result<Bytes> {
    if range.start > range.end {
      bail!(
        "StorageObject::read_range: inverted range {}..{}",
        range.start,
        range.end
      );
    }
    if range.end > self.stat.len {
      bail!(
        "StorageObject::read_range: range {}..{} exceeds object length {}",
        range.start,
        range.end,
        self.stat.len
      );
    }
    if range.start == range.end {
      return Ok(Bytes::new());
    }
    // `Storage::open_read` returns a fresh stateful handle; `seek` +
    // `read_exact` issues the bounded read against the underlying FS
    // without slurping the whole file. The cached `stat.len` from
    // open time avoids re-stat'ing per read.
    let mut file = self.storage.open_read(&self.key)?;
    file.seek(SeekFrom::Start(range.start))?;
    let want = (range.end - range.start) as usize;
    let mut buf = vec![0u8; want];
    file.read_exact(&mut buf)?;
    Ok(Bytes::from(buf))
  }
}

/// `ObjectWriter` over `Storage`. Buffers writes in memory and flushes
/// via `Storage::atomic_write` on `complete`. `Storage` doesn't have a
/// streaming primitive so this is the simplest fit; it suits
/// small-payload writes (manifest, segment meta) but not multi-GB
/// segment data.
struct StorageObjectWriter {
  storage: Arc<dyn Storage>,
  key: PathBuf,
  buffer: Mutex<Vec<u8>>,
  finalized: bool,
}

#[async_trait]
impl ObjectWriter for StorageObjectWriter {
  async fn write(&mut self, chunk: Bytes) -> Result<()> {
    let mut buf = self
      .buffer
      .lock()
      .map_err(|e| anyhow!("StorageObjectWriter::write: poisoned: {e}"))?;
    buf.extend_from_slice(&chunk);
    Ok(())
  }

  async fn complete(mut self: Box<Self>) -> Result<ObjectStat> {
    let buf = std::mem::take(
      &mut *self
        .buffer
        .lock()
        .map_err(|e| anyhow!("StorageObjectWriter::complete: poisoned: {e}"))?,
    );
    self
      .storage
      .atomic_write(&self.key, &buf)
      .with_context(|| {
        format!(
          "StorageObjectWriter::complete atomic_write({})",
          self.key.display()
        )
      })?;
    self.finalized = true;
    Ok(ObjectStat {
      len: buf.len() as u64,
      provider_version: None,
      provider_checksum: None,
    })
  }

  async fn abort(mut self: Box<Self>) -> Result<()> {
    // Buffer never reached `Storage`; just drop it.
    self.finalized = true;
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::storage::FsStorage;
  use futures::executor::block_on;
  use tempfile::tempdir;

  fn fs_blob(dir: &Path) -> StorageAsBlobStore {
    StorageAsBlobStore::new(Arc::new(FsStorage::new(dir.to_path_buf())))
  }

  #[test]
  fn round_trip_put_get_via_storage() {
    let dir = tempdir().unwrap();
    let blob = fs_blob(dir.path());
    let key = dir.path().join("a/b.bin");
    block_on(blob.put(&key, Bytes::from_static(b"hello world"))).unwrap();
    let got = block_on(blob.get(&key)).unwrap();
    assert_eq!(got, Bytes::from_static(b"hello world"));

    // Raw bytes — no 37-byte header.
    let raw = std::fs::read(&key).unwrap();
    assert_eq!(raw, b"hello world");
  }

  #[test]
  fn get_range_returns_exact_bytes() {
    let dir = tempdir().unwrap();
    let blob = fs_blob(dir.path());
    let key = dir.path().join("payload");
    block_on(blob.put(&key, Bytes::from_static(b"0123456789"))).unwrap();

    let r = block_on(blob.get_range(&key, 2..5)).unwrap();
    assert_eq!(r, Bytes::from_static(b"234"));
  }

  #[test]
  fn open_object_serves_range_reads_via_seek() {
    let dir = tempdir().unwrap();
    let blob = fs_blob(dir.path());
    let key = dir.path().join("payload");
    block_on(blob.put(&key, Bytes::from_static(b"abcdefghij"))).unwrap();

    let obj = block_on(blob.open(&key)).unwrap();
    assert_eq!(obj.len(), 10);
    assert!(
      obj.stat().provider_version.is_none(),
      "Storage has no per-write version; provider_version must be None"
    );

    let r = block_on(obj.read_range(3..7)).unwrap();
    assert_eq!(r, Bytes::from_static(b"defg"));
  }

  #[test]
  #[allow(clippy::reversed_empty_ranges)] // intentional: testing rejection
  fn range_contract_inverted_and_oob_rejected() {
    let dir = tempdir().unwrap();
    let blob = fs_blob(dir.path());
    let key = dir.path().join("payload");
    block_on(blob.put(&key, Bytes::from_static(b"abcdef"))).unwrap();

    assert!(block_on(blob.get_range(&key, 5..2)).is_err());
    assert!(block_on(blob.get_range(&key, 0..100)).is_err());
    let obj = block_on(blob.open(&key)).unwrap();
    assert!(block_on(obj.read_range(5..2)).is_err());
    assert!(block_on(obj.read_range(0..100)).is_err());
  }

  #[test]
  fn capabilities_match_storage_constraints() {
    let dir = tempdir().unwrap();
    let blob = fs_blob(dir.path());
    let cap = blob.capabilities();
    assert!(!cap.conditional_put, "Storage trait has no CAS primitive");
    assert!(!cap.multipart_upload);
  }

  #[test]
  fn put_if_match_returns_other_unsupported_error() {
    let dir = tempdir().unwrap();
    let blob = fs_blob(dir.path());
    let key = dir.path().join("k");
    let err = block_on(blob.put_if_match(&key, Bytes::from_static(b"x"), None))
      .expect_err("StorageAsBlobStore must not support conditional PUT");
    match err {
      PutIfMatchError::Other(e) => {
        let msg = format!("{e:#}");
        assert!(msg.contains("conditional"), "got: {msg}");
      }
      PutIfMatchError::Conflict { .. } => panic!("expected Other, got Conflict"),
    }
  }

  #[test]
  fn put_stream_buffers_and_flushes_on_complete() {
    let dir = tempdir().unwrap();
    let blob = fs_blob(dir.path());
    let key = dir.path().join("streamed");

    let mut writer = block_on(blob.put_stream(&key)).unwrap();
    block_on(writer.write(Bytes::from_static(b"part1-"))).unwrap();
    block_on(writer.write(Bytes::from_static(b"part2"))).unwrap();
    let stat = block_on(writer.complete()).unwrap();
    assert_eq!(stat.len, b"part1-part2".len() as u64);

    let got = block_on(blob.get(&key)).unwrap();
    assert_eq!(got, Bytes::from_static(b"part1-part2"));
  }

  #[test]
  fn put_stream_abort_does_not_create_target() {
    let dir = tempdir().unwrap();
    let blob = fs_blob(dir.path());
    let key = dir.path().join("aborted");

    let mut writer = block_on(blob.put_stream(&key)).unwrap();
    block_on(writer.write(Bytes::from_static(b"discarded"))).unwrap();
    block_on(writer.abort()).unwrap();
    assert!(block_on(blob.stat(&key)).is_err());
  }
}
