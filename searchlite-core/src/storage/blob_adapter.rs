//! [`Storage`]-over-[`BlobStore`] adapter — Stage 6's expressiveness gate.
//!
//! Wraps any `Arc<dyn BlobStore>` and exposes the existing `Storage` trait
//! surface so the unmodified index code can run against a `BlobStore`
//! impl. This is the test fixture that proves Stage 5's trait shape is
//! sufficient to back everything `Storage` exposes.
//!
//! ## Sync-over-async bridging
//!
//! The adapter uses [`crate::runtime::block_on_blob`] (Stage 10a) to
//! drive `BlobStore` futures from the sync `Storage` methods. The
//! bridge selects an executor based on the calling context:
//!
//! * Default build: `futures::executor::block_on`. Lightweight; no
//!   Tokio dep. Fine for [`LocalBlobStore`](super::local_blob),
//!   [`StorageAsBlobStore`](super::storage_as_blob), and any other
//!   pure-sync-or-pollable BlobStore impl.
//! * `tokio-runtime` feature: a global lazy multi-thread Tokio runtime
//!   (with `block_in_place` fallback when called from inside an
//!   active multi-thread runtime). Required for Stage 10b's
//!   `S3BlobStore` futures, whose `hyper`/`tokio-rustls` internals
//!   need a real reactor.
//!
//! ## Filesystem-shaped methods
//!
//! `Storage` includes a few methods that have no clean blob-store
//! analogue: `ensure_dir`, `remove_dir_all`, and `open_append`'s atomic
//! O_APPEND semantics. The adapter handles these against the local
//! filesystem when `key` resolves to a local path, which works for
//! `LocalBlobStore` (the only Stage 6 impl) and any future
//! local-FS-backed BlobStore. The Stage 10b S3 backend will NOT go
//! through this adapter; it consumes `BlobStore` directly via the
//! segment reader migration completed in Stage 8.
//!
//! ## Buffered file handles
//!
//! `Storage::open_read` / `open_write` / `open_append` return a
//! `Box<dyn StorageFile>` (a stateful Read+Write+Seek handle). Blob
//! stores don't have stateful handles — they have whole-object reads and
//! writes. The adapter buffers in memory: reads slurp the whole object
//! into a `Cursor<Vec<u8>>`, writes accumulate until `sync_all` flushes
//! the buffer via `BlobStore::put`. This is correct for the existing
//! index code's usage (sequential write-then-close, sequential
//! read-with-occasional-seeks), but it does mean a 1 GB segment write
//! buffers 1 GB in memory before the put. Stage 8's hot-path migration
//! eliminated this for postings/docstore — segment readers consume
//! `BlobStore` directly via bounded `read_range` calls instead of
//! going through this adapter.

use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;

use super::blob::BlobStore;
use super::{DynFile, Storage, StorageFile};
use crate::runtime::block_on_blob as block_on;

/// Adapts an `Arc<dyn BlobStore>` to the existing `Storage` trait. See
/// the module docs for the impedance mismatches and how they're handled.
pub struct BlobStoreAdapter {
  blob: Arc<dyn BlobStore>,
  root: PathBuf,
}

impl BlobStoreAdapter {
  pub fn new(blob: Arc<dyn BlobStore>, root: PathBuf) -> Self {
    Self { blob, root }
  }

  fn resolve(&self, path: &Path) -> PathBuf {
    if path.is_absolute() {
      path.to_path_buf()
    } else {
      self.root.join(path)
    }
  }
}

impl Storage for BlobStoreAdapter {
  fn root(&self) -> &Path {
    &self.root
  }

  /// Stage 8a [P1] (Codex review): expose the inner blob store so
  /// `Index::*_with_storage` constructors can avoid double-wrapping
  /// (`StorageAsBlobStore` → `BlobStoreAdapter` → another `block_on`).
  /// See `default_blob_store` in `index/mod.rs`.
  fn as_blob_store(&self) -> Option<Arc<dyn BlobStore>> {
    Some(self.blob.clone())
  }

  fn ensure_dir(&self, path: &Path) -> Result<()> {
    // Filesystem-shaped: BlobStore has no notion of directories.
    // For local-FS-backed impls (Stage 6 LocalBlobStore), parent dirs
    // are automatically created on `put`, but the index code calls
    // `ensure_dir` for the index root before any put — so we materialize
    // it here against the local FS.
    std::fs::create_dir_all(self.resolve(path))?;
    Ok(())
  }

  fn exists(&self, path: &Path) -> bool {
    let resolved = self.resolve(path);
    block_on(self.blob.stat(&resolved)).is_ok()
  }

  fn open_read(&self, path: &Path) -> Result<DynFile> {
    let resolved = self.resolve(path);
    let bytes = block_on(self.blob.get(&resolved))
      .with_context(|| format!("BlobStoreAdapter::open_read({})", resolved.display()))?;
    Ok(Box::new(BufferedBlobFile::for_read(bytes)))
  }

  fn open_write(&self, path: &Path) -> Result<DynFile> {
    Ok(Box::new(BufferedBlobFile::for_write(
      self.blob.clone(),
      self.resolve(path),
    )))
  }

  fn open_append(&self, path: &Path) -> Result<DynFile> {
    // Best-effort: read existing bytes if the object exists, start empty
    // if it doesn't, and propagate any other error. Subsequent writes
    // append in-buffer and flush via `put` on `sync_all`.
    //
    // This is single-process append semantics — POSIX O_APPEND
    // atomic-append-across-processes is not preserved through the
    // adapter. Acceptable for tests because all append uses in
    // `searchlite-core` are single-writer (Wal owns its file
    // exclusively).
    //
    // Codex Stage 6 P2: `Err(_) => empty buffer` (the previous shape)
    // silently overwrote existing bytes on `sync_all` whenever the
    // initial `get` failed for any reason — including transient I/O,
    // permission, or backend errors. Distinguishing NotFound from other
    // failures is the load-bearing fix; only NotFound starts empty.
    let resolved = self.resolve(path);
    let initial = match block_on(self.blob.get(&resolved)) {
      Ok(b) => b.to_vec(),
      Err(e) if error_is_not_found(&e) => Vec::new(),
      Err(e) => {
        return Err(e).with_context(|| {
          format!(
            "BlobStoreAdapter::open_append({}): unable to read existing bytes",
            resolved.display()
          )
        })
      }
    };
    Ok(Box::new(BufferedBlobFile::for_append(
      self.blob.clone(),
      resolved,
      initial,
    )))
  }

  fn read_to_end(&self, path: &Path) -> Result<Vec<u8>> {
    let resolved = self.resolve(path);
    let bytes = block_on(self.blob.get(&resolved))
      .with_context(|| format!("BlobStoreAdapter::read_to_end({})", resolved.display()))?;
    Ok(bytes.to_vec())
  }

  fn write_all(&self, path: &Path, data: &[u8]) -> Result<()> {
    let resolved = self.resolve(path);
    block_on(self.blob.put(&resolved, Bytes::copy_from_slice(data)))
      .with_context(|| format!("BlobStoreAdapter::write_all({})", resolved.display()))?;
    Ok(())
  }

  fn atomic_write(&self, path: &Path, data: &[u8]) -> Result<()> {
    // `BlobStore::put` is atomic-by-contract: implementations either
    // tmp+rename (LocalBlobStore) or use the provider's atomic put
    // (object stores). Either way, partial writes are not observable.
    let resolved = self.resolve(path);
    block_on(self.blob.put(&resolved, Bytes::copy_from_slice(data)))
      .with_context(|| format!("BlobStoreAdapter::atomic_write({})", resolved.display()))?;
    Ok(())
  }

  fn remove(&self, path: &Path) -> Result<()> {
    let resolved = self.resolve(path);
    block_on(self.blob.delete(&resolved))
      .with_context(|| format!("BlobStoreAdapter::remove({})", resolved.display()))
  }

  fn remove_dir_all(&self, path: &Path) -> Result<()> {
    // Filesystem-shaped, like `ensure_dir`. The index code uses this for
    // wholesale cleanup of segment file directories during compaction.
    // BlobStore doesn't expose recursive-delete; for local-FS-backed
    // impls we satisfy the contract via `std::fs::remove_dir_all`.
    let resolved = self.resolve(path);
    if resolved.exists() {
      std::fs::remove_dir_all(&resolved)
        .with_context(|| format!("BlobStoreAdapter::remove_dir_all({})", resolved.display()))?;
    }
    Ok(())
  }
}

/// Walk an `anyhow::Error` chain looking for a `std::io::Error` whose
/// `kind()` is `NotFound`. Used by `open_append` to distinguish "object
/// doesn't exist yet, start empty" from "I/O failed, propagate error" —
/// the previous shape conflated both into an empty buffer and silently
/// overwrote existing WAL bytes on sync (Codex Stage 6 P2).
fn error_is_not_found(err: &anyhow::Error) -> bool {
  err.chain().any(|cause| {
    cause
      .downcast_ref::<std::io::Error>()
      .map(|e| e.kind() == std::io::ErrorKind::NotFound)
      .unwrap_or(false)
  })
}

/// In-memory buffered file handle that adapts blob-store whole-object
/// semantics to the stateful `StorageFile` trait. See the module docs for
/// the buffering trade-off.
struct BufferedBlobFile {
  cursor: Cursor<Vec<u8>>,
  /// On `sync_all`, if `backend` is `Some`, the buffer is flushed via
  /// `BlobStore::put`. `None` for read-only handles.
  backend: Option<(Arc<dyn BlobStore>, PathBuf)>,
  /// Tracks whether the buffer has been written since the last flush.
  /// `open_write` starts dirty (caller will write zero or more bytes; an
  /// empty file should still be persisted on close). `open_append`
  /// starts clean (initial bytes are already on disk).
  dirty: bool,
}

impl BufferedBlobFile {
  fn for_read(bytes: Bytes) -> Self {
    Self {
      cursor: Cursor::new(bytes.to_vec()),
      backend: None,
      dirty: false,
    }
  }

  fn for_write(blob: Arc<dyn BlobStore>, key: PathBuf) -> Self {
    Self {
      cursor: Cursor::new(Vec::new()),
      backend: Some((blob, key)),
      dirty: true,
    }
  }

  fn for_append(blob: Arc<dyn BlobStore>, key: PathBuf, initial: Vec<u8>) -> Self {
    let pos = initial.len() as u64;
    let mut cursor = Cursor::new(initial);
    cursor.set_position(pos);
    Self {
      cursor,
      backend: Some((blob, key)),
      dirty: false,
    }
  }
}

impl Read for BufferedBlobFile {
  fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
    self.cursor.read(buf)
  }
}

impl Write for BufferedBlobFile {
  fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
    let n = self.cursor.write(buf)?;
    if n > 0 {
      self.dirty = true;
    }
    Ok(n)
  }

  fn flush(&mut self) -> io::Result<()> {
    // `flush` is a buffer-level concern; durability is tied to
    // `sync_all` per `StorageFile`'s contract.
    Ok(())
  }
}

impl Seek for BufferedBlobFile {
  fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
    self.cursor.seek(pos)
  }
}

impl StorageFile for BufferedBlobFile {
  fn set_len(&mut self, len: u64) -> Result<()> {
    let inner = self.cursor.get_mut();
    let len_usize = usize::try_from(len)
      .map_err(|_| anyhow!("BufferedBlobFile::set_len: length {len} overflows usize"))?;
    inner.resize(len_usize, 0);
    if self.cursor.position() > len {
      self.cursor.set_position(len);
    }
    self.dirty = true;
    Ok(())
  }

  fn sync_all(&mut self) -> Result<()> {
    if let Some((backend, key)) = &self.backend {
      if self.dirty {
        let bytes = Bytes::from(self.cursor.get_ref().clone());
        block_on(backend.put(key, bytes))
          .with_context(|| format!("BufferedBlobFile::sync_all put({})", key.display()))?;
        self.dirty = false;
      }
    }
    Ok(())
  }
}

impl Drop for BufferedBlobFile {
  fn drop(&mut self) {
    if self.dirty && self.backend.is_some() {
      // Best-effort flush on drop. The existing FsStorage impl writes
      // to a real `File` whose handle is closed on drop without any
      // user-level sync; some callers rely on writes-being-durable
      // only after an explicit `sync_all`. We mirror the existing
      // semantics: flush so the data is visible to subsequent reads,
      // but we don't return errors from Drop.
      let _ = self.sync_all();
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::storage::LocalBlobStore;
  use tempfile::tempdir;

  fn adapter(dir: &Path) -> BlobStoreAdapter {
    let blob: Arc<dyn BlobStore> = Arc::new(LocalBlobStore::new(dir.to_path_buf()));
    BlobStoreAdapter::new(blob, dir.to_path_buf())
  }

  #[test]
  fn atomic_write_and_read_to_end_round_trip() {
    let dir = tempdir().unwrap();
    let adapter = adapter(dir.path());
    let key = Path::new("a/b.bin");
    adapter.atomic_write(key, b"payload").unwrap();
    let got = adapter.read_to_end(key).unwrap();
    assert_eq!(got, b"payload");
  }

  #[test]
  fn open_write_buffers_and_persists_on_sync_all() {
    let dir = tempdir().unwrap();
    let adapter = adapter(dir.path());
    let key = Path::new("buffered.bin");
    let mut f = adapter.open_write(key).unwrap();
    f.write_all(b"first ").unwrap();
    f.write_all(b"second").unwrap();
    f.sync_all().unwrap();
    let got = adapter.read_to_end(key).unwrap();
    assert_eq!(got, b"first second");
  }

  #[test]
  fn open_append_continues_from_existing_bytes() {
    let dir = tempdir().unwrap();
    let adapter = adapter(dir.path());
    let key = Path::new("appended.bin");
    adapter.atomic_write(key, b"head:").unwrap();

    let mut f = adapter.open_append(key).unwrap();
    f.write_all(b"tail").unwrap();
    f.sync_all().unwrap();

    let got = adapter.read_to_end(key).unwrap();
    assert_eq!(got, b"head:tail");
  }

  #[test]
  fn open_read_yields_seekable_buffered_view() {
    let dir = tempdir().unwrap();
    let adapter = adapter(dir.path());
    let key = Path::new("seekable.bin");
    adapter.atomic_write(key, b"0123456789").unwrap();

    let mut f = adapter.open_read(key).unwrap();
    f.seek(SeekFrom::Start(3)).unwrap();
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf).unwrap();
    assert_eq!(&buf, b"3456");
  }

  #[test]
  fn exists_reflects_blob_state() {
    let dir = tempdir().unwrap();
    let adapter = adapter(dir.path());
    let key = Path::new("present.bin");
    assert!(!adapter.exists(key));
    adapter.atomic_write(key, b"x").unwrap();
    assert!(adapter.exists(key));
    adapter.remove(key).unwrap();
    assert!(!adapter.exists(key));
  }

  #[test]
  fn remove_dir_all_against_local_fs_layout() {
    let dir = tempdir().unwrap();
    let adapter = adapter(dir.path());
    let nested = Path::new("a/b/c.bin");
    adapter.atomic_write(nested, b"x").unwrap();
    let parent = dir.path().join("a");
    assert!(parent.exists());
    adapter.remove_dir_all(&parent).unwrap();
    assert!(!parent.exists());
  }

  #[test]
  fn remove_is_idempotent() {
    let dir = tempdir().unwrap();
    let adapter = adapter(dir.path());
    adapter.remove(Path::new("never-existed")).unwrap();
  }

  #[test]
  fn ensure_dir_creates_directory() {
    let dir = tempdir().unwrap();
    let adapter = adapter(dir.path());
    let nested = dir.path().join("a/b/c");
    adapter.ensure_dir(&nested).unwrap();
    assert!(nested.is_dir());
  }

  /// Stage 6 P2 fix: relative paths must resolve through the adapter's
  /// own `root`, not the underlying blob store's. With the previous shape
  /// the adapter only resolved for `ensure_dir`/`remove_dir_all`; other
  /// methods passed the relative path through to `LocalBlobStore::resolve`,
  /// which would join against the blob store's root instead.
  #[test]
  fn relative_paths_resolve_through_adapter_root_not_blob_root() {
    // Two distinct roots so the bug surfaces: the adapter advertises
    // `adapter_root` via `Storage::root()`, but the blob store is
    // configured with `blob_root`. A correctly-resolving adapter must
    // route writes/reads to `adapter_root/<relative>`, never to
    // `blob_root/<relative>`.
    let outer = tempdir().unwrap();
    let adapter_root = outer.path().join("a-side");
    let blob_root = outer.path().join("b-side");
    std::fs::create_dir_all(&adapter_root).unwrap();
    std::fs::create_dir_all(&blob_root).unwrap();

    let blob: Arc<dyn BlobStore> = Arc::new(LocalBlobStore::new(blob_root.clone()));
    let adapter = BlobStoreAdapter::new(blob, adapter_root.clone());

    let key = Path::new("nested/payload.bin");
    adapter.atomic_write(key, b"adapter-routed").unwrap();

    // Writes land under the adapter's root, NOT the blob store's root.
    assert!(
      adapter_root.join("nested/payload.bin").is_file(),
      "atomic_write must resolve relative paths through the adapter root"
    );
    assert!(
      !blob_root.join("nested/payload.bin").exists(),
      "atomic_write must NOT route through the blob store's own root"
    );

    // Read-side methods agree with the write-side resolution.
    let got = adapter.read_to_end(key).unwrap();
    assert_eq!(got, b"adapter-routed");
    assert!(adapter.exists(key));

    adapter.remove(key).unwrap();
    assert!(!adapter.exists(key));
    assert!(!adapter_root.join("nested/payload.bin").exists());
  }

  /// Stage 6 P2 fix: `open_append` must distinguish "object doesn't
  /// exist yet" (start empty) from "I/O failed for some other reason"
  /// (propagate). Previously every error became an empty buffer, so a
  /// permission denial silently overwrote existing bytes on the next
  /// `sync_all`. Test by pointing at a path inside a non-readable
  /// directory and asserting the open errors instead of producing a
  /// would-truncate handle.
  ///
  /// Note: chmod 000 is a Unix-ism. Skipped on Windows.
  #[test]
  #[cfg(unix)]
  fn open_append_propagates_non_not_found_errors() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempdir().unwrap();
    let adapter = adapter(dir.path());
    let secret_dir = dir.path().join("locked");
    std::fs::create_dir(&secret_dir).unwrap();
    let secret_file = secret_dir.join("payload");
    std::fs::write(&secret_file, b"sensitive").unwrap();
    // Drop directory permissions so file metadata + open both fail with
    // PermissionDenied — distinct from NotFound.
    std::fs::set_permissions(&secret_dir, std::fs::Permissions::from_mode(0o000)).unwrap();

    let result = adapter.open_append(&secret_file);

    // Restore perms before any panic so tempdir cleanup can succeed.
    std::fs::set_permissions(&secret_dir, std::fs::Permissions::from_mode(0o755)).unwrap();

    let err = match result {
      Ok(_) => panic!("permission-denied open_append must propagate, not silently empty"),
      Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
      !msg.contains("not found"),
      "permission errors must not be conflated with NotFound: {msg}"
    );

    // The original bytes are intact: open_append did NOT publish an
    // empty buffer to overwrite them.
    let after = std::fs::read(&secret_file).unwrap();
    assert_eq!(after, b"sensitive");
  }

  /// Companion test: `open_append` against a NotFound path must succeed
  /// with an empty starting buffer. This is the legitimate "first
  /// append" case (e.g. fresh WAL) that the P2 fix must not regress.
  #[test]
  fn open_append_starts_empty_when_object_missing() {
    let dir = tempdir().unwrap();
    let adapter = adapter(dir.path());
    let key = Path::new("fresh-wal");
    assert!(!adapter.exists(key));

    let mut f = adapter.open_append(key).unwrap();
    f.write_all(b"first-record").unwrap();
    f.sync_all().unwrap();

    let got = adapter.read_to_end(key).unwrap();
    assert_eq!(got, b"first-record");
  }
}
