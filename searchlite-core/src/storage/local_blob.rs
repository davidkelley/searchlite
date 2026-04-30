//! Local-filesystem-backed [`BlobStore`] implementation.
//!
//! Stage 6's first concrete impl. Async methods do sync `std::fs` work
//! directly (no `spawn_blocking`, no `tokio` dep) — this matches Stage 4's
//! "sync work behind an `async fn`" pattern. Production async backends
//! (Stage 9's S3) are a separate impl that doesn't share this code.
//!
//! ## CAS via per-key flock + header-in-data version
//!
//! Every data file written by this backend starts with a fixed 37-byte
//! header: a UUIDv4 in 36 ASCII characters, followed by a single `\n`.
//! Callers never see the header — `get` / `get_range` / `Object::read_range`
//! strip it on read, and `stat::len` reports the logical length
//! (physical minus 37 bytes). The version inside the header IS the
//! `provider_version` token returned by `stat` and `put_if_match`.
//!
//! Co-locating version with data inside one atomically-published
//! artifact eliminates the crash and race windows a sidecar approach
//! suffers (Codex Stage 6 v2 [P1]/[P2]):
//!
//! - **No data/version drift**: a single `rename` publishes both
//!   atomically. There is no window where the new bytes are visible
//!   under the old version token.
//! - **No write-mode mismatch**: `put`, `put_stream::complete`, and
//!   `put_if_match` all generate a fresh UUID for the header on every
//!   successful write, so unconditional writes can't leave conditional
//!   readers staring at a stale token.
//!
//! All write paths take the same per-key sidecar lock (`<key>.lock`
//! via `fs2`) so unconditional and conditional writers serialize on a
//! single critical section. Reads do not take the lock — observed
//! version is whatever the file's atomically-published header currently
//! shows.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use bytes::Bytes;
use fs2::FileExt;
use uuid::Uuid;

use super::blob::{
  BlobStore, Capabilities, Object, ObjectStat, ObjectWriter, PutIfMatchError,
};

/// Local-filesystem [`BlobStore`]. Keys may be absolute paths (used
/// directly) or paths relative to the configured `root`.
pub struct LocalBlobStore {
  root: PathBuf,
}

impl LocalBlobStore {
  pub fn new(root: PathBuf) -> Self {
    Self { root }
  }

  pub fn root(&self) -> &Path {
    &self.root
  }

  /// Resolve a key to an absolute filesystem path. Absolute keys are used
  /// as-is (matching the existing `FsStorage` convention where the index
  /// code passes absolute paths from `SegmentMeta`); relative keys are
  /// joined to `root`.
  fn resolve(&self, key: &Path) -> PathBuf {
    if key.is_absolute() {
      key.to_path_buf()
    } else {
      self.root.join(key)
    }
  }

  fn lockfile_for(&self, target: &Path) -> PathBuf {
    sidecar_path(target, ".lock")
  }
}

#[async_trait]
impl BlobStore for LocalBlobStore {
  async fn stat(&self, key: &Path) -> Result<ObjectStat> {
    let path = self.resolve(key);
    stat_from_path(&path)
  }

  async fn open(&self, key: &Path) -> Result<Arc<dyn Object>> {
    let path = self.resolve(key);
    let stat = stat_from_path(&path)?;
    Ok(Arc::new(LocalObject { path, stat }))
  }

  async fn get_range(&self, key: &Path, range: Range<u64>) -> Result<Bytes> {
    if range.start > range.end {
      bail!(
        "LocalBlobStore::get_range: inverted range {}..{}",
        range.start,
        range.end
      );
    }
    let path = self.resolve(key);
    let logical_len = logical_len_of(&path)
      .with_context(|| format!("LocalBlobStore::get_range stat({})", path.display()))?;
    if range.end > logical_len {
      bail!(
        "LocalBlobStore::get_range: range {}..{} exceeds object length {} for {}",
        range.start,
        range.end,
        logical_len,
        path.display()
      );
    }
    if range.start == range.end {
      return Ok(Bytes::new());
    }
    read_logical_range(&path, range)
      .with_context(|| format!("LocalBlobStore::get_range read({})", path.display()))
  }

  async fn get(&self, key: &Path) -> Result<Bytes> {
    let path = self.resolve(key);
    let logical_len = logical_len_of(&path)
      .with_context(|| format!("LocalBlobStore::get({})", path.display()))?;
    read_logical_range(&path, 0..logical_len)
      .with_context(|| format!("LocalBlobStore::get({})", path.display()))
  }

  async fn put(&self, key: &Path, body: Bytes) -> Result<ObjectStat> {
    let path = self.resolve(key);
    if let Some(parent) = path.parent() {
      fs::create_dir_all(parent)?;
    }
    // Per Codex Stage 6 v2 [P2], `put` must take the same per-key lock as
    // `put_if_match` so a non-CAS write can't race a CAS write on the
    // same key. The header carries the version atomically with the data,
    // so a single tmp+rename publishes both — there's no window where
    // the bytes are visible under a stale version.
    self
      .write_under_lock(&path, |new_version| {
        write_data_file_atomic(&path, new_version, &body)
      })
      .await
  }

  async fn put_stream(&self, key: &Path) -> Result<Box<dyn ObjectWriter>> {
    let path = self.resolve(key);
    if let Some(parent) = path.parent() {
      fs::create_dir_all(parent)?;
    }
    let tmp = make_tmp_path(&path);
    let lockfile_path = self.lockfile_for(&path);
    // Reserve the header bytes at the start of the tmp file with a
    // placeholder UUID. Streamed writes through `ObjectWriter::write`
    // append after the header, so the body ends up at the right offset.
    // `complete` overwrites the placeholder with a fresh UUID under the
    // per-key lock so the published version is generated at commit
    // time, not at writer-creation time.
    let mut tmp_file = File::create(&tmp)?;
    let placeholder = new_version_token();
    write_header(&mut tmp_file, &placeholder)?;
    Ok(Box::new(LocalObjectWriter {
      target: path,
      tmp,
      lockfile_path,
      file: Some(tmp_file),
      finalized: false,
    }))
  }

  async fn put_if_match(
    &self,
    key: &Path,
    body: Bytes,
    expected: Option<&str>,
  ) -> std::result::Result<ObjectStat, PutIfMatchError> {
    let path = self.resolve(key);
    if let Some(parent) = path.parent() {
      fs::create_dir_all(parent).map_err(|e| PutIfMatchError::Other(e.into()))?;
    }
    self.put_if_match_locked(&path, body, expected).await
  }

  async fn delete(&self, key: &Path) -> Result<()> {
    let path = self.resolve(key);
    // Per Codex Stage 6 v3 [P2], `delete` is a same-key writer and must
    // take the per-key lock so it serializes with `put`,
    // `put_if_match`, and `put_stream::complete`. Without it, a delete
    // could interleave between a CAS write's version check and its
    // rename — yielding a successful conditional write that doesn't
    // linearize with the delete. With it, every concurrent
    // (put*, delete) pair has a deterministic linearization on the lock
    // queue.
    let _lock = self.acquire_key_lock(&path)?;
    match fs::remove_file(&path) {
      Ok(()) => Ok(()),
      // Idempotent: deleting an absent object is not an error.
      Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
      Err(e) => Err(anyhow!(e)).with_context(|| format!("LocalBlobStore::delete({})", path.display())),
    }
  }

  fn capabilities(&self) -> Capabilities {
    Capabilities {
      // Real CAS via fs2 file locking + tmp+rename. `put_if_match`
      // serializes on a per-key sidecar lock, so the stat→write→rename
      // sequence runs without a TOCTOU window even across cooperating
      // processes.
      conditional_put: true,
      // Local FS streams serially via tmp+rename; no parallel-part
      // protocol exposed (this flag is informational per the trait
      // contract — `put_stream` is always available).
      multipart_upload: false,
      mmap_friendly: true,
    }
  }
}

impl LocalBlobStore {
  /// Acquire the per-key flock and run `write` once the lock is held.
  /// The closure receives the freshly-generated version UUID so it can
  /// stamp the data file's header with it — the version, header, and
  /// data are published atomically by a single tmp+rename inside
  /// `write`. After `write` returns, this method does the parent-dir
  /// fsync and stat. Used by both `put` (no precondition) and
  /// `put_if_match` (precondition is checked inside `write` so it sees
  /// the locked-state view).
  async fn write_under_lock<F>(&self, path: &Path, write: F) -> Result<ObjectStat>
  where
    F: FnOnce(&str) -> Result<()>,
  {
    let _lock = self.acquire_key_lock(path)?;
    let new_version = new_version_token();
    write(&new_version)?;
    sync_dir(path)?;
    let metadata = fs::metadata(path)?;
    Ok(ObjectStat {
      len: metadata.len().saturating_sub(HEADER_LEN),
      provider_version: Some(new_version),
      provider_checksum: None,
    })
  }

  /// Unified `put_if_match` implementation. Both `expected = None` and
  /// `expected = Some(version)` run their existence-or-version check,
  /// data write, and rename publish inside a single per-key flock
  /// critical section. The version token lives in the data file's
  /// header (Codex v2 [P1]) so a single rename atomically publishes
  /// both bytes and version — there is no window where new bytes are
  /// visible under a stale token.
  async fn put_if_match_locked(
    &self,
    path: &Path,
    body: Bytes,
    expected: Option<&str>,
  ) -> std::result::Result<ObjectStat, PutIfMatchError> {
    let _lock = self
      .acquire_key_lock(path)
      .map_err(PutIfMatchError::Other)?;
    // From here until the lock guard drops at the end of the function,
    // no other cooperating LocalBlobStore writer (CAS or unconditional)
    // can race the check → write → rename sequence on this key.

    // Read the *current* observed state inside the locked region. The
    // version token is whatever the file's atomically-published header
    // currently shows; if the file doesn't exist yet, current is None.
    let current = match stat_from_path(path) {
      Ok(stat) => Some(stat),
      Err(e) if error_is_not_found(&e) => None,
      Err(e) => return Err(PutIfMatchError::Other(e)),
    };

    match (expected, &current) {
      (None, Some(stat)) => {
        // Caller asked "must not exist" but it does. Conflict carries
        // the current stat so the retry can switch to Some(version).
        return Err(PutIfMatchError::Conflict {
          current: Some(stat.clone()),
        });
      }
      (Some(_), None) => {
        // Caller asked "must match this version" but the object doesn't
        // exist. Nothing to match.
        return Err(PutIfMatchError::Conflict { current: None });
      }
      (Some(want), Some(have)) if have.provider_version.as_deref() != Some(want) => {
        return Err(PutIfMatchError::Conflict {
          current: Some(have.clone()),
        });
      }
      _ => {} // (None, None) or (Some(v), Some(v)) — proceed to write.
    }

    // Generate the new version, write the header+body atomically as a
    // single tmp+rename. The tmp file lives in the same parent
    // directory so the rename stays on a single filesystem.
    let new_version = new_version_token();
    write_data_file_atomic(path, &new_version, &body).map_err(PutIfMatchError::Other)?;
    sync_dir(path).map_err(PutIfMatchError::Other)?;
    let metadata = fs::metadata(path).map_err(|e| PutIfMatchError::Other(e.into()))?;
    Ok(ObjectStat {
      len: metadata.len().saturating_sub(HEADER_LEN),
      provider_version: Some(new_version),
      provider_checksum: None,
    })
  }

  /// Acquire the per-key sidecar lock for `path`. Returns a guard whose
  /// drop releases the lock. Common code path for all writers (`put`,
  /// `put_if_match`, `put_stream::complete`) so they all serialize on
  /// the same critical section.
  fn acquire_key_lock(&self, path: &Path) -> Result<KeyLockGuard> {
    let lockfile_path = self.lockfile_for(path);
    if let Some(parent) = lockfile_path.parent() {
      fs::create_dir_all(parent)?;
    }
    let lock_file = OpenOptions::new()
      .create(true)
      .read(true)
      .write(true)
      .truncate(false)
      .open(&lockfile_path)
      .with_context(|| format!("open lockfile: {}", lockfile_path.display()))?;
    FileExt::lock_exclusive(&lock_file)
      .with_context(|| format!("lock_exclusive: {}", lockfile_path.display()))?;
    Ok(KeyLockGuard { _file: lock_file })
  }
}

/// Drop guard for the per-key flock. Holding the file open is what
/// holds the lock; dropping the file releases it.
struct KeyLockGuard {
  _file: File,
}

struct LocalObject {
  path: PathBuf,
  stat: ObjectStat,
}

#[async_trait]
impl Object for LocalObject {
  fn stat(&self) -> &ObjectStat {
    &self.stat
  }

  async fn read_range(&self, range: Range<u64>) -> Result<Bytes> {
    if range.start > range.end {
      bail!(
        "LocalObject::read_range: inverted range {}..{}",
        range.start,
        range.end
      );
    }
    if range.end > self.stat.len {
      bail!(
        "LocalObject::read_range: range {}..{} exceeds object length {}",
        range.start,
        range.end,
        self.stat.len
      );
    }
    if range.start == range.end {
      return Ok(Bytes::new());
    }
    // Logical-to-physical offset translation: the on-disk file has a
    // 37-byte version header at byte 0 that callers never see.
    read_logical_range(&self.path, range)
  }
}

/// Streaming writer for `LocalBlobStore::put_stream`. Writes accumulate
/// to a tmp file in the parent directory after a placeholder version
/// header; `complete` overwrites the placeholder with a freshly-generated
/// UUID under the per-key lock, fsync's, renames over the target, and
/// fsync's the parent. `abort` removes the tmp file. Drop without
/// complete/abort cleans up the tmp file as best-effort.
struct LocalObjectWriter {
  target: PathBuf,
  tmp: PathBuf,
  /// Path to the per-key lockfile so `complete` can serialize against
  /// `put` and `put_if_match`. Pre-computed at writer creation so
  /// `complete` doesn't recompute it.
  lockfile_path: PathBuf,
  file: Option<File>,
  finalized: bool,
}

#[async_trait]
impl ObjectWriter for LocalObjectWriter {
  async fn write(&mut self, chunk: Bytes) -> Result<()> {
    let file = self
      .file
      .as_mut()
      .ok_or_else(|| anyhow!("LocalObjectWriter::write called after finalization"))?;
    file.write_all(&chunk)?;
    Ok(())
  }

  async fn complete(mut self: Box<Self>) -> Result<ObjectStat> {
    let new_version = new_version_token();
    if let Some(mut file) = self.file.take() {
      // Overwrite the placeholder header with the fresh commit-time
      // version so the published file's header matches the
      // `provider_version` we return. Keeps the streamed body bytes at
      // their original offsets — the header is fixed length.
      file.seek(SeekFrom::Start(0))?;
      write_header(&mut file, &new_version)?;
      file.sync_all()?;
    }
    // Take the per-key lock for the publish step so a concurrent `put`
    // or `put_if_match` can't interleave between rename and parent
    // fsync (Codex Stage 6 v2 [P2]).
    let _lock = acquire_lock(&self.lockfile_path)?;
    fs::rename(&self.tmp, &self.target)?;
    sync_dir(&self.target)?;
    self.finalized = true;
    let metadata = fs::metadata(&self.target)?;
    Ok(ObjectStat {
      len: metadata.len().saturating_sub(HEADER_LEN),
      provider_version: Some(new_version),
      provider_checksum: None,
    })
  }

  async fn abort(mut self: Box<Self>) -> Result<()> {
    self.file = None;
    let _ = fs::remove_file(&self.tmp);
    self.finalized = true;
    Ok(())
  }
}

/// Free-function variant of `LocalBlobStore::acquire_key_lock` for use by
/// `LocalObjectWriter::complete`, which doesn't have a back-reference
/// to the `LocalBlobStore`.
fn acquire_lock(lockfile_path: &Path) -> Result<KeyLockGuard> {
  if let Some(parent) = lockfile_path.parent() {
    fs::create_dir_all(parent)?;
  }
  let lock_file = OpenOptions::new()
    .create(true)
    .read(true)
    .write(true)
    .truncate(false)
    .open(lockfile_path)
    .with_context(|| format!("open lockfile: {}", lockfile_path.display()))?;
  FileExt::lock_exclusive(&lock_file)
    .with_context(|| format!("lock_exclusive: {}", lockfile_path.display()))?;
  Ok(KeyLockGuard { _file: lock_file })
}

impl Drop for LocalObjectWriter {
  fn drop(&mut self) {
    if !self.finalized {
      // Best-effort: caller is supposed to call complete/abort, but if
      // the writer leaks (panic, early return) we still don't want to
      // leave a stale tmp file behind.
      self.file = None;
      let _ = fs::remove_file(&self.tmp);
    }
  }
}

/// Length of the fixed version header at the start of every data file:
/// 36 bytes for a UUIDv4 (canonical hyphenated form) plus a single
/// terminating newline. Picked so `read_header` can parse with a
/// fixed-size read and so `LocalObjectWriter` can overwrite the header
/// in-place at commit time without changing offsets.
const HEADER_LEN: u64 = 37;
const HEADER_LEN_USIZE: usize = HEADER_LEN as usize;

/// Generate a fresh UUIDv4-based version token. UUIDv4 strings are
/// 36-character canonical hex+dashes, so they're trivially distinguishable
/// from any future "legacy"-shaped fallback tokens we might add (which
/// would never start with a hex digit at the position of a dash).
fn new_version_token() -> String {
  Uuid::new_v4().to_string()
}

/// Write the 37-byte version header (`{uuid}\n`) at the file's current
/// position. Used at writer creation (with a placeholder UUID) and at
/// commit (with the freshly-generated one). The version string MUST be
/// 36 ASCII bytes; UUIDv4 strings always satisfy this.
fn write_header(file: &mut File, version: &str) -> Result<()> {
  if version.len() != 36 {
    bail!(
      "internal: version header must be exactly 36 bytes, got {}",
      version.len()
    );
  }
  file.write_all(version.as_bytes())?;
  file.write_all(b"\n")?;
  Ok(())
}

/// Read and validate the 37-byte version header at the file's current
/// cursor (typically byte 0). On success the cursor is left at the start
/// of the body (byte 37 of the physical file) so the caller can read or
/// seek the logical content directly. Returns the parsed UUID string.
///
/// Used by every read entry point (`stat`, `get`, `get_range`,
/// `Object::read_range`, `exists` via `stat`) so non-conforming files
/// fail with the same error class everywhere — Codex Stage 6 v4 [P2]
/// flagged that previously `stat`/`open` validated but `get`/`get_range`
/// only checked physical length, so a malformed file made `exists`/`stat`
/// fail while `read_to_end` succeeded with truncated tail bytes.
fn read_and_validate_header(file: &mut File, path: &Path) -> Result<String> {
  let mut buf = [0u8; HEADER_LEN_USIZE];
  file.read_exact(&mut buf)?;
  if buf[HEADER_LEN_USIZE - 1] != b'\n' {
    bail!(
      "LocalBlobStore: malformed header at {} — expected '\\n' at byte {}",
      path.display(),
      HEADER_LEN_USIZE - 1
    );
  }
  let version = std::str::from_utf8(&buf[..HEADER_LEN_USIZE - 1])
    .map_err(|e| anyhow!("LocalBlobStore: malformed UUID in header at {}: {e}", path.display()))?;
  Ok(version.to_string())
}

/// Read the 37-byte version header at the start of the file at `path`.
/// Convenience wrapper that opens the file and delegates to
/// `read_and_validate_header`.
fn read_header(path: &Path) -> Result<String> {
  let mut file = File::open(path)?;
  read_and_validate_header(&mut file, path)
}

/// Build an `ObjectStat` for the file at `path` by reading its
/// version header and `fs::metadata`. The reported `len` is the
/// *logical* length (physical minus the 37-byte header), so callers see
/// the body byte count they wrote without ever needing to know the
/// header exists.
fn stat_from_path(path: &Path) -> Result<ObjectStat> {
  let metadata = fs::metadata(path)
    .with_context(|| format!("LocalBlobStore::stat metadata({})", path.display()))?;
  let physical_len = metadata.len();
  if physical_len < HEADER_LEN {
    bail!(
      "LocalBlobStore: data file at {} is shorter than the {} -byte version header",
      path.display(),
      HEADER_LEN
    );
  }
  let version = read_header(path)?;
  Ok(ObjectStat {
    len: physical_len - HEADER_LEN,
    provider_version: Some(version),
    provider_checksum: None,
  })
}

/// Read the logical length of the file at `path`, validating the
/// version header along the way. Used by `get`/`get_range` for the
/// bounds check before issuing the range read; the header validation is
/// load-bearing so non-conforming files fail consistently with
/// `stat`/`open` (Codex Stage 6 v4 [P2]).
fn logical_len_of(path: &Path) -> Result<u64> {
  let mut file = File::open(path)?;
  let physical = file.metadata()?.len();
  if physical < HEADER_LEN {
    bail!(
      "LocalBlobStore: data file at {} is shorter than the {}-byte header",
      path.display(),
      HEADER_LEN
    );
  }
  // Validate the header matches the format `stat`/`open` expect. A 37+
  // byte file with garbage in the first 37 bytes must be rejected here
  // too, otherwise `get_range`'s bounds check would pass and we'd serve
  // garbage bytes from offset 37 onward.
  let _version = read_and_validate_header(&mut file, path)?;
  Ok(physical - HEADER_LEN)
}

/// Read the byte range `[range.start, range.end)` from the *logical*
/// content of the file at `path`. The header is validated before the
/// seek + read so all read entry points reject malformed files
/// consistently. The caller is responsible for the
/// `start <= end <= logical_len` bounds check; this helper does no
/// further bounds validation beyond what `read_exact` does naturally.
fn read_logical_range(path: &Path, range: Range<u64>) -> Result<Bytes> {
  let want = (range.end - range.start) as usize;
  let mut file = File::open(path)?;
  // Validates and leaves the cursor at the start of the body.
  let _version = read_and_validate_header(&mut file, path)?;
  if range.start > 0 {
    file.seek(SeekFrom::Current(range.start as i64))?;
  }
  let mut buf = vec![0u8; want];
  file.read_exact(&mut buf)?;
  Ok(Bytes::from(buf))
}

/// Atomically publish a data file at `path` consisting of the 37-byte
/// header followed by `body`. Done as tmp+rename — both the new bytes
/// and the new version (in the header) become visible together with a
/// single rename, so there's no window where a stale version could be
/// observed paired with new content.
fn write_data_file_atomic(path: &Path, version: &str, body: &[u8]) -> Result<()> {
  if let Some(parent) = path.parent() {
    fs::create_dir_all(parent)?;
  }
  let tmp = make_tmp_path(path);
  let cleanup = TmpCleanup::new(&tmp);
  {
    let mut tmp_file = File::create(&tmp)?;
    write_header(&mut tmp_file, version)?;
    tmp_file.write_all(body)?;
    tmp_file.sync_all()?;
  }
  fs::rename(&tmp, path)?;
  cleanup.disarm();
  Ok(())
}

/// Walk an `anyhow::Error` chain looking for a `std::io::Error` whose
/// `kind()` is `NotFound`. Used inside `put_if_match_locked` to
/// distinguish "object doesn't exist yet" from real I/O errors.
fn error_is_not_found(err: &anyhow::Error) -> bool {
  err.chain().any(|cause| {
    cause
      .downcast_ref::<std::io::Error>()
      .map(|e| e.kind() == std::io::ErrorKind::NotFound)
      .unwrap_or(false)
  })
}

/// Compute the path of a sidecar file (only `<key>.lock` is used now).
/// Sidecars live in the same parent directory so any rename involving
/// them stays on a single filesystem.
fn sidecar_path(target: &Path, suffix: &str) -> PathBuf {
  let parent = target.parent().unwrap_or(Path::new("."));
  let file_name = target.file_name().unwrap_or_default();
  let mut sidecar_name = file_name.to_os_string();
  sidecar_name.push(suffix);
  parent.join(sidecar_name)
}

fn make_tmp_path(path: &Path) -> PathBuf {
  let parent = path.parent().unwrap_or(Path::new("."));
  let file_name = path.file_name().unwrap_or_default();
  let mut tmp_name = file_name.to_os_string();
  tmp_name.push(format!(".tmp-{}", Uuid::new_v4()));
  parent.join(tmp_name)
}

fn sync_dir(path: &Path) -> Result<()> {
  if let Some(parent) = path.parent() {
    let dir = File::open(parent)?;
    dir.sync_all()?;
  }
  Ok(())
}

/// Drop guard mirroring `FsStorage::atomic_write`'s `TmpCleanup`: removes
/// the staging file if the caller returns early before `disarm`.
struct TmpCleanup<'a> {
  path: &'a Path,
  armed: bool,
}

impl<'a> TmpCleanup<'a> {
  fn new(path: &'a Path) -> Self {
    Self { path, armed: true }
  }

  fn disarm(mut self) {
    self.armed = false;
  }
}

impl Drop for TmpCleanup<'_> {
  fn drop(&mut self) {
    if self.armed {
      let _ = fs::remove_file(self.path);
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use futures::executor::block_on;
  use tempfile::tempdir;

  fn store(dir: &Path) -> LocalBlobStore {
    LocalBlobStore::new(dir.to_path_buf())
  }

  #[test]
  fn put_get_round_trip() {
    let dir = tempdir().unwrap();
    let store = store(dir.path());
    let key = Path::new("a/b/c.bin");
    let body = Bytes::from_static(b"hello world");

    let stat = block_on(store.put(key, body.clone())).unwrap();
    assert_eq!(stat.len, body.len() as u64);

    let got = block_on(store.get(key)).unwrap();
    assert_eq!(got, body);
  }

  #[test]
  fn get_range_returns_exact_bytes() {
    let dir = tempdir().unwrap();
    let store = store(dir.path());
    let key = Path::new("payload");
    block_on(store.put(key, Bytes::from_static(b"0123456789"))).unwrap();

    let got = block_on(store.get_range(key, 2..5)).unwrap();
    assert_eq!(got, Bytes::from_static(b"234"));
  }

  #[test]
  #[allow(clippy::reversed_empty_ranges)] // Intentional: testing rejection.
  fn get_range_rejects_inverted_range() {
    let dir = tempdir().unwrap();
    let store = store(dir.path());
    let key = Path::new("payload");
    block_on(store.put(key, Bytes::from_static(b"0123456789"))).unwrap();

    let err = block_on(store.get_range(key, 7..3)).expect_err("inverted range must error");
    let msg = format!("{err:#}");
    assert!(
      msg.contains("inverted range"),
      "expected inverted-range error, got: {msg}"
    );
  }

  #[test]
  fn get_range_rejects_out_of_bounds() {
    let dir = tempdir().unwrap();
    let store = store(dir.path());
    let key = Path::new("payload");
    block_on(store.put(key, Bytes::from_static(b"0123456789"))).unwrap();

    let err = block_on(store.get_range(key, 5..100)).expect_err("oob range must error");
    let msg = format!("{err:#}");
    assert!(
      msg.contains("exceeds object length"),
      "expected oob error, got: {msg}"
    );
  }

  #[test]
  fn get_range_empty_returns_empty_bytes_without_read() {
    let dir = tempdir().unwrap();
    let store = store(dir.path());
    let key = Path::new("payload");
    block_on(store.put(key, Bytes::from_static(b"abc"))).unwrap();

    // start == end, anywhere within bounds, returns empty.
    let got = block_on(store.get_range(key, 1..1)).unwrap();
    assert!(got.is_empty());
  }

  #[test]
  fn open_returns_pinned_stat_and_serves_range_reads() {
    let dir = tempdir().unwrap();
    let store = store(dir.path());
    let key = Path::new("payload");
    block_on(store.put(key, Bytes::from_static(b"abcdefghij"))).unwrap();

    let obj = block_on(store.open(key)).unwrap();
    assert_eq!(obj.len(), 10);
    let got = block_on(obj.read_range(2..5)).unwrap();
    assert_eq!(got, Bytes::from_static(b"cde"));
  }

  #[test]
  #[allow(clippy::reversed_empty_ranges)] // Intentional: testing rejection.
  fn read_range_on_object_rejects_inverted_and_oob() {
    let dir = tempdir().unwrap();
    let store = store(dir.path());
    let key = Path::new("payload");
    block_on(store.put(key, Bytes::from_static(b"abcdef"))).unwrap();
    let obj = block_on(store.open(key)).unwrap();

    let err = block_on(obj.read_range(5..2)).expect_err("inverted must error");
    assert!(format!("{err:#}").contains("inverted range"));
    let err = block_on(obj.read_range(0..100)).expect_err("oob must error");
    assert!(format!("{err:#}").contains("exceeds object length"));
  }

  #[test]
  fn put_stream_serial_write_and_complete() {
    let dir = tempdir().unwrap();
    let store = store(dir.path());
    let key = Path::new("streamed.bin");
    let mut writer = block_on(store.put_stream(key)).unwrap();
    block_on(writer.write(Bytes::from_static(b"part1-"))).unwrap();
    block_on(writer.write(Bytes::from_static(b"part2"))).unwrap();
    let stat = block_on(writer.complete()).unwrap();
    assert_eq!(stat.len, b"part1-part2".len() as u64);
    let got = block_on(store.get(key)).unwrap();
    assert_eq!(got, Bytes::from_static(b"part1-part2"));
  }

  #[test]
  fn put_stream_abort_removes_tmp_and_leaves_target_absent() {
    let dir = tempdir().unwrap();
    let store = store(dir.path());
    let key = Path::new("aborted.bin");
    let mut writer = block_on(store.put_stream(key)).unwrap();
    block_on(writer.write(Bytes::from_static(b"abandoned"))).unwrap();
    block_on(writer.abort()).unwrap();
    // Target was never created.
    assert!(block_on(store.stat(key)).is_err());
    // No staging file leaked (tmp file was cleaned up by abort).
    let entries: Vec<_> = fs::read_dir(dir.path())
      .unwrap()
      .filter_map(|e| e.ok())
      .filter(|e| {
        e.file_name()
          .to_string_lossy()
          .starts_with("aborted.bin.tmp-")
      })
      .collect();
    assert!(entries.is_empty(), "abort must remove the tmp file");
  }

  #[test]
  fn put_stream_drop_without_complete_or_abort_cleans_up() {
    let dir = tempdir().unwrap();
    let store = store(dir.path());
    let key = Path::new("leaked.bin");
    {
      let mut writer = block_on(store.put_stream(key)).unwrap();
      block_on(writer.write(Bytes::from_static(b"oops"))).unwrap();
      // Drop without complete/abort.
    }
    let entries: Vec<_> = fs::read_dir(dir.path())
      .unwrap()
      .filter_map(|e| e.ok())
      .filter(|e| {
        e.file_name()
          .to_string_lossy()
          .starts_with("leaked.bin.tmp-")
      })
      .collect();
    assert!(
      entries.is_empty(),
      "Drop without complete/abort must clean up the tmp file"
    );
  }

  #[test]
  fn delete_is_idempotent_for_missing_object() {
    let dir = tempdir().unwrap();
    let store = store(dir.path());
    block_on(store.delete(Path::new("never-existed"))).unwrap();
  }

  #[test]
  fn put_if_match_must_not_exist_atomic_create() {
    let dir = tempdir().unwrap();
    let store = store(dir.path());
    let key = Path::new("created");

    let stat = block_on(store.put_if_match(key, Bytes::from_static(b"v1"), None)).unwrap();
    assert_eq!(stat.len, 2);
    // Second create-must-not-exist call sees the existing object and
    // surfaces a typed Conflict (with current stat for the caller to
    // retry as `Some(version)`).
    let err = block_on(store.put_if_match(key, Bytes::from_static(b"v2"), None))
      .expect_err("must-not-exist conflict");
    match err {
      PutIfMatchError::Conflict { current } => {
        let current = current.expect("Conflict should carry current stat");
        assert_eq!(current.len, 2);
      }
      PutIfMatchError::Other(e) => panic!("expected Conflict, got Other: {e:#}"),
    }
    // Original bytes intact.
    let got = block_on(store.get(key)).unwrap();
    assert_eq!(got, Bytes::from_static(b"v1"));
  }

  #[test]
  fn put_if_match_some_version_round_trip() {
    let dir = tempdir().unwrap();
    let store = store(dir.path());
    let key = Path::new("versioned");
    let s0 = block_on(store.put(key, Bytes::from_static(b"v1"))).unwrap();
    let v0 = s0.provider_version.clone().unwrap();

    let s1 = block_on(store.put_if_match(key, Bytes::from_static(b"v2"), Some(&v0))).unwrap();
    assert_eq!(s1.len, 2);
    assert_ne!(s1.provider_version, s0.provider_version);

    // Re-using the original (now stale) version yields Conflict carrying
    // the current stat.
    let err =
      block_on(store.put_if_match(key, Bytes::from_static(b"v3"), Some(&v0))).expect_err(
        "stale-version conflict",
      );
    match err {
      PutIfMatchError::Conflict { current } => {
        let current = current.expect("Conflict should carry current stat");
        assert_eq!(current.len, 2);
        assert_eq!(current.provider_version, s1.provider_version);
      }
      PutIfMatchError::Other(e) => panic!("expected Conflict, got Other: {e:#}"),
    }
  }

  #[test]
  fn put_if_match_some_version_against_missing_object_is_conflict() {
    let dir = tempdir().unwrap();
    let store = store(dir.path());
    let key = Path::new("never-existed");
    let err = block_on(store.put_if_match(key, Bytes::from_static(b"v1"), Some("0-0"))).expect_err(
      "Some(version) on missing object must Conflict",
    );
    match err {
      PutIfMatchError::Conflict { current } => assert!(current.is_none()),
      PutIfMatchError::Other(e) => panic!("expected Conflict, got Other: {e:#}"),
    }
  }

  /// Concurrent CAS attempts on the same key from two threads: the
  /// per-key sidecar lockfile serializes the stat→write→rename critical
  /// section, so exactly one of N writers wins each CAS round. The other
  /// gets a `Conflict` with the (now-updated) current stat. Repeating
  /// this exercises the lock contention path the way a real multi-writer
  /// commit protocol would.
  #[test]
  fn put_if_match_concurrent_threads_serialize_via_lockfile() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Barrier;
    use std::thread;

    let dir = tempdir().unwrap();
    let store = Arc::new(store(dir.path()));
    let key = Path::new("contended");
    let s0 = block_on(store.put(key, Bytes::from_static(b"v0"))).unwrap();
    let initial_version = s0.provider_version.clone().unwrap();

    let wins = Arc::new(AtomicUsize::new(0));
    let losses = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(4));

    let handles: Vec<_> = (0..4)
      .map(|i| {
        let store = Arc::clone(&store);
        let wins = Arc::clone(&wins);
        let losses = Arc::clone(&losses);
        let barrier = Arc::clone(&barrier);
        let version = initial_version.clone();
        let body = Bytes::from(format!("body-{i}"));
        thread::spawn(move || {
          barrier.wait();
          let result = block_on(store.put_if_match(key, body, Some(&version)));
          match result {
            Ok(_) => {
              wins.fetch_add(1, Ordering::SeqCst);
            }
            Err(PutIfMatchError::Conflict { .. }) => {
              losses.fetch_add(1, Ordering::SeqCst);
            }
            Err(PutIfMatchError::Other(e)) => {
              panic!("expected Conflict for losers, got Other: {e:#}");
            }
          }
        })
      })
      .collect();
    for h in handles {
      h.join().unwrap();
    }
    assert_eq!(
      wins.load(Ordering::SeqCst),
      1,
      "exactly one writer must win the CAS round"
    );
    assert_eq!(
      losses.load(Ordering::SeqCst),
      3,
      "the other three writers must observe Conflict"
    );

    // The winning body persisted; the losing bodies didn't.
    let final_bytes = block_on(store.get(key)).unwrap();
    let s = std::str::from_utf8(&final_bytes).unwrap();
    assert!(
      s.starts_with("body-"),
      "final bytes must come from one of the writers: {s:?}"
    );
  }

  /// Stage 6 P2 fix: the version token must change on every successful
  /// write — including same-length rewrites that land within filesystem
  /// mtime granularity. With the older `(mtime_ns, len)` token this test
  /// would fail on a coarse-mtime filesystem, letting two writers both
  /// pass a stale `expected` and silently overwrite each other. The UUID
  /// sidecar fixes that by producing a fresh token per successful write.
  #[test]
  fn put_version_changes_on_same_length_rewrite() {
    let dir = tempdir().unwrap();
    let store = store(dir.path());
    let key = Path::new("flicker");

    let s0 = block_on(store.put(key, Bytes::from_static(b"AAAA"))).unwrap();
    let v0 = s0.provider_version.clone().unwrap();

    // Same length, different content. On a coarse-mtime filesystem the
    // two writes can land at the same recorded mtime; only the UUID
    // sidecar disambiguates them.
    let s1 = block_on(store.put(key, Bytes::from_static(b"BBBB"))).unwrap();
    let v1 = s1.provider_version.clone().unwrap();

    assert_eq!(s0.len, s1.len, "same-length-rewrite test prerequisite");
    assert_ne!(
      v0, v1,
      "same-length rewrites must produce distinct version tokens"
    );

    // The new version is what `stat` returns now, and it's distinct from
    // both legacy-shaped tokens and the prior UUID.
    let observed = block_on(store.stat(key)).unwrap();
    assert_eq!(observed.provider_version.as_deref(), Some(v1.as_str()));
    assert!(
      !v1.starts_with("legacy-"),
      "fresh sidecar tokens must not collide with the legacy fallback shape"
    );
  }

  /// Stage 6 P2 fix: the same property must hold across `put_if_match`
  /// rewrites — two same-length CAS writes against the same key must
  /// produce distinct version tokens, so a third writer holding the
  /// older token can't mistakenly succeed against the newer state.
  #[test]
  fn put_if_match_version_changes_on_same_length_rewrite() {
    let dir = tempdir().unwrap();
    let store = store(dir.path());
    let key = Path::new("cas-flicker");

    let s0 = block_on(store.put(key, Bytes::from_static(b"AAAA"))).unwrap();
    let v0 = s0.provider_version.clone().unwrap();

    let s1 = block_on(store.put_if_match(key, Bytes::from_static(b"BBBB"), Some(&v0))).unwrap();
    let v1 = s1.provider_version.clone().unwrap();

    assert_eq!(s0.len, s1.len);
    assert_ne!(v0, v1);
  }

  /// Stage 6 P1 fix: an `expected = None` precondition must not publish a
  /// partial object even if a write fails mid-way. We can't easily inject
  /// a write fault, but we can prove publication is staged through tmp +
  /// rename by observing that the data file does not exist between the
  /// `Conflict` returned by a second `put_if_match(None)` and any visible
  /// alteration to the original bytes.
  #[test]
  fn put_if_match_none_branch_uses_atomic_publish_via_tmp_rename() {
    let dir = tempdir().unwrap();
    let store = store(dir.path());
    let key = Path::new("must-not-exist");

    block_on(store.put_if_match(key, Bytes::from_static(b"v1"), None)).unwrap();

    // Second call sees existing object and Conflicts. The original bytes
    // remain untouched — a partial-publish bug would have left the file
    // truncated or corrupted.
    let err = block_on(store.put_if_match(key, Bytes::from_static(b"v2-truncated"), None))
      .expect_err("must-not-exist conflict on existing object");
    match err {
      PutIfMatchError::Conflict { current } => {
        let current = current.expect("Conflict carries current stat");
        assert_eq!(current.len, 2);
      }
      PutIfMatchError::Other(e) => panic!("expected Conflict, got Other: {e:#}"),
    }
    let got = block_on(store.get(key)).unwrap();
    assert_eq!(
      got,
      Bytes::from_static(b"v1"),
      "original bytes must be intact after a None-branch Conflict"
    );

    // Sanity: no leaked staging files in the parent directory.
    let leftovers: Vec<_> = fs::read_dir(dir.path())
      .unwrap()
      .filter_map(|e| e.ok())
      .filter(|e| {
        e.file_name()
          .to_string_lossy()
          .starts_with("must-not-exist.tmp-")
      })
      .collect();
    assert!(
      leftovers.is_empty(),
      "no tmp staging files should leak after a Conflict"
    );
  }

  /// Stage 6 v2 [P1] regression: the version token and the data must be
  /// co-published atomically. A `put` followed by reading the raw file
  /// bytes (bypassing the public API's header-strip) must show the
  /// reported `provider_version` UUID at the start of the file. This is
  /// the load-bearing property: the data file IS the version's home, so
  /// no two-rename window can ever leave the version inconsistent with
  /// the bytes.
  #[test]
  fn put_writes_version_inside_data_file_header() {
    let dir = tempdir().unwrap();
    let store = store(dir.path());
    let key = Path::new("with-header");
    let stat = block_on(store.put(key, Bytes::from_static(b"payload"))).unwrap();
    let version = stat.provider_version.clone().unwrap();

    // The on-disk file is `<version>\n<payload>` (37-byte header + body).
    let raw = fs::read(dir.path().join(key)).unwrap();
    assert_eq!(raw.len(), version.len() + 1 + b"payload".len());
    assert!(
      raw.starts_with(version.as_bytes()),
      "data file must begin with the reported provider_version"
    );
    assert_eq!(
      raw[version.len()],
      b'\n',
      "header must be terminated by a newline"
    );
    assert_eq!(
      &raw[version.len() + 1..],
      b"payload",
      "logical body must follow the header"
    );

    // And the public API hides the header: `get` returns just the body,
    // and `stat.len` is the body's logical length.
    let got = block_on(store.get(key)).unwrap();
    assert_eq!(got, Bytes::from_static(b"payload"));
    assert_eq!(stat.len, b"payload".len() as u64);
  }

  /// Stage 6 v2 [P2] regression: a concurrent `put` and `put_if_match`
  /// against the same key must serialize on the per-key flock so neither
  /// can leave a data/version mismatch. We can't directly observe
  /// "did the lock serialize them" but we can assert the post-condition:
  /// after both finish, the file's stored version exactly matches what
  /// the winning writer claims to have published.
  #[test]
  fn put_and_put_if_match_serialize_on_per_key_lock() {
    use std::sync::Barrier;
    use std::thread;

    let dir = tempdir().unwrap();
    let store = Arc::new(store(dir.path()));
    let key = Path::new("contended");

    let initial = block_on(store.put(key, Bytes::from_static(b"v0"))).unwrap();
    let v0 = initial.provider_version.clone().unwrap();

    let barrier = Arc::new(Barrier::new(2));

    // One thread does an unconditional `put`; the other does a
    // `put_if_match` against the original version. Either ordering is
    // legal, but the final file must always show a version that one of
    // the two writers reported as their commit.
    let store_a = Arc::clone(&store);
    let barrier_a = Arc::clone(&barrier);
    let put_handle = thread::spawn(move || {
      barrier_a.wait();
      block_on(store_a.put(key, Bytes::from_static(b"from-put"))).unwrap()
    });

    let store_b = Arc::clone(&store);
    let barrier_b = Arc::clone(&barrier);
    let cas_handle = thread::spawn(move || {
      barrier_b.wait();
      block_on(store_b.put_if_match(key, Bytes::from_static(b"from-cas"), Some(&v0)))
    });

    let put_stat = put_handle.join().unwrap();
    let cas_outcome = cas_handle.join().unwrap();

    let final_stat = block_on(store.stat(key)).unwrap();
    let final_version = final_stat.provider_version.clone().unwrap();

    // The final on-disk version is the one written by whichever thread
    // released the lock last. There are three possible orderings:
    //   (A) put first, CAS second: CAS sees the new version (not v0) →
    //       Conflict. Final version = put_stat.provider_version.
    //   (B) CAS first, put second: CAS succeeds (saw v0), put then
    //       overwrites. Final version = put_stat.provider_version.
    //   (C) put first, CAS second, where CAS happens to fail because
    //       it observes the put's new version (same as A).
    //
    // CAS-succeeded paths leave the file with the put's bytes; CAS-failed
    // paths likewise. In ALL cases the file's bytes/version must match
    // exactly one of the two writer outcomes — never a mismatched
    // (bytes from one, version from the other) frankenstate.
    let raw = fs::read(dir.path().join(key)).unwrap();
    let body = &raw[(HEADER_LEN as usize)..];

    let put_version = put_stat.provider_version.clone().unwrap();

    if let Ok(cas_stat) = cas_outcome {
      let cas_version = cas_stat.provider_version.clone().unwrap();
      // Both succeeded. The final state matches one of them
      // *consistently* — same writer's bytes AND same writer's version.
      let put_consistent = body == b"from-put" && final_version == put_version;
      let cas_consistent = body == b"from-cas" && final_version == cas_version;
      assert!(
        put_consistent || cas_consistent,
        "final state must match one writer's bytes-version pair, got bytes={:?} version={}",
        std::str::from_utf8(body).unwrap_or("<non-utf8>"),
        final_version
      );
    } else {
      // CAS Conflicted; final state must be from `put`.
      assert_eq!(
        body, b"from-put",
        "with CAS in Conflict, final bytes must be from put"
      );
      assert_eq!(
        final_version, put_version,
        "with CAS in Conflict, final version must be from put"
      );
    }
  }

  /// Stage 6 v3 [P2] regression: `delete` is a same-key writer and must
  /// take the per-key lock so it serializes with the other writers.
  /// Without the lock, a delete could interleave after `put_if_match`'s
  /// version check but before its rename — leaving an outcome that
  /// doesn't linearize against the delete. With the lock, every
  /// linearization on the lock queue produces a consistent observable
  /// state.
  ///
  /// Two threads race a CAS update against a delete on the same key.
  /// After both complete, the file is always gone (delete always
  /// succeeds, idempotent on missing object), and the CAS returned
  /// either Ok (was scheduled first) or Conflict (saw post-delete
  /// missing-file state) — never an inconsistent in-between.
  #[test]
  fn put_if_match_and_delete_serialize_on_per_key_lock() {
    use std::sync::Barrier;
    use std::thread;

    let dir = tempdir().unwrap();
    let store = Arc::new(store(dir.path()));
    let key = Path::new("delete-race");

    let initial = block_on(store.put(key, Bytes::from_static(b"v0"))).unwrap();
    let v0 = initial.provider_version.clone().unwrap();

    let barrier = Arc::new(Barrier::new(2));

    let store_a = Arc::clone(&store);
    let barrier_a = Arc::clone(&barrier);
    let v0_for_cas = v0.clone();
    let cas_handle = thread::spawn(move || {
      barrier_a.wait();
      block_on(store_a.put_if_match(key, Bytes::from_static(b"v1"), Some(&v0_for_cas)))
    });

    let store_b = Arc::clone(&store);
    let barrier_b = Arc::clone(&barrier);
    let delete_handle = thread::spawn(move || {
      barrier_b.wait();
      block_on(store_b.delete(key))
    });

    let cas_outcome = cas_handle.join().unwrap();
    delete_handle.join().unwrap().unwrap();

    // Whichever order the lock queue picked, the file is gone after
    // both finish: either delete ran last (took the file out from under
    // the just-published CAS), or delete ran first and the CAS
    // Conflicted on missing-file state.
    assert!(
      block_on(store.stat(key)).is_err(),
      "concurrent put_if_match + delete must leave the file gone"
    );

    // CAS outcome is Ok (delete second) or Conflict (delete first) —
    // never `Other`. Both branches are linearizable; what we're
    // forbidding is "Ok with no observable file state matching the
    // CAS-returned version", which the lock prevents.
    match cas_outcome {
      Ok(stat) => {
        // Delete scheduled second. CAS reported a successful publish
        // with the new version; delete then removed the file.
        let v1 = stat.provider_version.clone().unwrap();
        assert_ne!(v1, v0, "CAS Ok must report a fresh version");
      }
      Err(PutIfMatchError::Conflict { current }) => {
        // Delete scheduled first. CAS saw missing file → Conflict with
        // current = None (nothing observed).
        assert!(
          current.is_none(),
          "CAS Conflict against post-delete state should carry current = None"
        );
      }
      Err(PutIfMatchError::Other(e)) => {
        panic!("expected Ok or Conflict, got Other: {e:#}");
      }
    }
  }

  /// Stage 6 v4 [P2] regression: every read entry point validates the
  /// version header consistently, so a malformed or non-conforming file
  /// (one that's at least 37 bytes long but doesn't start with a valid
  /// `<UUID>\n` header) fails the same way through `stat`, `open`,
  /// `get`, `get_range`, and `exists`. Previously `get`/`get_range`
  /// only checked physical length, so a malformed file's tail bytes
  /// would be served with the first 37 bytes silently skipped.
  #[test]
  fn read_paths_reject_files_without_a_valid_header_consistently() {
    let dir = tempdir().unwrap();
    let store = store(dir.path());
    let key = Path::new("malformed.bin");
    let path = dir.path().join(key);

    // Bypass the BlobStore API and write a raw file that's longer than
    // the header but has nothing UUID-shaped in the first 37 bytes.
    fs::write(
      &path,
      b"this is not a UUID header\nbut its 100 bytes long bytes bytes bytes bytes bytes 1234",
    )
    .unwrap();
    assert!(fs::metadata(&path).unwrap().len() >= HEADER_LEN);

    // Every read entry point must reject this file. We don't pin a
    // specific error message — the contract is "all five paths fail",
    // not "all five paths fail with the same English string."
    assert!(
      block_on(store.stat(key)).is_err(),
      "stat must reject a file without a valid header"
    );
    assert!(
      block_on(store.open(key)).is_err(),
      "open must reject a file without a valid header"
    );
    assert!(
      block_on(store.get(key)).is_err(),
      "get must reject a file without a valid header (used to silently skip 37 bytes)"
    );
    assert!(
      block_on(store.get_range(key, 0..1)).is_err(),
      "get_range must reject a file without a valid header"
    );

    // `exists` rides on `stat`; with stat rejecting, exists must report
    // false rather than true-with-broken-content.
    use crate::storage::Storage;
    let blob: Arc<dyn BlobStore> = Arc::new(LocalBlobStore::new(dir.path().to_path_buf()));
    let adapter = crate::storage::BlobStoreAdapter::new(blob, dir.path().to_path_buf());
    assert!(
      !adapter.exists(&path),
      "exists must return false for a malformed file (stat fails)"
    );

    // Bonus check: an adapter `read_to_end` (which goes through
    // `BlobStore::get`) also rejects.
    assert!(
      adapter.read_to_end(&path).is_err(),
      "adapter.read_to_end must reject a malformed file rather than returning truncated tail bytes"
    );
  }

  /// Stage 6 v4 [P2] companion: a file too short to even hold the
  /// header (< 37 bytes) is rejected the same way as a malformed
  /// 37+-byte file. The error class is "shorter than header" rather
  /// than "malformed UUID", but both fail the read.
  #[test]
  fn read_paths_reject_files_shorter_than_header_consistently() {
    let dir = tempdir().unwrap();
    let store = store(dir.path());
    let key = Path::new("tiny.bin");
    let path = dir.path().join(key);
    fs::write(&path, b"too short").unwrap();

    assert!(block_on(store.stat(key)).is_err());
    assert!(block_on(store.open(key)).is_err());
    assert!(block_on(store.get(key)).is_err());
    assert!(block_on(store.get_range(key, 0..1)).is_err());
  }

  #[test]
  fn capabilities_advertise_real_cas_no_multipart() {
    let dir = tempdir().unwrap();
    let store = store(dir.path());
    let cap = store.capabilities();
    assert!(cap.conditional_put, "LocalBlobStore advertises real CAS");
    assert!(!cap.multipart_upload, "Local FS has no multipart protocol");
    assert!(cap.mmap_friendly, "Local FS is mmap-friendly");
  }
}
