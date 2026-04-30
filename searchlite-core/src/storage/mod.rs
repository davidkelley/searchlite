use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use parking_lot::RwLock;
use uuid::Uuid;

pub mod blob;
pub use blob::{
  ArtifactIdentity, BlobStore, Capabilities, ContentHash, Object, ObjectStat, ObjectWriter,
  ProviderChecksum, PutIfMatchError,
};

pub trait StorageFile: Read + Write + Seek + Send {
  fn set_len(&mut self, len: u64) -> Result<()>;
  fn sync_all(&mut self) -> Result<()>;
}

impl StorageFile for File {
  fn set_len(&mut self, len: u64) -> Result<()> {
    File::set_len(self, len).map_err(Into::into)
  }

  fn sync_all(&mut self) -> Result<()> {
    File::sync_all(self).map_err(Into::into)
  }
}

pub type DynFile = Box<dyn StorageFile>;

pub trait Storage: Send + Sync {
  fn root(&self) -> &Path;
  fn ensure_dir(&self, path: &Path) -> Result<()>;
  fn exists(&self, path: &Path) -> bool;
  fn open_read(&self, path: &Path) -> Result<DynFile>;
  fn open_write(&self, path: &Path) -> Result<DynFile>;
  fn open_append(&self, path: &Path) -> Result<DynFile>;
  fn read_to_end(&self, path: &Path) -> Result<Vec<u8>>;
  fn write_all(&self, path: &Path, data: &[u8]) -> Result<()>;
  fn atomic_write(&self, path: &Path, data: &[u8]) -> Result<()>;
  fn remove(&self, path: &Path) -> Result<()>;
  fn remove_dir_all(&self, path: &Path) -> Result<()>;
}

pub struct FsStorage {
  root: PathBuf,
}

impl FsStorage {
  pub fn new(root: PathBuf) -> Self {
    Self { root }
  }
}

impl Storage for FsStorage {
  fn root(&self) -> &Path {
    &self.root
  }

  fn ensure_dir(&self, path: &Path) -> Result<()> {
    fs::create_dir_all(path)?;
    Ok(())
  }

  fn exists(&self, path: &Path) -> bool {
    path.exists()
  }

  fn open_read(&self, path: &Path) -> Result<DynFile> {
    Ok(Box::new(File::open(path)?))
  }

  fn open_write(&self, path: &Path) -> Result<DynFile> {
    if let Some(parent) = path.parent() {
      fs::create_dir_all(parent)?;
    }
    Ok(Box::new(File::create(path)?))
  }

  fn open_append(&self, path: &Path) -> Result<DynFile> {
    if let Some(parent) = path.parent() {
      fs::create_dir_all(parent)?;
    }
    let file = File::options()
      .create(true)
      .append(true)
      .read(true)
      .open(path)?;
    Ok(Box::new(file))
  }

  fn read_to_end(&self, path: &Path) -> Result<Vec<u8>> {
    Ok(fs::read(path)?)
  }

  fn write_all(&self, path: &Path, data: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
      fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    file.write_all(data)?;
    file.sync_all()?;
    sync_dir(path)?;
    Ok(())
  }

  fn atomic_write(&self, path: &Path, data: &[u8]) -> Result<()> {
    // The staging file must not collide with any other in-flight `atomic_write`
    // targeting a sibling file. `Path::with_extension("tmp")` replaces the
    // existing extension, so `foo.json` and `foo.meta` both reduce to
    // `foo.tmp` and race on the same file — one caller's payload can end up
    // under the other caller's final path (see BUG-019 / #157). Appending a
    // per-call UUID suffix to the full file name keeps siblings isolated while
    // still placing the staging file next to the target so the `rename` stays
    // on the same filesystem (and therefore atomic).
    let file_name = path
      .file_name()
      .ok_or_else(|| anyhow!("atomic_write target has no file name: {path:?}"))?;
    let mut tmp_name = file_name.to_os_string();
    tmp_name.push(format!(".tmp-{}", Uuid::new_v4()));
    let tmp = path.with_file_name(tmp_name);
    if let Some(parent) = path.parent() {
      fs::create_dir_all(parent)?;
    }
    // RAII guard: if any step between `File::create` and a successful `rename`
    // returns early, the staging file is best-effort removed so repeated I/O
    // failures (disk full, EIO, interrupted sync) don't accumulate stale
    // `.tmp-<uuid>` entries next to the target.
    let mut guard = TmpCleanup::new(&tmp);
    {
      let mut file = File::create(&tmp)?;
      file.write_all(data)?;
      file.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    // Rename succeeded: the staging name no longer refers to our bytes, so
    // there is nothing for the guard to clean up.
    guard.disarm();
    sync_dir(path)?;
    Ok(())
  }

  fn remove(&self, path: &Path) -> Result<()> {
    if path.exists() {
      if let Some(parent) = path.parent() {
        if !parent.exists() {
          return Ok(());
        }
      }
      fs::remove_file(path)
        .map_err(|e| anyhow!("failed to remove file {}: {e}", path.display()))?;
    }
    Ok(())
  }

  fn remove_dir_all(&self, path: &Path) -> Result<()> {
    if path.exists() {
      fs::remove_dir_all(path)
        .map_err(|e| anyhow!("failed to remove directory {}: {e}", path.display()))?;
    }
    Ok(())
  }
}

fn sync_dir(path: &Path) -> Result<()> {
  if let Some(parent) = path.parent() {
    let dir = File::open(parent)?;
    dir.sync_all()?;
  }
  Ok(())
}

/// Drop-guard that best-effort removes an `atomic_write` staging file if the
/// caller returns early (create / write / sync / rename failure). After a
/// successful rename the guard is disarmed because the staging name no longer
/// refers to the bytes we wrote.
struct TmpCleanup<'a> {
  path: &'a Path,
  armed: bool,
}

impl<'a> TmpCleanup<'a> {
  fn new(path: &'a Path) -> Self {
    Self { path, armed: true }
  }

  fn disarm(&mut self) {
    self.armed = false;
  }
}

impl Drop for TmpCleanup<'_> {
  fn drop(&mut self) {
    if self.armed {
      // Best-effort: we're already unwinding or returning an error, so a
      // cleanup failure is just a leaked staging file — strictly worse to
      // mask the original error by propagating this one.
      let _ = fs::remove_file(self.path);
    }
  }
}

pub struct InMemoryStorage {
  root: PathBuf,
  files: RwLock<HashMap<PathBuf, Arc<RwLock<Vec<u8>>>>>,
}

impl InMemoryStorage {
  pub fn new(root: PathBuf) -> Self {
    Self {
      root,
      files: RwLock::new(HashMap::new()),
    }
  }

  fn entry(&self, path: &Path) -> Arc<RwLock<Vec<u8>>> {
    let mut map = self.files.write();
    map
      .entry(path.to_path_buf())
      .or_insert_with(|| Arc::new(RwLock::new(Vec::new())))
      .clone()
  }

  fn open_with_mode(&self, path: &Path, truncate: bool, append: bool) -> Result<DynFile> {
    let data = self.entry(path);
    if truncate {
      data.write().clear();
    }
    let pos = if append { data.read().len() as u64 } else { 0 };
    Ok(Box::new(MemFile { data, pos }))
  }
}

impl Storage for InMemoryStorage {
  fn root(&self) -> &Path {
    &self.root
  }

  fn ensure_dir(&self, _path: &Path) -> Result<()> {
    Ok(())
  }

  fn exists(&self, path: &Path) -> bool {
    self.files.read().contains_key(path)
  }

  fn open_read(&self, path: &Path) -> Result<DynFile> {
    if !self.exists(path) {
      return Err(anyhow!("file {path:?} missing in memory storage"));
    }
    self.open_with_mode(path, false, false)
  }

  fn open_write(&self, path: &Path) -> Result<DynFile> {
    self.open_with_mode(path, true, false)
  }

  fn open_append(&self, path: &Path) -> Result<DynFile> {
    self.open_with_mode(path, false, true)
  }

  fn read_to_end(&self, path: &Path) -> Result<Vec<u8>> {
    if let Some(buf) = self.files.read().get(path) {
      return Ok(buf.read().clone());
    }
    Err(anyhow!("file {path:?} missing in memory storage"))
  }

  fn write_all(&self, path: &Path, data: &[u8]) -> Result<()> {
    let entry = self.entry(path);
    let mut guard = entry.write();
    guard.clear();
    guard.extend_from_slice(data);
    Ok(())
  }

  fn atomic_write(&self, path: &Path, data: &[u8]) -> Result<()> {
    self.write_all(path, data)
  }

  fn remove(&self, path: &Path) -> Result<()> {
    let mut map = self.files.write();
    map.remove(path);
    Ok(())
  }

  fn remove_dir_all(&self, path: &Path) -> Result<()> {
    let mut map = self.files.write();
    map.retain(|p, _| !p.starts_with(path));
    Ok(())
  }
}

struct MemFile {
  data: Arc<RwLock<Vec<u8>>>,
  pos: u64,
}

impl Read for MemFile {
  fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
    let data = self.data.read();
    if self.pos as usize >= data.len() {
      return Ok(0);
    }
    let available = data.len() - self.pos as usize;
    let len = available.min(buf.len());
    buf[..len].copy_from_slice(&data[self.pos as usize..self.pos as usize + len]);
    self.pos += len as u64;
    Ok(len)
  }
}

impl Write for MemFile {
  fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
    let mut data = self.data.write();
    let end = (self.pos as usize).saturating_add(buf.len());
    if end > data.len() {
      data.resize(end, 0);
    }
    data[self.pos as usize..end].copy_from_slice(buf);
    self.pos = end as u64;
    Ok(buf.len())
  }

  fn flush(&mut self) -> std::io::Result<()> {
    Ok(())
  }
}

impl Seek for MemFile {
  fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
    let new = match pos {
      SeekFrom::Start(off) => off as i64,
      SeekFrom::End(off) => {
        let len = self.data.read().len() as i64;
        len + off
      }
      SeekFrom::Current(off) => self.pos as i64 + off,
    };
    if new < 0 {
      return Err(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        "negative seek",
      ));
    }
    self.pos = new as u64;
    Ok(self.pos)
  }
}

impl StorageFile for MemFile {
  fn set_len(&mut self, len: u64) -> Result<()> {
    let mut data = self.data.write();
    data.resize(len as usize, 0);
    if self.pos > len {
      self.pos = len;
    }
    Ok(())
  }

  fn sync_all(&mut self) -> Result<()> {
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::sync::Barrier;
  use std::thread;

  #[test]
  fn atomic_write_persists_payload_under_target_path() {
    let dir = tempfile::tempdir().unwrap();
    let storage = FsStorage::new(dir.path().to_path_buf());
    let target = dir.path().join("foo.json");
    storage.atomic_write(&target, b"hello").unwrap();
    let got = fs::read(&target).unwrap();
    assert_eq!(got, b"hello");
  }

  /// Regression for BUG-019: `Path::with_extension("tmp")` collapsed siblings
  /// that share a stem onto the same staging path, so interleaved writes could
  /// swap payloads or leave a target missing. The staging path must now be
  /// unique per call, so concurrent writes to sibling files both land intact.
  #[test]
  fn atomic_write_isolates_concurrent_sibling_writes() {
    // Repeat a few times so interleavings that the old implementation would
    // lose are very likely to surface if the fix regresses.
    for _ in 0..16 {
      let dir = tempfile::tempdir().unwrap();
      let storage = Arc::new(FsStorage::new(dir.path().to_path_buf()));
      let a = dir.path().join("shared.json");
      let b = dir.path().join("shared.meta");
      let c = dir.path().join("shared.checksum");
      let barrier = Arc::new(Barrier::new(3));

      let handles: Vec<_> = [
        (a.clone(), b"A-payload".to_vec()),
        (b.clone(), b"B-payload".to_vec()),
        (c.clone(), b"C-payload".to_vec()),
      ]
      .into_iter()
      .map(|(path, data)| {
        let storage = Arc::clone(&storage);
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
          barrier.wait();
          storage.atomic_write(&path, &data).unwrap();
        })
      })
      .collect();

      for h in handles {
        h.join().unwrap();
      }

      assert_eq!(fs::read(&a).unwrap(), b"A-payload");
      assert_eq!(fs::read(&b).unwrap(), b"B-payload");
      assert_eq!(fs::read(&c).unwrap(), b"C-payload");

      // No staging files should be left behind after a successful run.
      let leftovers: Vec<_> = fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().contains(".tmp-"))
        .collect();
      assert!(
        leftovers.is_empty(),
        "unexpected staging files left behind: {leftovers:?}"
      );
    }
  }

  /// Regression for Copilot feedback on BUG-019: when any step between
  /// `File::create` and a successful `rename` fails (write, sync, rename), the
  /// staging file must be cleaned up — otherwise repeated I/O failures (disk
  /// full, EIO, interrupted sync) accumulate stale `.tmp-<uuid>` files next to
  /// the target. We drive the failure by pointing `atomic_write` at a target
  /// path that is an existing non-empty directory: `File::create` on the
  /// staging path still succeeds, but `fs::rename(tmp, target)` fails because
  /// the target directory is not empty. After the failure the parent must
  /// contain no leftover staging files.
  #[test]
  fn atomic_write_removes_staging_file_on_rename_failure() {
    let dir = tempfile::tempdir().unwrap();
    let storage = FsStorage::new(dir.path().to_path_buf());
    let target = dir.path().join("occupied");
    fs::create_dir(&target).unwrap();
    // Put a child inside the target directory so the eventual rename fails
    // with ENOTEMPTY / EISDIR on every supported platform.
    fs::write(target.join("bystander"), b"child").unwrap();

    let err = storage
      .atomic_write(&target, b"payload")
      .expect_err("rename over a non-empty directory must fail");
    // Surface the error to the assertion message if anything changes.
    let _ = err;

    let leftovers: Vec<_> = fs::read_dir(dir.path())
      .unwrap()
      .filter_map(|e| e.ok())
      .filter(|e| e.file_name().to_string_lossy().starts_with("occupied.tmp-"))
      .collect();
    assert!(
      leftovers.is_empty(),
      "staging file was not cleaned up after rename failure: {leftovers:?}"
    );
    // The pre-existing target directory and its child must be untouched.
    assert!(target.is_dir(), "target directory unexpectedly removed");
    assert_eq!(fs::read(target.join("bystander")).unwrap(), b"child");
  }

  /// The staging file must not reuse the target's stem in a way that removes
  /// the original extension — otherwise callers who rely on the extension to
  /// disambiguate sibling files can still collide.
  #[test]
  fn atomic_write_staging_path_preserves_target_extension() {
    let dir = tempfile::tempdir().unwrap();
    let storage = FsStorage::new(dir.path().to_path_buf());
    // Writing "shared.meta" while "shared.tmp" pre-exists on disk must not
    // clobber the pre-existing `shared.tmp`. The pre-existing file is our
    // stand-in for another writer's unrelated file that happens to sit at the
    // old (buggy) staging location.
    let bystander = dir.path().join("shared.tmp");
    fs::write(&bystander, b"bystander").unwrap();
    let target = dir.path().join("shared.meta");
    storage.atomic_write(&target, b"payload").unwrap();
    assert_eq!(fs::read(&target).unwrap(), b"payload");
    assert_eq!(
      fs::read(&bystander).unwrap(),
      b"bystander",
      "atomic_write must not touch an unrelated sibling file sharing the stem"
    );
  }
}
