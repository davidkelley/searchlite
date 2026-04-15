use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use crc32fast::Hasher;

use crate::api::types::Document;
use crate::storage::{Storage, StorageFile};
use crate::util::varint::{read_u64, write_u64};

#[derive(Debug, Clone)]
pub enum WalEntry {
  AddDoc(Document),
  DeleteDocId(String),
  Commit,
  WriteBinding(Vec<u8>),
}

pub struct Wal {
  _storage: Arc<dyn Storage>,
  _path: PathBuf,
  file: Box<dyn StorageFile>,
}

impl Wal {
  pub fn open(storage: Arc<dyn Storage>, path: &Path) -> Result<Self> {
    let file = storage
      .open_append(path)
      .with_context(|| format!("opening wal at {path:?}"))?;
    Ok(Self {
      _storage: storage,
      _path: path.to_path_buf(),
      file,
    })
  }

  pub fn append_add_doc(&mut self, doc: &Document) -> Result<()> {
    let payload = serde_json::to_vec(doc)?;
    self.append_entry(1, &payload)
  }

  /// Append a commit record and fsync the file to disk.
  ///
  /// The commit record is the WAL's durability boundary: once `append_commit`
  /// returns `Ok`, every preceding `AddDoc` / `DeleteDocId` entry is guaranteed
  /// to have been forced out of the kernel page cache, so a subsequent crash
  /// cannot resurrect "committed" state as pending ops on replay.
  ///
  /// `append_add_doc` and `append_delete_doc_id` deliberately defer fsync so
  /// that a burst of writes under a single commit stays fast; callers do not
  /// need to call [`Wal::sync`] separately after `append_commit`.
  pub fn append_commit(&mut self) -> Result<()> {
    self.append_entry(2, &[])?;
    self.file.sync_all()?;
    Ok(())
  }

  pub fn append_delete_doc_id(&mut self, doc_id: &str) -> Result<()> {
    self.append_entry(3, doc_id.as_bytes())
  }

  pub fn append_binding(&mut self, binding: &[u8]) -> Result<()> {
    self.append_entry(4, binding)
  }

  fn append_entry(&mut self, entry_type: u8, payload: &[u8]) -> Result<()> {
    let mut buf = Vec::with_capacity(16 + payload.len());
    write_u64(payload.len() as u64, &mut buf);
    buf.push(entry_type);
    buf.extend_from_slice(payload);
    let mut hasher = Hasher::new();
    hasher.update(&buf[buf.len() - payload.len() - 1..]);
    let checksum = hasher.finalize();
    buf.extend_from_slice(&checksum.to_le_bytes());
    self.file.write_all(&buf)?;
    self.file.flush()?;
    Ok(())
  }

  pub fn len(&mut self) -> Result<u64> {
    let current = self.file.stream_position()?;
    let end = self.file.seek(SeekFrom::End(0))?;
    self.file.seek(SeekFrom::Start(current))?;
    Ok(end)
  }

  pub fn is_empty(&mut self) -> Result<bool> {
    Ok(self.len()? == 0)
  }

  pub fn truncate_to(&mut self, len: u64) -> Result<()> {
    self.file.set_len(len)?;
    self.file.seek(SeekFrom::Start(len))?;
    self.file.sync_all()
  }

  pub fn sync(&mut self) -> Result<()> {
    self.file.flush()?;
    self.file.sync_all()
  }

  pub fn truncate(&mut self) -> Result<()> {
    self.file.set_len(0)?;
    self.file.seek(SeekFrom::Start(0))?;
    self.file.sync_all()
  }

  pub fn replay(storage: &dyn Storage, path: &Path) -> Result<Vec<WalEntry>> {
    if !storage.exists(path) {
      return Ok(Vec::new());
    }
    let data = storage.read_to_end(path)?;
    let mut cursor = 0usize;
    let mut entries = Vec::new();
    while cursor < data.len() {
      // Remember where this entry started so diagnostics can point at it
      // regardless of how far the cursor is advanced below.
      let entry_start = cursor;
      let (len, len_bytes) = match read_u64(&data[cursor..]) {
        Ok(v) => v,
        Err(_) => break,
      };
      cursor += len_bytes;
      if cursor >= data.len() {
        break;
      }
      let entry_type = data[cursor];
      cursor += 1;
      let len_usize = if let Ok(v) = usize::try_from(len) {
        v
      } else {
        break;
      };
      let payload_end = if let Some(end) = cursor.checked_add(len_usize) {
        end
      } else {
        break;
      };
      let checksum_end = if let Some(end) = payload_end.checked_add(4) {
        end
      } else {
        break;
      };
      if checksum_end > data.len() {
        break;
      }
      let payload = &data[cursor..payload_end];
      cursor = payload_end;
      let checksum_bytes = &data[cursor..checksum_end];
      cursor = checksum_end;
      let mut hasher = Hasher::new();
      hasher.update(&[entry_type]);
      hasher.update(payload);
      let checksum = hasher.finalize();
      if checksum.to_le_bytes() != checksum_bytes {
        break;
      }
      // Past this point the framing and CRC32 both check out, so the payload
      // is bit-for-bit what was written. A decode failure here cannot be
      // confused with a torn/truncated tail — it indicates a real semantic
      // mismatch (e.g. a schema change between versions). Surface those
      // failures rather than silently dropping the entry; see BUG-007.
      match entry_type {
        1 => {
          let doc = serde_json::from_slice::<Document>(payload).with_context(|| {
            format!(
              "WAL AddDoc entry at offset {entry_start} is checksum-valid but failed to deserialize"
            )
          })?;
          entries.push(WalEntry::AddDoc(doc));
        }
        2 => entries.push(WalEntry::Commit),
        3 => {
          let id = std::str::from_utf8(payload)
            .with_context(|| {
              format!(
                "WAL DeleteDocId entry at offset {entry_start} is checksum-valid but is not valid UTF-8"
              )
            })?
            .to_string();
          entries.push(WalEntry::DeleteDocId(id));
        }
        4 => entries.push(WalEntry::WriteBinding(payload.to_vec())),
        _ => {}
      }
    }
    Ok(entries)
  }

  pub fn replay_with_binding(
    storage: &dyn Storage,
    path: &Path,
  ) -> Result<(Option<Vec<u8>>, Vec<WalEntry>)> {
    let entries = if storage.exists(path) {
      Self::replay(storage, path)?
    } else {
      Vec::new()
    };
    let mut binding: Option<Vec<u8>> = None;
    let mut filtered = Vec::new();
    for entry in entries {
      match entry {
        WalEntry::WriteBinding(b) => binding = Some(b),
        other => filtered.push(other),
      }
    }
    Ok((binding, filtered))
  }

  pub fn last_pending_ops(
    storage: &dyn Storage,
    path: &Path,
  ) -> Result<(Option<Vec<u8>>, Vec<WalEntry>)> {
    let (binding, entries) = if storage.exists(path) {
      Self::replay_with_binding(storage, path)?
    } else {
      (None, Vec::new())
    };
    let mut pending = Vec::new();
    for entry in entries {
      match entry {
        WalEntry::AddDoc(_) | WalEntry::DeleteDocId(_) => pending.push(entry),
        WalEntry::Commit => pending.clear(),
        WalEntry::WriteBinding(_) => {} // already stripped
      }
    }
    Ok((binding, pending))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::tempdir;

  #[test]
  fn replays_entries_and_handles_commit() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("wal.log");
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    let mut wal = Wal::open(Arc::new(storage), &path).unwrap();
    let doc = Document {
      fields: [("body".into(), serde_json::json!("hello"))]
        .into_iter()
        .collect(),
    };
    wal.append_add_doc(&doc).unwrap();
    wal.append_commit().unwrap();
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    let entries = Wal::replay(&storage, &path).unwrap();
    assert!(matches!(entries.last(), Some(WalEntry::Commit)));
    let (_, pending) = Wal::last_pending_ops(&storage, &path).unwrap();
    assert!(pending.is_empty());
  }

  #[test]
  fn stops_on_invalid_checksum() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("wal.log");
    std::fs::write(&path, vec![1u8, 2, 3, 4]).unwrap();
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    let entries = Wal::replay(&storage, &path).unwrap();
    assert!(entries.is_empty());
  }

  #[test]
  fn handles_truncated_large_length_without_panic() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("wal.log");
    let mut buf = Vec::new();
    // Write an entry length that cannot fit in usize and provide no payload.
    crate::util::varint::write_u64(u64::MAX, &mut buf);
    std::fs::write(&path, buf).unwrap();
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    let entries = Wal::replay(&storage, &path).unwrap();
    assert!(entries.is_empty());
  }

  #[test]
  fn records_delete_entries() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("wal.log");
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    let mut wal = Wal::open(Arc::new(storage), &path).unwrap();
    wal.append_delete_doc_id("abc").unwrap();
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    let entries = Wal::replay(&storage, &path).unwrap();
    assert!(matches!(
      entries.first(),
      Some(WalEntry::DeleteDocId(id)) if id == "abc"
    ));
  }

  #[test]
  fn len_and_is_empty_track_bytes() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("wal.log");
    let storage = Arc::new(crate::storage::FsStorage::new(dir.path().to_path_buf()));
    let mut wal = Wal::open(storage.clone(), &path).unwrap();
    assert!(wal.is_empty().unwrap());
    let doc = Document {
      fields: [("body".into(), serde_json::json!("hello len"))]
        .into_iter()
        .collect(),
    };
    wal.append_add_doc(&doc).unwrap();
    let len_after = wal.len().unwrap();
    assert!(len_after > 0);
    assert!(!wal.is_empty().unwrap());
    wal.truncate().unwrap();
    assert!(wal.is_empty().unwrap());
  }

  /// Regression for BUG-007: a CRC-valid `AddDoc` payload that fails to
  /// deserialize into `Document` must surface an error instead of being
  /// silently dropped. The CRC has already guaranteed the bytes are
  /// bit-for-bit what was written, so decode failure is a real semantic
  /// mismatch (schema drift) that the operator must see.
  #[test]
  fn replay_surfaces_crc_valid_adddoc_decode_failure() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("wal.log");
    let mut buf = Vec::new();
    // Hand-craft an AddDoc entry (type 1) whose payload is valid bytes but
    // not a valid `Document` JSON (a bare integer is not an object).
    let payload: &[u8] = b"123";
    crate::util::varint::write_u64(payload.len() as u64, &mut buf);
    buf.push(1u8);
    buf.extend_from_slice(payload);
    let mut hasher = Hasher::new();
    hasher.update(&[1u8]);
    hasher.update(payload);
    let checksum = hasher.finalize();
    buf.extend_from_slice(&checksum.to_le_bytes());
    std::fs::write(&path, buf).unwrap();

    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    let err = Wal::replay(&storage, &path).expect_err(
      "CRC-valid but undecodable AddDoc payload must be surfaced, not silently dropped",
    );
    let msg = format!("{err:#}");
    assert!(
      msg.contains("AddDoc") && msg.contains("checksum-valid"),
      "unexpected error message: {msg}"
    );
  }

  /// Regression for BUG-007: a CRC-valid `DeleteDocId` payload that is not
  /// valid UTF-8 must surface an error. Same reasoning as the AddDoc case —
  /// CRC already confirmed the bytes are authentic, so the failure is real.
  #[test]
  fn replay_surfaces_crc_valid_delete_decode_failure() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("wal.log");
    let mut buf = Vec::new();
    // 0xFF is never valid UTF-8 on its own.
    let payload: &[u8] = &[0xFFu8];
    crate::util::varint::write_u64(payload.len() as u64, &mut buf);
    buf.push(3u8);
    buf.extend_from_slice(payload);
    let mut hasher = Hasher::new();
    hasher.update(&[3u8]);
    hasher.update(payload);
    let checksum = hasher.finalize();
    buf.extend_from_slice(&checksum.to_le_bytes());
    std::fs::write(&path, buf).unwrap();

    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    let err = Wal::replay(&storage, &path).expect_err(
      "CRC-valid but non-UTF-8 DeleteDocId payload must be surfaced, not silently dropped",
    );
    let msg = format!("{err:#}");
    assert!(
      msg.contains("DeleteDocId") && msg.contains("checksum-valid"),
      "unexpected error message: {msg}"
    );
  }

  /// A genuinely torn / truncated tail (bad checksum) must still stop replay
  /// cleanly without surfacing an error — BUG-007 is specifically about
  /// entries that pass the CRC but cannot be decoded.
  #[test]
  fn replay_still_stops_silently_on_bad_checksum_tail() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("wal.log");
    let storage = Arc::new(crate::storage::FsStorage::new(dir.path().to_path_buf()));
    let mut wal = Wal::open(storage.clone(), &path).unwrap();
    let doc = Document {
      fields: [("body".into(), serde_json::json!("good"))]
        .into_iter()
        .collect(),
    };
    wal.append_add_doc(&doc).unwrap();
    // Append a synthetic second entry by hand with intentionally invalid
    // checksum bytes so replay stops at the bad tail entry.
    let mut raw = std::fs::read(&path).unwrap();
    let mut bad = Vec::new();
    let payload = serde_json::to_vec(&doc).unwrap();
    crate::util::varint::write_u64(payload.len() as u64, &mut bad);
    bad.push(1u8);
    bad.extend_from_slice(&payload);
    // Intentionally wrong checksum bytes.
    bad.extend_from_slice(&[0u8, 0, 0, 0]);
    raw.extend_from_slice(&bad);
    std::fs::write(&path, &raw).unwrap();

    let entries = Wal::replay(storage.as_ref(), &path).unwrap();
    assert_eq!(
      entries.len(),
      1,
      "bad-checksum tail must stop replay silently, not abort",
    );
  }

  #[test]
  fn truncate_to_restores_previous_length() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("wal.log");
    let storage = Arc::new(crate::storage::FsStorage::new(dir.path().to_path_buf()));
    let mut wal = Wal::open(storage.clone(), &path).unwrap();
    let doc = Document {
      fields: [("body".into(), serde_json::json!("hello truncate"))]
        .into_iter()
        .collect(),
    };
    wal.append_add_doc(&doc).unwrap();
    let len_after_add = wal.len().unwrap();
    wal.append_commit().unwrap();
    let len_after_commit = wal.len().unwrap();
    assert!(len_after_commit > len_after_add);
    wal.truncate_to(len_after_add).unwrap();
    let len_restored = wal.len().unwrap();
    assert_eq!(len_restored, len_after_add);
  }

  /// A [`StorageFile`] that proxies to an inner file and records each
  /// `sync_all` call so tests can assert on durability behaviour without
  /// inspecting the host kernel's page cache.
  struct SyncCountingFile {
    inner: crate::storage::DynFile,
    sync_count: Arc<std::sync::atomic::AtomicUsize>,
  }

  impl std::io::Read for SyncCountingFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
      self.inner.read(buf)
    }
  }

  impl std::io::Write for SyncCountingFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
      self.inner.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
      self.inner.flush()
    }
  }

  impl std::io::Seek for SyncCountingFile {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
      self.inner.seek(pos)
    }
  }

  impl crate::storage::StorageFile for SyncCountingFile {
    fn set_len(&mut self, len: u64) -> Result<()> {
      self.inner.set_len(len)
    }

    fn sync_all(&mut self) -> Result<()> {
      self
        .sync_count
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
      self.inner.sync_all()
    }
  }

  struct SyncCountingStorage {
    inner: crate::storage::FsStorage,
    sync_count: Arc<std::sync::atomic::AtomicUsize>,
  }

  impl crate::storage::Storage for SyncCountingStorage {
    fn root(&self) -> &Path {
      self.inner.root()
    }

    fn ensure_dir(&self, path: &Path) -> Result<()> {
      self.inner.ensure_dir(path)
    }

    fn exists(&self, path: &Path) -> bool {
      self.inner.exists(path)
    }

    fn open_read(&self, path: &Path) -> Result<crate::storage::DynFile> {
      self.inner.open_read(path)
    }

    fn open_write(&self, path: &Path) -> Result<crate::storage::DynFile> {
      let file = self.inner.open_write(path)?;
      Ok(Box::new(SyncCountingFile {
        inner: file,
        sync_count: Arc::clone(&self.sync_count),
      }))
    }

    fn open_append(&self, path: &Path) -> Result<crate::storage::DynFile> {
      let file = self.inner.open_append(path)?;
      Ok(Box::new(SyncCountingFile {
        inner: file,
        sync_count: Arc::clone(&self.sync_count),
      }))
    }

    fn read_to_end(&self, path: &Path) -> Result<Vec<u8>> {
      self.inner.read_to_end(path)
    }

    fn write_all(&self, path: &Path, data: &[u8]) -> Result<()> {
      self.inner.write_all(path, data)
    }

    fn atomic_write(&self, path: &Path, data: &[u8]) -> Result<()> {
      self.inner.atomic_write(path, data)
    }

    fn remove(&self, path: &Path) -> Result<()> {
      self.inner.remove(path)
    }

    fn remove_dir_all(&self, path: &Path) -> Result<()> {
      self.inner.remove_dir_all(path)
    }
  }

  #[test]
  fn append_commit_fsyncs_before_returning() {
    use std::sync::atomic::Ordering;

    let dir = tempdir().unwrap();
    let path = dir.path().join("wal.log");
    let sync_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let storage = Arc::new(SyncCountingStorage {
      inner: crate::storage::FsStorage::new(dir.path().to_path_buf()),
      sync_count: Arc::clone(&sync_count),
    });
    let mut wal = Wal::open(storage, &path).unwrap();
    let doc = Document {
      fields: [("body".into(), serde_json::json!("durable"))]
        .into_iter()
        .collect(),
    };

    // Plain appends must not fsync so that batching stays cheap.
    wal.append_add_doc(&doc).unwrap();
    assert_eq!(sync_count.load(Ordering::SeqCst), 0);

    // `append_commit` is the durability boundary: it must fsync before
    // returning `Ok`, without requiring the caller to also invoke `sync`.
    wal.append_commit().unwrap();
    assert_eq!(sync_count.load(Ordering::SeqCst), 1);
  }
}
