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
      .with_context(|| format!("opening wal at {:?}", path))?;
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

  pub fn append_commit(&mut self) -> Result<()> {
    self.append_entry(2, &[])
  }

  pub fn append_delete_doc_id(&mut self, doc_id: &str) -> Result<()> {
    self.append_entry(3, doc_id.as_bytes())
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
    Ok(self.file.seek(SeekFrom::End(0))?)
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
      let payload_end = cursor.saturating_add(len as usize);
      if payload_end + 4 > data.len() {
        break;
      }
      let payload = &data[cursor..payload_end];
      cursor = payload_end;
      let checksum_bytes = &data[cursor..cursor + 4];
      cursor += 4;
      let mut hasher = Hasher::new();
      hasher.update(&[entry_type]);
      hasher.update(payload);
      let checksum = hasher.finalize();
      if checksum.to_le_bytes() != checksum_bytes {
        break;
      }
      match entry_type {
        1 => {
          if let Ok(doc) = serde_json::from_slice::<Document>(payload) {
            entries.push(WalEntry::AddDoc(doc));
          }
        }
        2 => entries.push(WalEntry::Commit),
        3 => {
          if let Ok(id) = std::str::from_utf8(payload) {
            entries.push(WalEntry::DeleteDocId(id.to_string()));
          }
        }
        _ => {}
      }
    }
    Ok(entries)
  }

  pub fn last_pending_ops(storage: &dyn Storage, path: &Path) -> Result<Vec<WalEntry>> {
    let entries = if storage.exists(path) {
      Self::replay(storage, path)?
    } else {
      Vec::new()
    };
    let mut pending = Vec::new();
    for entry in entries {
      match entry {
        WalEntry::AddDoc(_) | WalEntry::DeleteDocId(_) => pending.push(entry),
        WalEntry::Commit => pending.clear(),
      }
    }
    Ok(pending)
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
    let pending = Wal::last_pending_ops(&storage, &path).unwrap();
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
}
