use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::{Context, Result};
use crc32fast::Hasher;

use crate::api::types::Document;
use crate::util::varint::{read_u64, write_u64};

#[derive(Debug, Clone)]
pub enum WalEntry {
  AddDoc(Document),
  Commit,
}

pub struct Wal {
  file: File,
}

impl Wal {
  pub fn open(path: &Path) -> Result<Self> {
    let file = File::options()
      .create(true)
      .append(true)
      .read(true)
      .open(path)
      .with_context(|| format!("opening wal at {:?}", path))?;
    Ok(Self { file })
  }

  pub fn append_add_doc(&mut self, doc: &Document) -> Result<()> {
    let payload = serde_json::to_vec(doc)?;
    self.append_entry(1, &payload)
  }

  pub fn append_commit(&mut self) -> Result<()> {
    self.append_entry(2, &[])
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

  pub fn truncate(&mut self) -> Result<()> {
    self.file.set_len(0)?;
    self.file.seek(SeekFrom::Start(0))?;
    Ok(())
  }

  pub fn replay(path: &Path) -> Result<Vec<WalEntry>> {
    if !path.exists() {
      return Ok(Vec::new());
    }
    let mut file = File::open(path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;
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
        _ => {}
      }
    }
    Ok(entries)
  }

  pub fn last_pending_documents(path: &Path) -> Result<Vec<Document>> {
    let entries = Self::replay(path)?;
    let mut pending = Vec::new();
    for entry in entries {
      match entry {
        WalEntry::AddDoc(doc) => pending.push(doc),
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
    let mut wal = Wal::open(&path).unwrap();
    let doc = Document {
      fields: [("body".into(), serde_json::json!("hello"))]
        .into_iter()
        .collect(),
    };
    wal.append_add_doc(&doc).unwrap();
    wal.append_commit().unwrap();
    let entries = Wal::replay(&path).unwrap();
    assert!(matches!(entries.last(), Some(WalEntry::Commit)));
    let pending = Wal::last_pending_documents(&path).unwrap();
    assert!(pending.is_empty());
  }

  #[test]
  fn stops_on_invalid_checksum() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("wal.log");
    std::fs::write(&path, vec![1u8, 2, 3, 4]).unwrap();
    let entries = Wal::replay(&path).unwrap();
    assert!(entries.is_empty());
  }
}
