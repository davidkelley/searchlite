use std::path::Path;

use anyhow::Result;

use crate::storage::Storage;
use crate::util::checksum::checksum;
use crate::util::fst::TinyFst;
use crate::util::varint::{read_u64, write_u64};

pub fn write_terms(storage: &dyn Storage, path: &Path, terms: &[(String, u64)]) -> Result<()> {
  let mut file = storage.open_write(path)?;
  file.write_all(&(terms.len() as u64).to_le_bytes())?;
  let mut buf = Vec::new();
  for (term, offset) in terms {
    write_u64(term.len() as u64, &mut buf);
    buf.extend_from_slice(term.as_bytes());
    buf.extend_from_slice(&offset.to_le_bytes());
  }
  let crc = checksum(&buf);
  file.write_all(&buf)?;
  file.write_all(&crc.to_le_bytes())?;
  file.sync_all()?;
  Ok(())
}

pub fn read_terms(storage: &dyn Storage, path: &Path) -> Result<TinyFst> {
  let buf = storage.read_to_end(path)?;
  if buf.len() < 12 {
    return Ok(TinyFst::default());
  }
  let term_count = u64::from_le_bytes([
    buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
  ]);
  let payload = &buf[8..];
  let (data, crc_bytes) = payload.split_at(payload.len() - 4);
  let expected = u32::from_le_bytes([crc_bytes[0], crc_bytes[1], crc_bytes[2], crc_bytes[3]]);
  let actual = checksum(data);
  if expected != actual {
    return Ok(TinyFst::default());
  }
  let mut cursor = 0usize;
  let mut pairs = Vec::with_capacity(term_count as usize);
  for _ in 0..term_count {
    let (len, consumed) = read_u64(&data[cursor..])?;
    cursor += consumed;
    let end = cursor + len as usize;
    if end > data.len() {
      break;
    }
    let term = String::from_utf8_lossy(&data[cursor..end]).into_owned();
    cursor = end;
    if cursor + 8 > data.len() {
      break;
    }
    let offset = u64::from_le_bytes([
      data[cursor],
      data[cursor + 1],
      data[cursor + 2],
      data[cursor + 3],
      data[cursor + 4],
      data[cursor + 5],
      data[cursor + 6],
      data[cursor + 7],
    ]);
    cursor += 8;
    pairs.push((term, offset));
  }
  Ok(TinyFst::from_terms(&pairs))
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::tempdir;

  #[test]
  fn roundtrips_terms_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("terms");
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    let pairs = vec![
      ("alpha".to_string(), 10),
      ("beta".to_string(), 20),
      ("gamma".to_string(), 30),
    ];
    write_terms(&storage, &path, &pairs).unwrap();
    let fst = read_terms(&storage, &path).unwrap();
    assert_eq!(fst.get("beta"), Some(20));
    assert_eq!(fst.get("missing"), None);
  }

  #[test]
  fn invalid_checksum_returns_empty() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("terms");
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    write_terms(&storage, &path, &[("term".to_string(), 1)]).unwrap();
    let mut data = std::fs::read(&path).unwrap();
    let last = data.last_mut().unwrap();
    *last = last.wrapping_add(1);
    std::fs::write(&path, data).unwrap();
    let fst = read_terms(&storage, &path).unwrap();
    assert!(fst.get("term").is_none());
  }
}
