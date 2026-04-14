use std::path::Path;

use anyhow::{bail, Result};

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
    bail!("terms file at {path:?} is truncated");
  }
  let term_count = u64::from_le_bytes([
    buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
  ]);
  let payload = &buf[8..];
  let (data, crc_bytes) = payload.split_at(payload.len() - 4);
  let expected = u32::from_le_bytes([crc_bytes[0], crc_bytes[1], crc_bytes[2], crc_bytes[3]]);
  let actual = checksum(data);
  if expected != actual {
    bail!("terms file at {path:?} failed checksum validation");
  }
  let mut cursor = 0usize;
  let mut pairs = Vec::with_capacity(term_count as usize);
  for _ in 0..term_count {
    let (len, consumed) = read_u64(&data[cursor..])?;
    cursor += consumed;
    let end = cursor + len as usize;
    if end > data.len() {
      bail!("terms file at {path:?} ended unexpectedly while reading term");
    }
    let term = std::str::from_utf8(&data[cursor..end])
      .map_err(|e| {
        anyhow::anyhow!("terms file at {path:?} contains non-UTF-8 term at offset {cursor}: {e}")
      })?
      .to_string();
    cursor = end;
    if cursor + 8 > data.len() {
      bail!("terms file at {path:?} ended unexpectedly while reading offset");
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
  fn invalid_checksum_errors() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("terms");
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    write_terms(&storage, &path, &[("term".to_string(), 1)]).unwrap();
    let mut data = std::fs::read(&path).unwrap();
    let last = data.last_mut().unwrap();
    *last = last.wrapping_add(1);
    std::fs::write(&path, data).unwrap();
    let err = read_terms(&storage, &path).unwrap_err();
    assert!(err.to_string().contains("failed checksum"));
  }

  #[test]
  fn invalid_utf8_term_errors() {
    // Regression test for BUG-010: read_terms should refuse to decode terms that
    // contain non-UTF-8 bytes rather than silently replacing them with U+FFFD.
    let dir = tempdir().unwrap();
    let path = dir.path().join("terms");
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());

    // Write a valid terms file containing a single ASCII term, then overwrite
    // its payload with invalid UTF-8 bytes while keeping the CRC consistent.
    write_terms(&storage, &path, &[("valid".to_string(), 1)]).unwrap();
    let raw = std::fs::read(&path).unwrap();

    // Layout: [term_count:u64][payload][crc32:u32]
    let header_len = 8;
    let payload = &raw[header_len..raw.len() - 4];
    let mut payload = payload.to_vec();

    // Locate the first byte of the term bytes: after the varint-encoded length.
    // `write_u64(5, ..)` emits a single byte (0x05), followed by the 5 term bytes.
    // Replace them with an overlong / invalid UTF-8 sequence (0xC3 is a 2-byte
    // UTF-8 lead but is followed by ASCII bytes that are not valid continuation
    // bytes, producing an ill-formed sequence that `from_utf8_lossy` would
    // coerce to U+FFFD).
    let term_offset = 1; // byte index within `payload` where the term bytes start
    payload[term_offset] = 0xC3;
    payload[term_offset + 1] = 0x28;
    payload[term_offset + 2] = 0xA0;
    payload[term_offset + 3] = 0xA1;
    payload[term_offset + 4] = 0xFF;

    // Recompute CRC over the mutated payload so the checksum check passes and
    // the decoder actually reaches the UTF-8 validation step.
    let new_crc = checksum(&payload);

    let mut rebuilt = Vec::with_capacity(raw.len());
    rebuilt.extend_from_slice(&raw[..header_len]);
    rebuilt.extend_from_slice(&payload);
    rebuilt.extend_from_slice(&new_crc.to_le_bytes());
    std::fs::write(&path, rebuilt).unwrap();

    let err = read_terms(&storage, &path).unwrap_err();
    let msg = err.to_string();
    assert!(
      msg.contains("non-UTF-8 term"),
      "expected non-UTF-8 term error, got: {msg}"
    );
  }
}
