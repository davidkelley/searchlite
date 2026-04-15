use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};

use crate::storage::Storage;
use crate::util::checksum::checksum;
use crate::util::fst::TinyFst;
use crate::util::varint::{read_u64, write_u64};

/// Reject `term_count` values that cannot possibly be backed by the bytes
/// still available in the terms payload. `term_count` is an 8-byte header
/// read directly from the file and is **not** covered by the inner CRC
/// (the CRC is scoped to the payload only), so a single-byte flip in that
/// header — or a legacy/merge-produced segment with an empty outer
/// checksum map — would otherwise drive a multi-gigabyte
/// `Vec::with_capacity` before the per-entry read loop ever discovered
/// the file is too short. The minimum per-term stride is 9 bytes: a
/// 1-byte varint length (smallest LEB128 encoding) + 0 bytes for an
/// empty term + 8 bytes for the `u64` offset. Mirrors the helper
/// introduced by BUG-012 for `fastfields::read_fields`.
fn checked_count(count: u64, min_stride: u64, remaining: usize) -> Result<usize> {
  let needed = count
    .checked_mul(min_stride)
    .ok_or_else(|| anyhow!("term_count {count} * stride {min_stride} overflows u64"))?;
  if needed > remaining as u64 {
    return Err(anyhow!(
      "term_count {count} would need {needed} bytes but only {remaining} remain in terms payload"
    ));
  }
  // At this point `count * min_stride <= remaining` and `remaining: usize`,
  // so `count` fits in a `usize` without truncation.
  Ok(count as usize)
}

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
  // Validate the untrusted `term_count` header against the bytes still
  // available in the payload before it reaches `Vec::with_capacity`. The
  // payload excludes the 8-byte header and the trailing 4-byte CRC, so
  // `data.len()` is the exact byte budget the per-entry loop has left to
  // work with. See BUG-207; the header is not covered by the inner CRC.
  let term_count = checked_count(term_count, 9, data.len())?;
  let mut cursor = 0usize;
  let mut pairs = Vec::with_capacity(term_count);
  for _ in 0..term_count {
    let (len, consumed) = read_u64(&data[cursor..])?;
    cursor += consumed;
    // `len` is read from an untrusted varint on disk. On 32-bit targets
    // `len as usize` would truncate; on any target `cursor + len as usize`
    // could wrap past `usize::MAX` for a crafted `u64::MAX`-ish length,
    // letting the subsequent bounds check pass on the wrapped value and
    // panicking the slice index below. Use `try_from` + `checked_add` so
    // the overflow surfaces as a structured error instead.
    let len_usize = usize::try_from(len).map_err(|_| {
      anyhow!("terms file at {path:?} declares term length {len} that exceeds usize")
    })?;
    let end = cursor.checked_add(len_usize).ok_or_else(|| {
      anyhow!("terms file at {path:?} declares term length {len} that overflows cursor")
    })?;
    if end > data.len() {
      bail!("terms file at {path:?} ended unexpectedly while reading term");
    }
    // `buf` prepended an 8-byte term_count header before this payload, so translate
    // the payload-relative `cursor` to an absolute file offset to help operators
    // locate the corruption. `with_context` preserves the underlying `Utf8Error`
    // so the error chain keeps the low-level failure as a source.
    let file_offset = cursor + 8;
    let term = std::str::from_utf8(&data[cursor..end])
      .with_context(|| {
        format!("terms file at {path:?} contains non-UTF-8 term at file offset {file_offset}")
      })?
      .to_string();
    cursor = end;
    // Same wrap concern as above, but with a fixed 8-byte stride — still
    // worth guarding defensively so the bounds check cannot be bypassed
    // by a `cursor` near `usize::MAX`.
    let offset_end = cursor
      .checked_add(8)
      .ok_or_else(|| anyhow!("terms file at {path:?} cursor overflow reading offset"))?;
    if offset_end > data.len() {
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
    let original_term = "valid";
    write_terms(&storage, &path, &[(original_term.to_string(), 1)]).unwrap();
    let raw = std::fs::read(&path).unwrap();

    // Layout: [term_count:u64][payload][crc32:u32]
    let header_len = 8;
    let mut payload = raw[header_len..raw.len() - 4].to_vec();

    // Locate the term bytes by parsing the varint-encoded length prefix so the
    // test stays correct regardless of how many bytes the varint encoding uses.
    let (term_len, varint_len) = read_u64(&payload).unwrap();
    assert_eq!(term_len as usize, original_term.len());
    let term_start = varint_len;
    let term_end = term_start + term_len as usize;

    // Overwrite the term bytes with an ill-formed UTF-8 sequence: 0xC3 is a
    // 2-byte UTF-8 lead followed by bytes that are not valid continuation
    // bytes. `from_utf8_lossy` would coerce this to U+FFFD; strict `from_utf8`
    // must reject it.
    let invalid = [0xC3, 0x28, 0xA0, 0xA1, 0xFF];
    assert_eq!(term_end - term_start, invalid.len());
    payload[term_start..term_end].copy_from_slice(&invalid);

    // Recompute CRC over the mutated payload so the checksum check passes and
    // the decoder actually reaches the UTF-8 validation step.
    let new_crc = checksum(&payload);

    let mut rebuilt = Vec::with_capacity(raw.len());
    rebuilt.extend_from_slice(&raw[..header_len]);
    rebuilt.extend_from_slice(&payload);
    rebuilt.extend_from_slice(&new_crc.to_le_bytes());
    std::fs::write(&path, rebuilt).unwrap();

    let err = read_terms(&storage, &path).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
      msg.contains("non-UTF-8 term"),
      "expected non-UTF-8 term error, got: {msg}"
    );
    // The underlying Utf8Error should be preserved as the source of the chain.
    assert!(
      err.chain().any(|cause| cause.is::<std::str::Utf8Error>()),
      "expected Utf8Error in error chain, got: {msg}"
    );
  }

  /// Regression for BUG-207: a tampered terms file whose 8-byte
  /// `term_count` header claims `u64::MAX` entries must be rejected
  /// before `Vec::with_capacity` commits a multi-gigabyte allocation.
  /// The inner CRC intentionally does not cover the header, so
  /// flipping its bytes alone cannot invalidate the checksum — the
  /// bounds-check against the remaining payload is the load-bearing
  /// guard here. Mirrors BUG-205's regression in `postings.rs` and
  /// BUG-012's in `fastfields.rs`.
  #[test]
  fn read_terms_rejects_oversized_term_count_on_short_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("terms");
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());

    write_terms(&storage, &path, &[("only".to_string(), 1)]).unwrap();
    let mut raw = std::fs::read(&path).unwrap();
    // Overwrite only the 8-byte term_count header. The inner CRC is
    // computed over the payload only, so it remains valid.
    raw[..8].copy_from_slice(&u64::MAX.to_le_bytes());
    std::fs::write(&path, raw).unwrap();

    let err = read_terms(&storage, &path).expect_err("oversized term_count must be rejected");
    let msg = format!("{err:#}").to_lowercase();
    // `u64::MAX` trips the `checked_mul` guard; any moderately-oversized
    // count trips the `needed > remaining` branch. Either message is a
    // correct rejection — what matters is that the error surfaces
    // `term_count` before a multi-gigabyte allocation is committed.
    assert!(
      msg.contains("term_count") && (msg.contains("remain") || msg.contains("overflow")),
      "expected bounds-check error, got: {msg}"
    );
  }

  /// Also verify that an oversized but not `u64::MAX` header — one
  /// large enough to commit gigabytes of capacity but small enough to
  /// not trigger the multiplication overflow branch — is still
  /// rejected. Targets the `needed > remaining` branch specifically.
  #[test]
  fn read_terms_rejects_moderately_oversized_term_count() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("terms");
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());

    write_terms(&storage, &path, &[("only".to_string(), 1)]).unwrap();
    let mut raw = std::fs::read(&path).unwrap();
    // 4_000_000_000 * 9 B stride = 36 GB claimed; the written file is
    // only a handful of bytes so this must be rejected without ever
    // reaching `Vec::with_capacity`.
    raw[..8].copy_from_slice(&4_000_000_000u64.to_le_bytes());
    std::fs::write(&path, raw).unwrap();

    let err = read_terms(&storage, &path).expect_err("oversized term_count must be rejected");
    let msg = format!("{err:#}").to_lowercase();
    assert!(
      msg.contains("term_count") && msg.contains("remain"),
      "expected bounds-check error, got: {msg}"
    );
  }

  /// Defense-in-depth companion to BUG-207: a crafted per-term varint
  /// length near `u64::MAX` must be rejected via a structured error
  /// rather than wrapping `cursor + len as usize` and panicking on the
  /// subsequent slice index. Addresses the panic vector flagged by
  /// Copilot review on the BUG-207 PR.
  #[test]
  fn read_terms_rejects_oversized_per_term_length() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("terms");
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());

    // Assemble a valid terms file by hand whose single per-term varint
    // encodes `u64::MAX` as the term byte length. The CRC is computed
    // over the payload so the checksum guard passes and execution
    // reaches the cursor arithmetic we want to exercise.
    let mut payload = Vec::new();
    write_u64(u64::MAX, &mut payload);
    // No term bytes — the cursor-wrap guard fires before any term data
    // could be read. Append a plausible 8-byte offset so the payload
    // is not rejected for being structurally incomplete earlier on.
    payload.extend_from_slice(&0u64.to_le_bytes());
    let crc = checksum(&payload);

    let mut raw = Vec::new();
    // term_count = 1 — passes checked_count (1 * 9 B <= payload bytes).
    raw.extend_from_slice(&1u64.to_le_bytes());
    raw.extend_from_slice(&payload);
    raw.extend_from_slice(&crc.to_le_bytes());
    std::fs::write(&path, raw).unwrap();

    let err = read_terms(&storage, &path).expect_err("oversized per-term length must be rejected");
    let msg = format!("{err:#}").to_lowercase();
    assert!(
      msg.contains("term length"),
      "expected per-term length error, got: {msg}"
    );
  }
}
