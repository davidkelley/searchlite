use std::io::{Read, Seek, SeekFrom, Write};

use anyhow::{bail, Result};

use crate::DocId;

/// Hard cap on stored document payload size to avoid OOM or corrupt reads.
pub const MAX_DOCSTORE_BYTES: usize = 32 * 1024 * 1024; // 32 MiB

pub struct DocStoreWriter<'a, W: Write + Seek + ?Sized> {
  file: &'a mut W,
  offsets: Vec<u64>,
  #[cfg_attr(not(feature = "zstd"), allow(dead_code))]
  use_zstd: bool,
  /// Set when a previous `add_document` failed and the post-failure rewind
  /// could not restore the file cursor. In that state the stream has
  /// unknown trailing bytes, so all subsequent `add_document` calls must
  /// fail fast rather than paper over a torn record.
  poisoned: bool,
}

impl<'a, W: Write + Seek + ?Sized> DocStoreWriter<'a, W> {
  pub fn new(file: &'a mut W, use_zstd: bool) -> Self {
    Self {
      file,
      offsets: Vec::new(),
      use_zstd,
      poisoned: false,
    }
  }

  pub fn add_document(&mut self, doc: &serde_json::Value) -> Result<()> {
    if self.poisoned {
      bail!(
        "DocStoreWriter is poisoned after a prior IO failure whose cursor \
         rewind could not be recovered; drop the writer and rebuild the segment"
      );
    }
    let offset = self.file.stream_position()?;
    #[allow(unused_mut)]
    let mut data = serde_json::to_vec(doc)?;
    if data.len() > MAX_DOCSTORE_BYTES {
      bail!(
        "stored document too large ({} bytes, max {})",
        data.len(),
        MAX_DOCSTORE_BYTES
      );
    }
    #[cfg(feature = "zstd")]
    if self.use_zstd {
      data = zstd::stream::encode_all(&data[..], 0)?;
    }
    if data.len() > MAX_DOCSTORE_BYTES {
      bail!(
        "stored document too large ({} bytes, max {})",
        data.len(),
        MAX_DOCSTORE_BYTES
      );
    }
    let len = data.len() as u32;
    // Only publish the offset once both writes succeed. On IO failure rewind
    // the file cursor so a retry or the next `add_document` call overwrites
    // the partial record instead of leaving a torn header/payload that would
    // desynchronise doc ids from offsets on the next write. If the rewind
    // itself fails, mark the writer poisoned so the caller cannot silently
    // append on top of an unknown cursor position.
    match self
      .file
      .write_all(&len.to_le_bytes())
      .and_then(|_| self.file.write_all(&data))
    {
      Ok(()) => {
        self.offsets.push(offset);
        Ok(())
      }
      Err(write_err) => match self.file.seek(SeekFrom::Start(offset)) {
        Ok(_) => Err(anyhow::Error::new(write_err).context(format!(
          "DocStoreWriter failed to append doc {}; file cursor rewound to {} for retry",
          self.offsets.len(),
          offset
        ))),
        Err(seek_err) => {
          self.poisoned = true;
          Err(anyhow::Error::new(write_err).context(format!(
            "DocStoreWriter failed to append doc {} and could not rewind file cursor \
             to {}: {seek_err}; writer is now poisoned",
            self.offsets.len(),
            offset
          )))
        }
      },
    }
  }

  pub fn offsets(&self) -> &[u64] {
    &self.offsets
  }
}

/// Stateful seek+read docstore reader. Stage 8b switched
/// `SegmentReader::get_doc` to issue one bounded `Object::read_range`
/// per fetch instead of the seek+two-read shape this struct
/// implements, so production code no longer constructs this type.
/// It is kept around for unit-test coverage of the on-wire format
/// and as a documented reference impl until Stage 8c removes it.
#[allow(dead_code)]
pub struct DocStoreReader<R: Read + Seek> {
  file: R,
  offsets: Vec<u64>,
  #[cfg_attr(not(feature = "zstd"), allow(dead_code))]
  use_zstd: bool,
}

#[allow(dead_code)]
impl<R: Read + Seek> DocStoreReader<R> {
  pub fn new(file: R, offsets: Vec<u64>, use_zstd: bool) -> Self {
    Self {
      file,
      offsets,
      use_zstd,
    }
  }

  pub fn get(&mut self, doc_id: DocId) -> Result<serde_json::Value> {
    let offset = *self
      .offsets
      .get(doc_id as usize)
      .ok_or_else(|| anyhow::anyhow!("doc id out of bounds"))?;
    self.file.seek(SeekFrom::Start(offset))?;
    let mut len_bytes = [0u8; 4];
    self.file.read_exact(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes) as usize;
    if len > MAX_DOCSTORE_BYTES {
      bail!("stored document length {len} exceeds maximum {MAX_DOCSTORE_BYTES}");
    }
    let mut buf = vec![0u8; len];
    self.file.read_exact(&mut buf)?;
    // Reuse the shared parse path so this seek+read shape and Stage
    // 8b's BlobStore range-read path can never drift on
    // length/MAX/zstd/JSON handling. The bundled byte slice is
    // `len_bytes ++ buf` per the on-wire format.
    let mut bundle = Vec::with_capacity(4 + buf.len());
    bundle.extend_from_slice(&len_bytes);
    bundle.extend_from_slice(&buf);
    decode_docstore_record(&bundle, self.use_zstd)
  }
}

/// Decode one docstore record from a `[u32 LE length][payload]` byte
/// bundle. The bundle MUST be exactly `4 + length` bytes; trailing
/// bytes are treated as corruption (offset table claimed a longer
/// record than the embedded length actually encodes).
///
/// This is the single source of truth for docstore parse semantics:
/// the legacy `DocStoreReader::get` (seek + 2 read_exacts) and the
/// Stage 8b `SegmentReader::get_doc` (one bounded `Object::read_range`)
/// both call into here so they cannot drift on:
///
/// * `MAX_DOCSTORE_BYTES` enforcement (pre- and post-decompress).
/// * zstd handling (gated on the `zstd` feature flag).
/// * JSON decode error context.
/// * Length-prefix / range-length consistency checks.
///
/// `use_zstd` reflects the segment-meta flag (`SegmentFileMeta::use_zstd`).
pub fn decode_docstore_record(bundle: &[u8], use_zstd: bool) -> Result<serde_json::Value> {
  if bundle.len() < 4 {
    bail!(
      "docstore record truncated: need at least 4 length-prefix bytes, got {}",
      bundle.len()
    );
  }
  let len = u32::from_le_bytes(bundle[..4].try_into().expect("4-byte slice")) as usize;
  if len > MAX_DOCSTORE_BYTES {
    bail!("stored document length {len} exceeds maximum {MAX_DOCSTORE_BYTES}");
  }
  if 4 + len != bundle.len() {
    bail!(
      "docstore record length mismatch: header says {len} bytes, range has {} payload bytes \
       (4 + {len} != {}); offset table or file may be corrupt",
      bundle.len().saturating_sub(4),
      bundle.len()
    );
  }
  let payload = &bundle[4..];
  #[cfg(feature = "zstd")]
  let owned;
  #[cfg(feature = "zstd")]
  let payload: &[u8] = if use_zstd {
    owned = zstd::stream::decode_all(payload)?;
    if owned.len() > MAX_DOCSTORE_BYTES {
      bail!(
        "stored document length {} exceeds maximum {MAX_DOCSTORE_BYTES} after decompression",
        owned.len()
      );
    }
    &owned
  } else {
    payload
  };
  #[cfg(not(feature = "zstd"))]
  let _ = use_zstd;
  let json: serde_json::Value = serde_json::from_slice(payload)?;
  Ok(json)
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::NamedTempFile;

  #[test]
  fn stores_and_loads_documents() {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();
    let mut file = tmp.reopen().unwrap();
    let mut writer = DocStoreWriter::new(&mut file, false);
    writer
      .add_document(&serde_json::json!({"title": "Rust", "year": 2024}))
      .unwrap();
    writer
      .add_document(&serde_json::json!({"title": "Search", "year": 2023}))
      .unwrap();

    let offsets = writer.offsets().to_vec();
    drop(writer);
    drop(file);
    let reader_file = std::fs::File::open(path).unwrap();
    let mut reader = DocStoreReader::new(reader_file, offsets, false);
    let first = reader.get(0).unwrap();
    assert_eq!(first["title"], "Rust");
    assert!(reader.get(2).is_err());
  }

  #[test]
  fn rejects_oversized_documents() {
    let tmp = NamedTempFile::new().unwrap();
    let mut file = tmp.reopen().unwrap();
    let mut writer = DocStoreWriter::new(&mut file, false);
    // Build a string whose serialized JSON length is MAX_DOCSTORE_BYTES + 1 to
    // exceed the bound regardless of compression.
    let inner = String::from_utf8(vec![b'a'; MAX_DOCSTORE_BYTES - 1]).unwrap();
    let huge = serde_json::json!(inner);
    let err = writer.add_document(&huge).unwrap_err();
    assert!(err.to_string().contains("too large"));
  }

  /// Write + Seek shim used to simulate mid-record IO failures in
  /// `add_document`. The writer accepts bytes up to `fail_after` in total,
  /// then every further `write` returns `ErrorKind::BrokenPipe`. A `write`
  /// call that straddles the threshold returns a short count and the next
  /// call produces the error — i.e. the failure surfaces *after* `fail_after`
  /// bytes have been accepted, not *on* the `fail_after`-th byte.
  struct FailingWriter {
    inner: std::io::Cursor<Vec<u8>>,
    fail_after: usize,
    written: usize,
  }

  impl FailingWriter {
    fn new(fail_after: usize) -> Self {
      Self {
        inner: std::io::Cursor::new(Vec::new()),
        fail_after,
        written: 0,
      }
    }
  }

  impl Write for FailingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
      if self.written >= self.fail_after {
        return Err(std::io::Error::new(
          std::io::ErrorKind::BrokenPipe,
          "injected failure",
        ));
      }
      let remaining = self.fail_after - self.written;
      let to_write = buf.len().min(remaining);
      let n = self.inner.write(&buf[..to_write])?;
      self.written += n;
      // If the caller's buffer was truncated we return the short count — the
      // next `write` call will observe the injected failure and `write_all`
      // will loop, surface the error, and the caller sees a real half-failed
      // IO device.
      Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
      self.inner.flush()
    }
  }

  impl Seek for FailingWriter {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
      self.inner.seek(pos)
    }
  }

  #[test]
  fn add_document_preserves_offsets_on_io_failure() {
    // Drive two writes through FailingWriter. The first runs with an
    // effectively infinite byte budget and succeeds. The second lowers the
    // byte budget so that the record's write_all only partially completes
    // before FailingWriter injects BrokenPipe. After the failure the offsets
    // vector must still describe only the first successful document so the
    // caller can retry with the original doc_id.
    let mut file = FailingWriter::new(usize::MAX);
    let mut writer = DocStoreWriter::new(&mut file, false);
    writer
      .add_document(&serde_json::json!({"title": "Rust"}))
      .unwrap();
    assert_eq!(writer.offsets().len(), 1);

    // Lock the failure threshold to the current position so the next write
    // partially succeeds then errors.
    let pos_after_first = writer.file.inner.position() as usize;
    writer.file.fail_after = pos_after_first + 2;
    writer.file.written = pos_after_first;

    let err = writer
      .add_document(&serde_json::json!({"title": "Search"}))
      .unwrap_err();
    // Use alternate-debug formatting so the full anyhow context chain is
    // rendered (includes both the wrapping "cursor rewound" message and the
    // original BrokenPipe root cause).
    let chain = format!("{err:#}");
    assert!(
      chain.to_lowercase().contains("pipe") || chain.contains("injected"),
      "expected BrokenPipe / injected in error chain, got: {chain}"
    );
    assert!(
      chain.contains("rewound"),
      "expected wrapping context mentioning rewind in error chain, got: {chain}"
    );
    // Critically: offsets was NOT advanced, so the next successful add_document
    // re-uses doc_id == 1 (not 2).
    assert_eq!(writer.offsets().len(), 1);
    // And the cursor was rewound so the partial bytes get overwritten.
    assert_eq!(
      writer.file.inner.position() as usize,
      pos_after_first,
      "cursor should be rewound to the pre-failed-write position"
    );

    // Retry with an unbounded byte budget and confirm doc_id 1 is assigned.
    writer.file.fail_after = usize::MAX;
    writer.file.written = pos_after_first;
    writer
      .add_document(&serde_json::json!({"title": "Retry"}))
      .unwrap();
    assert_eq!(writer.offsets().len(), 2);

    let offsets = writer.offsets().to_vec();
    drop(writer);
    // Read back both docs from the in-memory buffer via a cursor.
    let buffer = file.inner.into_inner();
    let cursor = std::io::Cursor::new(buffer);
    let mut reader = DocStoreReader::new(cursor, offsets, false);
    let first = reader.get(0).unwrap();
    assert_eq!(first["title"], "Rust");
    let second = reader.get(1).unwrap();
    assert_eq!(
      second["title"], "Retry",
      "doc_id 1 must resolve to the retried document, not the abandoned one"
    );
  }

  /// Write + Seek shim whose `seek` returns `ErrorKind::Other` after the
  /// first call, letting us reach the unrecoverable-rewind branch in
  /// `add_document` without otherwise affecting writes.
  struct RewindFailingWriter {
    inner: std::io::Cursor<Vec<u8>>,
    fail_after: usize,
    written: usize,
    seek_calls: usize,
    fail_seek_from: usize,
  }

  impl RewindFailingWriter {
    fn new(fail_after: usize, fail_seek_from: usize) -> Self {
      Self {
        inner: std::io::Cursor::new(Vec::new()),
        fail_after,
        written: 0,
        seek_calls: 0,
        fail_seek_from,
      }
    }
  }

  impl Write for RewindFailingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
      if self.written >= self.fail_after {
        return Err(std::io::Error::new(
          std::io::ErrorKind::BrokenPipe,
          "injected failure",
        ));
      }
      let remaining = self.fail_after - self.written;
      let to_write = buf.len().min(remaining);
      let n = self.inner.write(&buf[..to_write])?;
      self.written += n;
      Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
      self.inner.flush()
    }
  }

  impl Seek for RewindFailingWriter {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
      let call = self.seek_calls;
      self.seek_calls += 1;
      if call >= self.fail_seek_from {
        return Err(std::io::Error::other("injected seek failure"));
      }
      self.inner.seek(pos)
    }
  }

  #[test]
  fn add_document_poisons_writer_when_rewind_fails() {
    // stream_position() is the first seek call (call #0). Writes for the
    // initial doc all succeed. On the second add_document we force write_all
    // to fail, and also configure seek to start failing so the recovery
    // rewind cannot restore the cursor. The writer must poison itself so
    // later add_document calls fail fast rather than silently appending on
    // top of a half-written record.
    let mut file = RewindFailingWriter::new(usize::MAX, usize::MAX);
    let mut writer = DocStoreWriter::new(&mut file, false);
    writer
      .add_document(&serde_json::json!({"title": "Rust"}))
      .unwrap();

    // From this point seek calls fail AND the write budget is exhausted.
    let pos_after_first = writer.file.inner.position() as usize;
    writer.file.fail_after = pos_after_first;
    writer.file.written = pos_after_first;
    writer.file.fail_seek_from = writer.file.seek_calls + 1; // allow the stream_position seek
                                                             // Actually: stream_position() counts as one seek. Let the next seek fail.
                                                             // `stream_position` inside add_document increments seek_calls by 1, so we
                                                             // mark the _rewind_ seek (the one immediately after the failed write) as
                                                             // the failing call.

    let err = writer
      .add_document(&serde_json::json!({"title": "Search"}))
      .unwrap_err();
    let chain = format!("{err:#}");
    assert!(
      chain.contains("poisoned"),
      "expected poisoned surface in error chain, got: {chain}"
    );
    assert_eq!(writer.offsets().len(), 1);

    // The next add_document must fail fast with the poisoned error even
    // though the underlying writer has been notionally "healed".
    writer.file.fail_after = usize::MAX;
    writer.file.fail_seek_from = usize::MAX;
    let err = writer
      .add_document(&serde_json::json!({"title": "Retry"}))
      .unwrap_err();
    assert!(
      err.to_string().contains("poisoned"),
      "poisoned writer must refuse further add_document calls, got: {err}"
    );
    assert_eq!(writer.offsets().len(), 1);
  }

  #[test]
  fn rejects_corrupt_length_header() {
    use std::io::Write;

    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();
    // Write a bogus length header that exceeds the limit; no body is needed because
    // the reader should fail before attempting to read the payload.
    {
      let mut file = std::fs::File::create(&path).unwrap();
      let len = (MAX_DOCSTORE_BYTES as u32).saturating_add(1);
      file.write_all(&len.to_le_bytes()).unwrap();
    }
    let mut reader = DocStoreReader::new(std::fs::File::open(&path).unwrap(), vec![0], false);
    let err = reader.get(0).unwrap_err();
    assert!(
      err.to_string().contains("stored document length") && err.to_string().contains("exceeds")
    );
  }
}
