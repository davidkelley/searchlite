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
}

impl<'a, W: Write + Seek + ?Sized> DocStoreWriter<'a, W> {
  pub fn new(file: &'a mut W, use_zstd: bool) -> Self {
    Self {
      file,
      offsets: Vec::new(),
      use_zstd,
    }
  }

  pub fn add_document(&mut self, doc: &serde_json::Value) -> Result<()> {
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
    // desynchronise doc ids from offsets on the next write.
    match self
      .file
      .write_all(&len.to_le_bytes())
      .and_then(|_| self.file.write_all(&data))
    {
      Ok(()) => {
        self.offsets.push(offset);
        Ok(())
      }
      Err(e) => {
        // Best-effort: put the cursor back where it was so the partial record
        // is overwritten on the next write. If this rewind itself fails the
        // writer is abandoned and the original IO error is what callers see.
        let _ = self.file.seek(SeekFrom::Start(offset));
        Err(e.into())
      }
    }
  }

  pub fn offsets(&self) -> &[u64] {
    &self.offsets
  }
}

pub struct DocStoreReader<R: Read + Seek> {
  file: R,
  offsets: Vec<u64>,
  #[cfg_attr(not(feature = "zstd"), allow(dead_code))]
  use_zstd: bool,
}

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
    #[cfg(feature = "zstd")]
    let buf = if self.use_zstd {
      let decoded = zstd::stream::decode_all(&buf[..])?;
      if decoded.len() > MAX_DOCSTORE_BYTES {
        bail!(
          "stored document length {} exceeds maximum {} after decompression",
          decoded.len(),
          MAX_DOCSTORE_BYTES
        );
      }
      decoded
    } else {
      buf
    };
    #[cfg(not(feature = "zstd"))]
    let buf = buf;
    let json: serde_json::Value = serde_json::from_slice(&buf)?;
    Ok(json)
  }
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

  /// Write + Seek shim that fails the Nth byte of output, used to simulate a
  /// mid-record IO failure during `add_document`.
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
      if n < buf.len() {
        // Part of the buffer was written; the next call will observe the
        // injected failure. Return what we managed to flush so `write_all`
        // loops and eventually surfaces the error, matching how a real
        // half-failing IO device behaves.
        Ok(n)
      } else {
        Ok(n)
      }
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
    // The first document is small; the second will be forced to fail mid-write
    // by the injected writer. After the failure the offsets vector must still
    // describe only the first successful document so the caller can retry the
    // second add_document with the original doc_id.
    // Build the writer with a healthy cap; the second add_document will be
    // forced to fail by narrowing the cap below the record length.
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
    assert!(
      err.to_string().to_lowercase().contains("pipe") || err.to_string().contains("injected")
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

    // Retry with a healthy cap and confirm doc_id 1 is assigned.
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
