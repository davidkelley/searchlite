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
    self.offsets.push(offset);
    let len = data.len() as u32;
    self.file.write_all(&len.to_le_bytes())?;
    self.file.write_all(&data)?;
    Ok(())
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
      bail!(
        "stored document length {} exceeds maximum {}",
        len,
        MAX_DOCSTORE_BYTES
      );
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
