use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

use anyhow::Result;

use crate::DocId;

pub struct DocStoreWriter<'a> {
  file: &'a mut File,
  offsets: Vec<u64>,
  #[cfg_attr(not(feature = "zstd"), allow(dead_code))]
  use_zstd: bool,
}

impl<'a> DocStoreWriter<'a> {
  pub fn new(file: &'a mut File, use_zstd: bool) -> Self {
    Self {
      file,
      offsets: Vec::new(),
      use_zstd,
    }
  }

  pub fn add_document(&mut self, doc: &serde_json::Value) -> Result<()> {
    let offset = self.file.stream_position()?;
    self.offsets.push(offset);
    #[allow(unused_mut)]
    let mut data = serde_json::to_vec(doc)?;
    #[cfg(feature = "zstd")]
    if self.use_zstd {
      data = zstd::stream::encode_all(&data[..], 0)?;
    }
    let len = data.len() as u32;
    self.file.write_all(&len.to_le_bytes())?;
    self.file.write_all(&data)?;
    Ok(())
  }

  pub fn offsets(&self) -> &[u64] {
    &self.offsets
  }
}

pub struct DocStoreReader {
  file: File,
  offsets: Vec<u64>,
  #[cfg_attr(not(feature = "zstd"), allow(dead_code))]
  use_zstd: bool,
}

impl DocStoreReader {
  pub fn new(file: File, offsets: Vec<u64>, use_zstd: bool) -> Self {
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
    let mut buf = vec![0u8; len];
    self.file.read_exact(&mut buf)?;
    #[cfg(feature = "zstd")]
    let buf = if self.use_zstd {
      zstd::stream::decode_all(&buf[..])?
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
    let reader_file = File::open(path).unwrap();
    let mut reader = DocStoreReader::new(reader_file, offsets, false);
    let first = reader.get(0).unwrap();
    assert_eq!(first["title"], "Rust");
    assert!(reader.get(2).is_err());
  }
}
