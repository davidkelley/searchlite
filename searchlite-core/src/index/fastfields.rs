use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{anyhow, Result};

use crate::storage::Storage;
use crate::DocId;

#[derive(Debug, Clone)]
pub enum FastValue {
  I64(i64),
  F64(f64),
  Str(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FieldType {
  I64,
  F64,
  Str,
}

impl FieldType {
  fn as_u8(self) -> u8 {
    match self {
      FieldType::I64 => 0,
      FieldType::F64 => 1,
      FieldType::Str => 2,
    }
  }

  fn from_u8(v: u8) -> Option<Self> {
    match v {
      0 => Some(FieldType::I64),
      1 => Some(FieldType::F64),
      2 => Some(FieldType::Str),
      _ => None,
    }
  }
}

#[derive(Debug, Default)]
struct StrColumnBuilder {
  dict: Vec<String>,
  dict_index: HashMap<String, u32>,
  values: Vec<Option<u32>>,
}

impl StrColumnBuilder {
  fn push(&mut self, doc_id: usize, value: &str) {
    if self.values.len() <= doc_id {
      self.values.resize(doc_id + 1, None);
    }
    let idx = if let Some(&idx) = self.dict_index.get(value) {
      idx
    } else {
      let idx = self.dict.len() as u32;
      self.dict.push(value.to_string());
      self.dict_index.insert(value.to_string(), idx);
      idx
    };
    self.values[doc_id] = Some(idx);
  }
}

#[derive(Debug)]
enum ColumnBuilder {
  I64(Vec<Option<i64>>),
  F64(Vec<Option<f64>>),
  Str(StrColumnBuilder),
}

pub struct FastFieldsWriter {
  data: HashMap<String, ColumnBuilder>,
}

impl FastFieldsWriter {
  pub fn new() -> Self {
    Self {
      data: HashMap::new(),
    }
  }

  pub fn set(&mut self, field: &str, doc_id: DocId, value: FastValue) {
    let idx = doc_id as usize;
    match value {
      FastValue::I64(v) => {
        let col = self
          .data
          .entry(field.to_string())
          .or_insert_with(|| ColumnBuilder::I64(Vec::new()));
        match col {
          ColumnBuilder::I64(values) => {
            if values.len() <= idx {
              values.resize(idx + 1, None);
            }
            values[idx] = Some(v);
          }
          _ => panic!("fast field type mismatch for {}", field),
        }
      }
      FastValue::F64(v) => {
        let col = self
          .data
          .entry(field.to_string())
          .or_insert_with(|| ColumnBuilder::F64(Vec::new()));
        match col {
          ColumnBuilder::F64(values) => {
            if values.len() <= idx {
              values.resize(idx + 1, None);
            }
            values[idx] = Some(v);
          }
          _ => panic!("fast field type mismatch for {}", field),
        }
      }
      FastValue::Str(v) => {
        let col = self
          .data
          .entry(field.to_string())
          .or_insert_with(|| ColumnBuilder::Str(StrColumnBuilder::default()));
        match col {
          ColumnBuilder::Str(builder) => builder.push(idx, &v),
          _ => panic!("fast field type mismatch for {}", field),
        }
      }
    }
  }

  pub fn write_to(&self, storage: &dyn Storage, path: &Path) -> Result<()> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"FFV1");
    let field_count = self.data.len() as u32;
    buf.extend_from_slice(&field_count.to_le_bytes());
    for (name, col) in self.data.iter() {
      write_field(name, col, &mut buf)?;
    }
    let mut handle = storage.open_write(path)?;
    let mut writer = BufWriter::new(&mut *handle);
    writer.write_all(&buf)?;
    writer.flush()?;
    drop(writer);
    handle.sync_all()?;
    Ok(())
  }
}

#[derive(Debug)]
enum Column {
  I64(Vec<Option<i64>>),
  F64(Vec<Option<f64>>),
  Str {
    dict: Vec<String>,
    values: Vec<Option<u32>>,
  },
}

pub struct FastFieldsReader {
  fields: HashMap<String, Column>,
}

impl FastFieldsReader {
  pub fn open(storage: &dyn Storage, path: &Path) -> Result<Self> {
    let data = storage.read_to_end(path)?;
    let fields = read_fields(&data)?;
    Ok(Self { fields })
  }

  pub fn matches_keyword(&self, field: &str, doc_id: DocId, value: &str) -> bool {
    match self.fields.get(field) {
      Some(Column::Str { dict, values }) => values
        .get(doc_id as usize)
        .and_then(|opt| opt.and_then(|idx| dict.get(idx as usize)))
        .map(|s| s == value)
        .unwrap_or(false),
      _ => false,
    }
  }

  pub fn matches_keyword_in(&self, field: &str, doc_id: DocId, values: &[String]) -> bool {
    match self.fields.get(field) {
      Some(Column::Str {
        dict,
        values: lookup,
      }) => lookup
        .get(doc_id as usize)
        .and_then(|opt| opt.and_then(|idx| dict.get(idx as usize)))
        .map(|s| values.iter().any(|v| v == s))
        .unwrap_or(false),
      _ => false,
    }
  }

  pub fn matches_i64_range(&self, field: &str, doc_id: DocId, min: i64, max: i64) -> bool {
    match self.fields.get(field) {
      Some(Column::I64(values)) => values
        .get(doc_id as usize)
        .and_then(|opt| *opt)
        .map(|v| v >= min && v <= max)
        .unwrap_or(false),
      _ => false,
    }
  }

  pub fn matches_f64_range(&self, field: &str, doc_id: DocId, min: f64, max: f64) -> bool {
    match self.fields.get(field) {
      Some(Column::F64(values)) => values
        .get(doc_id as usize)
        .and_then(|opt| *opt)
        .map(|v| v >= min && v <= max)
        .unwrap_or(false),
      _ => false,
    }
  }
}

fn write_field(name: &str, col: &ColumnBuilder, buf: &mut Vec<u8>) -> Result<()> {
  let name_bytes = name.as_bytes();
  buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
  buf.extend_from_slice(name_bytes);
  match col {
    ColumnBuilder::I64(values) => {
      buf.push(FieldType::I64.as_u8());
      buf.extend_from_slice(&(values.len() as u32).to_le_bytes());
      write_presence(values.iter().map(|v| v.is_some()), buf);
      for v in values {
        buf.extend_from_slice(&v.unwrap_or(0).to_le_bytes());
      }
    }
    ColumnBuilder::F64(values) => {
      buf.push(FieldType::F64.as_u8());
      buf.extend_from_slice(&(values.len() as u32).to_le_bytes());
      write_presence(values.iter().map(|v| v.is_some()), buf);
      for v in values {
        buf.extend_from_slice(&v.unwrap_or(0.0).to_le_bytes());
      }
    }
    ColumnBuilder::Str(builder) => {
      buf.push(FieldType::Str.as_u8());
      buf.extend_from_slice(&(builder.values.len() as u32).to_le_bytes());
      let dict_len = builder.dict.len() as u32;
      buf.extend_from_slice(&dict_len.to_le_bytes());
      for entry in builder.dict.iter() {
        let b = entry.as_bytes();
        buf.extend_from_slice(&(b.len() as u32).to_le_bytes());
        buf.extend_from_slice(b);
      }
      for v in builder.values.iter() {
        let idx = v.map(|i| i).unwrap_or(u32::MAX);
        buf.extend_from_slice(&idx.to_le_bytes());
      }
    }
  }
  Ok(())
}

fn write_presence(iter: impl Iterator<Item = bool>, buf: &mut Vec<u8>) {
  for present in iter {
    buf.push(present as u8);
  }
}

fn read_fields(data: &[u8]) -> Result<HashMap<String, Column>> {
  if data.len() < 8 {
    return Ok(HashMap::new());
  }
  if &data[..4] != b"FFV1" {
    return Err(anyhow!("invalid fast field header"));
  }
  let mut cursor = 4;
  let field_count = read_u32(&mut cursor, data)? as usize;
  let mut fields = HashMap::with_capacity(field_count);
  for _ in 0..field_count {
    let name_len = read_u32(&mut cursor, data)? as usize;
    if cursor + name_len > data.len() {
      return Err(anyhow!("invalid fast field name length"));
    }
    let name = String::from_utf8_lossy(&data[cursor..cursor + name_len]).into_owned();
    cursor += name_len;
    let ty = FieldType::from_u8(read_u8(&mut cursor, data)?)
      .ok_or_else(|| anyhow!("invalid fast field type"))?;
    let doc_len = read_u32(&mut cursor, data)? as usize;
    let column = match ty {
      FieldType::I64 => {
        let presence = read_presence(doc_len, &mut cursor, data)?;
        let mut vals = Vec::with_capacity(doc_len);
        for present in presence.into_iter() {
          if cursor + 8 > data.len() {
            return Err(anyhow!("unexpected end of fast field i64"));
          }
          let mut arr = [0u8; 8];
          arr.copy_from_slice(&data[cursor..cursor + 8]);
          cursor += 8;
          if present {
            vals.push(Some(i64::from_le_bytes(arr)));
          } else {
            vals.push(None);
          }
        }
        Column::I64(vals)
      }
      FieldType::F64 => {
        let presence = read_presence(doc_len, &mut cursor, data)?;
        let mut vals = Vec::with_capacity(doc_len);
        for present in presence.into_iter() {
          if cursor + 8 > data.len() {
            return Err(anyhow!("unexpected end of fast field f64"));
          }
          let mut arr = [0u8; 8];
          arr.copy_from_slice(&data[cursor..cursor + 8]);
          cursor += 8;
          if present {
            vals.push(Some(f64::from_le_bytes(arr)));
          } else {
            vals.push(None);
          }
        }
        Column::F64(vals)
      }
      FieldType::Str => {
        let dict_len = read_u32(&mut cursor, data)? as usize;
        let mut dict = Vec::with_capacity(dict_len);
        for _ in 0..dict_len {
          let slen = read_u32(&mut cursor, data)? as usize;
          if cursor + slen > data.len() {
            return Err(anyhow!("unexpected end of fast field dict"));
          }
          let s = String::from_utf8_lossy(&data[cursor..cursor + slen]).into_owned();
          cursor += slen;
          dict.push(s);
        }
        let mut vals = Vec::with_capacity(doc_len);
        for _ in 0..doc_len {
          let idx = read_u32(&mut cursor, data)?;
          if idx == u32::MAX {
            vals.push(None);
          } else {
            if idx as usize >= dict.len() {
              return Err(anyhow!("invalid fast field dict index"));
            }
            vals.push(Some(idx));
          }
        }
        Column::Str { dict, values: vals }
      }
    };
    fields.insert(name, column);
  }
  Ok(fields)
}

fn read_u32(cursor: &mut usize, buf: &[u8]) -> Result<u32> {
  if *cursor + 4 > buf.len() {
    return Err(anyhow!("unexpected end of buffer"));
  }
  let mut arr = [0u8; 4];
  arr.copy_from_slice(&buf[*cursor..*cursor + 4]);
  *cursor += 4;
  Ok(u32::from_le_bytes(arr))
}

fn read_u8(cursor: &mut usize, buf: &[u8]) -> Result<u8> {
  if *cursor >= buf.len() {
    return Err(anyhow!("unexpected end of buffer"));
  }
  let b = buf[*cursor];
  *cursor += 1;
  Ok(b)
}

fn read_presence(len: usize, cursor: &mut usize, buf: &[u8]) -> Result<Vec<bool>> {
  if *cursor + len > buf.len() {
    return Err(anyhow!("unexpected end of presence data"));
  }
  let slice = &buf[*cursor..*cursor + len];
  *cursor += len;
  Ok(slice.iter().map(|b| *b != 0).collect())
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::tempdir;

  #[test]
  fn writes_and_reads_fast_fields() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("fast.json");
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    let mut writer = FastFieldsWriter::new();
    writer.set("tag", 0, FastValue::Str("news".into()));
    writer.set("year", 0, FastValue::I64(2024));
    writer.set("score", 0, FastValue::F64(0.42));
    writer.write_to(&storage, &path).unwrap();

    let reader = FastFieldsReader::open(&storage, &path).unwrap();
    assert!(reader.matches_keyword("tag", 0, "news"));
    assert!(reader.matches_keyword_in("tag", 0, &vec!["sports".into(), "news".into()]));
    assert!(reader.matches_i64_range("year", 0, 2020, 2025));
    assert!(reader.matches_f64_range("score", 0, 0.0, 1.0));
    assert!(!reader.matches_keyword("tag", 1, "news"));
  }
}
