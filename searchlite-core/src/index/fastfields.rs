use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::DocId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FastValue {
  I64(i64),
  F64(f64),
  Str(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FastFieldFile {
  pub fields: HashMap<String, Vec<FastValue>>,
}

pub struct FastFieldsWriter {
  data: FastFieldFile,
}

impl FastFieldsWriter {
  pub fn new() -> Self {
    Self {
      data: FastFieldFile::default(),
    }
  }

  pub fn set(&mut self, field: &str, doc_id: DocId, value: FastValue) {
    let col = self.data.fields.entry(field.to_string()).or_default();
    if col.len() <= doc_id as usize {
      col.resize(doc_id as usize + 1, FastValue::Str(String::new()));
    }
    col[doc_id as usize] = value;
  }

  pub fn write_to(&self, path: &Path) -> Result<()> {
    let writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(writer, &self.data)?;
    Ok(())
  }
}

pub struct FastFieldsReader {
  data: FastFieldFile,
}

impl FastFieldsReader {
  pub fn open(path: &Path) -> Result<Self> {
    let reader = BufReader::new(File::open(path)?);
    let data: FastFieldFile = serde_json::from_reader(reader)?;
    Ok(Self { data })
  }

  pub fn matches_keyword(&self, field: &str, doc_id: DocId, value: &str) -> bool {
    if let Some(values) = self.data.fields.get(field) {
      if let Some(FastValue::Str(s)) = values.get(doc_id as usize) {
        return s == value;
      }
    }
    false
  }

  pub fn matches_keyword_in(&self, field: &str, doc_id: DocId, values: &[String]) -> bool {
    if let Some(values_col) = self.data.fields.get(field) {
      if let Some(FastValue::Str(s)) = values_col.get(doc_id as usize) {
        return values.iter().any(|v| v == s);
      }
    }
    false
  }

  pub fn matches_i64_range(&self, field: &str, doc_id: DocId, min: i64, max: i64) -> bool {
    if let Some(values) = self.data.fields.get(field) {
      if let Some(FastValue::I64(v)) = values.get(doc_id as usize) {
        return *v >= min && *v <= max;
      }
    }
    false
  }

  pub fn matches_f64_range(&self, field: &str, doc_id: DocId, min: f64, max: f64) -> bool {
    if let Some(values) = self.data.fields.get(field) {
      if let Some(FastValue::F64(v)) = values.get(doc_id as usize) {
        return *v >= min && *v <= max;
      }
    }
    false
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::tempdir;

  #[test]
  fn writes_and_reads_fast_fields() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("fast.json");
    let mut writer = FastFieldsWriter::new();
    writer.set("tag", 0, FastValue::Str("news".into()));
    writer.set("year", 0, FastValue::I64(2024));
    writer.set("score", 0, FastValue::F64(0.42));
    writer.write_to(&path).unwrap();

    let reader = FastFieldsReader::open(&path).unwrap();
    assert!(reader.matches_keyword("tag", 0, "news"));
    assert!(reader.matches_keyword_in("tag", 0, &vec!["sports".into(), "news".into()]));
    assert!(reader.matches_i64_range("year", 0, 2020, 2025));
    assert!(reader.matches_f64_range("score", 0, 0.0, 1.0));
    assert!(!reader.matches_keyword("tag", 1, "news"));
  }
}
