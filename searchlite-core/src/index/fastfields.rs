use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::mem;
use std::path::Path;

use anyhow::{anyhow, Result};

use crate::storage::Storage;
use crate::DocId;

#[derive(Debug, Clone)]
pub enum FastValue {
  I64(i64),
  F64(f64),
  Str(String),
  I64List(Vec<i64>),
  F64List(Vec<f64>),
  StrList(Vec<String>),
  I64Nested { object: usize, values: Vec<i64> },
  F64Nested { object: usize, values: Vec<f64> },
  StrNested { object: usize, values: Vec<String> },
  NestedCount { objects: usize },
  NestedParent { object: usize, parent: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FieldType {
  I64,
  F64,
  Str,
  I64List,
  F64List,
  StrList,
  I64Nested,
  F64Nested,
  StrNested,
  NestedCount,
  NestedParent,
}

impl FieldType {
  fn as_u8(self) -> u8 {
    match self {
      FieldType::I64 => 0,
      FieldType::F64 => 1,
      FieldType::Str => 2,
      FieldType::I64List => 3,
      FieldType::F64List => 4,
      FieldType::StrList => 5,
      FieldType::I64Nested => 6,
      FieldType::F64Nested => 7,
      FieldType::StrNested => 8,
      FieldType::NestedCount => 9,
      FieldType::NestedParent => 10,
    }
  }

  fn from_u8(v: u8) -> Option<Self> {
    match v {
      0 => Some(FieldType::I64),
      1 => Some(FieldType::F64),
      2 => Some(FieldType::Str),
      3 => Some(FieldType::I64List),
      4 => Some(FieldType::F64List),
      5 => Some(FieldType::StrList),
      6 => Some(FieldType::I64Nested),
      7 => Some(FieldType::F64Nested),
      8 => Some(FieldType::StrNested),
      9 => Some(FieldType::NestedCount),
      10 => Some(FieldType::NestedParent),
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

#[derive(Debug, Default)]
struct StrListColumnBuilder {
  dict: Vec<String>,
  dict_index: HashMap<String, u32>,
  values: Vec<Vec<u32>>,
}

impl StrListColumnBuilder {
  fn push(&mut self, doc_id: usize, entries: &[String]) {
    if self.values.len() <= doc_id {
      self.values.resize(doc_id + 1, Vec::new());
    }
    let doc_values = &mut self.values[doc_id];
    doc_values.clear();
    for value in entries {
      let idx = if let Some(&idx) = self.dict_index.get(value) {
        idx
      } else {
        let idx = self.dict.len() as u32;
        self.dict.push(value.to_string());
        self.dict_index.insert(value.to_string(), idx);
        idx
      };
      doc_values.push(idx);
    }
  }
}

#[derive(Debug, Default)]
struct StrNestedColumnBuilder {
  dict: Vec<String>,
  dict_index: HashMap<String, u32>,
  values: Vec<Vec<Vec<u32>>>,
}

impl StrNestedColumnBuilder {
  fn push(&mut self, doc_id: usize, object: usize, entries: &[String]) {
    if self.values.len() <= doc_id {
      self.values.resize(doc_id + 1, Vec::new());
    }
    let doc_entries = &mut self.values[doc_id];
    if doc_entries.len() <= object {
      doc_entries.resize(object + 1, Vec::new());
    }
    let target = &mut doc_entries[object];
    target.clear();
    for value in entries {
      let idx = if let Some(&idx) = self.dict_index.get(value) {
        idx
      } else {
        let idx = self.dict.len() as u32;
        self.dict.push(value.to_string());
        self.dict_index.insert(value.to_string(), idx);
        idx
      };
      target.push(idx);
    }
  }
}

#[derive(Debug)]
enum ColumnBuilder {
  I64(Vec<Option<i64>>),
  I64List(Vec<Vec<i64>>),
  I64Nested(Vec<Vec<Vec<i64>>>),
  F64(Vec<Option<f64>>),
  F64List(Vec<Vec<f64>>),
  F64Nested(Vec<Vec<Vec<f64>>>),
  Str(StrColumnBuilder),
  StrList(StrListColumnBuilder),
  StrNested(StrNestedColumnBuilder),
  NestedCount(Vec<u32>),
  NestedParent(Vec<Vec<u32>>),
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
          ColumnBuilder::I64List(entries) => {
            if entries.len() <= idx {
              entries.resize(idx + 1, Vec::new());
            }
            entries[idx] = vec![v];
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
          ColumnBuilder::F64List(entries) => {
            if entries.len() <= idx {
              entries.resize(idx + 1, Vec::new());
            }
            entries[idx] = vec![v];
          }
          _ => panic!("fast field type mismatch for {}", field),
        }
      }
      FastValue::I64List(values) => {
        let col = self
          .data
          .entry(field.to_string())
          .or_insert_with(|| ColumnBuilder::I64List(Vec::new()));
        match col {
          ColumnBuilder::I64List(entries) => {
            if entries.len() <= idx {
              entries.resize(idx + 1, Vec::new());
            }
            entries[idx] = values;
          }
          ColumnBuilder::I64(existing) => {
            let existing_values = mem::take(existing);
            let mut list_entries: Vec<Vec<i64>> = existing_values
              .into_iter()
              .map(|opt| opt.map(|v| vec![v]).unwrap_or_default())
              .collect();
            if list_entries.len() <= idx {
              list_entries.resize(idx + 1, Vec::new());
            }
            list_entries[idx] = values;
            *col = ColumnBuilder::I64List(list_entries);
          }
          _ => panic!("fast field type mismatch for {}", field),
        }
      }
      FastValue::I64Nested { object, values } => {
        let col = self
          .data
          .entry(field.to_string())
          .or_insert_with(|| ColumnBuilder::I64Nested(Vec::new()));
        match col {
          ColumnBuilder::I64Nested(entries) => {
            if entries.len() <= idx {
              entries.resize(idx + 1, Vec::new());
            }
            let doc_entries = &mut entries[idx];
            if doc_entries.len() <= object {
              doc_entries.resize(object + 1, Vec::new());
            }
            doc_entries[object] = values;
          }
          _ => panic!("fast field type mismatch for {}", field),
        }
      }
      FastValue::F64List(values) => {
        let col = self
          .data
          .entry(field.to_string())
          .or_insert_with(|| ColumnBuilder::F64List(Vec::new()));
        match col {
          ColumnBuilder::F64List(entries) => {
            if entries.len() <= idx {
              entries.resize(idx + 1, Vec::new());
            }
            entries[idx] = values;
          }
          ColumnBuilder::F64(existing) => {
            let existing_values = mem::take(existing);
            let mut list_entries: Vec<Vec<f64>> = existing_values
              .into_iter()
              .map(|opt| opt.map(|v| vec![v]).unwrap_or_default())
              .collect();
            if list_entries.len() <= idx {
              list_entries.resize(idx + 1, Vec::new());
            }
            list_entries[idx] = values;
            *col = ColumnBuilder::F64List(list_entries);
          }
          _ => panic!("fast field type mismatch for {}", field),
        }
      }
      FastValue::F64Nested { object, values } => {
        let col = self
          .data
          .entry(field.to_string())
          .or_insert_with(|| ColumnBuilder::F64Nested(Vec::new()));
        match col {
          ColumnBuilder::F64Nested(entries) => {
            if entries.len() <= idx {
              entries.resize(idx + 1, Vec::new());
            }
            let doc_entries = &mut entries[idx];
            if doc_entries.len() <= object {
              doc_entries.resize(object + 1, Vec::new());
            }
            doc_entries[object] = values;
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
          ColumnBuilder::StrList(builder) => {
            let single = [v];
            builder.push(idx, &single);
          }
          _ => panic!("fast field type mismatch for {}", field),
        }
      }
      FastValue::StrList(values) => {
        let col = self
          .data
          .entry(field.to_string())
          .or_insert_with(|| ColumnBuilder::StrList(StrListColumnBuilder::default()));
        match col {
          ColumnBuilder::StrList(builder) => builder.push(idx, &values),
          ColumnBuilder::Str(existing) => {
            let dict = mem::take(&mut existing.dict);
            let dict_index = mem::take(&mut existing.dict_index);
            let existing_values = mem::take(&mut existing.values);
            let mut list_builder = StrListColumnBuilder {
              dict,
              dict_index,
              values: existing_values
                .into_iter()
                .map(|opt| opt.map(|v| vec![v]).unwrap_or_default())
                .collect(),
            };
            list_builder.push(idx, &values);
            *col = ColumnBuilder::StrList(list_builder);
          }
          _ => panic!("fast field type mismatch for {}", field),
        }
      }
      FastValue::StrNested { object, values } => {
        let col = self
          .data
          .entry(field.to_string())
          .or_insert_with(|| ColumnBuilder::StrNested(StrNestedColumnBuilder::default()));
        match col {
          ColumnBuilder::StrNested(builder) => builder.push(idx, object, &values),
          _ => panic!("fast field type mismatch for {}", field),
        }
      }
      FastValue::NestedCount { objects } => {
        let col = self
          .data
          .entry(field.to_string())
          .or_insert_with(|| ColumnBuilder::NestedCount(Vec::new()));
        match col {
          ColumnBuilder::NestedCount(counts) => {
            if counts.len() <= idx {
              counts.resize(idx + 1, 0);
            }
            counts[idx] = objects as u32;
          }
          _ => panic!("fast field type mismatch for {}", field),
        }
      }
      FastValue::NestedParent { object, parent } => {
        let col = self
          .data
          .entry(field.to_string())
          .or_insert_with(|| ColumnBuilder::NestedParent(Vec::new()));
        match col {
          ColumnBuilder::NestedParent(entries) => {
            if entries.len() <= idx {
              entries.resize(idx + 1, Vec::new());
            }
            let doc_entries = &mut entries[idx];
            if doc_entries.len() <= object {
              doc_entries.resize(object + 1, u32::MAX);
            }
            doc_entries[object] = parent as u32;
          }
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
  I64List {
    offsets: Vec<u32>,
    values: Vec<i64>,
  },
  I64Nested {
    doc_offsets: Vec<u32>,
    object_offsets: Vec<u32>,
    values: Vec<i64>,
  },
  F64(Vec<Option<f64>>),
  F64List {
    offsets: Vec<u32>,
    values: Vec<f64>,
  },
  F64Nested {
    doc_offsets: Vec<u32>,
    object_offsets: Vec<u32>,
    values: Vec<f64>,
  },
  Str {
    dict: Vec<String>,
    values: Vec<Option<u32>>,
  },
  StrList {
    dict: Vec<String>,
    offsets: Vec<u32>,
    values: Vec<u32>,
  },
  StrNested {
    dict: Vec<String>,
    doc_offsets: Vec<u32>,
    object_offsets: Vec<u32>,
    values: Vec<u32>,
  },
  NestedCount(Vec<u32>),
  NestedParent {
    offsets: Vec<u32>,
    parents: Vec<u32>,
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
      Some(Column::StrList {
        dict,
        offsets,
        values,
      }) => {
        if let Some((start, end)) = doc_range(offsets, doc_id as usize) {
          values[start..end]
            .iter()
            .any(|idx| dict.get(*idx as usize).map(|s| s == value).unwrap_or(false))
        } else {
          false
        }
      }
      Some(Column::StrNested {
        dict,
        doc_offsets,
        object_offsets,
        values,
      }) => {
        if let Some((obj_start, obj_end)) = doc_range(doc_offsets, doc_id as usize) {
          for obj_idx in obj_start..obj_end {
            if let Some((start, end)) = object_range(object_offsets, obj_idx) {
              if values[start..end]
                .iter()
                .any(|idx| dict.get(*idx as usize).map(|s| s == value).unwrap_or(false))
              {
                return true;
              }
            }
          }
        }
        false
      }
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
      Some(Column::StrList {
        dict,
        offsets,
        values: lookup,
      }) => {
        if let Some((start, end)) = doc_range(offsets, doc_id as usize) {
          lookup[start..end].iter().any(|idx| {
            dict
              .get(*idx as usize)
              .map(|s| values.iter().any(|v| v == s))
              .unwrap_or(false)
          })
        } else {
          false
        }
      }
      Some(Column::StrNested {
        dict,
        doc_offsets,
        object_offsets,
        values: lookup,
      }) => {
        if let Some((obj_start, obj_end)) = doc_range(doc_offsets, doc_id as usize) {
          for obj_idx in obj_start..obj_end {
            if let Some((start, end)) = object_range(object_offsets, obj_idx) {
              if lookup[start..end].iter().any(|idx| {
                dict
                  .get(*idx as usize)
                  .map(|s| values.iter().any(|v| v == s))
                  .unwrap_or(false)
              }) {
                return true;
              }
            }
          }
        }
        false
      }
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
      Some(Column::I64List { offsets, values }) => {
        if let Some((start, end)) = doc_range(offsets, doc_id as usize) {
          values[start..end].iter().any(|v| *v >= min && *v <= max)
        } else {
          false
        }
      }
      Some(Column::I64Nested {
        doc_offsets,
        object_offsets,
        values,
      }) => {
        if let Some((obj_start, obj_end)) = doc_range(doc_offsets, doc_id as usize) {
          for obj_idx in obj_start..obj_end {
            if let Some((start, end)) = object_range(object_offsets, obj_idx) {
              if values[start..end].iter().any(|v| *v >= min && *v <= max) {
                return true;
              }
            }
          }
        }
        false
      }
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
      Some(Column::F64List { offsets, values }) => {
        if let Some((start, end)) = doc_range(offsets, doc_id as usize) {
          values[start..end].iter().any(|v| *v >= min && *v <= max)
        } else {
          false
        }
      }
      Some(Column::F64Nested {
        doc_offsets,
        object_offsets,
        values,
      }) => {
        if let Some((obj_start, obj_end)) = doc_range(doc_offsets, doc_id as usize) {
          for obj_idx in obj_start..obj_end {
            if let Some((start, end)) = object_range(object_offsets, obj_idx) {
              if values[start..end].iter().any(|v| *v >= min && *v <= max) {
                return true;
              }
            }
          }
        }
        false
      }
      _ => false,
    }
  }

  pub fn str_value(&self, field: &str, doc_id: DocId) -> Option<&str> {
    match self.fields.get(field) {
      Some(Column::Str { dict, values }) => values
        .get(doc_id as usize)
        .and_then(|opt| opt.and_then(|idx| dict.get(idx as usize)))
        .map(|s| s.as_str()),
      Some(Column::StrList {
        dict,
        offsets,
        values,
      }) => {
        if let Some((start, _end)) = doc_range(offsets, doc_id as usize) {
          values
            .get(start)
            .and_then(|idx| dict.get(*idx as usize))
            .map(|s| s.as_str())
        } else {
          None
        }
      }
      _ => None,
    }
  }

  pub fn i64_value(&self, field: &str, doc_id: DocId) -> Option<i64> {
    match self.fields.get(field) {
      Some(Column::I64(values)) => values.get(doc_id as usize).and_then(|opt| *opt),
      Some(Column::I64List { offsets, values }) => {
        if let Some((start, _)) = doc_range(offsets, doc_id as usize) {
          values.get(start).copied()
        } else {
          None
        }
      }
      _ => None,
    }
  }

  pub fn f64_value(&self, field: &str, doc_id: DocId) -> Option<f64> {
    match self.fields.get(field) {
      Some(Column::F64(values)) => values.get(doc_id as usize).and_then(|opt| *opt),
      Some(Column::F64List { offsets, values }) => {
        if let Some((start, _)) = doc_range(offsets, doc_id as usize) {
          values.get(start).copied()
        } else {
          None
        }
      }
      _ => None,
    }
  }

  pub fn str_values(&self, field: &str, doc_id: DocId) -> Vec<&str> {
    match self.fields.get(field) {
      Some(Column::Str { dict, values }) => values
        .get(doc_id as usize)
        .and_then(|opt| opt.and_then(|idx| dict.get(idx as usize)))
        .map(|s| vec![s.as_str()])
        .unwrap_or_default(),
      Some(Column::StrList {
        dict,
        offsets,
        values,
      }) => {
        if let Some((start, end)) = doc_range(offsets, doc_id as usize) {
          values[start..end]
            .iter()
            .filter_map(|idx| dict.get(*idx as usize).map(|s| s.as_str()))
            .collect()
        } else {
          Vec::new()
        }
      }
      _ => Vec::new(),
    }
  }

  pub fn i64_values(&self, field: &str, doc_id: DocId) -> Vec<i64> {
    match self.fields.get(field) {
      Some(Column::I64(values)) => values
        .get(doc_id as usize)
        .and_then(|opt| *opt)
        .map(|v| vec![v])
        .unwrap_or_default(),
      Some(Column::I64List { offsets, values }) => {
        if let Some((start, end)) = doc_range(offsets, doc_id as usize) {
          values[start..end].to_vec()
        } else {
          Vec::new()
        }
      }
      _ => Vec::new(),
    }
  }

  pub fn f64_values(&self, field: &str, doc_id: DocId) -> Vec<f64> {
    match self.fields.get(field) {
      Some(Column::F64(values)) => values
        .get(doc_id as usize)
        .and_then(|opt| *opt)
        .map(|v| vec![v])
        .unwrap_or_default(),
      Some(Column::F64List { offsets, values }) => {
        if let Some((start, end)) = doc_range(offsets, doc_id as usize) {
          values[start..end].to_vec()
        } else {
          Vec::new()
        }
      }
      _ => Vec::new(),
    }
  }

  pub fn numeric_values(&self, field: &str, doc_id: DocId) -> Vec<f64> {
    match self.fields.get(field) {
      Some(Column::F64(values)) => values
        .get(doc_id as usize)
        .and_then(|opt| *opt)
        .map(|v| vec![v])
        .unwrap_or_default(),
      Some(Column::F64List { offsets, values }) => {
        if let Some((start, end)) = doc_range(offsets, doc_id as usize) {
          values[start..end].to_vec()
        } else {
          Vec::new()
        }
      }
      Some(Column::I64(values)) => values
        .get(doc_id as usize)
        .and_then(|opt| *opt)
        .map(|v| vec![v as f64])
        .unwrap_or_default(),
      Some(Column::I64List { offsets, values }) => {
        if let Some((start, end)) = doc_range(offsets, doc_id as usize) {
          values[start..end].iter().map(|v| *v as f64).collect()
        } else {
          Vec::new()
        }
      }
      _ => Vec::new(),
    }
  }

  pub fn nested_object_count(&self, path: &str, doc_id: DocId) -> usize {
    let key = nested_count_key(path);
    match self.fields.get(&key) {
      Some(Column::NestedCount(counts)) => {
        counts.get(doc_id as usize).copied().unwrap_or(0) as usize
      }
      _ => 0,
    }
  }

  pub fn nested_str_values(&self, field: &str, doc_id: DocId) -> Vec<Vec<&str>> {
    match self.fields.get(field) {
      Some(Column::StrNested {
        dict,
        doc_offsets,
        object_offsets,
        values,
      }) => {
        if let Some((obj_start, obj_end)) = doc_range(doc_offsets, doc_id as usize) {
          let mut out = Vec::with_capacity(obj_end.saturating_sub(obj_start));
          for obj_idx in obj_start..obj_end {
            if let Some((start, end)) = object_range(object_offsets, obj_idx) {
              let vals = values[start..end]
                .iter()
                .filter_map(|idx| dict.get(*idx as usize).map(|s| s.as_str()))
                .collect();
              out.push(vals);
            }
          }
          out
        } else {
          Vec::new()
        }
      }
      _ => Vec::new(),
    }
  }

  pub fn nested_i64_values(&self, field: &str, doc_id: DocId) -> Vec<Vec<i64>> {
    match self.fields.get(field) {
      Some(Column::I64Nested {
        doc_offsets,
        object_offsets,
        values,
      }) => {
        if let Some((obj_start, obj_end)) = doc_range(doc_offsets, doc_id as usize) {
          let mut out = Vec::with_capacity(obj_end.saturating_sub(obj_start));
          for obj_idx in obj_start..obj_end {
            if let Some((start, end)) = object_range(object_offsets, obj_idx) {
              out.push(values[start..end].to_vec());
            }
          }
          out
        } else {
          Vec::new()
        }
      }
      _ => Vec::new(),
    }
  }

  pub fn nested_f64_values(&self, field: &str, doc_id: DocId) -> Vec<Vec<f64>> {
    match self.fields.get(field) {
      Some(Column::F64Nested {
        doc_offsets,
        object_offsets,
        values,
      }) => {
        if let Some((obj_start, obj_end)) = doc_range(doc_offsets, doc_id as usize) {
          let mut out = Vec::with_capacity(obj_end.saturating_sub(obj_start));
          for obj_idx in obj_start..obj_end {
            if let Some((start, end)) = object_range(object_offsets, obj_idx) {
              out.push(values[start..end].to_vec());
            }
          }
          out
        } else {
          Vec::new()
        }
      }
      _ => Vec::new(),
    }
  }

  pub fn nested_parents(&self, path: &str, doc_id: DocId) -> Vec<Option<usize>> {
    let key = nested_parent_key(path);
    match self.fields.get(&key) {
      Some(Column::NestedParent { offsets, parents }) => {
        if let Some((start, end)) = doc_range(offsets, doc_id as usize) {
          parents[start..end]
            .iter()
            .map(|p| {
              if *p == u32::MAX {
                None
              } else {
                Some(*p as usize)
              }
            })
            .collect()
        } else {
          Vec::new()
        }
      }
      _ => Vec::new(),
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
    ColumnBuilder::I64List(values) => {
      buf.push(FieldType::I64List.as_u8());
      buf.extend_from_slice(&(values.len() as u32).to_le_bytes());
      let mut offsets = Vec::with_capacity(values.len() + 1);
      offsets.push(0);
      for vals in values.iter() {
        let next = *offsets.last().unwrap() + vals.len() as u32;
        offsets.push(next);
      }
      for off in offsets.iter() {
        buf.extend_from_slice(&off.to_le_bytes());
      }
      for vals in values.iter() {
        for v in vals.iter() {
          buf.extend_from_slice(&v.to_le_bytes());
        }
      }
    }
    ColumnBuilder::I64Nested(values) => {
      buf.push(FieldType::I64Nested.as_u8());
      buf.extend_from_slice(&(values.len() as u32).to_le_bytes());
      let mut doc_offsets = Vec::with_capacity(values.len() + 1);
      doc_offsets.push(0);
      let mut object_offsets: Vec<u32> = Vec::new();
      object_offsets.push(0);
      for objects in values.iter() {
        let next = *doc_offsets.last().unwrap() + objects.len() as u32;
        doc_offsets.push(next);
        for vals in objects.iter() {
          let next_obj = *object_offsets.last().unwrap() + vals.len() as u32;
          object_offsets.push(next_obj);
        }
      }
      for off in doc_offsets.iter() {
        buf.extend_from_slice(&off.to_le_bytes());
      }
      for off in object_offsets.iter() {
        buf.extend_from_slice(&off.to_le_bytes());
      }
      for objects in values.iter() {
        for vals in objects.iter() {
          for v in vals.iter() {
            buf.extend_from_slice(&v.to_le_bytes());
          }
        }
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
    ColumnBuilder::F64List(values) => {
      buf.push(FieldType::F64List.as_u8());
      buf.extend_from_slice(&(values.len() as u32).to_le_bytes());
      let mut offsets = Vec::with_capacity(values.len() + 1);
      offsets.push(0);
      for vals in values.iter() {
        let next = *offsets.last().unwrap() + vals.len() as u32;
        offsets.push(next);
      }
      for off in offsets.iter() {
        buf.extend_from_slice(&off.to_le_bytes());
      }
      for vals in values.iter() {
        for v in vals.iter() {
          buf.extend_from_slice(&v.to_le_bytes());
        }
      }
    }
    ColumnBuilder::F64Nested(values) => {
      buf.push(FieldType::F64Nested.as_u8());
      buf.extend_from_slice(&(values.len() as u32).to_le_bytes());
      let mut doc_offsets = Vec::with_capacity(values.len() + 1);
      doc_offsets.push(0);
      let mut object_offsets: Vec<u32> = Vec::new();
      object_offsets.push(0);
      for objects in values.iter() {
        let next = *doc_offsets.last().unwrap() + objects.len() as u32;
        doc_offsets.push(next);
        for vals in objects.iter() {
          let next_obj = *object_offsets.last().unwrap() + vals.len() as u32;
          object_offsets.push(next_obj);
        }
      }
      for off in doc_offsets.iter() {
        buf.extend_from_slice(&off.to_le_bytes());
      }
      for off in object_offsets.iter() {
        buf.extend_from_slice(&off.to_le_bytes());
      }
      for objects in values.iter() {
        for vals in objects.iter() {
          for v in vals.iter() {
            buf.extend_from_slice(&v.to_le_bytes());
          }
        }
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
    ColumnBuilder::StrList(builder) => {
      buf.push(FieldType::StrList.as_u8());
      buf.extend_from_slice(&(builder.values.len() as u32).to_le_bytes());
      let dict_len = builder.dict.len() as u32;
      buf.extend_from_slice(&dict_len.to_le_bytes());
      for entry in builder.dict.iter() {
        let b = entry.as_bytes();
        buf.extend_from_slice(&(b.len() as u32).to_le_bytes());
        buf.extend_from_slice(b);
      }
      let mut offsets = Vec::with_capacity(builder.values.len() + 1);
      offsets.push(0);
      for vals in builder.values.iter() {
        let next = *offsets.last().unwrap() + vals.len() as u32;
        offsets.push(next);
      }
      for off in offsets.iter() {
        buf.extend_from_slice(&off.to_le_bytes());
      }
      for vals in builder.values.iter() {
        for idx in vals.iter() {
          buf.extend_from_slice(&idx.to_le_bytes());
        }
      }
    }
    ColumnBuilder::StrNested(builder) => {
      buf.push(FieldType::StrNested.as_u8());
      buf.extend_from_slice(&(builder.values.len() as u32).to_le_bytes());
      let dict_len = builder.dict.len() as u32;
      buf.extend_from_slice(&dict_len.to_le_bytes());
      for entry in builder.dict.iter() {
        let b = entry.as_bytes();
        buf.extend_from_slice(&(b.len() as u32).to_le_bytes());
        buf.extend_from_slice(b);
      }
      let mut doc_offsets = Vec::with_capacity(builder.values.len() + 1);
      doc_offsets.push(0);
      let mut object_offsets: Vec<u32> = Vec::new();
      object_offsets.push(0);
      for objects in builder.values.iter() {
        let next = *doc_offsets.last().unwrap() + objects.len() as u32;
        doc_offsets.push(next);
        for vals in objects.iter() {
          let next_obj = *object_offsets.last().unwrap() + vals.len() as u32;
          object_offsets.push(next_obj);
        }
      }
      for off in doc_offsets.iter() {
        buf.extend_from_slice(&off.to_le_bytes());
      }
      for off in object_offsets.iter() {
        buf.extend_from_slice(&off.to_le_bytes());
      }
      for objects in builder.values.iter() {
        for vals in objects.iter() {
          for idx in vals.iter() {
            buf.extend_from_slice(&idx.to_le_bytes());
          }
        }
      }
    }
    ColumnBuilder::NestedCount(counts) => {
      buf.push(FieldType::NestedCount.as_u8());
      buf.extend_from_slice(&(counts.len() as u32).to_le_bytes());
      for count in counts.iter() {
        buf.extend_from_slice(&count.to_le_bytes());
      }
    }
    ColumnBuilder::NestedParent(values) => {
      buf.push(FieldType::NestedParent.as_u8());
      buf.extend_from_slice(&(values.len() as u32).to_le_bytes());
      let mut offsets = Vec::with_capacity(values.len() + 1);
      offsets.push(0);
      for parents in values.iter() {
        let next = *offsets.last().unwrap() + parents.len() as u32;
        offsets.push(next);
      }
      for off in offsets.iter() {
        buf.extend_from_slice(&off.to_le_bytes());
      }
      for parents in values.iter() {
        for p in parents.iter() {
          buf.extend_from_slice(&p.to_le_bytes());
        }
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

fn doc_range(offsets: &[u32], doc: usize) -> Option<(usize, usize)> {
  if offsets.len() < doc + 2 {
    return None;
  }
  let start = offsets[doc] as usize;
  let end = offsets[doc + 1] as usize;
  Some((start, end))
}

fn object_range(offsets: &[u32], object_idx: usize) -> Option<(usize, usize)> {
  if offsets.len() < object_idx + 2 {
    return None;
  }
  let start = offsets[object_idx] as usize;
  let end = offsets[object_idx + 1] as usize;
  Some((start, end))
}

pub fn nested_count_key(path: &str) -> String {
  format!("_nested_count:{path}")
}

pub fn nested_parent_key(path: &str) -> String {
  format!("_nested_parent:{path}")
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
      FieldType::I64List => {
        let mut offsets = Vec::with_capacity(doc_len + 1);
        for _ in 0..doc_len + 1 {
          offsets.push(read_u32(&mut cursor, data)?);
        }
        let total_vals = *offsets.last().unwrap_or(&0) as usize;
        let mut vals = Vec::with_capacity(total_vals);
        for _ in 0..total_vals {
          if cursor + 8 > data.len() {
            return Err(anyhow!("unexpected end of fast field i64 list"));
          }
          let mut arr = [0u8; 8];
          arr.copy_from_slice(&data[cursor..cursor + 8]);
          cursor += 8;
          vals.push(i64::from_le_bytes(arr));
        }
        Column::I64List {
          offsets,
          values: vals,
        }
      }
      FieldType::I64Nested => {
        let mut doc_offsets = Vec::with_capacity(doc_len + 1);
        for _ in 0..doc_len + 1 {
          doc_offsets.push(read_u32(&mut cursor, data)?);
        }
        let total_objects = *doc_offsets.last().unwrap_or(&0) as usize;
        let mut object_offsets = Vec::with_capacity(total_objects + 1);
        for _ in 0..total_objects + 1 {
          object_offsets.push(read_u32(&mut cursor, data)?);
        }
        let total_vals = *object_offsets.last().unwrap_or(&0) as usize;
        let mut vals = Vec::with_capacity(total_vals);
        for _ in 0..total_vals {
          if cursor + 8 > data.len() {
            return Err(anyhow!("unexpected end of fast field nested i64"));
          }
          let mut arr = [0u8; 8];
          arr.copy_from_slice(&data[cursor..cursor + 8]);
          cursor += 8;
          vals.push(i64::from_le_bytes(arr));
        }
        Column::I64Nested {
          doc_offsets,
          object_offsets,
          values: vals,
        }
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
      FieldType::F64List => {
        let mut offsets = Vec::with_capacity(doc_len + 1);
        for _ in 0..doc_len + 1 {
          offsets.push(read_u32(&mut cursor, data)?);
        }
        let total_vals = *offsets.last().unwrap_or(&0) as usize;
        let mut vals = Vec::with_capacity(total_vals);
        for _ in 0..total_vals {
          if cursor + 8 > data.len() {
            return Err(anyhow!("unexpected end of fast field f64 list"));
          }
          let mut arr = [0u8; 8];
          arr.copy_from_slice(&data[cursor..cursor + 8]);
          cursor += 8;
          vals.push(f64::from_le_bytes(arr));
        }
        Column::F64List {
          offsets,
          values: vals,
        }
      }
      FieldType::F64Nested => {
        let mut doc_offsets = Vec::with_capacity(doc_len + 1);
        for _ in 0..doc_len + 1 {
          doc_offsets.push(read_u32(&mut cursor, data)?);
        }
        let total_objects = *doc_offsets.last().unwrap_or(&0) as usize;
        let mut object_offsets = Vec::with_capacity(total_objects + 1);
        for _ in 0..total_objects + 1 {
          object_offsets.push(read_u32(&mut cursor, data)?);
        }
        let total_vals = *object_offsets.last().unwrap_or(&0) as usize;
        let mut vals = Vec::with_capacity(total_vals);
        for _ in 0..total_vals {
          if cursor + 8 > data.len() {
            return Err(anyhow!("unexpected end of fast field nested f64"));
          }
          let mut arr = [0u8; 8];
          arr.copy_from_slice(&data[cursor..cursor + 8]);
          cursor += 8;
          vals.push(f64::from_le_bytes(arr));
        }
        Column::F64Nested {
          doc_offsets,
          object_offsets,
          values: vals,
        }
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
      FieldType::StrList => {
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
        let mut offsets = Vec::with_capacity(doc_len + 1);
        for _ in 0..doc_len + 1 {
          offsets.push(read_u32(&mut cursor, data)?);
        }
        let total_vals = *offsets.last().unwrap_or(&0) as usize;
        let mut vals = Vec::with_capacity(total_vals);
        for _ in 0..total_vals {
          let idx = read_u32(&mut cursor, data)?;
          if idx as usize >= dict.len() {
            return Err(anyhow!("invalid fast field dict index"));
          }
          vals.push(idx);
        }
        Column::StrList {
          dict,
          offsets,
          values: vals,
        }
      }
      FieldType::StrNested => {
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
        let mut doc_offsets = Vec::with_capacity(doc_len + 1);
        for _ in 0..doc_len + 1 {
          doc_offsets.push(read_u32(&mut cursor, data)?);
        }
        let total_objects = *doc_offsets.last().unwrap_or(&0) as usize;
        let mut object_offsets = Vec::with_capacity(total_objects + 1);
        for _ in 0..total_objects + 1 {
          object_offsets.push(read_u32(&mut cursor, data)?);
        }
        let total_vals = *object_offsets.last().unwrap_or(&0) as usize;
        let mut vals = Vec::with_capacity(total_vals);
        for _ in 0..total_vals {
          let idx = read_u32(&mut cursor, data)?;
          if idx as usize >= dict.len() {
            return Err(anyhow!("invalid fast field dict index"));
          }
          vals.push(idx);
        }
        Column::StrNested {
          dict,
          doc_offsets,
          object_offsets,
          values: vals,
        }
      }
      FieldType::NestedCount => {
        let mut counts = Vec::with_capacity(doc_len);
        for _ in 0..doc_len {
          counts.push(read_u32(&mut cursor, data)?);
        }
        Column::NestedCount(counts)
      }
      FieldType::NestedParent => {
        let mut offsets = Vec::with_capacity(doc_len + 1);
        for _ in 0..doc_len + 1 {
          offsets.push(read_u32(&mut cursor, data)?);
        }
        let total = *offsets.last().unwrap_or(&0) as usize;
        let mut parents = Vec::with_capacity(total);
        for _ in 0..total {
          parents.push(read_u32(&mut cursor, data)?);
        }
        Column::NestedParent { offsets, parents }
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
    writer.set(
      "tags",
      0,
      FastValue::StrList(vec!["news".into(), "tech".into()]),
    );
    writer.set("year", 0, FastValue::I64(2024));
    writer.set("years", 0, FastValue::I64List(vec![2022, 2024]));
    writer.set("score", 0, FastValue::F64(0.42));
    writer.set("scores", 0, FastValue::F64List(vec![0.1, 0.42]));
    writer.set(
      &nested_count_key("comment"),
      0,
      FastValue::NestedCount { objects: 2 },
    );
    writer.set(
      &nested_parent_key("comment"),
      0,
      FastValue::NestedParent {
        object: 0,
        parent: u32::MAX as usize,
      },
    );
    writer.set(
      &nested_parent_key("comment"),
      0,
      FastValue::NestedParent {
        object: 1,
        parent: u32::MAX as usize,
      },
    );
    writer.set(
      &nested_count_key("comment.reply"),
      0,
      FastValue::NestedCount { objects: 1 },
    );
    writer.set(
      "comment.author",
      0,
      FastValue::StrNested {
        object: 0,
        values: vec!["alice".into()],
      },
    );
    writer.set(
      &nested_parent_key("comment.reply"),
      0,
      FastValue::NestedParent {
        object: 0,
        parent: 0,
      },
    );
    writer.set(
      "comment.author",
      0,
      FastValue::StrNested {
        object: 1,
        values: vec!["bob".into()],
      },
    );
    writer.set(
      "comment.score",
      0,
      FastValue::I64Nested {
        object: 0,
        values: vec![10],
      },
    );
    writer.write_to(&storage, &path).unwrap();

    let reader = FastFieldsReader::open(&storage, &path).unwrap();
    assert!(reader.matches_keyword("tag", 0, "news"));
    assert!(reader.matches_keyword("tags", 0, "tech"));
    assert!(reader.matches_keyword_in("tag", 0, &["sports".into(), "news".into()]));
    assert!(reader.matches_keyword_in("tags", 0, &["sports".into(), "tech".into()]));
    assert!(reader.matches_i64_range("year", 0, 2020, 2025));
    assert!(reader.matches_i64_range("years", 0, 2020, 2023));
    assert!(reader.matches_f64_range("score", 0, 0.0, 1.0));
    assert!(reader.matches_f64_range("scores", 0, 0.4, 0.5));
    assert!(!reader.matches_keyword("tag", 1, "news"));
    assert_eq!(reader.str_values("tags", 0).len(), 2);
    assert_eq!(reader.i64_values("years", 0), vec![2022, 2024]);
    assert_eq!(reader.f64_values("scores", 0), vec![0.1, 0.42]);
    assert_eq!(reader.numeric_values("year", 0), vec![2024.0]);
    assert_eq!(reader.numeric_values("years", 0), vec![2022.0, 2024.0]);
    assert_eq!(reader.numeric_values("score", 0), vec![0.42]);
    assert_eq!(reader.numeric_values("scores", 0), vec![0.1, 0.42]);
    assert_eq!(reader.nested_object_count("comment", 0), 2);
    let nested = reader.nested_str_values("comment.author", 0);
    assert_eq!(nested.len(), 2);
    assert!(nested[0].contains(&"alice"));
    assert!(nested[1].contains(&"bob"));
    let nested_nums = reader.nested_i64_values("comment.score", 0);
    assert_eq!(nested_nums[0], vec![10]);
    let parents = reader.nested_parents("comment", 0);
    assert_eq!(parents.len(), 2);
    assert!(parents.iter().all(|p| p.is_none()));
    let reply_parents = reader.nested_parents("comment.reply", 0);
    assert_eq!(reply_parents, vec![Some(0)]);
  }
}
