use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::mem;
use std::path::Path;

use anyhow::{anyhow, Context, Result};

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
          _ => panic!("fast field type mismatch for {field}"),
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
          _ => panic!("fast field type mismatch for {field}"),
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
          _ => panic!("fast field type mismatch for {field}"),
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
          _ => panic!("fast field type mismatch for {field}"),
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
          _ => panic!("fast field type mismatch for {field}"),
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
          _ => panic!("fast field type mismatch for {field}"),
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
          _ => panic!("fast field type mismatch for {field}"),
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
          _ => panic!("fast field type mismatch for {field}"),
        }
      }
      FastValue::StrNested { object, values } => {
        let col = self
          .data
          .entry(field.to_string())
          .or_insert_with(|| ColumnBuilder::StrNested(StrNestedColumnBuilder::default()));
        match col {
          ColumnBuilder::StrNested(builder) => builder.push(idx, object, &values),
          _ => panic!("fast field type mismatch for {field}"),
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
          _ => panic!("fast field type mismatch for {field}"),
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
          _ => panic!("fast field type mismatch for {field}"),
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

pub(crate) fn case_insensitive_equals(a: &str, b: &str) -> bool {
  if a.is_ascii() && b.is_ascii() {
    a.eq_ignore_ascii_case(b)
  } else {
    // Compare char-by-char via the Unicode lowercase mapping without
    // allocating two temporary Strings. Each `char::to_lowercase()` yields
    // an iterator of 1–3 chars; we flatten both sides and compare element-wise.
    let mut a_lower = a.chars().flat_map(|c| c.to_lowercase());
    let mut b_lower = b.chars().flat_map(|c| c.to_lowercase());
    loop {
      match (a_lower.next(), b_lower.next()) {
        (Some(ac), Some(bc)) if ac == bc => continue,
        (None, None) => return true,
        _ => return false,
      }
    }
  }
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
        .map(|s| case_insensitive_equals(s, value))
        .unwrap_or(false),
      Some(Column::StrList {
        dict,
        offsets,
        values,
      }) => {
        if let Some((start, end)) = doc_range(offsets, doc_id as usize) {
          values[start..end].iter().any(|idx| {
            dict
              .get(*idx as usize)
              .map(|s| case_insensitive_equals(s, value))
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
        values,
      }) => {
        if let Some((obj_start, obj_end)) = doc_range(doc_offsets, doc_id as usize) {
          for obj_idx in obj_start..obj_end {
            if let Some((start, end)) = object_range(object_offsets, obj_idx) {
              if values[start..end].iter().any(|idx| {
                dict
                  .get(*idx as usize)
                  .map(|s| case_insensitive_equals(s, value))
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

  /// Check if a document's keyword field matches any of the given values.
  ///
  /// Uses zero-allocation `case_insensitive_equals` for each comparison,
  /// which is O(stored_values × filter_values) but avoids all heap
  /// allocations. For typical filter sizes this is faster than building
  /// a `HashSet` (which requires `to_lowercase()` allocations).
  pub fn matches_keyword_in(&self, field: &str, doc_id: DocId, values: &[String]) -> bool {
    match self.fields.get(field) {
      Some(Column::Str {
        dict,
        values: lookup,
      }) => lookup
        .get(doc_id as usize)
        .and_then(|opt| opt.and_then(|idx| dict.get(idx as usize)))
        .map(|s| values.iter().any(|v| case_insensitive_equals(s, v)))
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
              .map(|s| values.iter().any(|v| case_insensitive_equals(s, v)))
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
                  .map(|s| values.iter().any(|v| case_insensitive_equals(s, v)))
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

  pub fn nested_str_values_at<'a>(
    &'a self,
    field: &str,
    doc_id: DocId,
    object_idx: usize,
  ) -> Vec<&'a str> {
    match self.fields.get(field) {
      Some(Column::StrNested {
        dict,
        doc_offsets,
        object_offsets,
        values,
      }) => {
        if let Some((obj_start, obj_end)) = doc_range(doc_offsets, doc_id as usize) {
          if object_idx >= obj_end.saturating_sub(obj_start) {
            return Vec::new();
          }
          let absolute_idx = obj_start.saturating_add(object_idx);
          if let Some((start, end)) = object_range(object_offsets, absolute_idx) {
            values[start..end]
              .iter()
              .filter_map(|idx| dict.get(*idx as usize).map(|s| s.as_str()))
              .collect()
          } else {
            Vec::new()
          }
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
  let start = *offsets.get(doc)? as usize;
  let end = *offsets.get(doc.checked_add(1)?)? as usize;
  if start > end {
    return None;
  }
  Some((start, end))
}

fn object_range(offsets: &[u32], object_idx: usize) -> Option<(usize, usize)> {
  let start = *offsets.get(object_idx)? as usize;
  let end = *offsets.get(object_idx.checked_add(1)?)? as usize;
  if start > end {
    return None;
  }
  Some((start, end))
}

pub fn nested_count_key(path: &str) -> String {
  format!("_nested_count:{path}")
}

pub fn nested_parent_key(path: &str) -> String {
  format!("_nested_parent:{path}")
}

pub fn doc_length_key(field: &str) -> String {
  format!("_len:{field}")
}

fn read_fields(data: &[u8]) -> Result<HashMap<String, Column>> {
  if data.len() < 8 {
    return Ok(HashMap::new());
  }
  if &data[..4] != b"FFV1" {
    return Err(anyhow!("invalid fast field header"));
  }
  let mut cursor = 4;
  // Each field header is at least 9 bytes on disk (4-byte name length +
  // at least 0 name bytes + 1-byte type tag + 4-byte doc_len). Validate the
  // count against the remaining buffer so a crafted `field_count` near
  // `u32::MAX` cannot drive a multi-gigabyte `HashMap::with_capacity` before
  // the first field read even runs.
  let field_count = checked_count(
    read_u32(&mut cursor, data)? as usize,
    9,
    data.len() - cursor,
  )?;
  let mut fields = HashMap::with_capacity(field_count);
  for _ in 0..field_count {
    let name_len = read_u32(&mut cursor, data)? as usize;
    if cursor + name_len > data.len() {
      return Err(anyhow!("invalid fast field name length"));
    }
    // Fast-field names are written from `&str` by `write_field`, so they are
    // valid UTF-8 by construction. Any non-UTF-8 bytes on disk are therefore
    // a corruption signal — surface them as a structured error instead of
    // silently mapping to U+FFFD via `from_utf8_lossy`. Mirrors BUG-010's
    // fix in `terms.rs::read_terms`; see BUG-217.
    let name = std::str::from_utf8(&data[cursor..cursor + name_len])
      .with_context(|| {
        format!("fast-field file contains non-UTF-8 bytes at offset {cursor} (field name)")
      })?
      .to_string();
    cursor += name_len;
    let ty = FieldType::from_u8(read_u8(&mut cursor, data)?)
      .ok_or_else(|| anyhow!("invalid fast field type"))?;
    let doc_len = read_u32(&mut cursor, data)? as usize;
    let column = match ty {
      FieldType::I64 => {
        let presence = read_presence(doc_len, &mut cursor, data)?;
        let vals_count = checked_count(doc_len, 8, data.len() - cursor)?;
        let mut vals = Vec::with_capacity(vals_count);
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
        let offsets_count = checked_count(checked_add_one(doc_len)?, 4, data.len() - cursor)?;
        let mut offsets = Vec::with_capacity(offsets_count);
        for _ in 0..offsets_count {
          offsets.push(read_u32(&mut cursor, data)?);
        }
        validate_monotonic_offsets(&offsets, "I64List")?;
        let total_vals = checked_count(
          *offsets.last().unwrap_or(&0) as usize,
          8,
          data.len() - cursor,
        )?;
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
        let doc_offsets_count = checked_count(checked_add_one(doc_len)?, 4, data.len() - cursor)?;
        let mut doc_offsets = Vec::with_capacity(doc_offsets_count);
        for _ in 0..doc_offsets_count {
          doc_offsets.push(read_u32(&mut cursor, data)?);
        }
        validate_monotonic_offsets(&doc_offsets, "I64Nested doc")?;
        let object_offsets_count = checked_count(
          checked_add_one(*doc_offsets.last().unwrap_or(&0) as usize)?,
          4,
          data.len() - cursor,
        )?;
        let mut object_offsets = Vec::with_capacity(object_offsets_count);
        for _ in 0..object_offsets_count {
          object_offsets.push(read_u32(&mut cursor, data)?);
        }
        validate_monotonic_offsets(&object_offsets, "I64Nested object")?;
        let total_vals = checked_count(
          *object_offsets.last().unwrap_or(&0) as usize,
          8,
          data.len() - cursor,
        )?;
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
        let vals_count = checked_count(doc_len, 8, data.len() - cursor)?;
        let mut vals = Vec::with_capacity(vals_count);
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
        let offsets_count = checked_count(checked_add_one(doc_len)?, 4, data.len() - cursor)?;
        let mut offsets = Vec::with_capacity(offsets_count);
        for _ in 0..offsets_count {
          offsets.push(read_u32(&mut cursor, data)?);
        }
        validate_monotonic_offsets(&offsets, "F64List")?;
        let total_vals = checked_count(
          *offsets.last().unwrap_or(&0) as usize,
          8,
          data.len() - cursor,
        )?;
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
        let doc_offsets_count = checked_count(checked_add_one(doc_len)?, 4, data.len() - cursor)?;
        let mut doc_offsets = Vec::with_capacity(doc_offsets_count);
        for _ in 0..doc_offsets_count {
          doc_offsets.push(read_u32(&mut cursor, data)?);
        }
        validate_monotonic_offsets(&doc_offsets, "F64Nested doc")?;
        let object_offsets_count = checked_count(
          checked_add_one(*doc_offsets.last().unwrap_or(&0) as usize)?,
          4,
          data.len() - cursor,
        )?;
        let mut object_offsets = Vec::with_capacity(object_offsets_count);
        for _ in 0..object_offsets_count {
          object_offsets.push(read_u32(&mut cursor, data)?);
        }
        validate_monotonic_offsets(&object_offsets, "F64Nested object")?;
        let total_vals = checked_count(
          *object_offsets.last().unwrap_or(&0) as usize,
          8,
          data.len() - cursor,
        )?;
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
        // Each dict entry has at least a 4-byte `u32` length prefix on disk,
        // so `dict_len * 4` is a hard lower bound on the serialized size.
        let dict_len = checked_count(
          read_u32(&mut cursor, data)? as usize,
          4,
          data.len() - cursor,
        )?;
        let mut dict = Vec::with_capacity(dict_len);
        for _ in 0..dict_len {
          let slen = read_u32(&mut cursor, data)? as usize;
          if cursor + slen > data.len() {
            return Err(anyhow!("unexpected end of fast field dict"));
          }
          let s = std::str::from_utf8(&data[cursor..cursor + slen])
            .with_context(|| {
              format!(
                "fast-field file contains non-UTF-8 bytes at offset {cursor} (Str dict entry)"
              )
            })?
            .to_string();
          cursor += slen;
          dict.push(s);
        }
        let vals_count = checked_count(doc_len, 4, data.len() - cursor)?;
        let mut vals = Vec::with_capacity(vals_count);
        for _ in 0..vals_count {
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
        let dict_len = checked_count(
          read_u32(&mut cursor, data)? as usize,
          4,
          data.len() - cursor,
        )?;
        let mut dict = Vec::with_capacity(dict_len);
        for _ in 0..dict_len {
          let slen = read_u32(&mut cursor, data)? as usize;
          if cursor + slen > data.len() {
            return Err(anyhow!("unexpected end of fast field dict"));
          }
          let s = std::str::from_utf8(&data[cursor..cursor + slen])
            .with_context(|| {
              format!(
                "fast-field file contains non-UTF-8 bytes at offset {cursor} (StrList dict entry)"
              )
            })?
            .to_string();
          cursor += slen;
          dict.push(s);
        }
        let offsets_count = checked_count(checked_add_one(doc_len)?, 4, data.len() - cursor)?;
        let mut offsets = Vec::with_capacity(offsets_count);
        for _ in 0..offsets_count {
          offsets.push(read_u32(&mut cursor, data)?);
        }
        validate_monotonic_offsets(&offsets, "StrList")?;
        let total_vals = checked_count(
          *offsets.last().unwrap_or(&0) as usize,
          4,
          data.len() - cursor,
        )?;
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
        let dict_len = checked_count(
          read_u32(&mut cursor, data)? as usize,
          4,
          data.len() - cursor,
        )?;
        let mut dict = Vec::with_capacity(dict_len);
        for _ in 0..dict_len {
          let slen = read_u32(&mut cursor, data)? as usize;
          if cursor + slen > data.len() {
            return Err(anyhow!("unexpected end of fast field dict"));
          }
          let s = std::str::from_utf8(&data[cursor..cursor + slen])
            .with_context(|| {
              format!(
                "fast-field file contains non-UTF-8 bytes at offset {cursor} (StrNested dict entry)"
              )
            })?
            .to_string();
          cursor += slen;
          dict.push(s);
        }
        let doc_offsets_count = checked_count(checked_add_one(doc_len)?, 4, data.len() - cursor)?;
        let mut doc_offsets = Vec::with_capacity(doc_offsets_count);
        for _ in 0..doc_offsets_count {
          doc_offsets.push(read_u32(&mut cursor, data)?);
        }
        validate_monotonic_offsets(&doc_offsets, "StrNested doc")?;
        let object_offsets_count = checked_count(
          checked_add_one(*doc_offsets.last().unwrap_or(&0) as usize)?,
          4,
          data.len() - cursor,
        )?;
        let mut object_offsets = Vec::with_capacity(object_offsets_count);
        for _ in 0..object_offsets_count {
          object_offsets.push(read_u32(&mut cursor, data)?);
        }
        validate_monotonic_offsets(&object_offsets, "StrNested object")?;
        let total_vals = checked_count(
          *object_offsets.last().unwrap_or(&0) as usize,
          4,
          data.len() - cursor,
        )?;
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
        let counts_count = checked_count(doc_len, 4, data.len() - cursor)?;
        let mut counts = Vec::with_capacity(counts_count);
        for _ in 0..counts_count {
          counts.push(read_u32(&mut cursor, data)?);
        }
        Column::NestedCount(counts)
      }
      FieldType::NestedParent => {
        let offsets_count = checked_count(checked_add_one(doc_len)?, 4, data.len() - cursor)?;
        let mut offsets = Vec::with_capacity(offsets_count);
        for _ in 0..offsets_count {
          offsets.push(read_u32(&mut cursor, data)?);
        }
        validate_monotonic_offsets(&offsets, "NestedParent")?;
        let total = checked_count(
          *offsets.last().unwrap_or(&0) as usize,
          4,
          data.len() - cursor,
        )?;
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

/// Validate that reading `count` entries, each serialized as at least
/// `min_stride` bytes in the tail of the segment buffer, cannot run past the
/// remaining `remaining` bytes. Returns the unchanged count on success so it
/// can be used verbatim as a `Vec::with_capacity` size and loop bound.
///
/// Every element-count field in a fast-field column is a `u32` read directly
/// from the segment file. A crafted or corrupted file can therefore supply a
/// count approaching `u32::MAX`, whose naive `Vec::with_capacity` would ask
/// for multi-gigabyte allocations (e.g. `4G * 8B = 32 GiB` for an `i64`
/// list) before the per-element loop ever discovered that the file is too
/// short. Bounding the count against the bytes still available keeps the
/// pre-allocation cost proportional to the file itself.
fn checked_count(count: usize, min_stride: usize, remaining: usize) -> Result<usize> {
  let needed = count.checked_mul(min_stride).ok_or_else(|| {
    anyhow!("fast field element count {count} * stride {min_stride} overflows usize")
  })?;
  if needed > remaining {
    return Err(anyhow!(
      "fast field claims {needed} bytes but only {remaining} remain"
    ));
  }
  Ok(count)
}

/// `count + 1`, but returns an error instead of overflowing `usize`. `count`
/// is a `u32` cast to `usize` in practice, so on 64-bit targets the add
/// cannot overflow; on 32-bit targets it can, and a malicious segment with a
/// count of `u32::MAX` would otherwise panic in debug or wrap in release.
fn checked_add_one(count: usize) -> Result<usize> {
  count
    .checked_add(1)
    .ok_or_else(|| anyhow!("fast field element count {count} + 1 overflows usize"))
}

/// Validate that an offset array is monotonically non-decreasing. Corrupt or
/// adversarially crafted fast-field files can contain non-monotonic offsets
/// which would cause `values[start..end]` to panic when `start > end`.
/// Surfacing this at load time prevents silent panics at query time.
fn validate_monotonic_offsets(offsets: &[u32], column_kind: &str) -> Result<()> {
  for window in offsets.windows(2) {
    if window[0] > window[1] {
      return Err(anyhow!(
        "non-monotonic offsets in fast field {column_kind} column: {} > {}",
        window[0],
        window[1]
      ));
    }
  }
  Ok(())
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

  #[test]
  fn nested_str_values_at_stays_within_doc_object_range() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("fast-nested-range.json");
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    let mut writer = FastFieldsWriter::new();

    writer.set(
      "comment.author",
      1,
      FastValue::StrNested {
        object: 0,
        values: vec!["bob".into()],
      },
    );
    writer.write_to(&storage, &path).unwrap();

    let reader = FastFieldsReader::open(&storage, &path).unwrap();
    assert_eq!(
      reader.nested_str_values_at("comment.author", 0, 0),
      Vec::<&str>::new()
    );
    assert_eq!(
      reader.nested_str_values_at("comment.author", 1, 0),
      vec!["bob"]
    );
  }

  // --- Corrupt-segment regression tests (BUG-012) ---
  //
  // Each test below hand-crafts a fast-field buffer whose on-disk counts
  // would drive a multi-gigabyte `Vec::with_capacity` / `HashMap::with_capacity`
  // in the pre-fix code. `read_fields` must reject them before allocating.
  // The buffers are intentionally tiny so a regression would fail on most
  // CI runners via OOM / allocator abort.

  fn header_with(field_count: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(8);
    buf.extend_from_slice(b"FFV1");
    buf.extend_from_slice(&field_count.to_le_bytes());
    buf
  }

  fn field_prefix(buf: &mut Vec<u8>, ty: FieldType, doc_len: u32) {
    buf.extend_from_slice(&0u32.to_le_bytes()); // empty name
    buf.push(ty.as_u8());
    buf.extend_from_slice(&doc_len.to_le_bytes());
  }

  fn assert_capacity_rejected(err: anyhow::Error) {
    let msg = err.to_string();
    assert!(
      msg.contains("fast field claims") || msg.contains("overflows usize"),
      "expected a bounded-allocation error, got: {msg}"
    );
  }

  #[test]
  fn read_fields_rejects_oversized_field_count() {
    let buf = header_with(u32::MAX);
    assert_capacity_rejected(read_fields(&buf).unwrap_err());
  }

  #[test]
  fn read_fields_rejects_i64_list_oversized_doc_len() {
    let mut buf = header_with(1);
    field_prefix(&mut buf, FieldType::I64List, u32::MAX);
    assert_capacity_rejected(read_fields(&buf).unwrap_err());
  }

  #[test]
  fn read_fields_rejects_f64_list_oversized_doc_len() {
    let mut buf = header_with(1);
    field_prefix(&mut buf, FieldType::F64List, u32::MAX);
    assert_capacity_rejected(read_fields(&buf).unwrap_err());
  }

  #[test]
  fn read_fields_rejects_i64_nested_oversized_doc_len() {
    let mut buf = header_with(1);
    field_prefix(&mut buf, FieldType::I64Nested, u32::MAX);
    assert_capacity_rejected(read_fields(&buf).unwrap_err());
  }

  #[test]
  fn read_fields_rejects_f64_nested_oversized_doc_len() {
    let mut buf = header_with(1);
    field_prefix(&mut buf, FieldType::F64Nested, u32::MAX);
    assert_capacity_rejected(read_fields(&buf).unwrap_err());
  }

  #[test]
  fn read_fields_rejects_i64_nested_oversized_total_objects() {
    // doc_len = 0 so the doc_offsets allocation is fine (1 entry), but the
    // single offset claims `u32::MAX` objects, which is what the issue
    // called out as the second-layer OOM vector.
    let mut buf = header_with(1);
    field_prefix(&mut buf, FieldType::I64Nested, 0);
    buf.extend_from_slice(&u32::MAX.to_le_bytes()); // doc_offsets[0]
    assert_capacity_rejected(read_fields(&buf).unwrap_err());
  }

  #[test]
  fn read_fields_rejects_f64_nested_oversized_total_objects() {
    let mut buf = header_with(1);
    field_prefix(&mut buf, FieldType::F64Nested, 0);
    buf.extend_from_slice(&u32::MAX.to_le_bytes()); // doc_offsets[0]
    assert_capacity_rejected(read_fields(&buf).unwrap_err());
  }

  #[test]
  fn read_fields_rejects_str_oversized_dict_len() {
    let mut buf = header_with(1);
    field_prefix(&mut buf, FieldType::Str, 0);
    buf.extend_from_slice(&u32::MAX.to_le_bytes()); // dict_len
    assert_capacity_rejected(read_fields(&buf).unwrap_err());
  }

  #[test]
  fn read_fields_rejects_str_list_oversized_dict_len() {
    let mut buf = header_with(1);
    field_prefix(&mut buf, FieldType::StrList, 0);
    buf.extend_from_slice(&u32::MAX.to_le_bytes()); // dict_len
    assert_capacity_rejected(read_fields(&buf).unwrap_err());
  }

  #[test]
  fn read_fields_rejects_str_nested_oversized_dict_len() {
    let mut buf = header_with(1);
    field_prefix(&mut buf, FieldType::StrNested, 0);
    buf.extend_from_slice(&u32::MAX.to_le_bytes()); // dict_len
    assert_capacity_rejected(read_fields(&buf).unwrap_err());
  }

  #[test]
  fn read_fields_rejects_nested_count_oversized_doc_len() {
    let mut buf = header_with(1);
    field_prefix(&mut buf, FieldType::NestedCount, u32::MAX);
    assert_capacity_rejected(read_fields(&buf).unwrap_err());
  }

  #[test]
  fn read_fields_rejects_nested_parent_oversized_doc_len() {
    let mut buf = header_with(1);
    field_prefix(&mut buf, FieldType::NestedParent, u32::MAX);
    assert_capacity_rejected(read_fields(&buf).unwrap_err());
  }

  #[test]
  fn read_fields_rejects_nested_parent_oversized_total() {
    let mut buf = header_with(1);
    field_prefix(&mut buf, FieldType::NestedParent, 0);
    buf.extend_from_slice(&u32::MAX.to_le_bytes()); // offsets[0]
    assert_capacity_rejected(read_fields(&buf).unwrap_err());
  }

  // --- Non-UTF-8 corruption regression tests (BUG-217) ---
  //
  // `write_field` always emits field names and string-dictionary entries from
  // Rust `&str` values, which are valid UTF-8 by construction. Any non-UTF-8
  // bytes observed by `read_fields` are therefore a corruption signal. Each
  // test below hand-crafts a minimal `FFV1` buffer whose only corruption is
  // an ill-formed UTF-8 sequence at one of the four affected sites and
  // asserts that `read_fields` surfaces a structured error (preserving the
  // underlying `Utf8Error`) rather than silently mapping to `U+FFFD`. The
  // sequence `[0xC3, 0x28, 0xA0]` is rejected by strict UTF-8 decoding
  // because `0xC3` is a 2-byte lead that requires a continuation byte in
  // `0x80..=0xBF`, and `0x28` falls outside that range.
  const INVALID_UTF8: [u8; 3] = [0xC3, 0x28, 0xA0];

  fn assert_non_utf8_rejected(err: anyhow::Error, label: &str) {
    let msg = format!("{err:#}");
    assert!(
      msg.contains("non-UTF-8 bytes"),
      "expected non-UTF-8 error for {label}, got: {msg}"
    );
    assert!(
      msg.contains(label),
      "expected error for {label} to mention the label, got: {msg}"
    );
    // Preserve the underlying `Utf8Error` in the chain so operators can
    // identify the low-level cause without relying on message text.
    assert!(
      err.chain().any(|cause| cause.is::<std::str::Utf8Error>()),
      "expected Utf8Error in error chain for {label}, got: {msg}"
    );
  }

  #[test]
  fn read_fields_rejects_non_utf8_field_name() {
    let mut buf = header_with(1);
    buf.extend_from_slice(&(INVALID_UTF8.len() as u32).to_le_bytes());
    buf.extend_from_slice(&INVALID_UTF8);
    // The `checked_count` guard at the top of the field loop enforces a
    // 9-byte minimum stride per field (4-byte name length + 0 bytes name
    // + 1-byte type tag + 4-byte doc_len), so append a type byte and a
    // `doc_len` of zero to satisfy it. The buffer still reaches the UTF-8
    // check before any of those trailing bytes are consumed.
    buf.push(FieldType::I64.as_u8());
    buf.extend_from_slice(&0u32.to_le_bytes());
    assert_non_utf8_rejected(read_fields(&buf).unwrap_err(), "field name");
  }

  /// Push an empty-name `Str`/`StrList`/`StrNested` field header whose
  /// dictionary contains a single entry with non-UTF-8 bytes. The loop
  /// rejects the entry before any `doc_len`-sized allocation matters, so
  /// `doc_len = 0` keeps the buffer minimal.
  fn push_str_field_with_invalid_dict_entry(buf: &mut Vec<u8>, ty: FieldType) {
    field_prefix(buf, ty, 0);
    buf.extend_from_slice(&1u32.to_le_bytes()); // dict_len = 1
    buf.extend_from_slice(&(INVALID_UTF8.len() as u32).to_le_bytes());
    buf.extend_from_slice(&INVALID_UTF8);
  }

  #[test]
  fn read_fields_rejects_non_utf8_str_dict_entry() {
    let mut buf = header_with(1);
    push_str_field_with_invalid_dict_entry(&mut buf, FieldType::Str);
    assert_non_utf8_rejected(read_fields(&buf).unwrap_err(), "Str dict entry");
  }

  #[test]
  fn read_fields_rejects_non_utf8_str_list_dict_entry() {
    let mut buf = header_with(1);
    push_str_field_with_invalid_dict_entry(&mut buf, FieldType::StrList);
    assert_non_utf8_rejected(read_fields(&buf).unwrap_err(), "StrList dict entry");
  }

  #[test]
  fn read_fields_rejects_non_utf8_str_nested_dict_entry() {
    let mut buf = header_with(1);
    push_str_field_with_invalid_dict_entry(&mut buf, FieldType::StrNested);
    assert_non_utf8_rejected(read_fields(&buf).unwrap_err(), "StrNested dict entry");
  }

  /// End-to-end variant: round-trip a valid `Str` column through
  /// `FastFieldsWriter` + `FastFieldsReader::open`, mutate the on-disk
  /// dictionary entry to be ill-formed UTF-8, and confirm the reader
  /// surfaces the corruption instead of returning a `U+FFFD`-substituted
  /// column. Mirrors `terms::tests::invalid_utf8_term_errors`.
  #[test]
  fn open_rejects_non_utf8_str_dict_entry_on_disk() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("fast.ff");
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());

    let mut writer = FastFieldsWriter::new();
    writer.set("tag", 0, FastValue::Str("valid".into()));
    writer.write_to(&storage, &path).unwrap();

    // The writer emits the ASCII `valid` byte sequence verbatim into the
    // dictionary; find and overwrite its leading byte with the start of a
    // 2-byte UTF-8 sequence so the entry is no longer valid UTF-8.
    let mut bytes = std::fs::read(&path).unwrap();
    let needle = b"valid";
    let idx = bytes
      .windows(needle.len())
      .position(|w| w == needle)
      .expect("expected ASCII dict entry in on-disk buffer");
    bytes[idx] = 0xC3;
    std::fs::write(&path, bytes).unwrap();

    // `FastFieldsReader` does not implement `Debug`, so `.unwrap_err()` is
    // not available here — match the `Result` explicitly.
    let err = match FastFieldsReader::open(&storage, &path) {
      Ok(_) => panic!("expected non-UTF-8 error, got Ok"),
      Err(e) => e,
    };
    assert_non_utf8_rejected(err, "Str dict entry");
  }

  #[test]
  fn checked_count_accepts_zero_count() {
    // Guard against the helper over-rejecting legitimate empty columns.
    assert_eq!(checked_count(0, 8, 0).unwrap(), 0);
    assert_eq!(checked_count(0, 8, 1024).unwrap(), 0);
  }

  #[test]
  fn checked_count_accepts_exact_fit() {
    assert_eq!(checked_count(4, 8, 32).unwrap(), 4);
  }

  #[test]
  fn checked_count_rejects_one_over() {
    checked_count(5, 8, 32).unwrap_err();
  }

  #[test]
  fn checked_count_rejects_overflow() {
    // `count * stride` overflows `usize` even though `remaining` would
    // otherwise allow the claim.
    checked_count(usize::MAX, 8, usize::MAX).unwrap_err();
  }

  #[test]
  fn checked_add_one_rejects_usize_max() {
    checked_add_one(usize::MAX).unwrap_err();
    assert_eq!(checked_add_one(0).unwrap(), 1);
    assert_eq!(
      checked_add_one(u32::MAX as usize).unwrap(),
      u32::MAX as usize + 1
    );
  }

  // --- Non-monotonic offset regression tests (BUG-253) ---
  //
  // Each test below hand-crafts a fast-field buffer whose offset array
  // contains a non-monotonic pair (start > end). Before the fix,
  // `values[start..end]` would panic at query time. `read_fields` must
  // now reject these at load time.

  fn assert_monotonicity_rejected(err: anyhow::Error) {
    let msg = err.to_string();
    assert!(
      msg.contains("non-monotonic offsets"),
      "expected a non-monotonic offsets error, got: {msg}"
    );
  }

  #[test]
  fn read_fields_rejects_non_monotonic_i64_list_offsets() {
    let mut buf = header_with(1);
    field_prefix(&mut buf, FieldType::I64List, 2); // 2 docs → 3 offsets
                                                   // offsets: [0, 5, 3] — doc 1 has start=5, end=3
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&5u32.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    assert_monotonicity_rejected(read_fields(&buf).unwrap_err());
  }

  #[test]
  fn read_fields_rejects_non_monotonic_f64_list_offsets() {
    let mut buf = header_with(1);
    field_prefix(&mut buf, FieldType::F64List, 2);
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&5u32.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    assert_monotonicity_rejected(read_fields(&buf).unwrap_err());
  }

  #[test]
  fn read_fields_rejects_non_monotonic_i64_nested_doc_offsets() {
    let mut buf = header_with(1);
    field_prefix(&mut buf, FieldType::I64Nested, 2);
    // doc_offsets: [0, 3, 1] — non-monotonic
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes());
    assert_monotonicity_rejected(read_fields(&buf).unwrap_err());
  }

  #[test]
  fn read_fields_rejects_non_monotonic_i64_nested_object_offsets() {
    let mut buf = header_with(1);
    field_prefix(&mut buf, FieldType::I64Nested, 1);
    // doc_offsets: [0, 2] — valid, 2 objects
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&2u32.to_le_bytes());
    // object_offsets: [0, 5, 3] — non-monotonic at index 1→2
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&5u32.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    assert_monotonicity_rejected(read_fields(&buf).unwrap_err());
  }

  #[test]
  fn read_fields_rejects_non_monotonic_f64_nested_doc_offsets() {
    let mut buf = header_with(1);
    field_prefix(&mut buf, FieldType::F64Nested, 2);
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes());
    assert_monotonicity_rejected(read_fields(&buf).unwrap_err());
  }

  #[test]
  fn read_fields_rejects_non_monotonic_f64_nested_object_offsets() {
    let mut buf = header_with(1);
    field_prefix(&mut buf, FieldType::F64Nested, 1);
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&2u32.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&5u32.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    assert_monotonicity_rejected(read_fields(&buf).unwrap_err());
  }

  #[test]
  fn read_fields_rejects_non_monotonic_str_list_offsets() {
    let mut buf = header_with(1);
    field_prefix(&mut buf, FieldType::StrList, 2);
    buf.extend_from_slice(&0u32.to_le_bytes()); // dict_len = 0
                                                // offsets: [0, 5, 3] — non-monotonic
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&5u32.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    assert_monotonicity_rejected(read_fields(&buf).unwrap_err());
  }

  #[test]
  fn read_fields_rejects_non_monotonic_str_nested_doc_offsets() {
    let mut buf = header_with(1);
    field_prefix(&mut buf, FieldType::StrNested, 2);
    buf.extend_from_slice(&0u32.to_le_bytes()); // dict_len = 0
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes());
    assert_monotonicity_rejected(read_fields(&buf).unwrap_err());
  }

  #[test]
  fn read_fields_rejects_non_monotonic_str_nested_object_offsets() {
    let mut buf = header_with(1);
    field_prefix(&mut buf, FieldType::StrNested, 1);
    buf.extend_from_slice(&0u32.to_le_bytes()); // dict_len = 0
                                                // doc_offsets: [0, 2] — valid
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&2u32.to_le_bytes());
    // object_offsets: [0, 5, 3] — non-monotonic
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&5u32.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    assert_monotonicity_rejected(read_fields(&buf).unwrap_err());
  }

  #[test]
  fn read_fields_rejects_non_monotonic_nested_parent_offsets() {
    let mut buf = header_with(1);
    field_prefix(&mut buf, FieldType::NestedParent, 2);
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes());
    assert_monotonicity_rejected(read_fields(&buf).unwrap_err());
  }

  #[test]
  fn read_fields_accepts_monotonic_offsets() {
    // Monotonically non-decreasing offsets (including equal adjacent
    // values, which represent empty ranges) must be accepted.
    let mut buf = header_with(1);
    field_prefix(&mut buf, FieldType::I64List, 3); // 3 docs → 4 offsets
                                                   // offsets: [0, 0, 2, 2] — doc 0 empty, doc 1 has 2 values, doc 2 empty
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&2u32.to_le_bytes());
    buf.extend_from_slice(&2u32.to_le_bytes());
    // 2 i64 values
    buf.extend_from_slice(&42i64.to_le_bytes());
    buf.extend_from_slice(&99i64.to_le_bytes());
    let fields = read_fields(&buf).unwrap();
    assert!(fields.contains_key(""));
  }

  #[test]
  fn doc_range_returns_none_for_inverted_offsets() {
    // Defense-in-depth: even if offsets somehow bypass load-time validation,
    // doc_range must return None rather than yielding a start > end pair.
    let offsets = vec![0u32, 5, 3, 10];
    assert_eq!(doc_range(&offsets, 0), Some((0, 5)));
    assert_eq!(doc_range(&offsets, 1), None); // 5 > 3, inverted
    assert_eq!(doc_range(&offsets, 2), Some((3, 10)));
  }

  #[test]
  fn object_range_returns_none_for_inverted_offsets() {
    let offsets = vec![0u32, 5, 3, 10];
    assert_eq!(object_range(&offsets, 0), Some((0, 5)));
    assert_eq!(object_range(&offsets, 1), None);
    assert_eq!(object_range(&offsets, 2), Some((3, 10)));
  }

  #[test]
  fn validate_monotonic_offsets_accepts_empty() {
    validate_monotonic_offsets(&[], "test").unwrap();
  }

  #[test]
  fn validate_monotonic_offsets_accepts_single() {
    validate_monotonic_offsets(&[42], "test").unwrap();
  }

  #[test]
  fn validate_monotonic_offsets_accepts_equal_adjacent() {
    validate_monotonic_offsets(&[0, 0, 5, 5, 10], "test").unwrap();
  }

  #[test]
  fn validate_monotonic_offsets_rejects_decreasing() {
    let err = validate_monotonic_offsets(&[0, 5, 3], "TestCol").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("non-monotonic offsets"));
    assert!(msg.contains("TestCol"));
    assert!(msg.contains("5 > 3"));
  }
}
