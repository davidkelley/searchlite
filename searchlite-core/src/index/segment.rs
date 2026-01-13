use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
#[cfg(feature = "vectors")]
use bincode::Options;
#[cfg(feature = "vectors")]
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use hashbrown::{HashMap as FastHashMap, HashSet as FastHashSet};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::api::types::Document;
#[cfg(feature = "vectors")]
use crate::api::types::VectorMetric as ApiVectorMetric;
use crate::index::directory;
use crate::index::docstore::{DocStoreReader, DocStoreWriter};
use crate::index::fastfields::{
  doc_length_key, nested_count_key, nested_parent_key, FastFieldsReader, FastFieldsWriter,
  FastValue,
};
use crate::index::manifest::{
  FieldKind, NestedField, NestedProperty, ResolvedField, Schema, SegmentMeta, SegmentPaths,
};
use crate::index::postings::{read_doc_freq, InvertedIndexBuilder, PostingsReader, PostingsWriter};
use crate::index::terms::{read_terms, write_terms};
use crate::storage::{Storage, StorageFile};
use crate::util::checksum::checksum;
#[cfg(feature = "vectors")]
use crate::vectors::hnsw::HnswParams;
#[cfg(feature = "vectors")]
use crate::vectors::hnsw::{HnswGraph, HnswIndex};
#[cfg(feature = "vectors")]
use crate::vectors::VectorStore;
use crate::DocId;
#[cfg(feature = "vectors")]
use std::io::Cursor;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentFileMeta {
  pub doc_offsets: Vec<u64>,
  #[serde(default)]
  pub doc_ids: Vec<String>,
  pub avg_field_lengths: HashMap<String, f32>,
  #[cfg(feature = "vectors")]
  #[serde(default)]
  pub vector_fields: HashMap<String, VectorFieldMeta>,
  pub use_zstd: bool,
}

#[cfg(feature = "vectors")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorFieldMeta {
  pub dim: usize,
  pub metric: crate::index::manifest::VectorMetric,
  #[serde(default)]
  pub vectors: u32,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub hnsw: Option<HnswParams>,
}

#[derive(Default)]
struct CollectedDocument {
  doc_id: Option<String>,
  text: HashMap<String, Vec<String>>,
  keywords: HashMap<String, Vec<String>>,
  i64s: HashMap<String, Vec<i64>>,
  f64s: HashMap<String, Vec<f64>>,
  stored: HashMap<String, Vec<serde_json::Value>>,
  nested_keywords: HashMap<String, Vec<Vec<String>>>,
  nested_i64s: HashMap<String, Vec<Vec<i64>>>,
  nested_f64s: HashMap<String, Vec<Vec<f64>>>,
  nested_counts: HashMap<String, usize>,
  nested_parents: HashMap<String, Vec<usize>>,
  nested_stored: HashMap<String, serde_json::Value>,
  #[cfg(feature = "vectors")]
  vectors: HashMap<String, Option<Vec<f32>>>,
}

impl CollectedDocument {
  fn push_stored(&mut self, path: &str, values: impl IntoIterator<Item = serde_json::Value>) {
    let entry = self.stored.entry(path.to_string()).or_default();
    entry.extend(values);
  }

  fn finalize_stored(self) -> serde_json::Map<String, serde_json::Value> {
    let mut out: serde_json::Map<String, serde_json::Value> = self
      .stored
      .into_iter()
      .map(|(k, vals)| {
        let value = if vals.len() == 1 {
          vals.into_iter().next().unwrap()
        } else {
          serde_json::Value::Array(vals)
        };
        (k, value)
      })
      .collect();
    for (k, v) in self.nested_stored.into_iter() {
      out.insert(k, v);
    }
    out
  }
}

fn collect_strings(value: &serde_json::Value) -> Vec<String> {
  match value {
    serde_json::Value::String(s) => vec![s.clone()],
    serde_json::Value::Array(arr) => arr
      .iter()
      .filter_map(|v| v.as_str().map(|s| s.to_string()))
      .collect(),
    _ => Vec::new(),
  }
}

fn collect_i64s(value: &serde_json::Value) -> Vec<i64> {
  match value {
    serde_json::Value::Number(n) => n.as_i64().into_iter().collect(),
    serde_json::Value::Array(arr) => arr.iter().filter_map(|v| v.as_i64()).collect(),
    _ => Vec::new(),
  }
}

fn collect_f64s(value: &serde_json::Value) -> Vec<f64> {
  match value {
    serde_json::Value::Number(n) => n.as_f64().into_iter().collect(),
    serde_json::Value::Array(arr) => arr.iter().filter_map(|v| v.as_f64()).collect(),
    _ => Vec::new(),
  }
}

fn handle_field(
  meta: &ResolvedField,
  value: &serde_json::Value,
  collected: &mut CollectedDocument,
  store_value: bool,
) {
  match meta.kind {
    FieldKind::Text => {
      let vals = collect_strings(value);
      if meta.indexed && !vals.is_empty() {
        collected
          .text
          .entry(meta.path.clone())
          .or_default()
          .extend(vals.iter().cloned());
      }
      if meta.stored && store_value {
        collected.push_stored(&meta.path, vals.into_iter().map(serde_json::Value::String));
      }
    }
    FieldKind::Keyword => {
      let vals = collect_strings(value);
      if !vals.is_empty() {
        collected
          .keywords
          .entry(meta.path.clone())
          .or_default()
          .extend(vals.iter().cloned());
      }
      if meta.stored && store_value {
        collected.push_stored(&meta.path, vals.into_iter().map(serde_json::Value::String));
      }
    }
    FieldKind::Numeric => {
      if meta.numeric_i64.unwrap_or(false) {
        let vals = collect_i64s(value);
        if !vals.is_empty() {
          collected
            .i64s
            .entry(meta.path.clone())
            .or_default()
            .extend(vals.iter().cloned());
        }
        if meta.stored && store_value {
          collected.push_stored(&meta.path, vals.into_iter().map(serde_json::Value::from));
        }
      } else {
        let vals = collect_f64s(value);
        if !vals.is_empty() {
          collected
            .f64s
            .entry(meta.path.clone())
            .or_default()
            .extend(vals.iter().cloned());
        }
        if meta.stored && store_value {
          collected.push_stored(&meta.path, vals.into_iter().map(serde_json::Value::from));
        }
      }
    }
    FieldKind::Unknown => {}
  }
}

#[allow(clippy::too_many_arguments)]
fn collect_nested(
  schema: &Schema,
  nested: &NestedField,
  value: &serde_json::Value,
  prefix: &str,
  collected: &mut CollectedDocument,
  resolved: &FastHashMap<String, ResolvedField>,
  store_value: bool,
  parent_idx: Option<usize>,
) -> Result<()> {
  match value {
    serde_json::Value::Null => {
      if nested.nullable {
        return Ok(());
      }
      bail!("nested field {prefix} cannot be null");
    }
    serde_json::Value::Array(arr) => {
      collected
        .nested_counts
        .insert(prefix.to_string(), arr.len());
      if let Some(p) = parent_idx {
        let entry = collected
          .nested_parents
          .entry(prefix.to_string())
          .or_insert_with(|| vec![usize::MAX; arr.len()]);
        if entry.len() < arr.len() {
          entry.resize(arr.len(), usize::MAX);
        }
        for slot in entry.iter_mut().take(arr.len()) {
          *slot = p;
        }
      } else {
        collected
          .nested_parents
          .entry(prefix.to_string())
          .or_insert_with(|| vec![usize::MAX; arr.len()]);
      }
      for (idx, v) in arr.iter().enumerate() {
        if v.is_null() {
          if nested.nullable {
            continue;
          }
          bail!("nested field {prefix} cannot be null");
        }
        let map = v
          .as_object()
          .ok_or_else(|| anyhow!("nested field {prefix} must contain objects"))?;
        collect_nested_object(schema, nested, map, prefix, idx, collected, resolved)?;
      }
    }
    serde_json::Value::Object(map) => {
      collected.nested_counts.insert(prefix.to_string(), 1);
      collected
        .nested_parents
        .entry(prefix.to_string())
        .or_insert_with(|| vec![parent_idx.unwrap_or(usize::MAX)]);
      collect_nested_object(schema, nested, map, prefix, 0, collected, resolved)?;
    }
    _ => bail!("nested field {prefix} must be object or array"),
  }
  if store_value {
    if let Some(filtered) = stored_nested_value(nested, value) {
      collected.nested_stored.insert(prefix.to_string(), filtered);
    }
  }
  Ok(())
}

fn record_nested_strings(
  collected: &mut CollectedDocument,
  field: &str,
  object_count: usize,
  object_idx: usize,
  values: Vec<String>,
) {
  let entry = collected
    .nested_keywords
    .entry(field.to_string())
    .or_insert_with(|| vec![Vec::new(); object_count]);
  if entry.len() < object_count {
    entry.resize(object_count, Vec::new());
  }
  if object_idx < entry.len() {
    entry[object_idx].extend(values);
  }
}

fn record_nested_i64(
  collected: &mut CollectedDocument,
  field: &str,
  object_count: usize,
  object_idx: usize,
  values: Vec<i64>,
) {
  let entry = collected
    .nested_i64s
    .entry(field.to_string())
    .or_insert_with(|| vec![Vec::new(); object_count]);
  if entry.len() < object_count {
    entry.resize(object_count, Vec::new());
  }
  if object_idx < entry.len() {
    entry[object_idx].extend(values);
  }
}

fn record_nested_f64(
  collected: &mut CollectedDocument,
  field: &str,
  object_count: usize,
  object_idx: usize,
  values: Vec<f64>,
) {
  let entry = collected
    .nested_f64s
    .entry(field.to_string())
    .or_insert_with(|| vec![Vec::new(); object_count]);
  if entry.len() < object_count {
    entry.resize(object_count, Vec::new());
  }
  if object_idx < entry.len() {
    entry[object_idx].extend(values);
  }
}

fn collect_nested_object(
  schema: &Schema,
  nested: &NestedField,
  map: &serde_json::Map<String, serde_json::Value>,
  prefix: &str,
  object_idx: usize,
  collected: &mut CollectedDocument,
  resolved: &FastHashMap<String, ResolvedField>,
) -> Result<()> {
  let object_count = *collected.nested_counts.get(prefix).unwrap_or(&0);
  for (k, v) in map.iter() {
    if let Some(prop) = nested.fields.iter().find(|p| p.name() == k) {
      match prop {
        NestedProperty::Object(obj) => {
          let next_prefix = format!("{prefix}.{}", obj.name);
          if v.is_null() {
            if obj.nullable {
              continue;
            }
            bail!("nested field {next_prefix} cannot be null");
          }
          collect_nested(
            schema,
            obj,
            v,
            &next_prefix,
            collected,
            resolved,
            false,
            Some(object_idx),
          )?;
        }
        _ => {
          let full_path = format!("{prefix}.{k}");
          if let Some(meta) = resolved.get(&full_path) {
            handle_field(meta, v, collected, false);
            if meta.fast {
              match meta.kind {
                FieldKind::Keyword => {
                  let vals = collect_strings(v);
                  if !vals.is_empty() {
                    record_nested_strings(collected, &full_path, object_count, object_idx, vals);
                  }
                }
                FieldKind::Numeric => {
                  if meta.numeric_i64.unwrap_or(false) {
                    let vals = collect_i64s(v);
                    if !vals.is_empty() {
                      record_nested_i64(collected, &full_path, object_count, object_idx, vals);
                    }
                  } else {
                    let vals = collect_f64s(v);
                    if !vals.is_empty() {
                      record_nested_f64(collected, &full_path, object_count, object_idx, vals);
                    }
                  }
                }
                FieldKind::Text | FieldKind::Unknown => {}
              }
            }
          } else {
            bail!("unknown nested field {prefix}.{k}");
          }
        }
      }
    } else {
      bail!("unknown nested field {prefix}.{k}");
    }
  }
  Ok(())
}

fn stored_nested_value(
  nested: &NestedField,
  value: &serde_json::Value,
) -> Option<serde_json::Value> {
  match value {
    serde_json::Value::Array(arr) => {
      let mut filtered = Vec::new();
      for v in arr.iter() {
        if let Some(v) = stored_nested_value(nested, v) {
          filtered.push(v);
        }
      }
      if filtered.is_empty() {
        None
      } else {
        Some(serde_json::Value::Array(filtered))
      }
    }
    serde_json::Value::Object(map) => {
      let mut out = serde_json::Map::new();
      for prop in nested.fields.iter() {
        if let Some(raw) = map.get(prop.name()) {
          match prop {
            NestedProperty::Text(f) => {
              if raw.is_null() {
                continue;
              }
              if f.stored {
                out.insert(prop.name().to_string(), raw.clone());
              }
            }
            NestedProperty::Keyword(f) => {
              if raw.is_null() {
                continue;
              }
              if f.stored {
                out.insert(prop.name().to_string(), raw.clone());
              }
            }
            NestedProperty::Numeric(f) => {
              if raw.is_null() {
                continue;
              }
              if f.stored {
                out.insert(prop.name().to_string(), raw.clone());
              }
            }
            NestedProperty::Object(obj) => {
              if raw.is_null() {
                continue;
              }
              if let Some(child) = stored_nested_value(obj, raw) {
                out.insert(prop.name().to_string(), child);
              }
            }
          }
        }
      }
      if out.is_empty() {
        None
      } else {
        Some(serde_json::Value::Object(out))
      }
    }
    _ => None,
  }
}

#[cfg(feature = "vectors")]
fn collect_vector_value(
  schema: &Schema,
  field: &str,
  value: &serde_json::Value,
) -> Result<Option<Vec<f32>>> {
  use crate::index::manifest::VectorMetric;
  use crate::vectors::normalize_in_place;
  let Some(vf) = schema.vector_field(field) else {
    bail!("unknown vector field {field}");
  };
  if value.is_null() {
    return Ok(None);
  }
  let arr = value
    .as_array()
    .ok_or_else(|| anyhow!("vector field {field} must be an array"))?;
  let mut vecvals: Vec<f32> = Vec::with_capacity(arr.len());
  for v in arr.iter() {
    let Some(num) = v.as_f64() else {
      bail!("vector field {field} must contain numbers");
    };
    vecvals.push(num as f32);
  }
  if vecvals.len() != vf.dim {
    bail!(
      "vector field {field} expected dimension {}, got {}",
      vf.dim,
      vecvals.len()
    );
  }
  if matches!(vf.metric, VectorMetric::Cosine) {
    normalize_in_place(&mut vecvals);
  }
  Ok(Some(vecvals))
}

fn collect_document(
  schema: &Schema,
  doc: &Document,
  resolved: &FastHashMap<String, ResolvedField>,
) -> Result<CollectedDocument> {
  let mut collected = CollectedDocument::default();
  let doc_id = doc
    .fields
    .get(schema.doc_id_field())
    .and_then(|v| v.as_str())
    .expect("doc ids validated upstream");
  collected.doc_id = Some(doc_id.to_string());
  collected.push_stored(
    schema.doc_id_field(),
    [serde_json::Value::String(doc_id.to_string())],
  );
  for (field, value) in doc.fields.iter() {
    if field == schema.doc_id_field() {
      continue;
    }
    #[cfg(feature = "vectors")]
    if schema.vector_fields.iter().any(|vf| vf.name == *field) {
      let vec_value = collect_vector_value(schema, field, value)?;
      collected.vectors.insert(field.clone(), vec_value);
      continue;
    }
    if let Some(meta) = resolved.get(field) {
      handle_field(meta, value, &mut collected, true);
    } else if let Some(nested) = schema.nested_fields.iter().find(|n| n.name == *field) {
      if value.is_null() {
        if nested.nullable {
          continue;
        }
        bail!("nested field {} cannot be null", nested.name);
      }
      collect_nested(
        schema,
        nested,
        value,
        &nested.name,
        &mut collected,
        resolved,
        true,
        None,
      )?;
    } else {
      bail!("unknown field {field}");
    }
  }
  Ok(collected)
}

pub struct SegmentWriter<'a> {
  root: &'a Path,
  schema: &'a Schema,
  enable_positions: bool,
  use_zstd: bool,
  storage: Arc<dyn Storage>,
}

impl<'a> SegmentWriter<'a> {
  pub fn new(
    root: &'a Path,
    schema: &'a Schema,
    enable_positions: bool,
    use_zstd: bool,
    storage: Arc<dyn Storage>,
  ) -> Self {
    Self {
      root,
      schema,
      enable_positions,
      use_zstd,
      storage,
    }
  }

  pub fn write_segment(&self, docs: &[Document], generation: u32) -> Result<SegmentMeta> {
    self.write_segment_stream(docs.iter().map(|doc| Ok(Cow::Borrowed(doc))), generation)
  }

  #[allow(dead_code)]
  pub fn write_segment_from_iter<I>(&self, docs: I, generation: u32) -> Result<SegmentMeta>
  where
    I: IntoIterator<Item = Result<Document>>,
  {
    self.write_segment_stream(docs.into_iter().map(|doc| doc.map(Cow::Owned)), generation)
  }

  fn write_segment_stream<'doc, I>(&self, docs: I, generation: u32) -> Result<SegmentMeta>
  where
    I: IntoIterator<Item = Result<Cow<'doc, Document>>>,
  {
    let id = Uuid::new_v4().simple().to_string();
    let paths = directory::segment_paths(self.root, &id);
    let analyzers = self.schema.build_analyzers()?;

    let mut postings_builder = InvertedIndexBuilder::new();
    let mut total_doc_lengths: HashMap<String, u64> = HashMap::new();
    let mut fast_writer = FastFieldsWriter::new();
    let resolved: FastHashMap<String, ResolvedField> = self
      .schema
      .resolved_fields()
      .into_iter()
      .map(|f| (f.path.clone(), f))
      .collect();
    let keyword_fast: FastHashSet<&str> = resolved
      .values()
      .filter(|f| matches!(f.kind, FieldKind::Keyword) && f.fast)
      .map(|f| f.path.as_str())
      .collect();
    let numeric_info: FastHashMap<&str, (bool, bool)> = resolved
      .values()
      .filter(|f| matches!(f.kind, FieldKind::Numeric))
      .map(|f| (f.path.as_str(), (f.numeric_i64.unwrap_or(false), f.fast)))
      .collect();

    let mut docstore_file = self.storage.open_write(Path::new(&paths.docstore))?;
    let mut doc_writer = DocStoreWriter::new(&mut *docstore_file, self.use_zstd);

    #[cfg(feature = "vectors")]
    let mut vector_fields: HashMap<String, Vec<Option<Vec<f32>>>> = HashMap::new();
    let mut doc_ids: Vec<String> = Vec::new();

    for doc_res in docs.into_iter() {
      let doc = doc_res?;
      let doc_ref = doc.as_ref();
      let doc_ord = doc_ids.len() as DocId;
      self.schema.validate_document(doc_ref)?;
      let collected = collect_document(self.schema, doc_ref, &resolved)?;
      let doc_key = collected
        .doc_id
        .clone()
        .expect("collect_document should enforce doc id presence");
      doc_ids.push(doc_key.clone());
      fast_writer.set(
        self.schema.doc_id_field(),
        doc_ord,
        FastValue::Str(doc_key.clone()),
      );

      for (field, values) in collected.text.iter() {
        if let Some(meta) = resolved.get(field) {
          if !meta.indexed {
            continue;
          }
        }
        let Some(analyzer) = analyzers.index_analyzer(field) else {
          bail!("no analyzer configured for field `{field}`");
        };
        let mut position_offset: u32 = 0;
        let mut doc_len: u32 = 0;
        for text in values.iter() {
          let tokens = analyzer.analyze(text);
          let token_count = tokens.len() as u32;
          doc_len = doc_len.saturating_add(token_count);
          total_doc_lengths
            .entry(field.clone())
            .and_modify(|v| *v += token_count as u64)
            .or_insert(token_count as u64);
          for tok in tokens.iter() {
            let mut term_key = String::with_capacity(field.len() + tok.text.len() + 1);
            term_key.push_str(field);
            term_key.push(':');
            term_key.push_str(&tok.text);
            postings_builder.add_term(
              &term_key,
              doc_ord,
              position_offset + tok.position,
              self.enable_positions,
            );
          }
          if let Some(max_pos) = tokens.iter().map(|t| t.position).max() {
            position_offset += max_pos + 1;
          } else {
            // Preserve a position gap between successive values even when filters drop all tokens.
            position_offset += 1;
          }
        }
        fast_writer.set(
          &doc_length_key(field),
          doc_ord,
          FastValue::I64(doc_len as i64),
        );
      }

      for (field, values) in collected.keywords.iter() {
        let mut seen_terms = FastHashSet::new();
        let indexed = resolved.get(field).map(|m| m.indexed).unwrap_or(true);
        let is_nested_field = field.contains('.');
        for value in values.iter() {
          if indexed {
            let lower = value.to_ascii_lowercase();
            if seen_terms.insert(lower.clone()) {
              let mut term_key = String::with_capacity(field.len() + lower.len() + 1);
              term_key.push_str(field);
              term_key.push(':');
              term_key.push_str(&lower);
              postings_builder.add_term(&term_key, doc_ord, 0, false);
            }
          }
        }
        if keyword_fast.contains(field.as_str()) && !is_nested_field {
          if values.len() == 1 {
            fast_writer.set(field, doc_ord, FastValue::Str(values[0].clone()));
          } else if !values.is_empty() {
            fast_writer.set(field, doc_ord, FastValue::StrList(values.clone()));
          }
        }
      }

      for (field, values) in collected.i64s.iter() {
        if let Some((_, fast)) = numeric_info.get(field.as_str()) {
          if *fast && !field.contains('.') {
            if values.len() == 1 {
              fast_writer.set(field, doc_ord, FastValue::I64(values[0]));
            } else {
              fast_writer.set(field, doc_ord, FastValue::I64List(values.clone()));
            }
          }
        }
      }

      for (field, values) in collected.f64s.iter() {
        if let Some((_, fast)) = numeric_info.get(field.as_str()) {
          if *fast && !field.contains('.') {
            if values.len() == 1 {
              fast_writer.set(field, doc_ord, FastValue::F64(values[0]));
            } else {
              fast_writer.set(field, doc_ord, FastValue::F64List(values.clone()));
            }
          }
        }
      }

      for (path, count) in collected.nested_counts.iter() {
        fast_writer.set(
          &nested_count_key(path),
          doc_ord,
          FastValue::NestedCount { objects: *count },
        );
      }

      for (path, parents) in collected.nested_parents.iter() {
        for (object_idx, parent) in parents.iter().enumerate() {
          fast_writer.set(
            &nested_parent_key(path),
            doc_ord,
            FastValue::NestedParent {
              object: object_idx,
              parent: *parent,
            },
          );
        }
      }

      for (field, objects) in collected.nested_keywords.iter() {
        for (object_idx, vals) in objects.iter().enumerate() {
          if !vals.is_empty() {
            fast_writer.set(
              field,
              doc_ord,
              FastValue::StrNested {
                object: object_idx,
                values: vals.clone(),
              },
            );
          }
        }
      }

      for (field, objects) in collected.nested_i64s.iter() {
        for (object_idx, vals) in objects.iter().enumerate() {
          if !vals.is_empty() {
            fast_writer.set(
              field,
              doc_ord,
              FastValue::I64Nested {
                object: object_idx,
                values: vals.clone(),
              },
            );
          }
        }
      }

      for (field, objects) in collected.nested_f64s.iter() {
        for (object_idx, vals) in objects.iter().enumerate() {
          if !vals.is_empty() {
            fast_writer.set(
              field,
              doc_ord,
              FastValue::F64Nested {
                object: object_idx,
                values: vals.clone(),
              },
            );
          }
        }
      }

      #[cfg(feature = "vectors")]
      let collected_vectors = collected.vectors.clone();
      let stored = collected.finalize_stored();

      #[cfg(feature = "vectors")]
      for vf in self.schema.vector_fields.iter() {
        let entry = vector_fields.entry(vf.name.clone()).or_default();
        entry.push(collected_vectors.get(&vf.name).cloned().unwrap_or(None));
      }

      doc_writer.add_document(&serde_json::Value::Object(stored))?;
    }
    let doc_offsets = doc_writer.offsets().to_vec();
    drop(doc_writer);
    docstore_file.sync_all()?;
    drop(docstore_file);

    let mut postings_file = self.storage.open_write(Path::new(&paths.postings))?;
    let mut postings_writer = PostingsWriter::new(&mut *postings_file, self.enable_positions);
    let mut term_offsets = Vec::new();
    for (term, postings) in postings_builder.into_terms() {
      let offset = postings_writer.write_term(&postings)?;
      term_offsets.push((term, offset));
    }
    postings_file.sync_all()?;

    write_terms(
      self.storage.as_ref(),
      Path::new(&paths.terms),
      &term_offsets,
    )?;

    let total_docs = doc_ids.len();
    let avg_field_lengths = compute_avg_lengths(&total_doc_lengths, total_docs as u64);

    fast_writer.write_to(self.storage.as_ref(), Path::new(&paths.fast))?;

    #[cfg(feature = "vectors")]
    let mut vector_meta: HashMap<String, VectorFieldMeta> = HashMap::new();
    #[cfg(feature = "vectors")]
    {
      if !self.schema.vector_fields.is_empty() {
        if let Some(dir) = paths.vector_dir.as_deref() {
          self.storage.ensure_dir(Path::new(dir))?;
        }
      }
      for vf in self.schema.vector_fields.iter() {
        let field_vectors = vector_fields
          .remove(&vf.name)
          .unwrap_or_else(|| vec![None; total_docs]);
        if field_vectors.len() != total_docs {
          bail!("vector field {} missing values", vf.name);
        }
        let (store, present) = build_vector_store(vf, &field_vectors)?;
        let (vec_path, hnsw_path) = vector_paths(&paths, &vf.name)?;
        write_vector_file(self.storage.as_ref(), &vec_path, &store)?;
        let params = vf.hnsw.unwrap_or_default();
        let store_arc = Arc::new(store);
        let mut index = HnswIndex::new(store_arc.clone(), params);
        for doc_id in 0..total_docs {
          if store_arc.vector(doc_id as u32).is_some() {
            index.add_vector(doc_id as u32);
          }
        }
        let graph = index.into_graph();
        let graph_bytes = serde_json::to_vec(&graph)?;
        self.storage.write_all(&hnsw_path, &graph_bytes)?;
        vector_meta.insert(
          vf.name.clone(),
          VectorFieldMeta {
            dim: vf.dim,
            metric: vf.metric.clone(),
            vectors: present,
            hnsw: Some(params),
          },
        );
      }
    }

    let seg_file_meta = SegmentFileMeta {
      doc_offsets,
      doc_ids,
      avg_field_lengths: avg_field_lengths.clone(),
      #[cfg(feature = "vectors")]
      vector_fields: vector_meta.clone(),
      use_zstd: self.use_zstd,
    };
    write_segment_meta(
      self.storage.as_ref(),
      Path::new(&paths.meta),
      &seg_file_meta,
    )?;

    #[cfg(feature = "vectors")]
    let mut checksums = collect_checksums(self.storage.as_ref(), &paths)?;
    #[cfg(not(feature = "vectors"))]
    let checksums = collect_checksums(self.storage.as_ref(), &paths)?;
    #[cfg(feature = "vectors")]
    for (field, _meta) in vector_meta.iter() {
      let (vec_path, hnsw_path) = vector_paths(&paths, field)?;
      let vec_buf = self.storage.read_to_end(&vec_path)?;
      let hnsw_buf = self.storage.read_to_end(&hnsw_path)?;
      checksums.insert(format!("vector_{}_bin", field), checksum(&vec_buf));
      checksums.insert(format!("vector_{}_hnsw", field), checksum(&hnsw_buf));
    }

    let meta = SegmentMeta {
      id,
      generation,
      paths,
      doc_count: total_docs as u32,
      max_doc_id: total_docs.saturating_sub(1) as u32,
      blockmax: true,
      deleted_docs: Vec::new(),
      avg_field_lengths,
      checksums,
    };
    Ok(meta)
  }
}

fn write_segment_meta(storage: &dyn Storage, path: &Path, meta: &SegmentFileMeta) -> Result<()> {
  let mut handle = storage.open_write(path)?;
  let mut writer = BufWriter::new(&mut *handle);
  serde_json::to_writer_pretty(&mut writer, meta)?;
  writer.flush()?;
  drop(writer);
  handle.sync_all()?;
  Ok(())
}

fn compute_avg_lengths(lengths: &HashMap<String, u64>, total_docs: u64) -> HashMap<String, f32> {
  let mut out = HashMap::new();
  for (field, sum) in lengths {
    let avg = if total_docs == 0 {
      0.0
    } else {
      *sum as f32 / total_docs as f32
    };
    out.insert(field.clone(), avg);
  }
  out
}

#[cfg(feature = "vectors")]
const VECTOR_FILE_MAGIC: u32 = 0x56435452; // "VCTR"
#[cfg(feature = "vectors")]
const VECTOR_FILE_VERSION: u32 = 1;

#[cfg(feature = "vectors")]
fn vector_paths(
  paths: &SegmentPaths,
  field: &str,
) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
  let dir = paths
    .vector_dir
    .as_deref()
    .ok_or_else(|| anyhow!("segment missing vector directory path"))?;
  let base = Path::new(dir);
  Ok((
    base.join(format!("{field}.bin")),
    base.join(format!("{field}.hnsw")),
  ))
}

#[cfg(feature = "vectors")]
fn metric_code(metric: &ApiVectorMetric) -> u8 {
  match metric {
    ApiVectorMetric::Cosine => 0,
    ApiVectorMetric::L2 => 1,
  }
}

#[cfg(feature = "vectors")]
fn metric_from_code(code: u8) -> Option<ApiVectorMetric> {
  match code {
    0 => Some(ApiVectorMetric::Cosine),
    1 => Some(ApiVectorMetric::L2),
    _ => None,
  }
}

#[cfg(feature = "vectors")]
fn build_vector_store(
  field: &crate::index::manifest::VectorField,
  vectors: &[Option<Vec<f32>>],
) -> Result<(VectorStore, u32)> {
  let mut offsets = vec![u32::MAX; vectors.len()];
  let mut values = Vec::new();
  let mut present = 0u32;
  for (doc_id, vec_opt) in vectors.iter().enumerate() {
    if let Some(vecvals) = vec_opt {
      if vecvals.len() != field.dim {
        bail!(
          "vector field {} expected dim {}, got {} on doc {}",
          field.name,
          field.dim,
          vecvals.len(),
          doc_id
        );
      }
      let vals = vecvals.clone();
      offsets[doc_id] = present;
      present = present.saturating_add(1);
      values.extend_from_slice(&vals);
    }
  }
  let metric: ApiVectorMetric = field.metric.clone().into();
  Ok((
    VectorStore::new(field.dim, metric, offsets, values),
    present,
  ))
}

#[cfg(feature = "vectors")]
fn write_vector_file(storage: &dyn Storage, path: &Path, store: &VectorStore) -> Result<()> {
  let mut buf: Vec<u8> = Vec::new();
  buf.write_u32::<LittleEndian>(VECTOR_FILE_MAGIC)?;
  buf.write_u32::<LittleEndian>(VECTOR_FILE_VERSION)?;
  buf.write_u32::<LittleEndian>(store.dim() as u32)?;
  buf.write_u8(metric_code(&store.metric()))?;
  buf.write_u8(0)?;
  buf.write_u16::<LittleEndian>(0)?;
  buf.write_u32::<LittleEndian>(store.len() as u32)?;
  let value_len = store
    .offsets()
    .iter()
    .filter(|&&off| off != u32::MAX)
    .count();
  buf.write_u32::<LittleEndian>(value_len as u32)?;
  for off in store.offsets().iter() {
    buf.write_u32::<LittleEndian>(*off)?;
  }
  let values = store.values();
  for v in values.iter() {
    buf.write_f32::<LittleEndian>(*v)?;
  }
  storage.write_all(path, &buf)
}

#[cfg(feature = "vectors")]
fn read_vector_file(
  storage: &dyn Storage,
  path: &Path,
  expected_docs: usize,
  expected_dim: usize,
  expected_metric: &ApiVectorMetric,
) -> Result<VectorStore> {
  let bytes = storage.read_to_end(path)?;
  let mut cursor = Cursor::new(bytes);
  let magic = cursor.read_u32::<LittleEndian>()?;
  if magic != VECTOR_FILE_MAGIC {
    bail!("invalid vector file magic for {:?}", path);
  }
  let version = cursor.read_u32::<LittleEndian>()?;
  if version != VECTOR_FILE_VERSION {
    bail!("unsupported vector file version {} for {:?}", version, path);
  }
  let dim = cursor.read_u32::<LittleEndian>()? as usize;
  if dim != expected_dim {
    bail!(
      "vector dim mismatch for {:?}: expected {}, found {}",
      path,
      expected_dim,
      dim
    );
  }
  let metric_code_raw = cursor.read_u8()?;
  let Some(metric) = metric_from_code(metric_code_raw) else {
    bail!(
      "unknown vector metric code {} in {:?}",
      metric_code_raw,
      path
    );
  };
  if &metric != expected_metric {
    bail!(
      "vector metric mismatch for {:?}: expected {:?}, found {:?}",
      path,
      expected_metric,
      metric
    );
  }
  // skip reserved bytes
  let _ = cursor.read_u8()?;
  let _ = cursor.read_u16::<LittleEndian>()?;
  let doc_count = cursor.read_u32::<LittleEndian>()? as usize;
  if doc_count != expected_docs {
    bail!(
      "vector doc count mismatch for {:?}: expected {}, found {}",
      path,
      expected_docs,
      doc_count
    );
  }
  let vector_count = cursor.read_u32::<LittleEndian>()? as usize;
  let mut offsets = Vec::with_capacity(doc_count);
  for _ in 0..doc_count {
    offsets.push(cursor.read_u32::<LittleEndian>()?);
  }
  let mut values = Vec::with_capacity(vector_count.saturating_mul(dim));
  for _ in 0..vector_count.saturating_mul(dim) {
    values.push(cursor.read_f32::<LittleEndian>()?);
  }
  Ok(VectorStore::new(dim, metric, offsets, values))
}

fn collect_checksums(storage: &dyn Storage, paths: &SegmentPaths) -> Result<HashMap<String, u32>> {
  let mut map = HashMap::new();
  for (name, path_str) in [
    ("terms", &paths.terms),
    ("postings", &paths.postings),
    ("docstore", &paths.docstore),
    ("fast", &paths.fast),
    ("meta", &paths.meta),
  ] {
    let buf = storage.read_to_end(Path::new(path_str))?;
    map.insert(name.to_string(), checksum(&buf));
  }
  Ok(map)
}

fn verify_checksums(
  storage: &dyn Storage,
  meta: &SegmentMeta,
  _seg_meta: &SegmentFileMeta,
  seg_meta_bytes: &[u8],
) -> Result<()> {
  let verify = |label: &str, path: &Path, expected: Option<&u32>, data: Option<&[u8]>| {
    if let Some(expected) = expected {
      let actual = if let Some(bytes) = data {
        checksum(bytes)
      } else {
        checksum(&storage.read_to_end(path)?)
      };
      if actual != *expected {
        bail!(
          "segment {} failed checksum for {} (expected {}, found {})",
          meta.id,
          label,
          expected,
          actual
        );
      }
    }
    Ok(())
  };
  verify(
    "meta",
    Path::new(&meta.paths.meta),
    meta.checksums.get("meta"),
    Some(seg_meta_bytes),
  )?;
  verify(
    "terms",
    Path::new(&meta.paths.terms),
    meta.checksums.get("terms"),
    None,
  )?;
  verify(
    "postings",
    Path::new(&meta.paths.postings),
    meta.checksums.get("postings"),
    None,
  )?;
  verify(
    "docstore",
    Path::new(&meta.paths.docstore),
    meta.checksums.get("docstore"),
    None,
  )?;
  verify(
    "fast fields",
    Path::new(&meta.paths.fast),
    meta.checksums.get("fast"),
    None,
  )?;
  #[cfg(feature = "vectors")]
  {
    let seg_meta = _seg_meta;
    if let Some(dir) = meta.paths.vector_dir.as_deref() {
      if !dir.is_empty() {
        for field in seg_meta.vector_fields.keys() {
          let (vec_path, hnsw_path) = vector_paths(&meta.paths, field)?;
          verify(
            &format!("vector {} bin", field),
            &vec_path,
            meta.checksums.get(&format!("vector_{}_bin", field)),
            None,
          )?;
          verify(
            &format!("vector {} hnsw", field),
            &hnsw_path,
            meta.checksums.get(&format!("vector_{}_hnsw", field)),
            None,
          )?;
        }
      }
    }
  }
  Ok(())
}

#[cfg(feature = "vectors")]
struct VectorFieldReader {
  store: Arc<VectorStore>,
  index: HnswIndex,
}

pub struct SegmentReader {
  pub meta: SegmentMeta,
  terms: TinyTerms,
  postings: RefCell<Box<dyn StorageFile>>,
  docstore: RefCell<DocStoreReader<Box<dyn StorageFile>>>,
  doc_ids: Vec<String>,
  deleted: FastHashSet<DocId>,
  fast_fields: FastFieldsReader,
  keep_positions: bool,
  seg_meta: SegmentFileMeta,
  #[cfg(feature = "vectors")]
  vectors: HashMap<String, VectorFieldReader>,
}

impl SegmentReader {
  pub fn open(storage: Arc<dyn Storage>, meta: SegmentMeta, keep_positions: bool) -> Result<Self> {
    let seg_meta_bytes = storage.read_to_end(Path::new(&meta.paths.meta))?;
    let seg_meta: SegmentFileMeta = serde_json::from_slice(&seg_meta_bytes)?;
    #[cfg(not(feature = "zstd"))]
    if seg_meta.use_zstd {
      bail!(
        "segment {} uses zstd-compressed docstore, but this build was compiled without the `zstd` feature; rebuild with `--features zstd` or reindex without compression",
        meta.id
      );
    }
    verify_checksums(storage.as_ref(), &meta, &seg_meta, &seg_meta_bytes)?;
    let terms = read_terms(storage.as_ref(), Path::new(&meta.paths.terms))?;
    let postings = storage.open_read(Path::new(&meta.paths.postings))?;
    let doc_file = storage.open_read(Path::new(&meta.paths.docstore))?;
    if seg_meta.doc_ids.len() != seg_meta.doc_offsets.len() {
      bail!(
        "segment {} is missing document ids; reindex or re-commit documents with doc_id support",
        meta.id
      );
    }
    #[cfg(not(feature = "zstd"))]
    if seg_meta.use_zstd {
      eprintln!(
        "warning: index uses zstd-compressed docstore, but this binary was built without the `zstd` feature; stored fields may be unavailable"
      );
    }
    let docstore = DocStoreReader::new(doc_file, seg_meta.doc_offsets.clone(), seg_meta.use_zstd);
    let fast_fields = FastFieldsReader::open(storage.as_ref(), Path::new(&meta.paths.fast))?;
    let deleted: FastHashSet<DocId> = meta.deleted_docs.iter().copied().collect();
    #[cfg(feature = "vectors")]
    let mut vector_fields = HashMap::new();
    #[cfg(feature = "vectors")]
    {
      for (field, vmeta) in seg_meta.vector_fields.iter() {
        let (vec_path, hnsw_path) = vector_paths(&meta.paths, field)?;
        let expected_metric: ApiVectorMetric = vmeta.metric.clone().into();
        let store = read_vector_file(
          storage.as_ref(),
          &vec_path,
          meta.doc_count as usize,
          vmeta.dim,
          &expected_metric,
        )?;
        let graph_bytes = storage.read_to_end(&hnsw_path)?;
        let graph: HnswGraph = bincode::options()
          .with_fixint_encoding()
          .deserialize(&graph_bytes)
          .or_else(|_| serde_json::from_slice(&graph_bytes))
          .map_err(|e| {
            anyhow!(
              "failed to read HNSW graph for field {} in segment {}: {}",
              field,
              meta.id,
              e
            )
          })?;
        if graph.dim != vmeta.dim || graph.metric != expected_metric {
          bail!(
            "vector index metadata mismatch for field {} in segment {}",
            field,
            meta.id
          );
        }
        let store_arc = Arc::new(store);
        let index = HnswIndex::from_graph(graph, store_arc.clone());
        vector_fields.insert(
          field.clone(),
          VectorFieldReader {
            store: store_arc,
            index,
          },
        );
      }
    }
    Ok(Self {
      meta,
      terms: TinyTerms(terms),
      postings: RefCell::new(postings),
      docstore: RefCell::new(docstore),
      doc_ids: seg_meta.doc_ids.clone(),
      deleted,
      fast_fields,
      keep_positions,
      seg_meta,
      #[cfg(feature = "vectors")]
      vectors: vector_fields,
    })
  }

  pub fn postings(&self, term: &str) -> Option<PostingsReader> {
    let offset = self.terms.0.get(term)?;
    let mut file = self.postings.borrow_mut();
    PostingsReader::read_at(&mut *file, offset, self.keep_positions).ok()
  }

  pub fn doc_freq(&self, term: &str) -> Option<u32> {
    let offset = self.terms.0.get(term)?;
    let mut file = self.postings.borrow_mut();
    read_doc_freq(&mut *file, offset).ok()
  }

  pub fn terms_with_prefix<'a>(&'a self, prefix: &'a str) -> impl Iterator<Item = &'a String> + 'a {
    self.terms.0.iter_prefix(prefix).map(|(term, _)| term)
  }

  pub fn avg_field_length(&self, field: &str) -> f32 {
    self
      .seg_meta
      .avg_field_lengths
      .get(field)
      .copied()
      .unwrap_or(0.0)
  }

  pub fn get_doc(&self, doc_id: DocId) -> Result<serde_json::Value> {
    self.docstore.borrow_mut().get(doc_id)
  }

  pub fn doc_id(&self, doc_id: DocId) -> Option<&str> {
    self.doc_ids.get(doc_id as usize).map(|s| s.as_str())
  }

  pub fn is_deleted(&self, doc_id: DocId) -> bool {
    self.deleted.contains(&doc_id)
  }

  pub fn live_docs(&self) -> u32 {
    self
      .meta
      .doc_count
      .saturating_sub(self.deleted.len() as u32)
  }

  pub fn fast_fields(&self) -> &FastFieldsReader {
    &self.fast_fields
  }

  #[cfg(feature = "vectors")]
  pub fn vector(&self, field: &str, doc_id: DocId) -> Option<Vec<f32>> {
    self
      .vectors
      .get(field)
      .and_then(|vf| vf.store.vector(doc_id).map(|v| v.to_vec()))
  }

  #[cfg(feature = "vectors")]
  pub fn vector_components(&self, field: &str) -> Option<(&HnswIndex, Arc<VectorStore>)> {
    self
      .vectors
      .get(field)
      .map(|vf| (&vf.index, vf.store.clone()))
  }
}

struct TinyTerms(TinyFst);

use crate::util::fst::TinyFst;

#[cfg(test)]
mod tests {
  use super::*;
  use crate::api::types::{Document, Schema};
  use std::collections::HashMap;
  use tempfile::tempdir;

  fn sample_schema() -> Schema {
    Schema {
      doc_id_field: crate::index::manifest::default_doc_id_field(),
      analyzers: Vec::new(),
      text_fields: vec![crate::index::manifest::TextField {
        name: "body".into(),
        analyzer: "default".into(),
        search_analyzer: None,
        stored: true,
        indexed: true,
        nullable: false,
        search_as_you_type: None,
      }],
      keyword_fields: vec![crate::index::manifest::KeywordField {
        name: "tag".into(),
        stored: true,
        indexed: true,
        fast: true,
        nullable: false,
      }],
      numeric_fields: vec![crate::index::manifest::NumericField {
        name: "year".into(),
        i64: true,
        fast: true,
        stored: true,
        nullable: false,
      }],
      nested_fields: Vec::new(),
      #[cfg(feature = "vectors")]
      vector_fields: Vec::new(),
    }
  }

  fn doc(body: &str, tag: &str, year: i64) -> Document {
    Document {
      fields: [
        (
          "_id".into(),
          serde_json::json!(format!("{body}-{tag}-{year}")),
        ),
        ("body".into(), serde_json::json!(body)),
        ("tag".into(), serde_json::json!(tag)),
        ("year".into(), serde_json::json!(year)),
      ]
      .into_iter()
      .collect(),
    }
  }

  #[test]
  fn writes_and_reads_segment() {
    let dir = tempdir().unwrap();
    let schema = sample_schema();
    let storage = Arc::new(crate::storage::FsStorage::new(dir.path().to_path_buf()));
    let writer = SegmentWriter::new(dir.path(), &schema, true, false, storage.clone());
    let meta = writer
      .write_segment(
        &[
          doc("Rust search engine", "news", 2024),
          doc("Rust language", "tech", 2023),
        ],
        1,
      )
      .unwrap();
    let reader = SegmentReader::open(storage, meta.clone(), true).unwrap();
    let postings = reader.postings("body:rust").unwrap();
    assert_eq!(postings.len(), 2);
    let fast = reader.fast_fields();
    assert!(fast.matches_keyword("tag", 0, "news"));
    assert!(fast.matches_i64_range("year", 1, 2020, 2024));
    let stored_doc = reader.get_doc(0).unwrap();
    assert_eq!(stored_doc["tag"], "news");
    assert!(reader.avg_field_length("body") > 0.0);
  }

  #[test]
  fn writes_segment_from_iterator() {
    let dir = tempdir().unwrap();
    let schema = sample_schema();
    let storage = Arc::new(crate::storage::FsStorage::new(dir.path().to_path_buf()));
    let writer = SegmentWriter::new(dir.path(), &schema, true, false, storage.clone());
    let docs = vec![
      doc("Iter body one", "alpha", 2022),
      doc("Iter body two", "beta", 2023),
    ]
    .into_iter()
    .map(Ok);
    let meta = writer.write_segment_from_iter(docs, 2).unwrap();
    assert_eq!(meta.doc_count, 2);
    assert_eq!(meta.max_doc_id, 1);
    let reader = SegmentReader::open(storage, meta, true).unwrap();
    assert_eq!(reader.doc_id(0), Some("Iter body one-alpha-2022"));
    assert_eq!(reader.doc_id(1), Some("Iter body two-beta-2023"));
  }

  #[test]
  fn rejects_unknown_fields() {
    let dir = tempdir().unwrap();
    let schema = sample_schema();
    let storage = Arc::new(crate::storage::FsStorage::new(dir.path().to_path_buf()));
    let writer = SegmentWriter::new(dir.path(), &schema, true, false, storage);
    let mut bad_doc = doc("Rust search engine", "news", 2024);
    bad_doc
      .fields
      .insert("unexpected".into(), serde_json::json!("oops"));
    let err = writer.write_segment(&[bad_doc], 1).unwrap_err();
    assert!(
      err.to_string().contains("unknown field unexpected"),
      "unexpected error: {err}"
    );
  }

  #[test]
  fn computes_average_lengths() {
    let lengths = HashMap::from([("body".to_string(), 4u64), ("title".to_string(), 0u64)]);
    let avg = compute_avg_lengths(&lengths, 2);
    assert_eq!(avg.get("body"), Some(&2.0));
    assert_eq!(avg.get("title"), Some(&0.0));
  }

  #[cfg(not(feature = "zstd"))]
  #[test]
  fn opening_zstd_segment_without_feature_errors() {
    let dir = tempdir().unwrap();
    let paths = directory::segment_paths(dir.path(), "zstd");
    let seg_file_meta = SegmentFileMeta {
      doc_offsets: Vec::new(),
      doc_ids: Vec::new(),
      avg_field_lengths: HashMap::new(),
      use_zstd: true,
      #[cfg(feature = "vectors")]
      vector_fields: HashMap::new(),
    };
    std::fs::write(&paths.meta, serde_json::to_vec(&seg_file_meta).unwrap()).unwrap();
    let storage = Arc::new(crate::storage::FsStorage::new(dir.path().to_path_buf()));
    let meta = crate::index::manifest::SegmentMeta {
      id: "zstd".into(),
      generation: 1,
      paths,
      doc_count: 0,
      max_doc_id: 0,
      blockmax: true,
      deleted_docs: Vec::new(),
      avg_field_lengths: HashMap::new(),
      checksums: HashMap::new(),
    };
    let err = SegmentReader::open(storage, meta, true);
    assert!(err.is_err(), "expected zstd error for missing feature");
    let err = err.err().unwrap();
    assert!(
      err.to_string().contains("zstd"),
      "expected a clear zstd feature error, got {err}"
    );
  }
}
