use std::borrow::Cow;
use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine as _;
#[cfg(feature = "vectors")]
use bincode::Options;
#[cfg(feature = "vectors")]
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use hashbrown::{HashMap as FastHashMap, HashSet as FastHashSet};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::api::types::{ChecksumAuditFailureHook, ChecksumPolicy, Document, IndexOptions};
use crate::storage::{BlobStore, Object as BlobObject, StorageAsBlobStore};
#[cfg(feature = "vectors")]
use crate::api::types::VectorMetric as ApiVectorMetric;
use crate::index::directory;
use crate::index::docstore::{decode_docstore_record, DocStoreWriter, MAX_DOCSTORE_BYTES};
use crate::index::fastfields::{
  doc_length_key, nested_count_key, nested_parent_key, FastFieldsReader, FastFieldsWriter,
  FastValue,
};
use crate::index::manifest::{
  FieldKind, NestedField, NestedProperty, ResolvedField, Schema, SegmentMeta, SegmentPaths,
};
use crate::index::postings::{read_doc_freq, InvertedIndexBuilder, PostingsReader, PostingsWriter};
use crate::index::terms::{read_terms, write_terms};
use crate::storage::Storage;
use crate::util::case_fold::fold_keyword;
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
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub write_binding_b64: Option<String>,
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
      let base_count = *collected.nested_counts.get(prefix).unwrap_or(&0);
      let mut non_null_count = 0usize;
      for v in arr.iter() {
        if v.is_null() {
          if nested.nullable {
            continue;
          }
          bail!("nested field {prefix} cannot be null");
        }
        if !v.is_object() {
          bail!("nested field {prefix} must contain objects");
        }
        non_null_count = non_null_count.saturating_add(1);
      }

      let next_count = base_count.saturating_add(non_null_count);
      collected
        .nested_counts
        .insert(prefix.to_string(), next_count);
      let parent = parent_idx.unwrap_or(usize::MAX);
      let entry = collected
        .nested_parents
        .entry(prefix.to_string())
        .or_default();
      if entry.len() < base_count {
        entry.resize(base_count, usize::MAX);
      }
      entry.extend(std::iter::repeat_n(parent, non_null_count));
      let mut object_idx = base_count;
      for v in arr.iter() {
        if v.is_null() {
          continue;
        }
        let map = v
          .as_object()
          .ok_or_else(|| anyhow!("nested field {prefix} must contain objects"))?;
        collect_nested_object(schema, nested, map, prefix, object_idx, collected, resolved)?;
        object_idx = object_idx.saturating_add(1);
      }
    }
    serde_json::Value::Object(map) => {
      let base_count = *collected.nested_counts.get(prefix).unwrap_or(&0);
      collected
        .nested_counts
        .insert(prefix.to_string(), base_count.saturating_add(1));
      collected
        .nested_parents
        .entry(prefix.to_string())
        .or_default()
        .push(parent_idx.unwrap_or(usize::MAX));
      collect_nested_object(schema, nested, map, prefix, base_count, collected, resolved)?;
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
  for prop in nested.fields.iter() {
    if map.contains_key(prop.name()) {
      continue;
    }
    if prop.is_nullable() {
      continue;
    }
    bail!("missing required nested field {}.{}", prefix, prop.name());
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
    let num_f32 = num as f32;
    if !num_f32.is_finite() {
      bail!("vector field {field} contains non-finite component");
    }
    vecvals.push(num_f32);
  }
  if vecvals.len() != vf.dim {
    bail!(
      "vector field {field} expected dimension {}, got {}",
      vf.dim,
      vecvals.len()
    );
  }
  // BUG-386: reject any vector whose squared magnitudes sum past `f32::MAX`,
  // regardless of metric. Each component passes the per-value finitude check
  // above, but their sum-of-squares can still overflow (e.g. `[3e19, 3e19]`).
  //
  // For cosine this originally surfaced as `normalize_in_place` dividing by
  // `+inf` and silently zeroing the vector (BUG-384); that fix rejected only
  // the cosine path. For L2 the same overflow is just as user-visible — the
  // vector is persisted, then `l2_distance(v, 0)` accumulates `v[i]^2` into
  // `+inf`, `metric_similarity(L2) = -inf`, and `compute_hybrid_score` drops
  // the doc via the BUG-328 guard so the caller silently gets an empty / wrong
  // hit set. Guarding both metrics uniformly gives a diagnosable failure at
  // write time instead of invisible misbehaviour at read time.
  let sum_sq = vecvals.iter().map(|v| v * v).sum::<f32>();
  if !sum_sq.is_finite() {
    bail!(
      "vector field {field} has components whose sum-of-squares overflows f32; reduce component magnitudes"
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
  write_binding: Option<Vec<u8>>,
}

impl<'a> SegmentWriter<'a> {
  pub fn new(
    root: &'a Path,
    schema: &'a Schema,
    enable_positions: bool,
    use_zstd: bool,
    storage: Arc<dyn Storage>,
    write_binding: Option<Vec<u8>>,
  ) -> Self {
    Self {
      root,
      schema,
      enable_positions,
      use_zstd,
      storage,
      write_binding,
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
            // `fold_keyword` returns a borrowed `Cow` for already-lowercase
            // ASCII input, so we probe `seen_terms` before allocating and
            // only materialize an owned `String` on the insert path. This
            // preserves the ASCII fast path for the common duplicate case.
            let lower = fold_keyword(value);
            if !seen_terms.contains(lower.as_ref()) {
              let mut term_key = String::with_capacity(field.len() + lower.len() + 1);
              term_key.push_str(field);
              term_key.push(':');
              term_key.push_str(&lower);
              postings_builder.add_term(&term_key, doc_ord, 0, false);
              seen_terms.insert(lower.into_owned());
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
        // Match write_vector_file: the HNSW graph is an immutable segment
        // artifact referenced from the manifest, so it must go through the
        // same tmp-write → fsync → rename → fsync-dir path to avoid a
        // truncated live file if the write is interrupted.
        self.storage.atomic_write(&hnsw_path, &graph_bytes)?;
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
      write_binding_b64: self
        .write_binding
        .as_ref()
        .map(|b| base64::engine::general_purpose::STANDARD.encode(b)),
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
      checksums.insert(format!("vector_{field}_bin"), checksum(&vec_buf));
      checksums.insert(format!("vector_{field}_hnsw"), checksum(&hnsw_buf));
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
      write_binding_b64: self
        .write_binding
        .as_ref()
        .map(|b| base64::engine::general_purpose::STANDARD.encode(b)),
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

/// Serialize a [`VectorStore`] to its on-disk representation and persist it
/// through the storage's atomic-write path.
///
/// The vector file is an immutable segment artifact referenced from the
/// manifest. The manifest itself is written with [`Storage::atomic_write`], so
/// using the same tmp-write → fsync → rename → fsync-dir pattern here keeps
/// the two artifacts on equivalent durability and atomicity footing: a reader
/// recovering after a crash either sees the complete vector file or no file
/// at that path, never a truncated body with a valid magic/version header.
/// By contrast, `storage.write_all` writes/truncates the live path in place,
/// so an interrupted write can leave a partially-written file visible to
/// readers even if the written bytes are later synced.
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
  storage.atomic_write(path, &buf)
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
    bail!("invalid vector file magic for {path:?}");
  }
  let version = cursor.read_u32::<LittleEndian>()?;
  if version != VECTOR_FILE_VERSION {
    bail!("unsupported vector file version {version} for {path:?}");
  }
  let dim = cursor.read_u32::<LittleEndian>()? as usize;
  if dim != expected_dim {
    bail!("vector dim mismatch for {path:?}: expected {expected_dim}, found {dim}");
  }
  let metric_code_raw = cursor.read_u8()?;
  let Some(metric) = metric_from_code(metric_code_raw) else {
    bail!("unknown vector metric code {metric_code_raw} in {path:?}");
  };
  if &metric != expected_metric {
    bail!("vector metric mismatch for {path:?}: expected {expected_metric:?}, found {metric:?}");
  }
  // skip reserved bytes
  let _ = cursor.read_u8()?;
  let _ = cursor.read_u16::<LittleEndian>()?;
  let doc_count = cursor.read_u32::<LittleEndian>()? as usize;
  if doc_count != expected_docs {
    bail!("vector doc count mismatch for {path:?}: expected {expected_docs}, found {doc_count}");
  }
  let vector_count = cursor.read_u32::<LittleEndian>()? as usize;
  // `vector_count` is the number of dense rows stored in the values block.
  // Every row is pointed at by at most one u32 offset in the `offsets` table,
  // so it can never legally exceed `doc_count`. Rejecting early prevents a
  // crafted/corrupt header from driving the capacity / read-loop bound off a
  // multi-gigabyte allocation before we discover the file is too short.
  if vector_count > doc_count {
    bail!("vector file {path:?}: vector_count {vector_count} exceeds doc_count {doc_count}");
  }
  let mut offsets = Vec::with_capacity(doc_count);
  for _ in 0..doc_count {
    offsets.push(cursor.read_u32::<LittleEndian>()?);
  }
  // Validate the implied `vector_count * dim * sizeof(f32)` byte footprint
  // against the bytes still in the buffer. This catches both `usize` overflow
  // (via `checked_mul`) and a header whose counts would require more bytes
  // than the file actually contains, before any allocation happens.
  let values_len = vector_count
    .checked_mul(dim)
    .ok_or_else(|| anyhow!("vector file {path:?}: vector_count * dim overflows usize"))?;
  let values_bytes = values_len
    .checked_mul(std::mem::size_of::<f32>())
    .ok_or_else(|| anyhow!("vector file {path:?}: values byte size overflows usize"))?;
  let total_len = cursor.get_ref().len();
  let position = cursor.position() as usize;
  let remaining = total_len.saturating_sub(position);
  if values_bytes > remaining {
    bail!("vector file {path:?} claims {values_bytes} value bytes but only {remaining} remain");
  }
  let mut values = Vec::with_capacity(values_len);
  for _ in 0..values_len {
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
            &format!("vector {field} bin"),
            &vec_path,
            meta.checksums.get(&format!("vector_{field}_bin")),
            None,
          )?;
          verify(
            &format!("vector {field} hnsw"),
            &hnsw_path,
            meta.checksums.get(&format!("vector_{field}_hnsw")),
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

/// Immutable, parsed-once segment data shared across `IndexReader` instances.
///
/// `SegmentCore` is the cacheable half of a segment: terms dictionary, fast
/// fields, doc-id table, vectors, and segment-level metadata. Once a core is
/// loaded for a given (segment_id, content fingerprint), it can be reused by
/// any number of `SegmentReader` views without re-reading or re-parsing the
/// underlying files. Per-manifest mutable state (`deleted_docs`), the
/// resolved file paths (which may move under a future portable-manifest
/// scheme), and per-view stateful file handles all live on `SegmentReader`,
/// not here, so the core stays safe to share via `Arc` and survives manifest
/// rewrites that don't change segment file content.
pub struct SegmentCore {
  seg_meta: SegmentFileMeta,
  terms: TinyTerms,
  doc_ids: Vec<Arc<str>>,
  fast_fields: FastFieldsReader,
  keep_positions: bool,
  #[cfg(feature = "vectors")]
  vectors: HashMap<String, VectorFieldReader>,
}

/// Bundle of per-load options threaded through `SegmentCore::load` and
/// `SegmentCache::get_or_load`. Cheap to clone (one bool, one enum, one
/// `Option<Arc<…>>`).
#[derive(Clone)]
pub struct SegmentLoadCtx {
  pub keep_positions: bool,
  pub checksum_policy: ChecksumPolicy,
  pub audit_hook: Option<ChecksumAuditFailureHook>,
}

impl SegmentLoadCtx {
  pub fn from_options(options: &IndexOptions) -> Self {
    Self {
      keep_positions: options.enable_positions,
      checksum_policy: options.checksum_policy,
      audit_hook: options.checksum_audit_failure_hook.clone(),
    }
  }

  /// Strict-policy default with no audit hook. Used by the legacy
  /// `SegmentReader::open(storage, meta, keep_positions)` convenience
  /// constructor (which existing tests depend on) so callers that don't
  /// care about checksum policy keep the strictest behavior.
  fn strict(keep_positions: bool) -> Self {
    Self {
      keep_positions,
      checksum_policy: ChecksumPolicy::Strict,
      audit_hook: None,
    }
  }
}

impl SegmentCore {
  /// Load a segment's immutable data from storage. Performs term-dict load,
  /// fast-field parse, and vector load unconditionally; whole-file checksum
  /// verification is gated by `ctx.checksum_policy`:
  /// - `Strict`: verify all five segment files synchronously (existing
  ///   behavior); fails the load on mismatch.
  /// - `TrustManifest`: skip whole-file verification entirely. Postings and
  ///   docstore are NOT read during load — they're opened lazily by
  ///   per-view handles in `SegmentReader::from_core`.
  /// - `Audit`: skip synchronous verification but dispatch a background
  ///   task (via `rayon::spawn`) that re-runs `verify_checksums` and
  ///   surfaces failures via `ctx.audit_hook` (or `log::error!` if none).
  ///
  /// The returned `Arc` does not retain `storage` — view-level handles in
  /// `SegmentReader::from_core` are opened against the manifest-current
  /// storage, so a later manifest that relocates files (Stage 9) can still
  /// reuse this core without serving reads through stale paths.
  pub fn load(
    storage: Arc<dyn Storage>,
    meta: &SegmentMeta,
    ctx: &SegmentLoadCtx,
  ) -> Result<Arc<Self>> {
    let seg_meta_bytes = storage.read_to_end(Path::new(&meta.paths.meta))?;
    let mut seg_meta: SegmentFileMeta = serde_json::from_slice(&seg_meta_bytes)?;
    #[cfg(not(feature = "zstd"))]
    if seg_meta.use_zstd {
      bail!(
        "segment {} uses zstd-compressed docstore, but this build was compiled without the `zstd` feature; rebuild with `--features zstd` or reindex without compression",
        meta.id
      );
    }
    match ctx.checksum_policy {
      ChecksumPolicy::Strict => {
        verify_checksums(storage.as_ref(), meta, &seg_meta, &seg_meta_bytes)?;
      }
      ChecksumPolicy::TrustManifest => {
        // Manifest is the trust anchor. No whole-file reads beyond what
        // the data path itself requires (terms / fast fields / vectors).
        // Postings and docstore are not touched during load.
      }
      ChecksumPolicy::Audit => {
        // Open succeeds immediately; a background task re-runs the same
        // verification and surfaces any mismatch via the audit hook (or
        // `log::error!`). Captures clones of the inputs because the
        // foreground load must not block on this.
        dispatch_checksum_audit(
          storage.clone(),
          meta.clone(),
          seg_meta.clone(),
          seg_meta_bytes.clone(),
          ctx.audit_hook.clone(),
        );
      }
    }
    let terms = read_terms(storage.as_ref(), Path::new(&meta.paths.terms))?;
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
    let fast_fields = FastFieldsReader::open(storage.as_ref(), Path::new(&meta.paths.fast))?;
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
    let doc_ids: Vec<Arc<str>> = std::mem::take(&mut seg_meta.doc_ids)
      .into_iter()
      .map(Arc::<str>::from)
      .collect();
    // `storage` was used above for the load reads but is intentionally not
    // retained on the core: per-view file handles are opened against the
    // manifest-current storage in `SegmentReader::from_core`.
    let _ = storage;
    Ok(Arc::new(Self {
      seg_meta,
      terms: TinyTerms(terms),
      doc_ids,
      fast_fields,
      keep_positions: ctx.keep_positions,
      #[cfg(feature = "vectors")]
      vectors: vector_fields,
    }))
  }
}

/// Re-verify a cached `SegmentCore`'s on-disk segment files against the
/// manifest's recorded checksums. Called on every `Strict` cache hit so
/// the policy's pre-Stage-1 guarantee holds: a second `Index::reader()`
/// against the same `Index` detects external mutation that happened
/// between the first reader open and the second.
///
/// The cached core's parsed structures (terms, fast fields, doc-id table)
/// are reused — verification re-reads the files but does not re-parse
/// them. The bytes are dropped after CRC computation.
fn verify_cached_core(
  storage: &dyn Storage,
  meta: &SegmentMeta,
  core: &SegmentCore,
) -> Result<()> {
  // `seg_meta_bytes` is not retained on the cached core (the parsed
  // `SegmentFileMeta` is). Re-read it so `verify_checksums` can compare
  // the live meta-file bytes against the manifest's recorded checksum;
  // this catches the "external mutation invalidated the meta file"
  // case as well as the postings/docstore/fast/terms cases.
  let seg_meta_bytes = storage.read_to_end(Path::new(&meta.paths.meta))?;
  verify_checksums(storage, meta, &core.seg_meta, &seg_meta_bytes)
}

/// Background checksum audit dispatched by `ChecksumPolicy::Audit`. Runs on
/// the global `rayon` thread pool (bounded to roughly `num_cpus` workers by
/// default), so opening an index with hundreds of segments under `Audit`
/// won't fan out to hundreds of OS threads.
///
/// Re-runs `verify_checksums` against the same storage with the segment
/// metadata that the foreground load already parsed. On failure, invokes
/// `audit_hook` if provided; otherwise emits a `log::error!`. Successes are
/// silent (no allocation, no log noise).
fn dispatch_checksum_audit(
  storage: Arc<dyn Storage>,
  meta: SegmentMeta,
  seg_meta: SegmentFileMeta,
  seg_meta_bytes: Vec<u8>,
  audit_hook: Option<ChecksumAuditFailureHook>,
) {
  rayon::spawn(move || {
    let segment_id = meta.id.clone();
    if let Err(err) = verify_checksums(storage.as_ref(), &meta, &seg_meta, &seg_meta_bytes) {
      match audit_hook {
        Some(hook) => hook.invoke(&segment_id, &err),
        None => log::error!("checksum audit failed for segment {segment_id}: {err:#}"),
      }
    }
  });
}

/// Cache key for `SegmentCore`. The fingerprint hashes `SegmentMeta.checksums`
/// deterministically; if any underlying segment file changes, its checksum
/// changes, so the fingerprint changes and the cache misses. The segment id
/// stays in the key so two segments that happen to collide on fingerprint
/// (impossible in practice for `u64`-of-content-hashes, but defended against)
/// don't share a core.
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub(crate) struct SegmentCacheKey {
  pub id: String,
  pub fingerprint: u64,
}

impl SegmentCacheKey {
  pub fn from_meta(meta: &SegmentMeta) -> Self {
    Self {
      id: meta.id.clone(),
      fingerprint: fingerprint_checksums(&meta.checksums),
    }
  }
}

fn fingerprint_checksums(checksums: &HashMap<String, u32>) -> u64 {
  use std::collections::hash_map::DefaultHasher;
  use std::hash::{Hash, Hasher};
  // `HashMap` iteration order is non-deterministic, so sort first to keep the
  // fingerprint stable across processes and runs.
  let mut entries: Vec<(&String, &u32)> = checksums.iter().collect();
  entries.sort_by(|a, b| a.0.cmp(b.0));
  let mut hasher = DefaultHasher::new();
  for (k, v) in entries {
    k.hash(&mut hasher);
    v.hash(&mut hasher);
  }
  hasher.finish()
}

/// Default capacity of the per-`InnerIndex` segment-core cache. Sized to
/// comfortably hold every segment in any reasonable working index without
/// touching evictions in normal operation; capacity is shared across
/// generations so compaction churn can push older entries out before a new
/// reader's references protect them.
const DEFAULT_SEGMENT_CACHE_CAPACITY: usize = 1024;

/// Process-wide cache of immutable `SegmentCore`s, owned by `InnerIndex`.
///
/// Stores `Arc<SegmentCore>` (strong refs) under a bounded-LRU policy so that
/// the *typical* sequential reader pattern — open `IndexReader`, serve a
/// query, drop the reader, open another reader for the next request — hits
/// the cache instead of reloading from storage. The previous `Weak`-only
/// design only retained cores while at least one `SegmentReader` view was
/// alive, which collapses to a no-op for non-overlapping reader lifecycles.
///
/// In-flight readers hold their own `Arc<SegmentCore>` strong refs, so an
/// LRU eviction (or a manifest rewrite that drops a segment) doesn't yank
/// the core out from under them: the core stays alive until the last reader
/// drops. This is the property that makes the cache safe under
/// `Index::compact` mid-search.
pub struct SegmentCache {
  inner: parking_lot::Mutex<SegmentCacheInner>,
  loads: std::sync::atomic::AtomicUsize,
}

struct SegmentCacheInner {
  capacity: usize,
  entries: HashMap<SegmentCacheKey, CacheEntry>,
  next_seq: u64,
}

struct CacheEntry {
  core: Arc<SegmentCore>,
  /// Monotonic recency tag, bumped on every hit and on insert. The entry
  /// with the smallest `seq` is the LRU candidate for eviction.
  seq: u64,
}

impl Default for SegmentCache {
  fn default() -> Self {
    Self::new()
  }
}

impl SegmentCache {
  pub fn new() -> Self {
    Self::with_capacity(DEFAULT_SEGMENT_CACHE_CAPACITY)
  }

  pub fn with_capacity(capacity: usize) -> Self {
    let capacity = capacity.max(1);
    Self {
      inner: parking_lot::Mutex::new(SegmentCacheInner {
        capacity,
        entries: HashMap::with_capacity(capacity),
        next_seq: 0,
      }),
      loads: std::sync::atomic::AtomicUsize::new(0),
    }
  }

  /// Get an existing core or load a fresh one. Single-flight is best-effort:
  /// under heavy concurrent contention for the same key, two loaders may both
  /// load the same core, and the second insertion's core (equivalent to the
  /// first because they share a fingerprint) is dropped in favor of the
  /// already-cached one.
  pub fn get_or_load(
    &self,
    meta: &SegmentMeta,
    ctx: &SegmentLoadCtx,
    storage: Arc<dyn Storage>,
  ) -> Result<Arc<SegmentCore>> {
    let key = SegmentCacheKey::from_meta(meta);
    {
      let mut inner = self.inner.lock();
      inner.next_seq = inner.next_seq.saturating_add(1);
      let seq = inner.next_seq;
      if let Some(entry) = inner.entries.get_mut(&key) {
        let core = entry.core.clone();
        entry.seq = seq;
        // Drop the cache lock before any I/O.
        drop(inner);
        // Cache-hit semantics by policy:
        // - `Strict`: re-verify the on-disk bytes against the manifest's
        //   recorded checksums. The cached `SegmentCore` (parsed terms,
        //   fast fields, etc.) is reused — we don't re-allocate or re-
        //   parse — but the file contents must still match what the
        //   manifest says they should. This restores the pre-Stage-1
        //   guarantee that two `Index::reader()` calls within a single
        //   process detect external mutation between opens.
        // - `TrustManifest`: return immediately. The manifest is the
        //   trust anchor by definition.
        // - `Audit`: return immediately. Audit is a per-fresh-load
        //   concern; re-firing on every cache hit would produce N*M
        //   audit runs and defeat the bounded-execution model.
        if matches!(ctx.checksum_policy, ChecksumPolicy::Strict) {
          verify_cached_core(storage.as_ref(), meta, &core)?;
        }
        return Ok(core);
      }
    }
    // Slow path: load outside the lock so concurrent misses for *different*
    // keys aren't serialized on this load's I/O.
    let core = SegmentCore::load(storage, meta, ctx)?;
    self
      .loads
      .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut inner = self.inner.lock();
    // Re-check: a peer may have inserted while we loaded.
    inner.next_seq = inner.next_seq.saturating_add(1);
    let seq = inner.next_seq;
    if let Some(entry) = inner.entries.get_mut(&key) {
      entry.seq = seq;
      return Ok(entry.core.clone());
    }
    // Evict the LRU entry if at capacity. Any in-flight reader holding an
    // `Arc<SegmentCore>` for the evicted key keeps that core alive in memory
    // until the reader drops; only the cache's own strong ref goes away.
    if inner.entries.len() >= inner.capacity {
      if let Some(victim) = inner
        .entries
        .iter()
        .min_by_key(|(_, e)| e.seq)
        .map(|(k, _)| k.clone())
      {
        inner.entries.remove(&victim);
      }
    }
    inner.entries.insert(
      key,
      CacheEntry {
        core: core.clone(),
        seq,
      },
    );
    Ok(core)
  }

  /// Number of times a core was actually loaded from storage. Cache hits do
  /// not increment this. Used by tests to assert that segments are loaded at
  /// most once across N reader opens for a stable manifest, and that
  /// tombstone-only commits reuse the existing cached core.
  pub fn loads(&self) -> usize {
    self.loads.load(std::sync::atomic::Ordering::Relaxed)
  }

  #[cfg(test)]
  pub(crate) fn len(&self) -> usize {
    self.inner.lock().entries.len()
  }

  #[cfg(test)]
  pub(crate) fn contains_key(&self, key: &SegmentCacheKey) -> bool {
    self.inner.lock().entries.contains_key(key)
  }
}

/// Per-`IndexReader` view over an immutable `SegmentCore`, plus the
/// manifest-specific state (`deleted_docs`) and per-view handles for
/// postings/docstore reads.
///
/// Stage 8 (a + b): both postings and docstore reads now go through
/// `Arc<dyn Object>` handles opened from `BlobStore::open` (typically
/// wrapping the inner `Storage` via `StorageAsBlobStore`).
///
/// * **Postings** (Stage 8a): each lookup issues a bounded
///   `Object::read_range` driven by `TinyFst::range_for(term, postings_len)`
///   — for cloud backends this becomes a single `bytes=start-end` GET per
///   term instead of a whole-file read.
/// * **Docstore** (Stage 8b): `get_doc` derives the per-doc byte range from
///   the offsets table cached in `SegmentCore::seg_meta.doc_offsets` and
///   issues exactly one `Object::read_range` per fetched doc. The span is
///   bounded by `MAX_DOCSTORE_BYTES + 4` *before* issuing the read so a
///   corrupt offset table can't trigger an oversized GET. See
///   [`SegmentReader::get_doc`] for the full strict-validation contract.
///
/// Object handles deliberately live on the view, not on `SegmentCore` —
/// matching Stage 1's "core stays parsed-only; per-storage state lives
/// on the view" principle so manifest rewrites that relocate files
/// (Stage 9 portable manifest) don't read through stale paths.
pub struct SegmentReader {
  pub(crate) core: Arc<SegmentCore>,
  pub meta: SegmentMeta,
  deleted: FastHashSet<DocId>,
  /// Postings handle pinned at view-open time. `Object::stat()` exposes
  /// the object length we need to drive `TinyFst::range_for`.
  postings: Arc<dyn BlobObject>,
  /// Cached postings object length so `range_for` calls don't re-stat
  /// per term lookup. Equal to `postings.stat().len` at open time.
  postings_len: u64,
  /// Stage 8b: docstore handle pinned at view-open time. `get_doc` issues
  /// exactly one bounded `read_range` call per fetched doc — see the
  /// comment on `get_doc` for the offsets→range derivation and the
  /// strict validation contract.
  docstore: Arc<dyn BlobObject>,
  /// Stage 8b: cached docstore object length, used as the upper bound
  /// for the **last** doc's range (`offsets[N-1]..docstore_len`) so
  /// `get_doc` doesn't re-stat per fetch. Equal to
  /// `docstore.stat().len` at open time.
  docstore_len: u64,
}

impl SegmentReader {
  /// Build a per-manifest view over an already-loaded `SegmentCore`.
  /// Opens fresh per-view handles against the manifest-current
  /// `meta.paths`: postings and docstore both via `BlobStore::open`
  /// (Stages 8a + 8b). `Storage` is no longer needed here — both hot
  /// reads now flow through `BlobStore`.
  pub fn from_core(
    core: Arc<SegmentCore>,
    meta: SegmentMeta,
    blob_store: Arc<dyn BlobStore>,
  ) -> Result<Self> {
    // Stage 8a: postings is opened as an `Object` so `postings()` and
    // `doc_freq()` can issue bounded `read_range` calls using
    // `TinyFst::range_for`. We use `block_on` to bridge the
    // synchronous `from_core` API to the async `BlobStore::open`;
    // this is a transitional bridge documented in the module-level
    // comment of `storage_as_blob.rs` — Stage 8/9 may push async up
    // the call stack, but Stage 8a keeps the read path sync.
    let postings = futures::executor::block_on(
      blob_store.open(Path::new(&meta.paths.postings)),
    )?;
    let postings_len = postings.stat().len;

    // Stage 8b: docstore now uses the same Object shape. `get_doc`
    // computes the per-doc byte range from the offsets table cached
    // in `SegmentCore::seg_meta.doc_offsets` and issues exactly one
    // `Object::read_range` per fetch.
    let docstore = futures::executor::block_on(
      blob_store.open(Path::new(&meta.paths.docstore)),
    )?;
    let docstore_len = docstore.stat().len;

    let deleted: FastHashSet<DocId> = meta.deleted_docs.iter().copied().collect();
    Ok(Self {
      core,
      meta,
      deleted,
      postings,
      postings_len,
      docstore,
      docstore_len,
    })
  }

  /// Convenience constructor that loads a fresh `SegmentCore` under the
  /// strict checksum policy and immediately builds a view from it. Bypasses
  /// any reader-side cache; callers that want caching or a non-strict
  /// policy should go through `Index::reader()`. Wraps `storage` with a
  /// default `StorageAsBlobStore` so the test/legacy convenience API
  /// doesn't need to thread a separate `BlobStore` argument.
  pub fn open(storage: Arc<dyn Storage>, meta: SegmentMeta, keep_positions: bool) -> Result<Self> {
    let ctx = SegmentLoadCtx::strict(keep_positions);
    let core = SegmentCore::load(storage.clone(), &meta, &ctx)?;
    let blob_store: Arc<dyn BlobStore> = Arc::new(StorageAsBlobStore::new(storage));
    Self::from_core(core, meta, blob_store)
  }

  /// Stage 8a: bounded postings range read.
  ///
  /// Looks up the term's offset and length in the in-memory term
  /// dictionary (`TinyFst::range_for`) and issues a single
  /// `Object::read_range(start..end)` against the postings object.
  /// On a cloud backend this becomes one `bytes=start-(end-1)` GET
  /// per term; on local FS it's a single seek+read. Either way the
  /// scan-from-EOF-and-decode logic in `PostingsReader::read_at`
  /// operates on a `Cursor` over the bounded byte slice rather than
  /// the whole postings file.
  pub fn postings(&self, term: &str) -> Option<PostingsReader> {
    let range = self.core.terms.0.range_for(term, self.postings_len)?;
    let bytes = futures::executor::block_on(self.postings.read_range(range)).ok()?;
    let mut cursor = std::io::Cursor::new(bytes);
    PostingsReader::read_at(&mut cursor, 0, self.core.keep_positions).ok()
  }

  /// Stage 8a: tiny range read for the leading `doc_freq` `u32`.
  ///
  /// `doc_freq` is the first 4 bytes of a term's postings payload. We
  /// read just those 4 bytes via a bounded range — substantially less
  /// I/O than the full postings list when the caller only needs the
  /// frequency (e.g. for BM25 stats before deciding whether to
  /// iterate).
  pub fn doc_freq(&self, term: &str) -> Option<u32> {
    let offset = self.core.terms.0.get(term)?;
    let end = offset.checked_add(4)?;
    if end > self.postings_len {
      return None;
    }
    let bytes = futures::executor::block_on(self.postings.read_range(offset..end)).ok()?;
    let mut cursor = std::io::Cursor::new(bytes);
    read_doc_freq(&mut cursor, 0).ok()
  }

  pub fn terms_with_prefix<'a>(&'a self, prefix: &'a str) -> impl Iterator<Item = &'a str> + 'a {
    self.core.terms.0.iter_prefix(prefix).map(|(term, _)| term)
  }

  pub fn avg_field_length(&self, field: &str) -> f32 {
    self
      .core
      .seg_meta
      .avg_field_lengths
      .get(field)
      .copied()
      .unwrap_or(0.0)
  }

  /// Stage 8b: fetch a single stored document via one bounded
  /// `Object::read_range` against the docstore object.
  ///
  /// The byte range is derived from the offset table cached in
  /// `SegmentCore::seg_meta.doc_offsets` (loaded once at segment-load
  /// time): `start = offsets[doc_id]`, and `end = offsets[doc_id + 1]`
  /// for any non-last doc, or `end = docstore_len` for the last doc.
  /// The returned bundle is the exact `[u32 LE length][payload]`
  /// record; parsing/decompression goes through the shared
  /// [`decode_docstore_record`] helper so this path can never drift
  /// from the legacy `DocStoreReader::get` semantics.
  ///
  /// Strict validation (Codex Stage 8b review, including the v2 P1
  /// pre-read span guard):
  /// * `doc_id` must be in bounds for the offset table.
  /// * `start < end <= docstore_len` (rejects empty, inverted, and
  ///   out-of-bounds ranges as corruption).
  /// * `end - start` must be in `[4, MAX_DOCSTORE_BYTES + 4]` —
  ///   enforced **before** issuing `read_range` so a corrupt offset
  ///   table or appended/sparse docstore can't trigger a multi-GB
  ///   object-store GET / Vec allocation before parse-time validation
  ///   has a chance to reject.
  /// * `decode_docstore_record` then enforces (post-read) that the
  ///   embedded length is `<= MAX_DOCSTORE_BYTES` and that
  ///   `4 + embedded_len == fetched_range.len()` — a longer offset-
  ///   table-implied range than the embedded length actually encodes
  ///   is treated as corruption rather than silently ignored.
  pub fn get_doc(&self, doc_id: DocId) -> Result<serde_json::Value> {
    let offsets = &self.core.seg_meta.doc_offsets;
    let idx = doc_id as usize;
    let start = *offsets.get(idx).ok_or_else(|| {
      anyhow!(
        "doc id {doc_id} out of bounds: offsets table has {} entries",
        offsets.len()
      )
    })?;
    let end = offsets
      .get(idx + 1)
      .copied()
      .unwrap_or(self.docstore_len);
    if start >= end {
      bail!(
        "docstore: invalid range {start}..{end} for doc {doc_id} \
         (offsets must be strictly increasing within the file)"
      );
    }
    if end > self.docstore_len {
      bail!(
        "docstore: range {start}..{end} for doc {doc_id} exceeds \
         docstore object length {}",
        self.docstore_len
      );
    }
    // Stage 8b [P1] (Codex review): bound the offset-derived span
    // *before* issuing `read_range`. Without this guard, a corrupt
    // offset table or a docstore that's been appended to / sparsely
    // extended out-of-band can produce an arbitrarily large span,
    // causing `read_range` to issue a multi-GB object-store GET (and
    // a `Vec::with_capacity` of the same size) before
    // `decode_docstore_record` ultimately rejects the length-prefix
    // mismatch. The legitimate upper bound is `4 + MAX_DOCSTORE_BYTES`
    // — the writer enforces `MAX_DOCSTORE_BYTES` post-compression on
    // the payload, plus 4 bytes for the LE u32 length prefix. The
    // lower bound is 4 because the bundle MUST contain at least the
    // length prefix.
    let span = end - start;
    const MAX_BUNDLE_BYTES: u64 = MAX_DOCSTORE_BYTES as u64 + 4;
    if span < 4 {
      bail!(
        "docstore: span {span} for doc {doc_id} too small for the 4-byte length \
         prefix (offset table or file may be corrupt)"
      );
    }
    if span > MAX_BUNDLE_BYTES {
      bail!(
        "docstore: span {span} for doc {doc_id} exceeds maximum bundle size \
         {MAX_BUNDLE_BYTES} (= 4 + MAX_DOCSTORE_BYTES); refusing to issue \
         oversized read_range — offset table or docstore file may be corrupt"
      );
    }
    let bytes =
      futures::executor::block_on(self.docstore.read_range(start..end)).with_context(|| {
        format!("docstore read_range({start}..{end}) for doc {doc_id} failed")
      })?;
    decode_docstore_record(&bytes, self.core.seg_meta.use_zstd)
      .with_context(|| format!("decoding docstore record for doc {doc_id}"))
  }

  pub fn doc_id(&self, doc_id: DocId) -> Option<&str> {
    self.core.doc_ids.get(doc_id as usize).map(|s| s.as_ref())
  }

  pub fn find_doc_id(&self, id: &str) -> Option<DocId> {
    self
      .core
      .doc_ids
      .iter()
      .position(|d| d.as_ref() == id)
      .map(|i| i as DocId)
  }

  pub fn doc_ids(&self) -> &[Arc<str>] {
    &self.core.doc_ids
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
    &self.core.fast_fields
  }

  #[cfg(feature = "vectors")]
  pub fn vector(&self, field: &str, doc_id: DocId) -> Option<Vec<f32>> {
    self
      .core
      .vectors
      .get(field)
      .and_then(|vf| vf.store.vector(doc_id).map(|v| v.to_vec()))
  }

  #[cfg(feature = "vectors")]
  pub fn vector_components(&self, field: &str) -> Option<(&HnswIndex, Arc<VectorStore>)> {
    self
      .core
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
    let writer = SegmentWriter::new(dir.path(), &schema, true, false, storage.clone(), None);
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
    let writer = SegmentWriter::new(dir.path(), &schema, true, false, storage.clone(), None);
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
    let writer = SegmentWriter::new(dir.path(), &schema, true, false, storage, None);
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
      write_binding_b64: None,
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
      write_binding_b64: None,
    };
    let err = SegmentReader::open(storage, meta, true);
    assert!(err.is_err(), "expected zstd error for missing feature");
    let err = err.err().unwrap();
    assert!(
      err.to_string().contains("zstd"),
      "expected a clear zstd feature error, got {err}"
    );
  }

  #[cfg(feature = "vectors")]
  mod vector_file_bounds {
    use super::*;
    use crate::api::types::VectorMetric as ApiVectorMetric;
    use crate::vectors::VectorStore;
    use byteorder::{LittleEndian, WriteBytesExt};

    /// Build a minimal vector file header with caller-chosen counts. Does not
    /// append the offsets/values payload — callers that want a syntactically
    /// complete file append it themselves.
    fn build_header(dim: u32, metric_code: u8, doc_count: u32, vector_count: u32) -> Vec<u8> {
      let mut buf = Vec::new();
      buf.write_u32::<LittleEndian>(VECTOR_FILE_MAGIC).unwrap();
      buf.write_u32::<LittleEndian>(VECTOR_FILE_VERSION).unwrap();
      buf.write_u32::<LittleEndian>(dim).unwrap();
      buf.write_u8(metric_code).unwrap();
      buf.write_u8(0).unwrap();
      buf.write_u16::<LittleEndian>(0).unwrap();
      buf.write_u32::<LittleEndian>(doc_count).unwrap();
      buf.write_u32::<LittleEndian>(vector_count).unwrap();
      buf
    }

    #[test]
    fn round_trip_preserves_vectors() {
      let dir = tempdir().unwrap();
      let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
      let path = dir.path().join("vec.bin");
      let store = VectorStore::new(
        2,
        ApiVectorMetric::Cosine,
        vec![0, u32::MAX, 1],
        vec![0.1, 0.2, 0.3, 0.4],
      );
      write_vector_file(&storage, &path, &store).unwrap();
      let loaded = read_vector_file(&storage, &path, 3, 2, &ApiVectorMetric::Cosine).unwrap();
      assert_eq!(loaded.offsets(), &[0, u32::MAX, 1]);
      assert_eq!(loaded.values().as_slice(), &[0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn rejects_vector_count_exceeding_doc_count() {
      let dir = tempdir().unwrap();
      let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
      let path = dir.path().join("vec.bin");
      // doc_count = 2, but header claims 100 dense rows — enough that a naive
      // reader would allocate 100 * dim * sizeof(f32) before the short read
      // fires.
      let mut buf = build_header(4, metric_code(&ApiVectorMetric::Cosine), 2, 100);
      for offset in [0u32, 1u32] {
        buf.write_u32::<LittleEndian>(offset).unwrap();
      }
      std::fs::write(&path, &buf).unwrap();
      let err = read_vector_file(&storage, &path, 2, 4, &ApiVectorMetric::Cosine)
        .expect_err("must reject vector_count > doc_count");
      let msg = err.to_string();
      assert!(
        msg.contains("vector_count 100") && msg.contains("doc_count 2"),
        "unexpected error: {msg}"
      );
    }

    #[test]
    fn rejects_values_block_larger_than_remaining_bytes() {
      let dir = tempdir().unwrap();
      let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
      let path = dir.path().join("vec.bin");
      // vector_count = doc_count = 2 (passes the first guard), dim = 1024.
      // The values block should be 2 * 1024 * 4 = 8192 bytes, but we only
      // write 12 value bytes (3 f32s) so remaining < claimed.
      let mut buf = build_header(1024, metric_code(&ApiVectorMetric::Cosine), 2, 2);
      for offset in [0u32, 1u32] {
        buf.write_u32::<LittleEndian>(offset).unwrap();
      }
      for _ in 0..3 {
        buf.write_f32::<LittleEndian>(0.0).unwrap();
      }
      std::fs::write(&path, &buf).unwrap();
      let err = read_vector_file(&storage, &path, 2, 1024, &ApiVectorMetric::Cosine)
        .expect_err("must reject file when values bytes exceed remaining");
      let msg = err.to_string();
      assert!(
        msg.contains("value bytes") && msg.contains("remain"),
        "unexpected error: {msg}"
      );
    }

    #[test]
    fn rejects_near_u32_max_vector_count_without_allocating() {
      let dir = tempdir().unwrap();
      let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
      let path = dir.path().join("vec.bin");
      // Exact reproducer shape from BUG-014: header advertises a huge
      // vector_count that would previously drive
      // `Vec::with_capacity(vector_count * dim)` into an allocator abort.
      // After the fix, the `vector_count > doc_count` guard fires first so
      // the test terminates in constant memory.
      let mut buf = build_header(1024, metric_code(&ApiVectorMetric::Cosine), 2, u32::MAX);
      for offset in [0u32, 1u32] {
        buf.write_u32::<LittleEndian>(offset).unwrap();
      }
      std::fs::write(&path, &buf).unwrap();
      let err = read_vector_file(&storage, &path, 2, 1024, &ApiVectorMetric::Cosine)
        .expect_err("must reject crafted u32::MAX vector_count");
      let msg = err.to_string();
      assert!(msg.contains("exceeds doc_count"), "unexpected error: {msg}");
    }
  }

  // Regression coverage for BUG-013: every segment artifact that is referenced
  // from a manifest (vector file, HNSW graph) must be persisted through the
  // atomic tmp-write → fsync → rename → fsync-dir path, matching what the
  // manifest itself does. Using `write_all` leaves a truncated live file at
  // the target path if the write is interrupted mid-way, which turns into a
  // dangling manifest pointer.
  #[cfg(feature = "vectors")]
  mod vector_file_atomicity {
    use super::*;
    use crate::api::types::VectorMetric as ApiVectorMetric;
    use crate::storage::{DynFile, Storage};
    use crate::vectors::VectorStore;
    use parking_lot::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Storage wrapper around `FsStorage` that records which persistence
    /// method each write went through, so tests can assert that atomicity-
    /// critical artifacts use `atomic_write` rather than `write_all`.
    struct RecordingStorage {
      inner: crate::storage::FsStorage,
      write_all_paths: Mutex<Vec<std::path::PathBuf>>,
      atomic_write_paths: Mutex<Vec<std::path::PathBuf>>,
      atomic_write_count: AtomicUsize,
      write_all_count: AtomicUsize,
    }

    impl RecordingStorage {
      fn new(root: std::path::PathBuf) -> Self {
        Self {
          inner: crate::storage::FsStorage::new(root),
          write_all_paths: Mutex::new(Vec::new()),
          atomic_write_paths: Mutex::new(Vec::new()),
          atomic_write_count: AtomicUsize::new(0),
          write_all_count: AtomicUsize::new(0),
        }
      }
    }

    impl Storage for RecordingStorage {
      fn root(&self) -> &Path {
        self.inner.root()
      }
      fn ensure_dir(&self, path: &Path) -> Result<()> {
        self.inner.ensure_dir(path)
      }
      fn exists(&self, path: &Path) -> bool {
        self.inner.exists(path)
      }
      fn open_read(&self, path: &Path) -> Result<DynFile> {
        self.inner.open_read(path)
      }
      fn open_write(&self, path: &Path) -> Result<DynFile> {
        self.inner.open_write(path)
      }
      fn open_append(&self, path: &Path) -> Result<DynFile> {
        self.inner.open_append(path)
      }
      fn read_to_end(&self, path: &Path) -> Result<Vec<u8>> {
        self.inner.read_to_end(path)
      }
      fn write_all(&self, path: &Path, data: &[u8]) -> Result<()> {
        self.write_all_count.fetch_add(1, Ordering::SeqCst);
        self.write_all_paths.lock().push(path.to_path_buf());
        self.inner.write_all(path, data)
      }
      fn atomic_write(&self, path: &Path, data: &[u8]) -> Result<()> {
        self.atomic_write_count.fetch_add(1, Ordering::SeqCst);
        self.atomic_write_paths.lock().push(path.to_path_buf());
        self.inner.atomic_write(path, data)
      }
      fn remove(&self, path: &Path) -> Result<()> {
        self.inner.remove(path)
      }
      fn remove_dir_all(&self, path: &Path) -> Result<()> {
        self.inner.remove_dir_all(path)
      }
    }

    #[test]
    fn write_vector_file_uses_atomic_write() {
      let dir = tempdir().unwrap();
      let storage = RecordingStorage::new(dir.path().to_path_buf());
      let path = dir.path().join("vec.bin");
      let store = VectorStore::new(
        2,
        ApiVectorMetric::Cosine,
        vec![0, u32::MAX, 1],
        vec![0.1, 0.2, 0.3, 0.4],
      );

      write_vector_file(&storage, &path, &store).unwrap();

      assert_eq!(
        storage.atomic_write_count.load(Ordering::SeqCst),
        1,
        "vector file must be persisted through atomic_write exactly once"
      );
      assert_eq!(
        storage.write_all_count.load(Ordering::SeqCst),
        0,
        "vector file must not fall back to non-atomic write_all"
      );
      assert_eq!(
        storage.atomic_write_paths.lock().as_slice(),
        std::slice::from_ref(&path),
        "atomic_write was called on an unexpected path"
      );

      // Sanity-check the file round-trips correctly after an atomic write.
      let loaded = read_vector_file(&storage.inner, &path, 3, 2, &ApiVectorMetric::Cosine).unwrap();
      assert_eq!(loaded.offsets(), &[0, u32::MAX, 1]);
      assert_eq!(loaded.values().as_slice(), &[0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn write_vector_file_cleans_up_tmp_file_after_successful_atomic_write() {
      // This test verifies the successful-write cleanup behavior of
      // atomic_write: once the write completes, the final target file
      // should exist and no `.tmp` staging file should remain alongside it.
      let dir = tempdir().unwrap();
      let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
      let path = dir.path().join("subdir").join("vec.bin");
      let store = VectorStore::new(1, ApiVectorMetric::Cosine, vec![0], vec![1.0]);

      // Pre-create the parent directory via storage so ensure_dir semantics
      // are exercised the same way the real caller invokes them.
      storage.ensure_dir(path.parent().unwrap()).unwrap();
      write_vector_file(&storage, &path, &store).unwrap();

      // After a successful write, only the final target file should exist;
      // no `.tmp` sibling should be left around.
      assert!(path.exists(), "target vector file must exist");
      let leftover_tmp = path.with_extension("tmp");
      assert!(
        !leftover_tmp.exists(),
        "atomic_write must clean up the tmp staging file"
      );
    }

    /// `Storage` impl whose `atomic_write` always fails. Used to assert that
    /// when persistence errors out, the live target path is not clobbered —
    /// an existing file stays intact and no partial file is left behind.
    struct FailingAtomicWriteStorage {
      inner: crate::storage::FsStorage,
    }

    impl Storage for FailingAtomicWriteStorage {
      fn root(&self) -> &Path {
        self.inner.root()
      }
      fn ensure_dir(&self, path: &Path) -> Result<()> {
        self.inner.ensure_dir(path)
      }
      fn exists(&self, path: &Path) -> bool {
        self.inner.exists(path)
      }
      fn open_read(&self, path: &Path) -> Result<DynFile> {
        self.inner.open_read(path)
      }
      fn open_write(&self, path: &Path) -> Result<DynFile> {
        self.inner.open_write(path)
      }
      fn open_append(&self, path: &Path) -> Result<DynFile> {
        self.inner.open_append(path)
      }
      fn read_to_end(&self, path: &Path) -> Result<Vec<u8>> {
        self.inner.read_to_end(path)
      }
      fn write_all(&self, path: &Path, data: &[u8]) -> Result<()> {
        self.inner.write_all(path, data)
      }
      fn atomic_write(&self, _path: &Path, _data: &[u8]) -> Result<()> {
        Err(anyhow!("injected atomic_write failure"))
      }
      fn remove(&self, path: &Path) -> Result<()> {
        self.inner.remove(path)
      }
      fn remove_dir_all(&self, path: &Path) -> Result<()> {
        self.inner.remove_dir_all(path)
      }
    }

    #[test]
    fn write_vector_file_does_not_clobber_existing_file_on_failure() {
      // If atomic_write fails, an already-committed vector file at the
      // target path must remain byte-identical — the whole point of going
      // through the rename-based atomic path is that a failed write never
      // becomes visible to readers.
      let dir = tempdir().unwrap();
      let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
      let path = dir.path().join("vec.bin");
      let store = VectorStore::new(
        2,
        ApiVectorMetric::Cosine,
        vec![0, u32::MAX, 1],
        vec![0.1, 0.2, 0.3, 0.4],
      );

      // First commit a valid file with the real storage so the target path
      // has known-good bytes we can compare against after the failing write.
      write_vector_file(&storage, &path, &store).unwrap();
      let baseline = std::fs::read(&path).unwrap();

      // Now attempt a second write through a storage that fails atomic_write.
      let failing = FailingAtomicWriteStorage {
        inner: crate::storage::FsStorage::new(dir.path().to_path_buf()),
      };
      let overwrite_store = VectorStore::new(
        2,
        ApiVectorMetric::Cosine,
        vec![0, 1, 2],
        vec![9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
      );
      let err = write_vector_file(&failing, &path, &overwrite_store)
        .expect_err("failing atomic_write must surface an error");
      assert!(
        err.to_string().contains("injected atomic_write failure"),
        "unexpected error: {err}"
      );

      // The original file must still be exactly what we wrote first — the
      // failure path cannot leave a half-overwritten live file behind.
      let after = std::fs::read(&path).unwrap();
      assert_eq!(
        after, baseline,
        "failed atomic_write must not overwrite the live target path"
      );
    }
  }

  /// Stage 2 round-trip: for every term in a real segment's term dictionary,
  /// `TinyFst::range_for(term, postings_len)` must return exactly the byte
  /// range that `PostingsReader::read_at` consumes when decoding that term's
  /// postings list. This is the load-bearing correctness contract for
  /// future bounded `get_range` reads against an object-storage backend.
  #[test]
  fn range_for_matches_postings_reader_consumption_for_every_term() {
    use std::io::Seek;

    let dir = tempdir().unwrap();
    let schema = sample_schema();
    let storage = Arc::new(crate::storage::FsStorage::new(dir.path().to_path_buf()));

    // Multiple docs with overlapping and disjoint terms so the resulting FST
    // exercises both common-prefix and isolated-term layouts. Position-bearing
    // postings (`keep_positions=true`) maximize per-term byte length so the
    // test catches off-by-one errors in `read_at`'s inner loops, not just the
    // outer doc-freq varint.
    let writer = SegmentWriter::new(dir.path(), &schema, true, false, storage.clone(), None);
    let meta = writer
      .write_segment(
        &[
          doc("rust search engine fast", "news", 2024),
          doc("rust language tooling", "tech", 2023),
          doc("search engine indexing fast", "news", 2024),
          doc("indexing pipeline", "infra", 2025),
          doc("rust async tooling", "tech", 2024),
        ],
        1,
      )
      .unwrap();

    let reader = SegmentReader::open(storage.clone(), meta.clone(), true).unwrap();
    let postings_path = std::path::PathBuf::from(&meta.paths.postings);
    let postings_len = std::fs::metadata(&postings_path).unwrap().len();
    assert!(
      postings_len > 0,
      "test prerequisite: segment must have a non-empty postings file"
    );

    // Walk the term dictionary in sorted order and check each term's range.
    // Iterating with an empty prefix yields every term; the existing
    // `terms_with_prefix` already binary-searches into the sorted vec.
    let all_terms: Vec<String> = reader
      .terms_with_prefix("")
      .map(|t| t.to_string())
      .collect();
    assert!(
      all_terms.len() >= 4,
      "test prerequisite: schema + corpus must produce several distinct terms; got {}",
      all_terms.len()
    );

    let mut last_term_seen = false;
    for term in &all_terms {
      let computed = reader
        .core
        .terms
        .0
        .range_for(term, postings_len)
        .unwrap_or_else(|| panic!("range_for returned None for present term {term:?}"));

      // What did the postings reader actually consume? `read_at` seeks to
      // `offset` and decodes one postings list; the file's stream position
      // afterwards is the first byte past this term's payload.
      // Use `&mut file` (= `&mut Box<dyn StorageFile>`, which is `Sized`)
      // rather than `&mut *file` (which would give a `&mut dyn StorageFile`
      // unsized borrow that `PostingsReader::read_at`'s `R: Read + Seek`
      // bound rejects).
      let mut file = storage.open_read(&postings_path).unwrap();
      let _decoded = PostingsReader::read_at(&mut file, computed.start, true)
        .unwrap_or_else(|e| panic!("read_at failed for term {term:?} at offset {}: {e}", computed.start));
      let consumed_end = file.stream_position().unwrap();

      assert_eq!(
        computed.start,
        reader.core.terms.0.get(term).unwrap(),
        "range_for({term:?}) start must equal the FST's recorded offset"
      );
      assert_eq!(
        computed.end, consumed_end,
        "range_for({term:?}) end ({}) must equal bytes consumed by read_at ({}); \
         wrong end means an object-store range read would either truncate the \
         postings list or overshoot into the next term",
        computed.end, consumed_end
      );

      if computed.end == postings_len {
        last_term_seen = true;
      }
    }

    assert!(
      last_term_seen,
      "test must exercise the last-term branch (where range.end == postings_len); \
       did the loop iterate over the lexicographically-greatest term?"
    );
  }
}
