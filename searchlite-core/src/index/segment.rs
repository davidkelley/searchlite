use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use hashbrown::{HashMap as FastHashMap, HashSet as FastHashSet};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::analysis::tokenizer::tokenize;
use crate::api::types::Document;
use crate::index::directory;
use crate::index::docstore::{DocStoreReader, DocStoreWriter};
use crate::index::fastfields::{FastFieldsReader, FastFieldsWriter, FastValue};
use crate::index::manifest::{Schema, SegmentMeta, SegmentPaths};
use crate::index::postings::{InvertedIndexBuilder, PostingsReader, PostingsWriter};
use crate::index::terms::{read_terms, write_terms};
use crate::storage::{Storage, StorageFile};
use crate::util::checksum::checksum;
use crate::DocId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentFileMeta {
  pub doc_offsets: Vec<u64>,
  pub avg_field_lengths: HashMap<String, f32>,
  #[cfg(feature = "vectors")]
  pub vectors: HashMap<String, Vec<Vec<f32>>>,
  pub use_zstd: bool,
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
    let id = Uuid::new_v4().simple().to_string();
    let paths = directory::segment_paths(self.root, &id);

    let mut postings_builder = InvertedIndexBuilder::new();
    let mut doc_lengths: HashMap<String, u64> = HashMap::new();
    let mut fast_writer = FastFieldsWriter::new();
    let keyword_fast: FastHashSet<&str> = self
      .schema
      .keyword_fields
      .iter()
      .filter(|f| f.fast)
      .map(|f| f.name.as_str())
      .collect();
    let numeric_info: FastHashMap<&str, (bool, bool)> = self
      .schema
      .numeric_fields
      .iter()
      .map(|f| (f.name.as_str(), (f.i64, f.fast)))
      .collect();

    let mut docstore_file = self.storage.open_write(Path::new(&paths.docstore))?;
    let mut doc_writer = DocStoreWriter::new(&mut *docstore_file, self.use_zstd);

    #[cfg(feature = "vectors")]
    let mut vector_fields: HashMap<String, Vec<Vec<f32>>> = HashMap::new();

    for (doc_id_u64, doc) in docs.iter().enumerate() {
      let doc_id = doc_id_u64 as DocId;
      let mut stored = serde_json::Map::new();
      for (k, v) in doc.fields.iter() {
        stored.insert(k.clone(), v.clone());
      }
      for (field, value) in doc.fields.iter() {
        match self.schema.field_kind(field) {
          crate::index::manifest::FieldKind::Text => {
            if let Some(text) = value.as_str() {
              let tokens = tokenize(text);
              doc_lengths
                .entry(field.clone())
                .and_modify(|v| *v += tokens.len() as u64)
                .or_insert(tokens.len() as u64);
              for (pos, tok) in tokens.iter().enumerate() {
                let mut term_key = String::with_capacity(field.len() + tok.len() + 1);
                term_key.push_str(field);
                term_key.push(':');
                term_key.push_str(tok);
                postings_builder.add_term(&term_key, doc_id, pos as u32, self.enable_positions);
              }
            }
          }
          crate::index::manifest::FieldKind::Keyword => {
            if let Some(s) = value.as_str() {
              let lower = s.to_ascii_lowercase();
              let mut term_key = String::with_capacity(field.len() + lower.len() + 1);
              term_key.push_str(field);
              term_key.push(':');
              term_key.push_str(&lower);
              postings_builder.add_term(&term_key, doc_id, 0, false);
              if keyword_fast.contains(field.as_str()) {
                fast_writer.set(field, doc_id, FastValue::Str(s.to_string()));
              }
            }
          }
          crate::index::manifest::FieldKind::Numeric => {
            if let Some((is_i64, fast)) = numeric_info.get(field.as_str()) {
              if *is_i64 {
                if let Some(n) = value.as_i64() {
                  if *fast {
                    fast_writer.set(field, doc_id, FastValue::I64(n));
                  }
                }
              } else if let Some(f) = value.as_f64() {
                if *fast {
                  fast_writer.set(field, doc_id, FastValue::F64(f));
                }
              }
            }
          }
          crate::index::manifest::FieldKind::Unknown => {}
        }
        if self.schema.is_stored_field(field) {
          stored.insert(field.clone(), value.clone());
        }
      }

      #[cfg(feature = "vectors")]
      for vf in self.schema.vector_fields.iter() {
        let entry = vector_fields.entry(vf.name.clone()).or_default();
        if let Some(val) = doc.fields.get(&vf.name) {
          if let Some(arr) = val.as_array() {
            let mut vecvals: Vec<f32> = arr
              .iter()
              .filter_map(|v| v.as_f64())
              .map(|v| v as f32)
              .collect();
            if vecvals.len() < vf.dim {
              vecvals.resize(vf.dim, 0.0);
            } else if vecvals.len() > vf.dim {
              vecvals.truncate(vf.dim);
            }
            entry.push(vecvals);
            continue;
          }
        }
        entry.push(vec![0.0; vf.dim]);
      }

      doc_writer.add_document(&serde_json::Value::Object(stored))?;
    }
    let doc_offsets = doc_writer.offsets().to_vec();
    drop(doc_writer);
    docstore_file.sync_all()?;

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

    let avg_field_lengths = compute_avg_lengths(&doc_lengths, docs.len() as u64);

    fast_writer.write_to(self.storage.as_ref(), Path::new(&paths.fast))?;

    let seg_file_meta = SegmentFileMeta {
      doc_offsets,
      avg_field_lengths: avg_field_lengths.clone(),
      #[cfg(feature = "vectors")]
      vectors: vector_fields,
      use_zstd: self.use_zstd,
    };
    write_segment_meta(
      self.storage.as_ref(),
      Path::new(&paths.meta),
      &seg_file_meta,
    )?;

    let checksums = collect_checksums(self.storage.as_ref(), &paths)?;

    let meta = SegmentMeta {
      id,
      generation,
      paths,
      doc_count: docs.len() as u32,
      max_doc_id: docs.len() as u32 - 1,
      blockmax: true,
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

pub struct SegmentReader {
  pub meta: SegmentMeta,
  terms: TinyTerms,
  postings: RefCell<Box<dyn StorageFile>>,
  docstore: RefCell<DocStoreReader<Box<dyn StorageFile>>>,
  fast_fields: FastFieldsReader,
  keep_positions: bool,
  seg_meta: SegmentFileMeta,
}

impl SegmentReader {
  pub fn open(storage: Arc<dyn Storage>, meta: SegmentMeta, keep_positions: bool) -> Result<Self> {
    let terms = read_terms(storage.as_ref(), Path::new(&meta.paths.terms))?;
    let postings = storage.open_read(Path::new(&meta.paths.postings))?;
    let doc_file = storage.open_read(Path::new(&meta.paths.docstore))?;
    let seg_meta_bytes = storage.read_to_end(Path::new(&meta.paths.meta))?;
    let seg_meta: SegmentFileMeta = serde_json::from_slice(&seg_meta_bytes)?;
    let docstore = DocStoreReader::new(doc_file, seg_meta.doc_offsets.clone(), seg_meta.use_zstd);
    let fast_fields = FastFieldsReader::open(storage.as_ref(), Path::new(&meta.paths.fast))?;
    Ok(Self {
      meta,
      terms: TinyTerms(terms),
      postings: RefCell::new(postings),
      docstore: RefCell::new(docstore),
      fast_fields,
      keep_positions,
      seg_meta,
    })
  }

  pub fn postings(&self, term: &str) -> Option<PostingsReader> {
    let offset = self.terms.0.get(term)?;
    let mut file = self.postings.borrow_mut();
    PostingsReader::read_at(&mut *file, offset, self.keep_positions).ok()
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

  pub fn fast_fields(&self) -> &FastFieldsReader {
    &self.fast_fields
  }

  #[cfg(feature = "vectors")]
  pub fn vector(&self, field: &str, doc_id: DocId) -> Option<Vec<f32>> {
    self
      .seg_meta
      .vectors
      .get(field)?
      .get(doc_id as usize)
      .cloned()
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
      text_fields: vec![crate::index::manifest::TextField {
        name: "body".into(),
        tokenizer: "default".into(),
        stored: true,
        indexed: true,
      }],
      keyword_fields: vec![crate::index::manifest::KeywordField {
        name: "tag".into(),
        stored: true,
        indexed: true,
        fast: true,
      }],
      numeric_fields: vec![crate::index::manifest::NumericField {
        name: "year".into(),
        i64: true,
        fast: true,
        stored: true,
      }],
      #[cfg(feature = "vectors")]
      vector_fields: Vec::new(),
    }
  }

  fn doc(body: &str, tag: &str, year: i64) -> Document {
    Document {
      fields: [
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
  fn computes_average_lengths() {
    let lengths = HashMap::from([("body".to_string(), 4u64), ("title".to_string(), 0u64)]);
    let avg = compute_avg_lengths(&lengths, 2);
    assert_eq!(avg.get("body"), Some(&2.0));
    assert_eq!(avg.get("title"), Some(&0.0));
  }
}
