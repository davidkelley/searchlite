use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use serde_json::Value;

use searchlite_core::api::builder::IndexBuilder;
use searchlite_core::api::types::{Document, IndexOptions, SearchRequest, StorageType};
use searchlite_core::api::Index;
use searchlite_core::Manifest;

use super::{SurfaceHarness, SurfaceKind};

pub struct CoreHarness {
  index_path: PathBuf,
  index: Option<Index>,
}

impl CoreHarness {
  pub fn new(index_path: PathBuf) -> Self {
    Self {
      index_path,
      index: None,
    }
  }

  fn options(&self, create_if_missing: bool) -> IndexOptions {
    IndexOptions {
      path: self.index_path.clone(),
      create_if_missing,
      enable_positions: true,
      bm25_k1: 0.9,
      bm25_b: 0.4,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    }
  }

  fn index(&self) -> Result<&Index> {
    self
      .index
      .as_ref()
      .ok_or_else(|| anyhow!("core harness not initialized"))
  }
}

impl SurfaceHarness for CoreHarness {
  fn kind(&self) -> SurfaceKind {
    SurfaceKind::Core
  }

  fn init(&mut self, schema: &Value) -> Result<()> {
    let schema = serde_json::from_value(schema.clone()).context("parsing schema for core init")?;
    let index = IndexBuilder::create(self.index_path.as_path(), schema, self.options(true))
      .context("creating index for core harness")?;
    self.index = Some(index);
    Ok(())
  }

  fn add_ndjson(&mut self, ndjson: &str) -> Result<()> {
    let index = self.index()?;
    let mut writer = index.writer().context("opening core writer")?;
    for (line_no, line) in ndjson.lines().enumerate() {
      if line.trim().is_empty() {
        continue;
      }
      let value: Value = serde_json::from_str(line)
        .with_context(|| format!("parsing NDJSON line {}", line_no + 1))?;
      let obj = value
        .as_object()
        .ok_or_else(|| anyhow!("NDJSON document line {} is not an object", line_no + 1))?;
      let fields: BTreeMap<String, Value> = obj
        .iter()
        .map(|(key, val)| (key.clone(), val.clone()))
        .collect();
      writer
        .add_document(&Document { fields })
        .with_context(|| format!("adding document on line {}", line_no + 1))?;
    }
    Ok(())
  }

  fn commit(&mut self) -> Result<()> {
    let index = self.index()?;
    let mut writer = index.writer().context("opening core writer for commit")?;
    writer.commit().context("committing core writes")
  }

  fn refresh(&mut self) -> Result<()> {
    let index = self.index()?;
    let _ = index.reader().context("refreshing core reader")?;
    Ok(())
  }

  fn search(&mut self, request: &Value) -> Result<Value> {
    let index = self.index()?;
    let reader = index.reader().context("opening core reader")?;
    let request: SearchRequest =
      serde_json::from_value(request.clone()).context("parsing core search request")?;
    let result = reader.search(&request).context("executing core search")?;
    serde_json::to_value(result).context("serializing core search result")
  }

  fn mget(&mut self, ids: &[String], return_stored: bool) -> Result<Value> {
    let index = self.index()?;
    let reader = index.reader().context("opening core reader for mget")?;
    let docs = reader
      .mget(ids, return_stored)
      .context("executing core mget")?;
    serde_json::to_value(serde_json::json!({ "docs": docs }))
      .context("serializing core mget result")
  }

  fn update_doc(
    &mut self,
    id: &str,
    set: &serde_json::Map<String, Value>,
    unset: &[String],
  ) -> Result<()> {
    let index = self.index()?;
    let mut writer = index.writer().context("opening core writer for update")?;
    let set_map: BTreeMap<String, Value> = set
      .iter()
      .map(|(key, value)| (key.clone(), value.clone()))
      .collect();
    writer
      .apply_patch(id, &set_map, unset)
      .with_context(|| format!("patching document {id}"))?;
    Ok(())
  }

  fn delete_ids(&mut self, ids: &[String]) -> Result<()> {
    let index = self.index()?;
    let mut writer = index.writer().context("opening core writer for delete")?;
    writer
      .delete_documents(ids)
      .context("deleting core documents")
  }

  fn stats(&mut self) -> Result<Value> {
    let index = self.index()?;
    let manifest = index.manifest();
    let (documents, deleted_documents) = manifest_doc_counts(&manifest);
    Ok(serde_json::json!({
      "documents": documents,
      "deleted_documents": deleted_documents,
      "segments": manifest.segments.len(),
      "committed_at": manifest.committed_at,
      "index_uuid": manifest.uuid.to_string(),
      "index_name": "core",
    }))
  }

  fn inspect(&mut self) -> Result<Value> {
    let index = self.index()?;
    let manifest = index.manifest();
    Ok(serde_json::json!({ "manifest": manifest }))
  }

  fn compact(&mut self) -> Result<()> {
    let index = self.index()?;
    index.compact().context("compacting core index")
  }
}

fn manifest_doc_counts(manifest: &Manifest) -> (u64, u64) {
  let deleted: u64 = manifest
    .segments
    .iter()
    .map(|seg| seg.deleted_docs.len() as u64)
    .sum();
  let live: u64 = manifest
    .segments
    .iter()
    .map(|seg| {
      let doc_count = seg.doc_count as u64;
      doc_count.saturating_sub(seg.deleted_docs.len() as u64)
    })
    .sum();
  (live, deleted)
}
