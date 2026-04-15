use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use chrono::Utc;

use crate::api::errors::PatchError;
use crate::api::reader::IndexReader;
use crate::api::types::Document;
use crate::index::manifest::{Manifest, NestedField, NestedProperty, Schema};
use crate::index::segment::{SegmentFileMeta, SegmentWriter};
use crate::index::wal::{Wal, WalEntry};
use crate::index::InnerIndex;
use crate::util::doc_id::validate_doc_id;
#[cfg(feature = "write-key")]
use crate::util::write_key::{binding_for_uuid, verify_binding, verify_write_key};
use crate::DocId;

#[derive(Debug, Clone)]
struct DocAddress {
  segment_id: String,
  doc_id: DocId,
}

#[derive(Debug, Clone)]
enum PendingOp {
  Add { doc_id: String, doc: Document },
  Delete { doc_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PatchReaderStamp {
  committed_at: String,
  segment_fingerprint: u64,
}

pub struct IndexWriter {
  inner: Arc<InnerIndex>,
  wal: Wal,
  pending_ops: Vec<PendingOp>,
  pending_latest: HashMap<String, Option<Document>>,
  schema: Schema,
  live_docs: HashMap<String, DocAddress>,
  live_generation: u32,
  write_binding: Option<Vec<u8>>,
  patch_reader: Option<IndexReader>,
  patch_reader_stamp: Option<PatchReaderStamp>,
}

#[derive(Debug, Clone, Copy)]
pub struct WriterCheckpoint {
  wal_len: u64,
  pending_len: usize,
}

impl IndexWriter {
  pub(crate) fn new(inner: Arc<InnerIndex>, write_key: Option<&str>) -> Result<Self> {
    // Hold the writer lock during initialization to avoid racing with a commit.
    let _guard = inner.writer_lock.lock();
    let wal_path = crate::index::directory::wal_path(&inner.path);
    let (wal_binding, pending_entries) = Wal::last_pending_ops(inner.storage.as_ref(), &wal_path)?;
    #[allow(unused_mut)]
    let mut wal = inner.wal()?;
    let manifest = inner.manifest.read().clone();
    let schema = manifest.schema.clone();
    let live_generation = manifest
      .segments
      .iter()
      .map(|s| s.generation)
      .max()
      .unwrap_or(0);
    #[allow(unused_mut)]
    let mut write_binding: Option<Vec<u8>> = None;

    let mut segments_binding: Vec<Vec<u8>> = Vec::new();
    #[allow(unused_mut)]
    let mut binding_required = manifest.write_key.is_some() || wal_binding.is_some();
    for seg in manifest.segments.iter() {
      if let Some(b64) = seg.write_binding_b64.as_deref() {
        let decoded = BASE64
          .decode(b64)
          .map_err(|e| anyhow!("invalid base64 in segment manifest write_binding_b64: {e}"))?;
        segments_binding.push(decoded);
        binding_required = true;
      }
    }
    for seg in manifest.segments.iter() {
      match inner.storage.read_to_end(Path::new(&seg.paths.meta)) {
        Ok(bytes) => match serde_json::from_slice::<SegmentFileMeta>(&bytes) {
          Ok(seg_meta) => {
            if let Some(b64) = seg_meta.write_binding_b64.as_deref() {
              let decoded = BASE64.decode(b64).map_err(|e| {
                anyhow!("invalid base64 in segment metadata write_binding_b64: {e}")
              })?;
              segments_binding.push(decoded);
              binding_required = true;
            }
          }
          Err(e) => {
            if binding_required {
              return Err(anyhow!(
                "failed to decode segment metadata from {}: {e}",
                seg.paths.meta
              ));
            }
          }
        },
        Err(e) => {
          if binding_required {
            return Err(anyhow!(
              "failed to read segment metadata from {}: {e}",
              seg.paths.meta
            ));
          }
        }
      }
    }
    if binding_required {
      #[cfg(feature = "write-key")]
      {
        let key = write_key.ok_or_else(|| anyhow!("write key required for this index"))?;
        if let Some(meta) = manifest.write_key.as_ref() {
          verify_write_key(key, meta)?;
        }
        let candidate = binding_for_uuid(key, &manifest.uuid);
        if let Some(b) = wal_binding.as_ref() {
          if !verify_binding(b, &candidate) {
            bail!("write key does not match WAL binding; index may be tampered");
          }
        }
        for seg_binding in segments_binding.iter() {
          if !verify_binding(seg_binding, &candidate) {
            bail!("write key does not match segment binding; index may be tampered");
          }
        }
        if manifest.write_key.is_none() && (wal_binding.is_some() || !segments_binding.is_empty()) {
          bail!(
            "write key metadata missing but bindings exist; index metadata was likely tampered"
          );
        }
        write_binding = Some(candidate.clone());
        if wal_binding.is_none() {
          wal.append_binding(&candidate)?;
          wal.sync()?;
        }
      }
      #[cfg(not(feature = "write-key"))]
      {
        let _ = write_key;
        crate::util::write_key::require_write_key_feature()?;
      }
    }
    let live_docs = load_live_docs(inner.as_ref(), &manifest)?;
    let mut pending_ops = Vec::new();
    for entry in pending_entries {
      match entry {
        WalEntry::AddDoc(doc) => {
          let doc_id = doc_id_from_document(&schema, &doc)?;
          pending_ops.push(PendingOp::Add { doc_id, doc });
        }
        WalEntry::DeleteDocId(doc_id) => pending_ops.push(PendingOp::Delete { doc_id }),
        WalEntry::Commit => {}
        WalEntry::WriteBinding(_) => {}
      }
    }
    let pending_latest = pending_latest_from_ops(&pending_ops);
    drop(_guard);
    Ok(Self {
      inner,
      wal,
      pending_ops,
      pending_latest,
      schema,
      live_docs,
      live_generation,
      write_binding,
      patch_reader: None,
      patch_reader_stamp: None,
    })
  }

  /// Capture the current WAL length and pending-op count so callers can roll back
  /// only the work done after the checkpoint.
  pub fn checkpoint(&mut self) -> Result<WriterCheckpoint> {
    let inner = self.inner.clone();
    let _guard = inner.writer_lock.lock();
    self.checkpoint_locked()
  }

  /// Truncate WAL and pending_ops back to a prior checkpoint without dropping
  /// earlier queued work.
  pub fn rollback_to(&mut self, checkpoint: WriterCheckpoint) -> Result<()> {
    let inner = self.inner.clone();
    let _guard = inner.writer_lock.lock();
    self.rollback_to_locked(checkpoint)
  }

  pub fn add_document(&mut self, doc: &Document) -> Result<u32> {
    let inner = self.inner.clone();
    let _guard = inner.writer_lock.lock();
    // BUG-224: enforce required-field presence only at the user-facing
    // ingest boundary. `add_document_locked` is also reached from
    // `apply_patch`, which re-inserts documents reconstructed from the
    // docstore; those reconstructed documents may legitimately be missing
    // top-level fields that serialize away (empty arrays, nested
    // containers whose stored children all serialize to null), so the
    // presence check cannot live inside the locked path.
    self.schema.check_required_fields_present(doc)?;
    self.add_document_locked(doc)
  }

  pub fn delete_document(&mut self, doc_id: &str) -> Result<()> {
    self.delete_documents(&[doc_id.to_string()])
  }

  pub fn delete_documents(&mut self, doc_ids: &[String]) -> Result<()> {
    let _guard = self.inner.writer_lock.lock();
    for id in doc_ids {
      self.wal.append_delete_doc_id(id)?;
      self
        .pending_ops
        .push(PendingOp::Delete { doc_id: id.clone() });
      self.pending_latest.insert(id.clone(), None);
    }
    Ok(())
  }

  pub fn apply_patch(
    &mut self,
    doc_id: &str,
    set: &BTreeMap<String, serde_json::Value>,
    unset: &[String],
  ) -> Result<()> {
    let inner = self.inner.clone();
    let _guard = inner.writer_lock.lock();
    ensure_patch_safe(&self.schema)?;
    validate_patch_fields(&self.schema, doc_id, set, unset)?;
    let mut doc = match pending_doc_for_patch(&self.pending_latest, doc_id) {
      Some(Some(doc)) => doc,
      Some(None) => return Err(PatchError::DocumentNotFound.into()),
      None => load_document_for_patch(
        inner.clone(),
        &mut self.patch_reader,
        &mut self.patch_reader_stamp,
        doc_id,
      )?
      .ok_or(PatchError::DocumentNotFound)?,
    };
    let mut value = document_to_value(&doc)?;
    let literal_top_level_paths = literal_top_level_dotted_paths(&self.schema);
    for path in unset.iter() {
      unset_path_with_literals(&mut value, path, Some(&literal_top_level_paths))?;
    }
    for (path, val) in set.iter() {
      set_path_with_literals(
        &mut value,
        path,
        val.clone(),
        Some(&literal_top_level_paths),
      )?;
    }
    doc = value_to_document(value)?;
    self.schema.validate_document(&doc)?;
    self.add_document_locked(&doc)?;
    Ok(())
  }

  pub fn commit(&mut self) -> Result<()> {
    let inner = self.inner.clone();
    let _guard = inner.writer_lock.lock();
    if self.pending_ops.is_empty() {
      return Ok(());
    }
    self.wal.sync()?;
    let manifest_snapshot = inner.manifest.read().clone();
    self.schema = manifest_snapshot.schema.clone();
    let manifest_generation = manifest_snapshot
      .segments
      .iter()
      .map(|s| s.generation)
      .max()
      .unwrap_or(0);
    let mut live_docs = if manifest_generation == self.live_generation {
      self.live_docs.clone()
    } else {
      load_live_docs(inner.as_ref(), &manifest_snapshot)?
    };
    let mut pending_new: BTreeMap<String, Document> = BTreeMap::new();
    let mut tombstones: HashMap<String, Vec<DocId>> = HashMap::new();
    for op in self.pending_ops.iter() {
      match op {
        PendingOp::Add { doc_id, doc } => {
          if let Some(addr) = live_docs.remove(doc_id) {
            tombstones
              .entry(addr.segment_id)
              .or_default()
              .push(addr.doc_id);
          }
          pending_new.insert(doc_id.clone(), doc.clone());
        }
        PendingOp::Delete { doc_id } => {
          pending_new.remove(doc_id);
          if let Some(addr) = live_docs.remove(doc_id) {
            tombstones
              .entry(addr.segment_id)
              .or_default()
              .push(addr.doc_id);
          }
        }
      }
    }
    let mut new_manifest = manifest_snapshot.clone();
    for seg in new_manifest.segments.iter_mut() {
      if let Some(deleted) = tombstones.remove(&seg.id) {
        let mut set: HashSet<DocId> = seg.deleted_docs.iter().copied().collect();
        set.extend(deleted.into_iter());
        let mut merged: Vec<DocId> = set.into_iter().collect();
        merged.sort_unstable();
        seg.deleted_docs = merged;
      }
    }
    let mut new_segments: Vec<crate::index::manifest::SegmentMeta> = Vec::new();
    if !pending_new.is_empty() {
      let generation = new_manifest
        .segments
        .iter()
        .map(|s| s.generation)
        .max()
        .unwrap_or(0)
        + 1;
      let writer = SegmentWriter::new(
        &self.inner.path,
        &self.schema,
        self.inner.options.enable_positions,
        cfg!(feature = "zstd"),
        self.inner.storage.clone(),
        self.write_binding.clone(),
      );
      let docs: Vec<Document> = pending_new.values().cloned().collect();
      let segment = writer.write_segment(&docs, generation)?;
      // Keep track of newly written segments so they can be cleaned up on rollback.
      new_segments.push(segment.clone());
      new_manifest.segments.push(segment.clone());
      for (offset, doc_id) in pending_new.keys().enumerate() {
        live_docs.insert(
          doc_id.clone(),
          DocAddress {
            segment_id: segment.id.clone(),
            doc_id: offset as DocId,
          },
        );
      }
    }
    let new_generation = new_manifest
      .segments
      .iter()
      .map(|s| s.generation)
      .max()
      .unwrap_or(0);
    new_manifest.committed_at = Utc::now().to_rfc3339();
    let manifest_path = self.inner.manifest_path();
    let wal_len = self.wal.len()?;
    if let Err(e) = (|| -> Result<()> {
      new_manifest.store(self.inner.storage.as_ref(), &manifest_path)?;
      // `append_commit` fsyncs before returning, so no separate `sync` call
      // is needed to guarantee the commit record is durable on disk.
      self.wal.append_commit()?;
      Ok(())
    })() {
      // Roll back manifest to the previous snapshot and restore WAL to its
      // pre-commit length so pending ops can be retried safely.
      if let Err(truncate_err) = self.wal.truncate_to(wal_len) {
        log::error!(
          "WAL rollback failed while handling commit error: \
           unable to truncate WAL back to length {wal_len}: {truncate_err}"
        );
      }
      if let Err(manifest_err) =
        manifest_snapshot.store(self.inner.storage.as_ref(), &manifest_path)
      {
        log::error!(
          "Manifest rollback failed while handling commit error: {manifest_err}. \
           The on-disk manifest and WAL may be inconsistent."
        );
      }
      if !new_segments.is_empty() {
        let _ = crate::index::cleanup_segments(self.inner.storage.as_ref(), &new_segments);
      }
      self.patch_reader = None;
      self.patch_reader_stamp = None;
      return Err(e);
    }
    {
      let mut manifest_guard = self.inner.manifest.write();
      *manifest_guard = new_manifest;
    }
    self.wal.truncate()?;
    self.pending_ops.clear();
    self.pending_latest.clear();
    self.patch_reader = None;
    self.patch_reader_stamp = None;
    self.live_docs = live_docs;
    self.live_generation = new_generation;
    Ok(())
  }

  /// Commit pending changes and optionally evaluate the tiered merge policy
  /// afterwards. When `merge` is `true`, any merge candidates identified by
  /// `TieredMergePolicy::default()` are merged inline before returning.
  ///
  /// The existing `commit()` behaviour is preserved unchanged; this method
  /// simply adds an optional post-commit merge step.
  /// Commit and optionally merge. For write-key-protected indexes, use
  /// `commit_with_merge_and_key` instead.
  pub fn commit_with_merge(&mut self, merge: bool) -> Result<()> {
    if merge && self.write_binding.is_some() {
      bail!(
        "this index requires a write key for merge; \
         use commit_with_merge_and_key(merge, Some(key)) instead"
      );
    }
    self.commit_with_merge_and_key(merge, None)
  }

  /// Commit pending operations and optionally run a tiered merge pass.
  ///
  /// `write_key` must be provided for write-key-protected indexes so that
  /// the post-commit merge can verify and bind segments correctly.
  pub fn commit_with_merge_and_key(&mut self, merge: bool, write_key: Option<&str>) -> Result<()> {
    self.commit()?;
    if merge {
      let manifest = self.inner.manifest.read().clone();
      let policy = crate::index::merge::TieredMergePolicy::default();
      let merge_groups = policy.find_merges(&manifest.segments);
      for group in merge_groups {
        let idx = crate::index::Index {
          inner: self.inner.clone(),
        };
        idx.merge_segments(&group, write_key)?;
      }
      // Refresh live_docs and generation after the merge.
      let new_manifest = self.inner.manifest.read().clone();
      self.live_generation = new_manifest
        .segments
        .iter()
        .map(|s| s.generation)
        .max()
        .unwrap_or(0);
      self.live_docs = load_live_docs(self.inner.as_ref(), &new_manifest)?;
    }
    Ok(())
  }

  pub fn rollback(&mut self) -> Result<()> {
    let _guard = self.inner.writer_lock.lock();
    self.pending_ops.clear();
    self.pending_latest.clear();
    self.patch_reader = None;
    self.patch_reader_stamp = None;
    self.wal.truncate()?;
    Ok(())
  }

  /// Add a batch atomically: validate all docs, append to WAL, and either queue all or none.
  pub fn add_documents_batch(&mut self, docs: &[Document]) -> Result<usize> {
    let inner = self.inner.clone();
    let _guard = inner.writer_lock.lock();
    let checkpoint = self.checkpoint_locked()?;
    for doc in docs {
      // BUG-224: enforce required-field presence at the ingest boundary;
      // see the note in `add_document` for why this is kept out of
      // `add_document_locked`.
      if let Err(e) = self
        .schema
        .check_required_fields_present(doc)
        .and_then(|()| self.add_document_locked(doc).map(|_| ()))
      {
        self.rollback_to_locked(checkpoint)?;
        return Err(e);
      }
    }
    Ok(docs.len())
  }

  fn checkpoint_locked(&mut self) -> Result<WriterCheckpoint> {
    let wal_len = self.wal.len()?;
    Ok(WriterCheckpoint {
      wal_len,
      pending_len: self.pending_ops.len(),
    })
  }

  fn rollback_to_locked(&mut self, checkpoint: WriterCheckpoint) -> Result<()> {
    self.wal.truncate_to(checkpoint.wal_len)?;
    if self.pending_ops.len() > checkpoint.pending_len {
      self.pending_ops.truncate(checkpoint.pending_len);
      self.pending_latest = pending_latest_from_ops(&self.pending_ops);
    }
    Ok(())
  }

  fn add_document_locked(&mut self, doc: &Document) -> Result<u32> {
    self.schema.validate_document(doc)?;
    let doc_id = doc_id_from_document(&self.schema, doc)?;
    self.wal.append_add_doc(doc)?;
    self.pending_ops.push(PendingOp::Add {
      doc_id: doc_id.clone(),
      doc: doc.clone(),
    });
    self
      .pending_latest
      .insert(doc_id.clone(), Some(doc.clone()));
    let add_count = self
      .pending_ops
      .iter()
      .filter(|op| matches!(op, PendingOp::Add { .. }))
      .count();
    Ok(add_count as u32 - 1)
  }
}

impl Drop for IndexWriter {
  fn drop(&mut self) {
    if !self.pending_ops.is_empty() {
      if let Err(e) = self.wal.sync() {
        eprintln!(
          "IndexWriter: failed to sync WAL on drop ({} pending ops): {e}",
          self.pending_ops.len()
        );
      }
    }
  }
}

fn doc_id_from_document(schema: &Schema, doc: &Document) -> Result<String> {
  let doc_id = doc
    .fields
    .get(schema.doc_id_field())
    .and_then(|v| v.as_str())
    .ok_or_else(|| {
      anyhow!(
        "missing or empty required document id field `{}`",
        schema.doc_id_field()
      )
    })?;
  if doc_id.trim().is_empty() {
    bail!(
      "missing or empty required document id field `{}`",
      schema.doc_id_field()
    );
  }
  validate_doc_id(doc_id)?;
  Ok(doc_id.to_string())
}

fn load_document_for_patch(
  inner: Arc<InnerIndex>,
  patch_reader: &mut Option<IndexReader>,
  patch_reader_stamp: &mut Option<PatchReaderStamp>,
  doc_id: &str,
) -> Result<Option<Document>> {
  let current_stamp = {
    let manifest = inner.manifest.read();
    patch_reader_stamp_for_manifest(&manifest)
  };
  if patch_reader.is_none() || patch_reader_stamp.as_ref() != Some(&current_stamp) {
    *patch_reader = Some(IndexReader::open(inner.clone())?);
    *patch_reader_stamp = Some(current_stamp);
  }
  let Some(reader) = patch_reader.as_mut() else {
    bail!("patch reader unavailable");
  };
  let mut docs = reader.mget(&[doc_id.to_string()], true)?;
  let Some(found) = docs.pop() else {
    return Ok(None);
  };
  if !found.found {
    return Ok(None);
  }
  let source = found
    ._source
    .ok_or_else(|| anyhow!("stored fields are unavailable for document {doc_id}"))?;
  value_to_document(source).map(Some)
}

fn patch_reader_stamp_for_manifest(manifest: &Manifest) -> PatchReaderStamp {
  let mut hasher = DefaultHasher::new();
  manifest.segments.len().hash(&mut hasher);
  for seg in manifest.segments.iter() {
    seg.id.hash(&mut hasher);
    seg.generation.hash(&mut hasher);
    seg.doc_count.hash(&mut hasher);
    seg.max_doc_id.hash(&mut hasher);
    seg.blockmax.hash(&mut hasher);
    seg.deleted_docs.hash(&mut hasher);
    seg.paths.terms.hash(&mut hasher);
    seg.paths.postings.hash(&mut hasher);
    seg.paths.docstore.hash(&mut hasher);
    seg.paths.fast.hash(&mut hasher);
    seg.paths.meta.hash(&mut hasher);
    #[cfg(feature = "vectors")]
    seg.paths.vector_dir.hash(&mut hasher);
  }
  PatchReaderStamp {
    committed_at: manifest.committed_at.clone(),
    segment_fingerprint: hasher.finish(),
  }
}

fn pending_latest_from_ops(pending_ops: &[PendingOp]) -> HashMap<String, Option<Document>> {
  let mut latest = HashMap::new();
  for op in pending_ops.iter() {
    match op {
      PendingOp::Add { doc_id: id, doc } => {
        latest.insert(id.clone(), Some(doc.clone()));
      }
      PendingOp::Delete { doc_id: id } => {
        latest.insert(id.clone(), None);
      }
    }
  }
  latest
}

fn pending_doc_for_patch(
  pending_latest: &HashMap<String, Option<Document>>,
  doc_id: &str,
) -> Option<Option<Document>> {
  pending_latest.get(doc_id).cloned()
}

fn ensure_patch_safe(schema: &Schema) -> Result<()> {
  #[cfg(feature = "vectors")]
  if !schema.vector_fields.is_empty() {
    return Err(PatchError::VectorFieldsUnsupported.into());
  }
  for field in schema.resolved_fields().into_iter() {
    if (field.indexed || field.fast) && !field.stored {
      bail!(
        "cannot update documents: field `{}` is indexed/fast but not stored",
        field.path
      );
    }
  }
  Ok(())
}

fn validate_patch_fields(
  schema: &Schema,
  doc_id: &str,
  set: &BTreeMap<String, serde_json::Value>,
  unset: &[String],
) -> Result<()> {
  if set.is_empty() && unset.is_empty() {
    bail!("update must include at least one of set or unset");
  }
  validate_doc_id(doc_id)?;
  let doc_id_field = schema.doc_id_field();
  let patchable_paths = patchable_schema_paths(schema);
  for path in set.keys().chain(unset.iter()) {
    if path == doc_id_field {
      bail!("cannot update doc_id_field `{doc_id_field}`");
    }
    if !patchable_paths.contains(path) {
      bail!("unknown field {path}");
    }
  }
  Ok(())
}

fn patchable_schema_paths(schema: &Schema) -> HashSet<String> {
  let mut paths: HashSet<String> = schema
    .resolved_fields()
    .into_iter()
    .map(|field| field.path)
    .collect();
  for nested in schema.nested_fields.iter() {
    collect_nested_patchable_paths(nested, None, &mut paths);
  }
  paths
}

fn collect_nested_patchable_paths(
  nested: &NestedField,
  prefix: Option<&str>,
  out: &mut HashSet<String>,
) {
  let mut path = String::new();
  if let Some(parent) = prefix {
    path.push_str(parent);
    path.push('.');
  }
  path.push_str(&nested.name);
  out.insert(path.clone());
  for property in nested.fields.iter() {
    if let NestedProperty::Object(object) = property {
      collect_nested_patchable_paths(object, Some(&path), out);
    }
  }
}

fn literal_top_level_dotted_paths(schema: &Schema) -> HashSet<String> {
  let mut out = HashSet::new();
  for field in schema
    .text_fields
    .iter()
    .map(|field| field.name.as_str())
    .chain(
      schema
        .keyword_fields
        .iter()
        .map(|field| field.name.as_str()),
    )
    .chain(
      schema
        .numeric_fields
        .iter()
        .map(|field| field.name.as_str()),
    )
  {
    if field.contains('.') {
      out.insert(field.to_string());
    }
  }
  out
}

fn document_to_value(doc: &Document) -> Result<serde_json::Value> {
  let mut map = serde_json::Map::new();
  for (k, v) in doc.fields.iter() {
    map.insert(k.clone(), v.clone());
  }
  Ok(serde_json::Value::Object(map))
}

fn value_to_document(value: serde_json::Value) -> Result<Document> {
  let Some(obj) = value.as_object() else {
    bail!("document must be a JSON object");
  };
  let mut fields = BTreeMap::new();
  for (k, v) in obj.iter() {
    fields.insert(k.clone(), v.clone());
  }
  Ok(Document { fields })
}

#[cfg(test)]
fn set_path(root: &mut serde_json::Value, path: &str, value: serde_json::Value) -> Result<()> {
  set_path_with_literals(root, path, value, None)
}

fn set_path_with_literals(
  root: &mut serde_json::Value,
  path: &str,
  value: serde_json::Value,
  literal_paths: Option<&HashSet<String>>,
) -> Result<()> {
  if path.is_empty() {
    bail!("path must not be empty");
  }
  if path.split('.').any(|part| part.is_empty()) {
    bail!("path must not contain empty path segment");
  }
  if literal_paths.is_some_and(|paths| paths.contains(path)) {
    let Some(map) = root.as_object_mut() else {
      bail!("path {path} cannot traverse non-object");
    };
    map.insert(path.to_string(), value);
    return Ok(());
  }
  let parts: Vec<&str> = path.split('.').collect();
  set_path_parts(root, &parts, path, &value)
}

fn set_path_parts(
  current: &mut serde_json::Value,
  parts: &[&str],
  path: &str,
  value: &serde_json::Value,
) -> Result<()> {
  let Some((part, rest)) = parts.split_first() else {
    return Ok(());
  };
  match current {
    serde_json::Value::Object(map) => {
      if rest.is_empty() {
        map.insert((*part).to_string(), value.clone());
        return Ok(());
      }
      let entry = map
        .entry(*part)
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
      set_path_parts(entry, rest, path, value)
    }
    serde_json::Value::Array(items) => {
      for item in items.iter_mut() {
        set_path_parts(item, parts, path, value)?;
      }
      Ok(())
    }
    _ => bail!("path {path} cannot traverse non-object"),
  }
}

#[cfg(test)]
fn unset_path(root: &mut serde_json::Value, path: &str) -> Result<()> {
  unset_path_with_literals(root, path, None)
}

fn unset_path_with_literals(
  root: &mut serde_json::Value,
  path: &str,
  literal_paths: Option<&HashSet<String>>,
) -> Result<()> {
  if path.is_empty() {
    bail!("path must not be empty");
  }
  if path.split('.').any(|part| part.is_empty()) {
    bail!("path must not contain empty path segment");
  }
  if literal_paths.is_some_and(|paths| paths.contains(path)) {
    let Some(map) = root.as_object_mut() else {
      bail!("path {path} cannot traverse non-object");
    };
    map.remove(path);
    return Ok(());
  }
  let parts: Vec<&str> = path.split('.').collect();
  unset_path_parts(root, &parts, path)
}

fn unset_path_parts(current: &mut serde_json::Value, parts: &[&str], path: &str) -> Result<()> {
  let Some((part, rest)) = parts.split_first() else {
    return Ok(());
  };
  match current {
    serde_json::Value::Object(map) => {
      if rest.is_empty() {
        map.remove(*part);
        return Ok(());
      }
      let Some(next) = map.get_mut(*part) else {
        return Ok(());
      };
      unset_path_parts(next, rest, path)
    }
    serde_json::Value::Array(items) => {
      for item in items.iter_mut() {
        unset_path_parts(item, parts, path)?;
      }
      Ok(())
    }
    _ => bail!("path {path} cannot traverse non-object"),
  }
}

fn load_live_docs(inner: &InnerIndex, manifest: &Manifest) -> Result<HashMap<String, DocAddress>> {
  let mut map = HashMap::new();
  for seg in manifest.segments.iter() {
    let meta_bytes = inner
      .storage
      .read_to_end(Path::new(&seg.paths.meta))
      .map_err(|e| anyhow!("reading segment meta for {}: {}", seg.id, e))?;
    let seg_meta: SegmentFileMeta = serde_json::from_slice(&meta_bytes)?;
    if seg_meta.doc_ids.len() != seg_meta.doc_offsets.len() {
      bail!(
        "segment {} is missing document ids; reindex or compact to repair",
        seg.id
      );
    }
    let deleted: HashSet<u32> = seg.deleted_docs.iter().copied().collect();
    for (idx, doc_id) in seg_meta.doc_ids.iter().enumerate() {
      if deleted.contains(&(idx as u32)) {
        continue;
      }
      map.insert(
        doc_id.clone(),
        DocAddress {
          segment_id: seg.id.clone(),
          doc_id: idx as DocId,
        },
      );
    }
  }
  Ok(map)
}

#[cfg(test)]
mod tests {
  use std::collections::BTreeMap;
  use std::path::PathBuf;
  use std::sync::atomic::{AtomicBool, Ordering};
  use std::sync::Arc;

  use anyhow::anyhow;
  use parking_lot::{Mutex, RwLock};

  use super::PendingOp;
  use crate::api::types::{
    Document, IndexOptions, KeywordField, NestedField, NestedProperty, Schema, StorageType,
  };
  use crate::index::{directory, manifest::Manifest, wal::Wal, Index, InnerIndex};
  use crate::storage::{InMemoryStorage, Storage};
  use tempfile::tempdir;

  fn opts(path: &std::path::Path) -> IndexOptions {
    IndexOptions {
      path: path.to_path_buf(),
      create_if_missing: true,
      enable_positions: true,
      bm25_k1: 1.2,
      bm25_b: 0.75,
      storage: StorageType::Filesystem,
      #[cfg(feature = "vectors")]
      vector_defaults: None,
    }
  }

  fn nested_schema_for_patch() -> Schema {
    let mut schema = Schema::default_text_body();
    schema.nested_fields = vec![NestedField {
      name: "comment".into(),
      fields: vec![
        NestedProperty::Keyword(KeywordField {
          name: "author".into(),
          stored: true,
          indexed: true,
          fast: false,
          nullable: false,
        }),
        NestedProperty::Object(NestedField {
          name: "reply".into(),
          fields: vec![NestedProperty::Keyword(KeywordField {
            name: "author".into(),
            stored: true,
            indexed: true,
            fast: false,
            nullable: false,
          })],
          nullable: false,
        }),
      ],
      nullable: false,
    }];
    schema
  }

  struct FailingManifestStorage {
    inner: InMemoryStorage,
    fail_manifest: AtomicBool,
  }

  impl FailingManifestStorage {
    fn new(root: PathBuf) -> Self {
      Self {
        inner: InMemoryStorage::new(root),
        fail_manifest: AtomicBool::new(false),
      }
    }

    fn fail_next_manifest_store(&self) {
      self.fail_manifest.store(true, Ordering::SeqCst);
    }

    fn should_fail(&self, path: &std::path::Path) -> bool {
      path.ends_with("MANIFEST.json") && self.fail_manifest.swap(false, Ordering::SeqCst)
    }
  }

  impl Storage for FailingManifestStorage {
    fn root(&self) -> &std::path::Path {
      self.inner.root()
    }

    fn ensure_dir(&self, path: &std::path::Path) -> anyhow::Result<()> {
      self.inner.ensure_dir(path)
    }

    fn exists(&self, path: &std::path::Path) -> bool {
      self.inner.exists(path)
    }

    fn open_read(&self, path: &std::path::Path) -> anyhow::Result<crate::storage::DynFile> {
      self.inner.open_read(path)
    }

    fn open_write(&self, path: &std::path::Path) -> anyhow::Result<crate::storage::DynFile> {
      self.inner.open_write(path)
    }

    fn open_append(&self, path: &std::path::Path) -> anyhow::Result<crate::storage::DynFile> {
      self.inner.open_append(path)
    }

    fn read_to_end(&self, path: &std::path::Path) -> anyhow::Result<Vec<u8>> {
      self.inner.read_to_end(path)
    }

    fn write_all(&self, path: &std::path::Path, data: &[u8]) -> anyhow::Result<()> {
      self.inner.write_all(path, data)
    }

    fn atomic_write(&self, path: &std::path::Path, data: &[u8]) -> anyhow::Result<()> {
      if self.should_fail(path) {
        return Err(anyhow!("manifest write failed"));
      }
      self.inner.atomic_write(path, data)
    }

    fn remove(&self, path: &std::path::Path) -> anyhow::Result<()> {
      self.inner.remove(path)
    }

    fn remove_dir_all(&self, path: &std::path::Path) -> anyhow::Result<()> {
      self.inner.remove_dir_all(path)
    }
  }

  #[test]
  fn wal_retains_pending_when_manifest_store_fails() {
    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let storage = Arc::new(FailingManifestStorage::new(dir.path().to_path_buf()));
    let manifest_path = Manifest::manifest_path(dir.path());
    let manifest = Manifest::new(schema.clone());
    manifest.store(storage.as_ref(), &manifest_path).unwrap();

    let mut opts = opts(dir.path());
    opts.storage = StorageType::InMemory;
    let inner = Arc::new(InnerIndex {
      path: dir.path().to_path_buf(),
      options: opts,
      manifest: RwLock::new(manifest),
      writer_lock: Mutex::new(()),
      storage: storage.clone(),
    });

    storage.fail_next_manifest_store();

    let mut writer = super::IndexWriter::new(inner, None).unwrap();
    writer
      .add_document(&Document {
        fields: [
          ("_id".into(), serde_json::json!("1")),
          ("body".into(), serde_json::json!("commit wal safety")),
        ]
        .into_iter()
        .collect(),
      })
      .unwrap();
    let err = writer.commit();
    assert!(err.is_err());
    assert_eq!(
      writer
        .pending_ops
        .iter()
        .filter(|op| matches!(op, PendingOp::Add { .. }))
        .count(),
      1
    );

    let wal_path = directory::wal_path(dir.path());
    let (_, pending) = Wal::last_pending_ops(storage.as_ref(), &wal_path).unwrap();
    assert!(
      !pending.is_empty(),
      "wal should retain pending ops when manifest persistence fails"
    );
  }

  #[test]
  fn replay_pending_from_wal() {
    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("pending doc")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      // Drop without commit so wal retains entry.
    }
    let restored = idx.writer().unwrap();
    assert_eq!(
      restored
        .pending_ops
        .iter()
        .filter(|op| matches!(op, PendingOp::Add { .. }))
        .count(),
      1
    );
  }

  #[test]
  fn replay_pending_delete_from_wal() {
    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("to delete")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
    }
    {
      let mut writer = idx.writer().unwrap();
      writer.delete_document("1").unwrap();
      // Drop without commit so wal retains delete entry.
    }
    let mut restored = idx.writer().unwrap();
    assert_eq!(
      restored
        .pending_ops
        .iter()
        .filter(|op| matches!(op, PendingOp::Delete { .. }))
        .count(),
      1
    );
    restored.commit().unwrap();
    let manifest = idx.manifest();
    assert_eq!(manifest.segments.len(), 1);
    assert_eq!(manifest.segments[0].deleted_docs, vec![0]);
  }

  #[test]
  fn replay_add_then_delete_same_id() {
    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("original")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
    }
    {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("updated")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.delete_document("1").unwrap();
      // Drop without commit so wal retains ordered ops.
    }
    let mut restored = idx.writer().unwrap();
    assert_eq!(
      restored
        .pending_ops
        .iter()
        .filter(|op| matches!(op, PendingOp::Add { .. }))
        .count(),
      1
    );
    assert_eq!(
      restored
        .pending_ops
        .iter()
        .filter(|op| matches!(op, PendingOp::Delete { .. }))
        .count(),
      1
    );
    restored.commit().unwrap();
    let manifest = idx.manifest();
    assert_eq!(manifest.segments.len(), 1);
    assert_eq!(manifest.segments[0].deleted_docs, vec![0]);
  }

  #[test]
  fn rollback_clears_pending_and_wal() {
    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
    let mut writer = idx.writer().unwrap();
    writer
      .add_document(&Document {
        fields: [
          ("_id".into(), serde_json::json!("1")),
          ("body".into(), serde_json::json!("to rollback")),
        ]
        .into_iter()
        .collect(),
      })
      .unwrap();
    assert_eq!(
      writer
        .pending_ops
        .iter()
        .filter(|op| matches!(op, PendingOp::Add { .. }))
        .count(),
      1
    );
    writer.rollback().unwrap();
    assert!(writer.pending_ops.is_empty());
    let wal_path = crate::index::directory::wal_path(&writer.inner.path);
    assert_eq!(std::fs::metadata(wal_path).unwrap().len(), 0);
  }

  #[test]
  fn rollback_discards_pending_delete_ops() {
    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("to keep")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
    }
    let mut writer = idx.writer().unwrap();
    writer.delete_document("1").unwrap();
    assert_eq!(
      writer
        .pending_ops
        .iter()
        .filter(|op| matches!(op, PendingOp::Delete { .. }))
        .count(),
      1
    );
    writer.rollback().unwrap();
    assert!(writer.pending_ops.is_empty());
    let wal_path = crate::index::directory::wal_path(&writer.inner.path);
    assert_eq!(std::fs::metadata(wal_path).unwrap().len(), 0);
    let manifest = idx.manifest();
    assert!(manifest.segments[0].deleted_docs.is_empty());
  }

  #[test]
  #[cfg(feature = "write-key")]
  fn write_key_enforced_for_writer_open() {
    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let key = "super-secret-key";
    let idx = Index::create_with_write_key(dir.path(), schema.clone(), opts(dir.path()), Some(key))
      .unwrap();

    // Missing key -> error.
    assert!(idx.writer_with_key(None).is_err());

    // Wrong key -> error.
    assert!(idx.writer_with_key(Some("wrong")).is_err());

    // Correct key works and allows commit.
    let mut writer = idx.writer_with_key(Some(key)).unwrap();
    writer
      .add_document(&Document {
        fields: [
          ("_id".into(), serde_json::json!("1")),
          ("body".into(), serde_json::json!("hello")),
        ]
        .into_iter()
        .collect(),
      })
      .unwrap();
    writer.commit().unwrap();
  }

  #[test]
  fn commit_clears_wal_and_pending_entries() {
    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("commit durability")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
    }
    let wal_path = directory::wal_path(dir.path());
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    let (_, pending) = Wal::last_pending_ops(&storage, &wal_path).unwrap();
    assert!(
      pending.is_empty(),
      "pending WAL ops should be cleared on commit"
    );
    let wal_len = std::fs::metadata(&wal_path).unwrap().len();
    assert_eq!(wal_len, 0, "wal should be truncated after commit");
    let manifest = idx.manifest();
    assert_eq!(manifest.segments.len(), 1);
    assert_eq!(manifest.segments[0].doc_count, 1);
  }

  #[test]
  fn rollback_to_checkpoint_preserves_prior_pending() {
    let dir = tempdir().unwrap();
    let schema = Schema::default_text_body();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("doc-existing")),
            ("body".into(), serde_json::json!("existing")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      // drop writer without commit to leave pending WAL entries.
    }
    let mut writer = idx.writer().unwrap();
    assert_eq!(writer.pending_ops.len(), 1);
    let checkpoint = writer.checkpoint().unwrap();
    writer
      .add_document(&Document {
        fields: [
          ("_id".into(), serde_json::json!("doc-batch")),
          ("body".into(), serde_json::json!("new batch doc")),
        ]
        .into_iter()
        .collect(),
      })
      .unwrap();
    assert_eq!(writer.pending_ops.len(), 2);
    writer.rollback_to(checkpoint).unwrap();
    assert_eq!(
      writer.pending_ops.len(),
      1,
      "rollback_to should keep prior pending ops intact"
    );
    writer.commit().unwrap();
    let manifest = idx.manifest();
    assert_eq!(manifest.segments.len(), 1);
    assert_eq!(manifest.segments[0].doc_count, 1);
  }

  #[test]
  fn validate_patch_fields_accepts_nested_roots() {
    let schema = nested_schema_for_patch();
    let mut set = BTreeMap::new();
    set.insert(
      "comment".to_string(),
      serde_json::json!([{ "author": "alice", "reply": { "author": "bob" } }]),
    );
    set.insert(
      "comment.reply".to_string(),
      serde_json::json!({ "author": "carol" }),
    );
    super::validate_patch_fields(&schema, "doc-1", &set, &[]).unwrap();
  }

  #[test]
  fn validate_patch_fields_rejects_unknown_nested_roots() {
    let schema = nested_schema_for_patch();
    let mut set = BTreeMap::new();
    set.insert("comment.missing".to_string(), serde_json::json!("nope"));
    let err = super::validate_patch_fields(&schema, "doc-1", &set, &[]).unwrap_err();
    assert!(err.to_string().contains("unknown field comment.missing"));
  }

  #[test]
  fn validate_patch_fields_rejects_literal_dotted_parent_path() {
    let mut schema = Schema::default_text_body();
    schema.keyword_fields.push(KeywordField {
      name: "a.b".into(),
      stored: true,
      indexed: true,
      fast: false,
      nullable: false,
    });
    let mut set = BTreeMap::new();
    set.insert("a".to_string(), serde_json::json!("nope"));
    let err = super::validate_patch_fields(&schema, "doc-1", &set, &[]).unwrap_err();
    assert!(err.to_string().contains("unknown field a"));
  }

  #[test]
  fn apply_patch_allows_replacing_nested_root() {
    let dir = tempdir().unwrap();
    let schema = nested_schema_for_patch();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("before")),
            (
              "comment".into(),
              serde_json::json!([{ "author": "alice", "reply": { "author": "bob" } }]),
            ),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
    }
    {
      let mut writer = idx.writer().unwrap();
      let mut set = BTreeMap::new();
      set.insert(
        "comment".to_string(),
        serde_json::json!([{ "author": "eve", "reply": { "author": "mallory" } }]),
      );
      writer.apply_patch("1", &set, &[]).unwrap();
      writer.commit().unwrap();
    }
    let reader = idx.reader().unwrap();
    let docs = reader.mget(&["1".to_string()], true).unwrap();
    assert_eq!(docs.len(), 1);
    assert!(docs[0].found);
    assert_eq!(
      docs[0]._source.as_ref().unwrap()["comment"],
      serde_json::json!([{ "author": "eve", "reply": { "author": "mallory" } }])
    );
  }

  #[test]
  fn apply_patch_updates_dotted_paths_through_nested_arrays() {
    let dir = tempdir().unwrap();
    let schema = nested_schema_for_patch();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("before")),
            (
              "comment".into(),
              serde_json::json!([
                { "author": "alice", "reply": { "author": "bob" } },
                { "author": "carol", "reply": { "author": "dave" } }
              ]),
            ),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
    }
    {
      let mut writer = idx.writer().unwrap();
      let mut set = BTreeMap::new();
      set.insert("comment.reply.author".to_string(), serde_json::json!("eve"));
      writer.apply_patch("1", &set, &[]).unwrap();
      writer.commit().unwrap();
    }
    let reader = idx.reader().unwrap();
    let docs = reader.mget(&["1".to_string()], true).unwrap();
    assert_eq!(docs.len(), 1);
    assert!(docs[0].found);
    let source = docs[0]._source.as_ref().unwrap();
    assert_eq!(source["comment"][0]["reply"]["author"], "eve");
    assert_eq!(source["comment"][1]["reply"]["author"], "eve");
  }

  #[test]
  fn apply_patch_reuses_fresh_committed_state_across_commits() {
    let dir = tempdir().unwrap();
    let schema = nested_schema_for_patch();
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("before")),
            (
              "comment".into(),
              serde_json::json!([{ "author": "alice", "reply": { "author": "bob" } }]),
            ),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
    }
    let mut writer = idx.writer().unwrap();
    let mut first = BTreeMap::new();
    first.insert(
      "comment".to_string(),
      serde_json::json!([{ "author": "eve", "reply": { "author": "mallory" } }]),
    );
    writer.apply_patch("1", &first, &[]).unwrap();
    writer.commit().unwrap();
    let mut second = BTreeMap::new();
    second.insert("body".to_string(), serde_json::json!("after"));
    writer.apply_patch("1", &second, &[]).unwrap();
    writer.commit().unwrap();

    let reader = idx.reader().unwrap();
    let docs = reader.mget(&["1".to_string()], true).unwrap();
    assert_eq!(docs.len(), 1);
    assert!(docs[0].found);
    let source = docs[0]._source.as_ref().unwrap();
    assert_eq!(source["body"], "after");
    assert_eq!(
      source["comment"],
      serde_json::json!([{ "author": "eve", "reply": { "author": "mallory" } }])
    );
  }

  #[test]
  fn apply_patch_refreshes_reader_after_external_commit() {
    let dir = tempdir().unwrap();
    let mut schema = Schema::default_text_body();
    schema.keyword_fields.push(KeywordField {
      name: "status".into(),
      stored: true,
      indexed: true,
      fast: false,
      nullable: false,
    });
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("first")),
            ("status".into(), serde_json::json!("old")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("2")),
            ("body".into(), serde_json::json!("second")),
            ("status".into(), serde_json::json!("old")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
    }

    let mut writer = idx.writer().unwrap();
    let mut warm_reader = BTreeMap::new();
    warm_reader.insert("body".to_string(), serde_json::json!("writer-a-first"));
    writer.apply_patch("1", &warm_reader, &[]).unwrap();

    {
      let mut other_writer = idx.writer().unwrap();
      let mut set = BTreeMap::new();
      set.insert("status".to_string(), serde_json::json!("updated-by-other"));
      other_writer.apply_patch("2", &set, &[]).unwrap();
      other_writer.commit().unwrap();
    }

    let mut second_patch = BTreeMap::new();
    second_patch.insert("body".to_string(), serde_json::json!("writer-a-second"));
    writer.apply_patch("2", &second_patch, &[]).unwrap();
    writer.commit().unwrap();

    let reader = idx.reader().unwrap();
    let docs = reader.mget(&["2".to_string()], true).unwrap();
    assert_eq!(docs.len(), 1);
    assert!(docs[0].found);
    let source = docs[0]._source.as_ref().unwrap();
    assert_eq!(source["status"], "updated-by-other");
    assert_eq!(source["body"], "writer-a-second");
  }

  #[test]
  fn apply_patch_treats_dotted_top_level_field_as_literal_key() {
    let dir = tempdir().unwrap();
    let mut schema = Schema::default_text_body();
    schema.keyword_fields.push(KeywordField {
      name: "a.b".into(),
      stored: true,
      indexed: true,
      fast: false,
      nullable: true,
    });
    let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
    {
      let mut writer = idx.writer().unwrap();
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!("1")),
            ("body".into(), serde_json::json!("before")),
            ("a.b".into(), serde_json::json!("old")),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
      writer.commit().unwrap();
    }
    {
      let mut writer = idx.writer().unwrap();
      let mut set = BTreeMap::new();
      set.insert("a.b".to_string(), serde_json::json!("new"));
      writer.apply_patch("1", &set, &[]).unwrap();
      writer.commit().unwrap();
    }

    let reader = idx.reader().unwrap();
    let docs = reader.mget(&["1".to_string()], true).unwrap();
    assert_eq!(docs.len(), 1);
    assert!(docs[0].found);
    let source = docs[0]._source.as_ref().unwrap();
    assert_eq!(source["a.b"], "new");
    assert!(source.get("a").is_none());
  }

  #[test]
  fn set_path_rejects_empty_segments() {
    let invalid_paths = ["a..b", ".field", "field."];
    for path in invalid_paths.iter() {
      let mut value = serde_json::json!({});
      let err = super::set_path(&mut value, path, serde_json::json!(1)).unwrap_err();
      assert!(err.to_string().contains("empty path segment"));
    }
  }

  #[test]
  fn unset_path_rejects_empty_segments() {
    let invalid_paths = ["a..b", ".field", "field."];
    for path in invalid_paths.iter() {
      let mut value = serde_json::json!({ "field": { "x": 1 } });
      let err = super::unset_path(&mut value, path).unwrap_err();
      assert!(err.to_string().contains("empty path segment"));
    }
  }

  #[test]
  fn set_path_traverses_arrays() {
    let mut value = serde_json::json!({
      "comment": [
        { "reply": { "author": "alice" } },
        { "reply": { "author": "bob" } }
      ]
    });
    super::set_path(&mut value, "comment.reply.author", serde_json::json!("eve")).unwrap();
    assert_eq!(value["comment"][0]["reply"]["author"], "eve");
    assert_eq!(value["comment"][1]["reply"]["author"], "eve");
  }

  #[test]
  fn unset_path_traverses_arrays() {
    let mut value = serde_json::json!({
      "comment": [
        { "reply": { "author": "alice", "score": 1 } },
        { "reply": { "author": "bob", "score": 2 } }
      ]
    });
    super::unset_path(&mut value, "comment.reply.author").unwrap();
    assert!(value["comment"][0]["reply"].get("author").is_none());
    assert!(value["comment"][1]["reply"].get("author").is_none());
    assert_eq!(value["comment"][0]["reply"]["score"], 1);
    assert_eq!(value["comment"][1]["reply"]["score"], 2);
  }
}
