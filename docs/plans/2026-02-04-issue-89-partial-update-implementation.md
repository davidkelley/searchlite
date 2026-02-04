# Partial Update Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add partial update APIs (set/unset) for documents with best-effort bulk updates, preserving WAL/commit behavior and preventing data loss.

**Architecture:** Implement `IndexWriter::apply_patch` to read the current stored doc (pending ops first, then committed), apply a JSON patch (unset then set), validate against schema, and enqueue a normal add. HTTP endpoints call this writer method; bulk updates parse NDJSON pairs and return per-item results.

**Tech Stack:** Rust, searchlite-core, searchlite-http, axum, serde_json.

---

### Task 1: Core Update Happy-Path Tests (TDD)

**Files:**
- Create: `searchlite-core/tests/partial_update.rs`

**Step 1: Write the failing tests**

```rust
use std::collections::BTreeMap;

use searchlite_core::api::types::{Document, IndexOptions, Schema, StorageType};
use searchlite_core::Index;
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

#[test]
fn update_set_unset_top_level_fields() {
  let dir = tempdir().unwrap();
  let schema = Schema::default_text_body();
  let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();

  let mut writer = idx.writer().unwrap();
  writer
    .add_document(&Document {
      fields: [
        ("_id".into(), serde_json::json!("doc-1")),
        ("body".into(), serde_json::json!("hello")),
        ("count".into(), serde_json::json!(5)),
      ]
      .into_iter()
      .collect(),
    })
    .unwrap();
  writer.commit().unwrap();

  let mut set = BTreeMap::new();
  set.insert("count".to_string(), serde_json::json!(10));
  let unset = vec!["body".to_string()];

  let mut writer = idx.writer().unwrap();
  writer.apply_patch("doc-1", &set, &unset).unwrap();
  writer.commit().unwrap();

  let reader = idx.reader().unwrap();
  let res = reader.mget(&["doc-1".to_string()], true).unwrap();
  let doc = res[0]._source.clone().unwrap();
  assert_eq!(doc["count"], 10);
  assert!(doc.get("body").is_none());
}

#[test]
fn update_supports_nested_paths() {
  let dir = tempdir().unwrap();
  let schema: Schema = serde_json::from_value(serde_json::json!({
    "doc_id_field": "_id",
    "text_fields": [{ "name": "body", "analyzer": "default", "stored": true, "indexed": true, "nullable": false }],
    "keyword_fields": [],
    "numeric_fields": [],
    "nested_fields": [
      { "name": "metadata", "nullable": true, "fields": [
          { "type": "keyword", "name": "alt", "stored": true, "indexed": true, "fast": false, "nullable": true }
        ]
      }
    ]
  }))
  .unwrap();
  let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();

  let mut writer = idx.writer().unwrap();
  writer
    .add_document(&Document {
      fields: [
        ("_id".into(), serde_json::json!("doc-2")),
        ("body".into(), serde_json::json!("hello")),
        ("metadata".into(), serde_json::json!({ "alt": "v1" })),
      ]
      .into_iter()
      .collect(),
    })
    .unwrap();
  writer.commit().unwrap();

  let mut set = BTreeMap::new();
  set.insert("metadata.alt".to_string(), serde_json::json!("v2"));

  let mut writer = idx.writer().unwrap();
  writer.apply_patch("doc-2", &set, &[]).unwrap();
  writer.commit().unwrap();

  let reader = idx.reader().unwrap();
  let res = reader.mget(&["doc-2".to_string()], true).unwrap();
  let doc = res[0]._source.clone().unwrap();
  assert_eq!(doc["metadata"]["alt"], "v2");
}
```

**Step 2: Run tests to verify failure**

Run: `cargo test -p searchlite-core partial_update -- --nocapture`
Expected: FAIL (missing `apply_patch` on `IndexWriter`).

---

### Task 2: Core Apply-Patch Implementation

**Files:**
- Modify: `searchlite-core/src/api/writer.rs`

**Step 1: Implement minimal `apply_patch` + helpers**

```rust
impl IndexWriter {
  pub fn apply_patch(
    &mut self,
    doc_id: &str,
    set: &BTreeMap<String, serde_json::Value>,
    unset: &[String],
  ) -> Result<()> {
    let _guard = self.inner.writer_lock.lock();
    ensure_patch_safe(&self.schema)?;
    validate_patch_fields(&self.schema, doc_id, set, unset)?;

    let mut doc = resolve_doc_for_patch(self, doc_id)?
      .ok_or_else(|| anyhow!("document not found"))?;

    let mut value = document_to_value(&doc)?;
    for path in unset {
      unset_path(&mut value, path)?;
    }
    for (path, val) in set.iter() {
      set_path(&mut value, path, val.clone())?;
    }

    doc = value_to_document(value)?;
    self.schema.validate_document(&doc)?;
    self.add_document_locked(&doc)?;
    Ok(())
  }
}
```

Add helper functions in the same module:
- `ensure_patch_safe(schema)` (reject indexed/fast fields that are not stored)
- `validate_patch_fields(schema, doc_id, set, unset)` (non-empty patch, no doc_id mutation, schema path validation)
- `resolve_doc_for_patch(writer, doc_id)` (pending ops first, then reader.mget with `return_stored=true`)
- `document_to_value` / `value_to_document`
- `set_path` / `unset_path` dot-walk helpers that only traverse objects and error on non-objects.

**Step 2: Run tests**

Run: `cargo test -p searchlite-core partial_update -- --nocapture`
Expected: PASS (happy-path tests).

**Step 3: Commit**

```bash
git add searchlite-core/src/api/writer.rs searchlite-core/tests/partial_update.rs
git commit -m "feat(core): add partial update writer support"
```

---

### Task 3: Core Error/Validation Tests and Fixes

**Files:**
- Modify: `searchlite-core/tests/partial_update.rs`
- Modify: `searchlite-core/src/api/writer.rs`

**Step 1: Add failing tests for error cases**

```rust
#[test]
fn update_rejects_missing_doc() {
  let dir = tempdir().unwrap();
  let schema = Schema::default_text_body();
  let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();

  let mut writer = idx.writer().unwrap();
  let err = writer.apply_patch("missing", &BTreeMap::new(), &[]).unwrap_err();
  assert!(err.to_string().contains("document not found"));
}

#[test]
fn update_rejects_doc_id_mutation() {
  let dir = tempdir().unwrap();
  let schema = Schema::default_text_body();
  let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();
  let mut writer = idx.writer().unwrap();
  writer
    .add_document(&Document {
      fields: [
        ("_id".into(), serde_json::json!("doc-1")),
        ("body".into(), serde_json::json!("hello")),
      ]
      .into_iter()
      .collect(),
    })
    .unwrap();
  writer.commit().unwrap();

  let mut set = BTreeMap::new();
  set.insert("_id".to_string(), serde_json::json!("other"));
  let mut writer = idx.writer().unwrap();
  let err = writer.apply_patch("doc-1", &set, &[]).unwrap_err();
  assert!(err.to_string().contains("doc_id_field"));
}

#[test]
fn update_rejects_nonstored_indexed_fields() {
  let dir = tempdir().unwrap();
  let schema: Schema = serde_json::from_value(serde_json::json!({
    "doc_id_field": "_id",
    "text_fields": [
      { "name": "body", "analyzer": "default", "stored": false, "indexed": true, "nullable": false }
    ],
    "keyword_fields": [],
    "numeric_fields": [],
    "nested_fields": []
  }))
  .unwrap();
  let idx = Index::create(dir.path(), schema, opts(dir.path())).unwrap();

  let mut writer = idx.writer().unwrap();
  writer
    .add_document(&Document {
      fields: [
        ("_id".into(), serde_json::json!("doc-1")),
        ("body".into(), serde_json::json!("hello")),
      ]
      .into_iter()
      .collect(),
    })
    .unwrap();
  writer.commit().unwrap();

  let mut set = BTreeMap::new();
  set.insert("body".to_string(), serde_json::json!("new"));
  let mut writer = idx.writer().unwrap();
  let err = writer.apply_patch("doc-1", &set, &[]).unwrap_err();
  assert!(err.to_string().contains("indexed/fast but not stored"));
}
```

**Step 2: Run tests to verify failure**

Run: `cargo test -p searchlite-core partial_update -- --nocapture`
Expected: FAIL with validation errors not yet implemented.

**Step 3: Implement validation fixes**
- Enforce non-empty patch, doc_id mutation rejection, schema-path validation, and non-stored indexed/fast field rejection.

**Step 4: Run tests**

Run: `cargo test -p searchlite-core partial_update -- --nocapture`
Expected: PASS.

**Step 5: Commit**

```bash
git add searchlite-core/src/api/writer.rs searchlite-core/tests/partial_update.rs
git commit -m "feat(core): validate partial update patches"
```

---

### Task 4: HTTP Single Update Endpoint

**Files:**
- Modify: `searchlite-http/src/lib.rs`

**Step 1: Add failing HTTP test**

```rust
#[tokio::test]
async fn http_supports_update_document() {
  init_tracing();
  let dir = tempdir().unwrap();
  let index_path = dir.path().join("idx-update");
  let (client, _base, index_base, handle, _state, _args) = setup_server(index_path).await;

  let schema = Schema::default_text_body();
  client.post(format!("{index_base}/init")).json(&schema).send().await.unwrap();

  let bulk = serde_json::json!({
    "docs": [ { "_id": "doc-1", "body": "hello" } ]
  });
  client.post(format!("{index_base}/bulk")).json(&bulk).send().await.unwrap();
  client.post(format!("{index_base}/commit")).send().await.unwrap();

  let update = serde_json::json!({
    "id": "doc-1",
    "set": { "body": "updated" },
    "unset": []
  });
  let res = client.post(format!("{index_base}/update")).json(&update).send().await.unwrap();
  assert!(res.status().is_success());
  client.post(format!("{index_base}/commit")).send().await.unwrap();

  let mget = serde_json::json!({ "ids": ["doc-1"], "return_stored": true });
  let res = client.post(format!("{index_base}/mget")).json(&mget).send().await.unwrap();
  let body: serde_json::Value = res.json().await.unwrap();
  assert_eq!(body["docs"][0]["_source"]["body"], "updated");

  handle.abort();
  let _ = handle.await;
}
```

**Step 2: Run test to verify failure**

Run: `cargo test -p searchlite-http http_supports_update_document -- --nocapture`
Expected: FAIL (route missing).

**Step 3: Implement endpoint**
- Add route: `.route("/indexes/:name/update", post(update_document))`
- Add `UpdateRequest { id: String, set: BTreeMap<String, Value>, unset: Vec<String> }`
- Add `UpdateResponse { updated: bool }`
- Implement `update_document` similar to `delete_documents` writer flow (write-key checks, writer lock, spawn_blocking), calling `writer.apply_patch` and mapping errors to 400/404.

**Step 4: Run test**

Run: `cargo test -p searchlite-http http_supports_update_document -- --nocapture`
Expected: PASS.

**Step 5: Commit**

```bash
git add searchlite-http/src/lib.rs
git commit -m "feat(http): add update document endpoint"
```

---

### Task 5: HTTP Bulk Update Endpoint (Best-Effort)

**Files:**
- Modify: `searchlite-http/src/lib.rs`

**Step 1: Add failing HTTP test**

```rust
#[tokio::test]
async fn http_supports_bulk_update_best_effort() {
  init_tracing();
  let dir = tempdir().unwrap();
  let index_path = dir.path().join("idx-bulk-update");
  let (client, _base, index_base, handle, _state, _args) = setup_server(index_path).await;

  let schema = Schema::default_text_body();
  client.post(format!("{index_base}/init")).json(&schema).send().await.unwrap();

  let bulk = serde_json::json!({
    "docs": [ { "_id": "doc-1", "body": "hello" } ]
  });
  client.post(format!("{index_base}/bulk")).json(&bulk).send().await.unwrap();
  client.post(format!("{index_base}/commit")).send().await.unwrap();

  let ndjson = [
    r#"{"update":{"_id":"doc-1"}}"#,
    r#"{"set":{"body":"updated"}}"#,
    r#"{"update":{"_id":"missing"}}"#,
    r#"{"set":{"body":"nope"}}"#,
    "",
  ]
  .join("\n");

  let res = client
    .post(format!("{index_base}/_bulk_update"))
    .body(ndjson)
    .send()
    .await
    .unwrap();
  assert!(res.status().is_success());
  let body: serde_json::Value = res.json().await.unwrap();
  assert_eq!(body["updated"], 1);
  assert_eq!(body["failed"], 1);

  handle.abort();
  let _ = handle.await;
}
```

**Step 2: Run test to verify failure**

Run: `cargo test -p searchlite-http http_supports_bulk_update_best_effort -- --nocapture`
Expected: FAIL (route missing).

**Step 3: Implement endpoint**
- Add route: `.route("/indexes/:name/_bulk_update", post(bulk_update))`
- Parse request body as NDJSON pairs: action line `{ "update": { "_id": "..." } }`, data line `{ "set": { ... }, "unset": [...] }`.
- Validate and accumulate per-item results. Best-effort: errors recorded per item and processing continues.
- Apply updates in a single `spawn_blocking` block with writer lock, iterating update list and calling `writer.apply_patch` per item.
- Response shape: `{ "updated": <u64>, "failed": <u64>, "items": [ { "id": "...", "status": 200|404|400, "error": "..." } ] }`.

**Step 4: Run test**

Run: `cargo test -p searchlite-http http_supports_bulk_update_best_effort -- --nocapture`
Expected: PASS.

**Step 5: Commit**

```bash
git add searchlite-http/src/lib.rs
git commit -m "feat(http): add bulk update endpoint"
```

---

### Task 6: OpenAPI + Docs

**Files:**
- Modify: `openapi.yaml`
- Modify: `README.md`

**Step 1: Document new endpoints**
- Add `/indexes/{name}/update` and `/indexes/{name}/_bulk_update` paths with request/response schemas.
- Include error responses and examples.

**Step 2: Update README**
- Add usage snippets for single update and bulk update.
- Document constraints (stored fields required, arrays not traversed, best-effort bulk).

**Step 3: Commit**

```bash
git add openapi.yaml README.md
git commit -m "docs: document partial update APIs"
```

---

### Task 7: Quality Gates + Coverage (>90%)

**Files:**
- Modify tests as needed to hit coverage target.

**Step 1: Format**

Run: `cargo fmt --all`
Expected: no output.

**Step 2: Lint**

Run: `cargo clippy --all --all-features --all-targets -- -D warnings`
Expected: PASS.

**Step 3: Build**

Run: `cargo build --all --all-features`
Expected: PASS.

**Step 4: Test**

Run: `cargo test --all --all-features`
Expected: PASS.

**Step 5: Coverage (>= 90%)**

Run: `cargo llvm-cov --all --all-features --workspace --summary`
Expected: `LINE` coverage >= 90%.
If below target, add focused tests in `searchlite-core/tests/partial_update.rs` and `searchlite-http/src/lib.rs` until coverage meets target.

**Step 6: Bench (if perf-sensitive)**

Run: `cargo bench -p searchlite-core`
Expected: PASS.

**Step 7: Final Commit (if additional tests added for coverage)**

```bash
git add searchlite-core/tests/partial_update.rs searchlite-http/src/lib.rs
git commit -m "test: boost coverage for partial update"
```
