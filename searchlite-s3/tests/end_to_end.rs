//! Stage 10c end-to-end smoke: bake locally → sync to S3 → open
//! read-only → search + mget.
//!
//! Drives the full intended deployment loop against a stateful
//! `wiremock` server that implements the subset of the S3 protocol
//! we use:
//!
//! * `HEAD bucket/key` → 200 + Content-Length + ETag (deterministic
//!   from content), or 404 if absent.
//! * `GET bucket/key` (no Range) → full body + ETag.
//! * `GET bucket/key` with `Range: bytes=N-M` → 206 + exact slice +
//!   Content-Range + ETag.
//! * `PUT bucket/key` → store bytes, return new ETag.
//! * `DELETE bucket/key` → remove + 204.
//!
//! This catches both path-resolution errors (Codex Stage 10c v1
//! finding) and range-read regressions in one shot — the local
//! reader's results have to match the S3-backed reader's results
//! exactly.
//!
//! ## Known macOS test-concurrency flake
//!
//! `aws-smithy-http-client` eagerly initializes a rustls TLS layer
//! during `aws_sdk_s3::Client::from_conf` even when the endpoint is
//! `http://` (we never actually negotiate TLS against wiremock).
//! That init walks the macOS keychain via `rustls-native-certs`,
//! which races between processes when multiple `cargo test` binaries
//! run in parallel and surfaces as:
//!
//!   `TrustStore configured to enable native roots but no valid
//!   root certificates parsed!`
//!
//! `S3BlobStore::new` mitigates the **within-process** variant via a
//! global `Mutex` around the synchronous `Client::from_conf` step;
//! cross-process serialization isn't possible from a library and
//! isn't worth a file-lock workaround for a test-only issue. Users
//! never hit this in production (one `S3BlobStore::new` per
//! process, at startup).
//!
//! If you see those panics under `cargo test --workspace
//! --all-features`, retry: the per-package run
//! (`cargo test -p searchlite-s3 --tests`) is the canonical
//! invocation and the failure is environmental, not a regression.

#![cfg(not(target_arch = "wasm32"))]

mod common;

use std::path::Path;
use std::sync::Arc;

use searchlite_core::api::types::{Document, IndexOptions};
use searchlite_core::{Index, Schema};
use searchlite_s3::{open_index_read_only, sync_to_s3, S3Config, S3Credentials};

use crate::common::stateful_mock::{spawn_stateful_s3_mock, StatefulS3Bucket};

fn local_opts(path: &Path) -> IndexOptions {
  // Defaults (FS storage, Strict checksum, BM25 defaults, etc.) plus
  // the two fields this test needs to override. Avoids re-listing
  // every field — `..Default::default()` is robust to future
  // additions on `IndexOptions`.
  IndexOptions {
    path: path.to_path_buf(),
    create_if_missing: true,
    ..Default::default()
  }
}

/// Build a small local index: 5 docs, two commits → two segments
/// → compact → one segment. Returns the tempdir, kept alive by
/// the caller.
fn bake_local_index() -> (tempfile::TempDir, Vec<(String, String)>) {
  let dir = tempfile::tempdir().unwrap();
  let docs: Vec<(String, String)> = vec![
    ("doc1".into(), "alpha bravo charlie".into()),
    ("doc2".into(), "bravo charlie delta".into()),
    ("doc3".into(), "charlie delta echo".into()),
    ("doc4".into(), "delta echo foxtrot".into()),
    ("doc5".into(), "echo foxtrot golf".into()),
  ];
  let schema = Schema::default_text_body();
  let idx = Index::create(dir.path(), schema, local_opts(dir.path())).unwrap();
  for chunk in docs.chunks(3) {
    let mut writer = idx.writer().unwrap();
    for (id, body) in chunk {
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!(id)),
            ("body".into(), serde_json::json!(body)),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
    }
    writer.commit().unwrap();
  }
  // Compact down to a single segment so the sync upload is small
  // and the smoke test exercises the post-compact shape.
  idx.compact().unwrap();
  drop(idx);
  (dir, docs)
}

fn config_for(server_uri: &str, bucket: &str, prefix: Option<&str>) -> S3Config {
  S3Config {
    endpoint_url: Some(server_uri.to_string()),
    region: "us-east-1".into(),
    bucket: bucket.into(),
    prefix: prefix.map(|s| s.to_string()),
    credentials: S3Credentials::Static {
      access_key_id: "test-key".into(),
      secret_access_key: "test-secret".into(),
      session_token: None,
    },
    conditional_put: true,
    force_path_style: true,
  }
}

/// Stage 10c smoke: bake → sync → open RO → search returns the
/// same hits as the local reader did pre-drop.
#[tokio::test(flavor = "multi_thread")]
async fn end_to_end_bake_sync_open_search_with_prefix() {
  let (local_dir, _) = bake_local_index();

  // Capture the local reader's ground truth for the same query
  // before we drop the local index.
  let baseline_hits = {
    let local_opts = local_opts(local_dir.path());
    let mut reopen = local_opts.clone();
    reopen.create_if_missing = false;
    let local = Index::open(reopen).unwrap();
    let reader = local.reader().unwrap();
    let req: searchlite_core::api::types::SearchRequest =
      serde_json::from_value(serde_json::json!({
        "query": "delta",
        "limit": 10,
        "track_total_hits": true,
        "return_stored": true,
      }))
      .unwrap();
    let result = reader.search(&req).unwrap();
    result.total_hits_estimate
  };

  let (server_uri, bucket) = spawn_stateful_s3_mock("test-bucket").await;

  // Sync local files to S3 under prefix "idx-baked".
  let report = sync_to_s3(
    local_dir.path(),
    config_for(&server_uri, &bucket, Some("idx-baked")),
  )
  .await
  .unwrap();
  assert!(
    report.files >= 5,
    "sync_to_s3 must upload at least the standard 5 segment files; got {report:?}"
  );

  // Open the index over S3 with the same prefix and verify it
  // serves the same hits as the local reader.
  let s3_idx = open_index_read_only(config_for(&server_uri, &bucket, Some("idx-baked")))
    .await
    .unwrap();
  let reader = s3_idx.reader().unwrap();
  let req: searchlite_core::api::types::SearchRequest = serde_json::from_value(serde_json::json!({
    "query": "delta",
    "limit": 10,
    "track_total_hits": true,
    "return_stored": true,
  }))
  .unwrap();
  let result = reader.search(&req).unwrap();
  assert_eq!(
    result.total_hits_estimate, baseline_hits,
    "S3-backed search must return the same hit count as the local reader"
  );
  assert!(
    result.hits.iter().all(|h| h.fields.is_some()),
    "_source must be served from S3"
  );
}

/// Stage 10c smoke variant: prefix = None (root-bucket addressing).
/// Proves the no-prefix shape resolves to bucket/MANIFEST.json,
/// not bucket//MANIFEST.json or any double-prefix variant.
#[tokio::test(flavor = "multi_thread")]
async fn end_to_end_bake_sync_open_search_without_prefix() {
  let (local_dir, _) = bake_local_index();
  let (server_uri, bucket) = spawn_stateful_s3_mock("test-bucket").await;

  let report = sync_to_s3(local_dir.path(), config_for(&server_uri, &bucket, None))
    .await
    .unwrap();
  assert!(report.files >= 5);

  let s3_idx = open_index_read_only(config_for(&server_uri, &bucket, None))
    .await
    .unwrap();
  let reader = s3_idx.reader().unwrap();
  let req: searchlite_core::api::types::SearchRequest = serde_json::from_value(serde_json::json!({
    "query": "alpha",
    "limit": 10,
    "track_total_hits": true,
  }))
  .unwrap();
  let result = reader.search(&req).unwrap();
  assert_eq!(result.total_hits_estimate, 1);
}

/// Stage 10c: mget through the S3-backed reader returns the same
/// stored payloads as the local reader.
#[tokio::test(flavor = "multi_thread")]
async fn end_to_end_mget_returns_source_from_s3() {
  let (local_dir, _) = bake_local_index();
  let (server_uri, bucket) = spawn_stateful_s3_mock("test-bucket").await;
  sync_to_s3(
    local_dir.path(),
    config_for(&server_uri, &bucket, Some("idx")),
  )
  .await
  .unwrap();
  let s3_idx = open_index_read_only(config_for(&server_uri, &bucket, Some("idx")))
    .await
    .unwrap();
  let reader = s3_idx.reader().unwrap();
  let results = reader
    .mget(&["doc1".to_string(), "doc3".to_string()], true)
    .unwrap();
  assert_eq!(results.len(), 2);
  for r in &results {
    assert!(r.found, "{} should be found", r.doc_id);
    let source = r._source.as_ref().expect("_source populated");
    let body = source.get("body").and_then(|v| v.as_str()).unwrap();
    assert!(
      body.contains("alpha") || body.contains("charlie"),
      "body for {} mismatched: {body}",
      r.doc_id
    );
  }
}

/// Stage 10c [Codex review]: every mutator must error on an
/// `open_index_read_only`-served index, regardless of how it was
/// opened. The Stage 10a check is unconditional; this asserts the
/// S3-backed open path actually sets `read_only = true`.
#[tokio::test(flavor = "multi_thread")]
async fn s3_backed_index_refuses_every_mutator() {
  let (local_dir, _) = bake_local_index();
  let (server_uri, bucket) = spawn_stateful_s3_mock("test-bucket").await;
  sync_to_s3(local_dir.path(), config_for(&server_uri, &bucket, None))
    .await
    .unwrap();
  let s3_idx = open_index_read_only(config_for(&server_uri, &bucket, None))
    .await
    .unwrap();
  let writer_err = match s3_idx.writer() {
    Ok(_) => panic!("S3-backed writer() must error"),
    Err(e) => e,
  };
  assert!(format!("{writer_err:#}").contains("read-only"));
  assert!(s3_idx.compact().is_err());
  assert!(s3_idx.merge_segments(&["abc".to_string()], None).is_err());
}

// ─────────────────────── path-shape assertions ────────────────────────

/// Stage 10c [Codex review]: prove keys land at exactly
/// `prefix/MANIFEST.json` etc. — never double-prefixed
/// (`prefix/prefix/MANIFEST.json`) or absolute. Captures the
/// stateful mock's stored keys and asserts each one starts with
/// `idx-baked/` and contains no double slash.
#[tokio::test(flavor = "multi_thread")]
async fn synced_keys_target_prefix_exactly_once() {
  let (local_dir, _) = bake_local_index();
  let bucket_state = Arc::new(StatefulS3Bucket::new("test-bucket"));
  let server_uri = bucket_state.spawn_server().await;
  sync_to_s3(
    local_dir.path(),
    config_for(&server_uri, "test-bucket", Some("idx-baked")),
  )
  .await
  .unwrap();
  let stored = bucket_state.snapshot();
  assert!(!stored.is_empty(), "sync produced no uploads");
  for key in stored.keys() {
    assert!(
      key.starts_with("idx-baked/"),
      "key {key:?} must be prefixed with idx-baked/ exactly once"
    );
    assert!(
      !key.starts_with("idx-baked/idx-baked/"),
      "key {key:?} must NOT be double-prefixed"
    );
    assert!(!key.starts_with('/'), "key {key:?} must not be absolute");
    assert!(
      !key.contains("//"),
      "key {key:?} must not contain double slashes"
    );
  }
  // The standard segment file kinds must each appear exactly once.
  let mut found = std::collections::BTreeSet::new();
  for key in stored.keys() {
    for ext in [".terms", ".post", ".docs", ".fast", ".meta"] {
      if key.ends_with(ext) {
        found.insert(ext);
      }
    }
  }
  assert_eq!(
    found.len(),
    5,
    "expected one of each segment file kind under prefix; saw: {stored:?}"
  );
  assert!(
    stored.contains_key("idx-baked/MANIFEST.json"),
    "manifest must be at idx-baked/MANIFEST.json; saw: {stored:?}"
  );
}

// ─────────────────────── sync corruption tests ────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn sync_errors_when_pending_manifest_exists() {
  let (local_dir, _) = bake_local_index();
  // Plant a `.pending` file.
  std::fs::write(
    local_dir.path().join("MANIFEST.json.pending"),
    b"\"untouched\"",
  )
  .unwrap();
  let (server_uri, bucket) = spawn_stateful_s3_mock("test-bucket").await;
  let err = sync_to_s3(local_dir.path(), config_for(&server_uri, &bucket, None))
    .await
    .expect_err("pending manifest must block sync");
  let msg = format!("{err:#}");
  assert!(
    msg.contains("MANIFEST.json.pending") && msg.contains("Recovery"),
    "expected pending-recovery error; got: {msg}"
  );
}

#[tokio::test(flavor = "multi_thread")]
async fn sync_errors_when_wal_is_non_empty() {
  let (local_dir, _) = bake_local_index();
  // Append junk to the WAL.
  let wal = local_dir.path().join("wal.log");
  std::fs::write(&wal, b"\x00\x00uncommitted-stuff").unwrap();
  let (server_uri, bucket) = spawn_stateful_s3_mock("test-bucket").await;
  let err = sync_to_s3(local_dir.path(), config_for(&server_uri, &bucket, None))
    .await
    .expect_err("non-empty WAL must block sync");
  let msg = format!("{err:#}");
  assert!(
    msg.contains("wal.log") && msg.contains("non-empty"),
    "expected non-empty WAL error; got: {msg}"
  );
}

// ─────────────────────── Stage 10c v2 regressions ────────────────────

/// Stage 10c v2 [P1] (Codex review): MANIFEST.json must be the
/// LAST PUT issued by `sync_to_s3`. Before this fix, files were
/// uploaded in `read_dir` order, so the manifest could land before
/// the segment artifacts it referenced — making a partially-published
/// index visible to readers if sync failed afterward.
#[tokio::test(flavor = "multi_thread")]
async fn sync_publishes_manifest_last_as_visibility_fence() {
  let (local_dir, _) = bake_local_index();
  let bucket = Arc::new(StatefulS3Bucket::new("test-bucket"));
  let server_uri = bucket.spawn_server().await;
  sync_to_s3(
    local_dir.path(),
    config_for(&server_uri, "test-bucket", Some("idx-baked")),
  )
  .await
  .unwrap();
  let order = bucket.put_order();
  assert!(!order.is_empty(), "sync_to_s3 must issue at least one PUT");
  let last = order.last().unwrap();
  assert_eq!(
    last, "idx-baked/MANIFEST.json",
    "MANIFEST.json must be the FINAL PUT (visibility fence); \
     observed order: {order:?}"
  );
  // Also: every PUT before the manifest must be a non-manifest key.
  for (idx, key) in order.iter().enumerate() {
    if idx + 1 < order.len() {
      assert!(
        !key.ends_with("MANIFEST.json"),
        "MANIFEST.json appeared at PUT #{idx} but was not the last; \
         observed order: {order:?}"
      );
    }
  }
}

/// Stage 10c v2 [P1] (Codex review): if a sync fails before the
/// final manifest PUT, no remote MANIFEST.json gets written. A
/// reader that subsequently calls `open_index_read_only` sees a
/// clean NotFound rather than a partially-published index pointing
/// at missing segment files.
#[tokio::test(flavor = "multi_thread")]
async fn failed_sync_leaves_no_remote_manifest() {
  let (local_dir, _) = bake_local_index();
  let bucket = Arc::new(StatefulS3Bucket::new("test-bucket"));
  let server_uri = bucket.spawn_server().await;

  // Inject a 500 on one of the segment file PUTs. The exact key
  // isn't known up-front (uuid-based), so we walk the local dir
  // and pick a `.terms` file. The sync should attempt that PUT,
  // hit the 500, and abort BEFORE the manifest publish.
  let target_key = pick_first_segment_terms_key(local_dir.path(), Some("idx-baked"));
  bucket.inject_put_failure(&target_key);

  let err = sync_to_s3(
    local_dir.path(),
    config_for(&server_uri, "test-bucket", Some("idx-baked")),
  )
  .await
  .expect_err("injected segment PUT failure must abort the sync");
  let msg = format!("{err:#}");
  assert!(
    msg.contains("put") || msg.contains("PUT") || msg.contains("500"),
    "expected sync error to mention the failed PUT; got: {msg}"
  );

  // The remote bucket must NOT contain the manifest. Other files
  // may have landed (PUTs before the failure target), but the
  // manifest-as-fence guarantees the prefix is not visible.
  let remote = bucket.snapshot();
  assert!(
    !remote.contains_key("idx-baked/MANIFEST.json"),
    "manifest MUST NOT be uploaded after a sync failure; \
     remote keys: {:?}",
    remote.keys().collect::<Vec<_>>()
  );

  // open_index_read_only against the prefix surfaces NotFound (not
  // a partial-index error).
  let open_err =
    match open_index_read_only(config_for(&server_uri, "test-bucket", Some("idx-baked"))).await {
      Ok(_) => panic!("open after failed sync must surface NotFound"),
      Err(e) => e,
    };
  let open_msg = format!("{open_err:#}");
  assert!(
    open_msg.contains("not found")
      || open_msg.contains("NotFound")
      || open_msg.contains("does not exist"),
    "expected NotFound on open after failed sync; got: {open_msg}"
  );
}

/// Pick the first `.terms` segment file under `local_root` and
/// return its S3 key (with optional prefix joined). Used by the
/// fail-mid-sync regression to inject a failure on a specific
/// non-manifest PUT.
fn pick_first_segment_terms_key(local_root: &Path, prefix: Option<&str>) -> String {
  for entry in std::fs::read_dir(local_root).unwrap() {
    let entry = entry.unwrap();
    let name = entry.file_name();
    let name_str = name.to_str().unwrap_or("");
    if name_str.ends_with(".terms") {
      return match prefix {
        Some(p) => format!("{p}/{name_str}"),
        None => name_str.to_string(),
      };
    }
  }
  panic!("no .terms segment file under {local_root:?}");
}

/// Stage 10c v2 [P2] (Codex review): a v1 (legacy) MANIFEST.json
/// must be rejected before any upload. The S3 open path resolves
/// segment keys against an empty logical root, so absolute or
/// root-prefixed-relative paths from a v1 manifest would silently
/// miss after upload. Sync's preflight catches this, naming the
/// version mismatch and pointing at the local-upgrade workflow.
/// Stage 10c v5 [P2] (Codex review): a v2 manifest can name an
/// artifact at a non-canonical lexical form (e.g. `./seg_X.post`)
/// that resolves to an existing local file but doesn't match the
/// key the upload walker would emit (`seg_X.post`). S3 stores keys
/// verbatim, so the manifest would publish a reference to
/// `./seg_X.post` while the bytes live at `seg_X.post` — visible
/// but unservable. Preflight rejects non-canonical keys before any
/// upload.
#[tokio::test(flavor = "multi_thread")]
async fn sync_errors_when_manifest_references_non_canonical_key() {
  let (local_dir, _) = bake_local_index();

  // Read the manifest, find the postings key, and rewrite it with a
  // leading `./` (lexical drift). The file on disk doesn't move —
  // `local_root.join("./seg_X.post")` resolves to the same regular
  // file, so the existence check passes; only the canonical-form
  // assertion catches the drift.
  let manifest_path = local_dir.path().join("MANIFEST.json");
  let mut value: serde_json::Value =
    serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
  let original_postings_key = value["segments"][0]["paths"]["postings"]
    .as_str()
    .expect("postings path in v2 manifest")
    .to_string();
  let drifted_key = format!("./{original_postings_key}");
  value["segments"][0]["paths"]["postings"] = serde_json::json!(drifted_key);
  std::fs::write(&manifest_path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();

  let bucket = Arc::new(StatefulS3Bucket::new("test-bucket"));
  let server_uri = bucket.spawn_server().await;
  let err = sync_to_s3(
    local_dir.path(),
    config_for(&server_uri, "test-bucket", None),
  )
  .await
  .expect_err("non-canonical key must be rejected at preflight");
  let msg = format!("{err:#}");
  assert!(
    msg.contains("canonical form") && msg.contains(&drifted_key),
    "expected canonical-form rejection naming the drifted key; got: {msg}"
  );
  // No PUT issued — preflight failed before any network write,
  // and crucially: no `MANIFEST.json` was published.
  let remote = bucket.snapshot();
  assert!(
    remote.is_empty(),
    "non-canonical-key rejection must abort BEFORE any upload; remote: {:?}",
    remote.keys().collect::<Vec<_>>()
  );
}

/// Stage 10c v4 [P2] (Codex review): if a v2 manifest names a
/// segment artifact at a path that the sync walker would SKIP
/// (e.g. a dot-file, `wal.log`, or the top-level
/// `MANIFEST.json`), preflight must reject it before any upload.
/// Otherwise the existence check passes, the upload silently
/// skips the path, and the manifest publish surfaces a remote
/// index pointing at a key that was never PUT.
///
/// We exercise the dot-file case: rename a real `.terms` file to
/// `.hidden.terms` and rewrite the manifest to point at it. The
/// file exists, the path is relative + `..`-free (so v2 validation
/// passes), but the uploader's skip rules would drop it.
#[tokio::test(flavor = "multi_thread")]
async fn sync_errors_when_manifest_references_skipped_dotfile_artifact() {
  let (local_dir, _) = bake_local_index();

  // Read the manifest, find the terms key, rename the file to a
  // dot-prefixed name, and rewrite the manifest path to match.
  let manifest_path = local_dir.path().join("MANIFEST.json");
  let mut value: serde_json::Value =
    serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
  let original_terms_key = value["segments"][0]["paths"]["terms"]
    .as_str()
    .expect("terms path in v2 manifest")
    .to_string();
  let hidden_key = format!(".hidden_{original_terms_key}");
  std::fs::rename(
    local_dir.path().join(&original_terms_key),
    local_dir.path().join(&hidden_key),
  )
  .unwrap();
  value["segments"][0]["paths"]["terms"] = serde_json::json!(hidden_key);
  std::fs::write(&manifest_path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();

  let bucket = Arc::new(StatefulS3Bucket::new("test-bucket"));
  let server_uri = bucket.spawn_server().await;
  let err = sync_to_s3(
    local_dir.path(),
    config_for(&server_uri, "test-bucket", None),
  )
  .await
  .expect_err("manifest pointing at a skipped dotfile must be rejected at preflight");
  let msg = format!("{err:#}");
  assert!(
    msg.contains("skip rules") && msg.contains(&hidden_key),
    "expected skip-rule rejection naming the dotfile artifact; got: {msg}"
  );
  // No PUT issued — preflight failed before any network write.
  assert!(
    bucket.snapshot().is_empty(),
    "skip-rule rejection must abort BEFORE any upload; remote: {:?}",
    bucket.snapshot().keys().collect::<Vec<_>>()
  );
}

/// Stage 10c v3 [P2] (Codex review): if the manifest references a
/// segment artifact that doesn't exist on disk (partial bake,
/// manual deletion, etc.), preflight must fail BEFORE any HTTP
/// request. Without this guard, `sync_to_s3` would upload whatever
/// files ARE present and then publish the manifest, making an
/// unservable index visible at the prefix.
#[tokio::test(flavor = "multi_thread")]
async fn sync_errors_when_manifest_references_missing_segment_file() {
  let (local_dir, _) = bake_local_index();

  // Identify the postings file from the manifest, then delete it.
  let manifest_bytes = std::fs::read(local_dir.path().join("MANIFEST.json")).unwrap();
  let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes).unwrap();
  let postings_key = manifest["segments"][0]["paths"]["postings"]
    .as_str()
    .expect("postings path in v2 manifest")
    .to_string();
  let postings_path = local_dir.path().join(&postings_key);
  std::fs::remove_file(&postings_path).expect("delete postings file");

  let bucket = Arc::new(StatefulS3Bucket::new("test-bucket"));
  let server_uri = bucket.spawn_server().await;
  let err = sync_to_s3(
    local_dir.path(),
    config_for(&server_uri, "test-bucket", None),
  )
  .await
  .expect_err("missing segment artifact must abort sync at preflight");
  let msg = format!("{err:#}");
  assert!(
    msg.contains("postings") && msg.contains(&postings_key),
    "expected error mentioning the missing postings artifact; got: {msg}"
  );
  // No PUT issued — preflight failed before any network write.
  assert!(
    bucket.snapshot().is_empty(),
    "missing-artifact rejection must abort BEFORE any upload; remote: {:?}",
    bucket.snapshot().keys().collect::<Vec<_>>()
  );
}

#[tokio::test(flavor = "multi_thread")]
async fn sync_errors_when_local_manifest_is_legacy_v1() {
  let (local_dir, _) = bake_local_index();
  // Surgically downgrade the manifest to v1 (without rewriting the
  // paths to absolute — the version check fires first).
  let manifest_path = local_dir.path().join("MANIFEST.json");
  let mut value: serde_json::Value =
    serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
  value["version"] = serde_json::json!(1);
  // v1 manifests had absolute segment paths; reflect that so the
  // file is internally self-consistent (validate_v1_legacy passes
  // for absolute paths).
  let segments = value["segments"].as_array_mut().unwrap();
  for seg in segments.iter_mut() {
    let paths = seg["paths"].as_object_mut().unwrap();
    for key in ["terms", "postings", "docstore", "fast", "meta"] {
      let rel = paths[key].as_str().unwrap().to_string();
      let abs = local_dir.path().join(&rel).to_string_lossy().into_owned();
      paths[key] = serde_json::json!(abs);
    }
  }
  std::fs::write(&manifest_path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();

  let bucket = Arc::new(StatefulS3Bucket::new("test-bucket"));
  let server_uri = bucket.spawn_server().await;
  let err = sync_to_s3(
    local_dir.path(),
    config_for(&server_uri, "test-bucket", None),
  )
  .await
  .expect_err("v1 manifest must be rejected at preflight");
  let msg = format!("{err:#}");
  assert!(
    msg.contains("version 1") || msg.contains("local manifest"),
    "expected legacy-version error; got: {msg}"
  );
  // No PUT should have been issued — preflight failed before the
  // sync reached the network.
  assert!(
    bucket.snapshot().is_empty(),
    "v1 rejection must abort BEFORE any upload; remote: {:?}",
    bucket.snapshot().keys().collect::<Vec<_>>()
  );
}

#[tokio::test(flavor = "multi_thread")]
async fn sync_errors_when_staging_file_present() {
  let (local_dir, _) = bake_local_index();
  // Plant an atomic_write staging artifact.
  std::fs::write(
    local_dir.path().join("MANIFEST.json.tmp-deadbeef-1234"),
    b"staging",
  )
  .unwrap();
  let (server_uri, bucket) = spawn_stateful_s3_mock("test-bucket").await;
  let err = sync_to_s3(local_dir.path(), config_for(&server_uri, &bucket, None))
    .await
    .expect_err("staging file must block sync");
  let msg = format!("{err:#}");
  assert!(
    msg.contains("staging file"),
    "expected staging-file error; got: {msg}"
  );
}
