mod common;

use std::collections::HashSet;

use anyhow::Result;
use serde_json::{json, Value};
use tempfile::tempdir;

use integration::fixtures::{load_example_fixtures, DatasetFixture, DatasetName};
use integration::matrix::MatrixMode;
use integration::surfaces::cli::CliHarness;
use integration::surfaces::core::CoreHarness;
use integration::surfaces::http::HttpHarness;
use integration::surfaces::{is_not_supported_error, SurfaceHarness, SurfaceKind};

#[test]
fn lifecycle_matrix_covers_all_surfaces() -> Result<()> {
  let fixtures = load_example_fixtures()?;
  let mode = MatrixMode::from_env();
  let searchlite_bin = common::searchlite_bin();

  for (dataset_name, dataset) in fixtures.datasets.iter() {
    for surface in [SurfaceKind::Core, SurfaceKind::Http, SurfaceKind::Cli] {
      for execution in ["bm25", "wand", "bmw"] {
        run_lifecycle_case(
          *dataset_name,
          dataset,
          surface,
          execution,
          mode,
          searchlite_bin.clone(),
        )?;
      }
    }
  }

  Ok(())
}

fn run_lifecycle_case(
  dataset_name: DatasetName,
  dataset: &DatasetFixture,
  surface: SurfaceKind,
  execution: &str,
  mode: MatrixMode,
  searchlite_bin: std::path::PathBuf,
) -> Result<()> {
  let dir = tempdir()?;
  let mut harness = build_harness(
    surface,
    dir.path().join(format!("idx-{surface:?}-{execution}")),
    searchlite_bin,
  )?;
  let capabilities = harness.capabilities();

  let schema_json = serde_json::to_value(&dataset.schema)?;
  let seed_docs = match mode {
    MatrixMode::Full => &dataset.seed_docs[..],
    MatrixMode::Quick => {
      let capped = dataset.seed_docs.len().min(250);
      &dataset.seed_docs[..capped]
    }
  };
  let seed_ndjson = common::docs_to_ndjson(seed_docs);

  harness.init(&schema_json)?;
  harness.add_ndjson(seed_ndjson.as_str())?;
  harness.commit()?;
  assert_supported_unit(
    "refresh-after-commit",
    capabilities.supports_refresh,
    harness.refresh(),
  );

  let insert_ndjson = common::docs_to_ndjson(&dataset.mutations.insert_docs);
  harness.add_ndjson(insert_ndjson.as_str())?;
  harness.commit()?;
  assert_supported_unit(
    "refresh-after-insert",
    capabilities.supports_refresh,
    harness.refresh(),
  );

  // --- Search and validate hits ---
  let mut search_req = lifecycle_request(execution);
  let first = harness.search(&search_req)?;
  let first_hits = first["hits"].as_array().cloned().unwrap_or_default();
  assert!(
    !first_hits.is_empty(),
    "expected non-empty hits for {dataset_name:?}/{surface:?}/{execution}"
  );
  // Verify doc_ids are valid non-empty strings
  for (i, hit) in first_hits.iter().enumerate() {
    let doc_id = hit["doc_id"].as_str();
    assert!(
      doc_id.is_some_and(|s| !s.is_empty()),
      "hit[{i}] missing valid doc_id for {dataset_name:?}/{surface:?}/{execution}"
    );
  }

  // --- search_after pagination: verify pages are disjoint ---
  if capabilities.supports_search_after {
    if let Some(token) = first
      .get("next_search_after")
      .cloned()
      .filter(|v| !v.is_null())
    {
      if let Some(obj) = search_req.as_object_mut() {
        obj.insert("search_after".to_string(), token);
      }
      let second = harness.search(&search_req)?;
      let second_hits = second["hits"].as_array().cloned().unwrap_or_default();
      let first_ids: HashSet<&str> = first_hits
        .iter()
        .filter_map(|h| h["doc_id"].as_str())
        .collect();
      let second_ids: HashSet<&str> = second_hits
        .iter()
        .filter_map(|h| h["doc_id"].as_str())
        .collect();
      assert!(
        first_ids.is_disjoint(&second_ids),
        "search_after pages should be disjoint for {dataset_name:?}/{surface:?}/{execution}: first={first_ids:?}, second={second_ids:?}"
      );
    }
  }

  // --- mget ---
  let mget_res = harness.mget(&dataset.mutations.mget_ids, true);
  if capabilities.supports_mget {
    let body = mget_res?;
    let docs = body["docs"].as_array().expect("mget docs array");
    assert_eq!(docs.len(), dataset.mutations.mget_ids.len());
  } else {
    assert_not_supported("mget", mget_res);
  }

  // --- update + verify via mget ---
  let update = dataset
    .mutations
    .update_docs
    .first()
    .expect("update fixture present");
  let update_res = harness.update_doc(update.id.as_str(), &update.set, &update.unset);
  if capabilities.supports_update {
    update_res?;
    harness.commit()?;
    assert_supported_unit(
      "refresh-after-update",
      capabilities.supports_refresh,
      harness.refresh(),
    );

    // Verify updated field contains marker via mget
    if capabilities.supports_mget {
      let mget_after = harness.mget(std::slice::from_ref(&update.id), true)?;
      let docs = mget_after["docs"].as_array().expect("mget docs");
      assert!(
        !docs.is_empty(),
        "mget after update should return document for {dataset_name:?}/{surface:?}/{execution}"
      );
      if let Some(fields) = docs[0].get("fields").and_then(Value::as_object) {
        if let Some(field_val) = fields.get(&update.updated_field) {
          let text = field_val.as_str().unwrap_or_default();
          assert!(
            text.contains("integration update marker"),
            "updated field '{}' should contain marker for {dataset_name:?}/{surface:?}/{execution}, got: {text}",
            update.updated_field
          );
        }
      }
    }
  } else {
    assert_not_supported("update", update_res);
  }

  // --- delete + verify absence ---
  let delete_res = harness.delete_ids(&dataset.mutations.delete_ids);
  if capabilities.supports_delete {
    delete_res?;
    harness.commit()?;
    assert_supported_unit(
      "refresh-after-delete",
      capabilities.supports_refresh,
      harness.refresh(),
    );

    // Search with high limit to verify deleted IDs are absent
    let post_delete = harness.search(&json!({
      "query": { "type": "match_all" },
      "limit": 100,
      "return_hits": true,
      "track_total_hits": true,
      "execution": execution,
    }))?;
    let empty = vec![];
    let post_delete_hits = post_delete["hits"].as_array().unwrap_or(&empty);
    let post_delete_ids: Vec<&str> = post_delete_hits
      .iter()
      .filter_map(|h| h["doc_id"].as_str())
      .collect();
    for deleted_id in &dataset.mutations.delete_ids {
      assert!(
        !post_delete_ids.contains(&deleted_id.as_str()),
        "deleted doc '{deleted_id}' should be absent for {dataset_name:?}/{surface:?}/{execution}"
      );
    }
  } else {
    assert_not_supported("delete", delete_res);
  }

  // --- stats: verify document count > 0 ---
  let stats_res = harness.stats();
  if capabilities.supports_stats {
    let stats = stats_res?;
    assert!(
      stats.get("documents").is_some(),
      "stats should expose documents field"
    );
    let doc_count = stats["documents"].as_u64().unwrap_or(0);
    assert!(
      doc_count > 0,
      "stats documents should be > 0 for {dataset_name:?}/{surface:?}/{execution}"
    );
  } else {
    assert_not_supported("stats", stats_res);
  }

  // --- inspect ---
  let inspect_res = harness.inspect();
  if capabilities.supports_inspect {
    let inspect = inspect_res?;
    assert!(
      inspect.get("manifest").is_some(),
      "inspect should expose manifest"
    );
  } else {
    assert_not_supported("inspect", inspect_res);
  }

  // --- compact + verify search still works ---
  let compact_res = harness.compact();
  if capabilities.supports_compact {
    compact_res?;
    let post_compact = harness.search(&lifecycle_request(execution))?;
    let post_compact_hits = post_compact["hits"].as_array().cloned().unwrap_or_default();
    assert!(
      !post_compact_hits.is_empty(),
      "search should return hits after compact for {dataset_name:?}/{surface:?}/{execution}"
    );
  } else {
    assert_not_supported("compact", compact_res);
  }

  Ok(())
}

fn build_harness(
  surface: SurfaceKind,
  index_path: std::path::PathBuf,
  searchlite_bin: std::path::PathBuf,
) -> Result<Box<dyn SurfaceHarness>> {
  match surface {
    SurfaceKind::Core => Ok(Box::new(CoreHarness::new(index_path))),
    SurfaceKind::Http => Ok(Box::new(HttpHarness::new(searchlite_bin, index_path)?)),
    SurfaceKind::Cli => Ok(Box::new(CliHarness::new(searchlite_bin, index_path))),
  }
}

fn lifecycle_request(execution: &str) -> Value {
  json!({
    "query": { "type": "match_all" },
    "limit": 2,
    "return_stored": true,
    "return_hits": true,
    "track_total_hits": true,
    "execution": execution,
    "sort": [
      { "field": "created_at", "order": "asc" }
    ]
  })
}

fn assert_supported_unit(label: &str, expected_supported: bool, result: Result<()>) {
  if expected_supported {
    if let Err(err) = result {
      panic!("{label} should be supported but failed: {err}");
    }
  } else {
    assert_not_supported(label, result);
  }
}

fn assert_not_supported<T>(label: &str, result: Result<T>) {
  match result {
    Ok(_) => panic!("{label} should be not-supported"),
    Err(err) => assert!(
      is_not_supported_error(&err),
      "{label} expected not-supported marker, got: {err}"
    ),
  }
}
