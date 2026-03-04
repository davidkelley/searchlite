mod common;

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

  let mut search_req = lifecycle_request(execution);
  let first = harness.search(&search_req)?;
  let first_hits = first["hits"].as_array().cloned().unwrap_or_default();
  assert!(
    !first_hits.is_empty(),
    "expected non-empty hits for {dataset_name:?}/{surface:?}/{execution}"
  );

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
      if let (Some(a), Some(b)) = (first_hits.first(), second_hits.first()) {
        let first_id = a["doc_id"].as_str().unwrap_or_default();
        let second_id = b["doc_id"].as_str().unwrap_or_default();
        assert_ne!(first_id, second_id, "search_after should advance results");
      }
    }
  }

  let mget_res = harness.mget(&dataset.mutations.mget_ids, true);
  if capabilities.supports_mget {
    let body = mget_res?;
    let docs = body["docs"].as_array().expect("mget docs array");
    assert_eq!(docs.len(), dataset.mutations.mget_ids.len());
  } else {
    assert_not_supported("mget", mget_res);
  }

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
  } else {
    assert_not_supported("update", update_res);
  }

  let delete_res = harness.delete_ids(&dataset.mutations.delete_ids);
  if capabilities.supports_delete {
    delete_res?;
    harness.commit()?;
    assert_supported_unit(
      "refresh-after-delete",
      capabilities.supports_refresh,
      harness.refresh(),
    );
  } else {
    assert_not_supported("delete", delete_res);
  }

  let stats_res = harness.stats();
  if capabilities.supports_stats {
    let stats = stats_res?;
    assert!(
      stats.get("documents").is_some(),
      "stats should expose documents field"
    );
  } else {
    assert_not_supported("stats", stats_res);
  }

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

  let compact_res = harness.compact();
  if capabilities.supports_compact {
    compact_res?;
    let post_compact = harness.search(&lifecycle_request(execution))?;
    assert!(
      post_compact["hits"].is_array(),
      "search should still work after compact"
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
