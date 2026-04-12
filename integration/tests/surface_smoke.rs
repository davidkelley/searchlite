mod common;

use anyhow::Result;
use serde_json::json;
use tempfile::tempdir;

use integration::surfaces::cli::CliHarness;
use integration::surfaces::core::CoreHarness;
use integration::surfaces::http::HttpHarness;
use integration::surfaces::SurfaceHarness;

fn schema_json() -> serde_json::Value {
  json!({
    "type": "object",
    "properties": {
      "body": { "type": "string" }
    }
  })
}

fn request_json() -> serde_json::Value {
  json!({
    "query": "rust",
    "limit": 5,
    "return_stored": true
  })
}

fn ndjson_docs() -> &'static str {
  "{\"_id\":\"1\",\"body\":\"rust search\"}\n{\"_id\":\"2\",\"body\":\"another doc\"}\n"
}

fn assert_smoke_hits(body: &serde_json::Value, surface_name: &str) {
  let hits = body["hits"].as_array().expect("hits array");
  assert!(!hits.is_empty(), "{surface_name}: expected non-empty hits");
  let doc_ids: Vec<&str> = hits.iter().filter_map(|h| h["doc_id"].as_str()).collect();
  assert!(
    doc_ids.contains(&"1"),
    "{surface_name}: search for 'rust' should return doc_id '1' (contains 'rust search'), got: {doc_ids:?}"
  );
}

#[test]
fn surface_smoke_core_happy_path() -> Result<()> {
  let dir = tempdir()?;
  let mut harness = CoreHarness::new(dir.path().join("idx-core"));
  harness.init(&schema_json())?;
  harness.add_ndjson(ndjson_docs())?;
  harness.commit()?;
  let body = harness.search(&request_json())?;
  assert_smoke_hits(&body, "Core");
  Ok(())
}

#[test]
fn surface_smoke_cli_happy_path() -> Result<()> {
  let dir = tempdir()?;
  let bin = common::searchlite_bin();
  let mut harness = CliHarness::new(bin, dir.path().join("idx-cli"));
  harness.init(&schema_json())?;
  harness.add_ndjson(ndjson_docs())?;
  harness.commit()?;
  let body = harness.search(&request_json())?;
  assert_smoke_hits(&body, "Cli");
  Ok(())
}

#[test]
fn surface_smoke_http_happy_path() -> Result<()> {
  let dir = tempdir()?;
  let bin = common::searchlite_bin();
  let mut harness = HttpHarness::new(bin, dir.path().join("idx-http"))?;
  harness.init(&schema_json())?;
  harness.add_ndjson(ndjson_docs())?;
  harness.commit()?;
  let body = harness.search(&request_json())?;
  assert_smoke_hits(&body, "Http");
  Ok(())
}
