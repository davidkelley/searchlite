mod common;

use anyhow::Result;
use reqwest::StatusCode;
use serde_json::json;
use tempfile::tempdir;

use integration::surfaces::http::HttpHarness;
use integration::surfaces::SurfaceHarness;

fn valid_schema() -> serde_json::Value {
  json!({
    "doc_id_field": "_id",
    "text_fields": [
      { "name": "body", "analyzer": "default", "stored": true, "indexed": true, "nullable": false }
    ],
    "keyword_fields": [],
    "numeric_fields": [],
    "nested_fields": [],
    "vector_fields": []
  })
}

fn assert_error_shape(body: &serde_json::Value) {
  assert!(
    body
      .get("error")
      .and_then(|e| e.get("type"))
      .and_then(|v| v.as_str())
      .is_some(),
    "missing error.type in body: {body}"
  );
  assert!(
    body
      .get("error")
      .and_then(|e| e.get("reason"))
      .and_then(|v| v.as_str())
      .is_some(),
    "missing error.reason in body: {body}"
  );
}

#[test]
fn http_contract_invalid_search_returns_structured_error() -> Result<()> {
  let searchlite_bin = common::searchlite_bin();
  let dir = tempdir()?;
  let mut harness = HttpHarness::new(searchlite_bin, dir.path().join("idx-http-contract-invalid"))?;
  harness.init(&valid_schema())?;

  let client = reqwest::blocking::Client::new();
  let res = client
    .post(format!("{}/search", harness.index_base_url()))
    .json(&json!({
      "query": "rust",
      "cursor": "abc",
      "search_after": ["token"]
    }))
    .send()?;

  assert_eq!(res.status(), StatusCode::BAD_REQUEST);
  let body: serde_json::Value = res.json()?;
  assert_error_shape(&body);
  Ok(())
}

#[test]
fn http_contract_unknown_index_returns_404_error_shape() -> Result<()> {
  let searchlite_bin = common::searchlite_bin();
  let dir = tempdir()?;
  let mut harness = HttpHarness::new(searchlite_bin, dir.path().join("idx-http-contract-unknown"))?;
  harness.init(&valid_schema())?;

  let client = reqwest::blocking::Client::new();
  let res = client
    .post(format!("{}/indexes/missing/search", harness.base_url()))
    .json(&json!({"query": "rust"}))
    .send()?;

  assert_eq!(res.status(), StatusCode::NOT_FOUND);
  let body: serde_json::Value = res.json()?;
  assert_error_shape(&body);
  Ok(())
}

#[test]
fn http_contract_init_conflict_returns_409_error_shape() -> Result<()> {
  let searchlite_bin = common::searchlite_bin();
  let dir = tempdir()?;
  let mut harness = HttpHarness::new(
    searchlite_bin,
    dir.path().join("idx-http-contract-conflict"),
  )?;
  harness.init(&valid_schema())?;

  let client = reqwest::blocking::Client::new();
  let res = client
    .post(format!("{}/init", harness.index_base_url()))
    .json(&valid_schema())
    .send()?;

  assert_eq!(res.status(), StatusCode::CONFLICT);
  let body: serde_json::Value = res.json()?;
  assert_error_shape(&body);
  Ok(())
}
