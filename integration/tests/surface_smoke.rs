use std::path::PathBuf;
use std::process::Command;

use anyhow::Result;
use serde_json::json;
use tempfile::tempdir;

use integration::surfaces::cli::CliHarness;
use integration::surfaces::core::CoreHarness;
use integration::surfaces::http::HttpHarness;
use integration::surfaces::SurfaceHarness;

fn schema_json() -> serde_json::Value {
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

fn searchlite_bin() -> PathBuf {
  if let Ok(path) = std::env::var("CARGO_BIN_EXE_searchlite") {
    return PathBuf::from(path);
  }
  if let Ok(path) = std::env::var("CARGO_BIN_EXE_searchlite-cli") {
    return PathBuf::from(path);
  }

  let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .parent()
    .expect("workspace root")
    .to_path_buf();
  let candidates = [
    workspace_root
      .join("target")
      .join("debug")
      .join(if cfg!(windows) {
        "searchlite.exe"
      } else {
        "searchlite"
      }),
    workspace_root
      .join("target")
      .join("debug")
      .join(if cfg!(windows) {
        "searchlite-cli.exe"
      } else {
        "searchlite-cli"
      }),
  ];
  for candidate in candidates {
    if candidate.exists() {
      return candidate;
    }
  }

  let status = Command::new("cargo")
    .arg("build")
    .arg("-p")
    .arg("searchlite-cli")
    .current_dir(&workspace_root)
    .status()
    .expect("build searchlite binary");
  assert!(status.success(), "building searchlite-cli failed");
  workspace_root
    .join("target")
    .join("debug")
    .join(if cfg!(windows) {
      "searchlite-cli.exe"
    } else {
      "searchlite-cli"
    })
}

#[test]
fn surface_smoke_core_happy_path() -> Result<()> {
  let dir = tempdir()?;
  let mut harness = CoreHarness::new(dir.path().join("idx-core"));
  harness.init(&schema_json())?;
  harness.add_ndjson(ndjson_docs())?;
  harness.commit()?;
  let body = harness.search(&request_json())?;
  let hits = body["hits"].as_array().expect("hits array");
  assert!(!hits.is_empty());
  Ok(())
}

#[test]
fn surface_smoke_cli_happy_path() -> Result<()> {
  let dir = tempdir()?;
  let mut harness = CliHarness::new(searchlite_bin(), dir.path().join("idx-cli"));
  harness.init(&schema_json())?;
  harness.add_ndjson(ndjson_docs())?;
  harness.commit()?;
  let body = harness.search(&request_json())?;
  let hits = body["hits"].as_array().expect("hits array");
  assert!(!hits.is_empty());
  Ok(())
}

#[test]
fn surface_smoke_http_happy_path() -> Result<()> {
  let dir = tempdir()?;
  let mut harness = HttpHarness::new(searchlite_bin(), dir.path().join("idx-http"))?;
  harness.init(&schema_json())?;
  harness.add_ndjson(ndjson_docs())?;
  harness.commit()?;
  let body = harness.search(&request_json())?;
  let hits = body["hits"].as_array().expect("hits array");
  assert!(!hits.is_empty());
  Ok(())
}
