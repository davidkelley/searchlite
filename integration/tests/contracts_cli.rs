mod common;

use std::fs;
use std::process::Command;

use anyhow::Result;
use serde_json::json;
use tempfile::tempdir;

#[test]
fn cli_contract_invalid_subcommand_exits_nonzero() -> Result<()> {
  let bin = common::searchlite_bin();
  let output = Command::new(bin)
    .arg("definitely-invalid-subcommand")
    .output()?;
  assert!(!output.status.success(), "invalid subcommand should fail");
  Ok(())
}

#[test]
fn cli_contract_search_emits_json() -> Result<()> {
  let bin = common::searchlite_bin();
  let dir = tempdir()?;

  let index_path = dir.path().join("idx-cli-contract");
  let schema_path = dir.path().join("schema.json");
  fs::write(
    &schema_path,
    serde_json::to_vec_pretty(&json!({
      "doc_id_field": "_id",
      "text_fields": [
        { "name": "body", "analyzer": "default", "stored": true, "indexed": true, "nullable": false }
      ],
      "keyword_fields": [],
      "numeric_fields": [],
      "nested_fields": [],
      "vector_fields": []
    }))?,
  )?;
  let docs_path = dir.path().join("docs.jsonl");
  fs::write(
    &docs_path,
    "{\"_id\":\"1\",\"body\":\"rust search\"}\n{\"_id\":\"2\",\"body\":\"another\"}\n",
  )?;
  let request_path = dir.path().join("request.json");
  fs::write(
    &request_path,
    serde_json::to_vec_pretty(&json!({
      "query": "rust",
      "limit": 5,
      "return_stored": true
    }))?,
  )?;

  let init = Command::new(&bin)
    .args([
      "init",
      index_path.to_str().unwrap(),
      schema_path.to_str().unwrap(),
    ])
    .output()?;
  assert!(init.status.success());

  let add = Command::new(&bin)
    .args([
      "add",
      index_path.to_str().unwrap(),
      docs_path.to_str().unwrap(),
    ])
    .output()?;
  assert!(add.status.success());

  let commit = Command::new(&bin)
    .args(["commit", index_path.to_str().unwrap()])
    .output()?;
  assert!(commit.status.success());

  let search = Command::new(&bin)
    .args([
      "search",
      index_path.to_str().unwrap(),
      "--request",
      request_path.to_str().unwrap(),
    ])
    .output()?;
  assert!(search.status.success());
  let body: serde_json::Value = serde_json::from_slice(&search.stdout)?;
  assert!(body["hits"].is_array());
  Ok(())
}

#[test]
fn cli_contract_inspect_emits_manifest_json() -> Result<()> {
  let bin = common::searchlite_bin();
  let dir = tempdir()?;

  let index_path = dir.path().join("idx-cli-inspect");
  let schema_path = dir.path().join("schema.json");
  fs::write(
    &schema_path,
    serde_json::to_vec_pretty(&json!({
      "doc_id_field": "_id",
      "text_fields": [
        { "name": "body", "analyzer": "default", "stored": true, "indexed": true, "nullable": false }
      ],
      "keyword_fields": [],
      "numeric_fields": [],
      "nested_fields": [],
      "vector_fields": []
    }))?,
  )?;

  let init = Command::new(&bin)
    .args([
      "init",
      index_path.to_str().unwrap(),
      schema_path.to_str().unwrap(),
    ])
    .output()?;
  assert!(init.status.success());

  let inspect = Command::new(&bin)
    .args(["inspect", index_path.to_str().unwrap()])
    .output()?;
  assert!(inspect.status.success());
  let stdout = String::from_utf8(inspect.stdout)?;
  let json_body = stdout
    .trim()
    .strip_prefix("manifest: ")
    .unwrap_or(stdout.trim());
  let manifest: serde_json::Value = serde_json::from_str(json_body)?;
  assert!(manifest["segments"].is_array());
  Ok(())
}
