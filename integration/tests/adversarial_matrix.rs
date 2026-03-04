mod common;

use anyhow::Result;
use serde_json::json;
use tempfile::tempdir;

use integration::surfaces::cli::CliHarness;
use integration::surfaces::core::CoreHarness;
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

fn invalid_schema() -> serde_json::Value {
  json!({ "doc_id_field": 7 })
}

fn docs_ndjson() -> &'static str {
  "{\"_id\":\"1\",\"body\":\"rust\"}\n{\"_id\":\"2\",\"body\":\"query\"}\n"
}

#[test]
fn adversarial_invalid_schema_rejected_across_surfaces() -> Result<()> {
  let searchlite_bin = common::searchlite_bin();

  {
    let dir = tempdir()?;
    let mut core = CoreHarness::new(dir.path().join("idx-core-invalid-schema"));
    assert!(core.init(&invalid_schema()).is_err());
  }

  {
    let dir = tempdir()?;
    let mut cli = CliHarness::new(
      searchlite_bin.clone(),
      dir.path().join("idx-cli-invalid-schema"),
    );
    assert!(cli.init(&invalid_schema()).is_err());
  }

  {
    let dir = tempdir()?;
    let mut http = HttpHarness::new(searchlite_bin, dir.path().join("idx-http-invalid-schema"))?;
    assert!(http.init(&invalid_schema()).is_err());
  }

  Ok(())
}

#[test]
fn adversarial_malformed_ndjson_rejected_across_surfaces() -> Result<()> {
  let searchlite_bin = common::searchlite_bin();

  {
    let dir = tempdir()?;
    let mut core = CoreHarness::new(dir.path().join("idx-core-bad-ndjson"));
    core.init(&valid_schema())?;
    assert!(core.add_ndjson("not-json\n").is_err());
  }

  {
    let dir = tempdir()?;
    let mut cli = CliHarness::new(
      searchlite_bin.clone(),
      dir.path().join("idx-cli-bad-ndjson"),
    );
    cli.init(&valid_schema())?;
    assert!(cli.add_ndjson("not-json\n").is_err());
  }

  {
    let dir = tempdir()?;
    let mut http = HttpHarness::new(searchlite_bin, dir.path().join("idx-http-bad-ndjson"))?;
    http.init(&valid_schema())?;
    assert!(http.add_ndjson("not-json\n").is_err());
  }

  Ok(())
}

#[test]
fn adversarial_pagination_conflicts_are_rejected() -> Result<()> {
  let searchlite_bin = common::searchlite_bin();
  let bad_request = json!({
    "query": "rust",
    "limit": 1,
    "return_stored": true,
    "cursor": "abc",
    "search_after": ["token"],
    "from": 1
  });

  {
    let dir = tempdir()?;
    let mut core = CoreHarness::new(dir.path().join("idx-core-bad-pagination"));
    core.init(&valid_schema())?;
    core.add_ndjson(docs_ndjson())?;
    core.commit()?;
    assert!(core.search(&bad_request).is_err());
  }

  {
    let dir = tempdir()?;
    let mut cli = CliHarness::new(
      searchlite_bin.clone(),
      dir.path().join("idx-cli-bad-pagination"),
    );
    cli.init(&valid_schema())?;
    cli.add_ndjson(docs_ndjson())?;
    cli.commit()?;
    assert!(cli.search(&bad_request).is_err());
  }

  {
    let dir = tempdir()?;
    let mut http = HttpHarness::new(searchlite_bin, dir.path().join("idx-http-bad-pagination"))?;
    http.init(&valid_schema())?;
    http.add_ndjson(docs_ndjson())?;
    http.commit()?;
    assert!(http.search(&bad_request).is_err());
  }

  Ok(())
}
