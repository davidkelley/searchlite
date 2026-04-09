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

/// Helper: run a closure against all three surfaces, collecting results.
fn for_each_surface<F>(f: F) -> Result<()>
where
  F: Fn(&mut dyn SurfaceHarness, &str) -> Result<()>,
{
  let searchlite_bin = common::searchlite_bin();

  {
    let dir = tempdir()?;
    let mut core = CoreHarness::new(dir.path().join("idx-core"));
    f(&mut core, "Core")?;
  }

  {
    let dir = tempdir()?;
    let mut cli = CliHarness::new(searchlite_bin.clone(), dir.path().join("idx-cli"));
    f(&mut cli, "Cli")?;
  }

  {
    let dir = tempdir()?;
    let mut http = HttpHarness::new(searchlite_bin, dir.path().join("idx-http"))?;
    f(&mut http, "Http")?;
  }

  Ok(())
}

// ---------------------------------------------------------------------------
// Original adversarial tests — now with error message validation
// ---------------------------------------------------------------------------

#[test]
fn adversarial_invalid_schema_rejected_across_surfaces() -> Result<()> {
  for_each_surface(|harness, surface| {
    let err = harness
      .init(&invalid_schema())
      .expect_err(&format!("{surface}: invalid schema should be rejected"));
    let msg = err.to_string().to_lowercase();
    assert!(
      msg.contains("schema") || msg.contains("pars") || msg.contains("invalid") || msg.contains("missing"),
      "{surface}: invalid schema error should mention schema/parsing, got: {msg}"
    );
    Ok(())
  })
}

#[test]
fn adversarial_malformed_ndjson_rejected_across_surfaces() -> Result<()> {
  for_each_surface(|harness, surface| {
    harness.init(&valid_schema())?;
    let err = harness
      .add_ndjson("not-json\n")
      .expect_err(&format!("{surface}: malformed NDJSON should be rejected"));
    let msg = err.to_string().to_lowercase();
    assert!(
      msg.contains("json") || msg.contains("pars") || msg.contains("invalid") || msg.contains("expected"),
      "{surface}: malformed NDJSON error should mention JSON/parsing, got: {msg}"
    );
    Ok(())
  })
}

#[test]
fn adversarial_pagination_conflicts_are_rejected() -> Result<()> {
  let bad_request = json!({
    "query": "rust",
    "limit": 1,
    "return_stored": true,
    "cursor": "abc",
    "search_after": ["token"],
    "from": 1
  });

  for_each_surface(|harness, surface| {
    harness.init(&valid_schema())?;
    harness.add_ndjson(docs_ndjson())?;
    harness.commit()?;
    let err = harness
      .search(&bad_request)
      .expect_err(&format!("{surface}: conflicting pagination should be rejected"));
    // Use the full error chain ({err:#}) since anyhow wraps the root cause
    let msg = format!("{err:#}").to_lowercase();
    assert!(
      msg.contains("cursor") || msg.contains("search_after") || msg.contains("pagination") || msg.contains("conflict") || msg.contains("mutually") || msg.contains("exclusive"),
      "{surface}: pagination conflict error should mention cursor/search_after, got: {msg}"
    );
    Ok(())
  })
}

// ---------------------------------------------------------------------------
// New adversarial tests
// ---------------------------------------------------------------------------

#[test]
fn adversarial_search_before_commit() -> Result<()> {
  for_each_surface(|harness, surface| {
    harness.init(&valid_schema())?;
    harness.add_ndjson(docs_ndjson())?;
    // Do NOT commit — search should return zero hits or error
    let result = harness.search(&json!({
      "query": { "type": "match_all" },
      "limit": 10,
      "return_hits": true,
    }));
    match result {
      Ok(body) => {
        let hits = body["hits"].as_array().map(|a| a.len()).unwrap_or(0);
        assert_eq!(
          hits, 0,
          "{surface}: search before commit should return 0 hits, got {hits}"
        );
      }
      Err(_) => {
        // Some surfaces may error on uncommitted index — acceptable
      }
    }
    Ok(())
  })
}

#[test]
fn adversarial_delete_nonexistent_ids() -> Result<()> {
  for_each_surface(|harness, surface| {
    harness.init(&valid_schema())?;
    harness.add_ndjson(docs_ndjson())?;
    harness.commit()?;
    // Delete IDs that never existed — should succeed silently (idempotent)
    let result = harness.delete_ids(&[
      "nonexistent-aaa".to_string(),
      "nonexistent-bbb".to_string(),
    ]);
    if let Err(err) = &result {
      // Only accept unsupported errors (CLI may not support delete in some configs)
      assert!(
        integration::surfaces::is_not_supported_error(err),
        "{surface}: delete of nonexistent IDs should be idempotent, got error: {err}"
      );
    }
    Ok(())
  })
}

#[test]
fn adversarial_double_commit_idempotency() -> Result<()> {
  for_each_surface(|harness, surface| {
    harness.init(&valid_schema())?;
    harness.add_ndjson(docs_ndjson())?;
    harness.commit()?;
    // Second commit with no new writes — should succeed
    harness
      .commit()
      .unwrap_or_else(|e| panic!("{surface}: double commit should be idempotent, got: {e}"));
    Ok(())
  })
}

#[test]
fn adversarial_unknown_query_type() -> Result<()> {
  let bad_query = json!({
    "query": { "type": "nonexistent_query_type" },
    "limit": 5,
  });

  for_each_surface(|harness, surface| {
    harness.init(&valid_schema())?;
    harness.add_ndjson(docs_ndjson())?;
    harness.commit()?;
    let result = harness.search(&bad_query);
    assert!(
      result.is_err(),
      "{surface}: unknown query type should be rejected"
    );
    Ok(())
  })
}

#[test]
fn adversarial_empty_ndjson_body() -> Result<()> {
  for_each_surface(|harness, surface| {
    harness.init(&valid_schema())?;
    let result = harness.add_ndjson("");
    // Empty body should either succeed (0 docs added) or error — both acceptable
    match result {
      Ok(()) => {} // fine — empty ingest
      Err(err) => {
        let msg = err.to_string().to_lowercase();
        // Should not be a crash/panic — any clean error is acceptable
        assert!(
          !msg.contains("panic") && !msg.contains("internal error"),
          "{surface}: empty NDJSON should not cause panic, got: {msg}"
        );
      }
    }
    Ok(())
  })
}
