use anyhow::Result;
use serde_json::json;
use tempfile::tempdir;

use integration::surfaces::core::CoreHarness;
use integration::surfaces::SurfaceHarness;

fn schema_json() -> serde_json::Value {
  json!({
    "type": "object",
    "properties": {
      "body": { "type": "string" },
      "lang": { "type": "string", "searchlite:kind": "keyword" }
    }
  })
}

#[test]
fn core_contract_stats_and_inspect_shapes() -> Result<()> {
  let dir = tempdir()?;
  let mut harness = CoreHarness::new(dir.path().join("idx-core-contract"));
  harness.init(&schema_json())?;
  harness.add_ndjson(
    "{\"_id\":\"1\",\"body\":\"rust\",\"lang\":\"en\"}\n{\"_id\":\"2\",\"body\":\"query\",\"lang\":\"en\"}\n",
  )?;
  harness.commit()?;

  let stats = harness.stats()?;
  assert!(stats["documents"].as_u64().unwrap_or(0) >= 2);
  assert!(stats.get("segments").is_some());

  let inspect = harness.inspect()?;
  assert!(inspect.get("manifest").is_some());
  assert!(inspect["manifest"]["segments"].is_array());
  Ok(())
}

#[test]
fn core_contract_mget_preserves_order_and_missing() -> Result<()> {
  let dir = tempdir()?;
  let mut harness = CoreHarness::new(dir.path().join("idx-core-mget"));
  harness.init(&schema_json())?;
  harness.add_ndjson(
    "{\"_id\":\"1\",\"body\":\"rust\",\"lang\":\"en\"}\n{\"_id\":\"2\",\"body\":\"query\",\"lang\":\"en\"}\n",
  )?;
  harness.commit()?;

  let body = harness.mget(
    &["1".to_string(), "missing".to_string(), "2".to_string()],
    true,
  )?;
  let docs = body["docs"].as_array().expect("mget docs array");
  assert_eq!(docs.len(), 3);
  assert_eq!(docs[0]["doc_id"], "1");
  assert_eq!(docs[1]["found"], false);
  assert_eq!(docs[2]["doc_id"], "2");
  Ok(())
}
