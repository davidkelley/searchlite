use anyhow::Result;
use serde_json::json;

use integration::scenario::{ExpectedOutcome, ScenarioCase};
use integration::surfaces::{SurfaceHarness, SurfaceKind};

struct StubHarness;

impl SurfaceHarness for StubHarness {
  fn kind(&self) -> SurfaceKind {
    SurfaceKind::Core
  }

  fn init(&mut self, _schema: &serde_json::Value) -> Result<()> {
    Ok(())
  }

  fn add_ndjson(&mut self, _ndjson: &str) -> Result<()> {
    Ok(())
  }

  fn commit(&mut self) -> Result<()> {
    Ok(())
  }

  fn search(&mut self, _request: &serde_json::Value) -> Result<serde_json::Value> {
    Ok(json!({"hits": []}))
  }
}

#[test]
fn integration_harness_compiles() {
  let case = ScenarioCase {
    id: "smoke".to_string(),
    surface: SurfaceKind::Core,
    dataset: "recipes".to_string(),
    expected: ExpectedOutcome::default(),
  };
  assert_eq!(case.id, "smoke");

  let mut harness = StubHarness;
  harness.init(&json!({"schema": "ok"})).unwrap();
  harness.add_ndjson("{\"_id\":\"1\"}\n").unwrap();
  harness.commit().unwrap();
  let body = harness.search(&json!({"query": "rust"})).unwrap();
  assert!(body.get("hits").is_some());
}
