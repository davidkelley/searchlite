use serde_json::Value;

use crate::surfaces::SurfaceKind;

#[derive(Debug, Clone)]
pub struct ExpectedOutcome {
  pub expect_success: bool,
  pub expected_error_type: Option<String>,
  pub metadata: Value,
}

impl Default for ExpectedOutcome {
  fn default() -> Self {
    Self {
      expect_success: true,
      expected_error_type: None,
      metadata: Value::Null,
    }
  }
}

#[derive(Debug, Clone)]
pub struct ScenarioCase {
  pub id: String,
  pub surface: SurfaceKind,
  pub dataset: String,
  pub expected: ExpectedOutcome,
}
