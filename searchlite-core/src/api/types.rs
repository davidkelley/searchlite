use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexOptions {
  pub path: PathBuf,
  pub create_if_missing: bool,
  pub enable_positions: bool,
  pub bm25_k1: f32,
  pub bm25_b: f32,
  #[cfg(feature = "vectors")]
  pub vector_defaults: Option<VectorOptions>,
}

#[cfg(feature = "vectors")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorOptions {
  pub dim: usize,
  pub metric: VectorMetric,
}

#[cfg(feature = "vectors")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorMetric {
  Cosine,
  L2,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Document {
  pub fields: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
  pub query: String,
  pub fields: Option<Vec<String>>,
  pub filters: Vec<Filter>,
  pub limit: usize,
  #[cfg(feature = "vectors")]
  pub vector_query: Option<(String, Vec<f32>, f32)>,
  pub return_stored: bool,
  pub highlight_field: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Filter {
  KeywordEq { field: String, value: String },
  KeywordIn { field: String, values: Vec<String> },
  I64Range { field: String, min: i64, max: i64 },
  F64Range { field: String, min: f64, max: f64 },
}

pub use crate::index::manifest::{KeywordField, NumericField, Schema, TextField};
