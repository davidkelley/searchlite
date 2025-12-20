use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionStrategy {
  Bm25,
  Wand,
  Bmw,
}

impl Default for ExecutionStrategy {
  fn default() -> Self {
    ExecutionStrategy::Wand
  }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexOptions {
  pub path: PathBuf,
  pub create_if_missing: bool,
  pub enable_positions: bool,
  pub bm25_k1: f32,
  pub bm25_b: f32,
  #[serde(default)]
  pub storage: StorageType,
  #[cfg(feature = "vectors")]
  pub vector_defaults: Option<VectorOptions>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum StorageType {
  Filesystem,
  InMemory,
}

impl Default for StorageType {
  fn default() -> Self {
    StorageType::Filesystem
  }
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
  #[serde(default)]
  pub execution: ExecutionStrategy,
  #[serde(default)]
  pub bmw_block_size: Option<usize>,
  #[cfg(feature = "vectors")]
  pub vector_query: Option<(String, Vec<f32>, f32)>,
  pub return_stored: bool,
  pub highlight_field: Option<String>,
  #[serde(default)]
  pub facets: Vec<FacetRequest>,
  #[serde(default)]
  pub aggregations: Vec<AggregationRequest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Filter {
  KeywordEq { field: String, value: String },
  KeywordIn { field: String, values: Vec<String> },
  I64Range { field: String, min: i64, max: i64 },
  F64Range { field: String, min: f64, max: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacetRequest {
  pub field: String,
  #[serde(flatten)]
  pub facet: FacetKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum FacetKind {
  Keyword,
  #[serde(rename_all = "camelCase")]
  Range {
    ranges: Vec<NumericRange>,
  },
  #[serde(rename_all = "camelCase")]
  Histogram {
    interval: f64,
    #[serde(default)]
    min: Option<f64>,
  },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericRange {
  pub min: f64,
  pub max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationRequest {
  pub field: String,
  #[serde(default)]
  pub operations: Vec<AggregationOp>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum AggregationOp {
  Min,
  Max,
  Sum,
  Avg,
}

pub use crate::index::manifest::{KeywordField, NumericField, Schema, TextField};
