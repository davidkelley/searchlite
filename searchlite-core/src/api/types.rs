use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionStrategy {
  Bm25,
  #[default]
  Wand,
  Bmw,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum StorageType {
  #[default]
  Filesystem,
  InMemory,
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
  pub aggs: BTreeMap<String, Aggregation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Filter {
  KeywordEq { field: String, value: String },
  KeywordIn { field: String, values: Vec<String> },
  I64Range { field: String, min: i64, max: i64 },
  F64Range { field: String, min: f64, max: f64 },
  Nested { path: String, filter: Box<Filter> },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Aggregations(pub BTreeMap<String, Aggregation>);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Aggregation {
  Terms(Box<TermsAggregation>),
  Range(Box<RangeAggregation>),
  DateRange(Box<DateRangeAggregation>),
  Histogram(Box<HistogramAggregation>),
  DateHistogram(Box<DateHistogramAggregation>),
  Stats(MetricAggregation),
  ExtendedStats(MetricAggregation),
  ValueCount(MetricAggregation),
  TopHits(TopHitsAggregation),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TermsAggregation {
  pub field: String,
  pub size: Option<usize>,
  pub shard_size: Option<usize>,
  pub min_doc_count: Option<u64>,
  pub missing: Option<serde_json::Value>,
  #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
  pub aggs: BTreeMap<String, Aggregation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RangeAggregation {
  pub field: String,
  pub keyed: bool,
  pub ranges: Vec<RangeBound>,
  pub missing: Option<serde_json::Value>,
  #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
  pub aggs: BTreeMap<String, Aggregation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DateRangeAggregation {
  pub field: String,
  pub keyed: bool,
  pub format: Option<String>,
  pub ranges: Vec<DateRangeBound>,
  pub missing: Option<serde_json::Value>,
  #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
  pub aggs: BTreeMap<String, Aggregation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HistogramAggregation {
  pub field: String,
  pub interval: f64,
  pub offset: Option<f64>,
  pub min_doc_count: Option<u64>,
  pub extended_bounds: Option<HistogramBounds>,
  pub hard_bounds: Option<HistogramBounds>,
  pub missing: Option<f64>,
  #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
  pub aggs: BTreeMap<String, Aggregation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DateHistogramAggregation {
  pub field: String,
  pub calendar_interval: Option<String>,
  pub fixed_interval: Option<String>,
  pub offset: Option<String>,
  pub format: Option<String>,
  pub min_doc_count: Option<u64>,
  pub extended_bounds: Option<DateHistogramBounds>,
  pub hard_bounds: Option<DateHistogramBounds>,
  pub missing: Option<String>,
  #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
  pub aggs: BTreeMap<String, Aggregation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricAggregation {
  pub field: String,
  pub missing: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TopHitsAggregation {
  pub size: usize,
  #[serde(default)]
  pub from: usize,
  #[serde(default)]
  pub fields: Option<Vec<String>>,
  #[serde(default)]
  pub sort: Option<String>,
  #[serde(default)]
  pub highlight_field: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RangeBound {
  pub key: Option<String>,
  pub from: Option<f64>,
  pub to: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DateRangeBound {
  pub key: Option<String>,
  pub from: Option<String>,
  pub to: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HistogramBounds {
  pub min: f64,
  pub max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DateHistogramBounds {
  pub min: String,
  pub max: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BucketResponse {
  pub key: serde_json::Value,
  pub doc_count: u64,
  #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
  pub aggregations: BTreeMap<String, AggregationResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AggregationResponse {
  Terms {
    buckets: Vec<BucketResponse>,
  },
  Range {
    buckets: Vec<BucketResponse>,
    keyed: bool,
  },
  DateRange {
    buckets: Vec<BucketResponse>,
    keyed: bool,
  },
  Histogram {
    buckets: Vec<BucketResponse>,
  },
  DateHistogram {
    buckets: Vec<BucketResponse>,
  },
  Stats(StatsResponse),
  ExtendedStats(ExtendedStatsResponse),
  ValueCount(ValueCountResponse),
  TopHits(TopHitsResponse),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StatsResponse {
  pub count: u64,
  pub min: f64,
  pub max: f64,
  pub sum: f64,
  pub avg: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExtendedStatsResponse {
  pub count: u64,
  pub min: f64,
  pub max: f64,
  pub sum: f64,
  pub avg: f64,
  pub variance: f64,
  pub std_deviation: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValueCountResponse {
  pub value: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TopHitsResponse {
  pub total: u64,
  pub hits: Vec<TopHit>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TopHit {
  pub doc_id: crate::DocId,
  pub score: Option<f32>,
  pub fields: Option<serde_json::Value>,
  pub snippet: Option<String>,
}

pub use crate::index::manifest::{
  KeywordField, NestedField, NestedProperty, NumericField, Schema, TextField,
};
