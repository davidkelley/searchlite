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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum VectorMetric {
  Cosine,
  L2,
}

#[cfg(feature = "vectors")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LegacyVectorQuery(pub String, pub Vec<f32>, pub f32);

#[cfg(feature = "vectors")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum VectorQuerySpec {
  Structured(VectorQuery),
  Legacy(LegacyVectorQuery),
}

#[cfg(feature = "vectors")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VectorQuery {
  pub field: String,
  pub vector: Vec<f32>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub k: Option<usize>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub alpha: Option<f32>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub ef_search: Option<usize>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub candidate_size: Option<usize>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub boost: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Document {
  pub fields: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FieldSpec {
  pub field: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub boost: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Query {
  String(String),
  Node(QueryNode),
}

impl From<String> for Query {
  fn from(value: String) -> Self {
    Self::String(value)
  }
}

impl From<&str> for Query {
  fn from(value: &str) -> Self {
    Self::String(value.to_string())
  }
}

impl From<QueryNode> for Query {
  fn from(value: QueryNode) -> Self {
    Self::Node(value)
  }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MatchOperator {
  Or,
  And,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MultiMatchType {
  #[default]
  BestFields,
  MostFields,
  CrossFields,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum MinimumShouldMatch {
  Value(usize),
  Percentage(String),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FunctionScoreMode {
  Sum,
  Multiply,
  Max,
  Min,
  Avg,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FunctionBoostMode {
  Multiply,
  Sum,
  Replace,
  Max,
  Min,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FieldValueModifier {
  None,
  Log,
  Log1p,
  Log2p,
  Sqrt,
  Reciprocal,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DecayFunction {
  Exp,
  Gauss,
  Linear,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FunctionSpec {
  Weight {
    weight: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    filter: Option<Filter>,
  },
  FieldValueFactor {
    field: String,
    #[serde(default = "default_factor")]
    factor: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    modifier: Option<FieldValueModifier>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    missing: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    filter: Option<Filter>,
  },
  Decay {
    field: String,
    origin: f64,
    scale: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    offset: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    decay: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    function: Option<DecayFunction>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    filter: Option<Filter>,
  },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum QueryNode {
  /// Match every document. `boost` is validated but does not affect scoring.
  MatchAll {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    boost: Option<f32>,
  },
  QueryString {
    query: String,
    #[serde(
      default,
      skip_serializing_if = "Option::is_none",
      deserialize_with = "deserialize_field_specs_opt"
    )]
    fields: Option<Vec<FieldSpec>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    boost: Option<f32>,
  },
  MultiMatch {
    query: String,
    #[serde(deserialize_with = "deserialize_field_specs")]
    fields: Vec<FieldSpec>,
    #[serde(default)]
    match_type: MultiMatchType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tie_breaker: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    operator: Option<MatchOperator>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    minimum_should_match: Option<MinimumShouldMatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    boost: Option<f32>,
  },
  DisMax {
    queries: Vec<QueryNode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tie_breaker: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    boost: Option<f32>,
  },
  Term {
    field: String,
    value: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    boost: Option<f32>,
  },
  Prefix {
    field: String,
    value: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_expansions: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    boost: Option<f32>,
  },
  Wildcard {
    field: String,
    value: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_expansions: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    boost: Option<f32>,
  },
  Regex {
    field: String,
    value: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_expansions: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    boost: Option<f32>,
  },
  /// Match documents containing the exact phrase. `boost` is validated but does not affect scoring.
  Phrase {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    field: Option<String>,
    terms: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    slop: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    boost: Option<f32>,
  },
  Bool {
    #[serde(default)]
    must: Vec<QueryNode>,
    #[serde(default)]
    should: Vec<QueryNode>,
    #[serde(default)]
    must_not: Vec<QueryNode>,
    #[serde(default)]
    filter: Vec<Filter>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    minimum_should_match: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    boost: Option<f32>,
  },
  ConstantScore {
    filter: Filter,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    boost: Option<f32>,
  },
  FunctionScore {
    query: Box<QueryNode>,
    functions: Vec<FunctionSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    score_mode: Option<FunctionScoreMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    boost_mode: Option<FunctionBoostMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_boost: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    min_score: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    boost: Option<f32>,
  },
  #[cfg(feature = "vectors")]
  Vector(VectorQuery),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum FieldSpecList {
  Names(Vec<String>),
  Specs(Vec<FieldSpec>),
}

fn deserialize_field_specs<'de, D>(deserializer: D) -> Result<Vec<FieldSpec>, D::Error>
where
  D: serde::Deserializer<'de>,
{
  let list = FieldSpecList::deserialize(deserializer)?;
  Ok(match list {
    FieldSpecList::Names(fields) => fields
      .into_iter()
      .map(|field| FieldSpec { field, boost: None })
      .collect(),
    FieldSpecList::Specs(specs) => specs,
  })
}

fn deserialize_field_specs_opt<'de, D>(deserializer: D) -> Result<Option<Vec<FieldSpec>>, D::Error>
where
  D: serde::Deserializer<'de>,
{
  let opt = Option::<FieldSpecList>::deserialize(deserializer)?;
  opt
    .map(|list| {
      Ok(match list {
        FieldSpecList::Names(fields) => fields
          .into_iter()
          .map(|field| FieldSpec { field, boost: None })
          .collect(),
        FieldSpecList::Specs(specs) => specs,
      })
    })
    .transpose()
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchRequest {
  pub query: Query,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub fields: Option<Vec<String>>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub filter: Option<Filter>,
  #[serde(default)]
  pub filters: Vec<Filter>,
  pub limit: usize,
  #[serde(default)]
  pub sort: Vec<SortSpec>,
  #[serde(default)]
  pub cursor: Option<String>,
  #[serde(default)]
  pub execution: ExecutionStrategy,
  #[serde(default)]
  pub bmw_block_size: Option<usize>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub fuzzy: Option<FuzzyOptions>,
  #[cfg(feature = "vectors")]
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub vector_query: Option<VectorQuerySpec>,
  #[cfg(feature = "vectors")]
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub vector_filter: Option<Filter>,
  pub return_stored: bool,
  pub highlight_field: Option<String>,
  #[serde(default)]
  pub aggs: BTreeMap<String, Aggregation>,
  #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
  pub suggest: BTreeMap<String, SuggestRequest>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub rescore: Option<RescoreRequest>,
  #[serde(default)]
  pub explain: bool,
  #[serde(default)]
  pub profile: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct SearchRequestHelper {
  pub query: Query,
  #[serde(default)]
  pub fields: Option<Vec<String>>,
  #[serde(default)]
  pub filter: Option<Filter>,
  #[serde(default)]
  pub filters: Vec<Filter>,
  pub limit: usize,
  #[serde(default)]
  pub sort: Vec<SortSpec>,
  #[serde(default)]
  pub cursor: Option<String>,
  #[serde(default)]
  pub execution: ExecutionStrategy,
  #[serde(default)]
  pub bmw_block_size: Option<usize>,
  #[serde(default)]
  pub fuzzy: Option<FuzzyOptions>,
  #[cfg(feature = "vectors")]
  #[serde(default)]
  pub vector_query: Option<VectorQuerySpec>,
  #[cfg(feature = "vectors")]
  #[serde(default)]
  pub vector_filter: Option<Filter>,
  pub return_stored: bool,
  pub highlight_field: Option<String>,
  #[serde(default)]
  pub aggs: BTreeMap<String, Aggregation>,
  #[serde(default)]
  pub suggest: BTreeMap<String, SuggestRequest>,
  #[serde(default)]
  pub rescore: Option<RescoreRequest>,
  #[serde(default)]
  pub explain: bool,
  #[serde(default)]
  pub profile: bool,
}

impl<'de> Deserialize<'de> for SearchRequest {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: serde::Deserializer<'de>,
  {
    let helper = SearchRequestHelper::deserialize(deserializer)?;
    if helper.filter.is_some() && !helper.filters.is_empty() {
      return Err(serde::de::Error::custom(
        "SearchRequest cannot set both `filter` and `filters`",
      ));
    }
    Ok(Self {
      query: helper.query,
      fields: helper.fields,
      filter: helper.filter,
      filters: helper.filters,
      limit: helper.limit,
      sort: helper.sort,
      cursor: helper.cursor,
      execution: helper.execution,
      bmw_block_size: helper.bmw_block_size,
      fuzzy: helper.fuzzy,
      #[cfg(feature = "vectors")]
      vector_query: helper.vector_query,
      #[cfg(feature = "vectors")]
      vector_filter: helper.vector_filter,
      return_stored: helper.return_stored,
      highlight_field: helper.highlight_field,
      aggs: helper.aggs,
      suggest: helper.suggest,
      rescore: helper.rescore,
      explain: helper.explain,
      profile: helper.profile,
    })
  }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RescoreRequest {
  pub window_size: usize,
  pub query: QueryNode,
  #[serde(default)]
  pub score_mode: RescoreMode,
}

/// How to combine the original document score with a rescore query score.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum RescoreMode {
  #[default]
  /// Sum the original score and the rescore score (`orig + rescore`).
  Total,
  /// Multiply the original score and the rescore score.
  Multiply,
  /// Backwards-compatible alias for [`RescoreMode::Total`].
  Sum,
  /// Use the maximum of the original and rescore scores.
  Max,
  /// Use the minimum of the original and rescore scores.
  Min,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyOptions {
  #[serde(default = "default_fuzzy_max_edits")]
  pub max_edits: u8,
  #[serde(default = "default_fuzzy_prefix_length")]
  pub prefix_length: usize,
  #[serde(default = "default_fuzzy_max_expansions")]
  pub max_expansions: usize,
  #[serde(default = "default_fuzzy_min_length")]
  pub min_length: usize,
}

impl Default for FuzzyOptions {
  fn default() -> Self {
    Self {
      max_edits: default_fuzzy_max_edits(),
      prefix_length: default_fuzzy_prefix_length(),
      max_expansions: default_fuzzy_max_expansions(),
      min_length: default_fuzzy_min_length(),
    }
  }
}

fn default_fuzzy_max_edits() -> u8 {
  1
}

fn default_fuzzy_prefix_length() -> usize {
  1
}

fn default_fuzzy_max_expansions() -> usize {
  50
}

fn default_fuzzy_min_length() -> usize {
  3
}

fn default_factor() -> f32 {
  1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SuggestRequest {
  Completion {
    field: String,
    prefix: String,
    #[serde(default = "default_suggest_size")]
    size: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    fuzzy: Option<FuzzyOptions>,
  },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SuggestResult {
  pub options: Vec<SuggestOption>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SuggestOption {
  pub text: String,
  pub score: f32,
  pub doc_freq: u64,
}

fn default_suggest_size() -> usize {
  5
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Filter {
  KeywordEq { field: String, value: String },
  KeywordIn { field: String, values: Vec<String> },
  I64Range { field: String, min: i64, max: i64 },
  F64Range { field: String, min: f64, max: f64 },
  Nested { path: String, filter: Box<Filter> },
  And(Vec<Filter>),
  Or(Vec<Filter>),
  Not(Box<Filter>),
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

/// Metric aggregations operate on numeric fast fields. When the field is
/// multi-valued each value contributes to stats/extended_stats; `BucketResponse::doc_count`
/// remains per-document.
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
  pub sort: Vec<SortSpec>,
  #[serde(default)]
  pub highlight_field: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Copy)]
#[serde(rename_all = "lowercase")]
pub enum SortOrder {
  Asc,
  Desc,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SortSpec {
  pub field: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub order: Option<SortOrder>,
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

/// Aggregate statistics over the numeric field values contributing to the bucket.
/// For multi-valued fields all values are included.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StatsResponse {
  /// Number of field values included (multi-valued fields contribute each entry).
  pub count: u64,
  pub min: f64,
  pub max: f64,
  pub sum: f64,
  pub avg: f64,
}

/// Extended stats computed over all numeric field values contributing to the bucket.
/// For multi-valued fields all values are included.
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
  pub doc_id: String,
  pub score: Option<f32>,
  pub fields: Option<serde_json::Value>,
  pub snippet: Option<String>,
}

pub use crate::index::manifest::{
  KeywordField, NestedField, NestedProperty, NumericField, Schema, SearchAsYouType, TextField,
};
