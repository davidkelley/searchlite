use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::de::{self, Unexpected, Visitor};
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultiMatchFuzziness {
  Auto,
  Edits(u8),
}

impl Serialize for MultiMatchFuzziness {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    match self {
      Self::Auto => serializer.serialize_str("AUTO"),
      Self::Edits(value) => serializer.serialize_u8(*value),
    }
  }
}

struct MultiMatchFuzzinessVisitor;

impl<'de> Visitor<'de> for MultiMatchFuzzinessVisitor {
  type Value = MultiMatchFuzziness;

  fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    formatter.write_str("`AUTO` (string) or an integer edit distance between 0 and 2")
  }

  fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
  where
    E: de::Error,
  {
    if value.eq_ignore_ascii_case("auto") {
      return Ok(MultiMatchFuzziness::Auto);
    }
    let parsed = value.parse::<u8>().map_err(|_| {
      E::invalid_value(
        Unexpected::Str(value),
        &"`AUTO` (string) or an integer edit distance between 0 and 2",
      )
    })?;
    if parsed > 2 {
      return Err(E::invalid_value(
        Unexpected::Unsigned(parsed as u64),
        &"an integer edit distance between 0 and 2",
      ));
    }
    Ok(MultiMatchFuzziness::Edits(parsed))
  }

  fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
  where
    E: de::Error,
  {
    self.visit_str(value.as_str())
  }

  fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
  where
    E: de::Error,
  {
    if value > 2 {
      return Err(E::invalid_value(
        Unexpected::Unsigned(value),
        &"an integer edit distance between 0 and 2",
      ));
    }
    Ok(MultiMatchFuzziness::Edits(value as u8))
  }

  fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
  where
    E: de::Error,
  {
    if !(0..=2).contains(&value) {
      return Err(E::invalid_value(
        Unexpected::Signed(value),
        &"an integer edit distance between 0 and 2",
      ));
    }
    Ok(MultiMatchFuzziness::Edits(value as u8))
  }
}

impl<'de> Deserialize<'de> for MultiMatchFuzziness {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: serde::Deserializer<'de>,
  {
    deserializer.deserialize_any(MultiMatchFuzzinessVisitor)
  }
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
pub enum RankFeatureModifier {
  None,
  Log,
  Log1p,
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
    fuzziness: Option<MultiMatchFuzziness>,
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
  RankFeature {
    field: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    boost: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    modifier: Option<RankFeatureModifier>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    missing: Option<f32>,
  },
  ScriptScore {
    query: Box<QueryNode>,
    script: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    params: Option<BTreeMap<String, f64>>,
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

#[cfg(feature = "vectors")]
pub fn parse_env_max_vector_candidates() -> Option<usize> {
  std::env::var("SEARCHLITE_MAX_VECTOR_CANDIDATES")
    .ok()
    .and_then(|s| s.parse::<usize>().ok())
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
  #[serde(default = "default_limit", alias = "size")]
  pub limit: usize,
  #[serde(default)]
  pub from: usize,
  #[serde(default = "default_return_hits")]
  pub return_hits: bool,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub candidate_size: Option<usize>,
  #[cfg(feature = "vectors")]
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub max_global_vector_candidates: Option<usize>,
  #[serde(default)]
  pub sort: Vec<SortSpec>,
  #[serde(default)]
  pub cursor: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub search_after: Option<Vec<serde_json::Value>>,
  #[serde(default)]
  pub execution: ExecutionStrategy,
  #[serde(default)]
  pub bmw_block_size: Option<usize>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub fuzzy: Option<FuzzyOptions>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub track_total_hits: Option<bool>,
  #[cfg(feature = "vectors")]
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub vector_query: Option<VectorQuerySpec>,
  #[cfg(feature = "vectors")]
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub vector_filter: Option<Filter>,
  #[serde(default = "default_return_stored")]
  pub return_stored: bool,
  pub highlight_field: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub highlight: Option<HighlightRequest>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub collapse: Option<CollapseRequest>,
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

impl SearchRequest {
  /// Create a new search request with sensible defaults.
  ///
  /// Sets `limit` to 10, `execution` to WAND, and all optional fields to
  /// `None` / empty. Use the `with_*` builder methods to customise.
  pub fn new(query: impl Into<Query>) -> Self {
    Self {
      query: query.into(),
      fields: None,
      filter: None,
      limit: 10,
      from: 0,
      return_hits: true,
      candidate_size: None,
      #[cfg(feature = "vectors")]
      max_global_vector_candidates: None,
      sort: Vec::new(),
      cursor: None,
      search_after: None,
      execution: ExecutionStrategy::default(),
      bmw_block_size: None,
      fuzzy: None,
      track_total_hits: None,
      #[cfg(feature = "vectors")]
      vector_query: None,
      #[cfg(feature = "vectors")]
      vector_filter: None,
      return_stored: false,
      highlight_field: None,
      highlight: None,
      collapse: None,
      aggs: BTreeMap::new(),
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    }
  }

  /// Set the maximum number of hits to return.
  pub fn with_limit(mut self, limit: usize) -> Self {
    self.limit = limit;
    self
  }

  /// Add a post-query filter.
  pub fn with_filter(mut self, filter: Filter) -> Self {
    self.filter = Some(filter);
    self
  }

  /// Return stored field values in each hit.
  pub fn with_return_stored(mut self, return_stored: bool) -> Self {
    self.return_stored = return_stored;
    self
  }

  /// Enable single-field highlighting (legacy shorthand).
  pub fn with_highlight_field(mut self, field: impl Into<String>) -> Self {
    self.highlight_field = Some(field.into());
    self
  }

  /// Enable multi-field highlighting.
  pub fn with_highlight(mut self, highlight: HighlightRequest) -> Self {
    self.highlight = Some(highlight);
    self
  }

  /// Add aggregations to the search request.
  pub fn with_aggs(mut self, aggs: BTreeMap<String, Aggregation>) -> Self {
    self.aggs = aggs;
    self
  }

  /// Enable fuzzy matching with the given options.
  pub fn with_fuzzy(mut self, fuzzy: FuzzyOptions) -> Self {
    self.fuzzy = Some(fuzzy);
    self
  }

  /// Set sort order.
  pub fn with_sort(mut self, sort: Vec<SortSpec>) -> Self {
    self.sort = sort;
    self
  }

  /// Skip the first `n` results (offset pagination).
  pub fn with_from(mut self, from: usize) -> Self {
    self.from = from;
    self
  }
}

#[derive(Debug, Clone, Deserialize)]
struct SearchRequestHelper {
  pub query: Query,
  #[serde(default)]
  pub fields: Option<Vec<String>>,
  #[serde(default)]
  pub filter: Option<Filter>,
  #[serde(default = "default_limit", alias = "size")]
  pub limit: usize,
  #[serde(default)]
  pub from: usize,
  #[serde(default = "default_return_hits")]
  pub return_hits: bool,
  #[serde(default)]
  pub candidate_size: Option<usize>,
  #[cfg(feature = "vectors")]
  #[serde(default)]
  pub max_global_vector_candidates: Option<usize>,
  #[serde(default)]
  pub sort: Vec<SortSpec>,
  #[serde(default)]
  pub cursor: Option<String>,
  #[serde(default)]
  pub search_after: Option<Vec<serde_json::Value>>,
  #[serde(default)]
  pub execution: ExecutionStrategy,
  #[serde(default)]
  pub bmw_block_size: Option<usize>,
  #[serde(default)]
  pub fuzzy: Option<FuzzyOptions>,
  #[serde(default)]
  pub track_total_hits: Option<bool>,
  #[cfg(feature = "vectors")]
  #[serde(default)]
  pub vector_query: Option<VectorQuerySpec>,
  #[cfg(feature = "vectors")]
  #[serde(default)]
  pub vector_filter: Option<Filter>,
  #[serde(default = "default_return_stored")]
  pub return_stored: bool,
  pub highlight_field: Option<String>,
  #[serde(default)]
  pub highlight: Option<HighlightRequest>,
  #[serde(default)]
  pub collapse: Option<CollapseRequest>,
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
    Ok(Self {
      query: helper.query,
      fields: helper.fields,
      filter: helper.filter,
      limit: helper.limit,
      from: helper.from,
      return_hits: helper.return_hits,
      candidate_size: helper.candidate_size,
      #[cfg(feature = "vectors")]
      max_global_vector_candidates: helper.max_global_vector_candidates,
      sort: helper.sort,
      cursor: helper.cursor,
      search_after: helper.search_after,
      execution: helper.execution,
      bmw_block_size: helper.bmw_block_size,
      fuzzy: helper.fuzzy,
      track_total_hits: helper.track_total_hits,
      #[cfg(feature = "vectors")]
      vector_query: helper.vector_query,
      #[cfg(feature = "vectors")]
      vector_filter: helper.vector_filter,
      return_stored: helper.return_stored,
      highlight_field: helper.highlight_field,
      highlight: helper.highlight,
      collapse: helper.collapse,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MgetRequest {
  pub ids: Vec<String>,
  #[serde(default)]
  pub return_stored: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MgetDoc {
  pub doc_id: String,
  pub found: bool,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub _source: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MgetResponse {
  pub docs: Vec<MgetDoc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSearchRequest {
  pub searches: Vec<SearchRequest>,
  #[serde(default)]
  pub parallel: bool,
  #[serde(default)]
  pub max_concurrency: Option<usize>,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CollapseRequest {
  pub field: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub inner_hits: Option<InnerHitsRequest>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InnerHitsRequest {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub size: Option<usize>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub from: Option<usize>,
  #[serde(default)]
  pub sort: Vec<SortSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct HighlightRequest {
  #[serde(default)]
  pub fields: BTreeMap<String, HighlightField>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HighlightField {
  #[serde(default = "default_pre_tag")]
  pub pre_tag: String,
  #[serde(default = "default_post_tag")]
  pub post_tag: String,
  #[serde(default = "default_fragment_size")]
  pub fragment_size: usize,
  #[serde(default = "default_num_fragments")]
  pub number_of_fragments: usize,
}

fn default_pre_tag() -> String {
  "<em>".to_string()
}

fn default_post_tag() -> String {
  "</em>".to_string()
}

fn default_fragment_size() -> usize {
  160
}

fn default_num_fragments() -> usize {
  1
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
pub struct FilterAggregation {
  pub filter: Filter,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub sampling: Option<AggregationSampling>,
  #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
  pub aggs: BTreeMap<String, Aggregation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NestedAggregation {
  pub path: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub sampling: Option<AggregationSampling>,
  #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
  pub aggs: BTreeMap<String, Aggregation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompositeAggregation {
  pub sources: Vec<CompositeSource>,
  pub size: usize,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub after: Option<serde_json::Value>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub sampling: Option<AggregationSampling>,
  #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
  pub aggs: BTreeMap<String, Aggregation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CompositeSource {
  Terms {
    name: String,
    field: String,
  },
  Histogram {
    name: String,
    field: String,
    interval: f64,
  },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CardinalityAggregation {
  pub field: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub precision_threshold: Option<usize>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub missing: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PercentilesAggregation {
  pub field: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub percents: Option<Vec<f64>>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub missing: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PercentileRanksAggregation {
  pub field: String,
  pub values: Vec<f64>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub missing: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DerivativeAggregation {
  pub buckets_path: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub gap_policy: Option<GapPolicy>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub unit: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MovingAvgAggregation {
  pub buckets_path: String,
  pub window: usize,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub predict: Option<usize>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub gap_policy: Option<GapPolicy>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BucketScriptAggregation {
  pub buckets_path: BTreeMap<String, String>,
  pub script: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BucketSortAggregation {
  pub sort: Vec<BucketSortSpec>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub from: Option<usize>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub size: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BucketSortSpec {
  pub field: String,
  pub order: SortOrder,
}

impl<'de> Deserialize<'de> for BucketSortSpec {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: serde::Deserializer<'de>,
  {
    let map: BTreeMap<String, SortOrder> = BTreeMap::deserialize(deserializer)?;
    if map.len() != 1 {
      return Err(serde::de::Error::custom(
        "bucket_sort sort entry must contain exactly one field",
      ));
    }
    let (field, order) = map.into_iter().next().unwrap();
    Ok(Self { field, order })
  }
}

impl Serialize for BucketSortSpec {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    let mut map = BTreeMap::new();
    map.insert(self.field.clone(), self.order);
    map.serialize(serializer)
  }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BucketMetricAggregation {
  pub buckets_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AggregationSampling {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub size: Option<usize>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub probability: Option<f64>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub seed: Option<u64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GapPolicy {
  Skip,
  InsertZeros,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Aggregation {
  Terms(Box<TermsAggregation>),
  SignificantTerms(Box<SignificantTermsAggregation>),
  RareTerms(Box<RareTermsAggregation>),
  Range(Box<RangeAggregation>),
  DateRange(Box<DateRangeAggregation>),
  Histogram(Box<HistogramAggregation>),
  DateHistogram(Box<DateHistogramAggregation>),
  Filter(Box<FilterAggregation>),
  Nested(Box<NestedAggregation>),
  Composite(Box<CompositeAggregation>),
  Stats(MetricAggregation),
  ExtendedStats(MetricAggregation),
  ValueCount(MetricAggregation),
  Cardinality(CardinalityAggregation),
  Percentiles(PercentilesAggregation),
  PercentileRanks(PercentileRanksAggregation),
  TopHits(TopHitsAggregation),
  BucketSort(BucketSortAggregation),
  AvgBucket(BucketMetricAggregation),
  SumBucket(BucketMetricAggregation),
  Derivative(DerivativeAggregation),
  MovingAvg(MovingAvgAggregation),
  BucketScript(BucketScriptAggregation),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TermsAggregation {
  pub field: String,
  pub size: Option<usize>,
  pub shard_size: Option<usize>,
  pub min_doc_count: Option<u64>,
  pub missing: Option<serde_json::Value>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub sampling: Option<AggregationSampling>,
  #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
  pub aggs: BTreeMap<String, Aggregation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SignificantTermsAggregation {
  pub field: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub size: Option<usize>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub min_doc_count: Option<u64>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub background_filter: Option<Filter>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub sampling: Option<AggregationSampling>,
  #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
  pub aggs: BTreeMap<String, Aggregation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RareTermsAggregation {
  pub field: String,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub max_doc_count: Option<u64>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub size: Option<usize>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub sampling: Option<AggregationSampling>,
  #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
  pub aggs: BTreeMap<String, Aggregation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RangeAggregation {
  pub field: String,
  pub keyed: bool,
  pub ranges: Vec<RangeBound>,
  pub missing: Option<serde_json::Value>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub sampling: Option<AggregationSampling>,
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
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub sampling: Option<AggregationSampling>,
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
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub sampling: Option<AggregationSampling>,
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
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub sampling: Option<AggregationSampling>,
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
pub struct SignificantBucketResponse {
  pub key: serde_json::Value,
  pub doc_count: u64,
  pub bg_count: u64,
  pub score: f64,
  #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
  pub aggregations: BTreeMap<String, AggregationResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OptionalBucketMetricResponse {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub value: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MovingAvgResponse {
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub value: Option<f64>,
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  pub predictions: Vec<f64>,
}

fn is_false(val: &bool) -> bool {
  !*val
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AggregationResponse {
  Terms {
    buckets: Vec<BucketResponse>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    aggregations: BTreeMap<String, AggregationResponse>,
    #[serde(default, skip_serializing_if = "is_false")]
    sampled: bool,
  },
  SignificantTerms {
    buckets: Vec<SignificantBucketResponse>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    aggregations: BTreeMap<String, AggregationResponse>,
    #[serde(default)]
    doc_count: u64,
    #[serde(default)]
    bg_count: u64,
    #[serde(default, skip_serializing_if = "is_false")]
    sampled: bool,
  },
  RareTerms {
    buckets: Vec<BucketResponse>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    aggregations: BTreeMap<String, AggregationResponse>,
    #[serde(default, skip_serializing_if = "is_false")]
    sampled: bool,
  },
  Range {
    buckets: Vec<BucketResponse>,
    keyed: bool,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    aggregations: BTreeMap<String, AggregationResponse>,
    #[serde(default, skip_serializing_if = "is_false")]
    sampled: bool,
  },
  DateRange {
    buckets: Vec<BucketResponse>,
    keyed: bool,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    aggregations: BTreeMap<String, AggregationResponse>,
    #[serde(default, skip_serializing_if = "is_false")]
    sampled: bool,
  },
  Histogram {
    buckets: Vec<BucketResponse>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    aggregations: BTreeMap<String, AggregationResponse>,
    #[serde(default, skip_serializing_if = "is_false")]
    sampled: bool,
  },
  DateHistogram {
    buckets: Vec<BucketResponse>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    aggregations: BTreeMap<String, AggregationResponse>,
    #[serde(default, skip_serializing_if = "is_false")]
    sampled: bool,
  },
  Filter {
    doc_count: u64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    aggregations: BTreeMap<String, AggregationResponse>,
    #[serde(default, skip_serializing_if = "is_false")]
    sampled: bool,
  },
  Nested {
    doc_count: u64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    aggregations: BTreeMap<String, AggregationResponse>,
    #[serde(default, skip_serializing_if = "is_false")]
    sampled: bool,
  },
  Composite {
    buckets: Vec<BucketResponse>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    after_key: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    aggregations: BTreeMap<String, AggregationResponse>,
    #[serde(default, skip_serializing_if = "is_false")]
    sampled: bool,
  },
  Stats(StatsResponse),
  ExtendedStats(ExtendedStatsResponse),
  ValueCount(ValueCountResponse),
  Cardinality(CardinalityResponse),
  Percentiles(PercentilesResponse),
  PercentileRanks(PercentileRanksResponse),
  TopHits(TopHitsResponse),
  BucketSort {
    #[serde(default)]
    from: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    size: Option<usize>,
  },
  AvgBucket(OptionalBucketMetricResponse),
  SumBucket(OptionalBucketMetricResponse),
  Derivative(OptionalBucketMetricResponse),
  MovingAvg(MovingAvgResponse),
  BucketScript(OptionalBucketMetricResponse),
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CardinalityResponse {
  pub value: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PercentilesResponse {
  /// Percentile values keyed by their percent (e.g. `"50"`, `"99.9"`). The
  /// value is `None` when the aggregation observed no documents with the
  /// requested field — matching Elasticsearch, which serializes those entries
  /// as JSON `null`. Pipeline aggregations treat `None` as a missing value.
  pub values: BTreeMap<String, Option<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PercentileRanksResponse {
  /// Rank values keyed by their target value. `None` signals that the
  /// aggregation observed no documents with the requested field (rather than
  /// a genuine `0.0` rank) so pipeline aggregations can skip the bucket.
  pub values: BTreeMap<String, Option<f64>>,
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
fn default_return_hits() -> bool {
  true
}

fn default_limit() -> usize {
  10
}

fn default_return_stored() -> bool {
  false
}

#[cfg(test)]
mod tests {
  use super::MgetDoc;

  #[test]
  fn mget_doc_serializes_doc_id_field_name() {
    let doc = MgetDoc {
      doc_id: "doc-1".to_string(),
      found: true,
      _source: None,
    };
    let serialized = serde_json::to_value(doc).unwrap();
    assert_eq!(
      serialized.get("doc_id"),
      Some(&serde_json::Value::String("doc-1".to_string()))
    );
    assert!(serialized.get("id").is_none());
  }
}
