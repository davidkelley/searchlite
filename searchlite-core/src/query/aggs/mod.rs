use std::cmp::Ordering;
use std::collections::{
  btree_map::Entry as BTreeEntry, hash_map::DefaultHasher, hash_map::Entry as HashEntry, BTreeMap,
  BinaryHeap, HashMap, HashSet, VecDeque,
};
use std::hash::{Hash, Hasher};

use crate::api::types::{
  Aggregation, AggregationResponse, AggregationSampling, BucketMetricResponse, BucketResponse,
  BucketScriptAggregation, BucketSortAggregation, BucketSortSpec, CardinalityResponse,
  CompositeAggregation, CompositeSource, DateHistogramAggregation, DateRangeAggregation,
  DerivativeAggregation, Filter, FilterAggregation, GapPolicy, HistogramAggregation,
  MovingAvgAggregation, MovingAvgResponse, NestedAggregation, OptionalBucketMetricResponse,
  PercentileRanksResponse, PercentilesResponse, RangeAggregation, RareTermsAggregation,
  SignificantBucketResponse, SignificantTermsAggregation, SortOrder, StatsResponse,
  TermsAggregation, TopHit, TopHitsAggregation, TopHitsResponse, ValueCountResponse,
};
use crate::index::fastfields::FastFieldsReader;
use crate::index::highlight::make_snippet;
use crate::index::manifest::Schema;
use crate::index::segment::SegmentReader;
use crate::query::collector::{AggregationSegmentCollector, DocCollector};
use crate::query::filters::passes_filter;
use crate::query::sort::{SortKey, SortPlan};
use crate::util::path_scope::resolve_scoped_path;
use crate::DocId;
use tdigest::TDigest;

#[derive(Clone)]
pub struct AggregationContext<'a> {
  pub fast_fields: &'a FastFieldsReader,
  pub segment: &'a SegmentReader,
  pub highlight_terms: &'a [String],
  pub schema: &'a Schema,
  pub segment_ord: u32,
}

/// Upper bound on the number of aggregation buckets we will materialize for a single request.
///
/// This protects against excessive memory/CPU usage in high-cardinality aggregations. The value
/// `10_000` is a pragmatic default that keeps memory bounded for typical analytics workloads while
/// still allowing thousands of buckets. Deployments that need a different limit can adjust this
/// constant at compile time.
pub(crate) const MAX_BUCKETS: usize = 10_000;
/// Upper bound on the number of forecast points a single `moving_avg` pipeline may emit.
///
/// `predict` controls a `Vec<f64>` allocation in [`apply_moving_avg_pipeline`] sized directly by
/// untrusted user input (BUG-221). Without a cap, a tiny request body can drive an unbounded heap
/// allocation during response finalization. We share the same `10_000` ceiling as `MAX_BUCKETS`
/// so the forecast horizon never grows past the materialization budget for any other bucketing
/// aggregation in the same request.
pub(crate) const MAX_PREDICTIONS: usize = MAX_BUCKETS;
/// Upper bound on the number of hits a single `top_hits` sub-aggregation may track per segment.
///
/// `size` and `from` control the per-segment `BinaryHeap<RankedDoc>` allocation in
/// [`TopHitsCollector`]; without a cap, a tiny request body can size the heap from untrusted user
/// input and drive an unbounded heap growth during collection (BUG-222). We share the same
/// `10_000` ceiling as `MAX_BUCKETS` so the materialized hit set never grows past the
/// materialization budget for any other bucketing aggregation in the same request.
pub(crate) const MAX_TOP_HITS: usize = MAX_BUCKETS;
const TDIGEST_MAX_SIZE: usize = 200;
const PERCENTILE_EXACT_LIMIT: usize = 256;

#[derive(Clone, Copy)]
enum SamplingMode {
  None,
  Probability(f64),
  TopN(usize),
}

#[derive(Clone)]
struct Sampler {
  mode: SamplingMode,
  seed: u64,
  accepted: usize,
}

impl Sampler {
  fn new(config: Option<&AggregationSampling>) -> Self {
    if let Some(cfg) = config {
      if let Some(limit) = cfg.size {
        return Self {
          mode: SamplingMode::TopN(limit),
          seed: cfg.seed.unwrap_or(0),
          accepted: 0,
        };
      }
      if let Some(prob) = cfg.probability {
        return Self {
          mode: SamplingMode::Probability(prob.clamp(0.0, 1.0)),
          seed: cfg.seed.unwrap_or(0),
          accepted: 0,
        };
      }
      if let Some(seed) = cfg.seed {
        return Self {
          mode: SamplingMode::None,
          seed,
          accepted: 0,
        };
      }
    }
    Self {
      mode: SamplingMode::None,
      seed: 0,
      accepted: 0,
    }
  }

  fn accept(&mut self, segment_ord: u32, doc_id: DocId) -> bool {
    self.accept_with_object(segment_ord, doc_id, None)
  }

  fn accept_object(&mut self, segment_ord: u32, doc_id: DocId, object_idx: usize) -> bool {
    self.accept_with_object(segment_ord, doc_id, Some(object_idx))
  }

  fn accept_with_object(
    &mut self,
    segment_ord: u32,
    doc_id: DocId,
    object_idx: Option<usize>,
  ) -> bool {
    match self.mode {
      SamplingMode::None => true,
      SamplingMode::Probability(p) => {
        if p <= 0.0 {
          return false;
        }
        if p >= 1.0 {
          return true;
        }
        let value = self.sample_value_with_object(segment_ord, doc_id, object_idx);
        let threshold = (p * (u64::MAX as f64)) as u64;
        value < threshold
      }
      SamplingMode::TopN(limit) => {
        if self.accepted < limit {
          self.accepted += 1;
          true
        } else {
          false
        }
      }
    }
  }

  fn sampled(&self) -> bool {
    !matches!(self.mode, SamplingMode::None)
  }

  #[cfg(test)]
  fn sample_value(&self, segment_ord: u32, doc_id: DocId) -> u64 {
    self.sample_value_with_object(segment_ord, doc_id, None)
  }

  fn sample_value_with_object(
    &self,
    segment_ord: u32,
    doc_id: DocId,
    object_idx: Option<usize>,
  ) -> u64 {
    let mut hasher = DefaultHasher::new();
    hasher.write_u64(self.seed);
    hasher.write_u32(segment_ord);
    hasher.write_u32(doc_id);
    if let Some(obj) = object_idx {
      hasher.write_u64(obj as u64);
    }
    hasher.finish()
  }
}

pub(crate) struct SignificantTermsCollector<'a> {
  field: String,
  size: Option<usize>,
  min_doc_count: u64,
  bg_counts: HashMap<String, u64>,
  bg_total: u64,
  buckets: HashMap<BucketKey<'a>, SignificantBucketState<'a>>,
  sub_aggs: BTreeMap<String, Aggregation>,
  pipeline_aggs: BTreeMap<String, Aggregation>,
  sampler: Sampler,
  doc_count: u64,
  ctx: AggregationContext<'a>,
}

struct SignificantBucketState<'a> {
  key: serde_json::Value,
  doc_count: u64,
  bg_count: u64,
  aggs: BTreeMap<String, AggregationNode<'a>>,
}

impl<'a> SignificantTermsCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &SignificantTermsAggregation) -> Self {
    let (sub_aggs, pipeline_aggs) = split_pipeline_aggs(&agg.aggs);
    let (bg_counts, bg_total) =
      compute_background_counts(&ctx, &agg.field, agg.background_filter.as_ref());
    Self {
      field: agg.field.clone(),
      size: agg.size,
      min_doc_count: agg.min_doc_count.unwrap_or(1),
      bg_counts,
      bg_total,
      buckets: HashMap::new(),
      sub_aggs,
      pipeline_aggs,
      sampler: Sampler::new(agg.sampling.as_ref()),
      doc_count: 0,
      ctx,
    }
  }

  fn get_bucket(
    &mut self,
    bucket_key: BucketKey<'a>,
    bg_count: u64,
  ) -> &mut SignificantBucketState<'a> {
    match self.buckets.entry(bucket_key) {
      HashEntry::Occupied(entry) => entry.into_mut(),
      HashEntry::Vacant(entry) => {
        let key_str = entry.key().as_str().to_string();
        entry.insert(SignificantBucketState {
          key: serde_json::Value::String(key_str),
          doc_count: 0,
          bg_count,
          aggs: build_children(&self.ctx, &self.sub_aggs),
        })
      }
    }
  }

  fn collect(&mut self, doc_id: DocId, score: f32) {
    if !self.sampler.accept(self.ctx.segment_ord, doc_id) {
      return;
    }
    let values = self.ctx.fast_fields.str_values(&self.field, doc_id);
    if values.is_empty() {
      return;
    }
    self.doc_count += 1;
    let mut seen = HashSet::new();
    for val in values.into_iter().filter(|v| seen.insert(*v)) {
      let bg_count = *self.bg_counts.get(val).unwrap_or(&0);
      let bucket = self.get_bucket(BucketKey::Borrowed(val), bg_count);
      bucket.doc_count += 1;
      for child in bucket.aggs.values_mut() {
        child.collect(doc_id, score);
      }
    }
  }

  fn finish(self) -> AggregationIntermediate {
    let mut buckets: Vec<SignificantBucketState<'a>> = self
      .buckets
      .into_values()
      .filter(|b| b.doc_count >= self.min_doc_count)
      .collect();
    // Sort by significance score proxy (doc_count / bg_count) descending
    // before truncation. Since the foreground/background totals are constant
    // across all buckets, this proxy is monotonically related to the full
    // significance score `(doc_count/fg_total) / (bg_count/bg_total)`.
    //
    // Buckets with bg_count == 0 are treated as score 0 to match the final
    // scoring guard in finalize_response. Compare ratios via integer
    // cross-multiplication to avoid float rounding.
    buckets.sort_by(|a, b| {
      let score_cmp = match (a.bg_count == 0, b.bg_count == 0) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater, // a has score 0, b > 0 → b first
        (false, true) => Ordering::Less,    // b has score 0, a > 0 → a first
        (false, false) => {
          let left = (a.doc_count as u128) * (b.bg_count as u128);
          let right = (b.doc_count as u128) * (a.bg_count as u128);
          right.cmp(&left)
        }
      };
      score_cmp.then_with(|| terms_bucket_cmp(&a.key, a.doc_count, &b.key, b.doc_count))
    });
    let limit = self.size.unwrap_or(buckets.len()).min(MAX_BUCKETS);
    buckets.truncate(limit);
    AggregationIntermediate::SignificantTerms {
      buckets: buckets
        .into_iter()
        .map(|b| SignificantBucketIntermediate {
          key: b.key,
          doc_count: b.doc_count,
          bg_count: b.bg_count,
          aggs: finalize_children(b.aggs),
        })
        .collect(),
      size: self.size,
      min_doc_count: self.min_doc_count,
      pipeline: self.pipeline_aggs,
      doc_count: self.doc_count,
      bg_count: self.bg_total,
      sampled: self.sampler.sampled(),
    }
  }
}

fn compute_background_counts(
  ctx: &AggregationContext<'_>,
  field: &str,
  filter: Option<&Filter>,
) -> (HashMap<String, u64>, u64) {
  if filter.is_none() && ctx.segment.meta.deleted_docs.is_empty() {
    // Fast path: use the term dictionary to pull doc frequencies without scanning every doc.
    let prefix = format!("{field}:");
    let field_prefix_len = prefix.len();
    let mut counts = HashMap::new();
    for key in ctx.segment.terms_with_prefix(&prefix) {
      if key.len() <= field_prefix_len {
        continue;
      }
      if let Some(df) = ctx.segment.doc_freq(key) {
        if df > 0 {
          counts.insert(key[field_prefix_len..].to_string(), df as u64);
        }
      }
    }
    let total = ctx.segment.live_docs() as u64;
    return (counts, total);
  }
  let mut counts = HashMap::new();
  let mut total = 0_u64;
  for doc_id in 0..ctx.segment.meta.doc_count {
    if ctx.segment.is_deleted(doc_id) {
      continue;
    }
    if let Some(f) = filter {
      if !passes_filter(ctx.fast_fields, doc_id, f) {
        continue;
      }
    }
    total += 1;
    let values = ctx.fast_fields.str_values(field, doc_id);
    let mut seen = HashSet::new();
    for val in values.into_iter().filter(|v| seen.insert(*v)) {
      *counts.entry(val.to_string()).or_insert(0) += 1;
    }
  }
  (counts, total)
}

pub(crate) struct RareTermsCollector<'a> {
  field: String,
  max_doc_count: u64,
  size: Option<usize>,
  buckets: HashMap<BucketKey<'a>, BucketState<'a>>,
  sub_aggs: BTreeMap<String, Aggregation>,
  pipeline_aggs: BTreeMap<String, Aggregation>,
  sampler: Sampler,
  ctx: AggregationContext<'a>,
}

impl<'a> RareTermsCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &RareTermsAggregation) -> Self {
    let (sub_aggs, pipeline_aggs) = split_pipeline_aggs(&agg.aggs);
    Self {
      field: agg.field.clone(),
      max_doc_count: agg.max_doc_count.unwrap_or(1),
      size: agg.size,
      buckets: HashMap::new(),
      sub_aggs,
      pipeline_aggs,
      sampler: Sampler::new(agg.sampling.as_ref()),
      ctx,
    }
  }

  fn collect(&mut self, doc_id: DocId, score: f32) {
    if !self.sampler.accept(self.ctx.segment_ord, doc_id) {
      return;
    }
    let values = self.ctx.fast_fields.str_values(&self.field, doc_id);
    if values.is_empty() {
      return;
    }
    let mut seen = HashSet::new();
    for val in values.into_iter().filter(|v| seen.insert(*v)) {
      let bucket = self
        .buckets
        .entry(BucketKey::Borrowed(val))
        .or_insert_with(|| BucketState {
          key: serde_json::Value::String(val.to_string()),
          doc_count: 0,
          aggs: build_children(&self.ctx, &self.sub_aggs),
        });
      bucket.doc_count += 1;
      for child in bucket.aggs.values_mut() {
        child.collect(doc_id, score);
      }
    }
  }

  fn finish(self) -> AggregationIntermediate {
    let mut buckets: Vec<BucketState<'a>> = self
      .buckets
      .into_values()
      .filter(|b| b.doc_count > 0 && b.doc_count <= self.max_doc_count)
      .collect();
    buckets.sort_by(|a, b| rare_terms_bucket_cmp(&a.key, a.doc_count, &b.key, b.doc_count));
    let limit = self.size.unwrap_or(buckets.len()).min(MAX_BUCKETS);
    buckets.truncate(limit);
    AggregationIntermediate::RareTerms {
      buckets: buckets
        .into_iter()
        .map(|b| BucketIntermediate {
          key: b.key,
          doc_count: b.doc_count,
          aggs: finalize_children(b.aggs),
        })
        .collect(),
      size: self.size,
      max_doc_count: self.max_doc_count,
      pipeline: self.pipeline_aggs,
      sampled: self.sampler.sampled(),
    }
  }
}

#[derive(Clone)]
pub struct BucketIntermediate {
  pub key: serde_json::Value,
  pub doc_count: u64,
  pub aggs: BTreeMap<String, AggregationIntermediate>,
}

#[derive(Clone)]
pub struct SignificantBucketIntermediate {
  pub key: serde_json::Value,
  pub doc_count: u64,
  pub bg_count: u64,
  pub aggs: BTreeMap<String, AggregationIntermediate>,
}

#[derive(Clone)]
pub enum AggregationIntermediate {
  Terms {
    buckets: Vec<BucketIntermediate>,
    size: Option<usize>,
    shard_size: Option<usize>,
    pipeline: BTreeMap<String, Aggregation>,
    sampled: bool,
  },
  SignificantTerms {
    buckets: Vec<SignificantBucketIntermediate>,
    size: Option<usize>,
    min_doc_count: u64,
    pipeline: BTreeMap<String, Aggregation>,
    doc_count: u64,
    bg_count: u64,
    sampled: bool,
  },
  RareTerms {
    buckets: Vec<BucketIntermediate>,
    size: Option<usize>,
    max_doc_count: u64,
    pipeline: BTreeMap<String, Aggregation>,
    sampled: bool,
  },
  Range {
    buckets: Vec<BucketIntermediate>,
    keyed: bool,
    pipeline: BTreeMap<String, Aggregation>,
    sampled: bool,
  },
  DateRange {
    buckets: Vec<BucketIntermediate>,
    keyed: bool,
    pipeline: BTreeMap<String, Aggregation>,
    sampled: bool,
  },
  Histogram {
    buckets: Vec<BucketIntermediate>,
    pipeline: BTreeMap<String, Aggregation>,
    sampled: bool,
  },
  DateHistogram {
    buckets: Vec<BucketIntermediate>,
    pipeline: BTreeMap<String, Aggregation>,
    sampled: bool,
  },
  Stats(StatsState),
  ExtendedStats(StatsState),
  ValueCount(ValueCountState),
  Cardinality(CardinalityState),
  Percentiles(PercentileState),
  PercentileRanks(PercentileRankState),
  TopHits(TopHitsState),
  Filter {
    bucket: BucketIntermediate,
    pipeline: BTreeMap<String, Aggregation>,
    sampled: bool,
  },
  Nested {
    bucket: BucketIntermediate,
    pipeline: BTreeMap<String, Aggregation>,
    sampled: bool,
  },
  Composite {
    buckets: Vec<BucketIntermediate>,
    size: usize,
    after: Option<serde_json::Value>,
    pipeline: BTreeMap<String, Aggregation>,
    sources: Vec<CompositeSource>,
    sampled: bool,
  },
}

#[derive(Clone, Copy, Default)]
pub struct StatsState {
  pub count: u64,
  pub min: f64,
  pub max: f64,
  pub sum: f64,
  pub m2: f64,
}

#[derive(Clone, Copy, Default)]
pub struct ValueCountState {
  pub value: u64,
}

#[derive(Clone, Default)]
pub struct CardinalityState {
  pub values: HashSet<u64>,
  pub precision_threshold: Option<usize>,
}

#[derive(Clone, Default)]
pub struct QuantileState {
  values: Vec<f64>,
  digest: Option<TDigest>,
  count: usize,
}

impl QuantileState {
  fn push(&mut self, value: f64) {
    self.count = self.count.saturating_add(1);
    if self.count <= PERCENTILE_EXACT_LIMIT && self.digest.is_none() {
      self.values.push(value);
      return;
    }
    self.ensure_digest();
    if let Some(digest) = self.digest.take() {
      self.digest = Some(digest.merge_unsorted(vec![value]));
    }
  }

  fn ensure_digest(&mut self) {
    let vals = std::mem::take(&mut self.values);
    if self.digest.is_none() {
      let base = TDigest::new_with_size(TDIGEST_MAX_SIZE);
      self.digest = Some(base.merge_unsorted(vals));
      return;
    }
    if vals.is_empty() {
      return;
    }
    if let Some(digest) = self.digest.take() {
      self.digest = Some(digest.merge_unsorted(vals));
    }
  }

  fn merge(&mut self, mut other: QuantileState) {
    self.count = self.count.saturating_add(other.count);
    if self.count <= PERCENTILE_EXACT_LIMIT
      && self.digest.is_none()
      && other.digest.is_none()
      && self.values.len() + other.values.len() <= PERCENTILE_EXACT_LIMIT
    {
      self.values.extend(other.values);
      return;
    }
    self.ensure_digest();
    let mut digest = self.digest.take().unwrap();
    if !other.values.is_empty() {
      digest = digest.merge_unsorted(other.values);
    }
    if let Some(other_digest) = other.digest.take() {
      digest = TDigest::merge_digests(vec![digest, other_digest]);
    }
    self.digest = Some(digest);
    self.values.clear();
  }

  fn percentile(&mut self, pct: f64) -> f64 {
    if self.count == 0 {
      return 0.0;
    }
    if self.count <= PERCENTILE_EXACT_LIMIT && self.digest.is_none() {
      let mut vals = self.values.clone();
      vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
      let n = vals.len() as f64;
      let rank = ((pct.clamp(0.0, 100.0) / 100.0) * (n - 1.0)).max(0.0);
      let low = rank.floor() as usize;
      let high = rank.ceil() as usize;
      if low == high {
        return vals[low];
      }
      let weight = rank - low as f64;
      return vals[low] * (1.0 - weight) + vals[high] * weight;
    }
    self.ensure_digest();
    let Some(digest) = self.digest.as_ref() else {
      return 0.0;
    };
    let q = pct.clamp(0.0, 100.0) / 100.0;
    digest.estimate_quantile(q)
  }

  fn percentile_rank(&mut self, target: f64) -> f64 {
    if self.count == 0 {
      return 0.0;
    }
    if self.count <= PERCENTILE_EXACT_LIMIT && self.digest.is_none() {
      let count = self.values.iter().filter(|v| **v <= target).count();
      return (count as f64 / self.values.len().max(1) as f64) * 100.0;
    }
    self.ensure_digest();
    let Some(digest) = self.digest.as_ref() else {
      return 0.0;
    };
    let min_val = digest.estimate_quantile(0.0);
    // Use a strict `<` here so that `target == min_val` falls through to the
    // binary search, which matches the exact path's inclusive semantics
    // (`count of v <= target`). A `<=` comparison would incorrectly short-circuit
    // to 0.0 whenever the caller targeted the observed minimum, even though one
    // or more values in the population are equal to it.
    if target < min_val {
      return 0.0;
    }
    let max_val = digest.estimate_quantile(1.0);
    if target >= max_val {
      return 100.0;
    }
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    for _ in 0..60 {
      let mid = (lo + hi) / 2.0;
      let value = digest.estimate_quantile(mid);
      if value <= target {
        lo = mid;
      } else {
        hi = mid;
      }
      if (hi - lo) < 1e-9 {
        break;
      }
    }
    lo * 100.0
  }
}

#[derive(Clone)]
pub struct PercentileState {
  pub quantiles: QuantileState,
  pub percents: Vec<f64>,
}

#[derive(Clone)]
pub struct PercentileRankState {
  pub quantiles: QuantileState,
  pub targets: Vec<f64>,
}

fn numeric_values(
  fast_fields: &FastFieldsReader,
  field: &str,
  doc_id: DocId,
  missing: Option<f64>,
) -> Vec<f64> {
  let mut values = fast_fields.numeric_values(field, doc_id);
  if values.is_empty() {
    if let Some(m) = missing {
      values.push(m);
    }
  }
  values
}

#[derive(Clone)]
pub struct TopHitsState {
  pub size: usize,
  pub from: usize,
  pub total: u64,
  pub(crate) hits: Vec<RankedTopHit>,
}

#[derive(Clone)]
pub(crate) struct RankedTopHit {
  key: SortKey,
  hit: TopHit,
}

impl PartialEq for RankedTopHit {
  fn eq(&self, other: &Self) -> bool {
    self.key == other.key
  }
}

impl Eq for RankedTopHit {}

impl PartialOrd for RankedTopHit {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

impl Ord for RankedTopHit {
  fn cmp(&self, other: &Self) -> Ordering {
    self.key.cmp(&other.key)
  }
}

#[derive(Clone, Copy)]
enum DateInterval {
  Fixed(i64),
  Calendar(CalendarUnit),
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum CalendarUnit {
  Day,
  Week,
  Month,
  Quarter,
  Year,
}

pub(crate) struct SegmentAggregationCollector<'a> {
  aggs: BTreeMap<String, AggregationNode<'a>>,
}

impl<'a> SegmentAggregationCollector<'a> {
  pub(crate) fn new(aggs: BTreeMap<String, AggregationNode<'a>>) -> Self {
    Self { aggs }
  }
}

impl DocCollector for SegmentAggregationCollector<'_> {
  fn collect(&mut self, doc_id: DocId, score: f32) {
    for agg in self.aggs.values_mut() {
      agg.collect(doc_id, score);
    }
  }
}

impl AggregationSegmentCollector for SegmentAggregationCollector<'_> {
  type Output = BTreeMap<String, AggregationIntermediate>;

  fn finish(self) -> Self::Output {
    self
      .aggs
      .into_iter()
      .map(|(name, agg)| (name, agg.finish()))
      .collect()
  }
}

#[derive(Clone, Copy)]
struct NestedCollectScope<'a> {
  path: &'a str,
  object_idx: usize,
}

pub(crate) enum AggregationNode<'a> {
  Terms(Box<TermsCollector<'a>>),
  SignificantTerms(Box<SignificantTermsCollector<'a>>),
  RareTerms(Box<RareTermsCollector<'a>>),
  Range(Box<RangeCollector<'a>>),
  DateRange(Box<DateRangeCollector<'a>>),
  Histogram(Box<HistogramCollector<'a>>),
  DateHistogram(Box<DateHistogramCollector<'a>>),
  Stats(Box<StatsCollector<'a>>),
  ExtendedStats(Box<StatsCollector<'a>>),
  ValueCount(Box<ValueCountCollector<'a>>),
  TopHits(Box<TopHitsCollector<'a>>),
  Cardinality(Box<CardinalityCollector<'a>>),
  Percentiles(Box<PercentilesCollector<'a>>),
  PercentileRanks(Box<PercentileRanksCollector<'a>>),
  Filter(Box<FilterCollector<'a>>),
  Nested(Box<NestedCollector<'a>>),
  Composite(Box<CompositeCollector<'a>>),
}

impl<'a> AggregationNode<'a> {
  pub fn from_request(ctx: AggregationContext<'a>, agg: &Aggregation) -> Self {
    match agg {
      Aggregation::Terms(t) => AggregationNode::Terms(Box::new(TermsCollector::new(ctx, t))),
      Aggregation::SignificantTerms(t) => {
        AggregationNode::SignificantTerms(Box::new(SignificantTermsCollector::new(ctx, t)))
      }
      Aggregation::RareTerms(t) => {
        AggregationNode::RareTerms(Box::new(RareTermsCollector::new(ctx, t)))
      }
      Aggregation::Range(r) => AggregationNode::Range(Box::new(RangeCollector::new(ctx, r))),
      Aggregation::DateRange(r) => {
        AggregationNode::DateRange(Box::new(DateRangeCollector::new(ctx, r)))
      }
      Aggregation::Histogram(h) => {
        AggregationNode::Histogram(Box::new(HistogramCollector::new(ctx, h)))
      }
      Aggregation::DateHistogram(h) => {
        AggregationNode::DateHistogram(Box::new(DateHistogramCollector::new(ctx, h)))
      }
      Aggregation::Stats(m) => AggregationNode::Stats(Box::new(StatsCollector::new(ctx, m))),
      Aggregation::ExtendedStats(m) => {
        AggregationNode::ExtendedStats(Box::new(StatsCollector::new(ctx, m)))
      }
      Aggregation::ValueCount(m) => {
        AggregationNode::ValueCount(Box::new(ValueCountCollector::new(ctx, m)))
      }
      Aggregation::TopHits(t) => AggregationNode::TopHits(Box::new(TopHitsCollector::new(ctx, t))),
      Aggregation::Cardinality(c) => {
        AggregationNode::Cardinality(Box::new(CardinalityCollector::new(ctx, c)))
      }
      Aggregation::Percentiles(p) => {
        AggregationNode::Percentiles(Box::new(PercentilesCollector::new(ctx, p)))
      }
      Aggregation::PercentileRanks(p) => {
        AggregationNode::PercentileRanks(Box::new(PercentileRanksCollector::new(ctx, p)))
      }
      Aggregation::Filter(f) => AggregationNode::Filter(Box::new(FilterCollector::new(ctx, f))),
      Aggregation::Nested(n) => AggregationNode::Nested(Box::new(NestedCollector::new(ctx, n))),
      Aggregation::Composite(c) => {
        AggregationNode::Composite(Box::new(CompositeCollector::new(ctx, c)))
      }
      Aggregation::BucketSort(_)
      | Aggregation::AvgBucket(_)
      | Aggregation::SumBucket(_)
      | Aggregation::Derivative(_)
      | Aggregation::MovingAvg(_)
      | Aggregation::BucketScript(_) => {
        unreachable!("pipeline aggregations are applied during finalize")
      }
    }
  }

  fn collect(&mut self, doc_id: DocId, score: f32) {
    self.collect_scoped(doc_id, score, None);
  }

  fn collect_scoped(&mut self, doc_id: DocId, score: f32, scope: Option<&NestedCollectScope<'_>>) {
    match self {
      AggregationNode::Terms(inner) => inner.collect_scoped(doc_id, score, scope),
      AggregationNode::SignificantTerms(inner) => inner.collect(doc_id, score),
      AggregationNode::RareTerms(inner) => inner.collect(doc_id, score),
      AggregationNode::Range(inner) => inner.collect(doc_id, score),
      AggregationNode::DateRange(inner) => inner.collect(doc_id, score),
      AggregationNode::Histogram(inner) => inner.collect(doc_id, score),
      AggregationNode::DateHistogram(inner) => inner.collect(doc_id, score),
      AggregationNode::Stats(inner) => inner.collect(doc_id, score),
      AggregationNode::ExtendedStats(inner) => inner.collect(doc_id, score),
      AggregationNode::ValueCount(inner) => inner.collect(doc_id, score),
      AggregationNode::TopHits(inner) => inner.collect(doc_id, score),
      AggregationNode::Cardinality(inner) => inner.collect(doc_id, score),
      AggregationNode::Percentiles(inner) => inner.collect(doc_id, score),
      AggregationNode::PercentileRanks(inner) => inner.collect(doc_id, score),
      AggregationNode::Filter(inner) => inner.collect(doc_id, score),
      AggregationNode::Nested(inner) => inner.collect_scoped(doc_id, score, scope),
      AggregationNode::Composite(inner) => inner.collect(doc_id, score),
    }
  }

  fn finish(self) -> AggregationIntermediate {
    match self {
      AggregationNode::Terms(inner) => inner.finish(),
      AggregationNode::SignificantTerms(inner) => inner.finish(),
      AggregationNode::RareTerms(inner) => inner.finish(),
      AggregationNode::Range(inner) => inner.finish(),
      AggregationNode::DateRange(inner) => inner.finish(),
      AggregationNode::Histogram(inner) => inner.finish(),
      AggregationNode::DateHistogram(inner) => inner.finish(),
      AggregationNode::Stats(inner) => AggregationIntermediate::Stats(inner.finish()),
      AggregationNode::ExtendedStats(inner) => {
        AggregationIntermediate::ExtendedStats(inner.finish())
      }
      AggregationNode::ValueCount(inner) => AggregationIntermediate::ValueCount(inner.finish()),
      AggregationNode::TopHits(inner) => AggregationIntermediate::TopHits(inner.finish()),
      AggregationNode::Cardinality(inner) => AggregationIntermediate::Cardinality(inner.finish()),
      AggregationNode::Percentiles(inner) => AggregationIntermediate::Percentiles(inner.finish()),
      AggregationNode::PercentileRanks(inner) => {
        AggregationIntermediate::PercentileRanks(inner.finish())
      }
      AggregationNode::Filter(inner) => inner.finish(),
      AggregationNode::Nested(inner) => inner.finish(),
      AggregationNode::Composite(inner) => inner.finish(),
    }
  }
}

pub(crate) struct TermsCollector<'a> {
  field: String,
  size: Option<usize>,
  shard_size: Option<usize>,
  min_doc_count: u64,
  missing: Option<serde_json::Value>,
  missing_key: Option<String>,
  // Cache the resolved field for the active nested scope to avoid per-hit string allocation.
  scoped_field_cache: Option<(String, String)>,
  buckets: HashMap<BucketKey<'a>, BucketState<'a>>,
  sub_aggs: BTreeMap<String, Aggregation>,
  pipeline_aggs: BTreeMap<String, Aggregation>,
  sampler: Sampler,
  ctx: AggregationContext<'a>,
}

pub(crate) struct BucketState<'a> {
  key: serde_json::Value,
  doc_count: u64,
  aggs: BTreeMap<String, AggregationNode<'a>>,
}

#[derive(Clone)]
enum BucketKey<'a> {
  Borrowed(&'a str),
  Owned(String),
}

impl BucketKey<'_> {
  fn as_str(&self) -> &str {
    match self {
      BucketKey::Borrowed(s) => s,
      BucketKey::Owned(s) => s.as_str(),
    }
  }
}

impl PartialEq for BucketKey<'_> {
  fn eq(&self, other: &Self) -> bool {
    self.as_str() == other.as_str()
  }
}

impl Eq for BucketKey<'_> {}

impl Hash for BucketKey<'_> {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.as_str().hash(state);
  }
}

impl<'a> TermsCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &TermsAggregation) -> Self {
    let min_doc_count = agg.min_doc_count.unwrap_or(1);
    let (sub_aggs, pipeline_aggs) = split_pipeline_aggs(&agg.aggs);
    Self {
      field: agg.field.clone(),
      size: agg.size,
      shard_size: agg.shard_size,
      min_doc_count,
      missing: agg.missing.clone(),
      missing_key: agg.missing.as_ref().map(|v| match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
      }),
      scoped_field_cache: None,
      buckets: HashMap::new(),
      sub_aggs,
      pipeline_aggs,
      sampler: Sampler::new(agg.sampling.as_ref()),
      ctx,
    }
  }

  fn get_bucket<F>(&mut self, bucket_key: BucketKey<'a>, make_value: F) -> &mut BucketState<'a>
  where
    F: FnOnce() -> serde_json::Value,
  {
    match self.buckets.entry(bucket_key) {
      HashEntry::Occupied(entry) => entry.into_mut(),
      HashEntry::Vacant(entry) => entry.insert(BucketState {
        key: make_value(),
        doc_count: 0,
        aggs: build_children(&self.ctx, &self.sub_aggs),
      }),
    }
  }

  fn collect_scoped(&mut self, doc_id: DocId, score: f32, scope: Option<&NestedCollectScope<'_>>) {
    let sampled = if let Some(scope) = scope {
      self
        .sampler
        .accept_object(self.ctx.segment_ord, doc_id, scope.object_idx)
    } else {
      self.sampler.accept(self.ctx.segment_ord, doc_id)
    };
    if !sampled {
      return;
    }
    let values = if let Some(scope) = scope {
      let should_refresh = match self.scoped_field_cache.as_ref() {
        Some((cached_scope, _)) => cached_scope.as_str() != scope.path,
        None => true,
      };
      if should_refresh {
        self.scoped_field_cache = Some((
          scope.path.to_string(),
          resolve_scoped_path(scope.path, &self.field),
        ));
      }
      let scoped_field = self
        .scoped_field_cache
        .as_ref()
        .map(|(_, field)| field.as_str())
        .expect("scoped field cache initialized");
      self
        .ctx
        .fast_fields
        .nested_str_values_at(scoped_field, doc_id, scope.object_idx)
    } else {
      self.ctx.fast_fields.str_values(&self.field, doc_id)
    };
    if !values.is_empty() {
      let mut seen = HashSet::new();
      for val in values.into_iter().filter(|v| seen.insert(*v)) {
        let bucket = self.get_bucket(BucketKey::Borrowed(val), || {
          serde_json::Value::String(val.to_string())
        });
        bucket.doc_count += 1;
        for child in bucket.aggs.values_mut() {
          child.collect_scoped(doc_id, score, scope);
        }
      }
      if !seen.is_empty() {
        return;
      }
    }
    let Some(missing) = self.missing.as_ref() else {
      return;
    };
    let bucket_key = BucketKey::Owned(self.missing_key.clone().unwrap_or_default());
    let bucket = match self.buckets.entry(bucket_key) {
      HashEntry::Occupied(entry) => entry.into_mut(),
      HashEntry::Vacant(entry) => entry.insert(BucketState {
        key: missing.clone(),
        doc_count: 0,
        aggs: build_children(&self.ctx, &self.sub_aggs),
      }),
    };
    bucket.doc_count += 1;
    for child in bucket.aggs.values_mut() {
      child.collect_scoped(doc_id, score, scope);
    }
  }

  fn finish(self) -> AggregationIntermediate {
    let mut buckets: Vec<BucketState<'a>> = self
      .buckets
      .into_values()
      .filter(|b| b.doc_count >= self.min_doc_count)
      .collect();
    buckets.sort_by(|a, b| terms_bucket_cmp(&a.key, a.doc_count, &b.key, b.doc_count));
    let limit = self
      .shard_size
      .or(self.size)
      .unwrap_or(buckets.len())
      .min(MAX_BUCKETS);
    buckets.truncate(limit);
    AggregationIntermediate::Terms {
      buckets: buckets
        .into_iter()
        .map(|b| BucketIntermediate {
          key: b.key,
          doc_count: b.doc_count,
          aggs: finalize_children(b.aggs),
        })
        .collect(),
      size: self.size,
      shard_size: self.shard_size,
      pipeline: self.pipeline_aggs,
      sampled: self.sampler.sampled(),
    }
  }
}

pub(crate) struct RangeCollector<'a> {
  field: String,
  keyed: bool,
  ranges: Vec<RangeEntry<'a>>,
  missing: Option<f64>,
  pipeline_aggs: BTreeMap<String, Aggregation>,
  sampler: Sampler,
  ctx: AggregationContext<'a>,
}

pub(crate) struct RangeEntry<'a> {
  key: Option<String>,
  from: Option<f64>,
  to: Option<f64>,
  bucket: BucketState<'a>,
}

impl<'a> RangeCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &RangeAggregation) -> Self {
    let (sub_aggs, pipeline_aggs) = split_pipeline_aggs(&agg.aggs);
    let ranges = agg
      .ranges
      .iter()
      .map(|r| RangeEntry {
        key: r.key.clone(),
        from: r.from,
        to: r.to,
        bucket: BucketState {
          key: serde_json::Value::Null,
          doc_count: 0,
          aggs: build_children(&ctx, &sub_aggs),
        },
      })
      .collect();
    let missing = agg.missing.as_ref().and_then(|v| {
      v.as_f64()
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
    });
    Self {
      field: agg.field.clone(),
      keyed: agg.keyed,
      ranges,
      missing,
      pipeline_aggs,
      sampler: Sampler::new(agg.sampling.as_ref()),
      ctx,
    }
  }

  fn collect(&mut self, doc_id: DocId, score: f32) {
    if !self.sampler.accept(self.ctx.segment_ord, doc_id) {
      return;
    }
    let values = numeric_values(self.ctx.fast_fields, &self.field, doc_id, self.missing);
    if values.is_empty() {
      return;
    }
    for entry in self.ranges.iter_mut() {
      if values.iter().any(|val| {
        let ge_from = entry.from.map(|f| *val >= f).unwrap_or(true);
        let lt_to = entry.to.map(|t| *val < t).unwrap_or(true);
        ge_from && lt_to
      }) {
        entry.bucket.doc_count += 1;
        for child in entry.bucket.aggs.values_mut() {
          child.collect(doc_id, score);
        }
      }
    }
  }

  fn finish(self) -> AggregationIntermediate {
    let buckets = self
      .ranges
      .into_iter()
      .map(|r| {
        let key = if let Some(key) = r.key {
          serde_json::Value::String(key)
        } else {
          serde_json::json!({"from": r.from, "to": r.to})
        };
        BucketIntermediate {
          key,
          doc_count: r.bucket.doc_count,
          aggs: finalize_children(r.bucket.aggs),
        }
      })
      .collect();
    AggregationIntermediate::Range {
      buckets,
      keyed: self.keyed,
      pipeline: self.pipeline_aggs,
      sampled: self.sampler.sampled(),
    }
  }
}

pub(crate) struct DateRangeCollector<'a> {
  inner: RangeCollector<'a>,
}

impl<'a> DateRangeCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &DateRangeAggregation) -> Self {
    let ranges = agg
      .ranges
      .iter()
      .map(|r| crate::api::types::RangeBound {
        key: r.key.clone(),
        from: r.from.as_deref().and_then(parse_date),
        to: r.to.as_deref().and_then(parse_date),
      })
      .collect();
    let numeric = RangeAggregation {
      field: agg.field.clone(),
      keyed: agg.keyed,
      ranges,
      missing: agg
        .missing
        .as_ref()
        .and_then(|val| match val {
          serde_json::Value::String(s) => parse_date(s),
          serde_json::Value::Number(n) => n.as_f64(),
          _ => None,
        })
        .and_then(|d| serde_json::Number::from_f64(d).map(serde_json::Value::Number))
        .or_else(|| agg.missing.clone()),
      aggs: agg.aggs.clone(),
      sampling: agg.sampling.clone(),
    };
    Self {
      inner: RangeCollector::new(ctx, &numeric),
    }
  }

  fn collect(&mut self, doc_id: DocId, score: f32) {
    self.inner.collect(doc_id, score);
  }

  fn finish(self) -> AggregationIntermediate {
    let keyed = self.inner.keyed;
    match self.inner.finish() {
      AggregationIntermediate::Range {
        buckets,
        pipeline,
        sampled,
        ..
      } => AggregationIntermediate::DateRange {
        keyed,
        buckets,
        pipeline,
        sampled,
      },
      _ => AggregationIntermediate::DateRange {
        keyed,
        buckets: Vec::new(),
        pipeline: BTreeMap::new(),
        sampled: false,
      },
    }
  }
}

pub(crate) struct HistogramCollector<'a> {
  field: String,
  interval: f64,
  offset: f64,
  min_doc_count: u64,
  buckets: HashMap<i64, BucketState<'a>>,
  extended_bounds: Option<(f64, f64)>,
  hard_bounds: Option<(f64, f64)>,
  missing: Option<f64>,
  sub_aggs: BTreeMap<String, Aggregation>,
  pipeline_aggs: BTreeMap<String, Aggregation>,
  sampler: Sampler,
  ctx: AggregationContext<'a>,
}

impl<'a> HistogramCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &HistogramAggregation) -> Self {
    let (sub_aggs, pipeline_aggs) = split_pipeline_aggs(&agg.aggs);
    let offset = agg.offset.unwrap_or(0.0);
    let extended_bounds = agg.extended_bounds.as_ref().map(|b| (b.min, b.max));
    let hard_bounds = agg.hard_bounds.as_ref().map(|b| (b.min, b.max));
    let has_bounds = agg.extended_bounds.is_some() || agg.hard_bounds.is_some();
    Self {
      field: agg.field.clone(),
      interval: agg.interval,
      offset,
      min_doc_count: agg.min_doc_count.unwrap_or(if has_bounds { 0 } else { 1 }),
      buckets: HashMap::new(),
      extended_bounds,
      hard_bounds,
      missing: agg.missing,
      sub_aggs,
      pipeline_aggs,
      sampler: Sampler::new(agg.sampling.as_ref()),
      ctx,
    }
  }

  fn bucket_key(&self, val: f64) -> i64 {
    ((val - self.offset) / self.interval).floor() as i64
  }

  fn collect(&mut self, doc_id: DocId, score: f32) {
    if !self.sampler.accept(self.ctx.segment_ord, doc_id) {
      return;
    }
    let values = numeric_values(self.ctx.fast_fields, &self.field, doc_id, self.missing);
    if values.is_empty() {
      return;
    }
    let mut seen = HashSet::new();
    for val in values {
      let bucket_id = self.bucket_key(val);
      if let Some((min, max)) = self.hard_bounds {
        let bucket_val = bucket_id as f64 * self.interval + self.offset;
        if bucket_val < min || bucket_val >= max {
          continue;
        }
      }
      if !seen.insert(bucket_id) {
        continue;
      }
      let bucket = self
        .buckets
        .entry(bucket_id)
        .or_insert_with(|| BucketState {
          key: serde_json::Value::Number(
            serde_json::Number::from_f64(bucket_id as f64 * self.interval + self.offset)
              .unwrap_or_else(|| serde_json::Number::from(0)),
          ),
          doc_count: 0,
          aggs: build_children(&self.ctx, &self.sub_aggs),
        });
      if bucket.aggs.is_empty() && !self.sub_aggs.is_empty() {
        bucket.aggs = build_children(&self.ctx, &self.sub_aggs);
      }
      bucket.doc_count += 1;
      for child in bucket.aggs.values_mut() {
        child.collect(doc_id, score);
      }
    }
  }

  fn finish(self) -> AggregationIntermediate {
    let interval = self.interval;
    let offset = self.offset;
    let min_doc_count = self.min_doc_count;
    let extended_bounds = self.extended_bounds;
    let hard_bounds = self.hard_bounds;
    let mut buckets = self.buckets;
    let bucket_key = |val: f64| ((val - offset) / interval).floor() as i64;
    let bucket_value = |bucket_id: i64| bucket_id as f64 * interval + offset;
    // Defense-in-depth: the request validator rejects non-finite / non-positive
    // intervals (see `validate_histogram_config`). Skip bounds materialization
    // entirely in the unlikely case we got here with a degenerate interval so
    // that the loop below cannot become unbounded (BUG-027).
    let bounds_materializable = interval.is_finite() && interval > 0.0;
    // Compute the effective fill range as the intersection of `extended_bounds`
    // with `hard_bounds`. `hard_bounds` is an absolute cap on emitted buckets,
    // so any empty buckets materialized from `extended_bounds` must be clipped
    // to stay within it. A plain `extended_bounds.or(hard_bounds)` fallback
    // would emit buckets outside `hard_bounds` whenever both are set (BUG-188).
    // The request validator already forbids `extended_bounds` from exceeding
    // `hard_bounds`, but we intersect here defensively so the collector cannot
    // violate the hard cap even if that validation is ever weakened or bypassed.
    let fill_range = intersect_fill_range_f64(extended_bounds, hard_bounds);
    if bounds_materializable {
      if let Some((min, max)) = fill_range {
        let mut bucket_id = bucket_key(min);
        let end = bucket_key(max);
        let mut materialized: usize = 0;
        while bucket_id <= end {
          buckets.entry(bucket_id).or_insert_with(|| BucketState {
            key: serde_json::Value::Number(
              serde_json::Number::from_f64(bucket_value(bucket_id))
                .unwrap_or_else(|| serde_json::Number::from(0)),
            ),
            doc_count: 0,
            aggs: BTreeMap::new(),
          });
          // Guard against saturating-cast + wrapping addition producing an
          // infinite loop if somehow `end == i64::MAX` (belt-and-braces: the
          // validator caps the span well below this).
          let Some(next) = bucket_id.checked_add(1) else {
            break;
          };
          bucket_id = next;
          materialized = materialized.saturating_add(1);
          if materialized >= MAX_BUCKETS {
            break;
          }
        }
      }
    }
    // BUG-269: the fill loop maps `hard_bounds` values to bucket keys via
    // `floor()`, which can produce keys below `hard_bounds.min` or at
    // `hard_bounds.max`. Drop any bucket whose key-value falls outside the
    // half-open range `[hard_bounds.min, hard_bounds.max)`.
    if let Some((hmin, hmax)) = hard_bounds {
      buckets.retain(|bucket_id, _| {
        let bv = bucket_value(*bucket_id);
        bv >= hmin && bv < hmax
      });
    }
    let mut buckets: Vec<BucketIntermediate> = buckets
      .into_values()
      .filter(|b| b.doc_count >= min_doc_count)
      .map(|b| BucketIntermediate {
        key: b.key,
        doc_count: b.doc_count,
        aggs: finalize_children(b.aggs),
      })
      .collect();
    buckets.sort_by(|a, b| cmp_bucket_value(&a.key, &b.key));
    AggregationIntermediate::Histogram {
      buckets,
      pipeline: self.pipeline_aggs,
      sampled: self.sampler.sampled(),
    }
  }
}

pub(crate) struct DateHistogramCollector<'a> {
  field: String,
  interval: DateInterval,
  offset_millis: i64,
  min_doc_count: u64,
  buckets: HashMap<i64, BucketState<'a>>,
  extended_bounds: Option<(i64, i64)>,
  hard_bounds: Option<(i64, i64)>,
  missing: Option<i64>,
  sub_aggs: BTreeMap<String, Aggregation>,
  pipeline_aggs: BTreeMap<String, Aggregation>,
  sampler: Sampler,
  ctx: AggregationContext<'a>,
}

impl<'a> DateHistogramCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &DateHistogramAggregation) -> Self {
    let (sub_aggs, pipeline_aggs) = split_pipeline_aggs(&agg.aggs);
    let interval = if let Some(cal) = agg
      .calendar_interval
      .as_ref()
      .and_then(|s| parse_calendar_interval(s))
    {
      DateInterval::Calendar(cal)
    } else {
      let millis = agg
        .fixed_interval
        .as_ref()
        .and_then(|s| parse_interval_seconds(s))
        .unwrap_or(86_400.0)
        * 1_000.0;
      DateInterval::Fixed(millis as i64)
    };
    let offset_millis = agg
      .offset
      .as_ref()
      .and_then(|s| parse_interval_seconds(s))
      .map(|s| (s * 1_000.0) as i64)
      .unwrap_or(0);
    let extended_bounds = agg
      .extended_bounds
      .as_ref()
      .and_then(|b| Some((parse_date(&b.min)? as i64, parse_date(&b.max)? as i64)));
    let hard_bounds = agg
      .hard_bounds
      .as_ref()
      .and_then(|b| Some((parse_date(&b.min)? as i64, parse_date(&b.max)? as i64)));
    let missing = agg
      .missing
      .as_ref()
      .and_then(|s| parse_date(s).or_else(|| s.parse::<f64>().ok()))
      .map(|v| v as i64);
    Self {
      field: agg.field.clone(),
      interval,
      offset_millis,
      min_doc_count: agg.min_doc_count.unwrap_or(0),
      buckets: HashMap::new(),
      extended_bounds,
      hard_bounds,
      missing,
      sub_aggs,
      pipeline_aggs,
      sampler: Sampler::new(agg.sampling.as_ref()),
      ctx,
    }
  }

  fn collect(&mut self, doc_id: DocId, score: f32) {
    if !self.sampler.accept(self.ctx.segment_ord, doc_id) {
      return;
    }
    let values: Vec<i64> = numeric_values(
      self.ctx.fast_fields,
      &self.field,
      doc_id,
      self.missing.map(|v| v as f64),
    )
    .into_iter()
    .map(|v| v as i64)
    .collect();
    if values.is_empty() {
      return;
    }
    let mut seen = HashSet::new();
    for val in values {
      let bucket_start = match bucket_start(val, self.offset_millis, &self.interval) {
        Some(v) => v,
        None => continue,
      };
      if let Some((min, max)) = self.hard_bounds {
        if bucket_start < min || bucket_start >= max {
          continue;
        }
      }
      if !seen.insert(bucket_start) {
        continue;
      }
      let bucket_entry = self
        .buckets
        .entry(bucket_start)
        .or_insert_with(|| BucketState {
          key: serde_json::Value::Number(serde_json::Number::from(bucket_start)),
          doc_count: 0,
          aggs: build_children(&self.ctx, &self.sub_aggs),
        });
      if bucket_entry.aggs.is_empty() && !self.sub_aggs.is_empty() {
        bucket_entry.aggs = build_children(&self.ctx, &self.sub_aggs);
      }
      bucket_entry.doc_count += 1;
      for child in bucket_entry.aggs.values_mut() {
        child.collect(doc_id, score);
      }
    }
  }

  fn finish(self) -> AggregationIntermediate {
    let mut buckets = self.buckets;
    // Intersect `extended_bounds` with `hard_bounds` before materializing empty
    // buckets so the hard cap is honored even when both are specified. The plain
    // `.or()` fallback would emit buckets outside `hard_bounds` whenever
    // `extended_bounds` was present (BUG-188). See the matching
    // `HistogramCollector::finish` comment for the full rationale.
    let fill_range = intersect_fill_range_i64(self.extended_bounds, self.hard_bounds);
    if let Some((min, max)) = fill_range {
      if let (Some(mut start), Some(mut end)) = (
        bucket_start(min, self.offset_millis, &self.interval),
        bucket_start(max, self.offset_millis, &self.interval),
      ) {
        if start > end {
          std::mem::swap(&mut start, &mut end);
        }
        let mut current = start;
        // Defense-in-depth: cap materialization at `MAX_BUCKETS` so a small
        // `fixed_interval` combined with a wide `extended_bounds` span cannot
        // push the process into an unbounded `HashMap` insert loop even if the
        // request validator is bypassed or weakened (BUG-200). Matches the
        // runtime cap already present in `HistogramCollector::finish`.
        let mut materialized: usize = 0;
        while current <= end {
          buckets.entry(current).or_insert_with(|| BucketState {
            key: serde_json::Value::Number(serde_json::Number::from(current)),
            doc_count: 0,
            aggs: BTreeMap::new(),
          });
          materialized = materialized.saturating_add(1);
          if materialized >= MAX_BUCKETS {
            break;
          }
          current = match add_interval(current, &self.interval) {
            Some(next) => next,
            None => break,
          };
        }
      }
    }
    // BUG-269: the fill loop maps `hard_bounds` timestamps to bucket starts
    // via `bucket_start()`, which can produce a bucket start below
    // `hard_bounds.min` or at `hard_bounds.max`. Drop any bucket whose start
    // falls outside the half-open range `[hard_bounds.min, hard_bounds.max)`.
    if let Some((hmin, hmax)) = self.hard_bounds {
      buckets.retain(|&bucket_start, _| bucket_start >= hmin && bucket_start < hmax);
    }
    let mut buckets: Vec<BucketIntermediate> = buckets
      .into_values()
      .filter(|b| b.doc_count >= self.min_doc_count)
      .map(|b| BucketIntermediate {
        key: b.key,
        doc_count: b.doc_count,
        aggs: finalize_children(b.aggs),
      })
      .collect();
    buckets.sort_by(|a, b| cmp_bucket_value(&a.key, &b.key));
    AggregationIntermediate::DateHistogram {
      buckets,
      pipeline: self.pipeline_aggs,
      sampled: self.sampler.sampled(),
    }
  }
}

pub(crate) struct StatsCollector<'a> {
  field: String,
  missing: Option<f64>,
  stats: StatsState,
  ctx: AggregationContext<'a>,
}

impl<'a> StatsCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &crate::api::types::MetricAggregation) -> Self {
    Self {
      field: agg.field.clone(),
      missing: agg.missing.as_ref().and_then(|v| {
        v.as_f64()
          .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
      }),
      stats: StatsState::default(),
      ctx,
    }
  }

  fn collect(&mut self, doc_id: DocId, _score: f32) {
    // Aggregate over every value; multi-valued fields contribute each entry (bucket doc_count
    // remains per-document).
    for val in numeric_values(self.ctx.fast_fields, &self.field, doc_id, self.missing) {
      self.stats = merge_stats(
        self.stats,
        StatsState {
          count: 1,
          min: val,
          max: val,
          sum: val,
          m2: 0.0,
        },
      );
    }
  }

  fn finish(self) -> StatsState {
    self.stats
  }
}

pub(crate) struct ValueCountCollector<'a> {
  field: String,
  missing: Option<f64>,
  state: ValueCountState,
  ctx: AggregationContext<'a>,
}

impl<'a> ValueCountCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &crate::api::types::MetricAggregation) -> Self {
    Self {
      field: agg.field.clone(),
      missing: agg.missing.as_ref().and_then(|v| {
        v.as_f64()
          .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
      }),
      state: ValueCountState::default(),
      ctx,
    }
  }

  fn collect(&mut self, doc_id: DocId, _score: f32) {
    let values = numeric_values(self.ctx.fast_fields, &self.field, doc_id, self.missing);
    self.state.value += values.len() as u64;
  }

  fn finish(self) -> ValueCountState {
    self.state
  }
}

pub(crate) struct CardinalityCollector<'a> {
  field: String,
  missing: Option<serde_json::Value>,
  kind: crate::index::manifest::FieldKind,
  numeric_i64: bool,
  state: CardinalityState,
  ctx: AggregationContext<'a>,
}

impl<'a> CardinalityCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &crate::api::types::CardinalityAggregation) -> Self {
    let meta = ctx.schema.field_meta(&agg.field);
    let kind = meta
      .as_ref()
      .map(|m| m.kind.clone())
      .unwrap_or(crate::index::manifest::FieldKind::Unknown);
    let numeric_i64 = meta.and_then(|m| m.numeric_i64).unwrap_or(false);
    Self {
      field: agg.field.clone(),
      missing: agg.missing.clone(),
      kind,
      numeric_i64,
      state: CardinalityState {
        values: HashSet::new(),
        precision_threshold: agg.precision_threshold,
      },
      ctx,
    }
  }

  fn collect(&mut self, doc_id: DocId, _score: f32) {
    match self.kind {
      crate::index::manifest::FieldKind::Keyword => {
        let mut values: Vec<String> = self
          .ctx
          .fast_fields
          .str_values(&self.field, doc_id)
          .iter()
          .map(|s| s.to_string())
          .collect();
        if values.is_empty() {
          if let Some(missing) = self.missing.as_ref().and_then(|v| v.as_str()) {
            values.push(missing.to_string());
          }
        }
        for v in values {
          self.state.values.insert(hash_cardinality(&v));
        }
      }
      crate::index::manifest::FieldKind::Numeric => {
        if self.numeric_i64 {
          let mut values = self.ctx.fast_fields.i64_values(&self.field, doc_id);
          if values.is_empty() {
            if let Some(m) = self.missing.as_ref().and_then(|v| v.as_i64()) {
              values.push(m);
            }
          }
          for v in values {
            self.state.values.insert(hash_cardinality(&v));
          }
        } else {
          let mut values = self.ctx.fast_fields.f64_values(&self.field, doc_id);
          if values.is_empty() {
            if let Some(m) = self.missing.as_ref().and_then(|v| {
              v.as_f64()
                .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
            }) {
              values.push(m);
            }
          }
          for v in values {
            self.state.values.insert(hash_cardinality(&v.to_bits()));
          }
        }
      }
      _ => {}
    }
  }

  fn finish(self) -> CardinalityState {
    self.state
  }
}

pub(crate) struct PercentilesCollector<'a> {
  field: String,
  missing: Option<f64>,
  quantiles: QuantileState,
  percents: Vec<f64>,
  ctx: AggregationContext<'a>,
}

impl<'a> PercentilesCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &crate::api::types::PercentilesAggregation) -> Self {
    let percents = agg
      .percents
      .clone()
      .unwrap_or_else(default_percentiles_list);
    Self {
      field: agg.field.clone(),
      missing: agg.missing.as_ref().and_then(|v| {
        v.as_f64()
          .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
      }),
      quantiles: QuantileState::default(),
      percents,
      ctx,
    }
  }

  fn collect(&mut self, doc_id: DocId, _score: f32) {
    let vals = numeric_values(self.ctx.fast_fields, &self.field, doc_id, self.missing);
    for v in vals {
      self.quantiles.push(v);
    }
  }

  fn finish(self) -> PercentileState {
    PercentileState {
      quantiles: self.quantiles,
      percents: self.percents,
    }
  }
}

pub(crate) struct PercentileRanksCollector<'a> {
  field: String,
  missing: Option<f64>,
  quantiles: QuantileState,
  targets: Vec<f64>,
  ctx: AggregationContext<'a>,
}

impl<'a> PercentileRanksCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &crate::api::types::PercentileRanksAggregation) -> Self {
    Self {
      field: agg.field.clone(),
      missing: agg.missing.as_ref().and_then(|v| {
        v.as_f64()
          .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
      }),
      quantiles: QuantileState::default(),
      targets: agg.values.clone(),
      ctx,
    }
  }

  fn collect(&mut self, doc_id: DocId, _score: f32) {
    let vals = numeric_values(self.ctx.fast_fields, &self.field, doc_id, self.missing);
    for v in vals {
      self.quantiles.push(v);
    }
  }

  fn finish(self) -> PercentileRankState {
    PercentileRankState {
      quantiles: self.quantiles,
      targets: self.targets,
    }
  }
}

pub(crate) struct FilterCollector<'a> {
  filter: Filter,
  bucket: BucketState<'a>,
  pipeline_aggs: BTreeMap<String, Aggregation>,
  sampler: Sampler,
  ctx: AggregationContext<'a>,
}

impl<'a> FilterCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &FilterAggregation) -> Self {
    let (sub_aggs, pipeline_aggs) = split_pipeline_aggs(&agg.aggs);
    Self {
      filter: agg.filter.clone(),
      bucket: BucketState {
        key: serde_json::Value::Null,
        doc_count: 0,
        aggs: build_children(&ctx, &sub_aggs),
      },
      pipeline_aggs,
      sampler: Sampler::new(agg.sampling.as_ref()),
      ctx,
    }
  }

  fn collect(&mut self, doc_id: DocId, score: f32) {
    if !self.sampler.accept(self.ctx.segment_ord, doc_id) {
      return;
    }
    if passes_filter(self.ctx.fast_fields, doc_id, &self.filter) {
      self.bucket.doc_count += 1;
      for child in self.bucket.aggs.values_mut() {
        child.collect(doc_id, score);
      }
    }
  }

  fn finish(self) -> AggregationIntermediate {
    AggregationIntermediate::Filter {
      bucket: BucketIntermediate {
        key: serde_json::Value::Null,
        doc_count: self.bucket.doc_count,
        aggs: finalize_children(self.bucket.aggs),
      },
      pipeline: self.pipeline_aggs,
      sampled: self.sampler.sampled(),
    }
  }
}

pub(crate) struct NestedCollector<'a> {
  path: String,
  bucket: BucketState<'a>,
  pipeline_aggs: BTreeMap<String, Aggregation>,
  sampler: Sampler,
  ctx: AggregationContext<'a>,
}

impl<'a> NestedCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &NestedAggregation) -> Self {
    let (sub_aggs, pipeline_aggs) = split_pipeline_aggs(&agg.aggs);
    Self {
      path: agg.path.clone(),
      bucket: BucketState {
        key: serde_json::Value::Null,
        doc_count: 0,
        aggs: build_children(&ctx, &sub_aggs),
      },
      pipeline_aggs,
      sampler: Sampler::new(agg.sampling.as_ref()),
      ctx,
    }
  }

  fn collect_scoped(
    &mut self,
    doc_id: DocId,
    score: f32,
    parent_scope: Option<&NestedCollectScope<'_>>,
  ) {
    let resolved_path = if let Some(parent_scope) = parent_scope {
      resolve_scoped_path(parent_scope.path, &self.path)
    } else {
      self.path.clone()
    };
    let object_count = self
      .ctx
      .fast_fields
      .nested_object_count(&resolved_path, doc_id);
    if object_count == 0 {
      return;
    }
    let parents = parent_scope.map(|_| self.ctx.fast_fields.nested_parents(&resolved_path, doc_id));
    for object_idx in 0..object_count {
      if let (Some(scope), Some(parents)) = (parent_scope, parents.as_ref()) {
        if parents.get(object_idx).and_then(|p| *p) != Some(scope.object_idx) {
          continue;
        }
      }
      if !self
        .sampler
        .accept_object(self.ctx.segment_ord, doc_id, object_idx)
      {
        continue;
      }
      self.bucket.doc_count += 1;
      let scope = NestedCollectScope {
        path: &resolved_path,
        object_idx,
      };
      for child in self.bucket.aggs.values_mut() {
        child.collect_scoped(doc_id, score, Some(&scope));
      }
    }
  }

  fn finish(self) -> AggregationIntermediate {
    AggregationIntermediate::Nested {
      bucket: BucketIntermediate {
        key: serde_json::Value::Null,
        doc_count: self.bucket.doc_count,
        aggs: finalize_children(self.bucket.aggs),
      },
      pipeline: self.pipeline_aggs,
      sampled: self.sampler.sampled(),
    }
  }
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct CompositeKey {
  parts: Vec<CompositeKeyPart>,
}

impl Ord for CompositeKey {
  fn cmp(&self, other: &Self) -> Ordering {
    for (a, b) in self.parts.iter().zip(other.parts.iter()) {
      let ord = a.cmp(b);
      if !ord.is_eq() {
        return ord;
      }
    }
    self.parts.len().cmp(&other.parts.len())
  }
}

impl PartialOrd for CompositeKey {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

#[derive(Clone, Hash, PartialEq, Eq)]
enum CompositeKeyPart {
  Str(String),
  F64(u64),
}

impl CompositeKeyPart {
  fn cmp(&self, other: &Self) -> Ordering {
    match (self, other) {
      (CompositeKeyPart::Str(a), CompositeKeyPart::Str(b)) => a.cmp(b),
      (CompositeKeyPart::F64(a), CompositeKeyPart::F64(b)) => {
        f64::from_bits(*a).total_cmp(&f64::from_bits(*b))
      }
      (CompositeKeyPart::Str(_), CompositeKeyPart::F64(_)) => Ordering::Less,
      (CompositeKeyPart::F64(_), CompositeKeyPart::Str(_)) => Ordering::Greater,
    }
  }

  fn to_json(&self) -> serde_json::Value {
    match self {
      CompositeKeyPart::Str(s) => serde_json::Value::String(s.clone()),
      CompositeKeyPart::F64(bits) => serde_json::Number::from_f64(f64::from_bits(*bits))
        .map(serde_json::Value::Number)
        .unwrap_or(serde_json::Value::Null),
    }
  }
}

pub(crate) struct CompositeCollector<'a> {
  sources: Vec<CompositeSource>,
  size: usize,
  after: Option<serde_json::Value>,
  buckets: HashMap<CompositeKey, BucketState<'a>>,
  sub_aggs: BTreeMap<String, Aggregation>,
  pipeline_aggs: BTreeMap<String, Aggregation>,
  sampler: Sampler,
  ctx: AggregationContext<'a>,
}

impl<'a> CompositeCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &CompositeAggregation) -> Self {
    let (sub_aggs, pipeline_aggs) = split_pipeline_aggs(&agg.aggs);
    Self {
      sources: agg.sources.clone(),
      size: agg.size,
      after: agg.after.clone(),
      buckets: HashMap::new(),
      sub_aggs,
      pipeline_aggs,
      sampler: Sampler::new(agg.sampling.as_ref()),
      ctx,
    }
  }

  fn collect(&mut self, doc_id: DocId, score: f32) {
    if !self.sampler.accept(self.ctx.segment_ord, doc_id) {
      return;
    }
    let mut per_source_values: Vec<Vec<CompositeKeyPart>> = Vec::with_capacity(self.sources.len());
    for source in self.sources.iter() {
      let values = match source {
        CompositeSource::Terms { field, .. } => self
          .ctx
          .fast_fields
          .str_values(field, doc_id)
          .into_iter()
          .map(|s| CompositeKeyPart::Str(s.to_string()))
          .collect::<Vec<_>>(),
        CompositeSource::Histogram {
          field, interval, ..
        } => self
          .ctx
          .fast_fields
          .f64_values(field, doc_id)
          .into_iter()
          .map(|v| {
            let bucket = (v / interval).floor() * interval;
            CompositeKeyPart::F64(bucket.to_bits())
          })
          .collect::<Vec<_>>(),
      };
      if values.is_empty() {
        return;
      }
      per_source_values.push(values);
    }
    let mut combos: Vec<CompositeKey> = Vec::new();
    build_composite_keys(&per_source_values, 0, &mut Vec::new(), &mut combos);
    let mut seen = HashSet::new();
    for key in combos.into_iter() {
      if !seen.insert(key.clone()) {
        continue;
      }
      let bucket = self
        .buckets
        .entry(key.clone())
        .or_insert_with(|| BucketState {
          key: composite_key_to_json(&key, &self.sources),
          doc_count: 0,
          aggs: build_children(&self.ctx, &self.sub_aggs),
        });
      if bucket.aggs.is_empty() && !self.sub_aggs.is_empty() {
        bucket.aggs = build_children(&self.ctx, &self.sub_aggs);
      }
      bucket.doc_count += 1;
      for child in bucket.aggs.values_mut() {
        child.collect(doc_id, score);
      }
    }
  }

  fn finish(self) -> AggregationIntermediate {
    let buckets: Vec<BucketIntermediate> = self
      .buckets
      .into_values()
      .map(|state| BucketIntermediate {
        key: state.key,
        doc_count: state.doc_count,
        aggs: finalize_children(state.aggs),
      })
      .collect();
    AggregationIntermediate::Composite {
      buckets,
      size: self.size,
      after: self.after,
      pipeline: self.pipeline_aggs,
      sources: self.sources,
      sampled: self.sampler.sampled(),
    }
  }
}
#[derive(Clone, Debug)]
struct RankedDoc {
  key: SortKey,
  score: f32,
  doc_id: DocId,
}

impl PartialEq for RankedDoc {
  fn eq(&self, other: &Self) -> bool {
    self.key == other.key
  }
}

impl Eq for RankedDoc {}

impl PartialOrd for RankedDoc {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

impl Ord for RankedDoc {
  fn cmp(&self, other: &Self) -> Ordering {
    self.key.cmp(&other.key)
  }
}

pub(crate) struct TopHitsCollector<'a> {
  size: usize,
  from: usize,
  limit: usize,
  heap: BinaryHeap<RankedDoc>,
  total: u64,
  fields: Option<Vec<String>>,
  highlight_field: Option<String>,
  highlight_terms: &'a [String],
  plan: SortPlan,
  segment_ord: u32,
  ctx: AggregationContext<'a>,
}

impl<'a> TopHitsCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &TopHitsAggregation) -> Self {
    let plan = SortPlan::from_request(ctx.schema, &agg.sort)
      .expect("top_hits sort validated during request planning");
    // Defense-in-depth: clamp `size` and `from` to `MAX_TOP_HITS` so an internal caller that
    // bypasses `validate_aggregations_in_scope` cannot drive an unbounded `BinaryHeap` here
    // (BUG-222). The request validator rejects values past the cap up-front; the `min` calls are
    // a hard ceiling on the per-segment heap. We persist the bounded values into the struct so
    // every downstream use (heap sizing, `finish`'s `start + size` arithmetic, the
    // `Vec::with_capacity` for hits) stays within the cap and consistent with `limit`. `.max(1)`
    // on `limit` preserves the legacy invariant that the collector retains the best hit even
    // when callers ask for `size = 0` so the merge step has a candidate to pick from.
    let bounded_size = agg.size.min(MAX_TOP_HITS);
    let bounded_from = agg.from.min(MAX_TOP_HITS);
    let limit = bounded_size
      .saturating_add(bounded_from)
      .clamp(1, MAX_TOP_HITS);
    Self {
      size: bounded_size,
      from: bounded_from,
      limit,
      heap: BinaryHeap::new(),
      total: 0,
      fields: agg.fields.clone(),
      highlight_field: agg.highlight_field.clone(),
      highlight_terms: ctx.highlight_terms,
      plan,
      segment_ord: ctx.segment_ord,
      ctx,
    }
  }

  fn collect(&mut self, doc_id: DocId, score: f32) {
    self.total += 1;
    let key = self
      .plan
      .build_key(self.ctx.segment, doc_id, score, self.segment_ord);
    let ranked = RankedDoc { key, score, doc_id };
    if self.heap.len() < self.limit {
      self.heap.push(ranked);
      return;
    }
    if let Some(worst) = self.heap.peek() {
      if ranked < *worst {
        self.heap.pop();
        self.heap.push(ranked);
      }
    }
  }

  fn finish(mut self) -> TopHitsState {
    let mut ranked: Vec<RankedDoc> = self.heap.drain().collect();
    ranked.sort_by(|a, b| a.key.cmp(&b.key));
    // Keep all top `(from + size)` ranked items for this segment; the final
    // `from` skip is applied once globally after segments are merged in
    // `finalize_response`. Applying the skip here would discard items whose
    // segment-local rank is `< from` but whose global rank is within the
    // requested `[from, from + size)` page — see BUG-215 for details.
    let mut hits = Vec::with_capacity(ranked.len());
    let need_doc = self.fields.is_some() || self.highlight_field.is_some();
    for doc in ranked.into_iter() {
      let fetched = if need_doc {
        self.ctx.segment.get_doc(doc.doc_id).ok()
      } else {
        None
      };
      let doc_id_str = self
        .ctx
        .segment
        .doc_id(doc.doc_id)
        .unwrap_or("")
        .to_string();
      let fields_val = fetched.as_ref().and_then(|d| {
        if let Some(sel) = &self.fields {
          let obj = d.as_object()?;
          let mut out = serde_json::Map::new();
          for key in sel {
            if let Some(v) = obj.get(key) {
              out.insert(key.clone(), v.clone());
            }
          }
          Some(serde_json::Value::Object(out))
        } else {
          Some(d.clone())
        }
      });
      let snippet = if let (Some(field), Some(doc_val)) = (&self.highlight_field, fetched.as_ref())
      {
        if let Some(text) = doc_val.get(field).and_then(|v| v.as_str()) {
          make_snippet(text, self.highlight_terms, &[])
        } else {
          None
        }
      } else {
        None
      };
      hits.push(RankedTopHit {
        key: doc.key,
        hit: TopHit {
          doc_id: doc_id_str,
          score: Some(doc.score),
          fields: fields_val,
          snippet,
        },
      });
    }
    TopHitsState {
      size: self.size,
      from: self.from,
      total: self.total,
      hits,
    }
  }
}

fn build_children<'a>(
  ctx: &AggregationContext<'a>,
  defs: &BTreeMap<String, Aggregation>,
) -> BTreeMap<String, AggregationNode<'a>> {
  defs
    .iter()
    .map(|(name, agg)| {
      (
        name.clone(),
        AggregationNode::from_request(ctx.clone(), agg),
      )
    })
    .collect()
}

fn finalize_children(
  aggs: BTreeMap<String, AggregationNode>,
) -> BTreeMap<String, AggregationIntermediate> {
  aggs.into_iter().map(|(k, v)| (k, v.finish())).collect()
}

fn split_pipeline_aggs(
  defs: &BTreeMap<String, Aggregation>,
) -> (BTreeMap<String, Aggregation>, BTreeMap<String, Aggregation>) {
  let mut bucket_aggs = BTreeMap::new();
  let mut pipeline_aggs = BTreeMap::new();
  for (name, agg) in defs.iter() {
    match agg {
      Aggregation::BucketSort(_)
      | Aggregation::AvgBucket(_)
      | Aggregation::SumBucket(_)
      | Aggregation::Derivative(_)
      | Aggregation::MovingAvg(_)
      | Aggregation::BucketScript(_) => {
        pipeline_aggs.insert(name.clone(), agg.clone());
      }
      _ => {
        bucket_aggs.insert(name.clone(), agg.clone());
      }
    }
  }
  (bucket_aggs, pipeline_aggs)
}

fn merge_stats(a: StatsState, b: StatsState) -> StatsState {
  if a.count == 0 {
    return b;
  }
  if b.count == 0 {
    return a;
  }
  let delta = b.sum / b.count as f64 - a.sum / a.count as f64;
  let count = a.count + b.count;
  let sum = a.sum + b.sum;
  let min = a.min.min(b.min);
  let max = a.max.max(b.max);
  let m2 = a.m2 + b.m2 + delta * delta * (a.count as f64 * b.count as f64 / count as f64);
  StatsState {
    count,
    min,
    max,
    sum,
    m2,
  }
}

pub fn merge_aggregation_results(
  results: Vec<BTreeMap<String, AggregationIntermediate>>,
) -> BTreeMap<String, AggregationResponse> {
  let mut merged: BTreeMap<String, AggregationIntermediate> = BTreeMap::new();
  for map in results.into_iter() {
    for (name, agg) in map.into_iter() {
      match merged.entry(name) {
        BTreeEntry::Vacant(entry) => {
          entry.insert(agg);
        }
        BTreeEntry::Occupied(mut entry) => merge_intermediate_in_place(entry.get_mut(), agg),
      }
    }
  }
  merged
    .into_iter()
    .map(|(name, agg)| (name, finalize_response(agg)))
    .collect()
}

fn merge_intermediate_in_place(
  target: &mut AggregationIntermediate,
  incoming: AggregationIntermediate,
) {
  match (target, incoming) {
    (
      AggregationIntermediate::Terms {
        buckets: target_buckets,
        size,
        shard_size,
        pipeline: target_pipeline,
        sampled: target_sampled,
      },
      AggregationIntermediate::Terms {
        buckets: incoming_buckets,
        size: incoming_size,
        shard_size: incoming_shard,
        pipeline: incoming_pipeline,
        sampled: incoming_sampled,
      },
    ) => {
      merge_bucket_lists(target_buckets, incoming_buckets);
      if size.is_none() {
        *size = incoming_size;
      }
      if shard_size.is_none() {
        *shard_size = incoming_shard;
      }
      if target_pipeline.is_empty() {
        *target_pipeline = incoming_pipeline;
      }
      *target_sampled |= incoming_sampled;
      let limit = shard_size
        .unwrap_or_else(|| target_buckets.len())
        .min(MAX_BUCKETS);
      target_buckets.sort_by(|a, b| terms_bucket_cmp(&a.key, a.doc_count, &b.key, b.doc_count));
      if target_buckets.len() > limit {
        target_buckets.truncate(limit);
      }
    }
    (
      AggregationIntermediate::SignificantTerms {
        buckets: target_buckets,
        size: target_size,
        min_doc_count: target_min,
        pipeline: target_pipeline,
        doc_count: target_doc_count,
        bg_count: target_bg_count,
        sampled: target_sampled,
      },
      AggregationIntermediate::SignificantTerms {
        buckets: incoming_buckets,
        size: incoming_size,
        min_doc_count: incoming_min,
        pipeline: incoming_pipeline,
        doc_count: incoming_doc_count,
        bg_count: incoming_bg_count,
        sampled: incoming_sampled,
      },
    ) => {
      merge_significant_bucket_lists(target_buckets, incoming_buckets);
      if target_size.is_none() {
        *target_size = incoming_size;
      }
      *target_min = (*target_min).min(incoming_min);
      *target_doc_count = target_doc_count.saturating_add(incoming_doc_count);
      *target_bg_count = target_bg_count.saturating_add(incoming_bg_count);
      if target_pipeline.is_empty() {
        *target_pipeline = incoming_pipeline;
      }
      *target_sampled |= incoming_sampled;
      let limit = target_size
        .unwrap_or_else(|| target_buckets.len())
        .min(MAX_BUCKETS);
      // Sort by significance score proxy (doc_count/bg_count) to preserve
      // high-significance low-frequency terms during truncation. Buckets with
      // bg_count == 0 are treated as score 0 to match finalize_response.
      // Compare ratios via integer cross-multiplication to avoid float rounding.
      target_buckets.sort_by(|a, b| {
        let score_cmp = match (a.bg_count == 0, b.bg_count == 0) {
          (true, true) => Ordering::Equal,
          (true, false) => Ordering::Greater,
          (false, true) => Ordering::Less,
          (false, false) => {
            let left = (a.doc_count as u128) * (b.bg_count as u128);
            let right = (b.doc_count as u128) * (a.bg_count as u128);
            right.cmp(&left)
          }
        };
        score_cmp.then_with(|| terms_bucket_cmp(&a.key, a.doc_count, &b.key, b.doc_count))
      });
      if target_buckets.len() > limit {
        target_buckets.truncate(limit);
      }
    }
    (
      AggregationIntermediate::RareTerms {
        buckets: target_buckets,
        size: target_size,
        max_doc_count: target_max,
        pipeline: target_pipeline,
        sampled: target_sampled,
      },
      AggregationIntermediate::RareTerms {
        buckets: incoming_buckets,
        size: incoming_size,
        max_doc_count: incoming_max,
        pipeline: incoming_pipeline,
        sampled: incoming_sampled,
      },
    ) => {
      merge_bucket_lists(target_buckets, incoming_buckets);
      if target_size.is_none() {
        *target_size = incoming_size;
      }
      *target_max = (*target_max).min(incoming_max);
      target_buckets.retain(|b| b.doc_count > 0 && b.doc_count <= *target_max);
      if target_pipeline.is_empty() {
        *target_pipeline = incoming_pipeline;
      }
      *target_sampled |= incoming_sampled;
      target_buckets
        .sort_by(|a, b| rare_terms_bucket_cmp(&a.key, a.doc_count, &b.key, b.doc_count));
      let limit = target_size
        .unwrap_or_else(|| target_buckets.len())
        .min(MAX_BUCKETS);
      if target_buckets.len() > limit {
        target_buckets.truncate(limit);
      }
    }
    (
      AggregationIntermediate::Range {
        buckets: target_buckets,
        pipeline: target_pipeline,
        keyed: _,
        sampled: target_sampled,
      },
      AggregationIntermediate::Range {
        buckets: incoming_buckets,
        pipeline: incoming_pipeline,
        keyed: _,
        sampled: incoming_sampled,
      },
    ) => {
      merge_bucket_lists(target_buckets, incoming_buckets);
      if target_pipeline.is_empty() {
        *target_pipeline = incoming_pipeline;
      }
      *target_sampled |= incoming_sampled;
    }
    (
      AggregationIntermediate::DateRange {
        buckets: target_buckets,
        pipeline: target_pipeline,
        keyed: _,
        sampled: target_sampled,
      },
      AggregationIntermediate::DateRange {
        buckets: incoming_buckets,
        pipeline: incoming_pipeline,
        keyed: _,
        sampled: incoming_sampled,
      },
    ) => {
      merge_bucket_lists(target_buckets, incoming_buckets);
      if target_pipeline.is_empty() {
        *target_pipeline = incoming_pipeline;
      }
      *target_sampled |= incoming_sampled;
    }
    (
      AggregationIntermediate::Histogram {
        buckets: target_buckets,
        pipeline: target_pipeline,
        sampled: target_sampled,
      },
      AggregationIntermediate::Histogram {
        buckets: incoming_buckets,
        pipeline: incoming_pipeline,
        sampled: incoming_sampled,
      },
    ) => {
      merge_bucket_lists(target_buckets, incoming_buckets);
      if target_pipeline.is_empty() {
        *target_pipeline = incoming_pipeline;
      }
      *target_sampled |= incoming_sampled;
    }
    (
      AggregationIntermediate::DateHistogram {
        buckets: target_buckets,
        pipeline: target_pipeline,
        sampled: target_sampled,
      },
      AggregationIntermediate::DateHistogram {
        buckets: incoming_buckets,
        pipeline: incoming_pipeline,
        sampled: incoming_sampled,
      },
    ) => {
      merge_bucket_lists(target_buckets, incoming_buckets);
      if target_pipeline.is_empty() {
        *target_pipeline = incoming_pipeline;
      }
      *target_sampled |= incoming_sampled;
    }
    (
      AggregationIntermediate::Stats(target_stats),
      AggregationIntermediate::Stats(incoming_stats),
    ) => {
      *target_stats = merge_stats(*target_stats, incoming_stats);
    }
    (
      AggregationIntermediate::ExtendedStats(target_stats),
      AggregationIntermediate::ExtendedStats(incoming_stats),
    ) => {
      *target_stats = merge_stats(*target_stats, incoming_stats);
    }
    (
      AggregationIntermediate::ValueCount(target_val),
      AggregationIntermediate::ValueCount(incoming_val),
    ) => {
      target_val.value += incoming_val.value;
    }
    (
      AggregationIntermediate::Cardinality(target_state),
      AggregationIntermediate::Cardinality(incoming_state),
    ) => {
      target_state.values.extend(incoming_state.values);
      if target_state.precision_threshold.is_none() {
        target_state.precision_threshold = incoming_state.precision_threshold;
      }
    }
    (
      AggregationIntermediate::Percentiles(target_state),
      AggregationIntermediate::Percentiles(incoming_state),
    ) => {
      target_state.quantiles.merge(incoming_state.quantiles);
      if target_state.percents.is_empty() {
        target_state.percents = incoming_state.percents;
      }
    }
    (
      AggregationIntermediate::PercentileRanks(target_state),
      AggregationIntermediate::PercentileRanks(incoming_state),
    ) => {
      target_state.quantiles.merge(incoming_state.quantiles);
      if target_state.targets.is_empty() {
        target_state.targets = incoming_state.targets;
      }
    }
    (
      AggregationIntermediate::TopHits(target_hits),
      AggregationIntermediate::TopHits(incoming_hits),
    ) => merge_top_hits(target_hits, incoming_hits),
    (
      AggregationIntermediate::Filter {
        bucket: target_bucket,
        pipeline: target_pipeline,
        sampled: target_sampled,
      },
      AggregationIntermediate::Filter {
        bucket: incoming_bucket,
        pipeline: incoming_pipeline,
        sampled: incoming_sampled,
      },
    ) => {
      target_bucket.doc_count += incoming_bucket.doc_count;
      for (name, agg) in incoming_bucket.aggs.into_iter() {
        match target_bucket.aggs.entry(name) {
          BTreeEntry::Vacant(entry) => {
            entry.insert(agg);
          }
          BTreeEntry::Occupied(mut entry) => {
            merge_intermediate_in_place(entry.get_mut(), agg);
          }
        }
      }
      if target_pipeline.is_empty() {
        *target_pipeline = incoming_pipeline;
      }
      *target_sampled |= incoming_sampled;
    }
    (
      AggregationIntermediate::Nested {
        bucket: target_bucket,
        pipeline: target_pipeline,
        sampled: target_sampled,
      },
      AggregationIntermediate::Nested {
        bucket: incoming_bucket,
        pipeline: incoming_pipeline,
        sampled: incoming_sampled,
      },
    ) => {
      target_bucket.doc_count += incoming_bucket.doc_count;
      for (name, agg) in incoming_bucket.aggs.into_iter() {
        match target_bucket.aggs.entry(name) {
          BTreeEntry::Vacant(entry) => {
            entry.insert(agg);
          }
          BTreeEntry::Occupied(mut entry) => {
            merge_intermediate_in_place(entry.get_mut(), agg);
          }
        }
      }
      if target_pipeline.is_empty() {
        *target_pipeline = incoming_pipeline;
      }
      *target_sampled |= incoming_sampled;
    }
    (
      AggregationIntermediate::Composite {
        buckets: target_buckets,
        size: target_size,
        after: target_after,
        pipeline: target_pipeline,
        sources: _,
        sampled: target_sampled,
      },
      AggregationIntermediate::Composite {
        buckets: incoming_buckets,
        size: incoming_size,
        after: incoming_after,
        pipeline: incoming_pipeline,
        sources: _,
        sampled: incoming_sampled,
      },
    ) => {
      merge_bucket_lists(target_buckets, incoming_buckets);
      *target_size = (*target_size).max(incoming_size);
      if target_after.is_none() {
        *target_after = incoming_after;
      }
      if target_pipeline.is_empty() {
        *target_pipeline = incoming_pipeline;
      }
      *target_sampled |= incoming_sampled;
    }
    _ => {}
  }
}

fn merge_bucket_lists(target: &mut Vec<BucketIntermediate>, incoming: Vec<BucketIntermediate>) {
  let mut index: HashMap<String, usize> = HashMap::with_capacity(target.len());
  for (idx, bucket) in target.iter().enumerate() {
    index.insert(bucket_key_string(&bucket.key), idx);
  }
  for bucket in incoming.into_iter() {
    let key = bucket_key_string(&bucket.key);
    if let Some(&idx) = index.get(&key) {
      let existing = &mut target[idx];
      existing.doc_count += bucket.doc_count;
      for (name, agg) in bucket.aggs.into_iter() {
        match existing.aggs.entry(name) {
          BTreeEntry::Vacant(entry) => {
            entry.insert(agg);
          }
          BTreeEntry::Occupied(mut entry) => {
            merge_intermediate_in_place(entry.get_mut(), agg);
          }
        }
      }
    } else {
      index.insert(key, target.len());
      target.push(bucket);
    }
  }
}

fn merge_significant_bucket_lists(
  target: &mut Vec<SignificantBucketIntermediate>,
  incoming: Vec<SignificantBucketIntermediate>,
) {
  let mut index: HashMap<String, usize> = HashMap::with_capacity(target.len());
  for (idx, bucket) in target.iter().enumerate() {
    index.insert(bucket_key_string(&bucket.key), idx);
  }
  for bucket in incoming.into_iter() {
    let key = bucket_key_string(&bucket.key);
    if let Some(&idx) = index.get(&key) {
      let existing = &mut target[idx];
      existing.doc_count += bucket.doc_count;
      existing.bg_count += bucket.bg_count;
      for (name, agg) in bucket.aggs.into_iter() {
        match existing.aggs.entry(name) {
          BTreeEntry::Vacant(entry) => {
            entry.insert(agg);
          }
          BTreeEntry::Occupied(mut entry) => {
            merge_intermediate_in_place(entry.get_mut(), agg);
          }
        }
      }
    } else {
      index.insert(key, target.len());
      target.push(bucket);
    }
  }
}

fn merge_top_hits(target: &mut TopHitsState, incoming: TopHitsState) {
  let limit = target
    .size
    .saturating_add(target.from)
    .max(target.size)
    .max(1);
  target.total += incoming.total;
  let total_hits = target.hits.len().saturating_add(incoming.hits.len());
  let min_capacity = target.size.max(1);
  let cap = limit.min(total_hits.max(min_capacity)).saturating_add(1);
  let mut heap: BinaryHeap<RankedTopHit> = BinaryHeap::with_capacity(cap);
  let mut push_hit = |hit: RankedTopHit| {
    if heap.len() < limit {
      heap.push(hit);
      return;
    }
    if let Some(worst) = heap.peek() {
      if hit < *worst {
        heap.pop();
        heap.push(hit);
      }
    }
  };
  for hit in target.hits.drain(..) {
    push_hit(hit);
  }
  for hit in incoming.hits {
    push_hit(hit);
  }
  let mut hits: Vec<_> = heap.into_iter().collect();
  hits.sort_by(|a, b| a.key.cmp(&b.key));
  // Keep the full merged top `(from + size)` window; the `from` skip is
  // applied once in `finalize_response` so that per-segment items at ranks
  // `[0, from)` are not discarded before the merge can compare them against
  // other segments (BUG-215).
  target.hits = hits;
}

fn bucket_key_string(key: &serde_json::Value) -> String {
  if let Some(s) = key.as_str() {
    s.to_string()
  } else {
    key.to_string()
  }
}

fn terms_bucket_cmp(
  a_key: &serde_json::Value,
  a_count: u64,
  b_key: &serde_json::Value,
  b_count: u64,
) -> Ordering {
  b_count
    .cmp(&a_count)
    .then_with(|| bucket_key_string(a_key).cmp(&bucket_key_string(b_key)))
}

fn rare_terms_bucket_cmp(
  a_key: &serde_json::Value,
  a_count: u64,
  b_key: &serde_json::Value,
  b_count: u64,
) -> Ordering {
  a_count
    .cmp(&b_count)
    .then_with(|| bucket_key_string(a_key).cmp(&bucket_key_string(b_key)))
}

fn finalize_response(intermediate: AggregationIntermediate) -> AggregationResponse {
  match intermediate {
    AggregationIntermediate::Terms {
      mut buckets,
      size,
      shard_size,
      pipeline,
      sampled,
    } => {
      buckets.sort_by(|a, b| terms_bucket_cmp(&a.key, a.doc_count, &b.key, b.doc_count));
      let limit = size
        .unwrap_or(shard_size.unwrap_or(buckets.len()))
        .min(MAX_BUCKETS);
      if buckets.len() > limit {
        buckets.truncate(limit);
      }
      let mut buckets: Vec<BucketResponse> = buckets.into_iter().map(finalize_bucket).collect();
      let aggregations = apply_pipeline_aggs(&pipeline, &mut buckets);
      AggregationResponse::Terms {
        buckets,
        aggregations,
        sampled,
      }
    }
    AggregationIntermediate::SignificantTerms {
      buckets,
      size,
      min_doc_count: _,
      pipeline,
      doc_count,
      bg_count,
      sampled,
    } => {
      let mut sig_buckets: Vec<SignificantBucketResponse> = buckets
        .into_iter()
        .map(|b| {
          let score = if doc_count > 0 && bg_count > 0 && b.bg_count > 0 {
            (b.doc_count as f64 / doc_count as f64) / (b.bg_count as f64 / bg_count as f64)
          } else {
            0.0
          };
          SignificantBucketResponse {
            key: b.key,
            doc_count: b.doc_count,
            bg_count: b.bg_count,
            score,
            aggregations: b
              .aggs
              .into_iter()
              .map(|(name, agg)| (name, finalize_response(agg)))
              .collect(),
          }
        })
        .collect();
      sig_buckets.sort_by(|a, b| {
        b.score
          .partial_cmp(&a.score)
          .unwrap_or(Ordering::Equal)
          .then_with(|| terms_bucket_cmp(&a.key, a.doc_count, &b.key, b.doc_count))
      });
      let limit = size.unwrap_or(sig_buckets.len()).min(MAX_BUCKETS);
      if sig_buckets.len() > limit {
        sig_buckets.truncate(limit);
      }
      let mut temp_buckets: Vec<BucketResponse> = sig_buckets
        .iter_mut()
        .map(|b| BucketResponse {
          key: b.key.clone(),
          doc_count: b.doc_count,
          aggregations: std::mem::take(&mut b.aggregations),
        })
        .collect();
      let aggregations = apply_pipeline_aggs(&pipeline, &mut temp_buckets);
      for (sig_bucket, bucket) in sig_buckets.iter_mut().zip(temp_buckets.into_iter()) {
        sig_bucket.aggregations = bucket.aggregations;
      }
      let buckets = sig_buckets;
      AggregationResponse::SignificantTerms {
        buckets,
        aggregations,
        doc_count,
        bg_count,
        sampled,
      }
    }
    AggregationIntermediate::RareTerms {
      mut buckets,
      size,
      max_doc_count: _,
      pipeline,
      sampled,
    } => {
      buckets.sort_by(|a, b| rare_terms_bucket_cmp(&a.key, a.doc_count, &b.key, b.doc_count));
      let limit = size.unwrap_or(buckets.len()).min(MAX_BUCKETS);
      if buckets.len() > limit {
        buckets.truncate(limit);
      }
      let mut buckets: Vec<BucketResponse> = buckets.into_iter().map(finalize_bucket).collect();
      let aggregations = apply_pipeline_aggs(&pipeline, &mut buckets);
      AggregationResponse::RareTerms {
        buckets,
        aggregations,
        sampled,
      }
    }
    AggregationIntermediate::Range {
      buckets,
      keyed,
      pipeline,
      sampled,
    } => {
      let mut buckets: Vec<BucketResponse> = buckets.into_iter().map(finalize_bucket).collect();
      let aggregations = apply_pipeline_aggs(&pipeline, &mut buckets);
      AggregationResponse::Range {
        buckets,
        keyed,
        aggregations,
        sampled,
      }
    }
    AggregationIntermediate::DateRange {
      buckets,
      keyed,
      pipeline,
      sampled,
    } => {
      let mut buckets: Vec<BucketResponse> = buckets.into_iter().map(finalize_bucket).collect();
      let aggregations = apply_pipeline_aggs(&pipeline, &mut buckets);
      AggregationResponse::DateRange {
        buckets,
        keyed,
        aggregations,
        sampled,
      }
    }
    AggregationIntermediate::Histogram {
      buckets,
      pipeline,
      sampled,
    } => {
      let mut buckets: Vec<BucketResponse> = buckets.into_iter().map(finalize_bucket).collect();
      buckets.sort_by(|a, b| cmp_bucket_value(&a.key, &b.key));
      let aggregations = apply_pipeline_aggs(&pipeline, &mut buckets);
      AggregationResponse::Histogram {
        buckets,
        aggregations,
        sampled,
      }
    }
    AggregationIntermediate::DateHistogram {
      buckets,
      pipeline,
      sampled,
    } => {
      let mut buckets: Vec<BucketResponse> = buckets.into_iter().map(finalize_bucket).collect();
      buckets.sort_by(|a, b| cmp_bucket_value(&a.key, &b.key));
      let aggregations = apply_pipeline_aggs(&pipeline, &mut buckets);
      AggregationResponse::DateHistogram {
        buckets,
        aggregations,
        sampled,
      }
    }
    AggregationIntermediate::Stats(stats) => AggregationResponse::Stats(StatsResponse {
      count: stats.count,
      min: stats.min,
      max: stats.max,
      sum: stats.sum,
      avg: if stats.count > 0 {
        stats.sum / stats.count as f64
      } else {
        0.0
      },
    }),
    AggregationIntermediate::ExtendedStats(stats) => {
      let variance = if stats.count > 0 {
        stats.m2 / stats.count as f64
      } else {
        0.0
      };
      AggregationResponse::ExtendedStats(crate::api::types::ExtendedStatsResponse {
        count: stats.count,
        min: stats.min,
        max: stats.max,
        sum: stats.sum,
        avg: if stats.count > 0 {
          stats.sum / stats.count as f64
        } else {
          0.0
        },
        variance,
        std_deviation: variance.sqrt(),
      })
    }
    AggregationIntermediate::ValueCount(val) => {
      AggregationResponse::ValueCount(ValueCountResponse { value: val.value })
    }
    AggregationIntermediate::Cardinality(state) => {
      AggregationResponse::Cardinality(CardinalityResponse {
        value: state.values.len() as u64,
      })
    }
    AggregationIntermediate::Percentiles(state) => {
      AggregationResponse::Percentiles(PercentilesResponse {
        values: compute_percentiles_from_state(state),
      })
    }
    AggregationIntermediate::PercentileRanks(state) => {
      AggregationResponse::PercentileRanks(PercentileRanksResponse {
        values: compute_percentile_ranks_from_state(state),
      })
    }
    AggregationIntermediate::TopHits(state) => {
      // `state.hits` holds the top `(from + size)` merged hits; apply the
      // final `from` skip and truncate to `size` to produce the response
      // page. Doing the skip here (instead of per-segment) ensures items at
      // segment-local ranks `[0, from)` can still win the global `[from,
      // from + size)` window after cross-segment merging.
      let start = state.from.min(state.hits.len());
      AggregationResponse::TopHits(TopHitsResponse {
        total: state.total,
        hits: state
          .hits
          .into_iter()
          .skip(start)
          .take(state.size)
          .map(|h| h.hit)
          .collect(),
      })
    }
    AggregationIntermediate::Filter {
      bucket,
      pipeline,
      sampled,
    } => {
      let mut bucket_resp = finalize_bucket(bucket);
      let mut bucket_list = vec![bucket_resp.clone()];
      let mut aggregations = apply_pipeline_aggs(&pipeline, &mut bucket_list);
      if let Some(mut b) = bucket_list.pop() {
        for (name, agg) in std::mem::take(&mut b.aggregations) {
          aggregations.insert(name, agg);
        }
        bucket_resp = b;
      }
      AggregationResponse::Filter {
        doc_count: bucket_resp.doc_count,
        aggregations,
        sampled,
      }
    }
    AggregationIntermediate::Nested {
      bucket,
      pipeline,
      sampled,
    } => {
      let mut bucket_list = vec![finalize_bucket(bucket)];
      let mut aggregations = apply_pipeline_aggs(&pipeline, &mut bucket_list);
      let mut bucket_resp = bucket_list.pop().expect("nested bucket response");
      for (name, agg) in std::mem::take(&mut bucket_resp.aggregations) {
        aggregations.insert(name, agg);
      }
      AggregationResponse::Nested {
        doc_count: bucket_resp.doc_count,
        aggregations,
        sampled,
      }
    }
    AggregationIntermediate::Composite {
      buckets,
      size,
      after,
      pipeline,
      sources,
      sampled,
    } => finalize_composite(buckets, size, after, pipeline, sources, sampled),
  }
}

fn finalize_bucket(bucket: BucketIntermediate) -> BucketResponse {
  BucketResponse {
    key: bucket.key,
    doc_count: bucket.doc_count,
    aggregations: bucket
      .aggs
      .into_iter()
      .map(|(name, agg)| (name, finalize_response(agg)))
      .collect(),
  }
}

fn pipeline_agg_dependencies<'a>(
  agg: &'a Aggregation,
  pipeline_keys: &HashSet<&str>,
) -> Vec<&'a str> {
  let paths: Vec<&str> = match agg {
    Aggregation::AvgBucket(cfg) => vec![&cfg.buckets_path],
    Aggregation::SumBucket(cfg) => vec![&cfg.buckets_path],
    Aggregation::Derivative(cfg) => vec![&cfg.buckets_path],
    Aggregation::MovingAvg(cfg) => vec![&cfg.buckets_path],
    Aggregation::BucketScript(cfg) => cfg.buckets_path.values().map(|s| s.as_str()).collect(),
    _ => vec![],
  };
  paths
    .into_iter()
    .filter_map(|path| {
      let agg_name = path.split('.').next().unwrap_or(path);
      if pipeline_keys.contains(agg_name) {
        Some(agg_name)
      } else {
        None
      }
    })
    .collect()
}

fn topological_sort_pipeline(pipeline: &BTreeMap<String, Aggregation>) -> Vec<&str> {
  let pipeline_keys: HashSet<&str> = pipeline
    .iter()
    .filter(|(_, agg)| {
      matches!(
        agg,
        Aggregation::Derivative(_) | Aggregation::MovingAvg(_) | Aggregation::BucketScript(_)
      )
    })
    .map(|(k, _)| k.as_str())
    .collect();
  let mut in_degree: BTreeMap<&str, usize> = BTreeMap::new();
  let mut dependents: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
  for key in pipeline.keys() {
    in_degree.entry(key.as_str()).or_insert(0);
  }
  for (name, agg) in pipeline.iter() {
    if matches!(agg, Aggregation::BucketSort(_)) {
      continue;
    }
    let deps = pipeline_agg_dependencies(agg, &pipeline_keys);
    *in_degree.entry(name.as_str()).or_insert(0) += deps.len();
    for dep in deps {
      dependents.entry(dep).or_default().push(name.as_str());
    }
  }
  let mut queue: VecDeque<&str> = in_degree
    .iter()
    .filter(|(_, &deg)| deg == 0)
    .map(|(&k, _)| k)
    .collect();
  let mut order = Vec::with_capacity(pipeline.len());
  while let Some(node) = queue.pop_front() {
    order.push(node);
    if let Some(deps) = dependents.get(node) {
      for &dep in deps {
        if let Some(deg) = in_degree.get_mut(dep) {
          *deg -= 1;
          if *deg == 0 {
            queue.push_back(dep);
          }
        }
      }
    }
  }
  // Append cycle members in BTreeMap key order so they still execute
  // (with unresolved deps) rather than being silently dropped.
  if order.len() < pipeline.len() {
    let in_order: HashSet<&str> = order.iter().copied().collect();
    for key in pipeline.keys() {
      if !in_order.contains(key.as_str()) {
        order.push(key.as_str());
      }
    }
  }
  order
}

fn apply_pipeline_aggs(
  pipeline: &BTreeMap<String, Aggregation>,
  buckets: &mut Vec<BucketResponse>,
) -> BTreeMap<String, AggregationResponse> {
  let mut responses = BTreeMap::new();
  let order = topological_sort_pipeline(pipeline);
  for name in &order {
    let agg = match pipeline.get(*name) {
      Some(a) => a,
      None => continue,
    };
    match agg {
      Aggregation::AvgBucket(cfg) => {
        let mut sum = 0.0_f64;
        let mut count = 0usize;
        for bucket in buckets.iter() {
          if let Some(val) = bucket_metric_value(bucket, &cfg.buckets_path) {
            sum += val;
            count += 1;
          }
        }
        let value = if count > 0 { sum / count as f64 } else { 0.0 };
        responses.insert(
          name.to_string(),
          AggregationResponse::AvgBucket(BucketMetricResponse { value }),
        );
      }
      Aggregation::SumBucket(cfg) => {
        let mut sum = 0.0_f64;
        for bucket in buckets.iter() {
          if let Some(val) = bucket_metric_value(bucket, &cfg.buckets_path) {
            sum += val;
          }
        }
        responses.insert(
          name.to_string(),
          AggregationResponse::SumBucket(BucketMetricResponse { value: sum }),
        );
      }
      Aggregation::Derivative(cfg) => {
        apply_derivative_pipeline(name, cfg, buckets, &mut responses);
      }
      Aggregation::MovingAvg(cfg) => {
        apply_moving_avg_pipeline(name, cfg, buckets, &mut responses);
      }
      Aggregation::BucketScript(cfg) => {
        apply_bucket_script_pipeline(name, cfg, buckets, &mut responses);
      }
      Aggregation::BucketSort(_) => {}
      _ => {}
    }
  }
  for (name, agg) in pipeline
    .iter()
    .filter(|(_, a)| matches!(a, Aggregation::BucketSort(_)))
  {
    if let Aggregation::BucketSort(cfg) = agg {
      bucket_sort_buckets(buckets, cfg);
      responses.insert(
        name.clone(),
        AggregationResponse::BucketSort {
          from: cfg.from.unwrap_or(0),
          size: cfg.size,
        },
      );
    }
  }
  responses
}

fn bucket_metric_series(buckets: &[BucketResponse], path: &str) -> Vec<Option<f64>> {
  buckets
    .iter()
    .map(|bucket| bucket_metric_value(bucket, path))
    .collect()
}

fn apply_derivative_pipeline(
  name: &str,
  cfg: &DerivativeAggregation,
  buckets: &mut [BucketResponse],
  responses: &mut BTreeMap<String, AggregationResponse>,
) {
  let series = bucket_metric_series(buckets, &cfg.buckets_path);
  let policy = cfg.gap_policy.unwrap_or(GapPolicy::Skip);
  let unit = cfg.unit.unwrap_or(1.0).max(f64::EPSILON);
  let mut prev: Option<f64> = None;
  for (idx, bucket) in buckets.iter_mut().enumerate() {
    let current = match (series.get(idx).and_then(|v| *v), policy) {
      (Some(v), _) => Some(v),
      (None, GapPolicy::InsertZeros) => Some(0.0),
      (None, GapPolicy::Skip) => None,
    };
    let value = match (current, prev) {
      (Some(cur), Some(prev_val)) => Some((cur - prev_val) / unit),
      _ => None,
    };
    if let Some(cur) = current {
      prev = Some(cur);
    }
    bucket.aggregations.insert(
      name.to_string(),
      AggregationResponse::Derivative(OptionalBucketMetricResponse { value }),
    );
  }
  responses.insert(
    name.to_string(),
    AggregationResponse::Derivative(OptionalBucketMetricResponse { value: None }),
  );
}

fn apply_moving_avg_pipeline(
  name: &str,
  cfg: &MovingAvgAggregation,
  buckets: &mut [BucketResponse],
  responses: &mut BTreeMap<String, AggregationResponse>,
) {
  let series = bucket_metric_series(buckets, &cfg.buckets_path);
  let policy = cfg.gap_policy.unwrap_or(GapPolicy::Skip);
  let mut window_values: VecDeque<f64> = VecDeque::new();
  // The request validator rejects `window = 0` (BUG-221) so this is the documented
  // precondition; assert it loudly in dev/test builds. The `.max(1)` survives as a
  // production safety net so an internal caller that bypasses validation still gets
  // a windowed average rather than the deque growing unboundedly to `buckets.len()`.
  debug_assert!(
    cfg.window >= 1,
    "moving_avg window must be >= 1; got {}",
    cfg.window
  );
  let window = cfg.window.max(1);
  let mut avgs = Vec::with_capacity(buckets.len());
  for (idx, bucket) in buckets.iter_mut().enumerate() {
    let current = match (series.get(idx).and_then(|v| *v), policy) {
      (Some(v), _) => Some(v),
      (None, GapPolicy::InsertZeros) => Some(0.0),
      (None, GapPolicy::Skip) => None,
    };
    if let Some(val) = current {
      if window_values.len() == window {
        window_values.pop_front();
      }
      window_values.push_back(val);
    }
    let avg = if window_values.is_empty() {
      None
    } else {
      Some(window_values.iter().copied().sum::<f64>() / window_values.len() as f64)
    };
    avgs.push(avg);
    bucket.aggregations.insert(
      name.to_string(),
      AggregationResponse::MovingAvg(MovingAvgResponse {
        value: avg,
        predictions: Vec::new(),
      }),
    );
  }
  let mut predictions = Vec::new();
  if let Some(predict) = cfg.predict {
    if let Some(last_avg) = avgs.last().and_then(|v| *v) {
      // Defense-in-depth: clamp `predict` to `MAX_PREDICTIONS` so an internal caller
      // that bypasses `validate_aggregations_in_scope` cannot drive an unbounded
      // allocation here (BUG-221). The request validator rejects values past the cap
      // up-front; this `min` is just a hard ceiling on the materialization step.
      let predict = predict.min(MAX_PREDICTIONS);
      // Simple forecast that repeats the last observed average to avoid feedback loops.
      predictions = vec![last_avg; predict];
    }
  }
  responses.insert(
    name.to_string(),
    AggregationResponse::MovingAvg(MovingAvgResponse {
      value: avgs.last().and_then(|v| *v),
      predictions,
    }),
  );
}

fn apply_bucket_script_pipeline(
  name: &str,
  cfg: &BucketScriptAggregation,
  buckets: &mut [BucketResponse],
  responses: &mut BTreeMap<String, AggregationResponse>,
) {
  let mut last_value: Option<f64> = None;
  for bucket in buckets.iter_mut() {
    let mut vars = BTreeMap::new();
    let mut missing = false;
    for (var, path) in cfg.buckets_path.iter() {
      if let Some(val) = bucket_metric_value(bucket, path) {
        vars.insert(var.clone(), val);
      } else {
        missing = true;
        break;
      }
    }
    let value = if missing {
      None
    } else {
      eval_bucket_script(&cfg.script, &vars)
    };
    if value.is_some() {
      last_value = value;
    }
    bucket.aggregations.insert(
      name.to_string(),
      AggregationResponse::BucketScript(OptionalBucketMetricResponse { value }),
    );
  }
  responses.insert(
    name.to_string(),
    AggregationResponse::BucketScript(OptionalBucketMetricResponse { value: last_value }),
  );
}

#[derive(Debug, Clone)]
enum ScriptToken {
  Number(f64),
  Var(String),
  Op(char),
  Neg,
  LParen,
  RParen,
}

fn op_precedence(op: char) -> u8 {
  match op {
    '+' | '-' => 1,
    '*' | '/' => 2,
    '~' => 3,
    _ => 0,
  }
}

fn eval_bucket_script(script: &str, vars: &BTreeMap<String, f64>) -> Option<f64> {
  let tokens = tokenize_script(script)?;
  let rpn = to_rpn(tokens)?;
  eval_rpn(rpn, vars)
}

fn tokenize_script(script: &str) -> Option<Vec<ScriptToken>> {
  let mut chars = script.chars().peekable();
  let mut tokens = Vec::new();
  let mut expect_unary = true;
  while let Some(ch) = chars.peek().copied() {
    if ch.is_whitespace() {
      chars.next();
      continue;
    }
    let looks_numeric = ch.is_ascii_digit()
      || ch == '.'
      || (expect_unary
        && ch == '-'
        && chars
          .clone()
          .nth(1)
          .map(|next| next.is_ascii_digit() || next == '.')
          .unwrap_or(false));
    if looks_numeric {
      let mut num = String::new();
      if ch == '-' {
        num.push('-');
        chars.next();
      }
      while let Some(next) = chars.peek() {
        if next.is_ascii_digit() || *next == '.' {
          num.push(*next);
          chars.next();
        } else {
          break;
        }
      }
      if num == "-" || num == "." || num == "-." {
        return None;
      }
      let value: f64 = num.parse().ok()?;
      tokens.push(ScriptToken::Number(value));
      expect_unary = false;
      continue;
    }
    if ch.is_ascii_alphabetic() || ch == '_' {
      let mut name = String::new();
      while let Some(next) = chars.peek() {
        if next.is_ascii_alphanumeric() || *next == '_' {
          name.push(*next);
          chars.next();
        } else {
          break;
        }
      }
      tokens.push(ScriptToken::Var(name));
      expect_unary = false;
      continue;
    }
    match ch {
      '+' | '-' | '*' | '/' => {
        if ch == '-' && expect_unary {
          tokens.push(ScriptToken::Neg);
          chars.next();
          continue;
        }
        tokens.push(ScriptToken::Op(ch));
        chars.next();
        expect_unary = true;
      }
      '(' => {
        tokens.push(ScriptToken::LParen);
        chars.next();
        expect_unary = true;
      }
      ')' => {
        tokens.push(ScriptToken::RParen);
        chars.next();
        expect_unary = false;
      }
      _ => return None,
    }
  }
  Some(tokens)
}

fn pop_op(op: char) -> ScriptToken {
  if op == '~' {
    ScriptToken::Neg
  } else {
    ScriptToken::Op(op)
  }
}

fn to_rpn(tokens: Vec<ScriptToken>) -> Option<Vec<ScriptToken>> {
  let mut output = Vec::new();
  let mut ops: Vec<char> = Vec::new();
  for token in tokens.into_iter() {
    match token {
      ScriptToken::Number(_) | ScriptToken::Var(_) => output.push(token),
      ScriptToken::Neg => {
        ops.push('~');
      }
      ScriptToken::Op(op) => {
        while let Some(&top) = ops.last() {
          if top == '(' {
            break;
          }
          if op_precedence(top) >= op_precedence(op) {
            output.push(pop_op(ops.pop().unwrap()));
          } else {
            break;
          }
        }
        ops.push(op);
      }
      ScriptToken::LParen => ops.push('('),
      ScriptToken::RParen => {
        let mut found_lparen = false;
        while let Some(op) = ops.pop() {
          if op == '(' {
            found_lparen = true;
            break;
          }
          output.push(pop_op(op));
        }
        if !found_lparen {
          return None;
        }
      }
    }
  }
  while let Some(op) = ops.pop() {
    if op == '(' {
      return None;
    }
    output.push(pop_op(op));
  }
  Some(output)
}

fn eval_rpn(tokens: Vec<ScriptToken>, vars: &BTreeMap<String, f64>) -> Option<f64> {
  let mut stack: Vec<f64> = Vec::new();
  for token in tokens.into_iter() {
    match token {
      ScriptToken::Number(v) => stack.push(v),
      ScriptToken::Var(name) => stack.push(*vars.get(&name)?),
      ScriptToken::Neg => {
        let a = stack.pop()?;
        stack.push(-a);
      }
      ScriptToken::Op(op) => {
        let b = stack.pop()?;
        let a = stack.pop()?;
        let result = match op {
          '+' => a + b,
          '-' => a - b,
          '*' => a * b,
          '/' => {
            if b.abs() < 1e-12 {
              return None;
            }
            a / b
          }
          _ => return None,
        };
        stack.push(result);
      }
      ScriptToken::LParen | ScriptToken::RParen => {}
    }
  }
  if stack.len() == 1 {
    stack.pop()
  } else {
    None
  }
}

fn bucket_sort_buckets(buckets: &mut Vec<BucketResponse>, cfg: &BucketSortAggregation) {
  buckets.sort_by(|a, b| bucket_sort_cmp(a, b, &cfg.sort));
  let from = cfg.from.unwrap_or(0);
  if from > 0 {
    buckets.drain(0..from.min(buckets.len()));
  }
  if let Some(size) = cfg.size {
    if buckets.len() > size {
      buckets.truncate(size);
    }
  }
}

#[derive(Clone)]
enum BucketSortComparable {
  Missing,
  F64(f64),
  Str(String),
}

fn bucket_sort_cmp(a: &BucketResponse, b: &BucketResponse, specs: &[BucketSortSpec]) -> Ordering {
  for spec in specs.iter() {
    let a_val = bucket_sort_value(a, spec);
    let b_val = bucket_sort_value(b, spec);
    let ord = compare_sort_values(&a_val, &b_val, spec.order);
    if !ord.is_eq() {
      return ord;
    }
  }
  bucket_key_string(&a.key).cmp(&bucket_key_string(&b.key))
}

fn bucket_sort_value(bucket: &BucketResponse, spec: &BucketSortSpec) -> BucketSortComparable {
  match spec.field.as_str() {
    "_count" => BucketSortComparable::F64(bucket.doc_count as f64),
    "key" | "_key" => BucketSortComparable::Str(bucket_key_string(&bucket.key)),
    path => bucket_metric_value(bucket, path)
      .map(BucketSortComparable::F64)
      .unwrap_or(BucketSortComparable::Missing),
  }
}

fn compare_sort_values(
  a: &BucketSortComparable,
  b: &BucketSortComparable,
  order: SortOrder,
) -> Ordering {
  match (a, b) {
    (BucketSortComparable::Missing, BucketSortComparable::Missing) => Ordering::Equal,
    (BucketSortComparable::Missing, _) => Ordering::Greater,
    (_, BucketSortComparable::Missing) => Ordering::Less,
    (BucketSortComparable::F64(va), BucketSortComparable::F64(vb)) => match order {
      SortOrder::Asc => va.total_cmp(vb),
      SortOrder::Desc => vb.total_cmp(va),
    },
    (BucketSortComparable::Str(sa), BucketSortComparable::Str(sb)) => match order {
      SortOrder::Asc => sa.cmp(sb),
      SortOrder::Desc => sb.cmp(sa),
    },
    (BucketSortComparable::F64(_), BucketSortComparable::Str(_)) => Ordering::Less,
    (BucketSortComparable::Str(_), BucketSortComparable::F64(_)) => Ordering::Greater,
  }
}

fn bucket_metric_value(bucket: &BucketResponse, path: &str) -> Option<f64> {
  if path == "_count" {
    return Some(bucket.doc_count as f64);
  }
  let (agg_name, sub_path) = match path.split_once('.') {
    Some((name, rest)) => (name, Some(rest)),
    None => (path, None),
  };
  let agg = bucket.aggregations.get(agg_name)?;
  extract_metric_from_response(agg, sub_path)
}

fn extract_metric_from_response(resp: &AggregationResponse, path: Option<&str>) -> Option<f64> {
  match resp {
    AggregationResponse::Stats(stats) => {
      let field = path.unwrap_or("avg");
      match field {
        "avg" => Some(stats.avg),
        "min" => Some(stats.min),
        "max" => Some(stats.max),
        "sum" => Some(stats.sum),
        "count" => Some(stats.count as f64),
        _ => None,
      }
    }
    AggregationResponse::ExtendedStats(stats) => {
      let field = path.unwrap_or("avg");
      match field {
        "avg" => Some(stats.avg),
        "min" => Some(stats.min),
        "max" => Some(stats.max),
        "sum" => Some(stats.sum),
        "count" => Some(stats.count as f64),
        "variance" => Some(stats.variance),
        "std_deviation" => Some(stats.std_deviation),
        _ => None,
      }
    }
    AggregationResponse::ValueCount(val) => Some(val.value as f64),
    AggregationResponse::Cardinality(val) => Some(val.value as f64),
    AggregationResponse::Percentiles(vals) => {
      let key = path?;
      vals.values.get(key).copied()
    }
    AggregationResponse::PercentileRanks(vals) => {
      let key = path?;
      vals.values.get(key).copied()
    }
    AggregationResponse::AvgBucket(val) | AggregationResponse::SumBucket(val) => Some(val.value),
    AggregationResponse::Derivative(val) => val.value,
    AggregationResponse::MovingAvg(val) => val.value,
    AggregationResponse::BucketScript(val) => val.value,
    _ => None,
  }
}

fn cmp_bucket_value(a: &serde_json::Value, b: &serde_json::Value) -> Ordering {
  if let (Some(va), Some(vb)) = (a.as_f64(), b.as_f64()) {
    return va.partial_cmp(&vb).unwrap_or(Ordering::Equal);
  }
  a.to_string().cmp(&b.to_string())
}

fn compute_percentiles_from_state(mut state: PercentileState) -> BTreeMap<String, f64> {
  let mut out = BTreeMap::new();
  for p in state.percents.iter() {
    out.insert(format!("{p}"), state.quantiles.percentile(*p));
  }
  out
}

fn compute_percentile_ranks_from_state(mut state: PercentileRankState) -> BTreeMap<String, f64> {
  let mut out = BTreeMap::new();
  for target in state.targets.iter() {
    out.insert(
      format!("{target}"),
      state.quantiles.percentile_rank(*target),
    );
  }
  out
}

fn finalize_composite(
  buckets: Vec<BucketIntermediate>,
  size: usize,
  after: Option<serde_json::Value>,
  pipeline: BTreeMap<String, Aggregation>,
  sources: Vec<CompositeSource>,
  sampled: bool,
) -> AggregationResponse {
  let mut buckets: Vec<BucketResponse> = buckets.into_iter().map(finalize_bucket).collect();
  buckets.sort_by(|a, b| cmp_composite_bucket(a, b, &sources));
  if let Some(after_val) = after
    .as_ref()
    .and_then(|v| composite_key_from_value(v, &sources))
  {
    buckets.retain(|b| {
      composite_key_from_value(&b.key, &sources)
        .map(|k| k > after_val)
        .unwrap_or(true)
    });
  }
  let has_more = buckets.len() > size;
  if has_more {
    buckets.truncate(size);
  }
  let aggregations = apply_pipeline_aggs(&pipeline, &mut buckets);
  let after_key = if has_more {
    buckets.last().map(|b| b.key.clone())
  } else {
    None
  };
  AggregationResponse::Composite {
    buckets,
    after_key,
    aggregations,
    sampled,
  }
}

fn cmp_composite_bucket(
  a: &BucketResponse,
  b: &BucketResponse,
  sources: &[CompositeSource],
) -> Ordering {
  let a_key = composite_key_from_value(&a.key, sources);
  let b_key = composite_key_from_value(&b.key, sources);
  match (a_key, b_key) {
    (Some(ka), Some(kb)) => ka.cmp(&kb),
    (Some(_), None) => Ordering::Less,
    (None, Some(_)) => Ordering::Greater,
    (None, None) => bucket_key_string(&a.key).cmp(&bucket_key_string(&b.key)),
  }
}

fn composite_key_from_value(
  value: &serde_json::Value,
  sources: &[CompositeSource],
) -> Option<CompositeKey> {
  let obj = value.as_object()?;
  let mut parts = Vec::with_capacity(sources.len());
  for source in sources.iter() {
    let (name, is_terms) = match source {
      CompositeSource::Terms { name, .. } => (name, true),
      CompositeSource::Histogram { name, .. } => (name, false),
    };
    let val = obj.get(name)?;
    let part = if is_terms {
      Some(CompositeKeyPart::Str(val.as_str()?.to_string()))
    } else {
      val.as_f64().map(|v| CompositeKeyPart::F64(v.to_bits()))
    }?;
    parts.push(part);
  }
  Some(CompositeKey { parts })
}

fn composite_key_to_json(key: &CompositeKey, sources: &[CompositeSource]) -> serde_json::Value {
  let mut obj = serde_json::Map::new();
  for (part, source) in key.parts.iter().zip(sources.iter()) {
    let name = match source {
      CompositeSource::Terms { name, .. } => name,
      CompositeSource::Histogram { name, .. } => name,
    };
    obj.insert(name.clone(), part.to_json());
  }
  serde_json::Value::Object(obj)
}

fn build_composite_keys(
  sources: &[Vec<CompositeKeyPart>],
  idx: usize,
  current: &mut Vec<CompositeKeyPart>,
  out: &mut Vec<CompositeKey>,
) {
  if idx == sources.len() {
    out.push(CompositeKey {
      parts: current.clone(),
    });
    return;
  }
  for val in sources[idx].iter() {
    current.push(val.clone());
    build_composite_keys(sources, idx + 1, current, out);
    current.pop();
  }
}

fn hash_cardinality<T: Hash>(value: &T) -> u64 {
  let mut hasher = DefaultHasher::new();
  value.hash(&mut hasher);
  hasher.finish()
}

fn default_percentiles_list() -> Vec<f64> {
  vec![1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0]
}

/// Compute the effective fill range for `HistogramCollector::finish`.
///
/// When both `extended_bounds` and `hard_bounds` are set, the empty-bucket fill
/// range is clipped to the intersection so that `hard_bounds` is honored as an
/// absolute cap on emitted buckets (BUG-188). Returns `None` when there are no
/// bounds to materialize or when the ranges do not overlap.
fn intersect_fill_range_f64(
  extended: Option<(f64, f64)>,
  hard: Option<(f64, f64)>,
) -> Option<(f64, f64)> {
  match (extended, hard) {
    (Some((emin, emax)), Some((hmin, hmax))) => {
      let lo = emin.max(hmin);
      let hi = emax.min(hmax);
      if lo <= hi {
        Some((lo, hi))
      } else {
        None
      }
    }
    (Some(eb), None) => Some(eb),
    (None, Some(hb)) => Some(hb),
    (None, None) => None,
  }
}

/// Integer-millisecond counterpart of [`intersect_fill_range_f64`] used by
/// [`DateHistogramCollector::finish`]. See that function for the full
/// rationale (BUG-188).
fn intersect_fill_range_i64(
  extended: Option<(i64, i64)>,
  hard: Option<(i64, i64)>,
) -> Option<(i64, i64)> {
  match (extended, hard) {
    (Some((emin, emax)), Some((hmin, hmax))) => {
      let lo = emin.max(hmin);
      let hi = emax.min(hmax);
      if lo <= hi {
        Some((lo, hi))
      } else {
        None
      }
    }
    (Some(eb), None) => Some(eb),
    (None, Some(hb)) => Some(hb),
    (None, None) => None,
  }
}

/// Returns `true` when the inclusive bucket count implied by a date_histogram's
/// `extended_bounds` (clipped to `hard_bounds`, if any) exceeds [`MAX_BUCKETS`].
///
/// The computation mirrors [`DateHistogramCollector::finish`] exactly: both
/// sides of the range are pushed through [`bucket_start`] first, then for
/// `Calendar` intervals we walk the range with [`add_interval`] (bailing out
/// the moment we cross `MAX_BUCKETS`), and for `Fixed` intervals we divide the
/// inclusive span by the step. Using a naive `(max - min) / interval` here
/// would miss the fence-post bucket and ignore `offset`, letting a pathological
/// request slip past the cap by one or two buckets.
///
/// Returns `false` when there are no bounds to materialize, when the parsed
/// interval is degenerate, or when the implied span is at or below the cap.
/// Callers are expected to have already rejected degenerate intervals.
pub(crate) fn date_histogram_span_exceeds_cap(
  extended: Option<(i64, i64)>,
  hard: Option<(i64, i64)>,
  offset_ms: i64,
  fixed_millis: Option<i64>,
  calendar: Option<CalendarUnit>,
) -> bool {
  let interval = if let Some(cal) = calendar {
    DateInterval::Calendar(cal)
  } else if let Some(millis) = fixed_millis {
    if millis <= 0 {
      return false;
    }
    DateInterval::Fixed(millis)
  } else {
    return false;
  };
  let Some((min, max)) = intersect_fill_range_i64(extended, hard) else {
    return false;
  };
  let (Some(mut start), Some(mut end)) = (
    bucket_start(min, offset_ms, &interval),
    bucket_start(max, offset_ms, &interval),
  ) else {
    return false;
  };
  if start > end {
    std::mem::swap(&mut start, &mut end);
  }
  match &interval {
    DateInterval::Fixed(step) => {
      let diff = end.saturating_sub(start);
      // `step > 0` guaranteed above; inclusive count = (end - start)/step + 1.
      let span = (diff / *step).saturating_add(1);
      span > MAX_BUCKETS as i64
    }
    DateInterval::Calendar(_) => {
      let mut cur = start;
      let mut count: usize = 1;
      while cur < end {
        cur = match add_interval(cur, &interval) {
          Some(next) => next,
          None => break,
        };
        count = count.saturating_add(1);
        if count > MAX_BUCKETS {
          return true;
        }
      }
      false
    }
  }
}

pub(crate) fn parse_calendar_interval(spec: &str) -> Option<CalendarUnit> {
  match spec.to_ascii_lowercase().as_str() {
    "day" | "1d" => Some(CalendarUnit::Day),
    "week" | "1w" => Some(CalendarUnit::Week),
    "month" | "1m" => Some(CalendarUnit::Month),
    "quarter" | "1q" => Some(CalendarUnit::Quarter),
    "year" | "1y" => Some(CalendarUnit::Year),
    _ => None,
  }
}

fn bucket_start(value: i64, offset: i64, interval: &DateInterval) -> Option<i64> {
  match interval {
    DateInterval::Fixed(step) => {
      // Guard against non-positive steps: the parser accepts `0ms` (and a
      // `DateInterval::Fixed(0)` step), which would divide by zero here and
      // make `add_interval` never advance during empty-bucket fill.
      if *step <= 0 {
        return None;
      }
      // NOTE: must use floor-style bucketing to match Elasticsearch
      // semantics and the behavior of the regular
      // `HistogramCollector::bucket_key`. Using `.ceil()` placed any
      // timestamp not exactly on a bucket boundary into the *next* bucket
      // (BUG-030, issue #186). Integer `div_euclid` rounds toward
      // negative infinity — identical to `.floor()` for this purpose —
      // and, unlike `as f64`, keeps full `i64` precision beyond 2^53.
      let bucket = value.saturating_sub(offset).div_euclid(*step);
      Some(bucket.saturating_mul(*step).saturating_add(offset))
    }
    DateInterval::Calendar(unit) => {
      truncate_calendar(value - offset, *unit).map(|start| start + offset)
    }
  }
}

fn add_interval(current: i64, interval: &DateInterval) -> Option<i64> {
  match interval {
    DateInterval::Fixed(step) => current.checked_add(*step),
    DateInterval::Calendar(unit) => add_calendar(current, *unit),
  }
}

fn truncate_calendar(value: i64, unit: CalendarUnit) -> Option<i64> {
  use chrono::{Datelike, Duration, Utc};
  let dt = chrono::DateTime::<Utc>::from_timestamp_millis(value)?;
  let date = dt.date_naive();
  let start_date = match unit {
    CalendarUnit::Day => date,
    CalendarUnit::Week => {
      date.checked_sub_signed(Duration::days(date.weekday().num_days_from_monday() as i64))?
    }
    CalendarUnit::Month => date.with_day(1)?,
    CalendarUnit::Quarter => {
      // Normalize the day to 1 before changing the month. `with_month` fails
      // when the resulting (year, target_month, original_day) triple is not a
      // real date (e.g. 2024-05-31 → April, which has only 30 days), so the
      // previous `with_month(..)?.with_day(1)?` ordering would short-circuit
      // to `None` and cause `DateHistogramCollector::collect` to silently drop
      // any `YYYY-05-31` document from the aggregation (BUG-233). Going
      // day-first is always safe: day 1 is valid in every month.
      let month = date.month();
      let quarter_start = ((month - 1) / 3) * 3 + 1;
      date.with_day(1)?.with_month(quarter_start)?
    }
    CalendarUnit::Year => date.with_month(1)?.with_day(1)?,
  };
  let start_dt = start_date.and_hms_opt(0, 0, 0)?;
  Some(chrono::DateTime::<Utc>::from_naive_utc_and_offset(start_dt, Utc).timestamp_millis())
}

fn last_day_of_month(year: i32, month: u32) -> u32 {
  use chrono::Datelike;
  if month == 12 {
    31
  } else {
    chrono::NaiveDate::from_ymd_opt(year, month + 1, 1)
      .and_then(|d| d.pred_opt())
      .map(|d| d.day())
      .unwrap_or(28)
  }
}

fn add_calendar(value: i64, unit: CalendarUnit) -> Option<i64> {
  use chrono::{Datelike, Duration, Utc};
  let dt = chrono::DateTime::<Utc>::from_timestamp_millis(value)?;
  let date = dt.date_naive();
  let next_date = match unit {
    CalendarUnit::Day => date.checked_add_signed(Duration::days(1))?,
    CalendarUnit::Week => date.checked_add_signed(Duration::days(7))?,
    CalendarUnit::Month => {
      let mut month = date.month();
      let mut year = date.year();
      month += 1;
      if month > 12 {
        month = 1;
        year += 1;
      }
      let original_day = date.day();
      // Normalize to day 1 before changing year/month to prevent
      // `with_month` from failing when the source day exceeds the
      // target month's length (BUG-233 pattern).
      let base = date.with_day(1)?.with_year(year)?.with_month(month)?;
      let max_day = last_day_of_month(year, month);
      base.with_day(original_day.min(max_day))?
    }
    CalendarUnit::Quarter => {
      let mut month = date.month();
      let mut year = date.year();
      month += 3;
      if month > 12 {
        month -= 12;
        year += 1;
      }
      let original_day = date.day();
      let base = date.with_day(1)?.with_year(year)?.with_month(month)?;
      let max_day = last_day_of_month(year, month);
      base.with_day(original_day.min(max_day))?
    }
    CalendarUnit::Year => {
      let new_year = date.year() + 1;
      let original_day = date.day();
      let original_month = date.month();
      let base = date.with_day(1)?.with_year(new_year)?;
      let max_day = last_day_of_month(new_year, original_month);
      base
        .with_month(original_month)?
        .with_day(original_day.min(max_day))?
    }
  };
  // Preserve the original time-of-day so that bucket keys remain aligned
  // with any sub-day offset applied by `bucket_start`. Previously this
  // hardcoded midnight via `and_hms_opt(0, 0, 0)`, which discarded the
  // offset and produced misaligned fill-loop keys (issue #251).
  let next_dt = next_date.and_time(dt.naive_utc().time());
  Some(chrono::DateTime::<Utc>::from_naive_utc_and_offset(next_dt, Utc).timestamp_millis())
}

pub(crate) fn parse_date(value: &str) -> Option<f64> {
  chrono::DateTime::parse_from_rfc3339(value)
    .map(|dt| dt.timestamp_millis() as f64)
    .ok()
    .or_else(|| value.parse::<f64>().ok())
}

pub(crate) fn parse_interval_seconds(spec: &str) -> Option<f64> {
  let mut idx = 0usize;
  for ch in spec.chars() {
    if ch.is_ascii_digit() || ch == '.' {
      idx += ch.len_utf8();
    } else {
      break;
    }
  }
  if idx == 0 {
    return None;
  }
  let value: f64 = spec[..idx].parse().ok()?;
  let suffix = &spec[idx..];
  let mult = match suffix {
    "" | "s" => 1.0,
    "ms" => 0.001,
    "m" => 60.0,
    "h" => 3600.0,
    "d" => 86_400.0,
    "w" => 604_800.0,
    _ => return None,
  };
  Some(value * mult)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn parse_interval_seconds_accepts_valid_units() {
    assert_eq!(parse_interval_seconds("10"), Some(10.0));
    assert_eq!(parse_interval_seconds("1500ms"), Some(1.5));
    assert_eq!(parse_interval_seconds("2s"), Some(2.0));
    assert_eq!(parse_interval_seconds("1m"), Some(60.0));
    assert_eq!(parse_interval_seconds("2.5m"), Some(150.0));
    assert_eq!(parse_interval_seconds("1h"), Some(3_600.0));
  }

  #[test]
  fn parse_interval_seconds_rejects_unknown_units() {
    assert_eq!(parse_interval_seconds("5x"), None);
    assert_eq!(parse_interval_seconds("10foo"), None);
  }

  #[test]
  fn quantile_state_handles_small_samples() {
    let mut q = QuantileState::default();
    for v in [1.0, 2.0, 3.0, 4.0] {
      q.push(v);
    }
    assert!((q.percentile(50.0) - 2.5).abs() < 1e-6);
    assert!((q.percentile_rank(2.0) - 50.0).abs() < 1e-6);
  }

  #[test]
  fn bucket_script_evaluates_basic_expression() {
    let mut vars = BTreeMap::new();
    vars.insert("a".to_string(), 2.0);
    vars.insert("b".to_string(), 4.0);
    let value = eval_bucket_script("a + b * 2", &vars).unwrap();
    assert!((value - 10.0).abs() < 1e-6);
  }

  #[test]
  fn bucket_script_handles_binary_subtraction_without_whitespace() {
    let mut vars = BTreeMap::new();
    vars.insert("a".to_string(), 5.0);
    let value = eval_bucket_script("a-2", &vars).unwrap();
    assert!((value - 3.0).abs() < 1e-6);
  }

  #[test]
  fn bucket_script_rejects_near_zero_division() {
    let mut vars = BTreeMap::new();
    vars.insert("a".to_string(), 1.0);
    vars.insert("b".to_string(), 1e-14);
    assert!(eval_bucket_script("a / b", &vars).is_none());
  }

  #[test]
  fn bucket_script_rejects_unmatched_rparen() {
    let mut vars = BTreeMap::new();
    vars.insert("a".to_string(), 2.0);
    vars.insert("b".to_string(), 3.0);
    vars.insert("c".to_string(), 4.0);
    assert!(eval_bucket_script("a + b) * c", &vars).is_none());
  }

  #[test]
  fn bucket_script_rejects_unmatched_lparen() {
    let mut vars = BTreeMap::new();
    vars.insert("a".to_string(), 2.0);
    vars.insert("b".to_string(), 3.0);
    vars.insert("c".to_string(), 4.0);
    assert!(eval_bucket_script("(a + b * c", &vars).is_none());
  }

  #[test]
  fn bucket_script_accepts_matched_parentheses() {
    let mut vars = BTreeMap::new();
    vars.insert("a".to_string(), 2.0);
    vars.insert("b".to_string(), 3.0);
    vars.insert("c".to_string(), 4.0);
    let value = eval_bucket_script("(a + b) * c", &vars).unwrap();
    assert!((value - 20.0).abs() < 1e-6);
  }

  #[test]
  fn bucket_script_unary_negation_of_variable() {
    let mut vars = BTreeMap::new();
    vars.insert("a".to_string(), 2.0);
    vars.insert("b".to_string(), 3.0);
    let value = eval_bucket_script("a * -b", &vars).unwrap();
    assert!((value - (-6.0)).abs() < 1e-6);
  }

  #[test]
  fn bucket_script_leading_unary_negation_variable() {
    let mut vars = BTreeMap::new();
    vars.insert("a".to_string(), 5.0);
    let value = eval_bucket_script("-a", &vars).unwrap();
    assert!((value - (-5.0)).abs() < 1e-6);
  }

  #[test]
  fn bucket_script_double_negation_variable() {
    let mut vars = BTreeMap::new();
    vars.insert("a".to_string(), 3.0);
    vars.insert("b".to_string(), 2.0);
    let value = eval_bucket_script("a - -b", &vars).unwrap();
    assert!((value - 5.0).abs() < 1e-6);
  }

  #[test]
  fn bucket_script_negation_in_parens() {
    let mut vars = BTreeMap::new();
    vars.insert("b".to_string(), 4.0);
    let value = eval_bucket_script("(-b)", &vars).unwrap();
    assert!((value - (-4.0)).abs() < 1e-6);
  }

  #[test]
  fn bucket_script_division_by_negated_variable() {
    let mut vars = BTreeMap::new();
    vars.insert("a".to_string(), 10.0);
    vars.insert("b".to_string(), 2.0);
    let value = eval_bucket_script("a / -b", &vars).unwrap();
    assert!((value - (-5.0)).abs() < 1e-6);
  }

  #[test]
  fn sampler_hash_includes_segment_ord() {
    let sampling = AggregationSampling {
      probability: Some(0.5),
      size: None,
      seed: Some(7),
    };
    let sampler = Sampler::new(Some(&sampling));
    let a = sampler.sample_value(0, 42);
    let b = sampler.sample_value(1, 42);
    assert_ne!(a, b);
  }

  #[test]
  fn intersect_fill_range_f64_clips_extended_to_hard() {
    // BUG-188: `extended_bounds` extending past `hard_bounds` must be clipped
    // to the hard range, not used verbatim.
    assert_eq!(
      intersect_fill_range_f64(Some((0.0, 100.0)), Some((20.0, 50.0))),
      Some((20.0, 50.0))
    );
    // Asymmetric overflow on the upper bound only.
    assert_eq!(
      intersect_fill_range_f64(Some((25.0, 80.0)), Some((20.0, 50.0))),
      Some((25.0, 50.0))
    );
    // Asymmetric overflow on the lower bound only.
    assert_eq!(
      intersect_fill_range_f64(Some((10.0, 45.0)), Some((20.0, 50.0))),
      Some((20.0, 45.0))
    );
  }

  #[test]
  fn intersect_fill_range_f64_falls_back_when_one_side_unset() {
    assert_eq!(
      intersect_fill_range_f64(Some((10.0, 40.0)), None),
      Some((10.0, 40.0))
    );
    assert_eq!(
      intersect_fill_range_f64(None, Some((5.0, 25.0))),
      Some((5.0, 25.0))
    );
    assert_eq!(intersect_fill_range_f64(None, None), None);
  }

  #[test]
  fn intersect_fill_range_f64_returns_none_for_disjoint_ranges() {
    // Disjoint ranges produce no fill at all — emitting phantom buckets
    // between them would violate hard_bounds.
    assert_eq!(
      intersect_fill_range_f64(Some((0.0, 10.0)), Some((20.0, 50.0))),
      None
    );
    assert_eq!(
      intersect_fill_range_f64(Some((60.0, 100.0)), Some((20.0, 50.0))),
      None
    );
  }

  #[test]
  fn intersect_fill_range_f64_preserves_ext_when_contained_in_hard() {
    // When `extended_bounds` is fully inside `hard_bounds` (the case the
    // request validator currently enforces), the intersection is exactly
    // `extended_bounds` — so the fix is a no-op for the validated path.
    assert_eq!(
      intersect_fill_range_f64(Some((25.0, 45.0)), Some((20.0, 50.0))),
      Some((25.0, 45.0))
    );
  }

  #[test]
  fn intersect_fill_range_i64_clips_extended_to_hard() {
    // Same BUG-188 guarantee for `DateHistogramCollector` (millisecond ints).
    assert_eq!(
      intersect_fill_range_i64(Some((0, 1_000)), Some((200, 500))),
      Some((200, 500))
    );
    assert_eq!(
      intersect_fill_range_i64(Some((100, 800)), Some((200, 500))),
      Some((200, 500))
    );
  }

  #[test]
  fn intersect_fill_range_i64_handles_missing_sides_and_disjoint() {
    assert_eq!(
      intersect_fill_range_i64(Some((10, 40)), None),
      Some((10, 40))
    );
    assert_eq!(intersect_fill_range_i64(None, Some((5, 25))), Some((5, 25)));
    assert_eq!(intersect_fill_range_i64(None, None), None);
    assert_eq!(
      intersect_fill_range_i64(Some((0, 10)), Some((20, 50))),
      None
    );
  }

  /// BUG-200: the helper feeding `validate_date_histogram_config` must use the
  /// same inclusive `bucket_start(min)..=bucket_start(max)` span the collector
  /// materializes — otherwise a pathologically small `fixed_interval` + wide
  /// bounds can either slip past validation by one bucket (fence-post) or
  /// silently allow a request that the runtime cap will later truncate.
  #[test]
  fn date_histogram_span_exceeds_cap_flags_wide_fixed_interval() {
    // 4-year span at 1ms — the exact repro from the issue.
    let four_years_ms: i64 = 4 * 365 * 86_400_000;
    let extended = Some((0_i64, four_years_ms));
    assert!(date_histogram_span_exceeds_cap(
      extended,
      None,
      0,
      Some(1),
      None
    ));
  }

  #[test]
  fn date_histogram_span_exceeds_cap_fence_post_rejected() {
    // 10_000 seconds at 1s → inclusive count = 10_001 (one above cap).
    let extended = Some((0_i64, 10_000_000_i64));
    assert!(date_histogram_span_exceeds_cap(
      extended,
      None,
      0,
      Some(1_000),
      None
    ));
  }

  #[test]
  fn date_histogram_span_exceeds_cap_exact_boundary_accepted() {
    // 9_999 seconds at 1s → inclusive count = 10_000 (exactly at cap).
    let extended = Some((0_i64, 9_999_000_i64));
    assert!(!date_histogram_span_exceeds_cap(
      extended,
      None,
      0,
      Some(1_000),
      None
    ));
  }

  #[test]
  fn date_histogram_span_exceeds_cap_honors_intersection_with_hard_bounds() {
    // extended is huge, but hard_bounds clips it to a tiny sub-range —
    // materialization is bounded by the intersection, not the union.
    let huge: (i64, i64) = (0, 4 * 365 * 86_400_000);
    let tiny_hard: (i64, i64) = (0, 1_000); // 1s at 1ms step = 1001 buckets.
    assert!(!date_histogram_span_exceeds_cap(
      Some(huge),
      Some(tiny_hard),
      0,
      Some(1),
      None
    ));
  }

  #[test]
  fn date_histogram_span_exceeds_cap_no_bounds_is_safe() {
    // With neither bound set, no empty buckets are materialized — safe.
    assert!(!date_histogram_span_exceeds_cap(
      None,
      None,
      0,
      Some(1),
      None
    ));
  }

  #[test]
  fn date_histogram_span_exceeds_cap_calendar_day_wide_range_rejected() {
    // ~100 years of day buckets is well above MAX_BUCKETS.
    let start = chrono::NaiveDate::from_ymd_opt(1900, 1, 1)
      .unwrap()
      .and_hms_opt(0, 0, 0)
      .unwrap()
      .and_utc()
      .timestamp_millis();
    let end = chrono::NaiveDate::from_ymd_opt(2000, 1, 1)
      .unwrap()
      .and_hms_opt(0, 0, 0)
      .unwrap()
      .and_utc()
      .timestamp_millis();
    assert!(date_histogram_span_exceeds_cap(
      Some((start, end)),
      None,
      0,
      None,
      Some(CalendarUnit::Day)
    ));
  }

  #[test]
  fn date_histogram_span_exceeds_cap_degenerate_fixed_is_safe() {
    // A `Fixed(0)` is rejected earlier in the validator chain; if it ever
    // reaches the span helper, return `false` rather than dividing by zero.
    assert!(!date_histogram_span_exceeds_cap(
      Some((0_i64, 1_000)),
      None,
      0,
      Some(0),
      None
    ));
  }

  /// BUG-233: `truncate_calendar` for `CalendarUnit::Quarter` used to change
  /// the month before normalizing the day. When the source day was 31 and the
  /// target quarter-start month was April (30 days), `with_month(4)` returned
  /// `None`, cascading back into `DateHistogramCollector::collect`, which
  /// silently dropped the document. The only real-world triggering date is
  /// day 31 of May (any year). This regression pins the fixed
  /// day-first ordering.
  #[test]
  fn truncate_calendar_quarter_handles_may_31_without_dropping_doc() {
    fn ts(year: i32, month: u32, day: u32, hour: u32) -> i64 {
      chrono::NaiveDate::from_ymd_opt(year, month, day)
        .unwrap()
        .and_hms_opt(hour, 0, 0)
        .unwrap()
        .and_utc()
        .timestamp_millis()
    }
    // 2024-05-31T12:00:00Z → should truncate to 2024-04-01T00:00:00Z (Q2).
    let input = ts(2024, 5, 31, 12);
    let expected = ts(2024, 4, 1, 0);
    assert_eq!(
      truncate_calendar(input, CalendarUnit::Quarter),
      Some(expected),
      "May 31 must fall into the Q2 bucket keyed 2024-04-01T00:00:00Z, \
       not disappear from the aggregation"
    );
    // The full collector path goes through `bucket_start`; assert that too.
    assert_eq!(
      bucket_start(input, 0, &DateInterval::Calendar(CalendarUnit::Quarter)),
      Some(expected),
      "bucket_start must propagate the truncated Q2 start, not None"
    );
  }

  /// BUG-233 follow-up: exhaustive sweep over every day of a leap year
  /// (which covers all 366 calendar slots, including Feb 29) × every calendar
  /// unit. `truncate_calendar` must produce `Some(_)` for every valid
  /// timestamp — `None` here is the exact failure mode that makes the
  /// collector silently drop documents. Guards against any future
  /// ordering regression in any branch of `truncate_calendar`.
  #[test]
  fn truncate_calendar_never_returns_none_for_valid_dates() {
    let units = [
      ("Day", CalendarUnit::Day),
      ("Week", CalendarUnit::Week),
      ("Month", CalendarUnit::Month),
      ("Quarter", CalendarUnit::Quarter),
      ("Year", CalendarUnit::Year),
    ];
    for ordinal in 1..=366u32 {
      // 2024 is a leap year, so every ordinal in 1..=366 yields a valid date.
      let date = chrono::NaiveDate::from_yo_opt(2024, ordinal).unwrap();
      let millis = date
        .and_hms_opt(23, 59, 59)
        .unwrap()
        .and_utc()
        .timestamp_millis();
      for (name, unit) in units {
        assert!(
          truncate_calendar(millis, unit).is_some(),
          "truncate_calendar returned None for {date} with unit {name}; \
           DateHistogramCollector would silently drop this document",
        );
      }
    }
  }

  #[test]
  fn derivative_pipeline_emits_expected_values() {
    let mut buckets = vec![
      BucketResponse {
        key: serde_json::json!(0),
        doc_count: 1,
        aggregations: BTreeMap::from([(
          "metric".to_string(),
          AggregationResponse::ValueCount(ValueCountResponse { value: 1 }),
        )]),
      },
      BucketResponse {
        key: serde_json::json!(1),
        doc_count: 1,
        aggregations: BTreeMap::from([(
          "metric".to_string(),
          AggregationResponse::ValueCount(ValueCountResponse { value: 3 }),
        )]),
      },
    ];
    let mut responses = BTreeMap::new();
    apply_derivative_pipeline(
      "diff",
      &DerivativeAggregation {
        buckets_path: "metric".to_string(),
        gap_policy: Some(GapPolicy::Skip),
        unit: Some(1.0),
      },
      &mut buckets,
      &mut responses,
    );
    if let Some(AggregationResponse::Derivative(OptionalBucketMetricResponse { value })) =
      buckets[1].aggregations.get("diff")
    {
      assert_eq!(value.unwrap(), 2.0);
    } else {
      panic!("missing derivative on bucket");
    }
    assert!(responses.contains_key("diff"));
  }

  /// Regression for #251: add_calendar must preserve the sub-day time
  /// component so that bucket keys stay aligned with the offset applied by
  /// bucket_start. Previously `and_hms_opt(0, 0, 0)` discarded the time,
  /// snapping every bucket after the first to midnight.
  #[test]
  fn add_calendar_preserves_sub_day_time_component() {
    use chrono::{NaiveDate, Utc};
    // Input: 2024-04-01T01:00:00Z (midnight + 1h offset)
    let dt = NaiveDate::from_ymd_opt(2024, 4, 1)
      .unwrap()
      .and_hms_opt(1, 0, 0)
      .unwrap();
    let ts = chrono::DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc).timestamp_millis();

    // Month: expect 2024-05-01T01:00:00Z, NOT 2024-05-01T00:00:00Z
    let next = add_calendar(ts, CalendarUnit::Month).unwrap();
    let expected = NaiveDate::from_ymd_opt(2024, 5, 1)
      .unwrap()
      .and_hms_opt(1, 0, 0)
      .unwrap();
    let expected_ts =
      chrono::DateTime::<Utc>::from_naive_utc_and_offset(expected, Utc).timestamp_millis();
    assert_eq!(
      next, expected_ts,
      "add_calendar(Month) must preserve 01:00:00 offset"
    );

    // Quarter: expect 2024-07-01T01:00:00Z
    let next_q = add_calendar(ts, CalendarUnit::Quarter).unwrap();
    let expected_q = NaiveDate::from_ymd_opt(2024, 7, 1)
      .unwrap()
      .and_hms_opt(1, 0, 0)
      .unwrap();
    let expected_q_ts =
      chrono::DateTime::<Utc>::from_naive_utc_and_offset(expected_q, Utc).timestamp_millis();
    assert_eq!(
      next_q, expected_q_ts,
      "add_calendar(Quarter) must preserve 01:00:00 offset"
    );

    // Year: expect 2025-04-01T01:00:00Z (preserves month and day)
    let next_y = add_calendar(ts, CalendarUnit::Year).unwrap();
    let expected_y = NaiveDate::from_ymd_opt(2025, 4, 1)
      .unwrap()
      .and_hms_opt(1, 0, 0)
      .unwrap();
    let expected_y_ts =
      chrono::DateTime::<Utc>::from_naive_utc_and_offset(expected_y, Utc).timestamp_millis();
    assert_eq!(
      next_y, expected_y_ts,
      "add_calendar(Year) must preserve 01:00:00 offset"
    );
  }

  /// Regression for #251: chained add_calendar calls (as the fill loop does)
  /// must produce a monotonically increasing sequence where every bucket key
  /// keeps the original sub-day offset.
  #[test]
  fn add_calendar_fill_loop_stays_aligned_with_offset() {
    use chrono::{NaiveDate, Utc};
    let offset_ms: i64 = 3_600_000; // 1 hour
    let start_dt = NaiveDate::from_ymd_opt(2024, 4, 1)
      .unwrap()
      .and_hms_opt(1, 0, 0)
      .unwrap();
    let start =
      chrono::DateTime::<Utc>::from_naive_utc_and_offset(start_dt, Utc).timestamp_millis();

    let expected_keys: Vec<i64> = [(2024, 4, 1), (2024, 5, 1), (2024, 6, 1), (2024, 7, 1)]
      .iter()
      .map(|(y, m, d)| {
        let dt = NaiveDate::from_ymd_opt(*y, *m, *d)
          .unwrap()
          .and_hms_opt(1, 0, 0)
          .unwrap();
        chrono::DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc).timestamp_millis()
      })
      .collect();

    let mut current = start;
    let mut keys = vec![current];
    for _ in 0..3 {
      current = add_calendar(current, CalendarUnit::Month).unwrap();
      keys.push(current);
    }
    assert_eq!(
      keys, expected_keys,
      "fill loop must produce keys at T01:00:00Z, not T00:00:00Z"
    );

    // Also verify bucket_start + add_interval round-trip consistency
    let interval = DateInterval::Calendar(CalendarUnit::Month);
    let mut cur = bucket_start(start, offset_ms, &interval).unwrap();
    for expected in &expected_keys {
      assert_eq!(cur, *expected);
      cur = add_interval(cur, &interval).unwrap();
    }
  }

  /// Regression for #257: add_calendar(Month) must preserve the day-of-month
  /// when the input has day > 1 (as happens with offsets >= 24h). Previously
  /// `.with_day(1)?` forced every bucket key to day 1, misaligning the fill
  /// loop with `bucket_start` output.
  #[test]
  fn add_calendar_preserves_day_with_large_offset() {
    use chrono::{NaiveDate, Utc};
    // Input: 2024-06-03T00:00:00Z (day 3 from a 2-day offset)
    let dt = NaiveDate::from_ymd_opt(2024, 6, 3)
      .unwrap()
      .and_hms_opt(0, 0, 0)
      .unwrap();
    let ts = chrono::DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc).timestamp_millis();

    // Month: expect 2024-07-03, NOT 2024-07-01
    let next = add_calendar(ts, CalendarUnit::Month).unwrap();
    let expected = NaiveDate::from_ymd_opt(2024, 7, 3)
      .unwrap()
      .and_hms_opt(0, 0, 0)
      .unwrap();
    let expected_ts =
      chrono::DateTime::<Utc>::from_naive_utc_and_offset(expected, Utc).timestamp_millis();
    assert_eq!(
      next, expected_ts,
      "add_calendar(Month) must preserve the day-of-month"
    );

    // Quarter: expect 2024-09-03
    let next_q = add_calendar(ts, CalendarUnit::Quarter).unwrap();
    let expected_q = NaiveDate::from_ymd_opt(2024, 9, 3)
      .unwrap()
      .and_hms_opt(0, 0, 0)
      .unwrap();
    let expected_q_ts =
      chrono::DateTime::<Utc>::from_naive_utc_and_offset(expected_q, Utc).timestamp_millis();
    assert_eq!(
      next_q, expected_q_ts,
      "add_calendar(Quarter) must preserve the day-of-month"
    );

    // Year: expect 2025-06-03
    let next_y = add_calendar(ts, CalendarUnit::Year).unwrap();
    let expected_y = NaiveDate::from_ymd_opt(2025, 6, 3)
      .unwrap()
      .and_hms_opt(0, 0, 0)
      .unwrap();
    let expected_y_ts =
      chrono::DateTime::<Utc>::from_naive_utc_and_offset(expected_y, Utc).timestamp_millis();
    assert_eq!(
      next_y, expected_y_ts,
      "add_calendar(Year) must preserve the day-of-month"
    );
  }

  /// Regression for #257: chained add_calendar calls with a 2-day offset must
  /// produce keys at day 3 each month, matching bucket_start output.
  #[test]
  fn add_calendar_fill_loop_aligned_with_multi_day_offset() {
    use chrono::{NaiveDate, Utc};
    let offset_ms: i64 = 172_800_000; // 2 days
    let start_dt = NaiveDate::from_ymd_opt(2024, 4, 3)
      .unwrap()
      .and_hms_opt(0, 0, 0)
      .unwrap();
    let start =
      chrono::DateTime::<Utc>::from_naive_utc_and_offset(start_dt, Utc).timestamp_millis();

    let expected_keys: Vec<i64> = [(2024, 4, 3), (2024, 5, 3), (2024, 6, 3), (2024, 7, 3)]
      .iter()
      .map(|(y, m, d)| {
        let dt = NaiveDate::from_ymd_opt(*y, *m, *d)
          .unwrap()
          .and_hms_opt(0, 0, 0)
          .unwrap();
        chrono::DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc).timestamp_millis()
      })
      .collect();

    let mut current = start;
    let mut keys = vec![current];
    for _ in 0..3 {
      current = add_calendar(current, CalendarUnit::Month).unwrap();
      keys.push(current);
    }
    assert_eq!(
      keys, expected_keys,
      "fill loop must produce keys at day 3, not day 1"
    );

    // Verify bucket_start + add_interval round-trip consistency
    let interval = DateInterval::Calendar(CalendarUnit::Month);
    let mut cur = bucket_start(start, offset_ms, &interval).unwrap();
    for expected in &expected_keys {
      assert_eq!(cur, *expected);
      cur = add_interval(cur, &interval).unwrap();
    }
  }

  /// Regression for #257: day clamping when the original day exceeds the
  /// target month's length (e.g. Jan 31 + 1 month → Feb 28/29).
  #[test]
  fn add_calendar_clamps_day_to_target_month_length() {
    use chrono::{NaiveDate, Utc};

    // Jan 31 + 1 month → Feb 29 (2024 is a leap year)
    let jan31 = NaiveDate::from_ymd_opt(2024, 1, 31)
      .unwrap()
      .and_hms_opt(0, 0, 0)
      .unwrap();
    let ts = chrono::DateTime::<Utc>::from_naive_utc_and_offset(jan31, Utc).timestamp_millis();

    let next = add_calendar(ts, CalendarUnit::Month).unwrap();
    let expected = NaiveDate::from_ymd_opt(2024, 2, 29)
      .unwrap()
      .and_hms_opt(0, 0, 0)
      .unwrap();
    let expected_ts =
      chrono::DateTime::<Utc>::from_naive_utc_and_offset(expected, Utc).timestamp_millis();
    assert_eq!(
      next, expected_ts,
      "Jan 31 + Month must clamp to Feb 29 in a leap year"
    );

    // Jan 31 + 1 month in a non-leap year → Feb 28
    let jan31_nl = NaiveDate::from_ymd_opt(2023, 1, 31)
      .unwrap()
      .and_hms_opt(0, 0, 0)
      .unwrap();
    let ts_nl =
      chrono::DateTime::<Utc>::from_naive_utc_and_offset(jan31_nl, Utc).timestamp_millis();

    let next_nl = add_calendar(ts_nl, CalendarUnit::Month).unwrap();
    let expected_nl = NaiveDate::from_ymd_opt(2023, 2, 28)
      .unwrap()
      .and_hms_opt(0, 0, 0)
      .unwrap();
    let expected_nl_ts =
      chrono::DateTime::<Utc>::from_naive_utc_and_offset(expected_nl, Utc).timestamp_millis();
    assert_eq!(
      next_nl, expected_nl_ts,
      "Jan 31 + Month must clamp to Feb 28 in a non-leap year"
    );

    // Feb 29 (leap) + 1 year → Feb 28 (non-leap)
    let feb29 = NaiveDate::from_ymd_opt(2024, 2, 29)
      .unwrap()
      .and_hms_opt(0, 0, 0)
      .unwrap();
    let ts_leap = chrono::DateTime::<Utc>::from_naive_utc_and_offset(feb29, Utc).timestamp_millis();

    let next_year = add_calendar(ts_leap, CalendarUnit::Year).unwrap();
    let expected_year = NaiveDate::from_ymd_opt(2025, 2, 28)
      .unwrap()
      .and_hms_opt(0, 0, 0)
      .unwrap();
    let expected_year_ts =
      chrono::DateTime::<Utc>::from_naive_utc_and_offset(expected_year, Utc).timestamp_millis();
    assert_eq!(
      next_year, expected_year_ts,
      "Feb 29 + Year must clamp to Feb 28 in a non-leap year"
    );
  }

  /// Regression test for #249: finalize_response must rank significant_terms
  /// buckets by significance score, not by doc_count. A low-frequency term
  /// with a very low bg_count must outrank a high-frequency term with a high
  /// bg_count when its significance score is higher.
  #[test]
  fn significant_terms_finalize_ranks_by_significance_score() {
    use serde_json::json;

    let intermediate = AggregationIntermediate::SignificantTerms {
      buckets: vec![
        // "common": high doc_count, high bg_count → low significance
        SignificantBucketIntermediate {
          key: json!("common"),
          doc_count: 80,
          bg_count: 5000,
          aggs: BTreeMap::new(),
        },
        // "frequent": medium doc_count, high bg_count → low significance
        SignificantBucketIntermediate {
          key: json!("frequent"),
          doc_count: 50,
          bg_count: 4000,
          aggs: BTreeMap::new(),
        },
        // "rare_sig": low doc_count, very low bg_count → very high significance
        SignificantBucketIntermediate {
          key: json!("rare_sig"),
          doc_count: 3,
          bg_count: 5,
          aggs: BTreeMap::new(),
        },
      ],
      size: Some(2),
      min_doc_count: 1,
      pipeline: BTreeMap::new(),
      doc_count: 100,
      bg_count: 10_000,
      sampled: false,
    };

    let response = finalize_response(intermediate);
    if let AggregationResponse::SignificantTerms { buckets, .. } = response {
      assert_eq!(buckets.len(), 2, "size=2 should yield 2 buckets");
      assert_eq!(
        buckets[0].key.as_str().unwrap(),
        "rare_sig",
        "rare_sig (score=60.0) must be ranked #1"
      );
      assert!(
        buckets[0].score > buckets[1].score,
        "first bucket should have higher score"
      );
      // Verify that "rare_sig" was not discarded by intermediate truncation
      let keys: Vec<&str> = buckets.iter().map(|b| b.key.as_str().unwrap()).collect();
      assert!(
        keys.contains(&"rare_sig"),
        "rare_sig must survive truncation"
      );
    } else {
      panic!("expected SignificantTerms response");
    }
  }

  /// Zero bg_count buckets must not displace genuinely significant terms.
  /// finalize_response assigns score 0.0 when bg_count == 0, so the
  /// intermediate proxy sort must also treat them as score 0.0.
  #[test]
  fn significant_terms_zero_bg_count_does_not_displace_real_terms() {
    use serde_json::json;

    let intermediate = AggregationIntermediate::SignificantTerms {
      buckets: vec![
        // "real_sig": genuinely significant (bg_count > 0)
        SignificantBucketIntermediate {
          key: json!("real_sig"),
          doc_count: 5,
          bg_count: 10,
          aggs: BTreeMap::new(),
        },
        // "zero_bg": high doc_count but bg_count == 0 → final score 0.0
        SignificantBucketIntermediate {
          key: json!("zero_bg"),
          doc_count: 90,
          bg_count: 0,
          aggs: BTreeMap::new(),
        },
      ],
      size: Some(1),
      min_doc_count: 1,
      pipeline: BTreeMap::new(),
      doc_count: 100,
      bg_count: 10_000,
      sampled: false,
    };

    let response = finalize_response(intermediate);
    if let AggregationResponse::SignificantTerms { buckets, .. } = response {
      assert_eq!(buckets.len(), 1, "size=1 should yield 1 bucket");
      assert_eq!(
        buckets[0].key.as_str().unwrap(),
        "real_sig",
        "zero bg_count bucket must not displace genuinely significant term"
      );
      assert!(
        buckets[0].score > 0.0,
        "real_sig should have a positive score"
      );
    } else {
      panic!("expected SignificantTerms response");
    }
  }

  /// Regression test for #249: cross-segment merge must also sort by
  /// significance score proxy before truncation.
  #[test]
  fn significant_terms_merge_preserves_high_significance_terms() {
    use serde_json::json;

    // Segment 1: contains "common" (high doc_count)
    let mut seg1 = AggregationIntermediate::SignificantTerms {
      buckets: vec![SignificantBucketIntermediate {
        key: json!("common"),
        doc_count: 40,
        bg_count: 2500,
        aggs: BTreeMap::new(),
      }],
      size: Some(1),
      min_doc_count: 1,
      pipeline: BTreeMap::new(),
      doc_count: 50,
      bg_count: 5000,
      sampled: false,
    };

    // Segment 2: contains "common" (more) and "rare_sig" (low doc_count, very low bg_count)
    let seg2 = AggregationIntermediate::SignificantTerms {
      buckets: vec![
        SignificantBucketIntermediate {
          key: json!("common"),
          doc_count: 40,
          bg_count: 2500,
          aggs: BTreeMap::new(),
        },
        SignificantBucketIntermediate {
          key: json!("rare_sig"),
          doc_count: 3,
          bg_count: 5,
          aggs: BTreeMap::new(),
        },
      ],
      size: Some(1),
      min_doc_count: 1,
      pipeline: BTreeMap::new(),
      doc_count: 50,
      bg_count: 5000,
      sampled: false,
    };

    merge_intermediate_in_place(&mut seg1, seg2);

    // After merge with size=1, "rare_sig" should survive because its
    // doc_count/bg_count ratio (3/5 = 0.6) is much higher than
    // "common"'s (80/5000 = 0.016).
    if let AggregationIntermediate::SignificantTerms { buckets, .. } = &seg1 {
      assert_eq!(buckets.len(), 1, "size=1 should yield 1 bucket after merge");
      assert_eq!(
        buckets[0].key.as_str().unwrap(),
        "rare_sig",
        "rare_sig must survive merge truncation due to higher significance"
      );
    } else {
      panic!("expected SignificantTerms intermediate");
    }
  }

  #[test]
  fn bucket_metric_value_resolves_decimal_percentile_key() {
    let mut aggs = BTreeMap::new();
    let mut pct_values = BTreeMap::new();
    pct_values.insert("50".to_string(), 10.0);
    pct_values.insert("99.9".to_string(), 42.5);
    pct_values.insert("99.99".to_string(), 100.0);
    aggs.insert(
      "latency_pct".to_string(),
      AggregationResponse::Percentiles(PercentilesResponse { values: pct_values }),
    );
    let bucket = BucketResponse {
      key: serde_json::json!(0),
      doc_count: 1,
      aggregations: aggs,
    };

    assert_eq!(bucket_metric_value(&bucket, "latency_pct.99.9"), Some(42.5));
    assert_eq!(
      bucket_metric_value(&bucket, "latency_pct.99.99"),
      Some(100.0)
    );
    assert_eq!(bucket_metric_value(&bucket, "latency_pct.50"), Some(10.0));
    assert_eq!(bucket_metric_value(&bucket, "latency_pct.unknown"), None);
  }

  #[test]
  fn bucket_metric_value_resolves_decimal_percentile_rank_key() {
    let mut aggs = BTreeMap::new();
    let mut rank_values = BTreeMap::new();
    rank_values.insert("50.5".to_string(), 72.0);
    rank_values.insert("100".to_string(), 99.0);
    aggs.insert(
      "rank_agg".to_string(),
      AggregationResponse::PercentileRanks(PercentileRanksResponse {
        values: rank_values,
      }),
    );
    let bucket = BucketResponse {
      key: serde_json::json!(0),
      doc_count: 1,
      aggregations: aggs,
    };

    assert_eq!(bucket_metric_value(&bucket, "rank_agg.50.5"), Some(72.0));
    assert_eq!(bucket_metric_value(&bucket, "rank_agg.100"), Some(99.0));
  }

  #[test]
  fn bucket_metric_value_stats_subfield_still_works() {
    let mut aggs = BTreeMap::new();
    aggs.insert(
      "my_stats".to_string(),
      AggregationResponse::Stats(StatsResponse {
        count: 10,
        min: 1.0,
        max: 100.0,
        avg: 50.0,
        sum: 500.0,
      }),
    );
    let bucket = BucketResponse {
      key: serde_json::json!(0),
      doc_count: 10,
      aggregations: aggs,
    };

    assert_eq!(bucket_metric_value(&bucket, "my_stats.max"), Some(100.0));
    assert_eq!(bucket_metric_value(&bucket, "my_stats.min"), Some(1.0));
    assert_eq!(bucket_metric_value(&bucket, "my_stats.avg"), Some(50.0));
    assert_eq!(bucket_metric_value(&bucket, "_count"), Some(10.0));
  }

  #[test]
  fn bucket_metric_value_no_subpath_returns_default_for_stats() {
    let mut aggs = BTreeMap::new();
    aggs.insert(
      "my_stats".to_string(),
      AggregationResponse::Stats(StatsResponse {
        count: 5,
        min: 0.0,
        max: 10.0,
        avg: 5.0,
        sum: 25.0,
      }),
    );
    let bucket = BucketResponse {
      key: serde_json::json!(0),
      doc_count: 5,
      aggregations: aggs,
    };

    // Without a subpath, Stats defaults to "avg"
    assert_eq!(bucket_metric_value(&bucket, "my_stats"), Some(5.0));
  }

  #[test]
  fn moving_avg_pipeline_with_decimal_percentile_path() {
    let mut pct_values = BTreeMap::new();
    pct_values.insert("99.9".to_string(), 10.0);
    let mut pct_values2 = BTreeMap::new();
    pct_values2.insert("99.9".to_string(), 20.0);
    let mut pct_values3 = BTreeMap::new();
    pct_values3.insert("99.9".to_string(), 30.0);

    let mut buckets = vec![
      BucketResponse {
        key: serde_json::json!(0),
        doc_count: 1,
        aggregations: BTreeMap::from([(
          "latency_pct".to_string(),
          AggregationResponse::Percentiles(PercentilesResponse { values: pct_values }),
        )]),
      },
      BucketResponse {
        key: serde_json::json!(1),
        doc_count: 1,
        aggregations: BTreeMap::from([(
          "latency_pct".to_string(),
          AggregationResponse::Percentiles(PercentilesResponse {
            values: pct_values2,
          }),
        )]),
      },
      BucketResponse {
        key: serde_json::json!(2),
        doc_count: 1,
        aggregations: BTreeMap::from([(
          "latency_pct".to_string(),
          AggregationResponse::Percentiles(PercentilesResponse {
            values: pct_values3,
          }),
        )]),
      },
    ];

    let mut responses = BTreeMap::new();
    apply_moving_avg_pipeline(
      "smoothed_p999",
      &MovingAvgAggregation {
        buckets_path: "latency_pct.99.9".to_string(),
        window: 2,
        predict: None,
        gap_policy: Some(GapPolicy::Skip),
      },
      &mut buckets,
      &mut responses,
    );

    // First bucket: moving_avg of window [10.0] = 10.0
    if let Some(AggregationResponse::MovingAvg(val)) = buckets[0].aggregations.get("smoothed_p999")
    {
      assert_eq!(val.value.unwrap(), 10.0);
    } else {
      panic!("expected moving_avg on bucket 0");
    }
    // Second bucket: moving_avg of window [10.0, 20.0] = 15.0
    if let Some(AggregationResponse::MovingAvg(val)) = buckets[1].aggregations.get("smoothed_p999")
    {
      assert_eq!(val.value.unwrap(), 15.0);
    } else {
      panic!("expected moving_avg on bucket 1");
    }
    // Third bucket: moving_avg of window [20.0, 30.0] = 25.0
    if let Some(AggregationResponse::MovingAvg(val)) = buckets[2].aggregations.get("smoothed_p999")
    {
      assert_eq!(val.value.unwrap(), 25.0);
    } else {
      panic!("expected moving_avg on bucket 2");
    }
  }

  #[test]
  fn pipeline_aggs_respect_dependency_order_not_alphabetical() {
    use crate::api::types::{BucketScriptAggregation, DerivativeAggregation};

    fn stats_bucket(key: i64, revenue: f64) -> BucketResponse {
      let mut m = BTreeMap::new();
      m.insert(
        "revenue".to_string(),
        AggregationResponse::Stats(StatsResponse {
          count: 1,
          min: revenue,
          max: revenue,
          sum: revenue,
          avg: revenue,
        }),
      );
      BucketResponse {
        key: serde_json::Value::Number(serde_json::Number::from(key)),
        doc_count: 1,
        aggregations: m,
      }
    }

    let mut buckets = vec![
      stats_bucket(1, 100.0),
      stats_bucket(2, 150.0),
      stats_bucket(3, 200.0),
    ];

    let mut pipeline = BTreeMap::new();
    // "adjusted" sorts before "daily_change" alphabetically, but depends on it
    pipeline.insert(
      "adjusted".to_string(),
      Aggregation::BucketScript(BucketScriptAggregation {
        buckets_path: {
          let mut bp = BTreeMap::new();
          bp.insert("d".to_string(), "daily_change".to_string());
          bp
        },
        script: "d * 100".to_string(),
      }),
    );
    pipeline.insert(
      "daily_change".to_string(),
      Aggregation::Derivative(DerivativeAggregation {
        buckets_path: "revenue".to_string(),
        gap_policy: None,
        unit: None,
      }),
    );

    let _responses = apply_pipeline_aggs(&pipeline, &mut buckets);

    // Bucket 0: derivative is None (no previous), so adjusted should be None
    let adj0 = buckets[0].aggregations.get("adjusted").unwrap();
    if let AggregationResponse::BucketScript(v) = adj0 {
      assert!(v.value.is_none(), "bucket 0 adjusted should be None");
    } else {
      panic!("expected BucketScript response");
    }

    // Bucket 1: derivative = 150 - 100 = 50, adjusted = 50 * 100 = 5000
    let adj1 = buckets[1].aggregations.get("adjusted").unwrap();
    if let AggregationResponse::BucketScript(v) = adj1 {
      assert!(
        v.value.is_some(),
        "bucket 1 adjusted must not be None (dependency ordering bug)"
      );
      assert!(
        (v.value.unwrap() - 5000.0).abs() < 1e-6,
        "expected 5000.0, got {:?}",
        v.value
      );
    } else {
      panic!("expected BucketScript response");
    }

    // Bucket 2: derivative = 200 - 150 = 50, adjusted = 50 * 100 = 5000
    let adj2 = buckets[2].aggregations.get("adjusted").unwrap();
    if let AggregationResponse::BucketScript(v) = adj2 {
      assert!(
        v.value.is_some(),
        "bucket 2 adjusted must not be None (dependency ordering bug)"
      );
      assert!(
        (v.value.unwrap() - 5000.0).abs() < 1e-6,
        "expected 5000.0, got {:?}",
        v.value
      );
    } else {
      panic!("expected BucketScript response");
    }
  }

  #[test]
  fn topological_sort_orders_dependencies_first() {
    use crate::api::types::{BucketScriptAggregation, DerivativeAggregation};

    let mut pipeline = BTreeMap::new();
    pipeline.insert(
      "adjusted".to_string(),
      Aggregation::BucketScript(BucketScriptAggregation {
        buckets_path: {
          let mut bp = BTreeMap::new();
          bp.insert("d".to_string(), "daily_change".to_string());
          bp
        },
        script: "d * 100".to_string(),
      }),
    );
    pipeline.insert(
      "daily_change".to_string(),
      Aggregation::Derivative(DerivativeAggregation {
        buckets_path: "revenue".to_string(),
        gap_policy: None,
        unit: None,
      }),
    );

    let order = topological_sort_pipeline(&pipeline);
    let dc_pos = order.iter().position(|&n| n == "daily_change").unwrap();
    let adj_pos = order.iter().position(|&n| n == "adjusted").unwrap();
    assert!(
      dc_pos < adj_pos,
      "daily_change must come before adjusted, got dc={dc_pos} adj={adj_pos}"
    );
  }
}
