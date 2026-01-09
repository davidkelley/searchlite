use std::cmp::Ordering;
use std::collections::{
  btree_map::Entry as BTreeEntry, hash_map::DefaultHasher, hash_map::Entry as HashEntry, BTreeMap,
  BinaryHeap, HashMap, HashSet,
};
use std::hash::{Hash, Hasher};

use crate::api::types::{
  Aggregation, AggregationResponse, AggregationSampling, BucketMetricResponse, BucketResponse,
  BucketScriptAggregation, BucketSortAggregation, BucketSortSpec, CardinalityResponse,
  CompositeAggregation, CompositeSource, DateHistogramAggregation, DateRangeAggregation,
  DerivativeAggregation, Filter, FilterAggregation, GapPolicy, HistogramAggregation,
  MovingAvgAggregation, MovingAvgResponse, OptionalBucketMetricResponse, PercentileRanksResponse,
  PercentilesResponse, RangeAggregation, RareTermsAggregation, SignificantBucketResponse,
  SignificantTermsAggregation, SortOrder, StatsResponse, TermsAggregation, TopHit,
  TopHitsAggregation, TopHitsResponse, ValueCountResponse,
};
use crate::index::fastfields::FastFieldsReader;
use crate::index::highlight::make_snippet;
use crate::index::manifest::Schema;
use crate::index::segment::SegmentReader;
use crate::query::collector::{AggregationSegmentCollector, DocCollector};
use crate::query::filters::passes_filter;
use crate::query::sort::{SortKey, SortPlan};
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

const MAX_BUCKETS: usize = 10_000;
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

  fn accept(&mut self, doc_id: DocId) -> bool {
    match self.mode {
      SamplingMode::None => true,
      SamplingMode::Probability(p) => {
        if p <= 0.0 {
          return false;
        }
        if p >= 1.0 {
          return true;
        }
        let mut hasher = DefaultHasher::new();
        hasher.write_u64(self.seed);
        hasher.write_u64(doc_id as u64);
        let value = hasher.finish();
        let scaled = (value as f64) / (u64::MAX as f64);
        scaled < p
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
    if !self.sampler.accept(doc_id) {
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
    buckets.sort_by(|a, b| terms_bucket_cmp(&a.key, a.doc_count, &b.key, b.doc_count));
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
    if !self.sampler.accept(doc_id) {
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
    if self.digest.is_none() {
      let base = TDigest::new_with_size(TDIGEST_MAX_SIZE);
      let vals = std::mem::take(&mut self.values);
      self.digest = Some(base.merge_unsorted(vals));
    } else if !self.values.is_empty() {
      let vals = std::mem::take(&mut self.values);
      let mut digest = self.digest.take().unwrap();
      let extra = TDigest::new_with_size(TDIGEST_MAX_SIZE).merge_unsorted(vals);
      digest = TDigest::merge_digests(vec![digest, extra]);
      self.digest = Some(digest);
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
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    for _ in 0..30 {
      let mid = (lo + hi) / 2.0;
      let value = digest.estimate_quantile(mid);
      if value <= target {
        lo = mid;
      } else {
        hi = mid;
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

#[derive(Clone, Copy)]
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
    match self {
      AggregationNode::Terms(inner) => inner.collect(doc_id, score),
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

  fn collect(&mut self, doc_id: DocId, score: f32) {
    if !self.sampler.accept(doc_id) {
      return;
    }
    let values = self.ctx.fast_fields.str_values(&self.field, doc_id);
    if !values.is_empty() {
      let mut seen = HashSet::new();
      for val in values.into_iter().filter(|v| seen.insert(*v)) {
        let bucket = self.get_bucket(BucketKey::Borrowed(val), || {
          serde_json::Value::String(val.to_string())
        });
        bucket.doc_count += 1;
        for child in bucket.aggs.values_mut() {
          child.collect(doc_id, score);
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
      child.collect(doc_id, score);
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
    if !self.sampler.accept(doc_id) {
      return;
    }
    let values = numeric_values(self.ctx.fast_fields, &self.field, doc_id, self.missing);
    if values.is_empty() {
      return;
    }
    for entry in self.ranges.iter_mut() {
      if values.iter().any(|val| {
        let ge_from = entry.from.map(|f| *val >= f).unwrap_or(true);
        let lt_to = entry.to.map(|t| *val <= t).unwrap_or(true);
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
    if !self.sampler.accept(doc_id) {
      return;
    }
    let values = numeric_values(self.ctx.fast_fields, &self.field, doc_id, self.missing);
    if values.is_empty() {
      return;
    }
    let mut seen = HashSet::new();
    for val in values {
      if let Some((min, max)) = self.hard_bounds {
        if val < min || val > max {
          continue;
        }
      }
      let bucket_id = self.bucket_key(val);
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
    if let Some((min, max)) = extended_bounds.or(hard_bounds) {
      let mut bucket_id = bucket_key(min);
      let end = bucket_key(max);
      while bucket_id <= end {
        buckets.entry(bucket_id).or_insert_with(|| BucketState {
          key: serde_json::Value::Number(
            serde_json::Number::from_f64(bucket_value(bucket_id))
              .unwrap_or_else(|| serde_json::Number::from(0)),
          ),
          doc_count: 0,
          aggs: BTreeMap::new(),
        });
        bucket_id += 1;
      }
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
    if !self.sampler.accept(doc_id) {
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
      if let Some((min, max)) = self.hard_bounds {
        if val < min || val > max {
          continue;
        }
      }
      let bucket_start = match bucket_start(val, self.offset_millis, &self.interval) {
        Some(v) => v,
        None => continue,
      };
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
    if let Some((min, max)) = self.extended_bounds.or(self.hard_bounds) {
      if let (Some(mut start), Some(mut end)) = (
        bucket_start(min, self.offset_millis, &self.interval),
        bucket_start(max, self.offset_millis, &self.interval),
      ) {
        if start > end {
          std::mem::swap(&mut start, &mut end);
        }
        let mut current = start;
        while current <= end {
          buckets.entry(current).or_insert_with(|| BucketState {
            key: serde_json::Value::Number(serde_json::Number::from(current)),
            doc_count: 0,
            aggs: BTreeMap::new(),
          });
          current = match add_interval(current, &self.interval) {
            Some(next) => next,
            None => break,
          };
        }
      }
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
    if !self.sampler.accept(doc_id) {
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
    if !self.sampler.accept(doc_id) {
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
    Self {
      size: agg.size,
      from: agg.from,
      limit: agg.size.saturating_add(agg.from).max(agg.size).max(1),
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
    let start = self.from.min(ranked.len());
    let end = (start + self.size).min(ranked.len());
    let mut hits = Vec::with_capacity(end.saturating_sub(start));
    for doc in ranked.into_iter().skip(start).take(self.size) {
      let need_doc = self.fields.is_some() || self.highlight_field.is_some();
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
      target_buckets.sort_by(|a, b| terms_bucket_cmp(&a.key, a.doc_count, &b.key, b.doc_count));
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
  let start = target.from.min(hits.len());
  target.hits = hits.into_iter().skip(start).take(target.size).collect();
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
        .iter()
        .map(|b| BucketResponse {
          key: b.key.clone(),
          doc_count: b.doc_count,
          aggregations: b.aggregations.clone(),
        })
        .collect();
      let aggregations = apply_pipeline_aggs(&pipeline, &mut temp_buckets);
      let mut by_key: HashMap<String, SignificantBucketResponse> = sig_buckets
        .into_iter()
        .map(|b| (bucket_key_string(&b.key), b))
        .collect();
      let mut buckets = Vec::new();
      for bucket in temp_buckets.into_iter() {
        let key = bucket_key_string(&bucket.key);
        if let Some(mut sig) = by_key.remove(&key) {
          sig.aggregations = bucket.aggregations;
          buckets.push(sig);
        }
      }
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
    AggregationIntermediate::TopHits(state) => AggregationResponse::TopHits(TopHitsResponse {
      total: state.total,
      hits: state.hits.into_iter().map(|h| h.hit).collect(),
    }),
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

fn apply_pipeline_aggs(
  pipeline: &BTreeMap<String, Aggregation>,
  buckets: &mut Vec<BucketResponse>,
) -> BTreeMap<String, AggregationResponse> {
  let mut responses = BTreeMap::new();
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
  for (name, agg) in pipeline.iter() {
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
          name.clone(),
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
          name.clone(),
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
  let mut window_values: Vec<f64> = Vec::new();
  let window = cfg.window.max(1);
  let mut avgs = Vec::with_capacity(buckets.len());
  for (idx, bucket) in buckets.iter_mut().enumerate() {
    let current = match (series.get(idx).and_then(|v| *v), policy) {
      (Some(v), _) => Some(v),
      (None, GapPolicy::InsertZeros) => Some(0.0),
      (None, GapPolicy::Skip) => None,
    };
    if let Some(val) = current {
      window_values.push(val);
      if window_values.len() > window {
        window_values.remove(0);
      }
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
    let mut forecast_window = window_values.clone();
    let mut last_val = avgs.last().and_then(|v| *v);
    for _ in 0..predict {
      if let Some(val) = last_val {
        forecast_window.push(val);
        if forecast_window.len() > window {
          forecast_window.remove(0);
        }
        let next = forecast_window.iter().copied().sum::<f64>() / forecast_window.len() as f64;
        predictions.push(next);
        last_val = Some(next);
      } else {
        break;
      }
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
  LParen,
  RParen,
}

fn op_precedence(op: char) -> u8 {
  match op {
    '+' | '-' => 1,
    '*' | '/' => 2,
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
  while let Some(ch) = chars.peek().copied() {
    if ch.is_whitespace() {
      chars.next();
      continue;
    }
    if ch.is_ascii_digit()
      || ch == '.'
      || (ch == '-'
        && chars
          .clone()
          .nth(1)
          .map(|next| next.is_ascii_digit() || next == '.')
          .unwrap_or(false))
    {
      let mut num = String::new();
      num.push(chars.next().unwrap());
      while let Some(next) = chars.peek() {
        if next.is_ascii_digit() || *next == '.' {
          num.push(*next);
          chars.next();
        } else {
          break;
        }
      }
      let value: f64 = num.parse().ok()?;
      tokens.push(ScriptToken::Number(value));
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
      continue;
    }
    match ch {
      '+' | '-' | '*' | '/' => {
        tokens.push(ScriptToken::Op(ch));
        chars.next();
      }
      '(' => {
        tokens.push(ScriptToken::LParen);
        chars.next();
      }
      ')' => {
        tokens.push(ScriptToken::RParen);
        chars.next();
      }
      _ => return None,
    }
  }
  Some(tokens)
}

fn to_rpn(tokens: Vec<ScriptToken>) -> Option<Vec<ScriptToken>> {
  let mut output = Vec::new();
  let mut ops: Vec<char> = Vec::new();
  for token in tokens.into_iter() {
    match token {
      ScriptToken::Number(_) | ScriptToken::Var(_) => output.push(token),
      ScriptToken::Op(op) => {
        while let Some(&top) = ops.last() {
          if top == '(' {
            break;
          }
          if op_precedence(top) >= op_precedence(op) {
            output.push(ScriptToken::Op(top));
            ops.pop();
          } else {
            break;
          }
        }
        ops.push(op);
      }
      ScriptToken::LParen => ops.push('('),
      ScriptToken::RParen => {
        while let Some(op) = ops.pop() {
          if op == '(' {
            break;
          }
          output.push(ScriptToken::Op(op));
        }
      }
    }
  }
  while let Some(op) = ops.pop() {
    if op != '(' {
      output.push(ScriptToken::Op(op));
    }
  }
  Some(output)
}

fn eval_rpn(tokens: Vec<ScriptToken>, vars: &BTreeMap<String, f64>) -> Option<f64> {
  let mut stack: Vec<f64> = Vec::new();
  for token in tokens.into_iter() {
    match token {
      ScriptToken::Number(v) => stack.push(v),
      ScriptToken::Var(name) => stack.push(*vars.get(&name)?),
      ScriptToken::Op(op) => {
        let b = stack.pop()?;
        let a = stack.pop()?;
        let result = match op {
          '+' => a + b,
          '-' => a - b,
          '*' => a * b,
          '/' => {
            if b == 0.0 {
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
  let mut parts: Vec<&str> = path.split('.').collect();
  if parts.is_empty() {
    return None;
  }
  let agg_name = parts.remove(0);
  let agg = bucket.aggregations.get(agg_name)?;
  extract_metric_from_response(agg, &parts)
}

fn extract_metric_from_response(resp: &AggregationResponse, path: &[&str]) -> Option<f64> {
  match resp {
    AggregationResponse::Stats(stats) => {
      let field = path.first().copied().unwrap_or("avg");
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
      let field = path.first().copied().unwrap_or("avg");
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
      let key = path.first().copied()?;
      vals.values.get(key).copied()
    }
    AggregationResponse::PercentileRanks(vals) => {
      let key = path.first().copied()?;
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
      let bucket = ((value - offset) as f64 / *step as f64).ceil() as i64;
      Some(bucket.saturating_mul(*step) + offset)
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
      let month = date.month();
      let quarter_start = ((month - 1) / 3) * 3 + 1;
      date.with_month(quarter_start)?.with_day(1)?
    }
    CalendarUnit::Year => date.with_month(1)?.with_day(1)?,
  };
  let start_dt = start_date.and_hms_opt(0, 0, 0)?;
  Some(chrono::DateTime::<Utc>::from_naive_utc_and_offset(start_dt, Utc).timestamp_millis())
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
      date.with_year(year)?.with_month(month)?.with_day(1)?
    }
    CalendarUnit::Quarter => {
      let mut month = date.month();
      let mut year = date.year();
      month += 3;
      if month > 12 {
        month -= 12;
        year += 1;
      }
      date.with_year(year)?.with_month(month)?.with_day(1)?
    }
    CalendarUnit::Year => date
      .with_year(date.year() + 1)?
      .with_month(1)?
      .with_day(1)?,
  };
  let next_dt = next_date.and_hms_opt(0, 0, 0)?;
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
    assert!((q.percentile_rank(2.0) - 50.0).abs() < 1e-3);
  }

  #[test]
  fn bucket_script_evalutes_basic_expression() {
    let mut vars = BTreeMap::new();
    vars.insert("a".to_string(), 2.0);
    vars.insert("b".to_string(), 4.0);
    let value = eval_bucket_script("a + b * 2", &vars).unwrap();
    assert!((value - 10.0).abs() < 1e-6);
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
}
