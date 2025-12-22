use std::collections::{BTreeMap, HashMap};

use crate::api::types::{
  Aggregation, AggregationResponse, BucketResponse, DateHistogramAggregation, DateRangeAggregation,
  HistogramAggregation, RangeAggregation, StatsResponse, TermsAggregation, TopHit,
  TopHitsAggregation, TopHitsResponse, ValueCountResponse,
};
use crate::index::fastfields::FastFieldsReader;
use crate::index::segment::SegmentReader;
use crate::query::collector::{AggregationSegmentCollector, DocCollector};
use crate::DocId;

#[derive(Clone)]
pub struct AggregationContext<'a> {
  pub fast_fields: &'a FastFieldsReader,
  pub segment: &'a SegmentReader,
}

#[derive(Clone)]
pub struct BucketIntermediate {
  pub key: serde_json::Value,
  pub doc_count: u64,
  pub aggs: BTreeMap<String, AggregationIntermediate>,
}

#[derive(Clone)]
pub enum AggregationIntermediate {
  Terms {
    buckets: Vec<BucketIntermediate>,
  },
  Range {
    buckets: Vec<BucketIntermediate>,
    keyed: bool,
  },
  DateRange {
    buckets: Vec<BucketIntermediate>,
    keyed: bool,
  },
  Histogram {
    buckets: Vec<BucketIntermediate>,
  },
  DateHistogram {
    buckets: Vec<BucketIntermediate>,
  },
  Stats(StatsState),
  ExtendedStats(StatsState),
  ValueCount(ValueCountState),
  TopHits(TopHitsState),
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

#[derive(Clone)]
pub struct TopHitsState {
  pub size: usize,
  pub from: usize,
  pub hits: Vec<TopHit>,
}

pub struct SegmentAggregationCollector<'a> {
  aggs: BTreeMap<String, AggregationNode<'a>>,
}

impl<'a> SegmentAggregationCollector<'a> {
  pub fn new(aggs: BTreeMap<String, AggregationNode<'a>>) -> Self {
    Self { aggs }
  }
}

impl<'a> DocCollector for SegmentAggregationCollector<'a> {
  fn collect(&mut self, doc_id: DocId, score: f32) {
    for agg in self.aggs.values_mut() {
      agg.collect(doc_id, score);
    }
  }
}

impl<'a> AggregationSegmentCollector for SegmentAggregationCollector<'a> {
  type Output = BTreeMap<String, AggregationIntermediate>;

  fn finish(self) -> Self::Output {
    self
      .aggs
      .into_iter()
      .map(|(name, agg)| (name, agg.finish()))
      .collect()
  }
}

pub enum AggregationNode<'a> {
  Terms(Box<TermsCollector<'a>>),
  Range(Box<RangeCollector<'a>>),
  DateRange(Box<DateRangeCollector<'a>>),
  Histogram(Box<HistogramCollector<'a>>),
  DateHistogram(Box<DateHistogramCollector<'a>>),
  Stats(Box<StatsCollector<'a>>),
  ExtendedStats(Box<StatsCollector<'a>>),
  ValueCount(Box<ValueCountCollector<'a>>),
  TopHits(Box<TopHitsCollector<'a>>),
}

impl<'a> AggregationNode<'a> {
  pub fn from_request(ctx: AggregationContext<'a>, agg: &Aggregation) -> Self {
    match agg {
      Aggregation::Terms(t) => AggregationNode::Terms(Box::new(TermsCollector::new(ctx, t))),
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
    }
  }

  fn collect(&mut self, doc_id: DocId, score: f32) {
    match self {
      AggregationNode::Terms(inner) => inner.collect(doc_id, score),
      AggregationNode::Range(inner) => inner.collect(doc_id, score),
      AggregationNode::DateRange(inner) => inner.collect(doc_id, score),
      AggregationNode::Histogram(inner) => inner.collect(doc_id, score),
      AggregationNode::DateHistogram(inner) => inner.collect(doc_id, score),
      AggregationNode::Stats(inner) => inner.collect(doc_id, score),
      AggregationNode::ExtendedStats(inner) => inner.collect(doc_id, score),
      AggregationNode::ValueCount(inner) => inner.collect(doc_id, score),
      AggregationNode::TopHits(inner) => inner.collect(doc_id, score),
    }
  }

  fn finish(self) -> AggregationIntermediate {
    match self {
      AggregationNode::Terms(inner) => inner.finish(),
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
    }
  }
}

struct TermsCollector<'a> {
  field: String,
  size: Option<usize>,
  shard_size: Option<usize>,
  min_doc_count: u64,
  missing: Option<serde_json::Value>,
  buckets: HashMap<String, BucketState<'a>>,
  sub_aggs: BTreeMap<String, Aggregation>,
  ctx: AggregationContext<'a>,
}

struct BucketState<'a> {
  key: serde_json::Value,
  doc_count: u64,
  aggs: BTreeMap<String, AggregationNode<'a>>,
}

impl<'a> TermsCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &TermsAggregation) -> Self {
    let min_doc_count = agg.min_doc_count.unwrap_or(1);
    Self {
      field: agg.field.clone(),
      size: agg.size,
      shard_size: agg.shard_size,
      min_doc_count,
      missing: agg.missing.clone(),
      buckets: HashMap::new(),
      sub_aggs: agg.aggs.clone(),
      ctx,
    }
  }

  fn bucket_key(value: &serde_json::Value) -> String {
    match value {
      serde_json::Value::String(s) => s.clone(),
      other => other.to_string(),
    }
  }

  fn get_bucket(&mut self, key: serde_json::Value) -> &mut BucketState<'a> {
    let bucket_key = Self::bucket_key(&key);
    self
      .buckets
      .entry(bucket_key)
      .or_insert_with(|| BucketState {
        key: key.clone(),
        doc_count: 0,
        aggs: build_children(&self.ctx, &self.sub_aggs),
      })
  }

  fn collect(&mut self, doc_id: DocId, score: f32) {
    let key_value = if let Some(val) = self.ctx.fast_fields.str_value(&self.field, doc_id) {
      Some(serde_json::Value::String(val.to_string()))
    } else {
      self.missing.clone()
    };
    let Some(key_value) = key_value else {
      return;
    };
    let bucket = self.get_bucket(key_value);
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
    buckets.sort_by(|a, b| {
      b.doc_count
        .cmp(&a.doc_count)
        .then_with(|| Self::bucket_key(&a.key).cmp(&Self::bucket_key(&b.key)))
    });
    let limit = self
      .shard_size
      .or(self.size)
      .unwrap_or_else(|| buckets.len());
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
    }
  }
}

struct RangeCollector<'a> {
  field: String,
  keyed: bool,
  ranges: Vec<RangeEntry<'a>>,
  missing: Option<f64>,
  ctx: AggregationContext<'a>,
}

struct RangeEntry<'a> {
  key: Option<String>,
  from: Option<f64>,
  to: Option<f64>,
  bucket: BucketState<'a>,
}

impl<'a> RangeCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &RangeAggregation) -> Self {
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
          aggs: build_children(&ctx, &agg.aggs),
        },
      })
      .collect();
    let missing = agg.missing.as_ref().and_then(|v| v.as_f64());
    Self {
      field: agg.field.clone(),
      keyed: agg.keyed,
      ranges,
      missing,
      ctx,
    }
  }

  fn collect(&mut self, doc_id: DocId, score: f32) {
    let mut value = self
      .ctx
      .fast_fields
      .i64_value(&self.field, doc_id)
      .map(|v| v as f64);
    if value.is_none() {
      value = self.ctx.fast_fields.f64_value(&self.field, doc_id);
    }
    if value.is_none() {
      value = self
        .ctx
        .fast_fields
        .str_value(&self.field, doc_id)
        .and_then(|s| s.parse::<f64>().ok());
    }
    let value = value.or(self.missing);
    let Some(val) = value else {
      return;
    };
    for entry in self.ranges.iter_mut() {
      let ge_from = entry.from.map(|f| val >= f).unwrap_or(true);
      let lt_to = entry.to.map(|t| val < t).unwrap_or(true);
      if ge_from && lt_to {
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
      .map(|mut r| {
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
    }
  }
}

struct DateRangeCollector<'a> {
  inner: RangeCollector<'a>,
}

impl<'a> DateRangeCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &DateRangeAggregation) -> Self {
    let ranges = agg
      .ranges
      .iter()
      .map(|r| crate::api::types::RangeBound {
        key: r.key.clone(),
        from: r
          .from
          .as_ref()
          .and_then(|s| parse_date(s).map(|d| d as f64)),
        to: r.to.as_ref().and_then(|s| parse_date(s).map(|d| d as f64)),
      })
      .collect();
    let numeric = RangeAggregation {
      field: agg.field.clone(),
      keyed: agg.keyed,
      ranges,
      missing: agg
        .missing
        .as_ref()
        .and_then(|s| parse_date(s).map(|d| serde_json::Value::Number((d as f64).into())))
        .or_else(|| agg.missing.clone()),
      aggs: agg.aggs.clone(),
    };
    Self {
      inner: RangeCollector::new(ctx, &numeric),
    }
  }

  fn collect(&mut self, doc_id: DocId, score: f32) {
    self.inner.collect(doc_id, score);
  }

  fn finish(self) -> AggregationIntermediate {
    AggregationIntermediate::DateRange {
      keyed: self.inner.keyed,
      buckets: if let AggregationIntermediate::Range { buckets, .. } = self.inner.finish() {
        buckets
      } else {
        Vec::new()
      },
    }
  }
}

struct HistogramCollector<'a> {
  field: String,
  interval: f64,
  offset: f64,
  min_doc_count: u64,
  buckets: HashMap<i64, BucketState<'a>>,
  bounds: Option<(f64, f64)>,
  sub_aggs: BTreeMap<String, Aggregation>,
  ctx: AggregationContext<'a>,
}

impl<'a> HistogramCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &HistogramAggregation) -> Self {
    let offset = agg.offset.unwrap_or(0.0);
    let bounds = agg
      .hard_bounds
      .as_ref()
      .map(|b| (b.min, b.max))
      .or_else(|| agg.extended_bounds.as_ref().map(|b| (b.min, b.max)));
    Self {
      field: agg.field.clone(),
      interval: agg.interval,
      offset,
      min_doc_count: agg.min_doc_count.unwrap_or(1),
      buckets: HashMap::new(),
      bounds,
      sub_aggs: agg.aggs.clone(),
      ctx,
    }
  }

  fn bucket_key(&self, val: f64) -> i64 {
    ((val - self.offset) / self.interval).floor() as i64
  }

  fn collect(&mut self, doc_id: DocId, score: f32) {
    let mut value = self.ctx.fast_fields.f64_value(&self.field, doc_id);
    if value.is_none() {
      value = self
        .ctx
        .fast_fields
        .i64_value(&self.field, doc_id)
        .map(|v| v as f64);
    }
    if value.is_none() {
      value = self
        .ctx
        .fast_fields
        .str_value(&self.field, doc_id)
        .and_then(|s| s.parse::<f64>().ok());
    }
    let Some(val) = value else {
      return;
    };
    if let Some((min, max)) = self.bounds {
      if val < min || val > max {
        return;
      }
    }
    let bucket_id = self.bucket_key(val);
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
    bucket.doc_count += 1;
    for child in bucket.aggs.values_mut() {
      child.collect(doc_id, score);
    }
  }

  fn finish(self) -> AggregationIntermediate {
    let mut buckets: Vec<BucketIntermediate> = self
      .buckets
      .into_values()
      .filter(|b| b.doc_count >= self.min_doc_count)
      .map(|b| BucketIntermediate {
        key: b.key,
        doc_count: b.doc_count,
        aggs: finalize_children(b.aggs),
      })
      .collect();
    buckets.sort_by(|a, b| a.key.to_string().cmp(&b.key.to_string()));
    AggregationIntermediate::Histogram { buckets }
  }
}

struct DateHistogramCollector<'a> {
  field: String,
  interval_secs: f64,
  offset: f64,
  buckets: HashMap<i64, BucketState<'a>>,
  sub_aggs: BTreeMap<String, Aggregation>,
  ctx: AggregationContext<'a>,
}

impl<'a> DateHistogramCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &DateHistogramAggregation) -> Self {
    let interval_secs = agg
      .fixed_interval
      .as_ref()
      .and_then(|s| parse_interval_seconds(s))
      .unwrap_or(86_400.0);
    let offset = agg
      .offset
      .as_ref()
      .and_then(|s| parse_interval_seconds(s))
      .unwrap_or(0.0);
    Self {
      field: agg.field.clone(),
      interval_secs,
      offset,
      buckets: HashMap::new(),
      sub_aggs: agg.aggs.clone(),
      ctx,
    }
  }

  fn collect(&mut self, doc_id: DocId, score: f32) {
    let value = self
      .ctx
      .fast_fields
      .str_value(&self.field, doc_id)
      .and_then(parse_date);
    let Some(val) = value else {
      return;
    };
    let bucket = (((val as f64) - self.offset) / self.interval_secs).floor() as i64;
    let bucket_entry = self.buckets.entry(bucket).or_insert_with(|| BucketState {
      key: serde_json::Value::Number(
        serde_json::Number::from_f64(bucket as f64 * self.interval_secs + self.offset)
          .unwrap_or_else(|| serde_json::Number::from(0)),
      ),
      doc_count: 0,
      aggs: build_children(&self.ctx, &self.sub_aggs),
    });
    bucket_entry.doc_count += 1;
    for child in bucket_entry.aggs.values_mut() {
      child.collect(doc_id, score);
    }
  }

  fn finish(self) -> AggregationIntermediate {
    let mut buckets: Vec<BucketIntermediate> = self
      .buckets
      .into_values()
      .map(|b| BucketIntermediate {
        key: b.key,
        doc_count: b.doc_count,
        aggs: finalize_children(b.aggs),
      })
      .collect();
    buckets.sort_by(|a, b| a.key.to_string().cmp(&b.key.to_string()));
    AggregationIntermediate::DateHistogram { buckets }
  }
}

struct StatsCollector<'a> {
  field: String,
  missing: Option<f64>,
  stats: StatsState,
  ctx: AggregationContext<'a>,
}

impl<'a> StatsCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &crate::api::types::MetricAggregation) -> Self {
    Self {
      field: agg.field.clone(),
      missing: agg.missing.as_ref().and_then(|v| v.as_f64()),
      stats: StatsState::default(),
      ctx,
    }
  }

  fn collect(&mut self, doc_id: DocId, _score: f32) {
    let mut value = self.ctx.fast_fields.f64_value(&self.field, doc_id);
    if value.is_none() {
      value = self
        .ctx
        .fast_fields
        .i64_value(&self.field, doc_id)
        .map(|v| v as f64);
    }
    if value.is_none() {
      value = self
        .ctx
        .fast_fields
        .str_value(&self.field, doc_id)
        .and_then(|s| s.parse::<f64>().ok());
    }
    let Some(val) = value.or(self.missing) else {
      return;
    };
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

  fn finish(self) -> StatsState {
    self.stats
  }
}

struct ValueCountCollector<'a> {
  field: String,
  missing: Option<f64>,
  state: ValueCountState,
  ctx: AggregationContext<'a>,
}

impl<'a> ValueCountCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &crate::api::types::MetricAggregation) -> Self {
    Self {
      field: agg.field.clone(),
      missing: agg.missing.as_ref().and_then(|v| v.as_f64()),
      state: ValueCountState::default(),
      ctx,
    }
  }

  fn collect(&mut self, doc_id: DocId, _score: f32) {
    let mut present = self
      .ctx
      .fast_fields
      .f64_value(&self.field, doc_id)
      .is_some();
    present = present
      || self
        .ctx
        .fast_fields
        .i64_value(&self.field, doc_id)
        .is_some();
    present = present
      || self
        .ctx
        .fast_fields
        .str_value(&self.field, doc_id)
        .is_some();
    if present || self.missing.is_some() {
      self.state.value += 1;
    }
  }

  fn finish(self) -> ValueCountState {
    self.state
  }
}

struct TopHitsCollector<'a> {
  size: usize,
  from: usize,
  hits: Vec<TopHit>,
  _ctx: AggregationContext<'a>,
}

impl<'a> TopHitsCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &TopHitsAggregation) -> Self {
    Self {
      size: agg.size,
      from: agg.from,
      hits: Vec::new(),
      _ctx: ctx,
    }
  }

  fn collect(&mut self, doc_id: DocId, score: f32) {
    self.hits.push(TopHit {
      doc_id,
      score: Some(score),
      fields: None,
      snippet: None,
    });
  }

  fn finish(mut self) -> TopHitsState {
    self.hits.sort_by(|a, b| {
      b.score
        .partial_cmp(&a.score)
        .unwrap_or(std::cmp::Ordering::Equal)
    });
    let total = self.hits.len();
    let start = self.from.min(total);
    let end = (start + self.size).min(total);
    TopHitsState {
      size: self.size,
      from: self.from,
      hits: self.hits[start..end].to_vec(),
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
      merged
        .entry(name)
        .and_modify(|existing| *existing = merge_intermediate(existing.clone(), agg.clone()))
        .or_insert(agg);
    }
  }
  merged
    .into_iter()
    .map(|(name, agg)| (name, finalize_response(agg)))
    .collect()
}

fn merge_intermediate(
  a: AggregationIntermediate,
  b: AggregationIntermediate,
) -> AggregationIntermediate {
  match (a, b) {
    (
      AggregationIntermediate::Terms { buckets: mut a },
      AggregationIntermediate::Terms { buckets: b },
    ) => {
      merge_bucket_lists(&mut a, b);
      AggregationIntermediate::Terms { buckets: a }
    }
    (
      AggregationIntermediate::Range {
        buckets: mut a,
        keyed,
      },
      AggregationIntermediate::Range { buckets: b, .. },
    ) => {
      merge_bucket_lists(&mut a, b);
      AggregationIntermediate::Range { buckets: a, keyed }
    }
    (
      AggregationIntermediate::DateRange {
        buckets: mut a,
        keyed,
      },
      AggregationIntermediate::DateRange { buckets: b, .. },
    ) => {
      merge_bucket_lists(&mut a, b);
      AggregationIntermediate::DateRange { buckets: a, keyed }
    }
    (
      AggregationIntermediate::Histogram { buckets: mut a },
      AggregationIntermediate::Histogram { buckets: b },
    ) => {
      merge_bucket_lists(&mut a, b);
      AggregationIntermediate::Histogram { buckets: a }
    }
    (
      AggregationIntermediate::DateHistogram { buckets: mut a },
      AggregationIntermediate::DateHistogram { buckets: b },
    ) => {
      merge_bucket_lists(&mut a, b);
      AggregationIntermediate::DateHistogram { buckets: a }
    }
    (AggregationIntermediate::Stats(a), AggregationIntermediate::Stats(b)) => {
      AggregationIntermediate::Stats(merge_stats(a, b))
    }
    (AggregationIntermediate::ExtendedStats(a), AggregationIntermediate::ExtendedStats(b)) => {
      AggregationIntermediate::ExtendedStats(merge_stats(a, b))
    }
    (AggregationIntermediate::ValueCount(a), AggregationIntermediate::ValueCount(b)) => {
      AggregationIntermediate::ValueCount(ValueCountState {
        value: a.value + b.value,
      })
    }
    (AggregationIntermediate::TopHits(mut a), AggregationIntermediate::TopHits(mut b)) => {
      a.hits.append(&mut b.hits);
      AggregationIntermediate::TopHits(a)
    }
    (left, _) => left,
  }
}

fn merge_bucket_lists(target: &mut Vec<BucketIntermediate>, incoming: Vec<BucketIntermediate>) {
  for bucket in incoming.into_iter() {
    if let Some(existing) = target.iter_mut().find(|b| b.key == bucket.key) {
      existing.doc_count += bucket.doc_count;
      let mut combined = existing.aggs.clone();
      for (name, agg) in bucket.aggs.into_iter() {
        combined
          .entry(name)
          .and_modify(|existing| *existing = merge_intermediate(existing.clone(), agg.clone()))
          .or_insert(agg);
      }
      existing.aggs = combined;
    } else {
      target.push(bucket);
    }
  }
}

fn finalize_response(intermediate: AggregationIntermediate) -> AggregationResponse {
  match intermediate {
    AggregationIntermediate::Terms { mut buckets } => {
      buckets.sort_by(|a, b| {
        b.doc_count
          .cmp(&a.doc_count)
          .then_with(|| a.key.to_string().cmp(&b.key.to_string()))
      });
      AggregationResponse::Terms {
        buckets: buckets.into_iter().map(finalize_bucket).collect(),
      }
    }
    AggregationIntermediate::Range { buckets, keyed } => AggregationResponse::Range {
      buckets: buckets.into_iter().map(finalize_bucket).collect(),
      keyed,
    },
    AggregationIntermediate::DateRange { buckets, keyed } => AggregationResponse::DateRange {
      buckets: buckets.into_iter().map(finalize_bucket).collect(),
      keyed,
    },
    AggregationIntermediate::Histogram { buckets } => AggregationResponse::Histogram {
      buckets: buckets.into_iter().map(finalize_bucket).collect(),
    },
    AggregationIntermediate::DateHistogram { buckets } => AggregationResponse::DateHistogram {
      buckets: buckets.into_iter().map(finalize_bucket).collect(),
    },
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
    AggregationIntermediate::TopHits(state) => AggregationResponse::TopHits(TopHitsResponse {
      total: state.hits.len() as u64,
      hits: state.hits,
    }),
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

fn parse_date(value: &str) -> Option<f64> {
  chrono::DateTime::parse_from_rfc3339(value)
    .map(|dt| dt.timestamp_millis() as f64)
    .ok()
}

fn parse_interval_seconds(spec: &str) -> Option<f64> {
  let digits: String = spec.chars().take_while(|c| c.is_ascii_digit()).collect();
  let value: f64 = digits.parse().ok()?;
  let suffix: String = spec.chars().skip_while(|c| c.is_ascii_digit()).collect();
  let mult = match suffix.as_str() {
    "ms" => 0.001,
    "s" => 1.0,
    "m" => 60.0,
    "h" => 3600.0,
    "d" => 86_400.0,
    _ => 1.0,
  };
  Some(value * mult)
}
