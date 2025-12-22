use std::cmp::Ordering;
use std::collections::{
  btree_map::Entry as BTreeEntry, hash_map::Entry as HashEntry, BTreeMap, BinaryHeap, HashMap,
};
use std::hash::{Hash, Hasher};

use crate::api::types::{
  Aggregation, AggregationResponse, BucketResponse, DateHistogramAggregation, DateRangeAggregation,
  HistogramAggregation, RangeAggregation, StatsResponse, TermsAggregation, TopHit,
  TopHitsAggregation, TopHitsResponse, ValueCountResponse,
};
use crate::index::fastfields::FastFieldsReader;
use crate::index::highlight::make_snippet;
use crate::index::segment::SegmentReader;
use crate::query::collector::{AggregationSegmentCollector, DocCollector};
use crate::DocId;

#[derive(Clone)]
pub struct AggregationContext<'a> {
  pub fast_fields: &'a FastFieldsReader,
  pub segment: &'a SegmentReader,
  pub highlight_terms: &'a [String],
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
    size: Option<usize>,
    shard_size: Option<usize>,
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

fn numeric_value(
  fast_fields: &FastFieldsReader,
  field: &str,
  doc_id: DocId,
  missing: Option<f64>,
) -> Option<f64> {
  fast_fields
    .f64_value(field, doc_id)
    .or_else(|| fast_fields.i64_value(field, doc_id).map(|v| v as f64))
    .or(missing)
}

fn has_numeric_value(
  fast_fields: &FastFieldsReader,
  field: &str,
  doc_id: DocId,
  allow_missing: bool,
) -> bool {
  fast_fields.f64_value(field, doc_id).is_some()
    || fast_fields.i64_value(field, doc_id).is_some()
    || allow_missing
}

#[derive(Clone)]
pub struct TopHitsState {
  pub size: usize,
  pub from: usize,
  pub total: u64,
  pub hits: Vec<TopHit>,
}

#[derive(Clone, Copy)]
enum DateInterval {
  Fixed(i64),
  Calendar(CalendarUnit),
}

#[derive(Clone, Copy)]
enum CalendarUnit {
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

pub(crate) struct TermsCollector<'a> {
  field: String,
  size: Option<usize>,
  shard_size: Option<usize>,
  min_doc_count: u64,
  missing: Option<serde_json::Value>,
  missing_key: Option<String>,
  buckets: HashMap<BucketKey<'a>, BucketState<'a>>,
  sub_aggs: BTreeMap<String, Aggregation>,
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
      sub_aggs: agg.aggs.clone(),
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
    if let Some(val) = self.ctx.fast_fields.str_value(&self.field, doc_id) {
      let bucket = self.get_bucket(BucketKey::Borrowed(val), || {
        serde_json::Value::String(val.to_string())
      });
      bucket.doc_count += 1;
      for child in bucket.aggs.values_mut() {
        child.collect(doc_id, score);
      }
      return;
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
    buckets.sort_by(|a, b| {
      b.doc_count
        .cmp(&a.doc_count)
        .then_with(|| bucket_key_string(&a.key).cmp(&bucket_key_string(&b.key)))
    });
    let limit = self.shard_size.or(self.size).unwrap_or(buckets.len());
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
    }
  }
}

pub(crate) struct RangeCollector<'a> {
  field: String,
  keyed: bool,
  ranges: Vec<RangeEntry<'a>>,
  missing: Option<f64>,
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
    let missing = agg.missing.as_ref().and_then(|v| {
      v.as_f64()
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
    });
    Self {
      field: agg.field.clone(),
      keyed: agg.keyed,
      ranges,
      missing,
      ctx,
    }
  }

  fn collect(&mut self, doc_id: DocId, score: f32) {
    let value = numeric_value(self.ctx.fast_fields, &self.field, doc_id, self.missing);
    let Some(val) = value else {
      return;
    };
    for entry in self.ranges.iter_mut() {
      let ge_from = entry.from.map(|f| val >= f).unwrap_or(true);
      let lt_to = entry.to.map(|t| val <= t).unwrap_or(true);
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
  ctx: AggregationContext<'a>,
}

impl<'a> HistogramCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &HistogramAggregation) -> Self {
    let offset = agg.offset.unwrap_or(0.0);
    let extended_bounds = agg.extended_bounds.as_ref().map(|b| (b.min, b.max));
    let hard_bounds = agg.hard_bounds.as_ref().map(|b| (b.min, b.max));
    Self {
      field: agg.field.clone(),
      interval: agg.interval,
      offset,
      min_doc_count: agg.min_doc_count.unwrap_or(1),
      buckets: HashMap::new(),
      extended_bounds,
      hard_bounds,
      missing: agg.missing,
      sub_aggs: agg.aggs.clone(),
      ctx,
    }
  }

  fn bucket_key(&self, val: f64) -> i64 {
    ((val - self.offset) / self.interval).ceil() as i64
  }

  fn collect(&mut self, doc_id: DocId, score: f32) {
    let value = numeric_value(self.ctx.fast_fields, &self.field, doc_id, self.missing);
    let Some(val) = value else {
      return;
    };
    if let Some((min, max)) = self.hard_bounds {
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
    if bucket.aggs.is_empty() && !self.sub_aggs.is_empty() {
      bucket.aggs = build_children(&self.ctx, &self.sub_aggs);
    }
    bucket.doc_count += 1;
    for child in bucket.aggs.values_mut() {
      child.collect(doc_id, score);
    }
  }

  fn finish(self) -> AggregationIntermediate {
    let interval = self.interval;
    let offset = self.offset;
    let min_doc_count = self.min_doc_count;
    let extended_bounds = self.extended_bounds;
    let hard_bounds = self.hard_bounds;
    let mut buckets = self.buckets;
    let bucket_key = |val: f64| ((val - offset) / interval).ceil() as i64;
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
    AggregationIntermediate::Histogram { buckets }
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
  ctx: AggregationContext<'a>,
}

impl<'a> DateHistogramCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &DateHistogramAggregation) -> Self {
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
      sub_aggs: agg.aggs.clone(),
      ctx,
    }
  }

  fn collect(&mut self, doc_id: DocId, score: f32) {
    let value = numeric_value(
      self.ctx.fast_fields,
      &self.field,
      doc_id,
      self.missing.map(|v| v as f64),
    )
    .map(|v| v as i64);
    let Some(val) = value else {
      return;
    };
    if let Some((min, max)) = self.hard_bounds {
      if val < min || val > max {
        return;
      }
    }
    let bucket_start = match bucket_start(val, self.offset_millis, &self.interval) {
      Some(v) => v,
      None => return,
    };
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
    AggregationIntermediate::DateHistogram { buckets }
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
    let value = numeric_value(self.ctx.fast_fields, &self.field, doc_id, self.missing);
    let Some(val) = value else {
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
    let present = has_numeric_value(
      self.ctx.fast_fields,
      &self.field,
      doc_id,
      self.missing.is_some(),
    );
    if present {
      self.state.value += 1;
    }
  }

  fn finish(self) -> ValueCountState {
    self.state
  }
}

#[derive(Clone, Copy, Debug)]
struct ScoredHit {
  doc_id: DocId,
  score: f32,
}

impl Eq for ScoredHit {}

impl PartialEq for ScoredHit {
  fn eq(&self, other: &Self) -> bool {
    self.doc_id == other.doc_id && self.score.to_bits() == other.score.to_bits()
  }
}

impl Ord for ScoredHit {
  fn cmp(&self, other: &Self) -> Ordering {
    self
      .score
      .total_cmp(&other.score)
      .then_with(|| self.doc_id.cmp(&other.doc_id))
  }
}

impl PartialOrd for ScoredHit {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

pub(crate) struct TopHitsCollector<'a> {
  size: usize,
  from: usize,
  limit: usize,
  heap: BinaryHeap<std::cmp::Reverse<ScoredHit>>,
  total: u64,
  fields: Option<Vec<String>>,
  highlight_field: Option<String>,
  highlight_terms: &'a [String],
  ctx: AggregationContext<'a>,
}

impl<'a> TopHitsCollector<'a> {
  fn new(ctx: AggregationContext<'a>, agg: &TopHitsAggregation) -> Self {
    Self {
      size: agg.size,
      from: agg.from,
      limit: agg.size.saturating_add(agg.from).max(agg.size).max(1),
      heap: BinaryHeap::new(),
      total: 0,
      fields: agg.fields.clone(),
      highlight_field: agg.highlight_field.clone(),
      highlight_terms: ctx.highlight_terms,
      ctx,
    }
  }

  fn collect(&mut self, doc_id: DocId, score: f32) {
    self.total += 1;
    self
      .heap
      .push(std::cmp::Reverse(ScoredHit { doc_id, score }));
    if self.heap.len() > self.limit {
      self.heap.pop();
    }
  }

  fn finish(mut self) -> TopHitsState {
    let mut scored: Vec<ScoredHit> = self.heap.drain().map(|r| r.0).collect();
    scored.sort_by(|a, b| {
      b.score
        .total_cmp(&a.score)
        .then_with(|| a.doc_id.cmp(&b.doc_id))
    });
    let start = self.from.min(scored.len());
    let end = (start + self.size).min(scored.len());
    let mut hits = Vec::with_capacity(end.saturating_sub(start));
    for h in scored.into_iter().skip(start).take(self.size) {
      let doc = self.ctx.segment.get_doc(h.doc_id).ok();
      let fields_val = doc.as_ref().and_then(|d| {
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
      let snippet = if let (Some(field), Some(doc_val)) = (&self.highlight_field, doc.as_ref()) {
        if let Some(text) = doc_val.get(field).and_then(|v| v.as_str()) {
          make_snippet(text, self.highlight_terms)
        } else {
          None
        }
      } else {
        None
      };
      hits.push(TopHit {
        doc_id: h.doc_id,
        score: Some(h.score),
        fields: fields_val,
        snippet,
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
      },
      AggregationIntermediate::Terms {
        buckets: incoming_buckets,
        size: incoming_size,
        shard_size: incoming_shard,
      },
    ) => {
      merge_bucket_lists(target_buckets, incoming_buckets);
      if size.is_none() {
        *size = incoming_size;
      }
      if shard_size.is_none() {
        *shard_size = incoming_shard;
      }
    }
    (
      AggregationIntermediate::Range {
        buckets: target_buckets,
        ..
      },
      AggregationIntermediate::Range {
        buckets: incoming_buckets,
        ..
      },
    ) => {
      merge_bucket_lists(target_buckets, incoming_buckets);
    }
    (
      AggregationIntermediate::DateRange {
        buckets: target_buckets,
        ..
      },
      AggregationIntermediate::DateRange {
        buckets: incoming_buckets,
        ..
      },
    ) => {
      merge_bucket_lists(target_buckets, incoming_buckets);
    }
    (
      AggregationIntermediate::Histogram {
        buckets: target_buckets,
      },
      AggregationIntermediate::Histogram {
        buckets: incoming_buckets,
      },
    ) => merge_bucket_lists(target_buckets, incoming_buckets),
    (
      AggregationIntermediate::DateHistogram {
        buckets: target_buckets,
      },
      AggregationIntermediate::DateHistogram {
        buckets: incoming_buckets,
      },
    ) => merge_bucket_lists(target_buckets, incoming_buckets),
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
      AggregationIntermediate::TopHits(target_hits),
      AggregationIntermediate::TopHits(incoming_hits),
    ) => merge_top_hits(target_hits, incoming_hits),
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

fn merge_top_hits(target: &mut TopHitsState, incoming: TopHitsState) {
  let limit = target
    .size
    .saturating_add(target.from)
    .max(target.size)
    .max(1);
  target.total += incoming.total;
  #[derive(Clone, Debug)]
  struct Ranked {
    score: f32,
    hit: TopHit,
    doc_id: crate::DocId,
  }
  impl PartialEq for Ranked {
    fn eq(&self, other: &Self) -> bool {
      self.score.to_bits() == other.score.to_bits() && self.doc_id == other.doc_id
    }
  }
  impl Eq for Ranked {}
  impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
      Some(self.cmp(other))
    }
  }
  impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> Ordering {
      self
        .score
        .total_cmp(&other.score)
        .then_with(|| other.doc_id.cmp(&self.doc_id))
    }
  }
  let mut heap: BinaryHeap<std::cmp::Reverse<Ranked>> = BinaryHeap::with_capacity(limit + 1);
  let mut push_hit = |hit: TopHit| {
    let ranked = Ranked {
      score: hit.score.unwrap_or(0.0),
      doc_id: hit.doc_id,
      hit,
    };
    heap.push(std::cmp::Reverse(ranked));
    if heap.len() > limit {
      heap.pop();
    }
  };
  for hit in target.hits.drain(..) {
    push_hit(hit);
  }
  for hit in incoming.hits {
    push_hit(hit);
  }
  let mut hits: Vec<_> = heap.into_iter().map(|r| r.0).collect();
  hits.sort_by(|a, b| {
    b.score
      .total_cmp(&a.score)
      .then_with(|| a.doc_id.cmp(&b.doc_id))
  });
  let start = target.from.min(hits.len());
  target.hits = hits
    .into_iter()
    .skip(start)
    .take(target.size)
    .map(|r| r.hit)
    .collect();
}

fn bucket_key_string(key: &serde_json::Value) -> String {
  if let Some(s) = key.as_str() {
    s.to_string()
  } else {
    key.to_string()
  }
}

fn finalize_response(intermediate: AggregationIntermediate) -> AggregationResponse {
  match intermediate {
    AggregationIntermediate::Terms {
      mut buckets,
      size,
      shard_size,
    } => {
      buckets.sort_by(|a, b| {
        b.doc_count
          .cmp(&a.doc_count)
          .then_with(|| bucket_key_string(&a.key).cmp(&bucket_key_string(&b.key)))
      });
      let limit = size.or(shard_size).unwrap_or(buckets.len());
      if buckets.len() > limit {
        buckets.truncate(limit);
      }
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
      total: state.total,
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

fn cmp_bucket_value(a: &serde_json::Value, b: &serde_json::Value) -> Ordering {
  if let (Some(va), Some(vb)) = (a.as_f64(), b.as_f64()) {
    return va.partial_cmp(&vb).unwrap_or(Ordering::Equal);
  }
  a.to_string().cmp(&b.to_string())
}

fn parse_calendar_interval(spec: &str) -> Option<CalendarUnit> {
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

fn parse_date(value: &str) -> Option<f64> {
  chrono::DateTime::parse_from_rfc3339(value)
    .map(|dt| dt.timestamp_millis() as f64)
    .ok()
    .or_else(|| value.parse::<f64>().ok())
}

fn parse_interval_seconds(spec: &str) -> Option<f64> {
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
    "ms" => 0.001,
    "s" => 1.0,
    "m" => 60.0,
    "h" => 3600.0,
    "d" => 86_400.0,
    "w" => 604_800.0,
    _ => 1.0,
  };
  Some(value * mult)
}
