use std::collections::{BTreeMap, HashMap, HashSet};

use crate::api::types::{AggregationOp, AggregationRequest, FacetKind, FacetRequest};
use crate::index::fastfields::FastFieldsReader;
use crate::index::manifest::Schema;
use crate::DocId;

#[derive(Debug, Clone)]
pub struct FacetPlan {
  configs: Vec<FacetConfig>,
}

impl FacetPlan {
  pub fn new(reqs: &[FacetRequest], schema: &Schema) -> Option<Self> {
    if reqs.is_empty() {
      return None;
    }
    let numeric_types: HashMap<&str, bool> = schema
      .numeric_fields
      .iter()
      .map(|f| (f.name.as_str(), f.i64))
      .collect();
    let mut configs = Vec::with_capacity(reqs.len());
    for r in reqs.iter() {
      let config = match &r.facet {
        FacetKind::Keyword => FacetConfig::Keyword {
          field: r.field.clone(),
        },
        FacetKind::Range { ranges } => {
          let bucketed: Vec<RangeBucket> = ranges
            .iter()
            .map(|rng| RangeBucket::new(rng.min, rng.max))
            .collect();
          FacetConfig::NumericRanges {
            field: r.field.clone(),
            buckets: bucketed,
          }
        }
        FacetKind::Histogram { interval, min } => FacetConfig::Histogram {
          field: r.field.clone(),
          interval: *interval,
          min: *min,
          is_i64: *numeric_types.get(r.field.as_str()).unwrap_or(&false),
        },
      };
      configs.push(config);
    }
    Some(Self { configs })
  }

  pub fn collector<'a>(&'a self, reader: &'a FastFieldsReader) -> FacetCollector<'a> {
    FacetCollector {
      reader,
      configs: self.configs.as_slice(),
      counts: vec![BTreeMap::new(); self.configs.len()],
    }
  }

  pub fn len(&self) -> usize {
    self.configs.len()
  }

  pub(crate) fn configs(&self) -> &[FacetConfig] {
    &self.configs
  }
}

#[derive(Debug)]
pub struct FacetCollector<'a> {
  reader: &'a FastFieldsReader,
  configs: &'a [FacetConfig],
  counts: Vec<BTreeMap<String, u64>>, // aligned with configs
}

impl<'a> FacetCollector<'a> {
  pub fn collect(&mut self, doc_id: DocId) {
    for (idx, config) in self.configs.iter().enumerate() {
      match config {
        FacetConfig::Keyword { field } => {
          if let Some(val) = self.reader.keyword_value(field, doc_id) {
            *self.counts[idx].entry(val.to_string()).or_insert(0) += 1;
          }
        }
        FacetConfig::NumericRanges { field, buckets } => {
          let value = self
            .reader
            .i64_value(field, doc_id)
            .map(|v| v as f64)
            .or_else(|| self.reader.f64_value(field, doc_id));
          if let Some(val) = value {
            for bucket in buckets.iter() {
              if bucket.contains(val) {
                *self.counts[idx].entry(bucket.label.clone()).or_insert(0) += 1;
              }
            }
          }
        }
        FacetConfig::Histogram {
          field,
          interval,
          min,
          is_i64,
        } => {
          let raw = if *is_i64 {
            self.reader.i64_value(field, doc_id).map(|v| v as f64)
          } else {
            self.reader.f64_value(field, doc_id)
          };
          if let Some(val) = raw {
            let start = min.unwrap_or(0.0);
            let offset_val = val - start;
            let bucket = (offset_val / interval).floor();
            let bucket_start = start + bucket * interval;
            let bucket_end = bucket_start + interval;
            let label = format!("{}..{}", bucket_start, bucket_end);
            *self.counts[idx].entry(label).or_insert(0) += 1;
          }
        }
      }
    }
  }

  pub fn into_counts(self) -> Vec<BTreeMap<String, u64>> {
    self.counts
  }
}

#[derive(Debug, Clone)]
pub(crate) enum FacetConfig {
  Keyword {
    field: String,
  },
  NumericRanges {
    field: String,
    buckets: Vec<RangeBucket>,
  },
  Histogram {
    field: String,
    interval: f64,
    min: Option<f64>,
    is_i64: bool,
  },
}

impl FacetConfig {
  pub(crate) fn field_name(&self) -> &str {
    match self {
      FacetConfig::Keyword { field }
      | FacetConfig::NumericRanges { field, .. }
      | FacetConfig::Histogram { field, .. } => field.as_str(),
    }
  }
}

#[derive(Debug, Clone)]
struct RangeBucket {
  min: f64,
  max: f64,
  label: String,
}

impl RangeBucket {
  fn new(min: f64, max: f64) -> Self {
    Self {
      min,
      max,
      label: format!("{}..{}", min, max),
    }
  }

  fn contains(&self, v: f64) -> bool {
    v >= self.min && v < self.max
  }
}

#[derive(Debug, Clone)]
pub struct AggregationPlan {
  states: Vec<AggregationState>,
}

impl AggregationPlan {
  pub fn new(reqs: &[AggregationRequest], schema: &Schema) -> Option<Self> {
    if reqs.is_empty() {
      return None;
    }
    let mut numeric_types: HashMap<&str, bool> = HashMap::new();
    for f in schema.numeric_fields.iter() {
      numeric_types.insert(f.name.as_str(), f.i64);
    }
    let mut states = Vec::with_capacity(reqs.len());
    for req in reqs.iter() {
      let is_i64 = *numeric_types.get(req.field.as_str()).unwrap_or(&false);
      let ops: HashSet<AggregationOp> = if req.operations.is_empty() {
        [
          AggregationOp::Min,
          AggregationOp::Max,
          AggregationOp::Sum,
          AggregationOp::Avg,
        ]
        .into_iter()
        .collect()
      } else {
        req.operations.iter().cloned().collect()
      };
      states.push(AggregationState::new(req.field.clone(), ops, is_i64));
    }
    Some(Self { states })
  }

  pub fn collector<'a>(&self, reader: &'a FastFieldsReader) -> AggregationCollector<'a> {
    AggregationCollector {
      reader,
      states: self.states.clone(),
    }
  }

  pub fn len(&self) -> usize {
    self.states.len()
  }
}

#[derive(Debug)]
pub struct AggregationCollector<'a> {
  reader: &'a FastFieldsReader,
  states: Vec<AggregationState>,
}

impl<'a> AggregationCollector<'a> {
  pub fn collect(&mut self, doc_id: DocId) {
    for state in self.states.iter_mut() {
      let val = if state.is_i64 {
        self
          .reader
          .i64_value(&state.field, doc_id)
          .map(|v| v as f64)
      } else {
        self.reader.f64_value(&state.field, doc_id)
      };
      if let Some(v) = val {
        state.apply(v);
      }
    }
  }

  pub fn into_states(self) -> Vec<AggregationState> {
    self.states
  }
}

#[derive(Debug, Clone)]
pub struct AggregationState {
  pub field: String,
  pub doc_count: u64,
  pub min: Option<f64>,
  pub max: Option<f64>,
  pub sum_f64: Option<f64>,
  pub sum_i128: Option<i128>,
  pub ops: HashSet<AggregationOp>,
  pub is_i64: bool,
}

impl AggregationState {
  fn new(field: String, ops: HashSet<AggregationOp>, is_i64: bool) -> Self {
    Self {
      field,
      doc_count: 0,
      min: None,
      max: None,
      sum_f64: None,
      sum_i128: None,
      ops,
      is_i64,
    }
  }

  fn apply(&mut self, value: f64) {
    self.doc_count += 1;
    if self.ops.contains(&AggregationOp::Min) {
      self.min = Some(self.min.map(|m| m.min(value)).unwrap_or(value));
    }
    if self.ops.contains(&AggregationOp::Max) {
      self.max = Some(self.max.map(|m| m.max(value)).unwrap_or(value));
    }
    if self.ops.contains(&AggregationOp::Sum) || self.ops.contains(&AggregationOp::Avg) {
      if self.is_i64 {
        let as_i128 = value as i128;
        self.sum_i128 = Some(self.sum_i128.unwrap_or(0) + as_i128);
      } else {
        self.sum_f64 = Some(self.sum_f64.unwrap_or(0.0) + value);
      }
    }
  }

  pub fn merge_from(&mut self, other: &AggregationState) {
    self.doc_count += other.doc_count;
    if self.ops.contains(&AggregationOp::Min) {
      if let Some(min) = other.min {
        self.min = Some(self.min.map(|m| m.min(min)).unwrap_or(min));
      }
    }
    if self.ops.contains(&AggregationOp::Max) {
      if let Some(max) = other.max {
        self.max = Some(self.max.map(|m| m.max(max)).unwrap_or(max));
      }
    }
    if self.ops.contains(&AggregationOp::Sum) || self.ops.contains(&AggregationOp::Avg) {
      if self.is_i64 {
        if let Some(part) = other.sum_i128 {
          self.sum_i128 = Some(self.sum_i128.unwrap_or(0) + part);
        }
      } else if let Some(part) = other.sum_f64 {
        self.sum_f64 = Some(self.sum_f64.unwrap_or(0.0) + part);
      }
    }
  }

  pub fn finalize(&self) -> AggregationOutput {
    let sum = if self.ops.contains(&AggregationOp::Sum) || self.ops.contains(&AggregationOp::Avg) {
      if self.is_i64 {
        self.sum_i128.map(|v| v as f64)
      } else {
        self.sum_f64
      }
    } else {
      None
    };
    let avg = if self.ops.contains(&AggregationOp::Avg) {
      match (sum, self.doc_count) {
        (Some(total), count) if count > 0 => Some(total / count as f64),
        _ => None,
      }
    } else {
      None
    };
    AggregationOutput {
      field: self.field.clone(),
      doc_count: self.doc_count,
      min: if self.ops.contains(&AggregationOp::Min) {
        self.min
      } else {
        None
      },
      max: if self.ops.contains(&AggregationOp::Max) {
        self.max
      } else {
        None
      },
      sum: if self.ops.contains(&AggregationOp::Sum) {
        sum
      } else {
        None
      },
      avg,
    }
  }
}

#[derive(Debug, Clone)]
pub struct AggregationOutput {
  pub field: String,
  pub doc_count: u64,
  pub min: Option<f64>,
  pub max: Option<f64>,
  pub sum: Option<f64>,
  pub avg: Option<f64>,
}
