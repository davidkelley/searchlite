use std::sync::Arc;

use crate::api::types::VectorMetric;

pub mod hnsw;
pub mod quant;

pub const DEFAULT_VECTOR_ALPHA: f32 = 0.5;

#[derive(Debug, Clone)]
pub struct VectorStore {
  dim: usize,
  metric: VectorMetric,
  offsets: Vec<u32>,
  values: Arc<Vec<f32>>,
  present: usize,
}

impl VectorStore {
  pub fn new(dim: usize, metric: VectorMetric, offsets: Vec<u32>, values: Vec<f32>) -> Self {
    debug_assert!(
      dim == 0 || (values.len().checked_rem(dim) == Some(0)),
      "vector store values must align to dim"
    );
    let present = offsets.iter().filter(|&&off| off != u32::MAX).count();
    Self {
      dim,
      metric,
      offsets,
      values: Arc::new(values),
      present,
    }
  }

  pub fn dim(&self) -> usize {
    self.dim
  }

  pub fn metric(&self) -> VectorMetric {
    self.metric.clone()
  }

  pub fn len(&self) -> usize {
    self.offsets.len()
  }

  pub fn is_empty(&self) -> bool {
    self.present == 0
  }

  pub fn present(&self) -> usize {
    self.present
  }

  pub fn offsets(&self) -> &[u32] {
    &self.offsets
  }

  pub fn values(&self) -> Arc<Vec<f32>> {
    self.values.clone()
  }

  pub fn vector(&self, doc_id: u32) -> Option<&[f32]> {
    let idx = self.offsets.get(doc_id as usize)?;
    if *idx == u32::MAX {
      return None;
    }
    let start = (*idx as usize).saturating_mul(self.dim);
    let end = start + self.dim;
    self.values.get(start..end)
  }
}

pub fn normalize_in_place(vec: &mut [f32]) {
  let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
  if norm > 0.0 {
    for v in vec.iter_mut() {
      *v /= norm;
    }
  }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
  let mut dot = 0.0f32;
  let mut norm_a = 0.0f32;
  let mut norm_b = 0.0f32;
  for (x, y) in a.iter().zip(b.iter()) {
    dot += x * y;
    norm_a += x * x;
    norm_b += y * y;
  }
  if norm_a == 0.0 || norm_b == 0.0 {
    return 0.0;
  }
  dot / (norm_a.sqrt() * norm_b.sqrt())
}

pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
  let mut sum = 0.0f32;
  for (x, y) in a.iter().zip(b.iter()) {
    let d = x - y;
    sum += d * d;
  }
  sum.sqrt()
}

pub fn metric_similarity(metric: &VectorMetric, a: &[f32], b: &[f32]) -> f32 {
  match metric {
    VectorMetric::Cosine => {
      // Cosine assumes normalized vectors; fall back to computing normalization if needed.
      let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
      if dot.is_nan() {
        0.0
      } else {
        dot
      }
    }
    VectorMetric::L2 => -l2_distance(a, b),
  }
}

pub fn blend_scores(bm25: f32, vector_score: f32, alpha: f32, higher_is_better: bool) -> f32 {
  let vec_component = if higher_is_better {
    vector_score
  } else {
    -vector_score
  };
  alpha * bm25 + (1.0 - alpha) * vec_component
}
