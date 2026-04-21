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
  // BUG-384: the sum-of-squares of individually-finite components can still
  // overflow `f32::MAX` to `+inf` (e.g. `[3e19, 3e19]`). Under `norm = +inf`,
  // `v / norm` silently collapses every component to `0.0`, poisoning any
  // downstream cosine similarity. Leave the vector un-normalized in that case
  // so the caller's overflow guard surfaces an actionable error rather than a
  // segment with all-zero cosine vectors that are invisible to every query.
  if norm > 0.0 && norm.is_finite() {
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

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn normalize_in_place_scales_finite_vector_to_unit_length() {
    let mut v = vec![3.0_f32, 4.0_f32];
    normalize_in_place(&mut v);
    assert!((v[0] - 0.6).abs() < 1e-6, "expected 0.6, got {}", v[0]);
    assert!((v[1] - 0.8).abs() < 1e-6, "expected 0.8, got {}", v[1]);
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    assert!((norm_sq - 1.0).abs() < 1e-6);
  }

  // BUG-384: `[3e19, 3e19]` has finite components but `(3e19)^2 = 9e38` alone
  // already exceeds `f32::MAX`, so the sum-of-squares saturates to `+inf`. The
  // pre-fix code computed `norm = sqrt(inf) = inf`, passed the `norm > 0.0`
  // check, and then divided every component by `inf` — silently zeroing the
  // vector and poisoning every downstream cosine score to exactly `0`.
  #[test]
  fn normalize_in_place_leaves_vector_untouched_when_sum_of_squares_overflows() {
    let input: Vec<f32> = vec![3.0e19_f32, 3.0e19_f32];
    let mut v = input.clone();
    normalize_in_place(&mut v);
    assert_eq!(
      v, input,
      "non-finite norm must leave the vector un-normalized, not zero it"
    );
    assert!(
      v.iter().all(|x| *x == 3.0e19_f32),
      "no component should be silently collapsed to zero"
    );
  }

  #[test]
  fn normalize_in_place_leaves_zero_vector_untouched() {
    let mut v = vec![0.0_f32, 0.0_f32];
    normalize_in_place(&mut v);
    assert_eq!(v, vec![0.0, 0.0]);
  }
}
