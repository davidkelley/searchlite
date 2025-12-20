pub mod hnsw;
pub mod quant;

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

pub fn blend_scores(bm25: f32, vector_score: f32, alpha: f32, higher_is_better: bool) -> f32 {
  let vec_component = if higher_is_better {
    vector_score
  } else {
    -vector_score
  };
  alpha * bm25 + (1.0 - alpha) * vec_component
}
