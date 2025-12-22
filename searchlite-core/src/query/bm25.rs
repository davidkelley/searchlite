pub fn bm25(tf: f32, df: f32, doc_len: f32, avgdl: f32, docs: f32, k1: f32, b: f32) -> f32 {
  let idf = ((docs - df + 0.5) / (df + 0.5)).ln().max(0.0) + 1.0;
  let norm_dl = if avgdl > 0.0 { doc_len / avgdl } else { 1.0 };
  let denom = tf + k1 * (1.0 - b + b * norm_dl);
  idf * (tf * (k1 + 1.0)) / denom.max(1e-6)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn computes_reasonable_scores() {
    let score = bm25(3.0, 5.0, 100.0, 120.0, 1000.0, 1.2, 0.75);
    assert!(score.is_finite());
    let zero_avgdl = bm25(1.0, 1.0, 0.0, 0.0, 10.0, 1.2, 0.75);
    assert!(zero_avgdl > 0.0);
  }
}
