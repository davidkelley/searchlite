/// GPU reranker stub. In v0.1 we simply echo the scores, but the module exists
/// so that an actual shader-backed implementation can be plugged in.
pub fn rerank(entries: &[(u32, f32)]) -> Vec<(u32, f32)> {
  entries.to_vec()
}
