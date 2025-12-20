//! Minimal placeholder for a vector index.

#[derive(Debug, Default)]
pub struct HnswIndex;

impl HnswIndex {
  pub fn new() -> Self {
    Self
  }

  pub fn add_vector(&mut self, _id: u32, _vec: &[f32]) {}

  pub fn search(&self, _query: &[f32], _k: usize) -> Vec<(u32, f32)> {
    Vec::new()
  }
}
