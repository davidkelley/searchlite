use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::api::types::VectorMetric;
use crate::vectors::{metric_similarity, VectorStore};

pub const DEFAULT_M: usize = 16;
pub const DEFAULT_EF_CONSTRUCTION: usize = 64;
pub const DEFAULT_EF_SEARCH: usize = 40;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswGraph {
  pub dim: usize,
  pub metric: VectorMetric,
  pub m: usize,
  pub ef_construction: usize,
  pub entry: Option<u32>,
  pub neighbors: Vec<Vec<u32>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HnswParams {
  pub m: usize,
  pub ef_construction: usize,
}

impl Default for HnswParams {
  fn default() -> Self {
    Self {
      m: DEFAULT_M,
      ef_construction: DEFAULT_EF_CONSTRUCTION,
    }
  }
}

#[derive(Debug, Clone)]
pub struct HnswIndex {
  graph: HnswGraph,
  store: Arc<VectorStore>,
}

// NOTE: This is a single-layer (flat) HNSW graph; hierarchical layers are
// omitted for now to keep construction/search simple for small/medium indexes.
// Add multi-layer support if larger-scale performance demands it.
#[derive(Debug, Clone, Copy)]
struct Scored {
  id: u32,
  score: f32,
}

impl Eq for Scored {}

impl PartialEq for Scored {
  fn eq(&self, other: &Self) -> bool {
    self.id == other.id && self.score.to_bits() == other.score.to_bits()
  }
}

impl Ord for Scored {
  fn cmp(&self, other: &Self) -> Ordering {
    match self.score.total_cmp(&other.score) {
      Ordering::Equal => self.id.cmp(&other.id),
      ord => ord,
    }
  }
}

impl PartialOrd for Scored {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

impl HnswIndex {
  pub fn new(store: Arc<VectorStore>, params: HnswParams) -> Self {
    let capacity = store.len();
    Self {
      graph: HnswGraph {
        dim: store.dim(),
        metric: store.metric(),
        m: params.m.max(1),
        ef_construction: params.ef_construction.max(1),
        entry: None,
        neighbors: vec![Vec::new(); capacity],
      },
      store,
    }
  }

  pub fn from_graph(graph: HnswGraph, store: Arc<VectorStore>) -> Self {
    Self { graph, store }
  }

  pub fn graph(&self) -> &HnswGraph {
    &self.graph
  }

  pub fn into_graph(self) -> HnswGraph {
    self.graph
  }

  pub fn len(&self) -> usize {
    self.store.present()
  }

  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  pub fn add_vector(&mut self, id: u32) {
    let Some(vector) = self.store.vector(id) else {
      return;
    };
    let target = id as usize;
    if target >= self.graph.neighbors.len() {
      self.graph.neighbors.resize(target + 1, Vec::new());
    }
    if self.graph.entry.is_none() {
      self.graph.entry = Some(id);
      return;
    }
    let ef = self.graph.ef_construction.max(self.graph.m * 2);
    let mut candidates = self.search_internal(vector, ef);
    candidates.retain(|cand| cand.id != id);
    candidates.sort_by(|a, b| b.cmp(a));
    if candidates.len() > self.graph.m {
      candidates.truncate(self.graph.m);
    }
    let neighbor_ids: Vec<u32> = candidates.iter().map(|s| s.id).collect();
    self.graph.neighbors[target] = neighbor_ids.clone();
    for &n in neighbor_ids.iter() {
      let mut owned = self
        .graph
        .neighbors
        .get(n as usize)
        .cloned()
        .unwrap_or_default();
      if !owned.contains(&id) {
        owned.push(id);
        self.prune_list(n, &mut owned);
        self.graph.neighbors[n as usize] = owned;
      }
    }
    if let Some(entry) = self.graph.entry {
      if self.graph.neighbors[entry as usize].is_empty() && id != entry {
        let mut entry_neighbors = self
          .graph
          .neighbors
          .get(entry as usize)
          .cloned()
          .unwrap_or_default();
        entry_neighbors.push(id);
        self.prune_list(entry, &mut entry_neighbors);
        self.graph.neighbors[entry as usize] = entry_neighbors;

        let mut target_neighbors = self
          .graph
          .neighbors
          .get(target)
          .cloned()
          .unwrap_or_default();
        target_neighbors.push(entry);
        self.prune_list(id, &mut target_neighbors);
        self.graph.neighbors[target] = target_neighbors;
      }
    }
  }

  pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(u32, f32)> {
    if k == 0 {
      return Vec::new();
    }
    let ef = ef_search.max(k).max(1);
    let mut results = self.search_internal(query, ef);
    results.sort_by(|a, b| b.cmp(a));
    results.truncate(k);
    results.into_iter().map(|s| (s.id, s.score)).collect()
  }

  fn search_internal(&self, query: &[f32], ef: usize) -> Vec<Scored> {
    let mut visited = HashSet::new();
    let entry = match self.graph.entry {
      Some(id) => id,
      None => return Vec::new(),
    };
    let entry_score = self.similarity(query, entry);
    let mut candidates = BinaryHeap::new();
    let mut results: BinaryHeap<Reverse<Scored>> = BinaryHeap::new();
    candidates.push(Scored {
      id: entry,
      score: entry_score,
    });
    results.push(Reverse(Scored {
      id: entry,
      score: entry_score,
    }));
    visited.insert(entry);
    while let Some(best) = candidates.pop() {
      let worst_score = results.peek().map(|s| s.0.score).unwrap_or(f32::MIN);
      if best.score < worst_score && results.len() >= ef {
        break;
      }
      for &neighbor in self
        .graph
        .neighbors
        .get(best.id as usize)
        .unwrap_or(&Vec::new())
      {
        if !visited.insert(neighbor) {
          continue;
        }
        let Some(score) = self.similarity_opt(query, neighbor) else {
          continue;
        };
        if results.len() < ef || score > worst_score {
          candidates.push(Scored {
            id: neighbor,
            score,
          });
          results.push(Reverse(Scored {
            id: neighbor,
            score,
          }));
          if results.len() > ef {
            results.pop();
          }
        }
      }
    }
    results.into_iter().map(|r| r.0).collect()
  }

  fn prune_list(&self, target: u32, list: &mut Vec<u32>) {
    list.sort_by(|a, b| {
      let sa = self.similarity_between(target, *a);
      let sb = self.similarity_between(target, *b);
      sb.total_cmp(&sa).then_with(|| a.cmp(b)) // Prefer deterministic ordering
    });
    if list.len() > self.graph.m {
      list.truncate(self.graph.m);
    }
  }

  fn similarity(&self, query: &[f32], doc: u32) -> f32 {
    self
      .store
      .vector(doc)
      .map(|v| metric_similarity(&self.graph.metric, query, v))
      .unwrap_or(f32::MIN)
  }

  fn similarity_opt(&self, query: &[f32], doc: u32) -> Option<f32> {
    self
      .store
      .vector(doc)
      .map(|v| metric_similarity(&self.graph.metric, query, v))
  }

  fn similarity_between(&self, a: u32, b: u32) -> f32 {
    let Some(va) = self.store.vector(a) else {
      return f32::MIN;
    };
    let Some(vb) = self.store.vector(b) else {
      return f32::MIN;
    };
    metric_similarity(&self.graph.metric, va, vb)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::vectors::normalize_in_place;

  #[test]
  fn hnsw_returns_nearest_for_cosine() {
    let offsets = vec![0, 1, 2];
    let mut values = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    for chunk in values.chunks_mut(2) {
      normalize_in_place(chunk);
    }
    let store = Arc::new(VectorStore::new(2, VectorMetric::Cosine, offsets, values));
    let mut index = HnswIndex::new(store.clone(), HnswParams::default());
    for id in 0..3 {
      index.add_vector(id);
    }
    let query = store.vector(0).unwrap().to_vec();
    let results = index.search(&query, 2, DEFAULT_EF_SEARCH);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, 0);
  }

  #[test]
  fn hnsw_supports_l2_metric() {
    let offsets = vec![0, 1];
    let values = vec![0.0, 0.0, 10.0, 10.0];
    let store = Arc::new(VectorStore::new(2, VectorMetric::L2, offsets, values));
    let mut index = HnswIndex::new(store.clone(), HnswParams::default());
    index.add_vector(0);
    index.add_vector(1);
    let results = index.search(&[1.0, 1.0], 2, DEFAULT_EF_SEARCH);
    assert_eq!(results[0].0, 0);
    assert_eq!(results[1].0, 1);
    assert!(results[0].1 > results[1].1);
  }
}
