use std::collections::HashSet;

use crate::fixtures::{DatasetName, ExampleFixtures};
use crate::surfaces::SurfaceKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixMode {
  Quick,
  Full,
}

impl MatrixMode {
  pub fn from_env() -> Self {
    match std::env::var("INTEGRATION_MODE") {
      Ok(value) if value.eq_ignore_ascii_case("full") => Self::Full,
      _ => Self::Quick,
    }
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardConfig {
  pub index: usize,
  pub total: usize,
}

impl ShardConfig {
  pub fn from_env() -> Option<Self> {
    let total = std::env::var("INTEGRATION_MATRIX_SHARDS")
      .ok()
      .and_then(|value| value.parse::<usize>().ok())?;
    let index = std::env::var("INTEGRATION_MATRIX_SHARD")
      .ok()
      .and_then(|value| value.parse::<usize>().ok())?;
    if total == 0 || index >= total {
      return None;
    }
    Some(Self { index, total })
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PaginationMode {
  None,
  Cursor,
  SearchAfter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LifecycleStage {
  PostCommit,
  PostUpdate,
  PostDelete,
  PostCompact,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FeatureMatrixCase {
  pub id: String,
  pub dataset: DatasetName,
  pub surface: SurfaceKind,
  pub query_name: String,
  pub execution: &'static str,
  pub pagination: PaginationMode,
  pub lifecycle_stage: LifecycleStage,
  pub return_stored: bool,
  pub return_hits: bool,
  pub track_total_hits: bool,
}

pub fn generate_feature_matrix_cases(
  fixtures: &ExampleFixtures,
  mode: MatrixMode,
  shard: Option<ShardConfig>,
) -> Vec<FeatureMatrixCase> {
  let executions = ["bm25", "wand", "bmw"];
  let paginations = [
    PaginationMode::None,
    PaginationMode::Cursor,
    PaginationMode::SearchAfter,
  ];
  let stages = [
    LifecycleStage::PostCommit,
    LifecycleStage::PostUpdate,
    LifecycleStage::PostDelete,
    LifecycleStage::PostCompact,
  ];
  let bools = [false, true];
  let surfaces = [SurfaceKind::Core, SurfaceKind::Http, SurfaceKind::Cli];

  let mut out = Vec::new();

  for (dataset_name, dataset) in fixtures.datasets.iter() {
    for query in dataset.queries.iter() {
      for surface in surfaces {
        for execution in executions {
          for pagination in paginations {
            for stage in stages {
              for return_stored in bools {
                for return_hits in bools {
                  for track_total_hits in bools {
                    let id = format!(
                      "{dataset_name:?}::{surface:?}::{query_name}::{execution}::{pagination:?}::{stage:?}::stored={return_stored}::hits={return_hits}::track={track_total_hits}",
                      query_name = query.name,
                    );

                    if !include_case(mode, id.as_str()) {
                      continue;
                    }
                    if let Some(shard_cfg) = shard {
                      if stable_hash(id.as_str()) % shard_cfg.total != shard_cfg.index {
                        continue;
                      }
                    }

                    out.push(FeatureMatrixCase {
                      id,
                      dataset: *dataset_name,
                      surface,
                      query_name: query.name.clone(),
                      execution,
                      pagination,
                      lifecycle_stage: stage,
                      return_stored,
                      return_hits,
                      track_total_hits,
                    });
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  out
}

fn include_case(mode: MatrixMode, id: &str) -> bool {
  match mode {
    MatrixMode::Full => true,
    MatrixMode::Quick => {
      // Pairwise-like thinning for local/PR speed while keeping deterministic spread.
      stable_hash(id).is_multiple_of(19)
    }
  }
}

/// FNV-1a 64-bit hash — deterministic across all platforms and Rust versions,
/// unlike `DefaultHasher` which may change between releases.
fn stable_hash(input: &str) -> usize {
  let mut hash: u64 = 0xcbf29ce484222325;
  for byte in input.as_bytes() {
    hash ^= *byte as u64;
    hash = hash.wrapping_mul(0x100000001b3);
  }
  hash as usize
}

pub fn assert_unique_ids(cases: &[FeatureMatrixCase]) {
  let mut seen = HashSet::with_capacity(cases.len());
  for case in cases {
    assert!(
      seen.insert(case.id.clone()),
      "duplicate case id: {}",
      case.id
    );
  }
}
