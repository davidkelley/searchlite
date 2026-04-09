use std::collections::HashSet;

use anyhow::Result;

use integration::fixtures::load_example_fixtures;
use integration::matrix::{
  assert_unique_ids, generate_feature_matrix_cases, MatrixMode, ShardConfig,
};

#[test]
fn feature_cross_matrix_cardinality_and_unique_ids() -> Result<()> {
  let fixtures = load_example_fixtures()?;
  let cases = generate_feature_matrix_cases(&fixtures, MatrixMode::Full, None);
  assert_unique_ids(&cases);
  assert!(
    cases.len() >= 2_500,
    "expected at least 2500 full-matrix cases, got {}",
    cases.len()
  );
  Ok(())
}

#[test]
fn feature_cross_matrix_quick_mode_is_smaller_than_full() -> Result<()> {
  let fixtures = load_example_fixtures()?;
  let full_cases = generate_feature_matrix_cases(&fixtures, MatrixMode::Full, None);
  let quick_cases = generate_feature_matrix_cases(&fixtures, MatrixMode::Quick, None);
  assert!(
    quick_cases.len() < full_cases.len(),
    "quick mode should emit fewer cases"
  );
  Ok(())
}

#[test]
fn feature_cross_matrix_sharding_is_deterministic_and_partitioned() -> Result<()> {
  let fixtures = load_example_fixtures()?;
  let full = generate_feature_matrix_cases(&fixtures, MatrixMode::Full, None);

  let shard_count = 3usize;
  let shards: Vec<Vec<_>> = (0..shard_count)
    .map(|idx| {
      generate_feature_matrix_cases(
        &fixtures,
        MatrixMode::Full,
        Some(ShardConfig {
          index: idx,
          total: shard_count,
        }),
      )
    })
    .collect();

  let full_ids: HashSet<_> = full.iter().map(|case| case.id.clone()).collect();
  let mut union_ids = HashSet::new();
  for shard in shards.iter() {
    for case in shard {
      union_ids.insert(case.id.clone());
    }
  }

  assert_eq!(full_ids, union_ids, "shards should cover the full case set");
  Ok(())
}
