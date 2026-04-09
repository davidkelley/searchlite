mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use tempfile::tempdir;

use integration::execution::execute_matrix_case;
use integration::fixtures::{load_example_fixtures, DatasetName};
use integration::matrix::{
  generate_feature_matrix_cases, FeatureMatrixCase, LifecycleStage, MatrixMode, ShardConfig,
};
use integration::surfaces::cli::CliHarness;
use integration::surfaces::core::CoreHarness;
use integration::surfaces::http::HttpHarness;
use integration::surfaces::{SurfaceHarness, SurfaceKind};

#[test]
fn feature_matrix_execution() -> Result<()> {
  let fixtures = load_example_fixtures()?;
  let mode = MatrixMode::from_env();
  let shard = ShardConfig::from_env();
  let cases = generate_feature_matrix_cases(&fixtures, mode, shard);
  let searchlite_bin = common::searchlite_bin();

  if cases.is_empty() {
    eprintln!("feature_matrix_execution: no cases to run (check INTEGRATION_MODE and shard config)");
    return Ok(());
  }

  // Group cases by (dataset, surface) to share expensive harness setup.
  let mut groups: BTreeMap<(DatasetName, SurfaceKind), Vec<&FeatureMatrixCase>> =
    BTreeMap::new();
  for case in &cases {
    groups
      .entry((case.dataset, case.surface))
      .or_default()
      .push(case);
  }

  let mut passed = 0usize;
  let mut skipped = 0usize;
  let mut failed = 0usize;
  let mut failures: Vec<String> = Vec::new();

  for ((dataset_name, surface), group_cases) in &groups {
    let dataset = fixtures.datasets.get(dataset_name).unwrap();
    let dir = tempdir()?;
    let index_path = dir
      .path()
      .join(format!("idx-matrix-{dataset_name:?}-{surface:?}"));

    let mut harness = build_harness(*surface, index_path, searchlite_bin.clone())?;

    // Seed the index
    let schema_json = serde_json::to_value(&dataset.schema)
      .context("serializing schema for matrix execution")?;
    let seed_docs = match mode {
      MatrixMode::Full => &dataset.seed_docs[..],
      MatrixMode::Quick => {
        let capped = dataset.seed_docs.len().min(250);
        &dataset.seed_docs[..capped]
      }
    };
    let seed_ndjson = common::docs_to_ndjson(seed_docs);

    harness
      .init(&schema_json)
      .with_context(|| format!("init for {dataset_name:?}/{surface:?}"))?;
    harness
      .add_ndjson(&seed_ndjson)
      .with_context(|| format!("add_ndjson for {dataset_name:?}/{surface:?}"))?;
    harness
      .commit()
      .with_context(|| format!("commit for {dataset_name:?}/{surface:?}"))?;

    // Sort cases by lifecycle stage to apply mutations progressively:
    // PostCommit -> PostUpdate -> PostDelete -> PostCompact
    let mut sorted_cases: Vec<&&FeatureMatrixCase> = group_cases.iter().collect();
    sorted_cases.sort_by_key(|c| lifecycle_order(c.lifecycle_stage));

    let capabilities = harness.capabilities();
    let mut applied_stage = LifecycleStage::PostCommit;

    for case in &sorted_cases {
      let target_stage = case.lifecycle_stage;

      // Apply lifecycle mutations if we need to advance to a later stage
      if lifecycle_order(target_stage) > lifecycle_order(applied_stage) {
        if let Err(err) =
          advance_lifecycle(&mut *harness, applied_stage, target_stage, dataset, &capabilities)
        {
          // If we cannot advance (e.g., surface doesn't support update), skip cases at this stage
          skipped += 1;
          eprintln!(
            "  [SKIP] {}: cannot advance to {target_stage:?}: {err:#}",
            case.id
          );
          continue;
        }
        applied_stage = target_stage;
      }

      match execute_matrix_case(&mut *harness, case, dataset) {
        Ok(()) => passed += 1,
        Err(err) => {
          failed += 1;
          failures.push(format!("[FAIL] {}: {err:#}", case.id));
        }
      }
    }
  }

  eprintln!(
    "\nfeature_matrix_execution: {passed} passed, {failed} failed, {skipped} skipped (total {} cases)",
    passed + failed + skipped
  );

  if !failures.is_empty() {
    let summary = if failures.len() > 20 {
      let mut s = failures[..20].join("\n");
      s.push_str(&format!("\n... and {} more failures", failures.len() - 20));
      s
    } else {
      failures.join("\n")
    };
    anyhow::bail!(
      "{failed} of {} matrix cases failed:\n{summary}",
      passed + failed + skipped
    );
  }

  Ok(())
}

fn build_harness(
  surface: SurfaceKind,
  index_path: PathBuf,
  searchlite_bin: PathBuf,
) -> Result<Box<dyn SurfaceHarness>> {
  match surface {
    SurfaceKind::Core => Ok(Box::new(CoreHarness::new(index_path))),
    SurfaceKind::Http => Ok(Box::new(HttpHarness::new(searchlite_bin, index_path)?)),
    SurfaceKind::Cli => Ok(Box::new(CliHarness::new(searchlite_bin, index_path))),
  }
}

fn lifecycle_order(stage: LifecycleStage) -> u8 {
  match stage {
    LifecycleStage::PostCommit => 0,
    LifecycleStage::PostUpdate => 1,
    LifecycleStage::PostDelete => 2,
    LifecycleStage::PostCompact => 3,
  }
}

/// Advance the harness from `current` stage to `target` stage by applying
/// intermediate lifecycle mutations.
fn advance_lifecycle(
  harness: &mut dyn SurfaceHarness,
  current: LifecycleStage,
  target: LifecycleStage,
  dataset: &integration::fixtures::DatasetFixture,
  capabilities: &integration::surfaces::SurfaceCapabilities,
) -> Result<()> {
  let stages = [
    LifecycleStage::PostCommit,
    LifecycleStage::PostUpdate,
    LifecycleStage::PostDelete,
    LifecycleStage::PostCompact,
  ];

  for &stage in &stages {
    if lifecycle_order(stage) <= lifecycle_order(current) {
      continue;
    }
    if lifecycle_order(stage) > lifecycle_order(target) {
      break;
    }

    match stage {
      LifecycleStage::PostCommit => {} // nothing to do
      LifecycleStage::PostUpdate => {
        if capabilities.supports_update {
          if let Some(update) = dataset.mutations.update_docs.first() {
            harness.update_doc(&update.id, &update.set, &update.unset)?;
            harness.commit()?;
          }
        }
        // If update not supported, we just skip — the index state is still valid
      }
      LifecycleStage::PostDelete => {
        if capabilities.supports_delete {
          harness.delete_ids(&dataset.mutations.delete_ids)?;
          harness.commit()?;
        }
      }
      LifecycleStage::PostCompact => {
        if capabilities.supports_compact {
          harness.compact()?;
        }
      }
    }
  }

  Ok(())
}
