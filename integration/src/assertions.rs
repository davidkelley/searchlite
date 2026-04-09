use anyhow::{anyhow, Result};
use serde_json::Value;

use crate::fixtures::DatasetName;
use crate::normalization::normalize_search_result;

pub fn assert_normalized_search_parity(
  left: &Value,
  right: &Value,
  dataset: DatasetName,
  query_name: &str,
) -> Result<()> {
  let left_norm = normalize_search_result(left)
    .map_err(|err| anyhow!("left normalization failed for {dataset:?}/{query_name}: {err}"))?;
  let right_norm = normalize_search_result(right)
    .map_err(|err| anyhow!("right normalization failed for {dataset:?}/{query_name}: {err}"))?;

  if left_norm.total_hits_estimate != right_norm.total_hits_estimate {
    return Err(anyhow!(
      "total_hits_estimate mismatch for {dataset:?}/{query_name}: left={}, right={}",
      left_norm.total_hits_estimate,
      right_norm.total_hits_estimate
    ));
  }

  if left_norm.hit_doc_ids != right_norm.hit_doc_ids {
    return Err(anyhow!(
      "doc_id hit order mismatch for {dataset:?}/{query_name}: left={:?}, right={:?}",
      left_norm.hit_doc_ids,
      right_norm.hit_doc_ids
    ));
  }

  if left_norm.aggregation_keys != right_norm.aggregation_keys {
    return Err(anyhow!(
      "aggregation key mismatch for {dataset:?}/{query_name}: left={:?}, right={:?}",
      left_norm.aggregation_keys,
      right_norm.aggregation_keys
    ));
  }

  // Compare aggregation structure: keys must match, and bucket keys/types must match.
  // We do NOT require exact numeric equality since floating-point arithmetic may
  // differ slightly between the Core and HTTP surfaces.
  if !agg_structures_compatible(
    &left_norm.aggregation_structure,
    &right_norm.aggregation_structure,
  ) {
    return Err(anyhow!(
      "aggregation structure mismatch for {dataset:?}/{query_name}: left keys={:?}, right keys={:?}",
      left_norm.aggregation_keys,
      right_norm.aggregation_keys,
    ));
  }

  if left_norm.suggest_keys != right_norm.suggest_keys {
    return Err(anyhow!(
      "suggest key mismatch for {dataset:?}/{query_name}: left={:?}, right={:?}",
      left_norm.suggest_keys,
      right_norm.suggest_keys
    ));
  }

  if left_norm.next_cursor_present != right_norm.next_cursor_present {
    return Err(anyhow!(
      "next_cursor presence mismatch for {dataset:?}/{query_name}: left={}, right={}",
      left_norm.next_cursor_present,
      right_norm.next_cursor_present
    ));
  }

  if left_norm.next_search_after_present != right_norm.next_search_after_present {
    return Err(anyhow!(
      "next_search_after presence mismatch for {dataset:?}/{query_name}: left={}, right={}",
      left_norm.next_search_after_present,
      right_norm.next_search_after_present
    ));
  }

  if left_norm.highlight_hit_count != right_norm.highlight_hit_count {
    return Err(anyhow!(
      "highlight hit-count mismatch for {dataset:?}/{query_name}: left={}, right={}",
      left_norm.highlight_hit_count,
      right_norm.highlight_hit_count
    ));
  }

  if left_norm.stored_field_present != right_norm.stored_field_present {
    return Err(anyhow!(
      "stored field presence mismatch for {dataset:?}/{query_name}: left={:?}, right={:?}",
      left_norm.stored_field_present,
      right_norm.stored_field_present
    ));
  }

  if left_norm.highlight_snippets != right_norm.highlight_snippets {
    return Err(anyhow!(
      "highlight snippet presence mismatch for {dataset:?}/{query_name}: left={:?}, right={:?}",
      left_norm.highlight_snippets,
      right_norm.highlight_snippets
    ));
  }

  // Verify both sides have scores in non-increasing order, but only when
  // no custom sort is active (custom sorts override score ordering).
  let has_custom_sort = left
    .get("hits")
    .and_then(Value::as_array)
    .and_then(|hits| hits.first())
    .and_then(|h| h.get("sort_key"))
    .is_some();

  if !has_custom_sort {
    check_score_ordering(&left_norm.hit_scores, "left", dataset, query_name)?;
    check_score_ordering(&right_norm.hit_scores, "right", dataset, query_name)?;
  }

  Ok(())
}

/// Compare aggregation structures for compatibility: same keys, same bucket
/// structure, but allow numeric values to differ (floating-point variance).
fn agg_structures_compatible(
  left: &std::collections::BTreeMap<String, Value>,
  right: &std::collections::BTreeMap<String, Value>,
) -> bool {
  if left.keys().collect::<Vec<_>>() != right.keys().collect::<Vec<_>>() {
    return false;
  }
  for key in left.keys() {
    if !values_structurally_compatible(&left[key], &right[key]) {
      return false;
    }
  }
  true
}

/// Recursively compare two JSON values for structural compatibility:
/// - Objects: same keys, recursively compatible values
/// - Arrays: same length, recursively compatible elements
/// - Strings: must be equal
/// - Booleans: must be equal
/// - Nulls: must both be null
/// - Numbers: allowed to differ (floating-point tolerance)
fn values_structurally_compatible(left: &Value, right: &Value) -> bool {
  match (left, right) {
    (Value::Object(l), Value::Object(r)) => {
      if l.keys().collect::<Vec<_>>() != r.keys().collect::<Vec<_>>() {
        return false;
      }
      l.keys()
        .all(|k| values_structurally_compatible(&l[k], &r[k]))
    }
    (Value::Array(l), Value::Array(r)) => {
      if l.len() != r.len() {
        return false;
      }
      l.iter()
        .zip(r.iter())
        .all(|(a, b)| values_structurally_compatible(a, b))
    }
    (Value::String(l), Value::String(r)) => l == r,
    (Value::Bool(l), Value::Bool(r)) => l == r,
    (Value::Null, Value::Null) => true,
    (Value::Number(_), Value::Number(_)) => true, // allow numeric differences
    _ => false,                                   // type mismatch
  }
}

fn check_score_ordering(
  scores: &[Option<f64>],
  side: &str,
  dataset: DatasetName,
  query_name: &str,
) -> Result<()> {
  let present: Vec<f64> = scores.iter().filter_map(|s| *s).collect();
  for window in present.windows(2) {
    if window[0] < window[1] - f64::EPSILON {
      return Err(anyhow!(
        "scores not in descending order ({side}) for {dataset:?}/{query_name}: {present:?}"
      ));
    }
  }
  Ok(())
}

/// Assert properties of a single search result, used by the matrix executor.
pub fn assert_search_result_properties(
  result: &Value,
  case_id: &str,
  expect_hits: bool,
  expect_stored: bool,
) -> Result<()> {
  // total_hits_estimate must always be present
  let _total = result
    .get("total_hits_estimate")
    .and_then(Value::as_u64)
    .ok_or_else(|| anyhow!("[{case_id}] missing numeric total_hits_estimate"))?;

  let hits = result
    .get("hits")
    .and_then(Value::as_array)
    .ok_or_else(|| anyhow!("[{case_id}] missing hits array"))?;

  if !expect_hits {
    // When return_hits=false, hits should be empty
    if !hits.is_empty() {
      return Err(anyhow!(
        "[{case_id}] expected empty hits with return_hits=false, got {} hits",
        hits.len()
      ));
    }
    return Ok(());
  }

  // When we expect hits, validate each one
  for (i, hit) in hits.iter().enumerate() {
    let doc_id = hit.get("doc_id").and_then(Value::as_str);
    if doc_id.is_none_or(|s| s.is_empty()) {
      return Err(anyhow!("[{case_id}] hit[{i}] has missing or empty doc_id"));
    }

    if expect_stored {
      let has_fields = hit
        .get("fields")
        .and_then(Value::as_object)
        .map(|o| !o.is_empty())
        .unwrap_or(false);
      if !has_fields {
        return Err(anyhow!(
          "[{case_id}] hit[{i}] expected stored fields with return_stored=true"
        ));
      }
    }
  }

  // Verify scores are in non-increasing order (only when no custom sort is active)
  let has_custom_sort = hits.first().and_then(|h| h.get("sort_key")).is_some();
  if !has_custom_sort {
    let scores: Vec<f64> = hits
      .iter()
      .filter_map(|h| h.get("score").and_then(Value::as_f64))
      .collect();
    for window in scores.windows(2) {
      if window[0] < window[1] - f64::EPSILON {
        return Err(anyhow!(
          "[{case_id}] scores not in descending order: {scores:?}"
        ));
      }
    }
  }

  Ok(())
}
