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

  Ok(())
}
