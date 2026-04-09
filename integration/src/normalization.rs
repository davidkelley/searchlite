use std::collections::BTreeMap;

use anyhow::{anyhow, Result};
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct NormalizedSearchResult {
  pub total_hits_estimate: u64,
  pub hit_doc_ids: Vec<String>,
  pub hit_scores: Vec<Option<f64>>,
  pub next_cursor_present: bool,
  pub next_search_after_present: bool,
  pub aggregation_keys: Vec<String>,
  pub aggregation_structure: BTreeMap<String, Value>,
  pub suggest_keys: Vec<String>,
  pub highlight_hit_count: usize,
  pub stored_field_present: Vec<bool>,
  pub highlight_snippets: Vec<bool>,
}

pub fn normalize_search_result(value: &Value) -> Result<NormalizedSearchResult> {
  let total_hits_estimate = value
    .get("total_hits_estimate")
    .and_then(Value::as_u64)
    .ok_or_else(|| anyhow!("search result missing numeric total_hits_estimate"))?;

  let hits = value
    .get("hits")
    .and_then(Value::as_array)
    .ok_or_else(|| anyhow!("search result missing hits array"))?;

  let mut hit_doc_ids = Vec::with_capacity(hits.len());
  let mut hit_scores = Vec::with_capacity(hits.len());
  let mut highlight_hit_count = 0usize;
  let mut stored_field_present = Vec::with_capacity(hits.len());
  let mut highlight_snippets = Vec::with_capacity(hits.len());

  for hit in hits {
    let doc_id = hit
      .get("doc_id")
      .and_then(Value::as_str)
      .ok_or_else(|| anyhow!("hit missing doc_id string"))?
      .to_string();
    hit_doc_ids.push(doc_id);

    hit_scores.push(hit.get("score").and_then(Value::as_f64));

    let has_stored = hit
      .get("fields")
      .and_then(Value::as_object)
      .map(|o| !o.is_empty())
      .unwrap_or(false);
    stored_field_present.push(has_stored);

    let has_highlights = hit
      .get("highlights")
      .and_then(Value::as_object)
      .map(|h| !h.is_empty())
      .unwrap_or(false);
    highlight_snippets.push(has_highlights);
    if has_highlights {
      highlight_hit_count += 1;
    }
  }

  let aggregation_structure: BTreeMap<String, Value> = value
    .get("aggregations")
    .and_then(Value::as_object)
    .map(|aggs| {
      aggs
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
    })
    .unwrap_or_default();

  let mut aggregation_keys: Vec<String> = aggregation_structure.keys().cloned().collect();
  aggregation_keys.sort();

  let mut suggest_keys = value
    .get("suggest")
    .and_then(Value::as_object)
    .map(|s| s.keys().cloned().collect::<Vec<_>>())
    .unwrap_or_default();
  suggest_keys.sort();

  Ok(NormalizedSearchResult {
    total_hits_estimate,
    hit_doc_ids,
    hit_scores,
    next_cursor_present: value.get("next_cursor").is_some_and(|v| !v.is_null()),
    next_search_after_present: value.get("next_search_after").is_some_and(|v| !v.is_null()),
    aggregation_keys,
    aggregation_structure,
    suggest_keys,
    highlight_hit_count,
    stored_field_present,
    highlight_snippets,
  })
}
