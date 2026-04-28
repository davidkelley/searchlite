use serde_json::{Map, Value};

use super::unsupported::Unsupported;

/// Copy ES `from`, `size`, and `search_after` onto the SearchLite request body.
/// `scroll` is rejected — recommend `search_after` migration.
pub fn apply_pagination(
  es_body: &Map<String, Value>,
  out: &mut Map<String, Value>,
) -> Result<(), Unsupported> {
  if es_body.contains_key("scroll") || es_body.contains_key("scroll_id") {
    return Err(Unsupported::with_detail(
      "scroll",
      "use `search_after` for cursor-style pagination",
    ));
  }
  if let Some(from) = es_body.get("from") {
    out.insert("from".to_string(), from.clone());
  }
  if let Some(size) = es_body.get("size") {
    out.insert("limit".to_string(), size.clone());
  }
  if let Some(search_after) = es_body.get("search_after") {
    out.insert("search_after".to_string(), search_after.clone());
  }
  if let Some(track) = es_body.get("track_total_hits") {
    // SearchLite's `track_total_hits` (Option<bool>) actually changes
    // execution: when true it disables WAND/BMW pruning so `total_hits` is
    // exact rather than an estimate. Forward it so `_count` and clients that
    // explicitly request exact totals get them.
    let normalized = match track {
      Value::Bool(b) => Value::Bool(*b),
      Value::Number(n) => {
        // ES allows an integer cap (track up to N exactly, then return a
        // lower bound). SearchLite has no lower-bound mode, so map any
        // positive cap to `true` and 0 to `false` — closer to user intent
        // than silently dropping it.
        let positive = n
          .as_i64()
          .map(|i| i > 0)
          .or_else(|| n.as_u64().map(|u| u > 0))
          .ok_or_else(|| {
            Unsupported::with_detail(
              "track_total_hits",
              "must be a boolean or non-negative integer",
            )
          })?;
        Value::Bool(positive)
      }
      _ => {
        return Err(Unsupported::with_detail(
          "track_total_hits",
          "must be a boolean or non-negative integer",
        ))
      }
    };
    out.insert("track_total_hits".to_string(), normalized);
  }
  Ok(())
}
