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
  if let Some(track_total_hits) = es_body.get("track_total_hits") {
    // SearchLite always returns total_hits_estimate; the request flag is
    // accepted but informational only.
    let _ = track_total_hits;
  }
  Ok(())
}
