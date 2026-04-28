use serde_json::{json, Value};

use super::unsupported::Unsupported;

/// Translate the ES `sort` field into SearchLite's `[{field, order}]` shape.
///
/// Accepted forms:
/// - `"field"` → `[{field, order: "asc"}]` (but `_score` defaults to `desc`,
///   matching Elasticsearch — score-sorting is normally relevance-descending)
/// - `["field", {field2: "desc"}]`
/// - `[{field: {order: "desc", missing: "_last"}}]`
/// - `"_score"` translates verbatim
pub fn translate_sort(es_sort: &Value) -> Result<Vec<Value>, Unsupported> {
  let entries = match es_sort {
    Value::String(_) | Value::Object(_) => vec![es_sort.clone()],
    Value::Array(items) => items.clone(),
    _ => {
      return Err(Unsupported::with_detail(
        "sort",
        "expected string, object, or array of either",
      ));
    }
  };

  let mut translated = Vec::with_capacity(entries.len());
  for entry in entries {
    translated.push(translate_sort_entry(&entry)?);
  }
  Ok(translated)
}

fn translate_sort_entry(entry: &Value) -> Result<Value, Unsupported> {
  match entry {
    Value::String(field) => Ok(json!({ "field": field, "order": default_order(field) })),
    Value::Object(map) => {
      if map.len() != 1 {
        return Err(Unsupported::with_detail(
          "sort",
          "sort object must contain exactly one field",
        ));
      }
      let (field, spec) = map.iter().next().unwrap();
      match spec {
        Value::String(order) => Ok(json!({ "field": field, "order": normalize_order(order)? })),
        Value::Object(opts) => {
          let order = opts
            .get("order")
            .and_then(Value::as_str)
            .map(normalize_order)
            .transpose()?
            .unwrap_or_else(|| default_order(field).to_string());

          if opts.contains_key("mode") {
            return Err(Unsupported::with_detail("sort.mode", "not implemented"));
          }
          if opts.contains_key("nested") || opts.contains_key("nested_path") {
            return Err(Unsupported::with_detail("sort.nested", "not implemented"));
          }
          if opts.contains_key("unmapped_type") {
            return Err(Unsupported::with_detail(
              "sort.unmapped_type",
              "not implemented",
            ));
          }
          // `missing` accepted but ignored (SearchLite default behavior applies).
          Ok(json!({ "field": field, "order": order }))
        }
        _ => Err(Unsupported::with_detail(
          "sort",
          "sort spec must be a string or object",
        )),
      }
    }
    _ => Err(Unsupported::with_detail(
      "sort",
      "sort entry must be a string or object",
    )),
  }
}

/// Elasticsearch defaults `_score` sorting to `desc` (relevance high → low) and
/// other fields to `asc`. Mirror that so `sort: "_score"` and `sort: {"_score": {}}`
/// behave like ES.
fn default_order(field: &str) -> &'static str {
  if field == "_score" {
    "desc"
  } else {
    "asc"
  }
}

fn normalize_order(order: &str) -> Result<String, Unsupported> {
  match order.to_ascii_lowercase().as_str() {
    "asc" => Ok("asc".to_string()),
    "desc" => Ok("desc".to_string()),
    other => Err(Unsupported::with_detail(
      "sort.order",
      format!("unknown order `{other}`, expected `asc` or `desc`"),
    )),
  }
}
