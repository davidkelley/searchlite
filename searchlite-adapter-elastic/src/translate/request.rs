use serde_json::{Map, Value};

use super::aggregation::translate_aggs;
use super::highlight::translate_highlight;
use super::pagination::apply_pagination;
use super::query::translate_query;
use super::sort::translate_sort;
use super::unsupported::Unsupported;

/// Translate an Elasticsearch `_search` request body into a SearchLite
/// `SearchRequest` body. Always sets `return_stored: true` so callers see
/// `_source` in hits.
pub fn translate_search_body(es_body: &Value) -> Result<Value, Unsupported> {
  // Accept only an object body or a literal `null`. Previously any non-object
  // value was silently coerced into an empty map and run as `match_all` —
  // for `_msearch` that meant a malformed body line scanned the whole index
  // instead of returning a parse error. ES treats a missing/null body as
  // match-all, so we keep that behaviour for `null` only.
  let map = match es_body {
    Value::Object(map) => map.clone(),
    Value::Null => Map::new(),
    _ => {
      return Err(Unsupported::with_detail(
        "body",
        "search body must be a JSON object",
      ))
    }
  };

  if map.contains_key("script_fields") {
    return Err(Unsupported::feature("script_fields"));
  }
  if map.contains_key("docvalue_fields") {
    return Err(Unsupported::feature("docvalue_fields"));
  }
  if map.contains_key("stored_fields") {
    return Err(Unsupported::feature("stored_fields"));
  }
  if map.contains_key("post_filter") {
    return Err(Unsupported::feature("post_filter"));
  }
  if map.contains_key("min_score") {
    return Err(Unsupported::feature("min_score"));
  }
  if map.contains_key("indices_boost") {
    return Err(Unsupported::feature("indices_boost"));
  }
  if map.contains_key("collapse") {
    return Err(Unsupported::feature("collapse"));
  }
  if map.contains_key("rescore") {
    return Err(Unsupported::feature("rescore"));
  }
  if map.contains_key("suggest") {
    return Err(Unsupported::feature("suggest"));
  }
  if map.contains_key("pit") {
    return Err(Unsupported::feature("point_in_time"));
  }

  let mut out = Map::new();

  let query = match map.get("query") {
    Some(q) => translate_query(q)?,
    None => translate_query(&Value::Object(Map::new()))?,
  };
  out.insert("query".into(), query);

  apply_pagination(&map, &mut out)?;

  if let Some(sort) = map.get("sort") {
    out.insert("sort".into(), Value::Array(translate_sort(sort)?));
  }

  if let Some(highlight) = map.get("highlight") {
    out.insert("highlight".into(), translate_highlight(highlight)?);
  }

  if let Some(source) = map.get("_source") {
    apply_source(source, &mut out)?;
  } else {
    out.insert("return_stored".into(), Value::Bool(true));
  }

  let aggs_value = map.get("aggs").or_else(|| map.get("aggregations")).cloned();
  if let Some(aggs_value) = aggs_value {
    let aggs_map = aggs_value
      .as_object()
      .ok_or_else(|| Unsupported::with_detail("aggs", "must be an object"))?;
    let translated = translate_aggs(aggs_map)?;
    out.insert("aggs".into(), Value::Object(translated));
  }

  if let Some(explain) = map.get("explain") {
    out.insert("explain".into(), explain.clone());
  }
  if let Some(profile) = map.get("profile") {
    out.insert("profile".into(), profile.clone());
  }

  Ok(Value::Object(out))
}

fn apply_source(source: &Value, out: &mut Map<String, Value>) -> Result<(), Unsupported> {
  match source {
    Value::Bool(false) => {
      out.insert("return_stored".into(), Value::Bool(false));
      out.insert("return_hits".into(), Value::Bool(true));
    }
    Value::Bool(true) => {
      out.insert("return_stored".into(), Value::Bool(true));
    }
    Value::Array(items) => {
      let fields: Vec<Value> = items
        .iter()
        .filter_map(Value::as_str)
        .map(|s| Value::String(s.to_string()))
        .collect();
      if !fields.is_empty() {
        out.insert("fields".into(), Value::Array(fields));
      }
      out.insert("return_stored".into(), Value::Bool(true));
    }
    Value::String(field) => {
      out.insert(
        "fields".into(),
        Value::Array(vec![Value::String(field.clone())]),
      );
      out.insert("return_stored".into(), Value::Bool(true));
    }
    Value::Object(opts) => {
      if let Some(includes) = opts.get("includes") {
        // ES accepts both the single-string form (`"includes": "title"`) and
        // the array form (`"includes": ["title", "category"]`). Previously
        // we only handled the array case via `filter_map(Value::as_str)`,
        // which silently dropped non-string entries (widening the payload)
        // and ignored the string form entirely (also widening). Validate
        // both shapes and reject invalid elements with a clear error.
        let fields = match includes {
          Value::String(s) => vec![Value::String(s.clone())],
          Value::Array(items) => {
            let mut fields = Vec::with_capacity(items.len());
            for item in items {
              match item.as_str() {
                Some(s) => fields.push(Value::String(s.to_string())),
                None => {
                  return Err(Unsupported::with_detail(
                    "_source.includes",
                    format!("array element must be a string, got {item}"),
                  ));
                }
              }
            }
            fields
          }
          _ => {
            return Err(Unsupported::with_detail(
              "_source.includes",
              "must be a string or array of strings",
            ));
          }
        };
        if !fields.is_empty() {
          out.insert("fields".into(), Value::Array(fields));
        }
      }
      if opts.get("excludes").is_some() {
        return Err(Unsupported::feature("_source.excludes"));
      }
      out.insert("return_stored".into(), Value::Bool(true));
    }
    _ => return Err(Unsupported::feature("_source")),
  }
  Ok(())
}
