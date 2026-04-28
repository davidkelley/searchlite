use serde_json::{json, Map, Value};

/// Translate a SearchLite SearchResult into an Elasticsearch search response
/// envelope. `took_ms` is computed by the route handler.
pub fn translate_search_response(index: &str, sl: &Value, took_ms: u64) -> Value {
  let mut env = Map::new();
  env.insert("took".into(), Value::from(took_ms));
  env.insert("timed_out".into(), Value::Bool(false));
  env.insert(
    "_shards".into(),
    json!({ "total": 1, "successful": 1, "skipped": 0, "failed": 0 }),
  );

  let total = sl
    .get("total_hits_estimate")
    .and_then(Value::as_u64)
    .unwrap_or(0);
  let hits_arr = sl
    .get("hits")
    .and_then(Value::as_array)
    .cloned()
    .unwrap_or_default();
  let mut max_score: Option<f64> = None;
  let translated_hits: Vec<Value> = hits_arr
    .iter()
    .map(|hit| {
      let translated = translate_hit(index, hit);
      if let Some(score) = translated.get("_score").and_then(Value::as_f64) {
        max_score = Some(max_score.map_or(score, |m| m.max(score)));
      }
      translated
    })
    .collect();

  env.insert(
    "hits".into(),
    json!({
      "total": { "value": total, "relation": "gte" },
      "max_score": max_score.map(Value::from).unwrap_or(Value::Null),
      "hits": translated_hits,
    }),
  );

  if let Some(aggs) = sl.get("aggregations").and_then(Value::as_object) {
    if !aggs.is_empty() {
      env.insert("aggregations".into(), translate_aggregations(aggs));
    }
  }

  Value::Object(env)
}

fn translate_hit(index: &str, hit: &Value) -> Value {
  let map = hit.as_object();
  let id = map
    .and_then(|m| m.get("doc_id"))
    .and_then(Value::as_str)
    .unwrap_or("")
    .to_string();
  let score = map.and_then(|m| m.get("score"));
  let fields = map.and_then(|m| m.get("fields"));

  let mut out = Map::new();
  out.insert("_index".into(), Value::String(index.to_string()));
  out.insert("_id".into(), Value::String(id));
  if let Some(s) = score {
    out.insert("_score".into(), s.clone());
  } else {
    out.insert("_score".into(), Value::Null);
  }
  if let Some(f) = fields {
    out.insert("_source".into(), f.clone());
  }
  // Precedence: prefer structured `highlights` (per-field map) over the legacy
  // single-field `snippet`. When both are present we keep `highlights` as the
  // primary payload but surface the snippet under `_snippet` so neither field
  // is silently dropped — the previous unconditional double-insert overwrote
  // the snippet.
  let snippet = map.and_then(|m| m.get("snippet")).cloned();
  let highlights = map.and_then(|m| m.get("highlights")).cloned();
  match (highlights, snippet) {
    (Some(h), Some(s)) => {
      let mut merged = h.as_object().cloned().unwrap_or_default();
      merged.insert("_snippet".to_string(), json!([s]));
      out.insert("highlight".into(), Value::Object(merged));
    }
    (Some(h), None) => {
      out.insert("highlight".into(), h);
    }
    (None, Some(s)) => {
      out.insert("highlight".into(), json!({ "_snippet": [s] }));
    }
    (None, None) => {}
  }
  if let Some(sort_key) = map.and_then(|m| m.get("sort_key")) {
    out.insert("sort".into(), sort_key.clone());
  }
  Value::Object(out)
}

fn translate_aggregations(sl_aggs: &Map<String, Value>) -> Value {
  let mut out = Map::new();
  for (name, agg) in sl_aggs {
    out.insert(name.clone(), translate_aggregation(agg));
  }
  Value::Object(out)
}

fn translate_aggregation(agg: &Value) -> Value {
  let Some(map) = agg.as_object() else {
    return agg.clone();
  };
  let kind = map.get("type").and_then(Value::as_str).unwrap_or("");
  match kind {
    "terms" | "rare_terms" | "range" | "date_range" | "histogram" | "date_histogram"
    | "composite" | "significant_terms" => {
      let mut out = Map::new();
      if let Some(buckets) = map.get("buckets").and_then(Value::as_array) {
        let translated_buckets: Vec<Value> = buckets.iter().map(translate_bucket).collect();
        out.insert("buckets".into(), Value::Array(translated_buckets));
      }
      if let Some(after_key) = map.get("after_key") {
        out.insert("after_key".into(), after_key.clone());
      }
      if let Some(doc_count) = map.get("doc_count") {
        out.insert("doc_count".into(), doc_count.clone());
      }
      Value::Object(out)
    }
    "filter" | "nested" => {
      let mut out = Map::new();
      if let Some(doc_count) = map.get("doc_count") {
        out.insert("doc_count".into(), doc_count.clone());
      }
      if let Some(sub) = map.get("aggregations").and_then(Value::as_object) {
        for (name, value) in sub {
          out.insert(name.clone(), translate_aggregation(value));
        }
      }
      Value::Object(out)
    }
    "stats" => {
      let mut out = Map::new();
      copy_field(map, "count", &mut out);
      copy_field(map, "min", &mut out);
      copy_field(map, "max", &mut out);
      copy_field(map, "sum", &mut out);
      copy_field(map, "avg", &mut out);
      Value::Object(out)
    }
    "extended_stats" => {
      let mut out = Map::new();
      copy_field(map, "count", &mut out);
      copy_field(map, "min", &mut out);
      copy_field(map, "max", &mut out);
      copy_field(map, "sum", &mut out);
      copy_field(map, "avg", &mut out);
      copy_field(map, "variance", &mut out);
      copy_field(map, "std_deviation", &mut out);
      Value::Object(out)
    }
    "value_count" | "cardinality" => {
      let mut out = Map::new();
      copy_field(map, "value", &mut out);
      Value::Object(out)
    }
    "percentiles" | "percentile_ranks" => {
      let mut out = Map::new();
      if let Some(values) = map.get("values") {
        out.insert("values".into(), values.clone());
      }
      Value::Object(out)
    }
    "top_hits" => {
      let total = map.get("total").and_then(Value::as_u64).unwrap_or(0);
      let hits = map
        .get("hits")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
      let translated_hits: Vec<Value> = hits.iter().map(top_hit_to_es).collect();
      json!({
        "hits": {
          "total": { "value": total, "relation": "eq" },
          "max_score": Value::Null,
          "hits": translated_hits,
        }
      })
    }
    "bucket_sort" | "avg_bucket" | "sum_bucket" | "derivative" | "moving_avg" | "bucket_script" => {
      let mut out = Map::new();
      copy_field(map, "value", &mut out);
      copy_field(map, "from", &mut out);
      copy_field(map, "size", &mut out);
      copy_field(map, "predictions", &mut out);
      Value::Object(out)
    }
    _ => agg.clone(),
  }
}

fn translate_bucket(bucket: &Value) -> Value {
  let Some(map) = bucket.as_object() else {
    return bucket.clone();
  };
  let mut out = Map::new();
  if let Some(key) = map.get("key") {
    out.insert("key".into(), key.clone());
    if let Some(s) = key.as_str() {
      out.insert("key_as_string".into(), Value::String(s.to_string()));
    }
  }
  if let Some(doc_count) = map.get("doc_count") {
    out.insert("doc_count".into(), doc_count.clone());
  }
  if let Some(bg_count) = map.get("bg_count") {
    out.insert("bg_count".into(), bg_count.clone());
  }
  if let Some(score) = map.get("score") {
    out.insert("score".into(), score.clone());
  }
  if let Some(sub) = map.get("aggregations").and_then(Value::as_object) {
    for (name, value) in sub {
      out.insert(name.clone(), translate_aggregation(value));
    }
  }
  Value::Object(out)
}

fn top_hit_to_es(hit: &Value) -> Value {
  let Some(map) = hit.as_object() else {
    return hit.clone();
  };
  let mut out = Map::new();
  if let Some(id) = map.get("doc_id") {
    out.insert("_id".into(), id.clone());
  }
  if let Some(score) = map.get("score") {
    out.insert("_score".into(), score.clone());
  } else {
    out.insert("_score".into(), Value::Null);
  }
  if let Some(fields) = map.get("fields") {
    out.insert("_source".into(), fields.clone());
  }
  if let Some(snippet) = map.get("snippet") {
    out.insert("highlight".into(), json!({ "_snippet": [snippet] }));
  }
  Value::Object(out)
}

fn copy_field(src: &Map<String, Value>, key: &str, dst: &mut Map<String, Value>) {
  if let Some(v) = src.get(key) {
    dst.insert(key.to_string(), v.clone());
  }
}
