use serde_json::{json, Map, Value};

use super::query::translate_to_filter;
use super::unsupported::Unsupported;

const AGG_KEYS: [&str; 2] = ["aggs", "aggregations"];

/// Walk a top-level ES `aggs`/`aggregations` map and collect each agg's
/// `meta` blob keyed by the agg name. SearchLite has no `meta` plumbing, so
/// the route handler stashes these and re-injects them into the response
/// after translation. Only top-level agg metadata is captured in v1 — nested
/// aggs would require recursive walking on both sides; tracked as a known
/// limitation.
pub fn extract_agg_meta(es_aggs: &Map<String, Value>) -> std::collections::BTreeMap<String, Value> {
  let mut out = std::collections::BTreeMap::new();
  for (name, spec) in es_aggs {
    if let Some(meta) = spec.as_object().and_then(|m| m.get("meta")) {
      out.insert(name.clone(), meta.clone());
    }
  }
  out
}

/// Inject collected agg `meta` entries back into a translated ES response.
/// Mutates `response.aggregations.<name>.meta` for each name in `meta`.
pub fn inject_agg_meta(response: &mut Value, meta: &std::collections::BTreeMap<String, Value>) {
  if meta.is_empty() {
    return;
  }
  let Some(aggs) = response
    .as_object_mut()
    .and_then(|m| m.get_mut("aggregations"))
    .and_then(Value::as_object_mut)
  else {
    return;
  };
  for (name, value) in meta {
    if let Some(entry) = aggs.get_mut(name).and_then(Value::as_object_mut) {
      entry.insert("meta".to_string(), value.clone());
    }
  }
}

/// Translate an ES `aggs`/`aggregations` map (name → spec) into SearchLite's
/// agg map (name → tagged Aggregation JSON). Sub-aggregations recurse.
pub fn translate_aggs(es_aggs: &Map<String, Value>) -> Result<Map<String, Value>, Unsupported> {
  let mut out = Map::new();
  for (name, spec) in es_aggs {
    let spec_obj = spec
      .as_object()
      .ok_or_else(|| Unsupported::with_detail("aggregations", "each entry must be an object"))?;
    out.insert(name.clone(), translate_one_agg(name, spec_obj)?);
  }
  Ok(out)
}

fn translate_one_agg(name: &str, spec: &Map<String, Value>) -> Result<Value, Unsupported> {
  let sub_aggs = collect_sub_aggs(spec)?;
  let agg_keys: Vec<&String> = spec
    .keys()
    .filter(|k| !AGG_KEYS.contains(&k.as_str()) && k.as_str() != "meta")
    .collect();
  if agg_keys.len() != 1 {
    return Err(Unsupported::with_detail(
      format!("aggregations.{name}"),
      "expected exactly one aggregation type per entry",
    ));
  }
  let agg_type = agg_keys[0].clone();
  let body = spec
    .get(&agg_type)
    .and_then(Value::as_object)
    .ok_or_else(|| {
      Unsupported::with_detail(format!("aggregations.{name}.{agg_type}"), "expected object")
    })?;

  let translated = match agg_type.as_str() {
    "terms" => translate_terms_agg(body, sub_aggs)?,
    "range" => translate_range_agg(body, sub_aggs)?,
    "date_range" => translate_date_range_agg(body, sub_aggs)?,
    "histogram" => translate_histogram_agg(body, sub_aggs)?,
    "date_histogram" => translate_date_histogram_agg(body, sub_aggs)?,
    "stats" => translate_metric_agg(body, "stats")?,
    "extended_stats" => translate_metric_agg(body, "extended_stats")?,
    "value_count" => translate_metric_agg(body, "value_count")?,
    "cardinality" => translate_cardinality_agg(body)?,
    "percentiles" => translate_percentiles_agg(body)?,
    "percentile_ranks" => translate_percentile_ranks_agg(body)?,
    "top_hits" => translate_top_hits_agg(body)?,
    "filter" => translate_filter_agg(body, sub_aggs)?,
    "nested" => translate_nested_agg(body, sub_aggs)?,
    "composite" => translate_composite_agg(body, sub_aggs)?,
    "bucket_sort" => translate_bucket_sort_agg(body)?,
    "avg_bucket" => translate_bucket_metric_agg(body, "avg_bucket")?,
    "sum_bucket" => translate_bucket_metric_agg(body, "sum_bucket")?,
    "derivative" => translate_derivative_agg(body)?,
    "moving_avg" => translate_moving_avg_agg(body)?,
    "bucket_script" => translate_bucket_script_agg(body)?,
    "significant_terms" => translate_significant_terms_agg(body, sub_aggs)?,
    "rare_terms" => translate_rare_terms_agg(body, sub_aggs)?,
    "avg" | "sum" | "min" | "max" => {
      return Err(Unsupported::with_detail(
        format!("aggregations.{name}.{agg_type}"),
        "use `stats` and read the corresponding field from the response",
      ))
    }
    other => return Err(Unsupported::feature(format!("aggregations.{name}.{other}"))),
  };
  Ok(translated)
}

fn collect_sub_aggs(spec: &Map<String, Value>) -> Result<Option<Value>, Unsupported> {
  let raw = spec.get("aggs").or_else(|| spec.get("aggregations"));
  let Some(raw) = raw else {
    return Ok(None);
  };
  let map = raw
    .as_object()
    .ok_or_else(|| Unsupported::with_detail("aggregations.aggs", "must be an object"))?;
  let translated = translate_aggs(map)?;
  Ok(Some(Value::Object(translated)))
}

fn merge_sub(mut out: Map<String, Value>, sub: Option<Value>) -> Map<String, Value> {
  if let Some(sub) = sub {
    out.insert("aggs".to_string(), sub);
  }
  out
}

// --- terms ---------------------------------------------------------------

fn translate_terms_agg(
  body: &Map<String, Value>,
  sub: Option<Value>,
) -> Result<Value, Unsupported> {
  let field = require_field(body, "terms")?;
  let mut out = Map::new();
  out.insert("type".into(), Value::String("terms".into()));
  out.insert("field".into(), Value::String(field));
  if let Some(size) = body.get("size") {
    out.insert("size".into(), size.clone());
  }
  if let Some(shard_size) = body.get("shard_size") {
    out.insert("shard_size".into(), shard_size.clone());
  }
  if let Some(min_doc) = body.get("min_doc_count") {
    out.insert("min_doc_count".into(), min_doc.clone());
  }
  if let Some(missing) = body.get("missing") {
    out.insert("missing".into(), missing.clone());
  }
  Ok(Value::Object(merge_sub(out, sub)))
}

fn translate_significant_terms_agg(
  body: &Map<String, Value>,
  sub: Option<Value>,
) -> Result<Value, Unsupported> {
  let field = require_field(body, "significant_terms")?;
  let mut out = Map::new();
  out.insert("type".into(), Value::String("significant_terms".into()));
  out.insert("field".into(), Value::String(field));
  if let Some(size) = body.get("size") {
    out.insert("size".into(), size.clone());
  }
  if let Some(min_doc) = body.get("min_doc_count") {
    out.insert("min_doc_count".into(), min_doc.clone());
  }
  if body.contains_key("background_filter") {
    return Err(Unsupported::with_detail(
      "significant_terms.background_filter",
      "not supported in v1 — pass via filter aggregation instead",
    ));
  }
  Ok(Value::Object(merge_sub(out, sub)))
}

fn translate_rare_terms_agg(
  body: &Map<String, Value>,
  sub: Option<Value>,
) -> Result<Value, Unsupported> {
  let field = require_field(body, "rare_terms")?;
  let mut out = Map::new();
  out.insert("type".into(), Value::String("rare_terms".into()));
  out.insert("field".into(), Value::String(field));
  if let Some(max) = body.get("max_doc_count") {
    out.insert("max_doc_count".into(), max.clone());
  }
  if let Some(size) = body.get("size") {
    out.insert("size".into(), size.clone());
  }
  Ok(Value::Object(merge_sub(out, sub)))
}

// --- range / date_range -------------------------------------------------

fn translate_range_agg(
  body: &Map<String, Value>,
  sub: Option<Value>,
) -> Result<Value, Unsupported> {
  let field = require_field(body, "range")?;
  let ranges = body
    .get("ranges")
    .and_then(Value::as_array)
    .ok_or_else(|| Unsupported::with_detail("range", "missing `ranges` array"))?;
  let mut translated_ranges = Vec::with_capacity(ranges.len());
  for r in ranges {
    let r_obj = r
      .as_object()
      .ok_or_else(|| Unsupported::with_detail("range.ranges", "each entry must be an object"))?;
    let mut entry = Map::new();
    if let Some(key) = r_obj.get("key") {
      entry.insert("key".into(), key.clone());
    } else {
      entry.insert("key".into(), Value::Null);
    }
    if let Some(from) = r_obj.get("from") {
      entry.insert("from".into(), from.clone());
    } else {
      entry.insert("from".into(), Value::Null);
    }
    if let Some(to) = r_obj.get("to") {
      entry.insert("to".into(), to.clone());
    } else {
      entry.insert("to".into(), Value::Null);
    }
    translated_ranges.push(Value::Object(entry));
  }
  let keyed = body.get("keyed").and_then(Value::as_bool).unwrap_or(false);
  let mut out = Map::new();
  out.insert("type".into(), Value::String("range".into()));
  out.insert("field".into(), Value::String(field));
  out.insert("keyed".into(), Value::Bool(keyed));
  out.insert("ranges".into(), Value::Array(translated_ranges));
  out.insert(
    "missing".into(),
    body.get("missing").cloned().unwrap_or(Value::Null),
  );
  Ok(Value::Object(merge_sub(out, sub)))
}

fn translate_date_range_agg(
  body: &Map<String, Value>,
  sub: Option<Value>,
) -> Result<Value, Unsupported> {
  let field = require_field(body, "date_range")?;
  let ranges = body
    .get("ranges")
    .and_then(Value::as_array)
    .ok_or_else(|| Unsupported::with_detail("date_range", "missing `ranges` array"))?;
  let translated_ranges: Vec<Value> = ranges
    .iter()
    .map(|r| {
      let r_obj = r.as_object().ok_or_else(|| {
        Unsupported::with_detail("date_range.ranges", "each entry must be an object")
      })?;
      let mut entry = Map::new();
      entry.insert(
        "key".into(),
        r_obj.get("key").cloned().unwrap_or(Value::Null),
      );
      entry.insert(
        "from".into(),
        r_obj.get("from").cloned().unwrap_or(Value::Null),
      );
      entry.insert("to".into(), r_obj.get("to").cloned().unwrap_or(Value::Null));
      Ok(Value::Object(entry))
    })
    .collect::<Result<Vec<_>, Unsupported>>()?;
  let keyed = body.get("keyed").and_then(Value::as_bool).unwrap_or(false);
  let mut out = Map::new();
  out.insert("type".into(), Value::String("date_range".into()));
  out.insert("field".into(), Value::String(field));
  out.insert("keyed".into(), Value::Bool(keyed));
  out.insert(
    "format".into(),
    body.get("format").cloned().unwrap_or(Value::Null),
  );
  out.insert("ranges".into(), Value::Array(translated_ranges));
  out.insert(
    "missing".into(),
    body.get("missing").cloned().unwrap_or(Value::Null),
  );
  Ok(Value::Object(merge_sub(out, sub)))
}

// --- histogram / date_histogram -----------------------------------------

fn translate_histogram_agg(
  body: &Map<String, Value>,
  sub: Option<Value>,
) -> Result<Value, Unsupported> {
  let field = require_field(body, "histogram")?;
  let interval = body
    .get("interval")
    .and_then(Value::as_f64)
    .ok_or_else(|| Unsupported::with_detail("histogram.interval", "must be a number"))?;
  let mut out = Map::new();
  out.insert("type".into(), Value::String("histogram".into()));
  out.insert("field".into(), Value::String(field));
  out.insert("interval".into(), Value::from(interval));
  out.insert(
    "offset".into(),
    body.get("offset").cloned().unwrap_or(Value::Null),
  );
  out.insert(
    "min_doc_count".into(),
    body.get("min_doc_count").cloned().unwrap_or(Value::Null),
  );
  out.insert(
    "extended_bounds".into(),
    body.get("extended_bounds").cloned().unwrap_or(Value::Null),
  );
  out.insert(
    "hard_bounds".into(),
    body.get("hard_bounds").cloned().unwrap_or(Value::Null),
  );
  out.insert(
    "missing".into(),
    body.get("missing").cloned().unwrap_or(Value::Null),
  );
  Ok(Value::Object(merge_sub(out, sub)))
}

fn translate_date_histogram_agg(
  body: &Map<String, Value>,
  sub: Option<Value>,
) -> Result<Value, Unsupported> {
  let field = require_field(body, "date_histogram")?;
  let mut out = Map::new();
  out.insert("type".into(), Value::String("date_histogram".into()));
  out.insert("field".into(), Value::String(field));
  out.insert(
    "calendar_interval".into(),
    body
      .get("calendar_interval")
      .cloned()
      .unwrap_or(Value::Null),
  );
  out.insert(
    "fixed_interval".into(),
    body.get("fixed_interval").cloned().unwrap_or(Value::Null),
  );
  out.insert(
    "offset".into(),
    body.get("offset").cloned().unwrap_or(Value::Null),
  );
  out.insert(
    "format".into(),
    body.get("format").cloned().unwrap_or(Value::Null),
  );
  out.insert(
    "min_doc_count".into(),
    body.get("min_doc_count").cloned().unwrap_or(Value::Null),
  );
  out.insert(
    "extended_bounds".into(),
    body.get("extended_bounds").cloned().unwrap_or(Value::Null),
  );
  out.insert(
    "hard_bounds".into(),
    body.get("hard_bounds").cloned().unwrap_or(Value::Null),
  );
  out.insert(
    "missing".into(),
    body.get("missing").cloned().unwrap_or(Value::Null),
  );
  Ok(Value::Object(merge_sub(out, sub)))
}

// --- metric aggregations -------------------------------------------------

fn translate_metric_agg(body: &Map<String, Value>, kind: &str) -> Result<Value, Unsupported> {
  let field = require_field(body, kind)?;
  Ok(json!({
    "type": kind,
    "field": field,
    "missing": body.get("missing").cloned().unwrap_or(Value::Null),
  }))
}

fn translate_cardinality_agg(body: &Map<String, Value>) -> Result<Value, Unsupported> {
  let field = require_field(body, "cardinality")?;
  let precision = body
    .get("precision_threshold")
    .cloned()
    .unwrap_or(Value::Null);
  Ok(json!({
    "type": "cardinality",
    "field": field,
    "precision_threshold": precision,
  }))
}

fn translate_percentiles_agg(body: &Map<String, Value>) -> Result<Value, Unsupported> {
  let field = require_field(body, "percentiles")?;
  let percents = body
    .get("percents")
    .cloned()
    .unwrap_or_else(|| json!([1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0]));
  Ok(json!({
    "type": "percentiles",
    "field": field,
    "percents": percents,
  }))
}

fn translate_percentile_ranks_agg(body: &Map<String, Value>) -> Result<Value, Unsupported> {
  let field = require_field(body, "percentile_ranks")?;
  let values = body
    .get("values")
    .cloned()
    .ok_or_else(|| Unsupported::with_detail("percentile_ranks.values", "missing array"))?;
  Ok(json!({
    "type": "percentile_ranks",
    "field": field,
    "values": values,
  }))
}

fn translate_top_hits_agg(body: &Map<String, Value>) -> Result<Value, Unsupported> {
  let size = body.get("size").and_then(Value::as_u64).unwrap_or(3);
  let from = body.get("from").and_then(Value::as_u64).unwrap_or(0);
  let mut out = Map::new();
  out.insert("type".into(), Value::String("top_hits".into()));
  out.insert("size".into(), Value::from(size));
  out.insert("from".into(), Value::from(from));
  if let Some(fields) = body
    .get("_source")
    .and_then(|v| v.as_object().and_then(|o| o.get("includes")))
  {
    out.insert("fields".into(), fields.clone());
  } else if let Some(fields) = body.get("fields") {
    out.insert("fields".into(), fields.clone());
  }
  if let Some(sort) = body.get("sort") {
    out.insert("sort".into(), super::sort::translate_sort(sort)?.into());
  }
  Ok(Value::Object(out))
}

// --- structural buckets --------------------------------------------------

fn translate_filter_agg(
  body: &Map<String, Value>,
  sub: Option<Value>,
) -> Result<Value, Unsupported> {
  let filter = translate_to_filter(&Value::Object(body.clone()))?;
  let mut out = Map::new();
  out.insert("type".into(), Value::String("filter".into()));
  out.insert("filter".into(), filter);
  Ok(Value::Object(merge_sub(out, sub)))
}

fn translate_nested_agg(
  body: &Map<String, Value>,
  sub: Option<Value>,
) -> Result<Value, Unsupported> {
  let path = body
    .get("path")
    .and_then(Value::as_str)
    .ok_or_else(|| Unsupported::with_detail("nested.path", "missing string"))?
    .to_string();
  let mut out = Map::new();
  out.insert("type".into(), Value::String("nested".into()));
  out.insert("path".into(), Value::String(path));
  Ok(Value::Object(merge_sub(out, sub)))
}

fn translate_composite_agg(
  body: &Map<String, Value>,
  sub: Option<Value>,
) -> Result<Value, Unsupported> {
  let sources = body
    .get("sources")
    .and_then(Value::as_array)
    .ok_or_else(|| Unsupported::with_detail("composite.sources", "missing array"))?;
  let translated_sources: Vec<Value> = sources
    .iter()
    .map(|src| {
      let src_obj = src.as_object().ok_or_else(|| {
        Unsupported::with_detail("composite.sources", "each source must be an object")
      })?;
      if src_obj.len() != 1 {
        return Err(Unsupported::with_detail(
          "composite.sources",
          "each source must contain exactly one named entry",
        ));
      }
      let (name, body) = src_obj.iter().next().unwrap();
      let body = body.as_object().ok_or_else(|| {
        Unsupported::with_detail("composite.sources", "source spec must be object")
      })?;
      if body.len() != 1 {
        return Err(Unsupported::with_detail(
          "composite.sources",
          "source spec must contain exactly one type",
        ));
      }
      let (kind, spec) = body.iter().next().unwrap();
      let spec = spec
        .as_object()
        .ok_or_else(|| Unsupported::with_detail("composite.sources", "type spec must be object"))?;
      let field = spec
        .get("field")
        .and_then(Value::as_str)
        .ok_or_else(|| Unsupported::with_detail("composite.sources.field", "missing"))?
        .to_string();
      match kind.as_str() {
        "terms" => Ok(json!({ "type": "terms", "name": name, "field": field })),
        "histogram" => {
          let interval = spec
            .get("interval")
            .and_then(Value::as_f64)
            .ok_or_else(|| Unsupported::with_detail("composite.histogram.interval", "missing"))?;
          Ok(json!({ "type": "histogram", "name": name, "field": field, "interval": interval }))
        }
        other => Err(Unsupported::feature(format!("composite.sources.{other}"))),
      }
    })
    .collect::<Result<Vec<_>, Unsupported>>()?;
  let size = body.get("size").and_then(Value::as_u64).unwrap_or(10) as usize;
  let mut out = Map::new();
  out.insert("type".into(), Value::String("composite".into()));
  out.insert("sources".into(), Value::Array(translated_sources));
  out.insert("size".into(), Value::from(size));
  if let Some(after) = body.get("after") {
    out.insert("after".into(), after.clone());
  }
  Ok(Value::Object(merge_sub(out, sub)))
}

// --- pipeline aggregations ----------------------------------------------

fn translate_bucket_sort_agg(body: &Map<String, Value>) -> Result<Value, Unsupported> {
  let mut out = Map::new();
  out.insert("type".into(), Value::String("bucket_sort".into()));
  out.insert(
    "from".into(),
    body.get("from").cloned().unwrap_or(Value::from(0u64)),
  );
  if let Some(size) = body.get("size") {
    out.insert("size".into(), size.clone());
  }
  if let Some(sort) = body.get("sort") {
    out.insert(
      "sort".into(),
      Value::Array(super::sort::translate_sort(sort)?),
    );
  }
  Ok(Value::Object(out))
}

fn translate_bucket_metric_agg(
  body: &Map<String, Value>,
  kind: &str,
) -> Result<Value, Unsupported> {
  let buckets_path = body
    .get("buckets_path")
    .and_then(Value::as_str)
    .ok_or_else(|| Unsupported::with_detail(format!("{kind}.buckets_path"), "missing"))?
    .to_string();
  Ok(json!({ "type": kind, "buckets_path": buckets_path }))
}

fn translate_derivative_agg(body: &Map<String, Value>) -> Result<Value, Unsupported> {
  let buckets_path = body
    .get("buckets_path")
    .and_then(Value::as_str)
    .ok_or_else(|| Unsupported::with_detail("derivative.buckets_path", "missing"))?
    .to_string();
  Ok(json!({ "type": "derivative", "buckets_path": buckets_path }))
}

fn translate_moving_avg_agg(body: &Map<String, Value>) -> Result<Value, Unsupported> {
  let buckets_path = body
    .get("buckets_path")
    .and_then(Value::as_str)
    .ok_or_else(|| Unsupported::with_detail("moving_avg.buckets_path", "missing"))?
    .to_string();
  let mut out = Map::new();
  out.insert("type".into(), Value::String("moving_avg".into()));
  out.insert("buckets_path".into(), Value::String(buckets_path));
  if let Some(window) = body.get("window") {
    out.insert("window".into(), window.clone());
  }
  if let Some(model) = body.get("model") {
    out.insert("model".into(), model.clone());
  }
  if let Some(predict) = body.get("predict") {
    out.insert("predict".into(), predict.clone());
  }
  Ok(Value::Object(out))
}

fn translate_bucket_script_agg(body: &Map<String, Value>) -> Result<Value, Unsupported> {
  let buckets_path = body
    .get("buckets_path")
    .cloned()
    .ok_or_else(|| Unsupported::with_detail("bucket_script.buckets_path", "missing"))?;
  let script = body
    .get("script")
    .cloned()
    .ok_or_else(|| Unsupported::with_detail("bucket_script.script", "missing"))?;
  Ok(json!({
    "type": "bucket_script",
    "buckets_path": buckets_path,
    "script": script,
  }))
}

// --- helpers -------------------------------------------------------------

fn require_field(body: &Map<String, Value>, kind: &str) -> Result<String, Unsupported> {
  body
    .get("field")
    .and_then(Value::as_str)
    .map(str::to_string)
    .ok_or_else(|| Unsupported::with_detail(format!("{kind}.field"), "missing string"))
}
