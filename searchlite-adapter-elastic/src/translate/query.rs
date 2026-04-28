use serde_json::{json, Map, Value};

use super::unsupported::Unsupported;

/// Translate a single ES query clause into SearchLite's tagged QueryNode JSON.
///
/// The input is the value of the `query` key from the ES `_search` body — i.e.
/// an object containing exactly one top-level clause name (e.g. `{"match_all": {}}`).
pub fn translate_query(es: &Value) -> Result<Value, Unsupported> {
  let map = es
    .as_object()
    .ok_or_else(|| Unsupported::with_detail("query", "expected an object"))?;
  if map.is_empty() {
    return Ok(json!({ "type": "match_all" }));
  }
  if map.len() != 1 {
    return Err(Unsupported::with_detail(
      "query",
      "expected exactly one top-level clause",
    ));
  }
  let (clause, body) = map.iter().next().unwrap();
  translate_clause(clause, body)
}

fn translate_clause(clause: &str, body: &Value) -> Result<Value, Unsupported> {
  match clause {
    "match_all" => translate_match_all(body),
    "match_none" => Err(Unsupported::feature("match_none")),
    "match" => translate_match(body),
    "match_phrase" => translate_match_phrase(body, false),
    "match_phrase_prefix" => Err(Unsupported::feature("match_phrase_prefix")),
    "multi_match" => translate_multi_match(body),
    "term" => translate_term(body),
    "terms" => translate_terms(body),
    "prefix" => translate_prefix(body),
    "wildcard" => translate_wildcard(body),
    "regexp" => translate_regexp(body),
    "range" => translate_range_clause(body),
    "exists" => translate_exists(body),
    "bool" => translate_bool(body),
    "constant_score" => translate_constant_score(body),
    "dis_max" => translate_dis_max(body),
    "query_string" => translate_query_string(body),
    "simple_query_string" => translate_query_string(body),
    other => Err(Unsupported::feature(other)),
  }
}

// --- match_all -----------------------------------------------------------

fn translate_match_all(body: &Value) -> Result<Value, Unsupported> {
  let mut out = Map::new();
  out.insert("type".into(), Value::String("match_all".into()));
  if let Some(boost) = body.as_object().and_then(|m| m.get("boost")) {
    out.insert("boost".into(), boost.clone());
  }
  Ok(Value::Object(out))
}

// --- match ---------------------------------------------------------------

fn translate_match(body: &Value) -> Result<Value, Unsupported> {
  let map = body
    .as_object()
    .ok_or_else(|| Unsupported::with_detail("match", "expected an object"))?;
  if map.len() != 1 {
    return Err(Unsupported::with_detail(
      "match",
      "expected exactly one field",
    ));
  }
  let (field, spec) = map.iter().next().unwrap();
  let parsed = parse_match_spec(spec)?;

  let mut out = Map::new();
  out.insert("type".into(), Value::String("query_string".into()));
  out.insert("query".into(), Value::String(parsed.query));
  out.insert(
    "fields".into(),
    Value::Array(vec![Value::String(field.clone())]),
  );
  if let Some(b) = parsed.boost {
    out.insert("boost".into(), Value::from(b));
  }
  if parsed.fuzziness.is_some() {
    return Err(Unsupported::with_detail(
      "match.fuzziness",
      "use multi_match for fuzziness",
    ));
  }
  if let Some(op) = parsed.operator {
    let upper = op.to_ascii_lowercase();
    if upper != "or" {
      return Err(Unsupported::with_detail(
        "match.operator",
        "only `or` is supported for `match`",
      ));
    }
  }
  Ok(Value::Object(out))
}

struct MatchSpec {
  query: String,
  boost: Option<f64>,
  fuzziness: Option<String>,
  operator: Option<String>,
}

fn parse_match_spec(spec: &Value) -> Result<MatchSpec, Unsupported> {
  match spec {
    Value::String(s) => Ok(MatchSpec {
      query: s.clone(),
      boost: None,
      fuzziness: None,
      operator: None,
    }),
    Value::Number(n) => Ok(MatchSpec {
      query: n.to_string(),
      boost: None,
      fuzziness: None,
      operator: None,
    }),
    Value::Bool(b) => Ok(MatchSpec {
      query: b.to_string(),
      boost: None,
      fuzziness: None,
      operator: None,
    }),
    Value::Object(map) => {
      let query = map
        .get("query")
        .and_then(|v| match v {
          Value::String(s) => Some(s.clone()),
          Value::Number(n) => Some(n.to_string()),
          Value::Bool(b) => Some(b.to_string()),
          _ => None,
        })
        .ok_or_else(|| Unsupported::with_detail("match", "missing string `query`"))?;
      Ok(MatchSpec {
        query,
        boost: map.get("boost").and_then(Value::as_f64),
        fuzziness: map
          .get("fuzziness")
          .and_then(Value::as_str)
          .map(str::to_string),
        operator: map
          .get("operator")
          .and_then(Value::as_str)
          .map(str::to_string),
      })
    }
    _ => Err(Unsupported::with_detail(
      "match",
      "spec must be string or object",
    )),
  }
}

// --- match_phrase --------------------------------------------------------

fn translate_match_phrase(body: &Value, _prefix: bool) -> Result<Value, Unsupported> {
  let map = body
    .as_object()
    .ok_or_else(|| Unsupported::with_detail("match_phrase", "expected an object"))?;
  if map.len() != 1 {
    return Err(Unsupported::with_detail(
      "match_phrase",
      "expected exactly one field",
    ));
  }
  let (field, spec) = map.iter().next().unwrap();
  let (text, slop, boost) = match spec {
    Value::String(s) => (s.clone(), None, None),
    Value::Object(opts) => {
      let q = opts
        .get("query")
        .and_then(Value::as_str)
        .ok_or_else(|| Unsupported::with_detail("match_phrase", "missing string `query`"))?
        .to_string();
      let slop = opts.get("slop").and_then(Value::as_u64);
      let boost = opts.get("boost").and_then(Value::as_f64);
      (q, slop, boost)
    }
    _ => {
      return Err(Unsupported::with_detail(
        "match_phrase",
        "spec must be string or object",
      ))
    }
  };

  // Slop=0 (the default) → emit a `query_string` with a quoted phrase so
  // tokenization runs through the field's analyzer (matching ES behaviour
  // for punctuation, contractions, non-ASCII). For slop > 0 we fall back
  // to the literal-token `phrase` translation since `query_string` quoted
  // phrases don't model token proximity.
  let slop_value = slop.unwrap_or(0);
  if slop_value == 0 {
    let mut out = Map::new();
    out.insert("type".into(), Value::String("query_string".into()));
    out.insert(
      "query".into(),
      Value::String(quote_phrase_for_query_string(&text)),
    );
    out.insert(
      "fields".into(),
      Value::Array(vec![Value::String(field.clone())]),
    );
    if let Some(b) = boost {
      out.insert("boost".into(), Value::from(b));
    }
    return Ok(Value::Object(out));
  }

  let terms: Vec<Value> = text
    .split_whitespace()
    .map(|t| Value::String(t.to_string()))
    .collect();
  let mut out = Map::new();
  out.insert("type".into(), Value::String("phrase".into()));
  out.insert("field".into(), Value::String(field.clone()));
  out.insert("terms".into(), Value::Array(terms));
  out.insert("slop".into(), Value::from(slop_value));
  if let Some(b) = boost {
    out.insert("boost".into(), Value::from(b));
  }
  Ok(Value::Object(out))
}

/// Wrap text in quotes for `query_string` syntax, escaping the only two
/// characters that have meaning inside a quoted phrase: `\` and `"`.
fn quote_phrase_for_query_string(text: &str) -> String {
  let mut out = String::with_capacity(text.len() + 2);
  out.push('"');
  for ch in text.chars() {
    if ch == '"' || ch == '\\' {
      out.push('\\');
    }
    out.push(ch);
  }
  out.push('"');
  out
}

// --- multi_match ---------------------------------------------------------

fn translate_multi_match(body: &Value) -> Result<Value, Unsupported> {
  let map = body
    .as_object()
    .ok_or_else(|| Unsupported::with_detail("multi_match", "expected an object"))?;
  let query = map
    .get("query")
    .and_then(Value::as_str)
    .ok_or_else(|| Unsupported::with_detail("multi_match", "missing string `query`"))?
    .to_string();
  let fields = map
    .get("fields")
    .and_then(Value::as_array)
    .ok_or_else(|| Unsupported::with_detail("multi_match", "missing `fields` array"))?
    .iter()
    .map(translate_field_spec)
    .collect::<Result<Vec<_>, _>>()?;

  let mut out = Map::new();
  out.insert("type".into(), Value::String("multi_match".into()));
  out.insert("query".into(), Value::String(query));
  out.insert("fields".into(), Value::Array(fields));

  if let Some(t) = map.get("type").and_then(Value::as_str) {
    let t_norm = match t {
      "best_fields" | "most_fields" | "cross_fields" => t.to_string(),
      "phrase" | "phrase_prefix" | "bool_prefix" => {
        // Don't recommend best_fields/most_fields/cross_fields here — they
        // are bag-of-words modes with different matching semantics. Phrase
        // intent is closer to `match_phrase` per field; we surface that
        // hint without claiming an exact equivalent.
        return Err(Unsupported::with_detail(
          "multi_match.type",
          format!(
            "`{t}` not supported in v1; consider issuing `match_phrase` per field combined under `dis_max`"
          ),
        ));
      }
      other => {
        return Err(Unsupported::with_detail(
          "multi_match.type",
          format!("unknown match_type `{other}`"),
        ))
      }
    };
    out.insert("match_type".into(), Value::String(t_norm));
  }

  if let Some(fz) = map.get("fuzziness") {
    out.insert("fuzziness".into(), translate_fuzziness(fz)?);
  }
  if let Some(tb) = map.get("tie_breaker") {
    out.insert("tie_breaker".into(), tb.clone());
  }
  if let Some(op) = map.get("operator").and_then(Value::as_str) {
    out.insert("operator".into(), Value::String(op.to_ascii_lowercase()));
  }
  if let Some(msm) = map.get("minimum_should_match") {
    out.insert("minimum_should_match".into(), msm.clone());
  }
  if let Some(boost) = map.get("boost") {
    out.insert("boost".into(), boost.clone());
  }
  Ok(Value::Object(out))
}

fn translate_field_spec(spec: &Value) -> Result<Value, Unsupported> {
  match spec {
    Value::String(s) => {
      if let Some((name, boost)) = s.split_once('^') {
        let boost: f64 = boost
          .parse()
          .map_err(|_| Unsupported::with_detail("fields", format!("invalid boost in `{s}`")))?;
        Ok(json!({ "field": name, "boost": boost }))
      } else {
        Ok(json!({ "field": s }))
      }
    }
    Value::Object(_) => Ok(spec.clone()),
    _ => Err(Unsupported::with_detail(
      "fields",
      "must be string or object",
    )),
  }
}

fn translate_fuzziness(value: &Value) -> Result<Value, Unsupported> {
  // SearchLite expects MultiMatchFuzziness shape — accept ES "AUTO" or numeric 0/1/2.
  match value {
    Value::String(s) => match s.to_ascii_uppercase().as_str() {
      "AUTO" => Ok(json!({ "max_edits": 2 })),
      "0" => Ok(json!({ "max_edits": 0 })),
      "1" => Ok(json!({ "max_edits": 1 })),
      "2" => Ok(json!({ "max_edits": 2 })),
      other => Err(Unsupported::with_detail(
        "fuzziness",
        format!("unknown value `{other}`"),
      )),
    },
    Value::Number(n) => {
      let edits = n
        .as_u64()
        .ok_or_else(|| Unsupported::with_detail("fuzziness", "must be a non-negative integer"))?;
      if edits > 2 {
        return Err(Unsupported::with_detail(
          "fuzziness",
          "max_edits must be 0, 1, or 2",
        ));
      }
      Ok(json!({ "max_edits": edits }))
    }
    _ => Err(Unsupported::with_detail(
      "fuzziness",
      "must be string or number",
    )),
  }
}

// --- term / terms --------------------------------------------------------

/// Classified ES `term`/`terms` value, preserving numeric type so we can route
/// integer/float terms to SearchLite's I64Range/F64Range filters instead of
/// stringifying them as keywords (which never matches a numeric field).
enum TermValue {
  Keyword(String),
  I64(i64),
  F64(f64),
}

fn classify_term_value(
  spec: &Value,
  clause: &str,
) -> Result<(TermValue, Option<f64>), Unsupported> {
  match spec {
    Value::String(s) => Ok((TermValue::Keyword(s.clone()), None)),
    Value::Bool(b) => Ok((TermValue::Keyword(b.to_string()), None)),
    Value::Number(n) => {
      let value = numeric_term(n, clause)?;
      Ok((value, None))
    }
    Value::Object(opts) => {
      let inner = opts
        .get("value")
        .or_else(|| opts.get("query"))
        .ok_or_else(|| Unsupported::with_detail(clause, "missing `value`"))?;
      let (value, _) = classify_term_value(inner, clause)?;
      let boost = opts.get("boost").and_then(Value::as_f64);
      Ok((value, boost))
    }
    _ => Err(Unsupported::with_detail(
      clause,
      "spec must be primitive or object",
    )),
  }
}

fn numeric_term(n: &serde_json::Number, clause: &str) -> Result<TermValue, Unsupported> {
  if let Some(i) = n.as_i64() {
    return Ok(TermValue::I64(i));
  }
  if let Some(u) = n.as_u64() {
    return match i64::try_from(u) {
      Ok(i) => Ok(TermValue::I64(i)),
      Err(_) => Ok(TermValue::F64(u as f64)),
    };
  }
  if let Some(f) = n.as_f64() {
    return Ok(TermValue::F64(f));
  }
  Err(Unsupported::with_detail(
    clause,
    "numeric value not representable as i64 or f64",
  ))
}

fn translate_term(body: &Value) -> Result<Value, Unsupported> {
  let map = body
    .as_object()
    .ok_or_else(|| Unsupported::with_detail("term", "expected an object"))?;
  if map.len() != 1 {
    return Err(Unsupported::with_detail(
      "term",
      "expected exactly one field",
    ));
  }
  let (field, spec) = map.iter().next().unwrap();
  let (value, boost) = classify_term_value(spec, "term")?;
  match value {
    TermValue::Keyword(s) => {
      let mut out = Map::new();
      out.insert("type".into(), Value::String("term".into()));
      out.insert("field".into(), Value::String(field.clone()));
      out.insert("value".into(), Value::String(s));
      if let Some(b) = boost {
        out.insert("boost".into(), Value::from(b));
      }
      Ok(Value::Object(out))
    }
    TermValue::I64(n) => Ok(numeric_term_query(
      field,
      json!({ "I64Range": { "field": field, "min": n, "max": n } }),
      boost,
    )),
    TermValue::F64(f) => Ok(numeric_term_query(
      field,
      json!({ "F64Range": { "field": field, "min": f, "max": f } }),
      boost,
    )),
  }
}

/// ES `term` against a numeric field is constant-score scoping to a single
/// value. SearchLite has no native numeric `term`, so wrap an equality range
/// (min == max) in `constant_score` to preserve scoring semantics.
fn numeric_term_query(_field: &str, filter: Value, boost: Option<f64>) -> Value {
  let mut out = Map::new();
  out.insert("type".into(), Value::String("constant_score".into()));
  out.insert("filter".into(), filter);
  if let Some(b) = boost {
    out.insert("boost".into(), Value::from(b));
  }
  Value::Object(out)
}

fn translate_terms(body: &Value) -> Result<Value, Unsupported> {
  let map = body
    .as_object()
    .ok_or_else(|| Unsupported::with_detail("terms", "expected an object"))?;
  // Only treat `boost` as the meta-key when its value is numeric (the only
  // shape ES documents). Otherwise it's a legitimate field name — silently
  // filtering it would make any schema with a `boost` column unqueryable.
  let mut field_iter = map
    .iter()
    .filter(|(k, v)| !(*k == "boost" && v.is_number()));
  let (field, spec) = field_iter
    .next()
    .ok_or_else(|| Unsupported::with_detail("terms", "missing field entry"))?;
  if field_iter.next().is_some() {
    return Err(Unsupported::with_detail(
      "terms",
      "expected exactly one field entry",
    ));
  }
  let values = spec
    .as_array()
    .ok_or_else(|| Unsupported::with_detail("terms", "field value must be an array"))?;
  // Per-value dispatch by type so numeric inputs route to range filters
  // (which match numeric fields upstream) instead of being stringified into
  // keyword `term` queries that never match.
  let shoulds: Vec<Value> = values
    .iter()
    .map(|v| terms_value_to_query_node(field, v))
    .collect::<Result<Vec<_>, Unsupported>>()?;

  Ok(json!({
    "type": "bool",
    "should": shoulds,
    "minimum_should_match": 1,
  }))
}

fn terms_value_to_query_node(field: &str, value: &Value) -> Result<Value, Unsupported> {
  match classify_terms_element(value)? {
    TermValue::Keyword(s) => Ok(json!({ "type": "term", "field": field, "value": s })),
    TermValue::I64(n) => Ok(json!({
      "type": "constant_score",
      "filter": { "I64Range": { "field": field, "min": n, "max": n } },
    })),
    TermValue::F64(f) => Ok(json!({
      "type": "constant_score",
      "filter": { "F64Range": { "field": field, "min": f, "max": f } },
    })),
  }
}

fn classify_terms_element(value: &Value) -> Result<TermValue, Unsupported> {
  match value {
    Value::String(s) => Ok(TermValue::Keyword(s.clone())),
    Value::Bool(b) => Ok(TermValue::Keyword(b.to_string())),
    Value::Number(n) => numeric_term(n, "terms"),
    _ => Err(Unsupported::with_detail(
      "terms",
      "values must be primitives (string, boolean, or number)",
    )),
  }
}

fn parse_value_or_object(spec: &Value, clause: &str) -> Result<(String, Option<f64>), Unsupported> {
  match spec {
    Value::String(s) => Ok((s.clone(), None)),
    Value::Number(n) => Ok((n.to_string(), None)),
    Value::Bool(b) => Ok((b.to_string(), None)),
    Value::Object(opts) => {
      let v = opts
        .get("value")
        .or_else(|| opts.get("query"))
        .ok_or_else(|| Unsupported::with_detail(clause, "missing `value`"))?;
      let primitive = match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        _ => {
          return Err(Unsupported::with_detail(
            clause,
            "value must be a primitive",
          ))
        }
      };
      let boost = opts.get("boost").and_then(Value::as_f64);
      Ok((primitive, boost))
    }
    _ => Err(Unsupported::with_detail(
      clause,
      "spec must be primitive or object",
    )),
  }
}

// --- prefix / wildcard / regexp -----------------------------------------

fn translate_prefix(body: &Value) -> Result<Value, Unsupported> {
  field_value_clause(body, "prefix", "prefix")
}

fn translate_wildcard(body: &Value) -> Result<Value, Unsupported> {
  field_value_clause(body, "wildcard", "wildcard")
}

fn translate_regexp(body: &Value) -> Result<Value, Unsupported> {
  field_value_clause(body, "regexp", "regex")
}

fn field_value_clause(body: &Value, es_clause: &str, sl_type: &str) -> Result<Value, Unsupported> {
  let map = body
    .as_object()
    .ok_or_else(|| Unsupported::with_detail(es_clause, "expected an object"))?;
  if map.len() != 1 {
    return Err(Unsupported::with_detail(
      es_clause,
      "expected exactly one field",
    ));
  }
  let (field, spec) = map.iter().next().unwrap();
  let (value, boost) = parse_value_or_object(spec, es_clause)?;
  let mut out = Map::new();
  out.insert("type".into(), Value::String(sl_type.into()));
  out.insert("field".into(), Value::String(field.clone()));
  out.insert("value".into(), Value::String(value));
  if let Some(b) = boost {
    out.insert("boost".into(), Value::from(b));
  }
  Ok(Value::Object(out))
}

// --- range ---------------------------------------------------------------

/// `range` is rendered as a SearchLite `constant_score` wrapping a Filter so
/// it can be used in any query position. When used inside `bool.filter` we
/// translate to a raw filter via [`translate_range_to_filter`].
fn translate_range_clause(body: &Value) -> Result<Value, Unsupported> {
  let filter = translate_range_to_filter(body)?;
  Ok(json!({ "type": "constant_score", "filter": filter }))
}

pub fn translate_range_to_filter(body: &Value) -> Result<Value, Unsupported> {
  let map = body
    .as_object()
    .ok_or_else(|| Unsupported::with_detail("range", "expected an object"))?;
  if map.len() != 1 {
    return Err(Unsupported::with_detail(
      "range",
      "expected exactly one field",
    ));
  }
  let (field, spec) = map.iter().next().unwrap();
  let opts = spec
    .as_object()
    .ok_or_else(|| Unsupported::with_detail("range", "spec must be an object"))?;

  let gte = opts.get("gte");
  let gt = opts.get("gt");
  let lte = opts.get("lte");
  let lt = opts.get("lt");

  if opts.get("format").is_some() {
    return Err(Unsupported::with_detail(
      "range.format",
      "date format strings not supported; supply numeric or epoch_ms values",
    ));
  }
  if opts.get("time_zone").is_some() {
    return Err(Unsupported::with_detail("range.time_zone", "not supported"));
  }
  if opts.get("relation").is_some() {
    return Err(Unsupported::with_detail("range.relation", "not supported"));
  }

  // Determine numeric kind from any provided bound.
  let any = gte.or(gt).or(lte).or(lt).ok_or_else(|| {
    Unsupported::with_detail("range", "must specify at least one of gte/gt/lte/lt")
  })?;

  let is_int = is_integer_value(any)
    && [gte, gt, lte, lt]
      .iter()
      .all(|b| b.map(is_integer_value).unwrap_or(true));

  if is_int {
    let min = i64_lower(gte, gt)?;
    let max = i64_upper(lte, lt)?;
    // `gt: i64::MAX` and `lt: i64::MIN` are exclusive bounds with no
    // representable integer satisfying them; either bound returning `None`
    // means the range is impossible. Emit a deliberately-empty range
    // (`min > max`) so the upstream filter matches zero documents.
    let (min, max) = match (min, max) {
      (Some(min), Some(max)) => (min, max),
      _ => (i64::MAX, i64::MIN),
    };
    Ok(json!({ "I64Range": { "field": field, "min": min, "max": max } }))
  } else {
    let min = f64_lower(gte, gt)?;
    let max = f64_upper(lte, lt)?;
    Ok(json!({ "F64Range": { "field": field, "min": min, "max": max } }))
  }
}

fn is_integer_value(v: &Value) -> bool {
  match v {
    Value::Number(n) => n.is_i64() || n.is_u64(),
    _ => false,
  }
}

/// Returns the inclusive lower bound, or `None` when the bound is
/// representable in input but unrepresentable in i64 after the exclusive
/// adjustment (e.g. `gt: i64::MAX` has no integer satisfying it).
fn i64_lower(gte: Option<&Value>, gt: Option<&Value>) -> Result<Option<i64>, Unsupported> {
  if let Some(v) = gte {
    let n = v
      .as_i64()
      .or_else(|| v.as_u64().and_then(|n| i64::try_from(n).ok()))
      .ok_or_else(|| Unsupported::with_detail("range.gte", "must fit in i64"))?;
    return Ok(Some(n));
  }
  if let Some(v) = gt {
    let n = v
      .as_i64()
      .or_else(|| v.as_u64().and_then(|n| i64::try_from(n).ok()))
      .ok_or_else(|| Unsupported::with_detail("range.gt", "must fit in i64"))?;
    // checked_add returns None for `gt: i64::MAX` since no integer is > i64::MAX.
    return Ok(n.checked_add(1));
  }
  Ok(Some(i64::MIN))
}

/// Returns the inclusive upper bound, or `None` when `lt` is set to
/// `i64::MIN` (no integer satisfies the predicate).
fn i64_upper(lte: Option<&Value>, lt: Option<&Value>) -> Result<Option<i64>, Unsupported> {
  if let Some(v) = lte {
    let n = v
      .as_i64()
      .or_else(|| v.as_u64().and_then(|n| i64::try_from(n).ok()))
      .ok_or_else(|| Unsupported::with_detail("range.lte", "must fit in i64"))?;
    return Ok(Some(n));
  }
  if let Some(v) = lt {
    let n = v
      .as_i64()
      .or_else(|| v.as_u64().and_then(|n| i64::try_from(n).ok()))
      .ok_or_else(|| Unsupported::with_detail("range.lt", "must fit in i64"))?;
    return Ok(n.checked_sub(1));
  }
  Ok(Some(i64::MAX))
}

fn f64_lower(gte: Option<&Value>, gt: Option<&Value>) -> Result<f64, Unsupported> {
  if let Some(v) = gte {
    return v
      .as_f64()
      .ok_or_else(|| Unsupported::with_detail("range.gte", "must be numeric"));
  }
  if let Some(v) = gt {
    let n = v
      .as_f64()
      .ok_or_else(|| Unsupported::with_detail("range.gt", "must be numeric"))?;
    // SearchLite F64Range is inclusive; nudge upward by next-representable.
    return Ok(next_up(n));
  }
  Ok(f64::NEG_INFINITY)
}

fn f64_upper(lte: Option<&Value>, lt: Option<&Value>) -> Result<f64, Unsupported> {
  if let Some(v) = lte {
    return v
      .as_f64()
      .ok_or_else(|| Unsupported::with_detail("range.lte", "must be numeric"));
  }
  if let Some(v) = lt {
    let n = v
      .as_f64()
      .ok_or_else(|| Unsupported::with_detail("range.lt", "must be numeric"))?;
    return Ok(next_down(n));
  }
  Ok(f64::INFINITY)
}

// Use the standard library's IEEE-754 next/prev helpers (stable since Rust 1.86).
// These handle ±0.0, subnormals, and infinities correctly, unlike the previous
// hand-rolled bit-twiddling.
fn next_up(x: f64) -> f64 {
  f64::next_up(x)
}

fn next_down(x: f64) -> f64 {
  f64::next_down(x)
}

// --- exists --------------------------------------------------------------

fn translate_exists(_body: &Value) -> Result<Value, Unsupported> {
  Err(Unsupported::with_detail(
    "exists",
    "no equivalent in searchlite; reject for v1",
  ))
}

// --- bool ----------------------------------------------------------------

fn translate_bool(body: &Value) -> Result<Value, Unsupported> {
  let map = body
    .as_object()
    .ok_or_else(|| Unsupported::with_detail("bool", "expected an object"))?;

  let must = collect_clause_array(map.get("must"))?;
  let should = collect_clause_array(map.get("should"))?;
  let must_not = collect_clause_array(map.get("must_not"))?;
  let filter_clauses = collect_clause_array(map.get("filter"))?;

  let must_translated = must
    .iter()
    .map(translate_query)
    .collect::<Result<Vec<_>, _>>()?;
  let should_translated = should
    .iter()
    .map(translate_query)
    .collect::<Result<Vec<_>, _>>()?;
  let must_not_translated = must_not
    .iter()
    .map(translate_query)
    .collect::<Result<Vec<_>, _>>()?;
  let filter_translated = filter_clauses
    .iter()
    .map(translate_to_filter)
    .collect::<Result<Vec<_>, _>>()?;

  // Capture the should-clause count before moving `should_translated` into
  // the output, since `resolve_bool_msm` needs it for percentage resolution.
  let should_count = should_translated.len();
  let mut out = Map::new();
  out.insert("type".into(), Value::String("bool".into()));
  out.insert("must".into(), Value::Array(must_translated));
  out.insert("should".into(), Value::Array(should_translated));
  out.insert("must_not".into(), Value::Array(must_not_translated));
  out.insert("filter".into(), Value::Array(filter_translated));

  if let Some(msm) = map.get("minimum_should_match") {
    // Core's `Bool.minimum_should_match` is `Option<usize>` only — it doesn't
    // accept ES's `"75%"` percentage form like `multi_match` does. Resolve
    // the percentage adapter-side against the should-clause count so the
    // common Kibana shape works.
    let resolved = resolve_bool_msm(msm, should_count)?;
    out.insert("minimum_should_match".into(), Value::from(resolved as u64));
  }
  if let Some(boost) = map.get("boost") {
    out.insert("boost".into(), boost.clone());
  }
  Ok(Value::Object(out))
}

/// Resolve a `bool.minimum_should_match` value against the count of `should`
/// clauses. Accepts:
///
/// - non-negative integer (`3`)
/// - integer-as-string (`"3"`)
/// - whole or fractional percentage (`"75%"`) → `floor(should_count * pct / 100)`
///
/// Rejects negatives, ES's combinator syntax (`"3<90%"`, `"-25%"`), and any
/// other shape with a clear error.
fn resolve_bool_msm(value: &Value, should_count: usize) -> Result<usize, Unsupported> {
  match value {
    Value::Number(n) => n
      .as_u64()
      .map(|n| (n as usize).min(should_count))
      .ok_or_else(|| {
        Unsupported::with_detail(
          "bool.minimum_should_match",
          "must be a non-negative integer",
        )
      }),
    Value::String(s) => parse_msm_string(s.trim(), should_count),
    _ => Err(Unsupported::with_detail(
      "bool.minimum_should_match",
      "must be an integer or a percentage string like \"75%\"",
    )),
  }
}

fn parse_msm_string(s: &str, should_count: usize) -> Result<usize, Unsupported> {
  if let Some(stripped) = s.strip_suffix('%') {
    // Reject combinator syntax (`3<90%`) and signed forms — supporting them
    // properly is non-trivial and silently mis-translating is worse than
    // returning a clear error.
    if stripped.contains('<') || stripped.starts_with('-') || stripped.starts_with('+') {
      return Err(Unsupported::with_detail(
        "bool.minimum_should_match",
        "combinator and signed percentage syntax (e.g. `3<90%`, `-25%`) is not supported",
      ));
    }
    let pct: f64 = stripped.parse().map_err(|_| {
      Unsupported::with_detail(
        "bool.minimum_should_match",
        format!("invalid percentage `{s}`"),
      )
    })?;
    if !(0.0..=100.0).contains(&pct) {
      return Err(Unsupported::with_detail(
        "bool.minimum_should_match",
        "percentage must be between 0 and 100",
      ));
    }
    let resolved = ((should_count as f64) * pct / 100.0).floor() as usize;
    return Ok(resolved.min(should_count));
  }
  // Bare integer string — accept defensively (some ES clients send "3").
  let n: i64 = s.parse().map_err(|_| {
    Unsupported::with_detail(
      "bool.minimum_should_match",
      format!("must be an integer or percentage, got `{s}`"),
    )
  })?;
  if n < 0 {
    return Err(Unsupported::with_detail(
      "bool.minimum_should_match",
      "negative integers are not supported (use a percentage instead)",
    ));
  }
  Ok((n as usize).min(should_count))
}

fn collect_clause_array(value: Option<&Value>) -> Result<Vec<Value>, Unsupported> {
  match value {
    None => Ok(Vec::new()),
    Some(Value::Array(items)) => Ok(items.clone()),
    Some(other) => Ok(vec![other.clone()]),
  }
}

/// Translate an ES query clause that appears in a filter context into a
/// SearchLite Filter (externally-tagged JSON).
pub fn translate_to_filter(es: &Value) -> Result<Value, Unsupported> {
  let map = es
    .as_object()
    .ok_or_else(|| Unsupported::with_detail("filter", "expected an object"))?;
  if map.len() != 1 {
    return Err(Unsupported::with_detail(
      "filter",
      "expected exactly one clause",
    ));
  }
  let (clause, body) = map.iter().next().unwrap();
  match clause.as_str() {
    "term" => term_to_filter(body),
    "terms" => terms_to_filter(body),
    "range" => translate_range_to_filter(body),
    "bool" => bool_to_filter(body),
    "match_all" => Ok(json!({ "And": [] })),
    other => Err(Unsupported::with_detail(
      format!("filter.{other}"),
      "not all query clauses are valid in filter context",
    )),
  }
}

fn term_to_filter(body: &Value) -> Result<Value, Unsupported> {
  let map = body
    .as_object()
    .ok_or_else(|| Unsupported::with_detail("term", "expected an object"))?;
  if map.len() != 1 {
    return Err(Unsupported::with_detail(
      "term",
      "expected exactly one field",
    ));
  }
  let (field, spec) = map.iter().next().unwrap();
  let (value, _boost) = classify_term_value(spec, "term")?;
  match value {
    TermValue::Keyword(s) => Ok(json!({ "KeywordEq": { "field": field, "value": s } })),
    TermValue::I64(n) => Ok(json!({ "I64Range": { "field": field, "min": n, "max": n } })),
    TermValue::F64(f) => Ok(json!({ "F64Range": { "field": field, "min": f, "max": f } })),
  }
}

fn terms_to_filter(body: &Value) -> Result<Value, Unsupported> {
  let map = body
    .as_object()
    .ok_or_else(|| Unsupported::with_detail("terms", "expected an object"))?;
  // Only treat `boost` as the meta-key when its value is numeric (the only
  // shape ES documents). Otherwise it's a legitimate field name — silently
  // filtering it would make any schema with a `boost` column unqueryable.
  let mut field_iter = map
    .iter()
    .filter(|(k, v)| !(*k == "boost" && v.is_number()));
  let (field, spec) = field_iter
    .next()
    .ok_or_else(|| Unsupported::with_detail("terms", "missing field entry"))?;
  if field_iter.next().is_some() {
    return Err(Unsupported::with_detail(
      "terms",
      "expected exactly one field entry",
    ));
  }
  let array = spec
    .as_array()
    .ok_or_else(|| Unsupported::with_detail("terms", "field value must be array"))?;
  let classified: Vec<TermValue> = array
    .iter()
    .map(classify_terms_element)
    .collect::<Result<Vec<_>, _>>()?;

  // All-string fast path keeps the more compact `KeywordIn` filter shape.
  // Mixed or numeric values fall through to per-value equality filters
  // wrapped in `Or`, since SearchLite has no native "numeric in" filter.
  if classified
    .iter()
    .all(|v| matches!(v, TermValue::Keyword(_)))
  {
    let values: Vec<String> = classified
      .into_iter()
      .map(|v| match v {
        TermValue::Keyword(s) => s,
        _ => unreachable!(),
      })
      .collect();
    return Ok(json!({ "KeywordIn": { "field": field, "values": values } }));
  }

  let filters: Vec<Value> = classified
    .into_iter()
    .map(|v| match v {
      TermValue::Keyword(s) => json!({ "KeywordEq": { "field": field, "value": s } }),
      TermValue::I64(n) => json!({ "I64Range": { "field": field, "min": n, "max": n } }),
      TermValue::F64(f) => json!({ "F64Range": { "field": field, "min": f, "max": f } }),
    })
    .collect();
  match filters.len() {
    0 => Err(Unsupported::with_detail(
      "terms",
      "values array must be non-empty",
    )),
    1 => Ok(filters.into_iter().next().unwrap()),
    _ => Ok(json!({ "Or": filters })),
  }
}

fn bool_to_filter(body: &Value) -> Result<Value, Unsupported> {
  let map = body
    .as_object()
    .ok_or_else(|| Unsupported::with_detail("bool", "expected an object"))?;
  let must = collect_clause_array(map.get("must"))?;
  let must_not = collect_clause_array(map.get("must_not"))?;
  let filter = collect_clause_array(map.get("filter"))?;
  let should = collect_clause_array(map.get("should"))?;
  if !should.is_empty() {
    return Err(Unsupported::with_detail(
      "filter.bool.should",
      "should clauses cannot appear inside filter context",
    ));
  }

  let mut and_parts: Vec<Value> = Vec::new();
  for clause in must.iter().chain(filter.iter()) {
    and_parts.push(translate_to_filter(clause)?);
  }
  for clause in must_not {
    let inner = translate_to_filter(&clause)?;
    and_parts.push(json!({ "Not": inner }));
  }
  if and_parts.len() == 1 {
    Ok(and_parts.into_iter().next().unwrap())
  } else {
    Ok(json!({ "And": and_parts }))
  }
}

// --- constant_score / dis_max -------------------------------------------

fn translate_constant_score(body: &Value) -> Result<Value, Unsupported> {
  let map = body
    .as_object()
    .ok_or_else(|| Unsupported::with_detail("constant_score", "expected an object"))?;
  let filter_node = map
    .get("filter")
    .ok_or_else(|| Unsupported::with_detail("constant_score.filter", "missing"))?;
  let filter = translate_to_filter(filter_node)?;
  let mut out = Map::new();
  out.insert("type".into(), Value::String("constant_score".into()));
  out.insert("filter".into(), filter);
  if let Some(boost) = map.get("boost") {
    out.insert("boost".into(), boost.clone());
  }
  Ok(Value::Object(out))
}

fn translate_dis_max(body: &Value) -> Result<Value, Unsupported> {
  let map = body
    .as_object()
    .ok_or_else(|| Unsupported::with_detail("dis_max", "expected an object"))?;
  let queries = map
    .get("queries")
    .and_then(Value::as_array)
    .ok_or_else(|| Unsupported::with_detail("dis_max.queries", "missing array"))?
    .iter()
    .map(translate_query)
    .collect::<Result<Vec<_>, _>>()?;
  let mut out = Map::new();
  out.insert("type".into(), Value::String("dis_max".into()));
  out.insert("queries".into(), Value::Array(queries));
  if let Some(tb) = map.get("tie_breaker") {
    out.insert("tie_breaker".into(), tb.clone());
  }
  if let Some(boost) = map.get("boost") {
    out.insert("boost".into(), boost.clone());
  }
  Ok(Value::Object(out))
}

// --- query_string --------------------------------------------------------

fn translate_query_string(body: &Value) -> Result<Value, Unsupported> {
  let map = body
    .as_object()
    .ok_or_else(|| Unsupported::with_detail("query_string", "expected an object"))?;
  let query = map
    .get("query")
    .and_then(Value::as_str)
    .ok_or_else(|| Unsupported::with_detail("query_string", "missing string `query`"))?
    .to_string();
  let mut out = Map::new();
  out.insert("type".into(), Value::String("query_string".into()));
  out.insert("query".into(), Value::String(query));

  if let Some(fields) = map.get("fields").and_then(Value::as_array) {
    let translated = fields
      .iter()
      .map(translate_field_spec)
      .collect::<Result<Vec<_>, _>>()?;
    out.insert("fields".into(), Value::Array(translated));
  } else if let Some(default_field) = map.get("default_field").and_then(Value::as_str) {
    out.insert(
      "fields".into(),
      Value::Array(vec![translate_field_spec(&Value::String(
        default_field.to_string(),
      ))?]),
    );
  }
  if let Some(boost) = map.get("boost") {
    out.insert("boost".into(), boost.clone());
  }
  Ok(Value::Object(out))
}
