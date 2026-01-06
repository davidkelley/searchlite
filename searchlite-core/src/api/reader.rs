use hashbrown::{HashMap, HashSet};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use smallvec::{smallvec, SmallVec};

use crate::analysis::analyzer::Analyzer;
use crate::api::types::{
  Aggregation, AggregationResponse, DateHistogramAggregation, DecayFunction, Filter,
  FunctionBoostMode, FunctionScoreMode, FuzzyOptions, HistogramAggregation, IndexOptions, Query,
  RescoreMode, RescoreRequest, SearchRequest, SortOrder, SuggestOption, SuggestRequest,
  SuggestResult,
};
use crate::api::AggregationError;
use crate::index::fastfields::FastFieldsReader;
use crate::index::highlight::make_snippet;
use crate::index::manifest::{FieldKind, Manifest, Schema, SchemaAnalyzers};
use crate::index::postings::{PostingEntry, PostingsReader};
use crate::index::segment::SegmentReader;
use crate::index::InnerIndex;
use crate::query::aggregation::AggregationPipeline;
use crate::query::aggs::{parse_calendar_interval, parse_date, parse_interval_seconds};
use crate::query::collector::{AggregationSegmentCollector, DocCollector};
use crate::query::filters::{passes_filter, passes_filters};
use crate::query::phrase::matches_phrase;
use crate::query::planner::{
  build_query_plan, PhraseSpec, QueryMatcher, ScoreExpr, ScoreNode, ScorePlan, TermExpansion,
  TermGroupSpec,
};
use crate::query::score_functions::{
  apply_boost_mode, combine_function_scores, compile_functions, CompiledFunction,
};
use crate::query::sort::{SortKey, SortKeyPart, SortPlan, SortValue};
use crate::query::wand::{
  execute_top_k_with_stats_and_mode_internal, score_tf, QueryStats, ScoreAdjustFn, ScoreMode,
  ScoredTerm,
};
use crate::util::regex::anchored_regex;
use crate::DocId;

const MAX_CURSOR_ADVANCE: usize = 50_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hit {
  pub doc_id: String,
  pub score: f32,
  pub fields: Option<serde_json::Value>,
  pub snippet: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub explanation: Option<HitExplanation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionExplanation {
  pub r#type: String,
  pub value: f32,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub field: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RescoreExplanation {
  pub rescore_score: f32,
  pub combined_score: f32,
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  pub functions: Vec<FunctionExplanation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HitExplanation {
  pub base_score: f32,
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  pub functions: Vec<FunctionExplanation>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub rescore: Option<RescoreExplanation>,
  pub final_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
  pub total_hits_estimate: u64,
  pub hits: Vec<Hit>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub next_cursor: Option<String>,
  #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
  pub aggregations: BTreeMap<String, AggregationResponse>,
  #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
  pub suggest: BTreeMap<String, SuggestResult>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub profile: Option<ProfileResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct ExecutionProfile {
  pub scored_docs: usize,
  pub candidates_examined: usize,
  pub postings_advanced: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileResult {
  pub execution: ExecutionProfile,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub rescore: Option<ExecutionProfile>,
  #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
  pub timings: BTreeMap<String, f64>,
}

fn score_sort_key(score: f32, segment_ord: u32, doc_id: DocId, order: SortOrder) -> SortKey {
  SortKey {
    parts: smallvec![SortKeyPart {
      order,
      value: SortValue::Score(score),
    }],
    segment_ord,
    doc_id,
  }
}

#[derive(Clone, Debug)]
enum CompiledScoreNode {
  Empty,
  Expr(ScoreExpr),
  Sum(Vec<CompiledScoreNode>),
  DisMax {
    children: Vec<CompiledScoreNode>,
    tie_breaker: f32,
  },
  Constant {
    score: f32,
    matcher: QueryMatcher,
  },
  FunctionScore {
    matcher: QueryMatcher,
    base: Box<CompiledScoreNode>,
    functions: Vec<CompiledFunction>,
    score_mode: FunctionScoreMode,
    boost_mode: FunctionBoostMode,
    max_boost: Option<f32>,
    min_score: Option<f32>,
    boost: f32,
  },
}

fn compile_score_node(node: &ScoreNode, schema: &Schema) -> Result<CompiledScoreNode> {
  Ok(match node {
    ScoreNode::Empty => CompiledScoreNode::Empty,
    ScoreNode::Expr(expr) => CompiledScoreNode::Expr(expr.clone()),
    ScoreNode::Sum(children) => {
      let mut out = Vec::with_capacity(children.len());
      for child in children.iter() {
        out.push(compile_score_node(child, schema)?);
      }
      CompiledScoreNode::Sum(out)
    }
    ScoreNode::DisMax {
      children,
      tie_breaker,
    } => {
      let mut out = Vec::with_capacity(children.len());
      for child in children.iter() {
        out.push(compile_score_node(child, schema)?);
      }
      CompiledScoreNode::DisMax {
        children: out,
        tie_breaker: *tie_breaker,
      }
    }
    ScoreNode::Constant { score, matcher } => CompiledScoreNode::Constant {
      score: *score,
      matcher: matcher.clone(),
    },
    ScoreNode::FunctionScore {
      matcher,
      base,
      functions,
      score_mode,
      boost_mode,
      max_boost,
      min_score,
      boost,
    } => CompiledScoreNode::FunctionScore {
      matcher: matcher.clone(),
      base: Box::new(compile_score_node(base, schema)?),
      functions: compile_functions(functions, schema)?,
      score_mode: *score_mode,
      boost_mode: *boost_mode,
      max_boost: *max_boost,
      min_score: *min_score,
      boost: *boost,
    },
  })
}

fn has_custom_scoring(node: &CompiledScoreNode) -> bool {
  match node {
    CompiledScoreNode::Empty | CompiledScoreNode::Expr(_) => false,
    CompiledScoreNode::Sum(children) | CompiledScoreNode::DisMax { children, .. } => {
      children.iter().any(has_custom_scoring)
    }
    CompiledScoreNode::Constant { .. } | CompiledScoreNode::FunctionScore { .. } => true,
  }
}

fn describe_function(func: &CompiledFunction, value: f32) -> FunctionExplanation {
  match func {
    CompiledFunction::Weight { .. } => FunctionExplanation {
      r#type: "weight".to_string(),
      value,
      field: None,
    },
    CompiledFunction::FieldValueFactor { field, .. } => FunctionExplanation {
      r#type: "field_value_factor".to_string(),
      value,
      field: Some(field.clone()),
    },
    CompiledFunction::Decay {
      field, function, ..
    } => {
      let name = match function {
        DecayFunction::Exp => "decay_exp",
        DecayFunction::Gauss => "decay_gauss",
        DecayFunction::Linear => "decay_linear",
      };
      FunctionExplanation {
        r#type: name.to_string(),
        value,
        field: Some(field.clone()),
      }
    }
  }
}

fn evaluate_compiled_score(
  node: &CompiledScoreNode,
  evaluator: &QueryEvaluator<'_>,
  fast_fields: &FastFieldsReader,
  doc_id: DocId,
  leaf_scores: &[f32],
  collect_functions: bool,
  out_functions: &mut Vec<FunctionExplanation>,
) -> Option<f32> {
  match node {
    CompiledScoreNode::Empty => Some(1.0),
    CompiledScoreNode::Expr(expr) => Some(expr.evaluate(leaf_scores)),
    CompiledScoreNode::Sum(children) => {
      let mut sum = 0.0_f32;
      for child in children.iter() {
        let score = evaluate_compiled_score(
          child,
          evaluator,
          fast_fields,
          doc_id,
          leaf_scores,
          collect_functions,
          out_functions,
        )?;
        sum += score;
      }
      Some(sum)
    }
    CompiledScoreNode::DisMax {
      children,
      tie_breaker,
    } => {
      if children.is_empty() {
        return Some(0.0);
      }
      let mut sum = 0.0_f32;
      let mut max = f32::NEG_INFINITY;
      for child in children.iter() {
        let score = evaluate_compiled_score(
          child,
          evaluator,
          fast_fields,
          doc_id,
          leaf_scores,
          collect_functions,
          out_functions,
        )?;
        max = max.max(score);
        sum += score;
      }
      Some(max + *tie_breaker * (sum - max))
    }
    CompiledScoreNode::Constant { score, matcher } => {
      if evaluator.matches_subquery(matcher, doc_id) {
        Some(*score)
      } else {
        Some(0.0)
      }
    }
    CompiledScoreNode::FunctionScore {
      matcher,
      base,
      functions,
      score_mode,
      boost_mode,
      max_boost,
      min_score,
      boost,
    } => {
      if !evaluator.matches_subquery(matcher, doc_id) {
        return Some(0.0);
      }
      let base_score = evaluate_compiled_score(
        base,
        evaluator,
        fast_fields,
        doc_id,
        leaf_scores,
        collect_functions,
        out_functions,
      )?;
      let mut function_values = Vec::new();
      let mut fn_expls = Vec::new();
      for func in functions.iter() {
        if let Some(val) = func.evaluate(fast_fields, doc_id) {
          function_values.push(val);
          if collect_functions {
            fn_expls.push(describe_function(func, val));
          }
        }
      }
      let mut effective_base = base_score;
      if effective_base.abs() <= f32::EPSILON && !function_values.is_empty() {
        // Preserve function contributions even when the base query scored 0.0,
        // so multiplicative boost modes do not erase function-only scoring.
        effective_base = 1.0;
      }
      let mut combined =
        if let Some(func_score) = combine_function_scores(&function_values, *score_mode) {
          apply_boost_mode(effective_base, func_score, *boost_mode)
        } else {
          effective_base
        };
      if let Some(max) = max_boost {
        combined = combined.min(*max);
      }
      if let Some(min) = min_score {
        if combined < *min {
          return None;
        }
      }
      combined *= *boost;
      if collect_functions {
        out_functions.extend(fn_expls);
      }
      Some(combined)
    }
  }
}
const CURSOR_VERSION: u8 = 1;
const CURSOR_BYTES: usize = 21;
const CURSOR_HEX_LEN: usize = CURSOR_BYTES * 2;
const SORT_CURSOR_VERSION: u8 = 2;
const DEFAULT_SUGGEST_SCAN: usize = 64;
const MAX_SUGGEST_CANDIDATES: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq)]
struct PaginationCursor {
  version: u8,
  generation: u32,
  key: SortKey,
  returned: u32,
}

impl PaginationCursor {
  fn encode(&self) -> String {
    let score_bits = self
      .key
      .score_bits()
      .expect("score cursor missing score value");
    let mut buf = [0u8; CURSOR_BYTES];
    buf[0] = self.version;
    buf[1..5].copy_from_slice(&self.generation.to_be_bytes());
    buf[5..9].copy_from_slice(&score_bits.to_be_bytes());
    buf[9..13].copy_from_slice(&self.key.segment_ord.to_be_bytes());
    buf[13..17].copy_from_slice(&self.key.doc_id.to_be_bytes());
    buf[17..].copy_from_slice(&self.returned.to_be_bytes());
    let mut encoded = String::with_capacity(CURSOR_HEX_LEN);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in buf {
      encoded.push(HEX[(byte >> 4) as usize] as char);
      encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
  }

  fn decode(raw: &str) -> Result<Self> {
    if raw.len() != CURSOR_HEX_LEN {
      bail!(
        "invalid cursor length: expected {CURSOR_HEX_LEN} hex chars, got {}",
        raw.len()
      );
    }
    let mut bytes = [0u8; CURSOR_BYTES];
    for (i, chunk) in raw.as_bytes().chunks_exact(2).enumerate() {
      let hex = std::str::from_utf8(chunk).unwrap();
      let value = u8::from_str_radix(hex, 16)
        .with_context(|| format!("decoding cursor at byte index {i}"))?;
      bytes[i] = value;
    }
    let version = bytes[0];
    if version != CURSOR_VERSION {
      bail!("unsupported cursor version {version}");
    }
    let generation = u32::from_be_bytes(bytes[1..5].try_into().unwrap());
    let score_bits = u32::from_be_bytes(bytes[5..9].try_into().unwrap());
    let segment_ord = u32::from_be_bytes(bytes[9..13].try_into().unwrap());
    let doc_id = u32::from_be_bytes(bytes[13..17].try_into().unwrap());
    let returned = u32::from_be_bytes(bytes[17..21].try_into().unwrap());
    if returned as usize > MAX_CURSOR_ADVANCE {
      bail!(
        "cursor requests {} hits, which exceeds max supported {MAX_CURSOR_ADVANCE}",
        returned
      );
    }
    Ok(Self {
      version,
      generation,
      key: score_sort_key(
        f32::from_bits(score_bits),
        segment_ord,
        doc_id,
        SortOrder::Desc,
      ),
      returned,
    })
  }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SortCursorState {
  version: u8,
  generation: u32,
  returned: u32,
  plan_hash: u32,
  segment_ord: u32,
  doc_id: DocId,
  values: Vec<CursorValue>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "t", content = "v", rename_all = "lowercase")]
enum CursorValue {
  Score(u32),
  I64(i64),
  F64(f64),
  Str(String),
  Missing,
}

impl From<SortValue> for CursorValue {
  fn from(value: SortValue) -> Self {
    match value {
      SortValue::Score(score) => CursorValue::Score(score.to_bits()),
      SortValue::I64(v) => CursorValue::I64(v),
      SortValue::F64(v) => CursorValue::F64(v),
      SortValue::Str(v) => CursorValue::Str(v),
      SortValue::Missing => CursorValue::Missing,
    }
  }
}

impl From<CursorValue> for SortValue {
  fn from(value: CursorValue) -> Self {
    match value {
      CursorValue::Score(bits) => SortValue::Score(f32::from_bits(bits)),
      CursorValue::I64(v) => SortValue::I64(v),
      CursorValue::F64(v) => SortValue::F64(v),
      CursorValue::Str(v) => SortValue::Str(v),
      CursorValue::Missing => SortValue::Missing,
    }
  }
}

fn hex_encode(bytes: &[u8]) -> String {
  const HEX: &[u8; 16] = b"0123456789abcdef";
  let mut out = String::with_capacity(bytes.len() * 2);
  for byte in bytes {
    out.push(HEX[(byte >> 4) as usize] as char);
    out.push(HEX[(byte & 0x0f) as usize] as char);
  }
  out
}

fn hex_decode(raw: &str) -> Result<Vec<u8>> {
  if raw.len() & 1 != 0 {
    bail!("invalid cursor: expected even-length hex string");
  }
  let mut bytes = Vec::with_capacity(raw.len() / 2);
  for (i, chunk) in raw.as_bytes().chunks_exact(2).enumerate() {
    let hex = std::str::from_utf8(chunk).unwrap();
    let value =
      u8::from_str_radix(hex, 16).with_context(|| format!("decoding cursor at byte index {i}"))?;
    bytes.push(value);
  }
  Ok(bytes)
}

#[derive(Clone, Debug)]
struct RankedHit {
  key: SortKey,
  score: f32,
  explanation: Option<HitExplanation>,
}

impl PartialEq for RankedHit {
  fn eq(&self, other: &Self) -> bool {
    self.key == other.key && self.score.to_bits() == other.score.to_bits()
  }
}

impl Eq for RankedHit {}

impl PartialOrd for RankedHit {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    Some(self.cmp(other))
  }
}

impl Ord for RankedHit {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    self.key.cmp(&other.key)
  }
}

#[derive(Default)]
struct NoopCollector;

impl DocCollector for NoopCollector {
  fn collect(&mut self, _doc_id: DocId, _score: f32) {}
}

fn push_ranked(heap: &mut BinaryHeap<RankedHit>, hit: RankedHit, limit: usize) {
  if limit == 0 {
    return;
  }
  if heap.len() < limit {
    heap.push(hit);
    return;
  }
  if let Some(worst) = heap.peek() {
    if hit < *worst {
      heap.pop();
      heap.push(hit);
    }
  }
}

struct CursorState {
  key: SortKey,
  returned: u32,
}

fn decode_cursor(
  raw: &str,
  manifest_generation: u32,
  sort_plan: &SortPlan,
  score_fast_path: bool,
) -> Result<CursorState> {
  if score_fast_path {
    let cur = PaginationCursor::decode(raw)?;
    if cur.generation != manifest_generation {
      bail!(
        "stale cursor for this index generation: expected {}, got {}",
        manifest_generation,
        cur.generation
      );
    }
    return Ok(CursorState {
      key: cur.key,
      returned: cur.returned,
    });
  }
  let bytes = hex_decode(raw)?;
  let state: SortCursorState =
    serde_json::from_slice(&bytes).context("parsing sort cursor payload")?;
  if state.version != SORT_CURSOR_VERSION {
    bail!("unsupported sort cursor version {}", state.version);
  }
  if state.generation != manifest_generation {
    bail!(
      "stale cursor for this index generation: expected {}, got {}",
      manifest_generation,
      state.generation
    );
  }
  if state.plan_hash != sort_plan.hash() {
    bail!("cursor sort order does not match this request");
  }
  if state.returned as usize > MAX_CURSOR_ADVANCE {
    bail!(
      "cursor requests {} hits, which exceeds max supported {MAX_CURSOR_ADVANCE}",
      state.returned
    );
  }
  let values: Vec<SortValue> = state.values.into_iter().map(SortValue::from).collect();
  let key = sort_plan.key_from_values(&values, state.segment_ord, state.doc_id)?;
  Ok(CursorState {
    key,
    returned: state.returned,
  })
}

fn encode_cursor(
  manifest_generation: u32,
  returned: u32,
  key: &SortKey,
  sort_plan: &SortPlan,
  score_fast_path: bool,
) -> Result<String> {
  if score_fast_path {
    return Ok(
      PaginationCursor {
        version: CURSOR_VERSION,
        generation: manifest_generation,
        key: key.clone(),
        returned,
      }
      .encode(),
    );
  }
  let values = sort_plan.values_from_key(key)?;
  let state = SortCursorState {
    version: SORT_CURSOR_VERSION,
    generation: manifest_generation,
    returned,
    plan_hash: sort_plan.hash(),
    segment_ord: key.segment_ord,
    doc_id: key.doc_id,
    values: values.into_iter().map(CursorValue::from).collect(),
  };
  let data = serde_json::to_vec(&state)?;
  Ok(hex_encode(&data))
}

#[derive(Clone, Debug)]
struct QualifiedTerm {
  field: String,
  term: String,
  key: String,
  weight: f32,
  leaf: usize,
}

#[derive(Clone, Debug)]
struct TermMatchGroup {
  keys: Vec<String>,
}

#[derive(Clone, Debug)]
struct PhraseFieldConfig {
  slop: u32,
  fields: Vec<(String, Vec<Vec<String>>)>,
}

#[derive(Clone, Debug)]
struct PhraseRuntime {
  slop: u32,
  variants: Vec<Vec<Vec<PostingEntry>>>,
}

#[derive(Clone, Copy)]
enum RootFilter<'a> {
  None,
  Node(&'a Filter),
  AndSlice(&'a [Filter]),
}

struct SegmentSearchParams<'a> {
  qualified_terms: &'a [QualifiedTerm],
  matcher: &'a QueryMatcher,
  term_groups: &'a [TermMatchGroup],
  phrase_fields: &'a [PhraseFieldConfig],
  scorer: Option<&'a ScorePlan>,
  score_tree: &'a CompiledScoreNode,
  needs_score_hook: bool,
  explain: bool,
  profile: bool,
  root_filter: RootFilter<'a>,
  agg_collector: Option<&'a mut dyn DocCollector>,
  match_counter: Option<&'a mut u64>,
  req: &'a SearchRequest,
  segment_ord: u32,
  rank_limit: usize,
  cursor_key: Option<SortKey>,
  saw_cursor: &'a mut bool,
  sort_plan: &'a SortPlan,
  collect_hits: Option<&'a mut dyn FnMut(SortKey, f32)>,
  stats: Option<&'a mut QueryStats>,
}

fn build_term_key(field: &str, term: &str) -> String {
  let mut key = String::with_capacity(field.len() + term.len() + 1);
  key.push_str(field);
  key.push(':');
  key.push_str(term);
  key
}

/// Returns the prefix by Unicode scalar value count (not bytes).
fn char_prefix(input: &str, len: usize) -> &str {
  if len == 0 {
    return "";
  }
  match input.char_indices().nth(len) {
    Some((idx, _)) => &input[..idx],
    None => input,
  }
}

fn distance_weight(distance: usize) -> f32 {
  1.0 / (distance as f32 + 1.0)
}

fn bounded_levenshtein(a: &str, b: &str, max_edits: usize) -> Option<usize> {
  let a_len = a.chars().count();
  let b_chars: SmallVec<[char; 32]> = b.chars().collect();
  let b_len = b_chars.len();
  if a_len.abs_diff(b_len) > max_edits {
    return None;
  }
  if a_len == 0 {
    return (b_len <= max_edits).then_some(b_len);
  }
  if b_len == 0 {
    return (a_len <= max_edits).then_some(a_len);
  }
  let mut prev: SmallVec<[usize; 64]> = (0..=b_len).collect();
  let mut curr: SmallVec<[usize; 64]> = smallvec![0; b_len + 1];
  for (i, ca) in a.chars().enumerate() {
    curr[0] = i + 1;
    let mut row_min = curr[0];
    for (j, cb) in b_chars.iter().enumerate() {
      let cost = if ca == *cb { 0 } else { 1 };
      let del = prev[j + 1] + 1;
      let ins = curr[j] + 1;
      let sub = prev[j] + cost;
      let val = del.min(ins).min(sub);
      curr[j + 1] = val;
      row_min = row_min.min(val);
    }
    if row_min > max_edits {
      return None;
    }
    std::mem::swap(&mut prev, &mut curr);
  }
  if prev[b_len] <= max_edits {
    Some(prev[b_len])
  } else {
    None
  }
}

fn expand_term_groups(
  segments: &[SegmentReader],
  groups: &[TermGroupSpec],
  fuzzy: Option<&FuzzyOptions>,
  analysis: &SchemaAnalyzers,
  schema: &Schema,
) -> Result<(Vec<QualifiedTerm>, Vec<TermMatchGroup>)> {
  let mut qualified_terms = Vec::new();
  let mut term_groups = Vec::with_capacity(groups.len());
  for group in groups.iter() {
    let mut keys = Vec::new();
    let mut seen_keys = HashSet::new();
    for field in group.fields.iter() {
      let target_leaf = field.leaf.or(group.leaf);
      let weight = group.boost * field.boost;
      match schema.field_kind(&field.field) {
        FieldKind::Text => {
          if let Some(analyzer) = analysis.search_analyzer(&field.field) {
            let mut seen_tokens = HashSet::new();
            let tokens: Vec<String> = match group.expansion {
              TermExpansion::Exact => analyzer
                .analyze(&group.term)
                .into_iter()
                .map(|t| t.text)
                .collect(),
              _ => analyze_pattern_tokens(analyzer, &group.term),
            };
            for token in tokens.into_iter() {
              if !seen_tokens.insert(token.clone()) {
                continue;
              }
              let (scored, mut expanded_keys) = expand_term_for_group(
                segments,
                &field.field,
                &token,
                weight,
                group.score,
                target_leaf,
                fuzzy,
                &group.expansion,
              )?;
              if group.score {
                qualified_terms.extend(scored);
              }
              for key in expanded_keys.drain(..) {
                if seen_keys.insert(key.clone()) {
                  keys.push(key);
                }
              }
            }
          }
        }
        FieldKind::Keyword => {
          let term = group.term.to_ascii_lowercase();
          let (scored, mut expanded_keys) = expand_term_for_group(
            segments,
            &field.field,
            &term,
            weight,
            group.score,
            target_leaf,
            fuzzy,
            &group.expansion,
          )?;
          if group.score {
            qualified_terms.extend(scored);
          }
          for key in expanded_keys.drain(..) {
            if seen_keys.insert(key.clone()) {
              keys.push(key);
            }
          }
        }
        FieldKind::Numeric | FieldKind::Unknown => {}
      }
    }
    term_groups.push(TermMatchGroup { keys });
  }
  Ok((qualified_terms, term_groups))
}

fn analyze_pattern_tokens(analyzer: &Analyzer, value: &str) -> Vec<String> {
  let tokens: Vec<String> = analyzer
    .analyze(value)
    .into_iter()
    .map(|t| t.text)
    .collect();
  if tokens.is_empty() {
    return vec![analyzer.normalize_pattern(value)];
  }
  if tokens.len() == 1 {
    return tokens;
  }
  // Wildcard/regex patterns often get split by analyzers; fall back to the raw pattern so we
  // preserve the literal structure, but still apply lightweight normalization (e.g. lowercase).
  vec![analyzer.normalize_pattern(value)]
}

#[allow(clippy::too_many_arguments)]
fn expand_term_for_group(
  segments: &[SegmentReader],
  field: &str,
  term: &str,
  boost: f32,
  score: bool,
  leaf: Option<usize>,
  fuzzy: Option<&FuzzyOptions>,
  expansion: &TermExpansion,
) -> Result<(Vec<QualifiedTerm>, Vec<String>)> {
  match expansion {
    TermExpansion::Exact => {
      if !score {
        return Ok((Vec::new(), vec![build_term_key(field, term)]));
      }
      let Some(leaf) = leaf else {
        return Ok((Vec::new(), vec![build_term_key(field, term)]));
      };
      let Some(fuzzy) = fuzzy else {
        return Ok(expand_term_exact(field, term, boost, leaf));
      };
      let max_edits = fuzzy.max_edits.min(2) as usize;
      if max_edits == 0 {
        return Ok(expand_term_exact(field, term, boost, leaf));
      }
      Ok(expand_term_fuzzy(segments, field, term, boost, leaf, fuzzy))
    }
    TermExpansion::Prefix { max_expansions } => Ok(expand_prefix(
      segments,
      field,
      term,
      boost,
      score,
      leaf,
      *max_expansions,
    )),
    TermExpansion::Wildcard { max_expansions } => {
      expand_wildcard(segments, field, term, boost, score, leaf, *max_expansions)
    }
    TermExpansion::Regex { max_expansions } => {
      expand_regex(segments, field, term, boost, score, leaf, *max_expansions)
    }
  }
}

fn expand_prefix(
  segments: &[SegmentReader],
  field: &str,
  prefix: &str,
  boost: f32,
  score: bool,
  leaf: Option<usize>,
  max_expansions: usize,
) -> (Vec<QualifiedTerm>, Vec<String>) {
  if max_expansions == 0 {
    return (Vec::new(), Vec::new());
  }
  let prefix_key = build_term_key(field, prefix);
  let field_prefix_len = field.len() + 1;
  let mut qualified = Vec::new();
  let mut keys = Vec::new();
  let mut seen = HashSet::new();
  for seg in segments.iter() {
    let mut expanded = 0usize;
    for key in seg.terms_with_prefix(&prefix_key) {
      if expanded >= max_expansions {
        break;
      }
      if key.len() <= field_prefix_len {
        continue;
      }
      if !seen.insert(key.clone()) {
        continue;
      }
      let term = key[field_prefix_len..].to_string();
      if score {
        if let Some(idx) = leaf {
          qualified.push(QualifiedTerm {
            field: field.to_string(),
            term: term.clone(),
            key: key.clone(),
            weight: boost,
            leaf: idx,
          });
        }
      }
      keys.push(key.clone());
      expanded += 1;
    }
  }
  (qualified, keys)
}

fn wildcard_literal_prefix(pattern: &str) -> &str {
  pattern.split(['*', '?']).next().unwrap_or("")
}

fn build_wildcard_regex(pattern: &str) -> Result<Regex> {
  let mut buf = String::from("^");
  for (i, ch) in pattern.char_indices() {
    match ch {
      '*' => buf.push_str(".*"),
      '?' => buf.push('.'),
      _ => {
        let end = i + ch.len_utf8();
        buf.push_str(&regex::escape(&pattern[i..end]));
      }
    }
  }
  buf.push('$');
  Regex::new(&buf).map_err(|e| anyhow::anyhow!("invalid wildcard `{pattern}`: {e}"))
}

fn expand_wildcard(
  segments: &[SegmentReader],
  field: &str,
  pattern: &str,
  boost: f32,
  score: bool,
  leaf: Option<usize>,
  max_expansions: usize,
) -> Result<(Vec<QualifiedTerm>, Vec<String>)> {
  if max_expansions == 0 {
    return Ok((Vec::new(), Vec::new()));
  }
  let regex = build_wildcard_regex(pattern)?;
  let literal_prefix = wildcard_literal_prefix(pattern);
  let prefix_key = build_term_key(field, literal_prefix);
  let field_prefix_len = field.len() + 1;
  let mut qualified = Vec::new();
  let mut keys = Vec::new();
  let mut seen = HashSet::new();
  for seg in segments.iter() {
    let mut expanded = 0usize;
    for key in seg.terms_with_prefix(&prefix_key) {
      if expanded >= max_expansions {
        break;
      }
      if key.len() <= field_prefix_len {
        continue;
      }
      let term = &key[field_prefix_len..];
      if !regex.is_match(term) {
        continue;
      }
      if !seen.insert(key.clone()) {
        continue;
      }
      if score {
        if let Some(idx) = leaf {
          qualified.push(QualifiedTerm {
            field: field.to_string(),
            term: term.to_string(),
            key: key.clone(),
            weight: boost,
            leaf: idx,
          });
        }
      }
      keys.push(key.clone());
      expanded += 1;
    }
  }
  Ok((qualified, keys))
}

fn regex_literal_prefix(pattern: &str) -> String {
  let mut prefix = String::new();
  let mut escaped = false;
  for (i, ch) in pattern.char_indices() {
    if escaped {
      match ch {
        '\\' => {
          // Escaped backslash is a literal backslash in the prefix.
          let end = i + ch.len_utf8();
          prefix.push_str(&pattern[i..end]);
          escaped = false;
          continue;
        }
        // Escape classes/boundaries mean we cannot keep extending the literal prefix.
        'd' | 'D' | 'w' | 'W' | 's' | 'S' | 'b' | 'B' => break,
        'p' | 'P' => break,
        _ => {
          let end = i + ch.len_utf8();
          prefix.push_str(&pattern[i..end]);
          escaped = false;
          continue;
        }
      }
    }
    match ch {
      '\\' => escaped = true,
      '^' if prefix.is_empty() => continue,
      '.' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '$' => break,
      _ => {
        let end = i + ch.len_utf8();
        prefix.push_str(&pattern[i..end]);
      }
    }
  }
  prefix
}

fn expand_regex(
  segments: &[SegmentReader],
  field: &str,
  pattern: &str,
  boost: f32,
  score: bool,
  leaf: Option<usize>,
  max_expansions: usize,
) -> Result<(Vec<QualifiedTerm>, Vec<String>)> {
  if max_expansions == 0 {
    return Ok((Vec::new(), Vec::new()));
  }
  let regex = anchored_regex(pattern)?;
  let literal_prefix = regex_literal_prefix(pattern);
  let prefix_key = build_term_key(field, &literal_prefix);
  let field_prefix_len = field.len() + 1;
  let mut qualified = Vec::new();
  let mut keys = Vec::new();
  let mut seen = HashSet::new();
  for seg in segments.iter() {
    let mut expanded = 0usize;
    for key in seg.terms_with_prefix(&prefix_key) {
      if expanded >= max_expansions {
        break;
      }
      if key.len() <= field_prefix_len {
        continue;
      }
      let term = &key[field_prefix_len..];
      if !regex.is_match(term) {
        continue;
      }
      if !seen.insert(key.clone()) {
        continue;
      }
      if score {
        if let Some(idx) = leaf {
          qualified.push(QualifiedTerm {
            field: field.to_string(),
            term: term.to_string(),
            key: key.clone(),
            weight: boost,
            leaf: idx,
          });
        }
      }
      keys.push(key.clone());
      expanded += 1;
    }
  }
  Ok((qualified, keys))
}

fn expand_term_exact(
  field: &str,
  term: &str,
  boost: f32,
  leaf: usize,
) -> (Vec<QualifiedTerm>, Vec<String>) {
  let key = build_term_key(field, term);
  (
    vec![QualifiedTerm {
      field: field.to_string(),
      term: term.to_string(),
      key: key.clone(),
      weight: boost,
      leaf,
    }],
    vec![key],
  )
}

fn expand_term_fuzzy(
  segments: &[SegmentReader],
  field: &str,
  term: &str,
  boost: f32,
  leaf: usize,
  fuzzy: &FuzzyOptions,
) -> (Vec<QualifiedTerm>, Vec<String>) {
  let term_len = term.chars().count();
  let exact_key = build_term_key(field, term);
  let mut qualified = vec![QualifiedTerm {
    field: field.to_string(),
    term: term.to_string(),
    key: exact_key.clone(),
    weight: boost * distance_weight(0),
    leaf,
  }];
  let mut keys = vec![exact_key.clone()];
  if term_len < fuzzy.min_length || fuzzy.max_expansions == 0 {
    return (qualified, keys);
  }
  let max_edits = fuzzy.max_edits.min(2) as usize;
  let prefix_len = fuzzy.prefix_length.min(term_len);
  let prefix = char_prefix(term, prefix_len);
  let mut prefix_key = String::with_capacity(field.len() + prefix.len() + 1);
  prefix_key.push_str(field);
  prefix_key.push(':');
  prefix_key.push_str(prefix);
  let field_prefix_len = field.len() + 1;
  let mut seen: HashSet<String> = HashSet::new();
  seen.insert(exact_key);
  let mut expansions = 0usize;
  'segments: for seg in segments.iter() {
    for key in seg.terms_with_prefix(&prefix_key) {
      if expansions >= fuzzy.max_expansions {
        break 'segments;
      }
      if key.len() <= field_prefix_len {
        continue;
      }
      let candidate = &key[field_prefix_len..];
      if candidate == term {
        continue;
      }
      let candidate_len = candidate.chars().count();
      if candidate_len.abs_diff(term_len) > max_edits {
        continue;
      }
      let Some(distance) = bounded_levenshtein(term, candidate, max_edits) else {
        continue;
      };
      if distance == 0 {
        continue;
      }
      if seen.insert(key.clone()) {
        qualified.push(QualifiedTerm {
          field: field.to_string(),
          term: candidate.to_string(),
          key: key.clone(),
          weight: boost * distance_weight(distance),
          leaf,
        });
        keys.push(key.clone());
        expansions += 1;
        if expansions >= fuzzy.max_expansions {
          break 'segments;
        }
      }
    }
  }
  (qualified, keys)
}

struct TermDocLists {
  lists: Vec<Vec<DocId>>,
  group_lists: Vec<Vec<usize>>,
}

struct QueryEvaluator<'a> {
  matcher: &'a QueryMatcher,
  term_docs: &'a [Vec<DocId>],
  term_group_lists: &'a [Vec<usize>],
  phrase_postings: &'a [PhraseRuntime],
  fast_fields: &'a FastFieldsReader,
}

impl<'a> QueryEvaluator<'a> {
  fn matches(&self, doc_id: DocId) -> bool {
    self.matches_node(self.matcher, doc_id)
  }

  fn matches_node(&self, node: &QueryMatcher, doc_id: DocId) -> bool {
    match node {
      QueryMatcher::MatchAll => true,
      QueryMatcher::Term(idx) => self.term_group_matches(*idx, doc_id),
      QueryMatcher::Phrase(idx) => self.phrase_matches(*idx, doc_id),
      QueryMatcher::QueryString(matcher) => {
        if matcher.term_groups.is_empty()
          && matcher.phrase_groups.is_empty()
          && matcher.not_term_groups.is_empty()
        {
          return false;
        }
        for idx in matcher.not_term_groups.iter().copied() {
          if self.term_group_matches(idx, doc_id) {
            return false;
          }
        }
        for idx in matcher.phrase_groups.iter().copied() {
          if !self.phrase_matches(idx, doc_id) {
            return false;
          }
        }
        if matcher.term_groups.is_empty() {
          return !matcher.phrase_groups.is_empty() || !matcher.not_term_groups.is_empty();
        }
        let matched_terms = matcher
          .term_groups
          .iter()
          .copied()
          .filter(|idx| self.term_group_matches(*idx, doc_id))
          .count();
        let required = matcher.minimum_should_match.unwrap_or(1);
        matched_terms >= required
      }
      QueryMatcher::DisMax(children) => {
        if children.is_empty() {
          return false;
        }
        children
          .iter()
          .any(|child| self.matches_node(child, doc_id))
      }
      QueryMatcher::Bool {
        must,
        should,
        must_not,
        filter,
        minimum_should_match,
      } => {
        for child in must.iter() {
          if !self.matches_node(child, doc_id) {
            return false;
          }
        }
        for child in must_not.iter() {
          if self.matches_node(child, doc_id) {
            return false;
          }
        }
        if !passes_filters(self.fast_fields, doc_id, filter) {
          return false;
        }
        let mut should_matches = 0usize;
        for child in should.iter() {
          if self.matches_node(child, doc_id) {
            should_matches += 1;
          }
        }
        let min_should = minimum_should_match.unwrap_or_else(|| {
          if should.is_empty() {
            0
          } else if must.is_empty() && filter.is_empty() {
            1
          } else {
            0
          }
        });
        should_matches >= min_should
      }
    }
  }

  fn matches_subquery(&self, matcher: &QueryMatcher, doc_id: DocId) -> bool {
    self.matches_node(matcher, doc_id)
  }

  fn term_group_matches(&self, group_idx: usize, doc_id: DocId) -> bool {
    let Some(group) = self.term_group_lists.get(group_idx) else {
      return false;
    };
    group.iter().copied().any(|list_idx| {
      self
        .term_docs
        .get(list_idx)
        .map(|docs| docs.binary_search(&doc_id).is_ok())
        .unwrap_or(false)
    })
  }

  fn phrase_matches(&self, phrase_idx: usize, doc_id: DocId) -> bool {
    let Some(runtime) = self.phrase_postings.get(phrase_idx) else {
      return false;
    };
    if runtime.variants.is_empty() {
      return false;
    }
    for per_term in runtime.variants.iter() {
      if matches_phrase(per_term.as_slice(), doc_id, runtime.slop) {
        return true;
      }
    }
    false
  }
}

fn expand_phrase_fields(
  phrase_specs: &[PhraseSpec],
  analysis: &SchemaAnalyzers,
  schema: &Schema,
) -> Vec<PhraseFieldConfig> {
  phrase_specs
    .iter()
    .map(|phrase| {
      let fields = phrase
        .fields
        .iter()
        .filter_map(|field| match schema.field_kind(field) {
          FieldKind::Text => analysis.search_analyzer(field).and_then(|analyzer| {
            let phrase_body = phrase.terms.join(" ");
            let tokens = analyzer.analyze(&phrase_body);
            if tokens.is_empty() {
              return None;
            }
            let mut positions: Vec<Vec<String>> = Vec::new();
            for token in tokens {
              let pos = token.position as usize;
              if positions.len() <= pos {
                positions.resize(pos + 1, Vec::new());
              }
              if !positions[pos].contains(&token.text) {
                positions[pos].push(token.text);
              }
            }
            Some((field.clone(), positions))
          }),
          FieldKind::Keyword => {
            let joined = phrase.terms.join(" ").to_ascii_lowercase();
            if joined.is_empty() {
              None
            } else {
              Some((field.clone(), vec![vec![joined]]))
            }
          }
          FieldKind::Numeric | FieldKind::Unknown => None,
        })
        .collect();
      PhraseFieldConfig {
        slop: phrase.slop,
        fields,
      }
    })
    .collect()
}

fn build_phrase_runtimes(
  seg: &SegmentReader,
  phrase_fields: &[PhraseFieldConfig],
) -> Vec<PhraseRuntime> {
  phrase_fields
    .iter()
    .map(|config| {
      let variants = config
        .fields
        .iter()
        .filter_map(|(field, positions)| {
          let mut per_position: Vec<Vec<PostingEntry>> = Vec::new();
          for alternatives in positions.iter() {
            let mut lists = Vec::new();
            for term in alternatives.iter() {
              let key = build_term_key(field, term);
              if let Some(posts) = seg.postings(&key) {
                lists.push(posts.iter().cloned().collect());
              }
            }
            if lists.is_empty() {
              return None;
            }
            per_position.push(merge_postings_lists(lists));
          }
          Some(per_position)
        })
        .collect::<Vec<Vec<Vec<PostingEntry>>>>();
      PhraseRuntime {
        slop: config.slop,
        variants,
      }
    })
    .collect()
}

fn build_term_doc_lists(seg: &SegmentReader, term_groups: &[TermMatchGroup]) -> TermDocLists {
  let mut lists = Vec::new();
  let mut indices: HashMap<String, usize> = HashMap::new();
  let mut group_lists = Vec::with_capacity(term_groups.len());
  for group in term_groups.iter() {
    let mut group_indices = Vec::new();
    for key in group.keys.iter() {
      let idx = if let Some(idx) = indices.get(key) {
        *idx
      } else {
        let docs = seg
          .postings(key)
          .map(|p| p.iter().map(|e| e.doc_id).collect())
          .unwrap_or_default();
        let idx = lists.len();
        lists.push(docs);
        indices.insert(key.clone(), idx);
        idx
      };
      group_indices.push(idx);
    }
    group_lists.push(group_indices);
  }
  TermDocLists { lists, group_lists }
}

fn merge_postings_lists(lists: Vec<Vec<PostingEntry>>) -> Vec<PostingEntry> {
  let mut merged: HashMap<DocId, PostingEntry> = HashMap::new();
  for list in lists.into_iter() {
    for entry in list.into_iter() {
      let positions = entry.positions;
      let doc_id = entry.doc_id;
      let target = merged.entry(doc_id).or_insert_with(|| PostingEntry {
        doc_id,
        term_freq: 0,
        positions: SmallVec::new(),
      });
      target.positions.extend(positions.into_iter());
    }
  }
  let mut values: Vec<_> = merged.into_values().collect();
  for entry in values.iter_mut() {
    entry.positions.sort_unstable();
    entry.positions.dedup();
    entry.term_freq = entry.positions.len() as u32;
  }
  values.sort_by_key(|e| e.doc_id);
  values
}

fn passes_root_filter(reader: &FastFieldsReader, doc_id: DocId, root: RootFilter<'_>) -> bool {
  match root {
    RootFilter::None => true,
    RootFilter::Node(filter) => passes_filter(reader, doc_id, filter),
    RootFilter::AndSlice(filters) => passes_filters(reader, doc_id, filters),
  }
}

#[derive(Default)]
struct SuggestCandidate {
  doc_freq: u64,
  score: f32,
}

fn collect_completion_candidates(
  segments: &[SegmentReader],
  field: &str,
  term: &str,
  size: usize,
  fuzzy: Option<&FuzzyOptions>,
) -> HashMap<String, SuggestCandidate> {
  let mut out: HashMap<String, SuggestCandidate> = HashMap::new();
  let max_candidates = size
    .saturating_mul(5)
    .clamp(DEFAULT_SUGGEST_SCAN, MAX_SUGGEST_CANDIDATES);
  match fuzzy {
    None => {
      let prefix_key = build_term_key(field, term);
      let field_prefix_len = field.len() + 1;
      for seg in segments.iter() {
        let mut expanded = 0usize;
        for key in seg.terms_with_prefix(&prefix_key) {
          if expanded >= max_candidates {
            break;
          }
          if key.len() <= field_prefix_len {
            continue;
          }
          let term_text = key[field_prefix_len..].to_string();
          let df = seg.postings(key).map(|p| p.len() as u64).unwrap_or(0);
          if df == 0 {
            continue;
          }
          let entry = out.entry(term_text).or_default();
          entry.doc_freq = entry.doc_freq.saturating_add(df);
          entry.score += df as f32;
          expanded += 1;
        }
      }
    }
    Some(fuzzy) => {
      let term_len = term.chars().count();
      if term_len < fuzzy.min_length || fuzzy.max_expansions == 0 {
        return out;
      }
      let max_edits = fuzzy.max_edits.min(2) as usize;
      if max_edits == 0 {
        return out;
      }
      let prefix_len = fuzzy.prefix_length.min(term_len);
      let prefix = char_prefix(term, prefix_len);
      let prefix_key = build_term_key(field, prefix);
      let field_prefix_len = field.len() + 1;
      let mut per_segment_cap = fuzzy.max_expansions.min(MAX_SUGGEST_CANDIDATES);
      per_segment_cap = per_segment_cap.max(size);
      for seg in segments.iter() {
        let mut expanded = 0usize;
        for key in seg.terms_with_prefix(&prefix_key) {
          if expanded >= per_segment_cap {
            break;
          }
          if key.len() <= field_prefix_len {
            continue;
          }
          let candidate = &key[field_prefix_len..];
          let candidate_len = candidate.chars().count();
          if candidate_len.abs_diff(term_len) > max_edits {
            continue;
          }
          let Some(distance) = bounded_levenshtein(term, candidate, max_edits) else {
            continue;
          };
          let df = seg.postings(key).map(|p| p.len() as u64).unwrap_or(0);
          if df == 0 {
            continue;
          }
          let entry = out.entry(candidate.to_string()).or_default();
          entry.doc_freq = entry.doc_freq.saturating_add(df);
          entry.score += distance_weight(distance) * df as f32;
          expanded += 1;
        }
      }
    }
  }
  out
}

pub struct IndexReader {
  pub manifest: Manifest,
  pub segments: Vec<SegmentReader>,
  analysis: SchemaAnalyzers,
  options: IndexOptions,
}

impl IndexReader {
  pub(crate) fn open(inner: Arc<InnerIndex>) -> Result<Self> {
    let manifest = inner.manifest.read().clone();
    let analysis = manifest.schema.build_analyzers()?;
    let mut segments = Vec::new();
    for seg in manifest.segments.iter() {
      segments.push(SegmentReader::open(
        inner.storage.clone(),
        seg.clone(),
        inner.options.enable_positions,
      )?);
    }
    Ok(Self {
      manifest,
      segments,
      options: IndexOptions {
        path: inner.path.clone(),
        create_if_missing: inner.options.create_if_missing,
        enable_positions: inner.options.enable_positions,
        bm25_k1: inner.options.bm25_k1,
        bm25_b: inner.options.bm25_b,
        storage: inner.options.storage.clone(),
        #[cfg(feature = "vectors")]
        vector_defaults: inner.options.vector_defaults.clone(),
      },
      analysis,
    })
  }

  fn completion_inputs(&self, field: &str, prefix: &str) -> Result<Vec<String>> {
    match self.manifest.schema.field_kind(field) {
      FieldKind::Text => {
        let analyzer = self
          .analysis
          .search_analyzer(field)
          .ok_or_else(|| anyhow::anyhow!("field `{field}` has no search analyzer"))?;
        let mut inputs = Vec::new();
        let tokens = analyzer.analyze(prefix);
        if let Some(last) = tokens.last() {
          inputs.push(last.text.clone());
        }
        if inputs.is_empty() {
          inputs.push(prefix.to_string());
        }
        inputs.sort();
        inputs.dedup();
        Ok(inputs)
      }
      FieldKind::Keyword => Ok(vec![prefix.to_ascii_lowercase()]),
      FieldKind::Numeric | FieldKind::Unknown => {
        bail!("completion suggest is only supported on text/keyword fields")
      }
    }
  }

  fn completion_suggest(
    &self,
    field: &str,
    prefix: &str,
    size: usize,
    fuzzy: Option<&FuzzyOptions>,
  ) -> Result<Vec<SuggestOption>> {
    if size == 0 {
      return Ok(Vec::new());
    }
    let inputs = self.completion_inputs(field, prefix)?;
    let mut merged: HashMap<String, SuggestCandidate> = HashMap::new();
    for term in inputs.into_iter() {
      let candidates = collect_completion_candidates(&self.segments, field, &term, size, fuzzy);
      for (text, cand) in candidates.into_iter() {
        let entry = merged.entry(text).or_default();
        entry.doc_freq = entry.doc_freq.saturating_add(cand.doc_freq);
        entry.score += cand.score;
      }
    }
    let mut options: Vec<SuggestOption> = merged
      .into_iter()
      .map(|(text, cand)| SuggestOption {
        text,
        score: cand.score,
        doc_freq: cand.doc_freq,
      })
      .collect();
    options.sort_by(|a, b| {
      b.score
        .partial_cmp(&a.score)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.text.cmp(&b.text))
    });
    options.truncate(size);
    Ok(options)
  }

  fn execute_suggest(
    &self,
    requests: &BTreeMap<String, SuggestRequest>,
  ) -> Result<BTreeMap<String, SuggestResult>> {
    let mut responses = BTreeMap::new();
    for (name, req) in requests.iter() {
      match req {
        SuggestRequest::Completion {
          field,
          prefix,
          size,
          fuzzy,
        } => {
          let options = self.completion_suggest(field, prefix, *size, fuzzy.as_ref())?;
          responses.insert(name.clone(), SuggestResult { options });
        }
      }
    }
    Ok(responses)
  }

  pub fn search(&self, req: &SearchRequest) -> Result<SearchResult> {
    if req.limit == 0 && req.cursor.is_some() {
      bail!("cursor is not supported when limit is 0; set a positive limit to page results");
    }
    if req.filter.is_some() && !req.filters.is_empty() {
      bail!("search request cannot set both `filter` and `filters`");
    }
    let sort_plan = SortPlan::from_request(&self.manifest.schema, &req.sort)?;
    let score_fast_path =
      sort_plan.is_score_only() && matches!(sort_plan.primary_order(), Some(SortOrder::Desc));
    let manifest_generation = self
      .manifest
      .segments
      .iter()
      .map(|s| s.generation)
      .max()
      .unwrap_or(0);
    let cursor_state = if req.limit == 0 {
      None
    } else if let Some(raw) = req.cursor.as_deref() {
      Some(decode_cursor(
        raw,
        manifest_generation,
        &sort_plan,
        score_fast_path,
      )?)
    } else {
      None
    };
    let cursor_key = cursor_state.as_ref().map(|c| c.key.clone());
    let cursor_returned = cursor_state
      .as_ref()
      .map(|c| c.returned as usize)
      .unwrap_or(0);
    let top_k = if req.limit == 0 {
      0
    } else {
      req.limit.saturating_add(1)
    };
    let default_fields: Vec<String> = if let Some(fields) = &req.fields {
      fields.clone()
    } else {
      self
        .manifest
        .schema
        .text_fields
        .iter()
        .map(|f| f.name.clone())
        .collect()
    };
    let query_plan = build_query_plan(&req.query, &default_fields)?;
    let compiled_score = compile_score_node(&query_plan.score_tree, &self.manifest.schema)?;
    let needs_score_hook = has_custom_scoring(&compiled_score);
    let (qualified_terms, term_groups) = expand_term_groups(
      &self.segments,
      &query_plan.term_groups,
      req.fuzzy.as_ref(),
      &self.analysis,
      &self.manifest.schema,
    )?;
    let highlight_terms: Vec<String> = {
      let mut dedup = HashSet::new();
      let mut terms = Vec::new();
      for term in qualified_terms.iter() {
        if dedup.insert(term.term.clone()) {
          terms.push(term.term.clone());
        }
      }
      terms
    };

    let phrase_fields = expand_phrase_fields(
      &query_plan.phrase_specs,
      &self.analysis,
      &self.manifest.schema,
    );
    let root_filter = if let Some(filter) = req.filter.as_ref() {
      RootFilter::Node(filter)
    } else if !req.filters.is_empty() {
      RootFilter::AndSlice(req.filters.as_slice())
    } else {
      RootFilter::None
    };

    let mut hits: Vec<RankedHit> = Vec::new();
    let mut heap = std::collections::BinaryHeap::<RankedHit>::new();
    let mut agg_results = Vec::new();
    let mut total_matches: u64 = 0;
    let mut saw_cursor = cursor_state.is_none() || req.limit == 0;
    let search_start = Instant::now();
    let mut timings: BTreeMap<String, f64> = BTreeMap::new();
    let mut search_stats = QueryStats::default();
    validate_aggregations(&self.manifest.schema, &req.aggs)?;
    let agg_pipeline =
      AggregationPipeline::from_request(&req.aggs, &highlight_terms, &self.manifest.schema);
    for (segment_ord, seg) in self.segments.iter().enumerate() {
      let mut agg_collector = agg_pipeline
        .as_ref()
        .map(|p| p.for_segment(seg, segment_ord as u32))
        .transpose()?;
      let mut noop_collector = NoopCollector;
      let mut collect_hits: Option<Box<dyn FnMut(SortKey, f32) + '_>> = None;
      if !score_fast_path && req.limit > 0 && !req.explain {
        let heap_limit = top_k;
        let heap_ref = &mut heap;
        collect_hits = Some(Box::new(move |key: SortKey, score: f32| {
          push_ranked(
            heap_ref,
            RankedHit {
              key,
              score,
              explanation: None,
            },
            heap_limit,
          );
        }));
      }
      let mut seg_hits = {
        let mut agg_ref = agg_collector
          .as_mut()
          .map(|collector| collector as &mut dyn DocCollector);
        if !score_fast_path && agg_ref.is_none() && req.limit > 0 {
          agg_ref = Some(&mut noop_collector);
        }
        let counter = if req.limit == 0 && agg_ref.is_some() {
          Some(&mut total_matches)
        } else {
          None
        };
        let segment_rank_limit = if score_fast_path {
          top_k
        } else if req.explain {
          seg.live_docs() as usize
        } else {
          0
        };
        let params = SegmentSearchParams {
          qualified_terms: &qualified_terms,
          matcher: &query_plan.matcher,
          term_groups: &term_groups,
          phrase_fields: &phrase_fields,
          scorer: query_plan.scorer.as_ref(),
          score_tree: &compiled_score,
          needs_score_hook,
          explain: req.explain,
          profile: req.profile,
          root_filter,
          agg_collector: agg_ref,
          match_counter: counter,
          req,
          segment_ord: segment_ord as u32,
          rank_limit: segment_rank_limit,
          cursor_key: cursor_key.clone(),
          saw_cursor: &mut saw_cursor,
          sort_plan: &sort_plan,
          collect_hits: collect_hits
            .as_mut()
            .map(|f| f as &mut dyn FnMut(SortKey, f32)),
          stats: if req.profile {
            Some(&mut search_stats)
          } else {
            None
          },
        };
        self.search_segment(seg, params)?
      };
      if let Some(collector) = agg_collector {
        agg_results.push(collector.finish());
      }
      hits.append(&mut seg_hits);
    }

    if !saw_cursor {
      bail!("stale or invalid cursor for this result set");
    }

    if !score_fast_path {
      hits.extend(heap);
    }
    hits.sort_by(|a, b| a.key.cmp(&b.key));
    let search_phase_end = if req.profile {
      Some(Instant::now())
    } else {
      None
    };
    let mut rescore_stats = QueryStats::default();
    if let Some(rescore_req) = req.rescore.as_ref() {
      let rescore_start = Instant::now();
      self.rescore_hits(
        hits.as_mut_slice(),
        rescore_req,
        &default_fields,
        &sort_plan,
        req,
        &mut rescore_stats,
      )?;
      if req.profile {
        timings.insert(
          "rescore_ms".to_string(),
          rescore_start.elapsed().as_secs_f64() * 1000.0,
        );
      }
    }
    if req.explain {
      for hit in hits.iter_mut() {
        if let Some(expl) = hit.explanation.as_mut() {
          expl.final_score = hit.score;
        } else {
          hit.explanation = Some(HitExplanation {
            base_score: hit.score,
            functions: Vec::new(),
            rescore: None,
            final_score: hit.score,
          });
        }
      }
    }
    if req.profile {
      let end = search_phase_end.unwrap_or_else(Instant::now);
      timings.insert(
        "search_ms".to_string(),
        end.duration_since(search_start).as_secs_f64() * 1000.0,
      );
    }
    let mut next_cursor = None;
    if req.limit > 0 && hits.len() > req.limit {
      let last = &hits[req.limit - 1];
      let returned = cursor_returned
        .saturating_add(req.limit)
        .try_into()
        .unwrap_or(u32::MAX);
      next_cursor = Some(encode_cursor(
        manifest_generation,
        returned,
        &last.key,
        &sort_plan,
        score_fast_path,
      )?);
      hits.truncate(req.limit);
    }
    let hits: Vec<Hit> = hits
      .into_iter()
      .filter_map(|h| self.materialize_hit(h, req, &highlight_terms))
      .collect();
    let aggregations = if let Some(pipeline) = agg_pipeline {
      pipeline.merge(agg_results)?
    } else {
      BTreeMap::new()
    };
    let suggest = if req.suggest.is_empty() {
      BTreeMap::new()
    } else {
      self.execute_suggest(&req.suggest)?
    };
    Ok(SearchResult {
      total_hits_estimate: if req.limit == 0 {
        total_matches
      } else {
        hits.len() as u64
      },
      hits,
      next_cursor,
      aggregations,
      suggest,
      profile: if req.profile {
        Some(ProfileResult {
          execution: to_execution_profile(&search_stats),
          rescore: if req.rescore.is_some() {
            Some(to_execution_profile(&rescore_stats))
          } else {
            None
          },
          timings,
        })
      } else {
        None
      },
    })
  }

  fn search_segment(
    &self,
    seg: &SegmentReader,
    params: SegmentSearchParams<'_>,
  ) -> Result<Vec<RankedHit>> {
    let SegmentSearchParams {
      qualified_terms,
      matcher,
      term_groups,
      phrase_fields,
      scorer,
      score_tree,
      needs_score_hook,
      explain,
      profile: _profile,
      root_filter,
      agg_collector,
      match_counter,
      req,
      segment_ord,
      rank_limit,
      cursor_key,
      saw_cursor,
      sort_plan,
      collect_hits,
      stats,
    } = params;
    let use_score_hook = needs_score_hook || explain;
    let score_mode = if sort_plan.uses_score() || use_score_hook {
      ScoreMode::Score
    } else {
      ScoreMode::MatchOnly
    };
    let term_doc_lists = build_term_doc_lists(seg, term_groups);
    let phrase_postings: Vec<PhraseRuntime> = build_phrase_runtimes(seg, phrase_fields);
    let query_eval = QueryEvaluator {
      matcher,
      term_docs: &term_doc_lists.lists,
      term_group_lists: &term_doc_lists.group_lists,
      phrase_postings: &phrase_postings,
      fast_fields: seg.fast_fields(),
    };
    if qualified_terms.is_empty() {
      return self.scan_segment(
        seg,
        &query_eval,
        root_filter,
        agg_collector,
        match_counter,
        segment_ord,
        rank_limit,
        cursor_key,
        saw_cursor,
        sort_plan,
        collect_hits,
        score_tree,
        needs_score_hook,
        explain,
        scorer,
        stats,
      );
    }
    let explanations: RefCell<HashMap<DocId, HitExplanation>> = RefCell::new(HashMap::new());
    let mut term_weights: HashMap<String, (String, f32, usize)> = HashMap::new();
    for term in qualified_terms.iter() {
      let entry =
        term_weights
          .entry(term.key.clone())
          .or_insert((term.field.clone(), 0.0, term.leaf));
      debug_assert_eq!(
        entry.2, term.leaf,
        "Inconsistent leaf for term key {} (expected {}, got {})",
        term.key, entry.2, term.leaf
      );
      entry.1 += term.weight;
    }

    let docs = seg.live_docs() as f32;
    let mut terms: Vec<ScoredTerm> = Vec::new();
    for (key, (field, weight, leaf)) in term_weights.into_iter() {
      if let Some(postings) = seg.postings(&key) {
        terms.push(ScoredTerm {
          postings,
          weight,
          avgdl: seg.avg_field_length(&field),
          docs,
          k1: self.options.bm25_k1,
          b: self.options.bm25_b,
          leaf,
        });
      }
    }
    if terms.is_empty() {
      return Ok(Vec::new());
    }

    let mut match_counter = match_counter;
    let mut collect_hits = collect_hits;
    let mut accept = |doc_id: DocId, score: f32| -> bool {
      if seg.is_deleted(doc_id) {
        return false;
      }
      if !query_eval.matches(doc_id) {
        return false;
      }
      if !passes_root_filter(seg.fast_fields(), doc_id, root_filter) {
        return false;
      }
      let key = sort_plan.build_key(seg, doc_id, score, segment_ord);
      if let Some(cur) = &cursor_key {
        let ord = key.cmp(cur);
        if ord.is_lt() || ord.is_eq() {
          if ord.is_eq() {
            *saw_cursor = true;
          }
          return false;
        }
      }
      if let Some(counter) = match_counter.as_mut() {
        **counter += 1;
      }
      if let Some(collector) = collect_hits.as_mut() {
        (*collector)(key, score);
      }
      true
    };

    let ranked = if use_score_hook {
      let score_plan = scorer;
      let score_tree_ref = score_tree;
      let eval_ref = &query_eval;
      let explain_enabled = explain;
      let fast_fields = seg.fast_fields();
      let explanations_ref = &explanations;
      let mut adjust: Box<ScoreAdjustFn<'_>> =
        Box::new(move |doc_id: DocId, raw_score: f32, leaves: &[f32]| {
          let mut fn_details = Vec::new();
          let final_score = evaluate_compiled_score(
            score_tree_ref,
            eval_ref,
            fast_fields,
            doc_id,
            leaves,
            explain_enabled,
            &mut fn_details,
          )?;
          if explain_enabled {
            let base_score = if let Some(plan) = score_plan {
              plan.evaluate(leaves)
            } else {
              raw_score
            };
            explanations_ref.borrow_mut().insert(
              doc_id,
              HitExplanation {
                base_score,
                functions: fn_details,
                rescore: None,
                final_score,
              },
            );
          }
          Some(final_score)
        });
      execute_top_k_with_stats_and_mode_internal(
        terms,
        rank_limit,
        req.execution.clone(),
        req.bmw_block_size,
        scorer,
        &mut accept,
        agg_collector,
        stats,
        score_mode,
        Some(&mut adjust),
      )
    } else {
      execute_top_k_with_stats_and_mode_internal(
        terms,
        rank_limit,
        req.execution.clone(),
        req.bmw_block_size,
        scorer,
        &mut accept,
        agg_collector,
        stats,
        score_mode,
        None,
      )
    };
    let mut explanations_map = explanations.into_inner();

    Ok(
      ranked
        .into_iter()
        .map(|rd| {
          let explanation = explanations_map.remove(&rd.doc_id).or_else(|| {
            if explain {
              Some(HitExplanation {
                base_score: rd.score,
                functions: Vec::new(),
                rescore: None,
                final_score: rd.score,
              })
            } else {
              None
            }
          });
          RankedHit {
            key: sort_plan.build_key(seg, rd.doc_id, rd.score, segment_ord),
            score: rd.score,
            explanation,
          }
        })
        .collect(),
    )
  }

  #[allow(clippy::too_many_arguments)]
  fn scan_segment(
    &self,
    seg: &SegmentReader,
    query_eval: &QueryEvaluator<'_>,
    root_filter: RootFilter<'_>,
    mut agg_collector: Option<&mut dyn DocCollector>,
    match_counter: Option<&mut u64>,
    segment_ord: u32,
    rank_limit: usize,
    cursor_key: Option<SortKey>,
    saw_cursor: &mut bool,
    sort_plan: &SortPlan,
    mut collect_hits: Option<&mut dyn FnMut(SortKey, f32)>,
    score_tree: &CompiledScoreNode,
    needs_score_hook: bool,
    explain: bool,
    scorer: Option<&ScorePlan>,
    mut stats: Option<&mut QueryStats>,
  ) -> Result<Vec<RankedHit>> {
    let mut local_heap = std::collections::BinaryHeap::<RankedHit>::new();
    let default_score = if sort_plan.uses_score() { 1.0 } else { 0.0 };
    let mut explanations: HashMap<DocId, HitExplanation> = HashMap::new();
    let mut match_counter = match_counter;
    for raw in 0..seg.meta.doc_count {
      let doc_id = raw as DocId;
      if seg.is_deleted(doc_id) {
        continue;
      }
      if !query_eval.matches(doc_id) {
        continue;
      }
      if !passes_root_filter(seg.fast_fields(), doc_id, root_filter) {
        continue;
      }
      let mut fn_details = Vec::new();
      let computed_score = if needs_score_hook || explain {
        let result = evaluate_compiled_score(
          score_tree,
          query_eval,
          seg.fast_fields(),
          doc_id,
          &[],
          explain,
          &mut fn_details,
        );
        match result {
          Some(score) => score,
          None => continue,
        }
      } else {
        default_score
      };
      if let Some(stats) = stats.as_deref_mut() {
        stats.candidates_examined += 1;
        stats.scored_docs += 1;
      }
      let key = sort_plan.build_key(seg, doc_id, computed_score, segment_ord);
      if let Some(cur) = &cursor_key {
        let ord = key.cmp(cur);
        if ord.is_lt() || ord.is_eq() {
          if ord.is_eq() {
            *saw_cursor = true;
          }
          continue;
        }
      }
      if let Some(counter) = match_counter.as_mut() {
        **counter += 1;
      }
      if explain {
        let base_score = if let Some(plan) = scorer {
          plan.evaluate(&[])
        } else {
          default_score
        };
        explanations.insert(
          doc_id,
          HitExplanation {
            base_score,
            functions: fn_details,
            rescore: None,
            final_score: computed_score,
          },
        );
      }
      if let Some(collector) = agg_collector.as_deref_mut() {
        collector.collect(doc_id, computed_score);
      }
      if let Some(collector) = collect_hits.as_mut() {
        (*collector)(key, computed_score);
      } else if rank_limit > 0 {
        let explanation = explanations.remove(&doc_id);
        push_ranked(
          &mut local_heap,
          RankedHit {
            key,
            score: computed_score,
            explanation,
          },
          rank_limit,
        );
      }
    }
    Ok(local_heap.into_iter().collect())
  }

  fn rescore_hits(
    &self,
    hits: &mut [RankedHit],
    rescore: &RescoreRequest,
    default_fields: &[String],
    sort_plan: &SortPlan,
    req: &SearchRequest,
    stats: &mut QueryStats,
  ) -> Result<()> {
    if hits.is_empty() {
      return Ok(());
    }
    let window = rescore.window_size.min(hits.len());
    if window == 0 {
      return Ok(());
    }
    let rescore_plan = build_query_plan(&Query::Node(rescore.query.clone()), default_fields)?;
    let compiled_score = compile_score_node(&rescore_plan.score_tree, &self.manifest.schema)?;
    let (qualified_terms, term_groups) = expand_term_groups(
      &self.segments,
      &rescore_plan.term_groups,
      req.fuzzy.as_ref(),
      &self.analysis,
      &self.manifest.schema,
    )?;
    let phrase_fields = expand_phrase_fields(
      &rescore_plan.phrase_specs,
      &self.analysis,
      &self.manifest.schema,
    );
    let mut per_segment: HashMap<u32, Vec<(DocId, usize)>> = HashMap::new();
    for (idx, hit) in hits.iter().take(window).enumerate() {
      per_segment
        .entry(hit.key.segment_ord)
        .or_default()
        .push((hit.key.doc_id, idx));
    }
    for (segment_ord, docs) in per_segment.into_iter() {
      let Some(seg) = self.segments.get(segment_ord as usize) else {
        continue;
      };
      let term_doc_lists = build_term_doc_lists(seg, &term_groups);
      let phrase_postings = build_phrase_runtimes(seg, &phrase_fields);
      let query_eval = QueryEvaluator {
        matcher: &rescore_plan.matcher,
        term_docs: &term_doc_lists.lists,
        term_group_lists: &term_doc_lists.group_lists,
        phrase_postings: &phrase_postings,
        fast_fields: seg.fast_fields(),
      };
      let mut term_weights: HashMap<String, (String, f32, usize)> = HashMap::new();
      for term in qualified_terms.iter() {
        let entry =
          term_weights
            .entry(term.key.clone())
            .or_insert((term.field.clone(), 0.0, term.leaf));
        entry.1 += term.weight;
      }
      let docs_count = seg.live_docs() as f32;
      let mut terms: Vec<ScoredTerm> = Vec::new();
      for (key, (field, weight, leaf)) in term_weights.into_iter() {
        if let Some(postings) = seg.postings(&key) {
          terms.push(ScoredTerm {
            postings,
            weight,
            avgdl: seg.avg_field_length(&field),
            docs: docs_count,
            k1: self.options.bm25_k1,
            b: self.options.bm25_b,
            leaf,
          });
        }
      }
      for (doc_id, hit_idx) in docs.into_iter() {
        if seg.is_deleted(doc_id) {
          continue;
        }
        if !query_eval.matches(doc_id) {
          continue;
        }
        stats.candidates_examined += 1;
        let mut leaf_scores = rescore_plan
          .scorer
          .as_ref()
          .map(|plan| vec![0.0_f32; plan.leaf_count])
          .unwrap_or_default();
        for term in terms.iter() {
          if let Some(tf) = term_freq_for_doc(&term.postings, doc_id) {
            let df = term.postings.len() as f32;
            let contribution =
              score_tf(tf, df, term.avgdl, term.docs, term.k1, term.b, term.weight);
            if let Some(buf) = leaf_scores.get_mut(term.leaf) {
              *buf += contribution;
            }
          }
        }
        let base_score = rescore_plan
          .scorer
          .as_ref()
          .map(|plan| plan.evaluate(&leaf_scores))
          .unwrap_or_else(|| leaf_scores.iter().copied().sum());
        let mut fn_details = Vec::new();
        let rescore_score = evaluate_compiled_score(
          &compiled_score,
          &query_eval,
          seg.fast_fields(),
          doc_id,
          &leaf_scores,
          req.explain,
          &mut fn_details,
        )
        .unwrap_or(base_score);
        stats.scored_docs += 1;
        stats.postings_advanced += terms.len();
        let hit = hits.get_mut(hit_idx).unwrap();
        let orig_score = hit.score;
        let combined = combine_rescore_scores(rescore.score_mode, orig_score, rescore_score);
        hit.score = combined;
        hit.key = sort_plan.build_key(seg, doc_id, combined, segment_ord);
        if req.explain {
          let mut expl = hit.explanation.take().unwrap_or(HitExplanation {
            base_score: orig_score,
            functions: Vec::new(),
            rescore: None,
            final_score: orig_score,
          });
          expl.rescore = Some(RescoreExplanation {
            rescore_score,
            combined_score: combined,
            functions: fn_details,
          });
          expl.final_score = combined;
          hit.explanation = Some(expl);
        }
      }
    }
    hits[..window].sort_by(|a, b| a.key.cmp(&b.key));
    Ok(())
  }

  fn materialize_hit(
    &self,
    ranked: RankedHit,
    req: &SearchRequest,
    highlight_terms: &[String],
  ) -> Option<Hit> {
    let seg = self.segments.get(ranked.key.segment_ord as usize)?;
    let doc_id_str = seg.doc_id(ranked.key.doc_id)?;
    let need_doc = req.return_stored || req.highlight_field.is_some();
    let mut doc_cache = None;
    if need_doc {
      doc_cache = seg.get_doc(ranked.key.doc_id).ok();
    }

    let snippet = if let (Some(field), Some(doc)) = (&req.highlight_field, doc_cache.as_ref()) {
      if let Some(text_val) = doc.get(field).and_then(|v| v.as_str()) {
        make_snippet(text_val, highlight_terms)
      } else {
        None
      }
    } else {
      None
    };

    let fields_val = if req.return_stored {
      doc_cache.clone()
    } else {
      None
    };

    Some(Hit {
      doc_id: doc_id_str.to_string(),
      score: ranked.score,
      fields: fields_val,
      snippet,
      explanation: ranked.explanation,
    })
  }
}

fn term_freq_for_doc(postings: &PostingsReader, doc_id: DocId) -> Option<f32> {
  let entries = postings.entries();
  let idx = entries.binary_search_by_key(&doc_id, |e| e.doc_id).ok()?;
  Some(entries.get(idx)?.term_freq as f32)
}

fn combine_rescore_scores(mode: RescoreMode, original: f32, rescore: f32) -> f32 {
  match mode {
    // Total is intentionally an alias for Sum to match Elasticsearch naming.
    RescoreMode::Total | RescoreMode::Sum => original + rescore,
    RescoreMode::Multiply => original * rescore,
    RescoreMode::Max => original.max(rescore),
    RescoreMode::Min => original.min(rescore),
  }
}

fn to_execution_profile(stats: &QueryStats) -> ExecutionProfile {
  ExecutionProfile {
    scored_docs: stats.scored_docs,
    candidates_examined: stats.candidates_examined,
    postings_advanced: stats.postings_advanced,
  }
}

fn validate_aggregations(schema: &Schema, aggs: &BTreeMap<String, Aggregation>) -> Result<()> {
  for (name, agg) in aggs.iter() {
    match agg {
      Aggregation::Terms(t) => {
        ensure_keyword_fast(schema, &t.field, name)?;
        validate_aggregations(schema, &t.aggs)?;
      }
      Aggregation::Range(r) => {
        ensure_numeric_fast(schema, &r.field, name)?;
        validate_aggregations(schema, &r.aggs)?;
      }
      Aggregation::DateRange(r) => {
        ensure_numeric_fast(schema, &r.field, name)?;
        validate_aggregations(schema, &r.aggs)?;
      }
      Aggregation::Histogram(h) => {
        ensure_numeric_fast(schema, &h.field, name)?;
        validate_histogram_config(name, h)?;
        validate_aggregations(schema, &h.aggs)?;
      }
      Aggregation::DateHistogram(h) => {
        ensure_numeric_fast(schema, &h.field, name)?;
        validate_date_histogram_config(name, h)?;
        validate_aggregations(schema, &h.aggs)?;
      }
      Aggregation::Stats(m) | Aggregation::ExtendedStats(m) | Aggregation::ValueCount(m) => {
        ensure_numeric_fast(schema, &m.field, name)?
      }
      Aggregation::TopHits(t) => {
        SortPlan::from_request(schema, &t.sort)
          .with_context(|| format!("invalid top_hits sort in aggregation `{name}`"))?;
      }
    }
  }
  Ok(())
}

fn ensure_keyword_fast(schema: &Schema, field: &str, agg: &str) -> Result<()> {
  if let Some(def) = schema.keyword_fields.iter().find(|f| f.name == field) {
    if def.fast {
      Ok(())
    } else {
      Err(
        AggregationError::MissingFastField {
          field: field.to_string(),
        }
        .into(),
      )
    }
  } else {
    Err(
      AggregationError::UnsupportedFieldType {
        agg: agg.to_string(),
        field: field.to_string(),
        expected: "fast keyword field".to_string(),
      }
      .into(),
    )
  }
}

fn ensure_numeric_fast(schema: &Schema, field: &str, agg: &str) -> Result<()> {
  if let Some(def) = schema.numeric_fields.iter().find(|f| f.name == field) {
    if def.fast {
      return Ok(());
    }
    return Err(
      AggregationError::MissingFastField {
        field: field.to_string(),
      }
      .into(),
    );
  }
  Err(
    AggregationError::UnsupportedFieldType {
      agg: agg.to_string(),
      field: field.to_string(),
      expected: "fast numeric field".to_string(),
    }
    .into(),
  )
}

fn validate_histogram_config(name: &str, agg: &HistogramAggregation) -> Result<()> {
  if agg.interval <= 0.0 {
    return Err(
      AggregationError::InvalidConfig {
        reason: format!("histogram `{name}` requires interval > 0"),
      }
      .into(),
    );
  }
  if let Some(bounds) = &agg.extended_bounds {
    if bounds.min > bounds.max {
      return Err(
        AggregationError::InvalidConfig {
          reason: format!("histogram `{name}` extended_bounds.min > max"),
        }
        .into(),
      );
    }
  }
  if let Some(bounds) = &agg.hard_bounds {
    if bounds.min > bounds.max {
      return Err(
        AggregationError::InvalidConfig {
          reason: format!("histogram `{name}` hard_bounds.min > max"),
        }
        .into(),
      );
    }
    if let Some(ext) = &agg.extended_bounds {
      if ext.min < bounds.min || ext.max > bounds.max {
        return Err(
          AggregationError::InvalidConfig {
            reason: format!("histogram `{name}` extended_bounds must be within hard_bounds"),
          }
          .into(),
        );
      }
    }
  }
  Ok(())
}

fn validate_date_histogram_config(name: &str, agg: &DateHistogramAggregation) -> Result<()> {
  let has_calendar = agg.calendar_interval.is_some();
  let has_fixed = agg.fixed_interval.is_some();
  if !has_calendar && !has_fixed {
    return Err(
      AggregationError::InvalidConfig {
        reason: format!("date_histogram `{name}` requires `calendar_interval` or `fixed_interval`"),
      }
      .into(),
    );
  }
  if let Some(cal) = &agg.calendar_interval {
    if parse_calendar_interval(cal).is_none() {
      return Err(
        AggregationError::InvalidConfig {
          reason: format!("date_histogram `{name}` calendar_interval `{cal}` is not supported"),
        }
        .into(),
      );
    }
  }
  if let Some(fixed) = &agg.fixed_interval {
    if parse_interval_seconds(fixed).is_none() {
      return Err(
        AggregationError::InvalidConfig {
          reason: format!("date_histogram `{name}` fixed_interval `{fixed}` is invalid"),
        }
        .into(),
      );
    }
  }
  if let Some(offset) = &agg.offset {
    if parse_interval_seconds(offset).is_none() {
      return Err(
        AggregationError::InvalidConfig {
          reason: format!("date_histogram `{name}` offset `{offset}` is invalid"),
        }
        .into(),
      );
    }
  }
  if let Some(bounds) = &agg.extended_bounds {
    let min = parse_date(&bounds.min).ok_or_else(|| AggregationError::InvalidConfig {
      reason: format!(
        "date_histogram `{name}` extended_bounds.min `{}` is not a valid date/number",
        bounds.min
      ),
    })?;
    let max = parse_date(&bounds.max).ok_or_else(|| AggregationError::InvalidConfig {
      reason: format!(
        "date_histogram `{name}` extended_bounds.max `{}` is not a valid date/number",
        bounds.max
      ),
    })?;
    if min > max {
      return Err(
        AggregationError::InvalidConfig {
          reason: format!("date_histogram `{name}` extended_bounds.min > max"),
        }
        .into(),
      );
    }
  }
  if let Some(bounds) = &agg.hard_bounds {
    let min = parse_date(&bounds.min).ok_or_else(|| AggregationError::InvalidConfig {
      reason: format!(
        "date_histogram `{name}` hard_bounds.min `{}` is not a valid date/number",
        bounds.min
      ),
    })?;
    let max = parse_date(&bounds.max).ok_or_else(|| AggregationError::InvalidConfig {
      reason: format!(
        "date_histogram `{name}` hard_bounds.max `{}` is not a valid date/number",
        bounds.max
      ),
    })?;
    if min > max {
      return Err(
        AggregationError::InvalidConfig {
          reason: format!("date_histogram `{name}` hard_bounds.min > max"),
        }
        .into(),
      );
    }
    if let Some(ext) = &agg.extended_bounds {
      let ext_min = parse_date(&ext.min).unwrap_or(min);
      let ext_max = parse_date(&ext.max).unwrap_or(max);
      if ext_min < min || ext_max > max {
        return Err(
          AggregationError::InvalidConfig {
            reason: format!("date_histogram `{name}` extended_bounds must be within hard_bounds"),
          }
          .into(),
        );
      }
    }
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::api::types::{
    ExecutionStrategy, FieldSpec, IndexOptions, MatchOperator, MultiMatchType, QueryNode, Schema,
    SearchRequest, TextField,
  };
  use crate::api::{Document, Index, Query, StorageType};
  use crate::query::wand::{execute_top_k_with_stats_and_mode_internal, ScoreMode, ScoredTerm};
  use std::collections::HashSet;

  #[test]
  fn pagination_cursor_roundtrips() {
    let cursor = PaginationCursor {
      version: CURSOR_VERSION,
      generation: 2,
      key: score_sort_key(1.5, 2, 3, SortOrder::Desc),
      returned: 42,
    };
    let encoded = cursor.encode();
    let decoded = PaginationCursor::decode(&encoded).unwrap();
    assert_eq!(decoded, cursor);
  }

  #[test]
  fn pagination_cursor_rejects_bad_length() {
    assert!(PaginationCursor::decode("deadbeef").is_err());
  }

  #[test]
  fn pagination_cursor_rejects_non_hex() {
    let invalid = "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"; // 42 chars, not hex
    assert!(PaginationCursor::decode(invalid).is_err());
  }

  #[test]
  fn pagination_cursor_rejects_excessive_advance() {
    let mut buf = [0u8; CURSOR_BYTES];
    buf[0] = CURSOR_VERSION;
    let returned = (MAX_CURSOR_ADVANCE as u32).saturating_add(1);
    buf[17..].copy_from_slice(&returned.to_be_bytes());
    let encoded = hex_encode(&buf);
    assert!(PaginationCursor::decode(&encoded).is_err());
  }

  #[test]
  fn multi_match_term_groups_cover_all_fields() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("idx");
    let mut schema = Schema::default_text_body();
    schema.text_fields.push(TextField {
      name: "title".into(),
      analyzer: "default".into(),
      search_analyzer: None,
      stored: true,
      indexed: true,
      nullable: false,
      search_as_you_type: None,
    });
    let idx = Index::create(
      &path,
      schema,
      IndexOptions {
        path: path.clone(),
        create_if_missing: true,
        enable_positions: true,
        bm25_k1: 0.9,
        bm25_b: 0.4,
        storage: StorageType::Filesystem,
        #[cfg(feature = "vectors")]
        vector_defaults: None,
      },
    )
    .unwrap();
    let mut writer = idx.writer().unwrap();
    let docs = vec![
      Document {
        fields: [
          ("_id".to_string(), serde_json::json!("doc-1")),
          ("title".to_string(), serde_json::json!("rust search")),
          ("body".to_string(), serde_json::json!("fast")),
        ]
        .into_iter()
        .collect(),
      },
      Document {
        fields: [
          ("_id".to_string(), serde_json::json!("doc-2")),
          ("title".to_string(), serde_json::json!("rust")),
          ("body".to_string(), serde_json::json!("search")),
        ]
        .into_iter()
        .collect(),
      },
      Document {
        fields: [
          ("_id".to_string(), serde_json::json!("doc-3")),
          ("title".to_string(), serde_json::json!("rust")),
          ("body".to_string(), serde_json::json!("rust search")),
        ]
        .into_iter()
        .collect(),
      },
      Document {
        fields: [
          ("_id".to_string(), serde_json::json!("doc-4")),
          ("title".to_string(), serde_json::json!("boring")),
          ("body".to_string(), serde_json::json!("rust")),
        ]
        .into_iter()
        .collect(),
      },
      Document {
        fields: [
          ("_id".to_string(), serde_json::json!("doc-5")),
          ("title".to_string(), serde_json::json!("none")),
          ("body".to_string(), serde_json::json!("rust fast search")),
        ]
        .into_iter()
        .collect(),
      },
    ];
    for doc in docs {
      writer.add_document(&doc).unwrap();
    }
    writer.commit().unwrap();
    let reader = idx.reader().unwrap();
    let default_fields: Vec<String> = reader
      .manifest
      .schema
      .text_fields
      .iter()
      .map(|f| f.name.clone())
      .collect();
    let plan = build_query_plan(
      &Query::Node(QueryNode::MultiMatch {
        query: "rust search".into(),
        fields: vec![
          FieldSpec {
            field: "title".into(),
            boost: None,
          },
          FieldSpec {
            field: "body".into(),
            boost: None,
          },
        ],
        match_type: MultiMatchType::BestFields,
        tie_breaker: None,
        operator: Some(MatchOperator::Or),
        minimum_should_match: None,
        boost: None,
      }),
      &default_fields,
    )
    .unwrap();
    let (qualified_terms, term_groups) = expand_term_groups(
      &reader.segments,
      &plan.term_groups,
      None,
      &reader.analysis,
      &reader.manifest.schema,
    )
    .unwrap();
    assert_eq!(term_groups.len(), 2);
    assert!(term_groups[0].keys.iter().any(|k| k == "title:rust"));
    assert!(term_groups[0].keys.iter().any(|k| k == "body:rust"));
    assert!(term_groups[1].keys.iter().any(|k| k == "title:search"));
    assert!(term_groups[1].keys.iter().any(|k| k == "body:search"));
    assert!(qualified_terms.iter().any(|t| t.key == "body:rust"));
    let seg = &reader.segments[0];
    let term_docs = build_term_doc_lists(seg, &term_groups);
    let doc2 = (0..seg.meta.doc_count)
      .map(|raw| raw as DocId)
      .find(|id| seg.doc_id(*id) == Some("doc-2"))
      .unwrap();
    let doc4 = (0..seg.meta.doc_count)
      .map(|raw| raw as DocId)
      .find(|id| seg.doc_id(*id) == Some("doc-4"))
      .unwrap();
    let evaluator = QueryEvaluator {
      matcher: &plan.matcher,
      term_docs: &term_docs.lists,
      term_group_lists: &term_docs.group_lists,
      phrase_postings: &[],
      fast_fields: seg.fast_fields(),
    };
    assert!(evaluator.matches(doc2));
    assert!(evaluator.matches(doc4));
    let mut term_weights: HashMap<String, (String, f32, usize)> = HashMap::new();
    for term in qualified_terms.iter() {
      let entry =
        term_weights
          .entry(term.key.clone())
          .or_insert((term.field.clone(), 0.0, term.leaf));
      debug_assert_eq!(
        entry.2, term.leaf,
        "Inconsistent leaf for term key {} (expected {}, got {})",
        term.key, entry.2, term.leaf
      );
      entry.1 += term.weight;
    }
    let docs = seg.live_docs() as f32;
    let mut scored_terms = Vec::new();
    for (key, (field, weight, leaf)) in term_weights.into_iter() {
      if let Some(postings) = seg.postings(&key) {
        scored_terms.push(ScoredTerm {
          postings,
          weight,
          avgdl: seg.avg_field_length(&field),
          docs,
          k1: reader.options.bm25_k1,
          b: reader.options.bm25_b,
          leaf,
        });
      }
    }
    let mut seen_matches: Vec<String> = Vec::new();
    let mut filter_doc = |doc_id: DocId, _score: f32| -> bool {
      if seg.is_deleted(doc_id) {
        return false;
      }
      if !evaluator.matches(doc_id) {
        return false;
      }
      if let Some(ext) = seg.doc_id(doc_id) {
        seen_matches.push(ext.to_string());
      }
      true
    };
    let ranked = execute_top_k_with_stats_and_mode_internal(
      scored_terms,
      6,
      ExecutionStrategy::Wand,
      None,
      plan.scorer.as_ref(),
      &mut filter_doc,
      None::<&mut crate::query::collector::MatchCountingCollector>,
      None,
      ScoreMode::Score,
      None,
    );
    let ranked_ids: Vec<_> = ranked
      .iter()
      .filter_map(|rd| seg.doc_id(rd.doc_id))
      .map(str::to_string)
      .collect();
    assert_eq!(
      seen_matches.len(),
      5,
      "accepted: {:?}, ranked: {:?}",
      seen_matches,
      ranked_ids
    );
    let res = reader
      .search(&SearchRequest {
        query: Query::Node(QueryNode::MultiMatch {
          query: "rust search".into(),
          fields: vec![
            FieldSpec {
              field: "title".into(),
              boost: None,
            },
            FieldSpec {
              field: "body".into(),
              boost: None,
            },
          ],
          match_type: MultiMatchType::BestFields,
          tie_breaker: None,
          operator: Some(MatchOperator::Or),
          minimum_should_match: None,
          boost: None,
        }),
        fields: None,
        filter: None,
        filters: vec![],
        limit: 5,
        sort: Vec::new(),
        cursor: None,
        execution: ExecutionStrategy::Wand,
        bmw_block_size: None,
        fuzzy: None,
        #[cfg(feature = "vectors")]
        vector_query: None,
        return_stored: false,
        highlight_field: None,
        aggs: BTreeMap::new(),
        suggest: BTreeMap::new(),
        rescore: None,
        explain: false,
        profile: false,
      })
      .unwrap();
    let ids: Vec<_> = res.hits.iter().map(|h| h.doc_id.as_str()).collect();
    assert_eq!(ids.len(), 5, "hits: {:?}", ids);
  }

  #[test]
  fn prefix_expansion_respects_max_expansions() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("idx");
    let idx = Index::create(
      &path,
      Schema::default_text_body(),
      IndexOptions {
        path: path.clone(),
        create_if_missing: true,
        enable_positions: true,
        bm25_k1: 0.9,
        bm25_b: 0.4,
        storage: StorageType::Filesystem,
        #[cfg(feature = "vectors")]
        vector_defaults: None,
      },
    )
    .unwrap();
    let mut writer = idx.writer().unwrap();
    for (id, body) in [("1", "ruby"), ("2", "rumor"), ("3", "rust")] {
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!(id)),
            ("body".into(), serde_json::json!(body)),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
    }
    writer.commit().unwrap();
    let reader = idx.reader().unwrap();
    let default_fields: Vec<String> = reader
      .manifest
      .schema
      .text_fields
      .iter()
      .map(|f| f.name.clone())
      .collect();
    let plan = build_query_plan(
      &Query::Node(QueryNode::Prefix {
        field: "body".into(),
        value: "ru".into(),
        max_expansions: Some(2),
        boost: None,
      }),
      &default_fields,
    )
    .unwrap();
    let (_, term_groups) = expand_term_groups(
      &reader.segments,
      &plan.term_groups,
      None,
      &reader.analysis,
      &reader.manifest.schema,
    )
    .unwrap();
    let keys: HashSet<_> = term_groups[0].keys.iter().cloned().collect();
    assert_eq!(keys.len(), 2);
    assert!(keys.contains("body:rumor"));
    assert!(keys.contains("body:ruby"));
    assert!(!keys.contains("body:rust"));
  }

  #[test]
  fn wildcard_expansion_handles_star_and_question() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("idx-wildcard");
    let idx = Index::create(
      &path,
      Schema::default_text_body(),
      IndexOptions {
        path: path.clone(),
        create_if_missing: true,
        enable_positions: true,
        bm25_k1: 0.9,
        bm25_b: 0.4,
        storage: StorageType::Filesystem,
        #[cfg(feature = "vectors")]
        vector_defaults: None,
      },
    )
    .unwrap();
    let mut writer = idx.writer().unwrap();
    for (id, body) in [("1", "rust"), ("2", "rest"), ("3", "roast"), ("4", "roost")] {
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!(id)),
            ("body".into(), serde_json::json!(body)),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
    }
    writer.commit().unwrap();
    let reader = idx.reader().unwrap();
    let default_fields: Vec<String> = reader
      .manifest
      .schema
      .text_fields
      .iter()
      .map(|f| f.name.clone())
      .collect();
    let star_plan = build_query_plan(
      &Query::Node(QueryNode::Wildcard {
        field: "body".into(),
        value: "r*st".into(),
        max_expansions: None,
        boost: None,
      }),
      &default_fields,
    )
    .unwrap();
    let (_, star_groups) = expand_term_groups(
      &reader.segments,
      &star_plan.term_groups,
      None,
      &reader.analysis,
      &reader.manifest.schema,
    )
    .unwrap();
    let star_keys: HashSet<_> = star_groups[0].keys.iter().cloned().collect();
    assert!(star_keys.contains("body:rust"));
    assert!(star_keys.contains("body:rest"));
    assert!(star_keys.contains("body:roast"));
    assert!(star_keys.contains("body:roost"));

    let question_plan = build_query_plan(
      &Query::Node(QueryNode::Wildcard {
        field: "body".into(),
        value: "ro?st".into(),
        max_expansions: None,
        boost: None,
      }),
      &default_fields,
    )
    .unwrap();
    let (_, question_groups) = expand_term_groups(
      &reader.segments,
      &question_plan.term_groups,
      None,
      &reader.analysis,
      &reader.manifest.schema,
    )
    .unwrap();
    let question_keys: HashSet<_> = question_groups[0].keys.iter().cloned().collect();
    assert!(question_keys.contains("body:roast"));
    assert!(question_keys.contains("body:roost"));
    assert_eq!(question_keys.len(), 2);
  }

  #[test]
  fn regex_expansion_applies_cap() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("idx-regex");
    let idx = Index::create(
      &path,
      Schema::default_text_body(),
      IndexOptions {
        path: path.clone(),
        create_if_missing: true,
        enable_positions: true,
        bm25_k1: 0.9,
        bm25_b: 0.4,
        storage: StorageType::Filesystem,
        #[cfg(feature = "vectors")]
        vector_defaults: None,
      },
    )
    .unwrap();
    let mut writer = idx.writer().unwrap();
    for (id, body) in [("1", "rust"), ("2", "ruby"), ("3", "rope")] {
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!(id)),
            ("body".into(), serde_json::json!(body)),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
    }
    writer.commit().unwrap();
    let reader = idx.reader().unwrap();
    let default_fields: Vec<String> = reader
      .manifest
      .schema
      .text_fields
      .iter()
      .map(|f| f.name.clone())
      .collect();
    let plan = build_query_plan(
      &Query::Node(QueryNode::Regex {
        field: "body".into(),
        value: "r(ust|uby)".into(),
        max_expansions: Some(1),
        boost: None,
      }),
      &default_fields,
    )
    .unwrap();
    let (_, groups) = expand_term_groups(
      &reader.segments,
      &plan.term_groups,
      None,
      &reader.analysis,
      &reader.manifest.schema,
    )
    .unwrap();
    let keys = groups[0].keys.to_vec();
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0], "body:ruby");
  }

  #[test]
  fn completion_suggest_prefers_higher_doc_freq() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("idx-suggest");
    let idx = Index::create(
      &path,
      Schema::default_text_body(),
      IndexOptions {
        path: path.clone(),
        create_if_missing: true,
        enable_positions: true,
        bm25_k1: 0.9,
        bm25_b: 0.4,
        storage: StorageType::Filesystem,
        #[cfg(feature = "vectors")]
        vector_defaults: None,
      },
    )
    .unwrap();
    let mut writer = idx.writer().unwrap();
    for (id, body) in [("1", "rust"), ("2", "rust"), ("3", "ruby")] {
      writer
        .add_document(&Document {
          fields: [
            ("_id".into(), serde_json::json!(id)),
            ("body".into(), serde_json::json!(body)),
          ]
          .into_iter()
          .collect(),
        })
        .unwrap();
    }
    writer.commit().unwrap();
    let reader = idx.reader().unwrap();
    let options = reader
      .completion_suggest("body", "ru", 1, None)
      .expect("completion suggest");
    assert_eq!(options.len(), 1);
    assert_eq!(options[0].text, "rust");
    assert_eq!(options[0].doc_freq, 2);
  }
}
