use hashbrown::{HashMap, HashSet};
use std::collections::{BTreeMap, BinaryHeap};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use smallvec::smallvec;

use crate::api::query::parse_query;
use crate::api::types::{
  Aggregation, AggregationResponse, DateHistogramAggregation, HistogramAggregation, IndexOptions,
  SearchRequest, SortOrder,
};
use crate::api::AggregationError;
use crate::index::highlight::make_snippet;
use crate::index::manifest::{Manifest, Schema};
use crate::index::postings::PostingEntry;
use crate::index::segment::SegmentReader;
use crate::index::InnerIndex;
use crate::query::aggregation::AggregationPipeline;
use crate::query::aggs::{parse_calendar_interval, parse_date, parse_interval_seconds};
use crate::query::collector::{AggregationSegmentCollector, DocCollector};
use crate::query::filters::passes_filters;
use crate::query::phrase::matches_phrase;
use crate::query::planner::{expand_not_terms, expand_terms};
use crate::query::sort::{SortKey, SortKeyPart, SortPlan, SortValue};
use crate::query::wand::{execute_top_k_with_mode, ScoreMode, ScoredTerm};
use crate::DocId;

const MAX_CURSOR_ADVANCE: usize = 50_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hit {
  pub doc_id: DocId,
  pub score: f32,
  pub fields: Option<serde_json::Value>,
  pub snippet: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
  pub total_hits_estimate: u64,
  pub hits: Vec<Hit>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub next_cursor: Option<String>,
  #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
  pub aggregations: BTreeMap<String, AggregationResponse>,
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

const CURSOR_VERSION: u8 = 1;
const CURSOR_BYTES: usize = 21;
const CURSOR_HEX_LEN: usize = CURSOR_BYTES * 2;
const SORT_CURSOR_VERSION: u8 = 2;

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

struct SegmentSearchParams<'a> {
  qualified_terms: &'a [(String, String, String)],
  qualified_not_terms: &'a [String],
  phrase_fields: &'a [Vec<(String, Vec<String>)>],
  agg_collector: Option<&'a mut dyn DocCollector>,
  match_counter: Option<&'a mut u64>,
  req: &'a SearchRequest,
  segment_ord: u32,
  rank_limit: usize,
  cursor_key: Option<SortKey>,
  saw_cursor: &'a mut bool,
  sort_plan: &'a SortPlan,
  collect_hits: Option<&'a mut dyn FnMut(SortKey, f32)>,
}

pub struct IndexReader {
  pub manifest: Manifest,
  pub segments: Vec<SegmentReader>,
  options: IndexOptions,
}

impl IndexReader {
  pub(crate) fn open(inner: Arc<InnerIndex>) -> Result<Self> {
    let manifest = inner.manifest.read().clone();
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
    })
  }

  pub fn search(&self, req: &SearchRequest) -> Result<SearchResult> {
    if req.limit == 0 && req.cursor.is_some() {
      bail!("cursor is not supported when limit is 0; set a positive limit to page results");
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
    let parsed = parse_query(&req.query);
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
    let term_keys = expand_terms(&parsed, &default_fields);
    let not_terms = expand_not_terms(&parsed, &default_fields);

    let qualified_terms: Vec<(String, String, String)> = term_keys
      .iter()
      .map(|(field, term)| {
        let mut key = String::with_capacity(field.len() + term.len() + 1);
        key.push_str(field);
        key.push(':');
        key.push_str(term);
        (field.clone(), term.clone(), key)
      })
      .collect();
    let qualified_not_terms: Vec<String> = not_terms
      .iter()
      .map(|(field, term)| {
        let mut key = String::with_capacity(field.len() + term.len() + 1);
        key.push_str(field);
        key.push(':');
        key.push_str(term);
        key
      })
      .collect();
    let highlight_terms: Vec<String> = {
      let mut dedup = HashSet::new();
      let mut terms = Vec::new();
      for (_, term, _) in qualified_terms.iter() {
        if dedup.insert(term) {
          terms.push(term.clone());
        }
      }
      terms
    };

    let phrase_fields: Vec<Vec<(String, Vec<String>)>> = parsed
      .phrases
      .iter()
      .map(|phrase| {
        let fields = if let Some(f) = &phrase.field {
          vec![f.clone()]
        } else {
          default_fields.clone()
        };
        fields
          .into_iter()
          .map(|field| {
            let term_keys = phrase
              .terms
              .iter()
              .map(|term| {
                let mut key = String::with_capacity(field.len() + term.len() + 1);
                key.push_str(&field);
                key.push(':');
                key.push_str(term);
                key
              })
              .collect();
            (field, term_keys)
          })
          .collect()
      })
      .collect();

    let mut hits: Vec<RankedHit> = Vec::new();
    let mut heap = std::collections::BinaryHeap::<RankedHit>::new();
    let mut agg_results = Vec::new();
    let mut total_matches: u64 = 0;
    let mut saw_cursor = cursor_state.is_none() || req.limit == 0;
    validate_aggregations(&self.manifest.schema, &req.aggs)?;
    let agg_pipeline = AggregationPipeline::from_request(&req.aggs, &highlight_terms);
    for (segment_ord, seg) in self.segments.iter().enumerate() {
      let mut agg_collector = agg_pipeline
        .as_ref()
        .map(|p| p.for_segment(seg))
        .transpose()?;
      let mut noop_collector = NoopCollector;
      let mut collect_hits: Option<Box<dyn FnMut(SortKey, f32) + '_>> = None;
      if !score_fast_path && req.limit > 0 {
        let heap_limit = top_k;
        let heap_ref = &mut heap;
        collect_hits = Some(Box::new(move |key: SortKey, score: f32| {
          push_ranked(heap_ref, RankedHit { key, score }, heap_limit);
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
        let params = SegmentSearchParams {
          qualified_terms: &qualified_terms,
          qualified_not_terms: &qualified_not_terms,
          phrase_fields: &phrase_fields,
          agg_collector: agg_ref,
          match_counter: counter,
          req,
          segment_ord: segment_ord as u32,
          rank_limit: if score_fast_path { top_k } else { 0 },
          cursor_key: cursor_key.clone(),
          saw_cursor: &mut saw_cursor,
          sort_plan: &sort_plan,
          collect_hits: collect_hits
            .as_mut()
            .map(|f| f as &mut dyn FnMut(SortKey, f32)),
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
    Ok(SearchResult {
      total_hits_estimate: if req.limit == 0 {
        total_matches
      } else {
        hits.len() as u64
      },
      hits,
      next_cursor,
      aggregations,
    })
  }

  fn search_segment(
    &self,
    seg: &SegmentReader,
    params: SegmentSearchParams<'_>,
  ) -> Result<Vec<RankedHit>> {
    let SegmentSearchParams {
      qualified_terms,
      qualified_not_terms,
      phrase_fields,
      agg_collector,
      match_counter,
      req,
      segment_ord,
      rank_limit,
      cursor_key,
      saw_cursor,
      sort_plan,
      collect_hits,
    } = params;
    let score_mode = if sort_plan.uses_score() {
      ScoreMode::Score
    } else {
      ScoreMode::MatchOnly
    };
    let mut term_counts: HashMap<String, (String, u32)> = HashMap::new();
    for (field, _, key) in qualified_terms.iter() {
      let entry = term_counts.entry(key.clone()).or_insert((field.clone(), 0));
      entry.1 += 1;
    }

    let docs = seg.meta.doc_count as f32;
    let mut terms: Vec<ScoredTerm> = Vec::new();
    for (key, (field, weight)) in term_counts.into_iter() {
      if let Some(postings) = seg.postings(&key) {
        terms.push(ScoredTerm {
          postings,
          weight,
          avgdl: seg.avg_field_length(&field),
          docs,
          k1: self.options.bm25_k1,
          b: self.options.bm25_b,
        });
      }
    }
    if terms.is_empty() {
      return Ok(Vec::new());
    }

    let not_doc_lists: Vec<Vec<DocId>> = qualified_not_terms
      .iter()
      .filter_map(|key| {
        seg
          .postings(key)
          .map(|p| p.iter().map(|e| e.doc_id).collect())
      })
      .collect();

    let phrase_postings: Vec<Vec<Vec<Vec<PostingEntry>>>> = phrase_fields
      .iter()
      .map(|fields| {
        fields
          .iter()
          .filter_map(|(_field, term_keys)| {
            let per_term: Vec<Vec<PostingEntry>> = term_keys
              .iter()
              .filter_map(|key| seg.postings(key).map(|p| p.iter().cloned().collect()))
              .collect();
            if per_term.len() == term_keys.len() {
              Some(per_term)
            } else {
              None
            }
          })
          .collect::<Vec<Vec<Vec<PostingEntry>>>>()
      })
      .collect();

    let mut match_counter = match_counter;
    let mut collect_hits = collect_hits;
    let mut accept = |doc_id: DocId, score: f32| -> bool {
      if !passes_filters(seg.fast_fields(), doc_id, &req.filters) {
        return false;
      }
      for list in not_doc_lists.iter() {
        if list.binary_search(&doc_id).is_ok() {
          return false;
        }
      }
      for variants in phrase_postings.iter() {
        if variants.is_empty() {
          return false;
        }
        let mut ok_any_field = false;
        for per_term in variants.iter() {
          if matches_phrase(per_term.as_slice(), doc_id) {
            ok_any_field = true;
            break;
          }
        }
        if !ok_any_field {
          return false;
        }
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

    let ranked = execute_top_k_with_mode(
      terms,
      rank_limit,
      req.execution.clone(),
      req.bmw_block_size,
      &mut accept,
      agg_collector,
      score_mode,
    );

    Ok(
      ranked
        .into_iter()
        .map(|rd| RankedHit {
          key: sort_plan.build_key(seg, rd.doc_id, rd.score, segment_ord),
          score: rd.score,
        })
        .collect(),
    )
  }

  fn materialize_hit(
    &self,
    ranked: RankedHit,
    req: &SearchRequest,
    highlight_terms: &[String],
  ) -> Option<Hit> {
    let seg = self.segments.get(ranked.key.segment_ord as usize)?;
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
      doc_id: ranked.key.doc_id,
      score: ranked.score,
      fields: fields_val,
      snippet,
    })
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
      Aggregation::TopHits(_) => {}
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
}
