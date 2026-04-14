use hashbrown::{HashMap, HashSet};
use std::cell::RefCell;
use std::collections::{BTreeMap, BinaryHeap};
use std::sync::Arc;
use std::sync::OnceLock;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::api::types::{
  Aggregation, AggregationResponse, AggregationSampling, DateHistogramAggregation, Filter,
  HistogramAggregation, IndexOptions, MgetDoc, Query, RescoreMode, RescoreRequest, SearchRequest,
  SortOrder, SuggestResult,
};
#[cfg(feature = "vectors")]
use crate::api::types::{LegacyVectorQuery, VectorQuery, VectorQuerySpec};
use crate::api::AggregationError;
use crate::index::fastfields::{doc_length_key, FastFieldsReader};
use crate::index::manifest::{
  FieldKind, Manifest, NestedField, NestedProperty, Schema, SchemaAnalyzers,
};
use crate::index::postings::PostingsReader;
use crate::index::segment::SegmentReader;
use crate::index::InnerIndex;
use crate::query::aggregation::AggregationPipeline;
use crate::query::aggs::{parse_calendar_interval, parse_date, parse_interval_seconds};
use crate::query::collector::{AggregationSegmentCollector, DocCollector};
use crate::query::filters::passes_filter;
use crate::query::planner::{build_query_plan, QueryMatcher, ScorePlan};
use crate::query::sort::{SortKey, SortPlan};
use crate::query::wand::{
  execute_top_k_with_stats_and_mode_internal, score_tf, QueryStats, ScoreAdjustFn, ScoreMode,
  ScoredTerm,
};
use crate::util::path_scope::resolve_optional_scoped_path;
#[cfg(feature = "vectors")]
use crate::vectors::hnsw::DEFAULT_EF_SEARCH;
#[cfg(feature = "vectors")]
use crate::vectors::{blend_scores, normalize_in_place, DEFAULT_VECTOR_ALPHA};
use crate::DocId;

use super::pagination::{
  decode_cursor, decode_search_after_token, encode_cursor, encode_search_after_token, CursorState,
};
use super::phrase::{
  build_phrase_runtimes, build_phrase_term_map, build_term_doc_lists, expand_phrase_fields,
  PhraseFieldConfig, PhraseRuntime, TermMatchGroup,
};

pub(crate) const MAX_CURSOR_ADVANCE: usize = 50_000;
const MAX_PAGE_SIZE: usize = 1_000;
const MAX_MGET_IDS: usize = 1_024;

#[cfg(not(target_arch = "wasm32"))]
fn monotonic_now_ms() -> f64 {
  static START: OnceLock<Instant> = OnceLock::new();
  START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
}

#[cfg(target_arch = "wasm32")]
fn monotonic_now_ms() -> f64 {
  js_sys::Date::now()
}

fn elapsed_ms_since(start_ms: f64) -> f64 {
  let elapsed = monotonic_now_ms() - start_ms;
  if elapsed.is_sign_negative() {
    0.0
  } else {
    elapsed
  }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hit {
  pub doc_id: String,
  pub score: f32,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub vector_score: Option<f32>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub sort_key: Option<Vec<serde_json::Value>>,
  pub fields: Option<serde_json::Value>,
  pub snippet: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub explanation: Option<HitExplanation>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub highlights: Option<BTreeMap<String, Vec<String>>>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub inner_hits: Option<Vec<Hit>>,
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
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub total_groups: Option<u64>,
  pub hits: Vec<Hit>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub next_cursor: Option<String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub next_search_after: Option<Vec<serde_json::Value>>,
  #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
  pub aggregations: BTreeMap<String, AggregationResponse>,
  #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
  pub suggest: BTreeMap<String, SuggestResult>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub profile: Option<ProfileResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSearchResponse {
  pub results: Vec<SearchResult>,
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

const MAX_CANDIDATE_SIZE: usize = 20_000;

#[cfg(feature = "vectors")]
const DEFAULT_MAX_VECTOR_GLOBAL_CANDIDATES: usize = 20_000;
#[cfg(feature = "vectors")]
const HARD_MAX_VECTOR_GLOBAL_CANDIDATES: usize = 100_000;

#[cfg(feature = "vectors")]
const MAX_VECTOR_CLAUSES: usize = 8;
#[cfg(feature = "vectors")]
const MAX_VECTOR_K: usize = 1024;
#[cfg(feature = "vectors")]
const MAX_VECTOR_CANDIDATE_SIZE: usize = 10_000;
#[cfg(feature = "vectors")]
const MAX_VECTOR_EF_SEARCH: usize = 65_536;

#[cfg(feature = "vectors")]
#[derive(Clone)]
struct VectorClausePlan {
  field: String,
  vector: Vec<f32>,
  k: usize,
  alpha: f32,
  ef_search: usize,
  candidate_size: usize,
  boost: f32,
  metric: crate::index::manifest::VectorMetric,
}

#[cfg(feature = "vectors")]
#[derive(Clone)]
struct VectorPlan {
  clauses: Vec<VectorClausePlan>,
  candidate_size: usize,
  vector_only: bool,
}

#[cfg(feature = "vectors")]
fn missing_vector_score(metric: &crate::index::manifest::VectorMetric) -> f32 {
  match metric {
    crate::index::manifest::VectorMetric::Cosine => -1.0,
    crate::index::manifest::VectorMetric::L2 => f32::MIN,
  }
}

#[cfg(feature = "vectors")]
fn compute_hybrid_score(
  key: (u32, DocId),
  bm25_score: f32,
  plan: &VectorPlan,
  vector_scores: &[HashMap<(u32, DocId), f32>],
) -> (f32, Option<f32>, bool) {
  let mut blended_sum = 0.0_f32;
  let mut vector_sum = 0.0_f32;
  let mut has_vector = false;
  for (clause, scores) in plan.clauses.iter().zip(vector_scores.iter()) {
    let raw_vec = scores.get(&key).copied();
    if let Some(vs) = raw_vec {
      vector_sum += vs;
      has_vector = true;
    }
    let vec_score = raw_vec.unwrap_or_else(|| missing_vector_score(&clause.metric));
    let blended = if clause.alpha >= 1.0 {
      bm25_score
    } else if clause.alpha <= 0.0 {
      vec_score
    } else {
      blend_scores(bm25_score, vec_score, clause.alpha, true)
    };
    blended_sum += blended;
  }
  let denom = plan.clauses.len().max(1) as f32;
  let final_score = blended_sum / denom;
  (final_score, has_vector.then_some(vector_sum), has_vector)
}

use super::scoring::{
  compile_score_node, evaluate_compiled_score, has_custom_scoring, CompiledScoreNode,
};
use super::term_expansion::{expand_term_groups, QualifiedTerm, WeightedTermEntry};

#[derive(Clone, Debug)]
pub(crate) struct RankedHit {
  pub(crate) key: SortKey,
  pub(crate) score: f32,
  pub(crate) vector_score: Option<f32>,
  pub(crate) explanation: Option<HitExplanation>,
}

impl PartialEq for RankedHit {
  fn eq(&self, other: &Self) -> bool {
    self.key == other.key
      && self.score.to_bits() == other.score.to_bits()
      && self.vector_score.map(f32::to_bits) == other.vector_score.map(f32::to_bits)
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

#[derive(Clone, Copy)]
enum RootFilter<'a> {
  None,
  Node(&'a Filter),
}

struct SegmentSearchParams<'a> {
  qualified_terms: &'a [QualifiedTerm],
  term_weights: &'a HashMap<String, WeightedTermEntry>,
  field_lengths_cache: &'a mut HashMap<String, CachedFieldLengths>,
  cross_lengths_cache: &'a mut HashMap<String, Arc<Vec<f32>>>,
  cross_avgdl_cache: &'a mut HashMap<String, f32>,
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
  skipped_by_cursor: &'a mut u64,
  req: &'a SearchRequest,
  segment_ord: u32,
  rank_limit: usize,
  cursor_key: Option<SortKey>,
  saw_cursor: &'a mut bool,
  sort_plan: &'a SortPlan,
  collect_hits: Option<&'a mut dyn FnMut(SortKey, f32)>,
  stats: Option<&'a mut QueryStats>,
}

pub(crate) use super::query_eval::QueryEvaluator;

fn passes_root_filter(reader: &FastFieldsReader, doc_id: DocId, root: RootFilter<'_>) -> bool {
  match root {
    RootFilter::None => true,
    RootFilter::Node(filter) => passes_filter(reader, doc_id, filter),
  }
}

pub struct IndexReader {
  pub manifest: Manifest,
  pub segments: Vec<SegmentReader>,
  doc_lookup: OnceLock<HashMap<String, Vec<(usize, DocId)>>>,
  pub(crate) analysis: SchemaAnalyzers,
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
      doc_lookup: OnceLock::new(),
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

  #[cfg(feature = "vectors")]
  fn build_vector_plan(&self, req: &SearchRequest) -> Result<Option<VectorPlan>> {
    use crate::api::types::QueryNode;
    fn collect_vectors(
      node: &QueryNode,
      vectors: &mut Vec<VectorQuery>,
      has_non_vector: &mut bool,
    ) {
      match node {
        QueryNode::Vector(vq) => vectors.push(vq.clone()),
        QueryNode::Bool {
          must,
          should,
          must_not,
          filter,
          ..
        } => {
          if !filter.is_empty() {
            *has_non_vector = true;
          }
          for q in must.iter().chain(should.iter()).chain(must_not.iter()) {
            collect_vectors(q, vectors, has_non_vector);
            if !matches!(q, QueryNode::Vector(_)) {
              *has_non_vector = true;
            }
          }
        }
        QueryNode::DisMax { queries, .. } => {
          for q in queries {
            collect_vectors(q, vectors, has_non_vector);
            if !matches!(q, QueryNode::Vector(_)) {
              *has_non_vector = true;
            }
          }
        }
        QueryNode::FunctionScore { query, .. } => {
          collect_vectors(query, vectors, has_non_vector);
          *has_non_vector = true;
        }
        QueryNode::ScriptScore { query, .. } => {
          collect_vectors(query, vectors, has_non_vector);
          *has_non_vector = true;
        }
        QueryNode::RankFeature { .. } => {
          *has_non_vector = true;
        }
        _ => {
          *has_non_vector = true;
        }
      }
    }
    fn find_vectors(query: &Query) -> (Vec<VectorQuery>, bool) {
      match query {
        Query::Node(node) => {
          let mut vectors = Vec::new();
          let mut has_non_vector = false;
          collect_vectors(node, &mut vectors, &mut has_non_vector);
          (vectors, has_non_vector)
        }
        _ => (Vec::new(), true),
      }
    }
    let (vector_nodes, has_non_vector_nodes) = find_vectors(&req.query);
    if !vector_nodes.is_empty() && req.vector_query.is_some() {
      bail!("cannot set both `vector_query` and a `vector` query node");
    }
    let mut vectors: Vec<VectorQuery> = if !vector_nodes.is_empty() {
      vector_nodes
    } else if let Some(spec) = req.vector_query.as_ref() {
      vec![match spec {
        VectorQuerySpec::Structured(v) => v.clone(),
        VectorQuerySpec::Legacy(LegacyVectorQuery(field, vec, alpha)) => VectorQuery {
          field: field.clone(),
          vector: vec.clone(),
          k: None,
          alpha: Some(*alpha),
          ef_search: None,
          candidate_size: None,
          boost: None,
        },
      }]
    } else {
      return Ok(None);
    };
    if vectors.len() > MAX_VECTOR_CLAUSES {
      bail!(
        "too many vector clauses: got {}, max supported {}",
        vectors.len(),
        MAX_VECTOR_CLAUSES
      );
    }
    let vector_only = !has_non_vector_nodes;
    let mut clauses = Vec::with_capacity(vectors.len());
    let mut max_k = 0usize;
    let mut total_k = 0usize;
    let base_candidate = req
      .candidate_size
      .unwrap_or_else(|| req.limit.max(10).saturating_mul(2))
      .max(req.limit)
      .min(MAX_CANDIDATE_SIZE);
    for vector_query in vectors.drain(..) {
      let schema_field = self
        .manifest
        .schema
        .vector_field(&vector_query.field)
        .ok_or_else(|| anyhow::anyhow!("unknown vector field `{}`", vector_query.field))?;
      if vector_query.vector.len() != schema_field.dim {
        bail!(
          "vector field `{}` expects dimension {}, got {}",
          schema_field.name,
          schema_field.dim,
          vector_query.vector.len()
        );
      }
      let mut query_vec = vector_query.vector.clone();
      if matches!(
        schema_field.metric,
        crate::index::manifest::VectorMetric::Cosine
      ) {
        normalize_in_place(&mut query_vec);
      }
      let alpha = vector_query.alpha.unwrap_or(DEFAULT_VECTOR_ALPHA);
      if !(0.0..=1.0).contains(&alpha) || !alpha.is_finite() {
        bail!("vector alpha must be a finite value between 0 and 1 inclusive");
      }
      if vector_only && query_vec.is_empty() {
        continue;
      }
      let default_k = if req.limit == 0 {
        vector_query.k.unwrap_or(10)
      } else {
        vector_query.k.unwrap_or(req.limit)
      };
      let mut k = default_k.max(1);
      if k > MAX_VECTOR_K {
        k = MAX_VECTOR_K;
      }
      let mut candidate_size = vector_query
        .candidate_size
        .unwrap_or_else(|| k.max(req.limit).max(10).saturating_mul(2));
      if candidate_size < k {
        candidate_size = k;
      }
      candidate_size = candidate_size.min(MAX_VECTOR_CANDIDATE_SIZE);
      let mut ef_search = vector_query
        .ef_search
        .unwrap_or_else(|| DEFAULT_EF_SEARCH.max(candidate_size));
      if ef_search > MAX_VECTOR_EF_SEARCH {
        ef_search = MAX_VECTOR_EF_SEARCH;
      }
      let boost = vector_query.boost.unwrap_or(1.0);
      if boost < 0.0 || !boost.is_finite() {
        bail!("vector boost must be finite and non-negative");
      }
      max_k = max_k.max(k);
      total_k = total_k.saturating_add(k);
      clauses.push(VectorClausePlan {
        field: vector_query.field.clone(),
        vector: query_vec,
        k,
        alpha,
        ef_search,
        candidate_size,
        boost,
        metric: schema_field.metric.clone(),
      });
    }
    if clauses.is_empty() {
      return Ok(None);
    }
    let global_cap = req
      .max_global_vector_candidates
      .unwrap_or(DEFAULT_MAX_VECTOR_GLOBAL_CANDIDATES)
      .clamp(1, HARD_MAX_VECTOR_GLOBAL_CANDIDATES);

    if global_cap < clauses.len() {
      bail!(
        "max_global_vector_candidates ({}) must be at least the number of vector clauses ({})",
        global_cap,
        clauses.len()
      );
    }

    let total_candidates: usize = clauses.iter().map(|c| c.candidate_size).sum();
    if total_candidates > global_cap {
      // Distribute the global budget evenly across vector clauses to avoid
      // unbounded candidate expansion when multiple vector queries are present.
      let per_clause_cap = (global_cap / clauses.len()).max(1);
      for clause in clauses.iter_mut() {
        clause.candidate_size = clause.candidate_size.min(per_clause_cap);
      }
      let mut capped_total: usize = clauses.iter().map(|c| c.candidate_size).sum();
      if capped_total > global_cap {
        // Trim any remaining excess while keeping at least one candidate per clause.
        let mut excess = capped_total - global_cap;
        for clause in clauses.iter_mut() {
          if excess == 0 {
            break;
          }
          let reducible = clause.candidate_size.saturating_sub(1);
          if reducible == 0 {
            continue;
          }
          let drop = reducible.min(excess);
          clause.candidate_size -= drop;
          excess -= drop;
        }
        capped_total = clauses.iter().map(|c| c.candidate_size).sum();
        debug_assert!(capped_total <= global_cap);
      }
    }
    let mut candidate_size = base_candidate.max(max_k);
    let available_candidates = clauses
      .iter()
      .map(|c| c.candidate_size)
      .sum::<usize>()
      .max(max_k);
    candidate_size = candidate_size.min(available_candidates).min(global_cap);
    if candidate_size == 0 {
      candidate_size = max_k.max(1);
    }
    Ok(Some(VectorPlan {
      clauses,
      candidate_size,
      vector_only,
    }))
  }

  #[cfg(feature = "vectors")]
  #[allow(clippy::too_many_arguments)]
  fn search_vector_only(
    &self,
    req: &SearchRequest,
    sort_plan: SortPlan,
    manifest_generation: u32,
    cursor_state: Option<CursorState>,
    plan: &VectorPlan,
  ) -> Result<SearchResult> {
    let track_total_hits = req.track_total_hits.unwrap_or(false);
    let score_fast_path = !track_total_hits
      && sort_plan.is_score_only()
      && matches!(sort_plan.primary_order(), Some(SortOrder::Desc));
    let cursor_key = cursor_state.as_ref().map(|c| c.key.clone());
    let cursor_returned = cursor_state
      .as_ref()
      .map(|c| c.returned as usize)
      .unwrap_or(0);
    let from = if cursor_state.is_some() { 0 } else { req.from };
    let page_cap = from.saturating_add(req.limit);
    let collect_hits = req.return_hits && page_cap > 0;
    let heap_limit = if collect_hits {
      plan.candidate_size.max(page_cap).saturating_add(1)
    } else {
      0
    };
    let root_filter = req
      .filter
      .as_ref()
      .map(RootFilter::Node)
      .unwrap_or(RootFilter::None);
    let vector_filter = req.vector_filter.as_ref();
    let mut heap = if collect_hits {
      Some(BinaryHeap::<RankedHit>::new())
    } else {
      None
    };
    let mut agg_results = Vec::new();
    let mut total_matches: u64 = 0;
    let mut skipped_by_cursor: u64 = 0;
    let mut saw_cursor = cursor_state.is_none() || !req.return_hits;
    let mut search_stats = QueryStats::default();
    validate_aggregations(&self.manifest.schema, &req.aggs)?;
    let agg_pipeline = AggregationPipeline::from_request(&req.aggs, &[], &self.manifest.schema);
    let vector_scores = self.collect_vector_maps(
      plan,
      root_filter,
      vector_filter,
      false,
      &[],
      &[],
      &QueryMatcher::MatchAll,
    )?;
    for (segment_ord, seg) in self.segments.iter().enumerate() {
      let mut agg_collector = agg_pipeline
        .as_ref()
        .map(|p| p.for_segment(seg, segment_ord as u32))
        .transpose()?;
      let mut seg_docs: HashSet<DocId> = HashSet::new();
      for scores in vector_scores.iter() {
        for ((seg_idx, doc_id), _) in scores.iter() {
          if *seg_idx == segment_ord as u32 {
            seg_docs.insert(*doc_id);
          }
        }
      }
      for doc_id in seg_docs.into_iter() {
        let key_tuple = (segment_ord as u32, doc_id);
        let (final_score, vector_score, _) =
          compute_hybrid_score(key_tuple, 0.0, plan, &vector_scores);
        let key = if req.return_hits {
          let key = sort_plan.build_key(seg, doc_id, final_score, segment_ord as u32);
          if let Some(cur) = &cursor_key {
            let ord = key.cmp(cur);
            if ord.is_lt() || ord.is_eq() {
              if ord.is_eq() {
                saw_cursor = true;
              }
              skipped_by_cursor += 1;
              continue;
            }
          }
          Some(key)
        } else {
          None
        };
        total_matches += 1;
        if let Some(col) = agg_collector.as_mut() {
          col.collect(doc_id, final_score);
        }
        if req.profile {
          search_stats.candidates_examined += 1;
          search_stats.scored_docs += 1;
        }
        if let (Some(heap_ref), Some(key)) = (heap.as_mut(), key) {
          let hit = RankedHit {
            key,
            score: final_score,
            vector_score,
            explanation: None,
          };
          if heap_limit == 0 {
            heap_ref.push(hit);
          } else {
            push_ranked(heap_ref, hit, heap_limit);
          }
        }
      }
      if let Some(collector) = agg_collector {
        agg_results.push(collector.finish());
      }
    }
    if !saw_cursor {
      bail!("stale or invalid cursor for this result set");
    }
    let mut hits: Vec<RankedHit> = heap
      .map(|h| h.into_iter().collect())
      .unwrap_or_else(Vec::new);
    if req.return_hits {
      hits.sort_by(|a, b| a.key.cmp(&b.key));
    } else {
      hits.clear();
    }
    let search_after_mode = req.search_after.is_some() && req.cursor.is_none();
    let total_hits_value = total_matches
      .saturating_add(cursor_returned as u64)
      .saturating_add(if search_after_mode {
        skipped_by_cursor
      } else {
        0
      });
    let mut total_groups = None;
    let mut group_inner_hits: Vec<Vec<RankedHit>> = Vec::new();
    if req.return_hits {
      if let Some(collapse) = req.collapse.as_ref() {
        let groups = self.collapse_hits(hits, collapse, &sort_plan)?;
        total_groups = Some(groups.len() as u64);
        group_inner_hits = groups.iter().map(|(_, inner)| inner.clone()).collect();
        hits = groups.into_iter().map(|(top, _)| top).collect();
      }
    }
    let mut next_cursor = None;
    let mut next_search_after = None;
    let empty_phrases: BTreeMap<String, Vec<Vec<String>>> = BTreeMap::new();
    let hits: Vec<Hit> = if req.return_hits {
      let total_needed = from.saturating_add(req.limit);
      let has_more = total_needed > 0 && hits.len() > total_needed;
      if has_more {
        let key = hits[total_needed - 1].key.clone();
        if !search_after_mode {
          next_cursor = Some(encode_cursor(
            manifest_generation,
            (cursor_returned + total_needed) as u32,
            &key,
            &sort_plan,
            score_fast_path,
          )?);
        }
      }
      let mut last_returned_key: Option<SortKey> = None;
      let return_sort_keys = req.search_after.is_some() || !req.sort.is_empty();
      let inner_sort_plan = if return_sort_keys {
        if let Some(collapse) = req.collapse.as_ref() {
          if let Some(cfg) = collapse.inner_hits.as_ref() {
            Some(
              SortPlan::from_request(&self.manifest.schema, &cfg.sort).with_context(|| {
                format!("invalid inner_hits sort for collapse on {}", collapse.field)
              })?,
            )
          } else {
            None
          }
        } else {
          None
        }
      } else {
        None
      };
      let out: Vec<Hit> = hits
        .into_iter()
        .enumerate()
        .skip(from)
        .take(req.limit)
        .filter_map(|(idx, h)| {
          last_returned_key = Some(h.key.clone());
          let sort_json = if return_sort_keys {
            encode_search_after_token(&sort_plan, &h.key, &self.segments).ok()
          } else {
            None
          };
          let mut hit = self.materialize_hit(
            h,
            req,
            &[],
            &empty_phrases,
            &sort_plan,
            return_sort_keys,
            sort_json.clone(),
          )?;
          if let Some(inner) = group_inner_hits.get(idx) {
            let inner_hits: Vec<Hit> = inner
              .iter()
              .filter_map(|ih| {
                let inner_sort = if return_sort_keys {
                  if let Some(inner_plan) = inner_sort_plan.as_ref() {
                    let seg = self.segments.get(ih.key.segment_ord as usize)?;
                    let inner_key =
                      inner_plan.build_key(seg, ih.key.doc_id, ih.score, ih.key.segment_ord);
                    encode_search_after_token(inner_plan, &inner_key, &self.segments).ok()
                  } else {
                    encode_search_after_token(&sort_plan, &ih.key, &self.segments).ok()
                  }
                } else {
                  None
                };
                self.materialize_hit(
                  ih.clone(),
                  req,
                  &[],
                  &empty_phrases,
                  &sort_plan,
                  return_sort_keys,
                  inner_sort,
                )
              })
              .collect();
            if !inner_hits.is_empty() {
              hit.inner_hits = Some(inner_hits);
            }
          }
          Some(hit)
        })
        .collect();
      if has_more {
        if let Some(key) = last_returned_key.as_ref() {
          next_search_after = encode_search_after_token(&sort_plan, key, &self.segments).ok();
        }
      }
      out
    } else {
      Vec::new()
    };
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
      total_hits_estimate: total_hits_value,
      total_groups,
      hits,
      next_cursor,
      next_search_after,
      aggregations,
      suggest,
      profile: if req.profile {
        Some(ProfileResult {
          execution: to_execution_profile(&search_stats),
          rescore: None,
          timings: BTreeMap::new(),
        })
      } else {
        None
      },
    })
  }

  #[cfg(feature = "vectors")]
  #[allow(clippy::too_many_arguments)]
  fn collect_vector_maps(
    &self,
    plan: &VectorPlan,
    root_filter: RootFilter<'_>,
    vector_filter: Option<&Filter>,
    require_text_match: bool,
    term_groups: &[TermMatchGroup],
    phrase_fields: &[PhraseFieldConfig],
    matcher: &QueryMatcher,
  ) -> Result<Vec<HashMap<(u32, DocId), f32>>> {
    #[derive(Clone, Copy)]
    struct VectorCandidate {
      segment_ord: u32,
      doc_id: DocId,
      score: f32,
    }
    let mut per_clause: Vec<Vec<VectorCandidate>> =
      plan.clauses.iter().map(|_| Vec::new()).collect();
    for (segment_ord, seg) in self.segments.iter().enumerate() {
      let mut pending: Vec<(usize, VectorCandidate)> = Vec::new();
      for (idx, clause) in plan.clauses.iter().enumerate() {
        let Some((index, _store)) = seg.vector_components(&clause.field) else {
          continue;
        };
        let available = index.len();
        if available == 0 {
          continue;
        }
        let search_k = clause.candidate_size.max(clause.k).min(available.max(1));
        let candidates = index.search(&clause.vector, search_k, clause.ef_search);
        for (doc_id, mut vscore) in candidates.into_iter() {
          if seg.is_deleted(doc_id) {
            continue;
          }
          if !passes_root_filter(seg.fast_fields(), doc_id, root_filter) {
            continue;
          }
          if let Some(filt) = vector_filter {
            if !passes_filter(seg.fast_fields(), doc_id, filt) {
              continue;
            }
          }
          vscore *= clause.boost;
          let cand = VectorCandidate {
            segment_ord: segment_ord as u32,
            doc_id,
            score: vscore,
          };
          if require_text_match {
            pending.push((idx, cand));
          } else {
            per_clause[idx].push(cand);
          }
        }
      }
      if require_text_match && !pending.is_empty() {
        let term_doc_lists = build_term_doc_lists(seg, term_groups);
        let phrase_postings: Vec<PhraseRuntime> = build_phrase_runtimes(seg, phrase_fields);
        let query_eval = QueryEvaluator {
          matcher,
          term_docs: &term_doc_lists.lists,
          term_group_lists: &term_doc_lists.group_lists,
          phrase_postings: &phrase_postings,
          fast_fields: seg.fast_fields(),
        };
        for (idx, cand) in pending.into_iter() {
          if query_eval.matches(cand.doc_id) {
            per_clause[idx].push(cand);
          }
        }
      }
    }
    let mut out = Vec::with_capacity(plan.clauses.len());
    for (idx, mut candidates) in per_clause.into_iter().enumerate() {
      candidates.sort_by(|a, b| {
        b.score
          .total_cmp(&a.score)
          .then_with(|| a.segment_ord.cmp(&b.segment_ord))
          .then_with(|| a.doc_id.cmp(&b.doc_id))
      });
      let max_candidates = plan.clauses.get(idx).map(|c| c.candidate_size).unwrap_or(0);
      if max_candidates > 0 && candidates.len() > max_candidates {
        candidates.truncate(max_candidates);
      }
      let mut map = HashMap::with_capacity(candidates.len());
      for cand in candidates.into_iter() {
        map.insert((cand.segment_ord, cand.doc_id), cand.score);
      }
      out.push(map);
    }
    Ok(out)
  }

  #[cfg(feature = "vectors")]
  #[allow(clippy::too_many_arguments)]
  fn merge_vector_hits(
    &self,
    hits: Vec<RankedHit>,
    vector_scores: &[HashMap<(u32, DocId), f32>],
    plan: &VectorPlan,
    sort_plan: &SortPlan,
    cursor_key: Option<&SortKey>,
    saw_cursor: &mut bool,
    skipped_by_cursor: &mut u64,
    heap_limit: usize,
  ) -> Result<Vec<RankedHit>> {
    let mut heap = BinaryHeap::new();
    let mut bm25_map: HashMap<(u32, DocId), RankedHit> = HashMap::new();
    for hit in hits.into_iter() {
      bm25_map.insert((hit.key.segment_ord, hit.key.doc_id), hit);
    }
    let mut candidate_keys: HashSet<(u32, DocId)> =
      bm25_map.keys().copied().collect::<HashSet<_>>();
    for map in vector_scores.iter() {
      candidate_keys.extend(map.keys().copied());
    }
    let all_vector_only = plan.clauses.iter().all(|c| c.alpha <= 0.0);
    for (seg_ord, doc_id) in candidate_keys.into_iter() {
      let mut bm25_score = 0.0_f32;
      let mut explanation = None;
      if let Some(existing) = bm25_map.remove(&(seg_ord, doc_id)) {
        bm25_score = existing.score;
        explanation = existing.explanation;
      }
      let (final_score, vector_score, has_vector) =
        compute_hybrid_score((seg_ord, doc_id), bm25_score, plan, vector_scores);
      if all_vector_only && !has_vector {
        continue;
      }
      if let Some(expl) = explanation.as_mut() {
        expl.final_score = final_score;
      }
      let seg = self
        .segments
        .get(seg_ord as usize)
        .ok_or_else(|| anyhow::anyhow!("missing segment {seg_ord}"))?;
      let key = sort_plan.build_key(seg, doc_id, final_score, seg_ord);
      if let Some(cur) = cursor_key {
        let ord = key.cmp(cur);
        if ord.is_lt() || ord.is_eq() {
          if ord.is_eq() {
            *saw_cursor = true;
          }
          *skipped_by_cursor += 1;
          continue;
        }
      }
      let ranked = RankedHit {
        key,
        score: final_score,
        vector_score,
        explanation,
      };
      if heap_limit == 0 {
        heap.push(ranked);
      } else {
        push_ranked(&mut heap, ranked, heap_limit);
      }
    }
    Ok(heap.into_iter().collect())
  }

  pub fn search(&self, req: &SearchRequest) -> Result<SearchResult> {
    if req.limit == 0 && req.cursor.is_some() {
      bail!("cursor is not supported when limit is 0");
    }
    if req.limit == 0 && req.from > 0 {
      bail!("from is not supported when limit is 0");
    }
    if !req.return_hits && req.search_after.is_some() {
      bail!("search_after is not supported when return_hits is false");
    }
    if req.limit == 0 && req.explain {
      bail!("explain is not supported when limit is 0");
    }
    if !req.return_hits && req.cursor.is_some() {
      bail!("cursor is not supported when return_hits is false");
    }
    if req.cursor.is_some() && req.search_after.is_some() {
      bail!("cursor cannot be combined with search_after; use one pagination method");
    }
    if req.cursor.is_some() && req.from > 0 {
      bail!("from must be 0 when using cursor pagination");
    }
    // Precedence: cursor wins; ignore search_after if both provided.
    if req.search_after.is_some() && req.from > 0 {
      bail!("search_after cannot be combined with from; use one pagination method");
    }
    if let Some(collapse) = req.collapse.as_ref() {
      ensure_keyword_fast(&self.manifest.schema, &collapse.field, "collapse", None)?;
    }
    let sort_plan = SortPlan::from_request(&self.manifest.schema, &req.sort)?;
    let track_total_hits = req.track_total_hits.unwrap_or(false);
    let score_fast_path = !track_total_hits
      && sort_plan.is_score_only()
      && matches!(sort_plan.primary_order(), Some(SortOrder::Desc));
    let manifest_generation = self
      .manifest
      .segments
      .iter()
      .map(|s| s.generation)
      .max()
      .unwrap_or(0);
    let mut from = req.from;
    let cursor_state = if req.limit == 0 {
      None
    } else if let Some(raw) = req.cursor.as_deref() {
      from = 0;
      Some(decode_cursor(
        raw,
        manifest_generation,
        &sort_plan,
        score_fast_path,
      )?)
    } else if let Some(token) = req.search_after.as_ref() {
      let key = decode_search_after_token(token, &sort_plan, &self.segments, self.doc_lookup())?;
      Some(CursorState { key, returned: 0 })
    } else {
      None
    };
    let cursor_key = cursor_state.as_ref().map(|c| c.key.clone());
    let cursor_returned = cursor_state
      .as_ref()
      .map(|c| c.returned as usize)
      .unwrap_or(0);
    let page_cap = from.saturating_add(req.limit);
    if req.return_hits && page_cap > MAX_PAGE_SIZE {
      bail!("from + size exceeds max page size {MAX_PAGE_SIZE}; adjust pagination");
    }
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
    #[cfg(feature = "vectors")]
    let mut vector_plan = self.build_vector_plan(req)?;
    #[cfg(feature = "vectors")]
    if let Some(plan) = vector_plan.as_ref() {
      if !plan.vector_only && plan.clauses.iter().all(|c| c.alpha >= 1.0) {
        vector_plan = None;
      }
    }
    let page_limit = if req.return_hits {
      from.saturating_add(req.limit)
    } else {
      0
    };
    let base_candidate = if page_limit == 0 {
      0
    } else {
      req
        .candidate_size
        .unwrap_or(page_limit)
        .max(page_limit)
        .min(MAX_CANDIDATE_SIZE)
    };
    #[cfg(feature = "vectors")]
    let effective_limit = if page_limit == 0 {
      0
    } else {
      vector_plan
        .as_ref()
        .map(|p| p.candidate_size.max(page_limit))
        .unwrap_or(base_candidate)
    };
    #[cfg(not(feature = "vectors"))]
    let effective_limit = base_candidate;
    let top_k = if !req.return_hits || effective_limit == 0 {
      0
    } else {
      effective_limit.saturating_add(1)
    };
    #[cfg(feature = "vectors")]
    if let Some(plan) = vector_plan.as_ref() {
      if plan.vector_only {
        return self.search_vector_only(req, sort_plan, manifest_generation, cursor_state, plan);
      }
    }
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
    let highlight_phrases = build_phrase_term_map(&query_plan.phrase_specs);
    let root_filter = req
      .filter
      .as_ref()
      .map(RootFilter::Node)
      .unwrap_or(RootFilter::None);

    let mut hits: Vec<RankedHit> = Vec::new();
    let mut heap = std::collections::BinaryHeap::<RankedHit>::new();
    let mut agg_results = Vec::new();
    let mut total_matches: u64 = 0;
    let mut skipped_by_cursor: u64 = 0;
    let mut saw_cursor = cursor_state.is_none() || !req.return_hits;
    let search_start_ms = req.profile.then(monotonic_now_ms);
    let mut timings: BTreeMap<String, f64> = BTreeMap::new();
    let mut search_stats = QueryStats::default();
    validate_aggregations(&self.manifest.schema, &req.aggs)?;
    let agg_pipeline =
      AggregationPipeline::from_request(&req.aggs, &highlight_terms, &self.manifest.schema);
    // Pre-build term_weights once (same for every segment) and allocate reusable caches.
    let mut term_weights: HashMap<String, WeightedTermEntry> = HashMap::new();
    for term in qualified_terms.iter() {
      let entry = term_weights.entry(term.key.clone()).or_insert((
        term.field.clone(),
        0.0,
        term.leaf,
        term.group_fields.clone(),
      ));
      entry.1 += term.weight;
      if entry.3.is_none() && term.group_fields.is_some() {
        entry.3 = term.group_fields.clone();
      }
    }
    let mut field_lengths_cache: HashMap<String, CachedFieldLengths> = HashMap::new();
    let mut cross_lengths_cache: HashMap<String, Arc<Vec<f32>>> = HashMap::new();
    let mut cross_avgdl_cache: HashMap<String, f32> = HashMap::new();
    for (segment_ord, seg) in self.segments.iter().enumerate() {
      let mut agg_collector = agg_pipeline
        .as_ref()
        .map(|p| p.for_segment(seg, segment_ord as u32))
        .transpose()?;
      let mut noop_collector = NoopCollector;
      let mut collect_hits: Option<Box<dyn FnMut(SortKey, f32) + '_>> = None;
      if req.return_hits && !score_fast_path && page_limit > 0 && !req.explain {
        let heap_limit = top_k;
        let heap_ref = &mut heap;
        collect_hits = Some(Box::new(move |key: SortKey, score: f32| {
          push_ranked(
            heap_ref,
            RankedHit {
              key,
              score,
              vector_score: None,
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
        if agg_ref.is_none()
          && (page_limit == 0 || !req.return_hits || (!score_fast_path && page_limit > 0))
        {
          agg_ref = Some(&mut noop_collector);
        }
        let segment_rank_limit = if !req.return_hits {
          0
        } else if score_fast_path {
          top_k
        } else if req.explain {
          seg.live_docs() as usize
        } else {
          0
        };
        // Clear per-segment caches; the allocations are reused across iterations.
        field_lengths_cache.clear();
        cross_lengths_cache.clear();
        cross_avgdl_cache.clear();
        let params = SegmentSearchParams {
          qualified_terms: &qualified_terms,
          term_weights: &term_weights,
          field_lengths_cache: &mut field_lengths_cache,
          cross_lengths_cache: &mut cross_lengths_cache,
          cross_avgdl_cache: &mut cross_avgdl_cache,
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
          match_counter: Some(&mut total_matches),
          skipped_by_cursor: &mut skipped_by_cursor,
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
      if req.return_hits {
        hits.append(&mut seg_hits);
      }
    }

    if !saw_cursor {
      bail!("stale or invalid cursor for this result set");
    }

    if req.return_hits && !score_fast_path {
      hits.extend(heap);
    }
    #[cfg(feature = "vectors")]
    if page_limit > 0 && req.return_hits {
      if let Some(plan) = vector_plan.as_ref() {
        let require_text_match = !plan.vector_only;
        let vector_scores = self.collect_vector_maps(
          plan,
          root_filter,
          req.vector_filter.as_ref(),
          require_text_match,
          &term_groups,
          &phrase_fields,
          &query_plan.matcher,
        )?;
        hits = self.merge_vector_hits(
          hits,
          &vector_scores,
          plan,
          &sort_plan,
          cursor_key.as_ref(),
          &mut saw_cursor,
          &mut skipped_by_cursor,
          top_k,
        )?;
      }
    }
    if req.return_hits {
      hits.sort_by(|a, b| a.key.cmp(&b.key));
    }
    let search_phase_end_ms = if req.profile {
      Some(monotonic_now_ms())
    } else {
      None
    };
    let mut rescore_stats = QueryStats::default();
    if req.return_hits {
      if let Some(rescore_req) = req.rescore.as_ref() {
        let rescore_start_ms = monotonic_now_ms();
        self.rescore_hits(
          &mut hits,
          rescore_req,
          &default_fields,
          &sort_plan,
          req,
          &mut rescore_stats,
        )?;
        if req.profile {
          timings.insert("rescore_ms".to_string(), elapsed_ms_since(rescore_start_ms));
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
    }
    if let Some(start_ms) = search_start_ms {
      let end_ms = search_phase_end_ms.unwrap_or_else(monotonic_now_ms);
      timings.insert("search_ms".to_string(), (end_ms - start_ms).max(0.0));
    }
    let search_after_mode = req.search_after.is_some() && req.cursor.is_none();
    let total_hits_value = total_matches
      .saturating_add(cursor_returned as u64)
      .saturating_add(if search_after_mode {
        skipped_by_cursor
      } else {
        0
      });
    let mut total_groups = None;
    let mut group_inner_hits: Vec<Vec<RankedHit>> = Vec::new();
    if req.return_hits {
      if let Some(collapse) = req.collapse.as_ref() {
        let groups = self.collapse_hits(hits, collapse, &sort_plan)?;
        total_groups = Some(groups.len() as u64);
        group_inner_hits = groups.iter().map(|(_, inner)| inner.clone()).collect();
        hits = groups.into_iter().map(|(top, _)| top).collect();
      }
    }
    let mut next_cursor = None;
    let mut next_search_after = None;
    let hits: Vec<Hit> = if req.return_hits {
      let total_needed = from.saturating_add(req.limit);
      let has_more = total_needed > 0 && hits.len() > total_needed;
      if has_more {
        let last = &hits[total_needed - 1];
        if !search_after_mode {
          let returned = cursor_returned
            .saturating_add(total_needed)
            .try_into()
            .unwrap_or(u32::MAX);
          next_cursor = Some(encode_cursor(
            manifest_generation,
            returned,
            &last.key,
            &sort_plan,
            score_fast_path,
          )?);
        }
      }
      let mut last_returned_key: Option<SortKey> = None;
      let return_sort_keys = req.search_after.is_some() || !req.sort.is_empty();
      let inner_sort_plan = if return_sort_keys {
        if let Some(collapse) = req.collapse.as_ref() {
          if let Some(cfg) = collapse.inner_hits.as_ref() {
            Some(
              SortPlan::from_request(&self.manifest.schema, &cfg.sort).with_context(|| {
                format!("invalid inner_hits sort for collapse on {}", collapse.field)
              })?,
            )
          } else {
            None
          }
        } else {
          None
        }
      } else {
        None
      };
      let out: Vec<Hit> = hits
        .into_iter()
        .enumerate()
        .skip(from)
        .take(req.limit)
        .filter_map(|(idx, h)| {
          last_returned_key = Some(h.key.clone());
          let sort_json = if return_sort_keys {
            encode_search_after_token(&sort_plan, &h.key, &self.segments).ok()
          } else {
            None
          };
          let mut hit = self.materialize_hit(
            h,
            req,
            &highlight_terms,
            &highlight_phrases,
            &sort_plan,
            return_sort_keys,
            sort_json.clone(),
          )?;
          if let Some(inner) = group_inner_hits.get(idx) {
            let inner_hits: Vec<Hit> = inner
              .iter()
              .filter_map(|ih| {
                let inner_sort = if return_sort_keys {
                  if let Some(inner_plan) = inner_sort_plan.as_ref() {
                    let seg = self.segments.get(ih.key.segment_ord as usize)?;
                    let inner_key =
                      inner_plan.build_key(seg, ih.key.doc_id, ih.score, ih.key.segment_ord);
                    encode_search_after_token(inner_plan, &inner_key, &self.segments).ok()
                  } else {
                    encode_search_after_token(&sort_plan, &ih.key, &self.segments).ok()
                  }
                } else {
                  None
                };
                self.materialize_hit(
                  ih.clone(),
                  req,
                  &highlight_terms,
                  &highlight_phrases,
                  &sort_plan,
                  return_sort_keys,
                  inner_sort,
                )
              })
              .collect();
            if !inner_hits.is_empty() {
              hit.inner_hits = Some(inner_hits);
            }
          }
          Some(hit)
        })
        .collect();
      if has_more {
        if let Some(key) = last_returned_key.as_ref() {
          next_search_after = encode_search_after_token(&sort_plan, key, &self.segments).ok();
        }
      }
      out
    } else {
      Vec::new()
    };
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
      total_hits_estimate: total_hits_value,
      total_groups,
      hits,
      next_cursor,
      next_search_after,
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

  pub fn mget(&self, ids: &[String], return_stored: bool) -> Result<Vec<MgetDoc>> {
    if ids.len() > MAX_MGET_IDS {
      bail!(
        "mget ids length {} exceeds max supported {}",
        ids.len(),
        MAX_MGET_IDS
      );
    }
    let doc_lookup = self.doc_lookup();
    let mut results: Vec<MgetDoc> = ids
      .iter()
      .map(|id| MgetDoc {
        doc_id: id.clone(),
        found: false,
        _source: None,
      })
      .collect();
    if results.is_empty() {
      return Ok(results);
    }
    let mut requested: HashMap<&str, Vec<usize>> = HashMap::new();
    for (idx, id) in ids.iter().enumerate() {
      requested.entry(id.as_str()).or_default().push(idx);
    }
    for (doc_id, positions) in requested.iter() {
      let Some(entries) = doc_lookup.get(*doc_id) else {
        continue;
      };
      let mut chosen: Option<(usize, DocId)> = None;
      for (seg_idx, doc_idx) in entries.iter().rev() {
        let seg = self
          .segments
          .get(*seg_idx)
          .ok_or_else(|| anyhow::anyhow!("segment {seg_idx} missing for mget"))?;
        if seg.is_deleted(*doc_idx) {
          continue;
        }
        chosen = Some((*seg_idx, *doc_idx));
        break;
      }
      let Some((seg_idx, doc_idx)) = chosen else {
        continue;
      };
      let seg = self
        .segments
        .get(seg_idx)
        .ok_or_else(|| anyhow::anyhow!("segment {seg_idx} missing for mget"))?;
      let source = if return_stored {
        Some(seg.get_doc(doc_idx)?)
      } else {
        None
      };
      for pos in positions.iter().copied() {
        results[pos].found = true;
        results[pos]._source = source.clone();
      }
    }
    Ok(results)
  }

  fn doc_lookup(&self) -> &HashMap<String, Vec<(usize, DocId)>> {
    self.doc_lookup.get_or_init(|| {
      let mut map = HashMap::new();
      for (seg_idx, seg) in self.segments.iter().enumerate() {
        for (doc_idx, doc_id) in seg.doc_ids().iter().enumerate() {
          if seg.is_deleted(doc_idx as DocId) {
            continue;
          }
          map
            .entry(doc_id.clone())
            .or_insert_with(Vec::new)
            .push((seg_idx, doc_idx as DocId));
        }
      }
      map
    })
  }

  pub fn multi_search(&self, requests: &[SearchRequest]) -> Result<Vec<SearchResult>> {
    let mut out = Vec::with_capacity(requests.len());
    for req in requests.iter() {
      out.push(self.search(req)?);
    }
    Ok(out)
  }

  fn search_segment(
    &self,
    seg: &SegmentReader,
    params: SegmentSearchParams<'_>,
  ) -> Result<Vec<RankedHit>> {
    let SegmentSearchParams {
      qualified_terms,
      term_weights,
      field_lengths_cache,
      cross_lengths_cache,
      cross_avgdl_cache,
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
      skipped_by_cursor,
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
        skipped_by_cursor,
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

    let docs = seg.live_docs() as f32;
    let mut terms: Vec<ScoredTerm> = Vec::new();
    for (key, (field, weight, leaf, group_fields)) in term_weights.iter() {
      if let Some(mut postings) = seg.postings(key) {
        // Scoring only needs doc_id + term_freq; drop position data to
        // free memory on high-frequency terms. Phrase matching loads its
        // own postings separately via build_phrase_runtimes.
        postings.strip_positions();
        let (avgdl, doc_lengths, min_doc_len) = if let Some(fields) = group_fields.as_deref() {
          let (avgdl, dl, mdl) = cross_fields_stats_for(
            field_lengths_cache,
            cross_lengths_cache,
            cross_avgdl_cache,
            fields,
            seg,
          );
          (avgdl, dl, mdl)
        } else {
          let (dl, mdl) = field_lengths_for(field_lengths_cache, field, seg);
          (seg.avg_field_length(field), dl, mdl)
        };
        terms.push(ScoredTerm {
          postings,
          weight: *weight,
          avgdl,
          docs,
          k1: self.options.bm25_k1,
          b: self.options.bm25_b,
          leaf: *leaf,
          doc_lengths,
          min_doc_len,
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
          *skipped_by_cursor += 1;
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
            vector_score: None,
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
    skipped_by_cursor: &mut u64,
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
          *skipped_by_cursor += 1;
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
            vector_score: None,
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
    hits: &mut Vec<RankedHit>,
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
    let mut to_remove: Vec<usize> = Vec::new();
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
      let mut field_lengths_cache: HashMap<String, CachedFieldLengths> = HashMap::new();
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
        if let Some(mut postings) = seg.postings(&key) {
          postings.strip_positions();
          let (doc_lengths, min_doc_len) = field_lengths_for(&mut field_lengths_cache, &field, seg);
          terms.push(ScoredTerm {
            postings,
            weight,
            avgdl: seg.avg_field_length(&field),
            docs: docs_count,
            k1: self.options.bm25_k1,
            b: self.options.bm25_b,
            leaf,
            doc_lengths,
            min_doc_len,
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
            let contribution = score_tf(
              tf,
              df,
              term.doc_len(doc_id),
              term.avgdl,
              term.docs,
              term.k1,
              term.b,
              term.weight,
            );
            if let Some(buf) = leaf_scores.get_mut(term.leaf) {
              *buf += contribution;
            }
          }
        }
        let mut fn_details = Vec::new();
        let rescore_score = match evaluate_compiled_score(
          &compiled_score,
          &query_eval,
          seg.fast_fields(),
          doc_id,
          &leaf_scores,
          req.explain,
          &mut fn_details,
        ) {
          Some(score) => score,
          None => {
            to_remove.push(hit_idx);
            continue;
          }
        };
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
    if !to_remove.is_empty() {
      to_remove.sort_unstable();
      to_remove.dedup();
      for idx in to_remove.into_iter().rev() {
        hits.remove(idx);
      }
    }
    let sort_window = rescore.window_size.min(hits.len());
    if sort_window > 0 {
      hits[..sort_window].sort_by(|a, b| a.key.cmp(&b.key));
    }
    Ok(())
  }
}

fn term_freq_for_doc(postings: &PostingsReader, doc_id: DocId) -> Option<f32> {
  let entries = postings.entries();
  let idx = entries.binary_search_by_key(&doc_id, |e| e.doc_id).ok()?;
  Some(entries.get(idx)?.term_freq as f32)
}

fn cross_fields_cache_key(fields: &[String]) -> String {
  fields.join("\u{1f}")
}

fn cross_fields_stats_for(
  field_lengths_cache: &mut HashMap<String, CachedFieldLengths>,
  cross_lengths_cache: &mut HashMap<String, Arc<Vec<f32>>>,
  cross_avgdl_cache: &mut HashMap<String, f32>,
  fields: &[String],
  seg: &SegmentReader,
) -> (f32, Option<Arc<Vec<f32>>>, Option<f32>) {
  let key = cross_fields_cache_key(fields);
  let avgdl = if let Some(value) = cross_avgdl_cache.get(&key).copied() {
    value
  } else {
    let mut total = 0.0_f32;
    let mut count = 0usize;
    for field in fields.iter() {
      total += seg.avg_field_length(field);
      count += 1;
    }
    let value = if count == 0 {
      1.0
    } else {
      total / count as f32
    };
    cross_avgdl_cache.insert(key.clone(), value);
    value
  };
  if let Some(lengths) = cross_lengths_cache.get(&key) {
    let min_pos = lengths
      .iter()
      .copied()
      .filter(|v| *v > 0.0)
      .reduce(f32::min);
    return (avgdl, Some(lengths.clone()), min_pos);
  }
  let mut combined = vec![0.0_f32; seg.meta.doc_count as usize];
  let mut contributing = 0usize;
  for field in fields.iter() {
    let (lengths, _) = field_lengths_for(field_lengths_cache, field, seg);
    if let Some(lengths) = lengths {
      contributing += 1;
      for (idx, len) in lengths.iter().enumerate().take(combined.len()) {
        combined[idx] += *len;
      }
    }
  }
  if contributing > 1 {
    let denom = contributing as f32;
    for len in combined.iter_mut() {
      *len /= denom;
    }
  }
  let min_pos = combined
    .iter()
    .copied()
    .filter(|v| *v > 0.0)
    .reduce(f32::min);
  let arc = Arc::new(combined);
  cross_lengths_cache.insert(key, arc.clone());
  (avgdl, Some(arc), min_pos)
}

struct CachedFieldLengths {
  lengths: Arc<Vec<f32>>,
  min_positive: Option<f32>,
}

fn field_lengths_for(
  cache: &mut HashMap<String, CachedFieldLengths>,
  field: &str,
  seg: &SegmentReader,
) -> (Option<Arc<Vec<f32>>>, Option<f32>) {
  if let Some(existing) = cache.get(field) {
    return (Some(existing.lengths.clone()), existing.min_positive);
  }
  let key = doc_length_key(field);
  let mut lengths = Vec::with_capacity(seg.meta.doc_count as usize);
  let mut min_positive = f32::INFINITY;
  for doc_id in 0..seg.meta.doc_count {
    let len = seg.fast_fields().i64_value(&key, doc_id).unwrap_or(0) as f32;
    if len > 0.0 {
      min_positive = min_positive.min(len);
    }
    lengths.push(len);
  }
  // When no positive lengths exist, report None so TermState::new uses
  // its own conservative fallback instead of a potentially misleading value.
  let min_opt = if min_positive.is_finite() {
    Some(min_positive)
  } else {
    None
  };
  let arc = Arc::new(lengths);
  cache.insert(
    field.to_string(),
    CachedFieldLengths {
      lengths: arc.clone(),
      min_positive: min_opt,
    },
  );
  (Some(arc), min_opt)
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
  validate_aggregations_in_scope(schema, aggs, None, false)
}

fn validate_aggregations_in_scope(
  schema: &Schema,
  aggs: &BTreeMap<String, Aggregation>,
  scope_path: Option<&str>,
  inside_nested: bool,
) -> Result<()> {
  for (name, agg) in aggs.iter() {
    if inside_nested {
      match agg {
        Aggregation::Terms(_) | Aggregation::Nested(_) => {}
        _ => {
          return Err(
            AggregationError::InvalidConfig {
              reason: format!(
                "aggregation `{name}` of type `{}` is not supported inside nested aggregations",
                aggregation_kind_name(agg)
              ),
            }
            .into(),
          );
        }
      }
    }
    match agg {
      Aggregation::Terms(t) => {
        ensure_keyword_fast(schema, &t.field, name, scope_path)?;
        if let (true, Some(scope_path)) = (inside_nested, scope_path) {
          let resolved_field = resolve_optional_scoped_path(Some(scope_path), &t.field);
          ensure_direct_scoped_child(
            schema,
            scope_path,
            &resolved_field,
            name,
            "field",
            DirectScopedChildKind::LeafField,
          )?;
        }
        validate_sampling(name, &t.sampling)?;
        validate_aggregations_in_scope(schema, &t.aggs, scope_path, inside_nested)?;
      }
      Aggregation::SignificantTerms(t) => {
        ensure_keyword_fast(schema, &t.field, name, scope_path)?;
        validate_sampling(name, &t.sampling)?;
        validate_aggregations_in_scope(schema, &t.aggs, scope_path, inside_nested)?;
      }
      Aggregation::RareTerms(r) => {
        ensure_keyword_fast(schema, &r.field, name, scope_path)?;
        validate_sampling(name, &r.sampling)?;
        validate_aggregations_in_scope(schema, &r.aggs, scope_path, inside_nested)?;
      }
      Aggregation::Range(r) => {
        ensure_numeric_fast(schema, &r.field, name, scope_path)?;
        validate_sampling(name, &r.sampling)?;
        validate_aggregations_in_scope(schema, &r.aggs, scope_path, inside_nested)?;
      }
      Aggregation::DateRange(r) => {
        ensure_numeric_fast(schema, &r.field, name, scope_path)?;
        validate_sampling(name, &r.sampling)?;
        validate_aggregations_in_scope(schema, &r.aggs, scope_path, inside_nested)?;
      }
      Aggregation::Histogram(h) => {
        ensure_numeric_fast(schema, &h.field, name, scope_path)?;
        validate_histogram_config(name, h)?;
        validate_sampling(name, &h.sampling)?;
        validate_aggregations_in_scope(schema, &h.aggs, scope_path, inside_nested)?;
      }
      Aggregation::DateHistogram(h) => {
        ensure_numeric_fast(schema, &h.field, name, scope_path)?;
        validate_date_histogram_config(name, h)?;
        validate_sampling(name, &h.sampling)?;
        validate_aggregations_in_scope(schema, &h.aggs, scope_path, inside_nested)?;
      }
      Aggregation::Stats(m) | Aggregation::ExtendedStats(m) | Aggregation::ValueCount(m) => {
        ensure_numeric_fast(schema, &m.field, name, scope_path)?
      }
      Aggregation::Percentiles(p) => ensure_numeric_fast(schema, &p.field, name, scope_path)?,
      Aggregation::PercentileRanks(p) => ensure_numeric_fast(schema, &p.field, name, scope_path)?,
      Aggregation::Cardinality(c) => {
        ensure_keyword_or_numeric_fast(schema, &c.field, name, scope_path)?
      }
      Aggregation::Filter(f) => {
        validate_sampling(name, &f.sampling)?;
        validate_aggregations_in_scope(schema, &f.aggs, scope_path, inside_nested)?;
      }
      Aggregation::Nested(n) => {
        let nested_path = ensure_nested_path(schema, &n.path, name, scope_path)?;
        if let Some(scope_path) = scope_path {
          ensure_direct_scoped_child(
            schema,
            scope_path,
            &nested_path,
            name,
            "path",
            DirectScopedChildKind::NestedPath,
          )?;
        }
        validate_sampling(name, &n.sampling)?;
        validate_aggregations_in_scope(schema, &n.aggs, Some(&nested_path), true)?;
      }
      Aggregation::Composite(c) => {
        for source in c.sources.iter() {
          match source {
            crate::api::types::CompositeSource::Terms { field, .. } => {
              ensure_keyword_fast(schema, field, name, scope_path)?
            }
            crate::api::types::CompositeSource::Histogram { field, .. } => {
              ensure_numeric_fast(schema, field, name, scope_path)?
            }
          }
        }
        validate_sampling(name, &c.sampling)?;
        validate_aggregations_in_scope(schema, &c.aggs, scope_path, inside_nested)?;
      }
      Aggregation::BucketSort(_)
      | Aggregation::AvgBucket(_)
      | Aggregation::SumBucket(_)
      | Aggregation::Derivative(_)
      | Aggregation::MovingAvg(_)
      | Aggregation::BucketScript(_) => {}
      Aggregation::TopHits(t) => {
        SortPlan::from_request(schema, &t.sort)
          .with_context(|| format!("invalid top_hits sort in aggregation `{name}`"))?;
      }
    }
  }
  Ok(())
}

fn aggregation_kind_name(agg: &Aggregation) -> &'static str {
  match agg {
    Aggregation::Terms(_) => "terms",
    Aggregation::SignificantTerms(_) => "significant_terms",
    Aggregation::RareTerms(_) => "rare_terms",
    Aggregation::Range(_) => "range",
    Aggregation::DateRange(_) => "date_range",
    Aggregation::Histogram(_) => "histogram",
    Aggregation::DateHistogram(_) => "date_histogram",
    Aggregation::Filter(_) => "filter",
    Aggregation::Nested(_) => "nested",
    Aggregation::Composite(_) => "composite",
    Aggregation::Stats(_) => "stats",
    Aggregation::ExtendedStats(_) => "extended_stats",
    Aggregation::ValueCount(_) => "value_count",
    Aggregation::Cardinality(_) => "cardinality",
    Aggregation::Percentiles(_) => "percentiles",
    Aggregation::PercentileRanks(_) => "percentile_ranks",
    Aggregation::TopHits(_) => "top_hits",
    Aggregation::BucketSort(_) => "bucket_sort",
    Aggregation::AvgBucket(_) => "avg_bucket",
    Aggregation::SumBucket(_) => "sum_bucket",
    Aggregation::Derivative(_) => "derivative",
    Aggregation::MovingAvg(_) => "moving_avg",
    Aggregation::BucketScript(_) => "bucket_script",
  }
}

fn nested_path(prefix: Option<&str>, name: &str) -> String {
  if let Some(prefix) = prefix {
    format!("{prefix}.{name}")
  } else {
    name.to_string()
  }
}

fn find_nested_by_path<'a>(schema: &'a Schema, target: &str) -> Option<&'a NestedField> {
  for nested in schema.nested_fields.iter() {
    if let Some(found) = find_nested_by_path_in(nested, None, target) {
      return Some(found);
    }
  }
  None
}

fn find_nested_by_path_in<'a>(
  nested: &'a NestedField,
  prefix: Option<&str>,
  target: &str,
) -> Option<&'a NestedField> {
  let current = nested_path(prefix, &nested.name);
  if current == target {
    return Some(nested);
  }
  for field in nested.fields.iter() {
    if let NestedProperty::Object(obj) = field {
      if let Some(found) = find_nested_by_path_in(obj, Some(&current), target) {
        return Some(found);
      }
    }
  }
  None
}

fn schema_has_nested_path(schema: &Schema, path: &str) -> bool {
  find_nested_by_path(schema, path).is_some()
}

fn schema_has_nested_leaf_field(schema: &Schema, field: &str) -> bool {
  schema
    .nested_fields
    .iter()
    .any(|nested| nested_has_leaf_field(nested, None, field))
}

fn nested_has_leaf_field(nested: &NestedField, prefix: Option<&str>, target: &str) -> bool {
  let current = nested_path(prefix, &nested.name);
  for field in nested.fields.iter() {
    match field {
      NestedProperty::Object(obj) => {
        if nested_has_leaf_field(obj, Some(&current), target) {
          return true;
        }
      }
      NestedProperty::Text(f) => {
        if nested_path(Some(&current), &f.name) == target {
          return true;
        }
      }
      NestedProperty::Keyword(f) => {
        if nested_path(Some(&current), &f.name) == target {
          return true;
        }
      }
      NestedProperty::Numeric(f) => {
        if nested_path(Some(&current), &f.name) == target {
          return true;
        }
      }
    }
  }
  false
}

fn top_level_field_kind_and_fast(schema: &Schema, field: &str) -> Option<(FieldKind, bool)> {
  if schema.text_fields.iter().any(|f| f.name == field) {
    return Some((FieldKind::Text, false));
  }
  if let Some(f) = schema.keyword_fields.iter().find(|f| f.name == field) {
    return Some((FieldKind::Keyword, f.fast));
  }
  if let Some(f) = schema.numeric_fields.iter().find(|f| f.name == field) {
    return Some((FieldKind::Numeric, f.fast));
  }
  None
}

fn ensure_nested_path(
  schema: &Schema,
  path: &str,
  agg: &str,
  scope_path: Option<&str>,
) -> Result<String> {
  let resolved_path = resolve_optional_scoped_path(scope_path, path);
  if schema_has_nested_path(schema, &resolved_path) {
    Ok(resolved_path)
  } else {
    Err(
      AggregationError::UnsupportedFieldType {
        agg: agg.to_string(),
        field: resolved_path,
        expected: "nested path".to_string(),
      }
      .into(),
    )
  }
}

enum DirectScopedChildKind {
  LeafField,
  NestedPath,
}

fn ensure_direct_scoped_child(
  schema: &Schema,
  scope_path: &str,
  resolved_path: &str,
  agg: &str,
  target: &str,
  kind: DirectScopedChildKind,
) -> Result<()> {
  let direct_child = match kind {
    DirectScopedChildKind::LeafField => {
      scope_has_direct_child_leaf_field(schema, scope_path, resolved_path)
    }
    DirectScopedChildKind::NestedPath => {
      scope_has_direct_child_nested_path(schema, scope_path, resolved_path)
    }
  };
  if direct_child {
    Ok(())
  } else {
    Err(
      AggregationError::InvalidConfig {
        reason: format!(
          "aggregation `{agg}` {target} `{resolved_path}` must be a direct child of nested scope `{scope_path}`"
        ),
      }
      .into(),
    )
  }
}

fn scope_has_direct_child_leaf_field(schema: &Schema, scope_path: &str, target_path: &str) -> bool {
  let Some(scope) = find_nested_by_path(schema, scope_path) else {
    return false;
  };
  scope.fields.iter().any(|field| match field {
    NestedProperty::Text(f) => nested_path(Some(scope_path), &f.name) == target_path,
    NestedProperty::Keyword(f) => nested_path(Some(scope_path), &f.name) == target_path,
    NestedProperty::Numeric(f) => nested_path(Some(scope_path), &f.name) == target_path,
    NestedProperty::Object(_) => false,
  })
}

fn scope_has_direct_child_nested_path(
  schema: &Schema,
  scope_path: &str,
  target_path: &str,
) -> bool {
  let Some(scope) = find_nested_by_path(schema, scope_path) else {
    return false;
  };
  scope.fields.iter().any(|field| match field {
    NestedProperty::Object(obj) => nested_path(Some(scope_path), &obj.name) == target_path,
    _ => false,
  })
}

fn ensure_keyword_fast(
  schema: &Schema,
  field: &str,
  agg: &str,
  scope_path: Option<&str>,
) -> Result<()> {
  let resolved = resolve_optional_scoped_path(scope_path, field);
  if scope_path.is_none() {
    if let Some((kind, fast)) = top_level_field_kind_and_fast(schema, &resolved) {
      if matches!(kind, FieldKind::Keyword) {
        if fast {
          return Ok(());
        }
        return Err(
          AggregationError::MissingFastField {
            field: resolved.to_string(),
          }
          .into(),
        );
      }
      return Err(
        AggregationError::UnsupportedFieldType {
          agg: agg.to_string(),
          field: resolved,
          expected: "fast keyword field".to_string(),
        }
        .into(),
      );
    }
    if schema_has_nested_leaf_field(schema, &resolved) {
      return Err(
        AggregationError::UnsupportedFieldType {
          agg: agg.to_string(),
          field: resolved,
          expected: "fast keyword field".to_string(),
        }
        .into(),
      );
    }
  }
  if let Some(def) = schema.field_meta(&resolved) {
    if matches!(def.kind, FieldKind::Keyword) {
      if def.fast {
        return Ok(());
      }
      return Err(
        AggregationError::MissingFastField {
          field: resolved.to_string(),
        }
        .into(),
      );
    }
    return Err(
      AggregationError::UnsupportedFieldType {
        agg: agg.to_string(),
        field: resolved,
        expected: "fast keyword field".to_string(),
      }
      .into(),
    );
  }
  Err(
    AggregationError::UnsupportedFieldType {
      agg: agg.to_string(),
      field: resolved,
      expected: "fast keyword field".to_string(),
    }
    .into(),
  )
}

fn ensure_numeric_fast(
  schema: &Schema,
  field: &str,
  agg: &str,
  scope_path: Option<&str>,
) -> Result<()> {
  let resolved = resolve_optional_scoped_path(scope_path, field);
  if scope_path.is_none() {
    if let Some((kind, fast)) = top_level_field_kind_and_fast(schema, &resolved) {
      if matches!(kind, FieldKind::Numeric) {
        if fast {
          return Ok(());
        }
        return Err(
          AggregationError::MissingFastField {
            field: resolved.to_string(),
          }
          .into(),
        );
      }
      return Err(
        AggregationError::UnsupportedFieldType {
          agg: agg.to_string(),
          field: resolved,
          expected: "fast numeric field".to_string(),
        }
        .into(),
      );
    }
    if schema_has_nested_leaf_field(schema, &resolved) {
      return Err(
        AggregationError::UnsupportedFieldType {
          agg: agg.to_string(),
          field: resolved,
          expected: "fast numeric field".to_string(),
        }
        .into(),
      );
    }
  }
  if let Some(def) = schema.field_meta(&resolved) {
    if matches!(def.kind, FieldKind::Numeric) {
      if def.fast {
        return Ok(());
      }
      return Err(
        AggregationError::MissingFastField {
          field: resolved.to_string(),
        }
        .into(),
      );
    }
    return Err(
      AggregationError::UnsupportedFieldType {
        agg: agg.to_string(),
        field: resolved,
        expected: "fast numeric field".to_string(),
      }
      .into(),
    );
  }
  Err(
    AggregationError::UnsupportedFieldType {
      agg: agg.to_string(),
      field: resolved,
      expected: "fast numeric field".to_string(),
    }
    .into(),
  )
}

fn ensure_keyword_or_numeric_fast(
  schema: &Schema,
  field: &str,
  agg: &str,
  scope_path: Option<&str>,
) -> Result<()> {
  let resolved = resolve_optional_scoped_path(scope_path, field);
  if scope_path.is_none() {
    if let Some((kind, fast)) = top_level_field_kind_and_fast(schema, &resolved) {
      if matches!(kind, FieldKind::Keyword | FieldKind::Numeric) {
        if fast {
          return Ok(());
        }
        return Err(
          AggregationError::MissingFastField {
            field: resolved.to_string(),
          }
          .into(),
        );
      }
      return Err(
        AggregationError::UnsupportedFieldType {
          agg: agg.to_string(),
          field: resolved,
          expected: "fast keyword or numeric field".to_string(),
        }
        .into(),
      );
    }
    if schema_has_nested_leaf_field(schema, &resolved) {
      return Err(
        AggregationError::UnsupportedFieldType {
          agg: agg.to_string(),
          field: resolved,
          expected: "fast keyword or numeric field".to_string(),
        }
        .into(),
      );
    }
  }
  if let Some(def) = schema.field_meta(&resolved) {
    let kind_ok = matches!(def.kind, FieldKind::Keyword | FieldKind::Numeric);
    if kind_ok && def.fast {
      return Ok(());
    }
    if kind_ok {
      return Err(
        AggregationError::MissingFastField {
          field: resolved.to_string(),
        }
        .into(),
      );
    }
    return Err(
      AggregationError::UnsupportedFieldType {
        agg: agg.to_string(),
        field: resolved,
        expected: "fast keyword or numeric field".to_string(),
      }
      .into(),
    );
  }
  Err(
    AggregationError::UnsupportedFieldType {
      agg: agg.to_string(),
      field: resolved,
      expected: "fast keyword or numeric field".to_string(),
    }
    .into(),
  )
}

fn validate_sampling(name: &str, sampling: &Option<AggregationSampling>) -> Result<()> {
  if let Some(s) = sampling {
    if s.size.is_some() && s.probability.is_some() {
      return Err(
        AggregationError::InvalidConfig {
          reason: format!("aggregation `{name}` sampling cannot set both size and probability"),
        }
        .into(),
      );
    }
    if let Some(prob) = s.probability {
      if !(0.0..=1.0).contains(&prob) {
        return Err(
          AggregationError::InvalidConfig {
            reason: format!("aggregation `{name}` sampling probability must be between 0 and 1"),
          }
          .into(),
        );
      }
    }
    if let Some(size) = s.size {
      if size == 0 {
        return Err(
          AggregationError::InvalidConfig {
            reason: format!("aggregation `{name}` sampling size must be greater than 0"),
          }
          .into(),
        );
      }
    }
    if s.seed.is_some() && s.size.is_none() && s.probability.is_none() {
      return Err(
        AggregationError::InvalidConfig {
          reason: format!(
            "aggregation `{name}` sampling seed requires size or probability to be set"
          ),
        }
        .into(),
      );
    }
  }
  Ok(())
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
    CollapseRequest, ExecutionStrategy, FieldSpec, IndexOptions, InnerHitsRequest, KeywordField,
    MatchOperator, MultiMatchType, NumericField, QueryNode, Schema, SearchRequest, SortOrder,
    SortSpec, TextField,
  };
  use crate::api::{Document, Index, Query, StorageType};
  #[cfg(feature = "vectors")]
  use crate::index::manifest::{VectorField, VectorMetric};
  use crate::query::wand::{execute_top_k_with_stats_and_mode_internal, ScoreMode, ScoredTerm};
  use serde_json::json;
  use std::collections::HashSet;

  #[test]
  fn search_after_round_trips_across_pages() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("idx");
    let schema = Schema::default_text_body();
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
    writer
      .add_document(&Document {
        fields: BTreeMap::from([
          ("_id".into(), json!("doc-1")),
          ("body".into(), json!("rust rust search")),
        ]),
      })
      .unwrap();
    writer
      .add_document(&Document {
        fields: BTreeMap::from([
          ("_id".into(), json!("doc-2")),
          ("body".into(), json!("rust query")),
        ]),
      })
      .unwrap();
    writer.commit().unwrap();

    let reader = idx.reader().unwrap();
    let first = reader
      .search(&SearchRequest {
        query: Query::String("rust".into()),
        fields: None,
        filter: None,
        limit: 1,
        from: 0,
        return_hits: true,
        candidate_size: None,
        #[cfg(feature = "vectors")]
        max_global_vector_candidates: None,
        sort: Vec::new(),
        cursor: None,
        search_after: None,
        execution: ExecutionStrategy::Wand,
        bmw_block_size: None,
        fuzzy: None,
        track_total_hits: None,
        #[cfg(feature = "vectors")]
        vector_query: None,
        #[cfg(feature = "vectors")]
        vector_filter: None,
        return_stored: false,
        highlight_field: None,
        highlight: None,
        collapse: None,
        aggs: BTreeMap::new(),
        suggest: BTreeMap::new(),
        rescore: None,
        explain: false,
        profile: false,
      })
      .unwrap();
    assert_eq!(first.hits.len(), 1);
    let token = first.next_search_after.clone().expect("next_search_after");

    let second = reader
      .search(&SearchRequest {
        query: Query::String("rust".into()),
        fields: None,
        filter: None,
        limit: 1,
        from: 0,
        return_hits: true,
        candidate_size: None,
        #[cfg(feature = "vectors")]
        max_global_vector_candidates: None,
        sort: Vec::new(),
        cursor: None,
        search_after: Some(token),
        execution: ExecutionStrategy::Wand,
        bmw_block_size: None,
        fuzzy: None,
        track_total_hits: None,
        #[cfg(feature = "vectors")]
        vector_query: None,
        #[cfg(feature = "vectors")]
        vector_filter: None,
        return_stored: false,
        highlight_field: None,
        highlight: None,
        collapse: None,
        aggs: BTreeMap::new(),
        suggest: BTreeMap::new(),
        rescore: None,
        explain: false,
        profile: false,
      })
      .unwrap();
    assert_eq!(second.hits.len(), 1);
    assert_ne!(second.hits[0].doc_id, first.hits[0].doc_id);
  }

  #[test]
  fn inner_hits_use_inner_sort_key() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("idx");
    let mut schema = Schema::default_text_body();
    schema.keyword_fields.push(KeywordField {
      name: "author".into(),
      stored: true,
      indexed: true,
      fast: true,
      nullable: false,
    });
    schema.keyword_fields.push(KeywordField {
      name: "title".into(),
      stored: true,
      indexed: true,
      fast: true,
      nullable: false,
    });
    schema.numeric_fields.push(NumericField {
      name: "rank".into(),
      i64: true,
      fast: true,
      stored: true,
      nullable: false,
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
    writer
      .add_document(&Document {
        fields: BTreeMap::from([
          ("_id".into(), json!("a-1")),
          ("body".into(), json!("rust search")),
          ("author".into(), json!("alice")),
          ("title".into(), json!("beta")),
          ("rank".into(), json!(2)),
        ]),
      })
      .unwrap();
    writer
      .add_document(&Document {
        fields: BTreeMap::from([
          ("_id".into(), json!("a-2")),
          ("body".into(), json!("rust systems")),
          ("author".into(), json!("alice")),
          ("title".into(), json!("alpha")),
          ("rank".into(), json!(1)),
        ]),
      })
      .unwrap();
    writer
      .add_document(&Document {
        fields: BTreeMap::from([
          ("_id".into(), json!("b-1")),
          ("body".into(), json!("rust memory")),
          ("author".into(), json!("bob")),
          ("title".into(), json!("gamma")),
          ("rank".into(), json!(3)),
        ]),
      })
      .unwrap();
    writer.commit().unwrap();

    let reader = idx.reader().unwrap();
    let resp = reader
      .search(&SearchRequest {
        query: Query::String("rust".into()),
        fields: None,
        filter: None,
        limit: 10,
        from: 0,
        return_hits: true,
        candidate_size: None,
        #[cfg(feature = "vectors")]
        max_global_vector_candidates: None,
        sort: vec![SortSpec {
          field: "rank".into(),
          order: Some(SortOrder::Asc),
        }],
        cursor: None,
        search_after: None,
        execution: ExecutionStrategy::Wand,
        bmw_block_size: None,
        fuzzy: None,
        track_total_hits: None,
        #[cfg(feature = "vectors")]
        vector_query: None,
        #[cfg(feature = "vectors")]
        vector_filter: None,
        return_stored: true,
        highlight_field: None,
        highlight: None,
        collapse: Some(CollapseRequest {
          field: "author".into(),
          inner_hits: Some(InnerHitsRequest {
            size: Some(2),
            from: Some(0),
            sort: vec![SortSpec {
              field: "title".into(),
              order: Some(SortOrder::Asc),
            }],
          }),
        }),
        aggs: BTreeMap::new(),
        suggest: BTreeMap::new(),
        rescore: None,
        explain: false,
        profile: false,
      })
      .unwrap();

    let alice_group = resp
      .hits
      .iter()
      .find(|hit| hit.inner_hits.as_ref().map(|h| h.len()) == Some(1))
      .expect("expected collapsed group with one inner hit");
    let inner = alice_group.inner_hits.as_ref().expect("inner hits present");
    assert_eq!(inner[0].doc_id, "a-1");
    let sort_key = inner[0].sort_key.as_ref().expect("inner hit sort_key");
    assert_eq!(sort_key.first().expect("sort value"), &json!("beta"));
  }

  #[test]
  fn cursor_rejects_from_offset() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("idx");
    let schema = Schema::default_text_body();
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
    writer
      .add_document(&Document {
        fields: BTreeMap::from([
          ("_id".into(), json!("doc-1")),
          ("body".into(), json!("rust rust search")),
        ]),
      })
      .unwrap();
    writer
      .add_document(&Document {
        fields: BTreeMap::from([
          ("_id".into(), json!("doc-2")),
          ("body".into(), json!("rust query")),
        ]),
      })
      .unwrap();
    writer.commit().unwrap();

    let reader = idx.reader().unwrap();
    let first = reader
      .search(&SearchRequest {
        query: Query::String("rust".into()),
        fields: None,
        filter: None,
        limit: 1,
        from: 0,
        return_hits: true,
        candidate_size: None,
        #[cfg(feature = "vectors")]
        max_global_vector_candidates: None,
        sort: Vec::new(),
        cursor: None,
        search_after: None,
        execution: ExecutionStrategy::Wand,
        bmw_block_size: None,
        fuzzy: None,
        track_total_hits: None,
        #[cfg(feature = "vectors")]
        vector_query: None,
        #[cfg(feature = "vectors")]
        vector_filter: None,
        return_stored: false,
        highlight_field: None,
        highlight: None,
        collapse: None,
        aggs: BTreeMap::new(),
        suggest: BTreeMap::new(),
        rescore: None,
        explain: false,
        profile: false,
      })
      .unwrap();
    let cursor = first.next_cursor.clone().expect("next_cursor");

    let err = reader
      .search(&SearchRequest {
        query: Query::String("rust".into()),
        fields: None,
        filter: None,
        limit: 1,
        from: 1,
        return_hits: true,
        candidate_size: None,
        #[cfg(feature = "vectors")]
        max_global_vector_candidates: None,
        sort: Vec::new(),
        cursor: Some(cursor),
        search_after: None,
        execution: ExecutionStrategy::Wand,
        bmw_block_size: None,
        fuzzy: None,
        track_total_hits: None,
        #[cfg(feature = "vectors")]
        vector_query: None,
        #[cfg(feature = "vectors")]
        vector_filter: None,
        return_stored: false,
        highlight_field: None,
        highlight: None,
        collapse: None,
        aggs: BTreeMap::new(),
        suggest: BTreeMap::new(),
        rescore: None,
        explain: false,
        profile: false,
      })
      .unwrap_err();
    assert!(err
      .to_string()
      .contains("from must be 0 when using cursor pagination"));
  }

  #[test]
  fn cursor_rejects_search_after() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("idx");
    let schema = Schema::default_text_body();
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
    writer
      .add_document(&Document {
        fields: BTreeMap::from([
          ("_id".into(), json!("doc-1")),
          ("body".into(), json!("rust rust search")),
        ]),
      })
      .unwrap();
    writer
      .add_document(&Document {
        fields: BTreeMap::from([
          ("_id".into(), json!("doc-2")),
          ("body".into(), json!("rust query")),
        ]),
      })
      .unwrap();
    writer.commit().unwrap();

    let reader = idx.reader().unwrap();
    let first = reader
      .search(&SearchRequest {
        query: Query::String("rust".into()),
        fields: None,
        filter: None,
        limit: 1,
        from: 0,
        return_hits: true,
        candidate_size: None,
        #[cfg(feature = "vectors")]
        max_global_vector_candidates: None,
        sort: Vec::new(),
        cursor: None,
        search_after: None,
        execution: ExecutionStrategy::Wand,
        bmw_block_size: None,
        fuzzy: None,
        track_total_hits: None,
        #[cfg(feature = "vectors")]
        vector_query: None,
        #[cfg(feature = "vectors")]
        vector_filter: None,
        return_stored: false,
        highlight_field: None,
        highlight: None,
        collapse: None,
        aggs: BTreeMap::new(),
        suggest: BTreeMap::new(),
        rescore: None,
        explain: false,
        profile: false,
      })
      .unwrap();
    let cursor = first.next_cursor.clone().expect("next_cursor");
    let token = first.next_search_after.clone().expect("next_search_after");

    let err = reader
      .search(&SearchRequest {
        query: Query::String("rust".into()),
        fields: None,
        filter: None,
        limit: 1,
        from: 0,
        return_hits: true,
        candidate_size: None,
        #[cfg(feature = "vectors")]
        max_global_vector_candidates: None,
        sort: Vec::new(),
        cursor: Some(cursor),
        search_after: Some(token),
        execution: ExecutionStrategy::Wand,
        bmw_block_size: None,
        fuzzy: None,
        track_total_hits: None,
        #[cfg(feature = "vectors")]
        vector_query: None,
        #[cfg(feature = "vectors")]
        vector_filter: None,
        return_stored: false,
        highlight_field: None,
        highlight: None,
        collapse: None,
        aggs: BTreeMap::new(),
        suggest: BTreeMap::new(),
        rescore: None,
        explain: false,
        profile: false,
      })
      .unwrap_err();
    assert!(err
      .to_string()
      .contains("cursor cannot be combined with search_after"));
  }

  #[test]
  fn mget_skips_deleted_docs_and_preserves_order() {
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
    writer
      .add_document(&Document {
        fields: BTreeMap::from([("_id".into(), json!("a")), ("body".into(), json!("first"))]),
      })
      .unwrap();
    writer
      .add_document(&Document {
        fields: BTreeMap::from([("_id".into(), json!("b")), ("body".into(), json!("second"))]),
      })
      .unwrap();
    writer.commit().unwrap();
    let mut writer = idx.writer().unwrap();
    writer.delete_documents(&["b".into()]).unwrap();
    writer.commit().unwrap();

    let reader = idx.reader().unwrap();
    let docs = reader
      .mget(&["a".into(), "b".into(), "missing".into()], true)
      .unwrap();
    assert_eq!(docs.len(), 3);
    assert!(docs[0].found);
    assert!(docs[0]._source.is_some());
    assert!(!docs[1].found);
    assert!(docs[1]._source.is_none());
    assert!(!docs[2].found);
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
        fuzziness: None,
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
    let mut term_weights: HashMap<String, WeightedTermEntry> = HashMap::new();
    for term in qualified_terms.iter() {
      let entry = term_weights.entry(term.key.clone()).or_insert((
        term.field.clone(),
        0.0,
        term.leaf,
        term.group_fields.clone(),
      ));
      debug_assert_eq!(
        entry.2, term.leaf,
        "Inconsistent leaf for term key {} (expected {}, got {})",
        term.key, entry.2, term.leaf
      );
      entry.1 += term.weight;
    }
    let docs = seg.live_docs() as f32;
    let mut scored_terms = Vec::new();
    for (key, (field, weight, leaf, _group_fields)) in term_weights.into_iter() {
      if let Some(postings) = seg.postings(&key) {
        scored_terms.push(ScoredTerm {
          postings,
          weight,
          avgdl: seg.avg_field_length(&field),
          docs,
          k1: reader.options.bm25_k1,
          b: reader.options.bm25_b,
          leaf,
          doc_lengths: None,
          min_doc_len: None,
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
      "accepted: {seen_matches:?}, ranked: {ranked_ids:?}"
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
          fuzziness: None,
          tie_breaker: None,
          operator: Some(MatchOperator::Or),
          minimum_should_match: None,
          boost: None,
        }),
        fields: None,
        filter: None,
        limit: 5,
        from: 0,
        return_hits: true,
        candidate_size: None,
        #[cfg(feature = "vectors")]
        max_global_vector_candidates: None,
        sort: Vec::new(),
        cursor: None,
        search_after: None,
        execution: ExecutionStrategy::Wand,
        bmw_block_size: None,
        fuzzy: None,
        track_total_hits: None,
        #[cfg(feature = "vectors")]
        vector_query: None,

        #[cfg(feature = "vectors")]
        vector_filter: None,
        return_stored: false,
        highlight_field: None,
        highlight: None,
        collapse: None,
        aggs: BTreeMap::new(),
        suggest: BTreeMap::new(),
        rescore: None,
        explain: false,
        profile: false,
      })
      .unwrap();
    let ids: Vec<_> = res.hits.iter().map(|h| h.doc_id.as_str()).collect();
    assert_eq!(ids.len(), 5, "hits: {ids:?}");
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

  #[cfg(feature = "vectors")]
  #[test]
  fn vector_candidate_size_respects_global_cap() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("idx-vectors");
    let schema = Schema {
      doc_id_field: "_id".into(),
      analyzers: Vec::new(),
      text_fields: Vec::new(),
      keyword_fields: Vec::new(),
      numeric_fields: Vec::new(),
      nested_fields: Vec::new(),
      vector_fields: vec![VectorField {
        name: "vec".into(),
        dim: 4,
        metric: VectorMetric::Cosine,
        hnsw: None,
      }],
    };
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
        vector_defaults: None,
      },
    )
    .unwrap();
    let reader = idx.reader().unwrap();
    let vector_clause = |field: &str| {
      QueryNode::Vector(VectorQuery {
        field: field.to_string(),
        vector: vec![0.1, 0.2, 0.3, 0.4],
        k: Some(10),
        alpha: Some(0.5),
        ef_search: None,
        candidate_size: Some(MAX_VECTOR_CANDIDATE_SIZE),
        boost: None,
      })
    };
    let query = Query::Node(QueryNode::Bool {
      must: Vec::new(),
      should: vec![
        vector_clause("vec"),
        vector_clause("vec"),
        vector_clause("vec"),
        vector_clause("vec"),
        vector_clause("vec"),
        vector_clause("vec"),
        vector_clause("vec"),
        vector_clause("vec"),
      ],
      must_not: Vec::new(),
      filter: Vec::new(),
      minimum_should_match: None,
      boost: None,
    });
    let req = SearchRequest {
      query,
      fields: None,
      filter: None,
      limit: 10,
      from: 0,
      return_hits: true,
      candidate_size: None,
      #[cfg(feature = "vectors")]
      max_global_vector_candidates: None,
      sort: Vec::new(),
      cursor: None,
      search_after: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
      track_total_hits: None,
      vector_query: None,
      vector_filter: None,
      return_stored: false,
      highlight_field: None,
      highlight: None,
      collapse: None,
      aggs: BTreeMap::new(),
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    };
    let plan = reader.build_vector_plan(&req).unwrap().expect("plan");
    let total_candidates: usize = plan.clauses.iter().map(|c| c.candidate_size).sum();
    assert!(
      total_candidates <= DEFAULT_MAX_VECTOR_GLOBAL_CANDIDATES,
      "total vector candidates {total_candidates} exceeds cap {DEFAULT_MAX_VECTOR_GLOBAL_CANDIDATES}"
    );
    assert!(
      plan.candidate_size <= DEFAULT_MAX_VECTOR_GLOBAL_CANDIDATES,
      "plan candidate_size {} exceeds cap {}",
      plan.candidate_size,
      DEFAULT_MAX_VECTOR_GLOBAL_CANDIDATES
    );
  }
}
