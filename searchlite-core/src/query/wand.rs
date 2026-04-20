use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::sync::Arc;

use crate::api::types::ExecutionStrategy;
use crate::index::postings::{PostingsReader, DEFAULT_BLOCK_SIZE};
use crate::query::bm25::bm25;
use crate::query::collector::DocCollector;
use crate::query::planner::ScorePlan;
use crate::DocId;

const DOCID_END: DocId = u32::MAX;

pub(crate) type ScoreAdjustFn<'a> = dyn FnMut(DocId, f32, &[f32]) -> Option<f32> + 'a;

#[derive(Debug, Clone, Copy)]
pub struct RankedDoc {
  pub doc_id: DocId,
  pub score: f32,
}

impl PartialEq for RankedDoc {
  fn eq(&self, other: &Self) -> bool {
    self.doc_id == other.doc_id && self.score.to_bits() == other.score.to_bits()
  }
}

impl Eq for RankedDoc {}

impl Ord for RankedDoc {
  fn cmp(&self, other: &Self) -> Ordering {
    match self.score.total_cmp(&other.score) {
      Ordering::Equal => other.doc_id.cmp(&self.doc_id),
      ord => ord,
    }
  }
}

impl PartialOrd for RankedDoc {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

#[derive(Debug, Default, Clone)]
pub struct QueryStats {
  pub scored_docs: usize,
  pub candidates_examined: usize,
  pub postings_advanced: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreMode {
  Score,
  MatchOnly,
}

impl ScoreMode {
  fn needs_scores(self) -> bool {
    matches!(self, ScoreMode::Score)
  }
}

#[derive(Debug, Clone)]
pub struct ScoredTerm {
  pub postings: PostingsReader,
  pub weight: f32,
  pub avgdl: f32,
  pub docs: f32,
  pub k1: f32,
  pub b: f32,
  pub leaf: usize,
  pub doc_lengths: Option<Arc<Vec<f32>>>,
  /// Precomputed minimum positive document length from `doc_lengths`.
  /// Avoids a full scan of the lengths vector inside `TermState::new`.
  pub min_doc_len: Option<f32>,
}

impl ScoredTerm {
  pub(crate) fn doc_len(&self, doc_id: DocId) -> f32 {
    self
      .doc_lengths
      .as_ref()
      .and_then(|lens| lens.get(doc_id as usize).copied())
      .filter(|v| *v > 0.0)
      .unwrap_or_else(|| self.avgdl.max(1.0))
  }
}

#[derive(Debug, Clone)]
struct TermState {
  postings: PostingsReader,
  idx: usize,
  weight: f32,
  df: f32,
  avgdl: f32,
  docs: f32,
  k1: f32,
  b: f32,
  leaf: usize,
  ub: f32,
  min_doc_len: f32,
  doc_lengths: Option<Arc<Vec<f32>>>,
  block_meta: Arc<crate::index::postings::BlockMeta>,
}

impl TermState {
  fn new(term: ScoredTerm, block_size: usize) -> Self {
    let df = term.postings.len() as f32;
    let clamped_block = block_size.max(1);
    let block_meta = build_block_meta(&term.postings, clamped_block);
    let doc_lengths = term.doc_lengths.clone();
    // The minimum doc length feeds WAND upper-bound calculations. Using a
    // value larger than the true minimum can underestimate bounds and cause
    // incorrect pruning, so we fall back to the most conservative value
    // (1.0) rather than avgdl when no precomputed hint is available.
    let min_doc_len = term
      .min_doc_len
      .filter(|v| v.is_finite() && *v > 0.0)
      .unwrap_or(1.0);
    let ub = upper_bound_tf(
      term.postings.max_tf,
      df,
      min_doc_len,
      term.avgdl,
      term.docs,
      term.k1,
      term.b,
      term.weight,
    );
    Self {
      postings: term.postings,
      idx: 0,
      weight: term.weight,
      df,
      avgdl: term.avgdl,
      docs: term.docs,
      k1: term.k1,
      b: term.b,
      leaf: term.leaf,
      ub,
      min_doc_len,
      doc_lengths,
      block_meta,
    }
  }

  fn is_done(&self) -> bool {
    self.idx >= self.postings.len()
  }

  fn doc_id(&self) -> DocId {
    if let Some(entry) = self.postings.entry(self.idx) {
      entry.doc_id
    } else {
      DOCID_END
    }
  }

  fn doc_len(&self, doc_id: DocId) -> f32 {
    self
      .doc_lengths
      .as_ref()
      .and_then(|lens| lens.get(doc_id as usize).copied())
      .filter(|v| *v > 0.0)
      .unwrap_or_else(|| self.avgdl.max(1.0))
  }

  fn tf(&self) -> f32 {
    self
      .postings
      .entry(self.idx)
      .map(|e| e.term_freq as f32)
      .unwrap_or(0.0)
  }

  fn score_current(&self) -> f32 {
    score_tf(
      self.tf(),
      self.df,
      self.doc_len(self.doc_id()),
      self.avgdl,
      self.docs,
      self.k1,
      self.b,
      self.weight,
    )
  }

  fn advance(&mut self) -> usize {
    if self.is_done() {
      return 0;
    }
    self.idx += 1;
    1
  }

  fn advance_to(&mut self, target: DocId) -> usize {
    if self.is_done() || self.doc_id() >= target {
      return 0;
    }
    let len = self.postings.len();
    let low = self.idx + 1;
    if low >= len {
      let delta = len.saturating_sub(self.idx);
      self.idx = len;
      return delta;
    }
    let mut step = 1usize;
    while low + step < len {
      if let Some(entry) = self.postings.entry(low + step) {
        if entry.doc_id >= target {
          break;
        }
      }
      step <<= 1;
    }
    let upper = (low + step).min(len);
    let slice = &self.postings.entries()[low..upper];
    let advance = slice.partition_point(|p| p.doc_id < target);
    let new_idx = (low + advance).min(len);
    let delta = new_idx.saturating_sub(self.idx);
    self.idx = new_idx;
    delta
  }

  fn block_index(&self) -> usize {
    self.idx / self.block_meta.block_size
  }

  fn block_upper_bound(&self) -> f32 {
    let block_idx = self.block_index();
    let tf = self.block_meta.tfs.get(block_idx).copied().unwrap_or(0.0);
    score_tf(
      tf,
      self.df,
      self.min_doc_len,
      self.avgdl,
      self.docs,
      self.k1,
      self.b,
      self.weight,
    )
  }

  fn upper_bound(&self) -> f32 {
    self.ub
  }

  fn skip_to_block(&mut self, target: DocId) -> usize {
    let prev = self.idx;
    let block_idx = self.block_meta.doc_ids.partition_point(|doc| *doc < target);
    let start = block_idx.saturating_mul(self.block_meta.block_size);
    if start > self.idx {
      self.idx = start.min(self.postings.len());
    }
    self.idx.saturating_sub(prev)
  }

  /// Skip forward through blocks whose per-block max-tf score contribution
  /// is below `min_contribution`. Stops at the first block whose upper bound
  /// meets the threshold, or at the end of the postings list.
  ///
  /// This is the BMW (Block-Max WAND) optimisation described in BUG-005:
  /// during the advancement phase, blocks whose max-tf cannot contribute
  /// enough to push a candidate past the heap threshold are skipped entirely,
  /// avoiding unnecessary `advance_to` work within those blocks.
  fn skip_blocks_below_bound(&mut self, min_contribution: f32) -> usize {
    if min_contribution <= 0.0 {
      return 0;
    }
    let prev = self.idx;
    let mut block_idx = self.block_index();
    while block_idx < self.block_meta.tfs.len() {
      let tf = self.block_meta.tfs[block_idx];
      let bound = score_tf(
        tf,
        self.df,
        self.min_doc_len,
        self.avgdl,
        self.docs,
        self.k1,
        self.b,
        self.weight,
      );
      if bound >= min_contribution {
        break;
      }
      block_idx += 1;
      self.idx = (block_idx * self.block_meta.block_size).min(self.postings.len());
    }
    self.idx.saturating_sub(prev)
  }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn score_tf(
  tf: f32,
  df: f32,
  doc_len: f32,
  avgdl: f32,
  docs: f32,
  k1: f32,
  b: f32,
  weight: f32,
) -> f32 {
  let norm_len = if doc_len > 0.0 {
    doc_len
  } else {
    avgdl.max(tf)
  };
  let base = bm25(tf, df, norm_len, avgdl, docs, k1, b);
  base * weight
}

#[allow(clippy::too_many_arguments)]
fn upper_bound_tf(
  tf: f32,
  df: f32,
  doc_len: f32,
  avgdl: f32,
  docs: f32,
  k1: f32,
  b: f32,
  weight: f32,
) -> f32 {
  if tf <= 0.0 {
    return 0.0;
  }
  score_tf(tf, df, doc_len, avgdl, docs, k1, b, weight)
}

fn build_block_meta(
  postings: &PostingsReader,
  block_size: usize,
) -> Arc<crate::index::postings::BlockMeta> {
  // When the requested block size matches what's already stored, share the
  // existing Arc — no allocation or copying at all.
  if block_size == postings.block_size() && !postings.block_max_doc_ids().is_empty() {
    return postings.block_meta();
  }
  let mut block_max_doc_ids = Vec::new();
  let mut block_max_tfs = Vec::new();
  let mut idx = 0usize;
  while idx < postings.len() {
    let end = (idx + block_size).min(postings.len());
    let mut tf_max = 0.0_f32;
    if let Some(last) = postings.entry(end - 1) {
      block_max_doc_ids.push(last.doc_id);
    }
    for i in idx..end {
      if let Some(entry) = postings.entry(i) {
        tf_max = tf_max.max(entry.term_freq as f32);
      }
    }
    block_max_tfs.push(tf_max);
    idx = end;
  }
  Arc::new(crate::index::postings::BlockMeta {
    doc_ids: block_max_doc_ids,
    tfs: block_max_tfs,
    block_size,
  })
}

fn with_stats(stats: &mut Option<&mut QueryStats>, f: impl FnOnce(&mut QueryStats)) {
  if let Some(s) = stats.as_deref_mut() {
    f(s);
  }
}

pub fn execute_top_k<F: FnMut(DocId, f32) -> bool, C: DocCollector + ?Sized>(
  terms: Vec<ScoredTerm>,
  k: usize,
  strategy: ExecutionStrategy,
  block_size: Option<usize>,
  accept: &mut F,
  collector: Option<&mut C>,
) -> Vec<RankedDoc> {
  execute_top_k_with_stats_and_mode_internal(
    terms,
    k,
    strategy,
    block_size,
    None,
    accept,
    collector,
    None,
    ScoreMode::Score,
    None,
  )
}

pub fn execute_top_k_with_mode<F: FnMut(DocId, f32) -> bool, C: DocCollector + ?Sized>(
  terms: Vec<ScoredTerm>,
  k: usize,
  strategy: ExecutionStrategy,
  block_size: Option<usize>,
  accept: &mut F,
  collector: Option<&mut C>,
  score_mode: ScoreMode,
) -> Vec<RankedDoc> {
  execute_top_k_with_stats_and_mode_internal(
    terms, k, strategy, block_size, None, accept, collector, None, score_mode, None,
  )
}

pub fn execute_top_k_with_stats<F: FnMut(DocId, f32) -> bool, C: DocCollector + ?Sized>(
  terms: Vec<ScoredTerm>,
  k: usize,
  strategy: ExecutionStrategy,
  block_size: Option<usize>,
  accept: &mut F,
  collector: Option<&mut C>,
  stats: Option<&mut QueryStats>,
) -> Vec<RankedDoc> {
  execute_top_k_with_stats_and_mode_internal(
    terms,
    k,
    strategy,
    block_size,
    None,
    accept,
    collector,
    stats,
    ScoreMode::Score,
    None,
  )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_top_k_with_stats_and_mode_internal<
  F: FnMut(DocId, f32) -> bool,
  C: DocCollector + ?Sized,
>(
  terms: Vec<ScoredTerm>,
  k: usize,
  strategy: ExecutionStrategy,
  block_size: Option<usize>,
  score_plan: Option<&ScorePlan>,
  accept: &mut F,
  collector: Option<&mut C>,
  stats: Option<&mut QueryStats>,
  score_mode: ScoreMode,
  score_adjust: Option<&mut ScoreAdjustFn<'_>>,
) -> Vec<RankedDoc> {
  let should_rank = k > 0;
  if terms.is_empty() || (!should_rank && collector.is_none()) {
    return Vec::new();
  }
  if !score_mode.needs_scores() {
    let bsize = block_size.unwrap_or(DEFAULT_BLOCK_SIZE).max(1);
    let states: Vec<TermState> = terms
      .into_iter()
      .filter(|t| t.postings.len() > 0)
      .map(|t| TermState::new(t, bsize))
      .collect();
    return match_only_loop(states, accept, collector, stats);
  }
  if matches!(strategy, ExecutionStrategy::Bm25) {
    return brute_force(
      &terms,
      k,
      should_rank,
      score_plan,
      accept,
      collector,
      stats,
      score_adjust,
    );
  }
  let bsize = block_size.unwrap_or(DEFAULT_BLOCK_SIZE).max(1);
  let states: Vec<TermState> = terms
    .into_iter()
    .filter(|t| t.postings.len() > 0)
    .map(|t| TermState::new(t, bsize))
    .collect();
  let use_block_bounds = matches!(strategy, ExecutionStrategy::Bmw);
  wand_loop(
    states,
    k,
    should_rank,
    use_block_bounds,
    score_plan,
    accept,
    collector,
    stats,
    score_adjust,
  )
}

#[allow(clippy::too_many_arguments)]
fn brute_force<F: FnMut(DocId, f32) -> bool, C: DocCollector + ?Sized>(
  terms: &[ScoredTerm],
  k: usize,
  rank_hits: bool,
  score_plan: Option<&ScorePlan>,
  accept: &mut F,
  mut collector: Option<&mut C>,
  mut stats: Option<&mut QueryStats>,
  mut score_adjust: Option<&mut ScoreAdjustFn<'_>>,
) -> Vec<RankedDoc> {
  if let Some(plan) = score_plan {
    let mut scores: hashbrown::HashMap<DocId, Vec<f32>> = hashbrown::HashMap::new();
    for term in terms.iter() {
      let df = term.postings.len() as f32;
      with_stats(&mut stats, |s| s.postings_advanced += term.postings.len());
      for entry in term.postings.iter() {
        let score = score_tf(
          entry.term_freq as f32,
          df,
          term.doc_len(entry.doc_id),
          term.avgdl,
          term.docs,
          term.k1,
          term.b,
          term.weight,
        );
        // Leaf ids are assigned densely by the planner; dense buffers keep accumulation cache-friendly.
        let buf = scores
          .entry(entry.doc_id)
          .or_insert_with(|| vec![0.0; plan.leaf_count]);
        assert!(
          term.leaf < buf.len(),
          "ScorePlan leaf_count ({}) is less than term leaf index ({})",
          buf.len(),
          term.leaf
        );
        buf[term.leaf] += score;
      }
    }
    let scored = scores.len();
    with_stats(&mut stats, |s| {
      s.scored_docs += scored;
      s.candidates_examined += scored;
    });
    let mut heap: BinaryHeap<Reverse<RankedDoc>> = BinaryHeap::new();
    for (doc_id, leaves) in scores.into_iter() {
      let mut score = plan.evaluate(&leaves);
      if let Some(adj) = score_adjust.as_deref_mut() {
        let Some(adjusted) = adj(doc_id, score, &leaves) else {
          continue;
        };
        score = adjusted;
      }
      // BUG-381: drop documents whose BM25 score is non-finite (typically
      // +inf from an accumulated boost product overflowing `f32::MAX`).
      // Without this guard the pure-BM25 path has no filter analogous to
      // `evaluate_compiled_score`, so a non-finite score reaches the heap
      // and serialises as an invalid JSON number, returning HTTP 500.
      if !score.is_finite() {
        continue;
      }
      if !accept(doc_id, score) {
        continue;
      }
      if let Some(collector) = collector.as_deref_mut() {
        collector.collect(doc_id, score);
      }
      if rank_hits {
        push_top_k(&mut heap, RankedDoc { doc_id, score }, k);
      }
    }
    return finalize_heap(heap);
  }
  let mut scores: hashbrown::HashMap<DocId, f32> = hashbrown::HashMap::new();
  for term in terms.iter() {
    let df = term.postings.len() as f32;
    with_stats(&mut stats, |s| s.postings_advanced += term.postings.len());
    for entry in term.postings.iter() {
      let score = score_tf(
        entry.term_freq as f32,
        df,
        term.doc_len(entry.doc_id),
        term.avgdl,
        term.docs,
        term.k1,
        term.b,
        term.weight,
      );
      *scores.entry(entry.doc_id).or_insert(0.0) += score;
    }
  }
  let scored = scores.len();
  with_stats(&mut stats, |s| {
    s.scored_docs += scored;
    s.candidates_examined += scored;
  });
  let mut heap: BinaryHeap<Reverse<RankedDoc>> = BinaryHeap::new();
  for (doc_id, mut score) in scores.into_iter() {
    if let Some(adj) = score_adjust.as_deref_mut() {
      let Some(adjusted) = adj(doc_id, score, &[]) else {
        continue;
      };
      score = adjusted;
    }
    // BUG-381: see the plan-driven branch above. `score_tf` returns
    // `bm25 * weight`; when accumulated boosts push `weight` past
    // `f32::MAX` the result is `+inf` and leaks into the heap unless we
    // drop it here.
    if !score.is_finite() {
      continue;
    }
    if !accept(doc_id, score) {
      continue;
    }
    if let Some(collector) = collector.as_deref_mut() {
      collector.collect(doc_id, score);
    }
    if rank_hits {
      push_top_k(&mut heap, RankedDoc { doc_id, score }, k);
    }
  }
  finalize_heap(heap)
}

fn match_only_loop<F: FnMut(DocId, f32) -> bool, C: DocCollector + ?Sized>(
  terms: Vec<TermState>,
  accept: &mut F,
  mut collector: Option<&mut C>,
  mut stats: Option<&mut QueryStats>,
) -> Vec<RankedDoc> {
  // Same wrapper as wand_loop
  #[derive(Debug)]
  struct TermWrapper(TermState);
  impl PartialEq for TermWrapper {
    fn eq(&self, other: &Self) -> bool {
      self.0.doc_id() == other.0.doc_id()
    }
  }
  impl Eq for TermWrapper {}
  impl PartialOrd for TermWrapper {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
      Some(self.cmp(other))
    }
  }
  impl Ord for TermWrapper {
    fn cmp(&self, other: &Self) -> Ordering {
      other.0.doc_id().cmp(&self.0.doc_id())
    }
  }

  let mut queue: BinaryHeap<TermWrapper> = terms
    .into_iter()
    .filter(|t| !t.is_done())
    .map(TermWrapper)
    .collect();

  let mut pending: Vec<TermWrapper> = Vec::with_capacity(queue.len());

  loop {
    if queue.is_empty() {
      break;
    }

    // Check head
    if queue.peek().map(|t| t.0.is_done()).unwrap_or(false) {
      queue.pop();
      continue;
    }

    let Some(smallest) = queue.peek() else {
      break;
    };
    let doc = smallest.0.doc_id();

    if doc == DOCID_END {
      break;
    }

    // Collect all terms at this doc_id
    while let Some(top) = queue.peek() {
      if top.0.doc_id() == doc {
        pending.push(queue.pop().unwrap());
      } else {
        break;
      }
    }

    // Advance all terms at this doc
    for wrapper in pending.iter_mut() {
      let moved = wrapper.0.advance();
      with_stats(&mut stats, |s| s.postings_advanced += moved);
    }

    with_stats(&mut stats, |s| {
      s.candidates_examined += 1;
      s.scored_docs += 1;
    });

    if accept(doc, 0.0) {
      if let Some(col) = collector.as_deref_mut() {
        col.collect(doc, 0.0);
      }
    }

    // Re-push terms
    for wrapper in pending.drain(..) {
      if !wrapper.0.is_done() {
        queue.push(wrapper);
      }
    }
  }
  Vec::new()
}

#[allow(clippy::too_many_arguments)]
fn wand_loop<F: FnMut(DocId, f32) -> bool, C: DocCollector + ?Sized>(
  terms: Vec<TermState>,
  k: usize,
  rank_hits: bool,
  use_block_bounds: bool,
  score_plan: Option<&ScorePlan>,
  accept: &mut F,
  mut collector: Option<&mut C>,
  mut stats: Option<&mut QueryStats>,
  mut score_adjust: Option<&mut ScoreAdjustFn<'_>>,
) -> Vec<RankedDoc> {
  let mut heap: BinaryHeap<Reverse<RankedDoc>> = BinaryHeap::new();

  // Use a sorted vector instead of BinaryHeap for the term queue.
  let mut queue: Vec<TermState> = terms.into_iter().filter(|t| !t.is_done()).collect();

  // Initial sort by doc_id
  queue.sort_unstable_by_key(|t| t.doc_id());

  let mut leaf_scores = score_plan.map(|plan| vec![0.0_f32; plan.leaf_count]);
  let mut touched: Vec<usize> = Vec::new();
  let mut touched_flags = leaf_scores.as_ref().map(|buf| vec![false; buf.len()]);
  let mut prune_done = false;

  fn bubble_reposition(queue: &mut [TermState], advanced: usize) {
    if advanced == 0 || queue.len() <= 1 {
      return;
    }
    // The queue is typically 5–20 elements (one per query term).
    // A full sort on such a small slice is faster than a manual merge,
    // and sort_unstable recognises nearly-sorted runs efficiently.
    queue.sort_unstable_by_key(|t| t.doc_id());
  }

  loop {
    if queue.is_empty() {
      break;
    }

    if prune_done {
      queue.retain(|t| !t.is_done());
      prune_done = false;
      if queue.is_empty() {
        break;
      }
    }

    let heap_threshold = if rank_hits && heap.len() >= k {
      heap.peek().map(|d| d.0.score).unwrap_or(0.0)
    } else {
      0.0
    };

    let pivot_threshold = if collector.is_some() {
      f32::NEG_INFINITY
    } else {
      heap_threshold
    };

    let mut pivot_idx = None;
    let mut acc = 0.0_f32;

    // Linear scan to find pivot
    for (i, term) in queue.iter().enumerate() {
      let bound = if use_block_bounds {
        term.block_upper_bound()
      } else {
        term.upper_bound()
      };

      // Skip NaN bounds only: NaN arithmetic poisons the accumulator and
      // `NaN >= threshold` is always false, so the term could never trigger
      // the pivot. Positive infinity, on the other hand, is a valid (if
      // loose) upper bound — it should immediately satisfy any finite
      // threshold and set the pivot, otherwise the WAND loop can terminate
      // early and silently drop documents whose global ub overflowed.
      if bound.is_nan() {
        continue;
      }
      acc += bound;
      if acc >= pivot_threshold {
        pivot_idx = Some(i);
        break;
      }
    }

    let Some(p_idx) = pivot_idx else {
      // Threshold not reachable with remaining terms
      break;
    };

    let pivot_doc = queue[p_idx].doc_id();
    let smallest_doc = queue[0].doc_id();

    if pivot_doc == DOCID_END {
      break;
    }

    if pivot_doc == smallest_doc {
      let doc_id = pivot_doc;
      let mut score_sum = 0.0;

      // Advance all terms matching this doc_id
      // Since queue is sorted, they are at the start
      let mut i = 0;
      while i < queue.len() {
        if queue[i].doc_id() != doc_id {
          break;
        }

        let term = &mut queue[i];
        let contribution = term.score_current();
        score_sum += contribution;

        if let (Some(buf), Some(flags)) = (leaf_scores.as_mut(), touched_flags.as_mut()) {
          let leaf = term.leaf;
          // leaf index is guaranteed by the planner to be in bounds
          if !flags[leaf] {
            flags[leaf] = true;
            touched.push(leaf);
          }
          buf[leaf] += contribution;
        }

        let moved = term.advance();
        with_stats(&mut stats, |s| s.postings_advanced += moved);
        if term.is_done() {
          prune_done = true;
        }
        i += 1;
      }

      // Reposition only the advanced prefix to maintain sorted order.
      bubble_reposition(&mut queue, i);

      with_stats(&mut stats, |s| {
        s.candidates_examined += 1;
        s.scored_docs += 1;
      });

      let mut score = score_sum;
      if let Some(plan) = score_plan {
        if let Some(buf) = leaf_scores.as_ref() {
          score = plan.evaluate(buf);
        }
      }

      let leaves_slice = leaf_scores.as_deref().unwrap_or(&[]);
      let score_opt = if let Some(adj) = score_adjust.as_deref_mut() {
        adj(doc_id, score, leaves_slice)
      } else {
        Some(score)
      }
      // BUG-381: a term's `weight` multiplies into `score_tf` on every
      // contribution, so accumulated query-boost products overflowing
      // `f32::MAX` turn `score_sum` into `+inf`. `evaluate_compiled_score`
      // already drops non-finite scores via `score_adjust`, but the plain
      // BM25 path has no such hook — filter here so non-finite scores
      // cannot reach the heap (they serialise as invalid JSON and return
      // HTTP 500 to the client).
      .filter(|s| s.is_finite());

      if let (Some(buf), Some(flags)) = (leaf_scores.as_mut(), touched_flags.as_mut()) {
        for idx in touched.drain(..) {
          buf[idx] = 0.0;
          flags[idx] = false;
        }
      }

      if let Some(final_score) = score_opt {
        if accept(doc_id, final_score) {
          if let Some(collector) = collector.as_deref_mut() {
            collector.collect(doc_id, final_score);
          }
          if rank_hits && (heap.len() < k || final_score > heap_threshold) {
            push_top_k(
              &mut heap,
              RankedDoc {
                doc_id,
                score: final_score,
              },
              k,
            );
          }
        }
      }
    } else {
      // Pivot > Smallest. Advance terms < pivot to pivot_doc.
      //
      // BMW optimisation (BUG-005): when we have a full top-k heap and
      // block-level bounds are available, skip blocks whose per-block
      // max-tf score cannot contribute enough to push a candidate past
      // the heap threshold. For each term being advanced, the minimum
      // contribution needed from this term is:
      //   min_needed = heap_threshold - Σ UB_global(other terms)
      // because the global upper bound of every other term is an upper
      // bound on what those terms could contribute at any doc_id.
      if use_block_bounds && rank_hits && heap.len() >= k {
        let total_ub: f32 = queue.iter().map(|t| t.upper_bound()).sum();
        for term in queue[0..p_idx].iter_mut() {
          let other_ub = total_ub - term.upper_bound();
          let min_needed = (heap_threshold - other_ub).max(0.0);
          let moved = term.skip_blocks_below_bound(min_needed);
          with_stats(&mut stats, |s| s.postings_advanced += moved);
          let moved = term.skip_to_block(pivot_doc);
          with_stats(&mut stats, |s| s.postings_advanced += moved);
          let moved = term.advance_to(pivot_doc);
          with_stats(&mut stats, |s| s.postings_advanced += moved);
          if term.is_done() {
            prune_done = true;
          }
        }
      } else {
        for term in queue[0..p_idx].iter_mut() {
          if use_block_bounds {
            let moved = term.skip_to_block(pivot_doc);
            with_stats(&mut stats, |s| s.postings_advanced += moved);
          }
          let moved = term.advance_to(pivot_doc);
          with_stats(&mut stats, |s| s.postings_advanced += moved);
          if term.is_done() {
            prune_done = true;
          }
        }
      }
      // Reposition only the advanced prefix.
      bubble_reposition(&mut queue, p_idx);
    }
  }

  finalize_heap(heap)
}

fn push_top_k(heap: &mut BinaryHeap<Reverse<RankedDoc>>, doc: RankedDoc, k: usize) {
  if heap.len() < k {
    heap.push(Reverse(doc));
    return;
  }
  if let Some(worst) = heap.peek() {
    if doc > worst.0 {
      heap.pop();
      heap.push(Reverse(doc));
    }
  }
}

fn finalize_heap(heap: BinaryHeap<Reverse<RankedDoc>>) -> Vec<RankedDoc> {
  let mut out: Vec<RankedDoc> = heap.into_iter().map(|r| r.0).collect();
  out.sort_by(|a, b| {
    b.score
      .total_cmp(&a.score)
      .then_with(|| a.doc_id.cmp(&b.doc_id))
  });
  out
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::index::postings::PostingEntry;
  use smallvec::smallvec;
  use std::sync::Arc;

  fn term_from_entries(entries: &[PostingEntry]) -> ScoredTerm {
    let reader = PostingsReader::from_entries_for_test(entries.to_vec(), DEFAULT_BLOCK_SIZE);
    let max_doc = entries.iter().map(|e| e.doc_id).max().unwrap_or(0) as usize;
    let doc_lengths = Arc::new(vec![10.0; max_doc.saturating_add(1)]);
    ScoredTerm {
      postings: reader,
      weight: 1.0,
      avgdl: 10.0,
      docs: 10.0,
      k1: 1.2,
      b: 0.75,
      leaf: 0,
      doc_lengths: Some(doc_lengths),
      min_doc_len: Some(10.0),
    }
  }

  #[test]
  fn ranked_doc_ordering_prefers_smaller_id_on_tie() {
    let a = RankedDoc {
      doc_id: 1,
      score: 1.0,
    };
    let b = RankedDoc {
      doc_id: 2,
      score: 1.0,
    };
    let mut heap = BinaryHeap::new();
    heap.push(Reverse(a));
    heap.push(Reverse(b));
    let worst = heap.peek().unwrap().0;
    assert_eq!(worst.doc_id, 2);
  }

  #[test]
  fn brute_force_matches_wand_results() {
    let term1 = term_from_entries(&[
      PostingEntry {
        doc_id: 1,
        term_freq: 2,
        positions: smallvec![],
      },
      PostingEntry {
        doc_id: 3,
        term_freq: 1,
        positions: smallvec![],
      },
    ]);
    let term2 = term_from_entries(&[PostingEntry {
      doc_id: 3,
      term_freq: 3,
      positions: smallvec![],
    }]);
    let mut accept = |_doc: DocId, _score: f32| true;
    let brute = brute_force::<_, crate::query::collector::MatchCountingCollector>(
      &[term1.clone(), term2.clone()],
      2,
      true,
      None,
      &mut accept,
      None,
      None,
      None,
    );
    let wand = execute_top_k::<_, crate::query::collector::MatchCountingCollector>(
      vec![term1, term2],
      2,
      ExecutionStrategy::Wand,
      None,
      &mut accept,
      None,
    );
    assert_eq!(brute.len(), wand.len());
    for (a, b) in brute.iter().zip(wand.iter()) {
      assert_eq!(a.doc_id, b.doc_id);
      assert!((a.score - b.score).abs() < 1e-6);
    }
  }

  #[test]
  fn bm25_penalizes_long_documents() {
    let short = score_tf(2.0, 1.0, 5.0, 10.0, 100.0, 1.2, 0.75, 1.0);
    let long = score_tf(2.0, 1.0, 100.0, 10.0, 100.0, 1.2, 0.75, 1.0);
    assert!(
      short > long,
      "short doc score {short} should exceed long doc score {long}"
    );
  }

  #[test]
  fn collectors_receive_all_matched_docs() {
    let term1 = term_from_entries(&[
      PostingEntry {
        doc_id: 1,
        term_freq: 1,
        positions: smallvec![],
      },
      PostingEntry {
        doc_id: 2,
        term_freq: 1,
        positions: smallvec![],
      },
    ]);
    let mut accept = |_doc: DocId, _score: f32| true;
    let mut collector = crate::query::collector::RecordingCollector::default();
    let results = execute_top_k(
      vec![term1],
      1,
      ExecutionStrategy::Bm25,
      None,
      &mut accept,
      Some(&mut collector),
    );
    assert_eq!(results.len(), 1);
    assert_eq!(collector.docs.len(), 2);
    let mut ids: Vec<DocId> = collector.docs.iter().map(|(id, _)| *id).collect();
    ids.sort_unstable();
    assert_eq!(ids, vec![1, 2]);
  }

  /// Helper that builds a ScoredTerm with a specific block size so the caller
  /// can control per-block tf upper bounds in tests.
  fn term_from_entries_with_block_size(entries: &[PostingEntry], block_size: usize) -> ScoredTerm {
    let reader = PostingsReader::from_entries_for_test(entries.to_vec(), block_size);
    let max_doc = entries.iter().map(|e| e.doc_id).max().unwrap_or(0) as usize;
    let doc_lengths = Arc::new(vec![10.0; max_doc.saturating_add(1)]);
    ScoredTerm {
      postings: reader,
      weight: 1.0,
      avgdl: 10.0,
      docs: 100.0,
      k1: 1.2,
      b: 0.75,
      leaf: 0,
      doc_lengths: Some(doc_lengths),
      min_doc_len: Some(10.0),
    }
  }

  #[test]
  fn bmw_produces_same_top_k_as_wand() {
    // Two terms with overlapping postings; verify BMW and WAND return
    // identical top-k results (correctness invariant).
    let term1 = term_from_entries(&[
      PostingEntry {
        doc_id: 0,
        term_freq: 3,
        positions: smallvec![],
      },
      PostingEntry {
        doc_id: 2,
        term_freq: 1,
        positions: smallvec![],
      },
      PostingEntry {
        doc_id: 5,
        term_freq: 5,
        positions: smallvec![],
      },
      PostingEntry {
        doc_id: 8,
        term_freq: 2,
        positions: smallvec![],
      },
    ]);
    let term2 = term_from_entries(&[
      PostingEntry {
        doc_id: 1,
        term_freq: 4,
        positions: smallvec![],
      },
      PostingEntry {
        doc_id: 5,
        term_freq: 1,
        positions: smallvec![],
      },
      PostingEntry {
        doc_id: 7,
        term_freq: 6,
        positions: smallvec![],
      },
    ]);
    let mut accept = |_doc: DocId, _score: f32| true;
    let wand = execute_top_k::<_, crate::query::collector::MatchCountingCollector>(
      vec![term1.clone(), term2.clone()],
      3,
      ExecutionStrategy::Wand,
      None,
      &mut accept,
      None,
    );
    let bmw = execute_top_k::<_, crate::query::collector::MatchCountingCollector>(
      vec![term1, term2],
      3,
      ExecutionStrategy::Bmw,
      None,
      &mut accept,
      None,
    );
    assert_eq!(
      wand.len(),
      bmw.len(),
      "WAND and BMW should return same count"
    );
    for (w, b) in wand.iter().zip(bmw.iter()) {
      assert_eq!(w.doc_id, b.doc_id, "doc_id mismatch");
      assert!(
        (w.score - b.score).abs() < 1e-6,
        "score mismatch: wand={} bmw={}",
        w.score,
        b.score
      );
    }
  }

  #[test]
  fn bmw_block_tf_skipping_exercises_advancement_path() {
    // Regression test for BUG-005: exercises the `pivot_doc > smallest_doc`
    // advancement branch where `skip_blocks_below_bound` is called.
    //
    // Layout (block_size = 4):
    //
    //   anchor: 8 entries at doc_ids [0..4, 40..44], all tf=20
    //     block 0 (entries 0-3): docs [0,1,2,3],     max_tf=20
    //     block 1 (entries 4-7): docs [40,41,42,43],  max_tf=20
    //
    //   spread: 44 entries at doc_ids [0..4] tf=20, [4..40] tf=1, [40..44] tf=20
    //     block 0  (entries 0-3):   docs [0,1,2,3],     max_tf=20
    //     block 1  (entries 4-7):   docs [4,5,6,7],     max_tf=1   ← skippable
    //     block 2  (entries 8-11):  docs [8,9,10,11],   max_tf=1   ← skippable
    //     ...
    //     block 9  (entries 36-39): docs [36,37,38,39], max_tf=1   ← skippable
    //     block 10 (entries 40-43): docs [40,41,42,43], max_tf=20
    //
    // Phase 1 (docs 0-3): both terms score together (tf=20 each).
    //   The top-k heap fills with high combined scores (k=3).
    //
    // Phase 2: anchor jumps to doc 40. spread is at doc 4.
    //   Queue: [spread(4), anchor(40)]
    //   Pivot scan accumulates spread.block_ub(tf=1) + anchor.block_ub(tf=20).
    //   pivot_doc = 40 > smallest_doc = 4 → advancement branch entered.
    //   spread must advance from doc 4 to doc 40 through 9 blocks of tf=1.
    //   skip_blocks_below_bound skips those low-tf blocks.
    let block_size = 4;

    // anchor: high-tf entries at low and high doc_ids with a gap.
    let mut anchor_entries = Vec::new();
    for doc_id in 0..4 {
      anchor_entries.push(PostingEntry {
        doc_id,
        term_freq: 20,
        positions: smallvec![],
      });
    }
    for doc_id in 40..44 {
      anchor_entries.push(PostingEntry {
        doc_id,
        term_freq: 20,
        positions: smallvec![],
      });
    }
    let anchor = term_from_entries_with_block_size(&anchor_entries, block_size);

    // spread: overlaps anchor at both ends, with many low-tf entries in between.
    let mut spread_entries = Vec::new();
    for doc_id in 0..4 {
      spread_entries.push(PostingEntry {
        doc_id,
        term_freq: 20,
        positions: smallvec![],
      });
    }
    for doc_id in 4..40 {
      spread_entries.push(PostingEntry {
        doc_id,
        term_freq: 1,
        positions: smallvec![],
      });
    }
    for doc_id in 40..44 {
      spread_entries.push(PostingEntry {
        doc_id,
        term_freq: 20,
        positions: smallvec![],
      });
    }
    let spread = term_from_entries_with_block_size(&spread_entries, block_size);

    let mut accept = |_doc: DocId, _score: f32| true;

    // Run with plain WAND (no block-tf skipping)
    let mut wand_stats = QueryStats::default();
    let wand_results = execute_top_k_with_stats::<_, crate::query::collector::MatchCountingCollector>(
      vec![anchor.clone(), spread.clone()],
      3,
      ExecutionStrategy::Wand,
      Some(block_size),
      &mut accept,
      None,
      Some(&mut wand_stats),
    );

    // Run with BMW (block-tf skipping active)
    let mut bmw_stats = QueryStats::default();
    let bmw_results = execute_top_k_with_stats::<_, crate::query::collector::MatchCountingCollector>(
      vec![anchor, spread],
      3,
      ExecutionStrategy::Bmw,
      Some(block_size),
      &mut accept,
      None,
      Some(&mut bmw_stats),
    );

    // Results must be identical (correctness).
    assert_eq!(
      wand_results.len(),
      bmw_results.len(),
      "WAND and BMW should return same number of results"
    );
    for (w, b) in wand_results.iter().zip(bmw_results.iter()) {
      assert_eq!(w.doc_id, b.doc_id, "doc_id mismatch");
      assert!(
        (w.score - b.score).abs() < 1e-6,
        "score mismatch at doc {}: wand={} bmw={}",
        w.doc_id,
        w.score,
        b.score
      );
    }

    // BMW should advance strictly fewer postings than plain WAND because
    // spread's low-tf blocks (doc_ids 4-39) are skipped in the advancement
    // phase rather than walked entry-by-entry.
    assert!(
      bmw_stats.postings_advanced < wand_stats.postings_advanced,
      "BMW should advance strictly fewer postings than WAND: bmw={} wand={}",
      bmw_stats.postings_advanced,
      wand_stats.postings_advanced
    );
  }

  #[test]
  fn skip_blocks_below_bound_skips_low_tf_blocks() {
    // Directly test the skip_blocks_below_bound method on a TermState
    // with known block-level tf values.
    let block_size = 2;
    // 3 blocks: block 0 (tf=1), block 1 (tf=1), block 2 (tf=10)
    let entries = vec![
      PostingEntry {
        doc_id: 0,
        term_freq: 1,
        positions: smallvec![],
      },
      PostingEntry {
        doc_id: 1,
        term_freq: 1,
        positions: smallvec![],
      },
      PostingEntry {
        doc_id: 2,
        term_freq: 1,
        positions: smallvec![],
      },
      PostingEntry {
        doc_id: 3,
        term_freq: 1,
        positions: smallvec![],
      },
      PostingEntry {
        doc_id: 4,
        term_freq: 10,
        positions: smallvec![],
      },
      PostingEntry {
        doc_id: 5,
        term_freq: 10,
        positions: smallvec![],
      },
    ];
    let term = ScoredTerm {
      postings: PostingsReader::from_entries_for_test(entries, block_size),
      weight: 1.0,
      avgdl: 10.0,
      docs: 100.0,
      k1: 1.2,
      b: 0.75,
      leaf: 0,
      doc_lengths: Some(Arc::new(vec![10.0; 6])),
      min_doc_len: Some(10.0),
    };
    let mut state = TermState::new(term, block_size);

    // Compute the score bound for tf=1 (low blocks) and tf=10 (high block)
    let low_bound = score_tf(1.0, 6.0, 10.0, 10.0, 100.0, 1.2, 0.75, 1.0);
    let high_bound = score_tf(10.0, 6.0, 10.0, 10.0, 100.0, 1.2, 0.75, 1.0);

    // Set min_contribution between low and high so low blocks are skipped
    let threshold = (low_bound + high_bound) / 2.0;
    assert!(
      threshold > low_bound,
      "threshold must exceed low block bound"
    );
    assert!(
      threshold < high_bound,
      "threshold must be below high block bound"
    );

    // Start at block 0 (idx=0). Calling skip_blocks_below_bound should skip
    // blocks 0 and 1 (both have tf=1 < threshold) and land at block 2 (idx=4).
    let skipped = state.skip_blocks_below_bound(threshold);
    assert_eq!(state.idx, 4, "should advance to block 2 (idx=4)");
    assert_eq!(skipped, 4, "should have skipped 4 postings");

    // Verify the block at idx=4 has doc_id=4
    assert_eq!(state.doc_id(), 4);
  }

  #[test]
  fn wand_does_not_skip_term_with_infinite_upper_bound() {
    // BUG-366: when a term's global upper bound overflows to +inf (e.g.
    // because its weight is huge), the pivot-finding loop must not silently
    // skip it. A +inf bound is a valid (if loose) upper bound — skipping it
    // can prevent the accumulator from reaching the heap threshold, causing
    // WAND to exit early and drop the term's documents.
    //
    // Setup: one "anchor" term with normal BM25 scores that seeds the heap
    // with finite scores, and one "overflow" term whose single posting lies
    // past the anchor's doc_ids and whose ub overflows f32.
    let anchor = term_from_entries(&[
      PostingEntry {
        doc_id: 1,
        term_freq: 1,
        positions: smallvec![],
      },
      PostingEntry {
        doc_id: 2,
        term_freq: 1,
        positions: smallvec![],
      },
      PostingEntry {
        doc_id: 3,
        term_freq: 1,
        positions: smallvec![],
      },
    ]);

    let overflow_postings = PostingsReader::from_entries_for_test(
      vec![PostingEntry {
        doc_id: 100,
        term_freq: 1,
        positions: smallvec![],
      }],
      DEFAULT_BLOCK_SIZE,
    );
    // Craft the overflow term so that its *upper bound* is `+inf` but the
    // per-doc runtime score at doc 100 is finite. The upper bound is
    // computed with `min_doc_len`, so a tiny `min_doc_len` inflates the
    // BM25 tf-norm factor (short docs score higher). The actual runtime
    // score at doc 100 uses `doc_lengths[100] = 100.0`, 10× `avgdl`, which
    // drops the tf-norm by ~7× and keeps `bm25 * weight` comfortably
    // below `f32::MAX`. This separates the two concerns exercised by
    // BUG-366 (pivot-scan admits +inf ub) and BUG-381 (finite runtime
    // scores are not filtered out by the new finitude guard).
    let mut overflow_doc_lengths = vec![10.0; 101];
    overflow_doc_lengths[100] = 100.0;
    let overflow = ScoredTerm {
      postings: overflow_postings,
      weight: 5.0e36,
      avgdl: 10.0,
      docs: 1.0e30,
      k1: 1.2,
      b: 0.75,
      leaf: 0,
      doc_lengths: Some(Arc::new(overflow_doc_lengths)),
      min_doc_len: Some(1.0),
    };

    // Precondition: the overflow term's global ub really is +inf. If bm25
    // math ever changes such that this no longer overflows, the test stops
    // exercising the bug and we need to pick new inputs.
    let overflow_state = TermState::new(overflow.clone(), DEFAULT_BLOCK_SIZE);
    assert!(
      overflow_state.upper_bound().is_infinite() && overflow_state.upper_bound().is_sign_positive(),
      "overflow term global ub should be +inf; got {}",
      overflow_state.upper_bound(),
    );
    // Precondition: the runtime score at doc 100 is finite, so the BUG-381
    // finitude guard in wand_loop will let it through. Without this, the
    // BUG-366 regression would accidentally exercise the BUG-381 path.
    let runtime_score = score_tf(
      1.0,
      overflow_state.df,
      overflow.doc_len(100),
      overflow.avgdl,
      overflow.docs,
      overflow.k1,
      overflow.b,
      overflow.weight,
    );
    assert!(
      runtime_score.is_finite(),
      "overflow term runtime score at doc 100 should be finite; got {runtime_score}",
    );

    let mut accept = |_doc: DocId, _score: f32| true;
    let results = execute_top_k::<_, crate::query::collector::MatchCountingCollector>(
      vec![anchor, overflow],
      2,
      ExecutionStrategy::Wand,
      None,
      &mut accept,
      None,
    );

    // doc 100's score is dominated by the overflowing weight, so it must be
    // ranked into the top-k. Before the fix, the `is_finite()` pivot guard
    // skipped the overflow term and doc 100 was silently dropped.
    assert!(
      results.iter().any(|r| r.doc_id == 100),
      "doc 100 (overflow term) was dropped from top-k: {:?}",
      results.iter().map(|r| r.doc_id).collect::<Vec<_>>(),
    );
  }

  // Helper: build a ScoredTerm whose per-doc BM25 score overflows `f32::MAX`
  // to `+inf`. The setup mimics the trigger described in BUG-381: the query
  // weight has been multiplied by a nested-boost product that pushes it to
  // `f32::MAX`, so `bm25 * weight` is always non-finite.
  fn overflow_scored_term(doc_ids: &[DocId]) -> ScoredTerm {
    let entries: Vec<PostingEntry> = doc_ids
      .iter()
      .map(|&doc_id| PostingEntry {
        doc_id,
        term_freq: 1,
        positions: smallvec![],
      })
      .collect();
    let reader = PostingsReader::from_entries_for_test(entries, DEFAULT_BLOCK_SIZE);
    let max_doc = doc_ids.iter().copied().max().unwrap_or(0) as usize;
    let doc_lengths = Arc::new(vec![10.0; max_doc.saturating_add(1)]);
    ScoredTerm {
      postings: reader,
      weight: f32::MAX,
      avgdl: 10.0,
      docs: 1.0e30,
      k1: 1.2,
      b: 0.75,
      leaf: 0,
      doc_lengths: Some(doc_lengths),
      min_doc_len: Some(10.0),
    }
  }

  /// BUG-381: a document whose BM25 score overflows to `+inf` must not reach
  /// the top-k heap on the WAND path. Before the fix, `score_sum` flowed
  /// straight from `score_tf` to `push_top_k` whenever no custom scoring
  /// hook was active, so `Hit.score = +inf` would cause `serde_json` to
  /// fail and the HTTP endpoint to return 500.
  #[test]
  fn wand_drops_doc_with_non_finite_bm25_score() {
    let anchor = term_from_entries(&[PostingEntry {
      doc_id: 1,
      term_freq: 1,
      positions: smallvec![],
    }]);
    let overflow = overflow_scored_term(&[5]);

    // Precondition: the runtime BM25 score for the overflow term really is
    // non-finite. If bm25 math changes so this is no longer true, pick
    // new inputs — the test is no longer exercising BUG-381.
    let probe_score = score_tf(
      1.0,
      overflow.postings.len() as f32,
      overflow.doc_len(5),
      overflow.avgdl,
      overflow.docs,
      overflow.k1,
      overflow.b,
      overflow.weight,
    );
    assert!(
      !probe_score.is_finite(),
      "overflow term runtime score should be non-finite; got {probe_score}",
    );

    let mut accept = |_doc: DocId, _score: f32| true;
    let results = execute_top_k::<_, crate::query::collector::MatchCountingCollector>(
      vec![anchor, overflow],
      10,
      ExecutionStrategy::Wand,
      None,
      &mut accept,
      None,
    );

    assert!(
      results.iter().all(|r| r.score.is_finite()),
      "non-finite scores leaked into top-k: {:?}",
      results.iter().map(|r| r.score).collect::<Vec<_>>(),
    );
    assert!(
      !results.iter().any(|r| r.doc_id == 5),
      "doc 5 (non-finite score) should have been dropped: {:?}",
      results.iter().map(|r| r.doc_id).collect::<Vec<_>>(),
    );
  }

  /// BUG-381: BMW path has the same score-accumulation as WAND, so the
  /// same finitude guard must apply when block-level bounds are enabled.
  #[test]
  fn bmw_drops_doc_with_non_finite_bm25_score() {
    let anchor = term_from_entries(&[PostingEntry {
      doc_id: 1,
      term_freq: 1,
      positions: smallvec![],
    }]);
    let overflow = overflow_scored_term(&[5]);

    let mut accept = |_doc: DocId, _score: f32| true;
    let results = execute_top_k::<_, crate::query::collector::MatchCountingCollector>(
      vec![anchor, overflow],
      10,
      ExecutionStrategy::Bmw,
      None,
      &mut accept,
      None,
    );

    assert!(
      results.iter().all(|r| r.score.is_finite()),
      "non-finite scores leaked into top-k via BMW: {:?}",
      results.iter().map(|r| r.score).collect::<Vec<_>>(),
    );
    assert!(
      !results.iter().any(|r| r.doc_id == 5),
      "doc 5 (non-finite score) should have been dropped on BMW path: {:?}",
      results.iter().map(|r| r.doc_id).collect::<Vec<_>>(),
    );
  }

  /// BUG-381: the brute-force path (`ExecutionStrategy::Bm25`, no score
  /// plan) accumulates `score_tf` contributions into a hashmap and then
  /// pushes each doc into the heap. The finitude guard must drop any doc
  /// whose accumulated contribution is non-finite.
  #[test]
  fn brute_force_drops_doc_with_non_finite_bm25_score() {
    let overflow = overflow_scored_term(&[5, 7]);
    let anchor = term_from_entries(&[PostingEntry {
      doc_id: 1,
      term_freq: 1,
      positions: smallvec![],
    }]);

    let mut accept = |_doc: DocId, _score: f32| true;
    let results = execute_top_k::<_, crate::query::collector::MatchCountingCollector>(
      vec![anchor, overflow],
      10,
      ExecutionStrategy::Bm25,
      None,
      &mut accept,
      None,
    );

    assert!(
      results.iter().all(|r| r.score.is_finite()),
      "non-finite scores leaked into top-k via brute-force: {:?}",
      results.iter().map(|r| r.score).collect::<Vec<_>>(),
    );
    let dropped_docs: Vec<DocId> = results.iter().map(|r| r.doc_id).collect();
    assert!(
      !dropped_docs.contains(&5) && !dropped_docs.contains(&7),
      "overflow docs should have been dropped: {dropped_docs:?}",
    );
    // The finite anchor doc should still be ranked.
    assert!(
      dropped_docs.contains(&1),
      "anchor doc 1 should still be ranked: {dropped_docs:?}",
    );
  }

  #[test]
  fn skip_blocks_below_bound_noop_when_threshold_zero() {
    let block_size = 2;
    let entries = vec![
      PostingEntry {
        doc_id: 0,
        term_freq: 1,
        positions: smallvec![],
      },
      PostingEntry {
        doc_id: 1,
        term_freq: 1,
        positions: smallvec![],
      },
    ];
    let term = ScoredTerm {
      postings: PostingsReader::from_entries_for_test(entries, block_size),
      weight: 1.0,
      avgdl: 10.0,
      docs: 100.0,
      k1: 1.2,
      b: 0.75,
      leaf: 0,
      doc_lengths: Some(Arc::new(vec![10.0; 2])),
      min_doc_len: Some(10.0),
    };
    let mut state = TermState::new(term, block_size);

    // With threshold <= 0, nothing should be skipped
    let skipped = state.skip_blocks_below_bound(0.0);
    assert_eq!(skipped, 0);
    assert_eq!(state.idx, 0);

    let skipped = state.skip_blocks_below_bound(-1.0);
    assert_eq!(skipped, 0);
    assert_eq!(state.idx, 0);
  }
}
