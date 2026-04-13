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
  block_max_doc_ids: Vec<DocId>,
  block_max_tfs: Vec<f32>,
  block_size: usize,
}

impl TermState {
  fn new(term: ScoredTerm, block_size: usize) -> Self {
    let df = term.postings.len() as f32;
    let clamped_block = block_size.max(1);
    let (block_max_doc_ids, block_max_tfs) = build_block_meta(&term.postings, clamped_block);
    let fallback = term.avgdl.max(1.0);
    let min_doc_len = term
      .min_doc_len
      .filter(|v| v.is_finite() && *v > 0.0)
      .unwrap_or(fallback);
    let doc_lengths = term.doc_lengths.clone();
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
      block_max_doc_ids,
      block_max_tfs,
      block_size: clamped_block,
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
    self.idx / self.block_size
  }

  fn block_upper_bound(&self) -> f32 {
    let block_idx = self.block_index();
    let tf = self.block_max_tfs.get(block_idx).copied().unwrap_or(0.0);
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
    let block_idx = self.block_max_doc_ids.partition_point(|doc| *doc < target);
    let start = block_idx.saturating_mul(self.block_size);
    if start > self.idx {
      self.idx = start.min(self.postings.len());
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

fn build_block_meta(postings: &PostingsReader, block_size: usize) -> (Vec<DocId>, Vec<f32>) {
  if block_size == postings.block_size && !postings.block_max_doc_ids.is_empty() {
    return (
      postings.block_max_doc_ids.clone(),
      postings.block_max_tfs.clone(),
    );
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
  (block_max_doc_ids, block_max_tfs)
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
    if advanced >= queue.len() {
      queue.sort_unstable_by_key(|t| t.doc_id());
      return;
    }
    // Sort the modified prefix (typically 1-5 elements).
    if advanced > 1 {
      queue[..advanced].sort_unstable_by_key(|t| t.doc_id());
    }
    // Merge two sorted runs in-place. For each element from the suffix
    // that belongs before the current prefix element, rotate it into
    // place. Since `advanced` is small, this is effectively O(n).
    let mut i = 0;
    let mut j = advanced;
    while i < j && j < queue.len() {
      if queue[i].doc_id() <= queue[j].doc_id() {
        i += 1;
      } else {
        queue[i..=j].rotate_right(1);
        i += 1;
        j += 1;
      }
    }
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

      if bound.is_finite() {
        acc += bound;
        if acc >= pivot_threshold {
          pivot_idx = Some(i);
          break;
        }
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
      };

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
      // Pivot > Smallest. Advance terms < pivot to pivot_doc
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
}
