use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use crate::api::types::ExecutionStrategy;
use crate::index::postings::{PostingsReader, DEFAULT_BLOCK_SIZE};
use crate::query::bm25::bm25;
use crate::query::collector::DocCollector;
use crate::DocId;

const DOCID_END: DocId = u32::MAX;

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
  pub weight: u32,
  pub avgdl: f32,
  pub docs: f32,
  pub k1: f32,
  pub b: f32,
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
  ub: f32,
  block_max_doc_ids: Vec<DocId>,
  block_max_tfs: Vec<f32>,
  block_size: usize,
}

impl TermState {
  fn new(term: ScoredTerm, block_size: usize) -> Self {
    let df = term.postings.len() as f32;
    let clamped_block = block_size.max(1);
    let (block_max_doc_ids, block_max_tfs) = build_block_meta(&term.postings, clamped_block);
    let ub = upper_bound_tf(
      term.postings.max_tf,
      df,
      term.avgdl,
      term.docs,
      term.k1,
      term.b,
      term.weight as f32,
    );
    Self {
      postings: term.postings,
      idx: 0,
      weight: term.weight as f32,
      df,
      avgdl: term.avgdl,
      docs: term.docs,
      k1: term.k1,
      b: term.b,
      ub,
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

fn score_tf(tf: f32, df: f32, avgdl: f32, docs: f32, k1: f32, b: f32, weight: f32) -> f32 {
  let doc_len = if avgdl > 0.0 { avgdl } else { tf };
  let base = bm25(tf, df, doc_len, avgdl, docs, k1, b);
  base * weight
}

fn upper_bound_tf(tf: f32, df: f32, avgdl: f32, docs: f32, k1: f32, b: f32, weight: f32) -> f32 {
  if tf <= 0.0 {
    return 0.0;
  }
  score_tf(tf, df, avgdl, docs, k1, b, weight)
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
  execute_top_k_with_mode(
    terms,
    k,
    strategy,
    block_size,
    accept,
    collector,
    ScoreMode::Score,
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
  execute_top_k_with_stats_and_mode(
    terms, k, strategy, block_size, accept, collector, None, score_mode,
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
  execute_top_k_with_stats_and_mode(
    terms,
    k,
    strategy,
    block_size,
    accept,
    collector,
    stats,
    ScoreMode::Score,
  )
}

#[allow(clippy::too_many_arguments)]
pub fn execute_top_k_with_stats_and_mode<F: FnMut(DocId, f32) -> bool, C: DocCollector + ?Sized>(
  terms: Vec<ScoredTerm>,
  k: usize,
  strategy: ExecutionStrategy,
  block_size: Option<usize>,
  accept: &mut F,
  collector: Option<&mut C>,
  stats: Option<&mut QueryStats>,
  score_mode: ScoreMode,
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
    return brute_force(&terms, k, should_rank, accept, collector, stats);
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
    accept,
    collector,
    stats,
  )
}

fn brute_force<F: FnMut(DocId, f32) -> bool, C: DocCollector + ?Sized>(
  terms: &[ScoredTerm],
  k: usize,
  rank_hits: bool,
  accept: &mut F,
  mut collector: Option<&mut C>,
  mut stats: Option<&mut QueryStats>,
) -> Vec<RankedDoc> {
  let mut scores: hashbrown::HashMap<DocId, f32> = hashbrown::HashMap::new();
  for term in terms.iter() {
    let df = term.postings.len() as f32;
    with_stats(&mut stats, |s| s.postings_advanced += term.postings.len());
    for entry in term.postings.iter() {
      let score = score_tf(
        entry.term_freq as f32,
        df,
        term.avgdl,
        term.docs,
        term.k1,
        term.b,
        term.weight as f32,
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
  for (doc_id, score) in scores.into_iter() {
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
  mut terms: Vec<TermState>,
  accept: &mut F,
  mut collector: Option<&mut C>,
  mut stats: Option<&mut QueryStats>,
) -> Vec<RankedDoc> {
  let mut order: Vec<usize> = (0..terms.len()).collect();
  order.sort_by_key(|&idx| terms[idx].doc_id());
  while !order.is_empty() {
    let doc = terms[order[0]].doc_id();
    if doc == DOCID_END {
      break;
    }
    let mut mutated: Vec<usize> = Vec::new();
    let mut idx = 0usize;
    while idx < order.len() && terms[order[idx]].doc_id() == doc {
      let term_idx = order[idx];
      let moved = terms[term_idx].advance();
      with_stats(&mut stats, |s| s.postings_advanced += moved);
      mutated.push(term_idx);
      idx += 1;
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
    requeue_terms(&mut order, &terms, &mut mutated);
  }
  Vec::new()
}

fn wand_loop<F: FnMut(DocId, f32) -> bool, C: DocCollector + ?Sized>(
  mut terms: Vec<TermState>,
  k: usize,
  rank_hits: bool,
  use_block_bounds: bool,
  accept: &mut F,
  mut collector: Option<&mut C>,
  mut stats: Option<&mut QueryStats>,
) -> Vec<RankedDoc> {
  let mut heap: BinaryHeap<Reverse<RankedDoc>> = BinaryHeap::new();
  let mut order: Vec<usize> = (0..terms.len()).collect();
  order.sort_by_key(|&idx| terms[idx].doc_id());
  loop {
    order.retain(|idx| !terms[*idx].is_done());
    if order.is_empty() {
      break;
    }
    let threshold = heap.peek().map(|d| d.0.score).unwrap_or(0.0);
    let pivot_idx = match choose_pivot(
      &order,
      &terms,
      if collector.is_some() {
        f32::NEG_INFINITY
      } else {
        threshold
      },
      use_block_bounds,
    ) {
      Some(idx) => idx,
      None => break,
    };
    let pivot_doc = terms[order[pivot_idx]].doc_id();
    let smallest_doc = terms[order[0]].doc_id();
    if pivot_doc == smallest_doc {
      let doc_id = pivot_doc;
      let mut score = 0.0;
      let mut mutated: Vec<usize> = Vec::new();
      let mut idx = 0usize;
      while idx < order.len() && terms[order[idx]].doc_id() == doc_id {
        let term_idx = order[idx];
        score += terms[term_idx].score_current();
        let moved = terms[term_idx].advance();
        mutated.push(term_idx);
        with_stats(&mut stats, |s| s.postings_advanced += moved);
        idx += 1;
      }
      with_stats(&mut stats, |s| {
        s.candidates_examined += 1;
        s.scored_docs += 1;
      });
      let accepted = accept(doc_id, score);
      if accepted {
        if let Some(collector) = collector.as_deref_mut() {
          collector.collect(doc_id, score);
        }
        if rank_hits && (heap.len() < k || score > threshold) {
          push_top_k(&mut heap, RankedDoc { doc_id, score }, k);
        }
      }
      requeue_terms(&mut order, &terms, &mut mutated);
    } else {
      let mut mutated: Vec<usize> = Vec::new();
      for &ord_idx in order.iter().take(pivot_idx) {
        let term = &mut terms[ord_idx];
        if use_block_bounds {
          let moved = term.skip_to_block(pivot_doc);
          with_stats(&mut stats, |s| s.postings_advanced += moved);
        }
        let moved = term.advance_to(pivot_doc);
        with_stats(&mut stats, |s| s.postings_advanced += moved);
        mutated.push(ord_idx);
      }
      requeue_terms(&mut order, &terms, &mut mutated);
    }
  }
  finalize_heap(heap)
}

fn requeue_terms(order: &mut Vec<usize>, terms: &[TermState], mutated: &mut Vec<usize>) {
  if mutated.is_empty() {
    return;
  }
  mutated.sort_unstable();
  order.retain(|idx| mutated.binary_search(idx).is_err() && !terms[*idx].is_done());
  for term_idx in mutated.drain(..) {
    if terms[term_idx].is_done() {
      continue;
    }
    let doc = terms[term_idx].doc_id();
    let pos = order
      .binary_search_by_key(&doc, |&i| terms[i].doc_id())
      .unwrap_or_else(|p| p);
    order.insert(pos, term_idx);
  }
}

fn choose_pivot(
  order: &[usize],
  terms: &[TermState],
  threshold: f32,
  use_block_bounds: bool,
) -> Option<usize> {
  let mut acc = 0.0_f32;
  for (i, &term_idx) in order.iter().enumerate() {
    let term = &terms[term_idx];
    let bound = if use_block_bounds {
      term.block_upper_bound()
    } else {
      term.upper_bound()
    };
    if !bound.is_finite() {
      continue;
    }
    acc += bound;
    if acc >= threshold {
      return Some(i);
    }
  }
  None
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

  fn term_from_entries(entries: &[PostingEntry]) -> ScoredTerm {
    let reader = PostingsReader::from_entries_for_test(entries.to_vec(), DEFAULT_BLOCK_SIZE);
    ScoredTerm {
      postings: reader,
      weight: 1,
      avgdl: 10.0,
      docs: 10.0,
      k1: 1.2,
      b: 0.75,
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
      &mut accept,
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
