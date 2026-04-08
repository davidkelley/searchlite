use crate::index::fastfields::FastFieldsReader;
use crate::query::filters::passes_filters;
use crate::query::phrase::matches_phrase;
use crate::query::planner::QueryMatcher;
use crate::DocId;

use super::phrase::PhraseRuntime;

pub(crate) struct QueryEvaluator<'a> {
  pub(crate) matcher: &'a QueryMatcher,
  pub(crate) term_docs: &'a [Vec<DocId>],
  pub(crate) term_group_lists: &'a [Vec<usize>],
  pub(crate) phrase_postings: &'a [PhraseRuntime],
  pub(crate) fast_fields: &'a FastFieldsReader,
}

impl<'a> QueryEvaluator<'a> {
  pub(crate) fn matches(&self, doc_id: DocId) -> bool {
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

  pub(crate) fn matches_subquery(&self, matcher: &QueryMatcher, doc_id: DocId) -> bool {
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
