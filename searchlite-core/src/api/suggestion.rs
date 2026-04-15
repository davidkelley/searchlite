use hashbrown::HashMap;
use std::cmp::Ordering;
use std::collections::BTreeMap;

use anyhow::{bail, Result};

use super::term_expansion::{
  bounded_levenshtein, build_term_key, char_prefix, distance_weight, DEFAULT_SUGGEST_SCAN,
  MAX_SUGGEST_CANDIDATES,
};
use crate::api::reader::IndexReader;
use crate::api::types::{FuzzyOptions, SuggestOption, SuggestRequest, SuggestResult};
use crate::index::manifest::FieldKind;
use crate::index::segment::SegmentReader;
use crate::util::case_fold::fold_keyword;

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
  let mut expanded_total: usize = 0;
  match fuzzy {
    None => {
      let prefix_key = build_term_key(field, term);
      let field_prefix_len = field.len() + 1;
      for seg in segments.iter() {
        for key in seg.terms_with_prefix(&prefix_key) {
          if expanded_total >= max_candidates {
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
          expanded_total += 1;
          if expanded_total >= max_candidates {
            break;
          }
        }
        if expanded_total >= max_candidates {
          break;
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
      let mut global_cap = fuzzy.max_expansions.min(MAX_SUGGEST_CANDIDATES);
      global_cap = global_cap.max(size);
      for seg in segments.iter() {
        for key in seg.terms_with_prefix(&prefix_key) {
          if expanded_total >= global_cap {
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
          expanded_total += 1;
          if expanded_total >= global_cap {
            break;
          }
        }
        if expanded_total >= global_cap {
          break;
        }
      }
    }
  }
  out
}

impl IndexReader {
  pub(crate) fn completion_inputs(&self, field: &str, prefix: &str) -> Result<Vec<String>> {
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
      FieldKind::Keyword => Ok(vec![fold_keyword(prefix).into_owned()]),
      FieldKind::Numeric | FieldKind::Unknown => {
        bail!("completion suggest is only supported on text/keyword fields")
      }
    }
  }

  pub(crate) fn completion_suggest(
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

  pub(crate) fn execute_suggest(
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
}
