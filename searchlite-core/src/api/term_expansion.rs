use hashbrown::HashSet;
use std::sync::Arc;

use anyhow::Result;
use regex::Regex;
use smallvec::{smallvec, SmallVec};

use crate::analysis::analyzer::Analyzer;
use crate::api::types::{FuzzyOptions, MultiMatchFuzziness};
use crate::index::manifest::{FieldKind, Schema, SchemaAnalyzers};
use crate::index::segment::SegmentReader;
use crate::query::planner::{TermExpansion, TermGroupMode, TermGroupSpec};
use crate::util::regex::anchored_regex;

use super::phrase::TermMatchGroup;

pub(super) const DEFAULT_SUGGEST_SCAN: usize = 64;
pub(super) const MAX_SUGGEST_CANDIDATES: usize = 256;

#[derive(Clone, Debug)]
pub(crate) struct QualifiedTerm {
  pub(crate) field: String,
  pub(crate) term: String,
  pub(crate) key: String,
  pub(crate) weight: f32,
  pub(crate) leaf: usize,
  pub(crate) group_fields: Option<Arc<Vec<String>>>,
}

pub(crate) type WeightedTermEntry = (String, f32, usize, Option<Arc<Vec<String>>>);

pub(super) fn build_term_key(field: &str, term: &str) -> String {
  let mut key = String::with_capacity(field.len() + term.len() + 1);
  key.push_str(field);
  key.push(':');
  key.push_str(term);
  key
}

/// Returns the prefix by Unicode scalar value count (not bytes).
pub(super) fn char_prefix(input: &str, len: usize) -> &str {
  if len == 0 {
    return "";
  }
  match input.char_indices().nth(len) {
    Some((idx, _)) => &input[..idx],
    None => input,
  }
}

pub(super) fn distance_weight(distance: usize) -> f32 {
  1.0 / (distance as f32 + 1.0)
}

fn auto_fuzziness_max_edits(term: &str) -> u8 {
  match term.chars().count() {
    0..=2 => 0,
    3..=5 => 1,
    _ => 2,
  }
}

fn resolve_multi_match_fuzzy_options(
  multi_match_fuzziness: Option<&MultiMatchFuzziness>,
  request_fuzzy: Option<&FuzzyOptions>,
  term: &str,
) -> Option<FuzzyOptions> {
  let Some(fuzziness) = multi_match_fuzziness else {
    return request_fuzzy.cloned();
  };
  let mut options = request_fuzzy.cloned().unwrap_or_default();
  options.max_edits = match fuzziness {
    MultiMatchFuzziness::Auto => auto_fuzziness_max_edits(term),
    MultiMatchFuzziness::Edits(value) => (*value).min(2),
  };
  Some(options)
}

pub(super) fn bounded_levenshtein(a: &str, b: &str, max_edits: usize) -> Option<usize> {
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

pub(crate) fn expand_term_groups(
  segments: &[SegmentReader],
  groups: &[TermGroupSpec],
  request_fuzzy: Option<&FuzzyOptions>,
  analysis: &SchemaAnalyzers,
  schema: &Schema,
) -> Result<(Vec<QualifiedTerm>, Vec<TermMatchGroup>)> {
  let mut qualified_terms = Vec::new();
  let mut term_groups = Vec::with_capacity(groups.len());
  for group in groups.iter() {
    let group_fields = if matches!(group.mode, TermGroupMode::CrossFields) {
      let mut deduped = Vec::with_capacity(group.fields.len());
      let mut seen = HashSet::new();
      for spec in group.fields.iter() {
        if seen.insert(spec.field.as_str()) {
          deduped.push(spec.field.clone());
        }
      }
      Some(Arc::new(deduped))
    } else {
      None
    };
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
              let term_fuzzy =
                resolve_multi_match_fuzzy_options(group.fuzziness.as_ref(), request_fuzzy, &token);
              let (scored, mut expanded_keys) = expand_term_for_group(
                segments,
                &field.field,
                &token,
                weight,
                group.score,
                target_leaf,
                term_fuzzy.as_ref(),
                &group.expansion,
                group_fields.clone(),
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
          let term_fuzzy =
            resolve_multi_match_fuzzy_options(group.fuzziness.as_ref(), request_fuzzy, &term);
          let (scored, mut expanded_keys) = expand_term_for_group(
            segments,
            &field.field,
            &term,
            weight,
            group.score,
            target_leaf,
            term_fuzzy.as_ref(),
            &group.expansion,
            group_fields.clone(),
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
  group_fields: Option<Arc<Vec<String>>>,
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
        return Ok(expand_term_exact(field, term, boost, leaf, group_fields));
      };
      let max_edits = fuzzy.max_edits.min(2) as usize;
      if max_edits == 0 {
        return Ok(expand_term_exact(field, term, boost, leaf, group_fields));
      }
      Ok(expand_term_fuzzy(
        segments,
        field,
        term,
        boost,
        leaf,
        fuzzy,
        group_fields,
      ))
    }
    TermExpansion::Prefix { max_expansions } => Ok(expand_prefix(
      segments,
      field,
      term,
      boost,
      score,
      leaf,
      *max_expansions,
      group_fields,
    )),
    TermExpansion::Wildcard { max_expansions } => expand_wildcard(
      segments,
      field,
      term,
      boost,
      score,
      leaf,
      *max_expansions,
      group_fields,
    ),
    TermExpansion::Regex { max_expansions } => expand_regex(
      segments,
      field,
      term,
      boost,
      score,
      leaf,
      *max_expansions,
      group_fields,
    ),
  }
}

#[allow(clippy::too_many_arguments)]
fn expand_prefix(
  segments: &[SegmentReader],
  field: &str,
  prefix: &str,
  boost: f32,
  score: bool,
  leaf: Option<usize>,
  max_expansions: usize,
  group_fields: Option<Arc<Vec<String>>>,
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
            group_fields: group_fields.clone(),
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

#[allow(clippy::too_many_arguments)]
fn expand_wildcard(
  segments: &[SegmentReader],
  field: &str,
  pattern: &str,
  boost: f32,
  score: bool,
  leaf: Option<usize>,
  max_expansions: usize,
  group_fields: Option<Arc<Vec<String>>>,
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
            group_fields: group_fields.clone(),
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

#[allow(clippy::too_many_arguments)]
fn expand_regex(
  segments: &[SegmentReader],
  field: &str,
  pattern: &str,
  boost: f32,
  score: bool,
  leaf: Option<usize>,
  max_expansions: usize,
  group_fields: Option<Arc<Vec<String>>>,
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
            group_fields: group_fields.clone(),
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
  group_fields: Option<Arc<Vec<String>>>,
) -> (Vec<QualifiedTerm>, Vec<String>) {
  let key = build_term_key(field, term);
  (
    vec![QualifiedTerm {
      field: field.to_string(),
      term: term.to_string(),
      key: key.clone(),
      weight: boost,
      leaf,
      group_fields,
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
  group_fields: Option<Arc<Vec<String>>>,
) -> (Vec<QualifiedTerm>, Vec<String>) {
  let term_len = term.chars().count();
  let exact_key = build_term_key(field, term);
  let mut qualified = vec![QualifiedTerm {
    field: field.to_string(),
    term: term.to_string(),
    key: exact_key.clone(),
    weight: boost * distance_weight(0),
    leaf,
    group_fields: group_fields.clone(),
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
          group_fields: group_fields.clone(),
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
