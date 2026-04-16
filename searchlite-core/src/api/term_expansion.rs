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
use crate::util::case_fold::fold_keyword;
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
              _ => analyze_pattern_tokens(analyzer, &group.term, &group.expansion),
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
                if seen_keys.insert(key.to_owned()) {
                  keys.push(key);
                }
              }
            }
          }
        }
        FieldKind::Keyword => {
          let term = fold_keyword(&group.term).into_owned();
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
            if seen_keys.insert(key.to_owned()) {
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

/// Returns `true` if `value` contains wildcard metacharacters (`*`, `?`).
fn contains_wildcard_meta(value: &str) -> bool {
  value.bytes().any(|b| matches!(b, b'*' | b'?'))
}

/// Returns `true` if `value` contains regex metacharacters that the default
/// tokenizer would discard (non-alphanumeric separators).
fn contains_regex_meta(value: &str) -> bool {
  value.bytes().any(|b| {
    matches!(
      b,
      b'*'
        | b'?'
        | b'.'
        | b'+'
        | b'['
        | b']'
        | b'('
        | b')'
        | b'{'
        | b'}'
        | b'|'
        | b'^'
        | b'$'
        | b'\\'
    )
  })
}

fn analyze_pattern_tokens(
  analyzer: &Analyzer,
  value: &str,
  expansion: &TermExpansion,
) -> Vec<String> {
  let tokens: Vec<String> = analyzer
    .analyze(value)
    .into_iter()
    .map(|t| t.text)
    .collect();
  if tokens.is_empty() {
    return vec![analyzer.normalize_pattern(value)];
  }
  if tokens.len() == 1 {
    // For wildcard/regex expansions: if the original value contains
    // metacharacters relevant to that expansion type, the analyzer will
    // have stripped them (they are non-alphanumeric token separators).
    // Fall back to normalize_pattern to preserve the pattern structure.
    // Wildcard only uses `*` and `?`; regex uses the full set. For prefix
    // expansions and plain terms we keep the analyzed form so that stemming,
    // Unicode folding, and other filters are honoured.
    let has_meta = match expansion {
      TermExpansion::Wildcard { .. } => contains_wildcard_meta(value),
      TermExpansion::Regex { .. } => contains_regex_meta(value),
      _ => false,
    };
    if has_meta {
      return vec![analyzer.normalize_pattern(value)];
    }
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
      if !seen.insert(key.to_owned()) {
        continue;
      }
      let term = key[field_prefix_len..].to_string();
      if score {
        if let Some(idx) = leaf {
          qualified.push(QualifiedTerm {
            field: field.to_string(),
            term: term.clone(),
            key: key.to_owned(),
            weight: boost,
            leaf: idx,
            group_fields: group_fields.clone(),
          });
        }
      }
      keys.push(key.to_owned());
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
      if !seen.insert(key.to_owned()) {
        continue;
      }
      if score {
        if let Some(idx) = leaf {
          qualified.push(QualifiedTerm {
            field: field.to_string(),
            term: term.to_string(),
            key: key.to_owned(),
            weight: boost,
            leaf: idx,
            group_fields: group_fields.clone(),
          });
        }
      }
      keys.push(key.to_owned());
      expanded += 1;
    }
  }
  Ok((qualified, keys))
}

/// Extracts a safe literal prefix from a regex pattern for use as a
/// term-dictionary scan bound.
///
/// The returned prefix is guaranteed to be a prefix of every string the
/// compiled regex can match (anchored via [`anchored_regex`]), so
/// `terms_with_prefix(field:<prefix>)` can be used to skip terms that
/// could not possibly match without missing any that could.
///
/// To preserve that invariant, the pattern is split at top-level `|` into
/// alternation branches, a per-branch safe prefix is extracted from each,
/// and the greatest common prefix of those is returned. Cases handled:
///
/// * Quantifiers that permit zero occurrences of the preceding atom
///   (`*`, `?`, `{0,…}`, `{,…}`, `{0}`) drop the preceding literal inside
///   a branch: e.g. `colou?r` yields branch prefix `colo`, not `colou`.
/// * Top-level alternation returns the LCP across branches. Simple shared
///   prefixes are preserved (e.g. `rust|ruby` → `ru`, `abc|ab` → `ab`),
///   while branches with no shared first character collapse to `""`
///   (e.g. `foo|bar` → `""`).
///
/// Alternation, groups, and character classes *inside* `(`, `[`, or `{`
/// never affect a branch's running prefix because the walker breaks at
/// those metacharacters. Only top-level constructs reach this logic.
fn regex_literal_prefix(pattern: &str) -> String {
  let mut branches = split_top_level_alternatives(pattern);
  let first = match branches.next() {
    Some(b) => b,
    None => return String::new(),
  };
  let mut prefix = regex_branch_literal_prefix(first);
  for branch in branches {
    if prefix.is_empty() {
      return prefix;
    }
    let next = regex_branch_literal_prefix(branch);
    let lcp_len = longest_common_prefix_len(&prefix, &next);
    prefix.truncate(lcp_len);
  }
  prefix
}

/// Extracts a safe literal prefix from a single alternation branch (i.e.
/// a pattern with no top-level `|`). See [`regex_literal_prefix`] for the
/// invariant the returned prefix upholds.
fn regex_branch_literal_prefix(pattern: &str) -> String {
  let chars: Vec<(usize, char)> = pattern.char_indices().collect();
  let mut prefix = String::new();
  let mut escaped = false;
  let mut i = 0usize;
  while i < chars.len() {
    let (pos, ch) = chars[i];
    if escaped {
      match ch {
        '\\' => {
          // Escaped backslash is a literal backslash. If a zero-permitting
          // quantifier follows, the backslash is optional and must not be
          // committed to the prefix.
          if quantifier_allows_zero(&chars, i + 1) {
            break;
          }
          let end = pos + ch.len_utf8();
          prefix.push_str(&pattern[pos..end]);
          escaped = false;
          i += 1;
          continue;
        }
        // Escape classes/boundaries mean we cannot keep extending the literal prefix.
        'd' | 'D' | 'w' | 'W' | 's' | 'S' | 'b' | 'B' => break,
        'p' | 'P' => break,
        _ => {
          // Escaped literal (e.g. `\.`, `\+`, `\?`). The same
          // quantifier-lookahead rule applies.
          if quantifier_allows_zero(&chars, i + 1) {
            break;
          }
          let end = pos + ch.len_utf8();
          prefix.push_str(&pattern[pos..end]);
          escaped = false;
          i += 1;
          continue;
        }
      }
    }
    match ch {
      '\\' => {
        escaped = true;
        i += 1;
      }
      '^' if prefix.is_empty() => {
        i += 1;
      }
      // Any top-level `|` should have been consumed by the caller's
      // splitter, but guard against it as a defensive break.
      '|' | '.' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '$' => break,
      _ => {
        if quantifier_allows_zero(&chars, i + 1) {
          // The preceding literal `ch` is optional, so it cannot extend
          // the prefix on its own. But if the next required atom is the
          // same literal `ch` (e.g. `ab?b`, `a*a`, `a{0,3}a`), then that
          // character is guaranteed at the current position regardless
          // of whether the optional was taken, so we can commit one
          // copy before stopping. We must stop here because the two
          // branches diverge in position after this, so subsequent
          // pattern chars no longer line up across both cases.
          if let Some(commit) = collapsed_literal_after_optional(&chars, i) {
            let end = commit + ch.len_utf8();
            prefix.push_str(&pattern[commit..end]);
          }
          break;
        }
        let end = pos + ch.len_utf8();
        prefix.push_str(&pattern[pos..end]);
        i += 1;
      }
    }
  }
  prefix
}

/// When `chars[i]` is a literal `x` followed by a zero-permitting
/// quantifier, checks whether the atom immediately after the quantifier
/// is also a required literal `x`. If so, returns the byte offset of
/// that second `x` (so the caller can commit it to the prefix). Returns
/// `None` otherwise.
///
/// The required-literal check:
///
/// * Must be a plain literal char (no backslash-escape, no metachar),
///   because anything else (wildcards, classes, groups, alternation)
///   doesn't guarantee a single specific byte at that position.
/// * Must not itself be followed by a zero-permitting quantifier — if
///   the second `x` is also optional, it isn't guaranteed to appear.
fn collapsed_literal_after_optional(chars: &[(usize, char)], i: usize) -> Option<usize> {
  let ch = chars.get(i)?.1;
  let q_end = quantifier_end(chars, i + 1)?;
  let (pos, next) = *chars.get(q_end)?;
  if !is_plain_literal(next) || next != ch {
    return None;
  }
  if quantifier_allows_zero(chars, q_end + 1) {
    return None;
  }
  Some(pos)
}

/// Returns the index in `chars` immediately after the quantifier that
/// starts at `pos`, or `None` if there is no quantifier at `pos`.
fn quantifier_end(chars: &[(usize, char)], pos: usize) -> Option<usize> {
  match chars.get(pos)?.1 {
    '*' | '?' | '+' => Some(pos + 1),
    '{' => {
      // Scan for the matching `}`; quantifier bodies contain only
      // digits and at most one `,`, but we stay tolerant and just look
      // for the terminator.
      let mut k = pos + 1;
      while k < chars.len() && chars[k].1 != '}' {
        k += 1;
      }
      if k < chars.len() {
        Some(k + 1)
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Returns true when `c` can safely be compared as a literal byte for
/// the `x?x`-style prefix extension — i.e. not a regex metacharacter,
/// escape prefix, or anchor.
fn is_plain_literal(c: char) -> bool {
  !matches!(
    c,
    '\\' | '.' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '$' | '^'
  )
}

/// Splits `pattern` at every `|` that sits at the top level — i.e. not
/// inside `(…)`, `[…]`, `{…}`, and not the target of a `\` escape. Returns
/// an iterator that yields at least one branch (the empty pattern yields a
/// single empty branch).
fn split_top_level_alternatives(pattern: &str) -> impl Iterator<Item = &str> {
  let mut out: SmallVec<[&str; 2]> = SmallVec::new();
  let mut escaped = false;
  let mut paren: i32 = 0;
  let mut bracket: i32 = 0;
  let mut brace: i32 = 0;
  let mut start = 0usize;
  for (i, ch) in pattern.char_indices() {
    if escaped {
      escaped = false;
      continue;
    }
    match ch {
      '\\' => escaped = true,
      '(' => paren = paren.saturating_add(1),
      ')' => paren = (paren - 1).max(0),
      '[' => bracket = bracket.saturating_add(1),
      ']' => bracket = (bracket - 1).max(0),
      '{' => brace = brace.saturating_add(1),
      '}' => brace = (brace - 1).max(0),
      '|' if paren == 0 && bracket == 0 && brace == 0 => {
        out.push(&pattern[start..i]);
        start = i + ch.len_utf8();
      }
      _ => {}
    }
  }
  out.push(&pattern[start..]);
  out.into_iter()
}

/// Returns the byte length of the longest common prefix of `a` and `b`,
/// measured at a UTF-8 character boundary. `a[..n]` and `b[..n]` are both
/// valid UTF-8 substrings of their respective inputs.
fn longest_common_prefix_len(a: &str, b: &str) -> usize {
  let mut len = 0usize;
  for (ac, bc) in a.char_indices().zip(b.chars()) {
    if ac.1 == bc {
      len = ac.0 + ac.1.len_utf8();
    } else {
      break;
    }
  }
  len
}

/// Returns true when `chars[pos]` starts a quantifier that permits zero
/// occurrences of the preceding atom (`*`, `?`, `{0,…}`, `{,…}`, `{0}`).
///
/// `{…}` forms must be properly terminated with `}`; unterminated cases
/// like `{0,` or `{,5` return false. Non-zero lower bounds (`{1,3}`,
/// `{5}`, …) also return false so the preceding literal is preserved.
fn quantifier_allows_zero(chars: &[(usize, char)], pos: usize) -> bool {
  if pos >= chars.len() {
    return false;
  }
  match chars[pos].1 {
    '*' | '?' => true,
    '{' => {
      // Parse the minimum-occurrences digits without allocating. An
      // overflowing or non-digit body disqualifies the quantifier from
      // being zero-permitting (conservative: return false).
      let digits_start = pos + 1;
      let mut j = digits_start;
      while j < chars.len() && chars[j].1.is_ascii_digit() {
        j += 1;
      }
      if j >= chars.len() {
        return false;
      }
      let lower_is_zero = if digits_start == j {
        // Empty lower bound (e.g. `{,5}`) is treated as 0.
        true
      } else {
        let mut lower: u64 = 0;
        let mut ok = true;
        for (_, c) in &chars[digits_start..j] {
          let digit = match c.to_digit(10) {
            Some(d) => d as u64,
            None => {
              ok = false;
              break;
            }
          };
          match lower.checked_mul(10).and_then(|n| n.checked_add(digit)) {
            Some(v) => lower = v,
            None => {
              ok = false;
              break;
            }
          }
        }
        ok && lower == 0
      };
      match chars[j].1 {
        '}' => lower_is_zero,
        ',' => {
          // Skip optional upper-bound digits and require a closing `}`
          // terminator; anything else is malformed and disqualifying.
          let mut k = j + 1;
          while k < chars.len() && chars[k].1.is_ascii_digit() {
            k += 1;
          }
          if k < chars.len() && chars[k].1 == '}' {
            lower_is_zero
          } else {
            false
          }
        }
        _ => false,
      }
    }
    _ => false,
  }
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
      if !seen.insert(key.to_owned()) {
        continue;
      }
      if score {
        if let Some(idx) = leaf {
          qualified.push(QualifiedTerm {
            field: field.to_string(),
            term: term.to_string(),
            key: key.to_owned(),
            weight: boost,
            leaf: idx,
            group_fields: group_fields.clone(),
          });
        }
      }
      keys.push(key.to_owned());
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
      key: key.to_owned(),
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
    key: exact_key.to_owned(),
    weight: boost * distance_weight(0),
    leaf,
    group_fields: group_fields.clone(),
  }];
  let mut keys = vec![exact_key.to_owned()];
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
      if seen.insert(key.to_owned()) {
        qualified.push(QualifiedTerm {
          field: field.to_string(),
          term: candidate.to_string(),
          key: key.to_owned(),
          weight: boost * distance_weight(distance),
          leaf,
          group_fields: group_fields.clone(),
        });
        keys.push(key.to_owned());
        expansions += 1;
        if expansions >= fuzzy.max_expansions {
          break 'segments;
        }
      }
    }
  }
  (qualified, keys)
}

#[cfg(test)]
mod tests {
  use super::{
    analyze_pattern_tokens, contains_regex_meta, contains_wildcard_meta, quantifier_allows_zero,
    regex_literal_prefix,
  };
  use crate::analysis::analyzer::AnalyzerRegistry;
  use crate::query::planner::TermExpansion;
  use crate::util::regex::anchored_regex;

  /// Pins the invariant the whole helper exists to uphold: every term the
  /// anchored regex could match must start with the returned prefix. If this
  /// property regresses, `expand_regex` silently drops matching terms.
  fn assert_prefix_is_safe(pattern: &str, matching: &[&str]) {
    let regex = anchored_regex(pattern).expect("valid regex");
    let prefix = regex_literal_prefix(pattern);
    for term in matching {
      assert!(
        regex.is_match(term),
        "test bug: pattern `{pattern}` must match `{term}`"
      );
      assert!(
        term.starts_with(&prefix),
        "prefix `{prefix}` is not a prefix of matching term `{term}` (pattern `{pattern}`)"
      );
    }
  }

  #[test]
  fn plain_literal_is_kept() {
    assert_eq!(regex_literal_prefix("color"), "color");
    assert_prefix_is_safe("color", &["color"]);
  }

  #[test]
  fn caret_anchor_is_stripped_from_prefix() {
    assert_eq!(regex_literal_prefix("^color"), "color");
    assert_prefix_is_safe("^color", &["color"]);
  }

  #[test]
  fn optional_char_is_not_committed() {
    // BUG-202 repro: `colou?r` must match both `color` and `colour`, so the
    // literal prefix cannot extend past `colo`.
    assert_eq!(regex_literal_prefix("colou?r"), "colo");
    assert_prefix_is_safe("colou?r", &["color", "colour"]);
  }

  #[test]
  fn question_at_end_trims_last_char() {
    assert_eq!(regex_literal_prefix("ab?"), "a");
    assert_prefix_is_safe("ab?", &["a", "ab"]);
  }

  #[test]
  fn star_quantifier_trims_last_char() {
    // `foo*` matches `fo`, `foo`, `fooo`, ... — common prefix is `fo`.
    assert_eq!(regex_literal_prefix("foo*"), "fo");
    assert_prefix_is_safe("foo*", &["fo", "foo", "fooo"]);
  }

  #[test]
  fn plus_quantifier_keeps_last_char() {
    // `+` requires at least one occurrence, so `foo+` still implies `foo`.
    assert_eq!(regex_literal_prefix("foo+"), "foo");
    assert_prefix_is_safe("foo+", &["foo", "fooo"]);
  }

  #[test]
  fn bounded_quantifier_with_zero_lower_trims_last_char() {
    assert_eq!(regex_literal_prefix("foo{0,3}"), "fo");
    assert_prefix_is_safe("foo{0,3}", &["fo", "foo", "fooo"]);
  }

  #[test]
  fn bounded_quantifier_with_empty_lower_trims_last_char() {
    // `{,n}` isn't accepted by the regex crate, but treating it as
    // zero-permitting is still the correct conservative choice: if a future
    // parser accepts it, the returned prefix must stay safe. So just assert
    // the computed prefix directly (no compilation through `anchored_regex`).
    assert_eq!(regex_literal_prefix("foo{,3}"), "fo");
  }

  #[test]
  fn bounded_quantifier_with_zero_exact_trims_last_char() {
    // `{0}` means "zero occurrences of the preceding atom" — the last char
    // is effectively removed. We still stop there to stay safe.
    assert_eq!(regex_literal_prefix("foo{0}"), "fo");
    assert_prefix_is_safe("foo{0}", &["fo"]);
  }

  #[test]
  fn bounded_quantifier_with_nonzero_lower_keeps_last_char() {
    // `{1,3}` and `{5}` both require at least one occurrence, so the last
    // literal char is still guaranteed to appear.
    assert_eq!(regex_literal_prefix("foo{1,3}"), "foo");
    assert_prefix_is_safe("foo{1,3}", &["foo", "fooo"]);
    assert_eq!(regex_literal_prefix("foo{5}"), "foo");
    assert_prefix_is_safe("foo{5}", &["foooooo"]);
  }

  #[test]
  fn optional_followed_by_matching_required_keeps_one_copy() {
    // When an optional atom is followed by the same required literal
    // (e.g. `ab?b`, `a*a`, `a{0,3}a`), the position after the earlier
    // literals is guaranteed to be that char regardless of whether the
    // optional was taken. We commit one copy and stop.
    assert_eq!(regex_literal_prefix("ab?b"), "ab");
    assert_prefix_is_safe("ab?b", &["ab", "abb"]);
    assert_eq!(regex_literal_prefix("a*a"), "a");
    assert_prefix_is_safe("a*a", &["a", "aa", "aaa"]);
    assert_eq!(regex_literal_prefix("a?a"), "a");
    assert_prefix_is_safe("a?a", &["a", "aa"]);
    assert_eq!(regex_literal_prefix("ab{0,3}b"), "ab");
    assert_prefix_is_safe("ab{0,3}b", &["ab", "abb", "abbb", "abbbb"]);
  }

  #[test]
  fn optional_followed_by_different_required_stops() {
    // `ab?c` diverges at position 1 (`c` vs `b`), so the prefix is just
    // `a`. `a*b` shares nothing, so prefix is empty.
    assert_eq!(regex_literal_prefix("ab?c"), "a");
    assert_prefix_is_safe("ab?c", &["ac", "abc"]);
    assert_eq!(regex_literal_prefix("a*b"), "");
    assert_prefix_is_safe("a*b", &["b", "ab", "aab"]);
  }

  #[test]
  fn optional_followed_by_optional_does_not_collapse() {
    // The required-literal check must reject the case where the char
    // after the first quantifier is itself optional (`ab?b?c`). The
    // second `b` isn't guaranteed to appear, so we can't commit it.
    assert_eq!(regex_literal_prefix("ab?b?c"), "a");
    assert_prefix_is_safe("ab?b?c", &["ac", "abc", "abbc"]);
  }

  #[test]
  fn optional_followed_by_metachar_stops() {
    // `.`, `\`, `[`, `(` etc. after the quantifier don't represent a
    // concrete literal, so no collapse can be applied.
    assert_eq!(regex_literal_prefix("ab?."), "a");
    assert_prefix_is_safe("ab?.", &["ax", "abx"]);
    assert_eq!(regex_literal_prefix("ab?\\."), "a");
    assert_prefix_is_safe("ab?\\.", &["a.", "ab."]);
  }

  #[test]
  fn top_level_alternation_returns_branch_lcp() {
    // BUG-202 repro: branches with no shared first character collapse to
    // an empty prefix.
    assert_eq!(regex_literal_prefix("foo|bar"), "");
    assert_prefix_is_safe("foo|bar", &["foo", "bar"]);
    // Branches that *do* share a literal prefix keep that shared prefix,
    // so the term-dictionary scan still benefits from a bound.
    assert_eq!(regex_literal_prefix("rust|ruby"), "ru");
    assert_prefix_is_safe("rust|ruby", &["rust", "ruby"]);
    assert_eq!(regex_literal_prefix("abc|ab"), "ab");
    assert_prefix_is_safe("abc|ab", &["abc", "ab"]);
    // More than two branches: LCP collapses to the shortest shared prefix.
    assert_eq!(regex_literal_prefix("rust|ruby|rails"), "r");
    assert_prefix_is_safe("rust|ruby|rails", &["rust", "ruby", "rails"]);
    assert_eq!(regex_literal_prefix("alpha|beta|alp"), "");
    assert_prefix_is_safe("alpha|beta|alp", &["alpha", "beta", "alp"]);
  }

  #[test]
  fn alternation_respects_escaped_pipe() {
    // An escaped `|` is a literal pipe character, not an alternation
    // operator, so the whole pattern is a single branch.
    assert_eq!(regex_literal_prefix("foo\\|bar"), "foo|bar");
    assert_prefix_is_safe("foo\\|bar", &["foo|bar"]);
  }

  #[test]
  fn alternation_ignores_pipe_inside_groups() {
    // `|` inside `(`, `[`, or `{` is part of the group, not a top-level
    // alternation, so the outer branch prefix is preserved.
    assert_eq!(regex_literal_prefix("foo(ab|cd)"), "foo");
    assert_eq!(regex_literal_prefix("foo[a|b]"), "foo");
  }

  #[test]
  fn grouped_alternation_keeps_outer_literal_prefix() {
    // `(` terminates the walk, so everything before it is preserved. The
    // existing `r(ust|uby)` expansion test relies on this.
    assert_eq!(regex_literal_prefix("r(ust|uby)"), "r");
    assert_prefix_is_safe("r(ust|uby)", &["rust", "ruby"]);
  }

  #[test]
  fn character_class_terminates_walk() {
    assert_eq!(regex_literal_prefix("fo[ou]"), "fo");
    assert_prefix_is_safe("fo[ou]", &["foo", "fou"]);
  }

  #[test]
  fn escaped_literal_followed_by_optional_is_dropped() {
    // `colou\??` is `colou` + literal `?` + optional-quantifier on that `?`.
    // The literal `?` is therefore optional, so it cannot extend the prefix.
    assert_eq!(regex_literal_prefix("colou\\??"), "colou");
    assert_prefix_is_safe("colou\\??", &["colou", "colou?"]);
  }

  #[test]
  fn escaped_literal_without_quantifier_is_kept() {
    assert_eq!(regex_literal_prefix("foo\\.bar"), "foo.bar");
    assert_prefix_is_safe("foo\\.bar", &["foo.bar"]);
  }

  #[test]
  fn escape_class_terminates_walk() {
    assert_eq!(regex_literal_prefix("foo\\d"), "foo");
    assert_prefix_is_safe("foo\\d", &["foo0", "foo9"]);
  }

  #[test]
  fn dollar_anchor_terminates_walk() {
    assert_eq!(regex_literal_prefix("foo$"), "foo");
    assert_prefix_is_safe("foo$", &["foo"]);
  }

  #[test]
  fn dot_terminates_walk() {
    assert_eq!(regex_literal_prefix("foo.bar"), "foo");
    assert_prefix_is_safe("foo.bar", &["fooxbar", "foo!bar"]);
  }

  #[test]
  fn unicode_literal_is_preserved() {
    assert_eq!(regex_literal_prefix("café"), "café");
    assert_prefix_is_safe("café", &["café"]);
  }

  #[test]
  fn quantifier_allows_zero_covers_bounded_forms() {
    fn chars(pattern: &str) -> Vec<(usize, char)> {
      pattern.char_indices().collect()
    }
    // Position is the *quantifier* char (or `{`), not the preceding atom.
    assert!(quantifier_allows_zero(&chars("?"), 0));
    assert!(quantifier_allows_zero(&chars("*"), 0));
    assert!(!quantifier_allows_zero(&chars("+"), 0));
    assert!(quantifier_allows_zero(&chars("{0}"), 0));
    assert!(quantifier_allows_zero(&chars("{0,3}"), 0));
    assert!(quantifier_allows_zero(&chars("{0,}"), 0));
    assert!(quantifier_allows_zero(&chars("{,5}"), 0));
    assert!(!quantifier_allows_zero(&chars("{1}"), 0));
    assert!(!quantifier_allows_zero(&chars("{1,3}"), 0));
    assert!(!quantifier_allows_zero(&chars("{2,}"), 0));
    assert!(!quantifier_allows_zero(&chars("foo"), 0));
    // Overflowing lower bound disqualifies.
    assert!(!quantifier_allows_zero(&chars("{99999999999999999999}"), 0));
    // Unterminated / malformed `{…` — conservatively false. All of these
    // are missing a closing `}`.
    assert!(!quantifier_allows_zero(&chars("{"), 0));
    assert!(!quantifier_allows_zero(&chars("{0"), 0));
    assert!(!quantifier_allows_zero(&chars("{0,"), 0));
    assert!(!quantifier_allows_zero(&chars("{0,3"), 0));
    assert!(!quantifier_allows_zero(&chars("{,5"), 0));
    assert!(!quantifier_allows_zero(&chars("{,"), 0));
    // Non-digit body disqualifies.
    assert!(!quantifier_allows_zero(&chars("{abc}"), 0));
    // Position past the end is false.
    assert!(!quantifier_allows_zero(&chars("?"), 1));
  }

  #[test]
  fn analyze_pattern_tokens_preserves_trailing_wildcard() {
    let registry = AnalyzerRegistry::with_default();
    let analyzer = registry.get("default").unwrap();
    let wildcard = TermExpansion::Wildcard {
      max_expansions: 100,
    };
    // Trailing wildcard: the analyzer absorbs "hello" as a single token and
    // discards the `*`. The fix must preserve it via normalize_pattern.
    let tokens = analyze_pattern_tokens(analyzer, "hello*", &wildcard);
    assert_eq!(tokens, vec!["hello*"]);
  }

  #[test]
  fn analyze_pattern_tokens_preserves_leading_wildcard() {
    let registry = AnalyzerRegistry::with_default();
    let analyzer = registry.get("default").unwrap();
    let wildcard = TermExpansion::Wildcard {
      max_expansions: 100,
    };
    let tokens = analyze_pattern_tokens(analyzer, "*world", &wildcard);
    assert_eq!(tokens, vec!["*world"]);
  }

  #[test]
  fn analyze_pattern_tokens_preserves_trailing_question_mark() {
    let registry = AnalyzerRegistry::with_default();
    let analyzer = registry.get("default").unwrap();
    let wildcard = TermExpansion::Wildcard {
      max_expansions: 100,
    };
    let tokens = analyze_pattern_tokens(analyzer, "hel?", &wildcard);
    assert_eq!(tokens, vec!["hel?"]);
  }

  #[test]
  fn analyze_pattern_tokens_preserves_regex_metacharacters() {
    let registry = AnalyzerRegistry::with_default();
    let analyzer = registry.get("default").unwrap();
    let regex = TermExpansion::Regex {
      max_expansions: 100,
    };
    // Regex pattern: `r.+` → tokenizer produces ["r"] (one token) and strips
    // the `.+`. The fix must fall back to normalize_pattern to preserve them.
    let tokens = analyze_pattern_tokens(analyzer, "r.+", &regex);
    assert_eq!(tokens, vec!["r.+"]);
  }

  #[test]
  fn analyze_pattern_tokens_returns_analyzed_form_for_plain_term() {
    let registry = AnalyzerRegistry::with_default();
    let analyzer = registry.get("default").unwrap();
    let wildcard = TermExpansion::Wildcard {
      max_expansions: 100,
    };
    // A plain term with no metacharacters should still return the analyzed form.
    let tokens = analyze_pattern_tokens(analyzer, "hello", &wildcard);
    assert_eq!(tokens, vec!["hello"]);
  }

  #[test]
  fn analyze_pattern_tokens_middle_wildcard_uses_normalize_pattern() {
    let registry = AnalyzerRegistry::with_default();
    let analyzer = registry.get("default").unwrap();
    let wildcard = TermExpansion::Wildcard {
      max_expansions: 100,
    };
    // Middle wildcard: produces multiple tokens → falls through to
    // normalize_pattern. This should still work as before.
    let tokens = analyze_pattern_tokens(analyzer, "r*st", &wildcard);
    assert_eq!(tokens, vec!["r*st"]);
  }

  #[test]
  fn analyze_pattern_tokens_prefix_keeps_analyzed_form_with_metacharacters() {
    let registry = AnalyzerRegistry::with_default();
    let analyzer = registry.get("default").unwrap();
    let prefix = TermExpansion::Prefix {
      max_expansions: 100,
    };
    // Prefix expansion with metachar-containing input (e.g. "c++") should
    // return the analyzed form, not the normalized pattern, so prefix search
    // matches indexed terms.
    let tokens = analyze_pattern_tokens(analyzer, "c++", &prefix);
    // The default analyzer tokenizes "c++" as ["c"] (stripping `++`).
    assert_eq!(tokens, vec!["c"]);
  }

  #[test]
  fn contains_wildcard_meta_only_detects_star_and_question() {
    assert!(contains_wildcard_meta("hello*"));
    assert!(contains_wildcard_meta("*world"));
    assert!(contains_wildcard_meta("hel?o"));
    // Regex-only metacharacters are NOT wildcard metacharacters.
    assert!(!contains_wildcard_meta("r.+"));
    assert!(!contains_wildcard_meta("c++"));
    assert!(!contains_wildcard_meta("node.js"));
    assert!(!contains_wildcard_meta("a[bc]d"));
    assert!(!contains_wildcard_meta("hello"));
    assert!(!contains_wildcard_meta(""));
  }

  #[test]
  fn contains_regex_meta_detects_full_metachar_set() {
    assert!(contains_regex_meta("hello*"));
    assert!(contains_regex_meta("hel?o"));
    assert!(contains_regex_meta("r.+"));
    assert!(contains_regex_meta("a[bc]d"));
    assert!(contains_regex_meta("^start"));
    assert!(contains_regex_meta("end$"));
    assert!(contains_regex_meta("a\\b"));
    assert!(contains_regex_meta("a|b"));
    assert!(contains_regex_meta("(group)"));
    assert!(contains_regex_meta("a{2,3}"));
    assert!(!contains_regex_meta("hello"));
    assert!(!contains_regex_meta("running"));
    assert!(!contains_regex_meta("café"));
    assert!(!contains_regex_meta(""));
  }

  #[test]
  fn analyze_pattern_tokens_wildcard_keeps_analyzed_form_for_non_wildcard_meta() {
    let registry = AnalyzerRegistry::with_default();
    let analyzer = registry.get("default").unwrap();
    let wildcard = TermExpansion::Wildcard {
      max_expansions: 100,
    };
    // "node.js" contains `.` which is a regex meta but NOT a wildcard meta.
    // Wildcard expansion should return the analyzed form, not normalize_pattern.
    let tokens = analyze_pattern_tokens(analyzer, "node.js", &wildcard);
    // The default analyzer tokenizes "node.js" as ["node", "js"] (two tokens)
    // → multi-token path → normalize_pattern("node.js") = "node.js".
    assert_eq!(tokens, vec!["node.js"]);
  }

  #[test]
  fn analyze_pattern_tokens_keeps_analyzed_form_when_no_metacharacters() {
    // Simulates the concern raised in review: when an analyzer transforms a
    // plain term (e.g. via stemming or Unicode folding), the analyzed token
    // differs from normalize_pattern(value). We must return the analyzed form,
    // not the normalized one, so that expansion matches indexed terms.
    // The default analyzer lowercases but does not stem, so "HELLO" → "hello"
    // matches normalize_pattern("HELLO") = "hello". This confirms the plain-
    // term path returns the analyzed form.
    let registry = AnalyzerRegistry::with_default();
    let analyzer = registry.get("default").unwrap();
    let wildcard = TermExpansion::Wildcard {
      max_expansions: 100,
    };
    let tokens = analyze_pattern_tokens(analyzer, "HELLO", &wildcard);
    assert_eq!(tokens, vec!["hello"]);
  }
}
