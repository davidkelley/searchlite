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
                if seen_keys.insert(key.to_owned()) {
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
/// To preserve that invariant, the walker must account for two regex
/// constructs that make the *last* accumulated character (or the whole
/// accumulated branch) optional — and therefore not a guaranteed prefix of
/// matching terms:
///
/// * Quantifiers that permit zero occurrences of the preceding atom
///   (`*`, `?`, `{0,…}`, `{,…}`, `{0}`). The preceding literal must be
///   dropped: e.g. `colou?r` requires prefix `colo`, not `colou`.
/// * Top-level alternation (`|`). No single literal prefix is shared by
///   every branch (e.g. `foo|bar` shares no common first character), so the
///   prefix must be cleared entirely.
///
/// Alternation, groups, and character classes *inside* `(`, `[`, or `{`
/// never affect the running prefix because the walker already breaks at
/// those metacharacters. Only top-level constructs reach this logic.
fn regex_literal_prefix(pattern: &str) -> String {
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
      '|' => {
        // Top-level alternation: no literal prefix is guaranteed to appear
        // across every branch, so invalidate anything accumulated so far.
        prefix.clear();
        break;
      }
      '.' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '$' => break,
      _ => {
        if quantifier_allows_zero(&chars, i + 1) {
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

/// Returns true when `chars[pos]` starts a quantifier that permits zero
/// occurrences of the preceding atom (`*`, `?`, `{0,…}`, `{,…}`, `{0}`).
/// Any other char, or a `{…}` whose lower bound is ≥ 1, returns false.
fn quantifier_allows_zero(chars: &[(usize, char)], pos: usize) -> bool {
  if pos >= chars.len() {
    return false;
  }
  match chars[pos].1 {
    '*' | '?' => true,
    '{' => {
      // Parse the minimum-occurrences digits; anything that isn't a valid
      // `{n…}` form is treated as "unknown" (conservative: return false).
      let digits_start = pos + 1;
      let mut j = digits_start;
      while j < chars.len() && chars[j].1.is_ascii_digit() {
        j += 1;
      }
      if j >= chars.len() {
        return false;
      }
      match chars[j].1 {
        ',' | '}' => {
          // Empty lower bound (e.g. `{,5}`) is treated as 0.
          if digits_start == j {
            return true;
          }
          let lower: u64 = chars[digits_start..j]
            .iter()
            .map(|(_, c)| *c)
            .collect::<String>()
            .parse()
            .unwrap_or(u64::MAX);
          lower == 0
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
  use super::{quantifier_allows_zero, regex_literal_prefix};
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
  fn top_level_alternation_clears_prefix() {
    // BUG-202 repro: `foo|bar` shares no common first character.
    assert_eq!(regex_literal_prefix("foo|bar"), "");
    assert_prefix_is_safe("foo|bar", &["foo", "bar"]);
    assert_eq!(regex_literal_prefix("rust|ruby"), "");
    assert_prefix_is_safe("rust|ruby", &["rust", "ruby"]);
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
    // Unterminated / malformed `{…` — conservatively false.
    assert!(!quantifier_allows_zero(&chars("{"), 0));
    assert!(!quantifier_allows_zero(&chars("{0"), 0));
    // Position past the end is false.
    assert!(!quantifier_allows_zero(&chars("?"), 1));
  }
}
