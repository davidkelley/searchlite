use regex::RegexBuilder;

pub struct HighlightOptions<'a> {
  pub pre_tag: &'a str,
  pub post_tag: &'a str,
  pub fragment_size: usize,
  pub number_of_fragments: usize,
}

/// Highlight terms and phrase sequences using a phrase-aware, token-boundary regex.
pub fn highlight_fragments(
  text: &str,
  terms: &[String],
  phrases: &[Vec<String>],
  opts: HighlightOptions<'_>,
) -> Vec<String> {
  if text.is_empty() || (terms.is_empty() && phrases.is_empty()) {
    return Vec::new();
  }
  let mut patterns: Vec<String> = Vec::new();
  // Phrase patterns first to prefer longer matches.
  for phrase in phrases.iter() {
    if phrase.is_empty() {
      continue;
    }
    let joined = phrase
      .iter()
      .map(|p| regex::escape(p))
      .collect::<Vec<_>>()
      .join(r"\W+");
    patterns.push(format!(r"\b{joined}\b"));
  }
  for term in terms.iter() {
    if term.is_empty() {
      continue;
    }
    patterns.push(format!(r"\b{}\b", regex::escape(term)));
  }
  if patterns.is_empty() {
    return Vec::new();
  }
  let pattern = patterns.join("|");
  let Ok(re) = RegexBuilder::new(&pattern).case_insensitive(true).build() else {
    return Vec::new();
  };
  let mut out = Vec::new();
  let mut offset = 0usize;
  for _ in 0..opts.number_of_fragments {
    if let Some(m) = re.find_at(text, offset) {
      let raw_start = m.start().saturating_sub(opts.fragment_size / 2);
      let raw_end = usize::min(text.len(), raw_start.saturating_add(opts.fragment_size));
      // Snap to character boundaries to avoid slicing mid-character.
      let start = snap_char_boundary_left(text, raw_start);
      let end = snap_char_boundary_right(text, raw_end);
      let fragment = text[start..end].to_string();
      let highlighted = re
        .replace_all(&fragment, |caps: &regex::Captures<'_>| {
          format!("{}{}{}", opts.pre_tag, &caps[0], opts.post_tag)
        })
        .into_owned();
      out.push(highlighted);
      offset = m.end();
    } else {
      break;
    }
  }
  out
}

/// Snap a byte index to the nearest char boundary at or before `idx`.
fn snap_char_boundary_left(text: &str, idx: usize) -> usize {
  let mut i = idx.min(text.len());
  while i > 0 && !text.is_char_boundary(i) {
    i -= 1;
  }
  i
}

/// Snap a byte index to the nearest char boundary at or after `idx`.
fn snap_char_boundary_right(text: &str, idx: usize) -> usize {
  let mut i = idx.min(text.len());
  while i < text.len() && !text.is_char_boundary(i) {
    i += 1;
  }
  i
}

pub fn make_snippet(text: &str, terms: &[String], phrases: &[Vec<String>]) -> Option<String> {
  let mut frags = highlight_fragments(
    text,
    terms,
    phrases,
    HighlightOptions {
      pre_tag: "**",
      post_tag: "**",
      fragment_size: 120,
      number_of_fragments: 1,
    },
  );
  frags.pop()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn highlights_first_term() {
    let text = "Rust is a systems programming language";
    let snippet = make_snippet(text, &[String::from("systems")], &[]).unwrap();
    assert!(snippet.contains("**systems**"));
    let none = make_snippet("", &[String::from("systems")], &[]);
    assert!(none.is_none());
  }

  #[test]
  fn highlights_multibyte_text_without_panic() {
    // The fragment window may land mid-character for multi-byte UTF-8.
    // Use accented Latin text so \b word boundaries work.
    let text = "Un résumé très détaillé du système de recherche avancée";
    let frags = highlight_fragments(
      text,
      &[String::from("système")],
      &[],
      HighlightOptions {
        pre_tag: "<em>",
        post_tag: "</em>",
        fragment_size: 20, // small size forces mid-char slicing
        number_of_fragments: 1,
      },
    );
    assert!(!frags.is_empty(), "should produce a fragment");
    assert!(
      frags[0].contains("<em>système</em>"),
      "fragment should contain highlighted term, got: {}",
      frags[0]
    );
  }

  #[test]
  fn snap_boundaries_are_correct() {
    // "café": c(0) a(1) f(2) é(3..5) — é is 2 bytes, index 4 is mid-char
    let text = "café";
    assert_eq!(snap_char_boundary_left(text, 3), 3); // start of é
    assert_eq!(snap_char_boundary_left(text, 4), 3); // mid-é snaps back
    assert_eq!(snap_char_boundary_left(text, 5), 5); // end of string
    assert_eq!(snap_char_boundary_right(text, 3), 3); // start of é
    assert_eq!(snap_char_boundary_right(text, 4), 5); // mid-é snaps forward
    assert_eq!(snap_char_boundary_right(text, 5), 5); // end of string
  }
}
