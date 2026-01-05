#[derive(Debug, Clone)]
pub struct QueryTerm {
  pub field: Option<String>,
  pub term: String,
}

#[derive(Debug, Clone)]
pub struct PhraseQuery {
  pub field: Option<String>,
  pub terms: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ParsedQuery {
  pub terms: Vec<QueryTerm>,
  pub phrases: Vec<PhraseQuery>,
  pub not_terms: Vec<QueryTerm>,
}

pub fn parse_query(input: &str) -> ParsedQuery {
  let mut terms = Vec::new();
  let mut phrases = Vec::new();
  let mut not_terms = Vec::new();
  let mut rest = input.trim();
  while let Some(start) = rest.find('"') {
    let (before, after) = rest.split_at(start);
    let before = before.trim();
    if !before.is_empty() {
      let (t, n) = parse_terms(before);
      terms.extend(t);
      not_terms.extend(n);
    }
    let after = &after[1..];
    if let Some(end_idx) = after.find('"') {
      let phrase_body = &after[..end_idx];
      let mut field = None;
      let mut body = phrase_body;
      if let Some(colon_idx) = phrase_body.find(':') {
        if phrase_body[..colon_idx]
          .chars()
          .all(|c| c.is_alphanumeric() || c == '_')
        {
          field = Some(phrase_body[..colon_idx].to_string());
          body = &phrase_body[colon_idx + 1..];
        }
      }
      let terms_vec = body
        .split_whitespace()
        .map(|t| t.to_string())
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>();
      if !terms_vec.is_empty() {
        phrases.push(PhraseQuery {
          field,
          terms: terms_vec,
        });
      }
      rest = &after[end_idx + 1..];
    } else {
      rest = "";
    }
  }
  if !rest.trim().is_empty() {
    let (t, n) = parse_terms(rest);
    terms.extend(t);
    not_terms.extend(n);
  }
  ParsedQuery {
    terms,
    phrases,
    not_terms,
  }
}

fn parse_terms(segment: &str) -> (Vec<QueryTerm>, Vec<QueryTerm>) {
  let mut out = Vec::new();
  let mut not_out = Vec::new();
  for raw in segment.split_whitespace() {
    if raw.is_empty() {
      continue;
    }
    let is_not = raw.starts_with('-');
    let token = raw.trim_start_matches('-');
    let (field, term) = if let Some(idx) = token.find(':') {
      let (f, rest) = token.split_at(idx);
      (Some(f.to_string()), rest[1..].to_string())
    } else {
      (None, token.to_string())
    };
    let qt = QueryTerm { field, term };
    if is_not {
      not_out.push(qt);
    } else {
      out.push(qt);
    }
  }
  (out, not_out)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn parses_fields_phrases_and_not_terms() {
    let parsed = parse_query("title:Rust body:safety -noise \"body:memory safety\"");
    assert_eq!(parsed.terms.len(), 2);
    assert_eq!(parsed.not_terms.len(), 1);
    let title = &parsed.terms[0];
    assert_eq!(title.field.as_deref(), Some("title"));
    assert_eq!(title.term, "Rust");
    let body = &parsed.terms[1];
    assert_eq!(body.field.as_deref(), Some("body"));
    assert_eq!(body.term, "safety");
    assert_eq!(parsed.not_terms[0].term, "noise");
    let phrase = &parsed.phrases[0];
    assert_eq!(phrase.field.as_deref(), Some("body"));
    assert_eq!(
      phrase.terms,
      vec!["memory".to_string(), "safety".to_string()]
    );
  }
}
