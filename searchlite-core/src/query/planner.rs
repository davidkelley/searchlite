use crate::api::query::ParsedQuery;

/// Expands query terms into field-qualified terms using default fields when no explicit field is given.
pub fn expand_terms(query: &ParsedQuery, default_fields: &[String]) -> Vec<(String, String)> {
  let mut out = Vec::new();
  for term in query.terms.iter() {
    if let Some(field) = &term.field {
      out.push((field.clone(), term.term.clone()));
    } else {
      for f in default_fields {
        out.push((f.clone(), term.term.clone()));
      }
    }
  }
  out
}

pub fn expand_not_terms(query: &ParsedQuery, default_fields: &[String]) -> Vec<(String, String)> {
  let mut out = Vec::new();
  for term in query.not_terms.iter() {
    if let Some(field) = &term.field {
      out.push((field.clone(), term.term.clone()));
    } else {
      for f in default_fields {
        out.push((f.clone(), term.term.clone()));
      }
    }
  }
  out
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::api::query::{ParsedQuery, QueryTerm};

  #[test]
  fn duplicates_terms_for_default_fields() {
    let query = ParsedQuery {
      terms: vec![QueryTerm {
        field: None,
        term: "rust".to_string(),
      }],
      phrases: Vec::new(),
      not_terms: vec![QueryTerm {
        field: None,
        term: "boring".to_string(),
      }],
    };
    let fields = vec!["title".to_string(), "body".to_string()];
    let expanded = expand_terms(&query, &fields);
    assert_eq!(
      expanded,
      vec![
        ("title".to_string(), "rust".to_string()),
        ("body".to_string(), "rust".to_string())
      ]
    );
    let not_expanded = expand_not_terms(&query, &fields);
    assert_eq!(
      not_expanded,
      vec![
        ("title".to_string(), "boring".to_string()),
        ("body".to_string(), "boring".to_string())
      ]
    );
  }
}
