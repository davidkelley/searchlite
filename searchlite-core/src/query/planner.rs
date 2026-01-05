use anyhow::{bail, Result};

use crate::api::query::{parse_query, ParsedQuery};
use crate::api::types::{Filter, Query, QueryNode};

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

#[derive(Debug, Clone)]
pub(crate) struct TermGroupSpec {
  pub fields: Vec<String>,
  pub term: String,
  pub boost: f32,
  pub score: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct PhraseSpec {
  pub fields: Vec<String>,
  pub terms: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct QueryStringMatcher {
  pub term_groups: Vec<usize>,
  pub phrase_groups: Vec<usize>,
  pub not_term_groups: Vec<usize>,
}

#[derive(Debug, Clone)]
pub(crate) enum QueryMatcher {
  MatchAll,
  Term(usize),
  Phrase(usize),
  QueryString(QueryStringMatcher),
  Bool {
    must: Vec<QueryMatcher>,
    should: Vec<QueryMatcher>,
    must_not: Vec<QueryMatcher>,
    filter: Vec<Filter>,
    minimum_should_match: Option<usize>,
  },
}

#[derive(Debug, Clone)]
pub(crate) struct QueryPlan {
  pub matcher: QueryMatcher,
  pub term_groups: Vec<TermGroupSpec>,
  pub phrase_specs: Vec<PhraseSpec>,
}

pub(crate) fn build_query_plan(query: &Query, default_fields: &[String]) -> Result<QueryPlan> {
  let node = match query {
    Query::String(raw) => QueryNode::QueryString {
      query: raw.clone(),
      fields: None,
      boost: None,
    },
    Query::Node(node) => node.clone(),
  };
  let mut builder = QueryPlanBuilder::new(default_fields);
  let matcher = builder.build_node(&node, true, 1.0)?;
  Ok(QueryPlan {
    matcher,
    term_groups: builder.term_groups,
    phrase_specs: builder.phrase_specs,
  })
}

struct QueryPlanBuilder<'a> {
  default_fields: &'a [String],
  term_groups: Vec<TermGroupSpec>,
  phrase_specs: Vec<PhraseSpec>,
}

impl<'a> QueryPlanBuilder<'a> {
  fn new(default_fields: &'a [String]) -> Self {
    Self {
      default_fields,
      term_groups: Vec::new(),
      phrase_specs: Vec::new(),
    }
  }

  fn build_node(&mut self, node: &QueryNode, score: bool, boost: f32) -> Result<QueryMatcher> {
    match node {
      QueryNode::MatchAll { boost: node_boost } => {
        // MatchAll is filter-only; boost is validated for API consistency.
        validate_boost(node_boost)?;
        Ok(QueryMatcher::MatchAll)
      }
      QueryNode::QueryString {
        query,
        fields,
        boost: node_boost,
      } => {
        let node_boost = validate_boost(node_boost)?;
        let parsed = parse_query(query);
        let field_list = fields.as_deref().unwrap_or(self.default_fields);
        let mut term_groups = Vec::new();
        for term in parsed.terms.iter() {
          let fields = match &term.field {
            Some(field) => vec![field.clone()],
            None => field_list.to_vec(),
          };
          let idx = self.push_term_group(fields, term.term.clone(), boost * node_boost, score);
          term_groups.push(idx);
        }
        let mut not_term_groups = Vec::new();
        for term in parsed.not_terms.iter() {
          let fields = match &term.field {
            Some(field) => vec![field.clone()],
            None => field_list.to_vec(),
          };
          let idx = self.push_term_group(fields, term.term.clone(), boost * node_boost, false);
          not_term_groups.push(idx);
        }
        let mut phrase_groups = Vec::new();
        for phrase in parsed.phrases.iter() {
          let fields = match &phrase.field {
            Some(field) => vec![field.clone()],
            None => field_list.to_vec(),
          };
          let idx = self.push_phrase(fields, phrase.terms.clone());
          phrase_groups.push(idx);
        }
        Ok(QueryMatcher::QueryString(QueryStringMatcher {
          term_groups,
          phrase_groups,
          not_term_groups,
        }))
      }
      QueryNode::Term {
        field,
        value,
        boost: node_boost,
      } => {
        let node_boost = validate_boost(node_boost)?;
        let idx = self.push_term_group(
          vec![field.clone()],
          value.clone(),
          boost * node_boost,
          score,
        );
        Ok(QueryMatcher::Term(idx))
      }
      QueryNode::Phrase {
        field,
        terms,
        boost: node_boost,
      } => {
        // Phrase matching is filter-only; boost is validated but not scored.
        validate_boost(node_boost)?;
        let fields = match field {
          Some(field) => vec![field.clone()],
          None => self.default_fields.to_vec(),
        };
        let idx = self.push_phrase(fields, terms.clone());
        Ok(QueryMatcher::Phrase(idx))
      }
      QueryNode::Bool {
        must,
        should,
        must_not,
        filter,
        minimum_should_match,
        boost: node_boost,
      } => {
        let node_boost = validate_boost(node_boost)?;
        let child_boost = boost * node_boost;
        let mut must_matchers = Vec::with_capacity(must.len());
        for child in must.iter() {
          must_matchers.push(self.build_node(child, score, child_boost)?);
        }
        let mut should_matchers = Vec::with_capacity(should.len());
        for child in should.iter() {
          should_matchers.push(self.build_node(child, score, child_boost)?);
        }
        let mut must_not_matchers = Vec::with_capacity(must_not.len());
        for child in must_not.iter() {
          must_not_matchers.push(self.build_node(child, false, child_boost)?);
        }
        Ok(QueryMatcher::Bool {
          must: must_matchers,
          should: should_matchers,
          must_not: must_not_matchers,
          filter: filter.clone(),
          minimum_should_match: *minimum_should_match,
        })
      }
    }
  }

  fn push_term_group(
    &mut self,
    fields: Vec<String>,
    term: String,
    boost: f32,
    score: bool,
  ) -> usize {
    let idx = self.term_groups.len();
    self.term_groups.push(TermGroupSpec {
      fields,
      term,
      boost,
      score,
    });
    idx
  }

  fn push_phrase(&mut self, fields: Vec<String>, terms: Vec<String>) -> usize {
    let idx = self.phrase_specs.len();
    self.phrase_specs.push(PhraseSpec { fields, terms });
    idx
  }
}

/// Validates and normalizes an optional boost value.
///
/// - `None` defaults to a boost of `1.0`.
/// - Any non-negative value is accepted.
/// - A boost of `0.0` disables scoring contribution while still matching.
fn validate_boost(boost: &Option<f32>) -> Result<f32> {
  let value = boost.unwrap_or(1.0);
  if value.is_sign_negative() {
    bail!("query boost must be non-negative (>= 0)");
  }
  Ok(value)
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
