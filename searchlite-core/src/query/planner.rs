use std::collections::BTreeMap;

use crate::util::regex::anchored_regex;
use anyhow::{bail, Result};

use crate::api::query::{parse_query, ParsedQuery};
use crate::api::types::{
  FieldSpec, Filter, FunctionBoostMode, FunctionScoreMode, FunctionSpec, MatchOperator,
  MinimumShouldMatch, MultiMatchType, Query, QueryNode, RankFeatureModifier,
};

const DEFAULT_PREFIX_MAX_EXPANSIONS: usize = 50;
const DEFAULT_WILDCARD_MAX_EXPANSIONS: usize = 100;
const DEFAULT_REGEX_MAX_EXPANSIONS: usize = 100;

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
pub(crate) enum TermGroupMode {
  PerField,
  CrossFields,
}

#[derive(Debug, Clone)]
pub(crate) struct FieldSpecInternal {
  pub field: String,
  pub boost: f32,
  pub leaf: Option<usize>,
}

#[derive(Debug, Clone)]
pub(crate) enum TermExpansion {
  Exact,
  Prefix { max_expansions: usize },
  Wildcard { max_expansions: usize },
  Regex { max_expansions: usize },
}

#[derive(Debug, Clone)]
pub(crate) struct TermGroupSpec {
  pub fields: Vec<FieldSpecInternal>,
  pub term: String,
  pub expansion: TermExpansion,
  pub boost: f32,
  pub score: bool,
  // NOTE: `mode` is set during planning to distinguish PerField vs CrossFields grouping,
  // but scoring currently assumes PerField. CrossFields-aware scoring will consume this
  // in a future iteration; until then we keep it and silence unused warnings.
  #[allow(dead_code)]
  pub mode: TermGroupMode,
  pub leaf: Option<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct PhraseSpec {
  pub fields: Vec<String>,
  pub terms: Vec<String>,
  pub slop: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct QueryStringMatcher {
  pub term_groups: Vec<usize>,
  pub phrase_groups: Vec<usize>,
  pub not_term_groups: Vec<usize>,
  pub minimum_should_match: Option<usize>,
}

#[derive(Debug, Clone)]
pub(crate) enum QueryMatcher {
  MatchAll,
  Term(usize),
  Phrase(usize),
  QueryString(QueryStringMatcher),
  DisMax(Vec<QueryMatcher>),
  Bool {
    must: Vec<QueryMatcher>,
    should: Vec<QueryMatcher>,
    must_not: Vec<QueryMatcher>,
    filter: Vec<Filter>,
    minimum_should_match: Option<usize>,
  },
}

#[derive(Debug, Clone)]
pub(crate) enum ScoreExpr {
  Leaf(usize),
  Sum(Vec<ScoreExpr>),
  DisMax {
    children: Vec<ScoreExpr>,
    tie_breaker: f32,
  },
}

impl ScoreExpr {
  fn max_leaf(&self) -> Option<usize> {
    match self {
      ScoreExpr::Leaf(idx) => Some(*idx),
      ScoreExpr::Sum(children) => children.iter().filter_map(|c| c.max_leaf()).max(),
      ScoreExpr::DisMax { children, .. } => children.iter().filter_map(|c| c.max_leaf()).max(),
    }
  }

  pub(crate) fn evaluate(&self, leaves: &[f32]) -> f32 {
    match self {
      ScoreExpr::Leaf(idx) => leaves.get(*idx).copied().unwrap_or(0.0),
      ScoreExpr::Sum(children) => children.iter().map(|c| c.evaluate(leaves)).sum(),
      ScoreExpr::DisMax {
        children,
        tie_breaker,
      } => {
        if children.is_empty() {
          return 0.0;
        }
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0_f32;
        for child in children.iter() {
          let score = child.evaluate(leaves);
          max = max.max(score);
          sum += score;
        }
        max + *tie_breaker * (sum - max)
      }
    }
  }
}

#[derive(Debug, Clone)]
pub(crate) struct ScorePlan {
  pub root: ScoreExpr,
  pub leaf_count: usize,
}

impl ScorePlan {
  pub fn evaluate(&self, leaves: &[f32]) -> f32 {
    self.root.evaluate(leaves)
  }
}

#[derive(Debug, Clone)]
pub(crate) enum ScoreNode {
  Empty,
  Expr(ScoreExpr),
  Sum(Vec<ScoreNode>),
  DisMax {
    children: Vec<ScoreNode>,
    tie_breaker: f32,
  },
  Constant {
    score: f32,
    matcher: QueryMatcher,
  },
  FunctionScore {
    matcher: QueryMatcher,
    base: Box<ScoreNode>,
    functions: Vec<FunctionSpec>,
    score_mode: FunctionScoreMode,
    boost_mode: FunctionBoostMode,
    max_boost: Option<f32>,
    min_score: Option<f32>,
    boost: f32,
  },
  RankFeature {
    matcher: QueryMatcher,
    field: String,
    modifier: Option<RankFeatureModifier>,
    missing: Option<f32>,
    boost: f32,
  },
  ScriptScore {
    matcher: QueryMatcher,
    base: Box<ScoreNode>,
    script: String,
    params: Option<BTreeMap<String, f64>>,
    boost: f32,
  },
}

#[derive(Debug, Clone)]
pub(crate) struct QueryPlan {
  pub matcher: QueryMatcher,
  pub term_groups: Vec<TermGroupSpec>,
  pub phrase_specs: Vec<PhraseSpec>,
  pub scorer: Option<ScorePlan>,
  pub score_tree: ScoreNode,
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
  let (matcher, score_expr, score_node) = builder.build_node(&node, true, 1.0)?;
  let mut leaf_count = builder.leaf_count();
  let scorer = score_expr.map(|expr| {
    if let Some(max_leaf) = expr.max_leaf() {
      leaf_count = leaf_count.max(max_leaf.saturating_add(1));
    }
    ScorePlan {
      root: expr,
      leaf_count,
    }
  });
  Ok(QueryPlan {
    matcher,
    term_groups: builder.term_groups,
    phrase_specs: builder.phrase_specs,
    scorer,
    score_tree: score_node,
  })
}

struct QueryPlanBuilder<'a> {
  default_fields: &'a [String],
  term_groups: Vec<TermGroupSpec>,
  phrase_specs: Vec<PhraseSpec>,
  next_leaf_idx: usize,
}

impl<'a> QueryPlanBuilder<'a> {
  fn new(default_fields: &'a [String]) -> Self {
    Self {
      default_fields,
      term_groups: Vec::new(),
      phrase_specs: Vec::new(),
      next_leaf_idx: 0,
    }
  }

  fn alloc_leaf(&mut self) -> usize {
    let idx = self.next_leaf_idx;
    self.next_leaf_idx += 1;
    idx
  }

  fn leaf_count(&self) -> usize {
    self.next_leaf_idx
  }

  fn build_node(
    &mut self,
    node: &QueryNode,
    score: bool,
    boost: f32,
  ) -> Result<(QueryMatcher, Option<ScoreExpr>, ScoreNode)> {
    match node {
      QueryNode::MatchAll { boost: node_boost } => {
        // MatchAll is filter-only; boost is validated for API consistency.
        validate_boost(node_boost)?;
        Ok((QueryMatcher::MatchAll, None, ScoreNode::Empty))
      }
      QueryNode::QueryString {
        query,
        fields,
        boost: node_boost,
      } => {
        let node_boost = validate_boost(node_boost)?;
        let parsed = parse_query(query);
        let base_fields = normalize_fields(fields.as_deref(), self.default_fields, None)?;
        let mut term_groups = Vec::new();
        let mut term_leaves = Vec::new();
        for term in parsed.terms.iter() {
          let fields = match &term.field {
            Some(field) => vec![FieldSpecInternal {
              field: field.clone(),
              boost: 1.0,
              leaf: None,
            }],
            None => base_fields.clone(),
          };
          let leaf = score.then(|| self.alloc_leaf());
          let idx = self.push_term_group(
            fields,
            term.term.clone(),
            TermExpansion::Exact,
            boost * node_boost,
            score,
            TermGroupMode::PerField,
            leaf,
          );
          term_groups.push(idx);
          if let Some(l) = leaf {
            term_leaves.push(ScoreExpr::Leaf(l));
          }
        }
        let mut not_term_groups = Vec::new();
        for term in parsed.not_terms.iter() {
          let fields = match &term.field {
            Some(field) => vec![FieldSpecInternal {
              field: field.clone(),
              boost: 1.0,
              leaf: None,
            }],
            None => base_fields.clone(),
          };
          let idx = self.push_term_group(
            fields,
            term.term.clone(),
            TermExpansion::Exact,
            boost * node_boost,
            false,
            TermGroupMode::PerField,
            None,
          );
          not_term_groups.push(idx);
        }
        let mut phrase_groups = Vec::new();
        for phrase in parsed.phrases.iter() {
          let fields = match &phrase.field {
            Some(field) => vec![field.clone()],
            None => base_fields.iter().map(|f| f.field.clone()).collect(),
          };
          let idx = self.push_phrase(fields, phrase.terms.clone(), 0);
          phrase_groups.push(idx);
        }
        let matcher = QueryMatcher::QueryString(QueryStringMatcher {
          term_groups,
          phrase_groups,
          not_term_groups,
          minimum_should_match: None,
        });
        let scorer = if term_leaves.is_empty() {
          None
        } else if term_leaves.len() == 1 {
          Some(term_leaves.pop().unwrap())
        } else {
          Some(ScoreExpr::Sum(term_leaves))
        };
        let score_node = scorer
          .as_ref()
          .map(|expr| ScoreNode::Expr(expr.clone()))
          .unwrap_or(ScoreNode::Empty);
        Ok((matcher, scorer, score_node))
      }
      QueryNode::MultiMatch {
        query,
        fields,
        match_type,
        tie_breaker,
        operator,
        minimum_should_match,
        boost: node_boost,
      } => {
        let node_boost = validate_boost(node_boost)?;
        let op = operator.clone().unwrap_or(MatchOperator::Or);
        let parsed = parse_query(query);
        let required = resolve_minimum_should_match(minimum_should_match, parsed.terms.len(), op)?;
        let tie = validate_tie_breaker(tie_breaker)?;
        let (field_specs, group_leaf, scorer, mode) = match match_type {
          MultiMatchType::BestFields => {
            let mut leaves = Vec::new();
            let mut specs = Vec::new();
            for spec in fields.iter() {
              let leaf = self.alloc_leaf();
              leaves.push(ScoreExpr::Leaf(leaf));
              specs.push(FieldSpecInternal {
                field: spec.field.clone(),
                boost: validate_boost(&spec.boost)?,
                leaf: Some(leaf),
              });
            }
            let scorer = if leaves.is_empty() {
              None
            } else {
              Some(ScoreExpr::DisMax {
                children: leaves,
                tie_breaker: tie,
              })
            };
            (specs, None, scorer, TermGroupMode::PerField)
          }
          MultiMatchType::MostFields => {
            let leaf = score.then(|| self.alloc_leaf());
            let specs = normalize_fields(Some(fields.as_slice()), self.default_fields, leaf)?;
            let scorer = leaf.map(ScoreExpr::Leaf);
            (specs, leaf, scorer, TermGroupMode::PerField)
          }
          MultiMatchType::CrossFields => {
            let leaf = score.then(|| self.alloc_leaf());
            let specs = normalize_fields(Some(fields.as_slice()), self.default_fields, leaf)?;
            let scorer = leaf.map(ScoreExpr::Leaf);
            (specs, leaf, scorer, TermGroupMode::CrossFields)
          }
        };
        let mut term_groups = Vec::new();
        for term in parsed.terms.iter() {
          let idx = self.push_term_group(
            field_specs.clone(),
            term.term.clone(),
            TermExpansion::Exact,
            boost * node_boost,
            score,
            mode.clone(),
            group_leaf,
          );
          term_groups.push(idx);
        }
        let mut not_term_groups = Vec::new();
        for term in parsed.not_terms.iter() {
          let idx = self.push_term_group(
            field_specs.clone(),
            term.term.clone(),
            TermExpansion::Exact,
            boost * node_boost,
            false,
            mode.clone(),
            None,
          );
          not_term_groups.push(idx);
        }
        let mut phrase_groups = Vec::new();
        for phrase in parsed.phrases.iter() {
          let fields = field_specs.iter().map(|f| f.field.clone()).collect();
          let idx = self.push_phrase(fields, phrase.terms.clone(), 0);
          phrase_groups.push(idx);
        }
        let matcher = QueryMatcher::QueryString(QueryStringMatcher {
          term_groups,
          phrase_groups,
          not_term_groups,
          minimum_should_match: required,
        });
        let score_node = scorer
          .as_ref()
          .map(|expr| ScoreNode::Expr(expr.clone()))
          .unwrap_or(ScoreNode::Empty);
        Ok((matcher, scorer, score_node))
      }
      QueryNode::DisMax {
        queries,
        tie_breaker,
        boost: node_boost,
      } => {
        let node_boost = validate_boost(node_boost)?;
        let tie = validate_tie_breaker(tie_breaker)?;
        let mut matchers = Vec::with_capacity(queries.len());
        let mut scorers = Vec::new();
        let mut score_nodes = Vec::new();
        for child in queries.iter() {
          let (matcher, scorer, score_node) = self.build_node(child, score, boost * node_boost)?;
          matchers.push(matcher);
          if let Some(expr) = scorer {
            scorers.push(expr);
          }
          if !matches!(score_node, ScoreNode::Empty) {
            score_nodes.push(score_node);
          }
        }
        let matcher = QueryMatcher::DisMax(matchers);
        let scorer = if scorers.is_empty() {
          None
        } else if scorers.len() == 1 {
          Some(scorers.pop().unwrap())
        } else {
          Some(ScoreExpr::DisMax {
            children: scorers,
            tie_breaker: tie,
          })
        };
        let score_node = if score_nodes.is_empty() {
          ScoreNode::Empty
        } else if score_nodes.len() == 1 {
          score_nodes.pop().unwrap()
        } else {
          ScoreNode::DisMax {
            children: score_nodes,
            tie_breaker: tie,
          }
        };
        Ok((matcher, scorer, score_node))
      }
      QueryNode::Term {
        field,
        value,
        boost: node_boost,
      } => {
        let node_boost = validate_boost(node_boost)?;
        let leaf = score.then(|| self.alloc_leaf());
        let idx = self.push_term_group(
          vec![FieldSpecInternal {
            field: field.clone(),
            boost: 1.0,
            leaf: None,
          }],
          value.clone(),
          TermExpansion::Exact,
          boost * node_boost,
          score,
          TermGroupMode::PerField,
          leaf,
        );
        let scorer = leaf.map(ScoreExpr::Leaf);
        let score_node = scorer
          .as_ref()
          .map(|expr| ScoreNode::Expr(expr.clone()))
          .unwrap_or(ScoreNode::Empty);
        Ok((QueryMatcher::Term(idx), scorer, score_node))
      }
      QueryNode::Prefix {
        field,
        value,
        max_expansions,
        boost: node_boost,
      } => {
        let node_boost = validate_boost(node_boost)?;
        let leaf = score.then(|| self.alloc_leaf());
        let idx = self.push_term_group(
          vec![FieldSpecInternal {
            field: field.clone(),
            boost: 1.0,
            leaf: None,
          }],
          value.clone(),
          TermExpansion::Prefix {
            max_expansions: max_expansions.unwrap_or(DEFAULT_PREFIX_MAX_EXPANSIONS),
          },
          boost * node_boost,
          score,
          TermGroupMode::PerField,
          leaf,
        );
        let scorer = leaf.map(ScoreExpr::Leaf);
        let score_node = scorer
          .as_ref()
          .map(|expr| ScoreNode::Expr(expr.clone()))
          .unwrap_or(ScoreNode::Empty);
        Ok((QueryMatcher::Term(idx), scorer, score_node))
      }
      QueryNode::Wildcard {
        field,
        value,
        max_expansions,
        boost: node_boost,
      } => {
        let node_boost = validate_boost(node_boost)?;
        let leaf = score.then(|| self.alloc_leaf());
        let idx = self.push_term_group(
          vec![FieldSpecInternal {
            field: field.clone(),
            boost: 1.0,
            leaf: None,
          }],
          value.clone(),
          TermExpansion::Wildcard {
            max_expansions: max_expansions.unwrap_or(DEFAULT_WILDCARD_MAX_EXPANSIONS),
          },
          boost * node_boost,
          score,
          TermGroupMode::PerField,
          leaf,
        );
        let scorer = leaf.map(ScoreExpr::Leaf);
        let score_node = scorer
          .as_ref()
          .map(|expr| ScoreNode::Expr(expr.clone()))
          .unwrap_or(ScoreNode::Empty);
        Ok((QueryMatcher::Term(idx), scorer, score_node))
      }
      QueryNode::Regex {
        field,
        value,
        max_expansions,
        boost: node_boost,
      } => {
        let node_boost = validate_boost(node_boost)?;
        let leaf = score.then(|| self.alloc_leaf());
        anchored_regex(value)?;
        let idx = self.push_term_group(
          vec![FieldSpecInternal {
            field: field.clone(),
            boost: 1.0,
            leaf: None,
          }],
          value.clone(),
          TermExpansion::Regex {
            max_expansions: max_expansions.unwrap_or(DEFAULT_REGEX_MAX_EXPANSIONS),
          },
          boost * node_boost,
          score,
          TermGroupMode::PerField,
          leaf,
        );
        let scorer = leaf.map(ScoreExpr::Leaf);
        let score_node = scorer
          .as_ref()
          .map(|expr| ScoreNode::Expr(expr.clone()))
          .unwrap_or(ScoreNode::Empty);
        Ok((QueryMatcher::Term(idx), scorer, score_node))
      }
      QueryNode::Phrase {
        field,
        terms,
        slop,
        boost: node_boost,
      } => {
        // Phrase matching is filter-only; boost is validated but not scored.
        validate_boost(node_boost)?;
        let fields = match field {
          Some(field) => vec![field.clone()],
          None => self.default_fields.to_vec(),
        };
        let idx = self.push_phrase(fields, terms.clone(), slop.unwrap_or(0) as u32);
        Ok((QueryMatcher::Phrase(idx), None, ScoreNode::Empty))
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
        let mut scorer_parts = Vec::new();
        let mut score_nodes = Vec::new();
        for child in must.iter() {
          let (m, s, score_node) = self.build_node(child, score, child_boost)?;
          must_matchers.push(m);
          if let Some(expr) = s {
            scorer_parts.push(expr);
          }
          if !matches!(score_node, ScoreNode::Empty) {
            score_nodes.push(score_node);
          }
        }
        let mut should_matchers = Vec::with_capacity(should.len());
        for child in should.iter() {
          let (m, s, score_node) = self.build_node(child, score, child_boost)?;
          should_matchers.push(m);
          if let Some(expr) = s {
            scorer_parts.push(expr);
          }
          if !matches!(score_node, ScoreNode::Empty) {
            score_nodes.push(score_node);
          }
        }
        let mut must_not_matchers = Vec::with_capacity(must_not.len());
        for child in must_not.iter() {
          let (m, s, score_node) = self.build_node(child, false, child_boost)?;
          must_not_matchers.push(m);
          if let Some(expr) = s {
            scorer_parts.push(expr);
          }
          if !matches!(score_node, ScoreNode::Empty) {
            score_nodes.push(score_node);
          }
        }
        let scorer = if scorer_parts.is_empty() {
          None
        } else if scorer_parts.len() == 1 {
          Some(scorer_parts.pop().unwrap())
        } else {
          Some(ScoreExpr::Sum(scorer_parts))
        };
        let score_node = if score_nodes.is_empty() {
          ScoreNode::Empty
        } else if score_nodes.len() == 1 {
          score_nodes.pop().unwrap()
        } else {
          ScoreNode::Sum(score_nodes)
        };
        Ok((
          QueryMatcher::Bool {
            must: must_matchers,
            should: should_matchers,
            must_not: must_not_matchers,
            filter: filter.clone(),
            minimum_should_match: *minimum_should_match,
          },
          scorer,
          score_node,
        ))
      }
      QueryNode::ConstantScore {
        filter,
        boost: node_boost,
      } => {
        let node_boost = validate_boost(node_boost)?;
        let matcher = QueryMatcher::Bool {
          must: Vec::new(),
          should: Vec::new(),
          must_not: Vec::new(),
          filter: vec![filter.clone()],
          minimum_should_match: None,
        };
        let score_node = ScoreNode::Constant {
          score: boost * node_boost,
          matcher: matcher.clone(),
        };
        Ok((matcher, None, score_node))
      }
      QueryNode::FunctionScore {
        query,
        functions,
        score_mode,
        boost_mode,
        max_boost,
        min_score,
        boost: node_boost,
      } => {
        let node_boost = validate_boost(node_boost)?;
        if let Some(val) = max_boost {
          if !val.is_finite() {
            bail!("function_score `max_boost` must be finite");
          }
        }
        if let Some(val) = min_score {
          if !val.is_finite() {
            bail!("function_score `min_score` must be finite");
          }
        }
        let (matcher, scorer, base_score_node) = self.build_node(query, score, boost)?;
        let score_node = ScoreNode::FunctionScore {
          matcher: matcher.clone(),
          base: Box::new(base_score_node),
          functions: functions.clone(),
          score_mode: (*score_mode).unwrap_or(FunctionScoreMode::Sum),
          boost_mode: (*boost_mode).unwrap_or(FunctionBoostMode::Multiply),
          max_boost: *max_boost,
          min_score: *min_score,
          boost: boost * node_boost,
        };
        Ok((matcher, scorer, score_node))
      }
      QueryNode::RankFeature {
        field,
        boost: node_boost,
        modifier,
        missing,
      } => {
        let node_boost = validate_boost(node_boost)?;
        let matcher = QueryMatcher::MatchAll;
        let score_node = ScoreNode::RankFeature {
          matcher: matcher.clone(),
          field: field.clone(),
          modifier: *modifier,
          missing: *missing,
          boost: boost * node_boost,
        };
        Ok((matcher, None, score_node))
      }
      QueryNode::ScriptScore {
        query,
        script,
        params,
        boost: node_boost,
      } => {
        let node_boost = validate_boost(node_boost)?;
        let (matcher, scorer, base_score_node) = self.build_node(query, score, boost)?;
        let score_node = ScoreNode::ScriptScore {
          matcher: matcher.clone(),
          base: Box::new(base_score_node),
          script: script.clone(),
          params: params.clone(),
          boost: boost * node_boost,
        };
        Ok((matcher, scorer, score_node))
      }
      #[cfg(feature = "vectors")]
      QueryNode::Vector(_) => {
        // Vector clauses are handled by the vector search path; treat as MatchAll
        // for BM25 planning so mixed queries can proceed.
        Ok((QueryMatcher::MatchAll, None, ScoreNode::Empty))
      }
    }
  }

  #[allow(clippy::too_many_arguments)]
  fn push_term_group(
    &mut self,
    fields: Vec<FieldSpecInternal>,
    term: String,
    expansion: TermExpansion,
    boost: f32,
    score: bool,
    mode: TermGroupMode,
    leaf: Option<usize>,
  ) -> usize {
    let idx = self.term_groups.len();
    self.term_groups.push(TermGroupSpec {
      fields,
      term,
      expansion,
      boost,
      score,
      mode,
      leaf,
    });
    idx
  }

  fn push_phrase(&mut self, fields: Vec<String>, terms: Vec<String>, slop: u32) -> usize {
    let idx = self.phrase_specs.len();
    self.phrase_specs.push(PhraseSpec {
      fields,
      terms,
      slop,
    });
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
  if !value.is_finite() || value.is_sign_negative() {
    bail!("query boost must be finite and non-negative (>= 0)");
  }
  Ok(value)
}

fn validate_tie_breaker(tie: &Option<f32>) -> Result<f32> {
  let value = tie.unwrap_or(0.0);
  if value < 0.0 {
    bail!("tie_breaker must be non-negative");
  }
  if value > 1.0 {
    bail!("tie_breaker must be <= 1.0");
  }
  Ok(value)
}

fn normalize_fields(
  fields: Option<&[FieldSpec]>,
  default_fields: &[String],
  leaf: Option<usize>,
) -> Result<Vec<FieldSpecInternal>> {
  match fields {
    Some(specs) => specs
      .iter()
      .map(|spec| {
        Ok(FieldSpecInternal {
          field: spec.field.clone(),
          boost: validate_boost(&spec.boost)?,
          leaf,
        })
      })
      .collect(),
    None => Ok(
      default_fields
        .iter()
        .map(|field| FieldSpecInternal {
          field: field.clone(),
          boost: 1.0,
          leaf,
        })
        .collect(),
    ),
  }
}

fn resolve_minimum_should_match(
  minimum_should_match: &Option<MinimumShouldMatch>,
  term_count: usize,
  op: MatchOperator,
) -> Result<Option<usize>> {
  if term_count == 0 {
    return Ok(None);
  }
  let base = match op {
    MatchOperator::And => term_count,
    MatchOperator::Or => 1,
  };
  let Some(spec) = minimum_should_match else {
    return Ok(Some(base));
  };
  let required = match spec {
    MinimumShouldMatch::Value(v) => (*v).min(term_count),
    MinimumShouldMatch::Percentage(pct) => {
      if !pct.ends_with('%') {
        bail!("minimum_should_match percentage must be a number with % suffix");
      }
      let without_percent_suffix = &pct[..pct.len() - 1];
      let percent: f32 = without_percent_suffix.parse().map_err(|_| {
        anyhow::anyhow!("minimum_should_match percentage must be a number with % suffix")
      })?;
      if !(0.0..=100.0).contains(&percent) {
        bail!("minimum_should_match percentage must be between 0 and 100");
      }
      let raw = (percent / 100.0) * term_count as f32;
      // 0% explicitly allows zero required matches; callers opt into this.
      raw.ceil() as usize
    }
  };
  Ok(Some(required.min(term_count)))
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

  #[test]
  fn multi_match_preserves_all_fields() {
    let default_fields = vec!["body".to_string(), "title".to_string()];
    let plan = build_query_plan(
      &Query::Node(QueryNode::MultiMatch {
        query: "rust".into(),
        fields: vec![
          FieldSpec {
            field: "title".into(),
            boost: None,
          },
          FieldSpec {
            field: "body".into(),
            boost: None,
          },
        ],
        match_type: MultiMatchType::BestFields,
        tie_breaker: None,
        operator: None,
        minimum_should_match: None,
        boost: None,
      }),
      &default_fields,
    )
    .unwrap();
    assert_eq!(plan.term_groups.len(), 1);
    let group = &plan.term_groups[0];
    let field_names: Vec<_> = group.fields.iter().map(|f| f.field.as_str()).collect();
    assert_eq!(field_names, vec!["title", "body"]);
  }
}
