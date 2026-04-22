use std::collections::{BTreeMap, HashMap, HashSet};

use crate::util::regex::anchored_regex;
use anyhow::{bail, Result};

use crate::api::query::{parse_query, ParsedQuery};
use crate::api::types::{
  FieldSpec, Filter, FunctionBoostMode, FunctionScoreMode, FunctionSpec, MatchOperator,
  MinimumShouldMatch, MultiMatchFuzziness, MultiMatchType, Query, QueryNode, RankFeatureModifier,
};

const DEFAULT_PREFIX_MAX_EXPANSIONS: usize = 50;
const DEFAULT_WILDCARD_MAX_EXPANSIONS: usize = 100;
const DEFAULT_REGEX_MAX_EXPANSIONS: usize = 100;

/// Absolute ceiling on `max_expansions` for `QueryNode::Prefix`. The default
/// is intentionally small because the planner enumerates matching terms from
/// the FST and builds an OR fan-out whose postings reads, memory, and WAND
/// heap costs scale linearly with the expansion size. A caller can raise
/// `max_expansions` above the default up to this ceiling; requests beyond
/// it are rejected rather than silently expanded, so a short HTTP body can
/// no longer translate into an unbounded server-side workload (BUG-022).
const MAX_PREFIX_EXPANSIONS_HARD: usize = 10_000;

/// Absolute ceiling on `max_expansions` for `QueryNode::Wildcard`.
/// See `MAX_PREFIX_EXPANSIONS_HARD` for rationale.
const MAX_WILDCARD_EXPANSIONS_HARD: usize = 10_000;

/// Absolute ceiling on `max_expansions` for `QueryNode::Regex`.
/// See `MAX_PREFIX_EXPANSIONS_HARD` for rationale.
const MAX_REGEX_EXPANSIONS_HARD: usize = 10_000;

/// Upper bound on phrase `slop` applied at query planning time. User-supplied
/// values (which flow in as `usize`) are saturated to this ceiling before
/// being narrowed to `u32` for the matcher. The ceiling is `i32::MAX` — the
/// largest value the matcher can faithfully represent in its `i32` "remaining
/// budget" — so every in-range request is preserved exactly and only values
/// beyond what the matcher could respect anyway get capped. Together with the
/// saturating `i32` cast inside `matches_phrase` this preserves the "more
/// slop → looser match" invariant for every input value.
const MAX_PHRASE_SLOP: u32 = i32::MAX as u32;

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
  pub mode: TermGroupMode,
  pub fuzziness: Option<MultiMatchFuzziness>,
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
      ScoreExpr::Sum(children) => {
        // Individual leaf scores are expected to be finite, but summing many
        // finite values can overflow `f32::MAX` to `±∞`, and any NaN input
        // poisons the sum. Clamp non-finite accumulators to `0.0` so they do
        // not leak into the sort-key heap via the pure-BM25 fast path (which
        // has no `score_adjust` hook to filter them). Mirrors the guard
        // applied to `CompiledScoreNode::Sum` in BUG-364.
        let sum: f32 = children.iter().map(|c| c.evaluate(leaves)).sum();
        if sum.is_finite() {
          sum
        } else {
          0.0
        }
      }
      ScoreExpr::DisMax {
        children,
        tie_breaker,
      } => {
        if children.is_empty() {
          return 0.0;
        }
        // Short-circuit the zero-tie-breaker case up-front. `0.0 * ∞ = NaN`
        // under IEEE-754, so the naïve `max + tie_breaker * (sum - max)`
        // formula would leak NaN into the heap when `sum` overflows even
        // though zero-tie-breaker DisMax semantics is simply `max`. Skipping
        // the `sum` accumulator here also avoids unnecessary work on the
        // per-candidate WAND hot path. Mirrors the guard applied to
        // `CompiledScoreNode::DisMax` in BUG-364.
        if *tie_breaker == 0.0 {
          let mut max = f32::NEG_INFINITY;
          for child in children.iter() {
            max = max.max(child.evaluate(leaves));
          }
          return if max.is_finite() { max } else { 0.0 };
        }
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0_f32;
        for child in children.iter() {
          let score = child.evaluate(leaves);
          max = max.max(score);
          sum += score;
        }
        // Clamp non-finite DisMax results to `0.0`: `sum` can still overflow
        // across many finite children, and `max + tie_breaker * (sum - max)`
        // may evaluate to `±∞`/`NaN` under IEEE-754 even with a non-zero
        // tie-breaker.
        let result = max + *tie_breaker * (sum - max);
        if result.is_finite() {
          result
        } else {
          0.0
        }
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
            combine_boost(boost, node_boost)?,
            score,
            TermGroupMode::PerField,
            None,
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
            combine_boost(boost, node_boost)?,
            false,
            TermGroupMode::PerField,
            None,
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
        fuzziness,
        tie_breaker,
        operator,
        minimum_should_match,
        boost: node_boost,
      } => {
        let node_boost = validate_boost(node_boost)?;
        let fuzziness = validate_multi_match_fuzziness(fuzziness)?;
        let op = operator.clone().unwrap_or(MatchOperator::Or);
        let parsed = parse_query(query);
        let required = resolve_minimum_should_match(minimum_should_match, parsed.terms.len(), op)?;
        let tie = validate_tie_breaker(tie_breaker)?;
        let (field_specs, group_leaf, scorer, mode) = match match_type {
          MultiMatchType::BestFields => {
            let normalized = normalize_multi_match_fields(fields.as_slice())?;
            let mut leaves = Vec::new();
            let mut specs = Vec::new();
            for (field, boost) in normalized.into_iter() {
              let leaf = self.alloc_leaf();
              leaves.push(ScoreExpr::Leaf(leaf));
              specs.push(FieldSpecInternal {
                field,
                boost,
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
            combine_boost(boost, node_boost)?,
            score,
            mode.clone(),
            fuzziness.clone(),
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
            combine_boost(boost, node_boost)?,
            false,
            mode.clone(),
            fuzziness.clone(),
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
        let combined = combine_boost(boost, node_boost)?;
        for child in queries.iter() {
          let (matcher, scorer, score_node) = self.build_node(child, score, combined)?;
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
          combine_boost(boost, node_boost)?,
          score,
          TermGroupMode::PerField,
          None,
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
            max_expansions: clamp_expansions(
              *max_expansions,
              DEFAULT_PREFIX_MAX_EXPANSIONS,
              MAX_PREFIX_EXPANSIONS_HARD,
              "prefix",
            )?,
          },
          combine_boost(boost, node_boost)?,
          score,
          TermGroupMode::PerField,
          None,
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
            max_expansions: clamp_expansions(
              *max_expansions,
              DEFAULT_WILDCARD_MAX_EXPANSIONS,
              MAX_WILDCARD_EXPANSIONS_HARD,
              "wildcard",
            )?,
          },
          combine_boost(boost, node_boost)?,
          score,
          TermGroupMode::PerField,
          None,
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
            max_expansions: clamp_expansions(
              *max_expansions,
              DEFAULT_REGEX_MAX_EXPANSIONS,
              MAX_REGEX_EXPANSIONS_HARD,
              "regex",
            )?,
          },
          combine_boost(boost, node_boost)?,
          score,
          TermGroupMode::PerField,
          None,
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
        let slop_raw = slop.unwrap_or(0);
        let slop = u32::try_from(slop_raw)
          .unwrap_or(MAX_PHRASE_SLOP)
          .min(MAX_PHRASE_SLOP);
        let idx = self.push_phrase(fields, terms.clone(), slop);
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
        let child_boost = combine_boost(boost, node_boost)?;
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
          score: combine_boost(boost, node_boost)?,
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
          boost: combine_boost(boost, node_boost)?,
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
          boost: combine_boost(boost, node_boost)?,
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
          boost: combine_boost(boost, node_boost)?,
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
    fuzziness: Option<MultiMatchFuzziness>,
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
      fuzziness,
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

/// Combine two individually-validated boost factors, rejecting overflow.
///
/// `validate_boost` confirms each individual boost is finite, but the
/// product of two finite f32 values can still overflow `f32::MAX` to
/// `+inf` (e.g. `1e38 * 1e38`). Before the guard, that `+inf` flowed
/// through term weights into BM25 scores and into `ScoreNode::Constant`
/// payloads, breaking serialisation and surfacing as HTTP 500 or as
/// silently dropped documents further down the pipeline (BUG-381).
/// Rejecting the overflow surfaces a deterministic, actionable
/// validation error instead.
///
/// Called at two layers:
/// - Plan time, from `build_node` for every nested `parent × node` boost
///   propagation (BUG-381 / #383).
/// - Expansion time, from `expand_term_groups`, where the already-combined
///   query boost is multiplied by the per-field multi_match boost when
///   materialising term weights (BUG-396 / #396). In that layer the
///   guard runs after planning has completed but before any scoring
///   takes place, so the error still bubbles out as a search-time
///   validation failure before any non-finite weight can leak.
pub(crate) fn combine_boost(boost: f32, node_boost: f32) -> Result<f32> {
  let combined = boost * node_boost;
  if !combined.is_finite() {
    bail!(
      "combined query boost overflows to non-finite ({boost} * {node_boost}); reduce the query, per-field, and nested boost factors"
    );
  }
  Ok(combined)
}

/// Resolve a caller-supplied `max_expansions` against the per-kind default and
/// hard ceiling, rejecting any value above the ceiling.
///
/// `requested.unwrap_or(default)` would normally suffice, but
/// `QueryNode::{Prefix, Wildcard, Regex}` deserialize `max_expansions`
/// directly from the wire, so a hostile client can supply `usize::MAX` (or
/// any value much larger than the index could reasonably expand to) and
/// force the planner into an unbounded fan-out. See BUG-022. The hard
/// ceiling is an absolute cap applied independently of the default so the
/// default can remain conservatively small without also being the ceiling.
fn clamp_expansions(
  requested: Option<usize>,
  default: usize,
  hard: usize,
  kind: &str,
) -> Result<usize> {
  let value = requested.unwrap_or(default);
  if value > hard {
    bail!("{kind} max_expansions {value} exceeds hard limit {hard}");
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

fn validate_multi_match_fuzziness(
  fuzziness: &Option<MultiMatchFuzziness>,
) -> Result<Option<MultiMatchFuzziness>> {
  let Some(fuzziness) = fuzziness else {
    return Ok(None);
  };
  match fuzziness {
    MultiMatchFuzziness::Auto => Ok(Some(MultiMatchFuzziness::Auto)),
    MultiMatchFuzziness::Edits(value) => {
      if *value > 2 {
        bail!("multi_match fuzziness edit distance must be between 0 and 2");
      }
      Ok(Some(MultiMatchFuzziness::Edits(*value)))
    }
  }
}

fn normalize_fields(
  fields: Option<&[FieldSpec]>,
  default_fields: &[String],
  leaf: Option<usize>,
) -> Result<Vec<FieldSpecInternal>> {
  match fields {
    Some(specs) => Ok(
      normalize_multi_match_fields(specs)?
        .into_iter()
        .map(|(field, boost)| FieldSpecInternal { field, boost, leaf })
        .collect(),
    ),
    None => {
      let mut seen = HashSet::new();
      Ok(
        default_fields
          .iter()
          .filter(|field| seen.insert((*field).clone()))
          .map(|field| FieldSpecInternal {
            field: field.clone(),
            boost: 1.0,
            leaf,
          })
          .collect(),
      )
    }
  }
}

fn normalize_multi_match_fields(fields: &[FieldSpec]) -> Result<Vec<(String, f32)>> {
  let mut out: Vec<(String, f32)> = Vec::new();
  let mut by_field: HashMap<String, usize> = HashMap::new();
  for spec in fields.iter() {
    let boost = validate_boost(&spec.boost)?;
    if let Some(existing) = by_field.get(&spec.field).copied() {
      if boost > out[existing].1 {
        out[existing].1 = boost;
      }
      continue;
    }
    by_field.insert(spec.field.clone(), out.len());
    out.push((spec.field.clone(), boost));
  }
  Ok(out)
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
      let percent: f64 = without_percent_suffix.parse().map_err(|_| {
        anyhow::anyhow!("minimum_should_match percentage must be a number with % suffix")
      })?;
      if !(0.0..=100.0).contains(&percent) {
        bail!("minimum_should_match percentage must be between 0 and 100");
      }
      // BUG-403: round *down* to match Elasticsearch `minimum_should_match`
      // percentage semantics (e.g. 75% of 3 terms = 2, not 3). `floor(0.0)`
      // also preserves the 0%-allows-zero-matches behaviour.
      //
      // For whole-number percentages, compute in integer arithmetic so that
      // mathematically exact products (e.g. 13% of 900 = 117) aren't
      // undercounted by f32/f64 rounding of the intermediate `percent/100`.
      // For fractional percentages, multiply before dividing in f64 so the
      // intermediate stays precise for realistic clause counts.
      if percent.fract() == 0.0 {
        let whole = percent as usize; // range-checked: 0..=100
        whole.saturating_mul(term_count) / 100
      } else {
        let raw = percent * term_count as f64 / 100.0;
        raw.floor() as usize
      }
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
  fn phrase_slop_is_clamped_to_ceiling() {
    // Regression test for BUG-026. User-supplied `slop` is a `usize`; before
    // the fix it was narrowed with `as u32`, which truncated high bits on
    // 64-bit platforms and then wrapped to a negative `i32` inside
    // `matches_phrase`. Every out-of-range value now collapses to the same
    // `MAX_PHRASE_SLOP` ceiling.
    // Out-of-range values saturate to MAX_PHRASE_SLOP rather than truncating
    // or wrapping. This covers usize values above u32::MAX (high-bit
    // truncation) and values above i32::MAX (downstream i32 wrap).
    for input in [
      Some(MAX_PHRASE_SLOP as usize + 1),
      Some(usize::MAX),
      Some(u32::MAX as usize),
    ] {
      let plan = build_query_plan(
        &Query::Node(QueryNode::Phrase {
          field: Some("body".into()),
          terms: vec!["hello".into(), "world".into()],
          slop: input,
          boost: None,
        }),
        &["body".to_string()],
      )
      .unwrap();
      assert_eq!(plan.phrase_specs.len(), 1);
      assert_eq!(plan.phrase_specs[0].slop, MAX_PHRASE_SLOP);
    }

    // In-range values — including legitimate large slops beyond typical usage
    // but still representable in the matcher's i32 budget — pass through
    // unchanged.
    for input in [0usize, 5, 500, 10_000, MAX_PHRASE_SLOP as usize] {
      let plan = build_query_plan(
        &Query::Node(QueryNode::Phrase {
          field: Some("body".into()),
          terms: vec!["hello".into(), "world".into()],
          slop: Some(input),
          boost: None,
        }),
        &["body".to_string()],
      )
      .unwrap();
      assert_eq!(plan.phrase_specs[0].slop as usize, input);
    }
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
        fuzziness: None,
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

  // -------- BUG-022 regression tests --------
  //
  // `QueryNode::{Prefix,Wildcard,Regex}::max_expansions` is deserialized
  // directly from the wire. Before the fix, any caller-supplied value was
  // forwarded verbatim to the term-expansion stage, turning a small HTTP body
  // (`max_expansions: usize::MAX`) into an unbounded server-side fan-out.
  //
  // The planner now clamps the requested expansion against a per-kind hard
  // ceiling (`MAX_{PREFIX,WILDCARD,REGEX}_EXPANSIONS_HARD`) and rejects any
  // value above the ceiling with an error that names both the offending
  // value and the limit.

  fn term_group_prefix_expansion(plan: &QueryPlan) -> usize {
    assert_eq!(plan.term_groups.len(), 1, "expected single term group");
    match plan.term_groups[0].expansion {
      TermExpansion::Prefix { max_expansions } => max_expansions,
      ref other => panic!("expected Prefix expansion, got {other:?}"),
    }
  }

  fn term_group_wildcard_expansion(plan: &QueryPlan) -> usize {
    assert_eq!(plan.term_groups.len(), 1, "expected single term group");
    match plan.term_groups[0].expansion {
      TermExpansion::Wildcard { max_expansions } => max_expansions,
      ref other => panic!("expected Wildcard expansion, got {other:?}"),
    }
  }

  fn term_group_regex_expansion(plan: &QueryPlan) -> usize {
    assert_eq!(plan.term_groups.len(), 1, "expected single term group");
    match plan.term_groups[0].expansion {
      TermExpansion::Regex { max_expansions } => max_expansions,
      ref other => panic!("expected Regex expansion, got {other:?}"),
    }
  }

  #[test]
  fn prefix_max_expansions_defaults_and_passes_through_within_ceiling() {
    // None → default.
    let plan = build_query_plan(
      &Query::Node(QueryNode::Prefix {
        field: "body".into(),
        value: "ru".into(),
        max_expansions: None,
        boost: None,
      }),
      &["body".to_string()],
    )
    .unwrap();
    assert_eq!(
      term_group_prefix_expansion(&plan),
      DEFAULT_PREFIX_MAX_EXPANSIONS
    );

    // In-range values up to and including the ceiling pass through unchanged.
    for input in [
      0usize,
      1,
      DEFAULT_PREFIX_MAX_EXPANSIONS,
      5_000,
      MAX_PREFIX_EXPANSIONS_HARD,
    ] {
      let plan = build_query_plan(
        &Query::Node(QueryNode::Prefix {
          field: "body".into(),
          value: "ru".into(),
          max_expansions: Some(input),
          boost: None,
        }),
        &["body".to_string()],
      )
      .unwrap();
      assert_eq!(term_group_prefix_expansion(&plan), input);
    }
  }

  #[test]
  fn prefix_max_expansions_rejects_above_ceiling() {
    // Values above the hard ceiling — including the pathological
    // `usize::MAX` — are rejected with a descriptive error rather than
    // silently expanding into an unbounded OR.
    for input in [MAX_PREFIX_EXPANSIONS_HARD + 1, 1_000_000, usize::MAX] {
      let err = build_query_plan(
        &Query::Node(QueryNode::Prefix {
          field: "body".into(),
          value: "ru".into(),
          max_expansions: Some(input),
          boost: None,
        }),
        &["body".to_string()],
      )
      .expect_err("prefix max_expansions above the ceiling must be rejected");
      let msg = err.to_string();
      assert!(
        msg.contains("prefix"),
        "error should mention query kind: {msg}"
      );
      assert!(
        msg.contains("exceeds hard limit"),
        "error should name the limit: {msg}"
      );
      // The full diagnostic is part of the contract — the error must name both
      // the offending value and the numeric ceiling so operators can spot
      // which client is tripping it and what the current cap is. Asserting
      // just "exceeds hard limit" leaves room for a future regression that
      // drops the numbers while keeping the phrase.
      assert!(
        msg.contains(&input.to_string()),
        "error should include offending value {input}: {msg}"
      );
      assert!(
        msg.contains(&MAX_PREFIX_EXPANSIONS_HARD.to_string()),
        "error should include hard ceiling {}: {msg}",
        MAX_PREFIX_EXPANSIONS_HARD
      );
    }
  }

  #[test]
  fn wildcard_max_expansions_defaults_and_passes_through_within_ceiling() {
    let plan = build_query_plan(
      &Query::Node(QueryNode::Wildcard {
        field: "body".into(),
        value: "ru*".into(),
        max_expansions: None,
        boost: None,
      }),
      &["body".to_string()],
    )
    .unwrap();
    assert_eq!(
      term_group_wildcard_expansion(&plan),
      DEFAULT_WILDCARD_MAX_EXPANSIONS
    );

    for input in [
      0usize,
      1,
      DEFAULT_WILDCARD_MAX_EXPANSIONS,
      5_000,
      MAX_WILDCARD_EXPANSIONS_HARD,
    ] {
      let plan = build_query_plan(
        &Query::Node(QueryNode::Wildcard {
          field: "body".into(),
          value: "ru*".into(),
          max_expansions: Some(input),
          boost: None,
        }),
        &["body".to_string()],
      )
      .unwrap();
      assert_eq!(term_group_wildcard_expansion(&plan), input);
    }
  }

  #[test]
  fn wildcard_max_expansions_rejects_above_ceiling() {
    for input in [MAX_WILDCARD_EXPANSIONS_HARD + 1, 1_000_000, usize::MAX] {
      let err = build_query_plan(
        &Query::Node(QueryNode::Wildcard {
          field: "body".into(),
          value: "ru*".into(),
          max_expansions: Some(input),
          boost: None,
        }),
        &["body".to_string()],
      )
      .expect_err("wildcard max_expansions above the ceiling must be rejected");
      let msg = err.to_string();
      assert!(
        msg.contains("wildcard"),
        "error should mention query kind: {msg}"
      );
      assert!(
        msg.contains("exceeds hard limit"),
        "error should name the limit: {msg}"
      );
      assert!(
        msg.contains(&input.to_string()),
        "error should include offending value {input}: {msg}"
      );
      assert!(
        msg.contains(&MAX_WILDCARD_EXPANSIONS_HARD.to_string()),
        "error should include hard ceiling {}: {msg}",
        MAX_WILDCARD_EXPANSIONS_HARD
      );
    }
  }

  #[test]
  fn regex_max_expansions_defaults_and_passes_through_within_ceiling() {
    let plan = build_query_plan(
      &Query::Node(QueryNode::Regex {
        field: "body".into(),
        value: "ru.*".into(),
        max_expansions: None,
        boost: None,
      }),
      &["body".to_string()],
    )
    .unwrap();
    assert_eq!(
      term_group_regex_expansion(&plan),
      DEFAULT_REGEX_MAX_EXPANSIONS
    );

    for input in [
      0usize,
      1,
      DEFAULT_REGEX_MAX_EXPANSIONS,
      5_000,
      MAX_REGEX_EXPANSIONS_HARD,
    ] {
      let plan = build_query_plan(
        &Query::Node(QueryNode::Regex {
          field: "body".into(),
          value: "ru.*".into(),
          max_expansions: Some(input),
          boost: None,
        }),
        &["body".to_string()],
      )
      .unwrap();
      assert_eq!(term_group_regex_expansion(&plan), input);
    }
  }

  #[test]
  fn regex_max_expansions_rejects_above_ceiling() {
    for input in [MAX_REGEX_EXPANSIONS_HARD + 1, 1_000_000, usize::MAX] {
      let err = build_query_plan(
        &Query::Node(QueryNode::Regex {
          field: "body".into(),
          value: "ru.*".into(),
          max_expansions: Some(input),
          boost: None,
        }),
        &["body".to_string()],
      )
      .expect_err("regex max_expansions above the ceiling must be rejected");
      let msg = err.to_string();
      assert!(
        msg.contains("regex"),
        "error should mention query kind: {msg}"
      );
      assert!(
        msg.contains("exceeds hard limit"),
        "error should name the limit: {msg}"
      );
      assert!(
        msg.contains(&input.to_string()),
        "error should include offending value {input}: {msg}"
      );
      assert!(
        msg.contains(&MAX_REGEX_EXPANSIONS_HARD.to_string()),
        "error should include hard ceiling {}: {msg}",
        MAX_REGEX_EXPANSIONS_HARD
      );
    }
  }

  // Regression tests for BUG-374. The BM25 fast-path `ScoreExpr::DisMax` and
  // `ScoreExpr::Sum` evaluators must not leak `NaN` or `±∞` into the sort-key
  // heap. Mirrors the `CompiledScoreNode` guards added in BUG-364.

  #[test]
  fn dis_max_zero_tie_breaker_short_circuits_when_sum_overflows() {
    // Target test for BUG-374: choose finite children whose `sum` overflows
    // to `+∞` while `max` stays finite. Without the `tie_breaker == 0`
    // short-circuit, the naïve formula evaluates to `f32::MAX + 0.0 * (∞ -
    // f32::MAX) = f32::MAX + 0.0 * ∞ = NaN`, which the final clamp would
    // mask by returning `0.0`. The short-circuit must instead return the
    // finite `max` directly.
    let expr = ScoreExpr::DisMax {
      children: vec![ScoreExpr::Leaf(0), ScoreExpr::Leaf(1)],
      tie_breaker: 0.0,
    };
    let leaves = [f32::MAX, f32::MAX];
    let score = expr.evaluate(&leaves);
    assert_eq!(
      score,
      f32::MAX,
      "DisMax with tie_breaker==0 must short-circuit to max when sum overflows"
    );
  }

  #[test]
  fn dis_max_nonzero_tie_breaker_guards_nonfinite_result() {
    // With `tie_breaker != 0`, `max + tie_breaker * (sum - max)` can still
    // produce a non-finite value when `sum` overflows. The final guard must
    // clamp to a finite fallback so pure-BM25 queries (no `score_adjust`)
    // never push `NaN`/`±∞` into the heap.
    let expr = ScoreExpr::DisMax {
      children: vec![ScoreExpr::Leaf(0), ScoreExpr::Leaf(1)],
      tie_breaker: 0.3,
    };
    let leaves = [f32::INFINITY, f32::INFINITY];
    let score = expr.evaluate(&leaves);
    assert!(
      score.is_finite(),
      "DisMax must guard non-finite results: got {score}"
    );
  }

  #[test]
  fn dis_max_zero_tie_breaker_preserves_finite_max() {
    // Sanity check: the zero-tie-breaker short-circuit must return the actual
    // `max` for finite inputs rather than collapsing to `0.0`.
    let expr = ScoreExpr::DisMax {
      children: vec![ScoreExpr::Leaf(0), ScoreExpr::Leaf(1)],
      tie_breaker: 0.0,
    };
    let leaves = [1.5_f32, 2.5_f32];
    assert_eq!(expr.evaluate(&leaves), 2.5);
  }

  #[test]
  fn dis_max_nonzero_tie_breaker_preserves_finite_formula() {
    // Sanity check for the finite path: `max + tb * (sum - max)` with
    // `max = 2.0`, `sum = 3.0`, `tie_breaker = 0.5` → `2.0 + 0.5 * 1.0 = 2.5`.
    let expr = ScoreExpr::DisMax {
      children: vec![ScoreExpr::Leaf(0), ScoreExpr::Leaf(1)],
      tie_breaker: 0.5,
    };
    let leaves = [1.0_f32, 2.0_f32];
    assert!((expr.evaluate(&leaves) - 2.5).abs() < 1e-6);
  }

  #[test]
  fn sum_guards_nonfinite_accumulator() {
    // `ScoreExpr::Sum` must reject non-finite sums so that overflow or NaN
    // propagation from leaf scores never leaks into the heap via the pure
    // BM25 path.
    let expr = ScoreExpr::Sum(vec![ScoreExpr::Leaf(0), ScoreExpr::Leaf(1)]);
    let leaves = [f32::INFINITY, f32::INFINITY];
    assert_eq!(expr.evaluate(&leaves), 0.0);

    let nan_leaves = [f32::NAN, 1.0];
    assert_eq!(expr.evaluate(&nan_leaves), 0.0);
  }

  #[test]
  fn sum_preserves_finite_accumulator() {
    let expr = ScoreExpr::Sum(vec![ScoreExpr::Leaf(0), ScoreExpr::Leaf(1)]);
    let leaves = [1.25_f32, 2.75_f32];
    assert_eq!(expr.evaluate(&leaves), 4.0);
  }

  #[test]
  fn combine_boost_rejects_overflow_product() {
    // Each factor on its own passes `validate_boost` (finite, non-negative),
    // but their product overflows `f32::MAX` to `+inf`. Before the guard,
    // that `+inf` would flow into `ScoredTerm.weight`, producing non-finite
    // BM25 scores and HTTP 500 on serialisation (BUG-381).
    let err = combine_boost(1e38, 1e38).unwrap_err();
    assert!(
      err.to_string().contains("overflows"),
      "expected overflow message, got: {err}",
    );
  }

  #[test]
  fn combine_boost_accepts_finite_product() {
    assert_eq!(combine_boost(2.0, 3.5).unwrap(), 7.0);
    assert_eq!(combine_boost(1.0, 1.0).unwrap(), 1.0);
    assert_eq!(combine_boost(0.0, 1e38).unwrap(), 0.0);
  }

  #[test]
  fn build_query_plan_rejects_nested_boost_overflow_via_bool_query_string() {
    // Mirrors the exact reproduction from BUG-381: a Bool with boost = 1e38
    // wrapping a QueryString with boost = 1e38. Each factor is individually
    // finite, but their product overflows `f32::MAX`. The planner now
    // rejects this at build time with a clear error instead of letting
    // `+inf` flow into term weights.
    let query = Query::Node(QueryNode::Bool {
      must: vec![QueryNode::QueryString {
        query: "hello".into(),
        fields: None,
        boost: Some(1e38),
      }],
      should: Vec::new(),
      must_not: Vec::new(),
      filter: Vec::new(),
      minimum_should_match: None,
      boost: Some(1e38),
    });
    let err = build_query_plan(&query, &["body".to_string()]).unwrap_err();
    assert!(
      err.to_string().contains("overflows"),
      "expected overflow error, got: {err}",
    );
  }

  #[test]
  fn build_query_plan_rejects_boost_overflow_on_constant_score() {
    // Before the guard, a `ConstantScore` wrapped in a boost-heavy Bool
    // produced `ScoreNode::Constant { score: +inf }` (BUG-370 addressed
    // the evaluation side, but the planner still silently built the
    // overflowed payload).
    let query = Query::Node(QueryNode::Bool {
      must: vec![QueryNode::ConstantScore {
        filter: Filter::KeywordEq {
          field: "tag".into(),
          value: "rust".into(),
        },
        boost: Some(1e38),
      }],
      should: Vec::new(),
      must_not: Vec::new(),
      filter: Vec::new(),
      minimum_should_match: None,
      boost: Some(1e38),
    });
    let err = build_query_plan(&query, &["body".to_string()]).unwrap_err();
    assert!(
      err.to_string().contains("overflows"),
      "expected overflow error, got: {err}",
    );
  }

  // -------- BUG-403 regression tests --------
  //
  // `resolve_minimum_should_match` must round percentage-based specs
  // *down* to match Elasticsearch's documented semantics: "The number
  // computed from the percentage is rounded down and used as the
  // minimum." Before the fix it used `ceil()`, which silently demanded
  // one more matching term than the user requested for any non-integer
  // product (e.g. `75%` of 3 terms became 3 instead of 2).
  #[test]
  fn resolve_minimum_should_match_percentage_rounds_down() {
    let cases: &[(&str, usize, usize)] = &[
      // (percentage, term_count, expected_required)
      ("75%", 3, 2),  // floor(2.25)   = 2, not ceil -> 3
      ("50%", 3, 1),  // floor(1.5)    = 1, not ceil -> 2
      ("34%", 3, 1),  // floor(1.02)   = 1, not ceil -> 2
      ("25%", 3, 0),  // floor(0.75)   = 0, not ceil -> 1
      ("60%", 5, 3),  // floor(3.0)    = 3 (integer product, unchanged)
      ("100%", 4, 4), // full match: integer product, unchanged
      ("0%", 3, 0),   // explicit zero-required: preserved
      ("10%", 1, 0),  // floor(0.1)    = 0, not ceil -> 1
      // Integer-product cases that would undercount if the intermediate
      // `percent/100` is stored as an inexact binary fraction and the
      // result is computed as `(percent / 100) * term_count` in f32/f64.
      // The fix must compute `(percent * term_count) / 100` (or, better,
      // use integer arithmetic for whole-number percents) so these land
      // on the exact integer rather than a hair below it.
      ("13%", 900, 117), // (13/100) * 900 in f32 is 116.9999…; must be 117
      ("70%", 10, 7),    // (70/100) * 10  can underflow to 6.9999…; must be 7
      ("7%", 1000, 70),  // (7/100)  * 1000 ≈ 69.9999… in f32; must be 70
      // Fractional percentages still exercise the float path.
      ("12.5%", 8, 1), // floor(1.0) = 1 (0.125 * 8 = 1.0 exactly in f64)
      ("33.3%", 3, 0), // floor(0.999) = 0
    ];
    for (pct, term_count, expected) in cases {
      let spec = Some(MinimumShouldMatch::Percentage((*pct).to_string()));
      let got = resolve_minimum_should_match(&spec, *term_count, MatchOperator::Or)
        .unwrap_or_else(|e| panic!("{pct} of {term_count} failed: {e}"));
      assert_eq!(
        got,
        Some(*expected),
        "{pct} of {term_count} terms: expected {expected}, got {got:?}"
      );
    }
  }

  #[test]
  fn resolve_minimum_should_match_value_caps_at_term_count() {
    // A raw `Value(v)` can exceed the number of terms, so this directly
    // exercises the final `.min(term_count)` cap. (Percentages can't
    // exceed it under the 0..=100 range check, so they don't reach the
    // cap — cover the path that actually does.)
    let spec = Some(MinimumShouldMatch::Value(10));
    let got = resolve_minimum_should_match(&spec, 4, MatchOperator::Or).unwrap();
    assert_eq!(got, Some(4));
  }

  #[test]
  fn build_query_plan_accepts_boost_product_within_range() {
    // Control case: a moderately large nested boost that stays finite
    // after multiplication must plan without error, so the guard does not
    // regress legitimate queries.
    let query = Query::Node(QueryNode::Bool {
      must: vec![QueryNode::Term {
        field: "body".into(),
        value: "rust".into(),
        boost: Some(1e9),
      }],
      should: Vec::new(),
      must_not: Vec::new(),
      filter: Vec::new(),
      minimum_should_match: None,
      boost: Some(1e9),
    });
    let plan = build_query_plan(&query, &["body".to_string()])
      .expect("finite nested boost product must plan cleanly");
    assert!(plan.scorer.is_some());
  }
}
