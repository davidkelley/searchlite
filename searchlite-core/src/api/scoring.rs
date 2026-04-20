use anyhow::{bail, Result};
use smallvec::smallvec;

use crate::api::query_eval::QueryEvaluator;
use crate::api::reader::FunctionExplanation;
use crate::api::types::{
  DecayFunction, FunctionBoostMode, FunctionScoreMode, RankFeatureModifier, SortOrder,
};
use crate::index::fastfields::FastFieldsReader;
use crate::index::manifest::Schema;
use crate::query::planner::{QueryMatcher, ScoreExpr, ScoreNode};
use crate::query::score_functions::{
  apply_boost_mode, combine_function_scores, compile_functions, CompiledFunction,
};
use crate::query::script::{compile_script, CompiledScript};
use crate::query::sort::{SortKey, SortKeyPart, SortValue};
use crate::query::util::ensure_numeric_fast as ensure_numeric_fast_field;
use crate::DocId;

pub(crate) fn score_sort_key(
  score: f32,
  segment_ord: u32,
  doc_id: DocId,
  order: SortOrder,
) -> SortKey {
  SortKey {
    parts: smallvec![SortKeyPart {
      order,
      value: SortValue::Score(score),
    }],
    segment_ord,
    doc_id,
  }
}

pub(crate) fn rank_numeric_value(
  reader: &FastFieldsReader,
  field: &str,
  doc_id: DocId,
  missing: f32,
) -> f64 {
  reader
    .f64_value(field, doc_id)
    .or_else(|| reader.i64_value(field, doc_id).map(|v| v as f64))
    .unwrap_or(missing as f64)
}

pub(crate) fn apply_rank_modifier(value: f64, modifier: &RankFeatureModifier) -> f64 {
  match modifier {
    RankFeatureModifier::None => value,
    RankFeatureModifier::Log => {
      if value <= 0.0 {
        0.0
      } else {
        value.log10()
      }
    }
    RankFeatureModifier::Log1p => {
      if value <= -1.0 {
        0.0
      } else {
        (1.0 + value).log10()
      }
    }
    RankFeatureModifier::Sqrt => {
      if value < 0.0 {
        0.0
      } else {
        value.sqrt()
      }
    }
    RankFeatureModifier::Reciprocal => {
      if value == 0.0 {
        0.0
      } else {
        1.0 / value
      }
    }
  }
}

#[derive(Clone, Debug)]
pub(crate) enum CompiledScoreNode {
  Empty,
  Expr(ScoreExpr),
  Sum(Vec<CompiledScoreNode>),
  DisMax {
    children: Vec<CompiledScoreNode>,
    tie_breaker: f32,
  },
  Constant {
    score: f32,
    matcher: QueryMatcher,
  },
  FunctionScore {
    matcher: QueryMatcher,
    base: Box<CompiledScoreNode>,
    functions: Vec<CompiledFunction>,
    score_mode: FunctionScoreMode,
    boost_mode: FunctionBoostMode,
    max_boost: Option<f32>,
    min_score: Option<f32>,
    boost: f32,
  },
  RankFeature {
    matcher: QueryMatcher,
    field: String,
    modifier: RankFeatureModifier,
    missing: f32,
    boost: f32,
  },
  ScriptScore {
    matcher: QueryMatcher,
    base: Box<CompiledScoreNode>,
    script: CompiledScript,
    boost: f32,
  },
}

pub(crate) fn compile_score_node(node: &ScoreNode, schema: &Schema) -> Result<CompiledScoreNode> {
  Ok(match node {
    ScoreNode::Empty => CompiledScoreNode::Empty,
    ScoreNode::Expr(expr) => CompiledScoreNode::Expr(expr.clone()),
    ScoreNode::Sum(children) => {
      let mut out = Vec::with_capacity(children.len());
      for child in children.iter() {
        out.push(compile_score_node(child, schema)?);
      }
      CompiledScoreNode::Sum(out)
    }
    ScoreNode::DisMax {
      children,
      tie_breaker,
    } => {
      let mut out = Vec::with_capacity(children.len());
      for child in children.iter() {
        out.push(compile_score_node(child, schema)?);
      }
      CompiledScoreNode::DisMax {
        children: out,
        tie_breaker: *tie_breaker,
      }
    }
    ScoreNode::Constant { score, matcher } => CompiledScoreNode::Constant {
      score: *score,
      matcher: matcher.clone(),
    },
    ScoreNode::FunctionScore {
      matcher,
      base,
      functions,
      score_mode,
      boost_mode,
      max_boost,
      min_score,
      boost,
    } => CompiledScoreNode::FunctionScore {
      matcher: matcher.clone(),
      base: Box::new(compile_score_node(base, schema)?),
      functions: compile_functions(functions, schema)?,
      score_mode: *score_mode,
      boost_mode: *boost_mode,
      max_boost: *max_boost,
      min_score: *min_score,
      boost: *boost,
    },
    ScoreNode::RankFeature {
      matcher,
      field,
      modifier,
      missing,
      boost,
    } => {
      let missing_val = missing.unwrap_or(0.0);
      if !missing_val.is_finite() {
        bail!("rank_feature `missing` must be finite");
      }
      ensure_numeric_fast_field(schema, field, "rank_feature")?;
      CompiledScoreNode::RankFeature {
        matcher: matcher.clone(),
        field: field.clone(),
        modifier: modifier.unwrap_or(RankFeatureModifier::None),
        missing: missing_val,
        boost: *boost,
      }
    }
    ScoreNode::ScriptScore {
      matcher,
      base,
      script,
      params,
      boost,
    } => CompiledScoreNode::ScriptScore {
      matcher: matcher.clone(),
      base: Box::new(compile_score_node(base, schema)?),
      script: compile_script(script, params, schema)?,
      boost: *boost,
    },
  })
}

pub(crate) fn has_custom_scoring(node: &CompiledScoreNode) -> bool {
  match node {
    CompiledScoreNode::Empty | CompiledScoreNode::Expr(_) => false,
    CompiledScoreNode::Sum(children) | CompiledScoreNode::DisMax { children, .. } => {
      children.iter().any(has_custom_scoring)
    }
    CompiledScoreNode::Constant { .. }
    | CompiledScoreNode::FunctionScore { .. }
    | CompiledScoreNode::RankFeature { .. }
    | CompiledScoreNode::ScriptScore { .. } => true,
  }
}

pub(crate) fn describe_function(func: &CompiledFunction, value: f32) -> FunctionExplanation {
  match func {
    CompiledFunction::Weight { .. } => FunctionExplanation {
      r#type: "weight".to_string(),
      value,
      field: None,
    },
    CompiledFunction::FieldValueFactor { field, .. } => FunctionExplanation {
      r#type: "field_value_factor".to_string(),
      value,
      field: Some(field.clone()),
    },
    CompiledFunction::Decay {
      field, function, ..
    } => {
      let name = match function {
        DecayFunction::Exp => "decay_exp",
        DecayFunction::Gauss => "decay_gauss",
        DecayFunction::Linear => "decay_linear",
      };
      FunctionExplanation {
        r#type: name.to_string(),
        value,
        field: Some(field.clone()),
      }
    }
  }
}

pub(crate) fn evaluate_compiled_score(
  node: &CompiledScoreNode,
  evaluator: &QueryEvaluator<'_>,
  fast_fields: &FastFieldsReader,
  doc_id: DocId,
  leaf_scores: &[f32],
  collect_functions: bool,
  out_functions: &mut Vec<FunctionExplanation>,
) -> Option<f32> {
  match node {
    CompiledScoreNode::Empty => Some(1.0),
    CompiledScoreNode::Expr(expr) => Some(expr.evaluate(leaf_scores)),
    CompiledScoreNode::Sum(children) => {
      let mut sum = 0.0_f32;
      let mut has_score = false;
      for child in children.iter() {
        if let Some(score) = evaluate_compiled_score(
          child,
          evaluator,
          fast_fields,
          doc_id,
          leaf_scores,
          collect_functions,
          out_functions,
        ) {
          has_score = true;
          sum += score;
        }
      }
      if has_score || children.is_empty() {
        // Individual child scores are guarded for finitude by their
        // respective nodes, but their sum can still overflow f32::MAX to
        // infinity when many finite children accumulate. Reject non-finite
        // sums so they do not leak into the sort key heap. Mirrors the
        // FunctionScore, RankFeature, ScriptScore, rescore, and hybrid
        // guards.
        if !sum.is_finite() {
          return None;
        }
        Some(sum)
      } else {
        None
      }
    }
    CompiledScoreNode::DisMax {
      children,
      tie_breaker,
    } => {
      if children.is_empty() {
        return Some(0.0);
      }
      let mut sum = 0.0_f32;
      let mut max = f32::NEG_INFINITY;
      let mut has_score = false;
      for child in children.iter() {
        if let Some(score) = evaluate_compiled_score(
          child,
          evaluator,
          fast_fields,
          doc_id,
          leaf_scores,
          collect_functions,
          out_functions,
        ) {
          has_score = true;
          max = max.max(score);
          sum += score;
        }
      }
      if has_score {
        // `sum` can overflow to infinity across many finite children and,
        // when `max` is also infinite, `sum - max` is NaN so the whole
        // expression becomes NaN. Reject non-finite results so they do not
        // leak into the sort key heap; matches the other scoring guards.
        //
        // Short-circuit when `tie_breaker == 0`: `0 * ∞` is `NaN` under
        // IEEE-754, so the naïve formula would drop the hit even though
        // zero-tie-breaker DisMax semantics is simply `max`. `max` is
        // always finite here because at least one child produced a finite
        // score (child scores are guarded by their respective nodes).
        let result = if *tie_breaker == 0.0 {
          max
        } else {
          max + *tie_breaker * (sum - max)
        };
        if !result.is_finite() {
          return None;
        }
        Some(result)
      } else {
        None
      }
    }
    CompiledScoreNode::Constant { score, matcher } => {
      if evaluator.matches_subquery(matcher, doc_id) {
        Some(*score)
      } else {
        Some(0.0)
      }
    }
    CompiledScoreNode::FunctionScore {
      matcher,
      base,
      functions,
      score_mode,
      boost_mode,
      max_boost,
      min_score,
      boost,
    } => {
      if !evaluator.matches_subquery(matcher, doc_id) {
        return Some(0.0);
      }
      let base_score = evaluate_compiled_score(
        base,
        evaluator,
        fast_fields,
        doc_id,
        leaf_scores,
        collect_functions,
        out_functions,
      )?;
      let mut function_values = Vec::new();
      let mut fn_expls = Vec::new();
      for func in functions.iter() {
        if let Some(val) = func.evaluate(fast_fields, doc_id) {
          function_values.push(val);
          if collect_functions {
            fn_expls.push(describe_function(func, val));
          }
        }
      }
      let mut effective_base = base_score;
      if effective_base.abs() <= f32::EPSILON
        && !function_values.is_empty()
        && *boost_mode == FunctionBoostMode::Multiply
      {
        // Preserve function contributions when the base query scored 0.0 and
        // the boost_mode is Multiply — otherwise `0 * func = 0` would erase
        // the function-only scoring. All other boost modes (Sum, Max, Min,
        // Replace) already preserve the function contribution without a
        // rewrite, so leaving `effective_base` at 0.0 gives the correct
        // result: Sum -> `0 + func = func`, Max -> `max(0, func)`, Min ->
        // `min(0, func)`, Replace ignores the base. Gating the rewrite on
        // Multiply prevents an artificial +1.0 bias in Sum, a 1.0 clamp in
        // Max when `func < 1.0`, and a 1.0 floor in Min when `func >= 1.0`.
        effective_base = 1.0;
      }
      let mut combined =
        if let Some(func_score) = combine_function_scores(&function_values, *score_mode) {
          // `func_score` can be non-finite if the combine step (Sum,
          // Multiply, or Avg) overflowed past f32::MAX to infinity, even
          // when every individual function value was finite. `max_boost`,
          // when set, caps infinity to a finite value because
          // `f32::INFINITY.min(finite) == finite`; when absent, we must
          // reject the doc rather than let infinity leak into the sort
          // key. Mirrors the RankFeature guards below and script.rs/aggs.
          let capped = match max_boost {
            Some(max) => func_score.min(*max),
            None => func_score,
          };
          if !capped.is_finite() {
            return None;
          }
          apply_boost_mode(effective_base, capped, *boost_mode)
        } else {
          effective_base
        };
      if !combined.is_finite() {
        return None;
      }
      if let Some(min) = min_score {
        if combined < *min {
          return None;
        }
      }
      combined *= *boost;
      if !combined.is_finite() {
        return None;
      }
      if collect_functions {
        out_functions.extend(fn_expls);
      }
      Some(combined)
    }
    CompiledScoreNode::RankFeature {
      matcher,
      field,
      modifier,
      missing,
      boost,
    } => {
      if !evaluator.matches_subquery(matcher, doc_id) {
        return Some(0.0);
      }
      let raw = rank_numeric_value(fast_fields, field, doc_id, *missing);
      let modified = apply_rank_modifier(raw, modifier);
      if !modified.is_finite() {
        return None;
      }
      let score = (modified as f32) * *boost;
      if !score.is_finite() {
        return None;
      }
      if collect_functions {
        out_functions.push(FunctionExplanation {
          r#type: "rank_feature".to_string(),
          value: score,
          field: Some(field.clone()),
        });
      }
      Some(score)
    }
    CompiledScoreNode::ScriptScore {
      matcher,
      base,
      script,
      boost,
    } => {
      if !evaluator.matches_subquery(matcher, doc_id) {
        return Some(0.0);
      }
      let base_score = evaluate_compiled_score(
        base,
        evaluator,
        fast_fields,
        doc_id,
        leaf_scores,
        collect_functions,
        out_functions,
      )?;
      let script_score = script.evaluate(fast_fields, doc_id, base_score)?;
      if !script_score.is_finite() {
        return None;
      }
      let score = script_score * *boost;
      if !score.is_finite() {
        return None;
      }
      if collect_functions {
        out_functions.push(FunctionExplanation {
          r#type: "script_score".to_string(),
          value: score,
          field: None,
        });
      }
      Some(score)
    }
  }
}
