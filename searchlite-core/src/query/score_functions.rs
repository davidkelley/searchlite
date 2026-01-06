use anyhow::{anyhow, bail, Result};

use crate::api::types::Filter;
use crate::api::types::{
  DecayFunction, FieldValueModifier, FunctionBoostMode, FunctionScoreMode, FunctionSpec,
};
use crate::index::fastfields::FastFieldsReader;
use crate::index::manifest::{FieldKind, Schema};
use crate::query::filters::passes_filter;
use crate::DocId;

#[derive(Debug, Clone)]
pub(crate) enum CompiledFunction {
  Weight {
    weight: f32,
    filter: Option<Filter>,
  },
  FieldValueFactor {
    field: String,
    factor: f32,
    modifier: FieldValueModifier,
    missing: f64,
    filter: Option<Filter>,
  },
  Decay {
    field: String,
    origin: f64,
    scale: f64,
    offset: f64,
    decay: f64,
    function: DecayFunction,
    filter: Option<Filter>,
  },
}

pub(crate) fn compile_functions(
  functions: &[FunctionSpec],
  schema: &Schema,
) -> Result<Vec<CompiledFunction>> {
  let mut compiled = Vec::with_capacity(functions.len());
  for func in functions.iter() {
    match func {
      FunctionSpec::Weight { weight, filter } => compiled.push(CompiledFunction::Weight {
        weight: *weight,
        filter: filter.clone(),
      }),
      FunctionSpec::FieldValueFactor {
        field,
        factor,
        modifier,
        missing,
        filter,
      } => {
        ensure_numeric_fast(schema, field)?;
        compiled.push(CompiledFunction::FieldValueFactor {
          field: field.clone(),
          factor: *factor,
          modifier: modifier.unwrap_or(FieldValueModifier::None),
          missing: missing.unwrap_or(0.0),
          filter: filter.clone(),
        });
      }
      FunctionSpec::Decay {
        field,
        origin,
        scale,
        offset,
        decay,
        function,
        filter,
      } => {
        ensure_numeric_fast(schema, field)?;
        if *scale <= 0.0 {
          bail!("decay scale must be > 0");
        }
        let decay = decay.unwrap_or(0.5);
        if decay <= 0.0 {
          bail!("decay factor must be > 0");
        }
        compiled.push(CompiledFunction::Decay {
          field: field.clone(),
          origin: *origin,
          scale: *scale,
          offset: offset.unwrap_or(0.0),
          decay,
          function: function.unwrap_or(DecayFunction::Exp),
          filter: filter.clone(),
        });
      }
    }
  }
  Ok(compiled)
}

pub(crate) fn combine_function_scores(values: &[f32], mode: FunctionScoreMode) -> Option<f32> {
  if values.is_empty() {
    return None;
  }
  let iter = values.iter().copied();
  match mode {
    FunctionScoreMode::Sum => Some(values.iter().copied().sum()),
    FunctionScoreMode::Multiply => Some(values.iter().copied().product()),
    FunctionScoreMode::Max => iter
      .reduce(|a, b| a.max(b))
      .or_else(|| values.first().copied()),
    FunctionScoreMode::Min => iter
      .reduce(|a, b| a.min(b))
      .or_else(|| values.first().copied()),
    FunctionScoreMode::Avg => Some(values.iter().copied().sum::<f32>() / values.len() as f32),
  }
}

pub(crate) fn apply_boost_mode(base: f32, func_score: f32, mode: FunctionBoostMode) -> f32 {
  match mode {
    FunctionBoostMode::Multiply => base * func_score,
    FunctionBoostMode::Sum => base + func_score,
    FunctionBoostMode::Replace => func_score,
    FunctionBoostMode::Max => base.max(func_score),
    FunctionBoostMode::Min => base.min(func_score),
  }
}

impl CompiledFunction {
  pub(crate) fn evaluate(&self, fast_fields: &FastFieldsReader, doc_id: DocId) -> Option<f32> {
    match self {
      CompiledFunction::Weight { weight, filter } => {
        if !filter_passes(filter, fast_fields, doc_id) {
          return None;
        }
        Some(*weight)
      }
      CompiledFunction::FieldValueFactor {
        field,
        factor,
        modifier,
        missing,
        filter,
      } => {
        if !filter_passes(filter, fast_fields, doc_id) {
          return None;
        }
        let raw = numeric_value(fast_fields, field, doc_id).unwrap_or(*missing);
        let scaled = raw * *factor as f64;
        let modified = apply_modifier(scaled, modifier);
        Some(modified as f32)
      }
      CompiledFunction::Decay {
        field,
        origin,
        scale,
        offset,
        decay,
        function,
        filter,
      } => {
        if !filter_passes(filter, fast_fields, doc_id) {
          return None;
        }
        let value = numeric_value(fast_fields, field, doc_id)?;
        let distance = (value - *origin).abs() - *offset;
        let norm = (distance.max(0.0)) / *scale;
        let score = decay_value(*decay, norm, function);
        if score.is_finite() {
          Some(score as f32)
        } else {
          None
        }
      }
    }
  }
}

fn decay_value(decay: f64, norm: f64, function: &DecayFunction) -> f64 {
  match function {
    DecayFunction::Exp => decay.powf(norm),
    DecayFunction::Gauss => decay.powf(norm * norm),
    DecayFunction::Linear => ((1.0 - norm) * (1.0 - decay) + decay).max(0.0),
  }
}

fn apply_modifier(value: f64, modifier: &FieldValueModifier) -> f64 {
  match modifier {
    FieldValueModifier::None => value,
    FieldValueModifier::Log => {
      if value <= 0.0 {
        0.0
      } else {
        value.ln()
      }
    }
    FieldValueModifier::Log1p => {
      if value <= -1.0 {
        0.0
      } else {
        value.ln_1p()
      }
    }
    FieldValueModifier::Log2p => {
      if value <= -1.0 {
        0.0
      } else {
        (value + 1.0).log2()
      }
    }
    FieldValueModifier::Sqrt => {
      if value < 0.0 {
        0.0
      } else {
        value.sqrt()
      }
    }
    FieldValueModifier::Reciprocal => {
      if value == 0.0 {
        0.0
      } else {
        1.0 / value
      }
    }
  }
}

fn numeric_value(reader: &FastFieldsReader, field: &str, doc_id: DocId) -> Option<f64> {
  reader
    .f64_value(field, doc_id)
    .or_else(|| reader.i64_value(field, doc_id).map(|v| v as f64))
}

fn ensure_numeric_fast(schema: &Schema, field: &str) -> Result<()> {
  let Some(meta) = schema.field_meta(field) else {
    bail!("function_score field `{field}` is not present in schema");
  };
  match meta.kind {
    FieldKind::Numeric => {
      if !meta.fast {
        bail!("function_score field `{field}` must be fast");
      }
      Ok(())
    }
    _ => Err(anyhow!(
      "function_score field `{field}` must be a numeric fast field"
    )),
  }
}

fn filter_passes(filter: &Option<Filter>, reader: &FastFieldsReader, doc_id: DocId) -> bool {
  match filter {
    None => true,
    Some(f) => passes_filter(reader, doc_id, f),
  }
}
