use anyhow::{anyhow, bail, Result};

use crate::index::manifest::{FieldKind, Schema};

/// Ensures a field exists, is numeric, and marked as fast for scoring contexts.
pub(crate) fn ensure_numeric_fast(schema: &Schema, field: &str, ctx: &str) -> Result<()> {
  let meta = schema
    .field_meta(field)
    .ok_or_else(|| anyhow!("{ctx} field `{field}` is not present in schema"))?;

  if !matches!(meta.kind, FieldKind::Numeric) {
    bail!("{ctx} field `{field}` must be a numeric fast field");
  }

  if !meta.fast {
    bail!("{ctx} field `{field}` must be fast");
  }

  Ok(())
}

/// Narrow a finite `f64` to `f32`, clamping values outside the `f32`
/// representable range to `f32::MIN` / `f32::MAX` instead of letting the
/// narrowing cast saturate to `±f32::INFINITY`.
///
/// Rust's `value as f32` cast saturates to `±f32::INFINITY` for any finite
/// `f64` whose magnitude exceeds `f32::MAX` (~3.4e38). Downstream score
/// paths reject non-finite values and silently drop the document, so a
/// legitimately large finite score would otherwise disappear from results.
/// Clamping mirrors the policy used by the `finite_or_zero` helper in
/// `query/aggs/mod.rs` for aggregation finalization.
///
/// `NaN` inputs propagate as-is (`clamp` is a no-op on `NaN`). Callers must
/// validate finitude on the `f64` input first if they want to reject `NaN`.
#[inline]
pub(crate) fn f64_to_finite_f32(value: f64) -> f32 {
  value.clamp(f32::MIN as f64, f32::MAX as f64) as f32
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn f64_to_finite_f32_clamps_overflow_to_f32_max() {
    assert_eq!(f64_to_finite_f32(1.0e40), f32::MAX);
    assert_eq!(f64_to_finite_f32(-1.0e40), f32::MIN);
  }

  #[test]
  fn f64_to_finite_f32_preserves_in_range_values() {
    assert_eq!(f64_to_finite_f32(0.0), 0.0);
    assert_eq!(f64_to_finite_f32(1.5), 1.5);
    assert_eq!(f64_to_finite_f32(-1.5), -1.5);
    assert_eq!(f64_to_finite_f32(f32::MAX as f64), f32::MAX);
    assert_eq!(f64_to_finite_f32(f32::MIN as f64), f32::MIN);
  }

  #[test]
  fn f64_to_finite_f32_clamps_f64_infinity() {
    assert_eq!(f64_to_finite_f32(f64::INFINITY), f32::MAX);
    assert_eq!(f64_to_finite_f32(f64::NEG_INFINITY), f32::MIN);
  }

  #[test]
  fn f64_to_finite_f32_propagates_nan() {
    assert!(f64_to_finite_f32(f64::NAN).is_nan());
  }

  #[test]
  fn f64_to_finite_f32_preserves_subnormals() {
    let tiny = 1.0e-40_f64;
    assert_eq!(f64_to_finite_f32(tiny), tiny as f32);
  }
}
