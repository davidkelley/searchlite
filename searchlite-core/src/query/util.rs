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
