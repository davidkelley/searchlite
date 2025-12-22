use thiserror::Error;

#[derive(Debug, Error)]
pub enum AggregationError {
  #[error("aggregation requires fast field `{field}`")]
  MissingFastField { field: String },

  #[error("aggregation `{agg}` is not supported for field `{field}` (expected {expected})")]
  UnsupportedFieldType {
    agg: String,
    field: String,
    expected: String,
  },

  #[error("invalid aggregation configuration: {reason}")]
  InvalidConfig { reason: String },
}
