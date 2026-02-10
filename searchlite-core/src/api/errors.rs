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

#[derive(Debug, Error)]
pub enum PatchError {
  #[error("document not found")]
  DocumentNotFound,

  #[error("vector fields are not supported for updates")]
  VectorFieldsUnsupported,
}
