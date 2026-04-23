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

/// Typed errors for the write-key auth path. Callers (notably the FFI) match
/// on these variants via `anyhow::Error::downcast_ref` rather than sniffing
/// the error's `Display` string — editing a message here will no longer
/// silently reclassify an auth failure as a generic write failure downstream.
#[derive(Debug, Error)]
pub enum WriteKeyError {
  /// The index enforces a write key, but the caller did not provide one.
  #[error("write key required for this index")]
  Required,

  /// The index enforces a write key, but the caller provided the wrong one
  /// (hash mismatch against the manifest) or a segment/WAL binding cannot be
  /// re-derived from it (suggesting the index metadata was tampered).
  #[error("write key does not match: {0}")]
  Mismatch(&'static str),

  /// The manifest records no write-key hash, yet at least one segment or
  /// the WAL carries a binding — the manifest file was edited after the fact.
  #[error("write key metadata missing but bindings exist; index metadata was likely tampered")]
  MetadataTampered,

  /// The caller supplied an empty string where a non-empty write key was
  /// expected (rejected before any key material is written to disk).
  #[error("write key cannot be empty")]
  Empty,

  /// The caller hit a write-key code path but the `write-key` Cargo feature
  /// was not enabled at build time.
  #[error("write-key feature is not enabled; rebuild with `--features write-key`")]
  FeatureDisabled,
}

impl WriteKeyError {
  /// Convenience predicate for the `Mismatch(_)` match arm so callers that
  /// just want "is this an auth error?" don't have to name a specific reason.
  pub const fn is_auth_variant(&self) -> bool {
    matches!(
      self,
      WriteKeyError::Required
        | WriteKeyError::Mismatch(_)
        | WriteKeyError::MetadataTampered
        | WriteKeyError::Empty
        | WriteKeyError::FeatureDisabled
    )
  }
}

#[cfg(test)]
mod tests {
  use super::WriteKeyError;

  #[test]
  fn downcast_recovers_variant_after_anyhow_conversion() {
    let err: anyhow::Error = WriteKeyError::Required.into();
    let inner = err.downcast_ref::<WriteKeyError>().expect("must downcast");
    assert!(matches!(inner, WriteKeyError::Required));
  }

  #[test]
  fn downcast_recovers_variant_through_context_chain() {
    // anyhow::Context wrapping must not hide the typed error from
    // downcast_ref — this is the property BUG-020's substring match
    // could not honour (context() shadows the Display string).
    let err: anyhow::Error = WriteKeyError::MetadataTampered.into();
    let err = err.context("opening writer").context("during compaction");
    let inner = err.downcast_ref::<WriteKeyError>().expect("must downcast");
    assert!(matches!(inner, WriteKeyError::MetadataTampered));
  }

  #[test]
  fn display_messages_are_stable_for_human_readers() {
    // Pin the message text so accidental edits become visible in review.
    // Downstream classification does NOT depend on these strings after
    // BUG-020's fix, but library users still read the messages.
    assert_eq!(
      WriteKeyError::Required.to_string(),
      "write key required for this index"
    );
    assert_eq!(
      WriteKeyError::Empty.to_string(),
      "write key cannot be empty"
    );
    assert_eq!(
      WriteKeyError::MetadataTampered.to_string(),
      "write key metadata missing but bindings exist; index metadata was likely tampered"
    );
    assert_eq!(
      WriteKeyError::FeatureDisabled.to_string(),
      "write-key feature is not enabled; rebuild with `--features write-key`"
    );
    assert_eq!(
      WriteKeyError::Mismatch("WAL binding; index may be tampered").to_string(),
      "write key does not match: WAL binding; index may be tampered"
    );
  }

  #[test]
  fn all_variants_are_classified_as_auth() {
    // Exhaustive match — adding a new non-auth variant in the future will
    // force the author to decide how to classify it here.
    for v in [
      WriteKeyError::Required,
      WriteKeyError::Mismatch("x"),
      WriteKeyError::MetadataTampered,
      WriteKeyError::Empty,
      WriteKeyError::FeatureDisabled,
    ] {
      assert!(v.is_auth_variant(), "{v:?} must be auth-classified");
    }
  }
}
