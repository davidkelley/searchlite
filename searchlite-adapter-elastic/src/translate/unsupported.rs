use thiserror::Error;

use crate::error::ESError;

#[derive(Debug, Clone, Error)]
#[error("elasticsearch feature `{feature}` is not supported by searchlite adapter: {detail}")]
pub struct Unsupported {
  pub feature: String,
  pub detail: String,
}

impl Unsupported {
  pub fn feature(feature: impl Into<String>) -> Self {
    Self {
      feature: feature.into(),
      detail: String::new(),
    }
  }

  pub fn with_detail(feature: impl Into<String>, detail: impl Into<String>) -> Self {
    Self {
      feature: feature.into(),
      detail: detail.into(),
    }
  }
}

impl From<Unsupported> for ESError {
  fn from(value: Unsupported) -> ESError {
    let reason = if value.detail.is_empty() {
      format!(
        "feature `{}` not supported by searchlite adapter",
        value.feature
      )
    } else {
      format!(
        "feature `{}` not supported by searchlite adapter: {}",
        value.feature, value.detail
      )
    };
    ESError::bad_request("x_content_parse_exception", reason)
  }
}
