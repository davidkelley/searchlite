pub fn path_is_within(base_path: &str, candidate: &str) -> bool {
  candidate == base_path
    || candidate
      .strip_prefix(base_path)
      .map(|suffix| suffix.starts_with('.'))
      .unwrap_or(false)
}

pub fn resolve_scoped_path(base_path: &str, maybe_relative: &str) -> String {
  if path_is_within(base_path, maybe_relative) {
    maybe_relative.to_string()
  } else {
    format!("{base_path}.{maybe_relative}")
  }
}

pub fn resolve_optional_scoped_path(scope_path: Option<&str>, maybe_relative: &str) -> String {
  if let Some(scope_path) = scope_path {
    resolve_scoped_path(scope_path, maybe_relative)
  } else {
    maybe_relative.to_string()
  }
}

#[cfg(test)]
mod tests {
  use super::{path_is_within, resolve_optional_scoped_path, resolve_scoped_path};

  #[test]
  fn path_scope_helpers_handle_absolute_and_relative_paths() {
    assert!(path_is_within("comment", "comment"));
    assert!(path_is_within("comment", "comment.reply"));
    assert!(!path_is_within("comment", "comments"));
    assert_eq!(
      resolve_scoped_path("comment", "author"),
      "comment.author".to_string()
    );
    assert_eq!(
      resolve_scoped_path("comment", "comment.author"),
      "comment.author".to_string()
    );
    assert_eq!(
      resolve_optional_scoped_path(Some("comment"), "author"),
      "comment.author".to_string()
    );
    assert_eq!(
      resolve_optional_scoped_path(None, "author"),
      "author".to_string()
    );
  }
}
