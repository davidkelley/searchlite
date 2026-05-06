//! `s3://` URL parsing for CLI / HTTP-style command lines.
//!
//! The grammar is deliberately narrow: bucket + optional prefix, no
//! credentials, region, or endpoint embedded. Connection-level knobs
//! (region, endpoint URL, force-path-style, conditional-put) live on
//! sibling flags so the same URL can target AWS S3, Cloudflare R2,
//! and MinIO without reshaping.
//!
//! ```text
//! s3://<bucket>                  → bucket="...", prefix=None
//! s3://<bucket>/                 → bucket="...", prefix=None
//! s3://<bucket>/<prefix>         → bucket="...", prefix=Some("...")
//! s3://<bucket>/<a>/<b>/<c>      → bucket="...", prefix=Some("a/b/c")
//! ```
//!
//! Trailing slashes on the prefix are normalized away (`s3://b/p/` →
//! `Some("p")`). Empty prefixes ("`s3://b/`", "`s3://b//`") collapse
//! to `None`. Scheme is case-insensitive (`S3://` is accepted) so
//! pasting from an AWS console URL works out of the box.

/// Parsed `s3://bucket/prefix` URL bits.
///
/// Both fields are owned `String`s so callers can move them into an
/// [`crate::S3Config`] without lifetime juggling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S3Url {
  /// Bucket name. Validated for non-emptiness only — full bucket-name
  /// rules (3-63 chars, RFC 1123 label, etc.) are enforced inside
  /// [`crate::S3BlobStore::new`] when the config is realized into an
  /// SDK client, so the same validation runs whether the user came
  /// in via this URL parser or built [`crate::S3Config`] by hand.
  pub bucket: String,
  /// Optional prefix within the bucket. `None` means "use the
  /// bucket root". The returned string never has leading or trailing
  /// `/`. Internal `/` separators between segments are preserved.
  pub prefix: Option<String>,
}

/// Parse `s3://<bucket>[/<prefix>]`. Returns a structured error
/// string suitable for surfacing on a `clap` parse failure.
///
/// See the module-level docs for the grammar.
pub fn parse_s3_url(url: &str) -> Result<S3Url, String> {
  let trimmed = url.trim();
  // Case-insensitive scheme check (so `S3://...` from a copy-paste
  // works), but the canonical lowercase form is what we'll use in
  // error messages.
  let after_scheme = trimmed
    .strip_prefix("s3://")
    .or_else(|| trimmed.strip_prefix("S3://"))
    .ok_or_else(|| format!("expected an s3://bucket[/prefix] URL, got {url:?}"))?;
  let (bucket_raw, prefix_raw) = match after_scheme.split_once('/') {
    Some((bucket, prefix)) => (bucket, prefix),
    None => (after_scheme, ""),
  };
  let bucket = bucket_raw.trim();
  if bucket.is_empty() {
    return Err(format!(
      "s3 URL is missing a bucket name: {url:?} (expected `s3://<bucket>[/<prefix>]`)"
    ));
  }
  // Normalize the prefix: trim leading/trailing slashes so an
  // accidental `s3://b/p/` matches `s3://b/p`. Collapsing runs of
  // `/` is intentionally NOT done here — the canonical-key check
  // inside `sync_to_s3`'s preflight rejects them, and we want that
  // single source of truth rather than silently fixing up the URL.
  let prefix_trimmed = prefix_raw.trim_matches('/');
  let prefix = if prefix_trimmed.is_empty() {
    None
  } else {
    Some(prefix_trimmed.to_string())
  };
  Ok(S3Url {
    bucket: bucket.to_string(),
    prefix,
  })
}

/// Heuristic: does this `endpoint_url` look like a Cloudflare R2
/// endpoint? Returns `true` for any host of the form
/// `<account-id>.r2.cloudflarestorage.com`, including with a
/// `https://` scheme prefix and arbitrary trailing path/port.
///
/// Used by CLI / HTTP front-ends to default
/// [`crate::S3Config::conditional_put`] to `false` when the user
/// passed an R2 endpoint without explicitly setting the flag — R2's
/// conditional-PUT support rolled out in late 2024 and is opt-in per
/// account, so the safer default is to refuse `If-Match` requests
/// rather than silently send headers an older endpoint would ignore.
/// Callers always retain explicit override.
pub fn is_r2_endpoint(endpoint_url: &str) -> bool {
  let trimmed = endpoint_url.trim();
  let host = trimmed
    .strip_prefix("https://")
    .or_else(|| trimmed.strip_prefix("http://"))
    .unwrap_or(trimmed);
  // Drop any path suffix (everything from the first `/` onward).
  // Falling back to `host` here is critical when there's no `/` —
  // we need the value computed up to this point, not the outer
  // pre-split string with its trailing slash still attached.
  let host = match host.split_once('/') {
    Some((h, _)) => h,
    None => host,
  };
  // Drop any port suffix (everything from the first `:` onward).
  let host = match host.split_once(':') {
    Some((h, _)) => h,
    None => host,
  };
  host.ends_with(".r2.cloudflarestorage.com")
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn parses_bucket_only() {
    let u = parse_s3_url("s3://my-bucket").unwrap();
    assert_eq!(u.bucket, "my-bucket");
    assert_eq!(u.prefix, None);
  }

  #[test]
  fn parses_bucket_with_trailing_slash_no_prefix() {
    let u = parse_s3_url("s3://my-bucket/").unwrap();
    assert_eq!(u.bucket, "my-bucket");
    assert_eq!(u.prefix, None);
  }

  #[test]
  fn parses_bucket_and_single_segment_prefix() {
    let u = parse_s3_url("s3://my-bucket/products").unwrap();
    assert_eq!(u.bucket, "my-bucket");
    assert_eq!(u.prefix.as_deref(), Some("products"));
  }

  #[test]
  fn parses_bucket_and_multi_segment_prefix() {
    let u = parse_s3_url("s3://my-bucket/products/v1/active").unwrap();
    assert_eq!(u.bucket, "my-bucket");
    assert_eq!(u.prefix.as_deref(), Some("products/v1/active"));
  }

  #[test]
  fn trims_trailing_slash_on_prefix() {
    let u = parse_s3_url("s3://my-bucket/products/v1/").unwrap();
    assert_eq!(u.prefix.as_deref(), Some("products/v1"));
  }

  #[test]
  fn empty_prefix_collapses_to_none() {
    let u = parse_s3_url("s3://my-bucket//").unwrap();
    assert_eq!(u.prefix, None);
  }

  #[test]
  fn accepts_uppercase_scheme() {
    let u = parse_s3_url("S3://my-bucket/products").unwrap();
    assert_eq!(u.bucket, "my-bucket");
    assert_eq!(u.prefix.as_deref(), Some("products"));
  }

  #[test]
  fn trims_surrounding_whitespace() {
    let u = parse_s3_url("  s3://my-bucket/products  ").unwrap();
    assert_eq!(u.bucket, "my-bucket");
    assert_eq!(u.prefix.as_deref(), Some("products"));
  }

  #[test]
  fn rejects_non_s3_scheme() {
    let err = parse_s3_url("https://my-bucket.s3.amazonaws.com").unwrap_err();
    assert!(err.contains("expected an s3://"));
  }

  #[test]
  fn rejects_missing_scheme() {
    let err = parse_s3_url("my-bucket/prefix").unwrap_err();
    assert!(err.contains("expected an s3://"));
  }

  #[test]
  fn rejects_empty_bucket() {
    let err = parse_s3_url("s3:///products").unwrap_err();
    assert!(err.contains("missing a bucket"));
  }

  #[test]
  fn rejects_empty_url() {
    assert!(parse_s3_url("").is_err());
    assert!(parse_s3_url("   ").is_err());
  }

  #[test]
  fn r2_endpoint_detection_matches_canonical_form() {
    assert!(is_r2_endpoint(
      "https://1234567890abcdef.r2.cloudflarestorage.com"
    ));
    assert!(is_r2_endpoint(
      "http://1234567890abcdef.r2.cloudflarestorage.com"
    ));
    assert!(is_r2_endpoint("1234567890abcdef.r2.cloudflarestorage.com"));
    assert!(is_r2_endpoint(
      "https://1234567890abcdef.r2.cloudflarestorage.com/"
    ));
    assert!(is_r2_endpoint(
      "https://1234567890abcdef.r2.cloudflarestorage.com:443"
    ));
  }

  #[test]
  fn r2_endpoint_detection_rejects_non_r2() {
    assert!(!is_r2_endpoint("https://s3.amazonaws.com"));
    assert!(!is_r2_endpoint(
      "https://my-bucket.s3.us-east-1.amazonaws.com"
    ));
    assert!(!is_r2_endpoint("http://localhost:9000"));
    assert!(!is_r2_endpoint("https://minio.example.com"));
    assert!(!is_r2_endpoint(""));
  }
}
