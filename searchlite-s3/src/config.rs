//! S3 configuration types.

/// Credentials for an S3-compatible endpoint.
#[derive(Debug, Clone)]
pub enum S3Credentials {
  /// Static credentials. Use when running outside an AWS environment
  /// or when targeting R2/MinIO.
  Static {
    access_key_id: String,
    secret_access_key: String,
    /// Optional session token. `None` for long-lived static credentials,
    /// `Some(...)` for temporary credentials issued by STS or an
    /// account-scoped R2 token.
    session_token: Option<String>,
  },
  /// Load credentials via [`aws_config`]'s default chain (env vars,
  /// IMDS, EC2 instance role, etc.). Use this when running inside AWS.
  LoadFromEnv,
}

/// Configuration for an [`crate::S3BlobStore`] instance.
#[derive(Debug, Clone)]
pub struct S3Config {
  /// Optional endpoint URL. `None` targets AWS S3; `Some(url)` for
  /// R2 (`https://<account>.r2.cloudflarestorage.com`), MinIO, or
  /// LocalStack.
  pub endpoint_url: Option<String>,
  /// Region. Required by SigV4 even for R2 (use `auto`) and MinIO.
  pub region: String,
  /// Bucket name. Validated at construction time (must be a legal
  /// S3 bucket name).
  pub bucket: String,
  /// Optional namespace within the bucket. Joined with each key as
  /// `{prefix}/{key}`. The prefix itself is validated against the
  /// same rules as keys: relative, non-empty after trim, no `..`,
  /// no backslashes, no platform prefix.
  pub prefix: Option<String>,
  /// Credential source. See [`S3Credentials`].
  pub credentials: S3Credentials,
  /// Whether the configured endpoint supports atomic conditional
  /// PUTs (`If-Match` / `If-None-Match`).
  ///
  /// * AWS S3: `true`.
  /// * R2: rolled out late 2024; default `false`. Opt in once
  ///   verified.
  /// * MinIO: `true`.
  /// * Endpoints that don't support conditional puts must set this
  ///   to `false`. Stage 10b's [`crate::S3BlobStore::put_if_match`]
  ///   refuses to issue a request when this flag is `false` rather
  ///   than silently sending an `If-Match` header that the endpoint
  ///   would ignore — better to surface a clear capability error.
  pub conditional_put: bool,
  /// Toggle path-style addressing (`https://endpoint/bucket/key`)
  /// vs virtual-hosted (`https://bucket.endpoint/key`).
  ///
  /// * AWS S3: `false` (virtual-hosted is preferred since 2020).
  /// * R2: `false`.
  /// * MinIO / LocalStack / **wiremock**: `true` — these don't
  ///   support virtual-hosted addressing.
  pub force_path_style: bool,
}

impl S3Config {
  /// Sensible defaults for a fresh AWS S3 connection. Caller must
  /// fill in `region`, `bucket`, and `credentials`.
  pub fn aws_default() -> Self {
    Self {
      endpoint_url: None,
      region: String::new(),
      bucket: String::new(),
      prefix: None,
      credentials: S3Credentials::LoadFromEnv,
      conditional_put: true,
      force_path_style: false,
    }
  }

  /// Sensible defaults for Cloudflare R2. Caller fills in `endpoint_url`,
  /// `bucket`, and `credentials`. Conditional PUTs default OFF — opt
  /// in only after verifying your account/bucket supports them.
  pub fn r2_default() -> Self {
    Self {
      endpoint_url: None,
      region: "auto".to_string(),
      bucket: String::new(),
      prefix: None,
      credentials: S3Credentials::Static {
        access_key_id: String::new(),
        secret_access_key: String::new(),
        session_token: None,
      },
      conditional_put: false,
      force_path_style: false,
    }
  }
}
