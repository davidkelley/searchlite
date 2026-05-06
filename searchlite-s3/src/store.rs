//! `S3BlobStore` — concrete [`BlobStore`] impl over `aws-sdk-s3`.

use std::ops::Range;
use std::path::{Component, Path};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use base64::Engine;
use bytes::Bytes;

use aws_credential_types::Credentials;
use aws_sdk_s3::config::{Builder as SdkConfigBuilder, Region};
use aws_sdk_s3::error::SdkError;
use aws_sdk_s3::primitives::ByteStream;
use aws_smithy_runtime_api::http::Response as SmithyResponse;

use searchlite_core::storage::blob::{
  ArtifactIdentity, BlobStore, Capabilities, ContentHash, Object, ObjectStat, ObjectWriter,
  ProviderChecksum, PutIfMatchError,
};

use crate::config::{S3Config, S3Credentials};
use crate::errors::{not_found_anyhow, S3StoreError};
use crate::object::{classify_conditional_status, S3Object};

/// Concrete [`BlobStore`] over S3-compatible APIs.
///
/// Tested against AWS S3, Cloudflare R2, MinIO, and (in unit tests)
/// `wiremock`. See the crate-level docs for the supported targets and
/// their config defaults.
pub struct S3BlobStore {
  client: Arc<aws_sdk_s3::Client>,
  config: Arc<S3Config>,
}

impl S3BlobStore {
  /// Construct an [`S3BlobStore`] from an [`S3Config`]. Validates the
  /// bucket name + optional prefix; loads credentials per
  /// `S3Credentials`.
  pub async fn new(config: S3Config) -> Result<Self> {
    validate_bucket_name(&config.bucket)?;
    if let Some(prefix) = &config.prefix {
      validate_relative_segment(prefix, "prefix")?;
    }
    let region = Region::new(config.region.clone());
    let mut builder = aws_sdk_s3::Config::builder()
      .region(region.clone())
      .force_path_style(config.force_path_style)
      .behavior_version(aws_sdk_s3::config::BehaviorVersion::latest());
    if let Some(endpoint) = &config.endpoint_url {
      builder = builder.endpoint_url(endpoint.clone());
    }
    builder = match &config.credentials {
      S3Credentials::Static {
        access_key_id,
        secret_access_key,
        session_token,
      } => builder.credentials_provider(Credentials::new(
        access_key_id,
        secret_access_key,
        session_token.clone(),
        None,
        "searchlite-s3-static",
      )),
      S3Credentials::LoadFromEnv => apply_env_credentials(builder, &region).await?,
    };
    // Stage 10c v3: serialize the synchronous Client::from_conf
    // step so the aws-smithy-http-client TLS provider's
    // native-roots load doesn't race across concurrent callers in
    // the same process. The panic surfaces on macOS as "TrustStore
    // configured to enable native roots but no valid root
    // certificates parsed!" when multiple threads hit the keychain
    // at the same time. The guard is held only across the sync
    // builder finalize (no .await inside), so the lock is fully
    // contained and clippy's `await_holding_lock` lint is happy.
    let client = {
      static SDK_INIT_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());
      let _guard = SDK_INIT_GUARD
        .lock()
        .map_err(|e| anyhow!("S3BlobStore::new: SDK init guard poisoned: {e}"))?;
      aws_sdk_s3::Client::from_conf(builder.build())
    };
    Ok(Self {
      client: Arc::new(client),
      config: Arc::new(config),
    })
  }

  /// Validate + prefix-join a `BlobStore` key onto the configured
  /// bucket namespace.
  ///
  /// Rejects: empty/whitespace-only keys, absolute paths
  /// (`/foo`), `..` traversal components, backslash separators
  /// (Windows-style), `Component::Prefix` (Windows drive letter), and
  /// non-UTF8 paths. Joins with `S3Config.prefix` if set.
  fn resolve_key(&self, key: &Path) -> Result<String> {
    let bare = path_to_relative_string(key)?;
    Ok(match &self.config.prefix {
      Some(prefix) => {
        // The prefix itself was validated at construction time, so
        // we just join. Strip a trailing `/` from prefix if present
        // to keep the joined key canonical.
        let trimmed = prefix.trim_end_matches('/');
        if trimmed.is_empty() {
          bare
        } else {
          format!("{trimmed}/{bare}")
        }
      }
      None => bare,
    })
  }

  fn bucket(&self) -> &str {
    &self.config.bucket
  }
}

#[async_trait]
impl BlobStore for S3BlobStore {
  async fn stat(&self, key: &Path) -> Result<ObjectStat> {
    let resolved = self.resolve_key(key)?;
    head_to_stat(self.client.as_ref(), self.bucket(), &resolved).await
  }

  async fn open(&self, key: &Path) -> Result<Arc<dyn Object>> {
    let resolved = self.resolve_key(key)?;
    let stat = head_to_stat(self.client.as_ref(), self.bucket(), &resolved).await?;
    Ok(Arc::new(S3Object {
      client: self.client.clone(),
      bucket: self.bucket().to_string(),
      key: resolved,
      stat,
    }))
  }

  async fn get_range(&self, key: &Path, range: Range<u64>) -> Result<Bytes> {
    if range.start > range.end {
      bail!(
        "S3BlobStore::get_range: inverted range {}..{}",
        range.start,
        range.end
      );
    }
    // Stage 10b v2 [P3] (Codex review): always validate the key and
    // HEAD the object before deciding whether to short-circuit. The
    // previous shape returned `Bytes::new()` for any `start == end`
    // input WITHOUT validating the key or the object's bounds, which
    // let invalid keys and out-of-bounds zero-width ranges silently
    // succeed. The trait contract requires `start <= end <= len`, so
    // an OOB zero-width range MUST error.
    let resolved = self.resolve_key(key)?;
    // Stage 10b [Codex review #1]: HEAD first to learn the current
    // stat, then issue GET with `If-Match` against that ETag. Closes
    // the stat/get race for top-level (non-`open`) callers.
    let stat = head_to_stat(self.client.as_ref(), self.bucket(), &resolved).await?;
    if range.end > stat.len {
      bail!(
        "S3BlobStore::get_range: range {}..{} exceeds object length {} for {}",
        range.start,
        range.end,
        stat.len,
        resolved
      );
    }
    if range.start == range.end {
      // Validation passed — zero-width range is logically empty,
      // skip the GET to avoid a malformed `bytes=N-(N-1)` header.
      return Ok(Bytes::new());
    }
    let header = format!("bytes={}-{}", range.start, range.end - 1);
    let mut req = self
      .client
      .get_object()
      .bucket(self.bucket())
      .key(&resolved)
      .range(header);
    let sent_conditional = stat.provider_version.is_some();
    if let Some(etag) = stat.provider_version.as_deref() {
      req = req.if_match(etag);
    }
    // Stage 10b v4 [P3] (Codex review): only use the conditional-
    // aware mapper when `If-Match` was actually sent.
    let resp = req.send().await.map_err(|sdk_err| {
      let ctx = format!("get_range {resolved}");
      if sent_conditional {
        map_conditional_sdk_error(sdk_err, &ctx)
      } else {
        map_sdk_error(sdk_err, &ctx)
      }
    })?;
    let bytes = resp
      .body
      .collect()
      .await
      .map_err(anyhow::Error::new)?
      .into_bytes();
    Ok(bytes)
  }

  async fn get(&self, key: &Path) -> Result<Bytes> {
    let resolved = self.resolve_key(key)?;
    let resp = self
      .client
      .get_object()
      .bucket(self.bucket())
      .key(&resolved)
      .send()
      .await
      .map_err(|sdk_err| map_sdk_error(sdk_err, &format!("get {resolved}")))?;
    let bytes = resp
      .body
      .collect()
      .await
      .map_err(anyhow::Error::new)?
      .into_bytes();
    Ok(bytes)
  }

  async fn put(&self, key: &Path, body: Bytes) -> Result<ObjectStat> {
    let resolved = self.resolve_key(key)?;
    let len = body.len() as u64;
    let resp = self
      .client
      .put_object()
      .bucket(self.bucket())
      .key(&resolved)
      .body(ByteStream::from(body))
      .send()
      .await
      .map_err(|sdk_err| map_sdk_error(sdk_err, &format!("put {resolved}")))?;
    Ok(ObjectStat {
      len,
      provider_version: resp.e_tag().map(|s| s.to_string()),
      provider_checksum: parse_response_checksum(&resp),
    })
  }

  async fn put_stream(&self, key: &Path) -> Result<Box<dyn ObjectWriter>> {
    let resolved = self.resolve_key(key)?;
    Ok(Box::new(crate::store::stream_writer::S3StreamWriter::new(
      self.client.clone(),
      self.bucket().to_string(),
      resolved,
    )))
  }

  async fn put_if_match(
    &self,
    key: &Path,
    body: Bytes,
    expected: Option<&str>,
  ) -> std::result::Result<ObjectStat, PutIfMatchError> {
    // Stage 10b [Codex review #5]: refuse the call when
    // `conditional_put` is false — silently sending the header to an
    // unsupported endpoint would let races slip through unnoticed.
    if !self.config.conditional_put {
      return Err(PutIfMatchError::Other(anyhow::Error::from(
        S3StoreError::ConditionalPutNotSupported,
      )));
    }
    let resolved = self.resolve_key(key).map_err(PutIfMatchError::Other)?;
    let len = body.len() as u64;
    let mut req = self
      .client
      .put_object()
      .bucket(self.bucket())
      .key(&resolved)
      .body(ByteStream::from(body));
    match expected {
      Some(etag) => req = req.if_match(etag),
      // None → must-not-exist contract; send `If-None-Match: *`.
      None => req = req.if_none_match("*"),
    }
    let resp = match req.send().await {
      Ok(resp) => resp,
      Err(sdk_err) => {
        // Detect 412/409 conditional failure. The SDK exposes the
        // raw response status via `ServiceError::raw().status()`.
        if let Some(stat) = classify_conditional_status_from_sdk_error(&sdk_err, &resolved) {
          // Best-effort: try to surface the current stat so the
          // caller can retry against the new identity.
          let current = head_to_stat(self.client.as_ref(), self.bucket(), &resolved)
            .await
            .ok();
          let _ = stat;
          return Err(PutIfMatchError::Conflict { current });
        }
        return Err(PutIfMatchError::Other(map_sdk_error(
          sdk_err,
          &format!("put_if_match {resolved}"),
        )));
      }
    };
    Ok(ObjectStat {
      len,
      provider_version: resp.e_tag().map(|s| s.to_string()),
      provider_checksum: parse_response_checksum(&resp),
    })
  }

  async fn delete(&self, key: &Path) -> Result<()> {
    let resolved = self.resolve_key(key)?;
    match self
      .client
      .delete_object()
      .bucket(self.bucket())
      .key(&resolved)
      .send()
      .await
    {
      Ok(_) => Ok(()),
      Err(sdk_err) => {
        // S3 DeleteObject is idempotent in spirit: 404 / NoSuchKey
        // is fine.
        if is_not_found(&sdk_err) {
          return Ok(());
        }
        Err(map_sdk_error(sdk_err, &format!("delete {resolved}")))
      }
    }
  }

  fn capabilities(&self) -> Capabilities {
    Capabilities {
      conditional_put: self.config.conditional_put,
      multipart_upload: true,
      mmap_friendly: false,
    }
  }
}

// ─────────────────────── helpers ──────────────────────────────────────

/// Issue HEAD against `bucket/key` and parse the response into an
/// [`ObjectStat`]. NotFound is mapped via [`not_found_anyhow`] so the
/// `BlobStoreAdapter` chain walk surfaces `io::ErrorKind::NotFound`.
async fn head_to_stat(client: &aws_sdk_s3::Client, bucket: &str, key: &str) -> Result<ObjectStat> {
  match client.head_object().bucket(bucket).key(key).send().await {
    Ok(resp) => Ok(ObjectStat {
      len: resp.content_length().unwrap_or(0).max(0) as u64,
      // ETag preserved verbatim; some providers wrap in quotes,
      // others don't, and `If-Match` requires byte-equality. Don't
      // strip.
      provider_version: resp.e_tag().map(|s| s.to_string()),
      provider_checksum: parse_head_checksum(&resp),
    }),
    Err(sdk_err) => {
      if is_not_found(&sdk_err) {
        Err(not_found_anyhow(key))
      } else {
        Err(map_sdk_error(sdk_err, &format!("head {key}")))
      }
    }
  }
}

/// Map an SDK error from a **non-conditional** request into an
/// `anyhow::Error`. Special-cases:
///
/// * **404 NotFound** → [`not_found_anyhow`] so the chain walk
///   surfaces a real `io::Error` of kind `NotFound` (the contract
///   `BlobStoreAdapter::error_is_not_found` depends on).
///
/// 412/409 are NOT special-cased here: a plain `get`/`put`/`delete`/
/// `head` did not send a precondition header, so labeling those
/// codes as [`S3StoreError::PreconditionFailed`] would be misleading.
/// Use [`map_conditional_sdk_error`] at call sites that DID send
/// `If-Match` / `If-None-Match`.
pub(crate) fn map_sdk_error<E>(err: SdkError<E, SmithyResponse>, ctx: &str) -> anyhow::Error
where
  E: std::error::Error + Send + Sync + 'static,
{
  if let SdkError::ServiceError(svc) = &err {
    let status = svc.raw().status().as_u16();
    if status == 404 {
      // Synthesize a NotFound chain rooted at `io::Error{kind:
      // NotFound}` so `BlobStoreAdapter::error_is_not_found` can
      // detect it. Wrap with the original SDK error (preserved via
      // anyhow context) and the call-site `ctx` so debug info isn't
      // lost when callers print the error chain.
      let key = ctx.split_whitespace().last().unwrap_or(ctx);
      return not_found_anyhow(key).context(format!("{ctx} ({err})"));
    }
  }
  anyhow::Error::new(err).context(ctx.to_string())
}

/// Conditional-aware variant of [`map_sdk_error`]. Adds 412/409 →
/// [`S3StoreError::PreconditionFailed`] on top of the standard
/// 404/generic mapping. Use this only at call sites that issue a
/// conditional request (`If-Match` / `If-None-Match`); the typed
/// `PreconditionFailed` discriminant is meaningful only when the
/// request actually carried a precondition header.
pub(crate) fn map_conditional_sdk_error<E>(
  err: SdkError<E, SmithyResponse>,
  ctx: &str,
) -> anyhow::Error
where
  E: std::error::Error + Send + Sync + 'static,
{
  if let SdkError::ServiceError(svc) = &err {
    let status = svc.raw().status().as_u16();
    if status == 404 {
      // Same shape as `map_sdk_error`: NotFound chain root + SDK
      // detail attached as context.
      let key = ctx.split_whitespace().last().unwrap_or(ctx);
      return not_found_anyhow(key).context(format!("{ctx} ({err})"));
    }
    if let Some(typed) = classify_conditional_status(status, ctx) {
      // Build a chain rooted at the typed `S3StoreError` so callers
      // can downcast. The original SDK error is attached as context
      // for debugging.
      return anyhow::Error::new(typed).context(format!("{ctx}: {err}"));
    }
  }
  anyhow::Error::new(err).context(ctx.to_string())
}

/// Detect S3 NotFound / NoSuchKey across the SDK's error variants.
fn is_not_found<E>(err: &SdkError<E, SmithyResponse>) -> bool
where
  E: std::error::Error,
{
  match err {
    SdkError::ServiceError(svc) => svc.raw().status().as_u16() == 404,
    _ => false,
  }
}

/// Inspect an SDK error for 412/409 conditional-conflict status codes.
fn classify_conditional_status_from_sdk_error<E>(
  err: &SdkError<E, SmithyResponse>,
  key: &str,
) -> Option<S3StoreError>
where
  E: std::error::Error,
{
  if let SdkError::ServiceError(svc) = err {
    let status = svc.raw().status().as_u16();
    if let Some(s) = classify_conditional_status(status, key) {
      return Some(s);
    }
    let _ = svc;
  }
  let _ = err;
  None
}

/// Validate that a string is a usable S3 key segment: non-empty,
/// no `..` traversal, no backslash, no `Prefix` component, no
/// absolute leading `/`. Used for both keys and prefix.
fn validate_relative_segment(s: &str, label: &str) -> Result<()> {
  if s.trim().is_empty() {
    bail!("S3 {label}: empty");
  }
  if s.starts_with('/') {
    bail!("S3 {label}: must not start with `/`: {s:?}");
  }
  if s.contains('\\') {
    bail!("S3 {label}: backslash separators are not supported: {s:?}");
  }
  let p = Path::new(s);
  for c in p.components() {
    match c {
      Component::ParentDir => bail!("S3 {label}: contains `..`: {s:?}"),
      Component::Prefix(_) => {
        bail!("S3 {label}: contains a platform prefix (e.g. drive letter): {s:?}")
      }
      Component::RootDir => bail!("S3 {label}: contains a root component: {s:?}"),
      _ => {}
    }
  }
  Ok(())
}

/// Convert a `Path` key to a UTF-8 S3 string in the canonical
/// forward-slash-separated form required by S3.
///
/// On Linux/macOS, `Path` already uses `/` as the separator, so the
/// to_str path is exact. On Windows, `Path::strip_prefix` produces
/// `\\`-separated paths; we walk `Path::components()` and join with
/// `/` so a Windows-host caller of [`crate::sync_to_s3`] uploads
/// keys in the same shape the open-side reader expects. Each
/// component is validated against the same rules as
/// [`validate_relative_segment`] (no `..`, no platform prefix, no
/// root component, non-empty after trim).
fn path_to_relative_string(key: &Path) -> Result<String> {
  let mut parts: Vec<&str> = Vec::new();
  for component in key.components() {
    match component {
      Component::Normal(name) => {
        let s = name
          .to_str()
          .ok_or_else(|| anyhow!("S3 key has a non-UTF-8 component: {key:?}"))?;
        if s.trim().is_empty() {
          // Catches both empty strings and whitespace-only segments
          // — the latter is what `Path::new("   ").components()`
          // produces on Linux and is never a valid S3 key shape.
          bail!("S3 key: empty / whitespace-only component: {key:?}");
        }
        if s.contains('/') {
          // A literal `/` inside a single OS component would mean
          // the OS layer let an embedded separator through; reject
          // rather than emit a key that wouldn't round-trip.
          bail!("S3 key: component {s:?} contains an embedded `/`: {key:?}");
        }
        if s.contains('\\') {
          // A `\` inside a *single* component can only mean we're
          // on a non-Windows platform AND the user passed a literal
          // backslash in the key (Windows would have split on it).
          // S3 keys are forward-slash-separated; reject so we don't
          // upload a key that other clients can't parse.
          bail!("S3 key: component {s:?} contains an embedded `\\`: {key:?}");
        }
        parts.push(s);
      }
      Component::ParentDir => bail!("S3 key: contains `..`: {key:?}"),
      Component::Prefix(_) => {
        bail!("S3 key: contains a platform prefix (e.g. drive letter): {key:?}")
      }
      Component::RootDir => bail!("S3 key: contains a root component: {key:?}"),
      Component::CurDir => {
        // `Path::components()` collapses interior `.` away, but a
        // standalone `.` (e.g. `Path::new(".")`) shows up here. An
        // S3 key can never be `.`; rejecting matches the behavior
        // of `validate_relative_segment` for a `.`-only string.
        bail!("S3 key: contains a `.` component: {key:?}");
      }
    }
  }
  if parts.is_empty() {
    bail!("S3 key: empty");
  }
  Ok(parts.join("/"))
}

/// Lightweight S3 bucket name validation. Doesn't replicate every
/// AWS rule (the SDK will reject malformed names at request time
/// anyway); just catches obviously wrong inputs early.
fn validate_bucket_name(bucket: &str) -> Result<()> {
  if bucket.is_empty() {
    bail!("S3Config::bucket: empty");
  }
  if bucket.len() > 63 {
    bail!("S3Config::bucket: too long (max 63 chars): {bucket:?}");
  }
  Ok(())
}

/// Parse `x-amz-checksum-*` headers from a HEAD response. The header
/// values are **base64-encoded** raw checksum bytes — not hex, not
/// integers. We validate the byte length matches the expected size
/// for each algorithm before constructing a [`ProviderChecksum`].
fn parse_head_checksum(
  resp: &aws_sdk_s3::operation::head_object::HeadObjectOutput,
) -> Option<ProviderChecksum> {
  if let Some(b64) = resp.checksum_crc32() {
    if let Some(c) = decode_b64_crc32(b64) {
      return Some(ProviderChecksum::Crc32(c));
    }
  }
  if let Some(b64) = resp.checksum_crc32_c() {
    if let Some(c) = decode_b64_crc32(b64) {
      return Some(ProviderChecksum::Crc32C(c));
    }
  }
  if let Some(b64) = resp.checksum_sha1() {
    if let Some(c) = decode_b64_fixed::<20>(b64) {
      return Some(ProviderChecksum::Sha1(c));
    }
  }
  if let Some(b64) = resp.checksum_sha256() {
    if let Some(c) = decode_b64_fixed::<32>(b64) {
      return Some(ProviderChecksum::Sha256(c));
    }
  }
  None
}

fn parse_response_checksum(
  resp: &aws_sdk_s3::operation::put_object::PutObjectOutput,
) -> Option<ProviderChecksum> {
  if let Some(b64) = resp.checksum_crc32() {
    if let Some(c) = decode_b64_crc32(b64) {
      return Some(ProviderChecksum::Crc32(c));
    }
  }
  if let Some(b64) = resp.checksum_crc32_c() {
    if let Some(c) = decode_b64_crc32(b64) {
      return Some(ProviderChecksum::Crc32C(c));
    }
  }
  if let Some(b64) = resp.checksum_sha1() {
    if let Some(c) = decode_b64_fixed::<20>(b64) {
      return Some(ProviderChecksum::Sha1(c));
    }
  }
  if let Some(b64) = resp.checksum_sha256() {
    if let Some(c) = decode_b64_fixed::<32>(b64) {
      return Some(ProviderChecksum::Sha256(c));
    }
  }
  None
}

fn decode_b64_crc32(b64: &str) -> Option<u32> {
  let bytes = base64::engine::general_purpose::STANDARD.decode(b64).ok()?;
  if bytes.len() != 4 {
    return None;
  }
  Some(u32::from_be_bytes(bytes.try_into().ok()?))
}

fn decode_b64_fixed<const N: usize>(b64: &str) -> Option<[u8; N]> {
  let bytes = base64::engine::general_purpose::STANDARD.decode(b64).ok()?;
  bytes.try_into().ok()
}

/// Apply credentials sourced from the `aws_config` default chain.
async fn apply_env_credentials(
  mut builder: SdkConfigBuilder,
  region: &Region,
) -> Result<SdkConfigBuilder> {
  let provided = aws_config::defaults(aws_sdk_s3::config::BehaviorVersion::latest())
    .region(region.clone())
    .load()
    .await;
  builder = builder.credentials_provider(
    provided
      .credentials_provider()
      .ok_or_else(|| anyhow!("aws_config returned no credentials provider for the env chain"))?,
  );
  Ok(builder)
}

/// Multipart upload writer with proper part-sizing discipline. See
/// [`stream_writer::S3StreamWriter`].
mod stream_writer {
  use super::{anyhow, ByteStream, Bytes, Context, Result};
  use anyhow::bail;
  use async_trait::async_trait;
  use aws_sdk_s3::types::CompletedMultipartUpload;
  use aws_sdk_s3::types::CompletedPart;
  use searchlite_core::storage::blob::{ObjectStat, ObjectWriter};
  use std::sync::Arc;

  /// S3 multipart minimum part size (5 MiB). The final part may be
  /// smaller. Smaller-than-minimum non-final parts are rejected by S3
  /// at `CompleteMultipartUpload` time.
  const MIN_PART_SIZE: usize = 5 * 1024 * 1024;

  /// Streaming `BlobStore::ObjectWriter` over S3 multipart uploads.
  ///
  /// Respects multipart sizing constraints. Buffers writes until the
  /// buffer reaches
  /// [`MIN_PART_SIZE`] and uploads at that boundary. The final flush
  /// happens on `complete()`. For empty or sub-min-part-size streams
  /// we fall back to single `PutObject` (no multipart upload at all),
  /// because completing a zero-part multipart upload is invalid in S3.
  ///
  /// `Drop` is best-effort only: if the writer is dropped without
  /// an explicit `abort()` or `complete()`, we cannot reliably await
  /// the abort RPC from a synchronous Drop. We log a warning and
  /// leave the multipart upload to be cleaned up by S3's lifecycle
  /// policy or an explicit ListMultipartUploads sweep.
  pub(super) struct S3StreamWriter {
    client: Arc<aws_sdk_s3::Client>,
    bucket: String,
    key: String,
    state: WriterState,
    /// Total bytes written across the whole stream, used to populate
    /// `ObjectStat.len` on `complete()`.
    total_bytes: u64,
  }

  enum WriterState {
    /// No upload yet; bytes are buffered. `complete()` from this
    /// state issues a single `PutObject` (no multipart).
    Buffered { buf: Vec<u8> },
    /// Multipart upload created. `parts` holds completed part
    /// metadata for the final `CompleteMultipartUpload`. `pending`
    /// holds bytes awaiting part-size alignment.
    Multipart {
      upload_id: String,
      parts: Vec<CompletedPart>,
      pending: Vec<u8>,
      next_part_number: i32,
    },
    /// Terminal state after `complete()` or `abort()`. Further
    /// writes error.
    Finished,
  }

  impl S3StreamWriter {
    pub(super) fn new(client: Arc<aws_sdk_s3::Client>, bucket: String, key: String) -> Self {
      Self {
        client,
        bucket,
        key,
        state: WriterState::Buffered {
          buf: Vec::with_capacity(MIN_PART_SIZE),
        },
        total_bytes: 0,
      }
    }
  }

  #[async_trait]
  impl ObjectWriter for S3StreamWriter {
    async fn write(&mut self, chunk: Bytes) -> Result<()> {
      self.total_bytes = self
        .total_bytes
        .checked_add(chunk.len() as u64)
        .ok_or_else(|| anyhow!("S3StreamWriter: total_bytes overflow"))?;
      // Stage 10b v2 [P1] (Codex review): the previous shape
      // appended `chunk` to the buffered state, then on the
      // threshold-crossing iteration `continue`-d into the multipart
      // branch which **re-appended** the same chunk to `pending`.
      // The 5 MiB threshold-crossing call uploaded duplicated bytes
      // while `total_bytes` reported only the original length.
      //
      // The new shape promotes the state ONCE per write call, then
      // flushes full parts from the (already-populated) pending —
      // never re-appending the chunk that triggered the promotion.
      match &mut self.state {
        WriterState::Finished => {
          bail!("S3StreamWriter: write after complete/abort")
        }
        WriterState::Buffered { buf } => {
          buf.extend_from_slice(&chunk);
          if buf.len() >= MIN_PART_SIZE {
            let init = self
              .client
              .create_multipart_upload()
              .bucket(&self.bucket)
              .key(&self.key)
              .send()
              .await
              .with_context(|| format!("create_multipart_upload {}", self.key))?;
            let upload_id = init
              .upload_id()
              .ok_or_else(|| anyhow!("create_multipart_upload returned no upload_id"))?
              .to_string();
            let pending = std::mem::take(buf);
            self.state = WriterState::Multipart {
              upload_id,
              parts: Vec::new(),
              pending,
              next_part_number: 1,
            };
            // Flush whole parts from the freshly-promoted pending.
            // Do NOT re-append `chunk` — it's already in `pending`.
            self.flush_full_parts().await?;
          }
          Ok(())
        }
        WriterState::Multipart { pending, .. } => {
          pending.extend_from_slice(&chunk);
          // Flush whole MIN_PART_SIZE chunks now. The final
          // (possibly short) part is flushed in `complete`.
          self.flush_full_parts().await?;
          Ok(())
        }
      }
    }

    async fn complete(mut self: Box<Self>) -> Result<ObjectStat> {
      // Take ownership of the state so we can move the bytes out
      // without leaving a half-built upload behind on Drop.
      let state = std::mem::replace(&mut self.state, WriterState::Finished);
      match state {
        WriterState::Finished => bail!("S3StreamWriter: already finished"),
        WriterState::Buffered { buf } => {
          // No multipart upload was started; small/empty stream
          // path. Use a single `PutObject`.
          let body = Bytes::from(buf);
          let len = body.len() as u64;
          let resp = self
            .client
            .put_object()
            .bucket(&self.bucket)
            .key(&self.key)
            .body(ByteStream::from(body))
            .send()
            .await
            .with_context(|| format!("put_object (single-shot) {}", self.key))?;
          Ok(ObjectStat {
            len,
            provider_version: resp.e_tag().map(|s| s.to_string()),
            provider_checksum: super::parse_response_checksum(&resp),
          })
        }
        WriterState::Multipart {
          upload_id,
          mut parts,
          pending,
          mut next_part_number,
        } => {
          // Upload the final (possibly short) part. S3 allows the
          // final part to be < MIN_PART_SIZE.
          if !pending.is_empty() {
            let part = upload_part(
              self.client.as_ref(),
              &self.bucket,
              &self.key,
              &upload_id,
              next_part_number,
              pending,
            )
            .await?;
            parts.push(part);
            next_part_number += 1;
          }
          let _ = next_part_number;
          if parts.is_empty() {
            // Defensive: shouldn't happen since the Multipart state
            // is only entered after MIN_PART_SIZE bytes were
            // buffered. But if it does, abort and surface a clear
            // error rather than completing a zero-part upload.
            let _ = self
              .client
              .abort_multipart_upload()
              .bucket(&self.bucket)
              .key(&self.key)
              .upload_id(&upload_id)
              .send()
              .await;
            bail!(
              "S3StreamWriter: no parts uploaded for multipart {} (this is a bug)",
              self.key
            );
          }
          let resp = self
            .client
            .complete_multipart_upload()
            .bucket(&self.bucket)
            .key(&self.key)
            .upload_id(&upload_id)
            .multipart_upload(
              CompletedMultipartUpload::builder()
                .set_parts(Some(parts))
                .build(),
            )
            .send()
            .await
            .with_context(|| format!("complete_multipart_upload {}", self.key))?;
          Ok(ObjectStat {
            len: self.total_bytes,
            provider_version: resp.e_tag().map(|s| s.to_string()),
            provider_checksum: None,
          })
        }
      }
    }

    async fn abort(mut self: Box<Self>) -> Result<()> {
      let state = std::mem::replace(&mut self.state, WriterState::Finished);
      match state {
        WriterState::Buffered { .. } | WriterState::Finished => Ok(()),
        WriterState::Multipart { upload_id, .. } => {
          self
            .client
            .abort_multipart_upload()
            .bucket(&self.bucket)
            .key(&self.key)
            .upload_id(upload_id)
            .send()
            .await
            .with_context(|| format!("abort_multipart_upload {}", self.key))?;
          Ok(())
        }
      }
    }
  }

  impl Drop for S3StreamWriter {
    fn drop(&mut self) {
      if let WriterState::Multipart { upload_id, .. } = &self.state {
        log::warn!(
          "S3StreamWriter dropped without complete/abort; multipart upload {upload_id} \
           on {} will be left dangling. Rely on bucket lifecycle policy or \
           ListMultipartUploads to clean up.",
          self.key
        );
      }
    }
  }

  impl S3StreamWriter {
    async fn flush_full_parts(&mut self) -> Result<()> {
      if let WriterState::Multipart {
        upload_id,
        parts,
        pending,
        next_part_number,
      } = &mut self.state
      {
        while pending.len() >= MIN_PART_SIZE {
          let chunk: Vec<u8> = pending.drain(..MIN_PART_SIZE).collect();
          let part = upload_part(
            self.client.as_ref(),
            &self.bucket,
            &self.key,
            upload_id,
            *next_part_number,
            chunk,
          )
          .await?;
          parts.push(part);
          *next_part_number += 1;
        }
      }
      Ok(())
    }
  }

  async fn upload_part(
    client: &aws_sdk_s3::Client,
    bucket: &str,
    key: &str,
    upload_id: &str,
    part_number: i32,
    body: Vec<u8>,
  ) -> Result<CompletedPart> {
    let resp = client
      .upload_part()
      .bucket(bucket)
      .key(key)
      .upload_id(upload_id)
      .part_number(part_number)
      .body(ByteStream::from(Bytes::from(body)))
      .send()
      .await
      .with_context(|| format!("upload_part {key} #{part_number}"))?;
    Ok(
      CompletedPart::builder()
        .set_part_number(Some(part_number))
        .set_e_tag(resp.e_tag().map(|s| s.to_string()))
        .build(),
    )
  }
}

#[allow(dead_code)]
fn _ensure_pub_types_used() {
  let _ = ArtifactIdentity {
    key: std::path::PathBuf::new(),
    len: 0,
    content_hash: ContentHash::new([0; 32]),
  };
}
