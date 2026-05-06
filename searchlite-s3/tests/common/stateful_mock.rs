//! Stage 10c: stateful S3 mock for end-to-end tests.
//!
//! Implements the subset of the S3 protocol the bake-and-serve
//! workflow uses, backed by an `Arc<Mutex<HashMap<String, Bytes>>>`:
//!
//! * `HEAD bucket/key` → 200 + `Content-Length` + `ETag` (CRC32 of
//!   the stored bytes, hex-formatted with quotes), or 404 if absent.
//! * `GET bucket/key` (no Range) → 200 + full body + ETag.
//! * `GET bucket/key` with `Range: bytes=N-M` → 206 + exact slice +
//!   `Content-Range` + ETag (same value as the HEAD).
//! * `PUT bucket/key` → store bytes, return new ETag.
//! * `DELETE bucket/key` → remove + 204 (idempotent on missing).
//!
//! The ETag is deterministic over the bytes (CRC32 hex), so a
//! PUT-then-HEAD-then-conditional-GET round-trip uses consistent
//! ETags and exercises Stage 10b's `If-Match` path. Real S3 ETags
//! aren't pure content hashes (multipart ETags include the part
//! count) but the test harness only needs **stable** ETags to
//! exercise the conditional protocol — content-derived hex is fine.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use wiremock::http::HeaderName;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

/// In-memory key-value store keyed by the S3 path-style "object
/// key" (e.g. `"idx-baked/MANIFEST.json"`). Shared across the
/// wiremock responder and the test that mounted it, so tests can
/// introspect:
///
/// * What was stored (`snapshot`).
/// * The order in which PUTs landed (`put_order`) — used by Stage
///   10c v2 [P1] to assert MANIFEST.json is the final PUT.
/// * Whether to inject a 5xx on a specific PUT key
///   (`fail_put_for`) — used by the fail-mid-sync regression.
pub struct StatefulS3Bucket {
  pub bucket: String,
  state: Arc<Mutex<HashMap<String, Bytes>>>,
  put_order: Arc<Mutex<Vec<String>>>,
  fail_put_for: Arc<Mutex<Option<String>>>,
}

impl StatefulS3Bucket {
  pub fn new(bucket: &str) -> Self {
    Self {
      bucket: bucket.to_string(),
      state: Arc::new(Mutex::new(HashMap::new())),
      put_order: Arc::new(Mutex::new(Vec::new())),
      fail_put_for: Arc::new(Mutex::new(None)),
    }
  }

  /// Mount the responder on a fresh `MockServer` and return its URI
  /// alongside the [`MockServer`] handle. Tests must hold the handle
  /// for as long as they want the mock alive — its `Drop` releases
  /// the listening socket so parallel tests don't leak ports under
  /// long CI runs.
  pub async fn spawn_server(self: &Arc<Self>) -> (String, MockServer) {
    let server = MockServer::start().await;
    Mock::given(wiremock::matchers::any())
      .respond_with(StatefulResponder {
        bucket: self.bucket.clone(),
        state: self.state.clone(),
        put_order: self.put_order.clone(),
        fail_put_for: self.fail_put_for.clone(),
      })
      .mount(&server)
      .await;
    let uri = server.uri();
    (uri, server)
  }

  /// Snapshot the stored keys + bytes (useful for path-shape
  /// assertions).
  pub fn snapshot(&self) -> HashMap<String, Bytes> {
    self.state.lock().unwrap().clone()
  }

  /// Order in which PUTs were received (deduped — only the FIRST
  /// PUT of a given key is recorded, because retries shouldn't
  /// shift the publish-order assertion).
  pub fn put_order(&self) -> Vec<String> {
    self.put_order.lock().unwrap().clone()
  }

  /// Configure the responder to return 500 for any PUT whose key
  /// matches `key`. Used to drive the fail-mid-sync regression.
  pub fn inject_put_failure(&self, key: &str) {
    *self.fail_put_for.lock().unwrap() = Some(key.to_string());
  }
}

/// Convenience helper — equivalent to constructing a
/// `StatefulS3Bucket` and calling `spawn_server`. Returns
/// `(server_uri, bucket_name, server_handle)`. Bind the handle to a
/// `_`-prefixed local in the test so the [`MockServer`] is dropped
/// only at scope end:
///
/// ```ignore
/// let (uri, bucket, _server) = spawn_stateful_s3_mock("test").await;
/// ```
///
/// If you need to introspect stored keys, use `StatefulS3Bucket` +
/// `snapshot` directly instead.
pub async fn spawn_stateful_s3_mock(bucket: &str) -> (String, String, MockServer) {
  let bucket_state = Arc::new(StatefulS3Bucket::new(bucket));
  let (uri, server) = bucket_state.spawn_server().await;
  (uri, bucket.to_string(), server)
}

struct StatefulResponder {
  bucket: String,
  state: Arc<Mutex<HashMap<String, Bytes>>>,
  put_order: Arc<Mutex<Vec<String>>>,
  fail_put_for: Arc<Mutex<Option<String>>>,
}

impl Respond for StatefulResponder {
  fn respond(&self, req: &Request) -> ResponseTemplate {
    // Path-style endpoint: /<bucket>/<key>...
    let path = req.url.path();
    let stripped = path.strip_prefix('/').unwrap_or(path);
    let bucket_prefix = format!("{}/", self.bucket);
    let key = match stripped.strip_prefix(&bucket_prefix) {
      Some(k) => k.to_string(),
      None => {
        // Misrouted — return 404 so the test sees a clear failure.
        return ResponseTemplate::new(404)
          .set_body_string(format!("missing bucket prefix in path: {path}"));
      }
    };
    let method = req.method.as_str();
    match method {
      "HEAD" => {
        let store = self.state.lock().unwrap();
        match store.get(&key) {
          Some(bytes) => head_response(bytes.len(), &etag_for(bytes)),
          None => ResponseTemplate::new(404),
        }
      }
      "GET" => {
        let store = self.state.lock().unwrap();
        let bytes = match store.get(&key) {
          Some(b) => b.clone(),
          None => return ResponseTemplate::new(404),
        };
        let etag = etag_for(&bytes);
        // Range header check.
        if let Some(range) = req.headers.get(HeaderName::from_static("range")) {
          let raw = range.to_str().unwrap_or("");
          if let Some((start, end_inclusive)) = parse_byte_range(raw) {
            // Validate every shape the slice below would otherwise
            // panic on: inverted ranges (`bytes=10-1`), out-of-bounds
            // start (`bytes=999-50` against a 100-byte body), and
            // out-of-bounds end. Real S3 responds 416 for all three;
            // the mock should match.
            if start > end_inclusive || start >= bytes.len() || end_inclusive >= bytes.len() {
              return ResponseTemplate::new(416)
                .set_body_string(format!("range OOB: {raw} for len {}", bytes.len()));
            }
            let slice: Vec<u8> = bytes[start..=end_inclusive].to_vec();
            return ResponseTemplate::new(206)
              .insert_header("ETag", etag)
              .insert_header(
                "Content-Range",
                format!("bytes {start}-{end_inclusive}/{}", bytes.len()),
              )
              .insert_header("Content-Length", slice.len().to_string())
              .set_body_bytes(slice);
          }
          return ResponseTemplate::new(416)
            .set_body_string(format!("malformed range header: {raw}"));
        }
        ResponseTemplate::new(200)
          .insert_header("ETag", etag)
          .insert_header("Content-Length", bytes.len().to_string())
          .set_body_bytes(bytes.to_vec())
      }
      "PUT" => {
        // Optional injected failure — used by the fail-mid-sync
        // regression to trigger a sync abort partway through.
        if let Some(target) = self.fail_put_for.lock().unwrap().as_deref() {
          if target == key {
            return ResponseTemplate::new(500)
              .set_body_string(format!("injected failure for PUT {key}"));
          }
        }
        let body = Bytes::copy_from_slice(&req.body);
        let etag = etag_for(&body);
        // Record the order of FIRST PUTs so callers can assert
        // publish ordering (e.g. manifest-last). Subsequent PUTs to
        // the same key (retries) don't re-record.
        {
          let mut order = self.put_order.lock().unwrap();
          if !order.iter().any(|k| k == &key) {
            order.push(key.clone());
          }
        }
        self.state.lock().unwrap().insert(key, body);
        ResponseTemplate::new(200).insert_header("ETag", etag)
      }
      "DELETE" => {
        self.state.lock().unwrap().remove(&key);
        ResponseTemplate::new(204)
      }
      other => ResponseTemplate::new(405)
        .set_body_string(format!("method {other} not implemented in stateful mock")),
    }
  }
}

fn head_response(len: usize, etag: &str) -> ResponseTemplate {
  ResponseTemplate::new(200)
    .insert_header("Content-Length", len.to_string())
    .insert_header("ETag", etag.to_string())
}

/// Parse a `bytes=N-M` Range header into `(start, end_inclusive)`.
fn parse_byte_range(raw: &str) -> Option<(usize, usize)> {
  let suffix = raw.strip_prefix("bytes=")?;
  let mut it = suffix.split('-');
  let start: usize = it.next()?.parse().ok()?;
  let end: usize = it.next()?.parse().ok()?;
  Some((start, end))
}

/// Stable, content-derived ETag. Real S3 wraps the ETag in quotes
/// and uses MD5 (or MD5-of-MD5s for multipart); we match the
/// quoting and use a CRC32 hex for simplicity. Tests only need
/// stability across PUT/HEAD/GET cycles, not cryptographic
/// strength.
fn etag_for(bytes: &[u8]) -> String {
  let crc = crc32fast::hash(bytes);
  format!("\"{:08x}\"", crc)
}
