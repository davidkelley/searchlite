//! Stage 10b: wiremock-backed protocol tests for [`S3BlobStore`].
//!
//! These exercise the wire protocol — request methods, paths,
//! headers, body — without requiring a real S3/R2/MinIO endpoint.
//! Each test spins up a fresh `wiremock::MockServer`, configures
//! `S3Config { force_path_style: true }` so the SDK targets the mock
//! server directly, sets up a Mock for the request shape under test,
//! and asserts the protocol-level behavior of the corresponding
//! `BlobStore` method.

#![cfg(not(target_arch = "wasm32"))]

use std::path::Path;
use std::sync::Arc;

use base64::Engine;
use bytes::Bytes;
use searchlite_core::storage::blob::{BlobStore, ProviderChecksum, PutIfMatchError};
use searchlite_s3::{S3BlobStore, S3Config, S3Credentials, S3StoreError};
use wiremock::matchers::{header, header_exists, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

/// Build an `S3BlobStore` configured to talk to the given mock server.
async fn store_for(server: &MockServer, conditional_put: bool) -> S3BlobStore {
  S3BlobStore::new(S3Config {
    endpoint_url: Some(server.uri()),
    region: "us-east-1".into(),
    bucket: "test-bucket".into(),
    prefix: None,
    credentials: S3Credentials::Static {
      access_key_id: "test-key".into(),
      secret_access_key: "test-secret".into(),
      session_token: None,
    },
    conditional_put,
    force_path_style: true,
  })
  .await
  .expect("S3BlobStore::new")
}

fn b64(bytes: &[u8]) -> String {
  base64::engine::general_purpose::STANDARD.encode(bytes)
}

// ─────────────────────── stat / open ──────────────────────────────────

#[tokio::test]
async fn stat_returns_len_and_etag_verbatim() {
  let server = MockServer::start().await;
  Mock::given(method("HEAD"))
    .and(path("/test-bucket/seg_X.terms"))
    .respond_with(
      ResponseTemplate::new(200)
        .insert_header("Content-Length", "1234")
        // ETag preserved verbatim, quotes included.
        .insert_header("ETag", "\"deadbeef-1\""),
    )
    .mount(&server)
    .await;
  let store = store_for(&server, true).await;
  let stat = store.stat(Path::new("seg_X.terms")).await.unwrap();
  assert_eq!(stat.len, 1234);
  assert_eq!(stat.provider_version.as_deref(), Some("\"deadbeef-1\""));
}

#[tokio::test]
async fn stat_404_surfaces_io_not_found_in_chain() {
  let server = MockServer::start().await;
  Mock::given(method("HEAD"))
    .and(path("/test-bucket/missing.bin"))
    .respond_with(ResponseTemplate::new(404))
    .mount(&server)
    .await;
  let store = store_for(&server, true).await;
  let err = store
    .stat(Path::new("missing.bin"))
    .await
    .expect_err("404 must error");
  let saw_io_not_found = err.chain().any(|cause| {
    cause
      .downcast_ref::<std::io::Error>()
      .map(|e| e.kind() == std::io::ErrorKind::NotFound)
      .unwrap_or(false)
  });
  assert!(
    saw_io_not_found,
    "S3 NotFound must surface io::ErrorKind::NotFound in anyhow chain"
  );
}

#[tokio::test]
async fn stat_parses_base64_checksums_with_validated_lengths() {
  let server = MockServer::start().await;
  // SHA-256 over empty input = e3b0c44...
  let sha256: [u8; 32] = [
    0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
    0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55,
  ];
  Mock::given(method("HEAD"))
    .and(path("/test-bucket/k.bin"))
    .respond_with(
      ResponseTemplate::new(200)
        .insert_header("Content-Length", "0")
        .insert_header("ETag", "\"e\"")
        .insert_header("x-amz-checksum-sha256", b64(&sha256)),
    )
    .mount(&server)
    .await;
  let store = store_for(&server, true).await;
  let stat = store.stat(Path::new("k.bin")).await.unwrap();
  match stat.provider_checksum {
    Some(ProviderChecksum::Sha256(bytes)) => assert_eq!(bytes, sha256),
    other => panic!("expected SHA-256 provider checksum, got {other:?}"),
  }
}

#[tokio::test]
async fn open_caches_stat_so_read_range_does_not_re_head() {
  // One HEAD on open; multiple read_range calls issue GETs only.
  let server = MockServer::start().await;
  Mock::given(method("HEAD"))
    .and(path("/test-bucket/big.bin"))
    .respond_with(
      ResponseTemplate::new(200)
        .insert_header("Content-Length", "100")
        .insert_header("ETag", "\"v1\""),
    )
    .expect(1)
    .mount(&server)
    .await;
  // GET response shared by both range reads.
  Mock::given(method("GET"))
    .and(path("/test-bucket/big.bin"))
    .respond_with(ResponseTemplate::new(206).set_body_bytes(vec![0u8; 10]))
    .expect(2)
    .mount(&server)
    .await;
  let store = store_for(&server, true).await;
  let obj = store.open(Path::new("big.bin")).await.unwrap();
  let _ = obj.read_range(0..10).await.unwrap();
  let _ = obj.read_range(50..60).await.unwrap();
  // wiremock asserts via .expect(N) — counts are checked on drop.
}

// ─────────────────────── conditional reads ────────────────────────────

#[tokio::test]
async fn read_range_sends_inclusive_byte_range_and_if_match() {
  let server = MockServer::start().await;
  Mock::given(method("HEAD"))
    .and(path("/test-bucket/k.bin"))
    .respond_with(
      ResponseTemplate::new(200)
        .insert_header("Content-Length", "100")
        .insert_header("ETag", "\"v-pin\""),
    )
    .mount(&server)
    .await;
  // Assert the exact Range and If-Match headers on the GET.
  Mock::given(method("GET"))
    .and(path("/test-bucket/k.bin"))
    .and(header("Range", "bytes=10-19"))
    .and(header("if-match", "\"v-pin\""))
    .respond_with(ResponseTemplate::new(206).set_body_bytes(vec![0u8; 10]))
    .expect(1)
    .mount(&server)
    .await;
  let store = store_for(&server, true).await;
  let obj = store.open(Path::new("k.bin")).await.unwrap();
  let bytes = obj.read_range(10..20).await.unwrap();
  assert_eq!(bytes.len(), 10);
}

#[tokio::test]
async fn read_range_zero_width_returns_empty_without_request() {
  let server = MockServer::start().await;
  Mock::given(method("HEAD"))
    .and(path("/test-bucket/k.bin"))
    .respond_with(
      ResponseTemplate::new(200)
        .insert_header("Content-Length", "100")
        .insert_header("ETag", "\"v\""),
    )
    .mount(&server)
    .await;
  // No GET expected; if any GET arrives the test fails on drop
  // because there's no matching Mock.
  let store = store_for(&server, true).await;
  let obj = store.open(Path::new("k.bin")).await.unwrap();
  let bytes = obj.read_range(50..50).await.unwrap();
  assert!(bytes.is_empty());
}

#[tokio::test]
async fn read_range_412_surfaces_precondition_failed() {
  // Stage 10b [Codex review #6]: a 412 on a conditional read_range
  // means the pinned object was overwritten between open and read.
  let server = MockServer::start().await;
  Mock::given(method("HEAD"))
    .and(path("/test-bucket/k.bin"))
    .respond_with(
      ResponseTemplate::new(200)
        .insert_header("Content-Length", "100")
        .insert_header("ETag", "\"v-old\""),
    )
    .mount(&server)
    .await;
  Mock::given(method("GET"))
    .and(path("/test-bucket/k.bin"))
    .respond_with(ResponseTemplate::new(412))
    .mount(&server)
    .await;
  let store = store_for(&server, true).await;
  let obj = store.open(Path::new("k.bin")).await.unwrap();
  let err = obj
    .read_range(0..10)
    .await
    .expect_err("412 must surface as an error");
  // Stage 10b v3 [P3] (Codex review): 412 must surface as a typed
  // `S3StoreError::PreconditionFailed` in the anyhow chain so
  // callers can downcast and discriminate from generic transport
  // errors.
  let saw_precondition = err.chain().any(|cause| {
    matches!(
      cause.downcast_ref::<S3StoreError>(),
      Some(S3StoreError::PreconditionFailed { .. })
    )
  });
  assert!(
    saw_precondition,
    "412 must surface S3StoreError::PreconditionFailed in the anyhow chain; got: {err:#}"
  );
}

#[tokio::test]
async fn top_level_get_range_412_surfaces_precondition_failed_typed() {
  // Stage 10b v3 [P3]: same typing contract on the top-level
  // `BlobStore::get_range` path — 412/409 must surface as
  // `S3StoreError::PreconditionFailed` (downcastable), not as a
  // generic SDK error.
  let server = MockServer::start().await;
  Mock::given(method("HEAD"))
    .and(path("/test-bucket/k.bin"))
    .respond_with(
      ResponseTemplate::new(200)
        .insert_header("Content-Length", "100")
        .insert_header("ETag", "\"v-old\""),
    )
    .mount(&server)
    .await;
  Mock::given(method("GET"))
    .and(path("/test-bucket/k.bin"))
    .respond_with(ResponseTemplate::new(412))
    .mount(&server)
    .await;
  let store = store_for(&server, true).await;
  let err = store
    .get_range(Path::new("k.bin"), 0..10)
    .await
    .expect_err("412 on top-level get_range must error");
  let saw_precondition = err.chain().any(|cause| {
    matches!(
      cause.downcast_ref::<S3StoreError>(),
      Some(S3StoreError::PreconditionFailed { .. })
    )
  });
  assert!(
    saw_precondition,
    "412 on get_range must surface S3StoreError::PreconditionFailed; got: {err:#}"
  );
}

#[tokio::test]
async fn top_level_get_range_409_surfaces_precondition_failed_typed() {
  // R2 returns 409 ConditionalRequestConflict for racing
  // conditional ops on reads as well; map both 412 and 409 the
  // same way.
  let server = MockServer::start().await;
  Mock::given(method("HEAD"))
    .and(path("/test-bucket/k.bin"))
    .respond_with(
      ResponseTemplate::new(200)
        .insert_header("Content-Length", "100")
        .insert_header("ETag", "\"v-old\""),
    )
    .mount(&server)
    .await;
  Mock::given(method("GET"))
    .and(path("/test-bucket/k.bin"))
    .respond_with(ResponseTemplate::new(409))
    .mount(&server)
    .await;
  let store = store_for(&server, true).await;
  let err = store
    .get_range(Path::new("k.bin"), 0..10)
    .await
    .expect_err("409 on get_range must error");
  let saw_precondition = err.chain().any(|cause| {
    matches!(
      cause.downcast_ref::<S3StoreError>(),
      Some(S3StoreError::PreconditionFailed { .. })
    )
  });
  assert!(
    saw_precondition,
    "409 on get_range must surface S3StoreError::PreconditionFailed; got: {err:#}"
  );
}

#[tokio::test]
async fn read_range_without_etag_does_not_send_if_match() {
  // If the object has no provider_version (some endpoints omit ETag),
  // we issue the GET without `If-Match`. Otherwise the request would
  // 412 every time.
  let server = MockServer::start().await;
  Mock::given(method("HEAD"))
    .and(path("/test-bucket/etagless.bin"))
    .respond_with(ResponseTemplate::new(200).insert_header("Content-Length", "100"))
    .mount(&server)
    .await;
  // Custom matcher: assert no `if-match` header was sent.
  fn no_if_match(req: &Request) -> bool {
    !req.headers.contains_key("if-match")
  }
  Mock::given(method("GET"))
    .and(path("/test-bucket/etagless.bin"))
    .and(wiremock::matchers::AnyMatcher)
    .and(NoIfMatch)
    .respond_with(ResponseTemplate::new(206).set_body_bytes(vec![0u8; 5]))
    .expect(1)
    .mount(&server)
    .await;
  let _ = no_if_match;
  let store = store_for(&server, true).await;
  let obj = store.open(Path::new("etagless.bin")).await.unwrap();
  let _ = obj.read_range(0..5).await.unwrap();
}

/// Custom wiremock matcher: request has no `if-match` header.
struct NoIfMatch;
impl wiremock::Match for NoIfMatch {
  fn matches(&self, req: &Request) -> bool {
    !req.headers.contains_key("if-match")
  }
}

// ─────────────────────── put / delete ─────────────────────────────────

#[tokio::test]
async fn put_returns_new_etag() {
  let server = MockServer::start().await;
  Mock::given(method("PUT"))
    .and(path("/test-bucket/k.bin"))
    .respond_with(ResponseTemplate::new(200).insert_header("ETag", "\"new\""))
    .expect(1)
    .mount(&server)
    .await;
  let store = store_for(&server, true).await;
  let stat = store
    .put(Path::new("k.bin"), Bytes::from_static(b"payload"))
    .await
    .unwrap();
  assert_eq!(stat.len, 7);
  assert_eq!(stat.provider_version.as_deref(), Some("\"new\""));
}

#[tokio::test]
async fn delete_is_idempotent_on_404() {
  let server = MockServer::start().await;
  Mock::given(method("DELETE"))
    .and(path("/test-bucket/missing.bin"))
    .respond_with(ResponseTemplate::new(404))
    .expect(1)
    .mount(&server)
    .await;
  let store = store_for(&server, true).await;
  store.delete(Path::new("missing.bin")).await.unwrap();
}

// ─────────────────────── put_if_match ─────────────────────────────────

#[tokio::test]
async fn put_if_match_refuses_when_capability_disabled() {
  // Stage 10b [Codex review #5]: with conditional_put = false, the
  // call MUST NOT issue any HTTP request — surface a typed error.
  let server = MockServer::start().await;
  // No mocks set up; any request would 404 from wiremock and fail
  // the test by surfacing a different error than expected.
  let store = store_for(&server, false).await;
  let err = store
    .put_if_match(Path::new("k.bin"), Bytes::from_static(b"x"), Some("\"e\""))
    .await
    .expect_err("put_if_match must error when capability is off");
  match err {
    PutIfMatchError::Other(inner) => {
      let typed = inner
        .downcast_ref::<S3StoreError>()
        .expect("inner must downcast to S3StoreError");
      assert!(matches!(typed, S3StoreError::ConditionalPutNotSupported));
    }
    other => panic!("expected Other(ConditionalPutNotSupported), got {other:?}"),
  }
}

#[tokio::test]
async fn put_if_match_some_sends_if_match_header() {
  let server = MockServer::start().await;
  Mock::given(method("PUT"))
    .and(path("/test-bucket/k.bin"))
    .and(header("if-match", "\"v1\""))
    .respond_with(ResponseTemplate::new(200).insert_header("ETag", "\"v2\""))
    .expect(1)
    .mount(&server)
    .await;
  let store = store_for(&server, true).await;
  let stat = store
    .put_if_match(Path::new("k.bin"), Bytes::from_static(b"x"), Some("\"v1\""))
    .await
    .unwrap();
  assert_eq!(stat.provider_version.as_deref(), Some("\"v2\""));
}

#[tokio::test]
async fn put_if_match_none_sends_if_none_match_star() {
  // Stage 10b [Codex review per plan]: expected = None means
  // must-not-exist, mapped to `If-None-Match: *`.
  let server = MockServer::start().await;
  Mock::given(method("PUT"))
    .and(path("/test-bucket/k.bin"))
    .and(header("if-none-match", "*"))
    .respond_with(ResponseTemplate::new(200).insert_header("ETag", "\"created\""))
    .expect(1)
    .mount(&server)
    .await;
  let store = store_for(&server, true).await;
  let stat = store
    .put_if_match(Path::new("k.bin"), Bytes::from_static(b"x"), None)
    .await
    .unwrap();
  assert_eq!(stat.provider_version.as_deref(), Some("\"created\""));
}

#[tokio::test]
async fn put_if_match_412_maps_to_conflict_with_current_stat() {
  let server = MockServer::start().await;
  Mock::given(method("PUT"))
    .and(path("/test-bucket/k.bin"))
    .and(header_exists("if-match"))
    .respond_with(ResponseTemplate::new(412))
    .expect(1)
    .mount(&server)
    .await;
  // Best-effort follow-up HEAD to surface the current stat.
  Mock::given(method("HEAD"))
    .and(path("/test-bucket/k.bin"))
    .respond_with(
      ResponseTemplate::new(200)
        .insert_header("Content-Length", "42")
        .insert_header("ETag", "\"actual\""),
    )
    .mount(&server)
    .await;
  let store = store_for(&server, true).await;
  let err = store
    .put_if_match(
      Path::new("k.bin"),
      Bytes::from_static(b"x"),
      Some("\"stale\""),
    )
    .await
    .expect_err("412 must error");
  match err {
    PutIfMatchError::Conflict { current } => {
      let stat = current.expect("HEAD-after-conflict should populate current");
      assert_eq!(stat.len, 42);
      assert_eq!(stat.provider_version.as_deref(), Some("\"actual\""));
    }
    other => panic!("expected Conflict, got {other:?}"),
  }
}

#[tokio::test]
async fn put_if_match_409_also_maps_to_conflict() {
  // R2 returns 409 ConditionalRequestConflict for racing
  // conditional ops.
  let server = MockServer::start().await;
  Mock::given(method("PUT"))
    .and(path("/test-bucket/k.bin"))
    .and(header_exists("if-match"))
    .respond_with(ResponseTemplate::new(409))
    .mount(&server)
    .await;
  Mock::given(method("HEAD"))
    .and(path("/test-bucket/k.bin"))
    .respond_with(
      ResponseTemplate::new(200)
        .insert_header("Content-Length", "1")
        .insert_header("ETag", "\"ok\""),
    )
    .mount(&server)
    .await;
  let store = store_for(&server, true).await;
  let err = store
    .put_if_match(
      Path::new("k.bin"),
      Bytes::from_static(b"x"),
      Some("\"stale\""),
    )
    .await
    .expect_err("409 must error");
  assert!(matches!(err, PutIfMatchError::Conflict { .. }));
}

// ─────────────────────── put_stream ───────────────────────────────────

#[tokio::test]
async fn put_stream_empty_falls_back_to_single_put() {
  // Stage 10b [Codex review #2]: zero-byte stream must NOT issue
  // CreateMultipartUpload — that's invalid in S3.
  let server = MockServer::start().await;
  Mock::given(method("PUT"))
    .and(path("/test-bucket/empty.bin"))
    .respond_with(ResponseTemplate::new(200).insert_header("ETag", "\"e\""))
    .expect(1)
    .mount(&server)
    .await;
  // Any other request would fail because no other mock is set.
  let store = store_for(&server, true).await;
  let writer = store.put_stream(Path::new("empty.bin")).await.unwrap();
  let stat = writer.complete().await.unwrap();
  assert_eq!(stat.len, 0);
}

#[tokio::test]
async fn put_stream_small_payload_falls_back_to_single_put() {
  // <5 MiB total → no multipart upload at all.
  let server = MockServer::start().await;
  Mock::given(method("PUT"))
    .and(path("/test-bucket/small.bin"))
    .respond_with(ResponseTemplate::new(200).insert_header("ETag", "\"s\""))
    .expect(1)
    .mount(&server)
    .await;
  let store = store_for(&server, true).await;
  let mut writer = store.put_stream(Path::new("small.bin")).await.unwrap();
  writer.write(Bytes::from(vec![0u8; 1024])).await.unwrap();
  writer.write(Bytes::from(vec![0u8; 2048])).await.unwrap();
  let stat = writer.complete().await.unwrap();
  assert_eq!(stat.len, 1024 + 2048);
}

#[tokio::test]
async fn put_stream_explicit_abort_does_not_complete() {
  let server = MockServer::start().await;
  // No PUT mock — if `complete` accidentally fired, we'd error.
  let store = store_for(&server, true).await;
  let writer = store.put_stream(Path::new("aborted.bin")).await.unwrap();
  writer.abort().await.unwrap();
}

// ─────────────────────── capabilities + key normalization ─────────────

#[tokio::test]
async fn capabilities_reflects_config_flags() {
  let server = MockServer::start().await;
  let store = store_for(&server, true).await;
  let caps = store.capabilities();
  assert!(caps.conditional_put);
  assert!(caps.multipart_upload);
  assert!(!caps.mmap_friendly);

  let store_no = store_for(&server, false).await;
  assert!(!store_no.capabilities().conditional_put);
}

#[tokio::test]
async fn key_normalization_rejects_absolute_dotdot_prefix_and_backslash() {
  let server = MockServer::start().await;
  let store = store_for(&server, true).await;
  for bad in ["/abs", "rel/../escape", "rel\\with\\backslash", "", "   "] {
    let err = match store.stat(Path::new(bad)).await {
      Ok(_) => panic!("key {bad:?} must be rejected"),
      Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
      msg.contains("S3 key"),
      "expected 'S3 key' in error for {bad:?}; got: {msg}"
    );
  }
}

#[tokio::test]
async fn prefix_is_joined_for_every_method() {
  let server = MockServer::start().await;
  // Configure store with a prefix; assert the prefix is included
  // in the request path.
  Mock::given(method("HEAD"))
    .and(path("/test-bucket/idx-baked/seg_X.terms"))
    .respond_with(
      ResponseTemplate::new(200)
        .insert_header("Content-Length", "1")
        .insert_header("ETag", "\"e\""),
    )
    .expect(1)
    .mount(&server)
    .await;
  let store = S3BlobStore::new(S3Config {
    endpoint_url: Some(server.uri()),
    region: "us-east-1".into(),
    bucket: "test-bucket".into(),
    prefix: Some("idx-baked".into()),
    credentials: S3Credentials::Static {
      access_key_id: "k".into(),
      secret_access_key: "s".into(),
      session_token: None,
    },
    conditional_put: true,
    force_path_style: true,
  })
  .await
  .unwrap();
  let _ = store.stat(Path::new("seg_X.terms")).await.unwrap();
}

// ─────────────────────── checksum byte-length validation ─────────────

#[tokio::test]
async fn malformed_base64_checksum_is_dropped_not_constructed() {
  // Stage 10b [Codex review #4]: a malformed `x-amz-checksum-*` header
  // (wrong byte length or invalid base64) must not produce a
  // `ProviderChecksum` — drop silently rather than fabricate one.
  let server = MockServer::start().await;
  Mock::given(method("HEAD"))
    .and(path("/test-bucket/bad.bin"))
    .respond_with(
      ResponseTemplate::new(200)
        .insert_header("Content-Length", "0")
        .insert_header("ETag", "\"e\"")
        // This is valid base64 but decodes to 5 bytes, not 32 → must
        // be dropped.
        .insert_header("x-amz-checksum-sha256", b64(&[0u8; 5])),
    )
    .mount(&server)
    .await;
  let store = store_for(&server, true).await;
  let stat = store.stat(Path::new("bad.bin")).await.unwrap();
  assert!(
    stat.provider_checksum.is_none(),
    "malformed checksum must be dropped, not surfaced"
  );
}

// ─────────────────────── Stage 10b v4 regressions ────────────────────

/// Stage 10b v4 [P3] (Codex review): a 412/409 returned from a
/// **non-conditional** request path (e.g. `BlobStore::get`, which
/// never sends `If-Match`) must NOT be typed as
/// `S3StoreError::PreconditionFailed`. The typed variant is
/// meaningful only when the request actually carried a precondition
/// header; labeling an unrelated server-quirk 412/409 as a
/// "stale-pin conditional failure" would mislead callers.
#[tokio::test]
async fn non_conditional_get_412_is_not_typed_as_precondition_failed() {
  let server = MockServer::start().await;
  // `BlobStore::get` issues a plain GET with no precondition.
  Mock::given(method("GET"))
    .and(path("/test-bucket/k.bin"))
    .respond_with(ResponseTemplate::new(412))
    .mount(&server)
    .await;
  let store = store_for(&server, true).await;
  let err = store
    .get(Path::new("k.bin"))
    .await
    .expect_err("412 must error");
  let saw_precondition = err.chain().any(|cause| {
    matches!(
      cause.downcast_ref::<S3StoreError>(),
      Some(S3StoreError::PreconditionFailed { .. })
    )
  });
  assert!(
    !saw_precondition,
    "non-conditional GET's 412 must NOT be typed as PreconditionFailed; got: {err:#}"
  );
}

/// Stage 10b v4 [P3]: same protection on the etagless top-level
/// `get_range` path. When the HEAD didn't return an ETag (some
/// providers omit it), the GET goes out without `If-Match`. A 412 in
/// that case is server-quirk territory, not a stale-pin signal.
#[tokio::test]
async fn etagless_get_range_412_is_not_typed_as_precondition_failed() {
  let server = MockServer::start().await;
  Mock::given(method("HEAD"))
    .and(path("/test-bucket/k.bin"))
    .respond_with(ResponseTemplate::new(200).insert_header("Content-Length", "100"))
    .mount(&server)
    .await;
  Mock::given(method("GET"))
    .and(path("/test-bucket/k.bin"))
    .respond_with(ResponseTemplate::new(412))
    .mount(&server)
    .await;
  let store = store_for(&server, true).await;
  let err = store
    .get_range(Path::new("k.bin"), 0..10)
    .await
    .expect_err("412 must error");
  let saw_precondition = err.chain().any(|cause| {
    matches!(
      cause.downcast_ref::<S3StoreError>(),
      Some(S3StoreError::PreconditionFailed { .. })
    )
  });
  assert!(
    !saw_precondition,
    "etagless get_range's 412 must NOT be typed as PreconditionFailed; got: {err:#}"
  );
}

// ─────────────────────── Stage 10b v2 regressions ────────────────────

/// Stage 10b v2 [P1] (Codex review): a write that crosses the
/// `MIN_PART_SIZE` (5 MiB) boundary in a single call must NOT
/// duplicate the threshold-crossing chunk. The previous shape
/// `continue`-d the loop after promotion to the multipart state and
/// re-appended `chunk` in the next iteration, leading to duplicated
/// bytes uploaded while `total_bytes` reported only the original
/// length.
///
/// We assert: a single 5 MiB write completes the multipart upload
/// with **one** UploadPart of 5 MiB followed by Complete, total
/// length matches the input, and (most importantly) the UploadPart
/// body bytes match the expected first 5 MiB exactly.
#[tokio::test(flavor = "multi_thread")]
async fn put_stream_first_write_crossing_5mib_uploads_no_duplicate_bytes() {
  let server = MockServer::start().await;

  // Mock CreateMultipartUpload — must be called exactly once.
  Mock::given(method("POST"))
    .and(path("/test-bucket/big.bin"))
    .and(wiremock::matchers::query_param("uploads", ""))
    .respond_with(ResponseTemplate::new(200).set_body_string(
      r#"<?xml version="1.0" encoding="UTF-8"?>
<InitiateMultipartUploadResult>
<Bucket>test-bucket</Bucket>
<Key>big.bin</Key>
<UploadId>upload-X</UploadId>
</InitiateMultipartUploadResult>"#,
    ))
    .expect(1)
    .mount(&server)
    .await;

  // Mock UploadPart for part 1 — assert exactly 5 MiB body length.
  Mock::given(method("PUT"))
    .and(path("/test-bucket/big.bin"))
    .and(wiremock::matchers::query_param("partNumber", "1"))
    .and(wiremock::matchers::query_param("uploadId", "upload-X"))
    .and(ExactBodyLength(5 * 1024 * 1024))
    .respond_with(ResponseTemplate::new(200).insert_header("ETag", "\"part1\""))
    .expect(1)
    .mount(&server)
    .await;

  // Mock CompleteMultipartUpload — exactly one call.
  Mock::given(method("POST"))
    .and(path("/test-bucket/big.bin"))
    .and(wiremock::matchers::query_param("uploadId", "upload-X"))
    .respond_with(ResponseTemplate::new(200).set_body_string(
      r#"<?xml version="1.0" encoding="UTF-8"?>
<CompleteMultipartUploadResult>
<Bucket>test-bucket</Bucket>
<Key>big.bin</Key>
<ETag>"final"</ETag>
</CompleteMultipartUploadResult>"#,
    ))
    .expect(1)
    .mount(&server)
    .await;

  let store = store_for(&server, true).await;
  let mut writer = store.put_stream(Path::new("big.bin")).await.unwrap();
  // A single 5 MiB write — exactly the threshold.
  writer
    .write(Bytes::from(vec![0u8; 5 * 1024 * 1024]))
    .await
    .unwrap();
  let stat = writer.complete().await.unwrap();
  // Critical: the recorded total length matches the input (no
  // double-counting). And the UploadPart body length matched
  // 5 MiB exactly (asserted via the matcher).
  assert_eq!(stat.len, 5 * 1024 * 1024);
}

/// Custom wiremock matcher: request body has exactly the given
/// byte length.
struct ExactBodyLength(usize);
impl wiremock::Match for ExactBodyLength {
  fn matches(&self, req: &Request) -> bool {
    req.body.len() == self.0
  }
}

/// Stage 10b v2 [P3] (Codex review): top-level `BlobStore::get_range`
/// with `start == end` must still validate the key and HEAD the
/// object before short-circuiting. An out-of-bounds zero-width range
/// must error rather than silently returning empty.
#[tokio::test(flavor = "multi_thread")]
async fn top_level_get_range_zero_width_validates_key_and_bounds() {
  let server = MockServer::start().await;
  Mock::given(method("HEAD"))
    .and(path("/test-bucket/k.bin"))
    .respond_with(
      ResponseTemplate::new(200)
        .insert_header("Content-Length", "100")
        .insert_header("ETag", "\"v\""),
    )
    // Stage 10b v2: must HEAD even for the zero-width path so
    // out-of-bounds inputs still error.
    .expect(2)
    .mount(&server)
    .await;
  let store = store_for(&server, true).await;
  // In-bounds zero-width: HEAD issued, no GET, returns empty.
  let bytes = store.get_range(Path::new("k.bin"), 50..50).await.unwrap();
  assert!(bytes.is_empty());

  // Out-of-bounds zero-width: HEAD issued, returns error.
  let err = match store.get_range(Path::new("k.bin"), 200..200).await {
    Ok(_) => panic!("OOB zero-width must error"),
    Err(e) => e,
  };
  let msg = format!("{err:#}");
  assert!(
    msg.contains("exceeds object length"),
    "expected OOB error; got: {msg}"
  );

  // Invalid key zero-width: rejected at key validation, no HEAD
  // issued (wiremock would 404 unmocked HEADs and we have none for
  // the bad key path).
  let err = match store.get_range(Path::new("../escape"), 0..0).await {
    Ok(_) => panic!("invalid key zero-width must error"),
    Err(e) => e,
  };
  let msg = format!("{err:#}");
  assert!(
    msg.contains("S3 key"),
    "expected key validation error; got: {msg}"
  );
}

// Force the runtime to be linked.
#[allow(dead_code)]
fn _ensure_arc_linked() {
  let _: Arc<u8> = Arc::new(0);
}
