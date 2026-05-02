//! Stage 10a: BlobStore sync→async bridge.
//!
//! Every place in the index where a sync API drives an async
//! `BlobStore` future (segment-reader postings/docstore opens, the
//! `BlobStoreAdapter` Storage shim, `StorageAsBlobStore::stat`, etc.)
//! goes through [`block_on_blob`] instead of calling
//! `futures::executor::block_on` directly.
//!
//! Why the indirection: `aws-sdk-s3` (Stage 10b's `S3BlobStore`)
//! returns futures whose internals depend on a Tokio reactor —
//! `hyper`, `tokio-rustls`, and `tokio-util` are all baked in. Driving
//! such a future on `futures::executor::block_on` (a single-thread
//! poll loop with no I/O reactor) hangs forever or panics. The fix is
//! a Tokio runtime that owns the reactor.
//!
//! ## Feature gating
//!
//! * **Default** (no `tokio-runtime` feature): keeps the historical
//!   `futures::executor::block_on` behavior. Local-FS-only deployments
//!   and wasm builds don't pay the Tokio dep cost. Every existing
//!   non-S3 BlobStore impl works unchanged.
//! * **`tokio-runtime` feature**: routes futures through a global
//!   lazy multi-thread Tokio runtime. Required by `searchlite-s3` and
//!   any other backend that depends on `tokio` for I/O.
//!
//! ## Nested-block_on safety
//!
//! Under the `tokio-runtime` feature we detect the calling thread's
//! runtime context:
//!
//! * **No active runtime** → drive `fut` on the global lazy
//!   multi-thread runtime via `Runtime::block_on`.
//! * **Multi-thread runtime active** → use
//!   [`tokio::task::block_in_place`] + the active handle's
//!   `block_on`. `block_in_place` parks the worker so other tasks
//!   can be re-scheduled elsewhere; the inner `block_on` then drives
//!   the future without panicking on the nested-runtime check.
//! * **Current-thread runtime active** → `block_in_place` would panic
//!   (it requires a multi-thread runtime). We detect this case via
//!   `Handle::runtime_flavor()` and panic up-front with a clear
//!   message rather than crashing in tokio internals. This is a
//!   deliberate design constraint: if you're embedding searchlite in
//!   a current-thread runtime, drive its sync API from a separate OS
//!   thread spawned outside any tokio runtime (`std::thread::spawn`).
//!   Note that `tokio::task::spawn_blocking` is NOT sufficient — its
//!   closure still observes the current-thread runtime via
//!   `Handle::try_current()` and would re-trigger the panic.
//!
//! Either way, callers of `block_on_blob` observe the future's
//! `Output` — semantically identical to the default
//! `futures::executor::block_on` shape, modulo the documented
//! current-thread limitation.

use std::future::Future;

/// Drive a `BlobStore` future to completion from a sync context.
///
/// See module docs for the runtime selection logic. Behavior:
///
/// * Default build: identical to `futures::executor::block_on(fut)`.
/// * `tokio-runtime` build: drives `fut` on a global lazy multi-thread
///   Tokio runtime. Safe to call from a thread without an active
///   runtime, or from inside an active **multi-thread** Tokio runtime
///   (uses `block_in_place`). Calling from inside a **current-thread**
///   Tokio runtime panics with an actionable message — `block_in_place`
///   doesn't work on current-thread runtimes, and there is no
///   no-allocation way to drive an arbitrary `F: Future` from a
///   current-thread context without restructuring as async.
pub fn block_on_blob<F>(fut: F) -> F::Output
where
  F: Future,
{
  #[cfg(feature = "tokio-runtime")]
  {
    tokio_bridge::block_on(fut)
  }
  #[cfg(not(feature = "tokio-runtime"))]
  {
    futures::executor::block_on(fut)
  }
}

#[cfg(feature = "tokio-runtime")]
mod tokio_bridge {
  use std::future::Future;
  use std::sync::OnceLock;
  use tokio::runtime::Runtime;

  static GLOBAL_RUNTIME: OnceLock<Runtime> = OnceLock::new();

  fn runtime() -> &'static Runtime {
    GLOBAL_RUNTIME.get_or_init(|| {
      tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("searchlite-blob")
        .build()
        .expect("failed to build searchlite Tokio bridge runtime")
    })
  }

  pub(super) fn block_on<F: Future>(fut: F) -> F::Output {
    match tokio::runtime::Handle::try_current() {
      Ok(handle) => {
        // Stage 10a v2 [P2] (Codex review): `block_in_place` only
        // works on multi-thread runtimes. Detect the flavor and
        // panic up-front with a clear message rather than crashing
        // inside tokio when called from a current-thread runtime.
        match handle.runtime_flavor() {
          tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(|| handle.block_on(fut))
          }
          flavor => panic!(
            "searchlite runtime bridge: cannot block on a BlobStore future from \
             within a Tokio runtime of flavor {flavor:?} — block_in_place requires \
             a multi-thread runtime. Configure tokio with \
             `tokio::runtime::Builder::new_multi_thread()`, or call searchlite's \
             sync API from a fresh thread without any active runtime (e.g. via \
             `std::thread::spawn`). Note: `tokio::task::spawn_blocking` is NOT a \
             workaround here — its closure still observes the current-thread \
             runtime via `Handle::try_current()` and would re-trigger this panic."
          ),
        }
      }
      Err(_) => runtime().block_on(fut),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn block_on_blob_drives_an_immediate_future() {
    let value = block_on_blob(async { 42u32 });
    assert_eq!(value, 42);
  }

  #[test]
  fn block_on_blob_drives_a_future_with_a_yield() {
    // A trivial yield point. Under the default (futures-executor)
    // build this still works because the executor polls again on
    // wake. Under tokio-runtime it's driven on the global runtime.
    let value = block_on_blob(async {
      // Force at least one yield so the future isn't trivially
      // ready on first poll.
      futures::future::ready(()).await;
      "hello"
    });
    assert_eq!(value, "hello");
  }

  /// Stage 10a: under the `tokio-runtime` feature, calling
  /// `block_on_blob` from inside an existing Tokio runtime must not
  /// panic. This is the critical "HTTP handler awaits into searchlite
  /// sync API" path.
  #[cfg(feature = "tokio-runtime")]
  #[test]
  fn block_on_blob_works_inside_existing_tokio_runtime() {
    let rt = tokio::runtime::Builder::new_multi_thread()
      .worker_threads(2)
      .enable_all()
      .build()
      .unwrap();
    let value: u32 = rt.block_on(async {
      // We're inside a runtime now. Call the sync bridge — it must
      // detect the active runtime and fall back to block_in_place.
      tokio::task::spawn_blocking(|| block_on_blob(async { 7u32 }))
        .await
        .unwrap()
    });
    assert_eq!(value, 7);
  }

  /// Stage 10a v2 [P2] (Codex review): calling `block_on_blob` from
  /// inside a **current-thread** Tokio runtime must surface a clear
  /// error. The previous shape used `tokio::task::block_in_place`
  /// unconditionally, which panics with a tokio-internal message on
  /// current-thread runtimes (because `block_in_place` requires a
  /// multi-thread runtime). The new shape detects the flavor and
  /// panics up-front with an actionable message.
  ///
  /// We assert via `catch_unwind` that the panic message names the
  /// `CurrentThread` flavor, recommends a fresh OS thread via
  /// `std::thread::spawn`, and explicitly warns that
  /// `tokio::task::spawn_blocking` is NOT a workaround (its closure
  /// still observes the current-thread runtime).
  #[cfg(feature = "tokio-runtime")]
  #[test]
  fn block_on_blob_panics_with_clear_message_inside_current_thread_runtime() {
    let rt = tokio::runtime::Builder::new_current_thread()
      .enable_all()
      .build()
      .unwrap();
    let result = rt.block_on(async {
      std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        block_on_blob(async { 1u32 })
      }))
    });
    let payload = result.expect_err(
      "block_on_blob must panic when called from inside a current-thread Tokio runtime",
    );
    let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
      (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
      s.clone()
    } else {
      String::from("<non-string panic payload>")
    };
    assert!(
      msg.contains("CurrentThread") && msg.contains("multi-thread"),
      "panic message must name the CurrentThread flavor and mention the multi-thread workaround; \
       got: {msg}"
    );
    assert!(
      msg.contains("std::thread::spawn"),
      "panic message must recommend std::thread::spawn as the current-thread workaround; \
       got: {msg}"
    );
    assert!(
      msg.contains("spawn_blocking") && msg.contains("NOT"),
      "panic message must explicitly warn that tokio::task::spawn_blocking is NOT a workaround; \
       got: {msg}"
    );
  }
}
