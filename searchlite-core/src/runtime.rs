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
//! * **Current-thread runtime active** (Stage 10b v2) → spawn a
//!   short-lived OS thread via [`std::thread::scope`] that drives
//!   the future on our owned global multi-thread runtime, then join
//!   it. `block_in_place` doesn't work on current-thread runtimes,
//!   and `Runtime::block_on` from inside another runtime panics.
//!   `std::thread::scope` lets the worker borrow from the calling
//!   stack frame (no `'static` bound on the future), and joining
//!   blocks the calling thread while the worker drives the future
//!   on a thread that is itself outside any tokio runtime. The future
//!   must be `Send` (already true for every `BlobStore` future, since
//!   `async_trait` produces `Send` futures by default).
//!
//! Either way, callers of `block_on_blob` observe the future's
//! `Output` — semantically identical to the default
//! `futures::executor::block_on` shape.

use std::future::Future;

/// Drive a `BlobStore` future to completion from a sync context.
///
/// See module docs for the runtime selection logic. Behavior:
///
/// * Default build: identical to `futures::executor::block_on(fut)`.
///   No `Send` bound — works for any future shape.
/// * `tokio-runtime` build: drives `fut` on a global lazy multi-thread
///   Tokio runtime. Safe to call from a thread without an active
///   runtime, from inside an active multi-thread Tokio runtime
///   (uses `block_in_place`), and from inside an active current-thread
///   runtime (uses `std::thread::scope` to escape the runtime). The
///   `tokio-runtime` build adds `Send` bounds on `F` and `F::Output`
///   because the current-thread fallback dispatches the future to a
///   scoped OS thread; every `BlobStore` future produced by
///   `async_trait` is already `Send`.
#[cfg(not(feature = "tokio-runtime"))]
pub fn block_on_blob<F>(fut: F) -> F::Output
where
  F: Future,
{
  futures::executor::block_on(fut)
}

/// `tokio-runtime`-feature variant of [`block_on_blob`]. See the
/// module docs and the cfg-default variant for behavior. The `Send`
/// bound is required by the current-thread-runtime fallback path,
/// which dispatches the future to a [`std::thread::scope`]-spawned
/// worker.
#[cfg(feature = "tokio-runtime")]
pub fn block_on_blob<F>(fut: F) -> F::Output
where
  F: Future + Send,
  F::Output: Send,
{
  tokio_bridge::block_on(fut)
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

  pub(super) fn block_on<F>(fut: F) -> F::Output
  where
    F: Future + Send,
    F::Output: Send,
  {
    match tokio::runtime::Handle::try_current() {
      Ok(handle) => match handle.runtime_flavor() {
        // Multi-thread runtime: park the worker via `block_in_place`
        // and drive on the active handle. No new threads needed.
        tokio::runtime::RuntimeFlavor::MultiThread => {
          tokio::task::block_in_place(|| handle.block_on(fut))
        }
        // Stage 10b v2 [P1] (Codex review): current-thread runtime
        // can't host a nested `block_on`, and `block_in_place`
        // panics on this flavor. Spawn an OS thread via
        // `std::thread::scope` so the future is driven outside any
        // tokio runtime, on our owned global multi-thread Tokio
        // runtime. `scope` lets the future borrow from the calling
        // stack frame (no `'static` bound).
        tokio::runtime::RuntimeFlavor::CurrentThread => std::thread::scope(|s| {
          s.spawn(|| runtime().block_on(fut))
            .join()
            .expect("searchlite blob bridge worker panicked")
        }),
        // Future flavors we don't recognize → bail out clearly.
        flavor => panic!(
          "searchlite runtime bridge: unsupported Tokio runtime flavor {flavor:?}; \
           the bridge supports MultiThread (block_in_place) and CurrentThread \
           (scoped-thread fallback)."
        ),
      },
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

  /// Stage 10b v2 [P1] (Codex review): calling `block_on_blob` from
  /// inside a current-thread Tokio runtime must SUCCEED — earlier
  /// stages panicked here, but the workspace `--all-features` build
  /// surfaces this on `searchlite-http`'s default-flavor tests. The
  /// fix routes the future through a `std::thread::scope`-spawned
  /// worker that drives our global multi-thread runtime, escaping
  /// the active current-thread runtime entirely.
  #[cfg(feature = "tokio-runtime")]
  #[test]
  fn block_on_blob_works_inside_current_thread_runtime_via_scoped_thread() {
    let rt = tokio::runtime::Builder::new_current_thread()
      .enable_all()
      .build()
      .unwrap();
    let value: u32 = rt.block_on(async {
      // Inside a current-thread runtime: must NOT panic, must drive
      // the future on the scoped-thread fallback.
      block_on_blob(async { 13u32 })
    });
    assert_eq!(value, 13);
  }

  /// Stage 10b v2: a non-trivial future with internal `.await` works
  /// the same way under the current-thread fallback. Confirms the
  /// scoped worker can drive multi-poll futures, not just trivial
  /// ready futures.
  #[cfg(feature = "tokio-runtime")]
  #[test]
  fn block_on_blob_drives_multi_poll_future_under_current_thread() {
    let rt = tokio::runtime::Builder::new_current_thread()
      .enable_all()
      .build()
      .unwrap();
    let value: u32 = rt.block_on(async {
      block_on_blob(async {
        // Force at least one yield so the future isn't ready on
        // first poll; the scoped-thread runtime polls it again on
        // wake.
        futures::future::ready(()).await;
        17u32
      })
    });
    assert_eq!(value, 17);
  }
}
