//! [`CachedBlobStore`] — RAM-tier read cache wrapping any
//! [`BlobStore`].
//!
//! Stage 7's deliverable. Caches `get_range` and `Object::read_range`
//! results with byte-weighted LRU+TinyLFU eviction (`moka::sync`),
//! single-flight populate-on-miss (so concurrent misses for the same
//! key don't double-fetch the backend), and observability counters
//! exposed via [`CacheStats`].
//!
//! ## Two cache modes
//!
//! Per Codex's Stage 7 framing, the cache supports two distinct
//! identity models:
//!
//! - **Trusted-identity** (`get_range_verified`): keyed on the
//!   manifest-recorded [`ContentHash`]. Used by Stage 8 segment readers
//!   that hold an [`ArtifactIdentity`] from the manifest. The hash is
//!   what we wrote; identical content (e.g., identical segment files)
//!   shares cache entries even across different keys or backends. This
//!   is the strongest cache key shape and the right one for immutable
//!   segment artifacts.
//!
//! - **Observed-identity** (`get_range`, `Object::read_range`): keyed on
//!   `(key, observed provider_version)`. Used by generic callers that
//!   don't have an `ArtifactIdentity`. Provider version is best-effort:
//!   correct for [`LocalBlobStore`] (UUID per write) and S3-shaped
//!   backends with strong ETags, but degrades to "always miss" on
//!   backends without a stable per-write token.
//!
//! Both modes share a single byte-weighted budget so callers don't have
//! to size two caches separately.
//!
//! ## Single-flight
//!
//! A concurrent miss on the same `(content-hash | observed-version,
//! range)` cache key triggers exactly one backend fetch. The first
//! caller becomes the leader and runs the fetch; subsequent callers
//! find the in-flight slot, register a `futures::channel::oneshot`
//! sender, and `await` the corresponding receiver. When the leader
//! finishes, it transitions the slot from `Pending(Vec<Sender>)` to
//! `Done(InFlightResult)` atomically and drains all senders with the
//! result. The wait is async — `oneshot::Receiver::await` yields the
//! executor instead of blocking the OS thread, so single-thread
//! runtimes don't deadlock when the leader and waiters share the
//! same executor (Codex Stage 7 v3 [P1]). No tokio dep — uses
//! `parking_lot::Mutex` for state and `futures::channel::oneshot`
//! for waker-aware notification.
//!
//! ## What is NOT cached
//!
//! - `BlobStore::get()` (whole-object reads) — would defeat the byte
//!   budget on multi-MB segments. Passes through to the inner store.
//! - `BlobStore::stat()` — small, cheap, and always a freshness check
//!   for callers; not worth caching.
//! - Writes (`put`, `put_if_match`, `put_stream`, `delete`) — pure
//!   pass-through.
//!
//! Callers that want whole-object caching should call
//! `get_range[_verified](key, 0..len)` themselves with their own
//! size guard.

use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::{bail, Result};
use async_trait::async_trait;
use bytes::Bytes;
use futures::channel::oneshot;
use moka::sync::Cache;
use parking_lot::Mutex;

use super::blob::{
  ArtifactIdentity, BlobStore, Capabilities, ContentHash, Object, ObjectStat, ObjectWriter,
  PutIfMatchError,
};

/// Default byte budget for the RAM cache. 64 MB is comfortable for an
/// in-process index cache; tune via [`CachedBlobStore::with_capacity`].
pub const DEFAULT_CACHE_CAPACITY_BYTES: u64 = 64 * 1024 * 1024;

/// Half-open byte range `[start, end)` used as part of cache keys.
/// `Range<u64>` is intentionally avoided here for cache keys: deriving
/// `Hash + Eq` on a tuple-shaped newtype is more obvious about the
/// invariant we want and avoids surprises if `Range`'s trait derivations
/// shift across std versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ByteRange {
  pub start: u64,
  pub end: u64,
}

impl ByteRange {
  pub fn new(start: u64, end: u64) -> Self {
    Self { start, end }
  }

  pub fn len(&self) -> u64 {
    self.end.saturating_sub(self.start)
  }

  pub fn is_empty(&self) -> bool {
    self.start == self.end
  }
}

impl From<Range<u64>> for ByteRange {
  fn from(r: Range<u64>) -> Self {
    Self {
      start: r.start,
      end: r.end,
    }
  }
}

impl From<ByteRange> for Range<u64> {
  fn from(r: ByteRange) -> Self {
    r.start..r.end
  }
}

/// Cache key. The two variants share a single underlying cache + byte
/// budget so callers don't size two caches separately.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CacheKey {
  /// Trusted-identity entries. The content hash is what the writer
  /// computed at segment-write time and recorded in the manifest;
  /// identical content shares cache entries even across different keys
  /// or backends.
  Content { hash: ContentHash, range: ByteRange },
  /// Observed-identity entries. `version` is the
  /// `ObjectStat::provider_version` at the time the entry was inserted.
  /// Stale entries (under a now-superseded version) are never served
  /// because lookups always carry the *current* version.
  Observed {
    key: PathBuf,
    version: String,
    range: ByteRange,
  },
}

/// Observability counters surfaced for tests and operators.
#[derive(Debug, Default)]
pub struct CacheStats {
  /// Range-cache hits. Bumped before bytes are returned to the caller.
  pub hits: AtomicU64,
  /// Range-cache misses. Includes misses that became leaders AND misses
  /// that became waiters; use `leader_fetches` to distinguish.
  pub misses: AtomicU64,
  /// Subset of misses where the caller did the actual backend fetch.
  /// `misses - leader_fetches == inflight_waits`.
  pub leader_fetches: AtomicU64,
  /// Subset of misses where the caller waited on an in-flight load
  /// led by another caller. Asserts the single-flight property.
  pub inflight_waits: AtomicU64,
}

impl CacheStats {
  pub fn hits(&self) -> u64 {
    self.hits.load(Ordering::Relaxed)
  }
  pub fn misses(&self) -> u64 {
    self.misses.load(Ordering::Relaxed)
  }
  pub fn leader_fetches(&self) -> u64 {
    self.leader_fetches.load(Ordering::Relaxed)
  }
  pub fn inflight_waits(&self) -> u64 {
    self.inflight_waits.load(Ordering::Relaxed)
  }
}

/// In-flight slot for a cache key whose bytes are currently being
/// fetched by a leader. Concurrent callers find this in
/// `CachedBlobStore::inflight`, register a `oneshot::Sender`, and
/// `await` the corresponding `Receiver`. The leader transitions
/// `state` from `Pending` to `Done` once and drains all senders with
/// the result.
///
/// Stage 7 v3: replaced `parking_lot::Condvar::wait` (which blocks
/// the OS thread) with an async-aware oneshot-channel design (Codex
/// [P1]). Inside an `async fn` the previous wait would deadlock a
/// single-thread executor: leader awaits the backend, waiter blocks
/// the only thread, leader can never resume.
struct InFlight {
  state: Mutex<InFlightState>,
}

enum InFlightState {
  /// The leader is still fetching. Late waiters push their senders
  /// here; the leader drains and sends to each on completion or
  /// cancellation.
  Pending(Vec<oneshot::Sender<InFlightResult>>),
  /// The leader has published. New waiters arriving after this
  /// transition take the cached result directly without registering
  /// a sender.
  Done(InFlightResult),
}

#[derive(Clone)]
enum InFlightResult {
  Ok(Bytes),
  /// Stringified error so it's `Clone`. `anyhow::Error` is not Clone
  /// and we want all waiters to observe the same error class.
  /// Stringifying loses the chain but preserves the message for
  /// diagnostics.
  Err(String),
}

/// Drop guard that holds the leader's stake in the in-flight slot.
/// On drop *without* an explicit `disarm()`, publishes a cancellation
/// error to the slot and removes it from the in-flight map — so a
/// dropped/cancelled leader future doesn't permanently poison the
/// slot and block waiters forever (Codex Stage 7 v2 [P1]).
///
/// Success path: the leader publishes the real result via
/// `publish_and_drain`, calls `LeaderGuard::disarm`, then drops the
/// guard as a no-op.
///
/// Cancellation path: the future containing this guard is dropped
/// before `disarm` is called. The guard's `drop` transitions state
/// to `Done(Err("cancelled"))` and fires every still-pending
/// `oneshot::Sender` so waiters wake up with the cancellation error.
struct LeaderGuard {
  inflight: Arc<Mutex<HashMap<CacheKey, Arc<InFlight>>>>,
  cache_key: CacheKey,
  slot: Arc<InFlight>,
  fired: bool,
}

impl LeaderGuard {
  fn disarm(mut self) {
    self.fired = true;
  }
}

impl Drop for LeaderGuard {
  fn drop(&mut self) {
    if self.fired {
      return;
    }
    let cancellation = InFlightResult::Err(
      "cached fetch leader was cancelled before publishing".into(),
    );
    let senders = take_pending_senders(&self.slot, cancellation.clone());
    for tx in senders {
      // Receivers may have been dropped (their futures cancelled too);
      // ignore send errors.
      let _ = tx.send(cancellation.clone());
    }
    self.inflight.lock().remove(&self.cache_key);
  }
}

/// Atomically transition `slot.state` to `Done(final_result)` and
/// return the senders that were pending. If the slot is already Done
/// (defensive — shouldn't happen in normal flow), returns an empty
/// vec so callers can no-op.
fn take_pending_senders(
  slot: &Arc<InFlight>,
  final_result: InFlightResult,
) -> Vec<oneshot::Sender<InFlightResult>> {
  let mut state = slot.state.lock();
  match std::mem::replace(&mut *state, InFlightState::Done(final_result)) {
    InFlightState::Pending(senders) => senders,
    InFlightState::Done(prev) => {
      // Restore — never overwrite an already-published result.
      *state = InFlightState::Done(prev);
      Vec::new()
    }
  }
}

/// Convert an `InFlightResult` into the `Result<Bytes>` shape callers
/// expect. Errors are reconstituted as fresh `anyhow::Error`s with the
/// stringified message.
fn outcome(result: InFlightResult) -> Result<Bytes> {
  match result {
    InFlightResult::Ok(bytes) => Ok(bytes),
    InFlightResult::Err(msg) => bail!("cached fetch failed: {msg}"),
  }
}

/// Wrapper over any [`BlobStore`] that caches range reads. See the
/// module docs for cache modes, single-flight semantics, and what is
/// NOT cached.
pub struct CachedBlobStore {
  inner: Arc<dyn BlobStore>,
  cache: Cache<CacheKey, Bytes>,
  /// `Arc` so `CachedObject` (returned from `open()`) can share the
  /// same in-flight map as the parent store. Both wrappers should
  /// observe the same single-flight slot when racing on the same
  /// `(key, version, range)` tuple.
  inflight: Arc<Mutex<HashMap<CacheKey, Arc<InFlight>>>>,
  stats: Arc<CacheStats>,
}

impl CachedBlobStore {
  /// Wrap `inner` with the default 64 MB byte budget.
  pub fn new(inner: Arc<dyn BlobStore>) -> Self {
    Self::with_capacity(inner, DEFAULT_CACHE_CAPACITY_BYTES)
  }

  /// Wrap `inner` with `capacity_bytes` of byte-weighted RAM budget.
  /// Eviction is moka's default LRU+TinyLFU; entries are weighed by
  /// their `Bytes` length, so the cache holds approximately
  /// `capacity_bytes` worth of payload across both cache modes.
  pub fn with_capacity(inner: Arc<dyn BlobStore>, capacity_bytes: u64) -> Self {
    let cache = Cache::builder()
      .max_capacity(capacity_bytes)
      .weigher(|_k: &CacheKey, v: &Bytes| u32::try_from(v.len()).unwrap_or(u32::MAX))
      .build();
    Self {
      inner,
      cache,
      inflight: Arc::new(Mutex::new(HashMap::new())),
      stats: Arc::new(CacheStats::default()),
    }
  }

  /// Trusted-identity range read. Cache key is the manifest-recorded
  /// content hash + byte range, so identical content (e.g. the same
  /// segment file under different keys or backends) shares cache
  /// entries. This is the API Stage 8 segment readers should use when
  /// they have an [`ArtifactIdentity`] from the manifest.
  ///
  /// The `key` field of `artifact` is what's passed to
  /// [`BlobStore::get_range`] on miss.
  pub async fn get_range_verified(
    &self,
    artifact: &ArtifactIdentity,
    range: ByteRange,
  ) -> Result<Bytes> {
    validate_range_against_len(range, artifact.len)?;
    if range.is_empty() {
      return Ok(Bytes::new());
    }
    let key = CacheKey::Content {
      hash: artifact.content_hash,
      range,
    };
    let inner = self.inner.clone();
    let artifact_key = artifact.key.clone();
    self
      .lookup_or_fetch(key, move || {
        // The closure returns a boxed future so `lookup_or_fetch` can
        // hold the leader's fetch as a generic `Future` without
        // naming an opaque type. The fetch body itself is plain
        // async — `lookup_or_fetch` `.await`s it directly; no
        // `block_on` is involved.
        Box::pin(async move { inner.get_range(&artifact_key, range.into()).await })
      })
      .await
  }

  /// Stats accessor for tests and metrics surfaces.
  pub fn stats(&self) -> &CacheStats {
    &self.stats
  }

  /// Number of cached entries (for tests). Sum of both cache modes.
  pub fn entry_count(&self) -> u64 {
    self.cache.entry_count()
  }

  /// Best-effort total bytes currently held in the cache (for tests).
  pub fn weighted_size(&self) -> u64 {
    self.cache.weighted_size()
  }

  /// Core lookup/fetch routine used by both cache modes. Either
  /// returns from the moka cache (a hit), waits for an in-flight
  /// leader (single-flight), or becomes the leader and runs `fetch`.
  async fn lookup_or_fetch<F>(&self, key: CacheKey, fetch: F) -> Result<Bytes>
  where
    F: FnOnce() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Bytes>> + Send>>,
  {
    // Fast path: hit.
    if let Some(bytes) = self.cache.get(&key) {
      self.stats.hits.fetch_add(1, Ordering::Relaxed);
      return Ok(bytes);
    }

    // Slow path: miss. Become leader, join existing in-flight as
    // waiter, or take a cached entry if the leader published while we
    // were racing.
    self.stats.misses.fetch_add(1, Ordering::Relaxed);
    enum Role {
      Leader(Arc<InFlight>),
      Waiter {
        slot: Arc<InFlight>,
        rx: oneshot::Receiver<InFlightResult>,
      },
    }
    let role = {
      let mut map = self.inflight.lock();
      // Re-check the cache under the inflight lock to catch the case
      // where the leader populated the cache and removed itself from
      // the map between our cache.get() and our inflight.lock().
      if let Some(bytes) = self.cache.get(&key) {
        // Someone published while we were racing. Treat this as a
        // hit for stats consistency. (Misses is already bumped; we
        // accept slight over-count rather than racing decrement.)
        self.stats.hits.fetch_add(1, Ordering::Relaxed);
        return Ok(bytes);
      }
      if let Some(existing) = map.get(&key) {
        // Existing leader is mid-fetch: register as a waiter
        // *under* the map lock so the leader's Pending → Done
        // transition (which also requires the slot's state lock)
        // can't race past us.
        let mut state = existing.state.lock();
        match &mut *state {
          InFlightState::Done(r) => {
            // Leader published between dropping the map lock and
            // taking the slot lock. Take the result directly; no
            // need to register.
            let result = r.clone();
            drop(state);
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            return outcome(result);
          }
          InFlightState::Pending(senders) => {
            let (tx, rx) = oneshot::channel();
            senders.push(tx);
            Role::Waiter {
              slot: existing.clone(),
              rx,
            }
          }
        }
      } else {
        let slot = Arc::new(InFlight {
          state: Mutex::new(InFlightState::Pending(Vec::new())),
        });
        map.insert(key.clone(), slot.clone());
        Role::Leader(slot)
      }
    };

    match role {
      Role::Leader(slot) => {
        self
          .stats
          .leader_fetches
          .fetch_add(1, Ordering::Relaxed);
        // Hold a `LeaderGuard` across the `fetch().await` so that if
        // the future is dropped (HTTP request cancelled mid-fetch,
        // etc.), the slot is published with a cancellation result
        // and removed from the map rather than left as a permanent
        // waiter trap.
        let leader_guard = LeaderGuard {
          inflight: self.inflight.clone(),
          cache_key: key.clone(),
          slot: slot.clone(),
          fired: false,
        };
        let result = fetch().await;
        let publishable = match &result {
          Ok(bytes) => {
            self.cache.insert(key.clone(), bytes.clone());
            InFlightResult::Ok(bytes.clone())
          }
          Err(e) => InFlightResult::Err(format!("{e:#}")),
        };
        // Atomically transition Pending → Done and drain registered
        // senders. Any waiter that observed `Pending` while we were
        // fetching has a sender in this list; everyone else either
        // already saw `Done` (stat-cached the result) or arrives
        // after we drop the in-flight map entry (becoming a fresh
        // leader for a new fetch).
        let senders = take_pending_senders(&slot, publishable.clone());
        for tx in senders {
          let _ = tx.send(publishable.clone());
        }
        self.inflight.lock().remove(&key);
        leader_guard.disarm();
        result
      }
      Role::Waiter { slot: _slot, rx } => {
        self
          .stats
          .inflight_waits
          .fetch_add(1, Ordering::Relaxed);
        // Async wait; doesn't block the executing thread, so
        // single-thread runtimes don't deadlock.
        let result = rx.await.unwrap_or_else(|_| {
          // Sender dropped without sending. With LeaderGuard::Drop
          // this should be unreachable in practice — the guard
          // sends to all senders on cancellation — but treat it as
          // a cancellation just in case.
          InFlightResult::Err(
            "cached fetch single-flight sender was dropped without sending".into(),
          )
        });
        outcome(result)
      }
    }
  }
}

/// Reject inverted and zero-length-prep ranges *before* the cache is
/// touched. Stage 5's range contract requires `start <= end <= len`;
/// we enforce it here so cache misses that would have been errors
/// don't leave behind in-flight slots or cache pollution.
fn validate_range_against_len(range: ByteRange, len: u64) -> Result<()> {
  if range.start > range.end {
    bail!(
      "CachedBlobStore: inverted range {}..{}",
      range.start,
      range.end
    );
  }
  if range.end > len {
    bail!(
      "CachedBlobStore: range {}..{} exceeds object length {}",
      range.start,
      range.end,
      len
    );
  }
  Ok(())
}

#[async_trait]
impl BlobStore for CachedBlobStore {
  /// Pass-through. Stat results aren't cached — they're cheap and
  /// callers usually want freshness.
  async fn stat(&self, key: &Path) -> Result<ObjectStat> {
    self.inner.stat(key).await
  }

  /// Open an object. Returns a [`CachedObject`] wrapping the inner
  /// object, so subsequent `read_range` calls go through the
  /// observed-identity cache mode and share the parent store's
  /// in-flight map for single-flight semantics.
  async fn open(&self, key: &Path) -> Result<Arc<dyn Object>> {
    let base = self.inner.open(key).await?;
    let cached = CachedObject {
      base,
      key: key.to_path_buf(),
      cache: self.cache.clone(),
      inflight: self.inflight.clone(),
      stats: self.stats.clone(),
    };
    Ok(Arc::new(cached))
  }

  /// Observed-identity range read. Stats the inner store for the
  /// current `provider_version`, then caches by
  /// `(key, version, range)`. A future read after a write that
  /// changed `provider_version` is a guaranteed miss — the new key
  /// doesn't match the old cache entry. The old entry sits until
  /// LRU evicts it.
  ///
  /// When the backend's `provider_version` is `None` (some backends
  /// don't expose a stable per-write token), we bypass the cache and
  /// single-flight entirely and pass through to the inner store. The
  /// alternative — using a placeholder like `""` as the version — would
  /// alias every generation of the same key to one cache entry and
  /// silently serve stale bytes after a write (Codex Stage 7 [P1]).
  ///
  /// **Race window** (Codex Stage 7 v2 [P2]): observed-mode does a
  /// `stat` to capture `provider_version`, then a separate
  /// unconditioned `get_range` to fetch bytes. If the object is
  /// replaced between the two calls, the bytes returned are from the
  /// *new* generation but cached under the *old* version key. A
  /// later caller that observes the same old version would receive
  /// the new bytes from cache. For mutable objects this is a known
  /// limitation — backends that expose conditional reads (S3
  /// `If-Match`) would close the window, but this method does not
  /// require or use them.
  ///
  /// **For correctness on immutable artifacts** (Stage 8 segments and
  /// any other content-addressed data), prefer
  /// [`CachedBlobStore::get_range_verified`], which keys on the
  /// manifest-recorded content hash and is unaffected by concurrent
  /// writes to the same key.
  async fn get_range(&self, key: &Path, range: Range<u64>) -> Result<Bytes> {
    let range: ByteRange = range.into();
    let stat = self.inner.stat(key).await?;
    validate_range_against_len(range, stat.len)?;
    if range.is_empty() {
      return Ok(Bytes::new());
    }
    let Some(version) = stat.provider_version.clone() else {
      // Backend doesn't expose a stable per-write version token; the
      // safe degradation is "always miss" — pass through every call so
      // a write is never aliased with a stale cache entry.
      return self.inner.get_range(key, range.into()).await;
    };
    let cache_key = CacheKey::Observed {
      key: key.to_path_buf(),
      version,
      range,
    };
    let inner = self.inner.clone();
    let path = key.to_path_buf();
    self
      .lookup_or_fetch(cache_key, move || {
        Box::pin(async move { inner.get_range(&path, range.into()).await })
      })
      .await
  }

  /// Pass-through. Whole-object reads are deliberately uncached to
  /// avoid silent multi-MB cache entries that defeat the byte budget.
  async fn get(&self, key: &Path) -> Result<Bytes> {
    self.inner.get(key).await
  }

  async fn put(&self, key: &Path, body: Bytes) -> Result<ObjectStat> {
    self.inner.put(key, body).await
  }

  async fn put_stream(&self, key: &Path) -> Result<Box<dyn ObjectWriter>> {
    self.inner.put_stream(key).await
  }

  async fn put_if_match(
    &self,
    key: &Path,
    body: Bytes,
    expected: Option<&str>,
  ) -> std::result::Result<ObjectStat, PutIfMatchError> {
    self.inner.put_if_match(key, body, expected).await
  }

  async fn delete(&self, key: &Path) -> Result<()> {
    self.inner.delete(key).await
  }

  fn capabilities(&self) -> Capabilities {
    self.inner.capabilities()
  }
}

/// Wraps `Arc<dyn Object>` with caching of `read_range` results, keyed
/// on the object's pinned `provider_version`. The wrapped `Object`'s
/// `stat()` is held so the version is consistent across all reads
/// through this handle.
struct CachedObject {
  base: Arc<dyn Object>,
  key: PathBuf,
  cache: Cache<CacheKey, Bytes>,
  inflight: Arc<Mutex<HashMap<CacheKey, Arc<InFlight>>>>,
  stats: Arc<CacheStats>,
}

#[async_trait]
impl Object for CachedObject {
  fn stat(&self) -> &ObjectStat {
    self.base.stat()
  }

  async fn read_range(&self, range: Range<u64>) -> Result<Bytes> {
    let range: ByteRange = range.into();
    let stat = self.base.stat();
    validate_range_against_len(range, stat.len)?;
    if range.is_empty() {
      return Ok(Bytes::new());
    }
    // Same `provider_version = None` bypass as `CachedBlobStore::get_range`
    // (Codex Stage 7 [P1]). A pinned `Object` whose stat carries `None`
    // can't safely participate in the observed-identity cache.
    let Some(version) = stat.provider_version.clone() else {
      return self.base.read_range(range.into()).await;
    };
    let cache_key = CacheKey::Observed {
      key: self.key.clone(),
      version,
      range,
    };
    // Inline the lookup_or_fetch logic since it lives on
    // `CachedBlobStore`. Sharing the routine cleanly between the two
    // owners is more friction than a small duplication.
    if let Some(bytes) = self.cache.get(&cache_key) {
      self.stats.hits.fetch_add(1, Ordering::Relaxed);
      return Ok(bytes);
    }
    self.stats.misses.fetch_add(1, Ordering::Relaxed);
    enum Role {
      Leader(Arc<InFlight>),
      Waiter {
        slot: Arc<InFlight>,
        rx: oneshot::Receiver<InFlightResult>,
      },
    }
    let role = {
      let mut map = self.inflight.lock();
      if let Some(bytes) = self.cache.get(&cache_key) {
        self.stats.hits.fetch_add(1, Ordering::Relaxed);
        return Ok(bytes);
      }
      if let Some(existing) = map.get(&cache_key) {
        let mut state = existing.state.lock();
        match &mut *state {
          InFlightState::Done(r) => {
            let result = r.clone();
            drop(state);
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            return outcome(result);
          }
          InFlightState::Pending(senders) => {
            let (tx, rx) = oneshot::channel();
            senders.push(tx);
            Role::Waiter {
              slot: existing.clone(),
              rx,
            }
          }
        }
      } else {
        let slot = Arc::new(InFlight {
          state: Mutex::new(InFlightState::Pending(Vec::new())),
        });
        map.insert(cache_key.clone(), slot.clone());
        Role::Leader(slot)
      }
    };

    match role {
      Role::Leader(slot) => {
        self
          .stats
          .leader_fetches
          .fetch_add(1, Ordering::Relaxed);
        let leader_guard = LeaderGuard {
          inflight: self.inflight.clone(),
          cache_key: cache_key.clone(),
          slot: slot.clone(),
          fired: false,
        };
        let result = self.base.read_range(range.into()).await;
        let publishable = match &result {
          Ok(bytes) => {
            self.cache.insert(cache_key.clone(), bytes.clone());
            InFlightResult::Ok(bytes.clone())
          }
          Err(e) => InFlightResult::Err(format!("{e:#}")),
        };
        let senders = take_pending_senders(&slot, publishable.clone());
        for tx in senders {
          let _ = tx.send(publishable.clone());
        }
        self.inflight.lock().remove(&cache_key);
        leader_guard.disarm();
        result
      }
      Role::Waiter { slot: _slot, rx } => {
        self
          .stats
          .inflight_waits
          .fetch_add(1, Ordering::Relaxed);
        let result = rx.await.unwrap_or_else(|_| {
          InFlightResult::Err(
            "cached fetch single-flight sender was dropped without sending".into(),
          )
        });
        outcome(result)
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::storage::LocalBlobStore;
  use futures::executor::block_on;
  use tempfile::tempdir;

  fn store_chain(dir: &Path) -> (Arc<dyn BlobStore>, CachedBlobStore) {
    let inner: Arc<dyn BlobStore> = Arc::new(LocalBlobStore::new(dir.to_path_buf()));
    let cached = CachedBlobStore::with_capacity(inner.clone(), 1024 * 1024);
    (inner, cached)
  }

  fn make_artifact(key: &Path, body: &[u8], content_hash: [u8; 32]) -> ArtifactIdentity {
    ArtifactIdentity {
      key: key.to_path_buf(),
      len: body.len() as u64,
      content_hash: ContentHash::new(content_hash),
    }
  }

  /// Observed-mode: second read of the same range hits the cache.
  /// Counters reflect the hit/miss and the leader-fetch path.
  #[test]
  fn get_range_observed_caches_second_read() {
    let dir = tempdir().unwrap();
    let (inner, cached) = store_chain(dir.path());
    let key = Path::new("data");
    block_on(inner.put(key, Bytes::from_static(b"abcdefghij"))).unwrap();

    let r1 = block_on(cached.get_range(key, 2..5)).unwrap();
    assert_eq!(r1, Bytes::from_static(b"cde"));
    assert_eq!(cached.stats().hits(), 0);
    assert_eq!(cached.stats().misses(), 1);
    assert_eq!(cached.stats().leader_fetches(), 1);
    assert_eq!(cached.stats().inflight_waits(), 0);

    let r2 = block_on(cached.get_range(key, 2..5)).unwrap();
    assert_eq!(r2, Bytes::from_static(b"cde"));
    assert_eq!(cached.stats().hits(), 1);
    assert_eq!(cached.stats().misses(), 1);
  }

  /// Observed-mode: a write that changes `provider_version` invalidates
  /// the cache. The new read is a guaranteed miss because its cache
  /// key carries the new version. The old cache entry sits until LRU
  /// evicts it; never served because lookups always carry the current
  /// version.
  #[test]
  fn get_range_observed_misses_after_provider_version_change() {
    let dir = tempdir().unwrap();
    let (inner, cached) = store_chain(dir.path());
    let key = Path::new("flicker");
    block_on(inner.put(key, Bytes::from_static(b"AAAAAAAAAA"))).unwrap();

    let r1 = block_on(cached.get_range(key, 0..4)).unwrap();
    assert_eq!(r1, Bytes::from_static(b"AAAA"));
    assert_eq!(cached.stats().misses(), 1);

    // Same length! Without the per-write UUID inside `provider_version`
    // we'd serve the stale "AAAA" out of cache. With it, the cache key
    // is different and we get a fresh miss for the new content.
    block_on(inner.put(key, Bytes::from_static(b"BBBBBBBBBB"))).unwrap();
    let r2 = block_on(cached.get_range(key, 0..4)).unwrap();
    assert_eq!(
      r2,
      Bytes::from_static(b"BBBB"),
      "version-change must invalidate the cache; got stale: {r2:?}"
    );
    assert_eq!(cached.stats().misses(), 2);
  }

  /// Trusted-mode: identical content under different keys shares cache
  /// entries (because the cache key is the manifest content hash, not
  /// the path). Write the same body to two different keys with the
  /// same `ArtifactIdentity::content_hash`, read via
  /// `get_range_verified` for the first; the second read against the
  /// other key is a hit because the content hash matches.
  #[test]
  fn get_range_verified_shares_cache_across_keys_with_identical_content() {
    let dir = tempdir().unwrap();
    let (inner, cached) = store_chain(dir.path());
    let body = b"shared-content-across-keys";
    let hash = [42u8; 32];

    block_on(inner.put(Path::new("a/data"), Bytes::from_static(body))).unwrap();
    block_on(inner.put(Path::new("b/data"), Bytes::from_static(body))).unwrap();

    let art_a = make_artifact(Path::new("a/data"), body, hash);
    let art_b = make_artifact(Path::new("b/data"), body, hash);

    let r1 = block_on(cached.get_range_verified(&art_a, ByteRange::new(0, 6))).unwrap();
    assert_eq!(r1, Bytes::from_static(b"shared"));
    assert_eq!(cached.stats().misses(), 1);

    // Same content hash → same cache key → hit, regardless of `key`.
    let r2 = block_on(cached.get_range_verified(&art_b, ByteRange::new(0, 6))).unwrap();
    assert_eq!(r2, Bytes::from_static(b"shared"));
    assert_eq!(cached.stats().hits(), 1);
    assert_eq!(cached.stats().misses(), 1);
  }

  /// Trusted-mode: rejects inverted and out-of-bounds ranges before
  /// touching the cache, so failed validations don't leave behind
  /// in-flight slots or pollute the cache. Verifies Stage 5's contract
  /// is preserved through the cache layer.
  #[test]
  #[allow(clippy::reversed_empty_ranges)] // Intentional: testing rejection.
  fn get_range_verified_rejects_invalid_ranges_without_caching() {
    let dir = tempdir().unwrap();
    let (inner, cached) = store_chain(dir.path());
    let body = b"abcdefg";
    let hash = [7u8; 32];
    block_on(inner.put(Path::new("k"), Bytes::from_static(body))).unwrap();
    let art = make_artifact(Path::new("k"), body, hash);

    let inv = block_on(cached.get_range_verified(&art, ByteRange::new(5, 2)));
    assert!(inv.is_err(), "inverted range must error");

    let oob = block_on(cached.get_range_verified(&art, ByteRange::new(0, 100)));
    assert!(oob.is_err(), "out-of-bounds range must error");

    // Neither failed call should have touched stats or cache.
    assert_eq!(cached.stats().hits(), 0);
    assert_eq!(cached.stats().misses(), 0);
    assert_eq!(cached.entry_count(), 0);
  }

  /// Whole-object `get` is deliberately uncached — caching it as
  /// `0..len` would silently store multi-MB segments and defeat the
  /// byte budget. Verifies cache stays at zero entries after `get`.
  #[test]
  fn whole_object_get_is_not_cached() {
    let dir = tempdir().unwrap();
    let (inner, cached) = store_chain(dir.path());
    let key = Path::new("blob");
    block_on(inner.put(key, Bytes::from_static(b"large body"))).unwrap();

    let _ = block_on(cached.get(key)).unwrap();
    let _ = block_on(cached.get(key)).unwrap();
    assert_eq!(cached.entry_count(), 0, "whole-object get must not cache");
    assert_eq!(cached.stats().hits(), 0);
    assert_eq!(cached.stats().misses(), 0);
  }

  /// `CachedBlobStore::open` returns a `CachedObject` that caches its
  /// own range reads. The object is pinned to the version observed at
  /// open time, so subsequent reads through it serve from the cache
  /// even after the underlying file changes. New readers (a fresh
  /// `open` after the change) get the new version and a fresh miss.
  #[test]
  fn cached_object_read_range_caches_against_pinned_version() {
    let dir = tempdir().unwrap();
    let (inner, cached) = store_chain(dir.path());
    let key = Path::new("pinned");
    block_on(inner.put(key, Bytes::from_static(b"abcdefghij"))).unwrap();

    let obj = block_on(cached.open(key)).unwrap();
    let r1 = block_on(obj.read_range(0..3)).unwrap();
    assert_eq!(r1, Bytes::from_static(b"abc"));
    assert_eq!(cached.stats().misses(), 1);

    let r2 = block_on(obj.read_range(0..3)).unwrap();
    assert_eq!(r2, Bytes::from_static(b"abc"));
    assert_eq!(cached.stats().hits(), 1);
  }

  /// Single-flight: many concurrent misses for the same key result in
  /// exactly one backend fetch (one leader, the rest waiters). This is
  /// the load-bearing property Codex required for Stage 7.
  #[test]
  fn concurrent_misses_for_same_key_single_flight() {
    use std::sync::Barrier;
    use std::thread;

    let dir = tempdir().unwrap();
    let (inner, cached) = store_chain(dir.path());
    let cached = Arc::new(cached);
    let key = Path::new("contended");
    block_on(inner.put(key, Bytes::from_static(b"0123456789"))).unwrap();

    // Wrap the inner store in a small "slow" shim that sleeps before
    // serving the read. This widens the in-flight window so concurrent
    // callers genuinely race instead of completing serially.
    //
    // The `inner` we already wired into `cached` doesn't sleep; the
    // single-flight property is observable via `leader_fetches` /
    // `inflight_waits` stats regardless. A `Barrier` synchronizes the
    // threads' calls so they all enter `lookup_or_fetch` together.
    let _ = inner;

    let n_threads = 8;
    let barrier = Arc::new(Barrier::new(n_threads));
    let handles: Vec<_> = (0..n_threads)
      .map(|_| {
        let cached = Arc::clone(&cached);
        let barrier = Arc::clone(&barrier);
        let key = key.to_path_buf();
        thread::spawn(move || {
          barrier.wait();
          block_on(cached.get_range(&key, 0..5))
        })
      })
      .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    for r in &results {
      assert_eq!(r.as_ref().unwrap(), &Bytes::from_static(b"01234"));
    }

    let stats = cached.stats();
    let leader = stats.leader_fetches();
    let waiters = stats.inflight_waits();
    let hits_observed = stats.hits();

    // Exactly one fetch should have hit the inner store. The remaining
    // misses are either inflight-waiters (before the leader populated
    // the cache) or hits (after, via the inflight-lock-held re-check
    // path). Sum must equal n_threads-1; leader is exactly 1.
    assert_eq!(
      leader, 1,
      "single-flight: exactly one leader fetch across {n_threads} threads, got {leader}"
    );
    assert_eq!(
      waiters as usize + hits_observed as usize,
      n_threads - 1,
      "the other {} threads must each be a waiter or a re-checked hit; got {waiters} waiters + {hits_observed} hits",
      n_threads - 1
    );
  }

  /// Stage 7 [P1] regression: when the inner backend's
  /// `provider_version` is `None`, observed-mode caching MUST bypass
  /// instead of aliasing every generation under `(key, "", range)`.
  /// Without the bypass, a write would be followed by a stale cache
  /// hit. The dummy `VersionlessStore` strips `provider_version` from
  /// every `stat`; with the bypass, two reads bracketing a write
  /// observe the new bytes; without it, the second read serves
  /// stale.
  #[test]
  fn observed_mode_bypasses_cache_when_provider_version_is_none() {
    use std::ops::Range;

    /// Wraps a real BlobStore but strips `provider_version` from
    /// every `stat` so the cache layer sees `None`. Test fixture
    /// only — production backends always populate the field.
    struct VersionlessStore {
      inner: Arc<dyn BlobStore>,
    }

    #[async_trait]
    impl BlobStore for VersionlessStore {
      async fn stat(&self, key: &Path) -> Result<ObjectStat> {
        let mut s = self.inner.stat(key).await?;
        s.provider_version = None;
        Ok(s)
      }
      async fn open(&self, key: &Path) -> Result<Arc<dyn Object>> {
        // Replace the base object's stat with a versionless one too,
        // so CachedObject::read_range sees the same `None` shape.
        let base = self.inner.open(key).await?;
        Ok(Arc::new(VersionlessObject {
          stat: ObjectStat {
            len: base.stat().len,
            provider_version: None,
            provider_checksum: base.stat().provider_checksum.clone(),
          },
          base,
        }))
      }
      async fn get_range(&self, key: &Path, range: Range<u64>) -> Result<Bytes> {
        self.inner.get_range(key, range).await
      }
      async fn get(&self, key: &Path) -> Result<Bytes> {
        self.inner.get(key).await
      }
      async fn put(&self, key: &Path, body: Bytes) -> Result<ObjectStat> {
        let mut s = self.inner.put(key, body).await?;
        s.provider_version = None;
        Ok(s)
      }
      async fn put_stream(&self, key: &Path) -> Result<Box<dyn ObjectWriter>> {
        self.inner.put_stream(key).await
      }
      async fn put_if_match(
        &self,
        key: &Path,
        body: Bytes,
        expected: Option<&str>,
      ) -> std::result::Result<ObjectStat, PutIfMatchError> {
        self.inner.put_if_match(key, body, expected).await
      }
      async fn delete(&self, key: &Path) -> Result<()> {
        self.inner.delete(key).await
      }
      fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
      }
    }

    struct VersionlessObject {
      base: Arc<dyn Object>,
      stat: ObjectStat,
    }

    #[async_trait]
    impl Object for VersionlessObject {
      fn stat(&self) -> &ObjectStat {
        &self.stat
      }
      async fn read_range(&self, range: Range<u64>) -> Result<Bytes> {
        self.base.read_range(range).await
      }
    }

    let dir = tempdir().unwrap();
    let raw: Arc<dyn BlobStore> = Arc::new(LocalBlobStore::new(dir.path().to_path_buf()));
    let versionless: Arc<dyn BlobStore> = Arc::new(VersionlessStore { inner: raw.clone() });
    let cached = CachedBlobStore::new(versionless);
    let key = Path::new("data");

    block_on(raw.put(key, Bytes::from_static(b"AAAA"))).unwrap();
    let r1 = block_on(cached.get_range(key, 0..4)).unwrap();
    assert_eq!(r1, Bytes::from_static(b"AAAA"));

    // Without the [P1] bypass, this read would alias to the same
    // cache key (`(key, "", 0..4)`) and serve "AAAA" from cache.
    block_on(raw.put(key, Bytes::from_static(b"BBBB"))).unwrap();
    let r2 = block_on(cached.get_range(key, 0..4)).unwrap();
    assert_eq!(
      r2,
      Bytes::from_static(b"BBBB"),
      "versionless backend must bypass cache; got stale: {r2:?}"
    );

    // Cache stays empty because the bypass path doesn't insert.
    assert_eq!(
      cached.entry_count(),
      0,
      "versionless backend must not populate the cache"
    );

    // Same property through CachedObject::read_range. Open a fresh
    // object after another write; ensure read_range serves fresh
    // bytes, not a cached entry under `("", ..)`.
    block_on(raw.put(key, Bytes::from_static(b"CCCC"))).unwrap();
    let obj = block_on(cached.open(key)).unwrap();
    let via_obj = block_on(obj.read_range(0..4)).unwrap();
    assert_eq!(
      via_obj,
      Bytes::from_static(b"CCCC"),
      "versionless CachedObject::read_range must also bypass cache"
    );
    assert_eq!(cached.entry_count(), 0);
  }

  /// Stage 7 [P1] regression: a leader future that gets dropped before
  /// publishing must not leave the in-flight slot in `result = None`
  /// forever, blocking waiters on the condvar. The `LeaderGuard`'s
  /// `Drop` impl publishes a cancellation error and removes the slot
  /// from the in-flight map.
  ///
  /// We exercise the property by constructing a `LeaderGuard` directly
  /// (bypassing `lookup_or_fetch` so we don't have to actually drop a
  /// future mid-await), then asserting a thread waiting on the slot
  /// wakes up with the cancellation error and that the inflight map
  /// no longer contains the slot.
  #[test]
  fn cancelled_leader_publishes_cancellation_to_waiters() {
    let dir = tempdir().unwrap();
    let (_inner, cached) = store_chain(dir.path());

    // Insert a synthetic in-flight slot into the cache's map and
    // pre-register a waiter so the leader's drop has someone to
    // notify. This mirrors the runtime shape of `lookup_or_fetch`'s
    // waiter branch (register sender → await receiver).
    let cache_key = CacheKey::Observed {
      key: PathBuf::from("test"),
      version: "v0".into(),
      range: ByteRange::new(0, 4),
    };
    let slot = Arc::new(InFlight {
      state: Mutex::new(InFlightState::Pending(Vec::new())),
    });
    let (tx, rx) = oneshot::channel();
    match &mut *slot.state.lock() {
      InFlightState::Pending(senders) => senders.push(tx),
      InFlightState::Done(_) => unreachable!(),
    }
    {
      let mut map = cached.inflight.lock();
      map.insert(cache_key.clone(), slot.clone());
    }

    // Construct a LeaderGuard, then drop it without calling disarm.
    // This simulates the leader future being dropped/cancelled
    // before publishing.
    {
      let _leader_guard = LeaderGuard {
        inflight: cached.inflight.clone(),
        cache_key: cache_key.clone(),
        slot: slot.clone(),
        fired: false,
      };
    } // dropped here → drains pending senders with Err("...cancelled...")
      // and removes the slot from the map.

    // The waiter's receiver wakes up with the cancellation error.
    let result = block_on(rx).expect("sender must have fired");
    match result {
      InFlightResult::Err(s) => assert!(
        s.contains("cancelled"),
        "expected cancellation error, got: {s}"
      ),
      InFlightResult::Ok(_) => panic!("expected Err, got Ok"),
    }

    // The map no longer holds the slot — late-arriving callers will
    // become a fresh leader rather than waiting on a dead slot.
    assert!(
      !cached.inflight.lock().contains_key(&cache_key),
      "LeaderGuard's Drop must remove the slot from the inflight map"
    );
  }

  /// Stage 7 v3 [P1] regression: with the oneshot-based single-flight,
  /// concurrent misses on a single-thread executor must not deadlock.
  /// Previously the waiter's `parking_lot::Condvar::wait` blocked the
  /// only executor thread, preventing the leader from making progress.
  ///
  /// The test wraps `LocalBlobStore` in a `YieldOnceStore` that yields
  /// the executor at least once during `get_range`, ensuring the
  /// leader's `fetch().await` actually returns `Pending` (otherwise
  /// `LocalBlobStore`'s sync-internal `get_range` would resolve on
  /// first poll and never expose the deadlock window). With Condvar,
  /// `join!`-ing two such futures inside `block_on` would hang
  /// forever; with oneshot, both complete in order.
  #[test]
  fn single_thread_executor_no_deadlock_under_concurrent_miss() {
    use futures::future::join;
    use std::ops::Range;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    /// Yields the executor exactly once on the first poll, then
    /// resolves. Stable Rust, no tokio. Used by `YieldOnceStore` to
    /// inject a real `Pending` return into the inner `get_range`
    /// future so the leader's await is observable.
    struct YieldOnce {
      yielded: bool,
    }
    impl std::future::Future for YieldOnce {
      type Output = ();
      fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.yielded {
          Poll::Ready(())
        } else {
          self.yielded = true;
          cx.waker().wake_by_ref();
          Poll::Pending
        }
      }
    }
    fn yield_once() -> YieldOnce {
      YieldOnce { yielded: false }
    }

    /// Wraps an inner `BlobStore` and yields the executor once before
    /// each `get_range`. Other methods pass through unchanged.
    struct YieldOnceStore {
      inner: Arc<dyn BlobStore>,
    }

    #[async_trait]
    impl BlobStore for YieldOnceStore {
      async fn stat(&self, key: &Path) -> Result<ObjectStat> {
        self.inner.stat(key).await
      }
      async fn open(&self, key: &Path) -> Result<Arc<dyn Object>> {
        self.inner.open(key).await
      }
      async fn get_range(&self, key: &Path, range: Range<u64>) -> Result<Bytes> {
        yield_once().await;
        self.inner.get_range(key, range).await
      }
      async fn get(&self, key: &Path) -> Result<Bytes> {
        self.inner.get(key).await
      }
      async fn put(&self, key: &Path, body: Bytes) -> Result<ObjectStat> {
        self.inner.put(key, body).await
      }
      async fn put_stream(&self, key: &Path) -> Result<Box<dyn ObjectWriter>> {
        self.inner.put_stream(key).await
      }
      async fn put_if_match(
        &self,
        key: &Path,
        body: Bytes,
        expected: Option<&str>,
      ) -> std::result::Result<ObjectStat, PutIfMatchError> {
        self.inner.put_if_match(key, body, expected).await
      }
      async fn delete(&self, key: &Path) -> Result<()> {
        self.inner.delete(key).await
      }
      fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
      }
    }

    let dir = tempdir().unwrap();
    let raw: Arc<dyn BlobStore> = Arc::new(LocalBlobStore::new(dir.path().to_path_buf()));
    let yielding: Arc<dyn BlobStore> = Arc::new(YieldOnceStore { inner: raw.clone() });
    let cached = CachedBlobStore::new(yielding);

    let key = Path::new("data");
    block_on(raw.put(key, Bytes::from_static(b"abcdef"))).unwrap();

    // Both futures created up-front; `join!` polls them alternately
    // on the single executor thread. The first poll of `f1` becomes
    // leader, runs `yield_once` (returns Pending), suspends. `f2` is
    // polled, finds the in-flight slot, registers a sender, awaits
    // the receiver (returns Pending). With the OLD Condvar impl,
    // `f2` would block the executor thread here; the leader never
    // resumes. With oneshot, `f2`'s Pending releases the thread back
    // to the executor; `f1` resumes, completes, drains senders;
    // `f2`'s receiver is Ready and produces bytes.
    let (r1, r2) = block_on(join(
      cached.get_range(key, 0..3),
      cached.get_range(key, 0..3),
    ));
    let r1 = r1.unwrap();
    let r2 = r2.unwrap();
    assert_eq!(r1, Bytes::from_static(b"abc"));
    assert_eq!(r2, Bytes::from_static(b"abc"));
    // Single-flight property holds: only one backend fetch.
    assert_eq!(cached.stats().leader_fetches(), 1);
  }

  /// Pass-through integrity: `stat`, `put`, `delete`, `capabilities`,
  /// and `put_if_match` all forward to the inner store unchanged.
  #[test]
  fn cached_blob_store_passes_through_non_read_methods() {
    let dir = tempdir().unwrap();
    let (inner, cached) = store_chain(dir.path());
    let key = Path::new("k");

    block_on(cached.put(key, Bytes::from_static(b"v0"))).unwrap();

    // stat through cached and stat through inner agree.
    let s_cached = block_on(cached.stat(key)).unwrap();
    let s_inner = block_on(inner.stat(key)).unwrap();
    assert_eq!(s_cached.len, s_inner.len);
    assert_eq!(s_cached.provider_version, s_inner.provider_version);

    // capabilities forward.
    assert_eq!(cached.capabilities(), inner.capabilities());

    // put_if_match forwards. Use the just-observed version.
    let v = s_cached.provider_version.clone().unwrap();
    let s2 = block_on(cached.put_if_match(key, Bytes::from_static(b"v1"), Some(&v))).unwrap();
    assert_ne!(s2.provider_version, s_cached.provider_version);

    // delete forwards.
    block_on(cached.delete(key)).unwrap();
    assert!(block_on(inner.stat(key)).is_err());
  }
}
