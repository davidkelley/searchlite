//! Stage 10c: bake-local-then-upload sync helper.
//!
//! Walks a local index directory and uploads every regular file
//! verbatim to an S3-compatible bucket via [`S3BlobStore::put`]. The
//! relative-to-root path becomes the BlobStore key; the
//! [`crate::S3Config::prefix`] (if any) is applied inside
//! [`crate::S3BlobStore::resolve_key`] — never here. This keeps the
//! single-source-of-truth namespace mapping that
//! [`crate::open_index_read_only`] depends on.
//!
//! ## Fail-closed contract (Codex Stage 10c review)
//!
//! Refuses to upload a partially-baked index. Errors before any
//! `put` is issued if any of the following hold for `local_root`:
//!
//! * `MANIFEST.json.pending` exists — recovery required. Reopen the
//!   source index mutably to reconcile (the writer's commit-time
//!   reconciler promotes valid pending bytes; the open path's
//!   recovery handles legacy v1), then re-sync.
//! * `wal.log` is non-empty — uncommitted/unflushed state. The
//!   reader-side serving path doesn't replay WAL, so anything in
//!   it would be silently dropped.
//! * Any `*.tmp-*` staging file (atomic-write artifacts left from
//!   a crashed write).
//! * `MANIFEST.json` is missing, malformed, below the latest
//!   manifest schema version, or contains non-portable segment
//!   paths (Stage 10c v2 [P2], Codex review). The S3 open path
//!   resolves keys against an empty logical root, so absolute or
//!   root-prefixed-relative segment paths from a legacy v1 manifest
//!   would silently miss after upload. Run a local mutable
//!   open-then-commit first to upgrade the manifest in place, then
//!   re-sync.
//!
//! ## Manifest publish ordering (Codex Stage 10c v2 [P1])
//!
//! `MANIFEST.json` is the visibility fence. The sync sequence is:
//!
//! 1. Validate the local manifest (preflight).
//! 2. Upload every NON-manifest file (segment artifacts, etc.).
//! 3. Upload the captured `MANIFEST.json` bytes LAST.
//!
//! If sync fails at any point before step 3, no remote manifest
//! exists and `open_index_read_only` against the prefix surfaces a
//! clean NotFound rather than a partially-published index pointing
//! at missing segment files.

use std::path::{Component, Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use bytes::Bytes;
use searchlite_core::storage::blob::BlobStore;
use searchlite_core::{Manifest, MANIFEST_LATEST_VERSION};

use crate::config::S3Config;
use crate::store::S3BlobStore;

const MANIFEST_FILE_NAME: &str = "MANIFEST.json";

/// Stage 10c v4 [P2] (Codex review): centralized "is this path
/// uploadable?" predicate, shared by [`upload_dir`] (deciding what
/// to skip) and [`preflight_manifest`] (refusing manifests that
/// reference paths the walker would skip).
///
/// A relative-to-`local_root` path is uploadable iff:
///
/// * No path component starts with `.` (dot-files / hidden dirs).
/// * No path component is named `wal.log` (the WAL is local-only;
///   the read-only open path never replays it).
/// * It is not the top-level `MANIFEST.json` (handled separately
///   by the publish-fence step).
///
/// Without sharing this predicate between the walker and the
/// preflight, a malformed manifest could legally name a
/// `.hidden.post` artifact, pass the existence check, get silently
/// SKIPPED during upload, and still publish `MANIFEST.json` —
/// leaving an unservable remote index pointing at a key that was
/// never PUT.
fn is_uploadable_relative_path(relative: &Path) -> bool {
  for component in relative.components() {
    if let Component::Normal(name) = component {
      let s = match name.to_str() {
        Some(s) => s,
        None => return false,
      };
      if s.starts_with('.') || s == "wal.log" {
        return false;
      }
    }
  }
  // Top-level MANIFEST.json: the manifest IS the fence and is
  // published separately.
  if relative == Path::new(MANIFEST_FILE_NAME) {
    return false;
  }
  true
}

/// Stage 10c v5 [P2] (Codex review): assert that a manifest key is
/// in the canonical form the sync walker would emit. The walker
/// builds keys via `read_dir` + `strip_prefix(local_root)`, which
/// produces no leading `./`, no trailing `/`, no `//`, no
/// backslash separators, and no `.`/`..`/`RootDir`/`Prefix`
/// components.
///
/// Without this check, a v2 manifest can name `./seg_X.post`:
/// `validate_v2_relative` accepts it (no `..`, no leading slash),
/// `local_root.join("./seg_X.post")` stats successfully (filesystem
/// collapses `./`), but the walker emits the key as `seg_X.post`
/// (no `./`) and S3 stores keys verbatim. The manifest would then
/// publish a reference to `./seg_X.post`, but the bytes live at
/// `seg_X.post` on the remote — a visible-but-unservable index.
///
/// Returns Ok(()) if the key is canonical; Err otherwise with a
/// reason. Callers in `preflight_manifest` then `bail!` with a
/// uniform error.
fn validate_canonical_segment_key(key: &str) -> Result<(), &'static str> {
  if key.is_empty() {
    return Err("key is empty");
  }
  if key.contains('\\') {
    return Err("key contains a backslash separator");
  }
  if key.starts_with('/') {
    return Err("key starts with `/`");
  }
  if key.ends_with('/') {
    return Err("key ends with `/`");
  }
  if key.contains("//") {
    return Err("key contains `//` (repeated separators)");
  }
  let p = Path::new(key);
  for component in p.components() {
    match component {
      Component::Normal(_) => {}
      Component::CurDir => return Err("key contains a `.` component"),
      Component::ParentDir => return Err("key contains a `..` component"),
      Component::RootDir => return Err("key contains a root component"),
      Component::Prefix(_) => return Err("key contains a platform prefix"),
    }
  }
  Ok(())
}

/// Per-call summary of a successful [`sync_to_s3`] run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyncReport {
  /// Number of regular files uploaded (segment artifacts +
  /// manifest).
  pub files: usize,
  /// Total bytes uploaded across every file.
  pub bytes: u64,
}

/// Bake-and-upload a local index to an S3-compatible bucket.
///
/// See module docs for the fail-closed cases this rejects and the
/// manifest-last publish ordering. The `S3Config.prefix` is applied
/// inside `S3BlobStore::resolve_key`; passing the prefix in
/// `local_root` would double-prefix.
pub async fn sync_to_s3(local_root: &Path, s3_config: S3Config) -> Result<SyncReport> {
  if !local_root.is_dir() {
    bail!("sync_to_s3: local_root {local_root:?} is not a directory");
  }
  // Pre-flight invariants (fail-closed). Errors here happen BEFORE
  // any HTTP request is issued.
  preflight_local_root(local_root)?;
  let manifest_bytes = preflight_manifest(local_root)?;

  let store = S3BlobStore::new(s3_config).await?;
  let mut report = SyncReport { files: 0, bytes: 0 };

  // Step 1: upload every non-manifest file. The manifest is the
  // visibility fence — if any of these fail, no remote manifest
  // gets written and the prefix surfaces NotFound on open.
  upload_dir(&store, local_root, local_root, SkipManifest, &mut report).await?;

  // Step 2: upload MANIFEST.json LAST. After this PUT, the prefix
  // is officially "published" — `open_index_read_only` will succeed
  // against it.
  let manifest_len = manifest_bytes.len() as u64;
  store
    .put(Path::new(MANIFEST_FILE_NAME), Bytes::from(manifest_bytes))
    .await
    .with_context(|| "sync_to_s3: final MANIFEST.json publish (visibility fence)")?;
  report.files += 1;
  report.bytes = report
    .bytes
    .checked_add(manifest_len)
    .ok_or_else(|| anyhow!("sync_to_s3: total bytes overflow"))?;

  Ok(report)
}

/// Pass to [`upload_dir`] to control whether the manifest is
/// uploaded inline (true) or skipped (caller publishes it later).
#[derive(Clone, Copy)]
struct SkipManifest;

/// Recursive directory walk that uploads regular files. Skips
/// `wal.log` (read-only opens don't replay it; preflight already
/// verified empty), `MANIFEST.json` (the caller publishes it last
/// as the visibility fence), and dot-files (hidden state).
fn upload_dir<'a>(
  store: &'a S3BlobStore,
  base: &'a Path,
  dir: &'a Path,
  _skip_manifest: SkipManifest,
  report: &'a mut SyncReport,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
  Box::pin(async move {
    for entry in std::fs::read_dir(dir).with_context(|| format!("sync_to_s3: read_dir({dir:?})"))? {
      let entry = entry.with_context(|| format!("sync_to_s3: read_dir entry under {dir:?}"))?;
      let path = entry.path();
      let relative = path.strip_prefix(base).map_err(|_| {
        anyhow!("sync_to_s3: file {path:?} is not under base {base:?} (this is a bug)")
      })?;
      // Stage 10c v4 [P2]: defer to the centralized predicate so
      // the walker's skip rules and `preflight_manifest`'s
      // uploadability assertion can never drift apart.
      if !is_uploadable_relative_path(relative) {
        continue;
      }
      let metadata = entry
        .metadata()
        .with_context(|| format!("sync_to_s3: metadata({path:?})"))?;
      if metadata.is_dir() {
        upload_dir(store, base, &path, SkipManifest, report).await?;
        continue;
      }
      if !metadata.is_file() {
        continue;
      }
      let bytes = std::fs::read(&path).with_context(|| format!("sync_to_s3: reading {path:?}"))?;
      let len = bytes.len() as u64;
      store
        .put(relative, Bytes::from(bytes))
        .await
        .with_context(|| format!("sync_to_s3: put {path:?} → {relative:?}"))?;
      report.files += 1;
      report.bytes = report
        .bytes
        .checked_add(len)
        .ok_or_else(|| anyhow!("sync_to_s3: total bytes overflow"))?;
    }
    Ok(())
  })
}

/// Stage 10c [Codex review]: pre-flight invariants for the local
/// directory shape. Fail before any upload is issued so a
/// partial-state index can't reach the cloud.
fn preflight_local_root(local_root: &Path) -> Result<()> {
  let pending = local_root.join("MANIFEST.json.pending");
  if pending.exists() {
    bail!(
      "sync_to_s3: refusing to upload — `MANIFEST.json.pending` exists at \
       {pending:?}. Recovery is required: reopen the source index mutably \
       (reads/commits will promote valid pending bytes or discard them), \
       then re-sync."
    );
  }
  let wal = local_root.join("wal.log");
  if wal.exists() {
    let metadata =
      std::fs::metadata(&wal).with_context(|| format!("sync_to_s3: metadata({wal:?})"))?;
    if metadata.len() > 0 {
      bail!(
        "sync_to_s3: refusing to upload — `wal.log` is non-empty at {wal:?} \
         ({} bytes). The bake-and-serve workflow requires a quiesced index: \
         commit any pending writes, drop active writers, and ensure the WAL \
         has been truncated/checkpointed. Read-only opens do NOT replay the \
         WAL, so any state still in it would be silently dropped on the \
         cloud side.",
        metadata.len()
      );
    }
  }
  // Reject any *.tmp-* staging file from atomic_write. The presence
  // of one indicates a crashed/in-flight write that hasn't been
  // cleaned up.
  for path in walkdir(local_root)? {
    let file_name = path
      .file_name()
      .and_then(|n| n.to_str())
      .unwrap_or_default();
    if file_name.contains(".tmp-") {
      bail!(
        "sync_to_s3: refusing to upload — staging file present at {path:?}. \
         Atomic-write staging artifacts should be cleaned up before sync; \
         their presence indicates an in-flight or crashed write."
      );
    }
  }
  Ok(())
}

/// Stage 10c v2 [P2] (Codex review): read + validate the local
/// manifest before any upload. Returns the raw bytes for the
/// final visibility-fence PUT.
///
/// The validations match what `open_index_read_only` requires:
///
/// * Latest version: `MANIFEST_LATEST_VERSION` (currently v2).
///   Lower versions would resolve segment paths against an empty
///   root and miss/reject them.
/// * Every segment's `paths` validates as relative-portable
///   (`SegmentPaths::validate_v2_relative`), so absolute or
///   root-prefixed-relative legacy paths can't slip through.
fn preflight_manifest(local_root: &Path) -> Result<Vec<u8>> {
  let manifest_path = local_root.join(MANIFEST_FILE_NAME);
  if !manifest_path.exists() {
    bail!(
      "sync_to_s3: refusing to upload — `MANIFEST.json` is missing at \
       {manifest_path:?}. Sync requires a fully-baked local index."
    );
  }
  let bytes = std::fs::read(&manifest_path)
    .with_context(|| format!("sync_to_s3: reading {manifest_path:?}"))?;
  let manifest: Manifest = serde_json::from_slice(&bytes)
    .with_context(|| format!("sync_to_s3: parsing {manifest_path:?}"))?;
  if manifest.version != MANIFEST_LATEST_VERSION {
    bail!(
      "sync_to_s3: refusing to upload — local manifest is version {} but the \
       S3 open path requires v{MANIFEST_LATEST_VERSION} (portable / relative \
       keys). Run a local mutable open-then-commit first to upgrade the \
       manifest in place: \
       `let idx = Index::open(opts)?; idx.writer()?.commit()?;` \
       (or any other mutator), then re-sync.",
      manifest.version
    );
  }
  for seg in &manifest.segments {
    seg.paths.validate_v2_relative().with_context(|| {
      format!(
        "sync_to_s3: segment {} has non-portable paths in MANIFEST.json; \
         the S3 open path resolves segment keys against an empty logical \
         root, so absolute or `..`-bearing paths would miss after upload",
        seg.id
      )
    })?;
    // Stage 10c v3 [P2] (Codex review): verify every artifact the
    // manifest references actually exists as a regular file under
    // `local_root`. Without this, a partial-bake (e.g. a missing
    // `.post` file) would let `sync_to_s3` upload whatever IS
    // present and then publish the manifest, surfacing an
    // unservable index. Catching this at preflight (before any
    // network write) keeps the manifest-as-fence guarantee intact.
    let resolved = seg.paths.resolve(local_root);
    for (label, relative_key, abs_path) in [
      ("terms", seg.paths.terms.as_str(), &resolved.terms),
      ("postings", seg.paths.postings.as_str(), &resolved.postings),
      ("docstore", seg.paths.docstore.as_str(), &resolved.docstore),
      ("fast", seg.paths.fast.as_str(), &resolved.fast),
      ("meta", seg.paths.meta.as_str(), &resolved.meta),
    ] {
      // Stage 10c v5 [P2] (Codex review): assert the manifest key
      // is in the **canonical** form the walker would emit. This
      // closes the "manifest references `./seg_X.post` but walker
      // uploads `seg_X.post`" lexical-drift gap.
      if let Err(reason) = validate_canonical_segment_key(relative_key) {
        bail!(
          "sync_to_s3: refusing to upload — segment {} references {label} \
           artifact at relative key {relative_key:?}, which is NOT in the \
           canonical form the sync walker would emit ({reason}). The walker \
           builds keys via `read_dir` + `strip_prefix(local_root)`, which \
           produces no leading `./`, no `..`, no repeated/leading/trailing \
           `/`, no backslashes, and no platform prefixes. Re-emit the \
           manifest with canonical keys before re-syncing.",
          seg.id
        );
      }
      require_regular_file(label, &seg.id, abs_path)?;
      // Stage 10c v4 [P2] (Codex review): also check that the
      // uploader will actually upload this path. A manifest that
      // names an existing-but-skipped path (e.g. `.hidden.post`,
      // `wal.log`, top-level `MANIFEST.json`) would otherwise pass
      // the existence check, get silently skipped during upload,
      // and still publish the manifest — leaving the remote prefix
      // pointing at a key that was never PUT.
      if !is_uploadable_relative_path(Path::new(relative_key)) {
        bail!(
          "sync_to_s3: refusing to upload — segment {} references {label} \
           artifact at relative key {relative_key:?}, which matches the \
           sync walker's skip rules (dot-file, `wal.log`, or top-level \
           `MANIFEST.json`). The local file exists but would NOT be \
           uploaded, so the manifest must NOT name it as a segment artifact.",
          seg.id
        );
      }
    }
  }
  Ok(bytes)
}

fn require_regular_file(label: &str, seg_id: &str, path: &Path) -> Result<()> {
  let metadata = std::fs::metadata(path).map_err(|e| {
    anyhow!(
      "sync_to_s3: refusing to upload — segment {seg_id} references {label} \
       artifact {path:?} but it cannot be stat'd: {e}. The manifest claims \
       this file exists; sync requires a fully-baked local index."
    )
  })?;
  if !metadata.is_file() {
    bail!(
      "sync_to_s3: refusing to upload — segment {seg_id} references {label} \
       artifact {path:?} but it is not a regular file."
    );
  }
  Ok(())
}

/// Minimal recursive directory walk. Avoids the `walkdir` crate dep
/// for one usage.
fn walkdir(root: &Path) -> Result<Vec<PathBuf>> {
  let mut out = Vec::new();
  walk(root, &mut out)?;
  Ok(out)
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
  for entry in std::fs::read_dir(dir).with_context(|| format!("walkdir: read_dir({dir:?})"))? {
    let entry = entry.with_context(|| format!("walkdir: entry under {dir:?}"))?;
    let path = entry.path();
    let metadata = entry.metadata()?;
    if metadata.is_dir() {
      walk(&path, out)?;
    } else if metadata.is_file() {
      out.push(path);
    }
  }
  Ok(())
}
