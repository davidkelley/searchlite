use std::collections::{BTreeMap, HashMap};
use std::path::{Component, Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::analysis::analyzer::{
  Analyzer, AnalyzerDef, AnalyzerRegistry, EdgeNgramConfig, TokenFilterDef,
};
use crate::storage::Storage;
use crate::util::doc_id::validate_doc_id;
use crate::util::write_key::WriteKeyMeta;

/// Stage 9a: latest supported manifest schema version. v2 records
/// segment paths as **relative-to-index-root keys** (no absolute
/// paths, no `..` components) so an index can be physically moved to
/// a new root and reopen unchanged. v1 (legacy) recorded absolute
/// filesystem paths; reading a v1 manifest still works (absolute
/// paths are accepted as-is on read), but commits always emit v2.
pub const MANIFEST_LATEST_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
  pub version: u32,
  pub uuid: Uuid,
  pub segments: Vec<SegmentMeta>,
  pub committed_at: String,
  pub schema: Schema,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub write_key: Option<WriteKeyMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentMeta {
  pub id: String,
  pub generation: u32,
  pub paths: SegmentPaths,
  pub doc_count: u32,
  pub max_doc_id: u32,
  pub blockmax: bool,
  #[serde(default)]
  pub deleted_docs: Vec<u32>,
  pub avg_field_lengths: HashMap<String, f32>,
  /// Stage 9b/9c: SHA-256 content hashes per segment artifact (`terms`,
  /// `postings`, `docstore`, `fast`, `meta`, plus `vector_{field}_bin`
  /// and `vector_{field}_hnsw` under `--features vectors`). Stored as
  /// 64-char lowercase hex. `BTreeMap` for deterministic JSON output.
  ///
  /// Stage 9c made this the sole integrity field. Manifests written
  /// before Stage 9b had a CRC32 `checksums` map and an empty
  /// `content_hashes`; deserializing such a manifest into the current
  /// struct shape silently drops the legacy `checksums` map and
  /// surfaces an empty `content_hashes`. `verify_checksums` rejects
  /// segments with empty `content_hashes` because there is no longer
  /// any fallback to fall back to — pre-9b indexes must be rebuilt to
  /// open under current code. Stage 9b+ writers always populate every
  /// expected artifact's hash; a non-empty map missing any expected
  /// artifact is treated as corruption (see `verify_checksums` in
  /// segment.rs).
  #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
  pub content_hashes: BTreeMap<String, String>,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub write_binding_b64: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentPaths {
  pub terms: String,
  pub postings: String,
  pub docstore: String,
  pub fast: String,
  pub meta: String,
  #[cfg(feature = "vectors")]
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub vector_dir: Option<String>,
}

/// Resolved (root-joined, absolute) form of [`SegmentPaths`]. Produced
/// by [`SegmentPaths::resolve`] at every read/write call site. The
/// underlying `String` keys in `SegmentPaths` are stored
/// relative-to-root for v2 manifests so an index can be relocated.
#[derive(Debug, Clone)]
pub struct ResolvedSegmentPaths {
  pub terms: PathBuf,
  pub postings: PathBuf,
  pub docstore: PathBuf,
  pub fast: PathBuf,
  pub meta: PathBuf,
  #[cfg(feature = "vectors")]
  pub vector_dir: Option<PathBuf>,
}

impl SegmentPaths {
  /// Resolve every key in this `SegmentPaths` against `root`. Absolute
  /// keys (legacy v1 manifests) are returned as-is; relative keys
  /// (v2) are joined under `root`. Always succeeds — validation of
  /// the v2-relative invariant is a separate concern handled by
  /// [`SegmentPaths::validate_v2_relative`].
  pub fn resolve(&self, root: &Path) -> ResolvedSegmentPaths {
    ResolvedSegmentPaths {
      terms: resolve_segment_path(root, &self.terms),
      postings: resolve_segment_path(root, &self.postings),
      docstore: resolve_segment_path(root, &self.docstore),
      fast: resolve_segment_path(root, &self.fast),
      meta: resolve_segment_path(root, &self.meta),
      #[cfg(feature = "vectors")]
      vector_dir: self
        .vector_dir
        .as_deref()
        .map(|d| resolve_segment_path(root, d)),
    }
  }

  /// Validate the v2 invariant: every key in this `SegmentPaths` is
  /// **relative** (no absolute paths) and contains **no `..`
  /// components** (no parent-traversal escape from the index root).
  /// Empty keys are also rejected as malformed.
  pub fn validate_v2_relative(&self) -> Result<()> {
    let check = |label: &str, s: &str| -> Result<()> {
      if s.is_empty() {
        bail!("v2 manifest: {label} segment path is empty");
      }
      let p = Path::new(s);
      if p.is_absolute() {
        bail!("v2 manifest: {label} segment path is absolute: {s:?}");
      }
      if p.components().any(|c| matches!(c, Component::ParentDir)) {
        bail!("v2 manifest: {label} segment path contains `..` component: {s:?}");
      }
      Ok(())
    };
    check("terms", &self.terms)?;
    check("postings", &self.postings)?;
    check("docstore", &self.docstore)?;
    check("fast", &self.fast)?;
    check("meta", &self.meta)?;
    #[cfg(feature = "vectors")]
    if let Some(dir) = self.vector_dir.as_deref() {
      check("vector_dir", dir)?;
    }
    Ok(())
  }

  /// Stage 9a [P2] (Codex review): validate the v1 *legacy* invariant
  /// — every key is non-empty and contains no `..` component. v1
  /// manifests produced by the pre-Stage-9 writer used
  /// `root.join(filename)`; that produced **absolute** paths when
  /// `IndexOptions.path` was absolute, and **root-prefixed relative**
  /// paths when it was relative (e.g. `idx/seg_X.terms` for
  /// `--index idx`). Both shapes are legitimate v1; only `..`
  /// traversal and empty paths are rejected as malformed.
  ///
  /// Stage 9a v4 [P2] update: previously rejected non-absolute v1 paths
  /// outright; that broke relative-root indexes created by older CLI
  /// invocations. The `..` check still prevents resolution from
  /// escaping the index root.
  pub fn validate_v1_legacy(&self) -> Result<()> {
    let check = |label: &str, s: &str| -> Result<()> {
      if s.is_empty() {
        bail!("v1 manifest: {label} segment path is empty");
      }
      let p = Path::new(s);
      if p.components().any(|c| matches!(c, Component::ParentDir)) {
        bail!("v1 manifest: {label} segment path contains `..` component: {s:?}");
      }
      Ok(())
    };
    check("terms", &self.terms)?;
    check("postings", &self.postings)?;
    check("docstore", &self.docstore)?;
    check("fast", &self.fast)?;
    check("meta", &self.meta)?;
    #[cfg(feature = "vectors")]
    if let Some(dir) = self.vector_dir.as_deref() {
      check("vector_dir", dir)?;
    }
    Ok(())
  }

  /// Stage 9a [P2] (Codex review): convert this `SegmentPaths` to its
  /// portable v2 form by stripping the index `root` prefix.
  ///
  /// Three shapes are handled, mirroring [`resolve_segment_path`]:
  ///
  /// * **Absolute key** — strip the (absolute) root prefix. Errors if
  ///   the absolute key is NOT under `root` (the segment file lives
  ///   elsewhere and an in-place upgrade would silently break reads).
  /// * **Relative key starting with `root`** — strip the root prefix.
  ///   This is the v1 relative-root case where the old writer
  ///   recorded e.g. `idx/seg_X.terms` for an index opened with a
  ///   relative `--index idx`.
  /// * **Bare relative key** — already in v2 shape; passed through
  ///   unchanged.
  ///
  /// Stage 9a v4 [P2] update: extended to handle root-prefixed
  /// relative v1 paths in addition to absolute paths. Previously this
  /// only stripped absolute prefixes, so a legacy relative-root
  /// manifest would round-trip its old prefix into the v2 form (e.g.
  /// `idx/idx/seg_X.terms` after one resolve cycle).
  pub fn relativize_under(&mut self, root: &Path) -> Result<()> {
    // Stage 9a v6 [P3] (Codex review): the absolute-form-of-root
    // step needs the process CWD when `root` is relative. We resolve
    // CWD once here and forward to `relativize_under_with_cwd` so
    // tests can exercise the path-candidate logic without mutating
    // the process-global CWD (which would race against parallel
    // tests).
    let cwd = if root.is_absolute() {
      None
    } else {
      Some(std::env::current_dir().with_context(|| {
        format!("resolving absolute form of relative index root {root:?} for upgrade")
      })?)
    };
    self.relativize_under_with_cwd(root, cwd.as_deref())
  }

  /// Stage 9a v6 [P3] (Codex review): testable variant of
  /// `relativize_under` that takes an explicit `cwd` for resolving
  /// relative roots, instead of pulling it from the process via
  /// `std::env::current_dir`. Production callers go through
  /// `relativize_under`; tests use this directly to avoid mutating
  /// process CWD under the parallel test harness.
  pub fn relativize_under_with_cwd(&mut self, root: &Path, cwd: Option<&Path>) -> Result<()> {
    // Build the set of candidate root forms an absolute v1 segment
    // key might have been recorded against. Path comparison is
    // lexical, so we need every form the on-disk path could
    // plausibly take:
    //
    // * Literal `root` (handles absolute root + absolute key, or
    //   relative root + relative key in the legacy relative-root
    //   case).
    // * Absolute form of `root` (cwd + root) — needed when the
    //   user opens with a relative root but the v1 manifest stored
    //   absolute paths.
    // * Canonical form of `root` (symlinks resolved) — needed on
    //   macOS where `/var/folders/...` is a symlink to
    //   `/private/var/folders/...` and the manifest may have stored
    //   either form depending on the writer's environment.
    let mut root_candidates: Vec<PathBuf> = Vec::with_capacity(4);
    root_candidates.push(root.to_path_buf());
    let absolute_root: Option<PathBuf> = if root.is_absolute() {
      None
    } else {
      let cwd = cwd.ok_or_else(|| {
        anyhow!("relativize_under_with_cwd: cwd is required when root {root:?} is relative")
      })?;
      Some(cwd.join(root))
    };
    if let Some(abs) = absolute_root.as_deref() {
      if !root_candidates.iter().any(|r| r == abs) {
        root_candidates.push(abs.to_path_buf());
      }
    }
    if let Ok(canon) = std::fs::canonicalize(root) {
      if !root_candidates.iter().any(|r| r == &canon) {
        root_candidates.push(canon);
      }
    }
    if let Some(abs) = absolute_root.as_deref() {
      if let Ok(canon) = std::fs::canonicalize(abs) {
        if !root_candidates.iter().any(|r| r == &canon) {
          root_candidates.push(canon);
        }
      }
    }
    fn relativize(
      label: &str,
      root: &Path,
      candidates: &[PathBuf],
      key: &mut String,
    ) -> Result<()> {
      if key.is_empty() {
        bail!("cannot relativize empty {label} path");
      }
      let p = Path::new(key.as_str());
      if p.is_absolute() {
        // Try every candidate root form against both the literal
        // key and its canonical (symlink-resolved) form. If any
        // combination succeeds, use the resulting bare key.
        let key_canon: Option<PathBuf> = std::fs::canonicalize(p).ok();
        for cand in candidates {
          if let Ok(stripped) = p.strip_prefix(cand) {
            *key = stripped.to_string_lossy().into_owned();
            return Ok(());
          }
          if let Some(kc) = key_canon.as_deref() {
            if let Ok(stripped) = kc.strip_prefix(cand) {
              *key = stripped.to_string_lossy().into_owned();
              return Ok(());
            }
          }
        }
        bail!(
          "cannot relativize {label} path {key:?}: not under index root {root:?} \
           (tried candidates {candidates:?})"
        );
      }
      if let Ok(stripped) = p.strip_prefix(root) {
        *key = stripped.to_string_lossy().into_owned();
        return Ok(());
      }
      // Already a bare relative key (v2 shape, or a v1 manifest from
      // an empty/CWD root). Leave as-is; post-upgrade
      // `validate_v2_relative` will catch any disallowed shape.
      Ok(())
    }
    relativize("terms", root, &root_candidates, &mut self.terms)?;
    relativize("postings", root, &root_candidates, &mut self.postings)?;
    relativize("docstore", root, &root_candidates, &mut self.docstore)?;
    relativize("fast", root, &root_candidates, &mut self.fast)?;
    relativize("meta", root, &root_candidates, &mut self.meta)?;
    #[cfg(feature = "vectors")]
    if let Some(dir) = self.vector_dir.as_mut() {
      relativize("vector_dir", root, &root_candidates, dir)?;
    }
    Ok(())
  }
}

/// Resolve a single segment path key against an index root.
///
/// Three legitimate shapes are recognized — chosen so v1 (legacy) and
/// v2 (Stage 9a) manifests both resolve to the right on-disk path:
///
/// * **Absolute** — passed through unchanged. v1's common shape, where
///   the writer did `root.join(filename)` and `root` was absolute.
/// * **Relative starting with `root`** — passed through unchanged. v1's
///   relative-root shape: when the caller used a relative
///   `IndexOptions.path` like `idx`, the writer recorded
///   `idx/seg_X.terms`, which already includes the root prefix.
///   Without this branch, naive `root.join` would double-prefix to
///   `idx/idx/seg_X.terms`.
/// * **Bare relative** — joined under `root`. v2's standard shape
///   (`seg_X.terms`).
pub fn resolve_segment_path(root: &Path, key: &str) -> PathBuf {
  let p = Path::new(key);
  if p.is_absolute() {
    return p.to_path_buf();
  }
  if p.strip_prefix(root).is_ok() {
    // Legacy v1 relative-root form. The path already includes the
    // root, so use it directly rather than joining a second time.
    return p.to_path_buf();
  }
  root.join(p)
}

impl Manifest {
  pub fn new(schema: Schema) -> Self {
    Self {
      version: MANIFEST_LATEST_VERSION,
      uuid: Uuid::new_v4(),
      segments: Vec::new(),
      committed_at: Utc::now().to_rfc3339(),
      schema,
      write_key: None,
    }
  }

  pub fn load(storage: &dyn Storage, path: &Path) -> Result<Self> {
    let data = storage
      .read_to_end(path)
      .with_context(|| format!("reading manifest at {path:?}"))?;
    let manifest: Manifest =
      serde_json::from_slice(&data).with_context(|| format!("parsing manifest at {path:?}"))?;
    // Stage 9a v4 [P3] (Codex review): assert the manifest version
    // **before** entering the per-segment validation loop. Previously
    // the unsupported-version check lived inside the loop, so a
    // malformed manifest with `version: 0` and an empty segment list
    // would skip the loop and load successfully.
    if manifest.version == 0 || manifest.version > MANIFEST_LATEST_VERSION {
      bail!(
        "manifest at {path:?} has unsupported version {} (supported: 1..={})",
        manifest.version,
        MANIFEST_LATEST_VERSION
      );
    }
    // Stage 9a: validate the per-version path invariant. v2 requires
    // relative-only keys; v1 (legacy) requires non-empty + `..`-free
    // keys (absolute or root-prefixed relative are both legitimate;
    // see `validate_v1_legacy`). A `..`-bearing path could otherwise
    // resolve outside the index root.
    for seg in &manifest.segments {
      match manifest.version {
        v if v >= 2 => seg
          .paths
          .validate_v2_relative()
          .with_context(|| format!("v2 manifest validation failed for segment {}", seg.id))?,
        1 => seg
          .paths
          .validate_v1_legacy()
          .with_context(|| format!("v1 manifest validation failed for segment {}", seg.id))?,
        // Pre-loop check above rejects all other versions.
        _ => unreachable!("version validated before per-segment loop"),
      }
    }
    Ok(manifest)
  }

  pub fn store(&self, storage: &dyn Storage, path: &Path) -> Result<()> {
    let data = self.serialize_for_write()?;
    storage
      .atomic_write(path, &data)
      .with_context(|| format!("writing manifest at {path:?}"))
  }

  /// Stage 9a [P2] (Codex review): in-place upgrade a (possibly v1)
  /// `Manifest` to the latest version under `root`. Strips the root
  /// prefix from any absolute legacy path, then bumps `version` to
  /// `MANIFEST_LATEST_VERSION` and validates the resulting v2
  /// invariant.
  ///
  /// This is the load-bearing migration step for legacy v1 manifests:
  /// without it, a `Writer::commit` against an open v1 index would
  /// re-emit a v1 manifest that mixes absolute legacy paths with
  /// freshly-relative new-segment paths, and the index would never
  /// become portable.
  pub fn upgrade_to_latest(&mut self, root: &Path) -> Result<()> {
    // Stage 9a v5 [P2] (Codex review): assert the supported-version
    // invariant **before** mutating. Recovery and leftover-pending
    // promote parse pending bytes directly and call this method
    // without going through `Manifest::load`, so without an explicit
    // guard a `version: 0` (or `version: 99`) pending file with no
    // segments would skip the per-segment relativize loop and become
    // a published v2 manifest. Mirror `Manifest::load`'s rejection
    // boundaries here so all write/recovery paths share the same
    // version contract.
    if self.version == 0 || self.version > MANIFEST_LATEST_VERSION {
      bail!(
        "refusing to upgrade manifest with unsupported version {} (supported: 1..={})",
        self.version,
        MANIFEST_LATEST_VERSION
      );
    }
    if self.version >= MANIFEST_LATEST_VERSION {
      // Already at latest. Re-validate defensively to catch any
      // in-process mutation that produced a non-portable v2 shape.
      for seg in &self.segments {
        seg.paths.validate_v2_relative().with_context(|| {
          format!(
            "v{} manifest validation failed for segment {}",
            self.version, seg.id
          )
        })?;
      }
      return Ok(());
    }
    for seg in self.segments.iter_mut() {
      seg.paths.relativize_under(root).with_context(|| {
        format!(
          "upgrading legacy manifest segment {} to v{MANIFEST_LATEST_VERSION}",
          seg.id
        )
      })?;
    }
    self.version = MANIFEST_LATEST_VERSION;
    for seg in &self.segments {
      seg.paths.validate_v2_relative().with_context(|| {
        format!(
          "post-upgrade validation failed for segment {} (this is a bug)",
          seg.id
        )
      })?;
    }
    Ok(())
  }

  /// Stage 9a [P2] (Codex review): single serialization path used by
  /// every commit-time write — `Manifest::store`, the pre-fence
  /// pending write, and the recovery promote step. Consolidates
  /// version + relative-key validation so no caller can bypass the
  /// portability invariant.
  ///
  /// Note: this does NOT mutate `version` or paths — call
  /// [`Manifest::upgrade_to_latest`] first if you have a legacy
  /// in-memory manifest. This separation lets callers control the
  /// upgrade timing (the writer upgrades before staging so the
  /// pending file is itself portable).
  pub fn serialize_for_write(&self) -> Result<Vec<u8>> {
    if self.version > MANIFEST_LATEST_VERSION {
      bail!(
        "refusing to write manifest with unsupported version {} (latest supported is {})",
        self.version,
        MANIFEST_LATEST_VERSION
      );
    }
    if self.version < MANIFEST_LATEST_VERSION {
      bail!(
        "refusing to write legacy v{} manifest; call Manifest::upgrade_to_latest before writing",
        self.version
      );
    }
    for seg in &self.segments {
      seg.paths.validate_v2_relative().with_context(|| {
        format!(
          "refusing to write v{MANIFEST_LATEST_VERSION} manifest with non-portable \
           path for segment {}",
          seg.id
        )
      })?;
    }
    Ok(serde_json::to_vec_pretty(self)?)
  }

  pub fn manifest_path(root: &Path) -> PathBuf {
    root.join("MANIFEST.json")
  }

  /// Path for a staged manifest that has been written but not yet promoted
  /// to the live `MANIFEST.json`.
  ///
  /// `Writer::commit` stages the new manifest here *before* appending the
  /// WAL commit record (the durability fence). On a crash between the WAL
  /// commit fence and the live manifest publish, [`Index::open`] reconciles
  /// the two by promoting this file to `MANIFEST.json` (see BUG-018).
  pub fn manifest_pending_path(root: &Path) -> PathBuf {
    root.join("MANIFEST.json.pending")
  }
}

#[derive(Debug, Clone, Default)]
pub struct Schema {
  pub doc_id_field: String,
  pub analyzers: Vec<AnalyzerDef>,
  pub text_fields: Vec<TextField>,
  pub keyword_fields: Vec<KeywordField>,
  pub numeric_fields: Vec<NumericField>,
  pub nested_fields: Vec<NestedField>,
  #[cfg(feature = "vectors")]
  pub vector_fields: Vec<VectorField>,
}

impl Serialize for Schema {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    let json_value = super::json_schema::schema_to_json_schema(self);
    json_value.serialize(serializer)
  }
}

impl<'de> Deserialize<'de> for Schema {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: serde::Deserializer<'de>,
  {
    let value = serde_json::Value::deserialize(deserializer)?;
    super::json_schema::parse_json_schema(&value).map_err(serde::de::Error::custom)
  }
}

pub fn default_doc_id_field() -> String {
  "_id".to_string()
}

impl Schema {
  pub fn default_text_body() -> Self {
    Self {
      doc_id_field: default_doc_id_field(),
      analyzers: Vec::new(),
      text_fields: vec![TextField {
        name: "body".to_string(),
        analyzer: "default".to_string(),
        search_analyzer: None,
        stored: true,
        indexed: true,
        nullable: false,
        search_as_you_type: None,
      }],
      keyword_fields: Vec::new(),
      numeric_fields: Vec::new(),
      nested_fields: Vec::new(),
      #[cfg(feature = "vectors")]
      vector_fields: Vec::new(),
    }
  }

  pub fn is_indexed_field(&self, field: &str) -> bool {
    self
      .resolved_fields()
      .iter()
      .any(|f| f.path == field && f.indexed)
  }

  pub fn is_stored_field(&self, field: &str) -> bool {
    self
      .resolved_fields()
      .iter()
      .any(|f| f.path == field && f.stored)
  }

  pub fn validate_config(&self) -> anyhow::Result<()> {
    if self.doc_id_field.contains('.') {
      anyhow::bail!("doc_id_field `{}` cannot be nested", self.doc_id_field);
    }
    self.build_analyzers()?;
    if self
      .resolved_fields()
      .iter()
      .any(|f| f.path == self.doc_id_field)
    {
      anyhow::bail!(
        "doc_id_field `{}` must not overlap with other schema fields",
        self.doc_id_field
      );
    }
    #[cfg(feature = "vectors")]
    for vf in self.vector_fields.iter() {
      if vf.dim == 0 {
        anyhow::bail!("vector field `{}` must have dim > 0", vf.name);
      }
      if self.resolved_fields().iter().any(|f| f.path == vf.name) {
        anyhow::bail!("vector field `{}` conflicts with another field", vf.name);
      }
      if let Some(hnsw) = &vf.hnsw {
        if hnsw.m == 0 {
          anyhow::bail!("vector field `{}` must set hnsw.m > 0", vf.name);
        }
        if hnsw.ef_construction == 0 {
          anyhow::bail!(
            "vector field `{}` must set hnsw.ef_construction > 0",
            vf.name
          );
        }
      }
    }
    Ok(())
  }

  pub fn build_analyzers(&self) -> anyhow::Result<SchemaAnalyzers> {
    let mut defs = self.analyzers.clone();
    let mut field_refs = Vec::new();
    let find_def = |name: &str, defs: &[AnalyzerDef]| -> Option<AnalyzerDef> {
      if name == "default" {
        return Some(AnalyzerDef {
          name: "default".to_string(),
          tokenizer: "default".to_string(),
          filters: Vec::new(),
        });
      }
      defs.iter().find(|d| d.name == name).cloned()
    };
    for (path, field) in self.text_field_map().into_iter() {
      let base_analyzer = field.analyzer.clone();
      let search_name = field
        .search_analyzer
        .clone()
        .unwrap_or_else(|| field.analyzer.clone());
      let index_analyzer = if let Some(cfg) = &field.search_as_you_type {
        let generated = format!("{}__saty_{}", base_analyzer, path.replace('.', "_"));
        if defs.iter().all(|d| d.name != generated) {
          let base_def = find_def(&base_analyzer, &defs).ok_or_else(|| {
            anyhow::anyhow!("field `{path}` references unknown analyzer `{base_analyzer}`")
          })?;
          let mut filters = base_def.filters.clone();
          filters.push(TokenFilterDef::EdgeNgram(EdgeNgramConfig {
            min: cfg.min_gram,
            max: cfg.max_gram,
          }));
          defs.push(AnalyzerDef {
            name: generated.clone(),
            tokenizer: base_def.tokenizer,
            filters,
          });
        }
        generated
      } else {
        base_analyzer.clone()
      };
      field_refs.push((
        path.clone(),
        FieldAnalyzerRefs {
          analyzer: index_analyzer,
          search_analyzer: search_name,
        },
      ));
    }
    let registry = AnalyzerRegistry::from_defs(&defs)?;
    let mut field_map = HashMap::new();
    for (path, refs) in field_refs.into_iter() {
      if registry.get(&refs.analyzer).is_none() {
        anyhow::bail!(
          "field `{path}` references unknown analyzer `{}`",
          refs.analyzer
        );
      }
      if registry.get(&refs.search_analyzer).is_none() {
        anyhow::bail!(
          "field `{path}` references unknown search analyzer `{}`",
          refs.search_analyzer
        );
      }
      if field_map.insert(path.clone(), refs).is_some() {
        anyhow::bail!("duplicate field `{path}` in analyzer map");
      }
    }
    Ok(SchemaAnalyzers {
      registry,
      field_map,
    })
  }

  fn text_field_map(&self) -> Vec<(String, &TextField)> {
    let mut out = Vec::new();
    for field in self.text_fields.iter() {
      out.push((field.name.clone(), field));
    }
    for nested in self.nested_fields.iter() {
      collect_nested_text_fields(nested, None, &mut out);
    }
    out
  }

  pub fn fast_fields(&self) -> Vec<String> {
    self
      .resolved_fields()
      .into_iter()
      .filter(|f| f.fast)
      .map(|f| f.path)
      .collect()
  }

  pub fn field_kind(&self, field: &str) -> FieldKind {
    self
      .resolved_fields()
      .into_iter()
      .find(|f| f.path == field)
      .map(|f| f.kind)
      .unwrap_or(FieldKind::Unknown)
  }

  pub fn field_meta(&self, field: &str) -> Option<ResolvedField> {
    self.resolved_fields().into_iter().find(|f| f.path == field)
  }

  pub fn resolved_fields(&self) -> Vec<ResolvedField> {
    let mut fields = Vec::new();
    for f in self.text_fields.iter() {
      fields.push(ResolvedField {
        path: f.name.clone(),
        kind: FieldKind::Text,
        indexed: f.indexed,
        stored: f.stored,
        fast: false,
        numeric_i64: None,
        nullable: f.nullable,
      });
    }
    for f in self.keyword_fields.iter() {
      fields.push(ResolvedField {
        path: f.name.clone(),
        kind: FieldKind::Keyword,
        indexed: f.indexed,
        stored: f.stored,
        fast: f.fast,
        numeric_i64: None,
        nullable: f.nullable,
      });
    }
    for f in self.numeric_fields.iter() {
      fields.push(ResolvedField {
        path: f.name.clone(),
        kind: FieldKind::Numeric,
        indexed: true,
        stored: f.stored,
        fast: f.fast,
        numeric_i64: Some(f.i64),
        nullable: f.nullable,
      });
    }
    for nested in self.nested_fields.iter() {
      nested.collect_fields(None, &mut fields);
    }
    fields
  }

  pub fn doc_id_field(&self) -> &str {
    &self.doc_id_field
  }

  pub fn validate_document(&self, doc: &crate::api::types::Document) -> anyhow::Result<()> {
    let Some(doc_id) = doc.fields.get(self.doc_id_field()).and_then(|v| v.as_str()) else {
      anyhow::bail!(
        "missing or empty required document id field `{}`",
        self.doc_id_field()
      );
    };
    validate_doc_id(doc_id)?;
    for (name, value) in doc.fields.iter() {
      if let Some(nested) = self.nested_fields.iter().find(|n| n.name == *name) {
        nested
          .validate(value)
          .with_context(|| format!("validating nested field {name}"))?;
        continue;
      }
      if let Some(meta) = self.field_meta(name) {
        validate_field_value(&meta, value)?;
      }
    }
    Ok(())
  }

  /// Reject documents that omit any non-nullable top-level field declared in
  /// the schema. Per `docs/schema.md`, every schema field is required unless
  /// it is explicitly marked nullable.
  ///
  /// Separated from `validate_document` because round-tripping a document
  /// through the docstore is lossy for several legitimate schema shapes
  /// (empty arrays and fields whose values all serialize away via
  /// `stored_nested_value`). The writer therefore invokes this check only
  /// on user-supplied documents at the ingest boundary (`add_document` /
  /// `add_documents_batch`). Reconstruction-based flows (`compact`,
  /// `merge_segments`, `apply_patch`) continue to use `validate_document`
  /// alone so they don't reject documents that were valid when ingested.
  ///
  /// Vector fields do not carry a nullability marker in the current schema
  /// model and are treated as implicitly optional by `collect_vector_value`,
  /// so they are excluded from this presence check by design.
  pub fn check_required_fields_present(
    &self,
    doc: &crate::api::types::Document,
  ) -> anyhow::Result<()> {
    let doc_id_name = self.doc_id_field();
    for f in self.text_fields.iter() {
      if f.nullable || f.name == doc_id_name {
        continue;
      }
      if !doc.fields.contains_key(&f.name) {
        anyhow::bail!("missing required field `{}`", f.name);
      }
    }
    for f in self.keyword_fields.iter() {
      if f.nullable || f.name == doc_id_name {
        continue;
      }
      if !doc.fields.contains_key(&f.name) {
        anyhow::bail!("missing required field `{}`", f.name);
      }
    }
    for f in self.numeric_fields.iter() {
      if f.nullable || f.name == doc_id_name {
        continue;
      }
      if !doc.fields.contains_key(&f.name) {
        anyhow::bail!("missing required field `{}`", f.name);
      }
    }
    for f in self.nested_fields.iter() {
      if f.nullable || f.name == doc_id_name {
        continue;
      }
      if !doc.fields.contains_key(&f.name) {
        anyhow::bail!("missing required field `{}`", f.name);
      }
    }
    Ok(())
  }

  #[cfg(feature = "vectors")]
  pub fn vector_field(&self, field: &str) -> Option<VectorField> {
    self.vector_fields.iter().find(|f| f.name == field).cloned()
  }
}

fn validate_field_value(meta: &ResolvedField, value: &serde_json::Value) -> anyhow::Result<()> {
  if value.is_null() {
    if meta.nullable {
      return Ok(());
    }
    anyhow::bail!("field `{}` cannot be null", meta.path);
  }
  match meta.kind {
    FieldKind::Text | FieldKind::Keyword => {
      if !is_string_or_string_array(value) {
        anyhow::bail!("field `{}` must be a string or array of strings", meta.path);
      }
    }
    FieldKind::Numeric => {
      if meta.numeric_i64.unwrap_or(false) {
        if !is_i64_or_array(value) {
          anyhow::bail!("field `{}` must be a number or array of numbers", meta.path);
        }
      } else if !is_f64_or_array(value) {
        anyhow::bail!("field `{}` must be a number or array of numbers", meta.path);
      }
    }
    FieldKind::Unknown => {}
  }
  Ok(())
}

fn is_string_or_string_array(value: &serde_json::Value) -> bool {
  match value {
    serde_json::Value::String(_) => true,
    serde_json::Value::Array(arr) => arr.iter().all(|v| v.as_str().is_some()),
    _ => false,
  }
}

fn is_i64_or_array(value: &serde_json::Value) -> bool {
  match value {
    serde_json::Value::Number(n) => n.as_i64().is_some(),
    serde_json::Value::Array(arr) => arr.iter().all(|v| v.as_i64().is_some()),
    _ => false,
  }
}

fn is_f64_or_array(value: &serde_json::Value) -> bool {
  match value {
    serde_json::Value::Number(n) => n.as_f64().is_some(),
    serde_json::Value::Array(arr) => arr.iter().all(|v| v.as_f64().is_some()),
    _ => false,
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::tempdir;

  #[test]
  fn doc_id_field_defaults_and_validates_presence() {
    let schema = Schema::default_text_body();
    assert_eq!(schema.doc_id_field(), "_id");
    let doc = crate::api::types::Document::default();
    let err = schema.validate_document(&doc).unwrap_err();
    assert!(err
      .to_string()
      .contains("missing or empty required document id field"));
  }

  #[test]
  fn doc_id_field_rejects_empty() {
    let schema = Schema::default_text_body();
    for value in ["", "   "] {
      let doc = crate::api::types::Document {
        fields: [("_id".into(), serde_json::json!(value))]
          .into_iter()
          .collect(),
      };
      let err = schema.validate_document(&doc).unwrap_err();
      assert!(err
        .to_string()
        .contains("document id cannot be empty or whitespace"));
    }
  }

  #[test]
  fn persists_manifest_and_schema_helpers() {
    let dir = tempdir().unwrap();
    let storage = crate::storage::FsStorage::new(dir.path().to_path_buf());
    let schema = Schema {
      doc_id_field: "pk".into(),
      analyzers: Vec::new(),
      text_fields: vec![TextField {
        name: "body".into(),
        analyzer: "default".into(),
        search_analyzer: None,
        stored: true,
        indexed: true,
        nullable: false,
        search_as_you_type: None,
      }],
      keyword_fields: vec![KeywordField {
        name: "tag".into(),
        stored: true,
        indexed: true,
        fast: true,
        nullable: false,
      }],
      numeric_fields: vec![NumericField {
        name: "year".into(),
        i64: true,
        fast: true,
        stored: true,
        nullable: false,
      }],
      nested_fields: vec![NestedField {
        name: "comment".into(),
        fields: vec![NestedProperty::Keyword(KeywordField {
          name: "author".into(),
          stored: true,
          indexed: true,
          fast: true,
          nullable: false,
        })],
        nullable: false,
      }],
      #[cfg(feature = "vectors")]
      vector_fields: Vec::new(),
    };
    let manifest = Manifest::new(schema.clone());
    let path = Manifest::manifest_path(dir.path());
    manifest.store(&storage, &path).unwrap();
    let loaded = Manifest::load(&storage, &path).unwrap();
    assert!(loaded.schema.is_indexed_field("body"));
    assert!(loaded.schema.is_stored_field("year"));
    let mut fast_fields = loaded.schema.fast_fields();
    fast_fields.sort();
    assert_eq!(
      fast_fields,
      vec![
        "comment.author".to_string(),
        "tag".to_string(),
        "year".to_string()
      ]
    );
    assert!(matches!(
      loaded.schema.field_kind("year"),
      FieldKind::Numeric
    ));
    assert!(matches!(
      loaded.schema.field_kind("comment.author"),
      FieldKind::Keyword
    ));
  }

  #[test]
  fn nested_nullable_fields_are_explicit() {
    let base_schema = Schema {
      doc_id_field: default_doc_id_field(),
      analyzers: Vec::new(),
      text_fields: Vec::new(),
      keyword_fields: Vec::new(),
      numeric_fields: Vec::new(),
      nested_fields: vec![NestedField {
        name: "game".into(),
        fields: vec![
          NestedProperty::Keyword(KeywordField {
            name: "name".into(),
            stored: true,
            indexed: true,
            fast: true,
            nullable: false,
          }),
          NestedProperty::Keyword(KeywordField {
            name: "franchise".into(),
            stored: true,
            indexed: false,
            fast: true,
            nullable: true,
          }),
        ],
        nullable: false,
      }],
      #[cfg(feature = "vectors")]
      vector_fields: Vec::new(),
    };

    let ok = crate::api::types::Document {
      fields: [
        ("_id".into(), serde_json::json!("game-1")),
        (
          "game".into(),
          serde_json::json!({ "name": "Skyline of Void", "franchise": null }),
        ),
      ]
      .into_iter()
      .collect(),
    };
    base_schema.validate_document(&ok).expect("nullable ok");

    let bad_null = crate::api::types::Document {
      fields: [
        ("_id".into(), serde_json::json!("game-2")),
        (
          "game".into(),
          serde_json::json!({ "name": null, "franchise": "Series" }),
        ),
      ]
      .into_iter()
      .collect(),
    };
    assert!(base_schema.validate_document(&bad_null).is_err());

    let nullable_game_schema = Schema {
      nested_fields: vec![NestedField {
        name: "game".into(),
        fields: vec![],
        nullable: true,
      }],
      ..base_schema.clone()
    };
    let null_game = crate::api::types::Document {
      fields: [
        ("_id".into(), serde_json::json!("game-3")),
        ("game".into(), serde_json::Value::Null),
      ]
      .into_iter()
      .collect(),
    };
    nullable_game_schema
      .validate_document(&null_game)
      .expect("nullable container ok");
  }

  #[test]
  fn nested_fields_require_present_non_nullable_properties() {
    let schema = Schema {
      doc_id_field: default_doc_id_field(),
      analyzers: Vec::new(),
      text_fields: Vec::new(),
      keyword_fields: Vec::new(),
      numeric_fields: Vec::new(),
      nested_fields: vec![NestedField {
        name: "comment".into(),
        fields: vec![
          NestedProperty::Keyword(KeywordField {
            name: "author".into(),
            stored: true,
            indexed: true,
            fast: false,
            nullable: false,
          }),
          NestedProperty::Numeric(NumericField {
            name: "score".into(),
            i64: true,
            fast: false,
            stored: false,
            nullable: true,
          }),
        ],
        nullable: false,
      }],
      #[cfg(feature = "vectors")]
      vector_fields: Vec::new(),
    };

    let missing = crate::api::types::Document {
      fields: [
        ("_id".into(), serde_json::json!("c1")),
        ("comment".into(), serde_json::json!({ "score": 10 })),
      ]
      .into_iter()
      .collect(),
    };
    let err = schema.validate_document(&missing).unwrap_err();
    let messages: Vec<String> = err.chain().map(|e| e.to_string()).collect();
    assert!(
      messages
        .iter()
        .any(|m| m.contains("missing required nested field comment.author")),
      "unexpected error chain: {messages:?}"
    );

    let complete = crate::api::types::Document {
      fields: [
        ("_id".into(), serde_json::json!("c2")),
        (
          "comment".into(),
          serde_json::json!({ "author": "Ada", "score": 10 }),
        ),
      ]
      .into_iter()
      .collect(),
    };
    schema
      .validate_document(&complete)
      .expect("complete nested object passes validation");
  }

  #[test]
  fn top_level_fields_enforce_nullability_and_type() {
    let schema = Schema {
      doc_id_field: default_doc_id_field(),
      analyzers: Vec::new(),
      text_fields: vec![TextField {
        name: "body".into(),
        analyzer: "default".into(),
        search_analyzer: None,
        stored: true,
        indexed: true,
        nullable: false,
        search_as_you_type: None,
      }],
      keyword_fields: Vec::new(),
      numeric_fields: vec![NumericField {
        name: "score".into(),
        i64: false,
        fast: false,
        stored: false,
        nullable: false,
      }],
      nested_fields: Vec::new(),
      #[cfg(feature = "vectors")]
      vector_fields: Vec::new(),
    };
    let null_doc = crate::api::types::Document {
      fields: [
        ("_id".into(), serde_json::json!("1")),
        ("body".into(), serde_json::Value::Null),
      ]
      .into_iter()
      .collect(),
    };
    let err = schema.validate_document(&null_doc).unwrap_err();
    assert!(err.to_string().contains("cannot be null"));

    let bad_type = crate::api::types::Document {
      fields: [
        ("_id".into(), serde_json::json!("2")),
        ("body".into(), serde_json::json!("ok")),
        ("score".into(), serde_json::json!("not a number")),
      ]
      .into_iter()
      .collect(),
    };
    let err = schema.validate_document(&bad_type).unwrap_err();
    assert!(err.to_string().contains("must be a number"));
  }

  #[test]
  fn nested_keyword_rejects_non_string_array_elements() {
    let schema = Schema {
      doc_id_field: default_doc_id_field(),
      analyzers: Vec::new(),
      text_fields: Vec::new(),
      keyword_fields: Vec::new(),
      numeric_fields: Vec::new(),
      nested_fields: vec![NestedField {
        name: "tags".into(),
        fields: vec![NestedProperty::Keyword(KeywordField {
          name: "value".into(),
          stored: true,
          indexed: true,
          fast: false,
          nullable: false,
        })],
        nullable: false,
      }],
      #[cfg(feature = "vectors")]
      vector_fields: Vec::new(),
    };

    // Array of numbers must be rejected for a keyword field.
    let bad_numbers = crate::api::types::Document {
      fields: [
        ("_id".into(), serde_json::json!("d1")),
        ("tags".into(), serde_json::json!({ "value": [1, 2, 3] })),
      ]
      .into_iter()
      .collect(),
    };
    let err = schema.validate_document(&bad_numbers).unwrap_err();
    let messages: Vec<String> = err.chain().map(|e| e.to_string()).collect();
    assert!(
      messages.iter().any(|m| m.contains("array of strings")),
      "unexpected error chain: {messages:?}"
    );
    // Error message should include the fully-qualified nested path so
    // sibling nested objects sharing property names stay distinguishable.
    assert!(
      messages.iter().any(|m| m.contains("tags.value")),
      "error chain should include the fully-qualified path: {messages:?}"
    );

    // Mixed string/non-string elements must also be rejected.
    let mixed = crate::api::types::Document {
      fields: [
        ("_id".into(), serde_json::json!("d2")),
        (
          "tags".into(),
          serde_json::json!({ "value": ["ok", 1, true] }),
        ),
      ]
      .into_iter()
      .collect(),
    };
    assert!(schema.validate_document(&mixed).is_err());

    // A pure array of strings continues to be accepted.
    let good = crate::api::types::Document {
      fields: [
        ("_id".into(), serde_json::json!("d3")),
        ("tags".into(), serde_json::json!({ "value": ["a", "b"] })),
      ]
      .into_iter()
      .collect(),
    };
    schema.validate_document(&good).expect("string array ok");
  }

  #[test]
  fn nested_text_rejects_non_string_array_elements() {
    let schema = Schema {
      doc_id_field: default_doc_id_field(),
      analyzers: Vec::new(),
      text_fields: Vec::new(),
      keyword_fields: Vec::new(),
      numeric_fields: Vec::new(),
      nested_fields: vec![NestedField {
        name: "comments".into(),
        fields: vec![NestedProperty::Text(TextField {
          name: "body".into(),
          analyzer: "default".into(),
          search_analyzer: None,
          stored: true,
          indexed: true,
          nullable: false,
          search_as_you_type: None,
        })],
        nullable: false,
      }],
      #[cfg(feature = "vectors")]
      vector_fields: Vec::new(),
    };

    let bad = crate::api::types::Document {
      fields: [
        ("_id".into(), serde_json::json!("d1")),
        (
          "comments".into(),
          serde_json::json!({ "body": [{ "x": 1 }, null] }),
        ),
      ]
      .into_iter()
      .collect(),
    };
    let err = schema.validate_document(&bad).unwrap_err();
    let messages: Vec<String> = err.chain().map(|e| e.to_string()).collect();
    assert!(
      messages.iter().any(|m| m.contains("array of strings")),
      "unexpected error chain: {messages:?}"
    );

    let ok = crate::api::types::Document {
      fields: [
        ("_id".into(), serde_json::json!("d2")),
        (
          "comments".into(),
          serde_json::json!({ "body": ["hello", "world"] }),
        ),
      ]
      .into_iter()
      .collect(),
    };
    schema.validate_document(&ok).expect("string array ok");
  }

  #[test]
  fn nested_numeric_rejects_non_number_array_elements() {
    fn schema_with(use_i64: bool) -> Schema {
      Schema {
        doc_id_field: default_doc_id_field(),
        analyzers: Vec::new(),
        text_fields: Vec::new(),
        keyword_fields: Vec::new(),
        numeric_fields: Vec::new(),
        nested_fields: vec![NestedField {
          name: "metrics".into(),
          fields: vec![NestedProperty::Numeric(NumericField {
            name: "values".into(),
            i64: use_i64,
            fast: false,
            stored: false,
            nullable: false,
          })],
          nullable: false,
        }],
        #[cfg(feature = "vectors")]
        vector_fields: Vec::new(),
      }
    }

    let i64_schema = schema_with(true);
    let bad_strings = crate::api::types::Document {
      fields: [
        ("_id".into(), serde_json::json!("d1")),
        (
          "metrics".into(),
          serde_json::json!({ "values": ["1", "2"] }),
        ),
      ]
      .into_iter()
      .collect(),
    };
    let err = i64_schema.validate_document(&bad_strings).unwrap_err();
    let messages: Vec<String> = err.chain().map(|e| e.to_string()).collect();
    assert!(
      messages.iter().any(|m| m.contains("array of numbers")),
      "unexpected error chain: {messages:?}"
    );

    // i64 fields must reject floats mixed into the array.
    let bad_floats = crate::api::types::Document {
      fields: [
        ("_id".into(), serde_json::json!("d2")),
        ("metrics".into(), serde_json::json!({ "values": [1, 2.5] })),
      ]
      .into_iter()
      .collect(),
    };
    assert!(i64_schema.validate_document(&bad_floats).is_err());

    let good_ints = crate::api::types::Document {
      fields: [
        ("_id".into(), serde_json::json!("d3")),
        ("metrics".into(), serde_json::json!({ "values": [1, 2, 3] })),
      ]
      .into_iter()
      .collect(),
    };
    i64_schema
      .validate_document(&good_ints)
      .expect("integer array ok");

    // f64 fields accept integers and floats but still reject strings.
    let f64_schema = schema_with(false);
    let good_floats = crate::api::types::Document {
      fields: [
        ("_id".into(), serde_json::json!("d4")),
        (
          "metrics".into(),
          serde_json::json!({ "values": [1, 2.5, 3] }),
        ),
      ]
      .into_iter()
      .collect(),
    };
    f64_schema
      .validate_document(&good_floats)
      .expect("number array ok");
    assert!(f64_schema.validate_document(&bad_strings).is_err());
  }

  #[test]
  fn check_required_fields_present_rejects_missing_non_nullable_top_level_fields() {
    // Regression for BUG-224: a user-submitted document that omits a
    // declared-required top-level field must be rejected at the ingest
    // boundary, mirroring the nested-field behaviour. This check lives in
    // `check_required_fields_present` (invoked by `add_document` /
    // `add_documents_batch`), not `validate_document`, so that rewrite
    // flows that re-validate reconstructed documents do not false-fail.
    let schema = Schema {
      doc_id_field: default_doc_id_field(),
      analyzers: Vec::new(),
      text_fields: vec![TextField {
        name: "body".into(),
        analyzer: "default".into(),
        search_analyzer: None,
        stored: true,
        indexed: true,
        nullable: false,
        search_as_you_type: None,
      }],
      keyword_fields: vec![KeywordField {
        name: "tag".into(),
        stored: true,
        indexed: true,
        fast: false,
        nullable: false,
      }],
      numeric_fields: vec![NumericField {
        name: "price".into(),
        i64: true,
        fast: false,
        stored: false,
        nullable: false,
      }],
      nested_fields: Vec::new(),
      #[cfg(feature = "vectors")]
      vector_fields: Vec::new(),
    };

    for missing in ["body", "tag", "price"] {
      let mut fields: std::collections::BTreeMap<String, serde_json::Value> = [
        ("_id".into(), serde_json::json!("doc-1")),
        ("body".into(), serde_json::json!("hello")),
        ("tag".into(), serde_json::json!("gadget")),
        ("price".into(), serde_json::json!(10)),
      ]
      .into_iter()
      .collect();
      fields.remove(missing);
      let doc = crate::api::types::Document { fields };
      let err = schema
        .check_required_fields_present(&doc)
        .expect_err("validation must reject missing required field");
      let msg = err.to_string();
      assert!(
        msg.contains(&format!("missing required field `{missing}`")),
        "unexpected error for missing field `{missing}`: {msg}"
      );
    }

    // Complete document passes both checks.
    let complete = crate::api::types::Document {
      fields: [
        ("_id".into(), serde_json::json!("doc-ok")),
        ("body".into(), serde_json::json!("hello")),
        ("tag".into(), serde_json::json!("gadget")),
        ("price".into(), serde_json::json!(10)),
      ]
      .into_iter()
      .collect(),
    };
    schema
      .validate_document(&complete)
      .expect("complete document passes validation");
    schema
      .check_required_fields_present(&complete)
      .expect("complete document passes required-fields check");
  }

  #[test]
  fn check_required_fields_present_allows_missing_nullable_top_level_fields() {
    // Complements the regression test above: nullable fields must still be
    // allowed to be omitted entirely at ingest.
    let schema = Schema {
      doc_id_field: default_doc_id_field(),
      analyzers: Vec::new(),
      text_fields: vec![TextField {
        name: "subtitle".into(),
        analyzer: "default".into(),
        search_analyzer: None,
        stored: true,
        indexed: true,
        nullable: true,
        search_as_you_type: None,
      }],
      keyword_fields: vec![KeywordField {
        name: "brand".into(),
        stored: true,
        indexed: true,
        fast: false,
        nullable: true,
      }],
      numeric_fields: vec![NumericField {
        name: "sale_price".into(),
        i64: true,
        fast: false,
        stored: false,
        nullable: true,
      }],
      nested_fields: Vec::new(),
      #[cfg(feature = "vectors")]
      vector_fields: Vec::new(),
    };
    let doc = crate::api::types::Document {
      fields: [("_id".into(), serde_json::json!("only-id"))]
        .into_iter()
        .collect(),
    };
    schema
      .check_required_fields_present(&doc)
      .expect("nullable-only schema permits omission");
  }

  #[test]
  fn validate_document_does_not_enforce_required_presence() {
    // BUG-224 split: `validate_document` must stay permissive about
    // missing top-level fields because rewrite flows (`compact`,
    // `merge_segments`, `apply_patch`) re-validate documents
    // reconstructed from the docstore, and round-tripping is lossy for
    // several legitimate schema shapes (empty arrays, nested containers
    // whose stored children serialize away). The presence check lives
    // in `check_required_fields_present` and is invoked only on
    // user-supplied documents at the ingest boundary.
    let schema = Schema {
      doc_id_field: default_doc_id_field(),
      analyzers: Vec::new(),
      text_fields: vec![TextField {
        name: "body".into(),
        analyzer: "default".into(),
        search_analyzer: None,
        stored: true,
        indexed: true,
        nullable: false,
        search_as_you_type: None,
      }],
      keyword_fields: Vec::new(),
      numeric_fields: Vec::new(),
      nested_fields: Vec::new(),
      #[cfg(feature = "vectors")]
      vector_fields: Vec::new(),
    };
    let reconstructed = crate::api::types::Document {
      fields: [("_id".into(), serde_json::json!("c1"))]
        .into_iter()
        .collect(),
    };
    schema
      .validate_document(&reconstructed)
      .expect("validate_document must not enforce required-field presence");
  }

  #[test]
  fn check_required_fields_present_requires_nested_top_level_container() {
    // A non-nullable nested container itself must be present at ingest,
    // not just its sub-fields. Complements the existing nested sub-field
    // check performed by `NestedField::validate` when the container is
    // present.
    let schema = Schema {
      doc_id_field: default_doc_id_field(),
      analyzers: Vec::new(),
      text_fields: Vec::new(),
      keyword_fields: Vec::new(),
      numeric_fields: Vec::new(),
      nested_fields: vec![NestedField {
        name: "comment".into(),
        fields: vec![NestedProperty::Keyword(KeywordField {
          name: "author".into(),
          stored: true,
          indexed: true,
          fast: false,
          nullable: false,
        })],
        nullable: false,
      }],
      #[cfg(feature = "vectors")]
      vector_fields: Vec::new(),
    };
    let doc = crate::api::types::Document {
      fields: [("_id".into(), serde_json::json!("c1"))]
        .into_iter()
        .collect(),
    };
    let err = schema
      .check_required_fields_present(&doc)
      .expect_err("missing nested container must error at ingest");
    assert!(
      err.to_string().contains("missing required field `comment`"),
      "unexpected error: {err}"
    );
  }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldKind {
  Text,
  Keyword,
  Numeric,
  Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedField {
  pub path: String,
  pub kind: FieldKind,
  pub indexed: bool,
  pub stored: bool,
  pub fast: bool,
  pub numeric_i64: Option<bool>,
  pub nullable: bool,
}

#[derive(Debug, Clone)]
pub struct FieldAnalyzerRefs {
  pub analyzer: String,
  pub search_analyzer: String,
}

#[derive(Debug, Clone)]
pub struct SchemaAnalyzers {
  pub(crate) registry: AnalyzerRegistry,
  pub(crate) field_map: HashMap<String, FieldAnalyzerRefs>,
}

impl SchemaAnalyzers {
  pub fn index_analyzer(&self, field: &str) -> Option<&Analyzer> {
    self
      .field_map
      .get(field)
      .and_then(|f| self.registry.get(&f.analyzer))
  }

  pub fn search_analyzer(&self, field: &str) -> Option<&Analyzer> {
    self
      .field_map
      .get(field)
      .and_then(|f| self.registry.get(&f.search_analyzer))
  }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(into = "SearchAsYouTypeDef", try_from = "SearchAsYouTypeDef")]
pub struct SearchAsYouType {
  pub min_gram: usize,
  pub max_gram: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SearchAsYouTypeDef {
  #[serde(default = "default_search_as_you_type_min")]
  min_gram: usize,
  #[serde(default = "default_search_as_you_type_max")]
  max_gram: usize,
}

impl From<SearchAsYouType> for SearchAsYouTypeDef {
  fn from(value: SearchAsYouType) -> Self {
    Self {
      min_gram: value.min_gram,
      max_gram: value.max_gram,
    }
  }
}

impl TryFrom<SearchAsYouTypeDef> for SearchAsYouType {
  type Error = anyhow::Error;

  fn try_from(value: SearchAsYouTypeDef) -> Result<Self, Self::Error> {
    let min = value.min_gram;
    let max = value.max_gram;
    if min == 0 || max == 0 {
      anyhow::bail!(
        "invalid search_as_you_type configuration: min_gram and max_gram must both be greater than zero (got min_gram={min}, max_gram={max})"
      );
    }
    if min > max {
      anyhow::bail!(
        "invalid search_as_you_type configuration: min_gram ({min}) must be less than or equal to max_gram ({max})"
      );
    }
    Ok(Self {
      min_gram: min,
      max_gram: max,
    })
  }
}

fn default_search_as_you_type_min() -> usize {
  1
}

fn default_search_as_you_type_max() -> usize {
  15
}

#[derive(Debug, Clone)]
pub struct TextField {
  pub name: String,
  pub analyzer: String,
  pub search_analyzer: Option<String>,
  pub stored: bool,
  pub indexed: bool,
  pub nullable: bool,
  pub search_as_you_type: Option<SearchAsYouType>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordField {
  pub name: String,
  pub stored: bool,
  pub indexed: bool,
  pub fast: bool,
  #[serde(default)]
  pub nullable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericField {
  pub name: String,
  pub i64: bool,
  pub fast: bool,
  #[serde(default)]
  pub stored: bool,
  #[serde(default)]
  pub nullable: bool,
}

#[derive(Debug, Clone)]
pub struct NestedField {
  pub name: String,
  pub fields: Vec<NestedProperty>,
  pub nullable: bool,
}

impl NestedField {
  fn validate(&self, value: &serde_json::Value) -> anyhow::Result<()> {
    match value {
      serde_json::Value::Null => {
        if self.nullable {
          return Ok(());
        }
        Err(anyhow!("nested field {} cannot be null", self.name))
      }
      serde_json::Value::Array(arr) => {
        for v in arr.iter() {
          self.validate(v)?;
        }
        Ok(())
      }
      serde_json::Value::Object(map) => {
        for (k, v) in map.iter() {
          let Some(prop) = self.fields.iter().find(|p| p.name() == k) else {
            return Err(anyhow!("unknown nested field {}.{}", self.name, k));
          };
          prop.validate_value(&format!("{}.{}", self.name, k), v)?;
        }
        for prop in self.fields.iter() {
          if map.contains_key(prop.name()) {
            continue;
          }
          if prop.is_nullable() {
            continue;
          }
          return Err(anyhow!(
            "missing required nested field {}.{}",
            self.name,
            prop.name()
          ));
        }
        Ok(())
      }
      _ => Err(anyhow!(
        "nested field {} must be object or array",
        self.name
      )),
    }
  }

  fn collect_fields(&self, prefix: Option<&str>, out: &mut Vec<ResolvedField>) {
    let mut full_prefix = String::new();
    if let Some(p) = prefix {
      full_prefix.push_str(p);
      full_prefix.push('.');
    }
    full_prefix.push_str(&self.name);
    for f in self.fields.iter() {
      f.collect_fields(&full_prefix, out);
    }
  }
}

#[derive(Debug, Clone)]
pub enum NestedProperty {
  Text(TextField),
  Keyword(KeywordField),
  Numeric(NumericField),
  Object(NestedField),
}

impl NestedProperty {
  pub fn name(&self) -> &str {
    match self {
      NestedProperty::Text(f) => &f.name,
      NestedProperty::Keyword(f) => &f.name,
      NestedProperty::Numeric(f) => &f.name,
      NestedProperty::Object(f) => &f.name,
    }
  }

  pub fn is_nullable(&self) -> bool {
    match self {
      NestedProperty::Text(f) => f.nullable,
      NestedProperty::Keyword(f) => f.nullable,
      NestedProperty::Numeric(f) => f.nullable,
      NestedProperty::Object(f) => f.nullable,
    }
  }

  fn validate_value(&self, key: &str, v: &serde_json::Value) -> anyhow::Result<()> {
    match self {
      NestedProperty::Text(f) => {
        if v.is_null() {
          if f.nullable {
            return Ok(());
          }
          return Err(anyhow!("nested field {key} cannot be null"));
        }
        if !is_string_or_string_array(v) {
          return Err(anyhow!(
            "nested field {key} must be a string or array of strings"
          ));
        }
        Ok(())
      }
      NestedProperty::Keyword(f) => {
        if v.is_null() {
          if f.nullable {
            return Ok(());
          }
          return Err(anyhow!("nested field {key} cannot be null"));
        }
        if !is_string_or_string_array(v) {
          return Err(anyhow!(
            "nested field {key} must be a string or array of strings"
          ));
        }
        Ok(())
      }
      NestedProperty::Numeric(f) => {
        if v.is_null() {
          if f.nullable {
            return Ok(());
          }
          return Err(anyhow!("nested field {key} cannot be null"));
        }
        let ok = if f.i64 {
          is_i64_or_array(v)
        } else {
          is_f64_or_array(v)
        };
        if !ok {
          return Err(anyhow!(
            "nested field {key} must be a number or array of numbers"
          ));
        }
        Ok(())
      }
      NestedProperty::Object(obj) => {
        if v.is_null() {
          if obj.nullable {
            return Ok(());
          }
          return Err(anyhow!("nested field {key} cannot be null"));
        }
        obj.validate(v)
      }
    }
  }

  fn collect_fields(&self, prefix: &str, out: &mut Vec<ResolvedField>) {
    match self {
      NestedProperty::Text(f) => out.push(ResolvedField {
        path: format!("{prefix}.{}", f.name),
        kind: FieldKind::Text,
        indexed: f.indexed,
        stored: f.stored,
        fast: false,
        numeric_i64: None,
        nullable: f.nullable,
      }),
      NestedProperty::Keyword(f) => out.push(ResolvedField {
        path: format!("{prefix}.{}", f.name),
        kind: FieldKind::Keyword,
        indexed: f.indexed,
        stored: f.stored,
        fast: f.fast,
        numeric_i64: None,
        nullable: f.nullable,
      }),
      NestedProperty::Numeric(f) => out.push(ResolvedField {
        path: format!("{prefix}.{}", f.name),
        kind: FieldKind::Numeric,
        indexed: true,
        stored: f.stored,
        fast: f.fast,
        numeric_i64: Some(f.i64),
        nullable: f.nullable,
      }),
      NestedProperty::Object(obj) => obj.collect_fields(Some(prefix), out),
    }
  }
}

fn collect_nested_text_fields<'a>(
  nested: &'a NestedField,
  prefix: Option<&str>,
  out: &mut Vec<(String, &'a TextField)>,
) {
  let mut full_prefix = String::new();
  if let Some(p) = prefix {
    full_prefix.push_str(p);
    full_prefix.push('.');
  }
  full_prefix.push_str(&nested.name);
  for f in nested.fields.iter() {
    match f {
      NestedProperty::Text(field) => {
        out.push((format!("{full_prefix}.{}", field.name), field));
      }
      NestedProperty::Object(obj) => collect_nested_text_fields(obj, Some(&full_prefix), out),
      _ => {}
    }
  }
}

#[cfg(feature = "vectors")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorField {
  pub name: String,
  pub dim: usize,
  pub metric: VectorMetric,
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub hnsw: Option<crate::vectors::hnsw::HnswParams>,
}

#[cfg(feature = "vectors")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorMetric {
  Cosine,
  L2,
}

#[cfg(feature = "vectors")]
impl From<crate::api::types::VectorMetric> for VectorMetric {
  fn from(v: crate::api::types::VectorMetric) -> Self {
    match v {
      crate::api::types::VectorMetric::Cosine => VectorMetric::Cosine,
      crate::api::types::VectorMetric::L2 => VectorMetric::L2,
    }
  }
}

#[cfg(feature = "vectors")]
impl From<VectorMetric> for crate::api::types::VectorMetric {
  fn from(v: VectorMetric) -> Self {
    match v {
      VectorMetric::Cosine => crate::api::types::VectorMetric::Cosine,
      VectorMetric::L2 => crate::api::types::VectorMetric::L2,
    }
  }
}
