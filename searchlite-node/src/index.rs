use std::path::PathBuf;
use std::sync::Mutex;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::collections::BTreeMap;

use searchlite_core::api::types::{
  ChecksumPolicy, ExecutionStrategy, IndexOptions, Query, SearchRequest, StorageType,
};
use searchlite_core::Index as CoreIndex;
use searchlite_core::Schema;
use searchlite_s3::{S3Config, S3Credentials};

use crate::convert::{value_to_document, value_to_documents};
use crate::error::{catch_panic, to_napi_error};

const BM25_K1: f32 = 0.9;
const BM25_B: f32 = 0.4;

#[napi(object)]
pub struct OpenOptions {
  pub write_key: Option<String>,
  pub schema: Option<serde_json::Value>,
}

/// Static credentials for an S3-compatible endpoint. Omit the
/// containing `credentials` field on `S3IndexConfig` to load
/// credentials from the standard AWS chain (env vars, shared
/// credentials file, IMDS, EC2 instance role).
#[napi(object)]
pub struct S3StaticCredentials {
  pub access_key_id: String,
  pub secret_access_key: String,
  pub session_token: Option<String>,
}

/// Configuration for opening a read-only Index against an
/// S3-compatible backend (AWS S3, Cloudflare R2, MinIO).
#[napi(object)]
pub struct S3IndexConfig {
  /// Bucket name.
  pub bucket: String,
  /// Region. Defaults to `us-east-1` when unset (required by SigV4
  /// even for R2 — pass `auto` for R2).
  pub region: Option<String>,
  /// Optional namespace within the bucket.
  pub prefix: Option<String>,
  /// Endpoint URL. Set for R2
  /// (`https://<account>.r2.cloudflarestorage.com`) or MinIO /
  /// LocalStack. Leave unset for AWS S3.
  pub endpoint_url: Option<String>,
  /// Path-style addressing (`https://endpoint/bucket/key`). Required
  /// for MinIO / LocalStack. Defaults to `false`.
  pub force_path_style: Option<bool>,
  /// Conditional PUT support (`If-Match` / `If-None-Match`). Defaults
  /// to `true` on AWS S3 and MinIO, and `false` on R2 (auto-detected
  /// from the endpoint hostname pattern `*.r2.cloudflarestorage.com`).
  pub conditional_put: Option<bool>,
  /// Credentials. When omitted, the standard AWS chain is used.
  pub credentials: Option<S3StaticCredentials>,
  /// Checksum policy: `"strict"` (default), `"trust-manifest"`, or
  /// `"audit"`. See `searchlite-core` docs for the trade-offs.
  pub checksum_policy: Option<String>,
}

/// Trim whitespace and treat the empty string as `None`. Used to keep
/// every user-supplied string forwarded into the AWS SDK on equal
/// footing — passing a value with stray padding into SigV4 signing
/// surfaces as an opaque `SignatureDoesNotMatch` rather than a clean
/// validation error.
fn normalize_string(s: Option<String>) -> Option<String> {
  s.map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

impl S3IndexConfig {
  fn into_parts(self) -> napi::Result<(S3Config, IndexOptions)> {
    let bucket = self.bucket.trim().to_string();
    if bucket.is_empty() {
      return Err(napi::Error::new(
        napi::Status::InvalidArg,
        "bucket must be a non-empty string",
      ));
    }
    let region = normalize_string(self.region).unwrap_or_else(|| "us-east-1".to_string());
    let endpoint_url = normalize_string(self.endpoint_url);
    let prefix = normalize_string(self.prefix);
    let is_r2 = endpoint_url
      .as_deref()
      .map(searchlite_s3::is_r2_endpoint)
      .unwrap_or(false);
    let conditional_put = self.conditional_put.unwrap_or(!is_r2);
    let credentials = match self.credentials {
      Some(c) => S3Credentials::Static {
        access_key_id: c.access_key_id,
        secret_access_key: c.secret_access_key,
        session_token: c.session_token,
      },
      None => S3Credentials::LoadFromEnv,
    };
    let s3 = S3Config {
      endpoint_url,
      region,
      bucket,
      prefix,
      credentials,
      conditional_put,
      force_path_style: self.force_path_style.unwrap_or(false),
    };

    let checksum_policy = match self.checksum_policy.as_deref() {
      None | Some("strict") => ChecksumPolicy::Strict,
      Some("trust-manifest") => ChecksumPolicy::TrustManifest,
      Some("audit") => ChecksumPolicy::Audit,
      Some(other) => {
        return Err(napi::Error::new(
          napi::Status::InvalidArg,
          format!("invalid checksumPolicy: {other}; expected strict, trust-manifest, or audit"),
        ));
      }
    };
    // Preserve the Node binding's BM25 tuning so search rankings match
    // those of filesystem-backed indexes opened via `Index::new`.
    let opts = IndexOptions {
      checksum_policy,
      bm25_k1: BM25_K1,
      bm25_b: BM25_B,
      ..IndexOptions::default()
    };
    Ok((s3, opts))
  }
}

#[napi]
pub struct Index {
  inner: Mutex<Option<CoreIndex>>,
  write_key: Option<String>,
}

#[napi]
impl Index {
  #[napi(constructor)]
  pub fn new(path: String, options: Option<OpenOptions>) -> napi::Result<Self> {
    catch_panic("Index::new", || {
      let opts = options.unwrap_or(OpenOptions {
        write_key: None,
        schema: None,
      });
      let path_buf = PathBuf::from(&path);
      let core_opts = IndexOptions {
        path: path_buf.clone(),
        create_if_missing: opts.schema.is_some(),
        enable_positions: true,
        bm25_k1: BM25_K1,
        bm25_b: BM25_B,
        storage: StorageType::Filesystem,
        checksum_policy: Default::default(),
        checksum_audit_failure_hook: None,
        read_only: false,
        #[cfg(feature = "vectors")]
        vector_defaults: None,
      };

      let provided_schema: Option<Schema> = match &opts.schema {
        Some(val) => Some(serde_json::from_value(val.clone()).map_err(|e| {
          napi::Error::new(napi::Status::InvalidArg, format!("invalid schema: {e}"))
        })?),
        None => None,
      };

      let manifest_path = path_buf.join("MANIFEST.json");
      let index = if manifest_path.exists() {
        // Index exists — open and optionally validate schema
        let index = CoreIndex::open(core_opts).map_err(to_napi_error)?;
        if let Some(ref schema) = provided_schema {
          let existing = index.manifest().schema;
          let existing_json =
            serde_json::to_value(&existing).map_err(|e| to_napi_error(e.into()))?;
          let provided_json = serde_json::to_value(schema).map_err(|e| to_napi_error(e.into()))?;
          if existing_json != provided_json {
            return Err(napi::Error::new(
              napi::Status::InvalidArg,
              format!(
                "schema mismatch: provided schema does not match existing index.\n  existing: {}\n  provided: {}",
                existing_json, provided_json
              ),
            ));
          }
        }
        index
      } else if let Some(schema) = provided_schema {
        // Index does not exist — create with schema
        CoreIndex::create_with_write_key(&path_buf, schema, core_opts, opts.write_key.as_deref())
          .map_err(to_napi_error)?
      } else {
        // No schema, no existing index — error
        return Err(napi::Error::new(
          napi::Status::GenericFailure,
          "index does not exist; provide a schema to create it",
        ));
      };

      Ok(Self {
        inner: Mutex::new(Some(index)),
        write_key: opts.write_key,
      })
    })
  }

  /// Open a read-only Index against an S3-compatible backend.
  ///
  /// The schema is read from the manifest in the bucket — there is
  /// no constructor-time schema for S3-backed indexes. Mutators
  /// (`add`, `addMany`, `commit`, `compact`) will error.
  ///
  /// This factory is async: opening involves at least one network
  /// round-trip (HEAD on `MANIFEST.json`) plus checksum-driven segment
  /// reads, and must not block Node's event loop.
  #[napi(factory, js_name = "fromS3")]
  pub async fn from_s3(config: S3IndexConfig) -> napi::Result<Self> {
    let (s3_config, opts) = config.into_parts()?;
    let index = searchlite_s3::open_index_read_only_with_options(s3_config, opts)
      .await
      .map_err(to_napi_error)?;
    Ok(Self {
      inner: Mutex::new(Some(index)),
      write_key: None,
    })
  }

  #[napi]
  pub fn add(&self, env: Env, doc: Unknown) -> napi::Result<()> {
    catch_panic("Index::add", || {
      let value: serde_json::Value = env.from_js_value(doc)?;
      let document = value_to_document(value)?;
      self.with_index(|index| {
        let mut writer = index
          .writer_with_key(self.write_key.as_deref())
          .map_err(to_napi_error)?;
        writer.add_document(&document).map_err(to_napi_error)?;
        Ok(())
      })
    })
  }

  #[napi(js_name = "addMany")]
  pub fn add_many(&self, env: Env, docs: Unknown) -> napi::Result<u32> {
    catch_panic("Index::addMany", || {
      let value: serde_json::Value = env.from_js_value(docs)?;
      let documents = value_to_documents(value)?;
      self.with_index(|index| {
        let mut writer = index
          .writer_with_key(self.write_key.as_deref())
          .map_err(to_napi_error)?;
        let count = writer
          .add_documents_batch(&documents)
          .map_err(to_napi_error)?;
        Ok(count as u32)
      })
    })
  }

  #[napi]
  pub fn commit(&self) -> napi::Result<()> {
    catch_panic("Index::commit", || {
      self.with_index(|index| {
        let mut writer = index
          .writer_with_key(self.write_key.as_deref())
          .map_err(to_napi_error)?;
        writer.commit().map_err(to_napi_error)
      })
    })
  }

  /// Delete a single document by id. Unlike `add`/`addMany` (which queue and
  /// require a separate `commit`), `delete`/`deleteMany` delete **and commit**
  /// within one writer session, so the removal is durable on return. A missing
  /// id is a no-op (the engine queues a tombstone regardless of existence).
  #[napi]
  pub fn delete(&self, id: String) -> napi::Result<()> {
    catch_panic("Index::delete", || {
      self.with_index(|index| {
        let mut writer = index
          .writer_with_key(self.write_key.as_deref())
          .map_err(to_napi_error)?;
        writer.delete_documents(&[id]).map_err(to_napi_error)?;
        writer.commit().map_err(to_napi_error)
      })
    })
  }

  /// Delete many documents by id, then commit — all in one writer session.
  /// Returns the number of ids submitted (not necessarily the number that
  /// existed). See `delete` for the commit semantics.
  #[napi(js_name = "deleteMany")]
  pub fn delete_many(&self, ids: Vec<String>) -> napi::Result<u32> {
    catch_panic("Index::deleteMany", || {
      self.with_index(|index| {
        let mut writer = index
          .writer_with_key(self.write_key.as_deref())
          .map_err(to_napi_error)?;
        writer.delete_documents(&ids).map_err(to_napi_error)?;
        writer.commit().map_err(to_napi_error)?;
        Ok(ids.len() as u32)
      })
    })
  }

  #[napi]
  pub fn search(&self, env: Env, query: Unknown) -> napi::Result<Unknown<'_>> {
    catch_panic("Index::search", || {
      self.with_index(|index| {
        let value: serde_json::Value = env.from_js_value(query)?;
        let request: SearchRequest = match value {
          serde_json::Value::String(s) => SearchRequest {
            query: Query::String(s),
            fields: None,
            filter: None,
            limit: 10,
            from: 0,
            return_hits: true,
            candidate_size: None,
            #[cfg(feature = "vectors")]
            max_global_vector_candidates: None,
            sort: Vec::new(),
            cursor: None,
            search_after: None,
            execution: ExecutionStrategy::Wand,
            bmw_block_size: None,
            fuzzy: None,
            track_total_hits: None,
            #[cfg(feature = "vectors")]
            vector_query: None,
            #[cfg(feature = "vectors")]
            vector_filter: None,
            return_stored: false,
            highlight_field: None,
            highlight: None,
            collapse: None,
            aggs: BTreeMap::new(),
            suggest: BTreeMap::new(),
            rescore: None,
            explain: false,
            profile: false,
          },
          obj @ serde_json::Value::Object(_) => serde_json::from_value(obj).map_err(|e| {
            napi::Error::new(
              napi::Status::InvalidArg,
              format!("invalid search request: {e}"),
            )
          })?,
          _ => {
            return Err(napi::Error::new(
              napi::Status::InvalidArg,
              "query must be a string or object",
            ));
          }
        };
        let reader = index.reader().map_err(to_napi_error)?;
        let result = reader.search(&request).map_err(to_napi_error)?;
        let result_value = serde_json::to_value(&result).map_err(|e| {
          napi::Error::new(
            napi::Status::GenericFailure,
            format!("failed to serialize result: {e}"),
          )
        })?;
        env.to_js_value(&result_value)
      })
    })
  }

  #[napi]
  pub fn compact(&self) -> napi::Result<()> {
    catch_panic("Index::compact", || {
      self.with_index(|index| {
        index
          .compact_with_key(self.write_key.as_deref())
          .map_err(to_napi_error)
      })
    })
  }

  #[napi]
  pub fn close(&self) -> napi::Result<()> {
    let mut guard = self
      .inner
      .lock()
      .map_err(|_| napi::Error::new(napi::Status::GenericFailure, "index lock poisoned"))?;
    guard.take();
    Ok(())
  }
}

impl Index {
  fn with_index<T>(&self, f: impl FnOnce(&CoreIndex) -> napi::Result<T>) -> napi::Result<T> {
    let guard = self
      .inner
      .lock()
      .map_err(|_| napi::Error::new(napi::Status::GenericFailure, "index lock poisoned"))?;
    let index = guard
      .as_ref()
      .ok_or_else(|| napi::Error::new(napi::Status::GenericFailure, "index is closed"))?;
    f(index)
  }
}
