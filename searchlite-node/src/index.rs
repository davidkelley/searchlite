use std::path::PathBuf;
use std::sync::Mutex;

use napi::bindgen_prelude::*;
use napi::JsUnknown;
use napi_derive::napi;
use std::collections::BTreeMap;

use searchlite_core::api::types::{
  ExecutionStrategy, IndexOptions, Query, SearchRequest, StorageType,
};
use searchlite_core::Index as CoreIndex;
use searchlite_core::Schema;

use crate::convert::{value_to_document, value_to_documents};
use crate::error::{catch_panic, to_napi_error};

const BM25_K1: f32 = 0.9;
const BM25_B: f32 = 0.4;

#[napi(object)]
pub struct OpenOptions {
  pub write_key: Option<String>,
  pub schema: Option<serde_json::Value>,
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

  #[napi]
  pub fn add(&self, env: Env, doc: JsUnknown) -> napi::Result<()> {
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
  pub fn add_many(&self, env: Env, docs: JsUnknown) -> napi::Result<u32> {
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

  #[napi]
  pub fn search(&self, env: Env, query: JsUnknown) -> napi::Result<JsUnknown> {
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
