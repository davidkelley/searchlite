pub mod cli;
pub mod core;
pub mod http;

use anyhow::{anyhow, Result};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SurfaceKind {
  Core,
  Http,
  Cli,
}

#[derive(Debug, Clone, Copy)]
pub struct SurfaceCapabilities {
  pub supports_refresh: bool,
  pub supports_search_after: bool,
  pub supports_mget: bool,
  pub supports_update: bool,
  pub supports_delete: bool,
  pub supports_status_codes: bool,
  pub supports_stats: bool,
  pub supports_inspect: bool,
  pub supports_compact: bool,
}

impl SurfaceCapabilities {
  pub const fn for_surface(kind: SurfaceKind) -> Self {
    match kind {
      SurfaceKind::Core => Self {
        supports_refresh: true,
        supports_search_after: true,
        supports_mget: true,
        supports_update: true,
        supports_delete: true,
        supports_status_codes: false,
        supports_stats: true,
        supports_inspect: true,
        supports_compact: true,
      },
      SurfaceKind::Http => Self {
        supports_refresh: true,
        supports_search_after: true,
        supports_mget: true,
        supports_update: true,
        supports_delete: true,
        supports_status_codes: true,
        supports_stats: true,
        supports_inspect: true,
        supports_compact: true,
      },
      SurfaceKind::Cli => Self {
        supports_refresh: false,
        supports_search_after: true,
        supports_mget: false,
        supports_update: false,
        supports_delete: true,
        supports_status_codes: false,
        supports_stats: false,
        supports_inspect: true,
        supports_compact: true,
      },
    }
  }
}

pub trait SurfaceHarness {
  fn kind(&self) -> SurfaceKind;

  fn capabilities(&self) -> SurfaceCapabilities {
    SurfaceCapabilities::for_surface(self.kind())
  }

  fn init(&mut self, _schema: &Value) -> Result<()>;
  fn add_ndjson(&mut self, _ndjson: &str) -> Result<()>;
  fn commit(&mut self) -> Result<()>;
  fn refresh(&mut self) -> Result<()> {
    Err(unsupported_operation(self.kind(), "refresh"))
  }
  fn search(&mut self, _request: &Value) -> Result<Value>;
  fn mget(&mut self, _ids: &[String], _return_stored: bool) -> Result<Value> {
    Err(unsupported_operation(self.kind(), "mget"))
  }
  fn update_doc(&mut self, _id: &str, _set: &Map<String, Value>, _unset: &[String]) -> Result<()> {
    Err(unsupported_operation(self.kind(), "update"))
  }
  fn delete_ids(&mut self, _ids: &[String]) -> Result<()> {
    Err(unsupported_operation(self.kind(), "delete"))
  }
  fn stats(&mut self) -> Result<Value> {
    Err(unsupported_operation(self.kind(), "stats"))
  }
  fn inspect(&mut self) -> Result<Value> {
    Err(unsupported_operation(self.kind(), "inspect"))
  }
  fn compact(&mut self) -> Result<()> {
    Err(unsupported_operation(self.kind(), "compact"))
  }
}

pub fn unsupported_operation(kind: SurfaceKind, operation: &str) -> anyhow::Error {
  anyhow!(
    "operation_not_supported:{}:{}",
    surface_name(kind),
    operation
  )
}

pub fn is_not_supported_error(err: &anyhow::Error) -> bool {
  err.to_string().starts_with("operation_not_supported:")
}

fn surface_name(kind: SurfaceKind) -> &'static str {
  match kind {
    SurfaceKind::Core => "core",
    SurfaceKind::Http => "http",
    SurfaceKind::Cli => "cli",
  }
}
