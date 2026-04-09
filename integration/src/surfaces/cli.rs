use std::fs;
use std::path::PathBuf;
use std::process::Command;

use anyhow::{anyhow, Context, Result};
use serde_json::{Map, Value};
use tempfile::TempDir;

use super::{unsupported_operation, SurfaceHarness, SurfaceKind};

pub struct CliHarness {
  bin: PathBuf,
  index_path: PathBuf,
  scratch: TempDir,
  file_counter: usize,
}

impl CliHarness {
  pub fn new(bin: PathBuf, index_path: PathBuf) -> Self {
    Self {
      bin,
      index_path,
      scratch: tempfile::tempdir().expect("create CLI harness scratch dir"),
      file_counter: 0,
    }
  }

  fn next_path(&mut self, prefix: &str, extension: &str) -> PathBuf {
    self.file_counter += 1;
    self
      .scratch
      .path()
      .join(format!("{prefix}-{}.{}", self.file_counter, extension))
  }

  fn run(&self, args: &[String]) -> Result<String> {
    let output = Command::new(&self.bin)
      .args(args)
      .output()
      .with_context(|| {
        format!(
          "running CLI command: {} {}",
          self.bin.display(),
          args.join(" ")
        )
      })?;

    if !output.status.success() {
      return Err(anyhow!(
        "CLI command failed: {}\nstderr:\n{}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr)
      ));
    }

    String::from_utf8(output.stdout).context("decoding CLI stdout as UTF-8")
  }

  fn index_arg(&self) -> String {
    self.index_path.display().to_string()
  }
}

impl SurfaceHarness for CliHarness {
  fn kind(&self) -> SurfaceKind {
    SurfaceKind::Cli
  }

  fn init(&mut self, schema: &Value) -> Result<()> {
    let schema_path = self.next_path("schema", "json");
    fs::write(&schema_path, serde_json::to_vec_pretty(schema)?).with_context(|| {
      format!(
        "writing CLI schema fixture to {}",
        schema_path.as_path().display()
      )
    })?;

    self.run(&[
      "init".to_string(),
      self.index_arg(),
      schema_path.display().to_string(),
    ])?;
    Ok(())
  }

  fn add_ndjson(&mut self, ndjson: &str) -> Result<()> {
    let docs_path = self.next_path("docs", "jsonl");
    fs::write(&docs_path, ndjson)
      .with_context(|| format!("writing CLI docs fixture to {}", docs_path.display()))?;

    self.run(&[
      "add".to_string(),
      self.index_arg(),
      docs_path.display().to_string(),
    ])?;
    Ok(())
  }

  fn commit(&mut self) -> Result<()> {
    self.run(&["commit".to_string(), self.index_arg()])?;
    Ok(())
  }

  fn search(&mut self, request: &Value) -> Result<Value> {
    let request_path = self.next_path("request", "json");
    fs::write(&request_path, serde_json::to_vec_pretty(request)?).with_context(|| {
      format!(
        "writing CLI request fixture to {}",
        request_path.as_path().display()
      )
    })?;

    let stdout = self.run(&[
      "search".to_string(),
      self.index_arg(),
      "--request".to_string(),
      request_path.display().to_string(),
    ])?;

    serde_json::from_str(stdout.as_str()).context("parsing CLI search JSON output")
  }

  fn delete_ids(&mut self, ids: &[String]) -> Result<()> {
    let ids_path = self.next_path("ids", "txt");
    fs::write(&ids_path, ids.join("\n"))
      .with_context(|| format!("writing CLI id list to {}", ids_path.display()))?;
    self.run(&[
      "delete".to_string(),
      self.index_arg(),
      ids_path.display().to_string(),
    ])?;
    Ok(())
  }

  fn inspect(&mut self) -> Result<Value> {
    let stdout = self.run(&["inspect".to_string(), self.index_arg()])?;
    let trimmed = stdout.trim();
    let json_body = trimmed.strip_prefix("manifest: ").unwrap_or(trimmed);
    let manifest: Value =
      serde_json::from_str(json_body).context("parsing CLI inspect manifest JSON")?;
    Ok(serde_json::json!({ "manifest": manifest }))
  }

  fn compact(&mut self) -> Result<()> {
    self.run(&["compact".to_string(), self.index_arg()])?;
    Ok(())
  }

  fn update_doc(&mut self, id: &str, set: &Map<String, Value>, unset: &[String]) -> Result<()> {
    // CLI `update` uses full upsert documents and does not support patch semantics.
    let _ = (id, set, unset);
    Err(unsupported_operation(self.kind(), "update"))
  }
}
