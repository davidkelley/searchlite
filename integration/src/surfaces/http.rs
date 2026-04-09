use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use reqwest::StatusCode;
use serde_json::{Map, Value};

use super::{SurfaceHarness, SurfaceKind};

const HTTP_INDEX_NAME: &str = "primary";

pub struct HttpHarness {
  _bin: PathBuf,
  _index_path: PathBuf,
  _base_url: String,
  index_base_url: String,
  client: reqwest::blocking::Client,
  child: Child,
  stderr_path: PathBuf,
}

impl HttpHarness {
  pub fn new(bin: PathBuf, index_path: PathBuf) -> Result<Self> {
    let port = portpicker::pick_unused_port().ok_or_else(|| anyhow!("failed to pick free port"))?;
    let bind = format!("127.0.0.1:{port}");

    let stderr_path = index_path.with_extension("stderr.log");
    let stderr_file = std::fs::File::create(&stderr_path)
      .with_context(|| format!("creating stderr log at {}", stderr_path.display()))?;

    let mut child = Command::new(&bin)
      .arg("http")
      .arg("--index")
      .arg(format!("{HTTP_INDEX_NAME}:{}", index_path.display()))
      .arg("--bind")
      .arg(bind.clone())
      .arg("--shutdown-grace-secs")
      .arg("0")
      .stdout(Stdio::null())
      .stderr(Stdio::from(stderr_file))
      .spawn()
      .with_context(|| format!("spawning HTTP server via {}", bin.display()))?;

    let client = reqwest::blocking::Client::builder()
      .timeout(Duration::from_secs(10))
      .build()
      .context("building HTTP harness client")?;

    let base_url = format!("http://{bind}");
    wait_for_health(&client, base_url.as_str(), &mut child)?;
    let index_base_url = format!("{base_url}/indexes/{HTTP_INDEX_NAME}");

    Ok(Self {
      _bin: bin,
      _index_path: index_path,
      _base_url: base_url,
      index_base_url,
      client,
      child,
      stderr_path,
    })
  }

  fn post_json(&self, path: &str, payload: &Value) -> Result<Value> {
    let response = self
      .client
      .post(format!("{}/{}", self.index_base_url, path))
      .json(payload)
      .send()
      .with_context(|| format!("POST {}/{}", self.index_base_url, path))?;
    parse_json_response(response, format!("POST {path}"))
  }

  fn post_ndjson(&self, path: &str, ndjson: &str) -> Result<Value> {
    let response = self
      .client
      .post(format!("{}/{}", self.index_base_url, path))
      .header("Content-Type", "application/x-ndjson")
      .body(ndjson.to_string())
      .send()
      .with_context(|| format!("POST {}/{}", self.index_base_url, path))?;
    parse_json_response(response, format!("POST {path}"))
  }

  fn post_empty(&self, path: &str) -> Result<Value> {
    let response = self
      .client
      .post(format!("{}/{}", self.index_base_url, path))
      .send()
      .with_context(|| format!("POST {}/{}", self.index_base_url, path))?;
    parse_json_response(response, format!("POST {path}"))
  }

  fn get_json(&self, path: &str) -> Result<Value> {
    let response = self
      .client
      .get(format!("{}/{}", self.index_base_url, path))
      .send()
      .with_context(|| format!("GET {}/{}", self.index_base_url, path))?;
    parse_json_response(response, format!("GET {path}"))
  }

  pub fn base_url(&self) -> &str {
    self._base_url.as_str()
  }

  pub fn index_base_url(&self) -> &str {
    self.index_base_url.as_str()
  }

  /// Read the HTTP server's stderr log. Useful for debugging server failures.
  #[allow(dead_code)]
  pub fn read_stderr_log(&self) -> String {
    std::fs::read_to_string(&self.stderr_path).unwrap_or_default()
  }
}

impl Drop for HttpHarness {
  fn drop(&mut self) {
    let _ = self.child.kill();
    let _ = self.child.wait();
  }
}

impl SurfaceHarness for HttpHarness {
  fn kind(&self) -> SurfaceKind {
    SurfaceKind::Http
  }

  fn init(&mut self, schema: &Value) -> Result<()> {
    let _ = self.post_json("init", schema)?;
    Ok(())
  }

  fn add_ndjson(&mut self, ndjson: &str) -> Result<()> {
    let _ = self.post_ndjson("add", ndjson)?;
    Ok(())
  }

  fn commit(&mut self) -> Result<()> {
    let _ = self.post_empty("commit")?;
    Ok(())
  }

  fn refresh(&mut self) -> Result<()> {
    let _ = self.post_empty("refresh")?;
    Ok(())
  }

  fn search(&mut self, request: &Value) -> Result<Value> {
    self.post_json("search", request)
  }

  fn mget(&mut self, ids: &[String], return_stored: bool) -> Result<Value> {
    self.post_json(
      "mget",
      &serde_json::json!({
        "ids": ids,
        "return_stored": return_stored,
      }),
    )
  }

  fn update_doc(&mut self, id: &str, set: &Map<String, Value>, unset: &[String]) -> Result<()> {
    let _ = self.post_json(
      "update",
      &serde_json::json!({
        "id": id,
        "set": set,
        "unset": unset,
      }),
    )?;
    Ok(())
  }

  fn delete_ids(&mut self, ids: &[String]) -> Result<()> {
    let _ = self.post_json("delete", &serde_json::json!({ "ids": ids }))?;
    Ok(())
  }

  fn stats(&mut self) -> Result<Value> {
    self.get_json("stats")
  }

  fn inspect(&mut self) -> Result<Value> {
    self.get_json("inspect")
  }

  fn compact(&mut self) -> Result<()> {
    let _ = self.post_empty("compact")?;
    Ok(())
  }
}

fn wait_for_health(
  client: &reqwest::blocking::Client,
  base_url: &str,
  child: &mut Child,
) -> Result<()> {
  let health_url = format!("{base_url}/healthz");
  for _ in 0..100 {
    if let Some(status) = child.try_wait().context("checking server process")? {
      return Err(anyhow!(
        "HTTP harness process exited early with status {status}"
      ));
    }

    if let Ok(resp) = client.get(health_url.as_str()).send() {
      if resp.status() == StatusCode::OK {
        return Ok(());
      }
    }
    thread::sleep(Duration::from_millis(50));
  }

  Err(anyhow!(
    "HTTP harness did not become healthy at {health_url} before timeout"
  ))
}

fn parse_json_response(response: reqwest::blocking::Response, operation: String) -> Result<Value> {
  let status = response.status();
  let body = response
    .text()
    .with_context(|| format!("reading response body for {operation}"))?;

  if !status.is_success() {
    return Err(anyhow!(
      "{operation} returned {} with body: {}",
      status,
      body
    ));
  }

  serde_json::from_str(&body).with_context(|| format!("parsing JSON body for {operation}"))
}
