use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use reqwest::StatusCode;
use serde_json::{json, Value};
use tempfile::TempDir;

mod common;

const INDEX: &str = "demo";

struct ElasticHarness {
  upstream: Child,
  adapter: Child,
  upstream_base: String,
  adapter_base: String,
  client: reqwest::blocking::Client,
  _index_dir: TempDir,
  _upstream_log: PathBuf,
  _adapter_log: PathBuf,
}

impl ElasticHarness {
  fn new() -> Result<Self> {
    Self::with_config(&[], &[])
  }

  /// Boot the harness with the default `demo` index plus extra index mounts
  /// and aliases. Used by tests that need to exercise alias resolution or
  /// multi-index endpoints.
  fn with_config(extra_indexes: &[&str], aliases: &[(&str, &str)]) -> Result<Self> {
    let upstream_bin = common::searchlite_bin();
    let adapter_bin = elastic_bin();
    let index_dir = tempfile::tempdir().context("creating temp index dir")?;
    let index_path = index_dir.path().join("index");

    let upstream_port =
      portpicker::pick_unused_port().ok_or_else(|| anyhow!("no free port for upstream"))?;
    let adapter_port =
      portpicker::pick_unused_port().ok_or_else(|| anyhow!("no free port for adapter"))?;
    let upstream_bind = format!("127.0.0.1:{upstream_port}");
    let adapter_bind = format!("127.0.0.1:{adapter_port}");

    let upstream_log = index_path.with_extension("upstream.log");
    let adapter_log = index_path.with_extension("adapter.log");
    let upstream_log_file = std::fs::File::create(&upstream_log)
      .with_context(|| format!("creating log {}", upstream_log.display()))?;
    let adapter_log_file = std::fs::File::create(&adapter_log)
      .with_context(|| format!("creating log {}", adapter_log.display()))?;

    let mut spawn = Command::new(&upstream_bin);
    spawn.arg("http");
    spawn
      .arg("--index")
      .arg(format!("{INDEX}:{}", index_path.display()));
    for extra in extra_indexes {
      spawn.arg("--index").arg(format!(
        "{extra}:{}",
        index_dir.path().join(extra).display()
      ));
    }
    for (alias, target) in aliases {
      spawn.arg("--alias").arg(format!("{alias}:{target}"));
    }
    spawn
      .arg("--bind")
      .arg(&upstream_bind)
      .arg("--shutdown-grace-secs")
      .arg("0")
      .stdout(Stdio::null())
      .stderr(Stdio::from(upstream_log_file));

    let mut upstream = spawn
      .spawn()
      .with_context(|| format!("spawning upstream via {}", upstream_bin.display()))?;

    let client = reqwest::blocking::Client::builder()
      .timeout(Duration::from_secs(10))
      .build()
      .context("building harness client")?;

    let upstream_base = format!("http://{upstream_bind}");
    if let Err(err) = wait_for_path(&client, &format!("{upstream_base}/healthz"), &mut upstream) {
      let _ = upstream.kill();
      return Err(err);
    }

    let mut adapter = Command::new(&adapter_bin)
      .arg("--bind")
      .arg(&adapter_bind)
      .arg("--upstream-url")
      .arg(&upstream_base)
      .arg("--shutdown-grace-secs")
      .arg("0")
      .stdout(Stdio::null())
      .stderr(Stdio::from(adapter_log_file))
      .spawn()
      .with_context(|| format!("spawning adapter via {}", adapter_bin.display()))?;

    let adapter_base = format!("http://{adapter_bind}");
    if let Err(err) = wait_for_path(&client, &format!("{adapter_base}/"), &mut adapter) {
      let _ = upstream.kill();
      let _ = adapter.kill();
      return Err(err);
    }

    Ok(Self {
      upstream,
      adapter,
      upstream_base,
      adapter_base,
      client,
      _index_dir: index_dir,
      _upstream_log: upstream_log,
      _adapter_log: adapter_log,
    })
  }

  fn init_index(&self, schema: &Value) -> Result<()> {
    self.init_named(INDEX, schema)
  }

  fn init_named(&self, name: &str, schema: &Value) -> Result<()> {
    let url = format!("{}/indexes/{name}/init", self.upstream_base);
    let resp = self.client.post(url).json(schema).send()?;
    let status = resp.status();
    let text = resp.text()?;
    if !status.is_success() {
      return Err(anyhow!("init {name} failed {status}: {text}"));
    }
    Ok(())
  }

  fn add_ndjson(&self, ndjson: &str) -> Result<()> {
    let url = format!("{}/indexes/{INDEX}/add", self.upstream_base);
    let resp = self
      .client
      .post(url)
      .header("Content-Type", "application/x-ndjson")
      .body(ndjson.to_string())
      .send()?;
    if !resp.status().is_success() {
      return Err(anyhow!("add failed {}: {}", resp.status(), resp.text()?));
    }
    Ok(())
  }

  fn commit(&self) -> Result<()> {
    let url = format!("{}/indexes/{INDEX}/commit", self.upstream_base);
    let resp = self.client.post(url).send()?;
    if !resp.status().is_success() {
      return Err(anyhow!("commit failed {}: {}", resp.status(), resp.text()?));
    }
    Ok(())
  }

  fn refresh(&self) -> Result<()> {
    let url = format!("{}/indexes/{INDEX}/refresh", self.upstream_base);
    let resp = self.client.post(url).send()?;
    if !resp.status().is_success() {
      return Err(anyhow!(
        "refresh failed {}: {}",
        resp.status(),
        resp.text()?
      ));
    }
    Ok(())
  }

  fn es_get(&self, path: &str) -> Result<(StatusCode, Value)> {
    let url = format!("{}{path}", self.adapter_base);
    let resp = self.client.get(url).send()?;
    let status = resp.status();
    let body = resp.text()?;
    let parsed = if body.is_empty() {
      Value::Null
    } else {
      serde_json::from_str(&body).unwrap_or(Value::String(body))
    };
    Ok((status, parsed))
  }

  fn es_head(&self, path: &str) -> Result<StatusCode> {
    let url = format!("{}{path}", self.adapter_base);
    let resp = self.client.head(url).send()?;
    Ok(resp.status())
  }

  fn es_post(&self, path: &str, body: &Value) -> Result<(StatusCode, Value)> {
    let url = format!("{}{path}", self.adapter_base);
    let resp = self.client.post(url).json(body).send()?;
    let status = resp.status();
    let text = resp.text()?;
    let parsed = if text.is_empty() {
      Value::Null
    } else {
      serde_json::from_str(&text).unwrap_or(Value::String(text))
    };
    Ok((status, parsed))
  }

  fn es_post_ndjson(&self, path: &str, body: &str) -> Result<(StatusCode, Value)> {
    let url = format!("{}{path}", self.adapter_base);
    let resp = self
      .client
      .post(url)
      .header("Content-Type", "application/x-ndjson")
      .body(body.to_string())
      .send()?;
    let status = resp.status();
    let text = resp.text()?;
    let parsed = if text.is_empty() {
      Value::Null
    } else {
      serde_json::from_str(&text).unwrap_or(Value::String(text))
    };
    Ok((status, parsed))
  }

  fn es_put(&self, path: &str) -> Result<StatusCode> {
    let url = format!("{}{path}", self.adapter_base);
    let resp = self.client.put(url).send()?;
    Ok(resp.status())
  }
}

impl Drop for ElasticHarness {
  fn drop(&mut self) {
    let _ = self.adapter.kill();
    let _ = self.upstream.kill();
    let _ = self.adapter.wait();
    let _ = self.upstream.wait();
  }
}

fn wait_for_path(client: &reqwest::blocking::Client, url: &str, child: &mut Child) -> Result<()> {
  for _ in 0..200 {
    if let Some(status) = child.try_wait()? {
      return Err(anyhow!("process exited early with status {status}"));
    }
    if let Ok(resp) = client.get(url).send() {
      if resp.status().is_success() {
        return Ok(());
      }
    }
    thread::sleep(Duration::from_millis(50));
  }
  Err(anyhow!("process did not become healthy at {url}"))
}

fn elastic_bin() -> PathBuf {
  if let Ok(path) = std::env::var("CARGO_BIN_EXE_searchlite-elastic") {
    return PathBuf::from(path);
  }
  let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .parent()
    .expect("workspace root")
    .to_path_buf();
  let candidate = workspace_root
    .join("target")
    .join("debug")
    .join(if cfg!(windows) {
      "searchlite-elastic.exe"
    } else {
      "searchlite-elastic"
    });
  if candidate.exists() {
    return candidate;
  }
  let status = Command::new("cargo")
    .arg("build")
    .arg("-p")
    .arg("searchlite-adapter-elastic")
    .arg("--bin")
    .arg("searchlite-elastic")
    .current_dir(&workspace_root)
    .status()
    .expect("build searchlite-elastic binary");
  assert!(status.success(), "building searchlite-elastic failed");
  workspace_root
    .join("target")
    .join("debug")
    .join(if cfg!(windows) {
      "searchlite-elastic.exe"
    } else {
      "searchlite-elastic"
    })
}

fn seed_index(h: &ElasticHarness) -> Result<()> {
  let schema = json!({
    "type": "object",
    "searchlite:docIdField": "_id",
    "properties": {
      "title": { "type": "string" },
      "category": { "type": "string", "searchlite:kind": "keyword" },
      "price": { "type": "integer" },
    }
  });
  h.init_index(&schema)?;
  let ndjson = "\
    {\"_id\":\"a\",\"title\":\"rust safety\",\"category\":\"books\",\"price\":20}\n\
    {\"_id\":\"b\",\"title\":\"go concurrency\",\"category\":\"books\",\"price\":15}\n\
    {\"_id\":\"c\",\"title\":\"music history\",\"category\":\"music\",\"price\":40}\n";
  h.add_ndjson(ndjson)?;
  h.commit()?;
  h.refresh()?;
  Ok(())
}

#[test]
fn root_returns_es_version_banner() {
  let h = ElasticHarness::new().expect("harness");
  let (status, body) = h.es_get("/").expect("get /");
  assert_eq!(status, StatusCode::OK);
  let version = body.get("version").expect("version key");
  assert!(version.get("number").is_some(), "missing version.number");
  assert_eq!(body.get("tagline").unwrap(), &json!("You Know, for Search"));
}

#[test]
fn cluster_health_returns_green() {
  let h = ElasticHarness::new().expect("harness");
  let (status, body) = h.es_get("/_cluster/health").expect("get health");
  assert_eq!(status, StatusCode::OK);
  assert_eq!(body.get("status").unwrap(), &json!("green"));
}

#[test]
fn head_index_finds_known_index() {
  let h = ElasticHarness::new().expect("harness");
  seed_index(&h).expect("seed");
  let status = h.es_head(&format!("/{INDEX}")).expect("head");
  assert_eq!(status, StatusCode::OK);
}

#[test]
fn head_index_missing_returns_404() {
  let h = ElasticHarness::new().expect("harness");
  seed_index(&h).expect("seed");
  let status = h.es_head("/nope").expect("head");
  assert_eq!(status, StatusCode::NOT_FOUND);
}

#[test]
fn get_mapping_returns_es_shape() {
  let h = ElasticHarness::new().expect("harness");
  seed_index(&h).expect("seed");
  let (status, body) = h
    .es_get(&format!("/{INDEX}/_mapping"))
    .expect("get mapping");
  assert_eq!(status, StatusCode::OK);
  let mappings = body.get(INDEX).unwrap().get("mappings").unwrap();
  let props = mappings.get("properties").unwrap();
  assert_eq!(
    props.get("title").unwrap().get("type").unwrap(),
    &json!("text")
  );
  assert_eq!(
    props.get("category").unwrap().get("type").unwrap(),
    &json!("keyword")
  );
  assert_eq!(
    props.get("price").unwrap().get("type").unwrap(),
    &json!("long")
  );
}

#[test]
fn search_match_all_returns_seeded_docs() {
  let h = ElasticHarness::new().expect("harness");
  seed_index(&h).expect("seed");
  let (status, body) = h
    .es_post(
      &format!("/{INDEX}/_search"),
      &json!({ "query": { "match_all": {} }, "size": 10 }),
    )
    .expect("search");
  assert_eq!(status, StatusCode::OK, "body: {body}");
  let total = body
    .get("hits")
    .unwrap()
    .get("total")
    .unwrap()
    .get("value")
    .unwrap()
    .as_u64()
    .unwrap();
  assert_eq!(total, 3);
  let hits = body
    .get("hits")
    .unwrap()
    .get("hits")
    .unwrap()
    .as_array()
    .unwrap();
  assert_eq!(hits.len(), 3);
  for hit in hits {
    assert_eq!(hit.get("_index").unwrap(), &json!(INDEX));
    assert!(hit.get("_id").is_some());
    assert!(hit.get("_source").is_some());
  }
}

#[test]
fn search_term_query_filters() {
  let h = ElasticHarness::new().expect("harness");
  seed_index(&h).expect("seed");
  let (status, body) = h
    .es_post(
      &format!("/{INDEX}/_search"),
      &json!({ "query": { "term": { "category": "music" } }, "size": 10 }),
    )
    .expect("search");
  assert_eq!(status, StatusCode::OK, "body: {body}");
  let hits = body
    .get("hits")
    .unwrap()
    .get("hits")
    .unwrap()
    .as_array()
    .unwrap();
  assert_eq!(hits.len(), 1);
  assert_eq!(hits[0].get("_id").unwrap(), &json!("c"));
}

#[test]
fn count_returns_total() {
  let h = ElasticHarness::new().expect("harness");
  seed_index(&h).expect("seed");
  let (status, body) = h
    .es_post(
      &format!("/{INDEX}/_count"),
      &json!({ "query": { "match_all": {} } }),
    )
    .expect("count");
  assert_eq!(status, StatusCode::OK, "body: {body}");
  assert_eq!(body.get("count").unwrap().as_u64().unwrap(), 3);
}

#[test]
fn mget_by_ids() {
  let h = ElasticHarness::new().expect("harness");
  seed_index(&h).expect("seed");
  let (status, body) = h
    .es_post(&format!("/{INDEX}/_mget"), &json!({ "ids": ["a", "c"] }))
    .expect("mget");
  assert_eq!(status, StatusCode::OK, "body: {body}");
  let docs = body.get("docs").unwrap().as_array().unwrap();
  assert_eq!(docs.len(), 2);
  let a = docs
    .iter()
    .find(|d| d.get("_id") == Some(&json!("a")))
    .unwrap();
  assert_eq!(a.get("found").unwrap(), &json!(true));
  assert_eq!(a.get("_index").unwrap(), &json!(INDEX));
}

#[test]
fn msearch_runs_two_searches() {
  let h = ElasticHarness::new().expect("harness");
  seed_index(&h).expect("seed");
  let ndjson = format!(
    "{}\n{}\n{}\n{}\n",
    json!({"index": INDEX}),
    json!({"query": {"match_all": {}}}),
    json!({"index": INDEX}),
    json!({"query": {"term": {"category": "music"}}}),
  );
  let (status, body) = h.es_post_ndjson("/_msearch", &ndjson).expect("msearch");
  assert_eq!(status, StatusCode::OK, "body: {body}");
  let responses = body.get("responses").unwrap().as_array().unwrap();
  assert_eq!(responses.len(), 2);
  let total_first = responses[0]
    .get("hits")
    .unwrap()
    .get("total")
    .unwrap()
    .get("value")
    .unwrap()
    .as_u64()
    .unwrap();
  assert_eq!(total_first, 3);
  let total_second = responses[1]
    .get("hits")
    .unwrap()
    .get("total")
    .unwrap()
    .get("value")
    .unwrap()
    .as_u64()
    .unwrap();
  assert_eq!(total_second, 1);
}

#[test]
fn put_index_is_rejected_with_400() {
  let h = ElasticHarness::new().expect("harness");
  let status = h.es_put("/some-index").expect("put");
  assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[test]
fn unsupported_geo_query_is_rejected() {
  let h = ElasticHarness::new().expect("harness");
  seed_index(&h).expect("seed");
  let (status, body) = h
    .es_post(
      &format!("/{INDEX}/_search"),
      &json!({ "query": { "geo_distance": { "distance": "5km", "loc": "0,0" } } }),
    )
    .expect("search");
  assert_eq!(status, StatusCode::BAD_REQUEST);
  let kind = body
    .get("error")
    .and_then(|e| e.get("type"))
    .and_then(Value::as_str)
    .unwrap_or("");
  assert!(kind.contains("parse"), "got error kind {kind}");
}

#[test]
fn cross_index_search_path_is_rejected() {
  let h = ElasticHarness::new().expect("harness");
  let (status, _) = h
    .es_post("/_search", &json!({ "query": { "match_all": {} } }))
    .expect("post");
  assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[test]
fn mapping_all_returns_every_mounted_index() {
  // Smoke test for the all-index mapping endpoint after parallelizing the
  // per-index inspect calls. Verifies cardinality and key-naming for two
  // mounted indexes.
  let h = ElasticHarness::with_config(&["secondary"], &[]).expect("harness");
  let schema = json!({
    "type": "object",
    "searchlite:docIdField": "_id",
    "properties": { "title": { "type": "string" } }
  });
  h.init_named(INDEX, &schema).expect("init demo");
  h.init_named("secondary", &schema).expect("init secondary");
  let (status, body) = h.es_get("/_mapping").expect("get all mappings");
  assert_eq!(status, StatusCode::OK, "body: {body}");
  let map = body.as_object().expect("object body");
  assert!(map.contains_key(INDEX), "missing demo: {body}");
  assert!(map.contains_key("secondary"), "missing secondary: {body}");
  assert_eq!(map.len(), 2, "expected exactly two index entries: {body}");
}

#[test]
fn settings_for_alias_returns_target_index_payload() {
  // Regression: get_settings used to fabricate a settings payload keyed by
  // the request path token, even when that token was an alias name.
  // Per ES, alias requests should resolve to the concrete target index and
  // key the response by the target — not the alias.
  let h = ElasticHarness::with_config(&[], &[("demo_alias", INDEX)]).expect("harness with alias");
  seed_index(&h).expect("seed");
  let (status, body) = h.es_get("/demo_alias/_settings").expect("get settings");
  assert_eq!(status, StatusCode::OK, "body: {body}");
  assert!(
    body.get(INDEX).is_some(),
    "settings should be keyed by the target index `{INDEX}`, got: {body}"
  );
  assert!(
    body.get("demo_alias").is_none(),
    "settings must not be keyed by the alias name; got: {body}"
  );
}

#[test]
fn settings_on_unknown_index_returns_404() {
  let h = ElasticHarness::new().expect("harness");
  let (status, body) = h.es_get("/no-such-index/_settings").expect("get");
  assert_eq!(status, StatusCode::NOT_FOUND);
  let kind = body
    .get("error")
    .and_then(|e| e.get("type"))
    .and_then(Value::as_str)
    .unwrap_or("");
  assert_eq!(kind, "index_not_found_exception");
}

#[test]
fn msearch_with_non_string_in_index_array_is_rejected() {
  // Regression: the index-array parser dropped non-string elements via
  // filter_map, so a malformed header like {"index":["demo",42]} was
  // silently accepted as a single-index ("demo") request, possibly
  // routing queries to an unintended index.
  let h = ElasticHarness::new().expect("harness");
  seed_index(&h).expect("seed");
  let ndjson = format!(
    "{}\n{}\n",
    json!({"index": [INDEX, 42]}),
    json!({"query": {"match_all": {}}}),
  );
  let (status, body) = h.es_post_ndjson("/_msearch", &ndjson).expect("post");
  assert_eq!(status, StatusCode::BAD_REQUEST, "body: {body}");
}

#[test]
fn msearch_with_single_string_in_index_array_still_works() {
  let h = ElasticHarness::new().expect("harness");
  seed_index(&h).expect("seed");
  let ndjson = format!(
    "{}\n{}\n",
    json!({"index": [INDEX]}),
    json!({"query": {"match_all": {}}}),
  );
  let (status, body) = h.es_post_ndjson("/_msearch", &ndjson).expect("post");
  assert_eq!(status, StatusCode::OK, "body: {body}");
}

#[test]
fn global_mget_with_source_false_omits_source_field() {
  // Regression: the global `_mget` handler always forwarded
  // `return_stored: true` to upstream, ignoring `_source: false` on the
  // request. The index-scoped handler honors it; this brings the global
  // form into parity.
  let h = ElasticHarness::new().expect("harness");
  seed_index(&h).expect("seed");
  let (status, body) = h
    .es_post(
      "/_mget",
      &json!({
        "_source": false,
        "docs": [
          { "_index": INDEX, "_id": "a" },
          { "_index": INDEX, "_id": "c" },
        ]
      }),
    )
    .expect("mget");
  assert_eq!(status, StatusCode::OK, "body: {body}");
  let docs = body.get("docs").unwrap().as_array().unwrap();
  assert_eq!(docs.len(), 2);
  for doc in docs {
    assert!(
      doc.get("_source").is_none(),
      "_source should be omitted when _source:false is requested; got {doc}"
    );
    assert_eq!(doc.get("found").unwrap(), &json!(true));
  }
}

#[test]
fn global_mget_with_source_true_includes_source_field() {
  let h = ElasticHarness::new().expect("harness");
  seed_index(&h).expect("seed");
  let (status, body) = h
    .es_post(
      "/_mget",
      &json!({
        "_source": true,
        "docs": [{ "_index": INDEX, "_id": "a" }]
      }),
    )
    .expect("mget");
  assert_eq!(status, StatusCode::OK, "body: {body}");
  let docs = body.get("docs").unwrap().as_array().unwrap();
  assert!(
    docs[0].get("_source").is_some(),
    "_source should be present when _source:true is requested"
  );
}

#[test]
fn search_get_sort_param_with_spaces_after_commas_is_parsed_cleanly() {
  // Regression: previously `?sort=foo:desc, _score` produced a field literally
  // named " _score" (leading space) because tokens were not trimmed after the
  // comma split. Both human-typed URLs and some SDKs emit spaces after commas.
  let h = ElasticHarness::new().expect("harness");
  seed_index(&h).expect("seed");
  let (status, body) = h
    .es_get(&format!("/{INDEX}/_search?sort=price:desc, _score&size=3"))
    .expect("get");
  assert_eq!(status, StatusCode::OK, "body: {body}");
  // Top hit should be the most expensive doc per the price:desc sort ordering.
  let first_id = body
    .pointer("/hits/hits/0/_id")
    .and_then(Value::as_str)
    .unwrap_or("");
  assert!(!first_id.is_empty(), "expected a top hit; got: {body}");
}

#[test]
fn search_get_sort_param_with_space_after_colon_is_parsed_cleanly() {
  // Both `field` and `order` should be trimmed.
  let h = ElasticHarness::new().expect("harness");
  seed_index(&h).expect("seed");
  let (status, body) = h
    .es_get(&format!("/{INDEX}/_search?sort=price : desc&size=3"))
    .expect("get");
  assert_eq!(status, StatusCode::OK, "body: {body}");
}

#[test]
fn search_get_with_integer_track_total_hits_query_param_is_accepted() {
  // Regression: ES allows `?track_total_hits=10000` (integer cap). Previously
  // the adapter typed the URL param as Option<bool> and rejected the integer
  // form during query-string deserialization before the handler even ran.
  let h = ElasticHarness::new().expect("harness");
  seed_index(&h).expect("seed");
  let (status, body) = h
    .es_get(&format!("/{INDEX}/_search?q=*&track_total_hits=10000"))
    .expect("get");
  assert_eq!(status, StatusCode::OK, "body: {body}");
}

#[test]
fn search_get_with_boolean_track_total_hits_query_param_is_accepted() {
  let h = ElasticHarness::new().expect("harness");
  seed_index(&h).expect("seed");
  let (status, _) = h
    .es_get(&format!("/{INDEX}/_search?q=*&track_total_hits=true"))
    .expect("get");
  assert_eq!(status, StatusCode::OK);
}

#[test]
fn settings_on_known_index_returns_payload() {
  let h = ElasticHarness::new().expect("harness");
  seed_index(&h).expect("seed");
  let (status, body) = h
    .es_get(&format!("/{INDEX}/_settings"))
    .expect("get settings");
  assert_eq!(status, StatusCode::OK);
  assert!(body.get(INDEX).is_some(), "expected `{INDEX}` key in body");
}

#[test]
fn aliases_for_index_with_no_aliases_returns_empty_entry() {
  let h = ElasticHarness::new().expect("harness");
  seed_index(&h).expect("seed");
  let (status, body) = h
    .es_get(&format!("/{INDEX}/_aliases"))
    .expect("get aliases");
  assert_eq!(status, StatusCode::OK);
  // Existing index with no aliases configured → ES returns
  // `{<index>: {aliases: {}}}`. Verifies the handler is path-scoped (would
  // return all indexes if it weren't) and that empty-alias indexes don't 404.
  assert_eq!(body, json!({ INDEX: { "aliases": {} } }), "got {body}",);
}

#[test]
fn aliases_for_unknown_index_returns_404() {
  let h = ElasticHarness::new().expect("harness");
  let (status, body) = h.es_get("/no-such-index/_aliases").expect("get");
  assert_eq!(status, StatusCode::NOT_FOUND);
  let kind = body
    .get("error")
    .and_then(|e| e.get("type"))
    .and_then(Value::as_str)
    .unwrap_or("");
  assert_eq!(kind, "index_not_found_exception");
}
