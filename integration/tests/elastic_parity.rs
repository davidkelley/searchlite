//! Parity test: load identical corpora into a real Elasticsearch 9 container
//! and into a searchlite-elastic adapter pointed at a searchlite-http upstream,
//! then run the same queries against both and assert their semantically-relevant
//! responses match.
//!
//! Auto-skips when Docker is unavailable. CI runs Docker on ubuntu-latest by
//! default, so this test runs there without extra setup.

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use reqwest::StatusCode;
use serde_json::{json, Value};
use tempfile::TempDir;

mod common;

const ES_IMAGE: &str = "docker.elastic.co/elasticsearch/elasticsearch:9.0.0";
const INDEX: &str = "parity";
const ES_BOOT_TIMEOUT: Duration = Duration::from_secs(180);
const SL_BOOT_TIMEOUT: Duration = Duration::from_secs(15);

// ── Docker availability gate ─────────────────────────────────────────────────

fn docker_available() -> bool {
  Command::new("docker")
    .arg("info")
    .stdout(Stdio::null())
    .stderr(Stdio::null())
    .status()
    .map(|s| s.success())
    .unwrap_or(false)
}

fn skip_unless_docker(test_name: &str) -> Option<()> {
  if docker_available() {
    return Some(());
  }
  eprintln!("[skip] {test_name}: docker is not available — start Docker to run this test");
  None
}

// ── Real Elasticsearch container ─────────────────────────────────────────────

struct EsContainer {
  name: String,
  base_url: String,
}

impl EsContainer {
  fn start() -> Result<Self> {
    pull_image(ES_IMAGE)?;
    let port = portpicker::pick_unused_port().ok_or_else(|| anyhow!("no free port for ES"))?;
    // Container name needs to be unique per test process to avoid
    // collisions. PID alone isn't enough — `cargo test` reuses PIDs across
    // runs, and on CI a stale container with the same PID-derived name from
    // a previous job could exist. Append nanosecond-precision time so two
    // concurrent test processes (or a fresh run racing a slow `docker rm`)
    // get distinct names.
    let nanos = std::time::SystemTime::now()
      .duration_since(std::time::UNIX_EPOCH)
      .map(|d| d.subsec_nanos())
      .unwrap_or(0);
    let name = format!("searchlite-es-parity-{}-{}", std::process::id(), nanos);
    // Best-effort cleanup if a previous run left a stale container around.
    let _ = Command::new("docker")
      .args(["rm", "-f", &name])
      .stdout(Stdio::null())
      .stderr(Stdio::null())
      .status();

    let status = Command::new("docker")
      .args([
        "run",
        "-d",
        "--rm",
        "--name",
        &name,
        "-p",
        &format!("{port}:9200"),
        "-e",
        "discovery.type=single-node",
        "-e",
        "xpack.security.enabled=false",
        "-e",
        "xpack.security.http.ssl.enabled=false",
        "-e",
        "ES_JAVA_OPTS=-Xms512m -Xmx512m",
        ES_IMAGE,
      ])
      .stdout(Stdio::null())
      .stderr(Stdio::piped())
      .output()
      .context("running `docker run` for ES")?;
    if !status.status.success() {
      bail!(
        "docker run failed: {}",
        String::from_utf8_lossy(&status.stderr)
      );
    }

    let base_url = format!("http://127.0.0.1:{port}");
    let client = blocking_client();
    let started = Instant::now();
    loop {
      match client.get(format!("{base_url}/_cluster/health")).send() {
        Ok(resp) if resp.status().is_success() => {
          if let Ok(body) = resp.json::<Value>() {
            let status = body.get("status").and_then(Value::as_str).unwrap_or("");
            if status == "yellow" || status == "green" {
              return Ok(Self { name, base_url });
            }
          }
        }
        _ => {}
      }
      if started.elapsed() > ES_BOOT_TIMEOUT {
        // Capture container logs for diagnostics before dropping.
        let logs = Command::new("docker")
          .args(["logs", "--tail", "80", &name])
          .output()
          .map(|o| {
            format!(
              "stdout: {}\nstderr: {}",
              String::from_utf8_lossy(&o.stdout),
              String::from_utf8_lossy(&o.stderr)
            )
          })
          .unwrap_or_else(|err| format!("could not read logs: {err}"));
        let _ = Command::new("docker")
          .args(["rm", "-f", &name])
          .stdout(Stdio::null())
          .stderr(Stdio::null())
          .status();
        bail!(
          "elasticsearch did not become healthy within {:?}\n{logs}",
          ES_BOOT_TIMEOUT
        );
      }
      thread::sleep(Duration::from_millis(500));
    }
  }
}

impl Drop for EsContainer {
  fn drop(&mut self) {
    let _ = Command::new("docker")
      .args(["rm", "-f", &self.name])
      .stdout(Stdio::null())
      .stderr(Stdio::null())
      .status();
  }
}

fn pull_image(image: &str) -> Result<()> {
  let output = Command::new("docker")
    .args(["pull", image])
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .output()
    .context("invoking `docker pull`")?;
  if !output.status.success() {
    bail!(
      "docker pull {image} failed: {}",
      String::from_utf8_lossy(&output.stderr)
    );
  }
  Ok(())
}

fn blocking_client() -> reqwest::blocking::Client {
  reqwest::blocking::Client::builder()
    .timeout(Duration::from_secs(15))
    .build()
    .expect("build reqwest client")
}

// ── searchlite + adapter pair ────────────────────────────────────────────────

struct AdapterStack {
  upstream: Child,
  adapter: Child,
  upstream_base: String,
  adapter_base: String,
  _index_dir: TempDir,
  _logs: (PathBuf, PathBuf),
}

impl AdapterStack {
  fn start() -> Result<Self> {
    let upstream_bin = common::searchlite_bin();
    let adapter_bin = elastic_bin();
    let index_dir = tempfile::tempdir().context("create temp index dir")?;
    let index_path = index_dir.path().join("index");

    let upstream_port =
      portpicker::pick_unused_port().ok_or_else(|| anyhow!("no port for upstream"))?;
    let adapter_port =
      portpicker::pick_unused_port().ok_or_else(|| anyhow!("no port for adapter"))?;
    let upstream_bind = format!("127.0.0.1:{upstream_port}");
    let adapter_bind = format!("127.0.0.1:{adapter_port}");

    let upstream_log = index_path.with_extension("upstream.log");
    let adapter_log = index_path.with_extension("adapter.log");
    let upstream_log_file = std::fs::File::create(&upstream_log)?;
    let adapter_log_file = std::fs::File::create(&adapter_log)?;

    let mut upstream = Command::new(&upstream_bin)
      .arg("http")
      .arg("--index")
      .arg(format!("{INDEX}:{}", index_path.display()))
      .arg("--bind")
      .arg(&upstream_bind)
      .arg("--shutdown-grace-secs")
      .arg("0")
      .stdout(Stdio::null())
      .stderr(Stdio::from(upstream_log_file))
      .spawn()
      .with_context(|| format!("spawn upstream {}", upstream_bin.display()))?;

    let client = blocking_client();
    let upstream_base = format!("http://{upstream_bind}");
    if let Err(err) = wait_for_path(
      &client,
      &format!("{upstream_base}/healthz"),
      &mut upstream,
      SL_BOOT_TIMEOUT,
    ) {
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
      .with_context(|| format!("spawn adapter {}", adapter_bin.display()))?;

    let adapter_base = format!("http://{adapter_bind}");
    if let Err(err) = wait_for_path(
      &client,
      &format!("{adapter_base}/"),
      &mut adapter,
      SL_BOOT_TIMEOUT,
    ) {
      let _ = upstream.kill();
      let _ = adapter.kill();
      return Err(err);
    }

    Ok(Self {
      upstream,
      adapter,
      upstream_base,
      adapter_base,
      _index_dir: index_dir,
      _logs: (upstream_log, adapter_log),
    })
  }
}

impl Drop for AdapterStack {
  fn drop(&mut self) {
    let _ = self.adapter.kill();
    let _ = self.upstream.kill();
    let _ = self.adapter.wait();
    let _ = self.upstream.wait();
  }
}

fn wait_for_path(
  client: &reqwest::blocking::Client,
  url: &str,
  child: &mut Child,
  timeout: Duration,
) -> Result<()> {
  let start = Instant::now();
  while start.elapsed() < timeout {
    if let Some(status) = child.try_wait()? {
      bail!("process exited early with status {status}");
    }
    if let Ok(resp) = client.get(url).send() {
      if resp.status().is_success() {
        return Ok(());
      }
    }
    thread::sleep(Duration::from_millis(50));
  }
  bail!("process did not become healthy at {url}");
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
    .expect("build searchlite-elastic");
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

// ── Corpus + schemas ─────────────────────────────────────────────────────────

fn corpus() -> Vec<Value> {
  // Each doc has a `description` field of long-form text so we can exercise
  // multi-field matching, longer-document relevance, stopword behaviour and
  // plural/singular tokenization probes alongside the short-title cases.
  vec![
    json!({
      "_id": "1",
      "title": "rust safety guide",
      "description": "A practical introduction to writing memory-safe code in rust. Covers ownership, borrowing, and common pitfalls for systems programmers learning the language for the first time.",
      "category": "books",
      "price": 25,
      "rating": 4.5
    }),
    json!({
      "_id": "2",
      "title": "go concurrency patterns",
      "description": "Patterns for managing goroutines, channels, and synchronization in modern services. The book covers worker pools, fan-in pipelines, and graceful shutdown of long-running daemons.",
      "category": "books",
      "price": 30,
      "rating": 4.2
    }),
    json!({
      "_id": "3",
      "title": "music history vol 1",
      "description": "From the early classical period through the baroque and into the romantic era of European music. Composer biographies and notation samples included.",
      "category": "music",
      "price": 15,
      "rating": 3.8
    }),
    json!({
      "_id": "4",
      "title": "music history vol 2",
      "description": "Modern jazz, rock, and electronic music traditions of the twentieth century. Discusses the rise of recording technology and its influence on composition.",
      "category": "music",
      "price": 18,
      "rating": 4.0
    }),
    json!({
      "_id": "5",
      "title": "kitchen essentials",
      "description": "Knives, pans, and cutting boards every home cook needs in the kitchen. Reviews of starter kits and care instructions to keep the gear sharp.",
      "category": "kitchen",
      "price": 50,
      "rating": 4.7
    }),
    json!({
      "_id": "6",
      "title": "kitchen tools advanced",
      "description": "Specialized tools for advanced cooking techniques such as sous vide, smoking, and fermentation. Recommended brands and storage tips for a working kitchen.",
      "category": "kitchen",
      "price": 75,
      "rating": 4.3
    }),
    json!({
      "_id": "7",
      "title": "rust web frameworks",
      "description": "Building production web services with axum, actix, and rocket. Performance comparisons, deployment guidance, and migration paths for systems written in other languages.",
      "category": "books",
      "price": 35,
      "rating": 4.6
    }),
    json!({
      "_id": "8",
      "title": "guitar basics",
      "description": "Learn to play guitar from scratch. Chords, strumming patterns, and your first songs. No prior music theory required.",
      "category": "music",
      "price": 22,
      "rating": 4.1
    }),
    json!({
      "_id": "9",
      "title": "advanced cooking",
      "description": "Sauces, stocks, and braising. Techniques borrowed from professional kitchens, presented for the home cook willing to spend an afternoon with a good book.",
      "category": "kitchen",
      "price": 65,
      "rating": 4.8
    }),
    json!({
      "_id": "10",
      "title": "rust async deep dive",
      "description": "Tokio, futures, and the async runtime model. For experienced rust developers ready to reason about cancellation, structured concurrency, and performance.",
      "category": "books",
      "price": 40,
      "rating": 4.9
    }),
  ]
}

fn es_mapping() -> Value {
  json!({
    "settings": { "number_of_shards": 1, "number_of_replicas": 0 },
    "mappings": {
      "properties": {
        "title": { "type": "text" },
        "description": { "type": "text" },
        "category": { "type": "keyword" },
        "price": { "type": "long" },
        "rating": { "type": "double" }
      }
    }
  })
}

fn searchlite_schema() -> Value {
  json!({
    "type": "object",
    "searchlite:docIdField": "_id",
    "properties": {
      "title": { "type": "string" },
      "description": { "type": "string" },
      "category": { "type": "string", "searchlite:kind": "keyword" },
      "price": { "type": "integer" },
      "rating": { "type": "number" }
    }
  })
}

// ── Ingest helpers ───────────────────────────────────────────────────────────

fn ingest_into_es(client: &reqwest::blocking::Client, base: &str, docs: &[Value]) -> Result<()> {
  let resp = client
    .put(format!("{base}/{INDEX}"))
    .json(&es_mapping())
    .send()?;
  if !resp.status().is_success() {
    bail!("ES create index failed: {} {}", resp.status(), resp.text()?);
  }

  let mut bulk = String::new();
  for doc in docs {
    let id = doc.get("_id").and_then(Value::as_str).unwrap();
    let mut body = doc.clone();
    body.as_object_mut().unwrap().remove("_id");
    bulk.push_str(&format!(
      r#"{{"index":{{"_index":"{INDEX}","_id":"{id}"}}}}"#
    ));
    bulk.push('\n');
    bulk.push_str(&serde_json::to_string(&body)?);
    bulk.push('\n');
  }
  let resp = client
    .post(format!("{base}/_bulk"))
    .header("Content-Type", "application/x-ndjson")
    .body(bulk)
    .send()?;
  if !resp.status().is_success() {
    bail!("ES bulk failed: {} {}", resp.status(), resp.text()?);
  }
  let body: Value = resp.json()?;
  if body.get("errors").and_then(Value::as_bool).unwrap_or(false) {
    bail!("ES bulk reported errors: {body}");
  }
  let resp = client.post(format!("{base}/{INDEX}/_refresh")).send()?;
  if !resp.status().is_success() {
    bail!("ES refresh failed: {} {}", resp.status(), resp.text()?);
  }
  Ok(())
}

fn ingest_into_searchlite(
  client: &reqwest::blocking::Client,
  base: &str,
  docs: &[Value],
) -> Result<()> {
  let resp = client
    .post(format!("{base}/indexes/{INDEX}/init"))
    .json(&searchlite_schema())
    .send()?;
  if !resp.status().is_success() {
    bail!("searchlite init failed: {} {}", resp.status(), resp.text()?);
  }
  let mut ndjson = String::new();
  for doc in docs {
    ndjson.push_str(&serde_json::to_string(doc)?);
    ndjson.push('\n');
  }
  let resp = client
    .post(format!("{base}/indexes/{INDEX}/add"))
    .header("Content-Type", "application/x-ndjson")
    .body(ndjson)
    .send()?;
  if !resp.status().is_success() {
    bail!("searchlite add failed: {} {}", resp.status(), resp.text()?);
  }
  let resp = client
    .post(format!("{base}/indexes/{INDEX}/commit"))
    .send()?;
  if !resp.status().is_success() {
    bail!(
      "searchlite commit failed: {} {}",
      resp.status(),
      resp.text()?
    );
  }
  let resp = client
    .post(format!("{base}/indexes/{INDEX}/refresh"))
    .send()?;
  if !resp.status().is_success() {
    bail!(
      "searchlite refresh failed: {} {}",
      resp.status(),
      resp.text()?
    );
  }
  Ok(())
}

// ── Query suite + parity projections ─────────────────────────────────────────

#[derive(Debug, Clone)]
enum Parity {
  /// Compare the unordered set of returned hit IDs.
  HitIdSet,
  /// Compare the ordered list of returned hit IDs (use only when sorting is stable).
  OrderedHitIds,
  /// Compare the count returned by `_count`.
  Count,
  /// Compare bucket key→doc_count map for a named terms aggregation.
  TermsAggBuckets(&'static str),
  /// Compare count/min/max/sum (avg derived) of a stats aggregation.
  StatsAgg(&'static str),
  /// Relevance check for full-text queries:
  /// - both sides' top-K hits must each be a subset of `relevant`
  ///   (catches spurious matches on either side)
  /// - both sides' top-K must overlap by at least `min_overlap`
  ///   (catches engines disagreeing on which docs match)
  ///
  /// Scoring/ordering differences below the top-K threshold are tolerated
  /// because BM25 implementations legitimately vary.
  Relevance {
    top_k: usize,
    relevant: &'static [&'static str],
    min_overlap: usize,
  },
  /// Both sides' top-1 hit must be one of the allowlisted IDs. Use when the
  /// query has an obvious best answer (e.g. exact title match).
  Top1OneOf(&'static [&'static str]),
}

struct Case {
  name: &'static str,
  endpoint: Endpoint,
  body: Value,
  parity: Parity,
}

#[derive(Debug, Clone, Copy)]
enum Endpoint {
  Search,
  Count,
}

impl Endpoint {
  fn path(&self) -> &'static str {
    match self {
      Endpoint::Search => "_search",
      Endpoint::Count => "_count",
    }
  }
}

fn cases() -> Vec<Case> {
  vec![
    Case {
      name: "match_all returns all docs",
      endpoint: Endpoint::Search,
      body: json!({ "query": { "match_all": {} }, "size": 50 }),
      parity: Parity::HitIdSet,
    },
    Case {
      name: "term filter on category=books",
      endpoint: Endpoint::Search,
      body: json!({
        "query": { "term": { "category": "books" } },
        "size": 50
      }),
      parity: Parity::HitIdSet,
    },
    Case {
      name: "term filter on category=music",
      endpoint: Endpoint::Search,
      body: json!({
        "query": { "term": { "category": "music" } },
        "size": 50
      }),
      parity: Parity::HitIdSet,
    },
    Case {
      name: "terms filter on multiple categories",
      endpoint: Endpoint::Search,
      body: json!({
        "query": {
          "bool": {
            "filter": [{ "terms": { "category": ["books", "music"] } }]
          }
        },
        "size": 50
      }),
      parity: Parity::HitIdSet,
    },
    Case {
      name: "range filter price 30..=60",
      endpoint: Endpoint::Search,
      body: json!({
        "query": {
          "bool": {
            "filter": [{ "range": { "price": { "gte": 30, "lte": 60 } } }]
          }
        },
        "size": 50
      }),
      parity: Parity::HitIdSet,
    },
    Case {
      name: "bool must match + filter category",
      endpoint: Endpoint::Search,
      body: json!({
        "query": {
          "bool": {
            "must": [{ "match": { "title": "rust" } }],
            "filter": [{ "term": { "category": "books" } }]
          }
        },
        "size": 50
      }),
      parity: Parity::HitIdSet,
    },
    Case {
      name: "sort by price asc",
      endpoint: Endpoint::Search,
      body: json!({
        "query": { "match_all": {} },
        "sort": [{ "price": "asc" }],
        "size": 50
      }),
      parity: Parity::OrderedHitIds,
    },
    Case {
      name: "sort by price desc",
      endpoint: Endpoint::Search,
      body: json!({
        "query": { "match_all": {} },
        "sort": [{ "price": "desc" }],
        "size": 50
      }),
      parity: Parity::OrderedHitIds,
    },
    Case {
      name: "_count with term filter",
      endpoint: Endpoint::Count,
      body: json!({ "query": { "term": { "category": "kitchen" } } }),
      parity: Parity::Count,
    },
    Case {
      name: "terms agg on category",
      endpoint: Endpoint::Search,
      body: json!({
        "query": { "match_all": {} },
        "size": 0,
        "aggs": {
          "by_cat": { "terms": { "field": "category", "size": 10 } }
        }
      }),
      parity: Parity::TermsAggBuckets("by_cat"),
    },
    Case {
      name: "stats agg on price",
      endpoint: Endpoint::Search,
      body: json!({
        "query": { "match_all": {} },
        "size": 0,
        "aggs": {
          "price_stats": { "stats": { "field": "price" } }
        }
      }),
      parity: Parity::StatsAgg("price_stats"),
    },
    // ── Full-text relevance cases ──────────────────────────────────────────
    // Corpus reminder (id → title):
    //   1: "rust safety guide"          7: "rust web frameworks"
    //   2: "go concurrency patterns"    8: "guitar basics"
    //   3: "music history vol 1"        9: "advanced cooking"
    //   4: "music history vol 2"       10: "rust async deep dive"
    //   5: "kitchen essentials"
    //   6: "kitchen tools advanced"
    Case {
      name: "match: rust → only rust-titled docs",
      endpoint: Endpoint::Search,
      body: json!({
        "query": { "match": { "title": "rust" } },
        "size": 3
      }),
      parity: Parity::Relevance {
        top_k: 3,
        relevant: &["1", "7", "10"],
        min_overlap: 3,
      },
    },
    Case {
      name: "match: kitchen → only kitchen-titled docs",
      endpoint: Endpoint::Search,
      body: json!({
        "query": { "match": { "title": "kitchen" } },
        "size": 5
      }),
      parity: Parity::Relevance {
        top_k: 2,
        relevant: &["5", "6"],
        min_overlap: 2,
      },
    },
    Case {
      name: "match: music → only docs with 'music' in title",
      endpoint: Endpoint::Search,
      body: json!({
        "query": { "match": { "title": "music" } },
        "size": 5
      }),
      parity: Parity::Relevance {
        top_k: 2,
        relevant: &["3", "4"],
        min_overlap: 2,
      },
    },
    Case {
      name: "match: advanced → only docs with 'advanced'",
      endpoint: Endpoint::Search,
      body: json!({
        "query": { "match": { "title": "advanced" } },
        "size": 5
      }),
      parity: Parity::Relevance {
        top_k: 2,
        relevant: &["6", "9"],
        min_overlap: 2,
      },
    },
    Case {
      name: "match: 'rust safety' → top-1 must be the doc with both terms",
      endpoint: Endpoint::Search,
      body: json!({
        "query": { "match": { "title": "rust safety" } },
        "size": 1
      }),
      parity: Parity::Top1OneOf(&["1"]),
    },
    Case {
      name: "match: RUST (uppercase) → still finds rust docs (case-insensitive)",
      endpoint: Endpoint::Search,
      body: json!({
        "query": { "match": { "title": "RUST" } },
        "size": 3
      }),
      parity: Parity::Relevance {
        top_k: 3,
        relevant: &["1", "7", "10"],
        min_overlap: 3,
      },
    },
    Case {
      name: "match_phrase: 'music history' → only the two phrase matches",
      endpoint: Endpoint::Search,
      body: json!({
        "query": { "match_phrase": { "title": "music history" } },
        "size": 5
      }),
      parity: Parity::Relevance {
        top_k: 2,
        relevant: &["3", "4"],
        min_overlap: 2,
      },
    },
    Case {
      name: "match: 'rust' inside bool with category filter — top hits all rust+books",
      endpoint: Endpoint::Search,
      body: json!({
        "query": {
          "bool": {
            "must": [{ "match": { "title": "rust" } }],
            "filter": [{ "term": { "category": "books" } }]
          }
        },
        "size": 3
      }),
      parity: Parity::Relevance {
        top_k: 3,
        relevant: &["1", "7", "10"],
        min_overlap: 3,
      },
    },
    // ── Description-field, multi-field, and longer-text cases ────────────
    Case {
      name: "match in description: 'goroutines' → only doc 2 (description-only term)",
      endpoint: Endpoint::Search,
      body: json!({
        "query": { "match": { "description": "goroutines" } },
        "size": 5
      }),
      parity: Parity::Relevance {
        top_k: 1,
        relevant: &["2"],
        min_overlap: 1,
      },
    },
    Case {
      name: "match in description: 'ownership' → only doc 1 (rust-only concept)",
      endpoint: Endpoint::Search,
      body: json!({
        "query": { "match": { "description": "ownership" } },
        "size": 5
      }),
      parity: Parity::Relevance {
        top_k: 1,
        relevant: &["1"],
        min_overlap: 1,
      },
    },
    Case {
      name: "multi_match: 'rust' over title+description → rust-titled docs at top",
      endpoint: Endpoint::Search,
      body: json!({
        "query": {
          "multi_match": {
            "query": "rust",
            "fields": ["title", "description"]
          }
        },
        "size": 3
      }),
      parity: Parity::Relevance {
        top_k: 3,
        relevant: &["1", "7", "10"],
        min_overlap: 3,
      },
    },
    Case {
      name: "multi_match: 'kitchen' over title+description → kitchen-titled docs at top",
      endpoint: Endpoint::Search,
      body: json!({
        "query": {
          "multi_match": {
            "query": "kitchen",
            "fields": ["title", "description"]
          }
        },
        "size": 2
      }),
      parity: Parity::Relevance {
        top_k: 2,
        relevant: &["5", "6"],
        min_overlap: 2,
      },
    },
    Case {
      name: "multi_match with field boost: title^3, description — top-1 must be exact title match",
      endpoint: Endpoint::Search,
      body: json!({
        "query": {
          "multi_match": {
            "query": "advanced",
            "fields": ["title^3", "description"]
          }
        },
        "size": 1
      }),
      parity: Parity::Top1OneOf(&["6", "9"]),
    },
    // ── Tokenization probes (parity, not relevance) ─────────────────────
    // These check that BOTH engines AGREE on edge-case tokenization. Whether
    // they stem or strip stopwords is fine as long as they do it consistently.
    Case {
      name: "stemming probe: 'patterns' (plural) — should agree on hit set",
      endpoint: Endpoint::Search,
      body: json!({
        "query": { "match": { "title": "patterns" } },
        "size": 5
      }),
      parity: Parity::HitIdSet,
    },
    Case {
      name: "stemming probe: 'pattern' (singular) — should agree on hit set",
      endpoint: Endpoint::Search,
      body: json!({
        "query": { "match": { "title": "pattern" } },
        "size": 5
      }),
      parity: Parity::HitIdSet,
    },
    Case {
      name: "stemming probe: 'tools' (plural in title 6) vs 'tool' (singular)",
      endpoint: Endpoint::Search,
      body: json!({
        "query": { "match": { "title": "tool" } },
        "size": 5
      }),
      parity: Parity::HitIdSet,
    },
    Case {
      name: "stopword probe: 'the' alone — engines must agree (likely most/all docs)",
      endpoint: Endpoint::Search,
      body: json!({
        "query": { "match": { "description": "the" } },
        "size": 50
      }),
      parity: Parity::HitIdSet,
    },
    Case {
      name: "stopword probe: 'of the music' — engines must agree on which docs match",
      endpoint: Endpoint::Search,
      body: json!({
        "query": { "match": { "description": "of the music" } },
        "size": 50
      }),
      parity: Parity::HitIdSet,
    },
    Case {
      name: "long-text scoring: 'kitchen' against description → kitchen-titled docs win",
      endpoint: Endpoint::Search,
      body: json!({
        "query": { "match": { "description": "kitchen" } },
        "size": 3
      }),
      parity: Parity::Relevance {
        top_k: 2,
        relevant: &["5", "6"],
        min_overlap: 2,
      },
    },
  ]
}

/// Compare an ES response and an adapter response under the given parity rule.
/// Returns a list of issues; an empty list means the case passed.
fn evaluate(parity: &Parity, es: &Value, adapter: &Value) -> Vec<String> {
  match parity {
    Parity::HitIdSet => {
      let mut a = collect_hit_ids(es);
      let mut b = collect_hit_ids(adapter);
      a.sort();
      b.sort();
      if a != b {
        vec![format!(
          "hit id sets differ:\n    es      = {a:?}\n    adapter = {b:?}"
        )]
      } else {
        vec![]
      }
    }
    Parity::OrderedHitIds => {
      let a = collect_hit_ids(es);
      let b = collect_hit_ids(adapter);
      if a != b {
        vec![format!(
          "ordered hit ids differ:\n    es      = {a:?}\n    adapter = {b:?}"
        )]
      } else {
        vec![]
      }
    }
    Parity::Count => {
      let a = total(es);
      let b = total(adapter);
      if a != b {
        vec![format!("counts differ: es={a} adapter={b}")]
      } else {
        vec![]
      }
    }
    Parity::TermsAggBuckets(name) => {
      let a = terms_buckets(es, name);
      let b = terms_buckets(adapter, name);
      if a != b {
        vec![format!(
          "terms-agg `{name}` buckets differ:\n    es      = {a:?}\n    adapter = {b:?}"
        )]
      } else {
        vec![]
      }
    }
    Parity::StatsAgg(name) => {
      let a = stats_quad(es, name);
      let b = stats_quad(adapter, name);
      if a != b {
        vec![format!(
          "stats-agg `{name}` differs:\n    es      = {a}\n    adapter = {b}"
        )]
      } else {
        vec![]
      }
    }
    Parity::Relevance {
      top_k,
      relevant,
      min_overlap,
    } => {
      let mut issues = Vec::new();
      let relevant_set: std::collections::HashSet<&str> = relevant.iter().copied().collect();
      let es_top: Vec<String> = collect_hit_ids(es).into_iter().take(*top_k).collect();
      let adapter_top: Vec<String> = collect_hit_ids(adapter).into_iter().take(*top_k).collect();

      let es_extras: Vec<&String> = es_top
        .iter()
        .filter(|id| !relevant_set.contains(id.as_str()))
        .collect();
      let adapter_extras: Vec<&String> = adapter_top
        .iter()
        .filter(|id| !relevant_set.contains(id.as_str()))
        .collect();
      if !es_extras.is_empty() {
        issues.push(format!(
          "elasticsearch top-{top_k} contains non-relevant ids {:?}; relevant set is {:?}",
          es_extras, relevant
        ));
      }
      if !adapter_extras.is_empty() {
        issues.push(format!(
          "ADAPTER top-{top_k} contains non-relevant ids {:?}; relevant set is {:?}",
          adapter_extras, relevant
        ));
      }

      let es_set: std::collections::HashSet<&String> = es_top.iter().collect();
      let overlap = adapter_top.iter().filter(|id| es_set.contains(id)).count();
      if overlap < *min_overlap {
        issues.push(format!(
          "es top-{top_k}={es_top:?} and adapter top-{top_k}={adapter_top:?} share only {overlap} ids (need {min_overlap})"
        ));
      }
      issues
    }
    Parity::Top1OneOf(allowlist) => {
      let mut issues = Vec::new();
      let allow: std::collections::HashSet<&str> = allowlist.iter().copied().collect();
      let es_top1 = collect_hit_ids(es).into_iter().next();
      let adapter_top1 = collect_hit_ids(adapter).into_iter().next();
      match &es_top1 {
        Some(id) if !allow.contains(id.as_str()) => issues.push(format!(
          "elasticsearch top-1 = {id:?} but expected one of {:?}",
          allowlist
        )),
        None => issues.push(format!(
          "elasticsearch returned no hits; expected one of {:?}",
          allowlist
        )),
        _ => {}
      }
      match &adapter_top1 {
        Some(id) if !allow.contains(id.as_str()) => issues.push(format!(
          "ADAPTER top-1 = {id:?} but expected one of {:?}",
          allowlist
        )),
        None => issues.push(format!(
          "ADAPTER returned no hits; expected one of {:?}",
          allowlist
        )),
        _ => {}
      }
      issues
    }
  }
}

fn total(response: &Value) -> u64 {
  response
    .get("count")
    .and_then(Value::as_u64)
    .or_else(|| {
      response
        .pointer("/hits/total/value")
        .and_then(Value::as_u64)
    })
    .unwrap_or(0)
}

fn terms_buckets(response: &Value, name: &str) -> Vec<(String, u64)> {
  let pointer = format!("/aggregations/{name}/buckets");
  let buckets = response
    .pointer(&pointer)
    .and_then(Value::as_array)
    .cloned()
    .unwrap_or_default();
  let mut pairs: Vec<(String, u64)> = buckets
    .iter()
    .map(|b| {
      let key = b
        .get("key")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| b.get("key").map(|v| v.to_string()).unwrap_or_default());
      let count = b.get("doc_count").and_then(Value::as_u64).unwrap_or(0);
      (key, count)
    })
    .collect();
  pairs.sort();
  pairs
}

fn stats_quad(response: &Value, name: &str) -> Value {
  let pointer = format!("/aggregations/{name}");
  let stats = response.pointer(&pointer).cloned().unwrap_or(Value::Null);
  json!({
    "count": stats.get("count").cloned().unwrap_or(Value::Null),
    "min": stats.get("min").cloned().unwrap_or(Value::Null),
    "max": stats.get("max").cloned().unwrap_or(Value::Null),
    "sum": stats.get("sum").cloned().unwrap_or(Value::Null),
  })
}

fn collect_hit_ids(response: &Value) -> Vec<String> {
  response
    .pointer("/hits/hits")
    .and_then(Value::as_array)
    .map(|arr| {
      arr
        .iter()
        .filter_map(|h| h.get("_id").and_then(Value::as_str).map(String::from))
        .collect()
    })
    .unwrap_or_default()
}

fn run_query(
  client: &reqwest::blocking::Client,
  base: &str,
  endpoint: Endpoint,
  body: &Value,
) -> Result<Value> {
  let url = format!("{base}/{INDEX}/{}", endpoint.path());
  let resp = client.post(url).json(body).send()?;
  let status = resp.status();
  let text = resp.text()?;
  if status != StatusCode::OK {
    bail!("query failed {status}: {text}");
  }
  Ok(serde_json::from_str(&text)?)
}

/// Compact projection of a response showing just the hits. Used in failure
/// diagnostics so the panic message is readable.
fn compact(response: &Value) -> String {
  if response.get("count").is_some() {
    return format!("{{count: {}}}", total(response));
  }
  let total = total(response);
  let hits: Vec<Value> = response
    .pointer("/hits/hits")
    .and_then(Value::as_array)
    .cloned()
    .unwrap_or_default()
    .into_iter()
    .map(|h| {
      json!({
        "_id": h.get("_id"),
        "_score": h.get("_score"),
        "title": h.pointer("/_source/title"),
      })
    })
    .collect();
  let aggs = response.get("aggregations");
  match aggs {
    Some(a) => format!(
      "{{total: {total}, hits: {}, aggs: {}}}",
      serde_json::to_string(&hits).unwrap_or_default(),
      serde_json::to_string(a).unwrap_or_default(),
    ),
    None => format!(
      "{{total: {total}, hits: {}}}",
      serde_json::to_string(&hits).unwrap_or_default()
    ),
  }
}

// ── The test ─────────────────────────────────────────────────────────────────

#[test]
fn elasticsearch_parity() {
  if skip_unless_docker("elasticsearch_parity").is_none() {
    return;
  }

  let es = EsContainer::start().expect("start elasticsearch");
  let stack = AdapterStack::start().expect("start searchlite adapter stack");
  let docs = corpus();

  let client = blocking_client();
  ingest_into_es(&client, &es.base_url, &docs).expect("ingest into es");
  ingest_into_searchlite(&client, &stack.upstream_base, &docs).expect("ingest into searchlite");

  let mut failures: Vec<String> = Vec::new();
  for case in cases() {
    let es_resp = match run_query(&client, &es.base_url, case.endpoint, &case.body) {
      Ok(v) => v,
      Err(err) => {
        failures.push(format!("[{}] elasticsearch query failed: {err}", case.name));
        continue;
      }
    };
    let adapter_resp = match run_query(&client, &stack.adapter_base, case.endpoint, &case.body) {
      Ok(v) => v,
      Err(err) => {
        failures.push(format!("[{}] adapter query failed: {err}", case.name));
        continue;
      }
    };
    let issues = evaluate(&case.parity, &es_resp, &adapter_resp);
    let es_top: Vec<String> = collect_hit_ids(&es_resp).into_iter().take(5).collect();
    let adapter_top: Vec<String> = collect_hit_ids(&adapter_resp).into_iter().take(5).collect();
    if issues.is_empty() {
      eprintln!(
        "  ok   [{}] es={es_top:?} adapter={adapter_top:?}",
        case.name
      );
    } else {
      eprintln!(
        "  FAIL [{}] es={es_top:?} adapter={adapter_top:?}",
        case.name
      );
      let raw = format!(
        "    es raw:      {}\n    adapter raw: {}",
        compact(&es_resp),
        compact(&adapter_resp),
      );
      failures.push(format!(
        "[{}]:\n  - {}\n{raw}",
        case.name,
        issues.join("\n  - ")
      ));
    }
  }

  if !failures.is_empty() {
    panic!(
      "{} parity case(s) failed:\n\n{}",
      failures.len(),
      failures.join("\n\n")
    );
  }
}
