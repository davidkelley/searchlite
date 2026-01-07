use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use searchlite_core::api::builder::IndexBuilder;
use searchlite_core::api::types::{
  Aggregation, Document, ExecutionStrategy, IndexOptions, Query, QueryNode, SearchRequest,
  SortOrder, SortSpec, StorageType, VectorQuery, VectorQuerySpec,
};
use searchlite_core::api::Index;

#[derive(Parser)]
#[command(name = "searchlite", version, about = "Embedded search engine CLI")]
struct Cli {
  #[command(subcommand)]
  command: Commands,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Commands {
  /// Initialize a new index with a schema
  Init { index: PathBuf, schema: PathBuf },
  /// Add documents from a JSONL file
  Add { index: PathBuf, doc: PathBuf },
  /// Update (upsert) documents from a JSONL file
  Update { index: PathBuf, doc: PathBuf },
  /// Delete documents by id (newline-delimited list)
  Delete { index: PathBuf, ids: PathBuf },
  /// Commit pending documents
  Commit { index: PathBuf },
  /// Execute a search query
  Search {
    index: PathBuf,
    #[arg(short = 'q', long = "query")]
    query: Option<String>,
    #[arg(long, default_value_t = 10)]
    limit: usize,
    #[arg(long, default_value = "wand")]
    execution: String,
    #[arg(long)]
    bmw_block_size: Option<usize>,
    #[arg(long)]
    fields: Option<String>,
    #[arg(long)]
    return_stored: bool,
    #[arg(long)]
    highlight: Option<String>,
    #[arg(long)]
    cursor: Option<String>,
    #[arg(long)]
    sort: Option<String>,
    #[arg(long)]
    request: Option<PathBuf>,
    #[arg(long, conflicts_with = "request")]
    request_stdin: bool,
    #[cfg(feature = "vectors")]
    #[arg(long)]
    vector_field: Option<String>,
    #[cfg(feature = "vectors")]
    #[arg(long)]
    vector: Option<String>,
    #[cfg(feature = "vectors")]
    #[arg(long, default_value_t = 0.5)]
    alpha: f32,
    #[cfg(feature = "vectors")]
    #[arg(long)]
    vector_k: Option<usize>,
    #[cfg(feature = "vectors")]
    #[arg(long)]
    vector_ef_search: Option<usize>,
    #[cfg(feature = "vectors")]
    #[arg(long)]
    vector_candidates: Option<usize>,
    /// Aggregations JSON (Elasticsearch-style map)
    #[arg(long)]
    aggs: Option<String>,
    /// Aggregations JSON file path
    #[arg(long)]
    aggs_file: Option<PathBuf>,
  },
  /// Inspect manifest and segments
  Inspect { index: PathBuf },
  /// Compact segments
  Compact { index: PathBuf },
}

fn main() -> Result<()> {
  env_logger::init();
  let cli = Cli::parse();
  match cli.command {
    Commands::Init { index, schema } => cmd_init(index.as_path(), schema.as_path()),
    Commands::Add { index, doc } => cmd_add(index.as_path(), doc.as_path()),
    Commands::Update { index, doc } => cmd_add(index.as_path(), doc.as_path()),
    Commands::Delete { index, ids } => cmd_delete(index.as_path(), ids.as_path()),
    Commands::Commit { index } => cmd_commit(index.as_path()),
    Commands::Search {
      index,
      query,
      limit,
      execution,
      bmw_block_size,
      fields,
      return_stored,
      highlight,
      cursor,
      sort,
      request,
      request_stdin,
      #[cfg(feature = "vectors")]
      vector_field,
      #[cfg(feature = "vectors")]
      vector,
      #[cfg(feature = "vectors")]
      alpha,
      #[cfg(feature = "vectors")]
      vector_k,
      #[cfg(feature = "vectors")]
      vector_ef_search,
      #[cfg(feature = "vectors")]
      vector_candidates,
      aggs,
      aggs_file,
    } => {
      let request = if let Some(req) = read_request(request, request_stdin)? {
        req
      } else {
        build_search_request_from_cli(SearchCliArgs {
          query,
          limit,
          execution,
          bmw_block_size,
          fields,
          return_stored,
          highlight,
          cursor,
          sort,
          #[cfg(feature = "vectors")]
          vector_field,
          #[cfg(feature = "vectors")]
          vector,
          #[cfg(feature = "vectors")]
          alpha,
          #[cfg(feature = "vectors")]
          vector_k,
          #[cfg(feature = "vectors")]
          vector_ef_search,
          #[cfg(feature = "vectors")]
          vector_candidates,
          aggs,
          aggs_file,
        })?
      };
      cmd_search(index, request)
    }
    Commands::Inspect { index } => cmd_inspect(index.as_path()),
    Commands::Compact { index } => cmd_compact(index.as_path()),
  }
}

fn options(path: &Path, create_if_missing: bool) -> IndexOptions {
  IndexOptions {
    path: path.to_path_buf(),
    create_if_missing,
    enable_positions: true,
    bm25_k1: 0.9,
    bm25_b: 0.4,
    storage: StorageType::Filesystem,
    #[cfg(feature = "vectors")]
    vector_defaults: None,
  }
}

struct SearchCliArgs {
  query: Option<String>,
  limit: usize,
  execution: String,
  bmw_block_size: Option<usize>,
  fields: Option<String>,
  return_stored: bool,
  highlight: Option<String>,
  cursor: Option<String>,
  sort: Option<String>,
  #[cfg(feature = "vectors")]
  vector_field: Option<String>,
  #[cfg(feature = "vectors")]
  vector: Option<String>,
  #[cfg(feature = "vectors")]
  alpha: f32,
  #[cfg(feature = "vectors")]
  vector_k: Option<usize>,
  #[cfg(feature = "vectors")]
  vector_ef_search: Option<usize>,
  #[cfg(feature = "vectors")]
  vector_candidates: Option<usize>,
  aggs: Option<String>,
  aggs_file: Option<PathBuf>,
}

fn cmd_init(index: &Path, schema_path: &Path) -> Result<()> {
  let opts = options(index, true);
  let schema_str = fs::read_to_string(schema_path)?;
  let schema: searchlite_core::api::types::Schema = serde_json::from_str(&schema_str)?;
  IndexBuilder::create(index, schema, opts)?;
  println!("initialized index at {:?}", index);
  Ok(())
}

fn cmd_add(index: &Path, doc_path: &Path) -> Result<()> {
  let opts = options(index, false);
  let idx = Index::open(opts)?;
  let mut writer = idx.writer()?;
  let content =
    fs::read_to_string(doc_path).with_context(|| format!("reading docs from {:?}", doc_path))?;
  for (line_no, line) in content.lines().enumerate() {
    if line.trim().is_empty() {
      continue;
    }
    let value: serde_json::Value = serde_json::from_str(line)
      .with_context(|| format!("invalid JSON on line {}", line_no + 1))?;
    let mut fields = std::collections::BTreeMap::new();
    if let Some(obj) = value.as_object() {
      for (k, v) in obj {
        fields.insert(k.clone(), v.clone());
      }
    }
    writer.add_document(&Document { fields })?;
  }
  println!("queued documents (upsert), run commit to persist");
  Ok(())
}

fn cmd_delete(index: &Path, ids_path: &Path) -> Result<()> {
  let opts = options(index, false);
  let idx = Index::open(opts)?;
  let mut writer = idx.writer()?;
  let content = fs::read_to_string(ids_path)
    .with_context(|| format!("reading document ids from {:?}", ids_path))?;
  let mut ids = Vec::new();
  for (line_no, line) in content.lines().enumerate() {
    let trimmed = line.trim();
    if trimmed.is_empty() {
      continue;
    }
    if trimmed.chars().any(|c| c.is_control()) {
      bail!("invalid id on line {}", line_no + 1);
    }
    ids.push(trimmed.to_string());
  }
  if ids.is_empty() {
    bail!("no document ids provided");
  }
  writer.delete_documents(&ids)?;
  println!("queued {} deletes, run commit to persist", ids.len());
  Ok(())
}

fn cmd_commit(index: &Path) -> Result<()> {
  let opts = options(index, false);
  let idx = Index::open(opts)?;
  let mut writer = idx.writer()?;
  writer.commit()?;
  println!("committed");
  Ok(())
}

fn cmd_search(index: PathBuf, request: SearchRequest) -> Result<()> {
  let opts = options(index.as_path(), false);
  let idx = Index::open(opts)?;
  let reader = idx.reader()?;
  let result = reader.search(&request)?;
  println!("{}", serde_json::to_string_pretty(&result)?);
  Ok(())
}

fn build_search_request_from_cli(args: SearchCliArgs) -> Result<SearchRequest> {
  let SearchCliArgs {
    query,
    limit,
    execution,
    bmw_block_size,
    fields,
    return_stored,
    highlight,
    cursor,
    sort,
    #[cfg(feature = "vectors")]
    vector_field,
    #[cfg(feature = "vectors")]
    vector,
    #[cfg(feature = "vectors")]
    alpha,
    #[cfg(feature = "vectors")]
    vector_k,
    #[cfg(feature = "vectors")]
    vector_ef_search,
    #[cfg(feature = "vectors")]
    vector_candidates,
    aggs,
    aggs_file,
  } = args;
  #[cfg(feature = "vectors")]
  let vector_opts = build_vector_query(
    vector_field,
    vector,
    alpha,
    vector_k,
    vector_ef_search,
    vector_candidates,
  )?;
  #[cfg(not(feature = "vectors"))]
  let vector_opts: Option<searchlite_core::api::types::VectorQuery> = None;
  let query = match query {
    Some(q) => Query::String(q),
    None => {
      #[cfg(feature = "vectors")]
      {
        if let Some(v) = vector_opts.clone() {
          Query::Node(QueryNode::Vector(v))
        } else {
          bail!("search query is required unless --request or --request-stdin is provided");
        }
      }
      #[cfg(not(feature = "vectors"))]
      {
        bail!("search query is required unless --request or --request-stdin is provided");
      }
    }
  };
  #[cfg(feature = "vectors")]
  let request_vector_query = match &query {
    Query::Node(QueryNode::Vector(_)) => None,
    _ => vector_opts.clone().map(VectorQuerySpec::Structured),
  };
  Ok(SearchRequest {
    query,
    fields: fields.map(|f| f.split(',').map(|s| s.trim().to_string()).collect()),
    filter: None,
    filters: Vec::new(),
    limit,
    sort: parse_sort(sort)?,
    execution: parse_execution(&execution),
    bmw_block_size,
    fuzzy: None,
    #[cfg(feature = "vectors")]
    vector_query: request_vector_query,
    #[cfg(feature = "vectors")]
    vector_filter: None,
    return_stored,
    highlight_field: highlight,
    cursor,
    aggs: load_aggs(aggs, aggs_file)?,
    suggest: BTreeMap::new(),
    rescore: None,
    explain: false,
    profile: false,
  })
}

fn read_request(path: Option<PathBuf>, request_stdin: bool) -> Result<Option<SearchRequest>> {
  if let Some(p) = path {
    let contents =
      fs::read_to_string(&p).with_context(|| format!("reading search request from {:?}", p))?;
    let request = serde_json::from_str::<SearchRequest>(&contents)
      .with_context(|| format!("parsing search request JSON from {:?}", p))?;
    return Ok(Some(request));
  }
  if request_stdin {
    let mut buf = String::new();
    io::stdin()
      .read_to_string(&mut buf)
      .context("reading search request from stdin")?;
    let request = serde_json::from_str::<SearchRequest>(&buf)
      .context("parsing search request JSON from stdin")?;
    return Ok(Some(request));
  }
  Ok(None)
}

fn load_aggs(
  aggs: Option<String>,
  aggs_file: Option<PathBuf>,
) -> Result<BTreeMap<String, Aggregation>> {
  let raw = if let Some(path) = aggs_file {
    Some(fs::read_to_string(&path).with_context(|| format!("reading aggs from {:?}", path))?)
  } else {
    aggs
  };
  if let Some(body) = raw {
    if body.trim().is_empty() {
      return Ok(BTreeMap::new());
    }
    let parsed: BTreeMap<String, Aggregation> =
      serde_json::from_str(&body).with_context(|| "invalid aggregations JSON".to_string())?;
    Ok(parsed)
  } else {
    Ok(BTreeMap::new())
  }
}

fn parse_execution(value: &str) -> ExecutionStrategy {
  match value.to_ascii_lowercase().as_str() {
    "bm25" => ExecutionStrategy::Bm25,
    "bmw" => ExecutionStrategy::Bmw,
    _ => ExecutionStrategy::Wand,
  }
}

fn parse_sort(value: Option<String>) -> Result<Vec<SortSpec>> {
  let mut out = Vec::new();
  if let Some(raw) = value {
    for clause in raw.split(',') {
      let trimmed = clause.trim();
      if trimmed.is_empty() {
        continue;
      }
      let mut parts = trimmed.splitn(2, ':');
      let field = parts.next().unwrap().to_string();
      let order = if let Some(ord) = parts.next() {
        match ord.to_ascii_lowercase().as_str() {
          "asc" => Some(SortOrder::Asc),
          "desc" => Some(SortOrder::Desc),
          _ => bail!("invalid sort order `{ord}` (expected asc or desc)"),
        }
      } else {
        None
      };
      out.push(SortSpec { field, order });
    }
  }
  Ok(out)
}

#[cfg(feature = "vectors")]
fn build_vector_query(
  vector_field: Option<String>,
  vector: Option<String>,
  alpha: f32,
  vector_k: Option<usize>,
  vector_ef_search: Option<usize>,
  vector_candidates: Option<usize>,
) -> Result<Option<VectorQuery>> {
  if let (Some(field), Some(vec_str)) = (vector_field, vector) {
    let parsed: Vec<f32> = serde_json::from_str(&vec_str)?;
    return Ok(Some(VectorQuery {
      field,
      vector: parsed,
      k: vector_k,
      alpha: Some(alpha),
      ef_search: vector_ef_search,
      candidate_size: vector_candidates,
      boost: None,
    }));
  }
  Ok(None)
}

#[cfg(not(feature = "vectors"))]
#[allow(dead_code)]
fn build_vector_query(
  _vector_field: Option<String>,
  _vector: Option<String>,
  _alpha: f32,
  _vector_k: Option<usize>,
  _vector_ef_search: Option<usize>,
  _vector_candidates: Option<usize>,
) -> Result<Option<VectorQuery>> {
  Ok(None)
}

fn cmd_inspect(index: &Path) -> Result<()> {
  let opts = options(index, false);
  let idx = Index::open(opts)?;
  let manifest = idx.manifest();
  println!("manifest: {}", serde_json::to_string_pretty(&manifest)?);
  Ok(())
}

fn cmd_compact(index: &Path) -> Result<()> {
  let opts = options(index, false);
  let idx = Index::open(opts)?;
  idx.compact()?;
  println!("compaction complete");
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::tempdir;

  #[test]
  fn runs_cli_commands_end_to_end() {
    let dir = tempdir().unwrap();
    let index = dir.path().join("idx");
    let schema_path = dir.path().join("schema.json");
    let schema = searchlite_core::api::types::Schema::default_text_body();
    fs::write(&schema_path, serde_json::to_string(&schema).unwrap()).unwrap();
    cmd_init(index.as_path(), schema_path.as_path()).unwrap();

    let docs_path = dir.path().join("docs.jsonl");
    fs::write(
      &docs_path,
      "{\"_id\":\"1\",\"body\":\"Rust search\"}\n{\"_id\":\"2\",\"body\":\"Another document\"}\n",
    )
    .unwrap();
    cmd_add(index.as_path(), docs_path.as_path()).unwrap();
    cmd_commit(index.as_path()).unwrap();
    let request = build_search_request_from_cli(SearchCliArgs {
      query: Some("rust".into()),
      limit: 5,
      cursor: None,
      execution: "wand".to_string(),
      bmw_block_size: None,
      fields: None,
      return_stored: true,
      highlight: Some("body".to_string()),
      sort: None,
      #[cfg(feature = "vectors")]
      vector_field: None,
      #[cfg(feature = "vectors")]
      vector: None,
      #[cfg(feature = "vectors")]
      alpha: 0.5,
      #[cfg(feature = "vectors")]
      vector_k: None,
      #[cfg(feature = "vectors")]
      vector_ef_search: None,
      #[cfg(feature = "vectors")]
      vector_candidates: None,
      aggs: None,
      aggs_file: None,
    })
    .unwrap();
    cmd_search(index.clone(), request).unwrap();
    cmd_inspect(index.as_path()).unwrap();
    cmd_compact(index.as_path()).unwrap();
  }

  #[test]
  fn search_request_from_json_file() {
    let dir = tempdir().unwrap();
    let index = dir.path().join("idx");
    let schema_path = dir.path().join("schema.json");
    let schema = searchlite_core::api::types::Schema::default_text_body();
    fs::write(&schema_path, serde_json::to_string(&schema).unwrap()).unwrap();
    cmd_init(index.as_path(), schema_path.as_path()).unwrap();

    let docs_path = dir.path().join("docs.jsonl");
    fs::write(&docs_path, "{\"_id\":\"1\",\"body\":\"Rust search\"}\n").unwrap();
    cmd_add(index.as_path(), docs_path.as_path()).unwrap();
    cmd_commit(index.as_path()).unwrap();

    let request = SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      filters: vec![],
      limit: 5,
      sort: Vec::new(),
      cursor: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
      #[cfg(feature = "vectors")]
      vector_query: None,

      #[cfg(feature = "vectors")]
      vector_filter: None,
      return_stored: true,
      highlight_field: Some("body".to_string()),
      aggs: BTreeMap::new(),
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    };
    let request_path = dir.path().join("request.json");
    fs::write(&request_path, serde_json::to_string(&request).unwrap()).unwrap();

    let parsed = read_request(Some(request_path), false).unwrap().unwrap();
    cmd_search(index.clone(), parsed).unwrap();
  }

  #[test]
  fn search_fails_when_index_missing() {
    let dir = tempdir().unwrap();
    let index = dir.path().join("idx-missing");
    let request = SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      filters: vec![],
      limit: 5,
      sort: Vec::new(),
      cursor: None,
      execution: ExecutionStrategy::Wand,
      bmw_block_size: None,
      fuzzy: None,
      #[cfg(feature = "vectors")]
      vector_query: None,
      #[cfg(feature = "vectors")]
      vector_filter: None,
      return_stored: true,
      highlight_field: None,
      aggs: BTreeMap::new(),
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    };
    let err = cmd_search(index, request).unwrap_err();
    assert!(
      err.to_string().contains("index does not exist"),
      "unexpected error: {err}"
    );
  }
}
