use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use searchlite_core::api::builder::IndexBuilder;
use searchlite_core::api::types::{
  Aggregation, Document, ExecutionStrategy, Filter, IndexOptions, SearchRequest, StorageType,
};
use searchlite_core::api::Index;

#[derive(Parser)]
#[command(name = "searchlite", version, about = "Embedded search engine CLI")]
struct Cli {
  #[command(subcommand)]
  command: Commands,
}

#[derive(Subcommand)]
enum Commands {
  /// Initialize a new index with a schema
  Init { index: PathBuf, schema: PathBuf },
  /// Add documents from a JSONL file
  Add { index: PathBuf, doc: PathBuf },
  /// Commit pending documents
  Commit { index: PathBuf },
  /// Execute a search query
  Search {
    index: PathBuf,
    #[arg(short = 'q')]
    query: String,
    #[arg(long, default_value_t = 10)]
    limit: usize,
    #[arg(long, default_value = "wand")]
    execution: String,
    #[arg(long)]
    bmw_block_size: Option<usize>,
    #[arg(long)]
    fields: Option<String>,
    #[arg(long)]
    filter: Vec<String>,
    #[arg(long)]
    return_stored: bool,
    #[arg(long)]
    highlight: Option<String>,
    #[cfg(feature = "vectors")]
    #[arg(long)]
    vector_field: Option<String>,
    #[cfg(feature = "vectors")]
    #[arg(long)]
    vector: Option<String>,
    #[cfg(feature = "vectors")]
    #[arg(long, default_value_t = 0.5)]
    alpha: f32,
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
    Commands::Commit { index } => cmd_commit(index.as_path()),
    Commands::Search {
      index,
      query,
      limit,
      execution,
      bmw_block_size,
      fields,
      filter,
      return_stored,
      highlight,
      #[cfg(feature = "vectors")]
      vector_field,
      #[cfg(feature = "vectors")]
      vector,
      #[cfg(feature = "vectors")]
      alpha,
      aggs,
      aggs_file,
    } => cmd_search(SearchArgs {
      index,
      query,
      limit,
      execution,
      bmw_block_size,
      fields,
      filters: filter,
      return_stored,
      highlight,
      #[cfg(feature = "vectors")]
      vector_field,
      #[cfg(feature = "vectors")]
      vector,
      #[cfg(feature = "vectors")]
      alpha,
      aggs,
      aggs_file,
    }),
    Commands::Inspect { index } => cmd_inspect(index.as_path()),
    Commands::Compact { index } => cmd_compact(index.as_path()),
  }
}

fn default_options(path: &Path) -> IndexOptions {
  IndexOptions {
    path: path.to_path_buf(),
    create_if_missing: true,
    enable_positions: true,
    bm25_k1: 0.9,
    bm25_b: 0.4,
    storage: StorageType::Filesystem,
    #[cfg(feature = "vectors")]
    vector_defaults: None,
  }
}

struct SearchArgs {
  index: PathBuf,
  query: String,
  limit: usize,
  execution: String,
  bmw_block_size: Option<usize>,
  fields: Option<String>,
  filters: Vec<String>,
  return_stored: bool,
  highlight: Option<String>,
  #[cfg(feature = "vectors")]
  vector_field: Option<String>,
  #[cfg(feature = "vectors")]
  vector: Option<String>,
  #[cfg(feature = "vectors")]
  alpha: f32,
  aggs: Option<String>,
  aggs_file: Option<PathBuf>,
}

fn cmd_init(index: &Path, schema_path: &Path) -> Result<()> {
  let opts = default_options(index);
  let schema_str = fs::read_to_string(schema_path)?;
  let schema: searchlite_core::api::types::Schema = serde_json::from_str(&schema_str)?;
  IndexBuilder::create(index, schema, opts)?;
  println!("initialized index at {:?}", index);
  Ok(())
}

fn cmd_add(index: &Path, doc_path: &Path) -> Result<()> {
  let opts = default_options(index);
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
  println!("added documents, run commit to persist");
  Ok(())
}

fn cmd_commit(index: &Path) -> Result<()> {
  let opts = default_options(index);
  let idx = Index::open(opts)?;
  let mut writer = idx.writer()?;
  writer.commit()?;
  println!("committed");
  Ok(())
}

fn cmd_search(args: SearchArgs) -> Result<()> {
  let SearchArgs {
    index,
    query,
    limit,
    execution,
    bmw_block_size,
    fields,
    filters,
    return_stored,
    highlight,
    #[cfg(feature = "vectors")]
    vector_field,
    #[cfg(feature = "vectors")]
    vector,
    #[cfg(feature = "vectors")]
    alpha,
    aggs,
    aggs_file,
  } = args;
  let opts = default_options(index.as_path());
  let idx = Index::open(opts)?;
  let reader = idx.reader()?;
  let parsed_filters = filters
    .iter()
    .filter_map(|f| parse_filter(f))
    .collect::<Vec<_>>();
  let request = SearchRequest {
    query,
    fields: fields.map(|f| f.split(',').map(|s| s.trim().to_string()).collect()),
    filters: parsed_filters,
    limit,
    execution: parse_execution(&execution),
    bmw_block_size,
    #[cfg(feature = "vectors")]
    vector_query: build_vector_query(vector_field, vector, alpha)?,
    return_stored,
    highlight_field: highlight,
    aggs: load_aggs(aggs, aggs_file)?,
  };
  let result = reader.search(&request)?;
  println!("{}", serde_json::to_string_pretty(&result)?);
  Ok(())
}

fn parse_filter(input: &str) -> Option<Filter> {
  if let Some(eq_idx) = input.find(':') {
    let field = input[..eq_idx].to_string();
    let rest = &input[eq_idx + 1..];
    if let Some(range_idx) = rest.find("[") {
      let body = &rest[range_idx + 1..rest.len() - 1];
      let parts: Vec<&str> = body.split_whitespace().collect();
      if parts.len() == 3 && parts[1].eq_ignore_ascii_case("TO") {
        if let (Ok(min), Ok(max)) = (parts[0].parse::<i64>(), parts[2].parse::<i64>()) {
          return Some(Filter::I64Range { field, min, max });
        }
      }
    } else if rest.contains(',') {
      let values = rest.split(',').map(|s| s.trim().to_string()).collect();
      return Some(Filter::KeywordIn { field, values });
    } else {
      return Some(Filter::KeywordEq {
        field,
        value: rest.to_string(),
      });
    }
  }
  None
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

#[cfg(feature = "vectors")]
fn build_vector_query(
  vector_field: Option<String>,
  vector: Option<String>,
  alpha: f32,
) -> Result<Option<(String, Vec<f32>, f32)>> {
  if let (Some(field), Some(vec_str)) = (vector_field, vector) {
    let parsed: Vec<f32> = serde_json::from_str(&vec_str)?;
    return Ok(Some((field, parsed, alpha)));
  }
  Ok(None)
}

#[cfg(not(feature = "vectors"))]
#[allow(dead_code)]
fn build_vector_query(
  _vector_field: Option<String>,
  _vector: Option<String>,
  _alpha: f32,
) -> Result<Option<(String, Vec<f32>, f32)>> {
  Ok(None)
}

fn cmd_inspect(index: &Path) -> Result<()> {
  let opts = default_options(index);
  let idx = Index::open(opts)?;
  let manifest = idx.manifest();
  println!("manifest: {}", serde_json::to_string_pretty(&manifest)?);
  Ok(())
}

fn cmd_compact(index: &Path) -> Result<()> {
  let opts = default_options(index);
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
  fn parses_filter_variants() {
    match parse_filter("tag:rust") {
      Some(Filter::KeywordEq { field, value }) => {
        assert_eq!(field, "tag");
        assert_eq!(value, "rust");
      }
      other => panic!("unexpected filter {:?}", other),
    }
    match parse_filter("tag:rust,systems") {
      Some(Filter::KeywordIn { field, values }) => {
        assert_eq!(field, "tag");
        assert_eq!(values, vec!["rust".to_string(), "systems".to_string()]);
      }
      other => panic!("unexpected filter {:?}", other),
    }
    match parse_filter("year:[2010 TO 2020]") {
      Some(Filter::I64Range { field, min, max }) => {
        assert_eq!(field, "year");
        assert_eq!(min, 2010);
        assert_eq!(max, 2020);
      }
      other => panic!("unexpected filter {:?}", other),
    }
    assert!(parse_filter("invalid").is_none());
  }

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
      "{\"body\":\"Rust search\"}\n{\"body\":\"Another document\"}\n",
    )
    .unwrap();
    cmd_add(index.as_path(), docs_path.as_path()).unwrap();
    cmd_commit(index.as_path()).unwrap();
    cmd_search(SearchArgs {
      index: index.clone(),
      query: "rust".to_string(),
      limit: 5,
      execution: "wand".to_string(),
      bmw_block_size: None,
      fields: None,
      filters: vec![],
      return_stored: true,
      highlight: Some("body".to_string()),
      #[cfg(feature = "vectors")]
      vector_field: None,
      #[cfg(feature = "vectors")]
      vector: None,
      #[cfg(feature = "vectors")]
      alpha: 0.5,
      aggs: None,
      aggs_file: None,
    })
    .unwrap();
    cmd_inspect(index.as_path()).unwrap();
    cmd_compact(index.as_path()).unwrap();
  }
}
