use std::collections::BTreeMap;
use std::fs;
use std::io::{self, BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
#[cfg(feature = "s3")]
use clap::Args;
use clap::{Parser, Subcommand, ValueEnum};
use searchlite_core::api::builder::IndexBuilder;
#[cfg(feature = "vectors")]
use searchlite_core::api::types::QueryNode;
use searchlite_core::api::types::{
  Aggregation, ChecksumPolicy, Document, ExecutionStrategy, IndexOptions, Query, SearchRequest,
  SortOrder, SortSpec, StorageType,
};
#[cfg(feature = "vectors")]
use searchlite_core::api::types::{VectorQuery, VectorQuerySpec};
use searchlite_core::api::Index;
use searchlite_core::util::doc_id::validate_doc_id;
use searchlite_http::{
  init_tracing as init_http_tracing, run as http_run, ServeArgs as HttpServeArgs,
};
use tokio::runtime::Runtime;
use tracing::error;

/// User-facing index location. Accepts either a local filesystem
/// path or, when the `s3` feature is enabled, an `s3://bucket/prefix`
/// URL. Constructed via `clap`'s `value_parser`.
#[derive(Debug, Clone)]
enum IndexLocator {
  Local(PathBuf),
  #[cfg(feature = "s3")]
  S3(searchlite_s3::S3Url),
}

impl IndexLocator {
  fn parse(s: &str) -> Result<Self, String> {
    let trimmed = s.trim();
    let lowered = trimmed.to_ascii_lowercase();
    if lowered.starts_with("s3://") {
      #[cfg(feature = "s3")]
      {
        return searchlite_s3::parse_s3_url(trimmed).map(Self::S3);
      }
      #[cfg(not(feature = "s3"))]
      {
        return Err(format!(
          "{trimmed:?} is an s3:// URL but this build was compiled without the `s3` feature; \
           rebuild searchlite-cli with `--features s3` (default), or pass a local index path"
        ));
      }
    }
    if trimmed.is_empty() {
      return Err("index path cannot be empty".into());
    }
    Ok(Self::Local(PathBuf::from(trimmed)))
  }

  /// Coerce to a local filesystem path, erroring out for s3:// URLs.
  /// Used by every writer command (init / add / commit / compact /
  /// merge), which only operate on local indexes — s3:// indexes are
  /// always opened read-only.
  fn require_local(&self) -> Result<&Path> {
    match self {
      Self::Local(p) => Ok(p.as_path()),
      #[cfg(feature = "s3")]
      Self::S3(_) => bail!(
        "this command does not support s3:// URLs — writes happen against a local index. \
         Bake the index locally (`searchlite init`/`add`/`commit`/`compact`), then publish \
         with `searchlite sync <local-path> <s3-url>`."
      ),
    }
  }
}

/// `--checksum-policy` flag. Maps onto
/// [`searchlite_core::api::types::ChecksumPolicy`].
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
#[clap(rename_all = "kebab-case")]
enum ChecksumPolicyArg {
  /// Re-verify SHA-256 of every segment artifact on each fresh
  /// `Index::reader()`. Safest, but expensive on remote backends —
  /// each reader pays one whole-object read per segment artifact.
  #[default]
  Strict,
  /// Trust the manifest's recorded hashes without re-verifying.
  /// Recommended for cloud serving where each reader open should
  /// avoid pulling whole segment files over the network.
  TrustManifest,
  /// Open immediately and verify in a background `rayon` task,
  /// surfacing failures via `log::error!`.
  Audit,
}

impl From<ChecksumPolicyArg> for ChecksumPolicy {
  fn from(value: ChecksumPolicyArg) -> Self {
    match value {
      ChecksumPolicyArg::Strict => ChecksumPolicy::Strict,
      ChecksumPolicyArg::TrustManifest => ChecksumPolicy::TrustManifest,
      ChecksumPolicyArg::Audit => ChecksumPolicy::Audit,
    }
  }
}

/// Connection-level S3 flags shared by `searchlite sync` and
/// `--index s3://...` on read-side subcommands. Mirrors the
/// equivalents on the `aws s3` CLI: endpoint URL, region, path-style
/// addressing, and conditional-PUT support. Credentials always come
/// from the standard AWS chain (env vars, shared credentials file,
/// IAM roles) so secrets never appear in `ps` or shell history.
#[cfg(feature = "s3")]
#[derive(Args, Debug, Clone, Default)]
struct S3ConnectionArgs {
  /// S3-compatible endpoint URL. Set this for Cloudflare R2
  /// (`https://<account>.r2.cloudflarestorage.com`) or MinIO /
  /// LocalStack (`http://localhost:9000`). Leave unset to target AWS
  /// S3 directly.
  #[arg(long = "s3-endpoint", env = "SEARCHLITE_S3_ENDPOINT")]
  endpoint: Option<String>,

  /// AWS region (e.g. `us-east-1`). Required by SigV4 even for R2
  /// (use `auto`) and MinIO. Defaults to the standard `AWS_REGION` /
  /// `AWS_DEFAULT_REGION` env vars; falls back to `us-east-1` if
  /// neither is set.
  #[arg(long = "s3-region", env = "AWS_REGION")]
  region: Option<String>,

  /// Use path-style addressing (`https://endpoint/bucket/key`)
  /// instead of virtual-hosted-style (`https://bucket.endpoint/key`).
  /// Required for MinIO / LocalStack / wiremock.
  #[arg(long = "s3-force-path-style")]
  force_path_style: bool,

  /// Enable conditional PUTs (`If-Match` / `If-None-Match`).
  /// Defaults to `true` on AWS S3 and MinIO, and to `false` on
  /// Cloudflare R2 (auto-detected from the endpoint hostname pattern
  /// `*.r2.cloudflarestorage.com`). Pass `--s3-conditional-put true`
  /// to opt in once you've confirmed your R2 account/bucket supports
  /// them.
  #[arg(long = "s3-conditional-put")]
  conditional_put: Option<bool>,
}

#[cfg(feature = "s3")]
impl S3ConnectionArgs {
  /// Compose this flag bundle with a parsed [`searchlite_s3::S3Url`]
  /// into a full [`searchlite_s3::S3Config`] ready for
  /// [`searchlite_s3::S3BlobStore::new`].
  fn into_config(self, url: &searchlite_s3::S3Url) -> searchlite_s3::S3Config {
    let region = self
      .region
      .or_else(|| std::env::var("AWS_DEFAULT_REGION").ok())
      .filter(|r| !r.trim().is_empty())
      .unwrap_or_else(|| "us-east-1".to_string());
    let endpoint_url = self.endpoint.filter(|e| !e.trim().is_empty());
    let is_r2 = endpoint_url
      .as_deref()
      .map(searchlite_s3::is_r2_endpoint)
      .unwrap_or(false);
    let conditional_put = self.conditional_put.unwrap_or(!is_r2);
    searchlite_s3::S3Config {
      endpoint_url,
      region,
      bucket: url.bucket.clone(),
      prefix: url.prefix.clone(),
      credentials: searchlite_s3::S3Credentials::LoadFromEnv,
      conditional_put,
      force_path_style: self.force_path_style,
    }
  }
}

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
  Init {
    #[arg(value_parser = IndexLocator::parse)]
    index: IndexLocator,
    schema: PathBuf,
    /// Optional write key required for all future writes
    #[arg(long = "write-key")]
    write_key: Option<String>,
  },
  /// Add documents from a JSONL file
  Add {
    #[arg(value_parser = IndexLocator::parse)]
    index: IndexLocator,
    doc: PathBuf,
    #[arg(long = "write-key")]
    write_key: Option<String>,
  },
  /// Update (upsert) documents from a JSONL file
  Update {
    #[arg(value_parser = IndexLocator::parse)]
    index: IndexLocator,
    doc: PathBuf,
    #[arg(long = "write-key")]
    write_key: Option<String>,
  },
  /// Delete documents by id (newline-delimited list)
  Delete {
    #[arg(value_parser = IndexLocator::parse)]
    index: IndexLocator,
    ids: PathBuf,
    #[arg(long = "write-key")]
    write_key: Option<String>,
  },
  /// Commit pending documents
  Commit {
    #[arg(value_parser = IndexLocator::parse)]
    index: IndexLocator,
    #[arg(long = "write-key")]
    write_key: Option<String>,
  },
  /// Execute a search query. Pass either a local path or an
  /// `s3://bucket/prefix` URL as `<INDEX>`.
  Search {
    #[arg(value_parser = IndexLocator::parse)]
    index: IndexLocator,
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
    #[arg(long, default_value_t = true)]
    return_hits: bool,
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
    /// Checksum verification policy applied at index open. Defaults
    /// to `strict`. Cloud serving (`s3://...`) typically benefits
    /// from `trust-manifest` to skip whole-file SHA-256 re-reads on
    /// every `Index::reader()`.
    #[arg(long = "checksum-policy", value_enum, default_value_t = ChecksumPolicyArg::default())]
    checksum_policy: ChecksumPolicyArg,
    #[cfg(feature = "s3")]
    #[command(flatten)]
    s3: S3ConnectionArgs,
  },
  /// Start the HTTP server for one or more indexes (NAME:PATH mounts)
  Http {
    #[command(flatten)]
    args: HttpServeArgs,
  },
  /// Inspect manifest and segments. Accepts either a local path or
  /// an `s3://bucket/prefix` URL as `<INDEX>`.
  Inspect {
    #[arg(value_parser = IndexLocator::parse)]
    index: IndexLocator,
    #[arg(long = "checksum-policy", value_enum, default_value_t = ChecksumPolicyArg::default())]
    checksum_policy: ChecksumPolicyArg,
    #[cfg(feature = "s3")]
    #[command(flatten)]
    s3: S3ConnectionArgs,
  },
  /// Compact segments
  Compact {
    #[arg(value_parser = IndexLocator::parse)]
    index: IndexLocator,
    #[arg(long = "write-key")]
    write_key: Option<String>,
  },
  /// Bake-and-publish a local index to an S3-compatible bucket.
  ///
  /// Mirrors the `aws s3 sync` shape: `<SOURCE>` is a local index
  /// directory, `<DEST>` is an `s3://bucket/prefix` URL. Refuses
  /// to publish a partially-baked index (pending manifest, non-empty
  /// WAL, missing artifacts, legacy v1 manifests, non-canonical
  /// keys) and uploads `MANIFEST.json` last as the visibility fence.
  ///
  /// AWS credentials come from the standard chain (env vars,
  /// shared credentials file, IAM roles); R2 / MinIO users set
  /// `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` and pass
  /// `--s3-endpoint`.
  #[cfg(feature = "s3")]
  Sync {
    /// Local index directory (created with `searchlite init` and
    /// committed via `searchlite commit`).
    source: PathBuf,
    /// Destination URL (`s3://bucket[/prefix]`).
    #[arg(value_parser = parse_s3_dest)]
    dest: searchlite_s3::S3Url,
    #[command(flatten)]
    s3: S3ConnectionArgs,
  },
}

/// `clap::value_parser` for the `<DEST>` arg of `searchlite sync`.
/// Adds a friendlier error than [`searchlite_s3::parse_s3_url`]'s
/// default if the user passes a local path by mistake.
#[cfg(feature = "s3")]
fn parse_s3_dest(s: &str) -> Result<searchlite_s3::S3Url, String> {
  let trimmed = s.trim();
  if !trimmed.to_ascii_lowercase().starts_with("s3://") {
    return Err(format!(
      "{trimmed:?} is not an s3:// URL. \
       Usage: `searchlite sync <local-path> s3://<bucket>/<prefix>`."
    ));
  }
  searchlite_s3::parse_s3_url(trimmed)
}

fn main() -> Result<()> {
  env_logger::init();
  let cli = Cli::parse();
  match cli.command {
    Commands::Init {
      index,
      schema,
      write_key,
    } => cmd_init(&index, schema.as_path(), write_key.as_deref()),
    Commands::Add {
      index,
      doc,
      write_key,
    } => cmd_add(&index, doc.as_path(), write_key.as_deref()),
    Commands::Update {
      index,
      doc,
      write_key,
    } => cmd_add(&index, doc.as_path(), write_key.as_deref()),
    Commands::Delete {
      index,
      ids,
      write_key,
    } => cmd_delete(&index, ids.as_path(), write_key.as_deref()),
    Commands::Commit { index, write_key } => cmd_commit(&index, write_key.as_deref()),
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
      return_hits,
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
      checksum_policy,
      #[cfg(feature = "s3")]
      s3,
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
          return_hits,
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
      let read_args = ReadOpenArgs {
        checksum_policy,
        #[cfg(feature = "s3")]
        s3,
      };
      cmd_search(index, read_args, request)
    }
    Commands::Http { args } => {
      init_http_tracing();
      let rt = Runtime::new()?;
      if let Err(err) = rt.block_on(http_run(args)) {
        error!("{err:?}");
        std::process::exit(1);
      }
      Ok(())
    }
    Commands::Inspect {
      index,
      checksum_policy,
      #[cfg(feature = "s3")]
      s3,
    } => cmd_inspect(
      index,
      ReadOpenArgs {
        checksum_policy,
        #[cfg(feature = "s3")]
        s3,
      },
    ),
    Commands::Compact { index, write_key } => cmd_compact(&index, write_key.as_deref()),
    #[cfg(feature = "s3")]
    Commands::Sync { source, dest, s3 } => cmd_sync(source.as_path(), dest, s3),
  }
}

/// Bundle of open-time arguments shared between read-side
/// subcommands. Cleaner than threading every flag individually.
struct ReadOpenArgs {
  checksum_policy: ChecksumPolicyArg,
  #[cfg(feature = "s3")]
  s3: S3ConnectionArgs,
}

fn options(path: &Path, create_if_missing: bool) -> IndexOptions {
  IndexOptions {
    path: path.to_path_buf(),
    create_if_missing,
    enable_positions: true,
    bm25_k1: 0.9,
    bm25_b: 0.4,
    storage: StorageType::Filesystem,
    checksum_policy: Default::default(),
    checksum_audit_failure_hook: None,
    read_only: false,
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
  return_hits: bool,
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

fn cmd_init(index: &IndexLocator, schema_path: &Path, write_key: Option<&str>) -> Result<()> {
  let index = index.require_local()?;
  let opts = options(index, true);
  let schema_str = fs::read_to_string(schema_path)?;
  let schema: searchlite_core::api::types::Schema = serde_json::from_str(&schema_str)?;
  IndexBuilder::create_with_write_key(index, schema, opts, write_key)?;
  println!("initialized index at {index:?}");
  Ok(())
}

fn cmd_add(index: &IndexLocator, doc_path: &Path, write_key: Option<&str>) -> Result<()> {
  let index = index.require_local()?;
  let opts = options(index, false);
  let idx = Index::open(opts)?;
  let mut writer = idx.writer_with_key(write_key)?;
  // Stream the NDJSON input line-by-line so memory stays bounded by the
  // longest single document rather than by the total file size.
  let file = fs::File::open(doc_path).with_context(|| format!("reading docs from {doc_path:?}"))?;
  let reader = BufReader::new(file);
  for (line_no, line) in reader.lines().enumerate() {
    let line =
      line.with_context(|| format!("reading docs from {doc_path:?} at line {}", line_no + 1))?;
    let line = line.strip_suffix('\r').unwrap_or(&line);
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

fn cmd_delete(index: &IndexLocator, ids_path: &Path, write_key: Option<&str>) -> Result<()> {
  let index = index.require_local()?;
  let opts = options(index, false);
  let idx = Index::open(opts)?;
  let mut writer = idx.writer_with_key(write_key)?;
  // Stream the id list line-by-line so memory stays bounded regardless of
  // total file size.
  let file =
    fs::File::open(ids_path).with_context(|| format!("reading document ids from {ids_path:?}"))?;
  let reader = BufReader::new(file);
  let mut ids = Vec::new();
  for (line_no, line) in reader.lines().enumerate() {
    let line = line.with_context(|| {
      format!(
        "reading document ids from {ids_path:?} at line {}",
        line_no + 1
      )
    })?;
    let line = line.strip_suffix('\r').unwrap_or(&line);
    if line.trim().is_empty() {
      continue;
    }
    if let Err(err) = validate_doc_id(line) {
      bail!("invalid id on line {}: {}", line_no + 1, err);
    }
    ids.push(line.to_string());
  }
  if ids.is_empty() {
    bail!("no document ids provided");
  }
  writer.delete_documents(&ids)?;
  println!("queued {} deletes, run commit to persist", ids.len());
  Ok(())
}

fn cmd_commit(index: &IndexLocator, write_key: Option<&str>) -> Result<()> {
  let index = index.require_local()?;
  let opts = options(index, false);
  let idx = Index::open(opts)?;
  let mut writer = idx.writer_with_key(write_key)?;
  writer.commit()?;
  println!("committed");
  Ok(())
}

fn cmd_search(index: IndexLocator, args: ReadOpenArgs, request: SearchRequest) -> Result<()> {
  let idx = open_index_for_read(index, args)?;
  let reader = idx.reader()?;
  let result = reader.search(&request)?;
  println!("{}", serde_json::to_string_pretty(&result)?);
  Ok(())
}

/// Open an index for read-side commands (`search`, `inspect`).
/// Local paths go through `Index::open` with the requested checksum
/// policy threaded through; `s3://...` URLs route to
/// `searchlite_s3::open_index_read_only_with_options`, which is
/// async — we drive it on a fresh tokio runtime that's dropped after
/// the index is constructed (subsequent BlobStore calls go through
/// the global runtime that `searchlite_core::runtime::block_on_blob`
/// owns under the `tokio-runtime` feature).
fn open_index_for_read(loc: IndexLocator, args: ReadOpenArgs) -> Result<Index> {
  let policy: ChecksumPolicy = args.checksum_policy.into();
  match loc {
    IndexLocator::Local(path) => {
      let mut opts = options(path.as_path(), false);
      opts.checksum_policy = policy;
      Index::open(opts)
    }
    #[cfg(feature = "s3")]
    IndexLocator::S3(url) => {
      let s3_config = args.s3.into_config(&url);
      let opts = IndexOptions {
        checksum_policy: policy,
        ..Default::default()
      };
      let rt = Runtime::new()?;
      rt.block_on(searchlite_s3::open_index_read_only_with_options(
        s3_config, opts,
      ))
    }
  }
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
    return_hits,
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
  let _vector_opts: Option<()> = None;
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
  let candidate_size = {
    #[cfg(feature = "vectors")]
    {
      vector_candidates
    }
    #[cfg(not(feature = "vectors"))]
    {
      None
    }
  };
  #[cfg(feature = "vectors")]
  let request_vector_query = match &query {
    Query::Node(QueryNode::Vector(_)) => None,
    _ => vector_opts.clone().map(VectorQuerySpec::Structured),
  };
  if limit == 0 && cursor.is_some() {
    bail!("cursor is not supported when limit is 0");
  }
  Ok(SearchRequest {
    query,
    fields: fields.map(|f| f.split(',').map(|s| s.trim().to_string()).collect()),
    filter: None,
    limit,
    from: 0,
    return_hits,
    candidate_size,
    #[cfg(feature = "vectors")]
    max_global_vector_candidates: searchlite_core::api::types::parse_env_max_vector_candidates(),
    sort: parse_sort(sort)?,
    cursor,
    search_after: None,
    execution: parse_execution(&execution),
    bmw_block_size,
    fuzzy: None,
    track_total_hits: None,
    #[cfg(feature = "vectors")]
    vector_query: request_vector_query,
    #[cfg(feature = "vectors")]
    vector_filter: None,
    return_stored,
    highlight_field: highlight,
    highlight: None,
    collapse: None,
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
      fs::read_to_string(&p).with_context(|| format!("reading search request from {p:?}"))?;
    let request = serde_json::from_str::<SearchRequest>(&contents)
      .with_context(|| format!("parsing search request JSON from {p:?}"))?;
    if request.limit == 0 && request.cursor.is_some() {
      bail!("cursor is not supported when limit is 0");
    }
    return Ok(Some(request));
  }
  if request_stdin {
    let mut buf = String::new();
    io::stdin()
      .read_to_string(&mut buf)
      .context("reading search request from stdin")?;
    let request = serde_json::from_str::<SearchRequest>(&buf)
      .context("parsing search request JSON from stdin")?;
    if request.limit == 0 && request.cursor.is_some() {
      bail!("cursor is not supported when limit is 0");
    }
    return Ok(Some(request));
  }
  Ok(None)
}

fn load_aggs(
  aggs: Option<String>,
  aggs_file: Option<PathBuf>,
) -> Result<BTreeMap<String, Aggregation>> {
  let raw = if let Some(path) = aggs_file {
    Some(fs::read_to_string(&path).with_context(|| format!("reading aggs from {path:?}"))?)
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
) -> Result<Option<()>> {
  Ok(None)
}

fn cmd_inspect(index: IndexLocator, args: ReadOpenArgs) -> Result<()> {
  let idx = open_index_for_read(index, args)?;
  let manifest = idx.manifest();
  println!("manifest: {}", serde_json::to_string_pretty(&manifest)?);
  Ok(())
}

fn cmd_compact(index: &IndexLocator, write_key: Option<&str>) -> Result<()> {
  let index = index.require_local()?;
  let opts = options(index, false);
  let idx = Index::open(opts)?;
  idx.compact_with_key(write_key)?;
  println!("compaction complete");
  Ok(())
}

#[cfg(feature = "s3")]
fn cmd_sync(source: &Path, dest: searchlite_s3::S3Url, s3_flags: S3ConnectionArgs) -> Result<()> {
  if !source.is_dir() {
    bail!(
      "sync: <SOURCE> must be a local index directory, got {source:?} \
       (which is not a directory)"
    );
  }
  let s3_config = s3_flags.into_config(&dest);
  let rt = Runtime::new()?;
  let report = rt
    .block_on(searchlite_s3::sync_to_s3(source, s3_config))
    .with_context(|| format!("syncing {source:?} → s3://{}/", dest.bucket))?;
  let prefix = dest.prefix.as_deref().unwrap_or("");
  let separator = if prefix.is_empty() { "" } else { "/" };
  println!(
    "synced {files} files / {bytes} bytes to s3://{bucket}{separator}{prefix}",
    files = report.files,
    bytes = report.bytes,
    bucket = dest.bucket,
  );
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::tempdir;

  /// Construct a local-path [`IndexLocator`] for tests. The full
  /// enum has an `s3://` variant under the `s3` feature, but every
  /// existing test exercises the local-FS path.
  fn local(p: &Path) -> IndexLocator {
    IndexLocator::Local(p.to_path_buf())
  }

  /// Default open-time read args (Strict policy, default S3 flags
  /// when the feature is on). Tests that only exercise local indexes
  /// don't need to vary these.
  fn read_args() -> ReadOpenArgs {
    ReadOpenArgs {
      checksum_policy: ChecksumPolicyArg::Strict,
      #[cfg(feature = "s3")]
      s3: S3ConnectionArgs::default(),
    }
  }

  #[test]
  fn runs_cli_commands_end_to_end() {
    let dir = tempdir().unwrap();
    let index = dir.path().join("idx");
    let schema_path = dir.path().join("schema.json");
    let schema = searchlite_core::api::types::Schema::default_text_body();
    fs::write(&schema_path, serde_json::to_string(&schema).unwrap()).unwrap();
    cmd_init(&local(index.as_path()), schema_path.as_path(), None).unwrap();

    let docs_path = dir.path().join("docs.jsonl");
    fs::write(
      &docs_path,
      "{\"_id\":\"1\",\"body\":\"Rust search\"}\n{\"_id\":\"2\",\"body\":\"Another document\"}\n",
    )
    .unwrap();
    cmd_add(&local(index.as_path()), docs_path.as_path(), None).unwrap();
    cmd_commit(&local(index.as_path()), None).unwrap();
    let request = build_search_request_from_cli(SearchCliArgs {
      query: Some("rust".into()),
      limit: 5,
      cursor: None,
      return_hits: true,
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
    cmd_search(local(index.as_path()), read_args(), request).unwrap();
    cmd_inspect(local(index.as_path()), read_args()).unwrap();
    cmd_compact(&local(index.as_path()), None).unwrap();
  }

  #[test]
  fn search_request_from_json_file() {
    let dir = tempdir().unwrap();
    let index = dir.path().join("idx");
    let schema_path = dir.path().join("schema.json");
    let schema = searchlite_core::api::types::Schema::default_text_body();
    fs::write(&schema_path, serde_json::to_string(&schema).unwrap()).unwrap();
    cmd_init(&local(index.as_path()), schema_path.as_path(), None).unwrap();

    let docs_path = dir.path().join("docs.jsonl");
    fs::write(&docs_path, "{\"_id\":\"1\",\"body\":\"Rust search\"}\n").unwrap();
    cmd_add(&local(index.as_path()), docs_path.as_path(), None).unwrap();
    cmd_commit(&local(index.as_path()), None).unwrap();

    let request = SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      limit: 5,
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
      return_stored: true,
      highlight_field: Some("body".to_string()),
      highlight: None,
      collapse: None,
      aggs: BTreeMap::new(),
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    };
    let request_path = dir.path().join("request.json");
    fs::write(&request_path, serde_json::to_string(&request).unwrap()).unwrap();

    let parsed = read_request(Some(request_path), false).unwrap().unwrap();
    cmd_search(local(index.as_path()), read_args(), parsed).unwrap();
  }

  #[test]
  fn search_fails_when_index_missing() {
    let dir = tempdir().unwrap();
    let index = dir.path().join("idx-missing");
    let request = SearchRequest {
      query: "rust".into(),
      fields: None,
      filter: None,
      limit: 5,
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
      return_stored: true,
      highlight_field: None,
      highlight: None,
      collapse: None,
      aggs: BTreeMap::new(),
      suggest: BTreeMap::new(),
      rescore: None,
      explain: false,
      profile: false,
    };
    let err = cmd_search(local(index.as_path()), read_args(), request).unwrap_err();
    assert!(
      err.to_string().contains("index does not exist"),
      "unexpected error: {err}"
    );
  }

  #[test]
  fn cmd_add_handles_crlf_line_endings_with_streaming_reader() {
    // Regression for BUG-017: the streaming BufReader path does not collapse
    // CRLF into LF the way `str::lines()` did, so the manual `\r` strip must
    // keep working across many lines of a realistic NDJSON file.
    let dir = tempdir().unwrap();
    let index = dir.path().join("idx-crlf");
    let schema_path = dir.path().join("schema.json");
    let schema = searchlite_core::api::types::Schema::default_text_body();
    fs::write(&schema_path, serde_json::to_string(&schema).unwrap()).unwrap();
    cmd_init(&local(index.as_path()), schema_path.as_path(), None).unwrap();

    let docs_path = dir.path().join("docs.jsonl");
    let mut contents = String::new();
    for i in 0..32 {
      contents.push_str(&format!("{{\"_id\":\"{i}\",\"body\":\"doc {i}\"}}\r\n"));
    }
    // Include a blank line (pure CRLF) to confirm the empty-line skip still
    // fires even when the reader hands us `\r` before the strip.
    contents.push_str("\r\n");
    fs::write(&docs_path, contents).unwrap();

    cmd_add(&local(index.as_path()), docs_path.as_path(), None).unwrap();
    cmd_commit(&local(index.as_path()), None).unwrap();

    let opts = options(index.as_path(), false);
    let idx = Index::open(opts).unwrap();
    let reader = idx.reader().unwrap();
    let request = SearchRequest {
      query: "doc".into(),
      fields: None,
      filter: None,
      limit: 100,
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
    };
    let result = reader.search(&request).unwrap();
    assert_eq!(
      result.hits.len(),
      32,
      "expected all CRLF-terminated docs ingested"
    );
  }

  #[test]
  fn cmd_add_reports_line_number_on_invalid_json() {
    // Regression for BUG-017: per-line error context must still reference the
    // 1-based line number even though the reader now hands us lines
    // incrementally rather than as a slice of a fully buffered string.
    let dir = tempdir().unwrap();
    let index = dir.path().join("idx-badline");
    let schema_path = dir.path().join("schema.json");
    let schema = searchlite_core::api::types::Schema::default_text_body();
    fs::write(&schema_path, serde_json::to_string(&schema).unwrap()).unwrap();
    cmd_init(&local(index.as_path()), schema_path.as_path(), None).unwrap();

    let docs_path = dir.path().join("docs.jsonl");
    fs::write(
      &docs_path,
      "{\"_id\":\"1\",\"body\":\"ok\"}\n\
       {\"_id\":\"2\",\"body\":\"ok\"}\n\
       not-valid-json\n",
    )
    .unwrap();

    let err = cmd_add(&local(index.as_path()), docs_path.as_path(), None).unwrap_err();
    let chain: String = err
      .chain()
      .map(|e| e.to_string())
      .collect::<Vec<_>>()
      .join(" | ");
    assert!(
      chain.contains("invalid JSON on line 3"),
      "expected per-line error referencing line 3, got: {chain}"
    );
  }

  #[test]
  fn cmd_delete_streaming_reports_line_number_on_invalid_id() {
    // Regression for BUG-017: cmd_delete must still surface the 1-based line
    // number for invalid IDs after switching to the streaming reader.
    let dir = tempdir().unwrap();
    let index = dir.path().join("idx-badid");
    let schema_path = dir.path().join("schema.json");
    let schema = searchlite_core::api::types::Schema::default_text_body();
    fs::write(&schema_path, serde_json::to_string(&schema).unwrap()).unwrap();
    cmd_init(&local(index.as_path()), schema_path.as_path(), None).unwrap();

    let ids_path = dir.path().join("ids.txt");
    // Second ID contains a NUL byte, which validate_doc_id rejects.
    fs::write(&ids_path, "good-id\nbad\0id\n").unwrap();

    let err = cmd_delete(&local(index.as_path()), ids_path.as_path(), None).unwrap_err();
    assert!(
      err.to_string().contains("invalid id on line 2"),
      "expected per-line error referencing line 2, got: {err}"
    );
  }

  #[test]
  fn delete_handles_whitespace_padded_ids() {
    let dir = tempdir().unwrap();
    let index = dir.path().join("idx-whitespace-delete");
    let schema_path = dir.path().join("schema.json");
    let schema = searchlite_core::api::types::Schema::default_text_body();
    fs::write(&schema_path, serde_json::to_string(&schema).unwrap()).unwrap();
    cmd_init(&local(index.as_path()), schema_path.as_path(), None).unwrap();

    let docs_path = dir.path().join("docs.jsonl");
    fs::write(
      &docs_path,
      "{\"_id\":\"  padded-id  \",\"body\":\"spaced\"}\n",
    )
    .unwrap();
    cmd_add(&local(index.as_path()), docs_path.as_path(), None).unwrap();
    cmd_commit(&local(index.as_path()), None).unwrap();

    let ids_path = dir.path().join("ids.txt");
    fs::write(&ids_path, "  padded-id  \n").unwrap();
    cmd_delete(&local(index.as_path()), ids_path.as_path(), None).unwrap();
    cmd_commit(&local(index.as_path()), None).unwrap();

    let opts = options(index.as_path(), false);
    let idx = Index::open(opts).unwrap();
    let reader = idx.reader().unwrap();
    let request = SearchRequest {
      query: "spaced".into(),
      fields: None,
      filter: None,
      limit: 5,
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
    };
    let result = reader.search(&request).unwrap();
    assert!(
      result.hits.is_empty(),
      "expected padded id document to be deleted"
    );
  }
}
