# searchlite

Embedded, SQLite-flavored search engine with a single on-disk index and an ergonomic Rust API, CLI, and optional C FFI.

**Crates**
- `searchlite-core`: indexing, storage, and retrieval (BM25 + block-level maxima, boolean/phrase matching, filters, optional vectors/GPU rerank stubs).
- `searchlite-cli`: CLI for init/add/commit/search/inspect/compact.
- `searchlite-ffi`: optional C ABI (enable with the `ffi` feature).

**Core capabilities**
- Single-writer, multi-reader index backed by a WAL and atomic manifest updates.
- BM25 scoring (`k1=0.9`, `b=0.4` by default) with phrase matching and basic highlighting.
- Block-level max scores per term (WAND/BMW pruning) for faster exact top-k.
- Filesystem-backed by default; toggle to an in-memory index for ephemeral workloads.
- Stored/fast fields for filters and snippets; optional `vectors`, `gpu`, `zstd`, and `ffi` feature flags.

## Development setup
- Rust toolchain is pinned to `1.78.0` (`rust-toolchain.toml`); install `rustfmt`/`clippy` if missing.
- Build everything with `cargo build --all --all-features` (or `just build`).
- Code quality: `cargo fmt --all`, `cargo clippy --all --all-features -- -D warnings`.
- The CLI runs directly from the workspace: `cargo run -p searchlite-cli -- <subcommand>`.

## Testing & benchmarks
- Tests: `cargo test --all --all-features` (or `just test`).
- Benches (Criterion): `cargo bench -p searchlite-core` (or `just bench`).
- Smoketest the CLI by running a small end-to-end flow (see examples below).

## Schema and documents
Schema lives in `schema.json` (example below). Text fields control tokenization and storage, keyword/numeric fields support filters and fast-field access.

```json
{
  "text_fields": [
    { "name": "body", "tokenizer": "default", "stored": true, "indexed": true }
  ],
  "keyword_fields": [
    { "name": "lang", "stored": true, "indexed": true, "fast": true }
  ],
  "numeric_fields": [
    { "name": "year", "i64": true, "fast": true }
  ],
  "vector_fields": []
}
```

`stored` fields are returned when `--return-stored`/`return_stored` is enabled. `fast` fields are memory-mapped for filters; numeric ranges use `field:[min TO max]`, keyword filters accept `field:value` or `field:v1,v2`.

## CLI workflow examples
Set an index location once:
```bash
INDEX=/tmp/searchlite_idx
```

- Create an index from a schema:
```bash
cargo run -p searchlite-cli -- init --index "$INDEX" --schema schema.json
```

- Add a single document (newline-delimited JSON):
```bash
cat > /tmp/one.jsonl <<'EOF'
{"body":"Rust is a systems programming language","lang":"en","year":2024}
EOF
cargo run -p searchlite-cli -- add --index "$INDEX" --doc /tmp/one.jsonl
cargo run -p searchlite-cli -- commit --index "$INDEX"
```

- Add multiple documents (uses the included sample):
```bash
cargo run -p searchlite-cli -- add --index "$INDEX" --doc docs.jsonl
cargo run -p searchlite-cli -- commit --index "$INDEX"
```

- Query the index (field scoping, filters, stored fields, snippets):
```bash
cargo run -p searchlite-cli -- search \
  --index "$INDEX" \
  --q "body:rust language" \
  --limit 5 \
  --filter "lang:en" \
  --filter "year:[2020 TO 2025]" \
  --return-stored \
  --highlight body
```

Query syntax supports `field:term`, phrases in quotes (`"field:exact phrase"`), and negation with a leading `-term`.

- Inspect or compact:
```bash
cargo run -p searchlite-cli -- inspect --index "$INDEX"
cargo run -p searchlite-cli -- compact --index "$INDEX"
```

## Using the Rust API
```rust
use searchlite_core::api::{
    builder::IndexBuilder, Index, Filter,
    types::{
        Document, ExecutionStrategy, IndexOptions, KeywordField, NumericField, Schema,
        SearchRequest, StorageType,
    },
};
use std::path::PathBuf;

let path = PathBuf::from("./example_idx");
let mut schema = Schema::default_text_body();
schema.keyword_fields.push(KeywordField { name: "lang".into(), stored: true, indexed: true, fast: true });
schema.numeric_fields.push(NumericField { name: "year".into(), i64: true, fast: true, stored: true });

let opts = IndexOptions {
    path: path.clone(),
    create_if_missing: true,
    enable_positions: true,
    bm25_k1: 0.9,
    bm25_b: 0.4,
    storage: StorageType::Filesystem,
    #[cfg(feature = "vectors")]
    vector_defaults: None,
};

// Create or open the index.
let idx = IndexBuilder::create(&path, schema, opts.clone())?;

// Insert one document.
let mut writer = idx.writer()?;
let doc = Document { fields: [
    ("body".to_string(), serde_json::json!("Rust is fast and reliable")),
    ("lang".to_string(), serde_json::json!("en")),
    ("year".to_string(), serde_json::json!(2024)),
].into_iter().collect() };
writer.add_document(&doc)?;

// Insert multiple documents in one batch.
let more_docs = vec![
    Document { fields: [("body".to_string(), serde_json::json!("SQLite vibes for search")), ("lang".to_string(), serde_json::json!("en")), ("year".to_string(), serde_json::json!(2023))].into_iter().collect() },
    Document { fields: [("body".to_string(), serde_json::json!("Embedded search engine demo")), ("lang".to_string(), serde_json::json!("en")), ("year".to_string(), serde_json::json!(2022))].into_iter().collect() },
];
for d in more_docs.iter() {
    writer.add_document(d)?;
}
writer.commit()?; // Flush WAL into a segment

// Search the index.
let reader = idx.reader()?;
let results = reader.search(&SearchRequest {
    query: "rust engine".into(),
    fields: None,
    filters: vec![Filter::I64Range { field: "year".into(), min: 2020, max: 2025 }],
    limit: 5,
    execution: ExecutionStrategy::Wand,
    bmw_block_size: None,
    return_stored: true,
    highlight_field: Some("body".into()),
    #[cfg(feature = "vectors")]
    vector_query: None,
})?;
for hit in results.hits {
    println!("doc {} score {:.3} fields {:?}", hit.doc_id, hit.score, hit.fields);
}
```

`Index::open(opts)` opens an existing index; `Index::compact()` rewrites all segments into one. WAL-backed writers queue documents until `commit` is called; `rollback` drops uncommitted changes.

### Query execution modes
- `execution`: choose `"bm25"` (full evaluation), `"wand"` (exact WAND pruning), or `"bmw"` (block-max WAND). Default is `wand`.
- `bmw_block_size`: optional block size when using BMW pruning.

The CLI exposes `--execution` and `--bmw-block-size` on `search`. A small synthetic benchmark that compares the strategies lives in `searchlite-core/examples/pruning.rs` (`cargo run -p searchlite-core --example pruning`).

### In-memory indexes
For ephemeral or test-heavy scenarios, set `storage: StorageType::InMemory` in `IndexOptions`. The API and search behavior stay the same, but no files are created on disk. (The CLI currently uses filesystem storage only.)

## Building the C library
Build the FFI crate to generate a shared library and header for C or other language bindings.
```bash
# Release build (macOS dylib, Linux .so, Windows .dll + Rust rlib)
cargo build -p searchlite-ffi --release --features ffi

# Enable optional capabilities on the library:
# cargo build -p searchlite-ffi --release --features "ffi,vectors,zstd"
```
Artifacts land in `target/release` (e.g., `libsearchlite_ffi.dylib` or `libsearchlite_ffi.so`) and the C header is at `searchlite-ffi/searchlite.h`.

## Feature flags
- `vectors`: store/query vector fields; search requests can blend BM25 with vector similarity.
- `gpu`: stub GPU reranker hooks.
- `zstd`: compress stored fields.
- `ffi`: build the C FFI surface (`searchlite-ffi` crate, also exposed on the CLI).
