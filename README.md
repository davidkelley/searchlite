# searchlite

Embedded, SQLite-flavored search engine with a single on-disk index and an ergonomic Rust API, CLI, and optional C FFI.

**Crates**

- `searchlite-core`: indexing, storage, and retrieval (BM25 + block-level maxima, boolean/phrase matching, filters, optional vectors/GPU rerank stubs).
- `searchlite-cli`: CLI for init/add/commit/search/inspect/compact.
- `searchlite-ffi`: optional C ABI (enable with the `ffi` feature).
- `searchlite-wasm`: experimental wasm bindings with an IndexedDB-backed `Storage` implementation (threaded wasm needs `wasm-bindgen-rayon`; you must configure COOP/COEP yourself).

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
- The CLI runs directly from the workspace: `cargo run -p searchlite-cli -- <subcommand> <index> ...` (e.g., `cargo run -p searchlite-cli -- init /tmp/idx schema.json`).

## Testing & benchmarks

- Tests: `cargo test --all --all-features` (or `just test`).
- Benches (Criterion): `cargo bench -p searchlite-core` (or `just bench`).
- Smoketest the CLI by running a small end-to-end flow (see examples below).

## Schema and documents

Schema lives in `schema.json` (example below). Text fields control tokenization and storage, keyword/numeric fields support filters and fast-field access.

```json
{
  "doc_id_field": "_id",
  "text_fields": [
    { "name": "body", "tokenizer": "default", "stored": true, "indexed": true }
  ],
  "keyword_fields": [
    { "name": "lang", "stored": true, "indexed": true, "fast": true }
  ],
  "numeric_fields": [{ "name": "year", "i64": true, "fast": true }],
  "nested_fields": [
    {
      "name": "comment",
      "fields": [
        {
          "type": "keyword",
          "name": "author",
          "stored": true,
          "indexed": true,
          "fast": true
        }
      ]
    }
  ],
  "vector_fields": []
}
```

`stored` fields are returned when `--return-stored`/`return_stored` is enabled. `fast` fields are memory-mapped for filters; numeric ranges use `field:[min TO max]`, keyword filters accept `field:value` or `field:v1,v2`. Nested objects are flattened into dotted field names (e.g., `comment.author`); you can either filter on the dotted path directly or wrap a clause with the `Nested` filter in the JSON API.
Nested filters are evaluated per object, and stored nested values preserve their original structure while omitting unstored fields.

Every document must include a string primary key under `doc_id_field` (defaults to `_id`). Skip listing that id in your `text_fields`/`keyword_fields`/`numeric_fields`; it is stored automatically, returned on hits, and used for upsert/delete semantics.

## CLI workflow examples

Set an index location once:

```bash
INDEX=/tmp/searchlite_idx
```

## CLI commands

Use `cargo run -p searchlite-cli -- <command> ...` to invoke the CLI. Each command maps to a lifecycle step:

- `init <index> <schema>`: creates a new index directory and writes the schema manifest so the index is ready to accept documents.
- `add <index> <doc.jsonl>`: upserts newline-delimited JSON documents (keyed by `doc_id_field`) into the writer buffer; changes are not visible to readers until you run `commit`.
- `update <index> <doc.jsonl>`: alias for `add` to emphasize upsert semantics.
- `delete <index> <ids.txt>`: queues deletions by id (one id per line, matching `doc_id_field`), applied on `commit`.
- `commit <index>`: flushes buffered documents, writes new segment files, and updates the manifest so searches can see the newly added data.
- `search <index> [options]`: executes a query, returning JSON hits (and optional aggregations) using either CLI flags or a full request payload.
- `inspect <index>`: prints the current manifest and segment metadata to help debug index contents and state.
- `compact <index>`: merges segments to reduce fragmentation and improve search performance.

Documents without the required id (`doc_id_field`) will be rejected. Upserts are effective on commit; deletes hide older documents immediately after commit and are dropped on the next compaction.

- Create an index from a schema:

```bash
cargo run -p searchlite-cli -- init "$INDEX" schema.json
```

- Add a single document (newline-delimited JSON):

```bash
cat > /tmp/one.jsonl <<'EOF'
{"_id":"doc-1","body":"Rust is a systems programming language","lang":"en","year":2024}
EOF
cargo run -p searchlite-cli -- add "$INDEX" /tmp/one.jsonl
cargo run -p searchlite-cli -- commit "$INDEX"
```

- Add multiple documents (uses the included sample):

```bash
cargo run -p searchlite-cli -- add "$INDEX" docs.jsonl
cargo run -p searchlite-cli -- commit "$INDEX"
```

- Query the index (field scoping, filters, stored fields, snippets):

```bash
cat > /tmp/request.json <<'EOF'
{
  "query": "body:rust language",
  "fields": ["body","title"],
  "filters": [
    { "KeywordEq": { "field": "lang", "value": "en" } },
    { "I64Range": { "field": "year", "min": 2020, "max": 2025 } }
  ],
  "limit": 5,
  "return_stored": true,
  "highlight_field": "body"
}
EOF
cargo run -p searchlite-cli -- search "$INDEX" --request /tmp/request.json
```

- Typo-tolerant search (fuzzy matching):

```bash
cat > /tmp/request.json <<'EOF'
{
  "query": "body:rusk",
  "fuzzy": { "max_edits": 1, "prefix_length": 1, "max_expansions": 20, "min_length": 3 },
  "limit": 5,
  "return_stored": true
}
EOF
cargo run -p searchlite-cli -- search "$INDEX" --request /tmp/request.json
```

Aggregations use Elasticsearch-style JSON and require Cob containing `hits` and `aggregations`.

```bash
cat > /tmp/aggs.json <<'EOF'
{
  "langs": { "type": "terms", "field": "lang", "size": 5 },
  "views_stats": { "type": "stats", "field": "year" }
}
EOF

cargo run -p searchlite-cli -- search \
  --index "$INDEX" \
  --q "rust" \
  --limit 0 \
  --aggs-file /tmp/aggs.json
```

If you prefer inline JSON, pass `--aggs '{"langs":{"type":"terms","field":"lang"}}'`.

### Sorting

- Provide a `sort` array in the search request (or via `--sort "field:order,other_field:asc"` in the CLI). Each entry looks like `{"field":"year","order":"desc"}`; `_score` is also allowed.
- Sort targets must be fast keyword or numeric fields; the default order is ascending (descending for `_score`).
- Multi-valued fields use the minimum value for ascending sorts and the maximum for descending sorts; documents missing the field are placed last.
- Ordering is stable and tiebroken by segment/doc id so cursor pagination works reliably.

### Aggregations quick reference

- Field requirements: `terms` needs a fast keyword field; `range`, `histogram`, `stats`, `date_histogram` need fast numeric fields (date histograms accept numeric millis or RFC3339 strings stored as fast numeric); `top_hits` has no field requirement but returns stored fields/snippets when enabled.
- Stats semantics: `stats`/`extended_stats` aggregate over all field values; multi-valued fields contribute each entry (bucket `doc_count` stays per-document while `count` is per-value).
- Value count semantics: `value_count` counts field values (each entry from multi-valued fields, plus one per `missing` fill), not documents-with-values; this mirrors Elasticsearch's `value_count`.
- Bucket options: `terms` supports `size`, `shard_size`, `min_doc_count`, and nested `aggs`; `range`/`date_range` accept `key`, `from`, `to`, `keyed`; `histogram` supports `interval`, `offset`, `min_doc_count`, `extended_bounds`, `hard_bounds`, `missing`; `date_histogram` supports `calendar_interval` (day/week/month/quarter/year) or `fixed_interval` (e.g., `1d`, `12h`), optional `offset`, `min_doc_count`, `extended_bounds`, `hard_bounds`, `missing`.
- Top hits: `{"type":"top_hits","size":N,"from":M,"fields":["field1",...],"highlight_field":"body"}` returns sorted hits per bucket with `total` and optional snippets.
- Aggregations run over all matched documents (not just top-k); when `--limit 0` the search skips hit ranking and only returns `aggregations` (cursors are not supported with `--limit 0`).

Query syntax supports `field:term`, phrases in quotes (`"field:exact phrase"`), and negation with a leading `-term`.

- You can also pass the full search payload as JSON (same shape used by the upcoming HTTP service):

```bash
cat > /tmp/search_request.json <<'EOF'
{
  "query": "body:rust language",
  "fields": null,
  "filters": [
    { "KeywordEq": { "field": "lang", "value": "en" } },
    { "I64Range": { "field": "year", "min": 2020, "max": 2025 } }
  ],
  "limit": 5,
  "sort": [
    { "field": "year", "order": "desc" }
  ],
  "execution": "wand",
  "bmw_block_size": null,
  "return_stored": true,
  "highlight_field": "body"
}
EOF
cargo run -p searchlite-cli -- search "$INDEX" --request /tmp/search_request.json
```

Use `--request-stdin` to read the payload from standard input. When a JSON request is supplied, individual CLI flags (like `--q`, `--filter`, etc.) are ignored.

## Filters: examples

Filters operate on fast fields (`fast: true` in the schema). Keyword filters are case-insensitive; numeric ranges are inclusive. Nested filters bind to the same nested object (parent/child lineage).

### Basic keyword equality

```json
{
  "filters": [{ "KeywordEq": { "field": "lang", "value": "en" } }]
}
```

### Keyword membership (`IN`)

```json
{
  "filters": [{ "KeywordIn": { "field": "lang", "values": ["en", "fr"] } }]
}
```

### Numeric ranges (i64 and f64)

```json
{
  "filters": [
    { "I64Range": { "field": "year", "min": 2018, "max": 2024 } },
    { "F64Range": { "field": "score", "min": 0.25, "max": 0.9 } }
  ]
}
```

### Multi-valued fast fields

If your document has `tags: ["rust", "search"]` and `tags` is a fast keyword field:

```json
{
  "filters": [{ "KeywordEq": { "field": "tags", "value": "rust" } }]
}
```

Any value in the multi-valued column can satisfy the clause.

### Nested objects (single level)

Schema excerpt:

```json
{
  "nested_fields": [
    {
      "name": "comment",
      "fields": [
        {
          "type": "keyword",
          "name": "author",
          "fast": true,
          "stored": true,
          "indexed": true
        },
        {
          "type": "keyword",
          "name": "tag",
          "fast": true,
          "stored": true,
          "indexed": true
        }
      ]
    }
  ]
}
```

Filter: match documents with any comment whose author is `alice` and tag is `rust`:

```json
{
  "filters": [
    {
      "Nested": {
        "path": "comment",
        "filter": {
          "KeywordEq": { "field": "author", "value": "alice" }
        }
      }
    },
    {
      "Nested": {
        "path": "comment",
        "filter": {
          "KeywordEq": { "field": "tag", "value": "rust" }
        }
      }
    }
  ]
}
```

Because nested filters are scoped to the same object, this only passes when a single `comment` object has both `author=alice` and `tag=rust`.

### Deeply nested hierarchy

Schema excerpt:

```json
{
  "nested_fields": [
    {
      "name": "comment",
      "fields": [
        {
          "type": "keyword",
          "name": "author",
          "fast": true,
          "stored": true,
          "indexed": true
        },
        {
          "type": "object",
          "name": "reply",
          "fields": [
            {
              "type": "keyword",
              "name": "tag",
              "fast": true,
              "stored": true,
              "indexed": true
            }
          ]
        }
      ]
    }
  ]
}
```

Filter: require a comment with `author=bob` that has a reply tagged `y` (parent-child binding is enforced):

```json
{
  "filters": [
    {
      "Nested": {
        "path": "comment",
        "filter": {
          "KeywordEq": { "field": "author", "value": "bob" }
        }
      }
    },
    {
      "Nested": {
        "path": "comment",
        "filter": {
          "Nested": {
            "path": "reply",
            "filter": {
              "KeywordEq": { "field": "tag", "value": "y" }
            }
          }
        }
      }
    }
  ]
}
```

The inner `Nested` is evaluated only against replies belonging to the same `comment` object that satisfies the outer `Nested`.

### Numeric fields inside nested objects

Filter on nested numeric properties alongside keywords:

```json
{
  "filters": [
    {
      "Nested": {
        "path": "review",
        "filter": { "KeywordEq": { "field": "user", "value": "alice" } }
      }
    },
    {
      "Nested": {
        "path": "review",
        "filter": { "I64Range": { "field": "rating", "min": 5, "max": 8 } }
      }
    },
    {
      "Nested": {
        "path": "review",
        "filter": { "F64Range": { "field": "score", "min": 0.7, "max": 0.8 } }
      }
    }
  ]
}
```

All three clauses must match the same `review` object.

### Mixed nested and non-nested filters

Combine a top-level numeric range with nested filters:

```json
{
  "filters": [
    { "I64Range": { "field": "year", "min": 2020, "max": 2025 } },
    {
      "Nested": {
        "path": "comment",
        "filter": {
          "KeywordEq": { "field": "author", "value": "alice" }
        }
      }
    }
  ]
}
```

### Quick tips

- Mark filterable fields with `"fast": true` in the schema.
- For nested filters, wrap child clauses in `Nested` blocks; use additional nested blocks for deeper levels.
- Stored nested fields preserve structure; unstored fields are omitted in responses.

- Inspect or compact:

```bash
cargo run -p searchlite-cli -- inspect "$INDEX"
cargo run -p searchlite-cli -- compact "$INDEX"
```

## Using the Rust API

```rust
use searchlite_core::api::{
    builder::IndexBuilder, Index, Filter,
    types::{
        Aggregation, Document, ExecutionStrategy, IndexOptions, KeywordField, NumericField, Schema,
        SearchRequest, SortOrder, SortSpec, StorageType,
    },
};
use std::{collections::BTreeMap, path::PathBuf};

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
    ("_id".to_string(), serde_json::json!("doc-1")),
    ("body".to_string(), serde_json::json!("Rust is fast and reliable")),
    ("lang".to_string(), serde_json::json!("en")),
    ("year".to_string(), serde_json::json!(2024)),
].into_iter().collect() };
writer.add_document(&doc)?;

// Insert multiple documents in one batch.
let more_docs = vec![
    Document { fields: [("_id".to_string(), serde_json::json!("doc-2")), ("body".to_string(), serde_json::json!("SQLite vibes for search")), ("lang".to_string(), serde_json::json!("en")), ("year".to_string(), serde_json::json!(2023))].into_iter().collect() },
    Document { fields: [("_id".to_string(), serde_json::json!("doc-3")), ("body".to_string(), serde_json::json!("Embedded search engine demo")), ("lang".to_string(), serde_json::json!("en")), ("year".to_string(), serde_json::json!(2022))].into_iter().collect() },
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
    sort: vec![SortSpec { field: "year".into(), order: Some(SortOrder::Desc) }],
    cursor: None,
    execution: ExecutionStrategy::Wand,
    bmw_block_size: None,
    fuzzy: None,
    return_stored: true,
    highlight_field: Some("body".into()),
    aggs: [(
        "langs".to_string(),
        Aggregation::Terms(Box::new(searchlite_core::api::types::TermsAggregation {
            field: "lang".into(),
            size: Some(3),
            shard_size: None,
            min_doc_count: None,
            missing: None,
            aggs: BTreeMap::new(),
        })),
    )]
    .into_iter()
    .collect(),
    #[cfg(feature = "vectors")]
    vector_query: None,
})?;
for hit in results.hits {
    println!("doc {} score {:.3} fields {:?}", hit.doc_id, hit.score, hit.fields);
}
```

Search responses include a `next_cursor` when additional hits remain.

- JSON/SDK: send that value in the `cursor` field to fetch the next page without computing offsets.
- CLI: `cargo run -p searchlite-cli -- search "$INDEX" --q "rust" --limit 5 --cursor "$NEXT_CURSOR"`.
- FFI: pass the cursor string to the `cursor` argument.
- Cursors are opaque and bounded (up to ~50k returned hits) to avoid unbounded memory use; very deep pagination returns an error instead of over-consuming resources.

`Index::open(opts)` opens an existing index; `Index::compact()` rewrites all segments into one. WAL-backed writers queue documents until `commit` is called; `rollback` drops uncommitted changes.

### Query execution modes

- `execution`: choose `"bm25"` (full evaluation), `"wand"` (exact WAND pruning), or `"bmw"` (block-max WAND). Default is `wand`.
- `bmw_block_size`: optional block size when using BMW pruning.

The CLI exposes `--execution` and `--bmw-block-size` on `search`. A small synthetic benchmark that compares the strategies lives in `searchlite-core/examples/pruning.rs` (`cargo run -p searchlite-core --example pruning`).

### In-memory indexes

For ephemeral or test-heavy scenarios, set `storage: StorageType::InMemory` in `IndexOptions`. The API and search behavior stay the same, but no files are created on disk. (The CLI currently uses filesystem storage only.)

### WASM

#### Builds and targets

- Install `wasm-pack` (e.g., `brew install wasm-pack` or `cargo install wasm-pack`) before building.
- Threaded wasm needs atomics/bulk-memory and build-std; `searchlite-wasm/rust-toolchain.toml` pins a nightly with `rust-src` and the wasm target, and `searchlite-wasm/.cargo/config.toml` sets the required rustflags/build-std. Rustup will fetch the nightly toolchain automatically when you build this crate.
- Default build for browsers and module workers (ESM): `wasm-pack build searchlite-wasm --target web --release`.
- Classic workers / `importScripts` need a separate build: `wasm-pack build searchlite-wasm --target no-modules --release` (or `--target bundler` if you want a bundler to wrap it).
- Threaded build (requires COOP/COEP + SharedArrayBuffer): `wasm-pack build searchlite-wasm --target web --release -- --features threads`.

#### Choosing the right build

- **Browser window + module workers**: use the `--target web` build; this is a single build that works in both environments.
- **Classic web worker / service worker (no modules)**: use `--target no-modules` (or `--target bundler`) because `importScripts` cannot load ES modules.
- **Threads**: build with `--features threads` and serve with COOP/COEP headers; this is a separate build and is not available in service workers.

#### Running the demo

- Serve the crate directory over HTTP with COOP/COEP headers so SharedArrayBuffer works (e.g., `cd searchlite-wasm && npx http-server -c-1 --cors -p 8080 -H "Cross-Origin-Opener-Policy: same-origin" -H "Cross-Origin-Embedder-Policy: require-corp"`).
- Open `http://localhost:8080/index.html`. The bundled page imports `pkg/searchlite_wasm.js`, initializes the module, and provides a lightweight schema/upload/search demo in the browser.

#### API usage

- Instantiate from JS with `await Searchlite.init("demo-db", JSON.stringify(schema), "indexeddb")` (default) or `"memory"` for ephemeral indexes. `init` reopens existing indexes with the same name and validates schemas; mismatches return an error.
- Prefer `add_documents([...])` for bulk ingest and call `commit()` to flush everything to the manifest.

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
