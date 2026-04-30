<p align="center">
  <h1 align="center">searchlite</h1>
  <p align="center">
    <strong>The SQLite of search.</strong>
    <br />
    An embedded full-text search engine for Rust. No JVM, no cluster, no external
    process &mdash; just add a crate and search. WAL-backed durability, BM25
    relevance with WAND/BMW pruning, aggregations, filters, highlights, and
    fuzzy matching out of the box.
  </p>
</p>

<p align="center">
  <a href="https://crates.io/crates/searchlite-core"><img src="https://img.shields.io/crates/v/searchlite-core.svg" alt="crates.io" /></a>
  <a href="https://github.com/davidkelley/searchlite/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license" /></a>
  <a href="https://github.com/davidkelley/searchlite/actions"><img src="https://img.shields.io/github/actions/workflow/status/davidkelley/searchlite/ci.yml?branch=main" alt="CI" /></a>
</p>

---

## Try it

You don't need Rust installed to try Searchlite. Pick whichever path is fastest for you:

### One-line install (macOS / Linux)

```bash
curl -fsSL https://searchlite.dev/install | sh
```

Then create an index, add some documents, and search:

```bash
# Create an index with a minimal schema
searchlite init /tmp/myindex schema.json

# Add documents (newline-delimited JSON)
searchlite add /tmp/myindex docs.jsonl
searchlite commit /tmp/myindex

# Search
searchlite search /tmp/myindex -q "rust search" --limit 5
```

### Docker (zero install)

```bash
docker run --rm -p 8080:8080 -v "$PWD:/data" \
  ghcr.io/davidkelley/searchlite:latest \
  http --index default:/data --bind 0.0.0.0:8080
```

Then query over HTTP:

```bash
curl -s http://localhost:8080/indexes/default/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "rust search", "limit": 5}'
```

### As a Rust crate

```toml
[dependencies]
searchlite-core = "0.5"
```

```rust
use searchlite_core::api::{
    builder::IndexBuilder,
    types::{Document, IndexOptions, KeywordField, NumericField, Schema, SearchRequest, StorageType},
    Filter,
};
use std::path::PathBuf;

// 1. Define a schema
let mut schema = Schema::default_text_body(); // text field "body" with default analyzer
schema.keyword_fields.push(KeywordField {
    name: "lang".into(), stored: true, indexed: true, fast: true,
});
schema.numeric_fields.push(NumericField {
    name: "year".into(), i64: true, fast: true, stored: true,
});

// 2. Create an index
let path = PathBuf::from("/tmp/example_idx");
let opts = IndexOptions {
    path: path.clone(),
    create_if_missing: true,
    bm25_k1: 0.9,
    bm25_b: 0.4,
    ..Default::default()
};
let index = IndexBuilder::create(&path, schema, opts)?;

// 3. Add documents and commit
let mut writer = index.writer()?;
writer.add_document(&Document {
    fields: [
        ("_id".into(), serde_json::json!("doc-1")),
        ("body".into(), serde_json::json!("Rust is a fast systems language")),
        ("lang".into(), serde_json::json!("en")),
        ("year".into(), serde_json::json!(2024)),
    ].into_iter().collect(),
})?;
writer.commit()?;

// 4. Search
let reader = index.reader()?;
let results = reader.search(
    &SearchRequest::new("rust language")
        .with_limit(10)
        .with_filter(Filter::KeywordEq { field: "lang".into(), value: "en".into() })
        .with_return_stored(true)
        .with_highlight_field("body"),
)?;

for hit in &results.hits {
    println!("{} (score: {:.2})", hit.doc_id, hit.score);
}
```

### As an npm package

```bash
npm install searchlite-js
```

```javascript
import { EmbeddedIndex } from 'searchlite-js';

const index = new EmbeddedIndex('./my-index', {
  schema: { title: 'text', body: 'text', tag: 'keyword' },
});

await index.addMany([
  { _id: '1', title: 'Getting Started', body: 'Hello, world!', tag: 'intro' },
  { _id: '2', title: 'Advanced Search', body: 'Filters, facets, and more', tag: 'guide' },
]);
await index.commit();

const results = await index.search('hello');
console.log(results.hits[0].docId); // "1"

await index.close();
```

See the [quickstart](docs/quickstart.md#nodejs--typescript) for a full TypeScript example with Zod-validated typed search.

---

## Features

- **BM25 scoring** &mdash; tunable relevance with WAND/BMW pruning for fast exact top-K, even over millions of documents
- **Rich query DSL** &mdash; bool, phrase, fuzzy, prefix, wildcard, regex, multi-match, function scores, and rescoring
- **Filters** &mdash; narrow results by keyword, numeric range, nested objects, or boolean combinations without affecting relevance
- **Aggregations** &mdash; build faceted navigation, dashboards, and analytics with terms, histograms, stats, percentiles, and pipelines
- **Highlighting** &mdash; show users why a result matched with multi-field, phrase-aware snippet extraction
- **Nested documents** &mdash; model arrays of objects (reviews, comments, metadata) with per-object filtering and aggregation
- **Field collapsing** &mdash; deduplicate results by author, publisher, or any keyword field, with inner hits for "more like this"
- **Pagination** &mdash; cursor-based, search_after, and offset modes for any result set size
- **Optional vector search** &mdash; HNSW approximate nearest neighbors with hybrid BM25+vector blending for semantic search
- **WAL durability** &mdash; crash-safe writes with atomic manifest, fsync, and WAL replay. No data loss, ever

---

## Performance

Benchmarked on Apple M3 Max (36 GB), Rust 1.92.0, in-memory storage. All times are Criterion medians; your hardware will vary. See [BENCHMARKS.md](BENCHMARKS.md) for methodology and full results.

| Benchmark | Docs | Median |
|---|---|---|
| Index 100K docs | 100,000 | **4.21 s** (~23,700 docs/sec) |
| Single-term search | 100,000 | **4.26 ms** |
| Bool AND search | 100,000 | **6.25 ms** |
| Filtered search | 100,000 | **4.42 ms** |
| Terms aggregation | 100,000 | **8.29 ms** |

---

## Multi-platform

| Platform | Crate | Status |
|---|---|---|
| **Rust** | [`searchlite-core`](searchlite-core/) | Stable &mdash; full API |
| **CLI** | [`searchlite-cli`](searchlite-cli/) | Stable &mdash; init, add, commit, search, compact |
| **HTTP** | [`searchlite-http`](searchlite-http/) | Stable &mdash; REST API over one or more indexes |
| **C FFI** | [`searchlite-ffi`](searchlite-ffi/) | Stable &mdash; shared library + C header |
| **Node.js** | [`searchlite-js`](searchlite-node/) | Stable &mdash; native bindings + HTTP client, TypeScript + Zod |
| **WASM** | [`searchlite-wasm`](searchlite-wasm/README.md) | Pre-1.0 &mdash; IndexedDB-backed browser search with worker runtime, migrations, quota handling |
| **Elasticsearch adapter** | [`searchlite-adapter-elastic`](searchlite-adapter-elastic/README.md) | Pre-1.0 &mdash; read-only Elasticsearch HTTP-API compatibility for existing ES clients (Kibana, official SDKs) |

---

## Architecture

```
                     ┌──────────────┐
                     │  IndexWriter │
                     └──────┬───────┘
                            │ add_document()
                            ▼
                     ┌──────────────┐
                     │     WAL      │  ← append-only, fsync'd
                     └──────┬───────┘
                            │ commit()
                  ┌─────────┴─────────┐
                  ▼                   ▼
           ┌────────────┐     ┌────────────┐
           │  Segment 0 │     │  Segment 1 │  ...
           │  (postings, │     │  (postings, │
           │   docstore, │     │   docstore, │
           │   fast cols)│     │   fast cols)│
           └─────────────┘     └─────────────┘
                  │                   │
                  └─────────┬─────────┘
                            ▼
                     ┌──────────────┐
                     │   Manifest   │  ← atomic rename + dir fsync
                     └──────┬───────┘
                            │ reader()
                            ▼
                     ┌──────────────┐
                     │  IndexReader │  ← search(), mget(), multi_search()
                     └──────────────┘
```

- **Segments** are immutable inverted indexes with block-max postings (128-doc blocks for WAND/BMW).
- **Fast fields** are memory-mapped columnar stores for zero-copy filter evaluation.
- **Tiered merge policy** automatically merges small segments on commit; `compact()` rewrites everything into one segment.
- **Crash safety**: if the process dies mid-commit, WAL replay recreates the segment on next open. No data loss.

---

## Documentation

| Topic | Link |
|---|---|
| Getting started (CLI) | [docs/quickstart.md](docs/quickstart.md) |
| Rust API guide | [docs/rust-api.md](docs/rust-api.md) |
| Schema and fields | [docs/schema.md](docs/schema.md) |
| CLI reference | [docs/cli.md](docs/cli.md) |
| HTTP service | [docs/http.md](docs/http.md) |
| Elasticsearch adapter | [docs/adapters/elasticsearch.md](docs/adapters/elasticsearch.md) |
| Query DSL | [docs/queries.md](docs/queries.md) |
| Filters | [docs/filters.md](docs/filters.md) |
| Aggregations | [docs/aggregations.md](docs/aggregations.md) |
| Collapsing and highlighting | [docs/collapsing-and-highlighting.md](docs/collapsing-and-highlighting.md) |
| Vector search | [docs/vectors.md](docs/vectors.md) |
| In-memory indexes | [docs/in-memory.md](docs/in-memory.md) |
| WASM bindings | [docs/wasm.md](docs/wasm.md) |
| C FFI | [docs/ffi.md](docs/ffi.md) |
| Write-key protection | [docs/write-key.md](docs/write-key.md) |
| Feature flags | [docs/feature-flags.md](docs/feature-flags.md) |
| Benchmarks | [BENCHMARKS.md](BENCHMARKS.md) |
| Architecture deep-dive | [docs/intro.md](docs/intro.md) |
| Node.js bindings | [searchlite-node/README.md](searchlite-node/README.md) |
| Binding behaviors | [docs/bindings.md](docs/bindings.md) |

---

## Development

```bash
# Build
cargo build --all --all-features        # or: just build

# Test
cargo test --all --all-features         # or: just test

# Benchmark
cargo bench -p searchlite-core          # or: just bench

# Lint
cargo fmt --all
cargo clippy --all --all-features -- -D warnings
```

Rust toolchain is pinned to **1.92.0** (`rust-toolchain.toml`). CI runs across 5 toolchains (1.88.0, 1.92.0, stable, beta, nightly).

---

## License

MIT
