# Searchlite in a Nutshell

Searchlite is a lightweight, SQLite-flavored search engine: a single on-disk index with WAL durability, BM25 relevance, and modern search features, all in a lean Rust binary (no JVM, no cluster to manage).

## Why Searchlite?

- **Minimal ops**: open an index path and go — no Zookeeper, no cluster sizing, and simple backups/compaction.
- **Predictable costs**: runs well on one NVMe-backed box; scale up before you have to scale out.
- **Developer friendly**: clear Rust API, CLI, optional C FFI and wasm; JSON-based requests mirror how you think about queries and filters.
- **SQLite mindset**: single-writer/multi-reader with WAL-style commits and atomic manifests; readers stay online during writes.
- **Performance without bloat**: BM25 with WAND/BMW pruning, fast fields, and optional vectors—all without a heavyweight runtime.

## Try It Fast

Walk through ingesting a small JSONL file and running your first search in minutes: see [Quickstart](quickstart.md) (coming next).

## Features

- **Relevance + pruning**: BM25 scoring with phrase matching, typo tolerance, prefixes/wildcards/regex, and WAND/BMW block-max pruning for low-latency top-k.
- **Filters, sorts, and aggregations**: fast fields for keyword/numeric filters, stable sorting, and Elasticsearch-style aggs (terms, stats, histograms, date histograms).
- **Nested correctness**: per-object nested filters so mixed/nested data stays consistent; stored nested fields preserve structure.
- **Stored fields and highlighting**: return stored fields with snippets/highlights; cursors and stable sorts enable deep pagination.
- **Ergonomic surfaces**: Rust API, CLI, optional C FFI, experimental wasm/IndexedDB; filesystem or in-memory storage.
- **Vector and hybrid search**: optional HNSW vectors with BM25+vector blending, plus zstd compression and GPU rerank hooks behind feature flags.

## Highlights

- **Single-node simplicity**: one on-disk index, WAL-backed writer, atomic manifest updates; easy to embed or run beside your app.
- **Low operational overhead**: no JVM/GC, no cluster services, straightforward compaction/inspect flows.
- **Cost efficiency**: uses local NVMe and RAM effectively; fewer moving parts and replicas compared to distributed stacks.
- **Clear schema and requests**: explicit analyzers, fields, filters, and aggs; the CLI and JSON payloads make behavior easy to reason about.

## Surfaces and Feature Flags

| Surface | Availability | Notes |
| --- | --- | --- |
| Rust API | ✅ | Default crate surface. |
| CLI | ✅ | Ships with the workspace. |
| C FFI | ⚙️ `ffi` feature | Enable the `ffi` feature to build the shared library/header. |
| WASM | ⚙️ `wasm` crate | Experimental IndexedDB/memory backends; needs wasm-pack build. |
| Vectors | ⚙️ `vectors` feature | Enables HNSW + hybrid BM25/vector blending. |
| GPU rerank | ⚙️ `gpu` feature | Stub hooks for GPU reranking. |
| Compression | ⚙️ `zstd` feature | Compress stored fields. |

## Limits and Expectations

- Single-node scope: no built-in sharding/replication; think “SQLite for search.”
- Concurrency model: single writer, many readers; readers stay live across commits.
- HA story: primary + standby via snapshots/backups rather than cluster replication.
- Workload fit: low-latency top-k queries with filters/aggregations; not a distributed analytics engine.

## Operational Basics

- Commit cadence: buffer writes, then `commit`; compact occasionally to reduce segment sprawl.
- Backups: snapshot/copy the index directory after a commit (or stop the writer), then restore by opening the path.
- Inspection and debug: `inspect` to view manifests/segments; `explain`/`profile` flags to trace scoring and timings.
- Storage: favor local NVMe with `noatime`; use in-memory mode for ephemeral/testing workloads.

## Compatibility and License

- Rust 1.88.0+ (CI also checks 1.92, stable, beta, nightly); Linux and macOS are primary targets.
- MIT licensed.

## Who Will Like It

- Product teams who want full-text search without running Solr/Elasticsearch/Meilisearch.
- Single-tenant SaaS or per-tenant instances that can live on one box.
- Edge/offline or appliance-style deployments where local-first matters.
- Cost-sensitive workloads that can accept single-node HA (primary + standby) rather than a full cluster.
- Enabling embedded search in desktop/mobile apps via wasm or FFI.
