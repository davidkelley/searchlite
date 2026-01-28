---
name: index-lifecycle
description: Understanding and modifying Searchlite’s storage/indexing pipeline (WAL → segment build → manifest/compaction) and storage backends; use when working on ingest, durability, fast fields, or segment layouts.
---

# Index Lifecycle Cheatsheet

## The Searchlite mental model (one paragraph)

Searchlite serves **one index directory** per instance (CLI path or HTTP `--index`). A **single writer** buffers operations, appends them to a WAL, and on `commit` writes a new immutable segment and atomically updates a manifest. Readers consult the manifest to know which segments exist; compaction merges segments and drops deleted/obsolete data.

---

## Glossary

- **Schema**: declares fields, analyzers, what’s stored vs fast vs indexed.
- **WAL**: write-ahead log for durability of add/delete operations.
- **Segment**: immutable set of postings/docstore/fast fields written on commit.
- **Manifest**: authoritative list of segments that define the current index state.
- **Refresh**: reload readers so they observe the latest manifest.
- **Compaction**: rewrite live docs into fewer segments; cleans duplicates/old versions.

---

## Lifecycle steps (authoritative sequence)

### 1) Init

- Creates the index directory and writes schema/manifest state so it can accept docs.

### 2) Ingest (add/update/delete)

- Docs are upserted by primary key (`doc_id_field`, default `_id`)
- Deletes are queued by id
- Important: writes are buffered/queued; they are not visible to search yet

### 3) Commit

Commit is the “durable + visible” boundary:

- writer flushes buffered ops
- builds new segment files
- persists them
- atomically updates manifest

Durability contract:

- segment files/manifests are fsync’d on write
- WAL truncation only happens after manifest is persisted/synced
- manifest uses atomic rename + directory fsync

Crash window (expected behavior):

- if the process dies after manifest persisted but before WAL truncation, WAL replay re-applies the last batch
- no data loss; compaction cleans extra generations

### 4) Refresh (reader visibility)

- With HTTP, visibility may require explicit `POST /refresh` unless configured with `--refresh-on-commit`.

### 5) Search

Search uses:

- BM25 scoring by default (with pruning modes)
- phrase and fuzzy matching
- filters on fast fields
- aggregations on fast fields
- optional highlights, collapse/inner_hits, suggest, rescore, profile/explain

Execution modes (perf/correctness relevant):

- `bm25` full evaluation
- `wand` exact pruning (default)
- `bmw` block-max WAND pruning (tunable block size)

### 6) Maintain

- Inspect: see manifest + segments
- Stats: see doc/segment counts
- Compact: merge segments and drop tombstoned/obsolete docs

---

## File layout expectations (conceptual)

The exact file names are an implementation detail, but the directory contains:

- a manifest file (authoritative segment list)
- a WAL file
- per-segment files (postings/docstore/fast/meta/etc)
- optional vector index structures when `vectors` is enabled

When changing segment layout, preserve:

- compatibility with existing indexes or provide a migration story
- checksum/validation (if present)
- atomicity during commit and cleanup after rollback/failed commit

---

## Schema invariants that drive the whole system

- `doc_id_field` is required and is the key for upsert/delete semantics.
- `stored: true` controls whether values can be returned/highlighted.
- `fast: true` controls filters + aggregations performance/feasibility.
- Nested fields flatten to dotted paths (e.g. `comment.author`) while preserving stored nested structure in responses.
- Nested filters must be expressed with `Nested` blocks so clauses bind to the same object instance.

---

## Aggregations/filters coupling (easy to break)

- `terms`, `significant_terms`, `rare_terms` require a fast keyword field.
- numeric/date aggregations require fast numeric fields.
- multi-valued fields contribute multiple values to metrics (doc_count remains per-doc).

If you modify fast-field encoding, bucket iteration, or numeric parsing:

- add tests that validate aggs outputs (not just existence of buckets).

---

## Vector search (feature `vectors`)

Vector search is approximate ANN (HNSW):

- dimension mismatches are rejected
- cosine vectors are normalized automatically
- tune recall/perf with `candidate_size` and `ef_search`
- `vector_filter` can reduce candidate selection set

If you touch vector integration:

- test both vector-only and hybrid search paths
- confirm response includes `vector_score` when vector search runs

---

## “Where to make changes” guidance

- Core indexing/query/durability lives in `searchlite-core`
- CLI wiring + UX lives in `searchlite-cli`
- HTTP server wiring + limits + endpoints live in `searchlite-http` (and/or the CLI `http` subcommand)
- wasm storage backend + bindings live in `searchlite-wasm`
- C ABI lives in `searchlite-ffi`

Keep boundaries clean:

- `searchlite-core` should remain the source of truth for correctness; CLI/HTTP adapt surfaces around it.
