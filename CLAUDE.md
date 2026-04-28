# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository

Searchlite is an embedded full-text search engine for Rust ("the SQLite of search"). The Rust workspace is the source of truth; all other surfaces (CLI, HTTP, FFI, Node, WASM) are thin adapters around `searchlite-core`.

## Workspace Layout

- `searchlite-core/` — indexing, query, scoring, durability. **Source of truth for correctness.**
- `searchlite-cli/` — `searchlite` binary (`init`/`add`/`commit`/`search`/`compact`/`http`).
- `searchlite-http/` — Axum-based REST server over one or more indexes.
- `searchlite-ffi/` — C ABI shared library (`cdylib` + `rlib`).
- `searchlite-node/` — N-API Node.js bindings + TypeScript client (npm package `searchlite-js`).
- `searchlite-wasm/` — `wasm-bindgen` browser bindings, IndexedDB-backed storage.
- `integration/` — cross-surface contract/matrix/lifecycle/adversarial tests (driven by the harnesses in `integration/src/surfaces/`).
- `examples/`, `docs/`, `BENCHMARKS.md` — quickstarts, deep-dives, perf methodology.

## Common Commands

Use `just` (recipes in `Justfile`) or `cargo` directly.

```bash
# Build / test / lint the whole workspace (mirrors CI)
cargo build --all --all-features          # just build
cargo test  --all --all-features          # just test
cargo fmt --all                           # just fmt
cargo clippy --all --all-features --all-targets -- -D warnings   # just lint

# Single test
cargo test -p searchlite-core --all-features <test_name>
cargo test -p integration --test contracts_core <test_name>

# Benchmarks (Criterion)
cargo bench -p searchlite-core            # just bench

# Integration matrix
INTEGRATION_MODE=quick cargo test -p integration --all-features    # just test-integration-quick
INTEGRATION_MODE=full  cargo test -p integration --all-features    # just test-integration-full
# Optional sharding: INTEGRATION_MATRIX_SHARDS=N INTEGRATION_MATRIX_SHARD=i

# Node bindings (from searchlite-node/)
npm run build:native:debug && npm run build:ts
npm run lint && npm run typecheck && npm test     # just check-node

# WASM
wasm-pack build  searchlite-wasm --target web --release
wasm-pack test   --headless --firefox searchlite-wasm
```

CI (`.github/workflows/ci.yml`) runs fmt-check + clippy + build + test on Rust 1.88 and stable on every PR (extended on push: beta, nightly), plus separate Node, WASM-Chrome (lib-only), WASM-Firefox (full), and benchmark jobs. **Format check only enforces on `stable`**; clippy runs on every matrix entry.

The Rust toolchain is pinned to **1.92.0** (`rust-toolchain.toml`); MSRV is **1.88.0**. `rustfmt.toml` sets `tab_spaces = 2`, `max_width = 100`.

## Architecture (read these together)

The lifecycle is **WAL → segment → manifest**, single-writer/multi-reader:

1. `IndexWriter::add_document` buffers ops and appends to the WAL (fsync'd, append-only).
2. `commit()` flushes the buffer, builds a new immutable segment (postings + docstore + fast columns), then atomically swaps the manifest (rename + dir fsync). WAL is truncated only after manifest is durable.
3. `IndexReader` reads the manifest to discover segments. After commit, HTTP visibility requires `POST /refresh` unless launched with `--refresh-on-commit`.
4. Tiered merge runs on commit; `compact()` rewrites everything into one segment.
5. Crash safety: dying after manifest persist but before WAL truncate triggers WAL replay on next open — no data loss.

Postings use **128-doc block-max** layout. Search supports three execution modes for the same query: `bm25` (full eval), `wand` (default, exact pruning), `bmw` (block-max WAND). Filters and aggregations operate on **fast fields** (memory-mapped columnar) — non-fast fields can't be filtered/aggregated.

### Where things live in `searchlite-core/src`

- `index/` — `wal.rs`, `segment.rs`, `manifest.rs`, `merge.rs`, `postings.rs`, `docstore.rs`, `fastfields.rs`, `terms.rs`, `codec.rs`, `directory.rs`, `highlight.rs`, `json_schema.rs`. **The durability/storage core.** `index/mod.rs` defines `Index`, `InnerIndex`, and create/open paths.
- `api/` — public surface: `builder.rs`, `writer.rs`, `reader.rs`, `types.rs`, `query.rs`, `scoring.rs`, `pagination.rs`, `phrase.rs`, `suggestion.rs`, `materialization.rs`, `query_eval.rs`, `term_expansion.rs`, `errors.rs`. Re-exported from `api/mod.rs`.
- `query/` — execution: `bm25.rs`, `wand.rs`, `boolean.rs`, `phrase.rs`, `planner.rs`, `collector.rs`, `filters.rs`, `score_functions.rs`, `script.rs`, `sort.rs`, `aggregation.rs`, `aggs/`.
- `analysis/` — tokenization/stemming/normalization.
- `storage/` — `Storage` trait with `FsStorage` and `InMemoryStorage` impls (used by all surfaces, including WASM via the `browser` feature).
- `vectors/` — HNSW ANN (gated by `vectors`). `gpu/` is a stub (gated by `gpu`).

### Schema invariants (drive the whole system)

- `doc_id_field` (default `_id`) is required and is the upsert/delete key.
- `stored: true` → returnable/highlightable. `fast: true` → filterable/aggregatable. `indexed: true` → searchable.
- Nested fields flatten to dotted paths (`comment.author`) but preserve nested structure in stored output. Nested filters need `Nested` blocks to bind clauses to the same object.
- `terms`/`significant_terms`/`rare_terms` require fast keyword fields; numeric/date aggs require fast numeric fields.

## Feature Flags

Defaults are off — opt in per crate. Source: `docs/feature-flags.md`.

- `vectors` (core/cli/http/ffi/wasm/integration) — HNSW ANN; adds `bytemuck`, `bincode`. Test both vector-only and hybrid paths; assert `vector_score` presence when running.
- `write-key` (core only) — Argon2 write protection. Always enabled in `searchlite-cli`, `searchlite-http`, and `searchlite-ffi` (they depend on core with `features = ["write-key"]`).
- `zstd` — stored-field compression.
- `ffi` — required for `searchlite-ffi` C exports.
- `browser` — used internally by `searchlite-wasm`; **don't set manually**.
- `gpu` — placeholder, no-op.
- `threads` (wasm only) — `SharedArrayBuffer` rayon; needs nightly toolchain pinned in `searchlite-wasm/rust-toolchain.toml` and COOP/COEP headers.

CI runs everything with `--all-features`, so any new code must compile and pass tests under all-features-on.

## Cross-surface Contracts

Behavior must be identical across CLI/HTTP/Core. The `integration/` crate enforces this:

- `integration/src/surfaces/` defines `SurfaceHarness` impls for `core`, `cli`, `http`.
- `tests/contracts_*.rs` drives each surface with the same fixtures and asserts shapes match.
- `tests/lifecycle_matrix.rs`, `feature_cross_matrix.rs`, `feature_matrix_execution.rs`, `adversarial_matrix.rs` enumerate (surface × pagination × lifecycle stage × execution mode × storage) combinations. `INTEGRATION_MODE=quick` (default) runs a sampled subset; `full` runs the whole matrix.
- HTTP error payloads have shape `{"error":{"type":"...","reason":"..."}}` — preserve this when adding/altering errors.
- Determinism: small fixed corpora, fixed doc ids, no time-based assertions. Sort hits before comparing unless an explicit `sort` is requested.
- All three execution modes (`bm25`/`wand`/`bmw`) must return the same top-K for deterministic fixtures.

When changing any user-visible behavior, update or add an integration contract test alongside the unit test.

## Conventions

- **Conventional Commits** for all messages (`feat:`, `fix:`, `perf:`, `chore:`, `docs:`, etc.) — `release-plz` and `git-cliff` consume them.
- Workspace version is `Cargo.toml [workspace.package].version`. CI fails if `searchlite-node/package.json` version drifts from the workspace version — bump together.
- `searchlite-core` and `searchlite-cli` have **independent** `version` fields (not workspace-inherited); other member crates inherit via `version.workspace = true`.
- Performance matters: this is a hot-path library. Prefer explicit, allocation-aware code over clever abstractions. Run `cargo bench -p searchlite-core` before landing perf-sensitive changes.
- Keep the boundary clean: correctness lives in `searchlite-core`; CLI/HTTP/FFI/Node/WASM are surfaces around it. Don't reimplement search logic in adapters.

## Reference Documentation

`docs/intro.md` (architecture deep-dive), `docs/quickstart.md`, `docs/rust-api.md`, `docs/schema.md`, `docs/queries.md`, `docs/filters.md`, `docs/aggregations.md`, `docs/cli.md`, `docs/http.md`, `docs/vectors.md`, `docs/wasm.md`, `docs/ffi.md`, `docs/write-key.md`, `docs/feature-flags.md`. JSON Schemas: `index-schema.json`, `search-request.schema.json`. HTTP contract: `openapi.yaml`. Benchmark methodology: `BENCHMARKS.md`.
