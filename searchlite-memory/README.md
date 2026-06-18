# searchlite-memory

A repository-local **memory** MCP (Model Context Protocol) server for AI coding
agents, backed by [searchlite](https://github.com/davidkelley/searchlite)'s
full-text (BM25) **and** vector (HNSW) search.

It gives agents four tools — `remember`, `recall`, `get`, `forget` — over stdio.
Memory is committed into the repository as a human-readable JSONL ledger (plus a
quantized vector sidecar) so it travels with the code and is shared across a
team. The searchlite binary index is a gitignored, rebuildable cache.

> Status: in development. This package is built in stages; see the repository
> plan for the roadmap. The CLI scaffold (`serve` / `rebuild` / `doctor` /
> `init`) is in place; tool/server behavior lands in later stages.

## Design at a glance

- **Committed source of truth:** `.searchlite-memory/memory.jsonl` (text) +
  `vectors.jsonl` (int8-quantized embeddings + a model fingerprint). The
  binary index under `.searchlite-memory/index/` is gitignored and rebuilt.
- **Hybrid recall:** two searchlite calls (BM25 + pure-vector) fused with
  Reciprocal Rank Fusion (RRF, k=60), then re-scored by recency + importance.
- **Pluggable embeddings:** a local ONNX model by default
  (`all-MiniLM-L6-v2`, 384-dim, via `@huggingface/transformers`), optional
  external providers, and a full-text-only fallback when no embedder is
  configured.

## Requirements

- Node.js >= 20.
- `searchlite-js` built with the `vectors` feature (the default).

## Development

```bash
npm install          # links the local ../searchlite-node build (file: dependency)
npm run build        # swc (ESM JS) + tsc (declarations)
npm test             # vitest
npm run typecheck
npm run lint
```

> Note: the `searchlite-js` dependency is a `file:` link to the in-repo
> `searchlite-node` package for local development. Before publishing,
> repin it to the released registry version (`>= ` the version that ships the
> `vectors` default + `delete` binding).
