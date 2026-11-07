# searchlite-memory

A repository-local **memory** MCP (Model Context Protocol) server for AI coding
agents, backed by [searchlite](https://github.com/davidkelley/searchlite)'s
full-text (BM25) **and** vector (HNSW) search.

It gives agents four tools — `remember`, `recall`, `get`, `forget` — over stdio.
Memory is committed into the repository as a human-readable JSONL ledger (plus a
quantized vector sidecar) so it travels with the code and is shared across a
team. The searchlite binary index is a gitignored, rebuildable cache.

## Quick start

```bash
# In your repo (a one-time scaffold of .mcp.json + .searchlite-memory/ config):
npx searchlite-memory init

# Your MCP client (e.g. Claude Code) launches the server from .mcp.json.
# CLI subcommands:
searchlite-memory serve            # MCP stdio server (default)
searchlite-memory rebuild [--reembed]
searchlite-memory doctor           # health report
```

The local embedder downloads a small ONNX model on first use. Set
`SEARCHLITE_MEMORY_EMBEDDER=none` for a zero-dependency, full-text-only mode.

## Configuration (env)

| Variable | Default | Purpose |
|---|---|---|
| `SEARCHLITE_MEMORY_DIR` | `$CLAUDE_PROJECT_DIR/.searchlite-memory` or `./.searchlite-memory` | Memory directory. **Set this to an absolute path when your MCP host does not inject `CLAUDE_PROJECT_DIR`** (e.g. Cursor, Cline) — otherwise it resolves from the server's cwd. |
| `SEARCHLITE_MEMORY_EMBEDDER` | `local` | `local` \| `none` \| `openai` \| `voyage` \| `cohere` |
| `SEARCHLITE_MEMORY_MODEL` / `_DIM` / `_QUANT` | all-MiniLM-L6-v2 / 384 / q8 | Local model pin (also the committed-vector fingerprint). |
| `SEARCHLITE_MEMORY_RRF_K`, `_WEIGHTS`, `_HALF_LIFE_HOURS`, `_POOL_SIZE`, `_RECALL_LIMIT` | 60 / 0.6/0.2/0.15/0.05 / 168 / 50 / 8 | Retrieval + scoring tunables. |
| `SEARCHLITE_MEMORY_LOCK_STALE`, `_LOCK_RETRIES`, `SEARCHLITE_MEMORY_NO_LOCK` | 30000 / 10 / off | Cross-process lock (set `NO_LOCK=1` on NFS/network mounts where file locking is unreliable). |

Run `searchlite-memory doctor` to verify the setup (it warns when
`CLAUDE_PROJECT_DIR` is unset and when the committed files are accidentally
gitignored).

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
