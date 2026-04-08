# CLI Reference

The Searchlite CLI lets you create, populate, and query indexes from the command line.
It's the fastest way to prototype a search schema, bulk-load data, and test queries
before integrating the Rust API or HTTP service into your application.

**When you'd use the CLI:**
- Exploring Searchlite for the first time
- Bulk-loading data from JSONL exports (database dumps, scraped content, log files)
- Scripting index management in CI/CD pipelines or cron jobs
- Quick ad-hoc searches during development

---

## Install

Prebuilt binaries are published on every GitHub release:

```bash
curl -fsSL https://searchlite.dev/install | sh
```

Environment variables for the installer:
- `SEARCHLITE_VERSION` -- pin a specific release tag (e.g., `v0.4.0`)
- `SEARCHLITE_INSTALL_DIR` -- override the install directory
- `SEARCHLITE_BIN_NAME` -- change the binary name (default: `searchlite`)

Supported platforms: `x86_64`/`aarch64` Linux and macOS, Windows via Git Bash/WSL.

---

## Commands

| Command | Purpose |
|---|---|
| `init <index> <schema>` | Create a new index from a JSON schema file |
| `add <index> <docs.jsonl>` | Add or upsert documents (NDJSON format) |
| `update <index> <docs.jsonl>` | Alias for `add` (emphasizes upsert semantics) |
| `delete <index> <ids.txt>` | Queue deletions by ID (one per line) |
| `commit <index>` | Flush buffered writes into a new segment |
| `search <index> [options]` | Query the index and return JSON results |
| `inspect <index>` | Print manifest and segment metadata |
| `compact <index>` | Merge all segments to reduce fragmentation |

Documents are **buffered** when added -- they won't appear in search results until
you run `commit`. This lets you batch thousands of documents before making them
visible, which is much faster than committing after each one.

---

## Walkthrough: build and search an index

```bash
# Pick a location for the index
INDEX=/tmp/my_search_index

# 1. Create the index with a schema
searchlite init "$INDEX" schema.json

# 2. Add documents (NDJSON: one JSON object per line)
cat > /tmp/docs.jsonl <<'EOF'
{"_id":"1","body":"Rust is a systems programming language","lang":"en","year":2024}
{"_id":"2","body":"SQLite is an embedded database","lang":"en","year":2023}
{"_id":"3","body":"Full-text search with BM25 scoring","lang":"en","year":2024}
EOF
searchlite add "$INDEX" /tmp/docs.jsonl

# 3. Make documents searchable
searchlite commit "$INDEX"

# 4. Search!
searchlite search "$INDEX" --q "rust language" --limit 5
```

### Structured queries with filters

For complex searches, pass a JSON request file:

```bash
cat > /tmp/request.json <<'EOF'
{
  "query": {
    "type": "query_string",
    "query": "rust language",
    "fields": ["body", "title"]
  },
  "filter": {
    "And": [
      { "KeywordEq": { "field": "lang", "value": "en" } },
      { "I64Range": { "field": "year", "min": 2020, "max": 2025 } }
    ]
  },
  "limit": 5,
  "return_stored": true,
  "highlight_field": "body"
}
EOF
searchlite search "$INDEX" --request /tmp/request.json
```

### Fuzzy (typo-tolerant) search

Users misspell words. Fuzzy matching handles that automatically:

```bash
cat > /tmp/request.json <<'EOF'
{
  "query": { "type": "query_string", "query": "body:rusk" },
  "fuzzy": { "max_edits": 1, "prefix_length": 1, "max_expansions": 20, "min_length": 3 },
  "limit": 5,
  "return_stored": true
}
EOF
searchlite search "$INDEX" --request /tmp/request.json
```

This matches "rust" even though the user typed "rusk".

### Aggregations

Run analytics queries to build faceted navigation or dashboards:

```bash
searchlite search "$INDEX" \
  --q "rust" \
  --limit 0 \
  --aggs '{"languages": {"type": "terms", "field": "lang", "size": 10}}'
```

Setting `--limit 0` skips hit ranking and returns only aggregation results -- useful
for pure analytics.

### Sorting

```bash
searchlite search "$INDEX" --q "rust" --sort "year:desc" --limit 10
```

Sort targets must be fast keyword or numeric fields. The default order is ascending
(descending for `_score`). Multi-valued fields use min for ascending, max for descending.

### Inspect and compact

```bash
searchlite inspect "$INDEX"   # View segments, doc counts, schema
searchlite compact "$INDEX"   # Merge all segments into one
```

Compaction reduces fragmentation and improves search performance after many commits.

---

## Pagination with cursors

Search responses include `next_cursor` when more results are available:

```bash
# First page
searchlite search "$INDEX" --q "rust" --limit 5

# Next page (use the cursor value from the previous response)
searchlite search "$INDEX" --q "rust" --limit 5 --cursor "$NEXT_CURSOR"
```

Cursors are opaque tokens that encode the current position. They're bounded to ~50K
results to prevent unbounded memory use.

---

## Query execution modes

Searchlite supports three query execution strategies with different performance
characteristics:

| Mode | Flag | Behavior |
|---|---|---|
| `wand` (default) | `--execution wand` | WAND pruning -- skips low-scoring documents early. Fast and exact. |
| `bmw` | `--execution bmw` | Block-max WAND -- even more aggressive pruning using block-level max scores. |
| `bm25` | `--execution bm25` | Full evaluation -- scores every matching document. Slowest but useful for debugging. |

For most workloads, the default `wand` mode is the best choice. Use `bmw` for very
large indexes where you need maximum throughput.

---

## Tips

- Use `--request-stdin` to pipe a JSON request from another command.
- When a `--request` file is provided, individual CLI flags (`--q`, `--filter`, etc.) are ignored.
- The CLI uses filesystem storage only; for in-memory indexes, use the Rust API.
- For development, you can run the CLI via cargo: `cargo run -p searchlite-cli -- <command> ...`
