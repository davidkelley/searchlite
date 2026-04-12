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
| `http [options]` | Run the HTTP server. See [http.md](http.md) |

Documents are **buffered** when added -- they won't appear in search results until
you run `commit`. This lets you batch thousands of documents before making them
visible, which is much faster than committing after each one.

### Common flags for every write command

All write commands (`init`, `add`, `update`, `delete`, `commit`, `compact`) accept
the optional `--write-key <KEY>` flag. If the index was created with a write key,
you **must** supply the same key for every subsequent write, or the operation
fails with an authorization error. See [write-key.md](write-key.md) for details.

```bash
# Create a protected index
searchlite init "$INDEX" schema.json --write-key "my-secret"

# Every subsequent write must pass the same key
searchlite add "$INDEX" docs.jsonl --write-key "my-secret"
searchlite commit "$INDEX" --write-key "my-secret"
searchlite compact "$INDEX" --write-key "my-secret"
```

Read commands (`search`, `inspect`) never require a write key.

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
searchlite search "$INDEX" -q "rust language" --limit 5
```

### Updating documents (upsert)

`update` is an alias for `add` -- both perform upserts keyed by `_id`. Use
`update` in scripts when you want the intent of "replace an existing document"
to be explicit:

```bash
# The file format is identical to `add` -- NDJSON keyed by _id
cat > /tmp/patch.jsonl <<'EOF'
{"_id":"1","body":"Rust is a safe systems language","lang":"en","year":2025}
EOF
searchlite update "$INDEX" /tmp/patch.jsonl
searchlite commit "$INDEX"
```

Documents that share an existing `_id` replace the old version on the next
commit. New IDs are inserted. The update is not visible until `commit` runs.

### Deleting documents

`delete` takes a plain text file with one document ID per line:

```bash
cat > /tmp/ids.txt <<'EOF'
2
3
EOF
searchlite delete "$INDEX" /tmp/ids.txt
searchlite commit "$INDEX"
```

Blank lines and trailing whitespace are ignored, and IDs are validated before
anything is queued.

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

## All `search` flags

| Flag | Purpose |
|---|---|
| `-q, --query <STRING>` | Free-text query. Searches all indexed text fields by default. |
| `--fields <A,B,C>` | Comma-separated list restricting which text fields are searched. |
| `--limit <N>` | Max hits to return (default: `10`). Set to `0` to skip hit ranking and only run aggregations. |
| `--execution <wand\|bmw\|bm25>` | Scoring strategy (default: `wand`). See [Query execution modes](#query-execution-modes) below. |
| `--bmw-block-size <N>` | Override the block size used by the `bmw` strategy. Only meaningful with `--execution bmw`. |
| `--return-stored` | Include stored fields in each hit. Off by default so results stay compact. |
| `--return-hits <bool>` | Whether to include the hits array. Set to `false` for pure aggregation queries (default: `true`). |
| `--highlight <FIELD>` | Single-field highlighting. The hit's `snippet` is populated with `<em>...</em>` markup. |
| `--sort <FIELD[:asc\|desc],…>` | Comma-separated sort spec, e.g. `year:desc,title:asc`. Sort fields must be fast. |
| `--cursor <TOKEN>` | Resume from a previous response's `next_cursor`. |
| `--aggs <JSON>` | Inline aggregations JSON. |
| `--aggs-file <PATH>` | Load aggregations JSON from a file (alternative to `--aggs`). |
| `--request <PATH>` | Use a full `SearchRequest` JSON file. Overrides all other flags. |
| `--request-stdin` | Read a full `SearchRequest` JSON from stdin. Overrides all other flags. |

When compiled with the `vectors` feature, the following additional flags are
available for simple vector/hybrid queries (for anything more complex, use
`--request` with a full JSON body):

| Flag | Purpose |
|---|---|
| `--vector-field <NAME>` | Vector field to search. |
| `--vector <JSON>` | Query vector as a JSON array, e.g. `[0.1,-0.2,0.5]`. |
| `--alpha <0.0-1.0>` | Hybrid blend (default `0.5`; `0.0` = pure vector, `1.0` = pure BM25). |
| `--vector-k <N>` | Number of ANN neighbours to retrieve. |
| `--vector-ef-search <N>` | HNSW beam width at query time. |
| `--vector-candidates <N>` | Over-sampling size before re-ranking. |

### Reading a request from stdin

`--request-stdin` is handy when you want to build a request from another tool
(e.g., `jq`, a shell variable, or another program):

```bash
cat request.json | searchlite search "$INDEX" --request-stdin

# Or build one on the fly
jq -n '{query: "rust", limit: 5, return_stored: true}' \
  | searchlite search "$INDEX" --request-stdin
```

When `--request` or `--request-stdin` is used, every other flag is ignored --
the JSON body is authoritative.

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
- When a `--request` file is provided, individual CLI flags (`-q`, `--filter`, etc.) are ignored.
- The CLI uses filesystem storage only; for in-memory indexes, use the Rust API.
- For development, you can run the CLI via cargo: `cargo run -p searchlite-cli -- <command> ...`

---

## A complete worked example

This end-to-end script walks through the full lifecycle -- schema, ingest,
search, maintenance -- using only the CLI.

```bash
#!/usr/bin/env bash
set -euo pipefail

INDEX=/tmp/blog_idx

# --- 1. Schema: title + body for text search, tag + year for filtering ---
cat > /tmp/schema.json <<'EOF'
{
  "doc_id_field": "_id",
  "analyzers": [
    { "name": "en", "tokenizer": "default",
      "filters": [{ "stopwords": "en" }, { "stemmer": "english" }] }
  ],
  "text_fields": [
    { "name": "title", "analyzer": "en", "stored": true, "indexed": true },
    { "name": "body",  "analyzer": "en", "stored": true, "indexed": true }
  ],
  "keyword_fields": [
    { "name": "tag",  "stored": true, "indexed": true, "fast": true }
  ],
  "numeric_fields": [
    { "name": "year", "i64": true,  "fast": true, "stored": true }
  ]
}
EOF

# --- 2. Create the index ---
searchlite init "$INDEX" /tmp/schema.json

# --- 3. Add a handful of documents ---
cat > /tmp/posts.jsonl <<'EOF'
{"_id":"p1","title":"Why Rust is winning",       "body":"Safe systems programming",       "tag":"rust",   "year":2024}
{"_id":"p2","title":"Intro to BM25",             "body":"How search engines rank results", "tag":"search", "year":2023}
{"_id":"p3","title":"SQLite for embedded data",  "body":"A single-file database",          "tag":"data",   "year":2022}
EOF
searchlite add    "$INDEX" /tmp/posts.jsonl
searchlite commit "$INDEX"

# --- 4. A plain full-text query ---
searchlite search "$INDEX" -q "rust" --return-stored --limit 5

# --- 5. Aggregations (faceted navigation) ---
searchlite search "$INDEX" -q "*" --limit 0 \
  --aggs '{"tags": {"type": "terms", "field": "tag", "size": 10}}'

# --- 6. Full JSON request (filters + highlighting) ---
cat > /tmp/req.json <<'EOF'
{
  "query": { "type": "query_string", "query": "rust search" },
  "filter": { "I64Range": { "field": "year", "min": 2023, "max": 2025 } },
  "return_stored": true,
  "highlight_field": "body",
  "limit": 5
}
EOF
searchlite search "$INDEX" --request /tmp/req.json

# --- 7. Maintenance ---
searchlite inspect "$INDEX"      # Inspect manifest and segments
searchlite compact "$INDEX"      # Merge segments when the index is hot
```

Run it once and you have a fully working, searchable index -- with no servers
to run, no cluster to provision, and no build tools beyond `sh` and `curl`.
