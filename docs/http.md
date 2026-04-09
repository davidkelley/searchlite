# HTTP Service

The HTTP server wraps Searchlite's full API in a REST interface. Use it when you want
to expose search to non-Rust clients (JavaScript frontends, Python services, mobile apps),
run Searchlite as a standalone microservice, or test interactively with curl.

The HTTP API accepts the same JSON query format used by the Rust API and CLI, so you can
develop locally with curl and deploy the same queries from any language.

> **Note:** This HTTP service provides no authentication, authorization, or rate limiting.
> Do not expose it directly to untrusted networks. Front it with your own proxy or API
> gateway that enforces access control.

---

## Running the server

```bash
# Installed CLI
searchlite http --index default:/tmp/myindex --bind 0.0.0.0:8080

# Via cargo (development)
cargo run -p searchlite-cli -- http --index default:/tmp/myindex --bind 0.0.0.0:8080

# Via Docker
docker run --rm -p 8080:8080 -v "$PWD:/data" \
  ghcr.io/davidkelley/searchlite:latest \
  http --index default:/data --bind 0.0.0.0:8080

# Environment variables (useful in containers; semicolon-delimited)
SEARCHLITE_INDEX_MAP="default:/data" \
SEARCHLITE_BIND_ADDR=0.0.0.0:8080 \
searchlite http --refresh-on-commit
```

You can mount multiple indexes with repeated `--index` flags (e.g., `--index products:/data/products --index articles:/data/articles`).

## Configuration

| Flag / Env Var | Default | Purpose |
|---|---|---|
| `--index` / `SEARCHLITE_INDEX_MAP` | -- | NAME:PATH index mounts (repeatable; semicolon-delimited for env). Per-index overrides: `--index "items:/data,auto_commit=30,auto_refresh=10,refresh_on_commit=true"` |
| `--alias` / `SEARCHLITE_INDEX_ALIASES` | -- | ALIAS:TARGET indirections (semicolon-delimited for env) |
| `--bind` / `SEARCHLITE_BIND_ADDR` | `127.0.0.1:8080` | Listen address |
| `--require-existing-index` | false | Fail at startup if manifest is missing |
| `--auto-commit-interval-secs` / `SEARCHLITE_AUTO_COMMIT_INTERVAL_SECS` | `0` (disabled) | Global auto-commit interval. Disabled on write-key-protected indexes |
| `--auto-refresh-interval-secs` / `SEARCHLITE_AUTO_REFRESH_INTERVAL_SECS` | `0` (disabled) | Global auto-refresh interval |
| `--max-body-bytes` | -- | Max request body size |
| `--max-concurrency` | -- | Max concurrent requests |
| `--request-timeout-secs` | -- | Per-request timeout |
| `--shutdown-grace-secs` | -- | Graceful shutdown window |
| `--refresh-on-commit` | false | Auto-refresh readers after each commit |

All errors return `{"error": {"type": "...", "reason": "..."}}`.

---

## API endpoints

### Index lifecycle

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/indexes/{name}/init` | Create a new index with a schema |
| POST | `/indexes/{name}/add` | Stream NDJSON documents into the writer |
| POST | `/indexes/{name}/bulk` | Bulk ingest a JSON array of documents |
| POST | `/indexes/{name}/commit` | Flush buffered writes to a new segment |
| POST | `/indexes/{name}/refresh` | Reload readers to see latest commits |
| POST | `/indexes/{name}/compact` | Merge all segments into one |
| GET | `/indexes/{name}/inspect` | View manifest and segment metadata |
| GET | `/indexes/{name}/stats` | Document and segment counts |

### Search and retrieval

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/indexes/{name}/search` | Full-text search with filters, aggregations, highlighting |
| POST | `/indexes/{name}/multi_search` | Batch multiple searches in one request |
| POST | `/indexes/{name}/mget` | Fetch documents by ID |

### Document updates

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/indexes/{name}/update` | Partial update (set/unset fields) |
| POST | `/indexes/{name}/_bulk_update` | Batch partial updates (NDJSON) |
| POST | `/indexes/{name}/delete` | Delete documents by ID |

Writes are **buffered** -- documents are not visible to search until you call `/commit`.

---

## Examples

### Create an index and add documents

```bash
# Create index with schema
curl -XPOST http://localhost:8080/indexes/products/init \
  -H 'Content-Type: application/json' \
  --data-binary @schema.json

# Add documents
curl -XPOST http://localhost:8080/indexes/products/add \
  -H 'Content-Type: application/x-ndjson' \
  --data-binary @products.ndjson

# Make them searchable
curl -XPOST http://localhost:8080/indexes/products/commit
```

### Search with filters, highlights, and aggregations

```bash
curl -XPOST http://localhost:8080/indexes/products/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "wireless headphones",
    "filter": { "KeywordEq": { "field": "category", "value": "electronics" } },
    "limit": 10,
    "return_stored": true,
    "highlight_field": "description",
    "aggs": {
      "brands": { "type": "terms", "field": "brand", "size": 10 },
      "price_stats": { "type": "stats", "field": "price_cents" }
    }
  }'
```

### Pagination

Three mutually exclusive pagination modes:

```bash
# Offset pagination (from + size, max 1000)
curl -XPOST .../search -d '{"query": "rust", "from": 10, "size": 5}'

# search_after (use next_search_after from previous response)
curl -XPOST .../search -d '{"query": "rust", "sort": [{"field": "year", "order": "asc"}], "size": 5, "search_after": [2024, "doc-42", 0]}'

# Cursor (use next_cursor from previous response)
curl -XPOST .../search -d '{"query": "rust", "limit": 5, "cursor": "..."}'
```

### Fetch documents by ID

```bash
curl -XPOST http://localhost:8080/indexes/products/mget \
  -H 'Content-Type: application/json' \
  -d '{"ids": ["product-1", "product-2"], "return_stored": true}'
```

### Partial update

```bash
curl -XPOST http://localhost:8080/indexes/products/update \
  -H 'Content-Type: application/json' \
  -d '{"id": "product-1", "set": {"in_stock": true, "price_cents": 2499}, "unset": ["sale_label"]}'
```

### Multi-search

Batch multiple queries in one request. `parallel: true` runs them concurrently (default: sequential); `max_concurrency` caps parallelism (default 4, max 16).

```bash
curl -XPOST http://localhost:8080/indexes/products/multi_search \
  -H 'Content-Type: application/json' \
  -d '{
    "searches": [
      {"query": "wireless headphones", "limit": 5},
      {"query": "bluetooth speakers", "limit": 3}
    ],
    "parallel": true,
    "max_concurrency": 4
  }'
```

The response wraps results in `{"results": [...]}` with one `SearchResult` per input query.

### Maintenance

```bash
curl -XPOST .../indexes/products/refresh   # reload readers
curl -XPOST .../indexes/products/compact    # merge segments
curl -XGET  .../indexes/products/inspect    # view manifest
curl -XGET  .../indexes/products/stats      # doc/segment counts
```

The full API surface is also documented in `openapi.yaml` at the repo root.
