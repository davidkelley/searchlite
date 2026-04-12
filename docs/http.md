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

| Flag | Env Var | Default | Purpose |
|---|---|---|---|
| `--index`, `-I` | `SEARCHLITE_INDEX_MAP` | *(required)* | `NAME:PATH` index mount (repeatable; semicolon-delimited for env). Supports per-index overrides, see below. |
| `--alias` | `SEARCHLITE_INDEX_ALIASES` | -- | `ALIAS:TARGET` indirections that point one name at another mounted index (semicolon-delimited for env) |
| `--bind` | `SEARCHLITE_BIND_ADDR` | `127.0.0.1:8080` | Listen address |
| `--require-existing-index` | `SEARCHLITE_REQUIRE_EXISTING_INDEX` | `false` | Fail at startup if an index's manifest is missing |
| `--max-body-bytes` | `SEARCHLITE_MAX_BODY_BYTES` | `52428800` (50 MiB) | Max request body size in bytes |
| `--max-concurrency` | `SEARCHLITE_MAX_CONCURRENCY` | `64` | Max concurrent in-flight requests |
| `--request-timeout-secs` | `SEARCHLITE_REQUEST_TIMEOUT_SECS` | `30` | Per-request timeout in seconds |
| `--shutdown-grace-secs` | `SEARCHLITE_GRACEFUL_SHUTDOWN_SECS` | `5` | Graceful shutdown window after a SIGTERM/SIGINT |
| `--refresh-on-commit` | `SEARCHLITE_REFRESH_ON_COMMIT` | `false` | Auto-refresh readers after each commit so writes become searchable immediately |
| `--auto-commit-interval-secs` | `SEARCHLITE_AUTO_COMMIT_INTERVAL_SECS` | `0` (disabled) | Default auto-commit interval for all indexes (seconds). Disabled on write-key-protected indexes. |
| `--auto-refresh-interval-secs` | `SEARCHLITE_AUTO_REFRESH_INTERVAL_SECS` | `0` (disabled) | Default auto-refresh interval for all indexes (seconds) |
| `--max-vector-candidates`&nbsp;⚙️ | `SEARCHLITE_MAX_VECTOR_CANDIDATES` | *(vectors feature only)* | Global cap on combined vector candidates across clauses |

All errors return `{"error": {"type": "...", "reason": "..."}}`.

### Per-index overrides

The `--index` flag accepts a comma-delimited tail that overrides global settings
for a single mount. Useful when you want different refresh cadences per index:

```bash
# Two indexes with different auto-commit/auto-refresh cadences:
searchlite http \
  --index "orders:/data/orders,auto_commit=5,auto_refresh=5" \
  --index "catalog:/data/catalog,auto_commit=300"
```

Supported override keys:
- `auto_commit=<seconds>` -- overrides `--auto-commit-interval-secs` for this index
- `auto_refresh=<seconds>` -- overrides `--auto-refresh-interval-secs` for this index

Per-index values take precedence over the global defaults.

### Environment variables

Every CLI flag can be set via environment variable. This is the typical pattern
for containerised deployments:

```bash
export SEARCHLITE_INDEX_MAP="default:/data;orders:/data/orders"
export SEARCHLITE_BIND_ADDR="0.0.0.0:8080"
export SEARCHLITE_REFRESH_ON_COMMIT=true
searchlite http
```

Use `;` (semicolon) to separate multiple entries in `SEARCHLITE_INDEX_MAP` and
`SEARCHLITE_INDEX_ALIASES`.

---

## API endpoints

### Service-level

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/healthz` | Liveness probe. Returns `{ "status": "ok" }` — use it from load balancers and Kubernetes liveness/readiness checks. |
| GET | `/indexes` | List every mounted index and alias, with document counts, last commit times, and per-index auto-commit/auto-refresh settings. |

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
| POST | `/indexes/{name}/update` | Partial update (set/unset fields on a single document) |
| POST | `/indexes/{name}/_bulk_update` | Batch partial updates (NDJSON action/patch pairs) |
| POST | `/indexes/{name}/delete` | Delete documents by ID |

Writes are **buffered** -- documents are not visible to search until you call `/commit`.

### Request body formats at a glance

| Endpoint | Content-Type | Body shape |
|---|---|---|
| `/init` | `application/json` | A `Schema` object (see [schema.md](schema.md)) |
| `/add` | `application/x-ndjson` | One JSON document per line (NDJSON) |
| `/bulk` | `application/json` | `{ "docs": [ {...}, {...} ] }` -- must contain at least one document |
| `/update` | `application/json` | `{ "id": "...", "set": {...}, "unset": [...] }` -- at least one of set/unset required |
| `/_bulk_update` | `application/x-ndjson` | Alternating lines: an action (`{"update": {"_id":"..."}}`) followed by the patch body |
| `/delete` | `application/json` | `{ "ids": ["id1", "id2"] }` -- at least one id required |
| `/search`, `/multi_search`, `/mget` | `application/json` | Fully-typed request objects (see below) |

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

Searchlite supports three mutually exclusive pagination modes. Pick the one
that best matches your UI:

| Mode | Best for | Scaling |
|---|---|---|
| **Offset** (`from` + `size`) | Classic "page 1, 2, 3" navigation with a known page count. | Bounded to `from + size <= 1000`. Use cursor or search_after for deep pagination. |
| **`search_after`** | "Next page" APIs that sort by stable fields (dates, IDs, prices). | Unbounded; requires an explicit `sort` clause. |
| **`cursor`** | Infinite scroll or "load more" where you don't sort. | Unbounded; tokens are opaque and bounded to ~50K results per cursor. |

```bash
# 1. Offset pagination
curl -XPOST .../search -d '{"query": "rust", "from": 10, "size": 5}'

# 2. search_after -- use next_search_after from the previous response
curl -XPOST .../search -d '{
  "query": "rust",
  "sort":  [{"field": "year", "order": "asc"}, {"field": "_id", "order": "asc"}],
  "size":  5,
  "search_after": [2024, "doc-42"]
}'

# 3. Cursor -- pull next_cursor from the previous response
curl -XPOST .../search -d '{"query": "rust", "limit": 5, "cursor": "AAEFMTIzNA=="}'
```

A typical client loop with `search_after`:

```bash
# page 1
resp=$(curl -s -XPOST .../search -d '{"query":"rust","sort":[{"field":"_id","order":"asc"}],"size":5}')
after=$(echo "$resp" | jq -c '.next_search_after')

# page 2
curl -s -XPOST .../search -d "{\"query\":\"rust\",\"sort\":[{\"field\":\"_id\",\"order\":\"asc\"}],\"size\":5,\"search_after\":$after}"
```

### Fetch documents by ID

```bash
curl -XPOST http://localhost:8080/indexes/products/mget \
  -H 'Content-Type: application/json' \
  -d '{"ids": ["product-1", "product-2"], "return_stored": true}'
```

### Partial update

A single-document patch. `set` writes or overwrites fields, `unset` removes
them. You must provide at least one of the two.

```bash
curl -XPOST http://localhost:8080/indexes/products/update \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "product-1",
    "set":   { "in_stock": true, "price_cents": 2499 },
    "unset": ["sale_label"]
  }'
```

### Bulk updates (NDJSON action stream)

For updating many documents in one request, use `_bulk_update`. The body is
NDJSON with two lines per operation: an action descriptor followed by the
patch body.

```bash
cat > /tmp/bulk.ndjson <<'EOF'
{"update":{"_id":"product-1"}}
{"set":{"in_stock":true,"price_cents":2499}}
{"update":{"_id":"product-2"}}
{"set":{"price_cents":3499},"unset":["sale_label"]}
EOF

curl -XPOST http://localhost:8080/indexes/products/_bulk_update \
  -H 'Content-Type: application/x-ndjson' \
  --data-binary @/tmp/bulk.ndjson
curl -XPOST http://localhost:8080/indexes/products/commit
```

### Delete documents

```bash
curl -XPOST http://localhost:8080/indexes/products/delete \
  -H 'Content-Type: application/json' \
  -d '{"ids": ["product-discontinued-1", "product-discontinued-2"]}'
curl -XPOST http://localhost:8080/indexes/products/commit
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

### Service-level: health and discovery

```bash
# Liveness probe (no index required)
curl -s http://localhost:8080/healthz
# -> {"status":"ok"}

# List every mounted index (and aliases)
curl -s http://localhost:8080/indexes
# -> {
#      "indexes": [
#        { "name": "products", "path": "/data/products", "exists": true,
#          "doc_count": 1042, "committed_at": "2025-01-03T12:01:04Z",
#          "auto_commit_secs": 0, "auto_refresh_secs": 0,
#          "refresh_on_commit": false }
#      ],
#      "aliases": []
#    }
```

`/healthz` is cheap and never touches the indexes -- wire it into your load
balancer's health check. `/indexes` is ideal for building admin UIs or
verifying a deployment mounted what you expected.

---

## End-to-end: a minimal HTTP tour

This is a complete, copy-paste-able script that exercises most of the HTTP API.
It assumes a server running on `localhost:8080` with `--index default:/tmp/http_tour`.

```bash
BASE=http://localhost:8080/indexes/default

# 1. Schema: one text field, one keyword field, one numeric
cat > /tmp/schema.json <<'EOF'
{
  "doc_id_field": "_id",
  "text_fields":    [{"name": "body", "analyzer": "default", "stored": true, "indexed": true}],
  "keyword_fields": [{"name": "tag",  "stored": true, "indexed": true, "fast": true}],
  "numeric_fields": [{"name": "year", "i64": true,  "fast": true, "stored": true}]
}
EOF
curl -s -XPOST "$BASE/init" -H 'Content-Type: application/json' --data-binary @/tmp/schema.json

# 2. Ingest via NDJSON
cat > /tmp/docs.ndjson <<'EOF'
{"_id":"a","body":"Rust search","tag":"rust","year":2024}
{"_id":"b","body":"SQLite internals","tag":"data","year":2023}
{"_id":"c","body":"BM25 explained","tag":"search","year":2024}
EOF
curl -s -XPOST "$BASE/add" -H 'Content-Type: application/x-ndjson' --data-binary @/tmp/docs.ndjson
curl -s -XPOST "$BASE/commit"

# 3. Search with a filter and an aggregation
curl -s -XPOST "$BASE/search" -H 'Content-Type: application/json' -d '{
  "query": "search",
  "filter": { "I64Range": { "field": "year", "min": 2024, "max": 2024 } },
  "return_stored": true,
  "aggs": { "tags": { "type": "terms", "field": "tag", "size": 5 } }
}'

# 4. Patch one document
curl -s -XPOST "$BASE/update" -H 'Content-Type: application/json' \
  -d '{"id":"a","set":{"year":2025}}'
curl -s -XPOST "$BASE/commit"

# 5. Clean up stale content
curl -s -XPOST "$BASE/delete" -H 'Content-Type: application/json' \
  -d '{"ids":["b"]}'
curl -s -XPOST "$BASE/commit"
curl -s -XPOST "$BASE/compact"
```

The full API surface is also documented in `openapi.yaml` at the repo root.
