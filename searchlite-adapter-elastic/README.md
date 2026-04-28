# searchlite-adapter-elastic

A drop-in **Elasticsearch HTTP-API adapter** for [searchlite-http](../searchlite-http). Point your existing Elasticsearch clients — Kibana, the official ES SDKs, query builders, curl — at `http://adapter:9200` and they hit a SearchLite index.

The adapter accepts Elasticsearch-shaped requests, translates them to SearchLite's native API, calls a `searchlite-http` upstream, and translates the response back into the Elasticsearch envelope format. No client changes required.

```
   ES client (Kibana, SDK, curl)
            │
            │ ES query DSL (port 9200)
            ▼
   ┌──────────────────────────┐
   │ searchlite-elastic       │  translates JSON → JSON, both directions
   └──────────────────────────┘
            │
            │ SearchLite API (port 8080)
            ▼
   ┌──────────────────────────┐
   │ searchlite-http          │  the real engine
   └──────────────────────────┘
            │
            ▼
       on-disk index
```

## Quick start (Docker Compose)

```yaml
# docker-compose.yml
services:
  searchlite:
    image: ghcr.io/davidkelley/searchlite:latest
    command:
      - http
      - --index
      - demo:/data
      - --bind
      - 0.0.0.0:8080
    volumes:
      - searchlite-data:/data

  elastic:
    image: ghcr.io/davidkelley/searchlite-elastic:latest
    depends_on:
      - searchlite
    command:
      - --bind
      - 0.0.0.0:9200
      - --upstream-url
      - http://searchlite:8080
    ports:
      - "9200:9200"
    restart: unless-stopped

volumes:
  searchlite-data:
```

```bash
docker compose up -d

# Smoke-test the adapter
curl http://localhost:9200/                    # version banner
curl http://localhost:9200/_cluster/health     # always green; single-node
```

The adapter is read-only in v1, so initialise indexes and load data through the upstream `searchlite-http` on port 8080. See the [walkthrough](../docs/adapters/elasticsearch.md#initialize-an-index-and-load-data) for a worked example.

For a complete walkthrough including index initialisation, sample data, and connecting Kibana / Python / JS clients, see [docs/adapters/elasticsearch.md](../docs/adapters/elasticsearch.md).

## Quick start (cargo, no Docker)

```bash
# Terminal 1: upstream searchlite-http
cargo run -p searchlite-cli -- http --index demo:/tmp/demo --bind 127.0.0.1:8080

# Terminal 2: the adapter
cargo run -p searchlite-adapter-elastic -- \
  --bind 127.0.0.1:9200 \
  --upstream-url http://127.0.0.1:8080
```

## Configuration

All flags can also be set via the matching `SEARCHLITE_ELASTIC_*` environment variable.

| Flag | Env Var | Default | Purpose |
|---|---|---|---|
| `--bind` | `SEARCHLITE_ELASTIC_BIND_ADDR` | `127.0.0.1:9200` | Listen address |
| `--upstream-url` | `SEARCHLITE_ELASTIC_UPSTREAM_URL` | `http://127.0.0.1:8080` | Base URL of the upstream `searchlite-http`. Trailing slash and path prefix preserved. http/https only. |
| `--write-key` | `SEARCHLITE_ELASTIC_WRITE_KEY` | -- | Optional write key forwarded to upstream as `x-searchlite-write-key` |
| `--version-banner` | `SEARCHLITE_ELASTIC_VERSION_BANNER` | `8.11.0` | Version string returned in `GET /` and `_nodes` (some clients pin to a major) |
| `--request-timeout-secs` | `SEARCHLITE_ELASTIC_REQUEST_TIMEOUT_SECS` | `30` | Per-request timeout for upstream calls |
| `--max-body-bytes` | `SEARCHLITE_ELASTIC_MAX_BODY_BYTES` | `100 MiB` | Max request body size |
| `--max-concurrency` | `SEARCHLITE_ELASTIC_MAX_CONCURRENCY` | `64` | Max concurrent in-flight requests |
| `--shutdown-grace-secs` | `SEARCHLITE_ELASTIC_GRACEFUL_SHUTDOWN_SECS` | `5` | Grace period after SIGTERM/SIGINT before forced shutdown |

> **Note:** The adapter forwards the configured write key to the upstream, but does not authenticate the inbound ES client. Front it with your own proxy if you need access control on the ES side.

## Supported API surface (v1, read-only)

- `GET /` — version banner
- `GET /_cluster/health`, `GET /_cluster/state`, `GET /_nodes` — single-node stubs
- `HEAD /{index}`, `GET /{index}` — index existence and metadata
- `GET /{index}/_mapping`, `GET /_mapping` — mapping read
- `GET /{index}/_settings` — minimal settings stub
- `GET /{index}/_aliases`, `GET /{index}/_alias` — index-scoped alias view
- `POST /{index}/_search`, `GET /{index}/_search` — full-text search with the ES query DSL
- `POST /{index}/_count` — exact count
- `POST /{index}/_mget`, `POST /_mget` — multi-get by ID
- `POST /_msearch`, `POST /{index}/_msearch` — NDJSON multi-search

For the supported subset of the query DSL and aggregations, see the [compatibility matrix](../docs/adapters/elasticsearch.md#compatibility-matrix).

## What's NOT supported (returns `400 not_supported_in_v1`)

- Writes: `_doc`, `_bulk`, `_update`, `_delete_by_query`
- DDL: `PUT /{index}`, `DELETE /{index}`, `PUT /_mapping`
- `_refresh`, scroll API, runtime mappings
- Cross-index search via `POST /_search` (path must specify exactly one index)
- ES authentication / security plugin / ILM / snapshots / transforms / watcher
- Painless scripting beyond a small `script_score` whitelist
- `geo_*` queries and aggregations, `more_like_this`, parent/child, nested-as-query

The adapter is intended primarily for **read traffic from existing ES tooling**. Writes still go through the SearchLite native HTTP API on port 8080.

## See also

- [docs/adapters/elasticsearch.md](../docs/adapters/elasticsearch.md) — full walkthrough with worked examples, Kibana setup, compatibility matrix, and troubleshooting
- [searchlite-http](../searchlite-http) — the upstream HTTP service
- [docs/http.md](../docs/http.md) — the SearchLite native HTTP API (used internally and for writes)
- [docs/queries.md](../docs/queries.md) — the SearchLite query DSL (the translation target)

## License

MIT — same as the rest of the workspace.
