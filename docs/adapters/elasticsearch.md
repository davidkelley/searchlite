# Elasticsearch Adapter

`searchlite-adapter-elastic` is a standalone HTTP service that speaks the Elasticsearch HTTP API on the front and translates each request to SearchLite's native API on the back. Existing Elasticsearch clients — Kibana, the official ES SDKs, query builders, curl, your own scripts — can point at the adapter on port 9200 and get results from a SearchLite index without code changes.

The adapter is **read-only in v1**: search, mget, msearch, mapping read, and cluster stubs all work; writes and DDL return `400 not_supported_in_v1` and should still go through the SearchLite native HTTP API on port 8080.

> **Note:** Like SearchLite's native HTTP service, the adapter ships with no built-in authentication or rate limiting. Front it with your own proxy or API gateway when exposing it beyond localhost.

---

## When to use it

- You have an existing tool that only speaks the Elasticsearch HTTP API (Kibana, an SDK, a query builder, an internal dashboard).
- You're evaluating SearchLite as a cheaper or simpler alternative to a small Elasticsearch deployment and want to test it with the queries you already have.
- You want a stepping-stone migration from a real Elasticsearch cluster: keep your read clients pointed at the adapter while you migrate ingestion to the native SearchLite API.

If you're starting fresh and writing your own integration, the [native SearchLite HTTP API](../http.md) is simpler, more capable (write + DDL), and has no protocol translation overhead.

## Architecture

```
   ES client (Kibana, SDK, curl)
            │
            │ Elasticsearch query DSL
            │ port 9200
            ▼
   ┌──────────────────────────┐
   │ searchlite-adapter-elastic │  JSON ↔ JSON translation; no SearchLite types
   │  ghcr.io/davidkelley/      │
   │  searchlite-elastic        │
   └────────────┬─────────────┘
                │
                │ SearchLite native API
                │ port 8080
                ▼
   ┌──────────────────────────┐
   │ searchlite-http          │  the real engine
   │  ghcr.io/davidkelley/    │
   │  searchlite              │
   └────────────┬─────────────┘
                │
                ▼
            on-disk index
```

Each adapter request becomes one (or two, in the case of `_count`) upstream HTTP calls. Adapter ↔ upstream traffic is keep-alive and uses connection pooling via `reqwest`.

---

## Quick start (Docker Compose)

This is the fastest way to try the adapter end-to-end. It uses the published images at `ghcr.io/davidkelley/searchlite:latest` and `ghcr.io/davidkelley/searchlite-elastic:latest`.

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
      - --refresh-on-commit
    volumes:
      - searchlite-data:/data
    # Expose 8080 too if you want to write directly via the native API
    ports:
      - "8080:8080"

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

Start the stack:

```bash
docker compose up -d
```

Confirm both services are reachable:

```bash
curl -s http://localhost:8080/healthz
# {"status":"ok"}

curl -s http://localhost:9200/ | jq .version.number
# "8.11.0"

curl -s http://localhost:9200/_cluster/health | jq .status
# "green"
```

### Initialize an index and load data

The adapter's v1 surface is read-only, so initialisation and writes go through the native API on port 8080:

```bash
# 1. Create the schema
curl -X POST http://localhost:8080/indexes/demo/init \
  -H 'Content-Type: application/json' \
  -d '{
    "type": "object",
    "searchlite:docIdField": "_id",
    "properties": {
      "title":       { "type": "string" },
      "description": { "type": "string" },
      "category":    { "type": "string", "searchlite:kind": "keyword" },
      "price":       { "type": "integer" }
    }
  }'

# 2. Add a few documents (NDJSON, one per line)
curl -X POST http://localhost:8080/indexes/demo/add \
  -H 'Content-Type: application/x-ndjson' \
  --data-binary $'{"_id":"1","title":"rust safety guide","description":"ownership and borrowing","category":"books","price":25}\n{"_id":"2","title":"go concurrency patterns","description":"goroutines and channels","category":"books","price":30}\n{"_id":"3","title":"kitchen essentials","description":"knives, pans, cutting boards","category":"kitchen","price":50}\n'

# 3. Commit + refresh so writes are searchable
curl -X POST http://localhost:8080/indexes/demo/commit
curl -X POST http://localhost:8080/indexes/demo/refresh
```

### Query through the adapter

Now everything goes through the ES-compatible port:

```bash
# Match-all
curl -s -X POST http://localhost:9200/demo/_search \
  -H 'Content-Type: application/json' \
  -d '{"query":{"match_all":{}},"size":10}' | jq

# Term filter on category
curl -s -X POST http://localhost:9200/demo/_search \
  -H 'Content-Type: application/json' \
  -d '{"query":{"term":{"category":"books"}},"size":10}' | jq '.hits.hits[]._id'

# Multi-token relevance match
curl -s -X POST http://localhost:9200/demo/_search \
  -H 'Content-Type: application/json' \
  -d '{"query":{"match":{"description":"goroutines"}},"size":3}' | jq '.hits.hits[0]._source'

# Aggregation
curl -s -X POST http://localhost:9200/demo/_search \
  -H 'Content-Type: application/json' \
  -d '{
    "size": 0,
    "aggs": { "by_category": { "terms": { "field": "category", "size": 10 } } }
  }' | jq '.aggregations.by_category.buckets'

# Mapping
curl -s http://localhost:9200/demo/_mapping | jq

# Cluster health (always green; single-node)
curl -s http://localhost:9200/_cluster/health | jq
```

### Tear down

```bash
docker compose down -v
```

---

## Connecting common ES clients

The adapter advertises itself as Elasticsearch 8.11.0 by default. Any client that accepts an arbitrary host should connect.

### Kibana

Add a Kibana service to the same Compose file:

```yaml
  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    depends_on:
      - elastic
    environment:
      ELASTICSEARCH_HOSTS: '["http://elastic:9200"]'
      # Disable security plugin features the adapter doesn't implement
      ELASTICSEARCH_REQUESTTIMEOUT: 60000
      XPACK_SECURITY_ENABLED: "false"
    ports:
      - "5601:5601"
```

Browse to `http://localhost:5601`. Discover and basic visualisations work against the supported query DSL surface; features that depend on writes, ILM, security, or transforms will not work.

### Python (`elasticsearch` client)

```python
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

resp = es.search(
    index="demo",
    query={"match": {"description": "goroutines"}},
    size=3,
)
for hit in resp["hits"]["hits"]:
    print(hit["_id"], hit["_score"], hit["_source"]["title"])
```

If the client tries to negotiate transport-level features (security headers, sniffing) and complains, disable them:

```python
es = Elasticsearch(
    "http://localhost:9200",
    verify_certs=False,
    request_timeout=30,
    # Some client versions probe /_security or /_xpack on connect — those
    # return 400 from the adapter; pass headers={'accept': 'application/json'}
    # if you see warnings.
)
```

### JavaScript / TypeScript (`@elastic/elasticsearch` client)

```js
import { Client } from '@elastic/elasticsearch';

const client = new Client({ node: 'http://localhost:9200' });

const result = await client.search({
  index: 'demo',
  query: { match: { description: 'goroutines' } },
  size: 3,
});

console.log(result.hits.hits.map(h => h._id));
```

### curl / shell

Already shown above.

---

## Configuration reference

All flags can also be set via the matching `SEARCHLITE_ELASTIC_*` environment variable.

| Flag | Env Var | Default | Purpose |
|---|---|---|---|
| `--bind` | `SEARCHLITE_ELASTIC_BIND_ADDR` | `127.0.0.1:9200` | Listen address. **WARNING:** Binding to `0.0.0.0` exposes this unauthenticated service. Use a proxy. |
| `--upstream-url` | `SEARCHLITE_ELASTIC_UPSTREAM_URL` | `http://127.0.0.1:8080` | Base URL of the upstream `searchlite-http`. http/https only; trailing slash and path prefix are preserved. |
| `--write-key` | `SEARCHLITE_ELASTIC_WRITE_KEY` | -- | Optional write key forwarded to upstream as `x-searchlite-write-key` |
| `--version-banner` | `SEARCHLITE_ELASTIC_VERSION_BANNER` | `8.11.0` | Version returned in `GET /` and `_nodes`. Some clients pin to a major. |
| `--request-timeout-secs` | `SEARCHLITE_ELASTIC_REQUEST_TIMEOUT_SECS` | `30` | Per-request timeout for upstream calls |
| `--max-body-bytes` | `SEARCHLITE_ELASTIC_MAX_BODY_BYTES` | `100 MiB` | Max request body size |
| `--max-concurrency` | `SEARCHLITE_ELASTIC_MAX_CONCURRENCY` | `64` | Max concurrent in-flight requests |
| `--shutdown-grace-secs` | `SEARCHLITE_ELASTIC_GRACEFUL_SHUTDOWN_SECS` | `5` | Grace period after SIGTERM/SIGINT before forced shutdown |

All errors return `{"error": {"type": "...", "reason": "...", "root_cause": [...]}, "status": <int>}`.

---

## Compatibility matrix

### Supported endpoints

| Endpoint | Notes |
|---|---|
| `GET /` | Static version banner |
| `GET /_cluster/health` | Always green; single-node |
| `GET /_cluster/state`, `GET /_nodes` | Minimal stubs |
| `GET /_mapping` | Returns mappings for all indexes |
| `HEAD /{index}` | 200/404 |
| `GET /{index}` | mappings + settings + aliases |
| `GET /{index}/_mapping` | Per-index mapping |
| `GET /{index}/_settings` | Minimal settings stub; 404 on unknown |
| `GET /{index}/_aliases`, `GET /{index}/_alias` | Path-scoped; resolves alias names too |
| `POST /{index}/_search`, `GET /{index}/_search` | Full search surface |
| `POST /{index}/_count`, `GET /{index}/_count` | Exact count |
| `POST /{index}/_mget`, `POST /_mget` | Multi-get by ID |
| `POST /_msearch`, `POST /{index}/_msearch` | NDJSON multi-search |

### Rejected endpoints (return `400 not_supported_in_v1`)

- All write endpoints: `PUT/POST /{index}/_doc[/{id}]`, `_bulk`, `_update`, `_delete_by_query`, `_update_by_query`
- All DDL: `PUT /{index}`, `DELETE /{index}`, `PUT /_mapping`
- `POST /{index}/_refresh`
- Cross-index search via `POST /_search` (specify exactly one index in the path instead)
- ES auth / security / ILM / snapshots / transforms / watcher / scroll

### Query DSL

| Clause | Status | Notes |
|---|---|---|
| `match_all`, `match_none` | ✓ / ✗ | `match_none` rejected |
| `match` | ✓ | Single-field; translated to `query_string` |
| `match_phrase` | ✓ | `slop=0` uses analyzer-aware quoted `query_string`; `slop > 0` uses literal-token phrase |
| `match_phrase_prefix` | ✗ | Not supported |
| `multi_match` | ✓ | `best_fields` / `most_fields` / `cross_fields` types; field boost (`title^2`); fuzziness; tie-breaker; minimum-should-match |
| `term` | ✓ | String → keyword equality; numeric → constant-score range with `min == max` |
| `terms` | ✓ | Mixed string/numeric arrays handled per element; field literally named `boost` is queryable |
| `prefix`, `wildcard`, `regexp` | ✓ | Single-field, string only |
| `range` | ✓ | i64/f64 inferred from value type; `gt`/`lt` converted to inclusive ranges via IEEE next-up/next-down |
| `bool` | ✓ | `must`, `should`, `must_not`, `filter`; `minimum_should_match` accepts integer, percentage `"75%"`, and integer-string |
| `query_string`, `simple_query_string` | ✓ | Forwarded; SearchLite's parser supports the common subset |
| `constant_score` | ✓ | Filter context |
| `dis_max` | ✓ | With tie-breaker |
| `function_score`, `script_score` | partial | Limited to SearchLite's whitelist |
| `exists` | ✗ | No SearchLite equivalent |
| `geo_*`, `more_like_this`, `terms_set`, `combined_fields`, `intervals`, `pinned`, `parent/child`, `has_child`, `has_parent`, `nested` (query form) | ✗ | Rejected with `x_content_parse_exception` |

### Aggregations

| Aggregation | Status |
|---|---|
| `terms`, `significant_terms`, `rare_terms` | ✓ |
| `range`, `date_range` | ✓ |
| `histogram`, `date_histogram` | ✓ |
| `composite` (terms + histogram sources) | ✓ |
| `filter`, `nested` | ✓ |
| `stats`, `extended_stats`, `value_count`, `cardinality` | ✓ |
| `percentiles`, `percentile_ranks` | ✓ |
| `top_hits` | ✓ |
| Pipeline: `bucket_sort`, `avg_bucket`, `sum_bucket`, `derivative`, `moving_avg`, `bucket_script` | ✓ |
| `avg`, `sum`, `min`, `max` (single-metric) | ✗ | Use `stats` and read the corresponding field |
| `geo_*`, `geohash_grid`, `adjacency_matrix`, `diversified_sampler`, `auto_date_histogram`, `serial_diff`, `cumulative_sum`, `cumulative_cardinality`, `scripted_metric` | ✗ |

### Sort / pagination / highlight

| Feature | Status |
|---|---|
| `sort` (asc/desc, `_score`, mixed) | ✓ — `_score` defaults to desc, other fields to asc, matching ES |
| `from` + `size` | ✓ |
| `search_after` | ✓ |
| `scroll` | ✗ — recommend `search_after` migration |
| `_source: true/false`, `_source: ["fields"]`, `_source.includes` | ✓ |
| `_source.excludes` | ✗ |
| `highlight` (per-field, fragment_size, number_of_fragments, pre/post tags) | ✓ |
| `track_total_hits` (boolean or integer cap; cap mapped to true/false) | ✓ |

---

## Mapping translation

The adapter exposes SearchLite's JSON Schema-style mapping as an Elasticsearch mapping when you call `GET /{index}/_mapping`:

| SearchLite | Elasticsearch |
|---|---|
| `string` (default kind) | `text` (with `analyzer` and `search_analyzer` carried) |
| `string` + `searchlite:kind: keyword` | `keyword` |
| `integer` | `long` |
| `number` | `double` |
| `boolean` | `boolean` |
| `array` of `object` (nested) | `nested` (with sub-properties recursed) |
| `object` + `searchlite:vector` | `dense_vector` (with `dims` and `similarity` derived from the metric) |

The reverse direction (`PUT /{index}/_mapping`) is **not supported** in v1. Define the schema using the SearchLite native API on port 8080 — see [docs/schema.md](../schema.md).

---

## Behavioural notes

A few places where SearchLite's defaults match (or surprise next to) real Elasticsearch:

- **`_score` defaults to descending** for both bare-string (`"sort": ["_score"]`) and object form (`"sort": [{"_score": {}}]`), matching ES.
- **`match_phrase` is analyzer-aware** when `slop = 0` (the default). The adapter emits a `query_string` with a quoted phrase so SearchLite's per-field analyzer drives tokenization. Punctuation, contractions, and non-ASCII text are handled the same way ES would. With `slop > 0`, the adapter falls back to a literal-token phrase to preserve proximity control — known limitation.
- **No stemming or stopword removal** by default on either engine (matches ES's `standard` analyzer). `match: pattern` will not find a doc with `patterns` unless you configure a stemming analyzer per field.
- **BM25 length normalization matches**: shorter documents with the same query-term frequency outrank longer ones on both engines. This was verified across 30 parity test cases against real Elasticsearch 9.0.0.
- **`_count` returns an exact count** because the adapter always sets `track_total_hits: true` for that endpoint.
- **`track_total_hits` integer cap** (e.g. `10000`) is mapped to `true` since SearchLite has no lower-bound mode. `0` maps to `false`. Negatives are rejected.
- **`bool.minimum_should_match: "75%"`** is resolved adapter-side by counting the `should` clauses and computing `floor(should_count * pct / 100)`. Combinator syntax like `"3<90%"` is rejected.
- **Single-shard pretense**: the adapter always returns `"_shards": { "total": 1, "successful": 1, "skipped": 0, "failed": 0 }` since SearchLite has no sharding concept.

---

## Operations

- **Single node only.** SearchLite has no clustering. The adapter returns single-node stubs for `_cluster/health` and `_nodes`.
- **Healthcheck.** The adapter has no dedicated `/healthz` endpoint (use `GET /` for liveness; it's always 200 once bound). For Kubernetes, an HTTP probe against `/` works.
- **TLS.** Terminate TLS at a reverse proxy in front of the adapter. The adapter only speaks HTTP.
- **Behind a path prefix.** If the upstream is hosted under a path prefix (e.g. `https://internal.example.com/searchlite/`), pass `--upstream-url https://internal.example.com/searchlite` — the adapter normalises the trailing slash and preserves the prefix on every upstream call.
- **Restart safety.** The adapter is stateless. All persistent state lives in the upstream's index directory.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `502 connection_exception` from adapter | Upstream `searchlite-http` not running or wrong URL | `curl http://upstream:8080/healthz` and check `--upstream-url` |
| `400 x_content_parse_exception` with "feature `X` not supported" | Query uses an ES feature outside the adapter's surface | Check the [compatibility matrix](#compatibility-matrix); rewrite the query or use the native SearchLite API |
| `400 not_supported_in_v1` on a write or DDL | Adapter is read-only in v1 | Send writes/DDL to the native HTTP API on port 8080 |
| `404 index_not_found_exception` on `GET /demo/_settings` | Index doesn't exist (or the upstream listing hasn't refreshed) | Confirm with `curl http://upstream:8080/indexes` |
| Adapter accepts query but returns zero hits where ES would match | Tokenization or stemming mismatch — adapter uses analyzer-aware `match_phrase` and SearchLite's per-field analyzers | Verify your schema's `searchlite:analyzer`; for stemming you need a stemming analyzer wired into the index manifest |
| `400 illegal_argument_exception` on `POST /_search` | Cross-index search not supported | Specify exactly one index in the path: `POST /myindex/_search` |
| `_score` ordering looks reversed | Likely you sorted by `_score` ascending; ES defaults to descending | Default is now respected — confirm you're on adapter ≥ the build that landed the `_score` desc fix |

For deeper investigation, the adapter logs every upstream call at `tracing` level info (set `RUST_LOG=info,searchlite_adapter_elastic=debug` for translation details).

---

## See also

- [searchlite-adapter-elastic crate README](../../searchlite-adapter-elastic/README.md)
- [Native HTTP API](../http.md) — what the adapter calls under the hood, and where writes go
- [Query DSL](../queries.md) — the SearchLite-side query language the adapter targets
- [Schema](../schema.md) — how to define the index
- [Aggregations](../aggregations.md) — SearchLite's aggregation reference
