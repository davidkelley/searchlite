# Quickstart

This guide gets you from zero to a working Searchlite index in a few minutes using the CLI. It assumes a junior/mid-level developer comfortable with a terminal and basic JSON.

## Prerequisites

- Linux or macOS (x86_64 or aarch64) with `curl` and `tar` available.
- Local SSD/NVMe recommended for best ingest/search performance.
- No build tools, package managers, or Node.js required—the installer downloads prebuilt binaries.

## 1) Install the CLI

```bash
curl -fsSL https://searchlite.dev/install | sh
```

The script installs a `searchlite` binary to `/usr/local/bin` or `~/.local/bin`. If your shell does not already include that directory, add it to `PATH` before continuing.

## 2) Create an index

Pick a location for the index (any writable directory). Set an environment variable for convenience:

```bash
INDEX=/tmp/searchlite_idx
```

Create a schema file that defines your fields and analyzers. Save the JSON below as `/tmp/schema.json`:

```json
{
  "type": "object",
  "searchlite:analyzers": [
    { "name": "english", "tokenizer": "default", "filters": [{ "stopwords": "en" }, { "stemmer": "english" }] }
  ],
  "properties": {
    "title": { "type": "string", "searchlite:analyzer": "english" },
    "body": { "type": "string", "searchlite:analyzer": "english" },
    "lang": { "type": "string", "searchlite:kind": "keyword" },
    "year": { "type": "integer", "searchlite:stored": true }
  }
}
```

`searchlite:stored` lets you return fields in results, and `searchlite:fast` enables efficient filters and aggregations. Keyword and numeric fields have `searchlite:fast` on by default, so `lang` and `year` are filterable without any extra configuration.

Initialize the index with that schema:

```bash
searchlite init "$INDEX" /tmp/schema.json
```

## 3) Add documents

Create a small JSONL file (`/tmp/docs.jsonl`) with your documents. Each line is one JSON object with a unique `_id`:

```bash
cat > /tmp/docs.jsonl <<'EOF'
{"_id":"doc-1","title":"Rust search engine","body":"Searchlite is a lightweight search engine written in Rust.","lang":"en","year":2024}
{"_id":"doc-2","title":"SQLite vibes","body":"Single-node search with a WAL and atomic manifests.","lang":"en","year":2023}
{"_id":"doc-3","title":"Edge ready","body":"Run full-text search at the edge or in appliances.","lang":"en","year":2022}
EOF
```

Ingest the documents (this buffers them):

```bash
searchlite add "$INDEX" /tmp/docs.jsonl
```

## 4) Commit the changes

Commit makes buffered documents visible to readers:

```bash
searchlite commit "$INDEX"
```

## 5) Run a search

Search by query string. This example looks for "search" across all indexed text fields:

```bash
searchlite search "$INDEX" -q "search" --return-stored
```

You should see hits with `_score`, `_id`, and stored fields.

## 6) Try a JSON request

For filters, sorting, aggregations, or highlighting, send a full JSON payload via `--request`. Save this as `/tmp/request.json`:

```json
{
  "query": { "type": "query_string", "query": "search", "fields": ["title", "body"] },
  "filter": { "KeywordEq": { "field": "lang", "value": "en" } },
  "limit": 5,
  "sort": [{ "field": "year", "order": "desc" }],
  "return_stored": true,
  "highlight_field": "body"
}
```

Run it:

```bash
searchlite search "$INDEX" --request /tmp/request.json
```

## 7) Serve over HTTP

**Security warning:** The HTTP server has no auth, no authorization, and no rate limiting. Keep it bound to localhost or behind a proxy/firewall that enforces access control.

Run the bundled HTTP server straight from the installed binary (no Rust toolchain needed):

```bash
searchlite http --index "default:$INDEX" --bind 127.0.0.1:8080
# Add --refresh-on-commit if you want searches to see new data immediately.
```

All endpoints are prefixed with `/indexes/{name}/` where `{name}` matches the mount name (here, `default`).

Send the same search over HTTP:

```bash
curl -s http://127.0.0.1:8080/indexes/default/search \
  -H "content-type: application/json" \
  -d @/tmp/request.json
```

You can also ingest over HTTP instead of the CLI:

```bash
curl -X POST \
  -H "content-type: application/x-ndjson" \
  --data-binary @/tmp/docs.jsonl \
  http://127.0.0.1:8080/indexes/default/add
curl -X POST http://127.0.0.1:8080/indexes/default/commit
# If you did not start the server with --refresh-on-commit, also call:
curl -X POST http://127.0.0.1:8080/indexes/default/refresh
```

Keep the server bound to localhost unless you front it with a proxy or firewall.

## 8) Inspect and maintain

- Inspect the index manifest and segments:

  ```bash
  searchlite inspect "$INDEX"
  ```

- Compact occasionally to merge segments and reclaim space:

  ```bash
  searchlite compact "$INDEX"
  ```

## Next Steps

- Read [Searchlite in a Nutshell](intro.md) for a high-level overview of features, limits, and operational basics.
- See the [Schema guide](schema.md) for field types, analyzers, and nested object configuration.
- Explore [Queries](queries.md), [Filters](filters.md), and [Aggregations](aggregations.md) for the full search DSL.
- See [HTTP Service](http.md) for the complete REST API reference.
- See [Binding Lifecycle](bindings.md) for FFI and WASM-specific details.
