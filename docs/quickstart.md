# Quickstart

This guide gets you from zero to a working Searchlite index in a few minutes using the CLI. It assumes a junior/mid-level developer comfortable with a terminal and basic JSON.

## Prerequisites

- Rust 1.88.0+ installed (`rustup show` to verify) if you are building from source.
- An x86_64 Linux/macOS machine with local SSD/NVMe recommended.
- Clone this repo and stay in its root directory.

Prefer a prebuilt binary instead of building from source? Install it directly:

```bash
curl -fsSL https://searchlite.dev/install | sh
```

After installing, replace `cargo run -p searchlite-cli -- ...` in the commands below with `searchlite ...`.

```bash
git clone https://github.com/davidkelley/searchlite.git
cd searchlite
```

## 1) Build the CLI

```bash
cargo build -p searchlite-cli --release
```

The CLI binary will be at `target/release/searchlite-cli`. You can also use `cargo run -p searchlite-cli -- ...` for the steps below.

## 2) Create an index

Pick a location for the index (any writable directory). Set an environment variable for convenience:

```bash
INDEX=/tmp/searchlite_idx
```

Create a schema file that defines your fields and analyzers. Save the JSON below as `/tmp/schema.json`:

```json
{
  "doc_id_field": "_id",
  "analyzers": [
    { "name": "english", "tokenizer": "default", "filters": [{ "stopwords": "en" }, { "stemmer": "english" }] }
  ],
  "text_fields": [
    { "name": "title", "analyzer": "english", "stored": true, "indexed": true },
    { "name": "body", "analyzer": "english", "stored": true, "indexed": true }
  ],
  "keyword_fields": [
    { "name": "lang", "stored": true, "indexed": true, "fast": true }
  ],
  "numeric_fields": [
    { "name": "year", "i64": true, "fast": true, "stored": true }
  ],
  "nested_fields": [],
  "vector_fields": []
}
```

Initialize the index with that schema:

```bash
cargo run -p searchlite-cli -- init "$INDEX" /tmp/schema.json
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
cargo run -p searchlite-cli -- add "$INDEX" /tmp/docs.jsonl
```

## 4) Commit the changes

Commit makes buffered documents visible to readers:

```bash
cargo run -p searchlite-cli -- commit "$INDEX"
```

## 5) Run a search

Search by query string plus filters. This example looks for “search” in `title`/`body` and filters by `year`:

```bash
cargo run -p searchlite-cli -- search "$INDEX" \
  --q "search" \
  --filter '{"I64Range":{"field":"year","min":2022,"max":2024}}' \
  --return-stored
```

You should see hits with `_score`, `_id`, and stored fields.

## 6) Try a JSON request

For more control (sorting, aggregations, highlighting), send a full JSON payload. Save this as `/tmp/request.json`:

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
cargo run -p searchlite-cli -- search "$INDEX" --request /tmp/request.json
```

## 7) Inspect and maintain

- Inspect the index manifest and segments:

  ```bash
  cargo run -p searchlite-cli -- inspect "$INDEX"
  ```

- Compact occasionally to merge segments and reclaim space:

  ```bash
  cargo run -p searchlite-cli -- compact "$INDEX"
  ```

## Next Steps

- Explore analyzers, nested filters, aggregations, and vectors in the feature docs.
- Embed Searchlite via the Rust API or FFI if you want to integrate directly into your app.
- Check the Searchlite documentation site for updates and cross-links to deeper guides once they land.
