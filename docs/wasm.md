# WASM (WebAssembly)

The `searchlite-wasm` crate compiles Searchlite to WebAssembly, bringing full-text
search directly into the browser. Indexes are stored in IndexedDB, so search works
entirely client-side -- no server round-trips, no API keys, no infrastructure.

**When you'd use WASM search:**
- A documentation site that lets users search offline (like Rust docs or MDN)
- A note-taking app where all data lives on the client
- A privacy-focused application where search data never leaves the browser
- Prototyping a search experience without setting up a backend
- PWAs (Progressive Web Apps) that need to work offline

> **Status:** Experimental. The API is functional but may change.

---

## Quick start

### 1. Build the WASM package

```bash
# Install wasm-pack if you haven't already
brew install wasm-pack  # or: cargo install wasm-pack

# Build for browsers and module workers
wasm-pack build searchlite-wasm --target web --release
```

This produces a `pkg/` directory with `searchlite_wasm.js` and the compiled `.wasm` file.

### 2. Use from JavaScript

```javascript
import init, { Searchlite } from './pkg/searchlite_wasm.js';

await init();

// Create an index backed by IndexedDB (persists across page reloads)
const schema = {
  doc_id_field: "_id",
  text_fields: [{ name: "title", analyzer: "default", stored: true, indexed: true }],
  keyword_fields: [{ name: "category", stored: true, indexed: true, fast: true }],
};
const db = await Searchlite.init("my-search-db", JSON.stringify(schema), "indexeddb");

// Add documents and commit
await db.add_documents([
  { _id: "1", title: "Getting started with Rust", category: "tutorial" },
  { _id: "2", title: "Advanced search techniques", category: "guide" },
]);
await db.commit();

// Search
const results = await db.search("rust", 10, true);
console.log(results);
```

Use `"memory"` instead of `"indexeddb"` for ephemeral indexes that don't persist.

---

## Build targets

| Target | Command | Use case |
|---|---|---|
| ESM (browsers + module workers) | `--target web` | Default. Works in `<script type="module">` and module workers. |
| Classic workers | `--target no-modules` | For `importScripts()` environments (classic web workers, service workers). |
| Bundler | `--target bundler` | For webpack, Vite, or other bundlers. |
| Threaded | `--target web -- --features threads` | Multi-threaded search (requires COOP/COEP headers). |

### Threaded builds

Threaded WASM uses `SharedArrayBuffer` for parallel search. This requires your server
to send COOP/COEP headers:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Threaded builds are not available in service workers.

---

## Running the demo

```bash
cd searchlite-wasm
npx http-server -c-1 --cors -p 8080
# Open http://localhost:8080/index.html
```

For threaded builds, add the COOP/COEP headers:

```bash
npx http-server -c-1 --cors -p 8080 \
  -H "Cross-Origin-Opener-Policy: same-origin" \
  -H "Cross-Origin-Embedder-Policy: require-corp"
```

---

## API notes

- `Searchlite.init(name, schema, storage)` reopens existing indexes with the same name.
  Schema mismatches return an error.
- Prefer `add_documents([...])` for bulk ingest over adding one at a time.
- Call `commit()` after adding documents to make them searchable.
- Searches default to `return_stored: false`. Pass `true` as the third argument to
  `search()` to include stored field values.
- Use `search_request_value` / `search_request` for advanced queries (filters,
  aggregations, highlighting).
- See [bindings.md](bindings.md) for a reference of binding behaviors.

---

## Full examples

### Advanced search with filters and facets

`search()` is a convenience wrapper. For filters, aggregations, sorting, or
highlighting, build a full request and pass it to `search_request_value`
(which takes a plain JS object) or `search_request` (which takes a JSON
string). These mirror the [HTTP API](http.md) payload exactly.

```javascript
import init, { Searchlite } from './pkg/searchlite_wasm.js';
await init();

const schema = {
  doc_id_field: "_id",
  text_fields:    [{ name: "title", analyzer: "default", stored: true, indexed: true }],
  keyword_fields: [{ name: "tag",  stored: true, indexed: true, fast: true }],
  numeric_fields: [{ name: "year", i64: true,  fast: true, stored: true }],
};

const db = await Searchlite.init("docs-demo", JSON.stringify(schema), "indexeddb");
await db.add_documents([
  { _id: "1", title: "Rust search",  tag: "rust",   year: 2024 },
  { _id: "2", title: "BM25 basics",  tag: "search", year: 2023 },
  { _id: "3", title: "Edge ranking", tag: "search", year: 2022 },
]);
await db.commit();

const response = await db.search_request_value({
  query:   { type: "query_string", query: "search" },
  filter:  { I64Range: { field: "year", min: 2023, max: 2025 } },
  aggs:    { tags: { type: "terms", field: "tag", size: 5 } },
  highlight_field: "title",
  return_stored: true,
  limit: 10,
});

console.log(response.hits);           // Each hit has doc_id, score, fields, snippet
console.log(response.aggregations);   // { tags: { buckets: [...] } }
```

### Threaded queries

Enable threading before the first search to let the engine use multiple cores.
The build and the page both need extra configuration -- see the bullets below:

```javascript
await Searchlite.init_threads();                    // uses navigator.hardwareConcurrency
// or:
await Searchlite.init_threads(4);

const db = await Searchlite.init("docs-demo", JSON.stringify(schema), "indexeddb");
// subsequent searches will now run across threads
```

Requirements:
- Build with `--features threads` (e.g.
  `wasm-pack build searchlite-wasm --target web --release -- --features threads`).
- Serve the page with COOP/COEP headers so `SharedArrayBuffer` is available.
- Not available in service workers.

### Memory-only mode for tests and previews

If the index is purely ephemeral (demos, tests, iframes that should not
persist data), pass `"memory"` instead of `"indexeddb"`:

```javascript
const db = await Searchlite.init("throwaway", JSON.stringify(schema), "memory");
```

Do not mix storage modes for the same `db_name`. Switching from
`indexeddb` to `memory` silently ignores the previously-persisted state -- use
a fresh `db_name` whenever you change the backend.
