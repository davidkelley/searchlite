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
  type: "object",
  properties: {
    title: { type: "string" },
    category: { type: "string", "searchlite:kind": "keyword" },
  },
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

## Browser test commands

```bash
# Firefox (local)
wasm-pack test --headless --firefox --geckodriver /path/to/geckodriver searchlite-wasm

# Chrome (CI/default environments)
wasm-pack test --headless --chrome searchlite-wasm

# Chrome (local Snap Chromium, avoids chromedriver SIGKILL flakes)
RUST_TEST_THREADS=1 wasm-pack test --headless --chrome \
  --chromedriver /snap/bin/chromium.chromedriver searchlite-wasm
```

---

## API notes

- `Searchlite.init(name, schema, storage)` reopens existing indexes with the same name.
  Schema mismatches return an error.
- `Searchlite.list_indexes()` lists IndexedDB-backed indexes previously initialised by
  the WASM binding.
- `Searchlite.clear_index(name)` removes all persisted files for an index while keeping
  the IndexedDB database container.
- `Searchlite.drop_index(name)` deletes the persisted IndexedDB database for that index.
- `Searchlite.storage_usage()` returns browser storage usage/quota estimates when supported:
  `{ supported, usage_bytes, quota_bytes, remaining_bytes, persisted, note? }`.
- `Searchlite.cleanup_indexes(stale_older_than_ms, dry_run?)` removes stale IndexedDB-backed
  indexes based on registry age and returns `{ scanned, matched, dropped, kept, dry_run }`.
- `Searchlite.plan_migration(name, schema)` returns a compatibility plan with
  `status: "missing" | "compatible" | "rebuild_required"` and schema hashes.
- `Searchlite.migrate_index(name, schema)` executes the migration plan and returns
  `status: "created" | "compatible" | "rebuilt"`. It rebuilds incompatible schemas
  with rollback to the previous snapshot if rebuild fails.
- Prefer `add_documents([...])` for bulk ingest over adding one at a time.
- `delete_document(id)` and `delete_documents(ids)` queue deletes by `_id`/doc id.
  Deletions are applied on `commit()`.
- `update_document({ id, set?, unset? })` queues a partial update using patch semantics.
  `set` is a field-value map and `unset` is a list of field paths to remove; apply with `commit()`.
- `mget({ ids, return_stored? })` fetches documents by id in request order and returns
  `found` plus optional `_source` per id.
- `multi_search({ searches, parallel?, max_concurrency? })` executes multiple search
  requests and returns ordered per-request results.
- Controlled search APIs:
  - `search_controlled(query, limit, returnStored?, abortSignal?, timeoutMs?)`
  - `search_request_controlled(json, abortSignal?, timeoutMs?)`
  - `search_request_value_controlled(value, abortSignal?, timeoutMs?)`
  These return typed `aborted` / `timeout` errors when cancellation or timeout checks trigger.
- `search_request_value_async(value, abortSignal?, timeoutMs?)` is an async worker-oriented
  entrypoint for Promise-based execution.
- Package target note: `wasm-pack build --target web` exports a default init function, while
  the default bundler target auto-initializes and exposes only named exports (for example `Searchlite`).
- `compact()` merges segments and returns `{ compacted }`.
- `inspect()` returns `{ manifest }` with write-key metadata redacted.
- `stats()` returns document, deletion, and segment counts plus index metadata.
- `cleanup_orphaned_files(dry_run?)` removes IndexedDB file blobs that are not referenced by
  the active manifest (`MANIFEST`, `wal.log`, metadata, and live segment files are preserved).
- Call `commit()` after adding documents to make them searchable.
- Searches default to `return_stored: false`. Pass `true` as the third argument to
  `search()` to include stored field values.
- Use `search_request_value` / `search_request` for advanced queries (filters,
  aggregations, highlighting).
- See [bindings.md](bindings.md) for a reference of binding behaviors.

All WASM errors are returned as structured payloads:

```json
{
  "type": "quota_exceeded",
  "reason": "indexeddb quota exceeded while committing index data; run compact(), remove stale indexes with Searchlite.cleanup_indexes(...), or clear/drop unused indexes before retrying. detail: ..."
}
```

---

## Worker-first runtime

The demo app now defaults to worker-first execution for search to keep the main UI thread responsive:

- Worker entrypoint: `searchlite-wasm/searchlite-demo-worker.mjs`
- Worker client wrapper: `searchlite-wasm/searchlite-worker-client.mjs`
- Demo UI integration: `searchlite-wasm/index.html`

The worker client restarts the worker after timeout/abort to stop in-flight work. For
IndexedDB mode this is transparent (state is reopened from persistence). For memory mode,
worker restarts lose state, so the demo falls back to main-thread execution.
The worker client validates `timeoutMs`/`delayMs` as non-negative finite numbers and surfaces
typed `invalid_timeout`/`invalid_argument` errors for invalid values.

### Runtime fallback matrix

| Runtime | Worker mode | Threads | Recommended path |
| --- | --- | --- | --- |
| Browser + module workers + IndexedDB | yes | optional | Worker client + `search_request_value_async` / controlled APIs |
| Browser + no module worker support | no | optional | Main-thread `search_request_controlled` |
| Browser + memory storage | limited | optional | Main-thread controlled APIs (worker restart would lose memory state) |
| Service worker / classic worker only | no (module worker client not available) | no | Main-thread controlled APIs or custom classic-worker glue |

### Cancellation and timeout example

```javascript
const controller = new AbortController();
const timeoutMs = 250;

setTimeout(() => controller.abort(), 100);

try {
  const result = db.search_request_value_controlled(
    { query: "rust", limit: 20, return_stored: true },
    controller.signal,
    timeoutMs
  );
  console.log(result);
} catch (err) {
  // err.type is "aborted" or "timeout"
  console.error(err.type, err.reason);
}
```

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

Enable threading before the first search so the engine can use multiple cores.
`init_threads` is an **instance method** on `Searchlite`, so you need to have
initialised the index first. The build and the page both need extra
configuration -- see the bullets below:

```javascript
const db = await Searchlite.init("docs-demo", JSON.stringify(schema), "indexeddb");

await db.init_threads();       // uses navigator.hardwareConcurrency
// or:
await db.init_threads(4);

// subsequent searches on this `db` run across threads
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
