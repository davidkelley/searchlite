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
