# searchlite-wasm

Browser-native full-text search for Searchlite. Runs entirely in the client —
indexes are stored in IndexedDB, search happens in a Web Worker, and no server
round-trips are required.

- **Schema-driven**: text, keyword, numeric, and nested fields with the same
  schema format as `searchlite-core`.
- **Persistent**: IndexedDB-backed storage survives page reloads.
- **Non-blocking**: a worker-first runtime keeps the main thread responsive.
- **Cancellable**: `AbortSignal` and `timeoutMs` on every controlled search.
- **Typed errors**: every failure surfaces a `{ type, reason }` payload with a
  stable error code.

> The JavaScript/TypeScript surface is functional and hardened, but the API may
> still change before 1.0.

---

## Install & build

This crate is built with [`wasm-pack`][wasm-pack]. From the repository root:

```bash
wasm-pack build searchlite-wasm --target web --release
```

The command writes a `searchlite-wasm/pkg/` directory containing:

| File | Purpose |
| --- | --- |
| `searchlite_wasm.js` | ESM entrypoint with a default `init()` export and named `Searchlite` class |
| `searchlite_wasm_bg.wasm` | The compiled WebAssembly module |
| `searchlite_wasm.d.ts` | TypeScript type definitions |

Copy `pkg/` into your static assets (or reference it directly in dev) and
import from there. See [docs/wasm.md](../docs/wasm.md#build-targets) for other
build targets (`--target bundler`, `--target no-modules`, threaded builds).

[wasm-pack]: https://rustwasm.github.io/wasm-pack/

---

## Quickstart (vanilla ESM)

Drop this `index.html` next to the generated `pkg/` directory and serve it
with any static HTTP server:

```html
<!doctype html>
<meta charset="utf-8" />
<title>searchlite demo</title>
<script type="module">
  import init, { Searchlite } from "./pkg/searchlite_wasm.js";

  await init();

  // Minimal schema: one text field called "body".
  const schema = {
    doc_id_field: "_id",
    text_fields: [
      { name: "body", analyzer: "default", stored: true, indexed: true },
    ],
  };

  // "indexeddb" persists across reloads; "memory" is ephemeral.
  const db = await Searchlite.init(
    "quickstart",
    JSON.stringify(schema),
    "indexeddb",
  );

  db.add_documents([
    { _id: "1", body: "Rust brings systems programming to the browser." },
    { _id: "2", body: "Searchlite is a full-text search engine in WebAssembly." },
    { _id: "3", body: "IndexedDB stores binary blobs for offline search." },
  ]);
  await db.commit();

  const result = db.search("rust", 10, true);
  document.body.innerText = JSON.stringify(result, null, 2);
</script>
```

Serve it:

```bash
npx http-server -c-1 --cors -p 8080
# open http://localhost:8080/
```

Reload the page — the documents are still there, because IndexedDB persisted
them.

---

## Storage modes

| Mode | Default | Persistence | Typical use |
| --- | --- | --- | --- |
| `"indexeddb"` | yes | Survives reloads and browser restarts | Real apps, offline docs, PWAs |
| `"memory"` | no | Gone when the tab closes | Demos, tests, iframes, previews |

Do not reuse a `db_name` across modes. If you need to switch, pick a new name
or `Searchlite.drop_index(old_name)` first.

---

## Worker-first execution (recommended)

Running search on the main thread blocks the UI. For non-trivial workloads use
the worker client that ships with the crate:

```html
<!doctype html>
<meta charset="utf-8" />
<title>searchlite worker demo</title>
<script type="module">
  import { SearchliteWorkerClient } from "./searchlite-worker-client.mjs";

  const client = new SearchliteWorkerClient();
  const schema = {
    doc_id_field: "_id",
    text_fields: [
      { name: "body", analyzer: "default", stored: true, indexed: true },
    ],
  };

  await client.initIndex("worker-demo", JSON.stringify(schema), "indexeddb");
  await client.addDocuments([
    { _id: "1", body: "Workers keep the UI responsive." },
    { _id: "2", body: "AbortController cancels in-flight searches." },
  ]);

  // Abort after 100ms, otherwise fail after 2s.
  const controller = new AbortController();
  setTimeout(() => controller.abort(), 100);

  try {
    const result = await client.searchRequest(
      { query: "workers", limit: 10, return_stored: true },
      { signal: controller.signal, timeoutMs: 2_000 },
    );
    console.log(result);
  } catch (err) {
    console.error(err.type, err.reason); // e.g. "aborted" / "timeout"
  } finally {
    await client.dispose();
  }
</script>
```

The worker client ships as `searchlite-wasm/searchlite-worker-client.mjs` and
uses `searchlite-wasm/searchlite-demo-worker.mjs` under the hood. On
`indexeddb` storage, worker restarts on abort/timeout preserve state
transparently. On `memory` storage, restarts lose state — fall back to the
main-thread controlled APIs instead.

---

## Running the bundled demo

The crate ships with an interactive demo at [`index.html`](index.html):

```bash
wasm-pack build searchlite-wasm --target web --release
cd searchlite-wasm
npx http-server -c-1 --cors -p 8080
# open http://localhost:8080/index.html
```

The demo exercises every public API: schema upload, ingest, search, filters,
aggregations, maintenance, worker vs main-thread execution, and cancellation.

---

## Further reading

- [**docs/wasm.md**](../docs/wasm.md) — full API surface, build targets,
  threaded builds, worker-first runtime, fallback matrix, advanced examples.
- [**docs/bindings.md**](../docs/bindings.md) — binding semantics (commit
  lifecycle, return-shape conventions, schema-mismatch handling).
- [**docs/wasm-errors.md**](../docs/wasm-errors.md) — full reference for every
  typed error code, when it's emitted, and the recommended recovery.
- [**docs/roadmaps/WASM-PRODUCTION-TASK-BOARD.md**](../docs/roadmaps/WASM-PRODUCTION-TASK-BOARD.md) — milestone status and exit gates.

---

## License

MIT, same as the rest of the Searchlite workspace.
