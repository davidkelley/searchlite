# Plan: searchlite-memory E2E hardening + correctness audit

## Goal
Make `searchlite-memory` provably ready-to-go on day 1: a deep correctness/completeness
audit of the existing implementation, plus a small set of **critical, isolated end-to-end
tests** that exercise the real paths the unit tests stub out — the real local ONNX embedder
(semantic recall) and the real spawned CLI/MCP stdio server — with one small real-model run
to confirm it actually works.

## Context
The package is implemented and consensus-clean at unit/integration level (40 memory tests +
318 binding tests). The existing `test/e2e/mcp.test.ts` uses the SDK **in-memory** transport
and a **stub** embedder — so two production paths are currently unproven by automated tests:
(1) the real `@huggingface/transformers` embedder producing genuine semantic recall, and
(2) the real `searchlite-memory serve` process speaking JSON-RPC over actual stdio.

## Assumptions
- "Real spend" here = the local ONNX model download + CPU inference (no paid API; API
  providers are unimplemented and out of scope). Keep any real run tiny (2–3 memories, 1 recall).
- E2E tests that download the model or spawn processes must NOT run in the default
  `npm test` / CI lane — they are gated behind `RUN_E2E=1` so the fast offline suite is unchanged.
- The local model `Xenova/all-MiniLM-L6-v2` (384-dim) is reachable from the HuggingFace Hub
  on the machine running the gated E2E.
- Node 20+; the searchlite-js native binding is built (file: dev-link).

## Non-goals
- External embedding providers (OpenAI/Voyage/Cohere) — not implemented; not tested here.
- Multi-machine / cross-OS reproducibility testing.
- Performance/load benchmarking.

## Stages

### Stage 1 — Isolated real-model E2E (gated) + a small real run
**Files:** `searchlite-memory/test/e2e/real-model.test.ts` (new), `searchlite-memory/vitest.config.ts` (ensure gated files are excluded by default), `searchlite-memory/package.json` (add `test:e2e` script).
- A `describe.runIf(process.env.RUN_E2E === "1")` suite using the **real** `local` embedder
  (no stub) against a real `EmbeddedIndex` in a tmp dir:
  - remember 3 memories with distinct meanings; assert a **paraphrase query** (no shared
    keywords) recalls the semantically-matching memory above the others (proves vectors work).
  - get → full content; forget → gone; rebuild from ledger → recall still works (vectors come
    from the committed int8 sidecar, not re-embedding); assert `vectors.jsonl` + `memory.jsonl`
    exist and the int8 sidecar dim == 384.
- `package.json`: `"test:e2e": "RUN_E2E=1 vitest run test/e2e"`.
- Run it once locally (small) to confirm green; record model/runtime in the report.

**Acceptance criteria:** with `RUN_E2E=1` the suite passes using the real model; paraphrase
recall returns the semantically-correct memory; rebuilt index (sidecar vectors, model not
invoked for existing records) still recalls; default `npm test` does not run or download anything.

### Stage 2 — Real stdio CLI/MCP E2E (gated, spawned process)
**Files:** `searchlite-memory/test/e2e/stdio-cli.test.ts` (new).
- Gated suite that spawns the **built** `node dist/cli.js serve` as a child process and drives
  it with the SDK `Client` over a real `StdioClientTransport` (FTS-only via
  `SEARCHLITE_MEMORY_EMBEDDER=none`, offline + fast):
  - `initialize` handshake; `tools/list` shows the 4 tools; `remember`→`recall`→`get`→`forget`
    round-trip with `structuredContent`.
  - assert the server writes nothing to stdout except JSON-RPC (stdout is the protocol channel).
- A subprocess smoke for `init` and `doctor` (exit codes, scaffolded files) against the built CLI.

**Acceptance criteria:** the spawned real server completes the tool round-trip over stdio;
`init`/`doctor` subprocesses behave; suite is gated behind `RUN_E2E=1`.

### Stage 3 — Correctness/completeness fixes from the audit
**Files:** TBD by the panel's deep-dive (likely `searchlite-memory/src/**`).
- Apply any critical/high correctness or completeness gaps the panel surfaces during the
  comprehensive analysis (e.g. error-handling, edge cases, missing wiring). Each fix gets a
  focused commit + its own review.

**Acceptance criteria:** all panel critical/high findings resolved; full suite + gated E2E green.

## Audit findings folded in (Stage 3)
Critical/high from the Codex + opencode deep-dive, applied before the E2E tests so the tests
validate fixed behavior:
- **Packaging readiness:** add `prepublishOnly: npm run build` (dist is gitignored built output; publish must ship it); pin `@modelcontextprotocol/sdk` to `^1.29.0` (subpath layout used).
- **Model-drift safety:** rebuild only reuses a committed sidecar vector when its `model`+`dim` match the current embedder; otherwise the record is indexed FTS-only (no silently-wrong vectors) and `doctor`/`rebuild --reembed` repairs.
- **External-change freshness:** a long-running server detects ledger/sidecar changes from a `git pull`/branch switch via file mtime (not only the gitignored `.ledger-hash`).
- **Graceful shutdown:** `serve` installs SIGINT/SIGTERM handlers that close the store (release lock) and exit.
- **Embedder errors surfaced:** `local.ts` logs the specific failure to stderr before falling back to FTS-only (no silent swallow of bad revision/dtype/network).
- **`#ensureIndexFresh`/staleness** compares all gate fields (incl. schema/vector fingerprint).
- **`rebuild --reembed`** reloads state under the lock before persisting the sidecar.

## Decisions made
- **Gate via `RUN_E2E=1`, not an always-on test:** the model download (~tens of MB) and process
  spawns are too slow/networked for the default lane; gating keeps `npm test` fast and offline
  while making the real paths runnable on demand and in a dedicated CI lane.
- **Use vitest `describe.runIf` (not `skipIf`) keyed on `RUN_E2E`** so the gated suites are
  inert unless explicitly enabled.
- **stdio E2E runs FTS-only** (`EMBEDDER=none`) to stay offline/fast; the real-model E2E is the
  one place the embedder is exercised — keeping the single networked test isolated.
- **Real run is intentionally tiny** (2–3 memories) to honor the "small run to constrain cost".

## Reviewer pushback (rejected)
- **`prepublishOnly` should run `test:e2e`** (Codex, final round): rejected running the gated E2E in `prepublishOnly` — it downloads a model + spawns processes and is network-dependent, so it would make `npm publish` flaky/offline-fragile. Set `prepublishOnly` to `npm run build && npm test` (fast offline units) instead; `test:e2e` stays a dedicated CI/dev lane.
- **`--help`/usage to stdout** (opencode B8): kept on stdout — standard CLI convention; `--help` short-circuits and never starts the server, so it cannot corrupt an active JSON-RPC channel.
- **Unknown flags error out** (opencode B9): kept lenient (unknown flags ignored, default to `serve`) — low value, conventional for a thin CLI.
- **`searchlite-js` file: dep is not registry-ready** (Codex): accepted as a documented release step (repin to the published vectors-enabled version before publishing) — cannot be resolved without publishing the binding; flagged in README + this plan.

## Reviewer availability
- **Gemini unavailable** this run — the local Gemini CLI fails with `IneligibleTierError` (free-tier client no longer supported) and an untrusted-directory refusal. **Grok excluded** per the user (usage limit). Panel = Codex + opencode.
