# Browser/WASM Production Task Board

Updated: 2026-04-14  
Owner: Searchlite maintainers  
Scope: `searchlite-wasm` browser surface and release gates

## Goal

Ship a browser surface that is safe for consumer use by closing the five readiness gaps:

1. Lifecycle and migration support
2. WASM API parity with core/HTTP operations
3. Non-blocking browser execution model
4. Storage scale and quota safety
5. Browser-specific CI and release verification

## Global "Feature Complete" Contract

Each task is only complete when all are true:

- API behavior documented in `docs/wasm.md` and `docs/bindings.md`
- Automated tests added or updated
- CI gate exists and is required
- Failure mode is explicit and typed
- Regression coverage exists for discovered edge cases

## Milestones

| Milestone | Target | Exit gate |
| --- | --- | --- |
| M1 | CI skeleton + lifecycle foundations | Browser CI job required; lifecycle APIs merged |
| M2 | API parity | Parity matrix green for supported operations |
| M3 | Storage reliability at scale | Large-index and quota tests green |
| M4 | UX/runtime hardening | Non-blocking worker path + cancellation verified |

## Task Board

| ID | Milestone | Task | Primary file targets | Validation | Status |
| --- | --- | --- | --- | --- | --- |
| WASM-001 | M1 | Add browser CI workflow job (wasm build + browser test runner) | `.github/workflows/ci.yml`, `searchlite-wasm/Cargo.toml` | CI job runs on PR and blocks merge on failure | `Done` |
| WASM-002 | M1 | Add release smoke test for packaged wasm artifact | `.github/workflows/release-artifacts.yml` | Release workflow validates wasm package exports and can initialize/add/commit in memory mode | `Done` |
| WASM-003 | M1 | Define typed WASM error model and error code mapping | `searchlite-wasm/src/wasm.rs`, `docs/bindings.md` | JS-visible errors include stable `type` and clear `reason` fields | `Done` |
| WASM-004 | M1 | Add index lifecycle APIs: `list_indexes`, `drop_index`, `clear_index` | `searchlite-wasm/src/wasm.rs`, `searchlite-wasm/Cargo.toml`, `docs/wasm.md` | wasm-bindgen tests validate create/list/drop/clear correctness | `Done` |
| WASM-005 | M1 | Add schema version metadata and migration planner surface | `searchlite-wasm/src/wasm.rs`, `docs/wasm.md`, `docs/bindings.md` | Reopen path reports migrate-needed state instead of opaque mismatch | `Done` |
| WASM-006 | M1 | Add migration rollback safety for partial/failed upgrades | `searchlite-wasm/src/wasm.rs` | Failure-injection test proves no unusable half-migrated index remains | `Done` |
| WASM-007 | M2 | Add delete APIs (`delete_document`, `delete_documents`) to WASM | `searchlite-wasm/src/wasm.rs`, `docs/wasm.md` | Add/commit/delete/commit/search roundtrip tests pass | `Done` |
| WASM-008 | M2 | Add update/patch API parity for stored fields | `searchlite-wasm/src/wasm.rs`, `docs/wasm.md` | Patch semantics match core for set/unset and validation errors | `Done` |
| WASM-009 | M2 | Add `mget` API to WASM bindings | `searchlite-wasm/src/wasm.rs`, `docs/wasm.md` | Found/missing response shape matches core expectations | `Done` |
| WASM-010 | M2 | Add `multi_search` API to WASM bindings | `searchlite-wasm/src/wasm.rs`, `docs/wasm.md` | Batched requests return ordered result list with per-request output | `Done` |
| WASM-011 | M2 | Add maintenance APIs: `compact`, `inspect`, `stats` | `searchlite-wasm/src/wasm.rs`, `docs/wasm.md`, `docs/bindings.md` | Maintenance ops produce stable payloads and pass regression tests | `Done` |
| WASM-012 | M2 | Add parity contract tests (core vs wasm normalized responses) | `searchlite-wasm/src/wasm.rs` (or dedicated test module), `integration/` if expanded | Shared fixture set compares behavior for search/mget/multi-search/update/delete | `Done` |
| WASM-013 | M3 | Replace full-snapshot load with incremental/chunked load path | `searchlite-wasm/src/wasm.rs` | Large-index startup memory profile is bounded and documented | `Done` |
| WASM-014 | M3 | Batch IndexedDB persistence writes and reduce per-file transaction overhead | `searchlite-wasm/src/wasm.rs` | Ingest benchmark shows fewer IDB transactions and lower commit latency | `Done` |
| WASM-015 | M3 | Add storage usage/quota introspection API | `searchlite-wasm/src/wasm.rs`, `docs/wasm.md` | Browser reports usage/remaining capacity when supported | `Done` |
| WASM-016 | M3 | Add typed quota exceeded handling and recovery guidance | `searchlite-wasm/src/wasm.rs`, `docs/bindings.md`, `docs/wasm.md` | Quota exhaustion test asserts explicit quota error type and recovery path | `Done` |
| WASM-017 | M3 | Add cleanup policies for stale indexes and orphaned files | `searchlite-wasm/src/wasm.rs`, `docs/wasm.md` | Cleanup command removes only targeted data and preserves active indexes | `Done` |
| WASM-018 | M4 | Add worker-first async execution API | `searchlite-wasm/src/wasm.rs`, `searchlite-wasm/index.html`, new worker example files | Long-running search does not block main thread in browser worker tests | `Done` |
| WASM-019 | M4 | Add cancellation (`AbortSignal`) and timeout semantics | `searchlite-wasm/src/wasm.rs`, `docs/wasm.md`, `docs/bindings.md` | Cancellation tests confirm operation stops and returns typed abort/timeout errors | `Done` |
| WASM-020 | M4 | Add fallback strategy matrix for threads/no-threads/service-worker constraints | `docs/wasm.md`, `docs/bindings.md`, `searchlite-wasm/index.html` | Docs and example app behave correctly with and without `threads` feature | `Done` |
| WASM-021 | M4 | Final browser readiness checklist and sign-off report | `docs/roadmaps/WASM-PRODUCTION-TASK-BOARD.md` | All tasks closed and sign-off checklist marked complete | `In Progress` |

## Acceptance Checklist by Gap

| Gap | Required tasks | Complete when |
| --- | --- | --- |
| Lifecycle + migration | WASM-004, WASM-005, WASM-006 | Existing users can upgrade without manual db-name rotation |
| API parity | WASM-007, WASM-008, WASM-009, WASM-010, WASM-011, WASM-012 | Browser can perform same critical lifecycle/query ops as core/HTTP |
| Non-blocking runtime | WASM-018, WASM-019, WASM-020 | UI remains responsive under realistic workloads |
| Storage scale + quota | WASM-013, WASM-014, WASM-015, WASM-016, WASM-017 | Large data sets and quota limits are handled predictably |
| CI + release gates | WASM-001, WASM-002 | Browser regressions and broken artifacts are caught pre-release |

## Test Matrix (must exist before closing M4)

| Test ID | Scenario | Surface | Pass condition |
| --- | --- | --- | --- |
| T-WASM-01 | Reopen existing index with same schema | wasm | Documents remain searchable across reload |
| T-WASM-02 | Reopen with schema upgrade path | wasm | Migration succeeds and preserves searchable docs |
| T-WASM-03 | Forced migration failure | wasm | Rollback leaves prior version usable |
| T-WASM-04 | Delete + patch + mget lifecycle | wasm | Responses match expected shape and values |
| T-WASM-05 | Multi-search contract | wasm | Result ordering and counts match request order |
| T-WASM-06 | Compact + inspect + stats | wasm | Maintenance APIs return consistent metadata |
| T-WASM-07 | Large ingest + commit | wasm | No unbounded memory growth on startup/commit |
| T-WASM-08 | Quota exceeded write path | wasm | Typed quota error surfaced; no silent corruption |
| T-WASM-09 | Main-thread responsiveness under load | wasm_bindgen browser | Main-thread timer remains responsive during delayed worker search (`worker_search_request_keeps_main_thread_responsive`) |
| T-WASM-10 | Cancel in-flight search | wasm_bindgen browser | Worker-client search aborted by `AbortController` returns typed `aborted` (`worker_client_search_request_abort_returns_typed_error`) |
| T-WASM-11 | Threads enabled path | browser E2E | Threaded init/search succeeds under COOP/COEP |
| T-WASM-12 | Threads unavailable fallback | wasm_bindgen browser + docs | `init_threads` returns `threads_feature_disabled`; fallback behavior documented |

## Definition of Done (M4 Sign-off)

- [ ] All `WASM-00x` tasks are `Done`
- [ ] All `T-WASM-xx` tests are automated and green in CI
- [x] `docs/wasm.md` and `docs/bindings.md` fully match actual behavior
- [x] Release workflow verifies packaged wasm artifact
- [ ] No remaining "experimental caveat" without an explicit mitigation or documented limit

## M4 Sign-off Status (WASM-021)

- M4 implementation tasks (`WASM-018`..`WASM-020`) are complete.
- Browser worker-path tests were added for responsiveness, timeout typing, and AbortSignal cancellation.
- Browser wasm-bindgen suite is green locally in Firefox (`wasm-pack test --headless --firefox ... searchlite-wasm`: 45/45).
- Request-value entrypoints now tolerate both direct JS object payloads and JSON-shaped map payloads used in wasm-bindgen Rust tests.
- Chrome browser tests are green locally when explicitly using the installed Chromium driver and single-threaded Rust harness:
  - `RUST_TEST_THREADS=1 wasm-pack test --headless --chrome --chromedriver /snap/bin/chromium.chromedriver searchlite-wasm` (45/45)
- Release smoke validation now checks the actual wasm-pack bundler export surface (named `Searchlite` export, no assumed default init export) and confirms init/add/commit in Node.
- IndexedDB batch writes/cursor scans were hardened for Chrome transaction lifecycle semantics (no per-request await in write batches; snapshot/path scans use `get_all_keys`/`get_all`).
- Local validation blockers remain:
  - Workspace-wide `cargo build/test/clippy --all --all-features` fails on this host due missing `openssl.pc` for `openssl-sys`.
- `WASM-021` can be closed after CI confirms browser tests and all-feature gates in a fully provisioned environment.
