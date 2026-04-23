# Feature Hardening Matrix: wasm-m2-parity-core-ops

- Branch: feat/wasm-m1-foundation
- Last updated: 2026-04-14 06:55:22Z

## Scope
- [x] Intended behavior: implement M2 API parity core ops in WASM (`delete`, `update/patch`, `mget`, `multi_search`, `compact/inspect/stats`) with typed JS errors and docs alignment.
- [x] Out-of-scope: M3/M4 storage/runtime tasks (quota handling, worker-first execution, cancellation, fallback matrix).

## Changed Files
<!-- BEGIN_CHANGED_FILES -->
- `.github/workflows/ci.yml`
- `.github/workflows/release-artifacts.yml`
- `docs/bindings.md`
- `docs/wasm.md`
- `searchlite-wasm/index.html`
- `searchlite-wasm/src/wasm.rs`
<!-- END_CHANGED_FILES -->

## Invariant Matrix
| Area | Scenario | Expected Result | Test Type | Test Reference | Status |
| --- | --- | --- | --- | --- | --- |
| Delete API | Delete single and batch ids | Deleted docs disappear after commit; unaffected docs remain | wasm-bindgen integration | `delete_document_roundtrip`, `delete_documents_roundtrip` | done |
| Update API | set/unset patch semantics | Updates apply after commit and follow core patch rules | wasm-bindgen integration | `update_document_set_and_unset_roundtrip` | done |
| Update validation | Missing patch / invalid id / unknown field / missing doc | Typed JS errors are stable and descriptive | wasm-bindgen integration | `update_document_rejects_missing_patch`, `update_document_rejects_invalid_id`, `update_document_reports_not_found`, `update_document_rejects_unknown_field` | done |
| MGET | Order + found/missing shape + return_stored behavior | Response order preserved, per-id `found` and optional `_source` correct | wasm-bindgen integration | `mget_returns_found_missing_and_preserves_order`, `mget_respects_return_stored_false`, `mget_rejects_invalid_ids` | done |
| Multi-search | Ordered batched search results | `results[]` order matches input `searches[]` | wasm-bindgen integration | `multi_search_returns_ordered_results`, `multi_search_rejects_invalid_request` | done |
| Maintenance | compact/inspect/stats payloads | Segment compaction works; inspect redacts sensitive metadata; stats stable | wasm-bindgen integration | `compact_stats_and_inspect_roundtrip` | done |
| Migration hash stability | Schema hash fingerprint determinism | Hash remains deterministic for identical schemas, changes when schema changes | wasm-bindgen regression | `schema_hash_is_deterministic` | done |

## Adversarial Cases
- [x] Null/invalid payload shapes for delete/update/mget/multi-search return typed errors.
- [x] Empty/whitespace ids rejected via `invalid_id`.
- [x] Unknown update paths rejected and surfaced as typed `update_failed`.
- [ ] Missing fast field / unsupported type edge cases (covered partially by core patch validation; extend with dedicated adversarial fixtures as needed).

## Verification Checklist
- [x] `cargo fmt --all`
- [ ] `cargo build --all --all-features` (blocked in this environment by missing OpenSSL dev metadata)
- [ ] `cargo test --all --all-features` (blocked in this environment by missing OpenSSL dev metadata)
- [ ] `cargo clippy --all --all-features --all-targets -- -D warnings` (blocked in this environment by missing OpenSSL dev metadata)
- [x] `cargo clippy -p searchlite-wasm --all-targets -- -D warnings`
- [x] `cargo check -p searchlite-wasm --all-targets --target wasm32-unknown-unknown`
- [x] `cargo test -p searchlite-wasm`
- [ ] Browser wasm-pack tests (local runner limits still apply; validate in CI)
- [ ] `cargo bench -p searchlite-core` when perf-sensitive (not perf-sensitive changes)

## Review Summary
- Key risks: local browser runner constraints still block full wasm-bindgen browser execution; CI is source of truth for browser-level runtime.
- Tests added: API parity tests for delete/update/mget/multi-search/maintenance + schema hash determinism + core-vs-wasm parity fixture (`search`/`mget`/`multi_search`/`update`/`delete`).
- Follow-ups: start M3 storage scale/quota tasks after CI validates browser-runner pass.
