# Feature Hardening Matrix: wasm-m3-storage-reliability

- Branch: feat/wasm-m1-foundation
- Last updated: 2026-04-14 12:20:00Z

## Scope
- [x] Describe intended behavior.
- [x] Describe out-of-scope behavior.
Intended behavior:
- IndexedDB snapshot bootstrap uses a single transaction with `get_all_keys` + `get_all` for browser-compatibility.
- Persistence coalesces writes/deletes into batched transactions and `flush()` waits for both.
- WASM exposes storage/quota inspection (`storage_usage`) and cleanup APIs for stale indexes/orphaned files.
- Quota failures are surfaced as typed `quota_exceeded` errors with actionable recovery guidance.
Out of scope:
- Browser-specific quota limits and storage estimate fidelity across engines.
- Full E2E browser perf benchmarks in this branch (covered by CI/runtime environment).

## Changed Files
<!-- BEGIN_CHANGED_FILES -->
- `.github/workflows/ci.yml`
- `.github/workflows/release-artifacts.yml`
- `docs/bindings.md`
- `docs/wasm.md`
- `searchlite-wasm/Cargo.toml`
- `searchlite-wasm/index.html`
- `searchlite-wasm/src/wasm.rs`
<!-- END_CHANGED_FILES -->

## Invariant Matrix
| Area | Scenario | Expected Result | Test Type | Test Reference | Status |
| --- | --- | --- | --- | --- | --- |
| Storage bootstrap | Large persisted stores load via `get_all_keys` + `get_all` in one readonly transaction | Snapshot restore remains correct across browsers with paired key/value arrays | wasm_bindgen | `js_storage_persists_entries` | done |
| Persistence batching | Multiple writes in one flush are batched | Transaction count is coalesced (`1` tx for 12 queued writes in test) | wasm_bindgen | `js_storage_flush_batches_indexeddb_transactions` | done |
| Delete durability | Deletes are included in flush completion contract | Reopen after flush does not resurrect deleted files | wasm_bindgen | `js_storage_flush_waits_for_deletes` | done |
| Quota handling | Persist path emits typed quota error | `commit()` returns `{ type: "quota_exceeded", reason: ... }` | wasm_bindgen | `commit_surfaces_quota_exceeded_error_type` | done |
| Quota introspection | Browser storage estimate API is exposed safely | Returns supported usage/quota payload or explicit unsupported note | wasm_bindgen | `storage_usage_returns_supported_or_note` | done |
| Stale index cleanup | Cleanup targets only stale registry entries | Stale db dropped; fresh db retained | wasm_bindgen | `cleanup_indexes_drops_only_stale_entries` | done |
| Orphan cleanup | Cleanup preserves live manifest files | Orphan file removed; active searchable docs remain | wasm_bindgen | `cleanup_orphaned_files_removes_only_unknown_paths` | done |

## Adversarial Cases
- [x] Null, empty, and whitespace inputs.
- [ ] Dotted names and special characters.
- [x] Cross-scope mismatches.
- [ ] Missing fast field / unsupported type.
Covered:
- Invalid cleanup request validation (`stale_older_than_ms < 0`) -> typed error.
- Quota failure injection path validates explicit typed failure and guidance.
- Orphan cleanup path validates no live-manifest file removal.

## Verification Checklist
- [x] `cargo fmt --all`
- [ ] `cargo build --all --all-features`
- [ ] `cargo test --all --all-features`
- [ ] `cargo clippy --all --all-features --all-targets -- -D warnings`
- [ ] `cargo bench -p searchlite-core` when perf-sensitive.
Executed:
- `cargo check -p searchlite-wasm --all-targets --target wasm32-unknown-unknown`
- `cargo clippy -p searchlite-wasm --all-targets -- -D warnings`
- `cargo clippy -p searchlite-wasm --all-targets --target wasm32-unknown-unknown -- -D warnings`
- `cargo test -p searchlite-wasm` (host target; wasm-bindgen tests are browser-runner tests)
- `wasm-pack test --headless --firefox --geckodriver /snap/bin/geckodriver searchlite-wasm` (45/45 passed locally)
- `RUST_TEST_THREADS=1 wasm-pack test --headless --chrome --chromedriver /snap/bin/chromium.chromedriver searchlite-wasm` (45/45 passed locally)

## Review Summary
- Key risks:
- Snapshot bootstrap currently materializes full key/value arrays (`get_all_keys` + `get_all`), so startup memory still scales with persisted index footprint.
- Browser storage estimate API support varies by engine; API returns `supported: false` + `note` when unavailable.
- Browser E2E validation still depends on CI runner/browser setup for wasm-bindgen tests.
- Tests added:
- `js_storage_flush_batches_indexeddb_transactions`
- `js_storage_flush_waits_for_deletes`
- `storage_usage_returns_supported_or_note`
- `commit_surfaces_quota_exceeded_error_type`
- `cleanup_indexes_drops_only_stale_entries`
- `cleanup_orphaned_files_removes_only_unknown_paths`
- Follow-ups:
- Run browser wasm-bindgen test suite on CI or a host with chromedriver binaries available.
