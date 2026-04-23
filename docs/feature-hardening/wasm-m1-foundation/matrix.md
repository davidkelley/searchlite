# Feature Hardening Matrix: wasm-m1-foundation

- Branch: feat/wasm-m1-foundation
- Last updated: 2026-04-13 20:57:16Z

## Scope
- [x] Intended behavior: M1 WASM production foundations (typed errors, lifecycle APIs, migration planning/execution with rollback, CI/release gates).
- [x] Out-of-scope: M2+ parity APIs (`delete`, `patch`, `mget`, `multi_search`, maintenance ops) and storage/runtime hardening tasks.

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
| Lifecycle | `list_indexes`/`clear_index`/`drop_index` on IndexedDB-backed DBs | Registry and persisted files stay consistent after create/clear/drop/recreate | wasm-bindgen integration | `list_indexes_includes_initialized_db`, `clear_index_resets_contents`, `drop_index_removes_registry_entry` | done |
| Migration planning | `plan_migration` for missing/compatible/schema-changed DBs | Returns stable status and schema hashes (`missing`/`compatible`/`rebuild_required`) | wasm-bindgen integration | `plan_migration_reports_compatibility_and_rebuild` | done |
| Migration execution | `migrate_index` executes create/rebuild flow | Returns `created`/`compatible`/`rebuilt` and leaves DB usable | wasm-bindgen integration | `migrate_index_creates_missing_index`, `migrate_index_rebuilds_on_schema_change` | done |
| Rollback safety | Injected failure after clear during rebuild | Prior snapshot and registry are restored; old schema remains usable | wasm-bindgen integration | `migrate_index_rolls_back_on_rebuild_failure` | done |
| Release artifact | Packaged wasm bundle smoke import | Release workflow asserts expected files and JS exports | CI workflow step | `.github/workflows/release-artifacts.yml` `Smoke test wasm bundle` | done |
| Error model | JS-visible failures are typed payloads | Errors include stable `type` and human-readable `reason` | integration/docs | `WasmErrorPayload`, docs in `docs/wasm.md` + `docs/bindings.md` | done |

## Adversarial Cases
- [x] Null, empty, and whitespace inputs (covered by existing request validation paths and typed errors).
- [x] Dotted names and special characters (schema mismatch and invalid request errors remain typed).
- [x] Cross-scope mismatches (migration failure-injection proves rollback to prior schema/storage state).
- [ ] Missing fast field / unsupported type (tracked for M2 parity/contract matrix).

## Verification Checklist
- [x] `cargo fmt --all`
- [ ] `cargo build --all --all-features` (blocked in this environment by missing OpenSSL dev metadata)
- [ ] `cargo test --all --all-features` (blocked in this environment by missing OpenSSL dev metadata)
- [ ] `cargo clippy --all --all-features --all-targets -- -D warnings` (blocked in this environment by missing OpenSSL dev metadata)
- [x] `cargo check -p searchlite-wasm`
- [x] `cargo check -p searchlite-wasm --all-targets --target wasm32-unknown-unknown`
- [x] `cargo test -p searchlite-wasm`
- [ ] `wasm-pack test --headless --chrome searchlite-wasm` (blocked: chromedriver unavailable for this host target)
- [ ] `wasm-pack test --headless --firefox searchlite-wasm` (environment/runtime failures; existing nightly `std::time` unsupported panic path)
- [ ] `cargo bench -p searchlite-core` when perf-sensitive (not perf-sensitive changes)

## Review Summary
- Key risks: Browser-runner coverage remains environment-sensitive locally; rely on CI for definitive browser execution results.
- Tests added: migration execution (`create`, `rebuild`, rollback failure-injection) and existing lifecycle tests retained.
- Follow-ups: Start M2 tasks (`WASM-007+`) after CI confirms M1 browser/release gates on supported runners.
