# Feature Hardening Matrix: wasm-m2-delete-apis

- Branch: feat/wasm-m1-foundation
- Last updated: 2026-04-14 06:29:51Z

## Scope
- [x] Intended behavior: expose WASM delete APIs (`delete_document`, `delete_documents`) with typed validation errors and commit-gated durability semantics.
- [x] Out-of-scope: M2 patch/mget/multi-search/maintenance parity tasks and M3/M4 storage/runtime features.

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
| Delete lifecycle | Add/commit/delete/commit/search (single id) | Deleted document disappears while untouched docs remain searchable | wasm-bindgen integration | `delete_document_roundtrip` | done |
| Delete lifecycle | Add/commit/delete/commit/search (batch ids) | All requested docs are deleted in one commit; retained docs remain | wasm-bindgen integration | `delete_documents_roundtrip` | done |
| Validation | Non-string values in `delete_documents` payload | Typed `invalid_doc_id_batch` error surfaced to JS | wasm-bindgen integration | `delete_documents_rejects_non_string_ids` | done |
| Migration fingerprint | Schema hash stability across toolchain/runtime changes | Deterministic schema hash used for metadata/planning | code-level invariant | `schema_hash` uses deterministic FNV-1a | done |
| Regression | Existing lifecycle/migration APIs remain unchanged | M1 APIs compile and preserve typed error contract | compile + docs | `cargo check` + docs updates | done |

## Adversarial Cases
- [x] Null/invalid delete payloads (non-string array members return typed errors).
- [x] Cross-scope mismatches (schema mismatch guidance updated to include migration path).
- [ ] Empty string doc IDs (currently forwarded to core writer semantics; follow-up for stricter validation if desired).
- [ ] Missing fast field / unsupported type (tracked by M2 patch/mget parity tasks).

## Verification Checklist
- [x] `cargo fmt --all`
- [ ] `cargo build --all --all-features` (blocked in this environment by missing OpenSSL dev metadata)
- [ ] `cargo test --all --all-features` (blocked in this environment by missing OpenSSL dev metadata)
- [ ] `cargo clippy --all --all-features --all-targets -- -D warnings` (blocked in this environment by missing OpenSSL dev metadata)
- [x] `cargo clippy -p searchlite-wasm --all-targets -- -D warnings`
- [x] `cargo check -p searchlite-wasm --all-targets --target wasm32-unknown-unknown`
- [x] `cargo test -p searchlite-wasm`
- [ ] Browser wasm-pack tests (environment runner limits still apply locally)
- [ ] `cargo bench -p searchlite-core` when perf-sensitive (not perf-sensitive changes)

## Review Summary
- Key risks: browser-runtime validation remains CI-dependent in this local environment.
- Tests added: delete single, delete batch, invalid delete payload regression tests.
- Follow-ups: proceed to WASM-008 (patch/update parity), then WASM-009 (`mget`) and WASM-010 (`multi_search`).
