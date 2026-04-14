# Feature Hardening Matrix: wasm-m4-worker-runtime

- Branch: feat/wasm-m1-foundation
- Last updated: 2026-04-14 11:29:00Z

## Scope
- [x] Describe intended behavior.
- [x] Describe out-of-scope behavior.
Intended behavior:
- Worker-first browser runtime path for search operations, with main-thread fallback.
- Controlled search APIs with `AbortSignal` and `timeoutMs` across string/JSON/object entrypoints.
- Promise-based worker-oriented search API (`search_request_value_async`).
- Documented fallback strategy matrix (worker support, threads, service-worker constraints).
Out of scope:
- Preemptive mid-query cancellation inside core search loops.
- Full all-feature workspace validation (OpenSSL env blocker).

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
Additional new files in this milestone:
- `searchlite-wasm/searchlite-worker-client.mjs`
- `searchlite-wasm/searchlite-demo-worker.mjs`
- `docs/feature-hardening/wasm-m4-worker-runtime/matrix.md`

## Invariant Matrix
| Area | Scenario | Expected Result | Test Type | Test Reference | Status |
| --- | --- | --- | --- | --- | --- |
| Controlled search | Pre-aborted signal | Returns typed `aborted` error payload | wasm_bindgen | `search_request_value_controlled_aborts_with_preaborted_signal` | done |
| Controlled search | Timeout input validation | Negative timeout returns typed `invalid_timeout` | wasm_bindgen | `search_request_value_controlled_rejects_invalid_timeout` | done |
| Controlled search | Timeout enforcement | Timeout `0` returns typed `timeout` | wasm_bindgen | `search_request_value_controlled_times_out` | done |
| Async worker API | Promise-based search entrypoint | Async method returns successful search payload | wasm_bindgen | `search_request_value_async_roundtrip` | done |
| Worker runtime | Main-thread responsiveness under worker load | Main-thread timer fires while worker search is delayed/in-flight | wasm_bindgen browser | `worker_search_request_keeps_main_thread_responsive` | done |
| Worker runtime | Worker timeout typing | Delayed worker search with small timeout returns typed `timeout` | wasm_bindgen browser | `worker_search_request_timeout_returns_typed_error` | done |
| Worker client | AbortSignal cancellation | In-flight worker-client search aborted via `AbortController` returns typed `aborted` | wasm_bindgen browser | `worker_client_search_request_abort_returns_typed_error` | done |
| Worker client | Invalid timeout validation | Negative/non-finite `timeoutMs` rejected before dispatch with typed `invalid_timeout` | wasm_bindgen browser | `worker_client_search_request_rejects_invalid_timeout` | done |
| Fallback path | Threads unavailable | `init_threads` returns typed `threads_feature_disabled` without `threads` feature | wasm_bindgen browser | `init_threads_without_feature_returns_typed_error` | done |
| Demo runtime | Worker-first path with fallback | IndexedDB+worker uses worker action path; docs/UI define no-worker and memory fallbacks | tests + docs | `searchlite-wasm/index.html`, `docs/wasm.md`, `docs/bindings.md` | done |

## Adversarial Cases
- [x] Null, empty, and whitespace inputs.
- [ ] Dotted names and special characters.
- [x] Cross-scope mismatches.
- [ ] Missing fast field / unsupported type.
Covered:
- Invalid `timeoutMs` values rejected with stable typed error.
- Pre-aborted signals rejected before execution with stable typed error.
- Worker delayed path is covered in browser worker tests without blocking the main thread.
- Worker timeout behavior is covered through worker action timeout tests.
- Worker unavailability / memory-mode constraints handled via documented and implemented fallback.

## Verification Checklist
- [x] `cargo fmt --all`
- [ ] `cargo build --all --all-features`
- [ ] `cargo test --all --all-features`
- [ ] `cargo clippy --all --all-features --all-targets -- -D warnings`
- [x] `cargo bench -p searchlite-core` when perf-sensitive.
Executed:
- `cargo check -p searchlite-wasm --all-targets`
- `cargo check -p searchlite-wasm --all-targets --target wasm32-unknown-unknown`
- `cargo clippy -p searchlite-wasm --all-targets --target wasm32-unknown-unknown -- -D warnings`
- `cargo clippy -p searchlite-wasm --all-targets -- -D warnings`
- `cargo test -p searchlite-wasm` (host target; wasm-bindgen tests are browser-runner tests)
- `cargo bench -p searchlite-core` (completed)
- `wasm-pack test --headless --firefox --geckodriver <local-geckodriver> searchlite-wasm` (45/45 tests passed locally)
- `RUST_TEST_THREADS=1 wasm-pack test --headless --chrome --chromedriver /snap/bin/chromium.chromedriver searchlite-wasm` (45/45 tests passed locally)
- `wasm-pack test --headless --chrome --chromedriver /snap/bin/chromium.chromedriver searchlite-wasm` (logic tests pass, but chromedriver can be SIGKILLed mid-run on this host without single-threading)
- `cargo build --all --all-features` / `cargo test --all --all-features` / `cargo clippy --all --all-features --all-targets -- -D warnings` (blocked: missing `openssl.pc` for `openssl-sys`)

## Review Summary
- Key risks:
- Worker restart on abort/timeout preserves IndexedDB-backed state but resets memory-only mode state.
- Cancellation is cooperative at API boundaries, not preemptive inside core search loops.
- Tests added:
- `search_request_value_controlled_rejects_invalid_timeout`
- `search_request_value_controlled_aborts_with_preaborted_signal`
- `search_request_value_controlled_times_out`
- `search_request_value_async_roundtrip`
- `worker_search_request_keeps_main_thread_responsive`
- `worker_search_request_timeout_returns_typed_error`
- `worker_client_search_request_abort_returns_typed_error`
- `worker_client_search_request_rejects_invalid_timeout`
- `init_threads_without_feature_returns_typed_error`
- Follow-ups:
- Evaluate preemptive cancellation hooks in core search loops if strict mid-query cancellation is required.
