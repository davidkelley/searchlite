# Feature Hardening Matrix: issue-92-auto-commit-refresh-indexes

- Branch: feat/issue-92-auto-commit-refresh-indexes
- Last updated: 2026-03-03 09:30:45Z

## Scope
- [x] Implement issue #92 behavior:
  - Per-index auto-commit interval timers (default off at `0`).
  - Per-index auto-refresh interval timers (default off at `0`) with commit-change guard to avoid redundant refreshes.
  - Enriched `GET /indexes` response including operational metadata and runtime config flags.
- [x] Out of scope for this change set:
  - Compaction policies or automatic segment compaction.
  - New authentication/authorization controls for HTTP endpoints.
  - Major scheduler framework changes outside `searchlite-http`.

## Changed Files
<!-- BEGIN_CHANGED_FILES -->
- `docs/feature-hardening/issue-92-auto-commit-refresh-indexes/matrix.md`
- `openapi.yaml`
- `searchlite-http/src/lib.rs`
<!-- END_CHANGED_FILES -->

## Invariant Matrix
| Area | Scenario | Expected Result | Test Type | Test Reference | Status |
| --- | --- | --- | --- | --- | --- |
| Parsing/config | `--index name:path` without options | Existing mount behavior remains unchanged | regression | `searchlite-http/src/lib.rs` parser tests | done |
| Parsing/config | `--index name:path,auto_commit=30,auto_refresh=10` | Per-index overrides parsed and applied | unit | `searchlite-http/src/lib.rs` parser tests | done |
| Parsing/config | Unknown option key or non-numeric/negative interval | Startup parse error with clear reason | unit | `searchlite-http/src/lib.rs` parser tests | done |
| Runtime commit | `auto_commit_secs=0` | No auto-commit task runs for index | unit/integration | `searchlite-http/src/lib.rs` task wiring tests | todo |
| Runtime commit | Pending writes + `auto_commit_secs>0` | Writes become durable/visible without explicit `/commit` | integration | `searchlite-http/src/lib.rs::auto_commit_persists_pending_writes` | done |
| Runtime commit | Manual `/commit` overlaps timer tick | No deadlock; writer serialization preserved | integration | `searchlite-http/src/lib.rs` concurrency test | todo |
| Runtime refresh | `auto_refresh_secs=0` | No periodic refresh task runs | unit/integration | `searchlite-http/src/lib.rs` task wiring tests | todo |
| Runtime refresh | Commit marker unchanged since last refresh | Timer skips refresh to avoid thrash | unit | `searchlite-http/src/lib.rs::refresh_guard_skips_unchanged_commit_marker` | done |
| Runtime refresh | Commit marker advances | Refresh executes once and marker updates | unit/integration | `searchlite-http/src/lib.rs::auto_refresh_runs_after_commit_change` | todo |
| API contract | `GET /indexes` for initialized index | Returns `exists=true`, `committed_at`, `doc_count`, timer/config flags | integration | `searchlite-http/src/lib.rs::list_indexes_exposes_runtime_metadata` | done |
| API contract | `GET /indexes` for uninitialized mount | Returns `exists=false` with nullable runtime stats fields | integration | `searchlite-http/src/lib.rs::list_indexes_exposes_runtime_metadata` | done |
| Security/redaction | `/indexes` response contents | No write-key hash/salt/segment binding fields exposed | regression | `searchlite-http/src/lib.rs::indexes_endpoint_redacts_sensitive_metadata` | todo |

## Adversarial Cases
- [ ] Invalid interval values: `auto_commit=`, `auto_refresh=-1`, `auto_commit=abc`, overflow values.
- [ ] Mixed per-index and global defaults where only one override is set.
- [ ] Multiple indexes with heterogeneous timer configs (`items` vs `listings`) started together.
- [ ] Timer ticks while index is missing/uninitialized (should log and continue, not panic).
- [ ] High-frequency intervals (1s) under concurrent ingest/commit requests.

## Verification Checklist
- [x] `cargo fmt --all`
- [x] `cargo build --all --all-features`
- [x] `cargo test --all --all-features`
- [x] `cargo clippy --all --all-features --all-targets -- -D warnings`
- [ ] `cargo bench -p searchlite-core` when perf-sensitive.

## Planned Touchpoints
- `searchlite-http/src/lib.rs`
  - parse `--index` runtime options and plumb effective per-index timer config.
  - add maintenance task spawning, commit/refresh loops, and cooperative shutdown handling.
  - extend `/indexes` response model and endpoint implementation.
  - add parser/scheduler/endpoint integration tests.
- `searchlite-http/Cargo.toml`
  - enable `tokio` time support for interval scheduling.
- `openapi.yaml`
  - update `/indexes` schemas for new fields.
- `README.md` (and optionally `docs/quickstart.md` if aligned)
  - document timer flags, per-index examples, and response fields.

## Review Summary
- Key risks:
  - Timer contention with ingest-heavy workloads causing unnecessary lock waits.
  - Contract drift between Rust response structs and OpenAPI docs.
  - Refresh loops creating avoidable overhead if commit-change guard is wrong.
- Tests added:
  - parser coverage for runtime mount options.
  - scheduler integration coverage for auto-commit behavior.
  - endpoint coverage for `/indexes` metadata + redaction.
- Follow-ups:
  - Consider optional jitter/backpressure controls for large multi-index deployments.
  - Consider exposing refresh/commit counters in `/indexes` for observability.
