# Feature Hardening Matrix: cross-surface-integration-suite

- Branch: main
- Last updated: 2026-03-04 15:56:53Z

## Scope
- [x] Build a large integration suite in `integration/` that validates Searchlite behavior through core Rust APIs, HTTP endpoints, and CLI commands using fixtures from `examples/`.
- [x] Cover lifecycle flows end-to-end: `init/add/commit/refresh/search/search_after/mget/update/delete/stats/inspect/compact` with explicit surface capability mapping.
- [x] Add adversarial and contract-focused coverage for status codes, error payload shape, pagination invariants, and user error handling.
- [x] Out of scope for this change set:
  - changing public API semantics for unsupported operations on a given surface.
  - adding brand-new product features unrelated to integration hardening.
  - replacing existing unit/bench suites; this augments them.

## Changed Files
<!-- BEGIN_CHANGED_FILES -->
- `.agents/skills/notify-on-completion/SKILL.md`
- `.codex/skills/bug-hunting/SKILL.md`
- `.codex/skills/code-quality/SKILL.md`
- `.codex/skills/debugging-playbook/SKILL.md`
- `.codex/skills/docs-style/SKILL.md`
- `.codex/skills/feature-hardening/SKILL.md`
- `.codex/skills/feature-hardening/agents/openai.yaml`
- `.codex/skills/feature-hardening/references/ci-snippet.md`
- `.codex/skills/feature-hardening/references/matrix-template.md`
- `.codex/skills/feature-hardening/scripts/init_feature_hardening.py`
- `.codex/skills/feature-hardening/scripts/install_pre_push_hook.sh`
- `.codex/skills/feature-hardening/scripts/run_feature_hardening.sh`
- `.codex/skills/feature-hardening/scripts/update_feature_matrix.py`
- `.codex/skills/index-lifecycle/SKILL.md`
- `.codex/skills/integration-testing/SKILL.md`
- `.codex/skills/notify-on-completion/SKILL.md`
- `.codex/skills/performance-improvements/SKILL.md`
- `.codex/skills/unit-testing/SKILL.md`
- `.dockerignore`
- `.github/workflows/release-artifacts.yml`
- `.github/workflows/release.yml`
- `.gitignore`
- `Cargo.lock`
- `Cargo.toml`
- `Dockerfile`
- `README.md`
- `docs/bindings.md`
- `docs/feature-hardening/issue-91-fuzzy-cross-fields-track-total-hits/matrix.md`
- `docs/feature-hardening/issue-92-auto-commit-refresh-indexes/matrix.md`
- `docs/quickstart.md`
- `examples/recipes/queries/agg-macros-by-diet.json`
- `examples/recipes/queries/collapse-quick-by-cuisine.json`
- `examples/recipes/queries/fuzzy-weeknight-orzo.json`
- `examples/recipes/queries/gluten-free-fruit-crisp.json`
- `examples/recipes/queries/instant-pot-chili-rescore.json`
- `examples/recipes/queries/meal-prep-vegan-chili.json`
- `examples/recipes/queries/mediterranean-romaine-salad.json`
- `examples/recipes/queries/pescatarian-shrimp-curry.json`
- `examples/recipes/queries/vegan-tofu-high-protein.json`
- `examples/recipes/queries/weeknight-orzo-vegetarian.json`
- `examples/video-games/queries/achievement-guide-platinum.json`
- `examples/video-games/queries/aggregations-era-platforms.json`
- `examples/video-games/queries/collapse-review-by-game.json`
- `examples/video-games/queries/emulation-notes-crt-shader.json`
- `examples/video-games/queries/fuzzy-meta-ps5-misspell.json`
- `examples/video-games/queries/high-score-modern-reviews.json`
- `examples/video-games/queries/modern-meta-ps5.json`
- `examples/video-games/queries/retro-cheat-infinite-lives.json`
- `examples/video-games/queries/speedrun-rescore-bmw.json`
- `examples/video-games/queries/speedrun-route-sub100.json`
- `examples/video-games/queries/wildcard-konami-code.json`
- `openapi.yaml`
- `release-plz.toml`
- `search-request.schema.json`
- `searchlite-cli/CHANGELOG.md`
- `searchlite-cli/Cargo.lock`
- `searchlite-cli/Cargo.toml`
- `searchlite-cli/src/main.rs`
- `searchlite-core/CHANGELOG.md`
- `searchlite-core/Cargo.lock`
- `searchlite-core/Cargo.toml`
- `searchlite-core/benches/aggs.rs`
- `searchlite-core/benches/end_to_end.rs`
- `searchlite-core/src/api/builder.rs`
- `searchlite-core/src/api/errors.rs`
- `searchlite-core/src/api/mod.rs`
- `searchlite-core/src/api/reader.rs`
- `searchlite-core/src/api/types.rs`
- `searchlite-core/src/api/writer.rs`
- `searchlite-core/src/index/fastfields.rs`
- `searchlite-core/src/index/manifest.rs`
- `searchlite-core/src/index/mod.rs`
- `searchlite-core/src/index/segment.rs`
- `searchlite-core/src/index/wal.rs`
- `searchlite-core/src/query/aggs/mod.rs`
- `searchlite-core/src/query/planner.rs`
- `searchlite-core/src/query/sort.rs`
- `searchlite-core/src/query/wand.rs`
- `searchlite-core/src/util/doc_id.rs`
- `searchlite-core/src/util/mod.rs`
- `searchlite-core/src/util/path_scope.rs`
- `searchlite-core/src/util/write_key.rs`
- `searchlite-core/tests/aggregation_bounds.rs`
- `searchlite-core/tests/aggregations.rs`
- `searchlite-core/tests/analyzers.rs`
- `searchlite-core/tests/coverage.rs`
- `searchlite-core/tests/function_score.rs`
- `searchlite-core/tests/multi_field.rs`
- `searchlite-core/tests/partial_update.rs`
- `searchlite-core/tests/prefix_and_suggest.rs`
- `searchlite-core/tests/pruning.rs`
- `searchlite-core/tests/query_ast.rs`
- `searchlite-core/tests/regressions.rs`
- `searchlite-core/tests/smoke.rs`
- `searchlite-core/tests/sorting.rs`
- `searchlite-core/tests/vector_search.rs`
- `searchlite-ffi/CHANGELOG.md`
- `searchlite-ffi/Cargo.lock`
- `searchlite-ffi/Cargo.toml`
- `searchlite-ffi/searchlite.h`
- `searchlite-ffi/src/lib.rs`
- `searchlite-http/CHANGELOG.md`
- `searchlite-http/Cargo.lock`
- `searchlite-http/src/lib.rs`
- `searchlite-wasm/CHANGELOG.md`
- `searchlite-wasm/Cargo.lock`
- `searchlite-wasm/Cargo.toml`
- `searchlite-wasm/index.html`
- `searchlite-wasm/src/wasm.rs`
<!-- END_CHANGED_FILES -->

## Invariant Matrix
| Area | Scenario | Expected Result | Test Type | Test Reference | Status |
| --- | --- | --- | --- | --- | --- |
| Fixtures | Load `examples/recipes` + `examples/video-games` schemas/data/queries | Fixture loader parses all files and preserves stable corpus counts | integration | `integration/tests/fixtures_loading.rs` | done |
| Core lifecycle | `init->add->commit->search` | Search returns seeded docs after commit | integration | `integration/tests/lifecycle_matrix.rs` | done |
| HTTP lifecycle | Full endpoint chain including `/refresh`, `/mget`, `/stats`, `/inspect`, `/compact` | Status codes and payloads match contract; state transitions visible | integration | `integration/tests/lifecycle_matrix.rs` + `integration/tests/contracts_http.rs` | done |
| CLI lifecycle | `init/add/update/delete/commit/search/inspect/compact` | Commands succeed with deterministic JSON/output contracts | integration | `integration/tests/lifecycle_matrix.rs` + `integration/tests/contracts_cli.rs` | done |
| Capability contract | Unsupported operation for a surface | Explicit `NotSupported` (or equivalent) assertion, never silent skip | integration | `integration/tests/lifecycle_matrix.rs` | done |
| Pagination | `search_after` continuation after sorted query | No duplicate/omitted hits across page boundary | integration | `integration/tests/lifecycle_matrix.rs` + `integration/tests/feature_cross_matrix.rs` | done |
| Pagination validation | `cursor + search_after` and `search_after + from` | Request rejected with stable error contract | integration | `integration/tests/adversarial_matrix.rs` + `integration/tests/contracts_http.rs` | done |
| Query parity | Example query corpus across `bm25/wand/bmw` | Expected output invariants hold across surfaces after normalization | integration | `integration/tests/expected_outputs.rs` + `integration/tests/feature_cross_matrix.rs` | done |
| Mutations | Update then delete then commit | Updated fields visible before delete; deleted docs absent after commit | integration | `integration/tests/lifecycle_matrix.rs` | done |
| Stats/inspect invariants | pre/post commit and pre/post compact | doc/segment counters evolve as expected without data loss | integration | `integration/tests/lifecycle_matrix.rs` + `integration/tests/contracts_core.rs` | done |
| Error shape | HTTP bad request / missing index / conflict / payload-too-large | `{"error":{"type":"...","reason":"..."}}` always present | integration | `integration/tests/contracts_http.rs` | done |
| CLI error contract | Invalid args / malformed request file | Non-zero exit and deterministic error marker | integration | `integration/tests/contracts_cli.rs` | done |
| Matrix generation | Pairwise + targeted full-cross generation | Stable case IDs, no duplicates, high-volume case count (thousands in full mode) | integration | `integration/tests/feature_cross_matrix.rs` | done |
| Runtime control | Quick/full modes and shard env vars | Local and CI runs remain deterministic and bounded | integration | `integration/tests/feature_cross_matrix.rs` | done |

## Adversarial Cases
- [x] Invalid schema body at init.
- [x] Malformed NDJSON in add/bulk update payloads.
- [ ] Missing `docs` or `ids` fields in bulk/delete operations.
- [ ] Invalid IDs (control chars, whitespace-only, empty).
- [x] Pagination misuse (`cursor+search_after`, `search_after+from`, `cursor+limit=0`).
- [x] Unknown index and alias resolution failure paths.
- [ ] HTTP max body size rejection and mget upper-bound rejection.
- [x] CLI unknown command and bad flag combinations.
- [x] Surface capability mismatch assertions (unsupported op attempted on CLI/core).

## Verification Checklist
- [x] `cargo fmt --all`
- [x] `cargo build --all --all-features`
- [x] `cargo test --all --all-features`
- [x] `cargo clippy --all --all-features --all-targets -- -D warnings`
- [ ] `cargo bench -p searchlite-core` when perf-sensitive.

## Planned Touchpoints
- `Cargo.toml`
  - add `integration` workspace member.
- `integration/Cargo.toml`
  - define integration harness dependencies.
- `integration/src/{fixtures,matrix,scenario,assertions,normalization}.rs`
  - fixture loading, matrix generation, expected-output and contract assertions.
- `integration/src/surfaces/{core,http,cli}.rs`
  - operation adapters and capability map.
- `integration/tests/{lifecycle_matrix,feature_cross_matrix,adversarial_matrix,contracts_http,contracts_cli,contracts_core,expected_outputs}.rs`
  - broad matrix, negative, and contract coverage.
- `Justfile`
  - quick/full integration entrypoints.

## Review Summary
- Key risks:
  - matrix combinatorics can explode runtime without sharding/quick mode controls.
  - cross-surface output normalization can hide real regressions if over-normalized.
  - unsupported operation handling can become ambiguous without explicit capability assertions.
- Tests added:
  - `integration/tests/fixtures_loading.rs`
  - `integration/tests/surface_smoke.rs`
  - `integration/tests/expected_outputs.rs`
  - `integration/tests/lifecycle_matrix.rs`
  - `integration/tests/feature_cross_matrix.rs`
  - `integration/tests/adversarial_matrix.rs`
  - `integration/tests/contracts_http.rs`
  - `integration/tests/contracts_cli.rs`
  - `integration/tests/contracts_core.rs`
- Follow-ups:
  - Add CI sharding strategy if full suite runtime exceeds target budget.
  - Consider nightly full-matrix runs with PR-time quick matrix subset.
