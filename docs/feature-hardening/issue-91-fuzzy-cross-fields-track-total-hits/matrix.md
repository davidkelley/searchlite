# Feature Hardening Matrix: issue-91-fuzzy-cross-fields-track-total-hits

- Branch: feat/nested-aggregations
- Last updated: 2026-03-02 09:46:40Z

## Scope
- [x] Implement issue #91 behavior:
  - `multi_match` per-query fuzziness (`AUTO` and explicit edit distance) without changing default non-fuzzy behavior.
  - `multi_match` `cross_fields` scoring that combines term contributions across fields (BM25F-like) rather than treating each field independently.
  - request-level exact total hit counting via `track_total_hits=true`.
- [x] Out of scope for this change set:
  - full Elasticsearch parity for every `multi_match` option not already in Searchlite.
  - large query-language additions beyond `multi_match` fuzziness and exact-hit toggle plumbing.
  - changing existing default performance behavior when `track_total_hits` is omitted/false.

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
| API/Serde | `multi_match.fuzziness` omitted | Existing behavior unchanged (no fuzzy expansion) | unit | `searchlite-core/tests/query_ast.rs` | done |
| API/Serde | `multi_match.fuzziness="AUTO"` | Parsed and mapped per-term length (`1-2=>0`, `3-5=>1`, `>=6=>2`) | unit | `searchlite-core/tests/query_ast.rs` | done |
| API/Serde | `multi_match.fuzziness` numeric (0..2) | Parsed and bounded edit distance applied | unit | `searchlite-core/tests/query_ast.rs` | done |
| Validation | Invalid `fuzziness` value/string | Structured parse/validation error (HTTP 400 surface) | integration | `searchlite-http/src/lib.rs` tests | done |
| Planner | `CrossFields` keeps term grouping across listed fields | Term group keys include all fields while scorer leaf strategy supports combined scoring | unit | `searchlite-core/src/query/planner.rs` tests | done |
| Scoring | `CrossFields` BM25F-like aggregation | Split-term docs across fields score/rank correctly vs single-field hits | integration | `searchlite-core/tests/multi_field.rs` | done |
| Scoring | Fuzzy multi-match weight decay | Edit distance reduces contribution (`d=0>d=1>d=2`) | unit/integration | `searchlite-core/tests/smoke.rs` + new reader tests | done |
| Matching | `minimum_should_match` + `operator` still enforced with fuzzy/cross-fields | Recall expansion never bypasses term-group requirements | integration | `searchlite-core/tests/multi_field.rs` | done |
| Total Hits | `track_total_hits=false` (default) | Keep current fast-path estimate behavior and latency profile | regression | `searchlite-core/tests/pruning.rs` | done |
| Total Hits | `track_total_hits=true` | `total_hits_estimate` reflects exact post-filter match count | integration | `searchlite-core/tests/smoke.rs` (new exact-hit tests) | done |
| Pagination | cursor/search_after with exact hits | Cursor math and `next_cursor` behavior unchanged; total count exact | integration | `searchlite-core/tests/smoke.rs` pagination tests | done |
| Explain/Profile | `explain`/`profile` + new paths | No panic/regression; profile timings remain populated | regression | `searchlite-core/tests/smoke.rs` + existing reader tests | done |
| Performance | Hot query paths changed (`planner`, `reader`, `wand`) | No material regression in baseline benches; fuzzy path bounded | bench/profile | `cargo bench -p searchlite-core` (`aggs`, `end_to_end`) | done |

## Adversarial Cases
- [x] Empty query string in `multi_match` with fuzziness set.
- [x] `fuzziness` provided on non-string query paths is ignored consistently on non-`multi_match` clauses (`searchlite-core/tests/query_ast.rs::fuzziness_on_non_multi_match_query_is_ignored`).
- [x] `fuzziness` numeric above supported max edits (e.g. `3`) returns validation error.
- [x] Analyzer emits zero tokens; behavior stays deterministic (`searchlite-core/tests/multi_field.rs::cross_fields_zero_token_analyzer_behavior_is_deterministic`).
- [x] Duplicate fields in `multi_match.fields` do not double-count unexpectedly (`searchlite-core/tests/multi_field.rs::cross_fields_duplicate_fields_do_not_change_scores`).
- [x] Cross-fields queries over mixed field kinds (text + keyword + numeric) remain safe and deterministic (`searchlite-core/tests/multi_field.rs::cross_fields_mixed_field_kinds_are_deterministic`).
- [x] `track_total_hits=true` with `limit=0`, `return_hits=false`, and aggregations still yields exact count.
- [x] Cursor/search_after + `track_total_hits=true` does not under/over count skipped docs.
- [x] Fuzzy expansion cap reached (`max_expansions`) does not panic and remains bounded (`searchlite-core/tests/smoke.rs::fuzzy_respects_max_expansions`).

## Verification Checklist
- [x] `cargo fmt --all`
- [x] `cargo build --all --all-features`
- [x] `cargo test --all --all-features`
- [x] `cargo clippy --all --all-features --all-targets -- -D warnings`
- [x] `cargo bench -p searchlite-core` when perf-sensitive.

## Planned Touchpoints
- `searchlite-core/src/api/types.rs`
  - add `MultiMatchFuzziness` type and `QueryNode::MultiMatch.fuzziness`.
  - add `SearchRequest.track_total_hits: Option<bool>` with serde defaults.
- `searchlite-core/src/query/planner.rs`
  - plumb per-group/per-term fuzzy settings from `MultiMatch`.
  - finalize `CrossFields` planning semantics for combined scoring.
- `searchlite-core/src/api/reader.rs`
  - apply per-group fuzzy options in term expansion.
  - implement exact-hit collection path when `track_total_hits=true` (disable pruning shortcut by forcing full accept/collect).
  - ensure cursor/search_after total math remains correct under exact mode.
- `searchlite-core/tests/multi_field.rs`
  - add ranking-focused `cross_fields` expectations (BM25F-like behavior).
- `searchlite-core/tests/smoke.rs`
  - add exact total hit toggle tests and fuzzy multi-match typo coverage.
- `searchlite-core/tests/pruning.rs`
  - lock in estimate vs exact behavior for `track_total_hits`.
- `searchlite-http/src/lib.rs` tests
  - JSON acceptance/rejection cases for new request fields.

## Review Summary
- Key risks:
  - Cross-fields scoring changes ranking semantics and can cause regressions in existing relevance expectations.
  - Exact hit counting can regress latency if pruning is unintentionally disabled in default mode.
  - New request fields (`fuzziness`, `track_total_hits`) require broad test fixture updates where `SearchRequest` is manually constructed.
- Tests added:
  - `searchlite-core/tests/query_ast.rs`:
    - `multi_match_fuzziness_parses_auto_and_numeric`
    - `multi_match_fuzziness_rejects_out_of_range_value`
    - `fuzziness_on_non_multi_match_query_is_ignored`
  - `searchlite-core/tests/multi_field.rs`:
    - `cross_fields_fuzziness_auto_recovers_typo`
    - `cross_fields_duplicate_fields_do_not_change_scores`
    - `cross_fields_mixed_field_kinds_are_deterministic`
    - `cross_fields_zero_token_analyzer_behavior_is_deterministic`
  - `searchlite-core/tests/smoke.rs`:
    - `track_total_hits_returns_exact_count_for_wand`
    - `track_total_hits_with_zero_limit_counts_all_matches`
    - `track_total_hits_keeps_cursor_totals_exact`
  - `searchlite-http/src/lib.rs`:
    - `invalid_multi_match_fuzziness_returns_bad_request`
- Follow-ups:
  - Consider adding an explicit `total_hits_exact` response flag in a follow-up if clients need certainty without inferring from request flags.
  - Criterion local baselines can show small run-to-run drift (for example, low single-digit changes in `aggs_nested_terms_metadata`); keep watching CI/stable-host benchmark trends before treating these as product regressions.
