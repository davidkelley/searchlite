# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.1](https://github.com/davidkelley/searchlite/compare/searchlite-core-v0.8.0...searchlite-core-v0.8.1) - 2026-04-25

### Other

- update Cargo.toml dependencies

## [0.8.0](https://github.com/davidkelley/searchlite/compare/searchlite-core-v0.7.0...searchlite-core-v0.8.0) - 2026-04-23

### Added

- *(node)* Zod-native schema authoring ([#244](https://github.com/davidkelley/searchlite/pull/244))

### Fixed

- *(aggs)* reject histogram bucket ids whose reconstruction overflows (BUG-410) ([#411](https://github.com/davidkelley/searchlite/pull/411))
- *(aggs)* reject seconds that overflow millis conversion (BUG-408) ([#409](https://github.com/davidkelley/searchlite/pull/409))
- *(query)* reject non-finite tie_breaker at plan time (BUG-399) ([#400](https://github.com/davidkelley/searchlite/pull/400))
- *(query)* floor percentage minimum_should_match (BUG-403) ([#405](https://github.com/davidkelley/searchlite/pull/405))
- *(score)* reject term_weights accumulation overflow (BUG-401) ([#402](https://github.com/davidkelley/searchlite/pull/402))
- *(vectors)* reject L2 vectors whose sum-of-squares overflows f32 (BUG-386) ([#387](https://github.com/davidkelley/searchlite/pull/387))
- *(score)* drop non-finite BM25 scores from WAND/brute-force heaps (BUG-381) ([#382](https://github.com/davidkelley/searchlite/pull/382))
- *(vectors)* saturate vscore * boost overflow to f32 range (BUG-394) ([#398](https://github.com/davidkelley/searchlite/pull/398))
- *(query)* reject group.boost * field.boost overflow at expansion time (BUG-396) ([#397](https://github.com/davidkelley/searchlite/pull/397))
- *(score)* reject non-finite field_value_factor `missing` at plan time (BUG-392) ([#393](https://github.com/davidkelley/searchlite/pull/393))
- *(vectors)* saturate l2_distance overflow to f32::MAX (BUG-388) ([#389](https://github.com/davidkelley/searchlite/pull/389))
- *(vectors)* reject cosine vectors whose sum-of-squares overflows f32 (BUG-384) ([#385](https://github.com/davidkelley/searchlite/pull/385))
- *(query)* reject nested-boost overflow at plan time (BUG-381) ([#383](https://github.com/davidkelley/searchlite/pull/383))
- *(score)* reject non-finite decay factor (BUG-379) ([#380](https://github.com/davidkelley/searchlite/pull/380))
- *(query)* disable WAND pruning when score hook can amplify scores (BUG-376) ([#378](https://github.com/davidkelley/searchlite/pull/378))
- *(score)* reject non-finite decay origin and offset (BUG-373) ([#377](https://github.com/davidkelley/searchlite/pull/377))
- *(score)* guard ScoreExpr::DisMax/Sum against non-finite scores (BUG-374) ([#375](https://github.com/davidkelley/searchlite/pull/375))
- *(score)* reject non-finite Constant score from boost-product overflow (BUG-370) ([#372](https://github.com/davidkelley/searchlite/pull/372))
- *(sort)* reject non-finite F64 search_after and cursor values (BUG-369) ([#371](https://github.com/davidkelley/searchlite/pull/371))
- *(vectors)* reject non-finite components in collect_vector_value (BUG-330) ([#331](https://github.com/davidkelley/searchlite/pull/331))
- *(score)* clamp f64→f32 cast in function_score, rank_feature, script_score (BUG-336) ([#337](https://github.com/davidkelley/searchlite/pull/337))
- *(score)* gate function_score zero-base rewrite on Multiply (BUG-362) ([#363](https://github.com/davidkelley/searchlite/pull/363))
- *(aggs)* reject non-finite and i64-overflow bucket ids in HistogramCollector (BUG-358) ([#359](https://github.com/davidkelley/searchlite/pull/359))
- *(query)* allow infinite WAND upper bounds through pivot finder (BUG-366) ([#367](https://github.com/davidkelley/searchlite/pull/367))
- *(aggs)* reject non-finite values in avg_bucket and sum_bucket pipelines (BUG-324) ([#325](https://github.com/davidkelley/searchlite/pull/325))
- *(score)* reject non-finite Sum and DisMax accumulated scores (BUG-364) ([#365](https://github.com/davidkelley/searchlite/pull/365))
- *(score)* use doc_count for BM25 N so it matches df after deletions (BUG-360) ([#361](https://github.com/davidkelley/searchlite/pull/361))
- *(rescore)* reject non-finite combined scores in rescore (BUG-326) ([#327](https://github.com/davidkelley/searchlite/pull/327))
- *(aggs)* reject non-finite composite histogram bucket from overflow (BUG-356) ([#357](https://github.com/davidkelley/searchlite/pull/357))
- *(aggs)* reject f64-overflow number literals in bucket_script (BUG-354) ([#355](https://github.com/davidkelley/searchlite/pull/355))
- *(score)* reject f64-overflow number literals in script_score (BUG-352) ([#353](https://github.com/davidkelley/searchlite/pull/353))
- *(analysis)* emit multi-token synonyms at first-match position (BUG-347) ([#351](https://github.com/davidkelley/searchlite/pull/351))
- *(pagination)* reject non-finite score bits in cursor decode (BUG-345) ([#350](https://github.com/davidkelley/searchlite/pull/350))
- *(aggs)* bucket_script div guard rejects only exact zero (BUG-346) ([#349](https://github.com/davidkelley/searchlite/pull/349))
- *(aggs)* reject non-finite values in parse_interval_seconds (BUG-344) ([#348](https://github.com/davidkelley/searchlite/pull/348))
- *(sort)* reject non-finite search_after _score after f64→f32 cast (BUG-342) ([#343](https://github.com/davidkelley/searchlite/pull/343))
- *(aggs)* reject non-finite values in derivative and moving_avg pipelines (BUG-322) ([#323](https://github.com/davidkelley/searchlite/pull/323))
- *(vectors)* reject non-finite components in query vector (BUG-340) ([#341](https://github.com/davidkelley/searchlite/pull/341))
- *(aggs)* reject non-finite values in parse_date fallback (BUG-338) ([#339](https://github.com/davidkelley/searchlite/pull/339))
- *(aggs)* reject non-finite `missing` values parsed via string (BUG-334) ([#335](https://github.com/davidkelley/searchlite/pull/335))
- *(aggs)* reject non-finite values in stats and extended_stats (BUG-332) ([#333](https://github.com/davidkelley/searchlite/pull/333))
- *(hybrid)* reject non-finite blended final_score in compute_hybrid_score (BUG-328) ([#329](https://github.com/davidkelley/searchlite/pull/329))
- *(aggs)* numeric tiebreaker in bucket_sort_cmp for numeric bucket keys (BUG-320) ([#321](https://github.com/davidkelley/searchlite/pull/321))
- *(score)* reject non-finite combined scores in function_score (BUG-315) ([#319](https://github.com/davidkelley/searchlite/pull/319))
- *(query)* apply max_expansions globally across segments (BUG-316) ([#318](https://github.com/davidkelley/searchlite/pull/318))
- *(query)* enforce max_expansions globally across segments in term expansion (BUG-316) ([#317](https://github.com/davidkelley/searchlite/pull/317))
- *(aggs)* handle unary `+` in bucket_script tokenizer (BUG-313) ([#314](https://github.com/davidkelley/searchlite/pull/314))
- *(score)* rank_feature Log/Log1p modifiers use log10 not natural log (BUG-311) ([#312](https://github.com/davidkelley/searchlite/pull/312))
- *(score)* handle unary `+` in script_score tokenizer (BUG-309) ([#310](https://github.com/davidkelley/searchlite/pull/310))
- *(score)* field_value_factor Log* modifiers use log10 not natural log (BUG-307) ([#308](https://github.com/davidkelley/searchlite/pull/308))
- *(aggs)* accept leading `+` sign in parse_interval_seconds (BUG-305) ([#306](https://github.com/davidkelley/searchlite/pull/306))
- *(aggs)* return null for empty percentiles/percentile_ranks (BUG-303) ([#304](https://github.com/davidkelley/searchlite/pull/304))
- *(aggs)* treat empty stats/extended_stats as null in pipeline aggs (BUG-301) ([#302](https://github.com/davidkelley/searchlite/pull/302))
- *(aggs)* sum_bucket returns null instead of 0.0 for empty input (BUG-298) ([#299](https://github.com/davidkelley/searchlite/pull/299))
- *(aggs)* use numeric comparison for bucket_sort _key on numeric keys (BUG-296) ([#300](https://github.com/davidkelley/searchlite/pull/300))
- *(aggs)* accept leading sign in parse_interval_seconds (BUG-295) ([#297](https://github.com/davidkelley/searchlite/pull/297))
- *(aggs)* recompute date_histogram fill-loop keys from canonical form (BUG-293) ([#294](https://github.com/davidkelley/searchlite/pull/294))
- *(rescore)* shrink sort_window by removed hits to exclude non-rescored (BUG-291) ([#292](https://github.com/davidkelley/searchlite/pull/292))
- *(aggs)* use checked arithmetic on date_histogram calendar bucket_start (BUG-289) ([#290](https://github.com/davidkelley/searchlite/pull/290))
- *(aggs)* reject non-finite intermediate and final results in eval_rpn (BUG-287) ([#288](https://github.com/davidkelley/searchlite/pull/288))
- *(highlight)* ensure fragment window fully contains match (BUG-285) ([#286](https://github.com/davidkelley/searchlite/pull/286))
- *(aggs)* avg_bucket returns null instead of 0.0 for empty input (BUG-283) ([#284](https://github.com/davidkelley/searchlite/pull/284))
- *(score)* apply max_boost to function score before boost_mode combination ([#280](https://github.com/davidkelley/searchlite/pull/280))
- *(aggs)* moving_avg uses look-back window excluding current bucket (BUG-277) ([#278](https://github.com/davidkelley/searchlite/pull/278))
- *(aggs)* handle unary negation of variables in bucket_script tokenizer (BUG-275) ([#276](https://github.com/davidkelley/searchlite/pull/276))
- *(aggs)* topologically sort pipeline aggs to resolve inter-pipeline dependencies (BUG-273) ([#274](https://github.com/davidkelley/searchlite/pull/274))
- *(aggs)* run bucket_sort after other pipeline aggregations (BUG-271) ([#272](https://github.com/davidkelley/searchlite/pull/272))
- *(aggs)* hard_bounds filters on bucket key, not raw value (BUG-269) ([#270](https://github.com/davidkelley/searchlite/pull/270))
- *(aggs)* bucket_metric_value splits buckets_path on '.' breaking percentile key lookup for non-integer levels (BUG-267) ([#268](https://github.com/davidkelley/searchlite/pull/268))
- *(aggs)* bucket_script to_rpn silently accepts mismatched parentheses, producing wrong evaluation results (BUG-265) ([#266](https://github.com/davidkelley/searchlite/pull/266))
- *(rescore)* non-matching docs receive zero contribution under Multiply/Min (BUG-263) ([#264](https://github.com/davidkelley/searchlite/pull/264))
- *(score)* Log2p modifier uses ln(value+2) instead of log₂(value+1) (BUG-261) ([#262](https://github.com/davidkelley/searchlite/pull/262))
- *(analysis)* skip edge_ngram emission for tokens shorter than min (BUG-259) ([#260](https://github.com/davidkelley/searchlite/pull/260))
- *(aggs)* preserve day-of-month in add_calendar for Month/Quarter/Year (BUG-257) ([#258](https://github.com/davidkelley/searchlite/pull/258))
- *(query)* align normalize_pattern case-folding with default tokenizer (BUG-255) ([#256](https://github.com/davidkelley/searchlite/pull/256))
- *(aggs)* normalize day before month in quarter truncation ([#245](https://github.com/davidkelley/searchlite/pull/245))
- *(fastfields)* reject non-monotonic offsets in list/nested columns ([#254](https://github.com/davidkelley/searchlite/pull/254))
- *(aggs)* date_histogram quarter_interval drops May 31 docs (BUG-233) ([#234](https://github.com/davidkelley/searchlite/pull/234))
- *(merge)* shrink batch instead of abandoning overflowing tier (BUG-008) ([#236](https://github.com/davidkelley/searchlite/pull/236))
- *(ffi)* classify write-key auth errors via typed downcast (BUG-020) ([#237](https://github.com/davidkelley/searchlite/pull/237))
- *(aggs)* preserve sub-day time in add_calendar for offset-aligned bucket keys ([#252](https://github.com/davidkelley/searchlite/pull/252))
- *(aggs)* sort significant_terms by significance score before truncation ([#250](https://github.com/davidkelley/searchlite/pull/250))
- *(query)* preserve wildcard/regex metacharacters in single-token patterns ([#248](https://github.com/davidkelley/searchlite/pull/248))
- *(suggest)* bound fuzzy candidate scan against MAX_SUGGEST_CANDIDATES (BUG-024) ([#228](https://github.com/davidkelley/searchlite/pull/228))
- *(schema)* reject documents that omit non-nullable top-level fields (BUG-224) ([#226](https://github.com/davidkelley/searchlite/pull/226))
- *(aggs)* normalize day before month in quarter calendar truncation (BUG-233) ([#242](https://github.com/davidkelley/searchlite/pull/242))
- *(aggs)* make range/date_range upper bound `to` exclusive ([#240](https://github.com/davidkelley/searchlite/pull/240))
- *(writer)* make WAL commit the durability fence in Writer::commit (BUG-018) ([#238](https://github.com/davidkelley/searchlite/pull/238))
- *(aggs)* bound top_hits size and from against unbounded allocation (BUG-222) ([#225](https://github.com/davidkelley/searchlite/pull/225))
- *(aggs)* bound moving_avg predict against unbounded allocation (BUG-221) ([#223](https://github.com/davidkelley/searchlite/pull/223))
- *(fastfields)* reject non-UTF-8 field names and dict entries (BUG-217) ([#218](https://github.com/davidkelley/searchlite/pull/218))
- *(aggs)* apply top_hits from skip after cross-segment merge (BUG-215) ([#216](https://github.com/davidkelley/searchlite/pull/216))
- *(core)* unify keyword case-folding across postings and filter paths ([#213](https://github.com/davidkelley/searchlite/pull/213))
- *(aggs)* align percentile_rank TDigest path with inclusive semantics (BUG-209) ([#211](https://github.com/davidkelley/searchlite/pull/211))
- *(terms)* bound Vec capacity against untrusted u64 term_count (BUG-207) ([#208](https://github.com/davidkelley/searchlite/pull/208))
- *(postings)* bound Vec capacity against untrusted u32 counts (BUG-205) ([#206](https://github.com/davidkelley/searchlite/pull/206))
- *(aggs)* cap date_histogram empty bucket span at MAX_BUCKETS (BUG-200) ([#204](https://github.com/davidkelley/searchlite/pull/204))
- *(fastfields)* bound Vec::with_capacity against untrusted counts (BUG-012) ([#196](https://github.com/davidkelley/searchlite/pull/196))
- *(segment)* persist vector and HNSW artefacts via atomic_write (BUG-013) ([#199](https://github.com/davidkelley/searchlite/pull/199))
- *(varint)* guard read_u32_var against final-byte shift overflow (BUG-198) ([#201](https://github.com/davidkelley/searchlite/pull/201))
- *(storage)* use unique staging filename in FsStorage::atomic_write (BUG-019) ([#195](https://github.com/davidkelley/searchlite/pull/195))
- *(query)* regex_literal_prefix is too greedy with quantifiers/alternation (BUG-202) ([#203](https://github.com/davidkelley/searchlite/pull/203))
- *(wal)* surface CRC-valid but undecodable entries during replay (BUG-007) ([#194](https://github.com/davidkelley/searchlite/pull/194))
- *(wal)* fsync in append_commit to guarantee durability (BUG-006) ([#193](https://github.com/davidkelley/searchlite/pull/193))
- *(docstore)* publish offsets only after write_all succeeds (BUG-011) ([#192](https://github.com/davidkelley/searchlite/pull/192))
- *(aggs)* clip histogram extended_bounds fill range to hard_bounds (BUG-188) ([#191](https://github.com/davidkelley/searchlite/pull/191))
- *(aggs)* use floor for date_histogram bucket_start (BUG-030) ([#190](https://github.com/davidkelley/searchlite/pull/190))
- *(schema)* validate nested field array element types (BUG-009) ([#189](https://github.com/davidkelley/searchlite/pull/189))
- *(query)* clamp prefix/wildcard/regex max_expansions to hard ceiling (BUG-022) ([#187](https://github.com/davidkelley/searchlite/pull/187))
- *(postings)* reject non-monotonic positions in write_term (BUG-004) ([#185](https://github.com/davidkelley/searchlite/pull/185))
- *(merge)* surface missing segment IDs in merge_segments (BUG-028) ([#184](https://github.com/davidkelley/searchlite/pull/184))
- *(vectors)* bound read_vector_file allocations against file size (BUG-014) ([#183](https://github.com/davidkelley/searchlite/pull/183))
- *(terms)* reject non-UTF-8 term bytes in read_terms (BUG-010) ([#180](https://github.com/davidkelley/searchlite/pull/180))
- *(varint)* guard read_u64 against shift overflow (BUG-002) ([#177](https://github.com/davidkelley/searchlite/pull/177))
- *(query)* clamp phrase slop and saturate i32 cast (BUG-026) ([#175](https://github.com/davidkelley/searchlite/pull/175))
- *(postings)* always skip on-disk position bytes regardless of caller flag (BUG-001) ([#173](https://github.com/davidkelley/searchlite/pull/173))
- *(aggs)* reject degenerate histogram intervals to prevent OOM (BUG-027) ([#169](https://github.com/davidkelley/searchlite/pull/169))
- *(api)* reject non-ASCII cursor input without panicking ([#168](https://github.com/davidkelley/searchlite/pull/168))

### Other

- *(query)* cover BUG-381 boost-overflow rejection across MultiMatch/DisMax/nested Bool ([#414](https://github.com/davidkelley/searchlite/pull/414))
- *(score)* cover BUG-381 score_adjust salvage path ([#404](https://github.com/davidkelley/searchlite/pull/404))
- *(reader)* share doc_id allocations via Arc<str> in doc_lookup ([#281](https://github.com/davidkelley/searchlite/pull/281))
- *(http)* share IndexReader pool across parallel multi_search ([#282](https://github.com/davidkelley/searchlite/pull/282))
- *(postings)* add test for position delta overflow ([#214](https://github.com/davidkelley/searchlite/pull/214))
- *(wand)* use block-level tf bounds during BMW advancement (BUG-005) ([#239](https://github.com/davidkelley/searchlite/pull/239))
- *(phrase)* binary search doc_id in sorted postings (BUG-021) ([#232](https://github.com/davidkelley/searchlite/pull/232))

## [0.7.0](https://github.com/davidkelley/searchlite/compare/searchlite-core-v0.6.4...searchlite-core-v0.7.0) - 2026-04-13

### Fixed

- address perf PR review feedback — correctness and tests ([#137](https://github.com/davidkelley/searchlite/pull/137))

### Other

- optimize scoring and filter hot paths ([#135](https://github.com/davidkelley/searchlite/pull/135))

## [0.6.4](https://github.com/davidkelley/searchlite/compare/searchlite-core-v0.6.3...searchlite-core-v0.6.4) - 2026-04-13

### Other

- replace TinyFst BTreeMap with HashMap + sorted Vec ([#129](https://github.com/davidkelley/searchlite/pull/129))
- eliminate per-document String allocations in filter evaluation ([#130](https://github.com/davidkelley/searchlite/pull/130))

## [0.6.3](https://github.com/davidkelley/searchlite/compare/searchlite-core-v0.6.2...searchlite-core-v0.6.3) - 2026-04-13

### Fixed

- WAND inner-loop sort and highlight UTF-8 boundary safety ([#128](https://github.com/davidkelley/searchlite/pull/128))

## [0.6.2](https://github.com/davidkelley/searchlite/compare/searchlite-core-v0.6.1...searchlite-core-v0.6.2) - 2026-04-13

### Added

- replace custom index schema with JSON Schema + searchlite: vocabulary ([#118](https://github.com/davidkelley/searchlite/pull/118))

## [0.6.1](https://github.com/davidkelley/searchlite/compare/searchlite-core-v0.6.0...searchlite-core-v0.6.1) - 2026-04-10

### Fixed

- WASM getrandom error and Docker missing workspace members ([#116](https://github.com/davidkelley/searchlite/pull/116))

## [0.6.0](https://github.com/davidkelley/searchlite/compare/searchlite-core-v0.5.0...searchlite-core-v0.6.0) - 2026-04-10

### Added

- product readiness — modularize reader, scale benchmarks, merge policy ([#107](https://github.com/davidkelley/searchlite/pull/107))
- Fuzzy/cross-field search + exact total hits ([#103](https://github.com/davidkelley/searchlite/pull/103))
- feat/nested aggregations ([#102](https://github.com/davidkelley/searchlite/pull/102))
- partial update APIs ([#101](https://github.com/davidkelley/searchlite/pull/101))
- mget & `search_after` ([#95](https://github.com/davidkelley/searchlite/pull/95))

### Fixed

- inline format args for clippy on Rust 1.88.0 ([#113](https://github.com/davidkelley/searchlite/pull/113))
- fix/mget doc id response ([#105](https://github.com/davidkelley/searchlite/pull/105))

### Other

- overhaul README, benchmarks, and documentation suite ([#108](https://github.com/davidkelley/searchlite/pull/108))

## [0.5.0](https://github.com/davidkelley/searchlite/compare/searchlite-core-v0.4.1...searchlite-core-v0.5.0) - 2026-01-30

### Added

- optional write keys ([#85](https://github.com/davidkelley/searchlite/pull/85))

## [0.4.1](https://github.com/davidkelley/searchlite/compare/searchlite-core-v0.4.0...searchlite-core-v0.4.1) - 2026-01-28

### Fixed

- make ffi/wasm search requests atomic and align   return_stored defaults ([#73](https://github.com/davidkelley/searchlite/pull/73))

## [0.4.0](https://github.com/davidkelley/searchlite/compare/searchlite-core-v0.3.1...searchlite-core-v0.4.0) - 2026-01-28

### Added

- vector cap env

### Fixed

- address remaining limit=0 review feedback
- unify doc id validation   across ingest and delete paths
- default search request limit and return_stored
- stream ndjson batches with single writer ([#65](https://github.com/davidkelley/searchlite/pull/65))

## [0.3.1](https://github.com/davidkelley/searchlite/compare/searchlite-core-v0.3.0...searchlite-core-v0.3.1) - 2026-01-16

### Other

- perf/optimize wand and http ([#63](https://github.com/davidkelley/searchlite/pull/63))

## [0.3.0](https://github.com/davidkelley/searchlite/compare/searchlite-core-v0.2.1...searchlite-core-v0.3.0) - 2026-01-15

### Other

- remove filters final ([#62](https://github.com/davidkelley/searchlite/pull/62))

## [0.2.1](https://github.com/davidkelley/searchlite/compare/searchlite-core-v0.2.0...searchlite-core-v0.2.1) - 2026-01-14

### Fixed

- harden nested validation and collapse grouping ([#55](https://github.com/davidkelley/searchlite/pull/55))
- harden core replay and aggs
- address review feedback
- address review feedback
- harden core durability and bounds
- fsync vector files via storage helper
- harden fsync durability and wal cleanup
- guard compaction and writer consistency
- harden core validation and durability
- improve vector recall and durability

### Other

- avoid Debug bound in zstd guard
- add wal coverage; clean rollback artifacts
- satisfy clippy seek-from-current
- add wal is_empty for clippy
- log WAL sync errors on drop

## [0.2.0](https://github.com/davidkelley/searchlite/compare/searchlite-core-v0.1.0...searchlite-core-v0.2.0) - 2026-01-09

### Added

- add multi-vector ranking and scoring hooks
- implemented http service

### Fixed

- address additional review feedback
- address review feedback
- return estimate counts regardless of limit value

## [0.1.0](https://github.com/davidkelley/searchlite/releases/tag/searchlite-core-v0.1.0) - 2026-01-08

### Added

- add P6 aggregations collapse highlight
- initial pass at expanded aggregations and collapsing fields
- improve vector search api and perf
- implemented vector search
- initial implementation for function scoring
- initial implementation of search-as-you-type
- initial implementation of multi-field matching
- tokenizer pipeline
- implemented improved structured query/filter ast
- implemented fuzzy searching
- implemented update/delete actions
- sorting implementation
- improved performance of cursor-based pagination
- add cursor-based pagination
- completed implementation for nested filters
- improved performance for nested-field filters
- adding receipes example
- implemented release-plz
- initial release version
- support in-memory option
- initial commit

### Fixed

- ensure correct workflow artifacts are released
- address clippy regressions
- address review feedback on highlighting and collapse
- address clippy warnings in aggregations
- address review feedback on aggs and collapse
- improve highlighting and add request schema
- address review feedback
- dedupe segment writer and vector filter defaults
- Consolidated write_segment_from_iter so there’s only one definition
- make segment iter writer fallible and non-exact
- add iterator-friendly segment writer entrypoint
- tighten vector query handling
- restore bench build without vectors
- enforce alpha on bm25-only hybrid hits
- honor function_score min_score semantics
- validate function score params and clarify scoring edges
- finalize P4 scoring tweaks and docs
- apply function scoring hooks
- tidy search_as_you_type errors and regex prefix handling
- address additional PR feedback
- address PR review feedback
- address latest review feedback
- address review feedback
- address review feedback
- address PR review feedback
- address PR review feedback
- validate doc_id and cover delete WAL replay
- re-ordering commit flow
- hold the writer lock so WAL replay and live doc loading cant race
- alterations for copilot feedback
- implemented match based sorting algorithm
- improving cursor implementation
- addressing feedback on performance
- updated rust to 1.92.0

### Other

- Merge pull request #34 from davidkelley/fix/ci-builds-and-releases
- Merge branch 'main' into feat/vector-search
- cap vector search work to available vectors
- clarify multi-field section and planner TODO
- Merge branch 'main' into feat/wasm
- fixing formatting error
- Merge branch 'main' into feat/plan-implementation-for-nested-property-filters
- Merge pull request #2 from davidkelley/feat/plan-elasticsearch-like-aggregations-implementation
- using `contains()` instead of `iter().any()` is more efficient
