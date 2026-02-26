# Feature Hardening Matrix: {{FEATURE}}

- Branch: {{BRANCH}}
- Last updated: {{TIMESTAMP}}

## Scope
- [ ] Describe intended behavior.
- [ ] Describe out-of-scope behavior.

## Changed Files
<!-- BEGIN_CHANGED_FILES -->
- (run `update_feature_matrix.py` to populate)
<!-- END_CHANGED_FILES -->

## Invariant Matrix
| Area | Scenario | Expected Result | Test Type | Test Reference | Status |
| --- | --- | --- | --- | --- | --- |
| Validation | Happy path input | Accept and return expected result | integration | TODO | todo |
| Validation | Invalid config/input | Return structured error | unit/integration | TODO | todo |
| Scope/pathing | Relative and absolute paths | Resolve correctly and reject invalid scope | integration | TODO | todo |
| Regression | Prior bug class | Existing bug class remains fixed | regression | TODO | todo |
| Performance | Hot path touched | No material regression | bench/profile | TODO | todo |

## Adversarial Cases
- [ ] Null, empty, and whitespace inputs.
- [ ] Dotted names and special characters.
- [ ] Cross-scope mismatches.
- [ ] Missing fast field / unsupported type.

## Verification Checklist
- [ ] `cargo fmt --all`
- [ ] `cargo build --all --all-features`
- [ ] `cargo test --all --all-features`
- [ ] `cargo clippy --all --all-features --all-targets -- -D warnings`
- [ ] `cargo bench -p searchlite-core` when perf-sensitive.

## Review Summary
- Key risks:
- Tests added:
- Follow-ups:

