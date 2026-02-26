---
name: feature-hardening
description: Use when implementing or updating Searchlite features that need pre-review hardening via per-feature invariant matrix maintenance, adversarial/regression coverage updates, and full quality-gate verification.
---

# Feature Hardening

## Workflow

1. Initialize a feature matrix once:
   - `python3 .codex/skills/feature-hardening/scripts/init_feature_hardening.py --feature <feature>`

2. Refresh matrix metadata from the current diff each implementation cycle:
   - `python3 .codex/skills/feature-hardening/scripts/update_feature_matrix.py --feature <feature> --base origin/main`

3. Expand the matrix with feature-specific invariants and adversarial cases:
   - Add rows for known failure classes (scope mismatch, invalid config, dotted names, missing fast fields, null/empty input, etc.).
   - Link each row to a test name or planned test file.

4. Run the hardening quality gate before requesting review:
   - `.codex/skills/feature-hardening/scripts/run_feature_hardening.sh --feature <feature> --base origin/main --bench-if-sensitive`

5. Post a short hardening summary in the PR:
   - New tests added
   - Invariants covered
   - Verification commands completed
   - Any known residual risks

## Automation

- Install an optional pre-push hook:
  - `.codex/skills/feature-hardening/scripts/install_pre_push_hook.sh`

- For CI integration, use the snippet in:
  - `.codex/skills/feature-hardening/references/ci-snippet.md`

## Expected Artifacts

- Matrix path:
  - `docs/feature-hardening/<feature>/matrix.md`

- The matrix must include:
  - current changed-file list (auto-updated by script markers)
  - explicit invariant rows
  - adversarial coverage checklist
  - test references and completion status
