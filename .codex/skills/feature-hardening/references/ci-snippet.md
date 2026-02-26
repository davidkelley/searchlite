# CI Integration Snippet

Use this as a starting point for a hardening gate job:

```yaml
- name: Feature hardening matrix update
  run: |
    python3 .codex/skills/feature-hardening/scripts/update_feature_matrix.py \
      --feature "${FEATURE_NAME}" \
      --base origin/main \
      --output-root docs/feature-hardening \
      --create-if-missing

- name: Feature hardening quality gate
  run: |
    .codex/skills/feature-hardening/scripts/run_feature_hardening.sh \
      --feature "${FEATURE_NAME}" \
      --base origin/main \
      --skip-bench
```

Notes:
- Set `FEATURE_NAME` from branch naming convention or workflow input.
- Keep bench optional in CI if runtime is too high; run bench locally for perf-sensitive changes.

