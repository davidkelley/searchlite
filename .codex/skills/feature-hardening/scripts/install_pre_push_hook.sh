#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
hook_path="$repo_root/.git/hooks/pre-push"
script_rel=".codex/skills/feature-hardening/scripts/run_feature_hardening.sh"

cat > "$hook_path" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

branch="$(git rev-parse --abbrev-ref HEAD)"
feature="${FEATURE_HARDENING_FEATURE:-${branch#*/}}"
base="${FEATURE_HARDENING_BASE:-origin/main}"

if [[ -x ".codex/skills/feature-hardening/scripts/run_feature_hardening.sh" ]]; then
  ".codex/skills/feature-hardening/scripts/run_feature_hardening.sh" --feature "$feature" --base "$base" --skip-bench
fi
EOF

chmod +x "$hook_path"
echo "[OK] Installed pre-push hook: $hook_path"
echo "[INFO] Override feature/base with FEATURE_HARDENING_FEATURE / FEATURE_HARDENING_BASE env vars."

