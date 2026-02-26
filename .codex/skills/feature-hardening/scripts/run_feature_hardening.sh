#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_feature_hardening.sh --feature <name> [--base <ref>] [--output-root <dir>] [--bench|--bench-if-sensitive|--skip-bench]

Examples:
  .codex/skills/feature-hardening/scripts/run_feature_hardening.sh --feature nested-aggregations --base origin/main --bench-if-sensitive
  .codex/skills/feature-hardening/scripts/run_feature_hardening.sh --feature auth-refresh --skip-bench
EOF
}

feature=""
base="origin/main"
output_root="docs/feature-hardening"
bench_mode="if-sensitive" # always|if-sensitive|never

while [[ $# -gt 0 ]]; do
  case "$1" in
    --feature)
      feature="${2:-}"
      shift 2
      ;;
    --base)
      base="${2:-}"
      shift 2
      ;;
    --output-root)
      output_root="${2:-}"
      shift 2
      ;;
    --bench)
      bench_mode="always"
      shift
      ;;
    --bench-if-sensitive)
      bench_mode="if-sensitive"
      shift
      ;;
    --skip-bench)
      bench_mode="never"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$feature" ]]; then
  echo "[ERROR] --feature is required" >&2
  usage
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$script_dir/update_feature_matrix.py" --feature "$feature" --base "$base" --output-root "$output_root" --create-if-missing

cargo fmt --all
cargo build --all --all-features
cargo test --all --all-features
cargo clippy --all --all-features --all-targets -- -D warnings

if [[ "$bench_mode" != "never" ]]; then
  changed_files="$(git diff --name-only "$base"...HEAD || true)"
  perf_sensitive_regex='^searchlite-core/src/(query|index|api/reader\.rs|api/writer\.rs)'
  if [[ "$bench_mode" == "always" ]] || printf '%s\n' "$changed_files" | rg -q "$perf_sensitive_regex"; then
    cargo bench -p searchlite-core
  fi
fi

echo "[OK] Feature hardening checks completed for: $feature"

