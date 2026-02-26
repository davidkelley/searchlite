#!/usr/bin/env python3
"""Create a per-feature hardening matrix from the local template."""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import re
import subprocess
import sys


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "feature"


def current_branch() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def render(template: str, feature: str, branch: str) -> str:
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    out = template.replace("{{FEATURE}}", feature)
    out = out.replace("{{BRANCH}}", branch)
    out = out.replace("{{TIMESTAMP}}", now)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Initialize docs/feature-hardening/<feature>/matrix.md"
    )
    parser.add_argument("--feature", required=True, help="Feature name (will be slugified)")
    parser.add_argument(
        "--output-root",
        default="docs/feature-hardening",
        help="Output root for feature matrices",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Branch name to include in the matrix header",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing matrix file",
    )
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).resolve().parent
    template_path = script_dir.parent / "references" / "matrix-template.md"
    if not template_path.exists():
        print(f"[ERROR] Missing template: {template_path}", file=sys.stderr)
        return 1

    feature = slugify(args.feature)
    root = pathlib.Path(args.output_root).resolve()
    feature_dir = root / feature
    matrix_path = feature_dir / "matrix.md"

    if matrix_path.exists() and not args.force:
        print(f"[SKIP] Matrix already exists: {matrix_path}")
        print("Use --force to overwrite.")
        return 0

    feature_dir.mkdir(parents=True, exist_ok=True)
    template = template_path.read_text(encoding="utf-8")
    branch = args.branch or current_branch()
    matrix_path.write_text(render(template, feature, branch), encoding="utf-8")

    print(f"[OK] Wrote matrix: {matrix_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

