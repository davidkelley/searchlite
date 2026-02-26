#!/usr/bin/env python3
"""Update matrix metadata (changed files + timestamp) from git diff."""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import re
import subprocess
import sys
from typing import List

BEGIN_MARKER = "<!-- BEGIN_CHANGED_FILES -->"
END_MARKER = "<!-- END_CHANGED_FILES -->"


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "feature"


def run_lines(cmd: List[str]) -> List[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def changed_files(base: str) -> List[str]:
    files = run_lines(["git", "diff", "--name-only", f"{base}...HEAD"])
    if files:
        return files
    files = run_lines(["git", "diff", "--name-only", "HEAD"])
    if files:
        return files
    status = run_lines(["git", "status", "--porcelain"])
    parsed = []
    for row in status:
        if len(row) > 3:
            parsed.append(row[3:].strip())
    return parsed


def replace_changed_files_block(content: str, files: List[str]) -> str:
    if files:
        new_block = "\n".join(f"- `{path}`" for path in files)
    else:
        new_block = "- (no changed files detected)"

    block = f"{BEGIN_MARKER}\n{new_block}\n{END_MARKER}"
    pattern = re.compile(
        rf"{re.escape(BEGIN_MARKER)}.*?{re.escape(END_MARKER)}",
        flags=re.DOTALL,
    )
    if pattern.search(content):
        return pattern.sub(block, content, count=1)
    return content + "\n\n## Changed Files\n" + block + "\n"


def update_timestamp(content: str) -> str:
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    return re.sub(
        r"^- Last updated: .*$",
        f"- Last updated: {now}",
        content,
        flags=re.MULTILINE,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update docs/feature-hardening/<feature>/matrix.md from git diff"
    )
    parser.add_argument("--feature", required=True, help="Feature name (will be slugified)")
    parser.add_argument(
        "--base",
        default="origin/main",
        help="Diff base for changed-file detection",
    )
    parser.add_argument(
        "--output-root",
        default="docs/feature-hardening",
        help="Root where feature matrices are stored",
    )
    parser.add_argument(
        "--create-if-missing",
        action="store_true",
        help="Initialize matrix automatically if missing",
    )
    args = parser.parse_args()

    feature = slugify(args.feature)
    matrix_path = pathlib.Path(args.output_root).resolve() / feature / "matrix.md"
    if not matrix_path.exists():
        if not args.create_if_missing:
            print(
                f"[ERROR] Missing matrix: {matrix_path}\n"
                "Run init_feature_hardening.py first, or pass --create-if-missing.",
                file=sys.stderr,
            )
            return 1
        init_script = pathlib.Path(__file__).resolve().parent / "init_feature_hardening.py"
        init_proc = subprocess.run(
            [sys.executable, str(init_script), "--feature", feature, "--output-root", args.output_root],
            check=False,
        )
        if init_proc.returncode != 0:
            return init_proc.returncode

    content = matrix_path.read_text(encoding="utf-8")
    files = changed_files(args.base)
    content = replace_changed_files_block(content, files)
    content = update_timestamp(content)
    matrix_path.write_text(content, encoding="utf-8")
    print(f"[OK] Updated matrix: {matrix_path}")
    print(f"[OK] Changed files captured: {len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

