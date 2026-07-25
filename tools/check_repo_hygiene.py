#!/usr/bin/env python3
"""Pre-commit repo hygiene: catch the mistakes that cannot be undone.

Three failures are effectively permanent once committed, which is why this runs
*before* the first commit rather than in review:

1. **Large files.** A prior extraction of this same code shipped 9 GB of outputs
   inside its own repository. Git history cannot be pruned of that without a
   rewrite that breaks every clone.
2. **Absolute paths.** That same repo baked one user's home directory into its
   path defaults, so a fresh checkout resolved to directories that did not
   exist and failed far from the cause. kinescore resolves everything through
   ``KINESCORE_*`` environment variables instead.
3. **Symlinks.** The source tree contains dangling symlinks into a deleted
   sibling checkout. A broken symlink in git is nearly invisible in review and
   breaks a fresh clone.

Run: ``python tools/check_repo_hygiene.py [--repo .]``
Exit code is non-zero when any check fails, so it works as a git hook or a CI
step. ``tests/test_repo_hygiene.py`` calls the same functions.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

#: Hard ceilings. The tracked tree is source plus small golden fixtures; if
#: either is exceeded, something that should be referenced got vendored instead.
MAX_TRACKED_FILE_BYTES = 1_000_000
MAX_TRACKED_TREE_BYTES = 20_000_000

#: Directories that are legitimately untracked build/venv/output noise.
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv",
             "node_modules", "build", "dist", "out", "outputs", "assets",
             "model_ckpt"}

#: Absolute-path shapes that must never appear in committed source. These are
#: the literal prefixes that made the prior extraction unrunnable on any other
#: machine, plus the usernames on this host.
FORBIDDEN_PATH_RE = re.compile(
    r"(/mnt/data/sftp/data/\w+|/home/\w+|/Users/\w+|/tmp/claude|"
    r"quangpt3|hunght23|marionette-workspace)")

#: Files that are allowed to mention host paths: the tooling that *takes* source
#: trees as arguments, the environment template, and documentation of provenance.
PATH_EXEMPT = {".env.example"}
PATH_EXEMPT_DIRS = {"provenance", "docs"}

TEXT_SUFFIXES = {".py", ".toml", ".cfg", ".ini", ".txt", ".yaml", ".yml", ".sh"}


def _iter_files(repo: Path):
    for p in repo.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.is_file() or p.is_symlink():
            yield p


def check_no_symlinks(repo: Path) -> list[str]:
    """No symlinks at all in the tracked tree."""
    return [f"symlink: {p.relative_to(repo)} -> {p.readlink()}"
            for p in _iter_files(repo) if p.is_symlink()]


def check_file_sizes(repo: Path) -> list[str]:
    """No single tracked file above the ceiling, and a bounded total."""
    problems, total = [], 0
    for p in _iter_files(repo):
        if p.is_symlink():
            continue
        size = p.stat().st_size
        total += size
        if size > MAX_TRACKED_FILE_BYTES:
            problems.append(
                f"large file: {p.relative_to(repo)} is {size / 1e6:.1f} MB "
                f"(limit {MAX_TRACKED_FILE_BYTES / 1e6:.1f} MB). Reference it "
                f"via a KINESCORE_* env var instead of vendoring it.")
    if total > MAX_TRACKED_TREE_BYTES:
        problems.append(
            f"tree too large: {total / 1e6:.1f} MB "
            f"(limit {MAX_TRACKED_TREE_BYTES / 1e6:.1f} MB)")
    return problems


def check_no_absolute_paths(repo: Path) -> list[str]:
    """No host-specific absolute paths in committed source.

    ``tools/`` is exempt: those scripts legitimately take source-tree locations
    as arguments. What matters is that nothing under ``src/`` hardcodes one.
    """
    problems = []
    for p in _iter_files(repo):
        if p.is_symlink() or p.suffix not in TEXT_SUFFIXES:
            continue
        rel = p.relative_to(repo)
        if rel.name in PATH_EXEMPT or rel.parts[0] in PATH_EXEMPT_DIRS:
            continue
        if rel.parts[0] == "tools":
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            m = FORBIDDEN_PATH_RE.search(line)
            if m:
                problems.append(
                    f"absolute path: {rel}:{lineno} contains {m.group(0)!r}")
    return problems


def check_golden_budget(repo: Path, limit: int = 4_194_304) -> list[str]:
    """Golden fixtures stay small enough to live in git comfortably."""
    golden = repo / "tests" / "golden"
    if not golden.is_dir():
        return []
    total = sum(p.stat().st_size for p in golden.glob("*.npz"))
    if total > limit:
        return [f"golden fixtures total {total / 1e6:.2f} MB "
                f"(budget {limit / 1e6:.2f} MB)"]
    return []


CHECKS = {
    "symlinks": check_no_symlinks,
    "file sizes": check_file_sizes,
    "absolute paths": check_no_absolute_paths,
    "golden budget": check_golden_budget,
}


def run_all(repo: Path) -> dict[str, list[str]]:
    return {name: fn(repo) for name, fn in CHECKS.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".", type=Path)
    args = ap.parse_args()
    repo = args.repo.resolve()

    results = run_all(repo)
    failed = False
    for name, problems in results.items():
        if problems:
            failed = True
            print(f"FAIL {name} ({len(problems)}):")
            for p in problems[:20]:
                print(f"  {p}")
            if len(problems) > 20:
                print(f"  ... and {len(problems) - 20} more")
        else:
            print(f"ok   {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
