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

4. **Undocumented scripts.** ``scripts/`` collects ad-hoc runbooks copied in from
   wherever an operator last ran them (see ``scripts/*.sh``'s own history) --
   their filenames alone do not say what data they touch or why they exist.
   Every file in ``scripts/`` must carry a 4-line ``What`` / ``Why`` / ``Input``
   / ``Output`` header (see any existing ``scripts/*.sh`` or
   ``scripts/convert_lerobot_to_train.py`` for the pattern) so that six months
   from now the script is still legible without archaeology.

5. **Broken doc cross-references.** A docs reorganisation (e.g. moving 11 files
   into ``legacy_docs/``) silently strands every ``docs/<name>.md`` citation in
   a docstring or comment that pointed at the old location -- 66 files did
   exactly this after one such move, none of them raising an ImportError or a
   test failure, because a citation in a comment is not code. This check scans
   ``src/`` and ``tests/`` for every ``(legacy_)?docs/*.md`` reference it can
   find (a regex over file content, not a hand-typed list of doc names -- so a
   future rename is caught the same way) and fails on any whose target does
   not exist on disk.

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

#: ``.json`` is included because machine-generated manifests are exactly where an
#: absolute host path slips through unnoticed -- tests/golden/MANIFEST.json carried
#: the developer's ``$HOME`` and workspace root all the way to the first public push
#: without this check ever looking at it.
TEXT_SUFFIXES = {".py", ".toml", ".cfg", ".ini", ".txt", ".yaml", ".yml", ".sh", ".json"}


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


#: How far into a file to look for the What/Why/Input/Output header -- generous
#: enough for a shebang line, `set -euo pipefail`, and a couple of blank lines
#: before the header starts, without scanning the whole file.
HEADER_SCAN_LINES = 40

#: One regex per required header field. Matches a `#`-prefixed (shell/Python)
#: or bare (docstring) comment line starting with the field name and a colon,
#: so both `# What:   ...` (scripts/*.sh) and a module-docstring-embedded
#: `What:` line would pass -- the check cares that the field is present and
#: human-legible near the top of the file, not which comment style carries it.
HEADER_FIELD_RE = {
    field: re.compile(rf"^\s*#*\s*{field}\s*:", re.IGNORECASE)
    for field in ("What", "Why", "Input", "Output")
}

#: Files that are legitimately not runbooks and so don't need the header
#: (an `__init__.py` marker, if one is ever added; extend as needed).
SCRIPTS_HEADER_EXEMPT = {"__init__.py"}


def check_scripts_have_header(repo: Path) -> list[str]:
    """Every file directly under ``scripts/`` must carry a What/Why/Input/Output header.

    ``scripts/`` is where this repo's ad-hoc operator runbooks live (see
    module docstring point 4) -- unlike ``src/``, filenames alone are not
    enough context for what a script touches or why it exists. This check
    is deliberately generous about *where* in the first
    :data:`HEADER_SCAN_LINES` lines each field appears and what comment
    style wraps it; it only refuses a script that is missing a field
    entirely.
    """
    scripts_dir = repo / "scripts"
    if not scripts_dir.is_dir():
        return []
    problems = []
    for p in sorted(scripts_dir.rglob("*")):
        if p.is_dir() or p.name in SCRIPTS_HEADER_EXEMPT:
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(repo).parts):
            continue
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        head = lines[:HEADER_SCAN_LINES]
        missing = [field for field, rx in HEADER_FIELD_RE.items()
                  if not any(rx.match(line) for line in head)]
        if missing:
            problems.append(
                f"missing header field(s) {missing} in first "
                f"{HEADER_SCAN_LINES} lines: {p.relative_to(repo)} -- every "
                f"file in scripts/ needs a What/Why/Input/Output header.")
    return problems


#: Matches a ``docs/NAME.md`` or ``legacy_docs/NAME.md`` reference anywhere in
#: a line -- deliberately a generic path shape, not an alternation over a
#: hand-typed list of known doc filenames, so a doc added or renamed after
#: this check was written is still covered without editing this file. An
#: optional trailing ``#anchor`` (as in ``docs/ARCHITECTURE.md#adding-a-robot``)
#: is captured separately and dropped before checking existence -- an anchor
#: is a location *within* a file, not part of its path.
DOC_LINK_RE = re.compile(r"\b((?:legacy_)?docs/[A-Za-z0-9_./-]+\.md)(?:#[A-Za-z0-9_-]+)?")

#: Directory-name suffixes to treat as generated/non-source for the doc-link
#: check -- ``*.egg-info`` is regenerated by ``pip install -e .`` from
#: whatever ``README.md`` said at build time, so it can reintroduce a stale
#: reference through no edit of its own; it is not something a contributor
#: hand-writes doc citations into.
_DOC_LINK_SKIP_SUFFIXES = (".egg-info",)


def check_doc_links(repo: Path) -> list[str]:
    """Every ``docs/*.md`` / ``legacy_docs/*.md`` reference under ``src/`` and
    ``tests/`` must resolve to a real file, relative to the repo root.

    See the module docstring's point 5: this is what stops a docs
    reorganisation from silently stranding a comment's citation at the old
    path. The set of references to check is discovered by scanning file
    content with :data:`DOC_LINK_RE`, not hand-copied, so it stays correct as
    both the doc tree and the citing source files change.
    """
    problems = []
    for sub in ("src", "tests"):
        base = repo / sub
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_dir() or p.is_symlink():
                continue
            rel_parts = p.relative_to(repo).parts
            if any(part in SKIP_DIRS for part in rel_parts):
                continue
            if any(part.endswith(_DOC_LINK_SKIP_SUFFIXES) for part in rel_parts):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                for m in DOC_LINK_RE.finditer(line):
                    target = m.group(1)
                    if not (repo / target).is_file():
                        problems.append(
                            f"broken doc link: {p.relative_to(repo)}:{lineno} "
                            f"references {target!r}, which does not exist "
                            f"under {repo}")
    return problems


CHECKS = {
    "symlinks": check_no_symlinks,
    "file sizes": check_file_sizes,
    "absolute paths": check_no_absolute_paths,
    "golden budget": check_golden_budget,
    "script headers": check_scripts_have_header,
    "doc links": check_doc_links,
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
