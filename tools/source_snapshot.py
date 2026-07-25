#!/usr/bin/env python
"""Freeze the OLD source code's identity before anyone edits a ported line.

Why this has to run before the first port
------------------------------------------
Once ``src/kinescore`` starts changing, the only way to know whether a metric
still matches the source benchmark is to compare against *something*. That
something cannot be "the source repos, as they currently are" -- Marionette-
ciasc (**A**) is someone's active working tree (``git status`` below is not
clean) and Marionette-fkjepa (**B**) is a *worktree whose ``.git`` pointer is
dead* (see :func:`_fkjepa_git_state`), so it has no commit we can cite at all.
If either mutates or the checkout disappears, "byte-identical to source" stops
being checkable. Hashing the files now, while they are still the pre-port
ground truth, is the only point at which that claim is verifiable.

This script does not read the *content* of the sources for correctness -- that
is :mod:`gen_golden`'s job, which imports and executes them under
``.venv-marionette``. This script only proves *which bytes* those executions
were run against, so a hash mismatch later means "the port drifted or the
source moved", never "we forgot which version we started from".

What "working tree, not HEAD" means for A
------------------------------------------
``git status --porcelain`` on A shows modified and untracked files. The
``judge/*.py`` files this project actually ports may or may not be among them
(check the dirty list this script prints), but even if they show clean, HEAD
alone does not prove what is on disk -- only a hash of the working tree does.
We record both: HEAD for a human-readable "as of roughly when", and the SHA-256
of the actual bytes read for what was really used.

Usage
-----
    python tools/source_snapshot.py --source-a PATH_TO_MARIONETTE_CIASC \\
                                     --source-b PATH_TO_MARIONETTE_FKJEPA \\
                                     --out-dir provenance/

No path defaults to anyone's home directory on purpose (see repo CLAUDE.md /
task rules) -- both source roots are required arguments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["SourceFile", "hash_file", "snapshot"]

#: Files this benchmark ports, relative to each source root. This list *is*
#: the scope of the port -- adding a file here means "we now also promise to
#: reproduce this module's behaviour byte-for-byte before editing it".
#: Keeping it here (not as a CLI flag) means the scope is reviewed as a code
#: change, not silently redefined by whoever's invocation happens to run.
#:
#: The list must cover EVERY source module the ported code cites. An earlier
#: revision hashed only seven files while the package claimed to port fifteen,
#: so eight "verbatim" claims were unverifiable -- a provenance record that is
#: silently incomplete is worse than none, because it reads as exhaustive.
#: ``tests/test_provenance_complete.py`` pins this by cross-checking the list
#: against every source path mentioned in ``src/``.
SOURCES_A: tuple[str, ...] = (
    "judge/fk.py",
    "judge/physics.py",
    "judge/pixel_judge.py",
    "judge/scorer.py",
    "scripts/precompute_dino_cache.py",
    "scripts/train_judge.py",
)
SOURCES_B: tuple[str, ...] = (
    "models/evaluation/motion_reference.py",
    "models/evaluation/corruptions.py",
    "models/evaluation/pixel_judge_gr1.py",
    "models/evaluation/readout_v2_scorer.py",
    "models/kinematics/gr1_fk.py",
    "models/physics/feasibility.py",
    "models/physics/robot_colliders.py",
    "models/posendf/readout_v2.py",
    "eval/bench/manifest.py",
    "eval/bench/scoring.py",
    "eval/bench/stats.py",
)

#: Top-level entries of A that are known to be environment-specific (symlinks
#: to sibling checkouts, caches, or venvs) and must never be copied into this
#: repo even by accident. Populated at runtime by :func:`_never_copy`; this
#: constant only documents the *reason* each category is excluded.
_NEVER_COPY_REASONS = {
    "broken_symlink": "dangling symlink -- target does not exist on this host",
    "venv": "python environment, host- and interpreter-specific",
    "checkpoint_dir": "trained weights, not source code",
}


@dataclass(frozen=True)
class SourceFile:
    """One snapshotted file: identity + provenance, nothing interpreted."""

    repo: str          # "A" or "B"
    rel_path: str
    abs_path: str
    sha256: str
    n_lines: int
    mtime_iso: str
    size_bytes: int


def hash_file(path: Path) -> tuple[str, int, int]:
    """Return ``(sha256_hex, n_lines, size_bytes)`` for ``path``.

    Line count is computed from the same bytes that are hashed (not
    re-decoded with universal-newline translation), so it agrees with what
    ``wc -l`` would report on the exact file that produced the hash.
    """
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    n_lines = data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)
    return sha, n_lines, len(data)


def _git(repo: Path, *args: str) -> str:
    """Run git and return stdout with the trailing newline removed only.

    NOT ``.strip()``: for multi-line output (``git status --porcelain``) a
    full strip only trims the *first* line's leading space, silently
    corrupting the first dirty-file path (`` M ciasc/io.py`` -> parsed as
    ``iasc/io.py``). Every consumer here either wants one line (rev-parse) or
    parses line-by-line itself, so only the outer trailing ``\\n`` is noise.
    """
    out = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    )
    return out.stdout.rstrip("\n")


def _ciasc_git_state(root: Path) -> dict:
    """HEAD + dirty-file list of A -- the working tree is what we port.

    A clean ``git status`` would let us cite a commit; A's is not clean (this
    is someone's active workspace), so every consumer of this snapshot must
    treat the SHA-256 in ``sources.sha256`` as authoritative and the HEAD
    recorded here as "approximately when", not "exactly what".
    """
    try:
        head = _git(root, "rev-parse", "HEAD")
        status = _git(root, "status", "--porcelain")
        dirty = [line[3:] for line in status.splitlines() if line.strip()]
        is_dirty = bool(dirty)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return {"error": str(exc)}
    return {
        "root": str(root),
        "head": head,
        "is_dirty": is_dirty,
        "dirty_files": dirty,
        "note": (
            "is_dirty=True means HEAD does not describe the files on disk. "
            "The SHA-256 in sources.sha256 is what was actually read; HEAD is "
            "context only."
        ),
    }


def _fkjepa_git_state(root: Path) -> dict:
    """B's ``.git`` is a worktree pointer into a repo that no longer exists.

    ``git -C B rev-parse HEAD`` fails (or worse, silently resolves to the
    *wrong* repo if something is later checked out at the referenced gitdir),
    so B has **no citable commit at all**. The SHA-256 hash of each B source
    file is the entire provenance record for that file -- there is no
    "as of commit X" fallback to lean on.
    """
    git_marker = root / ".git"
    result = {
        "root": str(root),
        "git_marker_path": str(git_marker),
        "has_citable_commit": False,
    }
    if not git_marker.exists():
        result["note"] = ".git marker absent entirely; no git metadata at all."
        return result
    if git_marker.is_dir():
        # A real repo, not a worktree pointer -- unexpected but handle it.
        try:
            result["head"] = _git(root, "rev-parse", "HEAD")
            result["has_citable_commit"] = True
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            result["error"] = str(exc)
        return result
    # ``.git`` is a file: worktree pointer ("gitdir: <path>").
    pointer_text = git_marker.read_text().strip()
    gitdir = pointer_text.removeprefix("gitdir:").strip()
    gitdir_path = Path(gitdir)
    result["pointer_text"] = pointer_text
    try:
        result["pointer_target_exists"] = gitdir_path.exists()
    except PermissionError as exc:
        # Observed on this host: the pointer resolves under a path
        # (/mnt/.../quangpt3/...) this user can no longer even stat, likely
        # because the workspace was later relocated to a per-user home
        # (/mnt/.../hunght23/quangpt3_data/...). That is itself confirming
        # evidence the worktree pointer is dead, not a bug in this script.
        result["pointer_target_exists"] = False
        result["pointer_target_error"] = f"{type(exc).__name__}: {exc}"
    result["note"] = (
        "B is a git WORKTREE whose gitdir pointer is dead (target directory "
        "gone from -- or inaccessible under -- the primary checkout's "
        ".git/worktrees/). `git -C B ...` fails outright, so B contributes NO "
        "commit hash to this snapshot -- only file-level SHA-256 provenance "
        "exists for B's sources."
    )
    return result


def _never_copy(root_a: Path) -> list[dict]:
    """Enumerate top-level entries of A that must never be copied verbatim.

    Discovers broken symlinks empirically (does not assume which names are
    broken -- that depends on which sibling checkouts happen to exist on the
    current host) plus the two known non-source categories (venvs, checkpoint
    dirs) that are symlinked in but are not this project's concern.
    """
    entries = []
    for child in sorted(root_a.iterdir()):
        if child.is_symlink():
            resolves = child.exists()  # exists() follows symlinks -> False if dangling
            target = os.readlink(child)
            if not resolves:
                entries.append({
                    "path": child.name,
                    "kind": "broken_symlink",
                    "target": target,
                    "reason": _NEVER_COPY_REASONS["broken_symlink"],
                })
            elif "venv" in child.name:
                entries.append({
                    "path": child.name,
                    "kind": "venv",
                    "target": target,
                    "reason": _NEVER_COPY_REASONS["venv"],
                })
            elif "checkpoint" in child.name:
                entries.append({
                    "path": child.name,
                    "kind": "checkpoint_dir",
                    "target": target,
                    "reason": _NEVER_COPY_REASONS["checkpoint_dir"],
                })
    return entries


def snapshot(source_a: Path, source_b: Path, out_dir: Path) -> list[SourceFile]:
    """Hash every file in :data:`SOURCES_A` / :data:`SOURCES_B` and write the
    provenance record. Returns the list of hashed files for the caller to
    print a summary from."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[SourceFile] = []

    for repo_label, root, rels in (("A", source_a, SOURCES_A), ("B", source_b, SOURCES_B)):
        for rel in rels:
            abs_path = root / rel
            if not abs_path.is_file():
                raise FileNotFoundError(
                    f"expected source file missing: {abs_path} "
                    f"(listed in SOURCES_{repo_label} in this script)"
                )
            sha, n_lines, size = hash_file(abs_path)
            mtime = datetime.fromtimestamp(abs_path.stat().st_mtime, tz=timezone.utc)
            files.append(SourceFile(
                repo=repo_label, rel_path=rel, abs_path=str(abs_path),
                sha256=sha, n_lines=n_lines, mtime_iso=mtime.isoformat(),
                size_bytes=size,
            ))

    # -- provenance/sources.sha256 (TSV): the hash ledger -----------------
    sha_path = out_dir / "sources.sha256"
    with sha_path.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["repo", "rel_path", "sha256", "n_lines", "size_bytes", "mtime_iso"])
        for f in files:
            w.writerow([f.repo, f.rel_path, f.sha256, f.n_lines, f.size_bytes, f.mtime_iso])

    # -- provenance/copy_manifest.tsv: what gen_golden.py actually read ----
    # "suggested_dest" is informational only (this tool does not own src/ and
    # does not know the final module layout); it exists so a reviewer can
    # eyeball "does every source file have a home" without guessing.
    manifest_path = out_dir / "copy_manifest.tsv"
    dest_guess = {
        "judge/fk.py": "kinescore.robots.franka",
        "judge/physics.py": "kinescore.metrics (KinematicConsistency/PhysicsConsistency)",
        "judge/pixel_judge.py": "kinescore.heads (AttentivePoseHead/DinoPoseHead)",
        "judge/scorer.py": "kinescore.core.scorer",
        "models/evaluation/motion_reference.py": "kinescore.reference",
        "models/kinematics/gr1_fk.py": "kinescore.robots.gr1",
        "models/physics/smoothness.py": "kinescore.metrics (not yet wired to a fixture)",
    }
    with manifest_path.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["repo", "rel_path", "sha256", "suggested_dest", "notes"])
        for f in files:
            notes = ""
            if f.repo == "B" and f.rel_path == "models/kinematics/gr1_fk.py":
                notes = "sibling models/kinematics/fk.py is byte-identical to A judge/fk.py"
            if f.rel_path == "models/physics/smoothness.py":
                notes = "listed source; no golden fixture in Phase 0/1 (see task table)"
            w.writerow([f.repo, f.rel_path, f.sha256, dest_guess.get(f.rel_path, ""), notes])

    # -- provenance/git_state.json: A's HEAD+dirty, B's dead worktree ------
    git_state = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ciasc_A": _ciasc_git_state(source_a),
        "fkjepa_B": _fkjepa_git_state(source_b),
    }
    (out_dir / "git_state.json").write_text(json.dumps(git_state, indent=2) + "\n")

    # -- provenance/never_copy.txt: broken symlinks + non-source dirs in A -
    never_copy = _never_copy(source_a)
    lines = [
        "# Top-level entries of source A that must NEVER be copied into kinescore.",
        "# Regenerated by tools/source_snapshot.py -- do not hand-edit.",
        "#",
        "# path\tkind\ttarget\treason",
    ]
    for e in never_copy:
        lines.append(f"{e['path']}\t{e['kind']}\t{e['target']}\t{e['reason']}")
    (out_dir / "never_copy.txt").write_text("\n".join(lines) + "\n")

    return files


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source-a", required=True, type=Path,
                    help="path to the Marionette-ciasc checkout")
    p.add_argument("--source-b", required=True, type=Path,
                    help="path to the Marionette-fkjepa checkout")
    p.add_argument("--out-dir", default=Path("provenance"), type=Path,
                    help="where to write sources.sha256 / copy_manifest.tsv / "
                         "git_state.json / never_copy.txt (default: provenance/)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if not args.source_a.is_dir():
        print(f"error: --source-a {args.source_a} is not a directory", file=sys.stderr)
        return 2
    if not args.source_b.is_dir():
        print(f"error: --source-b {args.source_b} is not a directory", file=sys.stderr)
        return 2

    files = snapshot(args.source_a, args.source_b, args.out_dir)

    print(f"snapshotted {len(files)} source files -> {args.out_dir}/")
    for f in files:
        print(f"  [{f.repo}] {f.rel_path}  sha256={f.sha256[:12]}...  "
              f"{f.n_lines} lines  {f.size_bytes} bytes")
    git_state = json.loads((args.out_dir / "git_state.json").read_text())
    a = git_state["ciasc_A"]
    if "head" in a:
        print(f"\nA (ciasc) HEAD: {a['head']}  dirty={a['is_dirty']} "
              f"({len(a.get('dirty_files', []))} changed paths)")
    b = git_state["fkjepa_B"]
    print(f"B (fkjepa) has_citable_commit={b['has_citable_commit']}: "
          f"{b.get('note', '')}")
    never_copy = (args.out_dir / "never_copy.txt").read_text().splitlines()
    n_entries = len([l for l in never_copy if l and not l.startswith("#")])
    print(f"never_copy.txt: {n_entries} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
