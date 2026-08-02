"""The provenance record must cover every source this package claims to port.

An earlier revision hashed seven source files while the package cited fifteen,
so eight "ported verbatim" docstring claims had no SHA-256 behind them. That is
a worse failure than having no provenance at all: `sources.sha256` *reads* as
exhaustive, so a reviewer checking it would conclude the whole port was
verified when half of it was not.

This test makes the scope self-maintaining. It scrapes the source modules named
anywhere in ``src/`` and asserts each one is either enumerated in
``tools/source_snapshot.py`` or listed in :data:`CITED_NOT_PORTED` below.
Porting a new module without extending the snapshot scope fails here rather
than silently shrinking the audit.

Matching is by **module basename**, deliberately. Citations appear in two
shapes -- filesystem (``Marionette-ciasc/judge/fk.py``) and dotted
(``models.posendf.readout_v2.clamp_for_fk``) -- and the dotted form gives no
reliable way to tell where the module path ends and the symbol begins. The
basename is the part that is unambiguous in both, and it is enough to catch the
failure this test exists for: a whole source file cited but never hashed.

CPU-only, no torch, no network -- pure text analysis of the repo.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: Source modules that ``src/`` mentions but deliberately does NOT port. Each
#: must have a matching entry in ``legacy_docs/PROVENANCE.md`` under "not ported".
#: Listing them here rather than loosening the check keeps the omission a
#: reviewed decision instead of an accident.
CITED_NOT_PORTED: frozenset[str] = frozenset({
    "ctrl_world",            # the world model itself; kinescore only scores its output
    "pipeline_ctrl_world",
    "geometry_head",         # geometry referee -- separate axis, needs a depth stack
    "geometric_consistency",
    "camera_solver",
    "arm_saliency",
    "consistency_metrics",   # folded into judge/physics.py upstream; same lineage
    "physics_metrics",       # AST-identical to judge/physics.py, which IS hashed
    "pixel_judge",           # B's older single-view ancestor of A's, which IS hashed
    "longhorizon_eval",
    "io",                    # ciasc/io.py -- superseded by the video-file contract
    "fk_encoder", "ik", "dynamics", "readout",   # training/grounding machinery
})

#: Citations of a source *tree* path ending in .py.
_PATH_RE = re.compile(r"Marionette-(?:ciasc|fkjepa)[/`]?\s*[/`]?([\w/]+\.py)")
#: Dotted citations rooted at a known top-level source package.
#: Generic source-tree roots. Too coarse to be evidence that a citation
#: refers to a hashed module -- every dotted citation contains one -- so they
#: are stripped from both the candidate set and the scope set. Including them
#: made this check pass vacuously: removing a real source from the snapshot
#: scope still "matched" via its top-level package name.
_GENERIC_ROOTS = frozenset({"models", "judge", "eval", "scripts"})

#: Dotted citations need at least two components below the root, so a prose
#: reference to a method on a local object (``judge.predict_pose``) is not
#: mistaken for a module path.
_DOTTED_RE = re.compile(r"\b((?:models|judge|eval|scripts)(?:\.\w+){2,5})\b")


def _citations() -> list[frozenset[str]]:
    """One entry per citation: the set of names that could be its module.

    A filesystem citation (``judge/fk.py``) resolves to exactly one candidate.
    A dotted citation (``models.posendf.readout_v2.clamp_for_fk``) is ambiguous
    -- ``clamp_for_fk`` is a function, ``readout_v2`` the module -- and nothing
    in the text distinguishes them, since both are snake_case. Rather than
    guess, every component becomes a candidate and the citation counts as
    covered if *any* of them names a hashed module. That is the correct
    semantics anyway: the question this test asks is "does this citation refer
    to a source we hashed", not "which component is the module".
    """
    out: list[frozenset[str]] = []
    for py in (REPO / "src").rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="ignore")
        for m in _PATH_RE.finditer(text):
            out.append(frozenset({Path(m.group(1)).stem}))
        for m in _DOTTED_RE.finditer(text):
            parts = {p for p in m.group(1).split(".")
                     if p and p != "py" and p not in _GENERIC_ROOTS}
            if parts:
                out.append(frozenset(parts))
    return out


def _snapshot_paths() -> set[str]:
    """Exactly the paths enumerated in ``SOURCES_A`` / ``SOURCES_B``.

    Scraped from those two tuples specifically, not from every ``.py`` string
    in the module -- its docstrings cite source paths in prose, and matching
    those would make this check fail on documentation.
    """
    text = (REPO / "tools" / "source_snapshot.py").read_text()
    paths: set[str] = set()
    for const in ("SOURCES_A", "SOURCES_B"):
        block = re.search(rf"{const}.*?=\s*\((.*?)\)", text, re.S)
        if block:
            paths.update(re.findall(r'"([^"]+\.py)"', block.group(1)))
    return paths


def _snapshot_basenames() -> set[str]:
    """Names covered by ``SOURCES_A`` / ``SOURCES_B``.

    Includes each enumerated file's stem *and* its parent package directories.
    A citation like ``judge.predict_pose`` names a method on an object, not a
    module -- but ``judge/`` is a hashed package, so the citation does point
    into audited source and should count as covered. Treating package names as
    candidates is what lets the check stay strict about genuinely unhashed
    sources without flagging every prose reference to a class or method.
    """
    text = (REPO / "tools" / "source_snapshot.py").read_text()
    names: set[str] = set()
    for const in ("SOURCES_A", "SOURCES_B"):
        block = re.search(rf"{const}.*?=\s*\((.*?)\)", text, re.S)
        if block:
            for rel in re.findall(r'"([^"]+\.py)"', block.group(1)):
                p = Path(rel)
                names.add(p.stem)
                names.update(set(p.parent.parts) - _GENERIC_ROOTS)
    return names


def test_snapshot_scope_is_non_trivial():
    """Guard against a regex change making the check pass vacuously."""
    scope = _snapshot_basenames()
    assert len(scope) >= 15, f"snapshot scope looks truncated: {sorted(scope)}"


def test_every_cited_source_is_hashed_or_declared_unported():
    """No port claim may lack a SHA-256 without being an explicit exclusion."""
    known = _snapshot_basenames() | CITED_NOT_PORTED
    uncovered = sorted(
        {"/".join(sorted(c)) for c in _citations() if not (c & known)})
    assert not uncovered, (
        "these citations in src/ name no module that tools/source_snapshot.py "
        "hashes, and none is declared in CITED_NOT_PORTED, so their port "
        "status is unverifiable:\n  " + "\n  ".join(uncovered)
        + "\nEither add the source to SOURCES_A / SOURCES_B and re-run the "
          "snapshot, or add it to CITED_NOT_PORTED with a "
          "legacy_docs/PROVENANCE.md entry.")


def test_sources_sha256_is_not_stale():
    """The generated record must contain every enumerated path."""
    record = REPO / "provenance" / "sources.sha256"
    if not record.exists():
        pytest.skip("provenance/sources.sha256 not generated in this checkout")
    text = record.read_text()
    # File paths only -- _snapshot_basenames() also yields package directory
    # names, which are candidates for citation matching but are never rows in
    # the hash record.
    missing = sorted(p for p in _snapshot_paths() if p not in text)
    assert not missing, (
        f"sources.sha256 is stale -- enumerated but absent: {missing}. "
        "Re-run tools/source_snapshot.py.")
