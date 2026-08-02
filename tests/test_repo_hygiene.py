"""Repo hygiene, enforced as a test so it cannot rot.

These checks guard against mistakes that are effectively permanent once
committed: a vendored multi-hundred-megabyte asset tree, a hardcoded path into
one developer's home directory, a dangling symlink into a deleted sibling
checkout, a golden-fixture set that quietly grows until the repo is
unpleasant to clone, an undocumented operator script, or a doc reorganisation
that strands a stale ``docs/*.md`` citation in a comment nothing else ever
re-checks.

All of them were real failures in the codebase this benchmark was extracted
from, which is why they are pinned here rather than left to review.

CPU-only, no torch, no network -- this module must stay importable in the
minimal test tier.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def hygiene():
    """Import ``tools/check_repo_hygiene.py`` without making ``tools`` a package."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "check_repo_hygiene", REPO / "tools" / "check_repo_hygiene.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_symlinks(hygiene):
    """A broken symlink in git is nearly invisible and breaks a fresh clone."""
    assert hygiene.check_no_symlinks(REPO) == []


def test_no_oversized_files(hygiene):
    """Assets and checkpoints are referenced via env vars, never vendored."""
    assert hygiene.check_file_sizes(REPO) == []


def test_no_absolute_paths(hygiene):
    """Nothing under ``src/`` may hardcode a host-specific path.

    The prior extraction of this code baked ``/mnt/.../<someone>/`` into its
    path defaults, so a fresh checkout resolved to directories that did not
    exist. kinescore routes every location through ``KINESCORE_*``.
    """
    assert hygiene.check_no_absolute_paths(REPO) == []


def test_golden_fixtures_within_budget(hygiene):
    """Golden fixtures pin behaviour; they must stay small enough to live in git."""
    assert hygiene.check_golden_budget(REPO) == []


def test_scripts_have_header(hygiene):
    """Every file in scripts/ must carry a What/Why/Input/Output header.

    scripts/ is a collection of ad-hoc operator runbooks copied in from
    wherever they were last run; a filename alone (e.g. `run_cache_camhigh_A.sh`)
    says nothing about what data it touches or why it exists. This is what
    caught that exact case: two scripts differed only in a two-letter suffix
    with no header explaining the difference until one was added.
    """
    assert hygiene.check_scripts_have_header(REPO) == []


def test_scripts_header_check_catches_missing_field(hygiene, tmp_path):
    """The check must actually fail a script missing a header field, not just
    pass real ones -- a checker that only ever returns [] is not a checker."""
    repo = tmp_path
    scripts_dir = repo / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "bad.sh").write_text(
        "#!/usr/bin/env bash\n"
        "# What: does a thing\n"
        "# Why: because\n"
        "# Input: none\n"
        "set -euo pipefail\n"
        "echo hi\n"
    )  # missing "Output:"
    problems = hygiene.check_scripts_have_header(repo)
    assert len(problems) == 1
    assert "bad.sh" in problems[0]
    assert "Output" in problems[0]


def test_no_broken_doc_links(hygiene):
    """Every ``docs/*.md``/``legacy_docs/*.md`` citation under src/tests must
    resolve to a real file.

    A docs reorganisation (moving a file into ``legacy_docs/``, or renaming
    one) does not raise anywhere on its own -- a stale citation in a
    docstring or comment is not code, so nothing fails until a human notices.
    66 files carried exactly this kind of stale reference after one such
    move; this pins the fix and catches the next one.
    """
    assert hygiene.check_doc_links(REPO) == []


def test_doc_link_check_catches_a_broken_reference(hygiene, tmp_path):
    """The checker must actually fail a dangling reference, not just pass a
    real tree -- a checker that only ever returns [] is not a checker.

    The doc-name fragments below are built via string concatenation rather
    than written out whole, so this test's own source never contains a
    contiguous ``docs/<name>.md``-shaped substring for :data:`DOC_LINK_RE` to
    match when the real ``check_doc_links(REPO)`` run (in
    :func:`test_no_broken_doc_links` above) scans this very file -- a false
    positive against the checker's own test fixtures, not a real broken
    reference.
    """
    repo = tmp_path
    (repo / "src").mkdir()
    (repo / "docs").mkdir()
    real_name = "RE" + "AL.md"
    ghost_name = "GHO" + "ST.md"
    (repo / "docs" / real_name).write_text("# real\n")
    (repo / "src" / "mod.py").write_text(
        f'"""See docs/{real_name} for the real one and docs/{ghost_name} '
        'for the one that was never created."""\n'
    )
    problems = hygiene.check_doc_links(repo)
    assert len(problems) == 1
    assert "mod.py" in problems[0]
    assert ghost_name in problems[0]


def test_doc_link_check_strips_anchor_before_checking_existence(hygiene, tmp_path):
    """``docs/ARCHITECTURE.md#adding-a-robot`` must be checked against
    ``docs/ARCHITECTURE.md`` (the anchor names a section, not a path)."""
    repo = tmp_path
    (repo / "src").mkdir()
    (repo / "docs").mkdir()
    (repo / "docs" / "ARCHITECTURE.md").write_text("# architecture\n### Adding a robot\n")
    (repo / "src" / "mod.py").write_text(
        '"""see docs/ARCHITECTURE.md#adding-a-robot for the protocol"""\n'
    )
    assert hygiene.check_doc_links(repo) == []


def test_scripts_header_check_passes_complete_header(hygiene, tmp_path):
    """A script with all four fields present, in any of the four scan-window
    lines, passes -- the counterpart to the missing-field test above."""
    repo = tmp_path
    scripts_dir = repo / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "good.sh").write_text(
        "#!/usr/bin/env bash\n"
        "# What: does a thing\n"
        "# Why: because\n"
        "# Input: none\n"
        "# Output: nothing\n"
        "set -euo pipefail\n"
        "echo hi\n"
    )
    assert hygiene.check_scripts_have_header(repo) == []
