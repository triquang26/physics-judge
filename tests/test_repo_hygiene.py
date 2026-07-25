"""Repo hygiene, enforced as a test so it cannot rot.

These four checks guard against mistakes that are effectively permanent once
committed: a vendored multi-hundred-megabyte asset tree, a hardcoded path into
one developer's home directory, a dangling symlink into a deleted sibling
checkout, or a golden-fixture set that quietly grows until the repo is
unpleasant to clone.

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
