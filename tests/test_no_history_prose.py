"""The shipped source describes what the code does, never what it used to do.

Comments and docstrings that narrate an earlier version, a prior repository or
a fixed defect are noise to a reader who only ever sees this version, so they
are a test failure rather than a review preference.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("src", "configs", "tests", "docs")

#: Phrases that only make sense if the reader knows a previous version.
BANNED = [
    r"\blegacy\b",
    r"\bdeprecated\b",
    r"\bsuperseded\b",
    r"\bpreviously\b",
    r"\bused to (?:be|have|keep|live|call|do)\b",
    r"\bbefore this (?:module|change|field|class|function)\b",
    r"\bthe old\b",
    r"\bearlier version\b",
    r"\bhistorically\b",
    r"\bprior round\b",
    r"\bported (?:verbatim|from)\b",
    r"\bdefect [A-Z]?\d+\b",
    r"\bMarionette\b",
    r"\bfkjepa\b",
    r"\blegacy_docs\b",
    r"\bbackward[- ]compat",
    r"\bno longer (?:matches|needs|exists)\b",
    r"\bthe prototype\b",
    r"\bthe source(?:'s)? (?:code|repo|benchmark)\b",
]

_PATTERN = re.compile("|".join(BANNED), re.IGNORECASE)


#: Top-level files that ship with the source and get the same treatment.
SOURCE_FILES = ("README.md", "data/README.md")


def _files():
    for name in SOURCE_FILES:
        path = ROOT / name
        if path.is_file():
            yield path
    for directory in SOURCE_DIRS:
        for path in sorted((ROOT / directory).rglob("*")):
            if path.suffix in {".py", ".yaml", ".yml", ".md"} and path.is_file():
                if path.name == Path(__file__).name:
                    continue
                yield path


@pytest.mark.parametrize("path", list(_files()), ids=lambda p: str(p.name))
def test_no_history_prose(path):
    hits = [
        f"{path.relative_to(ROOT)}:{i}: {line.strip()}"
        for i, line in enumerate(path.read_text().splitlines(), 1)
        if _PATTERN.search(line)
    ]
    assert not hits, (
        "history prose in shipped source -- describe the current behaviour "
        "instead:\n  " + "\n  ".join(hits))
