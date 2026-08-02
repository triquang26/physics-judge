"""Pins the dependency direction between ``kinescore`` subpackages.

Two rules, both learned the hard way:

* ``kinescore.bench`` must not import ``kinescore.cli``. It did once --
  ``bench/csv_export.py:485`` reached up into ``cli._suites`` to look up a
  metric suite, even though a suite is a benchmark concept, not a CLI one.
  Fixed by moving that lookup to ``bench/suites.py`` (``cli`` now imports
  *it*, like every other ``bench`` name it uses -- see that module's
  docstring). This test is what keeps the fix from quietly reverting.
* ``kinescore.core`` must not import ``kinescore.bench`` at runtime. ``core``
  is deliberately the lowest layer in this package (see
  ``core/__init__.py``'s module docstring); a live runtime import the other
  way would be circular. ``core/contracts.py`` already imports
  ``bench.cell``/``bench.config``/``bench.manifest`` under ``TYPE_CHECKING``
  for annotation purposes only -- that is the one tolerated exception below.

NOTE for whoever next owns ``kinescore.bench``/``core/contracts.py``: the
``TYPE_CHECKING`` exception is scoped to that one file. ``contracts.py``'s
own docstring already floats replacing it with a ``Protocol`` so ``core``
needs no ``bench`` reference at all, even under ``TYPE_CHECKING`` -- if that
happens, delete the exception below rather than widening it to "any file in
``core``".

Uses ``ast`` rather than ``grep`` so this only ever sees real, static
``import``/``from ... import ...`` statements -- not a dynamically built
module-path string (``kinescore.cli.main``'s own subcommand discovery does
exactly that via ``importlib.import_module(f"kinescore.cli.{stem}")``, which
is not a load-time dependency edge and must not trip this check).
"""
from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src" / "kinescore"

#: ``core/contracts.py``'s TYPE_CHECKING-only ``bench.*`` imports are
#: tolerated (see module docstring) -- this is the one and only exception.
_CORE_BENCH_TYPE_CHECKING_EXCEPTION = "src/kinescore/core/contracts.py"


def _kinescore_packages() -> list[str]:
    """Every ``kinescore.<pkg>`` with an ``__init__.py``, read off disk."""
    return sorted(p.name for p in _SRC.iterdir()
                 if p.is_dir() and (p / "__init__.py").is_file())


class _ImportVisitor(ast.NodeVisitor):
    """Collects ``(dotted_module, type_checking_guarded)`` for every import."""

    def __init__(self) -> None:
        self.imports: list[tuple[str, bool]] = []
        self._tc_depth = 0

    @staticmethod
    def _is_type_checking_test(test: ast.expr) -> bool:
        if isinstance(test, ast.Name):
            return test.id == "TYPE_CHECKING"
        if isinstance(test, ast.Attribute):
            return test.attr == "TYPE_CHECKING"
        return False

    def visit_If(self, node: ast.If) -> None:
        guarded = self._is_type_checking_test(node.test)
        if guarded:
            self._tc_depth += 1
        self.generic_visit(node)
        if guarded:
            self._tc_depth -= 1

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append((alias.name, self._tc_depth > 0))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:  # bare relative `from . import x` -- unused in this package
            self.imports.append((node.module, self._tc_depth > 0))


def _imports_of(path: Path) -> list[tuple[str, bool]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    visitor = _ImportVisitor()
    visitor.visit(tree)
    return visitor.imports


def _target_package(module: str) -> str | None:
    """``"kinescore.bench.foo"`` -> ``"bench"``; anything else -> ``None``."""
    parts = module.split(".")
    if len(parts) < 2 or parts[0] != "kinescore":
        return None
    return parts[1]


def _violations(from_package: str, to_package: str, *,
                allow_type_checking_in: frozenset[str] = frozenset()) -> list[str]:
    """Every ``path: module`` where a file under ``from_package`` imports ``to_package``."""
    problems = []
    for path in sorted((_SRC / from_package).rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for module, type_checking in _imports_of(path):
            if _target_package(module) != to_package:
                continue
            if type_checking and rel in allow_type_checking_in:
                continue
            problems.append(f"{rel}: import {module!r}"
                            + (" (TYPE_CHECKING)" if type_checking else ""))
    return problems


def test_kinescore_packages_are_found_on_disk():
    # Sanity check on the fixture itself: if this package's on-disk shape
    # changes so much that bench/cli/core stop existing, the two tests below
    # would silently pass by scanning nothing -- this keeps them honest.
    packages = _kinescore_packages()
    for expected in ("bench", "cli", "core"):
        assert expected in packages


def test_bench_does_not_import_cli():
    violations = _violations("bench", "cli")
    assert not violations, (
        "kinescore.bench must not import kinescore.cli (dependency runs the "
        "other way everywhere else in this package):\n  "
        + "\n  ".join(violations))


def test_core_does_not_import_bench_at_runtime():
    violations = _violations(
        "core", "bench",
        allow_type_checking_in=frozenset({_CORE_BENCH_TYPE_CHECKING_EXCEPTION}))
    assert not violations, (
        "kinescore.core must not import kinescore.bench at runtime (a "
        "TYPE_CHECKING-only import is tolerated only in "
        f"{_CORE_BENCH_TYPE_CHECKING_EXCEPTION}):\n  "
        + "\n  ".join(violations))
