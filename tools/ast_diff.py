#!/usr/bin/env python
"""Prove two Python files agree on function BEHAVIOUR, ignoring prose.

Why not just ``diff -u``
-------------------------
A byte diff treats a renamed variable, a re-wrapped docstring, or a
reformatted comment as a change, which is noise for the one question this
port actually needs answered: **did the executable body move verbatim?**
``judge/fk.py`` and ``models/kinematics/fk.py`` are a good example -- they
happen to be byte-identical (see the self-test below), but the more
interesting case is ``judge/physics.py`` vs
``models/evaluation/physics_metrics.py``, whose module docstrings and imports
differ (the former folds ``KinematicConsistency`` in locally) while the method
bodies are the same pose math. This tool is what turns "the docstring says
byte-identical" into a checked claim instead of an assertion to trust.

Method
------
Parse both files with :mod:`ast`, collect every ``FunctionDef`` /
``AsyncFunctionDef`` keyed by its dotted qualified name (``Class.method`` for
methods, bare name at module scope), strip each function's docstring (the
leading ``Expr(Constant(str))`` statement, if present -- this is prose, not
behaviour), and render what remains with :func:`ast.unparse`. ``ast.unparse``
already normalises whitespace/quote style/parenthesisation, so two functions
compare equal here iff they'd produce the same bytecode-relevant AST modulo
docstrings and comments (which the parser never sees in the first place).

This is a coarser tool than it sounds: it does NOT catch a change to a
function's decorators, its default-argument *values* it can't statically
resolve identically-but-differently-spelled (e.g. ``0.2`` vs ``2e-1`` DO
compare different -- ``ast.unparse`` renders literals as written, not
evaluated), or anything happening via monkeypatching / metaclasses. For this
project's purpose -- catching an accidental behavioural edit during a
copy-paste port -- that's the right level of strictness: too coarse and it
would pass through a real edit; exact-source comparison is already `diff`'s
job and is what :func:`hash_file` in ``source_snapshot.py`` does at the whole-
file level.
"""
from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = ["FunctionDiff", "diff_files", "strip_docstring", "qualified_functions"]


@dataclass(frozen=True)
class FunctionDiff:
    """Comparison result for one qualified function name."""

    qualname: str
    in_a: bool
    in_b: bool
    bodies_equal: bool  # meaningless (False) unless in_a and in_b


def strip_docstring(node: ast.AST) -> ast.AST:
    """Return ``node`` with its leading docstring statement removed, if any.

    Mutates a **copy** of ``node.body`` (not the parsed tree in place) so
    repeated calls on the same tree are safe.
    """
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr):
        val = body[0].value
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            body = body[1:]
    stripped = ast.parse("")  # throwaway container; only .body is read below
    stripped.body = body
    return stripped


def _qualname(stack: list[str], name: str) -> str:
    return ".".join([*stack, name]) if stack else name


def qualified_functions(tree: ast.Module) -> dict[str, ast.AST]:
    """Map dotted qualified name -> function/async-function node.

    Walks nested classes (``Outer.Inner.method``) but does NOT descend into
    function bodies looking for further ``def`` (a closure defined inside a
    function is part of that function's body, and is compared as such, not
    separately).
    """
    out: dict[str, ast.AST] = {}

    def visit(node: ast.AST, stack: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[_qualname(stack, child.name)] = child
                # do not descend into the function body
            elif isinstance(child, ast.ClassDef):
                visit(child, [*stack, child.name])
            elif isinstance(child, (ast.Module,)):
                visit(child, stack)
            # anything else at module scope (if/try/etc.) is not descended
            # into on purpose: a function hidden inside a conditional import
            # guard is a code smell this tool should surface as "not found",
            # not silently paper over.

    visit(tree, [])
    return out


def _normalized_body(func: ast.AST) -> str:
    stripped = strip_docstring(func)
    return "\n".join(ast.unparse(stmt) for stmt in stripped.body)


def diff_files(path_a: Path, path_b: Path) -> list[FunctionDiff]:
    """Compare every function in ``path_a`` and ``path_b`` by qualified name.

    Returns one :class:`FunctionDiff` per name that appears in the union of
    both files, sorted by qualname.
    """
    tree_a = ast.parse(path_a.read_text(), filename=str(path_a))
    tree_b = ast.parse(path_b.read_text(), filename=str(path_b))
    funcs_a = qualified_functions(tree_a)
    funcs_b = qualified_functions(tree_b)

    results = []
    for qualname in sorted(set(funcs_a) | set(funcs_b)):
        in_a, in_b = qualname in funcs_a, qualname in funcs_b
        bodies_equal = (
            in_a and in_b
            and _normalized_body(funcs_a[qualname]) == _normalized_body(funcs_b[qualname])
        )
        results.append(FunctionDiff(qualname, in_a, in_b, bodies_equal))
    return results


def _print_report(path_a: Path, path_b: Path, results: list[FunctionDiff]) -> bool:
    """Print a per-function table; return True iff every shared function's
    body matched (the ``all clear`` condition callers should exit 0 on)."""
    only_a = [r for r in results if r.in_a and not r.in_b]
    only_b = [r for r in results if r.in_b and not r.in_a]
    shared = [r for r in results if r.in_a and r.in_b]
    differing = [r for r in shared if not r.bodies_equal]

    print(f"{path_a}\n  vs\n{path_b}")
    print(f"  {len(shared)} shared functions, {len(only_a)} only in A, "
          f"{len(only_b)} only in B")
    for r in only_a:
        print(f"  ONLY-A   {r.qualname}")
    for r in only_b:
        print(f"  ONLY-B   {r.qualname}")
    for r in differing:
        print(f"  DIFFERS  {r.qualname}")
    if not differing:
        print(f"  body-identical: all {len(shared)} shared functions match "
              f"(docstrings/comments/imports may still differ)")
    return not differing


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("file_a", type=Path)
    p.add_argument("file_b", type=Path)
    p.add_argument("--self-test", action="store_true",
                    help="ignore file_a/file_b; run the built-in self-test "
                         "(A judge/fk.py vs B models/kinematics/fk.py, "
                         "passed as file_a/file_b anyway) and assert zero "
                         "body differences")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    for p in (args.file_a, args.file_b):
        if not p.is_file():
            print(f"error: {p} is not a file", file=sys.stderr)
            return 2

    results = diff_files(args.file_a, args.file_b)
    clean = _print_report(args.file_a, args.file_b, results)

    if args.self_test:
        # This tool's own self-test: judge/fk.py and models/kinematics/fk.py
        # are on-disk byte-identical (verified separately with `diff`), so
        # this must report zero differences AND zero only-A/only-B entries,
        # or ast_diff.py itself has a bug worth distrusting.
        only = [r for r in results if not (r.in_a and r.in_b)]
        if not clean or only:
            print("SELF-TEST FAILED: expected zero differences on files "
                  "known to be byte-identical", file=sys.stderr)
            return 1
        print("SELF-TEST PASSED")
        return 0

    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
