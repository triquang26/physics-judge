"""Load ``configs/robot_map.yaml`` -- the (embodiment, generator) -> robot table.

See that file's header comment for WHY this indirection exists (one
embodiment directory, e.g. ``humanoid``, covers two different physical
robots depending on generator). This module only parses and validates the
table; :mod:`kinescore.bench.matrix` is what actually uses it to resolve a
cell's ``robot`` from its on-disk ``embodiment``.

Kept as its own tiny module (not folded into ``bench.config``) because two
independent things read it: ``bench.matrix`` (resolving a scoring cell) and
``bench.layout``/``bench.ingest`` (walking the raw HF tree, which has no
``benchmark.yaml`` in scope at all -- ``kinescore data ingest`` runs before
any run config is chosen).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["RobotMapError", "RobotEntry", "RobotMap", "load_robot_map", "parse_robot_map"]


class RobotMapError(ValueError):
    """``robot_map.yaml`` (or an equivalent dict) failed validation."""


@dataclass(frozen=True)
class RobotEntry:
    """One robot's row: the embodiment directory it lives under, and which
    generators actually produce its clips."""

    embodiment: str
    generators: tuple[str, ...]


@dataclass(frozen=True)
class RobotMap:
    """The fully validated robot table."""

    robots: Mapping[str, RobotEntry]

    def resolve(self, *, embodiment: str, generator: str) -> str | None:
        """The robot whose (embodiment, generators) matches, or ``None``.

        ``None`` means no robot in this table claims that (embodiment,
        generator) pair -- a real, expected outcome (e.g. embodiment=humanoid,
        generator=ctrlworld resolves to ``airbot_mmk2``; embodiment=humanoid,
        generator=cosmos -- not a real generator -- resolves to ``None``),
        never itself an error; the caller decides what an unresolved cell means.
        """
        for robot, entry in self.robots.items():
            if entry.embodiment == embodiment and generator in entry.generators:
                return robot
        return None

    def embodiment_of(self, robot: str) -> str:
        """The embodiment directory ``robot``'s clips live under.

        Raises
        ------
        KeyError
            If ``robot`` is not in this table.
        """
        return self.robots[robot].embodiment

    def generators_of(self, robot: str) -> tuple[str, ...]:
        """The generators that actually produce ``robot``'s clips.

        Raises
        ------
        KeyError
            If ``robot`` is not in this table.
        """
        return self.robots[robot].generators


def parse_robot_map(raw: Any) -> RobotMap:
    """Validate an already-parsed ``robot_map.yaml`` mapping into a :class:`RobotMap`."""
    if not (isinstance(raw, Mapping) and isinstance(raw.get("robots"), Mapping)
            and raw["robots"]):
        raise RobotMapError(
            f"robot_map root must be a mapping with a non-empty 'robots' key, "
            f"got {raw!r}")
    out: dict[str, RobotEntry] = {}
    for robot, entry in raw["robots"].items():
        if not (isinstance(entry, Mapping) and {"embodiment", "generators"} <= set(entry)):
            raise RobotMapError(
                f"robots.{robot}: must have keys embodiment, generators, "
                f"got {sorted(entry) if isinstance(entry, Mapping) else entry!r}")
        embodiment = entry["embodiment"]
        if not (isinstance(embodiment, str) and embodiment):
            raise RobotMapError(
                f"robots.{robot}.embodiment must be a non-empty string, "
                f"got {embodiment!r}")
        generators = entry["generators"]
        if not (isinstance(generators, list) and generators
                and all(isinstance(g, str) and g for g in generators)):
            raise RobotMapError(
                f"robots.{robot}.generators must be a non-empty list of "
                f"non-empty strings, got {generators!r}")
        out[robot] = RobotEntry(embodiment=embodiment, generators=tuple(generators))
    return RobotMap(robots=out)


def load_robot_map(path: str | Path) -> RobotMap:
    """Read and validate a ``robot_map.yaml`` file."""
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)
    return parse_robot_map(raw)
