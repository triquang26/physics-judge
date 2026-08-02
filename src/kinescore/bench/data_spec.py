"""Load ``configs/data_spec.yaml`` -- the per-generator file/format contract.

Nothing here is a Python literal: resolution, fps, which of the three
on-disk shapes (``episode_dir``/``task_episode``/``flat_or_dir``) a generator
uses, and the exclusion globs are all read from the YAML file so a dataset
format change is a config edit. See that file's header comment for what each
field means; this module only parses and validates the shape.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DataSpecError", "GeneratorSpec", "DataSpec", "load_data_spec", "parse_data_spec",
]

_KNOWN_SHAPES = frozenset({"episode_dir", "task_episode", "flat_or_dir"})


class DataSpecError(ValueError):
    """``data_spec.yaml`` (or an equivalent dict) failed validation."""


@dataclass(frozen=True)
class GeneratorSpec:
    """One generator's file-naming + expected-format contract.

    Fields are a superset across the three ``shape`` values -- only the ones
    relevant to a generator's ``shape`` are populated by
    :func:`parse_data_spec`; the rest stay ``None``. Reading the wrong field
    for a given ``shape`` is a caller bug, not something this dataclass
    guards against (mirrors ``kinescore.bench.config.SourceConfig``, which
    has the same "some fields only make sense for some generators" shape).
    """

    generator: str
    shape: str
    has_iter_level: bool
    width: int
    height: int
    n_views: int
    has_ground_truth: bool
    fps_tolerant: bool
    fps: float | None = None
    fps_by_robot: Mapping[str, float] | None = None
    pred_filename: str | None = None
    gt_filename: str | None = None
    pred_glob: str | None = None
    flat_pred_glob: str | None = None
    flat_gt_glob: str | None = None
    dir_pred_filename: str | None = None
    dir_gt_filename: str | None = None

    def resolve_fps(self, *, robot: str | None) -> float | None:
        """This generator's expected fps, for ``robot`` if fps is per-robot."""
        if self.fps is not None:
            return self.fps
        if self.fps_by_robot is not None and robot is not None:
            return self.fps_by_robot.get(robot)
        return None


@dataclass(frozen=True)
class DataSpec:
    """A fully validated ``data_spec.yaml``."""

    generators: Mapping[str, GeneratorSpec]
    exclude_globs: tuple[str, ...]
    robots: Mapping[str, Mapping[str, Any]]


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise DataSpecError(msg)


def _parse_generator(name: str, entry: Any) -> GeneratorSpec:
    _require(isinstance(entry, Mapping), f"generators.{name}: must be a mapping")
    shape = entry.get("shape")
    _require(shape in _KNOWN_SHAPES,
             f"generators.{name}.shape={shape!r} must be one of {sorted(_KNOWN_SHAPES)}")
    for key in ("width", "height", "n_views"):
        _require(isinstance(entry.get(key), int) and entry[key] > 0,
                 f"generators.{name}.{key} must be a positive integer")
    _require(isinstance(entry.get("has_iter_level"), bool),
             f"generators.{name}.has_iter_level must be a bool")
    _require(isinstance(entry.get("has_ground_truth"), bool),
             f"generators.{name}.has_ground_truth must be a bool")
    _require(isinstance(entry.get("fps_tolerant"), bool),
             f"generators.{name}.fps_tolerant must be a bool")

    fps = entry.get("fps")
    fps_by_robot = entry.get("fps_by_robot")
    _require((fps is None) != (fps_by_robot is None),
             f"generators.{name}: exactly one of fps / fps_by_robot must be set")
    if fps is not None:
        _require(isinstance(fps, (int, float)) and fps > 0,
                 f"generators.{name}.fps must be a positive number")
        fps = float(fps)
    if fps_by_robot is not None:
        _require(isinstance(fps_by_robot, Mapping) and fps_by_robot,
                 f"generators.{name}.fps_by_robot must be a non-empty mapping")
        fps_by_robot = {str(r): float(v) for r, v in fps_by_robot.items()}

    if shape == "episode_dir":
        _require(isinstance(entry.get("pred_filename"), str),
                 f"generators.{name}.pred_filename must be a string (shape=episode_dir)")
    elif shape == "task_episode":
        _require(isinstance(entry.get("pred_glob"), str),
                 f"generators.{name}.pred_glob must be a string (shape=task_episode)")
    elif shape == "flat_or_dir":
        for key in ("flat_pred_glob", "flat_gt_glob", "dir_pred_filename", "dir_gt_filename"):
            _require(isinstance(entry.get(key), str),
                     f"generators.{name}.{key} must be a string (shape=flat_or_dir)")

    return GeneratorSpec(
        generator=name, shape=shape,
        has_iter_level=entry["has_iter_level"], width=entry["width"],
        height=entry["height"], n_views=entry["n_views"],
        has_ground_truth=entry["has_ground_truth"],
        fps_tolerant=entry["fps_tolerant"], fps=fps, fps_by_robot=fps_by_robot,
        pred_filename=entry.get("pred_filename"), gt_filename=entry.get("gt_filename"),
        pred_glob=entry.get("pred_glob"), flat_pred_glob=entry.get("flat_pred_glob"),
        flat_gt_glob=entry.get("flat_gt_glob"), dir_pred_filename=entry.get("dir_pred_filename"),
        dir_gt_filename=entry.get("dir_gt_filename"))


def parse_data_spec(raw: Any) -> DataSpec:
    """Validate an already-parsed ``data_spec.yaml`` mapping into a :class:`DataSpec`."""
    _require(isinstance(raw, Mapping), f"data_spec root must be a mapping, got {type(raw)}")
    gens_raw = raw.get("generators")
    _require(isinstance(gens_raw, Mapping) and gens_raw,
             "data_spec.generators must be a non-empty mapping")
    generators = {name: _parse_generator(name, entry) for name, entry in gens_raw.items()}

    excl = raw.get("exclude_globs", [])
    _require(isinstance(excl, Sequence) and not isinstance(excl, (str, bytes)),
             "data_spec.exclude_globs must be a list")
    _require(all(isinstance(g, str) and g for g in excl),
             "data_spec.exclude_globs entries must be non-empty strings")

    robots_raw = raw.get("robots", {})
    _require(isinstance(robots_raw, Mapping), "data_spec.robots must be a mapping")

    return DataSpec(generators=generators, exclude_globs=tuple(excl), robots=dict(robots_raw))


def load_data_spec(path: str | Path) -> DataSpec:
    """Read and validate a ``data_spec.yaml`` file."""
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)
    return parse_data_spec(raw)
