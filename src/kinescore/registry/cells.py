"""``cell_id -> everything a run needs``, read from ``configs/cells.yaml``.

A **reader** is one trained head, keyed by ``<robot>.<corpus>.<view_id>``: all
three change the weights. The robot follows from the corpus and is written
anyway, so swapping one without the other fails here rather than training
against the wrong forward kinematics.

A **cell** is one scored unit, keyed by ``<embodiment>.<view_id>.<model>``. It
names a reader and the clips it selects. ``method`` and ``split`` partition
clips *inside* a cell -- neither changes robot, packing or reader -- so they are
reported as sub-partitions rather than multiplied into the cell id.

Every path a run touches -- cache, checkpoint, train tree, score tree, output --
derives from these two ids through this module, so a training run and a scoring
run cannot mean different things by the same name.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kinescore.paths import env_path
from kinescore.registry.views import DEFAULT_VIEWS_PATH, ViewSpec, load_views

__all__ = [
    "TrainSource", "ReaderSpec", "CellSpec", "Registry", "load_registry",
    "DEFAULT_CELLS_PATH", "DEFAULT_ROBOTS_PATH", "SCENE_KEY_MODES",
]

#: Accepted :attr:`TrainSource.scene_key` values.
SCENE_KEY_MODES = frozenset({"prefix", "episode"})

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
DEFAULT_CELLS_PATH = _CONFIG_DIR / "cells.yaml"
DEFAULT_ROBOTS_PATH = _CONFIG_DIR / "robots.yaml"


def _expand(value: str) -> str:
    """Expand ``${KINESCORE_*}`` in a configured path.

    Raises
    ------
    kinescore.paths.MissingPathError
        If a referenced variable is unset -- naming it, rather than resolving
        to a path with an empty component.
    """
    out = value
    for key in ("KINESCORE_DATA_ROOT", "KINESCORE_CACHE_DIR",
                "KINESCORE_CKPT_DIR", "KINESCORE_ASSETS"):
        token = "${" + key + "}"
        if token in out:
            out = out.replace(token, str(env_path(key)))
    return os.path.expanduser(out)


@dataclass(frozen=True)
class TrainSource:
    """Where a reader's supervision comes from, and how to read its state.

    Attributes
    ----------
    corpus:
        Short name of the corpus, and the middle field of the reader id.
    adapter:
        Adapter id, resolved through :mod:`kinescore.adapters`.
    root:
        Corpus root, with ``${KINESCORE_*}`` already expanded.
    cameras:
        Source camera keys, in the order they become panels. Declared, not
        discovered: alphabetical order is not panel order. An entry may list
        alternatives as ``a|b``, for a corpus that names one viewpoint two
        ways.
    joint_field:
        Key in the source's own metadata holding the joint array.
    joint_columns:
        Which of that array's columns are the robot's joints, in the robot's
        canonical order. Empty means every column, in order.
    gripper_column:
        Column of ``joint_field`` holding gripper opening, or ``None``.
    gripper_field:
        A separate field holding gripper opening, for corpora that store it
        outside ``joint_field``. Ignored when ``gripper_column`` is set.
    scene_key:
        How an episode's scene is derived for the scene-disjoint split.
        ``"prefix"`` takes the episode id up to its last ``__``, the
        ``<scene>__<index>`` convention this benchmark's corpora name episodes
        with. ``"episode"`` makes every episode its own one-episode scene, for
        a corpus whose ids carry no scene structure to recover; the split is
        then a plain stratified sample over episodes and claims no scene
        disjointness.
    """

    corpus: str
    adapter: str
    root: str
    cameras: tuple[str, ...] = ()
    joint_field: str = "observation.state"
    joint_columns: tuple[int, ...] = ()
    gripper_column: int | None = None
    gripper_field: str = ""
    scene_key: str = "prefix"

    def __post_init__(self) -> None:
        if self.scene_key not in SCENE_KEY_MODES:
            raise ValueError(
                f"scene_key must be one of {sorted(SCENE_KEY_MODES)}, got "
                f"{self.scene_key!r}")
        if len(set(self.cameras)) != len(self.cameras):
            raise ValueError(f"cameras repeat a key: {list(self.cameras)}")


@dataclass(frozen=True)
class ReaderSpec:
    """One trained head: a robot seen through one packing.

    Attributes
    ----------
    reader_id:
        ``<robot>.<corpus>.<view_id>``.
    robot:
        Robot name, resolved through :func:`kinescore.robots.get_robot`.
    view:
        The packing, from ``views.yaml``.
    train:
        Its supervision, or ``None`` when no corpus supplies one.
    status:
        Empty when the reader is trainable; otherwise why it is not.
    """

    reader_id: str
    robot: str
    view: ViewSpec
    train: TrainSource | None = None
    status: str = ""

    @property
    def corpus(self) -> str:
        """The corpus this head is trained on, or ``""`` when it has none."""
        return self.train.corpus if self.train is not None else ""

    @property
    def trainable(self) -> bool:
        return self.train is not None and not self.status

    @property
    def checkpoint_path(self) -> Path:
        """``$KINESCORE_CKPT_DIR/<reader_id>.pt``."""
        return env_path("KINESCORE_CKPT_DIR") / f"{self.reader_id}.pt"

    @property
    def cache_dir(self) -> Path:
        """``$KINESCORE_CACHE_DIR/<reader_id>``, holding ``{split}/{ep}.pt``."""
        return env_path("KINESCORE_CACHE_DIR") / self.reader_id

    @property
    def train_tree(self) -> Path:
        """``$KINESCORE_DATA_ROOT/trees/<reader_id>``, rebuilt by ``data``."""
        return env_path("KINESCORE_DATA_ROOT") / "trees" / self.reader_id


@dataclass(frozen=True)
class CellSpec:
    """One scored unit.

    Attributes
    ----------
    cell_id:
        ``<embodiment>.<view_id>.<model>``.
    embodiment, view_id, model:
        The three axes the id is built from.
    reader:
        The head that reads this cell's clips.
    select:
        Catalog fields a clip must match to belong to this cell.
    status:
        Empty when the cell is scorable; otherwise why it is not, which is
        the reader's status when the reader has one.
    """

    cell_id: str
    embodiment: str
    view_id: str
    model: str
    reader: ReaderSpec
    select: dict[str, str]
    status: str = ""

    @property
    def view(self) -> ViewSpec:
        return self.reader.view

    @property
    def robot(self) -> str:
        return self.reader.robot

    @property
    def scorable(self) -> bool:
        return not self.status

    @property
    def output_dir(self) -> Path:
        """``$KINESCORE_OUTPUT_DIR/<cell_id>``."""
        from kinescore.paths import output_dir
        return output_dir() / self.cell_id


@dataclass(frozen=True)
class Registry:
    """Everything the four config files declare, cross-checked.

    Attributes
    ----------
    views, robots, readers, cells:
        Keyed by ``view_id`` / robot name / ``reader_id`` / ``cell_id``.
    sources:
        Absolute paths of the files this was read from, for the run manifest.
    """

    views: dict[str, ViewSpec]
    robots: dict[str, dict[str, Any]]
    readers: dict[str, ReaderSpec]
    cells: dict[str, CellSpec]
    sources: tuple[str, ...]

    def cell(self, cell_id: str) -> CellSpec:
        """Look up a cell, listing every id on a miss."""
        try:
            return self.cells[cell_id]
        except KeyError:
            raise KeyError(
                f"unknown cell {cell_id!r}; declared cells: "
                f"{sorted(self.cells)}") from None

    def reader(self, reader_id: str) -> ReaderSpec:
        """Look up a reader, listing every id on a miss."""
        try:
            return self.readers[reader_id]
        except KeyError:
            raise KeyError(
                f"unknown reader {reader_id!r}; declared readers: "
                f"{sorted(self.readers)}") from None

    def cells_for_reader(self, reader_id: str) -> tuple[CellSpec, ...]:
        """Every cell scored by one reader."""
        return tuple(c for c in self.cells.values()
                     if c.reader.reader_id == reader_id)


def _train_from_entry(reader_id: str, entry: dict[str, Any]) -> TrainSource:
    unknown = set(entry) - {"corpus", "adapter", "root", "cameras",
                            "joint_field", "joint_columns", "gripper_column",
                            "gripper_field", "scene_key"}
    if unknown:
        raise ValueError(
            f"reader {reader_id!r}: unknown train key(s) {sorted(unknown)}")
    return TrainSource(
        corpus=str(entry["corpus"]),
        adapter=str(entry["adapter"]),
        root=_expand(str(entry["root"])),
        cameras=tuple(str(c) for c in entry.get("cameras", ())),
        joint_field=str(entry.get("joint_field", "observation.state")),
        joint_columns=tuple(int(i) for i in entry.get("joint_columns", ())),
        gripper_column=(None if entry.get("gripper_column") is None
                        else int(entry["gripper_column"])),
        gripper_field=str(entry.get("gripper_field", "")),
        scene_key=str(entry.get("scene_key", "prefix")),
    )


def _reader_from_entry(reader_id: str, entry: dict[str, Any],
                       views: dict[str, ViewSpec],
                       robots: dict[str, dict[str, Any]]) -> ReaderSpec:
    unknown = set(entry) - {"robot", "view", "corpus", "train", "status"}
    if unknown:
        raise ValueError(
            f"reader {reader_id!r}: unknown key(s) {sorted(unknown)}")
    robot = str(entry["robot"])
    view_id = str(entry["view"])
    if robot not in robots:
        raise ValueError(
            f"reader {reader_id!r} names robot {robot!r}, which robots.yaml "
            f"does not declare: {sorted(robots)}")
    if view_id not in views:
        raise ValueError(
            f"reader {reader_id!r} names view {view_id!r}, which views.yaml "
            f"does not declare: {sorted(views)}")
    train_entry = entry.get("train")
    train = (None if train_entry is None
             else _train_from_entry(reader_id, train_entry))
    corpus = train.corpus if train is not None else str(entry.get("corpus", ""))
    if not corpus:
        raise ValueError(
            f"reader {reader_id!r} names no corpus; a reader with no train "
            f"source still needs `corpus:` so its id can be checked")
    expected_id = f"{robot}.{corpus}.{view_id}"
    if reader_id != expected_id:
        raise ValueError(
            f"reader id {reader_id!r} must be <robot>.<corpus>.<view_id>, i.e. "
            f"{expected_id!r}")
    return ReaderSpec(
        reader_id=reader_id, robot=robot, view=views[view_id], train=train,
        status=str(entry.get("status", "")),
    )


def _cell_from_entry(cell_id: str, entry: dict[str, Any],
                     readers: dict[str, ReaderSpec],
                     robots: dict[str, dict[str, Any]]) -> CellSpec:
    unknown = set(entry) - {"reader", "select", "status"}
    if unknown:
        raise ValueError(f"cell {cell_id!r}: unknown key(s) {sorted(unknown)}")
    parts = cell_id.split(".")
    if len(parts) != 3:
        raise ValueError(
            f"cell id {cell_id!r} must be <embodiment>.<view_id>.<model>")
    embodiment, view_id, model = parts
    reader_id = str(entry["reader"])
    if reader_id not in readers:
        raise ValueError(
            f"cell {cell_id!r} names reader {reader_id!r}, which is not "
            f"declared: {sorted(readers)}")
    reader = readers[reader_id]
    if reader.view.view_id != view_id:
        raise ValueError(
            f"cell {cell_id!r} is packed {view_id!r} but its reader "
            f"{reader_id!r} reads {reader.view.view_id!r}")
    declared = robots[reader.robot].get("embodiment")
    if declared != embodiment:
        raise ValueError(
            f"cell {cell_id!r} is embodiment {embodiment!r} but robot "
            f"{reader.robot!r} is declared {declared!r} in robots.yaml")
    select = {str(k): str(v) for k, v in (entry.get("select") or {}).items()}
    return CellSpec(
        cell_id=cell_id, embodiment=embodiment, view_id=view_id, model=model,
        reader=reader, select=select,
        status=str(entry.get("status", "")) or reader.status,
    )


def load_registry(cells_path: str | Path = DEFAULT_CELLS_PATH,
                  robots_path: str | Path = DEFAULT_ROBOTS_PATH,
                  views_path: str | Path = DEFAULT_VIEWS_PATH) -> Registry:
    """Read the three definition files and cross-check them.

    Every reference is resolved eagerly -- a cell naming an undeclared reader,
    a reader naming an undeclared view, or an embodiment that disagrees with
    ``robots.yaml`` fails here, not mid-run.
    """
    views = load_views(views_path)
    robots_doc = yaml.safe_load(Path(robots_path).read_text()) or {}
    robots = robots_doc.get("robots") or {}
    if not isinstance(robots, dict):
        raise ValueError(f"{robots_path}: `robots` must be a mapping")

    doc = yaml.safe_load(Path(cells_path).read_text()) or {}
    readers = {
        rid: _reader_from_entry(rid, entry, views, robots)
        for rid, entry in (doc.get("readers") or {}).items()
    }
    cells = {
        cid: _cell_from_entry(cid, entry, readers, robots)
        for cid, entry in (doc.get("cells") or {}).items()
    }
    return Registry(
        views=views, robots=robots, readers=readers, cells=cells,
        sources=(str(Path(views_path).resolve()),
                 str(Path(robots_path).resolve()),
                 str(Path(cells_path).resolve())),
    )
