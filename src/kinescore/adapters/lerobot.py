"""Episodes from a LeRobot v2 dataset directory.

Layout, with the task level present only in corpora that group by task::

    <root>/<split>/[<task>/]meta/info.json
    <root>/<split>/[<task>/]data/chunk-NNN/episode_NNNNNN.parquet
    <root>/<split>/[<task>/]videos/chunk-NNN/<camera_key>/episode_NNNNNN.mp4

Cameras are stored one file each; `TrainSource.cameras` declares which become
panels and in what order. One entry may list alternatives as ``a|b``: corpora
that merged two collection runs name the same viewpoint two ways, and the
first name present is taken.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from kinescore.adapters.base import RawEpisode, SkippedEpisode, register_adapter

if TYPE_CHECKING:
    from kinescore.registry.cells import TrainSource

__all__ = ["LeRobotAdapter"]

_VIDEO_PREFIX = "observation.images."


def _datasets(root: Path) -> Iterator[tuple[str, str, Path]]:
    """``(split, task, dataset_dir)`` for every directory holding meta/info.json."""
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if (split_dir / "meta" / "info.json").exists():
            yield split_dir.name, "", split_dir
            continue
        for task_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if (task_dir / "meta" / "info.json").exists():
                yield split_dir.name, task_dir.name, task_dir


def _resolve(cameras: tuple[str, ...], available: list[str]
             ) -> tuple[list[str], str]:
    """Source key per panel, taking the first alternative present."""
    out: list[str] = []
    for spec in cameras:
        for name in spec.split("|"):
            if name in available:
                out.append(name)
                break
        else:
            return [], spec
    return out, ""


def _column(table: Any, field: str) -> np.ndarray | None:
    """``field`` as ``[T, D]``, or ``None`` if the table has no such column."""
    if field not in table.column_names:
        return None
    rows = table.column(field).to_pylist()
    if not rows:
        return None
    return np.asarray(rows, dtype=np.float32).reshape(len(rows), -1)


class LeRobotAdapter:
    """Reads a LeRobot v2 corpus, one episode per parquet file."""

    SOURCE_ID = "lerobot"

    def episodes(self, source: TrainSource
                 ) -> Iterator[RawEpisode | SkippedEpisode]:
        import pyarrow.parquet as pq

        root = Path(source.root)
        if not root.is_dir():
            raise ValueError(f"corpus root does not exist: {root}")
        if not source.cameras:
            raise ValueError(
                f"adapter {self.SOURCE_ID!r} needs `cameras:`; {root} stores "
                f"one file per camera and panel order is not recoverable from "
                f"the tree")

        for split, task, dataset in _datasets(root):
            info = json.loads((dataset / "meta" / "info.json").read_text())
            fps = float(info.get("fps") or 0.0)
            available = sorted(
                k[len(_VIDEO_PREFIX):] for k in (info.get("features") or {})
                if k.startswith(_VIDEO_PREFIX))
            prefix = "__".join(p for p in (split, task) if p)

            cameras, missing = _resolve(source.cameras, available)
            if missing:
                yield SkippedEpisode(
                    prefix, f"camera {missing!r} not in this dataset "
                            f"(has {available})", str(dataset))
                continue

            for parquet in sorted(dataset.glob("data/chunk-*/episode_*.parquet")):
                stem = parquet.stem
                episode_id = f"{prefix}__{stem}" if prefix else stem
                chunk = parquet.parent.name
                try:
                    table = pq.read_table(parquet)
                except Exception as exc:  # noqa: BLE001 -- reported, not raised
                    yield SkippedEpisode(episode_id, f"parquet unreadable: {exc}",
                                         str(parquet))
                    continue

                state = _column(table, source.joint_field)
                if state is None:
                    yield SkippedEpisode(
                        episode_id,
                        f"no column {source.joint_field!r} (has "
                        f"{table.column_names})", str(parquet))
                    continue

                cols = source.joint_columns
                if cols and max(cols) >= state.shape[1]:
                    yield SkippedEpisode(
                        episode_id,
                        f"joint_columns reach index {max(cols)} but "
                        f"{source.joint_field!r} has {state.shape[1]} columns",
                        str(parquet))
                    continue
                joints = state[:, list(cols)] if cols else state

                gripper = None
                if source.gripper_column is not None:
                    if source.gripper_column >= state.shape[1]:
                        yield SkippedEpisode(
                            episode_id,
                            f"gripper_column {source.gripper_column} is past "
                            f"the {state.shape[1]} columns of "
                            f"{source.joint_field!r}", str(parquet))
                        continue
                    gripper = state[:, source.gripper_column]
                elif source.gripper_field:
                    grip = _column(table, source.gripper_field)
                    if grip is not None:
                        gripper = grip[:, 0]

                views = {
                    cam: str(dataset / "videos" / chunk / f"{_VIDEO_PREFIX}{cam}"
                             / f"{stem}.mp4")
                    for cam in cameras
                }
                absent = [c for c, p in views.items() if not Path(p).exists()]
                if absent:
                    yield SkippedEpisode(
                        episode_id, f"no video for camera(s) {absent}",
                        str(parquet))
                    continue

                yield RawEpisode(
                    episode_id=episode_id,
                    joints=joints,
                    gripper=gripper,
                    fps=fps,
                    scene_key=prefix or split,
                    source_path=str(parquet),
                    views=views,
                )


register_adapter(LeRobotAdapter.SOURCE_ID, LeRobotAdapter)
