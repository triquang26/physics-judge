"""Teleop episodes from a ctrlworld ``input/`` tree.

Layout::

    <root>/<subset>/episode_<name>/metadata.json
                                  /full_gt.mp4
                                  /view_0.mp4 ... view_{n-1}.mp4

``metadata.json`` carries ``fps`` and the logged state array under the key the
cell names (``states`` for the two-arm and humanoid corpora, ``joints`` for the
Franka one, whose ``states`` is a Cartesian pose rather than radians). Which of
that array's columns are joints is likewise the cell's declaration, because a
14-column ALOHA state interleaves two grippers among twelve arm joints and only
the cell knows the robot's canonical order.

``full_gt.mp4`` is the packed frame and is what this adapter reads: its frame
count matches ``num_frames`` and the state array, while the per-camera
``view_*.mp4`` files run one frame long. A tree without it falls back to the
per-camera files, which are then packed to the cell's view.
"""
from __future__ import annotations

import glob
import json
import os
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np

from kinescore.adapters.base import (
    RawEpisode,
    SkippedEpisode,
    register_adapter,
)

if TYPE_CHECKING:
    from kinescore.registry.cells import TrainSource

__all__ = ["CtrlWorldTeleopAdapter"]

_PACKED = "full_gt.mp4"
_VIEW_GLOB = "view_*.mp4"


def _scene_key(episode_dir: str) -> str:
    """Task identity: the episode name with its trailing index removed."""
    name = os.path.basename(episode_dir)
    name = name[len("episode_"):] if name.startswith("episode_") else name
    head = name.rsplit("__", 1)[0] if "__" in name else name
    return head or name


class CtrlWorldTeleopAdapter:
    """Reads ``<root>/<subset>/episode_*/`` directories."""

    SOURCE_ID = "ctrlworld_teleop"

    def episodes(self, source: TrainSource
                 ) -> Iterator[RawEpisode | SkippedEpisode]:
        """Yield one entry per episode directory under ``source.root``."""
        pattern = os.path.join(source.root, "*", "episode_*")
        for episode_dir in sorted(glob.glob(pattern)):
            if not os.path.isdir(episode_dir):
                continue
            subset = os.path.basename(os.path.dirname(episode_dir))
            name = os.path.basename(episode_dir)
            episode_id = f"{subset}_{name[len('episode_'):]}"

            meta_path = os.path.join(episode_dir, "metadata.json")
            try:
                meta = json.loads(open(meta_path).read())
            except (OSError, ValueError) as exc:
                yield SkippedEpisode(episode_id, f"metadata.json unreadable: {exc}",
                                     episode_dir)
                continue

            raw = meta.get(source.joint_field)
            if not raw:
                yield SkippedEpisode(
                    episode_id,
                    f"metadata.json has no {source.joint_field!r} array",
                    episode_dir)
                continue
            state = np.asarray(raw, dtype=np.float32)
            if state.ndim != 2:
                yield SkippedEpisode(
                    episode_id,
                    f"{source.joint_field!r} is {state.shape}, expected [T, J]",
                    episode_dir)
                continue

            cols = source.joint_columns or tuple(range(state.shape[1]))
            if max(cols) >= state.shape[1]:
                yield SkippedEpisode(
                    episode_id,
                    f"joint_columns reach index {max(cols)} but "
                    f"{source.joint_field!r} has {state.shape[1]} columns",
                    episode_dir)
                continue
            joints = state[:, list(cols)]
            gripper = (None if source.gripper_column is None
                       else state[:, source.gripper_column])

            packed = os.path.join(episode_dir, _PACKED)
            views: dict[str, str] = {}
            if not os.path.exists(packed):
                packed = ""
                views = {
                    os.path.splitext(os.path.basename(v))[0]: v
                    for v in sorted(glob.glob(os.path.join(episode_dir, _VIEW_GLOB)))
                }
                if not views:
                    yield SkippedEpisode(
                        episode_id, f"neither {_PACKED} nor {_VIEW_GLOB}",
                        episode_dir)
                    continue

            yield RawEpisode(
                episode_id=episode_id,
                joints=joints,
                gripper=gripper,
                fps=float(meta.get("fps", 0.0)),
                scene_key=_scene_key(episode_dir),
                source_path=episode_dir,
                views=views,
                packed=packed or None,
            )


register_adapter(CtrlWorldTeleopAdapter.SOURCE_ID, CtrlWorldTeleopAdapter)
