"""Episodes from a tree that is already in the canonical train shape.

Layout::

    <root>/videos/{train,val}/<episode_id>.mp4
    <root>/annotation/{train,val}/<episode_id>.json

The video is already packed, so it passes through untouched; the split is
re-derived downstream rather than inherited, so one corpus materialised twice
lands the same episodes on the same side.
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

__all__ = ["CanonicalTreeAdapter"]

_JOINT_KEY = "observation.state.joint_position"
_GRIPPER_KEY = "observation.state.gripper_position"


class CanonicalTreeAdapter:
    """Reads ``<root>/videos/<split>/*.mp4`` with their annotations."""

    SOURCE_ID = "canonical"

    def episodes(self, source: TrainSource
                 ) -> Iterator[RawEpisode | SkippedEpisode]:
        """Yield one entry per video under ``source.root``."""
        for video in sorted(glob.glob(
                os.path.join(source.root, "videos", "*", "*.mp4"))):
            split = os.path.basename(os.path.dirname(video))
            episode_id = os.path.splitext(os.path.basename(video))[0]
            annotation = os.path.join(source.root, "annotation", split,
                                      f"{episode_id}.json")
            try:
                label = json.loads(open(annotation).read())
            except (OSError, ValueError) as exc:
                yield SkippedEpisode(episode_id, f"annotation unreadable: {exc}",
                                     video)
                continue
            if label.get("joint_source") != "real":
                yield SkippedEpisode(
                    episode_id,
                    f"joint_source is {label.get('joint_source')!r}, not 'real'",
                    annotation)
                continue
            joints = np.asarray(label.get(_JOINT_KEY, []), dtype=np.float32)
            if joints.ndim != 2 or joints.size == 0:
                yield SkippedEpisode(
                    episode_id, f"{_JOINT_KEY} is not a [T, J] array", annotation)
                continue
            cols = source.joint_columns
            if cols:
                if max(cols) >= joints.shape[1]:
                    yield SkippedEpisode(
                        episode_id,
                        f"joint_columns reach index {max(cols)} but the "
                        f"annotation has {joints.shape[1]} columns", annotation)
                    continue
                joints = joints[:, list(cols)]
            gripper = label.get(_GRIPPER_KEY)
            yield RawEpisode(
                episode_id=episode_id,
                joints=joints,
                gripper=None if gripper is None
                else np.asarray(gripper, dtype=np.float32),
                fps=float(label.get("fps", 0.0)),
                scene_key=episode_id.rsplit("__", 1)[0],
                source_path=video,
                packed=video,
            )


register_adapter(CanonicalTreeAdapter.SOURCE_ID, CanonicalTreeAdapter)
