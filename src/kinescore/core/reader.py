"""Pose reader contract: pixels -> 3-D keypoints.

A ``PoseReader`` wraps a frozen vision backbone and a trained head, and turns
video frames into keypoint positions. It is the only learned component in
kinescore -- every violation detector downstream is analytic, so the benchmark
cannot drift with a model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

from kinescore.core.clip import ViewLayout

__all__ = ["Readout", "PoseReader"]


@dataclass
class Readout:
    """One clip's predicted pose.

    Attributes
    ----------
    P:
        ``(B, T, K, 3)`` keypoints in the robot-base frame, metres.
    extras:
        Anything else a detector may consume.
    """

    P: torch.Tensor
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.P.shape[1])


@runtime_checkable
class PoseReader(Protocol):
    """Frozen backbone plus trained head, reading keypoints from packed frames.

    Attributes
    ----------
    view_layout:
        Camera packing this reader was trained for. Checked against the clip's
        own layout before scoring, so a three-panel checkpoint cannot consume
        single-view frames.
    robot_name:
        Which :class:`~kinescore.core.robot.RobotSpec` frame ``P`` is in.
    reader_id:
        Stable identity for the output record, e.g.
        ``"dinov3_vitl16@1024/p2/mv3_row"``.
    """

    view_layout: ViewLayout
    robot_name: str
    reader_id: str

    def read(self, frames: torch.Tensor) -> Readout:
        """Read ``(B, T, H, W, 3)`` uint8 frames into a :class:`Readout`."""
        ...
