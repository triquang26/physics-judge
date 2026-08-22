"""The reader: packed video frames -> 3-D keypoints.

:class:`~kinescore.backbones.dino.FeatureBackbone` (frozen) encodes each frame
into patch tokens; :class:`~kinescore.heads.keypoint.KeypointHead` (trained)
reads ``K`` points out of them. Points are metres in the robot-base frame and
go straight to the violation detectors -- there is no kinematic chain in this
path, so no URDF, no joint limits and no forward kinematics at read time.

The robot is named, not loaded: ``robot_name`` says which base frame the
points live in.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from kinescore.backbones.dino import FeatureBackbone
from kinescore.core.clip import ViewLayout
from kinescore.core.reader import Readout
from kinescore.heads.keypoint import KeypointHead
from kinescore.readers._frames import normalize_frames

__all__ = ["KeypointReader"]


@dataclass
class KeypointReader:
    """Frozen backbone + trained head, read as ``(B, T, K, 3)``.

    Parameters
    ----------
    backbone:
        A :class:`~kinescore.backbones.dino.FeatureBackbone`.
    head:
        A :class:`~kinescore.heads.keypoint.KeypointHead`.
    view_layout:
        Camera packing the head was trained on. The token count of every
        encoded frame is checked against it, so a three-panel head cannot
        quietly consume single-view features.
    robot_name:
        Which robot's base frame the points are in.
    reader_id:
        Stable identity, recorded with every score.
    use_context:
        ``False`` reads each frame independently, bypassing the head's
        temporal stage.
    """

    backbone: FeatureBackbone
    head: KeypointHead
    view_layout: ViewLayout
    robot_name: str
    reader_id: str
    use_context: bool = True

    @property
    def n_keypoints(self) -> int:
        return self.head.n_keypoints

    def read(self, frames: torch.Tensor) -> Readout:
        """``(T,H,W,3)`` uint8 or ``(B,T,3,H,W)`` float in ``[0,1]`` -> Readout."""
        rgb, b, t = normalize_frames(frames)
        feat = self.backbone.encode(rgb)          # (B*T, V, P, D)
        _, v, p, d = feat.shape
        self.view_layout.assert_tokens(v * p)
        feat = feat.reshape(b, t, v * p, d).float()
        points = self.head(feat, use_context=self.use_context)
        return Readout(P=points)
