"""The reader: packed video frames -> 3-D keypoints.

:class:`~kinescore.backbones.dino.FeatureBackbone` (frozen) encodes each frame
into patch tokens; :class:`~kinescore.heads.keypoint.KeypointHead` (trained)
reads ``K`` points out of them. Points are metres in the robot-base frame and
go straight to the violation detectors -- there is no kinematic chain in this
path, so no URDF, no joint limits and no forward kinematics at read time.

The robot is named, not loaded: ``robot_name`` says which base frame the
points live in.

A clip is read in windows of the head's ``t_max`` rather than whole, so a clip
longer than the positional table still reads; temporal context spans a window,
not the clip.
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
    frame_chunk:
        Frames encoded per backbone call (``0`` = the whole clip at once).
        Attention is quadratic in tokens per frame, so a long clip at full
        resolution can exhaust the device in one batch. Frames are encoded
        independently, so chunking changes memory and time, never the numbers.
        Only a chunk is moved to the head's device, so device memory is bounded
        by this rather than by the clip's length.
    """

    backbone: FeatureBackbone
    head: KeypointHead
    view_layout: ViewLayout
    robot_name: str
    reader_id: str
    use_context: bool = True
    frame_chunk: int = 0

    @property
    def n_keypoints(self) -> int:
        return self.head.n_keypoints

    def read(self, frames: torch.Tensor, *, frame_chunk: int | None = None
             ) -> Readout:
        """``(T,H,W,3)`` uint8 or ``(B,T,3,H,W)`` float in ``[0,1]`` -> Readout."""
        rgb, b, t = normalize_frames(frames)
        step = self.frame_chunk if frame_chunk is None else frame_chunk
        step = rgb.shape[0] if step <= 0 else step
        device = next(self.head.parameters()).device
        feat = torch.cat([self.backbone.encode(rgb[i:i + step].to(device))
                          .half().cpu()
                          for i in range(0, rgb.shape[0], step)])  # (B*T,V,P,D)
        _, v, p, d = feat.shape
        self.view_layout.assert_tokens(v * p)
        feat = feat.reshape(b, t, v * p, d)
        window = max(1, self.head.t_max)
        points = torch.cat(
            [self.head(feat[:, i:i + window].to(device).float(),
                       use_context=self.use_context)
             for i in range(0, t, window)], dim=1)
        return Readout(P=points)
