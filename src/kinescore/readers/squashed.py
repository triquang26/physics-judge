"""The valid-by-construction reader: frames -> squashed joints + gripper.

Composes :class:`~kinescore.backbones.dino.FeatureBackbone` +
:class:`~kinescore.heads.attentive.AttentivePoseHead` +
:func:`~kinescore.heads.ranges.squash_to_limits`, mirroring
``PixelPhysicsJudge.predict_pose`` in the source. This is the reader for the
three real checkpoints this package targets compatibility with
(``judge_v3l``, ``judge_v3l_mv``, ``judge_reward`` -- see
``readers/checkpoint.py``).

Every prediction is inside ``[q_lo, q_hi]`` by construction, so
``limit_semantics = "squashed"`` and ``q_raw`` is always ``None`` (see
``core/reader.py`` and ``heads/ranges.py`` for why that matters -- defect
D7). Readers that need joint-limit violation to be an observable quantity
should use :class:`~kinescore.readers.heteroscedastic.HeteroscedasticPoseReader`
instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from kinescore.backbones.dino import FeatureBackbone
from kinescore.core.clip import ViewLayout
from kinescore.core.reader import LimitSemantics, Readout
from kinescore.heads.attentive import AttentivePoseHead
from kinescore.heads.ranges import squash_to_limits
from kinescore.readers._frames import normalize_frames

__all__ = ["SquashedPoseReader"]


@dataclass
class SquashedPoseReader:
    """Frozen backbone + :class:`AttentivePoseHead`, squashed into limits.

    Parameters
    ----------
    backbone:
        A :class:`~kinescore.backbones.dino.FeatureBackbone`. Its
        ``view_layout`` must match ``view_layout`` below (both describe the
        same clip's camera packing; the backbone's is pixel-space, this
        reader's is the token-space contract the head was trained on).
    head:
        A constructed :class:`~kinescore.heads.attentive.AttentivePoseHead`
        with ``head.n_cams == view_layout.n_views``, typically produced by
        ``readers.checkpoint.load``.
    q_lo, q_hi:
        ``(n_joints,)`` radians -- e.g. ``RobotSpec.q_lo`` / ``.q_hi``. Kept
        as plain tensors rather than a ``RobotSpec`` dependency so this
        reader is constructible (and testable) without any concrete robot
        plugin.
    view_layout:
        Token-space camera layout the head expects
        (``ViewLayout.assert_tokens`` is enforced inside ``head.forward``).
    robot_name, reader_id:
        See ``core/reader.py``.
    """

    backbone: FeatureBackbone
    head: AttentivePoseHead
    q_lo: torch.Tensor
    q_hi: torch.Tensor
    view_layout: ViewLayout
    robot_name: str
    reader_id: str
    limit_semantics: LimitSemantics = field(default="squashed", init=False)

    def __post_init__(self) -> None:
        if self.head.n_cams != self.view_layout.n_views:
            raise ValueError(
                f"head.n_cams={self.head.n_cams} != "
                f"view_layout.n_views={self.view_layout.n_views}")

    def read(self, frames: torch.Tensor) -> Readout:
        """``(T,H,W,3)`` uint8 or ``(B,T,3,H,W)`` float in ``[0,1]`` -> Readout."""
        rgb, B, T = normalize_frames(frames)
        feat = self.backbone.encode(rgb)  # (B*T, V, P, D)
        _, V, P, D = feat.shape
        feat = feat.reshape(B, T, V * P, D).float()
        raw = self.head(feat)  # (B,T,n_joints+1)
        q_raw, gripper_raw = raw[..., :-1], raw[..., -1:]
        q = squash_to_limits(q_raw, self.q_lo, self.q_hi)
        gripper = torch.sigmoid(gripper_raw)
        return Readout(q=q, q_raw=None, sigma=None, aux=gripper, extras={})
