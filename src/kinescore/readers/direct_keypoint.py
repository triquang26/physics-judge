"""The direct-keypoint reader: frames -> 3-D keypoints, no FK, no clamp.

Composes :class:`~kinescore.backbones.dino.FeatureBackbone` +
:class:`~kinescore.heads.views.ViewEmbedding` +
:class:`~kinescore.heads.heteroscedastic.ReadoutV2Head`, exactly the same
FORWARD path as :class:`~kinescore.readers.heteroscedastic.HeteroscedasticPoseReader`
-- but the DECODE differs. ``HeteroscedasticPoseReader`` treats every head
output as a joint angle and runs it through :func:`~kinescore.heads.ranges.clamp_for_fk`
so :class:`~kinescore.core.robot.RobotSpec` FK has something safe to consume.
This reader has no robot, no URDF and no joint limits in its path at all: the
head's ``mu`` is reshaped straight into ``(B,T,K,3)`` keypoints in the
robot-base frame (metres) and returned as :attr:`~kinescore.core.reader.Readout.P`.
``q``/``q_raw`` are ``None`` -- there is no joint-angle representation to put
there.

Why a checkpoint would ever train this instead of joints+FK
-------------------------------------------------------------
FK is exact but brittle to a wrong URDF/kinematic chain and cannot express a
prediction the model is uncertain about except through the joint's own
sigma. A keypoint head instead regresses the quantity most physics
violations (rigidity, jerk, teleport -- see ``kinescore.violations``) are
actually computed on, skipping the FK step and its assumptions entirely. The
tradeoff is that ``limit_semantics="keypoints"`` readers cannot report a
joint-limit violation at all (there are no joint limits here to violate) --
see ``core/reader.py``'s module docstring.

``ReadoutV2Head`` has no native multiview support (see
``readers/heteroscedastic.py``), so multiview is composed externally here too,
via the same zero-init :class:`~kinescore.heads.views.ViewEmbedding`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from kinescore.backbones.dino import FeatureBackbone
from kinescore.core.clip import ViewLayout
from kinescore.core.reader import LimitSemantics, Readout
from kinescore.heads.heteroscedastic import ReadoutV2Head
from kinescore.heads.views import ViewEmbedding
from kinescore.readers._frames import normalize_frames

__all__ = ["DirectKeypointPoseReader"]


@dataclass
class DirectKeypointPoseReader:
    """Frozen backbone + view bias + :class:`ReadoutV2Head`, decoded as
    ``(B,T,K,3)`` keypoints instead of joint angles.

    Parameters
    ----------
    backbone:
        A :class:`~kinescore.backbones.dino.FeatureBackbone`.
    head:
        A constructed :class:`~kinescore.heads.heteroscedastic.ReadoutV2Head`
        with ``head.n_out == 3 * n_keypoints`` -- this reader reshapes every
        head output into ``(K, 3)`` points, so ``n_out`` must be a multiple
        of 3.
    n_keypoints:
        ``K``, the number of 3-D points the head predicts per frame
        (``head.n_out == 3 * n_keypoints``, checked in :meth:`__post_init__`).
    view_layout:
        Token-space camera layout. Routed through a zero-init
        :class:`~kinescore.heads.views.ViewEmbedding` before pooling -- see
        ``readers/heteroscedastic.py``'s module docstring for why this is a
        no-op at ``n_views=1``.
    robot_name, reader_id:
        See ``core/reader.py``. ``robot_name`` identifies which robot's frame
        the keypoints are expressed in even though this reader never touches
        that robot's URDF/FK.
    use_context:
        Forwarded to ``ReadoutV2Head.forward`` -- ``False`` bypasses the
        temporal encoder (per-frame mode).
    """

    backbone: FeatureBackbone
    head: ReadoutV2Head
    n_keypoints: int
    view_layout: ViewLayout
    robot_name: str
    reader_id: str
    use_context: bool = True
    #: The keypoint head is trained on RAW cached DINO tokens (the
    #: ``KeypointTrainer`` feeds ``load_cache`` output straight to the head, no
    #: view bias), so applying an untrained ViewEmbedding at inference is a
    #: train/inference mismatch that corrupts the keypoints (~170 mm). Default
    #: off; set True only for a head trained *with* a matching view embedding.
    apply_view_embedding: bool = False
    limit_semantics: LimitSemantics = field(default="keypoints", init=False)
    view_embedding: ViewEmbedding = field(init=False)

    def __post_init__(self) -> None:
        if self.n_keypoints < 1:
            raise ValueError(f"n_keypoints must be >= 1, got {self.n_keypoints}")
        if self.head.n_out != 3 * self.n_keypoints:
            raise ValueError(
                f"head.n_out={self.head.n_out} != 3 * n_keypoints "
                f"(3 * {self.n_keypoints} = {3 * self.n_keypoints})")
        view_embedding = ViewEmbedding(in_dim=self.head.in_dim,
                                       view_layout=self.view_layout)
        # Same device-placement fix as HeteroscedasticPoseReader: `load_head`
        # moves `head` to its target device before this reader is
        # constructed, so a freshly built ViewEmbedding would default to CPU
        # and its zero bias would meet CUDA tokens in `forward`.
        head_param = next(self.head.parameters(), None)
        if head_param is not None:
            view_embedding = view_embedding.to(head_param.device)
        self.view_embedding = view_embedding

    def read(self, frames: torch.Tensor) -> Readout:
        """``(T,H,W,3)`` uint8 or ``(B,T,3,H,W)`` float in ``[0,1]`` -> Readout.

        ``Readout.q``/``q_raw`` are ``None``; ``Readout.P`` is ``(B,T,K,3)``
        keypoints in the robot-base frame, metres -- no clamp, no FK.
        """
        rgb, B, T = normalize_frames(frames)
        feat = self.backbone.encode(rgb)  # (B*T, V, P, D)
        _, V, P, D = feat.shape
        feat = feat.reshape(B, T, V * P, D).float()
        if self.apply_view_embedding:
            feat = self.view_embedding(feat)  # asserts token count; adds bias

        out = self.head(feat, use_context=self.use_context)
        mu, logvar = out["mu"], out["logvar"]
        points = mu.reshape(B, T, self.n_keypoints, 3)
        sigma = torch.exp(0.5 * logvar).reshape(B, T, self.n_keypoints, 3)
        return Readout(q=None, q_raw=None, sigma=sigma, aux=None, P=points,
                       extras={})
