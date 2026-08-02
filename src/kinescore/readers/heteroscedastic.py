"""The observable-limit-violation reader: frames -> raw joints + sigma.

Composes :class:`~kinescore.backbones.dino.FeatureBackbone` +
:class:`~kinescore.heads.views.ViewEmbedding` +
:class:`~kinescore.heads.heteroscedastic.ReadoutV2Head` +
:func:`~kinescore.heads.ranges.clamp_for_fk`, mirroring
``ReadoutV2Scorer.score_feat`` in the source (minus FK/physics scoring, which
belongs to the metric layer, not the reader).

``ReadoutV2Head`` has no native multiview support in the source at all (it
was only ever trained single-camera), so multiview here is composed
externally via :class:`~kinescore.heads.views.ViewEmbedding` -- with
``view_layout.n_views == 1`` (the default) it adds an exact-zero bias, so a
single-view clip through this reader is bit-identical to routing the raw
tokens straight into the head.

``limit_semantics = "raw_rad"``: ``q`` is FK-clamped, ``q_raw`` is the head's
unconstrained output, and ``clamp_rad`` (the clamp magnitude, i.e. the
observable joint-limit violation) rides along in ``Readout.extras`` -- see
``heads/ranges.py`` for why this reader family exists (defect D7).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from kinescore.backbones.dino import FeatureBackbone
from kinescore.core.clip import ViewLayout
from kinescore.core.reader import LimitSemantics, Readout
from kinescore.heads.heteroscedastic import ReadoutV2Head
from kinescore.heads.ranges import clamp_for_fk
from kinescore.heads.views import ViewEmbedding
from kinescore.readers._frames import normalize_frames

__all__ = ["HeteroscedasticPoseReader"]


@dataclass
class HeteroscedasticPoseReader:
    """Frozen backbone + view bias + :class:`ReadoutV2Head`, FK-clamped.

    Parameters
    ----------
    backbone:
        A :class:`~kinescore.backbones.dino.FeatureBackbone`.
    head:
        A constructed :class:`~kinescore.heads.heteroscedastic.ReadoutV2Head`
        with ``head.n_out == q_lo.shape[0]`` (this reader treats every head
        output as a joint angle to clamp; a robot with auxiliary non-angle
        DoF, e.g. hand flexion, should give those permissive ``[lo,hi]``
        bounds rather than being excluded here).
    q_lo, q_hi:
        ``(n_out,)`` radians.
    view_layout:
        Token-space camera layout. Routed through a zero-init
        :class:`~kinescore.heads.views.ViewEmbedding` before pooling --
        see module docstring.
    robot_name, reader_id:
        See ``core/reader.py``.
    use_context:
        Forwarded to ``ReadoutV2Head.forward`` -- ``False`` bypasses the
        temporal encoder (per-frame mode).
    """

    backbone: FeatureBackbone
    head: ReadoutV2Head
    q_lo: torch.Tensor
    q_hi: torch.Tensor
    view_layout: ViewLayout
    robot_name: str
    reader_id: str
    use_context: bool = True
    limit_semantics: LimitSemantics = field(default="raw_rad", init=False)
    view_embedding: ViewEmbedding = field(init=False)

    def __post_init__(self) -> None:
        if self.q_lo.shape != self.q_hi.shape:
            raise ValueError(
                f"q_lo shape {tuple(self.q_lo.shape)} != "
                f"q_hi shape {tuple(self.q_hi.shape)}")
        view_embedding = ViewEmbedding(in_dim=self.head.in_dim,
                                       view_layout=self.view_layout)
        # Place it on whatever device the head already lives on. `load_head`
        # does `head.to(device)` *before* this reader is constructed, so a
        # freshly built ViewEmbedding would default to CPU and its bias would
        # meet CUDA tokens in `forward` -- "Expected all tensors to be on the
        # same device". Zero-init means the bias is numerically a no-op at
        # n_views=1, so this placement fixes a crash without moving any number.
        head_param = next(self.head.parameters(), None)
        if head_param is not None:
            view_embedding = view_embedding.to(head_param.device)
        self.view_embedding = view_embedding

    def read(self, frames: torch.Tensor) -> Readout:
        """``(T,H,W,3)`` uint8 or ``(B,T,3,H,W)`` float in ``[0,1]`` -> Readout."""
        rgb, B, T = normalize_frames(frames)
        feat = self.backbone.encode(rgb)  # (B*T, V, P, D)
        _, V, P, D = feat.shape
        feat = feat.reshape(B, T, V * P, D).float()
        feat = self.view_embedding(feat)  # asserts token count; adds bias

        out = self.head(feat, use_context=self.use_context)
        mu, logvar = out["mu"], out["logvar"]
        q, clamp_rad = clamp_for_fk(mu, self.q_lo, self.q_hi)
        sigma = torch.exp(0.5 * logvar)
        return Readout(q=q, q_raw=mu, sigma=sigma, aux=None,
                       extras={"clamp_rad": clamp_rad})
