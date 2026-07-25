"""Save/load for :class:`~kinescore.heads.attentive.AttentivePoseHead` readers.

Adapted from ``PixelPhysicsJudge.save`` / ``.load``. Two real defects in the
source format are fixed here, and both are verified against the three real
checkpoints (``judge_v3l``, ``judge_v3l_mv``, ``judge_reward``) rather than
assumed:

* **D5 -- ``hidden``/``dropout`` were never persisted.** The source's
  ``save()`` wrote a 13-key cfg (12 without ``n_cams``, which was added
  later and is itself absent from ``judge_v3l``/``judge_reward``) that
  covered the backbone and pooling settings but not ``hidden`` or
  ``dropout``. ``load()`` then reconstructed the head with ``hidden``
  hardcoded to ``AttentivePoseHead``'s default (512). Any checkpoint trained
  with ``hidden != 512`` therefore could not be reloaded at all --
  ``load_state_dict(strict=True)`` would fail on the ``mlp.0.*`` shape.
  :func:`save` now persists both fields. :func:`load` still tolerates their
  absence in an old file by **inferring** ``hidden`` from
  ``state_dict["mlp.0.bias"].shape[0]`` -- verified against all three real
  checkpoints, which all give 512 -- and defaulting ``dropout`` (it does not
  affect any parameter shape, only training-time regularisation, so a wrong
  guess here cannot break loading).

  ``n_heads`` and ``embed_dim`` are cross-checked the same way (from
  ``score.weight.shape[0]`` and ``norm.weight.shape[0]``) even when the cfg
  *does* carry them, so a hand-edited or corrupted cfg fails loudly with both
  the cfg value and the inferred value named, instead of constructing a head
  whose declared shape doesn't match its own weights.

* **Missing ``n_cams`` in legacy cfgs.** ``judge_v3l`` and ``judge_reward``
  predate the multiview head entirely and have no ``n_cams`` key. ``load``
  preserves the source's ``cfg.get("n_cams", 1)`` fallback exactly.

On top of the source format, :func:`save` additionally persists
``view_layout_key``, ``robot_name`` and ``limit_semantics`` -- see
``core/reader.py``: a reader should declare its own semantics rather than
have the caller guess them from which checkpoint file was passed in.
``load_state_dict(strict=True)`` is preserved throughout; nothing here ever
loads partial or renamed weights.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from kinescore.core.clip import ViewLayout
from kinescore.core.reader import LimitSemantics
from kinescore.heads.attentive import AttentivePoseHead

__all__ = ["LoadedCheckpoint", "save", "load"]

# Default limit_semantics for a checkpoint that predates this field. Every
# real checkpoint at this format (judge_v3l, judge_v3l_mv, judge_reward) was
# trained with the sigmoid squash in PixelPhysicsJudge.predict_pose, so
# "squashed" is the historically accurate default, not a guess.
_LEGACY_LIMIT_SEMANTICS: LimitSemantics = "squashed"


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Everything :func:`load` recovers from a saved ``AttentivePoseHead``.

    ``head`` is ready to use (``eval()``'d, weights loaded). The rest is
    provenance: what the cfg declared, what was inferred from the weights
    themselves, and the reader-level semantics the checkpoint carries.
    """

    head: AttentivePoseHead
    n_cams: int
    hidden: int
    n_heads: int
    embed_dim: int
    n_joints: int
    dropout: float
    view_layout_key: Optional[str]
    robot_name: Optional[str]
    limit_semantics: LimitSemantics
    cfg: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


def save(path: str, head: AttentivePoseHead, *, view_layout: ViewLayout,
         robot_name: str, limit_semantics: LimitSemantics = "squashed",
         backbone_cfg: Optional[Dict[str, Any]] = None,
         meta: Optional[Dict[str, Any]] = None) -> None:
    """Persist ``head`` plus enough cfg to reconstruct and identify it.

    Parameters
    ----------
    path:
        Output ``.pt`` path; parent directories are created.
    head:
        A constructed, (optionally trained) :class:`AttentivePoseHead`.
    view_layout:
        The camera layout ``head`` was trained for. Only ``.key`` (a string)
        is persisted -- see :meth:`~kinescore.core.clip.ViewLayout.key` --
        not a reconstructable object; a loader that needs the full layout
        (with ``tokens_per_view``) must be told it independently, the same
        way the backbone's patch grid is.
    robot_name, limit_semantics:
        See ``core/reader.py``. Stored so a reader built from this checkpoint
        declares them itself.
    backbone_cfg:
        Optional dict of backbone settings (``dino_model``, ``embed_dim``,
        ...) to round-trip alongside the head, e.g. the output of
        ``vars()``-ing a :class:`~kinescore.backbones.dino.FeatureBackbone`'s
        constructor args. Not required -- callers that keep the backbone cfg
        elsewhere can omit it.
    meta:
        Free-form provenance (training run id, val error, ...).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cfg: Dict[str, Any] = {
        "n_joints": head.n_joints,
        "n_heads": head.n_heads,
        "n_cams": head.n_cams,
        "hidden": head.mlp[0].out_features,  # == "hidden" ctor arg
        "dropout": head.mlp[2].p,  # nn.Dropout.p
        "embed_dim": head.norm.normalized_shape[0],
        "view_layout_key": view_layout.key,
        "robot_name": robot_name,
        "limit_semantics": limit_semantics,
    }
    if backbone_cfg:
        cfg["backbone"] = dict(backbone_cfg)
    torch.save({"head": head.state_dict(), "cfg": cfg, "meta": meta or {}}, path)


def _infer_or_check(cfg: Dict[str, Any], key: str, inferred: int) -> int:
    """Return ``cfg[key]`` if present and consistent with ``inferred``, else
    ``inferred``. Raises, naming both values, on a real mismatch."""
    if key not in cfg:
        return inferred
    declared = int(cfg[key])
    if declared != inferred:
        raise ValueError(
            f"checkpoint cfg[{key!r}]={declared} does not match the value "
            f"inferred from the saved weights ({inferred}); the cfg and the "
            f"state_dict disagree about this head's shape")
    return declared


def load(path: str, device: str = "cpu") -> LoadedCheckpoint:
    """Load an :class:`AttentivePoseHead` saved by :func:`save`, or a legacy
    ``PixelPhysicsJudge``-format checkpoint (``judge_v3l``/``judge_v3l_mv``/
    ``judge_reward``).

    Every shape-bearing constructor argument is either read from ``cfg`` or
    inferred from the state dict itself (never left at
    :class:`AttentivePoseHead`'s constructor default), so a mismatched or
    incomplete cfg cannot silently produce a head with the wrong shape --
    ``load_state_dict(strict=True)`` would simply fail, loudly, instead.
    """
    ck = torch.load(path, map_location="cpu")
    sd: Dict[str, torch.Tensor] = ck["head"]
    cfg: Dict[str, Any] = dict(ck.get("cfg", {}))
    meta: Dict[str, Any] = dict(ck.get("meta", {}))

    if "mlp.0.bias" not in sd or "score.weight" not in sd or "norm.weight" not in sd:
        raise ValueError(
            f"{path!r} does not look like an AttentivePoseHead checkpoint "
            f"(missing mlp.0.bias / score.weight / norm.weight)")

    hidden = _infer_or_check(cfg, "hidden", int(sd["mlp.0.bias"].shape[0]))
    n_heads = _infer_or_check(cfg, "n_heads", int(sd["score.weight"].shape[0]))
    embed_dim = _infer_or_check(cfg, "embed_dim", int(sd["norm.weight"].shape[0]))
    n_joints = int(sd["mlp.3.bias"].shape[0]) - 1  # output is n_joints + gripper
    n_cams = int(cfg.get("n_cams", 1))  # preserved exactly: absent in legacy cfgs
    dropout = float(cfg.get("dropout", 0.1))  # shape-inert; source's default

    head = AttentivePoseHead(in_dim=embed_dim, hidden=hidden, n_joints=n_joints,
                             dropout=dropout, n_heads=n_heads, n_cams=n_cams)
    head.load_state_dict(sd, strict=True)
    head = head.to(device).eval()

    return LoadedCheckpoint(
        head=head, n_cams=n_cams, hidden=hidden, n_heads=n_heads,
        embed_dim=embed_dim, n_joints=n_joints, dropout=dropout,
        view_layout_key=cfg.get("view_layout_key"),
        robot_name=cfg.get("robot_name"),
        limit_semantics=cfg.get("limit_semantics", _LEGACY_LIMIT_SEMANTICS),
        cfg=cfg, meta=meta,
    )
