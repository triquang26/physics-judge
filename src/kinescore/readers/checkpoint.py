"""Persist and restore a keypoint reader.

A checkpoint is ``{"head": state_dict, "n_out": 3*K, "embed_dim": D, "cfg": {...}}``.
``cfg`` records the cell the head was trained for, and :func:`load_reader`
verifies it against what the caller asks for, so a head trained on three panels
cannot be handed two. It also records which head class to rebuild, under
``head``; a checkpoint that names none is a :data:`HEAD_DEFAULT_KIND` head.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from kinescore.core.clip import ViewLayout
from kinescore.heads import AnyKeypointHead, DiffusionKeypointHead, KeypointHead
from kinescore.readers.keypoint import KeypointReader

__all__ = [
    "ReaderExpectation", "CheckpointMismatch", "HEAD_CLASSES",
    "HEAD_DEFAULT_KIND", "HEAD_DEFAULTS", "HEAD_CTOR_KEYS", "build_backbone",
    "save_reader", "load_reader", "read_cfg",
]

#: The head classes a checkpoint may name in ``cfg["head"]``.
HEAD_CLASSES: dict[str, type[AnyKeypointHead]] = {
    KeypointHead.head_kind: KeypointHead,
    DiffusionKeypointHead.head_kind: DiffusionKeypointHead,
}

#: The head a checkpoint that names none was written by.
HEAD_DEFAULT_KIND = KeypointHead.head_kind

_SHARED_CTOR_KEYS = {
    "in_dim", "n_views", "tokens_per_view", "d_model", "decoder_nhead",
    "n_decoder_layers", "temporal_nhead", "ff", "n_temporal_layers", "t_max",
    "dropout",
}

#: Each head kind's keyword set. A saved ``cfg`` is filtered to its head's
#: keys before construction, so the identity fields it also carries never
#: reach the constructor.
HEAD_CTOR_KEYS: dict[str, frozenset[str]] = {
    "keypoint": frozenset(_SHARED_CTOR_KEYS),
    "diffusion": frozenset(_SHARED_CTOR_KEYS | {
        "n_coord_freqs", "sample_steps", "n_samples", "init_t"}),
}

_SHARED_DEFAULTS = {
    "d_model": 512, "decoder_nhead": 8, "n_decoder_layers": 2,
    "temporal_nhead": 8, "ff": 2048, "n_temporal_layers": 4, "t_max": 64,
    "dropout": 0.1, "n_views": 1, "tokens_per_view": 576,
}

#: Architecture each head kind is trained with. Applied for any of its
#: :data:`HEAD_CTOR_KEYS` a checkpoint's ``cfg`` omits.
HEAD_DEFAULTS: dict[str, dict[str, Any]] = {
    "keypoint": dict(_SHARED_DEFAULTS),
    "diffusion": {**_SHARED_DEFAULTS, "n_coord_freqs": 10, "sample_steps": 10,
                  "n_samples": 5, "init_t": 0.3},
}



class CheckpointMismatch(ValueError):
    """A checkpoint does not describe the cell it is being loaded for."""


@dataclass(frozen=True)
class ReaderExpectation:
    """What a caller requires of a checkpoint before it will use it."""

    cell_id: str | None = None
    robot: str | None = None
    view_id: str | None = None
    n_views: int | None = None
    packing: str | None = None
    n_keypoints: int | None = None

    def check(self, path: str, cfg: dict[str, Any], n_out: int) -> None:
        got = {
            "robot": cfg.get("robot"),
            "view_id": cfg.get("view_id"),
            "n_views": cfg.get("n_views"),
            "packing": cfg.get("packing"),
            "n_keypoints": n_out // 3,
        }
        bad = [
            f"{k}: checkpoint has {got[k]!r}, cell requires {want!r}"
            for k, want in (
                ("robot", self.robot), ("view_id", self.view_id),
                ("n_views", self.n_views), ("packing", self.packing),
                ("n_keypoints", self.n_keypoints),
            )
            if want is not None and got[k] is not None and got[k] != want
        ]
        if bad:
            raise CheckpointMismatch(
                f"{path!r} does not match cell {self.cell_id!r}:\n  "
                + "\n  ".join(bad))


def read_cfg(path: str) -> dict[str, Any]:
    """The checkpoint's ``cfg`` dict, without building anything."""
    ck = torch.load(path, map_location="cpu")
    return dict(ck.get("cfg", {}))


def build_backbone(view_layout: ViewLayout, embed_dim: int,
                   backbone_cfg: dict[str, Any] | None, device: str):
    """Construct the frozen feature backbone this checkpoint reads through."""
    from kinescore.backbones.default import build_backbone as _build

    overrides = dict(backbone_cfg or {})
    overrides.setdefault("embed_dim", embed_dim)
    return _build(view_layout, device=device, overrides=overrides)


def save_reader(path: str, head: AnyKeypointHead, *, cell_id: str, robot: str,
                view_id: str, view_layout: ViewLayout,
                meta: dict[str, Any] | None = None) -> None:
    """Write ``head`` plus the cell identity needed to load it back safely.

    Every keyword the head's class takes is read off the instance under its
    own name, so a head kind is saved by the same rule that rebuilds it.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cfg: dict[str, Any] = {
        k: getattr(head, k) for k in sorted(HEAD_CTOR_KEYS[head.head_kind])}
    cfg.update({
        "cell_id": cell_id,
        "robot": robot,
        "view_id": view_id,
        "view_layout_key": view_layout.key,
        "n_views": view_layout.n_views,
        "packing": view_layout.packing,
        "head": head.head_kind,
        "n_out": head.n_out,
    })
    torch.save({
        "head": head.state_dict(), "n_out": head.n_out,
        "embed_dim": head.in_dim, "cfg": cfg, "meta": dict(meta or {}),
    }, path)


def _head_tensors(path: str, head: AnyKeypointHead,
                  saved: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The saved tensors the head declares, refusing an incomplete file.

    A checkpoint may carry tensors the head has no parameter for; those are
    not part of the model and are dropped. A tensor the head *needs* and the
    file does not have is fatal.
    """
    wanted = set(head.state_dict())
    missing = sorted(wanted - set(saved))
    if missing:
        raise ValueError(
            f"{path!r} is missing {len(missing)} head tensor(s): "
            f"{missing[:8]}{' ...' if len(missing) > 8 else ''}")
    return {k: v for k, v in saved.items() if k in wanted}


def load_reader(path: str, *, robot: Any, view_layout: ViewLayout,
                expect: ReaderExpectation | None = None,
                device: str = "cpu", reader_id: str | None = None,
                backbone: Any | None = None,
                backbone_cfg: dict[str, Any] | None = None,
                use_context: bool | None = None):
    """Load ``path`` into a ready-to-score
    :class:`~kinescore.readers.keypoint.KeypointReader`.

    Parameters
    ----------
    robot:
        A :class:`~kinescore.core.robot.RobotSpec`. Read only for its ``name``:
        the reader never runs forward kinematics.
    expect:
        Checked against the checkpoint's ``cfg`` before anything is built.
        ``None`` skips the check.

    Raises
    ------
    ValueError
        If ``path`` is not a keypoint checkpoint, names a head class that
        does not exist, or its ``n_out``/``embed_dim`` cannot be resolved.
    CheckpointMismatch
        If ``expect`` disagrees with the checkpoint's own ``cfg``.
    """
    ck = torch.load(path, map_location="cpu")
    sd = ck.get("head")
    if sd is None:
        raise ValueError(
            f"{path!r} is not a keypoint checkpoint (no 'head' key; got "
            f"top-level keys {sorted(ck)})")
    cfg: dict[str, Any] = dict(ck.get("cfg", {}))

    n_out = int(cfg.get("n_out", ck.get("n_out", 0)) or 0)
    if n_out <= 0 or n_out % 3 != 0:
        raise ValueError(
            f"{path!r}: n_out must be a positive multiple of 3 (3 per "
            f"keypoint), got {n_out!r}")
    in_dim = int(cfg.get("in_dim", ck.get("embed_dim", 0)) or 0)
    if in_dim <= 0:
        raise ValueError(
            f"{path!r}: could not resolve embed_dim/in_dim (cfg keys "
            f"{sorted(cfg)}, top-level keys {sorted(ck)})")

    if expect is not None:
        expect.check(path, cfg, n_out)

    kind = str(cfg.get("head", HEAD_DEFAULT_KIND))
    if kind not in HEAD_CLASSES:
        raise ValueError(
            f"{path!r}: unknown head {kind!r}, known heads are "
            f"{sorted(HEAD_CLASSES)}")
    ctor_cfg: dict[str, Any] = dict(HEAD_DEFAULTS[kind])
    ctor_cfg.update({k: v for k, v in cfg.items() if k in HEAD_CTOR_KEYS[kind]})
    ctor_cfg["in_dim"] = in_dim
    ctor_cfg["n_keypoints"] = n_out // 3

    head = HEAD_CLASSES[kind](**ctor_cfg)
    head.load_state_dict(_head_tensors(path, head, sd), strict=True)
    head = head.to(device).eval()

    ctx = use_context if use_context is not None \
        else bool(cfg.get("use_context", True))
    bb = backbone or build_backbone(view_layout, in_dim, backbone_cfg, device)

    return KeypointReader(
        backbone=bb, head=head, view_layout=view_layout, use_context=ctx,
        robot_name=robot.name,
        reader_id=reader_id or f"keypoint/{view_layout.key}/{path}")
