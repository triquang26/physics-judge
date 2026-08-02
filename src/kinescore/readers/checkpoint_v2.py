"""Save/load for :class:`~kinescore.heads.heteroscedastic.ReadoutV2Head` readers
-- the GR-1 / production-page path.

Closes a gap where this head family (already correctly ported) had no
checkpoint-to-reader loader or CLI wiring at all, leaving only the secondary
Franka/squashed path runnable (that path has since been removed entirely --
see ``legacy_docs/PROVENANCE.md``'s D7 addendum -- so this is now the ONLY reader
load path). The pattern is the usual one for this package (cfg -> constructor
kwargs -> ``load_state_dict(strict=True)``), plus two
things a bare ``ReadoutV2Head(**cfg)`` would miss: sigma calibration
(:func:`resolve_sigma_scale`) and the FK/aux dimension split
(:class:`ReadoutV2PoseReader`, GR-1's 29 head outputs vs its 17 FK joints).
Neither ``heads/heteroscedastic.py`` nor ``readers/heteroscedastic.py`` is
modified to get either behaviour -- both are composed here instead. See
``legacy_docs/PROVENANCE.md`` D10 for the full rationale (why the loader was
missing, the sigma-scale precedence, the FK/aux split, and why
``use_context`` must be resolved per-checkpoint).

This module also carries :func:`load_direct_keypoint_reader`, the loader for
the sibling direct-keypoint head family (same ``ReadoutV2Head`` architecture,
different ``n_out``/decode -- see ``readers/direct_keypoint.py``). It reads a
different on-disk shape (the training side's own ``{"head", "n_out",
"embed_dim", "cfg"}``, not this module's ``{"state_dict", "cfg", "meta"}``),
so it duck-types its own cfg detection (:func:`is_direct_keypoint_cfg`)
rather than reusing :func:`load_head`.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Any

import torch

from kinescore.core.clip import ViewLayout
from kinescore.core.reader import LimitSemantics, Readout
from kinescore.heads.heteroscedastic import ReadoutV2Head
from kinescore.readers.heteroscedastic import HeteroscedasticPoseReader

__all__ = [
    "LoadedReadoutV2Checkpoint", "is_readout_v2_cfg", "resolve_sigma_scale",
    "resolve_use_context", "load_head", "save", "load_reader",
    "ReadoutV2PoseReader", "DEFAULT_ROBOT_NAME", "READOUT_V2_CTOR_KEYS",
    "pad_permissive", "build_default_backbone",
    "HEAD_TARGET_JOINTS", "HEAD_TARGET_KEYPOINTS", "is_direct_keypoint_cfg",
    "load_direct_keypoint_reader",
]

#: What a checkpoint's head regresses. ``"joints"`` is the ``ReadoutV2Head`` /
#: joint-angle+FK family this whole module was written for (:func:`save`'s
#: default -- unchanged behaviour for every checkpoint written before this
#: field existed). ``"keypoints"`` is the direct 3-D keypoint family (see
#: :func:`load_direct_keypoint_reader` and
#: ``readers/direct_keypoint.py::DirectKeypointPoseReader``) -- a checkpoint
#: declaring this has no joint angles, no FK, and no ``ReadoutV2PoseReader``
#: clamp/split logic in its path at all.
HEAD_TARGET_JOINTS = "joints"
HEAD_TARGET_KEYPOINTS = "keypoints"

#: Defaults the direct-keypoint head family was actually trained with
#: (``training/train_kp_head_v3.py``-equivalent) -- bigger than
#: ``ReadoutV2Head.__init__``'s own defaults (which target the smaller,
#: original ``readout_v2`` head). Used by :func:`load_direct_keypoint_reader`
#: for whatever :data:`READOUT_V2_CTOR_KEYS` a checkpoint's ``cfg`` omits, so
#: a minimal ``{"head_target": "keypoints"}`` cfg (no per-layer overrides)
#: still reconstructs the right architecture rather than silently falling
#: back to the smaller ``ReadoutV2Head`` defaults.
_DIRECT_KEYPOINT_HEAD_DEFAULTS: dict[str, Any] = {
    "d_model": 768, "n_heads": 8, "temporal_nhead": 8, "ff": 2048,
    "n_temporal_layers": 4, "t_max": 64, "dropout": 0.1,
}

#: ``ReadoutV2Head.__init__``'s keyword-argument set. Filtering the saved cfg
#: to this set is what lets :func:`load_head` tolerate extra bookkeeping keys
#: (``view_layout_key``, ``robot_name``, ...) :func:`save` adds on top of the
#: source format.
READOUT_V2_CTOR_KEYS = frozenset({
    "in_dim", "d_model", "n_heads", "temporal_nhead", "ff",
    "n_temporal_layers", "t_max", "dropout", "n_out", "logvar_min",
    "logvar_max",
})

#: A cfg for this head family always carries these two; neither key exists in
#: any ``AttentivePoseHead`` cfg format (current or legacy) this package
#: targets. Requiring both, not one, guards against a coincidental collision.
_READOUT_V2_CFG_MARKERS = frozenset({"d_model", "temporal_nhead"})

#: Default ``robot_name`` when a loaded cfg/meta has none -- every real
#: checkpoint at this head family targets GR-1 (``readout_v2_gr1.pt`` predates
#: this package's ``robot_name`` convention). Falling back to this is a
#: silent footgun if the checkpoint is not actually GR-1 (it is how
#: ``humanoid.pt``, which carries no ``robot_name`` at all, gets its robot),
#: so :func:`load_head` warns -- naming the file and this value -- every time
#: it actually falls back, rather than silently assuming.
DEFAULT_ROBOT_NAME = "fourier_gr1"

#: Wide-but-finite bound for auxiliary (non-FK) output dims (see
#: :func:`pad_permissive`) -- finite so ``clamp_for_fk``'s hinge never
#: produces a NaN from ``inf - inf``, and far outside any real hand-DoF
#: reading so it never actually clamps a real prediction.
_PERMISSIVE_BOUND = 1.0e3

#: Matches the production GR-1 judge backbone: DINOv3-L @768, patch_pool=2.
#: Used only when a checkpoint's cfg has no ``backbone`` sub-dict of its own
#: and none was supplied by the caller.
_DEFAULT_BACKBONE_CFG: dict[str, Any] = {
    "dino_model": "dinov3_vitl16", "dino_input": 768, "patch_pool": 2,
    "embed_dim": 1024, "hf_model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "n_register": 4,
}


def is_readout_v2_cfg(cfg: dict[str, Any]) -> bool:
    """``True`` if ``cfg`` looks like a saved :class:`ReadoutV2Head`'s.

    The auto-routing check ``readers/loader.py::load_reader`` uses to pick
    this loader over rejecting a legacy, now-unsupported checkpoint format --
    see ``legacy_docs/PROVENANCE.md`` D7's addendum and D10.
    """
    return _READOUT_V2_CFG_MARKERS <= set(cfg)


def resolve_use_context(meta: dict[str, Any], override: bool | None = None) -> bool:
    """Resolve whether a loaded head should run with temporal context at
    inference (``ReadoutV2Head.forward``'s ``use_context`` flag).

    Precedence: explicit ``override`` > ``meta["use_context"]`` > ``True``.
    This must be a per-checkpoint choice, not always ``True`` -- see
    ``legacy_docs/PROVENANCE.md`` D10 for the double-bug a mismatched default causes
    on a checkpoint whose ``TemporalEncoder`` was never trained.
    """
    if override is not None:
        return bool(override)
    return bool(meta.get("use_context", True))


def resolve_sigma_scale(meta: dict[str, Any], n_out: int,
                        override: Any | None = None) -> torch.Tensor:
    """Resolve the sigma temperature into a broadcastable fp32 cpu tensor.

    Precedence: explicit ``override`` > ``meta["sigma_scale"]`` > ``1.0``
    (no-op) -- see ``legacy_docs/PROVENANCE.md`` D10. A scalar becomes shape
    ``(1,)``, a length-``n_out`` vector stays ``(n_out,)``. Floored at
    ``1e-8`` so a corrupted/zero scale can't silently zero out every sigma.
    """
    val = override if override is not None else meta.get("sigma_scale", 1.0)
    t = torch.as_tensor(val, dtype=torch.float32).reshape(-1)
    if t.numel() not in (1, n_out):
        raise ValueError(
            f"sigma_scale must be a scalar or length-{n_out} vector, got "
            f"{t.numel()} element(s)")
    return t.clamp_min(1e-8)


def pad_permissive(limits: torch.Tensor, n_out: int, *, sign: float
                    ) -> torch.Tensor:
    """Extend a ``(n_fk,)`` limit vector to ``(n_out,)`` with permissive bounds.

    The extra ``n_out - n_fk`` dims are auxiliary (non-FK) head outputs --
    GR-1's 12 hand DoF. ``sign`` is ``-1.0`` for a lower bound, ``+1.0`` for
    upper. No-op when ``n_fk == n_out`` already.
    """
    n_fk = int(limits.shape[0])
    if n_fk == n_out:
        return limits
    if n_fk > n_out:
        raise ValueError(
            f"limits has {n_fk} entries but the head only predicts {n_out} "
            f"output(s) -- cannot pad a NEGATIVE number of dims")
    pad = torch.full((n_out - n_fk,), sign * _PERMISSIVE_BOUND,
                     dtype=limits.dtype, device=limits.device)
    return torch.cat([limits, pad], dim=0)


@dataclass(frozen=True)
class LoadedReadoutV2Checkpoint:
    """Everything :func:`load_head` recovers from a saved :class:`ReadoutV2Head`.

    ``head`` is ready to use (``eval()``'d, weights loaded, ``strict=True``);
    the rest is resolved provenance (see ``legacy_docs/PROVENANCE.md`` D10).
    """

    head: ReadoutV2Head
    cfg: dict[str, Any]
    meta: dict[str, Any]
    sigma_scale: torch.Tensor
    robot_name: str
    view_layout_key: str | None
    urdf_sha256: str | None
    use_context: bool = True
    limit_semantics: LimitSemantics = "raw_rad"


def load_head(path: str, device: str = "cpu",
              sigma_scale: Any | None = None,
              use_context: bool | None = None) -> LoadedReadoutV2Checkpoint:
    """Load a :class:`ReadoutV2Head` from ``path``, ``strict=True``.

    Accepts both the source's own save format (``{"state_dict", "cfg",
    "meta"}``, e.g. ``readout_v2_gr1.pt``) and :func:`save`'s (same keys,
    cfg extended with ``view_layout_key``/``robot_name``). Raises
    ``ValueError`` naming what was found if ``path`` isn't
    ``ReadoutV2Head``-shaped, rather than failing deep inside
    ``load_state_dict`` with a confusing shape-mismatch trace.
    """
    ck = torch.load(path, map_location="cpu")
    sd = ck.get("state_dict")
    if sd is None:
        raise ValueError(
            f"{path!r} does not look like a ReadoutV2Head checkpoint "
            f"(no 'state_dict' key; got top-level keys {sorted(ck)})")
    cfg: dict[str, Any] = dict(ck.get("cfg", {}))
    meta: dict[str, Any] = dict(ck.get("meta", {}))
    if not is_readout_v2_cfg(cfg):
        raise ValueError(
            f"{path!r} cfg does not look like a ReadoutV2Head cfg (expected "
            f"{sorted(_READOUT_V2_CFG_MARKERS)} keys, got {sorted(cfg)})")

    ctor_cfg = {k: v for k, v in cfg.items() if k in READOUT_V2_CTOR_KEYS}
    head = ReadoutV2Head(**ctor_cfg)
    head.load_state_dict(sd, strict=True)
    head = head.to(device).eval()

    scale = resolve_sigma_scale(meta, head.n_out, override=sigma_scale)
    ctx = resolve_use_context(meta, override=use_context)
    robot_name = cfg.get("robot_name") or meta.get("robot_name")
    if not robot_name:
        robot_name = DEFAULT_ROBOT_NAME
        warnings.warn(
            f"{path!r} has no robot_name in cfg or meta -- assuming "
            f"{DEFAULT_ROBOT_NAME!r} (DEFAULT_ROBOT_NAME). If this checkpoint "
            f"does not target that robot, its reader will be silently paired "
            f"with the wrong joint limits/FK; re-save it with an explicit "
            f"robot_name or pass one to the caller that resolves this reader.",
            stacklevel=2)

    return LoadedReadoutV2Checkpoint(
        head=head, cfg=cfg, meta=meta, sigma_scale=scale, robot_name=robot_name,
        view_layout_key=cfg.get("view_layout_key"),
        urdf_sha256=meta.get("urdf_sha256"), use_context=ctx,
    )


def save(path: str, head: ReadoutV2Head, *, view_layout: ViewLayout,
         robot_name: str = DEFAULT_ROBOT_NAME,
         sigma_scale: Any | None = None, urdf_sha256: str | None = None,
         meta: dict[str, Any] | None = None,
         head_target: str = HEAD_TARGET_JOINTS) -> None:
    """Persist ``head`` in the source's own format, plus this package's
    reader-identity fields (``view_layout_key``/``robot_name`` -- see
    ``core/reader.py``). ``sigma_scale``/``urdf_sha256``, when given, are
    folded into ``meta`` under the keys :func:`load_head` reads back.

    ``head_target`` declares what ``head`` regresses -- :data:`HEAD_TARGET_JOINTS`
    (the default, unchanged behaviour for every caller that predates this
    parameter) for the joint-angle+FK family this module was written for, or
    :data:`HEAD_TARGET_KEYPOINTS` for a head whose ``n_out`` is ``3 *
    n_keypoints``. This function does not itself route on the value -- it
    only records it in ``cfg`` for :func:`~kinescore.readers.loader.load_reader`
    (or a direct :func:`is_direct_keypoint_cfg` check) to dispatch on later;
    a keypoints-targeted head saved this way still round-trips through
    :func:`load_head`/:func:`load_reader` in this module unchanged (they
    never inspect ``head_target``), but a keypoint checkpoint would normally
    be loaded via :func:`load_direct_keypoint_reader` instead, which does.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cfg: dict[str, Any] = {
        "in_dim": head.in_dim, "d_model": head.d_model, "n_heads": head.n_heads,
        "temporal_nhead": head.temporal_nhead, "ff": head.ff,
        "n_temporal_layers": head.n_temporal_layers, "t_max": head.t_max,
        "dropout": head.dropout, "n_out": head.n_out,
        "logvar_min": head.logvar_min, "logvar_max": head.logvar_max,
        "view_layout_key": view_layout.key, "robot_name": robot_name,
        "head_target": head_target,
    }
    m = dict(meta or {})
    if sigma_scale is not None:
        m["sigma_scale"] = sigma_scale
    if urdf_sha256 is not None:
        m["urdf_sha256"] = urdf_sha256
    torch.save({"state_dict": head.state_dict(), "cfg": cfg, "meta": m}, path)


@dataclass
class ReadoutV2PoseReader:
    """Calibrated, FK/aux-splitting wrapper around
    :class:`~kinescore.readers.heteroscedastic.HeteroscedasticPoseReader`.

    Delegates every :class:`~kinescore.core.reader.PoseReader` attribute to
    ``inner`` and post-processes :meth:`read`'s output in two steps, both
    no-ops when they don't apply: **calibrate** (``sigma *= sigma_scale``)
    and **split** (dims beyond ``n_fk`` move into ``Readout.aux["hand"]``
    instead of reaching FK). See ``legacy_docs/PROVENANCE.md`` D10.

    Parameters
    ----------
    inner:
        A constructed :class:`HeteroscedasticPoseReader` (backbone + head +
        padded ``q_lo``/``q_hi`` already wired up by :func:`load_reader`).
    sigma_scale:
        ``(1,)`` or ``(n_out,)`` fp32 tensor, from :func:`resolve_sigma_scale`.
    n_fk:
        Number of leading output dims that are real FK joints (``robot.n_joints``).
    """

    inner: HeteroscedasticPoseReader
    sigma_scale: torch.Tensor
    n_fk: int
    limit_semantics: LimitSemantics = field(default="raw_rad", init=False)

    def __post_init__(self) -> None:
        if self.n_fk < 1:
            raise ValueError(f"n_fk must be >= 1, got {self.n_fk}")

    @property
    def view_layout(self) -> ViewLayout:
        return self.inner.view_layout

    @property
    def robot_name(self) -> str:
        return self.inner.robot_name

    @property
    def reader_id(self) -> str:
        return self.inner.reader_id

    def read(self, frames: torch.Tensor) -> Readout:
        """``(T,H,W,3)`` uint8 or ``(B,T,3,H,W)`` float in ``[0,1]`` -> Readout."""
        out = self.inner.read(frames)
        sigma = out.sigma
        if sigma is not None:
            scale = self.sigma_scale.to(device=sigma.device, dtype=sigma.dtype)
            sigma = sigma * scale

        n = self.n_fk
        width = out.q.shape[-1]
        if n >= width:  # no auxiliary dims to split off -- calibration only
            return Readout(q=out.q, q_raw=out.q_raw, sigma=sigma, aux=out.aux,
                           extras=dict(out.extras))

        extras = dict(out.extras)
        clamp_rad = extras.pop("clamp_rad", None)
        aux: dict[str, Any] = dict(out.aux) if isinstance(out.aux, dict) else {}
        aux["hand"] = out.q[..., n:]
        if out.q_raw is not None:
            aux["hand_raw"] = out.q_raw[..., n:]
        if sigma is not None:
            aux["hand_sigma"] = sigma[..., n:]
        if clamp_rad is not None:
            extras["clamp_rad"] = clamp_rad[..., :n]
            extras["clamp_rad_hand"] = clamp_rad[..., n:]

        return Readout(
            q=out.q[..., :n],
            q_raw=None if out.q_raw is None else out.q_raw[..., :n],
            sigma=None if sigma is None else sigma[..., :n],
            aux=aux, extras=extras,
        )


def build_default_backbone(view_layout: ViewLayout, embed_dim: int,
                            backbone_cfg: dict[str, Any] | None,
                            device: str):
    """Construct the default GR-1-judge-matching backbone (see
    ``_DEFAULT_BACKBONE_CFG``). Network-free: :class:`FeatureBackbone` loads
    its frozen weights lazily on first ``.encode()`` call, not here."""
    from kinescore.backbones.dino import FeatureBackbone

    cfg = dict(_DEFAULT_BACKBONE_CFG)
    if backbone_cfg:
        cfg.update(backbone_cfg)
    cfg.setdefault("embed_dim", embed_dim)
    bb = FeatureBackbone(view_layout=view_layout, **cfg)
    return bb.to(device).eval()


def load_reader(path: str, *, robot: Any, view_layout: ViewLayout,
                device: str = "cpu", reader_id: str | None = None,
                backbone: Any | None = None,
                backbone_cfg: dict[str, Any] | None = None,
                sigma_scale: Any | None = None,
                use_context: bool | None = None) -> ReadoutV2PoseReader:
    """Load ``path`` into a working :class:`ReadoutV2PoseReader` for ``robot``.

    Parameters
    ----------
    robot:
        A constructed :class:`~kinescore.core.robot.RobotSpec`.
        ``robot.n_joints`` may be less than the checkpoint's ``head.n_out``
        (extra dims land in ``Readout.aux`` -- see
        :class:`ReadoutV2PoseReader`), never greater.
    view_layout:
        Token-space camera layout; ``ReadoutV2Head`` has no native multiview
        support, so a non-default layout is composed in externally by
        ``HeteroscedasticPoseReader`` via
        :class:`~kinescore.heads.views.ViewEmbedding`.
    backbone / backbone_cfg:
        Supply a pre-built backbone directly, or override the default cfg;
        at most one is typically used.
    use_context:
        Override for :func:`resolve_use_context`. ``None`` (default) trusts
        the checkpoint's own declaration -- see ``legacy_docs/PROVENANCE.md`` D10.
    """
    loaded = load_head(path, device=device, sigma_scale=sigma_scale,
                       use_context=use_context)
    head = loaded.head
    n_out = head.n_out
    if robot.n_joints > n_out:
        raise ValueError(
            f"robot {robot.name!r} needs {robot.n_joints} FK joint(s) but "
            f"checkpoint {path!r} only predicts {n_out} output(s)")

    q_lo = pad_permissive(robot.q_lo, n_out, sign=-1.0)
    q_hi = pad_permissive(robot.q_hi, n_out, sign=+1.0)
    bb = backbone or build_default_backbone(view_layout, head.in_dim,
                                             backbone_cfg, device)
    rid = reader_id or f"readout_v2/{view_layout.key}/{path}"

    inner = HeteroscedasticPoseReader(
        backbone=bb, head=head, q_lo=q_lo, q_hi=q_hi, view_layout=view_layout,
        use_context=loaded.use_context,
        robot_name=robot.name, reader_id=rid)
    return ReadoutV2PoseReader(inner=inner, sigma_scale=loaded.sigma_scale,
                               n_fk=robot.n_joints)


def is_direct_keypoint_cfg(cfg: dict[str, Any]) -> bool:
    """``True`` if ``cfg`` declares a direct-keypoint checkpoint.

    The auto-routing check ``readers/loader.py::load_reader`` uses *ahead
    of* :func:`is_readout_v2_cfg`: a minimal keypoint cfg (the training side
    may write nothing but ``{"head_target": "keypoints"}``, letting
    :func:`load_direct_keypoint_reader` fill in the rest) carries none of
    that function's ``d_model``/``temporal_nhead`` markers, so it must be
    checked first rather than falling through to "not a ReadoutV2Head cfg".
    """
    return cfg.get("head_target") == HEAD_TARGET_KEYPOINTS


def load_direct_keypoint_reader(path: str, *, robot: Any, view_layout: ViewLayout,
                                device: str = "cpu", reader_id: str | None = None,
                                backbone: Any | None = None,
                                backbone_cfg: dict[str, Any] | None = None,
                                use_context: bool | None = None,
                                ):
    """Load a direct-keypoint checkpoint into a working
    :class:`~kinescore.readers.direct_keypoint.DirectKeypointPoseReader`.

    Accepts the training side's own save format -- ``{"head": state_dict,
    "n_out": 3*K, "embed_dim": D, "cfg": {..., "head_target": "keypoints"}}``
    -- which is *not* :func:`save`'s ``{"state_dict", "cfg", "meta"}`` shape
    (that format targets the joint-angle ``ReadoutV2Head`` family loaded by
    :func:`load_head`/:func:`load_reader`; a keypoint checkpoint saved
    through :func:`save` instead would need its own state-dict key
    translated before this function could read it -- it is not currently
    wired to do so). ``cfg`` is optional; whatever :data:`READOUT_V2_CTOR_KEYS`
    it omits falls back to ``_DIRECT_KEYPOINT_HEAD_DEFAULTS`` (``d_model=768``,
    ``n_heads=8``, ``temporal_nhead=8``, ``ff=2048``, ``n_temporal_layers=4``,
    ``t_max=64``, ``dropout=0.1``) -- the architecture this head family was
    actually trained with, not ``ReadoutV2Head``'s own (smaller) constructor
    defaults.

    Parameters
    ----------
    robot:
        A constructed :class:`~kinescore.core.robot.RobotSpec` (or any
        duck-typed stand-in with a ``.name``). Used only for
        :attr:`~kinescore.readers.direct_keypoint.DirectKeypointPoseReader.robot_name`
        -- this reader never runs FK, so ``robot.q_lo``/``q_hi``/``n_joints``
        are not touched.
    view_layout, backbone, backbone_cfg:
        See :func:`load_reader` -- same composition, no keypoint-specific
        difference.
    use_context:
        Override for :func:`resolve_use_context`; resolved against ``cfg``
        merged with any ``meta`` dict the checkpoint also carries (``None``
        trusts the checkpoint's own declaration, default ``True``).

    Raises
    ------
    ValueError
        If ``path`` has no ``"head"`` key (not this format at all), or its
        ``n_out``/``embed_dim`` cannot be resolved or don't imply an integer
        number of keypoints.
    """
    ck = torch.load(path, map_location="cpu")
    sd = ck.get("head")
    if sd is None:
        raise ValueError(
            f"{path!r} does not look like a direct-keypoint checkpoint (no "
            f"'head' key; got top-level keys {sorted(ck)})")
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

    ctor_cfg: dict[str, Any] = dict(_DIRECT_KEYPOINT_HEAD_DEFAULTS)
    ctor_cfg.update({k: v for k, v in cfg.items() if k in READOUT_V2_CTOR_KEYS})
    ctor_cfg["in_dim"] = in_dim
    ctor_cfg["n_out"] = n_out

    head = ReadoutV2Head(**ctor_cfg)
    head.load_state_dict(sd, strict=True)
    head = head.to(device).eval()

    meta_like = {**cfg, **dict(ck.get("meta", {}))}
    ctx = resolve_use_context(meta_like, override=use_context)
    bb = backbone or build_default_backbone(view_layout, in_dim, backbone_cfg,
                                            device)
    rid = reader_id or f"direct_keypoint/{view_layout.key}/{path}"

    from kinescore.readers.direct_keypoint import DirectKeypointPoseReader

    return DirectKeypointPoseReader(
        backbone=bb, head=head, n_keypoints=n_out // 3, view_layout=view_layout,
        use_context=ctx, robot_name=robot.name, reader_id=rid)
