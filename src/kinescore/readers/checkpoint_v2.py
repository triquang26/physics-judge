"""Save/load for :class:`~kinescore.heads.heteroscedastic.ReadoutV2Head` readers
-- the GR-1 / production-page path.

Why this file exists (the gap this closes)
-------------------------------------------
Before this module, ``kinescore.heads.heteroscedastic.ReadoutV2Head`` was
correctly ported (see that module's own docstring) but had **no
checkpoint-to-reader loader and no CLI wiring** -- ``cli/_scoring.py::build_scorer``
raised ``NotImplementedError`` for any checkpoint whose cfg wasn't an
:class:`~kinescore.heads.attentive.AttentivePoseHead`'s. That made the
Franka path (squashed, ``judge_v3l``/``judge_v3l_mv``) the *only* wired
reader, even though the GR-1 production judge
(``Marionette-fkjepa/models/evaluation/readout_v2_scorer.py::ReadoutV2Scorer``,
the thing the public kinescore page actually runs) is the ``ReadoutV2Head``
family. This module is that loader: :func:`load_head` mirrors
``readers/checkpoint.py::load``'s structure (cfg -> constructor kwargs ->
``load_state_dict(strict=True)``), :func:`load_reader` composes the loaded
head into a working :class:`~kinescore.core.reader.PoseReader`, and
:func:`is_readout_v2_cfg` is the shape-based cfg sniff ``readers/checkpoint.py
::load_reader`` (the single public auto-routing entry point) uses to decide
which loader a given ``.pt`` file needs.

Two things this loader does that a naive ``ReadoutV2Head(**cfg)`` would not
---------------------------------------------------------------------------
1. **sigma calibration.** ``readout_v2_gr1.pt``'s ``meta["sigma_scale"]`` is a
   post-hoc temperature (``sigma_cal = sigma_scale * sigma_raw``) fit on a
   clean validation split -- see
   ``models.posendf.readout_v2.fit_sigma_temperature`` and
   ``ReadoutV2Scorer._resolve_sigma_scale`` (precedence: explicit override >
   ``meta["sigma_scale"]`` > ``1.0``). Skipping it would silently under-report
   confidence by ~2x for the real checkpoint (``sigma_scale=1.9375``) -- every
   sigma-gate threshold tuned against the calibrated scale would pass roughly
   twice as many frames as intended. :func:`resolve_sigma_scale` ports that
   precedence rule; :class:`ReadoutV2PoseReader` applies it to every
   ``read()`` call, not just at load time, so it travels with the reader
   through an ensemble the same way ``HeteroscedasticPoseReader`` travels
   through :class:`~kinescore.readers.ensemble.EnsemblePoseReader`.
2. **the FK/aux split.** ``ReadoutV2Head`` predicts ``n_out=29`` dims for
   GR-1 (``17`` FK-relevant joints + ``12`` hand DoF -- see
   ``heads/heteroscedastic.py``'s module docstring), but
   ``kinescore.robots.gr1.spec.GR1Spec.n_joints == 17``: the hand DoF are not
   part of the FK chain this benchmark runs (``GR1FK`` ends at the wrist; see
   ``robots/gr1/fk.py``'s module docstring). The source's own
   ``ReadoutV2Scorer.score_feat`` handles this by splitting ``mu`` via
   ``ReadoutV2Head.split`` *before* ``clamp_for_fk`` -- only the first 17 dims
   ever reach FK; the trailing 12 are reported separately, never clamped
   against FK limits. :class:`ReadoutV2PoseReader` reproduces that split
   generically (parametrised by ``n_fk = robot.n_joints``, not hardcoded to
   17/12) on top of
   :class:`~kinescore.readers.heteroscedastic.HeteroscedasticPoseReader`,
   which -- by its own docstring -- treats every head output as a joint angle
   to clamp and expects the caller to supply permissive bounds for any
   auxiliary DoF; :func:`pad_permissive` is that padding. Without this split,
   ``Scorer.score_readout`` would hand a 29-wide ``q`` to
   ``GR1Spec.forward_transforms``, which asserts exactly 17
   (``GR1FK._full_theta``) and raises -- not a subtle numerical bug, but the
   loader would be unusable for the one checkpoint it exists to load.

Neither ``heads/heteroscedastic.py`` nor
``readers/heteroscedastic.py::HeteroscedasticPoseReader`` is modified to get
either behaviour -- both are composed here instead, so the already-verified
head math and the already-tested generic reader stay exactly as they are
(see ``tests/test_reader_limit_semantics.py``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import torch

from kinescore.core.clip import ViewLayout
from kinescore.core.reader import LimitSemantics, Readout
from kinescore.heads.heteroscedastic import ReadoutV2Head
from kinescore.readers.heteroscedastic import HeteroscedasticPoseReader

__all__ = [
    "LoadedReadoutV2Checkpoint", "is_readout_v2_cfg", "resolve_sigma_scale",
    "load_head", "save", "save_ensemble", "load_reader", "ReadoutV2PoseReader",
    "DEFAULT_ROBOT_NAME", "READOUT_V2_CTOR_KEYS", "pad_permissive",
    "build_default_backbone",
]

# ``ReadoutV2Head.__init__``'s keyword-argument set (kinescore's port -- see
# heads/heteroscedastic.py -- adds ``n_out`` as an explicit ctor arg where the
# source hardcodes ``N_OUT=29``; the real checkpoint's cfg has no ``n_out``
# key at all, so it falls through to that class's own default of 29, matching
# the source exactly). Filtering the saved cfg to this set is what lets
# `load_head` tolerate extra bookkeeping keys (``view_layout_key``,
# ``robot_name``, ...) this module's own :func:`save` adds on top of the
# source format, the same way `readers/checkpoint.py::load` tolerates
# `AttentivePoseHead`'s extra fields.
READOUT_V2_CTOR_KEYS = frozenset({
    "in_dim", "d_model", "n_heads", "temporal_nhead", "ff",
    "n_temporal_layers", "t_max", "dropout", "n_out", "logvar_min",
    "logvar_max",
})

# A cfg for this head family always carries these two -- neither key exists
# in any AttentivePoseHead cfg format this package targets (current
# ``save()`` format: n_joints/n_heads/n_cams/hidden/dropout/embed_dim/...;
# legacy ``judge_v3l``-vintage format: dino_model/dino_repo_dir/embed_dim/...
# -- verified by reading both, see readers/checkpoint.py and
# tests/test_checkpoint_legacy_cfg.py). Requiring both (not just one) is
# cheap insurance against a coincidental single-key collision.
_READOUT_V2_CFG_MARKERS = frozenset({"d_model", "temporal_nhead"})

#: Every real checkpoint at this head family targets GR-1 -- there is no
#: other target in either the source or this port (``ReadoutV2Head`` has no
#: multiview/embodiment parameter at all; see its module docstring). Used as
#: the historically-accurate default when a loaded cfg/meta carries no
#: explicit ``robot_name`` -- the real ``readout_v2_gr1.pt`` is exactly this
#: case (its cfg predates this package's ``robot_name`` convention).
DEFAULT_ROBOT_NAME = "fourier_gr1"

# A wide-but-finite bound for auxiliary (non-FK) output dims -- see
# `pad_permissive`. Finite (not +-inf) so `clamp_for_fk`'s `relu(q-hi)`
# arithmetic never produces a NaN from `inf - inf`, and so far outside any
# real hand-DoF reading (documented in `robots/gr1/fk.py` as living in
# ``[0, ~1.5]``) that it can never actually clamp a real prediction.
_PERMISSIVE_BOUND = 1.0e3

# Matches the production GR-1 judge backbone (see
# ``readout_v2_scorer.py``'s ``_DINO_DEFAULTS``): DINOv3-L @768 with
# patch_pool=2. Used only when a checkpoint's cfg carries no ``backbone``
# sub-dict of its own (true of the real ``readout_v2_gr1.pt`` -- its cfg is
# exactly the 10 ``ReadoutV2Head`` ctor keys, no backbone info at all) and no
# backbone was supplied by the caller.
_DEFAULT_BACKBONE_CFG: Dict[str, Any] = {
    "dino_model": "dinov3_vitl16", "dino_input": 768, "patch_pool": 2,
    "embed_dim": 1024, "hf_model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "n_register": 4,
}


def is_readout_v2_cfg(cfg: Dict[str, Any]) -> bool:
    """``True`` if ``cfg`` looks like a saved :class:`ReadoutV2Head`'s.

    The auto-routing check ``readers/checkpoint.py::load_reader`` uses to
    decide between the squashed :class:`~kinescore.heads.attentive.AttentivePoseHead`
    path and this one -- see module docstring and ``_READOUT_V2_CFG_MARKERS``.
    """
    return _READOUT_V2_CFG_MARKERS <= set(cfg)


def resolve_sigma_scale(meta: Dict[str, Any], n_out: int,
                        override: Optional[Any] = None) -> torch.Tensor:
    """Resolve the sigma temperature into a broadcastable fp32 cpu tensor.

    Mirrors ``ReadoutV2Scorer._resolve_sigma_scale`` exactly: precedence is
    explicit ``override`` > ``meta["sigma_scale"]`` > ``1.0`` (no-op). A
    scalar becomes shape ``(1,)``, a length-``n_out`` vector stays
    ``(n_out,)`` -- both broadcast against a ``(...,n_out)`` sigma tensor.
    Floored at ``1e-8`` so a corrupted/zero scale cannot silently zero out
    every reported sigma.
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
    GR-1's 12 hand DoF, for the real checkpoint. ``HeteroscedasticPoseReader``
    (see its own docstring) expects exactly this: give those dims bounds wide
    enough that clamping them is a no-op in practice, rather than excluding
    them. ``sign`` is ``-1.0`` for a lower-bound vector, ``+1.0`` for upper.
    A no-op (returns ``limits`` unchanged) when ``n_fk == n_out`` already.
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

    ``head`` is ready to use (``eval()``'d, weights loaded, ``strict=True``).
    The rest is provenance -- see module docstring for why ``sigma_scale``
    and ``robot_name`` need resolving rather than a bare ``cfg.get``.
    """

    head: ReadoutV2Head
    cfg: Dict[str, Any]
    meta: Dict[str, Any]
    sigma_scale: torch.Tensor
    robot_name: str
    view_layout_key: Optional[str]
    urdf_sha256: Optional[str]
    limit_semantics: LimitSemantics = "raw_rad"


def load_head(path: str, device: str = "cpu",
              sigma_scale: Optional[Any] = None) -> LoadedReadoutV2Checkpoint:
    """Load a :class:`ReadoutV2Head` from ``path``, ``strict=True``.

    Accepts both the source's own save format (``{"state_dict", "cfg",
    "meta"}``, e.g. ``readout_v2_gr1.pt``) and this module's own
    :func:`save` (same three keys, cfg extended with ``view_layout_key``/
    ``robot_name``). Raises ``ValueError`` -- naming what was found -- if
    ``path`` is not a ``ReadoutV2Head``-shaped checkpoint at all, rather than
    letting a wrong-family file fail deep inside ``load_state_dict`` with a
    confusing shape-mismatch trace.
    """
    ck = torch.load(path, map_location="cpu")
    sd = ck.get("state_dict")
    if sd is None:
        raise ValueError(
            f"{path!r} does not look like a ReadoutV2Head checkpoint "
            f"(no 'state_dict' key; got top-level keys {sorted(ck)})")
    cfg: Dict[str, Any] = dict(ck.get("cfg", {}))
    meta: Dict[str, Any] = dict(ck.get("meta", {}))
    if not is_readout_v2_cfg(cfg):
        raise ValueError(
            f"{path!r} cfg does not look like a ReadoutV2Head cfg (expected "
            f"{sorted(_READOUT_V2_CFG_MARKERS)} keys, got {sorted(cfg)})")

    ctor_cfg = {k: v for k, v in cfg.items() if k in READOUT_V2_CTOR_KEYS}
    head = ReadoutV2Head(**ctor_cfg)
    head.load_state_dict(sd, strict=True)
    head = head.to(device).eval()

    scale = resolve_sigma_scale(meta, head.n_out, override=sigma_scale)
    robot_name = cfg.get("robot_name") or meta.get("robot_name") or DEFAULT_ROBOT_NAME

    return LoadedReadoutV2Checkpoint(
        head=head, cfg=cfg, meta=meta, sigma_scale=scale, robot_name=robot_name,
        view_layout_key=cfg.get("view_layout_key"),
        urdf_sha256=meta.get("urdf_sha256"),
    )


def save(path: str, head: ReadoutV2Head, *, view_layout: ViewLayout,
         robot_name: str = DEFAULT_ROBOT_NAME,
         sigma_scale: Optional[Any] = None, urdf_sha256: Optional[str] = None,
         meta: Optional[Dict[str, Any]] = None) -> None:
    """Persist ``head`` in the source's own format, plus this package's
    reader-identity fields -- mirrors ``readers/checkpoint.py::save``'s
    relationship to ``PixelPhysicsJudge.save``, but for
    ``models.posendf.readout_v2.ReadoutV2Head.save`` instead: the 10-key cfg
    (``in_dim``...``logvar_max``) is exactly what the source persists (and
    what ``readout_v2_gr1.pt`` on disk actually has); ``view_layout_key``/
    ``robot_name`` are new, so a reader built from a file this function wrote
    declares its own identity instead of a caller having to guess it (see
    ``core/reader.py``). ``sigma_scale``/``urdf_sha256``, when given, are
    folded into ``meta`` under the same keys :func:`load_head` /
    ``ReadoutV2Scorer._resolve_sigma_scale`` read.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cfg: Dict[str, Any] = {
        "in_dim": head.in_dim, "d_model": head.d_model, "n_heads": head.n_heads,
        "temporal_nhead": head.temporal_nhead, "ff": head.ff,
        "n_temporal_layers": head.n_temporal_layers, "t_max": head.t_max,
        "dropout": head.dropout, "n_out": head.n_out,
        "logvar_min": head.logvar_min, "logvar_max": head.logvar_max,
        "view_layout_key": view_layout.key, "robot_name": robot_name,
    }
    m = dict(meta or {})
    if sigma_scale is not None:
        m["sigma_scale"] = sigma_scale
    if urdf_sha256 is not None:
        m["urdf_sha256"] = urdf_sha256
    torch.save({"state_dict": head.state_dict(), "cfg": cfg, "meta": m}, path)


def save_ensemble(path: str, heads: Sequence[ReadoutV2Head], *,
                  view_layout: ViewLayout, robot_name: str = DEFAULT_ROBOT_NAME,
                  sigma_scale: Optional[Any] = None,
                  urdf_sha256: Optional[str] = None,
                  meta: Optional[Dict[str, Any]] = None) -> None:
    """Persist ``K`` same-shaped heads as an ensemble bundle.

    Format: ``{"heads": [state_dict, ...], "cfg": <shared ctor cfg>, "meta":
    {...}}`` -- matches ``ReadoutV2Scorer._load_heads``'s ``ens`` branch (a
    dict with a ``"heads"`` list plus a shared ``cfg``), so a bundle this
    function writes is loadable by both the source scorer and
    :meth:`~kinescore.readers.ensemble.EnsemblePoseReader.from_checkpoint`.
    Every member must share the same ``cfg`` (same architecture) -- an
    ensemble of *different* head shapes is not what this format, or
    ``variance_decompose``, represents (the whole point is K forward passes
    of the *same* architecture to average).
    """
    if not heads:
        raise ValueError("save_ensemble needs at least one head")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ref = heads[0]
    cfg: Dict[str, Any] = {
        "in_dim": ref.in_dim, "d_model": ref.d_model, "n_heads": ref.n_heads,
        "temporal_nhead": ref.temporal_nhead, "ff": ref.ff,
        "n_temporal_layers": ref.n_temporal_layers, "t_max": ref.t_max,
        "dropout": ref.dropout, "n_out": ref.n_out,
        "logvar_min": ref.logvar_min, "logvar_max": ref.logvar_max,
        "view_layout_key": view_layout.key, "robot_name": robot_name,
    }
    m = dict(meta or {})
    if sigma_scale is not None:
        m["sigma_scale"] = sigma_scale
    if urdf_sha256 is not None:
        m["urdf_sha256"] = urdf_sha256
    torch.save({"heads": [h.state_dict() for h in heads], "cfg": cfg, "meta": m},
              path)


@dataclass
class ReadoutV2PoseReader:
    """Calibrated, FK/aux-splitting wrapper around
    :class:`~kinescore.readers.heteroscedastic.HeteroscedasticPoseReader`.

    See module docstring for why this exists instead of extending
    ``HeteroscedasticPoseReader`` itself. Delegates every
    :class:`~kinescore.core.reader.PoseReader` attribute to ``inner`` and
    post-processes :meth:`read`'s output in two steps, both no-ops when they
    don't apply:

    1. **calibrate.** ``sigma *= sigma_scale`` (see :func:`resolve_sigma_scale`).
    2. **split.** If ``inner``'s ``q``/``q_raw``/``sigma`` are wider than
       ``n_fk`` (the robot's actual FK joint count), the trailing dims move
       into ``Readout.aux["hand"]`` (+ ``"hand_raw"``/``"hand_sigma"`` when
       available) instead of being handed to FK. When ``n_fk`` already
       matches the head's output width (a robot with no auxiliary DoF), this
       is exactly a no-op -- ``read()`` returns ``inner.read()``'s own
       ``q``/``q_raw``/``aux`` unchanged (only sigma is calibrated).

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
        if n >= width:
            # No auxiliary dims to split off -- calibration only.
            return Readout(q=out.q, q_raw=out.q_raw, sigma=sigma, aux=out.aux,
                           extras=dict(out.extras))

        extras = dict(out.extras)
        clamp_rad = extras.pop("clamp_rad", None)
        aux: Dict[str, Any] = dict(out.aux) if isinstance(out.aux, dict) else {}
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
                            backbone_cfg: Optional[Dict[str, Any]],
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
                device: str = "cpu", reader_id: Optional[str] = None,
                backbone: Optional[Any] = None,
                backbone_cfg: Optional[Dict[str, Any]] = None,
                sigma_scale: Optional[Any] = None) -> ReadoutV2PoseReader:
    """Load ``path`` into a working :class:`ReadoutV2PoseReader` for ``robot``.

    Parameters
    ----------
    robot:
        A constructed :class:`~kinescore.core.robot.RobotSpec` (``.name``,
        ``.n_joints``, ``.q_lo``, ``.q_hi``). ``robot.n_joints`` may be less
        than the checkpoint's ``head.n_out`` (the GR-1 case: 17 < 29) -- the
        extra dims get permissive bounds and land in ``Readout.aux`` (see
        :class:`ReadoutV2PoseReader`); it may not be *greater* (a robot
        cannot need more FK joints than the head predicts at all).
    view_layout:
        Token-space camera layout. ``ReadoutV2Head`` has no native multiview
        support (see ``heads/heteroscedastic.py``); a non-default
        ``view_layout`` is composed in externally by
        ``HeteroscedasticPoseReader`` via
        :class:`~kinescore.heads.views.ViewEmbedding`.
    backbone / backbone_cfg:
        Supply a pre-built backbone directly (e.g. a test double, or one
        already loaded for another reader) or override the default cfg (see
        ``_DEFAULT_BACKBONE_CFG``); at most one is typically used.
    """
    loaded = load_head(path, device=device, sigma_scale=sigma_scale)
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
        robot_name=robot.name, reader_id=rid)
    return ReadoutV2PoseReader(inner=inner, sigma_scale=loaded.sigma_scale,
                               n_fk=robot.n_joints)
