"""K-member reader ensembles, and the aleatoric/epistemic variance split.

Generalises ``ReadoutV2Head.variance_decompose`` /
``ReadoutV2Scorer._disagreement`` from the source: there, ensembling was
wired specifically to ``ReadoutV2Head``'s ``mu``/``logvar`` dicts. Here
:func:`variance_decompose` operates on plain ``(mu, sigma)`` tensors instead,
so :class:`EnsemblePoseReader` can wrap *any* ``K`` members that satisfy
:class:`~kinescore.core.reader.PoseReader` -- ``K`` squashed heads, ``K``
heteroscedastic heads, or a mix -- rather than being specific to one head
class.

The two variance components answer different questions and must not be
collapsed into one number:

* **aleatoric** -- the mean of the members' own reported noise
  (``mean_k(sigma_k**2)``; zero for members with no ``sigma``, i.e. squashed
  heads). This is "how confident is a single member", and is present even
  with ``K=1``.
* **epistemic** -- the members' disagreement with each other
  (``var_k(mu_k)``, population variance so ``K=1`` gives exactly ``0`` rather
  than ``NaN``). This is the actual out-of-distribution signal: a scene the
  ensemble was never trained near produces disagreeing means even if every
  member is individually "confident".

``EnsemblePoseReader.from_checkpoint`` -- support it, don't fake it
----------------------------------------------------------------------
No real ``ReadoutV2Head`` ensemble bundle exists on disk today -- only the
single ``Marionette-fkjepa/model_ckpt/readout_v2_gr1.pt``. Rather than leave
ensemble support theoretical until one does, :meth:`EnsemblePoseReader.from_checkpoint`
reads *either* shape a file at that path could have: a single head (``K=1``)
or ``ReadoutV2Scorer``'s own ensemble-bundle format
(``{"heads": [state_dict, ...], "cfg": ..., "meta": ...}`` -- see
``readers/checkpoint_v2.py::save_ensemble``). The ``K=1`` case does **not**
construct an :class:`EnsemblePoseReader` around one member -- it returns that
member directly (a plain, calibrated
:class:`~kinescore.readers.checkpoint_v2.ReadoutV2PoseReader`) -- so
``epistemic``/member-disagreement are *structurally absent* from
``Readout.extras`` rather than the internal ``variance_decompose``
convention of a real (but degenerate) ``0`` for that case. Both are
"correct" in their own scope: ``variance_decompose`` returning ``0`` at
``K=1`` is what keeps ``total_variance = aleatoric + epistemic`` (hence
``sigma``) well-defined with one member, matching the source's own
``ReadoutV2Head.variance_decompose``/``_disagreement``. But a ``0`` *surfaced
to a caller* as "member disagreement" would read as "measured and found
zero" -- the same ``0``-vs-``unmeasured`` conflation this codebase's
``limit_semantics`` machinery exists to prevent for joint-limit violation
(defect D7, ``docs/PROVENANCE.md``). Omitting the key entirely at ``K=1`` is
the honest signal: the epistemic axis needs ``K>=2`` to mean anything, full
stop, and no caller can mistake an absent key for a measured zero.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch

from kinescore.core.clip import ViewLayout
from kinescore.core.reader import LimitSemantics, PoseReader, Readout

__all__ = ["variance_decompose", "EnsemblePoseReader"]


def variance_decompose(mus: torch.Tensor, sigmas: Optional[torch.Tensor] = None
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose an ensemble's predictive variance.

    Args:
        mus: Member means, stacked on axis 0: ``(K, ...)``.
        sigmas: Member aleatoric stds, same shape as ``mus``, or ``None`` if
            no member reports uncertainty (e.g. an ensemble of squashed
            heads) -- treated as zero aleatoric noise.

    Returns:
        ``(aleatoric, epistemic, total)``, each shaped like one member
        (``mus.shape[1:]``), where ``aleatoric = mean_k(sigma_k**2)``,
        ``epistemic = var_k(mu_k)`` (population variance, so identical
        members give exactly ``0``, not ``NaN``), ``total = aleatoric +
        epistemic``.
    """
    if mus.shape[0] == 0:
        raise ValueError("variance_decompose needs at least one member")
    mus = mus.float()
    if sigmas is not None:
        aleatoric = (sigmas.float() ** 2).mean(dim=0)
    else:
        aleatoric = torch.zeros_like(mus[0])
    epistemic = mus.var(dim=0, unbiased=False)
    total = aleatoric + epistemic
    return aleatoric, epistemic, total


class EnsemblePoseReader:
    """Wraps ``K`` :class:`~kinescore.core.reader.PoseReader` members.

    All members must agree on ``limit_semantics`` -- mixing a squashed and a
    raw-radian member would make ``q_raw`` and the clamp signal meaningless
    for half the ensemble, so this is checked at construction rather than
    left to silently average away.

    ``read`` runs every member (each does its own backbone forward -- this
    class does not share features between members, since members may use
    different backbones/checkpoints entirely) and returns the mean pose plus
    the aleatoric/epistemic decomposition in ``Readout.extras``.
    """

    def __init__(self, members: Sequence[PoseReader], robot_name: str,
                 view_layout: ViewLayout, reader_id: str) -> None:
        members = list(members)
        if not members:
            raise ValueError("EnsemblePoseReader needs at least one member")
        semantics = {m.limit_semantics for m in members}
        if len(semantics) != 1:
            raise ValueError(
                f"ensemble members disagree on limit_semantics: {semantics}")
        self.members: List[PoseReader] = members
        self.limit_semantics: LimitSemantics = members[0].limit_semantics
        self.robot_name = robot_name
        self.view_layout = view_layout
        self.reader_id = reader_id

    @classmethod
    def from_checkpoint(cls, path: str, *, robot: Any, view_layout: ViewLayout,
                        device: str = "cpu", backbone: Optional[Any] = None,
                        backbone_cfg: Optional[dict] = None,
                        reader_id: Optional[str] = None,
                        sigma_scale: Optional[Any] = None) -> PoseReader:
        """Load a ``ReadoutV2Head`` checkpoint at ``path`` into a reader.

        ``path`` may be a single-head file (``{"state_dict", "cfg", "meta"}``
        -- the real ``readout_v2_gr1.pt`` shape) or an ensemble bundle
        (``{"heads": [...], "cfg", "meta"}`` -- see
        ``readers/checkpoint_v2.py::save_ensemble``). See the module
        docstring's "``from_checkpoint`` -- support it, don't fake it"
        section for exactly what ``K=1`` returns and why.

        Every member shares one backbone (built once, or the caller-supplied
        ``backbone``) and one padded ``q_lo``/``q_hi`` -- see
        ``readers/checkpoint_v2.py::load_reader`` for the FK/aux split this
        also applies per member (``robot.n_joints`` may be less than the
        head's output width, e.g. GR-1's 17 FK joints + 12 hand DoF).
        """
        from kinescore.heads.heteroscedastic import ReadoutV2Head
        from kinescore.readers.checkpoint_v2 import (
            READOUT_V2_CTOR_KEYS,
            build_default_backbone,
            load_head,
            pad_permissive,
            resolve_sigma_scale,
        )
        from kinescore.readers.checkpoint_v2 import (
            ReadoutV2PoseReader as _RV2Reader,
        )
        from kinescore.readers.heteroscedastic import HeteroscedasticPoseReader

        ck = torch.load(path, map_location="cpu")
        if "heads" in ck:
            cfg = dict(ck.get("cfg", {}))
            meta = dict(ck.get("meta", {}))
            ctor_cfg = {k: v for k, v in cfg.items() if k in READOUT_V2_CTOR_KEYS}
            heads = []
            for sd in ck["heads"]:
                h = ReadoutV2Head(**ctor_cfg)
                h.load_state_dict(sd, strict=True)
                heads.append(h.to(device).eval())
            n_out = heads[0].n_out
            scale = resolve_sigma_scale(meta, n_out, override=sigma_scale)
        else:
            loaded = load_head(path, device=device, sigma_scale=sigma_scale)
            heads = [loaded.head]
            n_out = loaded.head.n_out
            scale = loaded.sigma_scale

        if robot.n_joints > n_out:
            raise ValueError(
                f"robot {robot.name!r} needs {robot.n_joints} FK joint(s) but "
                f"checkpoint {path!r} only predicts {n_out} output(s)")
        q_lo = pad_permissive(robot.q_lo, n_out, sign=-1.0)
        q_hi = pad_permissive(robot.q_hi, n_out, sign=+1.0)
        bb = backbone or build_default_backbone(view_layout, heads[0].in_dim,
                                                backbone_cfg, device)
        base_id = reader_id or f"readout_v2_ensemble/{view_layout.key}/{path}"

        members = [
            _RV2Reader(
                inner=HeteroscedasticPoseReader(
                    backbone=bb, head=h, q_lo=q_lo, q_hi=q_hi,
                    view_layout=view_layout, robot_name=robot.name,
                    reader_id=f"{base_id}#member{i}"),
                sigma_scale=scale, n_fk=robot.n_joints)
            for i, h in enumerate(heads)
        ]
        if len(members) == 1:
            # K=1 degrades to a plain heteroscedastic reader -- see module
            # docstring. `Readout.extras` then has no "epistemic"/"aleatoric"
            # key at all, rather than the internal variance_decompose 0.
            return members[0]
        return cls(members=members, robot_name=robot.name, view_layout=view_layout,
                   reader_id=base_id)

    def read(self, frames: torch.Tensor) -> Readout:
        outs = [m.read(frames) for m in self.members]

        # mu for the variance decomposition: q_raw when the head exposes
        # unconstrained angles, else the (already-valid) q -- squashed
        # members have no other notion of "mean prediction".
        mus = torch.stack([o.q_raw if o.q_raw is not None else o.q for o in outs],
                          dim=0)
        have_sigma = all(o.sigma is not None for o in outs)
        sigmas = torch.stack([o.sigma for o in outs], dim=0) if have_sigma else None
        aleatoric, epistemic, total = variance_decompose(mus, sigmas)

        q_stack = torch.stack([o.q for o in outs], dim=0)
        q = q_stack.mean(dim=0)
        q_raw = mus.mean(dim=0) if outs[0].q_raw is not None else None
        sigma = total.clamp_min(0).sqrt()

        return Readout(
            q=q, q_raw=q_raw, sigma=sigma, aux=outs[0].aux,
            extras={
                "aleatoric": aleatoric,
                "epistemic": epistemic,
                "member_q": q_stack,
                "n_members": len(self.members),
            },
        )
