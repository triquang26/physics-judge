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
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

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
