"""Per-DoF query attention + observability gate, and the FK-free keypoint path.

Ported **verbatim** (module math unchanged) from
``models.evaluation.pixel_judge_gr1.DisentangledPoseHead``. Not adapted for
this delivery: the GR-1-specific gamma-prior construction (kinematic-depth
init of the gate) and the aux waist/visibility head are judge-specific
wiring, not part of this class -- they belong to whatever robot plugin
composes it.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

__all__ = ["DisentangledPoseHead"]


class DisentangledPoseHead(nn.Module):
    """Per-DoF query attention + observability gate gamma + free-keypoint path.

    One learnable query per readout DoF (``n_joint``) gets its OWN soft
    attention map over patch tokens -- the disentangled "where each link
    looks". A per-DoF gate ``gamma`` mixes the locally-pooled feature with a
    global-context token, so occluded DoF (gamma -> 0) fall back to global
    context instead of faking a region. A parallel set of per-keypoint
    queries regresses UNCONSTRAINED 3-D keypoints ``P_free`` (no kinematic
    constraint) -- the "observation" for an FK-projection anomaly residual
    (``anomaly = gamma-weighted || P_free - FK(q) ||``, computed by the
    caller since it needs the robot's FK).

    Args:
        in_dim: Token embed width ``D``.
        n_joint: Number of scalar DoF queries (joint angles + any auxiliary
            flexion proxies).
        n_kp: Number of free-keypoint queries.
        hidden: MLP hidden width.
        dropout: MLP dropout.
        gamma_joint_init, gamma_kp_init: Optional ``(n_joint,)`` / ``(n_kp,)``
            initial gate values in ``(0,1)`` (converted to logits). ``None``
            (default) initialises gates at ``0`` (i.e. ``sigmoid(0)=0.5``).
    """

    def __init__(self, in_dim: int, n_joint: int = 29, n_kp: int = 12,
                 hidden: int = 512, dropout: float = 0.1,
                 gamma_joint_init: Optional[Sequence[float]] = None,
                 gamma_kp_init: Optional[Sequence[float]] = None) -> None:
        super().__init__()
        self.n_joint, self.n_kp = int(n_joint), int(n_kp)
        self.norm = nn.LayerNorm(in_dim)
        self.k_proj = nn.Linear(in_dim, in_dim)
        self.v_proj = nn.Linear(in_dim, in_dim)
        self.q_joint = nn.Parameter(torch.randn(n_joint, in_dim) * in_dim**-0.5)
        self.q_kp = nn.Parameter(torch.randn(n_kp, in_dim) * in_dim**-0.5)
        self.joint_mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(),
                                       nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.kp_mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(),
                                    nn.Dropout(dropout), nn.Linear(hidden, 3))
        self.gate_joint = nn.Parameter(self._init_gate(gamma_joint_init, n_joint))
        self.gate_kp = nn.Parameter(self._init_gate(gamma_kp_init, n_kp))
        self.scale = in_dim**-0.5

    @staticmethod
    def _init_gate(g: Optional[Sequence[float]], n: int) -> torch.Tensor:
        if g is None:
            return torch.zeros(n)
        gt = torch.clamp(torch.tensor(g, dtype=torch.float32), 0.05, 0.95)
        return torch.log(gt / (1 - gt))  # logit

    def _attend(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                gmean: torch.Tensor, gate: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = torch.einsum("btpd,qd->btqp", K, Q) * self.scale
        a = logits.softmax(-1)  # (B,T,Nq,P)
        pooled = torch.einsum("btqp,btpd->btqd", a, V)
        g = torch.sigmoid(gate).view(1, 1, -1, 1)
        return a, g * pooled + (1 - g) * gmean

    def forward(self, feat: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(B,T,P,D) -> (val_j (B,T,n_joint), P_free (B,T,n_kp,3), a_j, a_k)``."""
        x = self.norm(feat.float())
        K, V = self.k_proj(x), self.v_proj(x)
        gmean = V.mean(2, keepdim=True)
        a_j, feat_j = self._attend(self.q_joint, K, V, gmean, self.gate_joint)
        val_j = self.joint_mlp(feat_j).squeeze(-1)  # (B,T,n_joint)
        a_k, feat_k = self._attend(self.q_kp, K, V, gmean, self.gate_kp)
        P_free = self.kp_mlp(feat_k)  # (B,T,n_kp,3)
        return val_j, P_free, a_j, a_k
