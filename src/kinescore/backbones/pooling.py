"""Patch-token spatial pooling.

Ported **verbatim** from ``judge/pixel_judge.py::pool_patch_tokens`` in the
Marionette source. It is the one piece of the backbone that is genuinely
model-agnostic arithmetic (average-pooling a square grid), so there is no
reason to touch it -- and every reason not to, since ``patch_pool`` is baked
into every real checkpoint's cfg (``judge_v3l`` etc. use ``patch_pool=2``) and
a behavioural change here would silently shift what tokens a loaded head sees.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["pool_patch_tokens"]


def pool_patch_tokens(tokens: torch.Tensor, k: int) -> torch.Tensor:
    """Average-pool a square patch-token grid by factor ``k``.

    ``tokens``: ``(..., P, D)`` with ``P`` a perfect square (the DINO patch
    grid, e.g. 16x16=256). Returns ``(..., P/k**2, D)``. ``k<=1`` is a no-op.

    Used to shrink the cached patch grid (NFS/RAM cost) while keeping spatial
    structure for the attentive head -- and it must be applied identically at
    scoring time so the head always sees the same token layout it trained on.
    This is why it lives in ``backbones/`` rather than being inlined in a
    reader: every reader that shares a backbone must pool the same way.
    """
    if k <= 1:
        return tokens
    lead = tokens.shape[:-2]
    P, D = tokens.shape[-2], tokens.shape[-1]
    G = int(round(P**0.5))
    if G * G != P:
        raise ValueError(f"patch count {P} is not a square grid")
    x = tokens.reshape(-1, G, G, D).permute(0, 3, 1, 2)  # (B, D, G, G)
    x = F.avg_pool2d(x.float(), k).to(tokens.dtype)  # (B, D, G/k, G/k)
    x = x.permute(0, 2, 3, 1).reshape(*lead, -1, D)  # (..., (G/k)**2, D)
    return x
