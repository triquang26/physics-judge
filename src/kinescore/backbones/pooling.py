"""Patch-token spatial pooling."""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["pool_patch_tokens"]


def pool_patch_tokens(tokens: torch.Tensor, k: int) -> torch.Tensor:
    """Average-pool a square patch-token grid by factor ``k``.

    ``tokens``: ``(..., P, D)`` with ``P`` a perfect square (the DINO patch
    grid, e.g. 16x16=256). Returns ``(..., P/k**2, D)``. ``k<=1`` is a no-op.

    Shrinks the cached patch grid while keeping spatial structure for the
    attentive head. ``patch_pool`` is part of the backbone identity, so every
    stage pools the same way and a head always sees the token layout it
    trained on.
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
