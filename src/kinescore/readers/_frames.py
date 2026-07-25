"""Shared frame-shape normalisation for readers.

:class:`~kinescore.core.reader.PoseReader` accepts either an unbatched
``(T,H,W,3)`` clip (any dtype -- ``uint8`` frames straight off disk, or
already-float) or a batched ``(B,T,3,H,W)`` float clip in ``[0,1]``. Every
reader in this package needs the same conversion to the backbone's input
contract (``(N,3,H,W)`` float in ``[0,1]``, ``N = B*T``), so it lives here
once instead of being reimplemented per reader.
"""
from __future__ import annotations

from typing import Tuple

import torch

__all__ = ["normalize_frames"]


def normalize_frames(frames: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """``(T,H,W,3)`` or ``(B,T,3,H,W)`` -> ``((N,3,H,W) float in [0,1], B, T)``.

    ``uint8`` input is scaled by ``1/255``; already-float input is passed
    through unchanged (assumed already in ``[0,1]`` per the reader contract).
    """
    if frames.dim() == 4 and frames.shape[-1] == 3:
        # (T,H,W,3) -- unbatched single clip.
        x = frames
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
        x = x.permute(0, 3, 1, 2).contiguous()  # (T,3,H,W)
        return x, 1, x.shape[0]
    if frames.dim() == 5:
        # (B,T,3,H,W) float in [0,1].
        B, T = frames.shape[0], frames.shape[1]
        x = frames.reshape(B * T, *frames.shape[2:]).float()
        return x, B, T
    raise ValueError(
        f"expected frames shaped (T,H,W,3) or (B,T,3,H,W), got {tuple(frames.shape)}")
