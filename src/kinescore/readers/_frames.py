"""Shared frame-shape normalisation for readers."""
from __future__ import annotations

import torch

__all__ = ["normalize_frames"]


def normalize_frames(frames: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """``(T,H,W,3)``/``(T,3,H,W)``/``(B,T,3,H,W)`` -> ``((N,3,H,W) float, B, T)``.
    """
    if frames.dim() == 4:
        channels_last = frames.shape[-1] == 3
        channels_first = frames.shape[1] == 3
        if channels_last and channels_first:
            raise ValueError(
                "ambiguous 4-D frames "
                f"{tuple(frames.shape)}: axis 1 and axis 3 are both size 3, so "
                "(T,3,H,W) and (T,H,W,3) cannot be told apart. Pass a batched "
                "(B,T,3,H,W) tensor instead of guessing.")
        if not (channels_last or channels_first):
            raise ValueError(
                f"expected frames shaped (T,H,W,3), (T,3,H,W) or (B,T,3,H,W), "
                f"got {tuple(frames.shape)}: no axis of size 3 to read as "
                "colour channels")
        x = frames
        x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()
        if channels_last:  # (T,H,W,3) -> (T,3,H,W)
            x = x.permute(0, 3, 1, 2)
        return x.contiguous(), 1, x.shape[0]
    if frames.dim() == 5:
        # (B,T,3,H,W) float in [0,1].
        B, T = frames.shape[0], frames.shape[1]
        x = frames.reshape(B * T, *frames.shape[2:]).float()
        return x, B, T
    raise ValueError(
        "expected frames shaped (T,H,W,3), (T,3,H,W) or (B,T,3,H,W), got "
        f"{tuple(frames.shape)}")
