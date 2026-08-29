"""Decode frames for a :class:`~kinescore.core.clip.ClipSpec` -- and only that.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from kinescore.core.clip import ClipSpec

__all__ = ["load_rgb", "load_rgb_u8"]


def _decode_all(path: str) -> np.ndarray:
    """Decode every frame of ``path`` -> ``(N,H,W,3)`` uint8."""
    import imageio.v3 as iio

    if os.path.isdir(path):
        names = sorted(os.listdir(path))
        if not names:
            raise ValueError(f"frame directory {path!r} is empty")
        frames = np.stack([np.asarray(iio.imread(os.path.join(path, n)))
                           for n in names])
    else:
        frames = np.asarray(iio.imread(path))  # (N,H,W,3) uint8
    return frames


def load_rgb_u8(clip: ClipSpec, max_frames: int = 0) -> torch.Tensor:
    """Decode ``clip.path`` -> ``(T,3,H,W)`` uint8.

    Parameters
    ----------
    clip:
        The spec to decode. ``clip.stride`` (set by :meth:`ClipSpec.subsample`)
        is applied by slicing every ``stride``-th decoded frame; nothing else
        drops frames.
    max_frames:
        Optional hard cap, applied *after* striding, by keeping the first
        ``max_frames`` frames. A head truncation, deliberately not a uniform
        resample: it changes only *how much* of the clip is scored, never the
        interval between the frames that are.

    Returns
    -------
    torch.Tensor
        ``(T,3,H,W)`` uint8, where ``T`` is asserted to equal
        ``min(clip.n_frames, max_frames) if max_frames else clip.n_frames``.

    Raises
    ------
    ValueError
        If the decoded frame count doesn't match what ``clip`` declares --
        i.e. the file on disk does not match the spec that says how to
        interpret it (wrong file swapped in, spec built from a stale probe,
        stride applied twice). Refusing is the point of taking a ``ClipSpec``
        instead of a path.
    """
    frames = _decode_all(clip.path)
    if clip.stride > 1:
        frames = frames[::clip.stride]

    expected = clip.n_frames if not max_frames else min(clip.n_frames, max_frames)
    if max_frames and frames.shape[0] > max_frames:
        frames = frames[:max_frames]

    if frames.shape[0] != expected:
        raise ValueError(
            f"decoded {frames.shape[0]} frames from {clip.path!r} (stride="
            f"{clip.stride}) but the ClipSpec declares n_frames={clip.n_frames}"
            f"{f', max_frames={max_frames}' if max_frames else ''} "
            f"(expected {expected}). The timebase does not match the frames "
            f"being scored -- refusing to silently proceed.")

    return torch.tensor(np.asarray(frames)).permute(0, 3, 1, 2).contiguous()


def load_rgb(clip: ClipSpec, max_frames: int = 0) -> torch.Tensor:
    """:func:`load_rgb_u8` scaled to fp32 in ``[0,1]``."""
    return load_rgb_u8(clip, max_frames=max_frames).float().div_(255.0)
