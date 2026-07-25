"""Decode frames for a :class:`~kinescore.core.clip.ClipSpec` -- and only that.

The source's ``load_rgb(path: str, max_frames: int = 0)`` took a bare path.
Decimation (skip every other frame, take a stride-2 subset) happened at other
call sites as a plain slice on the decoded tensor, entirely decoupled from
whatever ``dt`` was passed to scoring alongside it -- that decoupling is
defect D1 (see ``kinescore.core.clip``): half of a benchmark's clips were
scored at a ``dt`` that no longer matched the frames actually fed to the
metrics.

Here :func:`load_rgb` takes the :class:`~kinescore.core.clip.ClipSpec` itself.
The *only* frame-dropping this function performs is ``clip.stride``, a field
that only :meth:`ClipSpec.subsample` can set (and which scales ``dt`` for you
when it does). There is no separate stride parameter to pass inconsistently.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from kinescore.core.clip import ClipSpec

__all__ = ["load_rgb"]


def _decode_all(path: str) -> np.ndarray:
    """Decode every frame of ``path`` -> ``(N,H,W,3)`` uint8.

    ``path`` is either a video file (mp4/webm/...) or a directory of frame
    images (``ClipSpec.path`` documents both as valid) -- a directory is read
    as its sorted file listing, one frame per image.
    """
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


def load_rgb(clip: ClipSpec, max_frames: int = 0) -> torch.Tensor:
    """Decode ``clip.path`` -> ``(T,3,H,W)`` fp32 in ``[0,1]``.

    Parameters
    ----------
    clip:
        The spec to decode. ``clip.stride`` (set by :meth:`ClipSpec.subsample`)
        is applied by slicing every ``stride``-th decoded frame; nothing else
        drops frames.
    max_frames:
        Optional hard cap, applied *after* striding, by keeping the first
        ``max_frames`` frames. This is a plain head-truncation -- not the
        source's ``np.linspace`` uniform resampling across the whole clip.
        Uniform resampling silently changes the spacing between kept frames
        (and therefore the effective ``dt``) without updating any metadata,
        which is exactly the class of bug this module exists to prevent; a
        truncation changes only *how much* of the clip is scored, not the
        interval between the frames that are.

    Returns
    -------
    torch.Tensor
        ``(T,3,H,W)`` where ``T`` is asserted to equal
        ``min(clip.n_frames, max_frames) if max_frames else clip.n_frames``.

    Raises
    ------
    ValueError
        If the decoded frame count doesn't match what ``clip`` declares --
        i.e. the file on disk no longer matches the spec that says how to
        interpret it (wrong file swapped in, spec built from a stale probe,
        stride applied twice, ...). Silently proceeding would let the
        timebase drift out from under the frames exactly like defect D1;
        refusing is the point of taking a ``ClipSpec`` instead of a path.
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
            f"(expected {expected}). The timebase no longer matches the frames "
            f"actually being scored -- refusing to silently proceed (defect D1).")

    rgb = torch.tensor(np.asarray(frames)).permute(0, 3, 1, 2).float() / 255.0
    return rgb
