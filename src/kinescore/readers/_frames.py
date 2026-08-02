"""Shared frame-shape normalisation for readers.

:class:`~kinescore.core.reader.PoseReader` accepts an unbatched ``(T,H,W,3)``
clip (any dtype -- ``uint8`` frames straight off disk, or already-float), an
unbatched ``(T,3,H,W)`` float clip, or a batched ``(B,T,3,H,W)`` float clip in
``[0,1]``. Every reader in this package needs the same conversion to the
backbone's input contract (``(N,3,H,W)`` float in ``[0,1]``, ``N = B*T``), so
it lives here once instead of being reimplemented per reader.

Why ``(T,3,H,W)`` is accepted
-----------------------------
It is what :func:`kinescore.video.reader.load_rgb` actually returns -- see its
docstring, which documents ``(T,3,H,W)`` explicitly -- and
:meth:`kinescore.core.scorer.Scorer.score` hands that tensor straight to
``reader.read``. Until the first end-to-end scoring run, no caller had ever
joined those two halves, so the mismatch sat undetected: every clip failed
with ``expected frames shaped (T,H,W,3) or (B,T,3,H,W), got (49,3,480,640)``
and, because ``bench/runner.py`` records failures rather than dropping them,
it surfaced as 24 recorded failures rather than a crash. Accepting the shape
the decoder documents is the fix, at the one place that already owns this
conversion, rather than a transpose bolted onto each call site.

The two unbatched layouts are told apart by which axis is 3, not by position:
``shape[-1] == 3`` is channels-last, ``shape[1] == 3`` is channels-first. A
tensor that is 3 on *both* axes (a 3-pixel-wide clip) is genuinely ambiguous
and raises rather than guessing -- guessing would silently transpose height
into channels and produce plausible, wrong joint angles.
"""
from __future__ import annotations

import torch

__all__ = ["normalize_frames"]


def normalize_frames(frames: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """``(T,H,W,3)``/``(T,3,H,W)``/``(B,T,3,H,W)`` -> ``((N,3,H,W) float, B, T)``.

    ``uint8`` input is scaled by ``1/255``; already-float input is passed
    through unchanged (assumed already in ``[0,1]`` per the reader contract).
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
