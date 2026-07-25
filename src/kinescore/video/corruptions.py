"""Controlled video corruptions for stress-testing a physics referee.

A valid physics judge must give *worse* scores to *worse* video. These
operators degrade decoded RGB frames in physically meaningful ways at a
tunable severity, so a stress sweep can check the judge's score moves
monotonically with damage:

* ``gaussian_noise`` -- sensor noise (pose reading becomes erratic),
* ``occlusion``      -- the arm gets hidden behind a block,
* ``blur``           -- defocus / motion blur,
* ``temporal_break``  -- frames swapped for random others -> a teleporting arm
  (the canonical "unphysical generation" failure). **Note:** the frame a given
  index is replaced with is drawn independently and uniformly from ``[0, T)``,
  so it can coincide with the index being replaced -- ``temporal_break`` can
  replace a frame with itself. This mirrors the source's behaviour; at low
  severity a small fraction of "corrupted" frames are therefore actually
  no-ops, which is expected, not a bug in the operator.

All operate on ``rgb (T, 3, H, W)`` in ``[0, 1]`` and return the same shape.

Reproducibility fix
--------------------
The source (``Marionette-fkjepa/models/evaluation/corruptions.py``) drew its
randomness from the global RNG (``torch.randn_like``, ``torch.randperm``,
``torch.randint`` with no ``generator``). A "reproducible" severity sweep was
therefore only reproducible by accident of whatever else in the process had
touched the global RNG state first -- reordering unrelated code could silently
change a stress-test result. Every operator here takes an explicit
``generator: torch.Generator | None`` and threads it through every random draw.
Passing the same seeded generator to two calls now *guarantees* identical
output; passing ``None`` falls back to the global RNG exactly as before, so
existing call sites that don't care about reproducibility are unaffected.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["VideoCorruptions"]


def _gaussian_kernel(sigma: float) -> torch.Tensor:
    k = int(2 * round(2 * sigma) + 1)
    x = torch.arange(k).float() - k // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return g / g.sum()


class VideoCorruptions:
    """Stateless severity-parameterised RGB corruption operators."""

    @staticmethod
    def gaussian_noise(rgb: torch.Tensor, sigma: float,
                       generator: torch.Generator | None = None) -> torch.Tensor:
        """Add i.i.d. Gaussian pixel noise of std ``sigma``."""
        if sigma <= 0:
            return rgb
        noise = torch.randn(rgb.shape, generator=generator,
                            dtype=rgb.dtype, device=rgb.device)
        return (rgb + sigma * noise).clamp(0, 1)

    @staticmethod
    def occlusion(rgb: torch.Tensor, frac: float,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        """Black out a centred square covering ``frac`` of the frame area.

        Deterministic given ``(rgb.shape, frac)`` -- ``generator`` is accepted
        only so every operator shares one call signature and can be driven
        uniformly from a severity sweep.
        """
        if frac <= 0:
            return rgb
        T, C, H, W = rgb.shape
        s = int((frac ** 0.5) * min(H, W))
        if s <= 0:
            return rgb
        y0, x0 = (H - s) // 2, (W - s) // 2
        out = rgb.clone()
        out[:, :, y0:y0 + s, x0:x0 + s] = 0.0
        return out

    @staticmethod
    def blur(rgb: torch.Tensor, sigma: float,
            generator: torch.Generator | None = None) -> torch.Tensor:
        """Separable Gaussian blur with std ``sigma`` pixels.

        Deterministic; ``generator`` accepted for signature uniformity only
        (see :meth:`occlusion`).
        """
        if sigma <= 0:
            return rgb
        g = _gaussian_kernel(sigma).to(rgb)
        k = g.numel()
        ker = (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)
        return F.conv2d(rgb, ker, padding=k // 2, groups=3).clamp(0, 1)

    @staticmethod
    def temporal_break(rgb: torch.Tensor, frac: float,
                       generator: torch.Generator | None = None) -> torch.Tensor:
        """Replace a fraction ``frac`` of frames with random other frames.

        Produces discontinuous "teleports" -- the motion-level failure a
        physics referee should catch even when each individual frame looks
        fine. See the module docstring: a replacement source frame can equal
        its own destination index, so this is not guaranteed to touch every
        selected frame.
        """
        if frac <= 0:
            return rgb
        T = rgb.shape[0]
        n = max(int(frac * T), 1)
        idx = torch.randperm(T, generator=generator)[:n]
        src = torch.randint(0, T, (n,), generator=generator)
        out = rgb.clone()
        out[idx] = rgb[src]
        return out

    #: Severity sweeps used by the stress test (0 = clean).
    SWEEPS = {
        "gaussian_noise": [0.0, 0.05, 0.1, 0.2, 0.4],
        "occlusion":      [0.0, 0.1, 0.2, 0.3, 0.5],
        "blur":           [0.0, 1.0, 2.0, 4.0, 8.0],
        "temporal_break": [0.0, 0.1, 0.2, 0.4, 0.6],
    }

    @classmethod
    def apply(cls, name: str, rgb: torch.Tensor, severity: float,
             generator: torch.Generator | None = None) -> torch.Tensor:
        return getattr(cls, name)(rgb, severity, generator=generator)
