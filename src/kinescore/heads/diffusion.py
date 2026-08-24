"""A denoising head: DINO patch tokens -> 3-D keypoints.

Points are read by denoising rather than by regression.
:class:`DiffusionKeypointHead` is an x0-denoiser: a query carries a noised
keypoint plus the noise level it was drawn at, cross-attends to the frame's
patch tokens through :class:`~kinescore.heads.keypoint.KeypointQueryDecoder`,
and predicts the clean coordinate. Reading a clip is then DDIM sampling, and
several samples are averaged into the point that ships.

Coordinates live in ``[-1, 1]`` inside the denoiser, mapped there by
:class:`WorkspaceNormalizer` from a box measured off the training targets and
carried in the checkpoint. Nothing guesses that box: a head asked to read or
train before it is fitted raises.

Output is ``(B, T, K, 3)`` metres in the robot-base frame, the same contract
:class:`~kinescore.heads.keypoint.KeypointHead` answers.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from kinescore.heads.keypoint import (
    KeypointQueryDecoder,
    TemporalEncoder,
    masked_smooth_l1,
    temporal_tracks,
)

__all__ = [
    "WorkspaceNormalizer", "CosineSchedule", "FourierCoordEmbed",
    "timestep_embedding", "DiffusionKeypointHead",
]


class WorkspaceNormalizer(nn.Module):
    """Affine map between metres and ``[-1, 1]``, one box per axis.

    The box is measured by :meth:`fit` and held as buffers, so it rides along
    in the state dict and a loaded head reads in the units it was trained on.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("lo", torch.zeros(3))
        self.register_buffer("hi", torch.zeros(3))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    def fit(self, lo: torch.Tensor, hi: torch.Tensor) -> None:
        """Set the box from two ``(3,)`` corners, in metres.

        Raises:
            ValueError: If ``hi`` is not above ``lo`` on every axis.
        """
        lo_t = torch.as_tensor(lo, dtype=torch.float32).reshape(3)
        hi_t = torch.as_tensor(hi, dtype=torch.float32).reshape(3)
        if not bool(torch.all(hi_t > lo_t)):
            raise ValueError(
                f"workspace hi must be above lo on every axis, got "
                f"lo={lo_t.tolist()} hi={hi_t.tolist()}")
        self.lo.copy_(lo_t.to(self.lo.device))
        self.hi.copy_(hi_t.to(self.hi.device))
        self.fitted.fill_(True)

    def _corners(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The fitted box.

        Raises:
            RuntimeError: If no box has been fitted.
        """
        if not bool(self.fitted):
            raise RuntimeError(
                "the workspace box is not fitted; call the head's calibrate() "
                "with the training targets before reading or training points")
        return self.lo, self.hi

    @property
    def half_extent(self) -> torch.Tensor:
        """``(3,)`` metres one normalised unit covers on each axis."""
        lo, hi = self._corners()
        return (hi - lo) / 2.0

    def encode(self, x_m: torch.Tensor) -> torch.Tensor:
        """``(..., 3)`` metres -> ``[-1, 1]``."""
        lo, hi = self._corners()
        return (x_m - lo) / (hi - lo) * 2.0 - 1.0

    def decode(self, x_n: torch.Tensor) -> torch.Tensor:
        """``(..., 3)`` in ``[-1, 1]`` -> metres."""
        lo, hi = self._corners()
        return (x_n + 1.0) / 2.0 * (hi - lo) + lo


class CosineSchedule:
    """Cosine ``alpha_bar(t)`` for continuous ``t`` in ``[0, 1]``.

    Args:
        s: Offset keeping ``alpha_bar`` short of 1 at ``t = 0``.
    """

    def __init__(self, s: float = 0.008) -> None:
        self.s = float(s)

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Signal fraction left at ``t``, in ``(0, 1]``."""
        f = torch.cos((t + self.s) / (1.0 + self.s) * math.pi / 2.0) ** 2
        f0 = math.cos(self.s / (1.0 + self.s) * math.pi / 2.0) ** 2
        return (f / f0).clamp(1e-5, 1.0)


def timestep_embedding(t: torch.Tensor, dim: int,
                       max_period: float = 10_000.0) -> torch.Tensor:
    """``(N,)`` noise levels in ``[0, 1]`` -> ``(N, dim)`` sinusoidal features."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None] * 1000.0
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return F.pad(emb, (0, 1)) if dim % 2 else emb


class FourierCoordEmbed(nn.Module):
    """``(..., 3)`` normalised coordinates -> ``(..., out_dim)``.

    Args:
        n_freqs: Octaves of sine/cosine features per axis.
        out_dim: Output width.
    """

    def __init__(self, n_freqs: int = 10, out_dim: int = 512) -> None:
        super().__init__()
        self.n_freqs = int(n_freqs)
        self.proj = nn.Sequential(
            nn.Linear(3 * (2 * self.n_freqs + 1), out_dim), nn.SiLU(),
            nn.Linear(out_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., 3) -> (..., out_dim)``."""
        feats = [x]
        for i in range(self.n_freqs):
            feats += [torch.sin((2.0 ** i) * math.pi * x),
                      torch.cos((2.0 ** i) * math.pi * x)]
        return self.proj(torch.cat(feats, dim=-1))


class DiffusionKeypointHead(nn.Module):
    """Denoise ``K`` points out of a frame's patch tokens.

    The readout is zero-initialised and enters through a residual around the
    noised input, so an untrained head predicts exactly what it was handed and
    training moves away from that rather than toward it.

    Args:
        in_dim: Token embed width ``D``.
        n_keypoints: ``K``, points predicted per frame.
        n_views: Views packed into a frame.
        tokens_per_view: Patch tokens one view contributes; must be square.
        d_model: Query and temporal token width.
        decoder_nhead: Cross-attention heads.
        n_decoder_layers: Cross-attention blocks.
        temporal_nhead: Temporal-attention heads.
        ff: Feed-forward width, decoder and temporal alike.
        n_temporal_layers: Temporal encoder layers.
        t_max: Frames the positional table covers.
        dropout: Dropout probability.
        n_coord_freqs: Octaves in the coordinate embedding.
        sample_steps: DDIM steps a read takes.
        n_samples: Samples averaged into the points a read returns.
        init_t: Noise level a read starts from when given an initial guess.

    Shape:
        - Input: ``(B, T, P, D)``.
        - Output: ``(B, T, K, 3)`` metres, robot-base frame.

    Raises:
        ValueError: If ``n_keypoints`` is below 1, or ``tokens_per_view`` is
            not a square patch grid.
    """

    head_kind = "diffusion"

    def __init__(self, in_dim: int = 1024, n_keypoints: int = 12,
                 n_views: int = 1, tokens_per_view: int = 576,
                 d_model: int = 512, decoder_nhead: int = 8,
                 n_decoder_layers: int = 2, temporal_nhead: int = 8,
                 ff: int = 2048, n_temporal_layers: int = 4, t_max: int = 64,
                 dropout: float = 0.1, n_coord_freqs: int = 10,
                 sample_steps: int = 10, n_samples: int = 5,
                 init_t: float = 0.3) -> None:
        super().__init__()
        if int(n_keypoints) < 1:
            raise ValueError(f"n_keypoints must be >= 1, got {n_keypoints}")
        self.in_dim = int(in_dim)
        self.n_keypoints = int(n_keypoints)
        self.n_views = int(n_views)
        self.tokens_per_view = int(tokens_per_view)
        self.d_model = int(d_model)
        self.decoder_nhead = int(decoder_nhead)
        self.n_decoder_layers = int(n_decoder_layers)
        self.temporal_nhead = int(temporal_nhead)
        self.ff = int(ff)
        self.n_temporal_layers = int(n_temporal_layers)
        self.t_max = int(t_max)
        self.dropout = float(dropout)
        self.n_coord_freqs = int(n_coord_freqs)
        self.sample_steps = int(sample_steps)
        self.n_samples = int(n_samples)
        self.init_t = float(init_t)

        self.decoder = KeypointQueryDecoder(
            in_dim=self.in_dim, n_keypoints=self.n_keypoints,
            n_views=self.n_views, tokens_per_view=self.tokens_per_view,
            d_model=self.d_model, nhead=self.decoder_nhead, ff=self.ff,
            n_layers=self.n_decoder_layers, dropout=dropout,
        )
        self.temporal = TemporalEncoder(
            d_model=self.d_model, nhead=self.temporal_nhead, ff=self.ff,
            n_layers=self.n_temporal_layers, t_max=self.t_max, dropout=dropout,
        )
        self.workspace = WorkspaceNormalizer()
        self.schedule = CosineSchedule()
        self.coord_embed = FourierCoordEmbed(self.n_coord_freqs, self.d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), nn.SiLU(),
            nn.Linear(self.d_model, self.d_model))
        self.out_norm = nn.LayerNorm(self.d_model)
        self.x0_head = nn.Linear(self.d_model, 3)
        nn.init.zeros_(self.x0_head.weight)
        nn.init.zeros_(self.x0_head.bias)

    @property
    def n_out(self) -> int:
        """Readout width, ``3 * n_keypoints``. The checkpoint records this."""
        return 3 * self.n_keypoints

    def calibrate(self, targets: Sequence[torch.Tensor], *,
                  margin: float = 0.05) -> None:
        """Fit the workspace box to the training targets, in metres.

        Each episode is reduced on its own, so the box costs one pair of
        corners per episode rather than a copy of every target.

        Args:
            targets: Per-episode ``(T, K, 3)`` points the head is fitted against.
            margin: Fraction of each axis' span left as headroom, so a point
                just outside the training range is still representable.

        Raises:
            ValueError: If ``targets`` holds no episode.
        """
        corners = [t.reshape(-1, 3).float().aminmax(dim=0) for t in targets]
        if not corners:
            raise ValueError("calibrate needs at least one episode of targets")
        lo = torch.stack([c.min for c in corners]).amin(dim=0)
        hi = torch.stack([c.max for c in corners]).amax(dim=0)
        pad = (hi - lo).clamp_min(1e-3) * margin
        self.workspace.fit(lo - pad, hi + pad)

    def _denoise(self, x_t: torch.Tensor, t: torch.Tensor, k: torch.Tensor,
                 v: torch.Tensor, use_context: bool = True) -> torch.Tensor:
        """``(B, T, K, 3)`` noised points at level ``(B,)`` -> clean points."""
        b, frames, points, _ = x_t.shape
        level = self.time_mlp(timestep_embedding(t, self.d_model))
        q = self.coord_embed(x_t) + level[:, None, None, :]
        q = q.reshape(b * frames, points, self.d_model)
        z = self.decoder.decode(q + self.decoder.queries(b * frames), k, v)
        z = z.reshape(b, frames, points, self.d_model)
        if use_context:
            z = z + temporal_tracks(self.temporal, z)
        return x_t + self.x0_head(self.out_norm(z))

    def training_loss(self, feat: torch.Tensor, target: torch.Tensor,
                      mask: torch.Tensor, *, beta: float = 0.05
                     ) -> torch.Tensor:
        """Masked smooth-L1 on the clean points recovered from noised targets.

        One noise level is drawn per window. ``beta`` is metres and is scaled
        into normalised units by the workspace, so it means the same knee
        wherever the loss is computed.
        """
        x0 = self.workspace.encode(target)
        t = torch.rand(x0.shape[0], device=x0.device).clamp_min(1e-4)
        ab = self.schedule.alpha_bar(t).reshape(-1, 1, 1, 1)
        x_t = ab.sqrt() * x0 + (1.0 - ab).sqrt() * torch.randn_like(x0)
        k, v = self.decoder.keys_values(feat)
        x0_hat = self._denoise(x_t, t, k, v)
        scale = float(self.workspace.half_extent.mean())
        return masked_smooth_l1(x0_hat, x0, mask, beta=beta / scale)

    @torch.no_grad()
    def sample(self, feat: torch.Tensor, *, n_steps: int | None = None,
               n_samples: int | None = None,
               init_m: torch.Tensor | None = None,
               init_t: float | None = None, use_context: bool = True
              ) -> torch.Tensor:
        """``(B, T, P, D) -> (B, T, K, 3)`` metres, by DDIM with ``eta = 0``.

        Every sample contributes its clean prediction at the last step, which
        carries no residual noise term. ``init_m`` starts from ``(B, T, K, 3)``
        metres noised to ``init_t`` and denoises only what is left, so a read
        that already has a guess spends its steps refining it.

        Raises:
            ValueError: If ``n_steps`` or ``n_samples`` is below 1.
        """
        steps = self.sample_steps if n_steps is None else int(n_steps)
        draws = self.n_samples if n_samples is None else int(n_samples)
        start = self.init_t if init_t is None else float(init_t)
        if steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {steps}")
        if draws < 1:
            raise ValueError(f"n_samples must be >= 1, got {draws}")

        k, v = self.decoder.keys_values(feat)
        b, frames = feat.shape[0], feat.shape[1]
        shape = (b, frames, self.n_keypoints, 3)
        ts = torch.linspace(start if init_m is not None else 1.0, 1e-4,
                            steps + 1, device=feat.device)
        x0_init = None if init_m is None else self.workspace.encode(init_m)

        drawn = []
        for _ in range(draws):
            if x0_init is None:
                x = torch.randn(shape, device=feat.device)
            else:
                ab = self.schedule.alpha_bar(ts[0])
                x = (ab.sqrt() * x0_init
                     + (1.0 - ab).sqrt() * torch.randn_like(x0_init))
            for i in range(steps):
                ab_now = self.schedule.alpha_bar(ts[i])
                ab_next = self.schedule.alpha_bar(ts[i + 1])
                x0_hat = self._denoise(x, ts[i].expand(b), k, v,
                                       use_context).clamp(-1.0, 1.0)
                eps = ((x - ab_now.sqrt() * x0_hat)
                       / (1.0 - ab_now).sqrt().clamp_min(1e-4))
                x = ab_next.sqrt() * x0_hat + (1.0 - ab_next).sqrt() * eps
            drawn.append(x0_hat)
        return self.workspace.decode(torch.stack(drawn).mean(dim=0))

    def forward(self, feat: torch.Tensor, use_context: bool = True
               ) -> torch.Tensor:
        """``(B, T, P, D) -> (B, T, K, 3)``."""
        return self.sample(feat, use_context=use_context)
