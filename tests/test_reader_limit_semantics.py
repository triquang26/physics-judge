"""``limit_semantics`` is what a reader declares, and it must be true.

See ``core/reader.py``: a ``"raw_rad"`` reader exposes the unconstrained
``q_raw`` and the clamp magnitude that makes joint-limit violation
observable -- the property every live reader in this package now has.
(The squashed-reader case -- ``q`` inside limits *by construction*,
``q_raw`` always ``None`` -- used to be exercised here too, against
``readers/squashed.py::SquashedPoseReader``; that reader was removed, see
``legacy_docs/PROVENANCE.md``'s D7 addendum, so this file now only covers the
``raw_rad`` half of the contract.)

No real backbone is loaded here (network/GPU/weights) -- a tiny stand-in with
the same ``encode(rgb) -> (N,V,P,D)`` contract is used instead, so this stays
a fast, offline CPU test.
"""
from __future__ import annotations

import torch

from kinescore.core.clip import ViewLayout
from kinescore.heads.heteroscedastic import ReadoutV2Head
from kinescore.readers.heteroscedastic import HeteroscedasticPoseReader

D = 12
P_TOKENS = 9  # 3x3 patch grid


class _FakeBackbone:
    """Stand-in for FeatureBackbone.encode: (N,3,H,W) -> (N,V,P,D)."""

    def __init__(self, view_layout: ViewLayout, d: int = D, p: int = P_TOKENS):
        self.view_layout = view_layout
        self.d = d
        self.p = p

    def encode(self, rgb: torch.Tensor) -> torch.Tensor:
        n = rgb.shape[0]
        v = self.view_layout.n_views
        return torch.randn(n, v, self.p, self.d)


def _clip(T: int = 5, H: int = 32, W: int = 32) -> torch.Tensor:
    return torch.randint(0, 255, (T, H, W, 3), dtype=torch.uint8)


def test_heteroscedastic_reader_exposes_q_raw_and_clamp_magnitude():
    n_out = 5
    layout = ViewLayout(n_views=1, tokens_per_view=P_TOKENS)
    backbone = _FakeBackbone(layout)
    head = ReadoutV2Head(in_dim=D, d_model=16, n_heads=2, temporal_nhead=2,
                         ff=16, n_temporal_layers=1, t_max=8, n_out=n_out)
    head.eval()
    q_lo = torch.full((n_out,), -1.0)
    q_hi = torch.full((n_out,), 1.0)
    reader = HeteroscedasticPoseReader(backbone=backbone, head=head, q_lo=q_lo,
                                       q_hi=q_hi, view_layout=layout,
                                       robot_name="fourier_gr1",
                                       reader_id="heteroscedastic/test")

    assert reader.limit_semantics == "raw_rad"
    out = reader.read(_clip())
    assert out.q_raw is not None
    assert out.q_raw.shape == (1, 5, n_out)
    assert out.sigma is not None
    assert "clamp_rad" in out.extras

    clamp_rad = out.extras["clamp_rad"]
    assert bool((clamp_rad >= 0).all())
    # q is always inside [lo, hi] after clamping, regardless of q_raw.
    assert bool((out.q >= q_lo.view(1, 1, -1) - 1e-6).all())
    assert bool((out.q <= q_hi.view(1, 1, -1) + 1e-6).all())
    # the clamp magnitude is exactly the raw-vs-clamped gap.
    expected_clamp = (out.q_raw - out.q).abs()
    torch.testing.assert_close(clamp_rad, expected_clamp, atol=1e-5, rtol=1e-5)


def test_heteroscedastic_reader_clamp_is_zero_when_raw_in_range():
    """When the head happens to emit an in-range pose, clamp_rad is exactly
    0 -- the violation signal is genuinely absent, not merely unobservable
    (contrast with the squashed reader, where it is unobservable always)."""
    n_out = 3
    layout = ViewLayout(n_views=1, tokens_per_view=P_TOKENS)
    backbone = _FakeBackbone(layout)
    head = ReadoutV2Head(in_dim=D, d_model=16, n_heads=2, temporal_nhead=2,
                         ff=16, n_temporal_layers=1, t_max=8, n_out=n_out)
    head.eval()
    # Zero the mu head so mu is always exactly 0 -- trivially in-range.
    with torch.no_grad():
        head.mu_head.weight.zero_()
        head.mu_head.bias.zero_()
    q_lo = torch.full((n_out,), -1.0)
    q_hi = torch.full((n_out,), 1.0)
    reader = HeteroscedasticPoseReader(backbone=backbone, head=head, q_lo=q_lo,
                                       q_hi=q_hi, view_layout=layout,
                                       robot_name="fourier_gr1",
                                       reader_id="heteroscedastic/test")
    out = reader.read(_clip())
    assert bool((out.extras["clamp_rad"].abs() < 1e-6).all())
    torch.testing.assert_close(out.q, out.q_raw, atol=1e-6, rtol=1e-6)
