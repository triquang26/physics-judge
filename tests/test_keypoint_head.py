"""The shipped head and reader: shapes, temporal context, and the token guard.

CPU-only and backbone-free -- the head consumes patch tokens, so a test can
hand it random ones.
"""
from __future__ import annotations

import pytest
import torch

from kinescore.core.clip import ViewLayout
from kinescore.core.reader import Readout
from kinescore.heads.keypoint import KeypointHead
from kinescore.readers.keypoint import KeypointReader

D = 32
TOKENS_PER_VIEW = 4


def _head(n_keypoints=5, **kwargs) -> KeypointHead:
    torch.manual_seed(0)
    return KeypointHead(in_dim=D, n_keypoints=n_keypoints, d_model=16,
                        n_heads=2, temporal_nhead=2, ff=32,
                        n_temporal_layers=1, t_max=8, **kwargs)


class _StubBackbone:
    """Returns ``(N, V, P, D)`` tokens without loading any weights."""

    def __init__(self, n_views: int) -> None:
        self.n_views = n_views

    def encode(self, rgb: torch.Tensor) -> torch.Tensor:
        n = rgb.shape[0]
        return torch.randn(n, self.n_views, TOKENS_PER_VIEW, D)


class TestHead:
    def test_output_is_points_not_a_flat_vector(self):
        head = _head(n_keypoints=5)
        out = head(torch.randn(2, 6, 12, D))
        assert out.shape == (2, 6, 5, 3)

    def test_n_out_is_three_per_keypoint(self):
        assert _head(n_keypoints=8).n_out == 24

    def test_any_token_count_is_pooled_away(self):
        head = _head()
        assert head(torch.randn(1, 3, 4, D)).shape == (1, 3, 5, 3)
        assert head(torch.randn(1, 3, 64, D)).shape == (1, 3, 5, 3)

    def test_temporal_context_changes_the_prediction(self):
        head = _head().eval()
        x = torch.randn(1, 6, 12, D)
        with torch.no_grad():
            with_ctx = head(x, use_context=True)
            without_ctx = head(x, use_context=False)
        assert not torch.allclose(with_ctx, without_ctx)

    def test_without_context_frames_are_independent(self):
        head = _head().eval()
        x = torch.randn(1, 4, 12, D)
        with torch.no_grad():
            whole = head(x, use_context=False)
            piecewise = torch.cat(
                [head(x[:, i:i + 1], use_context=False) for i in range(4)], dim=1)
        assert torch.allclose(whole, piecewise, atol=1e-5)

    def test_gradients_reach_every_parameter(self):
        head = _head()
        head(torch.randn(1, 3, 12, D)).sum().backward()
        missing = [n for n, p in head.named_parameters() if p.grad is None]
        assert not missing


class TestReader:
    def _reader(self, n_views, **kwargs) -> KeypointReader:
        layout = ViewLayout(n_views=n_views, tokens_per_view=TOKENS_PER_VIEW,
                            packing="width" if n_views > 1 else "none")
        return KeypointReader(
            backbone=_StubBackbone(n_views), head=_head().eval(),
            view_layout=layout, robot_name="franka_panda",
            reader_id=f"franka_panda.{n_views}v", **kwargs)

    def test_reads_uint8_frames_into_points(self):
        reader = self._reader(3)
        out = reader.read(torch.randint(0, 255, (5, 192, 960, 3),
                                        dtype=torch.uint8))
        assert isinstance(out, Readout)
        assert out.P.shape == (1, 5, 5, 3)
        assert out.n_frames == 5

    def test_token_count_must_match_the_layout(self):
        reader = self._reader(3)
        reader.backbone = _StubBackbone(2)  # two views into a three-view reader
        with pytest.raises(ValueError, match="token count"):
            reader.read(torch.randint(0, 255, (2, 192, 960, 3),
                                      dtype=torch.uint8))

    def test_channels_first_float_frames_are_accepted(self):
        reader = self._reader(1)
        out = reader.read(torch.rand(4, 3, 24, 32))
        assert out.P.shape == (1, 4, 5, 3)
