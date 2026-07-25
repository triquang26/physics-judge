"""``kinescore.training.cache``: a written cache round-trips its layout
metadata, and loading a 3-view cache with a 1-view reader raises (D4).

CPU-only, no backbone, no video decode -- :func:`write_cache`/:func:`load_cache`
operate on a header + tensor pair a test can construct by hand, exactly the
seam :func:`~kinescore.training.cache.encode_clip` would otherwise sit behind.
"""
from __future__ import annotations

import os

import pytest
import torch

from kinescore.core.clip import ViewLayout
from kinescore.training.cache import (
    CACHE_SCHEMA_VERSION,
    CacheHeader,
    backbone_id,
    load_cache,
    write_cache,
)


def _header(**overrides) -> CacheHeader:
    base = {
        "schema": CACHE_SCHEMA_VERSION, "view_layout_key": "1x4:unnamed",
        "n_views": 1, "tokens_per_view": 4, "backbone_id": "dinov3_vitl16@224:p2",
        "source_path": "/data/clips/ep0.mp4", "n_frames": 6, "embed_dim": 8,
    }
    base.update(overrides)
    return CacheHeader(**base)


class TestRoundTrip:
    def test_write_then_load_preserves_tensor_and_header(self, tmp_path):
        feat = torch.randn(6, 4, 8, dtype=torch.float16)
        header = _header()
        path = str(tmp_path / "ep0.pt")

        write_cache(path, feat, header)
        got_feat, got_header = load_cache(path)

        assert torch.equal(got_feat, feat)
        assert got_header == header

    def test_header_as_dict_from_dict_round_trip(self):
        header = _header()
        assert CacheHeader.from_dict(header.as_dict()) == header

    def test_creates_parent_directories(self, tmp_path):
        path = str(tmp_path / "nested" / "split" / "ep3.pt")
        write_cache(path, torch.zeros(1, 4, 8, dtype=torch.float16), _header(n_frames=1))
        assert os.path.exists(path)

    def test_load_missing_header_raises(self, tmp_path):
        # A bare-tensor "cache" (the legacy, non-self-describing format this
        # module's docstring says defect D4 traces back to).
        path = str(tmp_path / "legacy.pt")
        torch.save(torch.randn(6, 4, 8), path)
        with pytest.raises(ValueError, match="not a self-describing"):
            load_cache(path)


class TestD4ViewLayoutGuard:
    def test_matching_view_layout_loads_cleanly(self, tmp_path):
        feat = torch.randn(6, 4, 8, dtype=torch.float16)
        header = _header(n_views=1, view_layout_key=ViewLayout(n_views=1).key,
                         tokens_per_view=None)
        path = str(tmp_path / "ep0.pt")
        write_cache(path, feat, header)

        got_feat, got_header = load_cache(path, expected_view_layout=ViewLayout(n_views=1))
        assert got_feat.shape == (6, 4, 8)
        assert got_header.n_views == 1

    def test_three_view_cache_with_one_view_reader_raises(self, tmp_path):
        # The exact defect (D4) this format exists to catch: a multiview
        # cache silently feeding a single-view head.
        three_view_layout = ViewLayout(n_views=3)
        feat = torch.randn(6, 3 * 4, 8, dtype=torch.float16)  # (T, 3*P, D)
        header = _header(n_views=3, view_layout_key=three_view_layout.key,
                         tokens_per_view=4)
        path = str(tmp_path / "ep_multiview.pt")
        write_cache(path, feat, header)

        with pytest.raises(ValueError, match="defect D4"):
            load_cache(path, expected_view_layout=ViewLayout(n_views=1))

    def test_one_view_cache_with_three_view_reader_also_raises(self, tmp_path):
        # The guard fires in both directions, not just "cache has more views".
        one_view_layout = ViewLayout(n_views=1)
        feat = torch.randn(6, 4, 8, dtype=torch.float16)
        header = _header(n_views=1, view_layout_key=one_view_layout.key,
                         tokens_per_view=4)
        path = str(tmp_path / "ep_singleview.pt")
        write_cache(path, feat, header)

        with pytest.raises(ValueError, match="defect D4"):
            load_cache(path, expected_view_layout=ViewLayout(n_views=3))

    def test_no_expected_view_layout_skips_the_guard(self, tmp_path):
        # Loading without declaring an expectation is allowed (e.g. an
        # inspection tool that just wants to read the header) -- the guard is
        # opt-in via `expected_view_layout`, not unconditional.
        three_view_layout = ViewLayout(n_views=3)
        feat = torch.randn(6, 3 * 4, 8, dtype=torch.float16)
        header = _header(n_views=3, view_layout_key=three_view_layout.key,
                         tokens_per_view=4)
        path = str(tmp_path / "ep_multiview.pt")
        write_cache(path, feat, header)

        got_feat, got_header = load_cache(path)
        assert got_header.n_views == 3
        assert got_feat.shape[1] == 12


class TestBackboneId:
    def test_stable_id_string(self):
        class FakeBackbone:
            dino_model = "dinov3_vitl16"
            dino_input = 224
            patch_pool = 2

        assert backbone_id(FakeBackbone()) == "dinov3_vitl16@224:p2"
