"""``kinescore.training.cache``: a cache file describes itself, and a mismatch
between what it holds and what the caller asks for is an error, not a reshape.

CPU-only: :func:`write_cache` / :func:`load_cache` operate on a header +
tensor pair a test can build by hand, without a backbone or a video decode.
"""
from __future__ import annotations

import json

import pytest
import torch

from kinescore.core.clip import ViewLayout
from kinescore.training.cache import (
    CACHE_SCHEMA_VERSION,
    CacheHeader,
    assert_real_joint_source,
    load_cache,
    write_cache,
)


def _header(**overrides) -> CacheHeader:
    base = {
        "schema": CACHE_SCHEMA_VERSION,
        "reader_id": "airbot_mmk2.mv4_row",
        "view_layout_key": "3x4:unnamed:width:3p:0,1,2",
        "n_views": 3,
        "tokens_per_view": 4,
        "backbone_id": "dinov3_vitl16@768:p2",
        "source_path": "/data/clips/ep0.mp4",
        "n_frames": 6,
        "embed_dim": 8,
    }
    base.update(overrides)
    return CacheHeader(**base)


class TestRoundTrip:
    def test_tensor_and_header_survive(self, tmp_path):
        feat = torch.randn(6, 12, 8, dtype=torch.float16)
        path = str(tmp_path / "ep0.pt")
        write_cache(path, feat, _header())

        got, header = load_cache(path)
        assert torch.equal(got, feat)
        assert header == _header()

    def test_parent_directories_are_created(self, tmp_path):
        path = str(tmp_path / "deep" / "deeper" / "ep0.pt")
        write_cache(path, torch.zeros(1, 12, 8, dtype=torch.float16), _header())
        assert load_cache(path)[0].shape == (1, 12, 8)

    def test_mmap_read_matches_ordinary_read(self, tmp_path):
        feat = torch.randn(6, 12, 8, dtype=torch.float16)
        path = str(tmp_path / "ep0.pt")
        write_cache(path, feat, _header())

        mapped, header = load_cache(path, mmap=True)
        assert torch.equal(mapped, feat)
        assert header == _header()

    def test_mmap_read_still_checks_the_header(self, tmp_path):
        path = str(tmp_path / "ep0.pt")
        write_cache(path, torch.zeros(1, 12, 8, dtype=torch.float16), _header())
        with pytest.raises(ValueError):
            load_cache(path, reader_id="fourier_gr1.mv4_row", mmap=True)

    def test_mmap_slice_is_writable_once_copied(self, tmp_path):
        # The trainer slices a window and calls .float(); that copy must be
        # writable even though the mapped tensor behind it is not.
        path = str(tmp_path / "ep0.pt")
        write_cache(path, torch.ones(6, 12, 8, dtype=torch.float16), _header())
        window = load_cache(path, mmap=True)[0][:4].float()
        window += 1.0
        assert float(window[0, 0, 0]) == 2.0

    def test_unknown_header_fields_are_dropped(self, tmp_path):
        path = str(tmp_path / "ep0.pt")
        payload = {"feat": torch.zeros(1, 12, 8), "header": _header().as_dict()}
        payload["header"]["a_field_from_the_future"] = 1
        torch.save(payload, path)
        assert load_cache(path)[1].n_views == 3


class TestHeaderChecks:
    def test_wrong_reader_raises(self, tmp_path):
        path = str(tmp_path / "ep0.pt")
        write_cache(path, torch.zeros(1, 12, 8), _header())
        with pytest.raises(ValueError, match="was built for reader"):
            load_cache(path, reader_id="fourier_gr1.mv4_row")

    def test_view_count_mismatch_raises(self, tmp_path):
        path = str(tmp_path / "ep0.pt")
        write_cache(path, torch.zeros(1, 12, 8), _header())
        with pytest.raises(ValueError, match="view"):
            load_cache(path, view_layout=ViewLayout(n_views=1, tokens_per_view=4))

    def test_backbone_mismatch_raises(self, tmp_path):
        path = str(tmp_path / "ep0.pt")
        write_cache(path, torch.zeros(1, 12, 8), _header())
        with pytest.raises(ValueError, match="encoded by"):
            load_cache(path, backbone="dinov3_vits16@224:p1")

    def test_matching_expectations_pass(self, tmp_path):
        path = str(tmp_path / "ep0.pt")
        write_cache(path, torch.zeros(1, 12, 8), _header())
        feat, _ = load_cache(
            path, reader_id="airbot_mmk2.mv4_row",
            view_layout=ViewLayout(n_views=3, tokens_per_view=4,
                                   packing="width"),
            backbone="dinov3_vitl16@768:p2")
        assert feat.shape == (1, 12, 8)

    def test_a_bare_tensor_is_not_a_cache(self, tmp_path):
        path = str(tmp_path / "bare.pt")
        torch.save(torch.zeros(3), path)
        with pytest.raises(ValueError, match="not a kinescore cache file"):
            load_cache(path)


class TestJointSource:
    def _write(self, tmp_path, label) -> str:
        path = tmp_path / "ep0.json"
        path.write_text(json.dumps(label))
        return str(path)

    def test_real_joints_pass(self, tmp_path):
        path = self._write(tmp_path, {
            "joint_source": "real",
            "observation.state.joint_position": [[0.0] * 7],
        })
        assert assert_real_joint_source(path)["joint_source"] == "real"

    def test_synthetic_joints_raise(self, tmp_path):
        path = self._write(tmp_path, {
            "joint_source": "synthetic",
            "observation.state.joint_position": [[0.0] * 7],
        })
        with pytest.raises(ValueError, match="joint_source"):
            assert_real_joint_source(path)

    def test_missing_joint_array_raises(self, tmp_path):
        path = self._write(tmp_path, {"joint_source": "real"})
        with pytest.raises(ValueError, match="joint_position"):
            assert_real_joint_source(path)
