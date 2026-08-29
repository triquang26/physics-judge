"""A checkpoint carries the cell it was trained for, and refuses another.

``save_reader``/``load_reader`` round-trip without a backbone: the stub below
stands in for one, so nothing here downloads weights.
"""
from __future__ import annotations

import pytest
import torch

from kinescore.core.clip import ViewLayout
from kinescore.heads.diffusion import DiffusionKeypointHead
from kinescore.readers.checkpoint import (
    CheckpointMismatch,
    ReaderExpectation,
    load_reader,
    read_cfg,
    save_reader,
)

D = 32
LAYOUT = ViewLayout(n_views=3, tokens_per_view=4, packing="width",
                    order=("exterior_1", "exterior_2", "wrist"))


class _Robot:
    name = "airbot_mmk2"


class _StubBackbone:
    def encode(self, rgb):  # pragma: no cover - never called here
        raise AssertionError("the backbone must not be built to load a head")


def _head(n_keypoints=8) -> DiffusionKeypointHead:
    torch.manual_seed(0)
    return DiffusionKeypointHead(in_dim=D, n_keypoints=n_keypoints, n_views=3,
                        tokens_per_view=4, d_model=16, decoder_nhead=2,
                        n_decoder_layers=1, temporal_nhead=2, ff=32,
                        n_temporal_layers=1, t_max=8)


def _save(tmp_path, head=None, **overrides) -> str:
    path = str(tmp_path / "reader.pt")
    kwargs = {"cell_id": "single_arm.mv3_row.ctrlworld",
              "robot": "airbot_mmk2", "view_id": "mv3_row",
              "view_layout": LAYOUT}
    kwargs.update(overrides)
    save_reader(path, head or _head(), **kwargs)
    return path


class TestRoundTrip:
    def test_weights_and_geometry_survive(self, tmp_path):
        head = _head()
        path = _save(tmp_path, head)

        reader = load_reader(path, robot=_Robot(), view_layout=LAYOUT,
                             backbone=_StubBackbone())

        assert reader.head.n_keypoints == 8
        assert reader.robot_name == "airbot_mmk2"
        for name, want in head.state_dict().items():
            assert torch.equal(reader.head.state_dict()[name], want)

    def test_cfg_is_readable_without_building_anything(self, tmp_path):
        cfg = read_cfg(_save(tmp_path))
        assert cfg["cell_id"] == "single_arm.mv3_row.ctrlworld"
        assert cfg["view_layout_key"] == LAYOUT.key
        assert cfg["n_out"] == 24

    def test_meta_is_carried(self, tmp_path):
        path = _save(tmp_path, meta={"val_mm": 23.22})
        assert torch.load(path, weights_only=False)["meta"]["val_mm"] == 23.22


class TestExpectations:
    def _expect(self, **overrides) -> ReaderExpectation:
        base = {"cell_id": "single_arm.mv3_row.ctrlworld",
                "robot": "airbot_mmk2", "view_id": "mv3_row", "n_views": 3,
                "packing": "width", "n_keypoints": 8}
        base.update(overrides)
        return ReaderExpectation(**base)

    def test_a_matching_cell_loads(self, tmp_path):
        load_reader(_save(tmp_path), robot=_Robot(), view_layout=LAYOUT,
                    expect=self._expect(), backbone=_StubBackbone())

    @pytest.mark.parametrize("field,value", [
        ("robot", "fourier_gr1"),
        ("view_id", "sv1"),
        ("n_views", 1),
        ("packing", "grid2x2"),
        ("n_keypoints", 12),
    ])
    def test_a_different_cell_is_refused(self, tmp_path, field, value):
        with pytest.raises(CheckpointMismatch, match=field):
            load_reader(_save(tmp_path), robot=_Robot(), view_layout=LAYOUT,
                        expect=self._expect(**{field: value}),
                        backbone=_StubBackbone())


class TestMalformed:
    def test_a_file_without_a_head_is_refused(self, tmp_path):
        path = str(tmp_path / "not_a_reader.pt")
        torch.save({"cfg": {}}, path)
        with pytest.raises(ValueError, match="not a keypoint checkpoint"):
            load_reader(path, robot=_Robot(), view_layout=LAYOUT,
                        backbone=_StubBackbone())

    def test_n_out_must_be_three_per_keypoint(self, tmp_path):
        path = _save(tmp_path)
        payload = torch.load(path, weights_only=False)
        payload["cfg"]["n_out"] = 25
        torch.save(payload, path)
        with pytest.raises(ValueError, match="positive multiple of 3"):
            load_reader(path, robot=_Robot(), view_layout=LAYOUT,
                        backbone=_StubBackbone())

    def test_a_missing_tensor_is_fatal(self, tmp_path):
        path = _save(tmp_path)
        payload = torch.load(path, weights_only=False)
        payload["head"].pop("x0_head.weight")
        torch.save(payload, path)
        with pytest.raises(ValueError, match="missing 1 head tensor"):
            load_reader(path, robot=_Robot(), view_layout=LAYOUT,
                        backbone=_StubBackbone())

    def test_undeclared_tensors_are_dropped(self, tmp_path):
        path = _save(tmp_path)
        payload = torch.load(path, weights_only=False)
        payload["head"]["logvar_head.weight"] = torch.zeros(3, 16)
        torch.save(payload, path)
        reader = load_reader(path, robot=_Robot(), view_layout=LAYOUT,
                             backbone=_StubBackbone())
        assert "logvar_head.weight" not in reader.head.state_dict()
