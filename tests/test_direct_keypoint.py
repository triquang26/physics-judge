"""``DirectKeypointPoseReader`` -- the ``limit_semantics="keypoints"`` reader.

Mirrors ``tests/test_checkpoint_v2.py``'s pattern: a tiny stand-in backbone
with the same ``encode(rgb) -> (N,V,P,D)`` contract as
:class:`~kinescore.backbones.dino.FeatureBackbone`, and a duck-typed fake
robot (only ``.name`` is used -- this reader never touches FK/URDF/limits).
Covers both the reader itself (``read`` shape/semantics) and the
checkpoint-loading path: :func:`~kinescore.readers.checkpoint_v2.load_direct_keypoint_reader`
and its auto-routing through ``readers/loader.py::load_reader`` for a
checkpoint whose cfg declares ``head_target == "keypoints"``.

No real backbone or DINOv3 weights are loaded here -- fully offline, CPU-only.
"""
from __future__ import annotations

import pytest
import torch

from kinescore.core.clip import ViewLayout
from kinescore.core.reader import Readout
from kinescore.heads.heteroscedastic import ReadoutV2Head
from kinescore.readers import checkpoint_v2, loader
from kinescore.readers.direct_keypoint import DirectKeypointPoseReader

D = 12
P_TOKENS = 9  # 3x3 patch grid
K = 8  # keypoints -> n_out = 24


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

    def to(self, device):
        return self

    def eval(self):
        return self


class _FakeRobot:
    """Duck-typed RobotSpec stand-in -- only ``.name`` is used."""

    def __init__(self, name: str):
        self.name = name


def _clip(T: int = 5, H: int = 32, W: int = 32) -> torch.Tensor:
    return torch.randint(0, 255, (T, H, W, 3), dtype=torch.uint8)


def _make_head(n_out: int = 3 * K, seed: int = 0) -> ReadoutV2Head:
    torch.manual_seed(seed)
    head = ReadoutV2Head(in_dim=D, d_model=16, n_heads=2, temporal_nhead=2,
                         ff=16, n_temporal_layers=1, t_max=8, n_out=n_out)
    head.eval()
    return head


LAYOUT = ViewLayout(n_views=1, tokens_per_view=P_TOKENS)


# ── DirectKeypointPoseReader.read ────────────────────────────────────────────

class TestDirectKeypointPoseReaderRead:
    def test_declares_keypoints_limit_semantics(self):
        reader = DirectKeypointPoseReader(
            backbone=_FakeBackbone(LAYOUT), head=_make_head(), n_keypoints=K,
            view_layout=LAYOUT, robot_name="franka_panda",
            reader_id="direct_keypoint/test")
        assert reader.limit_semantics == "keypoints"

    def test_read_returns_P_shaped_b_t_k_3_and_no_q(self):
        reader = DirectKeypointPoseReader(
            backbone=_FakeBackbone(LAYOUT), head=_make_head(), n_keypoints=K,
            view_layout=LAYOUT, robot_name="franka_panda",
            reader_id="direct_keypoint/test")
        out = reader.read(_clip(T=5))
        assert isinstance(out, Readout)
        assert out.q is None
        assert out.q_raw is None
        assert out.P is not None
        assert out.P.shape == (1, 5, K, 3)
        assert out.sigma is not None
        assert out.sigma.shape == (1, 5, K, 3)

    def test_n_frames_falls_back_to_P_when_q_is_none(self):
        reader = DirectKeypointPoseReader(
            backbone=_FakeBackbone(LAYOUT), head=_make_head(), n_keypoints=K,
            view_layout=LAYOUT, robot_name="franka_panda",
            reader_id="direct_keypoint/test")
        out = reader.read(_clip(T=7))
        assert out.n_frames == 7

    def test_rejects_n_out_not_a_multiple_of_3(self):
        with pytest.raises(ValueError, match="n_out"):
            DirectKeypointPoseReader(
                backbone=_FakeBackbone(LAYOUT), head=_make_head(n_out=10),
                n_keypoints=K, view_layout=LAYOUT, robot_name="franka_panda",
                reader_id="direct_keypoint/test")

    def test_batched_input_preserves_batch_dim(self):
        reader = DirectKeypointPoseReader(
            backbone=_FakeBackbone(LAYOUT), head=_make_head(), n_keypoints=K,
            view_layout=LAYOUT, robot_name="franka_panda",
            reader_id="direct_keypoint/test")
        frames = torch.rand(3, 4, 3, 32, 32)  # (B,T,3,H,W)
        out = reader.read(frames)
        assert out.P.shape == (3, 4, K, 3)


# ── checkpoint_v2.load_direct_keypoint_reader ────────────────────────────────

def _save_training_format(path: str, head: ReadoutV2Head, n_out: int,
                          cfg_extra: dict | None = None) -> None:
    """Mimic the training side's own save format (not checkpoint_v2.save's)."""
    cfg = {"head_target": "keypoints"}
    cfg.update(cfg_extra or {})
    torch.save({"head": head.state_dict(), "n_out": n_out, "embed_dim": D,
               "cfg": cfg}, path)


class TestLoadDirectKeypointReader:
    def test_loads_with_default_architecture_when_cfg_is_minimal(self, tmp_path):
        head = ReadoutV2Head(in_dim=D, d_model=768, n_heads=8, temporal_nhead=8,
                             ff=2048, n_temporal_layers=4, t_max=64,
                             dropout=0.1, n_out=3 * K)
        head.eval()
        path = str(tmp_path / "kp.pt")
        _save_training_format(path, head, n_out=3 * K)

        reader = checkpoint_v2.load_direct_keypoint_reader(
            path, robot=_FakeRobot("franka_panda"), view_layout=LAYOUT,
            backbone=_FakeBackbone(LAYOUT))
        assert isinstance(reader, DirectKeypointPoseReader)
        assert reader.limit_semantics == "keypoints"
        assert reader.n_keypoints == K
        assert reader.robot_name == "franka_panda"
        out = reader.read(_clip(T=3))
        assert out.P.shape == (1, 3, K, 3)

    def test_loads_with_small_architecture_when_cfg_overrides_it(self, tmp_path):
        head = _make_head(n_out=3 * K)
        path = str(tmp_path / "kp_small.pt")
        _save_training_format(path, head, n_out=3 * K, cfg_extra={
            "d_model": 16, "n_heads": 2, "temporal_nhead": 2, "ff": 16,
            "n_temporal_layers": 1, "t_max": 8,
        })
        reader = checkpoint_v2.load_direct_keypoint_reader(
            path, robot=_FakeRobot("franka_panda"), view_layout=LAYOUT,
            backbone=_FakeBackbone(LAYOUT))
        sd1, sd2 = head.state_dict(), reader.head.state_dict()
        assert set(sd1) == set(sd2)
        for k in sd1:
            assert torch.equal(sd1[k], sd2[k])

    def test_rejects_file_with_no_head_key(self, tmp_path):
        path = str(tmp_path / "not_kp.pt")
        torch.save({"state_dict": {}, "cfg": {"d_model": 1}}, path)
        with pytest.raises(ValueError, match="'head' key"):
            checkpoint_v2.load_direct_keypoint_reader(
                path, robot=_FakeRobot("franka_panda"), view_layout=LAYOUT,
                backbone=_FakeBackbone(LAYOUT))

    def test_rejects_n_out_not_multiple_of_3(self, tmp_path):
        head = _make_head(n_out=10)
        path = str(tmp_path / "bad_nout.pt")
        _save_training_format(path, head, n_out=10)
        with pytest.raises(ValueError, match="multiple of 3"):
            checkpoint_v2.load_direct_keypoint_reader(
                path, robot=_FakeRobot("franka_panda"), view_layout=LAYOUT,
                backbone=_FakeBackbone(LAYOUT))


# ── is_direct_keypoint_cfg / auto-routing ────────────────────────────────────

class TestIsDirectKeypointCfg:
    def test_true_when_head_target_is_keypoints(self):
        assert checkpoint_v2.is_direct_keypoint_cfg({"head_target": "keypoints"})

    def test_false_when_head_target_is_joints_or_absent(self):
        assert not checkpoint_v2.is_direct_keypoint_cfg({"head_target": "joints"})
        assert not checkpoint_v2.is_direct_keypoint_cfg({})


class TestUnifiedLoadReaderRoutesKeypointCheckpoints:
    def test_load_reader_routes_head_target_keypoints_to_direct_keypoint_reader(
            self, tmp_path):
        head = _make_head(n_out=3 * K)
        path = str(tmp_path / "kp.pt")
        _save_training_format(path, head, n_out=3 * K, cfg_extra={
            "d_model": 16, "n_heads": 2, "temporal_nhead": 2, "ff": 16,
            "n_temporal_layers": 1, "t_max": 8,
        })
        reader = loader.load_reader(
            path, robot=_FakeRobot("franka_panda"), view_layout=LAYOUT,
            backbone=_FakeBackbone(LAYOUT))
        assert isinstance(reader, DirectKeypointPoseReader)
        assert reader.limit_semantics == "keypoints"
        out = reader.read(_clip(T=2))
        assert out.P.shape == (1, 2, K, 3)
        assert out.q is None
