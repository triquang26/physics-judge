"""``readers/checkpoint_v2.py`` -- the ReadoutV2 (heteroscedastic) loader.

CPU-only, offline: a tiny stand-in backbone with the same
``encode(rgb) -> (N,V,P,D)`` contract as :class:`~kinescore.backbones.dino.FeatureBackbone`
(see ``tests/test_reader_limit_semantics.py``) and a duck-typed fake
``RobotSpec`` stand in for the real GR-1 URDF/backbone, which this host may
not have configured (``KINESCORE_ASSETS``, network for DINOv3 weights).
``tests/test_checkpoint_v2_real_ckpt.py`` covers the real
``readout_v2_gr1.pt`` file, gated behind ``@pytest.mark.ckpt``.
"""
from __future__ import annotations

import pytest
import torch

from kinescore.core.clip import ViewLayout
from kinescore.heads.heteroscedastic import ReadoutV2Head
from kinescore.readers import checkpoint as ckpt_mod
from kinescore.readers import checkpoint_v2
from kinescore.readers.ensemble import EnsemblePoseReader

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

    def to(self, device):
        return self

    def eval(self):
        return self


class _FakeRobot:
    """Duck-typed RobotSpec stand-in -- no URDF, no pytorch_kinematics."""

    def __init__(self, name: str, n_joints: int, lo: float = -1.0, hi: float = 1.0):
        self.name = name
        self.n_joints = n_joints
        self.q_lo = torch.full((n_joints,), lo)
        self.q_hi = torch.full((n_joints,), hi)


def _clip(T: int = 5, H: int = 32, W: int = 32) -> torch.Tensor:
    return torch.randint(0, 255, (T, H, W, 3), dtype=torch.uint8)


def _make_head(n_out: int = 8, seed: int = 0) -> ReadoutV2Head:
    torch.manual_seed(seed)
    head = ReadoutV2Head(in_dim=D, d_model=16, n_heads=2, temporal_nhead=2,
                         ff=16, n_temporal_layers=1, t_max=8, n_out=n_out)
    head.eval()
    return head


LAYOUT = ViewLayout(n_views=1, tokens_per_view=P_TOKENS)


# ── cfg detection ──────────────────────────────────────────────────────────

class TestIsReadoutV2Cfg:
    def test_true_for_readout_v2_shaped_cfg(self):
        cfg = {"in_dim": 1024, "d_model": 512, "n_heads": 4,
              "temporal_nhead": 8, "ff": 1024, "n_temporal_layers": 2,
              "t_max": 64, "dropout": 0.1, "logvar_min": -10.0, "logvar_max": 4.0}
        assert checkpoint_v2.is_readout_v2_cfg(cfg) is True

    def test_false_for_attentive_shaped_cfg(self):
        cfg = {"n_joints": 7, "n_heads": 4, "n_cams": 1, "hidden": 512,
              "dropout": 0.1, "embed_dim": 1024}
        assert checkpoint_v2.is_readout_v2_cfg(cfg) is False

    def test_false_for_legacy_attentive_cfg(self):
        # the real judge_v3l-vintage 12-key cfg (see test_checkpoint_legacy_cfg.py)
        cfg = {"dino_model": "dinov3_vitl16", "dino_repo_dir": "",
              "embed_dim": 1024, "n_kp": 8, "fk_ee_link": "panda_link8",
              "pool": "attn", "n_heads": 4, "dino_input": 224, "patch_pool": 2,
              "hf_model_id": "x", "patch_size": 16, "n_register": 4}
        assert checkpoint_v2.is_readout_v2_cfg(cfg) is False

    def test_false_for_empty_cfg(self):
        assert checkpoint_v2.is_readout_v2_cfg({}) is False


# ── sigma_scale resolution ──────────────────────────────────────────────────

class TestResolveSigmaScale:
    def test_defaults_to_one_when_absent(self):
        t = checkpoint_v2.resolve_sigma_scale({}, n_out=8)
        assert torch.allclose(t, torch.tensor([1.0]))

    def test_reads_from_meta(self):
        t = checkpoint_v2.resolve_sigma_scale({"sigma_scale": 1.9375}, n_out=8)
        assert torch.allclose(t, torch.tensor([1.9375]))

    def test_override_wins_over_meta(self):
        t = checkpoint_v2.resolve_sigma_scale({"sigma_scale": 1.9375}, n_out=8,
                                              override=3.0)
        assert torch.allclose(t, torch.tensor([3.0]))

    def test_accepts_per_joint_vector(self):
        vec = [1.0] * 8
        t = checkpoint_v2.resolve_sigma_scale({}, n_out=8, override=vec)
        assert t.shape == (8,)

    def test_rejects_wrong_length_vector(self):
        with pytest.raises(ValueError):
            checkpoint_v2.resolve_sigma_scale({}, n_out=8, override=[1.0, 2.0])

    def test_floors_at_a_tiny_positive_value(self):
        t = checkpoint_v2.resolve_sigma_scale({}, n_out=8, override=0.0)
        assert float(t) > 0.0


# ── save/load_head round-trip ───────────────────────────────────────────────

class TestLoadHeadRoundtrip:
    def test_state_dict_and_forward_output_identical(self, tmp_path):
        head = _make_head(n_out=8)
        path = str(tmp_path / "rv2.pt")
        checkpoint_v2.save(path, head, view_layout=LAYOUT)
        loaded = checkpoint_v2.load_head(path)

        sd1, sd2 = head.state_dict(), loaded.head.state_dict()
        assert set(sd1) == set(sd2)
        for k in sd1:
            assert torch.equal(sd1[k], sd2[k]), f"mismatch at {k!r}"

        feat = torch.randn(2, 4, P_TOKENS, D)
        with torch.no_grad():
            o1 = head(feat)
            o2 = loaded.head(feat)
        assert torch.equal(o1["mu"], o2["mu"])
        assert torch.equal(o1["logvar"], o2["logvar"])

    def test_sigma_scale_persisted_through_meta(self, tmp_path):
        head = _make_head()
        path = str(tmp_path / "rv2.pt")
        checkpoint_v2.save(path, head, view_layout=LAYOUT, sigma_scale=1.9375)
        loaded = checkpoint_v2.load_head(path)
        assert torch.allclose(loaded.sigma_scale, torch.tensor([1.9375]), atol=1e-6)

    def test_robot_name_defaults_to_fourier_gr1_when_absent(self, tmp_path):
        # Mirrors the real readout_v2_gr1.pt: its cfg predates the
        # robot_name convention entirely.
        head = _make_head()
        path = str(tmp_path / "rv2_bare.pt")
        cfg = {"in_dim": D, "d_model": 16, "n_heads": 2, "temporal_nhead": 2,
              "ff": 16, "n_temporal_layers": 1, "t_max": 8, "dropout": 0.1,
              "n_out": 8, "logvar_min": -10.0, "logvar_max": 4.0}
        torch.save({"state_dict": head.state_dict(), "cfg": cfg,
                   "meta": {"urdf_path": "assets/grx/GRX/GR1/gr1t1/urdf/x.urdf"}},
                  path)
        loaded = checkpoint_v2.load_head(path)
        assert loaded.robot_name == checkpoint_v2.DEFAULT_ROBOT_NAME == "fourier_gr1"

    def test_view_layout_key_and_robot_name_round_trip_when_saved(self, tmp_path):
        head = _make_head()
        path = str(tmp_path / "rv2.pt")
        layout = ViewLayout(n_views=1, order=("ego",), tokens_per_view=P_TOKENS)
        checkpoint_v2.save(path, head, view_layout=layout, robot_name="fourier_gr1")
        loaded = checkpoint_v2.load_head(path)
        assert loaded.view_layout_key == layout.key
        assert loaded.robot_name == "fourier_gr1"

    def test_urdf_sha256_round_trips_through_meta(self, tmp_path):
        head = _make_head()
        path = str(tmp_path / "rv2.pt")
        checkpoint_v2.save(path, head, view_layout=LAYOUT, urdf_sha256="deadbeef")
        loaded = checkpoint_v2.load_head(path)
        assert loaded.urdf_sha256 == "deadbeef"

    def test_limit_semantics_is_always_raw_rad(self, tmp_path):
        head = _make_head()
        path = str(tmp_path / "rv2.pt")
        checkpoint_v2.save(path, head, view_layout=LAYOUT)
        loaded = checkpoint_v2.load_head(path)
        assert loaded.limit_semantics == "raw_rad"

    def test_rejects_non_readout_v2_file(self, tmp_path):
        path = str(tmp_path / "not_rv2.pt")
        cfg = {"n_joints": 7, "n_heads": 4, "n_cams": 1, "hidden": 512,
              "dropout": 0.1, "embed_dim": 1024}
        torch.save({"head": {}, "cfg": cfg, "meta": {}}, path)
        with pytest.raises(ValueError, match="does not look like"):
            checkpoint_v2.load_head(path)

    def test_rejects_file_with_no_state_dict_key(self, tmp_path):
        path = str(tmp_path / "bad.pt")
        torch.save({"cfg": {"d_model": 1, "temporal_nhead": 1}}, path)
        with pytest.raises(ValueError, match="state_dict"):
            checkpoint_v2.load_head(path)


# ── load_reader: no split (n_fk == n_out) ───────────────────────────────────

class TestLoadReaderNoSplit:
    def test_no_aux_split_when_head_matches_robot_exactly(self, tmp_path):
        head = _make_head(n_out=5)
        path = str(tmp_path / "rv2.pt")
        checkpoint_v2.save(path, head, view_layout=LAYOUT)
        robot = _FakeRobot("fourier_gr1", n_joints=5)
        reader = checkpoint_v2.load_reader(path, robot=robot, view_layout=LAYOUT,
                                           backbone=_FakeBackbone(LAYOUT))
        assert reader.limit_semantics == "raw_rad"
        assert reader.robot_name == "fourier_gr1"
        out = reader.read(_clip())
        assert out.q.shape == (1, 5, 5)
        assert out.aux is None  # inner's aux, untouched -- nothing to split off
        assert "clamp_rad" in out.extras
        assert "clamp_rad_hand" not in out.extras

    def test_sigma_is_calibrated_by_sigma_scale(self, tmp_path):
        head = _make_head(n_out=4)
        path = str(tmp_path / "rv2.pt")
        checkpoint_v2.save(path, head, view_layout=LAYOUT, sigma_scale=2.0)
        robot = _FakeRobot("fourier_gr1", n_joints=4)
        backbone = _FakeBackbone(LAYOUT)

        torch.manual_seed(42)
        reader = checkpoint_v2.load_reader(path, robot=robot, view_layout=LAYOUT,
                                           backbone=backbone, sigma_scale=None)
        raw_reader = checkpoint_v2.load_reader(path, robot=robot, view_layout=LAYOUT,
                                               backbone=backbone, sigma_scale=1.0)

        torch.manual_seed(7)
        clip = _clip()
        torch.manual_seed(7)
        out_scaled = reader.read(clip)
        torch.manual_seed(7)
        out_raw = raw_reader.read(clip)
        torch.testing.assert_close(out_scaled.sigma, out_raw.sigma * 2.0,
                                   atol=1e-5, rtol=1e-5)


# ── load_reader: FK/aux split (n_fk < n_out) ────────────────────────────────

class TestLoadReaderSplit:
    def test_splits_trailing_dims_into_aux(self, tmp_path):
        head = _make_head(n_out=8)  # 5 "fk" + 3 "hand"
        path = str(tmp_path / "rv2.pt")
        checkpoint_v2.save(path, head, view_layout=LAYOUT)
        robot = _FakeRobot("fourier_gr1", n_joints=5)
        reader = checkpoint_v2.load_reader(path, robot=robot, view_layout=LAYOUT,
                                           backbone=_FakeBackbone(LAYOUT))
        out = reader.read(_clip())
        assert out.q.shape == (1, 5, 5)
        assert out.q_raw.shape == (1, 5, 5)
        assert out.sigma.shape == (1, 5, 5)
        assert set(out.aux) == {"hand", "hand_raw", "hand_sigma"}
        assert out.aux["hand"].shape == (1, 5, 3)
        assert "clamp_rad" in out.extras and out.extras["clamp_rad"].shape == (1, 5, 5)
        assert out.extras["clamp_rad_hand"].shape == (1, 5, 3)

    def test_fk_slice_is_the_leading_dims_of_the_full_head_output(self, tmp_path):
        """The split must be a pure slice, not a re-derivation -- q[..., :n]
        from a robot needing all 8 dims (real, narrow limits on every dim)
        must equal the split reader's (5-joint robot, same limits on the
        leading 5) q exactly on the FK dims. ``q_raw`` (unclamped, so
        clamp-bound differences on the trailing dims cannot affect it) must
        match on *every* dim, split or not."""
        head = _make_head(n_out=8)
        path = str(tmp_path / "rv2.pt")
        checkpoint_v2.save(path, head, view_layout=LAYOUT)
        backbone = _FakeBackbone(LAYOUT)

        robot5 = _FakeRobot("fourier_gr1", n_joints=5)  # lo=-1, hi=1 (defaults)
        robot8 = _FakeRobot("fourier_gr1", n_joints=8)  # same lo/hi on every dim
        clip = _clip()

        torch.manual_seed(3)
        split_reader = checkpoint_v2.load_reader(path, robot=robot5, view_layout=LAYOUT,
                                                 backbone=backbone)
        torch.manual_seed(3)
        full_reader = checkpoint_v2.load_reader(path, robot=robot8, view_layout=LAYOUT,
                                                backbone=backbone)

        torch.manual_seed(11)
        out_split = split_reader.read(clip)
        torch.manual_seed(11)
        out_full = full_reader.read(clip)

        # FK dims: identical real limits on both sides -> identical clamp.
        torch.testing.assert_close(out_split.q, out_full.q[..., :5], atol=1e-6, rtol=1e-6)
        # q_raw is never clamped, so it matches on every dim regardless of
        # which robot's (differently-bounded) reader produced it.
        torch.testing.assert_close(out_split.q_raw, out_full.q_raw[..., :5],
                                   atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(out_split.aux["hand_raw"], out_full.q_raw[..., 5:],
                                   atol=1e-6, rtol=1e-6)

    def test_raises_when_robot_needs_more_joints_than_head_predicts(self, tmp_path):
        head = _make_head(n_out=4)
        path = str(tmp_path / "rv2.pt")
        checkpoint_v2.save(path, head, view_layout=LAYOUT)
        robot = _FakeRobot("fourier_gr1", n_joints=6)
        with pytest.raises(ValueError, match="only predicts"):
            checkpoint_v2.load_reader(path, robot=robot, view_layout=LAYOUT,
                                      backbone=_FakeBackbone(LAYOUT))

    def test_hand_dims_are_effectively_unclamped_by_permissive_bounds(self, tmp_path):
        """Zero the mu head so the "hand" dims are trivially in [-1,1] -- but
        the point of the permissive bound is that a LARGE raw hand value
        would also survive un-clamped; check that directly."""
        head = _make_head(n_out=6)  # 3 fk + 3 hand
        with torch.no_grad():
            head.mu_head.bias[3:] = 50.0  # far outside any real robot's rad range
        path = str(tmp_path / "rv2.pt")
        checkpoint_v2.save(path, head, view_layout=LAYOUT)
        robot = _FakeRobot("fourier_gr1", n_joints=3, lo=-1.0, hi=1.0)
        reader = checkpoint_v2.load_reader(path, robot=robot, view_layout=LAYOUT,
                                           backbone=_FakeBackbone(LAYOUT))
        out = reader.read(_clip(T=1))
        # the "hand" slice must NOT have been clamped down to the fk [-1,1]
        # range -- it should still read close to the 50.0 bias.
        assert bool((out.aux["hand"] > 10.0).all())


# ── ensemble round-trip (K=1 degrade, K=2 disagreement) ─────────────────────

class TestEnsembleFromCheckpoint:
    def test_k1_single_head_file_degrades_to_plain_reader(self, tmp_path):
        head = _make_head(n_out=5)
        path = str(tmp_path / "single.pt")
        checkpoint_v2.save(path, head, view_layout=LAYOUT, sigma_scale=1.5)
        robot = _FakeRobot("fourier_gr1", n_joints=5)

        reader = EnsemblePoseReader.from_checkpoint(
            path, robot=robot, view_layout=LAYOUT, backbone=_FakeBackbone(LAYOUT))

        assert not isinstance(reader, EnsemblePoseReader)
        assert isinstance(reader, checkpoint_v2.ReadoutV2PoseReader)
        out = reader.read(_clip())
        # No fabricated 0 -- the epistemic axis is structurally absent, not
        # present-and-zero. See ensemble.py's module docstring.
        assert "epistemic" not in out.extras
        assert "aleatoric" not in out.extras
        assert "member_q" not in out.extras

    def test_k1_ensemble_bundle_with_one_head_also_degrades(self, tmp_path):
        head = _make_head(n_out=5)
        path = str(tmp_path / "bundle_of_one.pt")
        checkpoint_v2.save_ensemble(path, [head], view_layout=LAYOUT)
        robot = _FakeRobot("fourier_gr1", n_joints=5)
        reader = EnsemblePoseReader.from_checkpoint(
            path, robot=robot, view_layout=LAYOUT, backbone=_FakeBackbone(LAYOUT))
        assert not isinstance(reader, EnsemblePoseReader)

    def test_k2_bundle_round_trip_computes_member_disagreement(self, tmp_path):
        head_a = _make_head(n_out=5, seed=0)
        head_b = _make_head(n_out=5, seed=1)
        path = str(tmp_path / "ens.pt")
        checkpoint_v2.save_ensemble(path, [head_a, head_b], view_layout=LAYOUT,
                                    sigma_scale=1.5, meta={"note": "roundtrip"})
        robot = _FakeRobot("fourier_gr1", n_joints=5)
        backbone = _FakeBackbone(LAYOUT)

        reader = EnsemblePoseReader.from_checkpoint(
            path, robot=robot, view_layout=LAYOUT, backbone=backbone)

        assert isinstance(reader, EnsemblePoseReader)
        assert len(reader.members) == 2
        assert reader.limit_semantics == "raw_rad"

        out = reader.read(_clip())
        assert out.extras["n_members"] == 2
        assert out.extras["member_q"].shape == (2, 1, 5, 5)
        assert "epistemic" in out.extras and "aleatoric" in out.extras
        # Two DIFFERENTLY-SEEDED heads reading the same clip should disagree
        # somewhere -- this is the actual "member disagreement is computed"
        # check the task asks for, not merely "the key exists".
        assert float(out.extras["epistemic"].abs().sum()) > 0.0

    def test_k2_bundle_applies_sigma_scale_to_every_member(self, tmp_path):
        head_a = _make_head(n_out=4, seed=0)
        head_b = _make_head(n_out=4, seed=1)
        path = str(tmp_path / "ens.pt")
        checkpoint_v2.save_ensemble(path, [head_a, head_b], view_layout=LAYOUT,
                                    sigma_scale=2.0)
        robot = _FakeRobot("fourier_gr1", n_joints=4)
        backbone = _FakeBackbone(LAYOUT)

        reader = EnsemblePoseReader.from_checkpoint(
            path, robot=robot, view_layout=LAYOUT, backbone=backbone)
        for m in reader.members:
            assert torch.allclose(m.sigma_scale, torch.tensor([2.0]))

    def test_ensemble_bundle_rejects_robot_needing_more_joints_than_heads_predict(
            self, tmp_path):
        head_a = _make_head(n_out=3, seed=0)
        head_b = _make_head(n_out=3, seed=1)
        path = str(tmp_path / "ens.pt")
        checkpoint_v2.save_ensemble(path, [head_a, head_b], view_layout=LAYOUT)
        robot = _FakeRobot("fourier_gr1", n_joints=6)
        with pytest.raises(ValueError, match="only predicts"):
            EnsemblePoseReader.from_checkpoint(
                path, robot=robot, view_layout=LAYOUT, backbone=_FakeBackbone(LAYOUT))


# ── unified auto-routing entry point: readers.checkpoint.load_reader ────────

class TestUnifiedLoadReaderRouting:
    def test_routes_readout_v2_cfg_to_readoutv2_pose_reader(self, tmp_path):
        head = _make_head(n_out=5)
        path = str(tmp_path / "rv2.pt")
        checkpoint_v2.save(path, head, view_layout=LAYOUT)
        robot = _FakeRobot("fourier_gr1", n_joints=5)
        reader = ckpt_mod.load_reader(path, robot=robot, view_layout=LAYOUT,
                                      backbone=_FakeBackbone(LAYOUT))
        assert isinstance(reader, checkpoint_v2.ReadoutV2PoseReader)
        assert reader.limit_semantics == "raw_rad"

    def test_routes_attentive_cfg_to_squashed_pose_reader(self, tmp_path):
        from kinescore.heads.attentive import AttentivePoseHead
        from kinescore.readers import checkpoint as ckpt
        from kinescore.readers.squashed import SquashedPoseReader

        head = AttentivePoseHead(in_dim=D, hidden=8, n_joints=5, n_heads=2, n_cams=1)
        head.eval()
        path = str(tmp_path / "attentive.pt")
        layout = ViewLayout(n_views=1, tokens_per_view=P_TOKENS)
        ckpt.save(path, head, view_layout=layout, robot_name="franka_panda",
                 limit_semantics="squashed")
        robot = _FakeRobot("franka_panda", n_joints=5)
        reader = ckpt_mod.load_reader(path, robot=robot, view_layout=layout,
                                      backbone=_FakeBackbone(layout))
        assert isinstance(reader, SquashedPoseReader)
        assert reader.limit_semantics == "squashed"
