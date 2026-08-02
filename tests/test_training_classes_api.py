"""``CacheBuilder`` / ``RawRadTrainer`` -- the class-based training/ API.

``training/`` used to expose only loose functions (``precompute_cache``,
``train_head_rawrad``); both are now thin wrappers around a class
(``CacheBuilder``, ``RawRadTrainer``) that holds the run's shared state as
instance attributes. These tests pin that the wrappers are BEHAVIOURALLY
identical to calling the class directly (not merely "both exist"), since
``kinescore.cli.cmd_cache`` / ``cmd_train_rawrad`` still import the free
functions and must keep working unchanged.
"""
import json
import os

import torch

from kinescore.core.clip import ViewLayout
from kinescore.heads.heteroscedastic import ReadoutV2Head
from kinescore.robots.synthetic import Synthetic2R
from kinescore.training import cache as cache_mod
from kinescore.training.cache import CacheBuilder, precompute_cache
from kinescore.training.trainer_rawrad import (
    RawRadTrainer,
    TrainConfigRawRad,
    train_head_rawrad,
)


class _FakeBackbone:
    dino_model = "fake"
    dino_input = 224
    patch_pool = 1


def _fake_encode_clip(backbone, clip_path, *, view_layout, max_frames=0,
                      device="cpu", fps_arg=None, dt_arg=None, frame_chunk=0):
    """Bypass real video decode: deterministic (T,P,D) tokens from the path."""
    class _Clip:
        n_frames = 4
    torch.manual_seed(abs(hash(clip_path)) % (2**31))
    feat = torch.randn(4, 2, 3, dtype=torch.float16)
    return feat, _Clip()


def _write_episode_sources(video_dir, ann_dir, ep_id):
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    open(os.path.join(video_dir, f"{ep_id}.mp4"), "wb").close()  # never read
    with open(os.path.join(ann_dir, f"{ep_id}.json"), "w") as f:
        json.dump({"joint_source": "real",
                   "observation.state.joint_position": [[0.0]] * 4}, f)


class TestCacheBuilderMatchesPrecomputeCache:
    def test_build_split_and_precompute_cache_write_identical_caches(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(cache_mod, "encode_clip", _fake_encode_clip)
        video_root = str(tmp_path / "video")
        ann_root = str(tmp_path / "ann")
        _write_episode_sources(os.path.join(video_root, "train"),
                               os.path.join(ann_root, "train"), "ep0")
        _write_episode_sources(os.path.join(video_root, "train"),
                               os.path.join(ann_root, "train"), "ep1")

        backbone = _FakeBackbone()
        view_layout = ViewLayout()

        out_a = str(tmp_path / "out_class")
        summary_a = CacheBuilder(backbone, view_layout).build_split(
            video_root=video_root, annotation_root=ann_root, out_root=out_a,
            split="train")

        out_b = str(tmp_path / "out_fn")
        summary_b = precompute_cache(
            video_root=video_root, annotation_root=ann_root, out_root=out_b,
            split="train", backbone=backbone, view_layout=view_layout)

        assert summary_a == summary_b == {
            "n_done": 2, "n_skipped_existing": 0, "n_skipped_no_annotation": 0}
        for ep in ("ep0", "ep1"):
            pa = torch.load(os.path.join(out_a, "train", f"{ep}.pt"), weights_only=False)
            pb = torch.load(os.path.join(out_b, "train", f"{ep}.pt"), weights_only=False)
            assert pa["header"] == pb["header"]
            assert torch.equal(pa["feat"], pb["feat"])


def _synthetic_split(n, n_tokens, embed_dim, n_joints, seed):
    g = torch.Generator().manual_seed(seed)
    feat = torch.randn(n, n_tokens, embed_dim, generator=g)
    q = (torch.rand(n, n_joints, generator=g) * 2 - 1) * 2.0
    return feat, q


class TestRawRadTrainerMatchesTrainHeadRawRad:
    def test_fit_and_function_wrapper_give_identical_loss_history(self):
        robot = Synthetic2R()
        n_tokens, embed_dim = 6, 8
        feat, q = _synthetic_split(32, n_tokens, embed_dim, robot.n_joints, seed=0)
        cfg = TrainConfigRawRad(steps=20, phase_a=8, batch_size=8, lr=5e-3,
                                eval_every=0, seed=0, log_every=0)

        torch.manual_seed(0)
        head_a = ReadoutV2Head(in_dim=embed_dim, d_model=16, n_heads=2,
                               temporal_nhead=2, ff=16, n_temporal_layers=1,
                               t_max=4, n_out=robot.n_joints)
        torch.manual_seed(0)
        head_b = ReadoutV2Head(in_dim=embed_dim, d_model=16, n_heads=2,
                               temporal_nhead=2, ff=16, n_temporal_layers=1,
                               t_max=4, n_out=robot.n_joints)

        result_class = RawRadTrainer(head_a, robot, cfg).fit(
            train_feat=feat, train_q=q)
        result_fn = train_head_rawrad(head_b, robot, train_feat=feat,
                                      train_q=q, cfg=cfg)

        assert result_class.loss_history == result_fn.loss_history
        assert result_class.train_keypoint_mm == result_fn.train_keypoint_mm

    def test_constructor_rejects_mismatched_n_out_before_any_step_runs(self):
        robot = Synthetic2R()
        head = ReadoutV2Head(in_dim=8, d_model=16, n_heads=2, temporal_nhead=2,
                             ff=16, n_temporal_layers=1, t_max=4,
                             n_out=robot.n_joints + 1)
        try:
            RawRadTrainer(head, robot)
        except ValueError as exc:
            assert "n_out" in str(exc)
        else:
            raise AssertionError("expected a ValueError on n_out mismatch")

    def test_static_eval_keypoint_mm_matches_module_function(self):
        from kinescore.training.trainer_rawrad import eval_keypoint_mm_rawrad
        robot = Synthetic2R()
        n_tokens, embed_dim = 4, 8
        head = ReadoutV2Head(in_dim=embed_dim, d_model=16, n_heads=2,
                             temporal_nhead=2, ff=16, n_temporal_layers=1,
                             t_max=4, n_out=robot.n_joints)
        feat, q = _synthetic_split(8, n_tokens, embed_dim, robot.n_joints, seed=1)
        a = RawRadTrainer.eval_keypoint_mm(head, robot, feat, q,
                                           q_lo=robot.q_lo, q_hi=robot.q_hi)
        b = eval_keypoint_mm_rawrad(head, robot, feat, q,
                                    q_lo=robot.q_lo, q_hi=robot.q_hi)
        assert a["keypoint_mm"] == b["keypoint_mm"]
