"""``kinescore.training.trainer``: FK targets, the masked loss, and the loop.

A tiny analytic robot and random cached tokens stand in for a corpus, so the
whole loop runs on CPU in seconds.
"""
from __future__ import annotations

import json

import pytest
import torch

from kinescore.core.clip import ViewLayout
from kinescore.heads.keypoint import KeypointHead
from kinescore.robots import get_robot
from kinescore.training.cache import CACHE_SCHEMA_VERSION, CacheHeader, write_cache
from kinescore.training.trainer import KeypointTrainer, TrainConfig

D = 32
TOKENS = 4
LAYOUT = ViewLayout(n_views=1, tokens_per_view=TOKENS, packing="none")
READER_ID = "synthetic_2r.sv1"


@pytest.fixture(scope="module")
def robot():
    return get_robot("synthetic_2r")


def _head(k) -> KeypointHead:
    torch.manual_seed(0)
    return KeypointHead(in_dim=D, n_keypoints=k, d_model=16, n_heads=2,
                        temporal_nhead=2, ff=32, n_temporal_layers=1, t_max=8)


def _trainer(robot, **cfg_kwargs) -> KeypointTrainer:
    k = KeypointTrainer.n_keypoints(robot)
    cfg = TrainConfig(steps=4, batch_size=2, window_size=4, eval_every=2,
                      log_every=0, **cfg_kwargs)
    return KeypointTrainer(_head(k), robot, reader_id=READER_ID,
                           view_layout=LAYOUT, cfg=cfg)


def _corpus(tmp_path, robot, n_episodes=4, n_frames=6):
    """Write a cache + annotation tree the trainer can load."""
    for split in ("train", "val"):
        (tmp_path / "cache" / split).mkdir(parents=True, exist_ok=True)
        (tmp_path / "annotation" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n_episodes):
            feat = torch.randn(n_frames, TOKENS, D, dtype=torch.float16)
            write_cache(str(tmp_path / "cache" / split / f"{i}.pt"), feat,
                        CacheHeader(
                            schema=CACHE_SCHEMA_VERSION, reader_id=READER_ID,
                            view_layout_key=LAYOUT.key, n_views=1,
                            tokens_per_view=TOKENS,
                            backbone_id="dinov3_vitl16@768:p2",
                            source_path=f"/corpus/{i}.mp4", n_frames=n_frames,
                            embed_dim=D))
            q = torch.rand(n_frames, robot.n_joints).tolist()
            (tmp_path / "annotation" / split / f"{i}.json").write_text(
                json.dumps({"joint_source": "real",
                            "observation.state.joint_position": q}))
    return str(tmp_path / "cache"), str(tmp_path / "annotation")


class TestTargets:
    def test_keypoint_count_comes_from_forward_kinematics(self, robot):
        k = KeypointTrainer.n_keypoints(robot)
        assert k == robot.forward_kinematics(
            torch.zeros(1, 1, robot.n_joints), None).shape[-2]

    def test_target_is_points_in_metres(self, robot):
        trainer = _trainer(robot)
        target = trainer.build_target(torch.zeros(5, robot.n_joints))
        assert target.shape == (5, KeypointTrainer.n_keypoints(robot), 3)
        assert target.dtype == torch.float32

    def test_head_and_robot_must_agree(self, robot):
        with pytest.raises(ValueError, match="forward kinematics produces"):
            KeypointTrainer(_head(99), robot, reader_id=READER_ID,
                            view_layout=LAYOUT)


class TestLoss:
    def test_padded_frames_do_not_enter_the_loss(self, robot):
        trainer = _trainer(robot)
        k = KeypointTrainer.n_keypoints(robot)
        pred = torch.zeros(1, 4, k, 3)
        target = torch.zeros(1, 4, k, 3)
        target[0, 2:] = 1e3  # padding, masked out
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        assert float(trainer.compute_loss(pred, target, mask)) == 0.0

    def test_error_on_real_frames_is_counted(self, robot):
        trainer = _trainer(robot)
        k = KeypointTrainer.n_keypoints(robot)
        pred = torch.zeros(1, 2, k, 3)
        target = torch.full((1, 2, k, 3), 0.5)
        mask = torch.ones(1, 2)
        assert float(trainer.compute_loss(pred, target, mask)) > 0.0


class TestLoop:
    def test_loading_checks_the_cache_against_the_reader(self, tmp_path, robot):
        cache_root, annotation_root = _corpus(tmp_path, robot)
        trainer = _trainer(robot)
        trainer.reader_id = "aloha_bimanual.sv1"
        with pytest.raises(ValueError, match="was built for reader"):
            trainer.load_episodes(cache_root, annotation_root, "train")

    def test_episodes_load_with_their_targets(self, tmp_path, robot):
        cache_root, annotation_root = _corpus(tmp_path, robot)
        trainer = _trainer(robot)
        episodes = trainer.load_episodes(cache_root, annotation_root, "train")
        assert len(episodes) == 4
        path, target = episodes[0]
        assert path.endswith(".pt")
        assert trainer.read_window(path, 0, target.shape[0]).shape[0] == target.shape[0]

    def test_read_window_returns_only_the_window(self, tmp_path, robot):
        cache_root, annotation_root = _corpus(tmp_path, robot)
        trainer = _trainer(robot)
        path, target = trainer.load_episodes(
            cache_root, annotation_root, "train")[0]
        assert trainer.read_window(path, 1, 3).shape[0] == 3

    def test_read_window_past_the_end_comes_back_short(self, tmp_path, robot):
        cache_root, annotation_root = _corpus(tmp_path, robot)
        trainer = _trainer(robot)
        path, target = trainer.load_episodes(
            cache_root, annotation_root, "train")[0]
        n = int(target.shape[0])
        assert trainer.read_window(path, n - 2, 10).shape[0] == 2

    def test_streamed_evaluate_matches_reading_the_clip_whole(self, tmp_path,
                                                               robot):
        # evaluate() scores an episode in t_max chunks rather than holding it.
        # n_frames is deliberately not a multiple of t_max, so the last chunk
        # is short and the ragged boundary is exercised.
        cache_root, annotation_root = _corpus(tmp_path, robot, n_episodes=1,
                                              n_frames=13)
        trainer = _trainer(robot)
        head = trainer.head
        episodes = trainer.load_episodes(cache_root, annotation_root, "val")
        got = trainer.evaluate(head, episodes)["keypoint_mm"]

        path, target = episodes[0]
        chunk = head.t_max
        assert target.shape[0] % chunk != 0
        head.eval()
        with torch.no_grad():
            whole = trainer.read_window(path, 0, target.shape[0]).float()[None]
            pred = torch.cat([head(whole[:, i:i + chunk])[0]
                              for i in range(0, target.shape[0], chunk)])
            e = (pred - target).norm(dim=-1)
            want = float((e ** 2).sum() / e.numel()) ** 0.5 * 1000.0
        assert abs(got - want) < 1e-6

    def test_read_window_is_writable(self, tmp_path, robot):
        # The mapped file is read-only; the copy handed back must not be.
        cache_root, annotation_root = _corpus(tmp_path, robot)
        trainer = _trainer(robot)
        path, _target = trainer.load_episodes(
            cache_root, annotation_root, "train")[0]
        w = trainer.read_window(path, 0, 2)
        w += 1.0

    def test_an_empty_split_is_an_error_not_an_empty_run(self, tmp_path, robot):
        (tmp_path / "cache" / "train").mkdir(parents=True)
        (tmp_path / "annotation" / "train").mkdir(parents=True)
        trainer = _trainer(robot)
        with pytest.raises(RuntimeError, match="real-joint annotation"):
            trainer.load_episodes(str(tmp_path / "cache"),
                                  str(tmp_path / "annotation"), "train")

    def test_fit_reports_both_splits_and_the_best_state(self, tmp_path, robot):
        cache_root, annotation_root = _corpus(tmp_path, robot)
        trainer = _trainer(robot)
        train_eps = trainer.load_episodes(cache_root, annotation_root, "train")
        val_eps = trainer.load_episodes(cache_root, annotation_root, "val")

        result = trainer.fit(train_episodes=train_eps, val_episodes=val_eps)

        assert len(result.loss_history) == 4
        assert result.train_mm > 0 and result.val_mm > 0
        assert 1 <= result.best_step <= 4
        assert set(result.best_state_dict) == set(trainer.head.state_dict())

    def test_fit_without_validation_reports_none(self, tmp_path, robot):
        cache_root, annotation_root = _corpus(tmp_path, robot)
        trainer = _trainer(robot)
        train_eps = trainer.load_episodes(cache_root, annotation_root, "train")
        result = trainer.fit(train_episodes=train_eps)
        assert result.val_mm is None and result.best_val_mm is None

    def test_a_clip_longer_than_t_max_still_scores(self, tmp_path, robot):
        cache_root, annotation_root = _corpus(tmp_path, robot, n_episodes=1,
                                              n_frames=20)  # t_max is 8
        trainer = _trainer(robot)
        episodes = trainer.load_episodes(cache_root, annotation_root, "val")
        assert trainer.evaluate(trainer.head, episodes)["keypoint_mm"] > 0
