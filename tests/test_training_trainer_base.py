"""``kinescore.training.trainer_base``: the shared windowed loop runs for
both recipes, and ``KeypointTrainer``'s loss drops.

The ``trainer_base`` sibling of ``tests/test_training_rawrad_smoke.py`` (see
that file's own docstring for the pattern this follows): CPU-only, tiny
synthetic ``(feat, q)`` episodes -- no cache files, no backbone, no real
checkpoint -- fed straight to :meth:`PoseTrainerBase.fit` via hand-built
``(feat, target)`` episode lists (bypassing :meth:`PoseTrainerBase.load_episodes`,
which needs real cache/annotation files on disk). What this pins is
:class:`~kinescore.training.trainer_base.KeypointTrainer` itself: ``n_out ==
3*K`` for a real robot, the loss trends down over a few steps, and
``head_out_width``/``validate_head`` reject a mismatched head -- plus a quick
check that ``JointTrainer`` (the legacy recipe) also runs one step on the
same shared loop.
"""
from __future__ import annotations

import torch

from kinescore.heads.heteroscedastic import ReadoutV2Head
from kinescore.robots import get_robot
from kinescore.training.trainer_base import (
    JointTrainer,
    KeypointTrainer,
    TrainConfigPose,
)


def _synthetic_episodes(n_eps: int, min_t: int, max_t: int, n_tokens: int,
                        embed_dim: int, n_joints: int, robot, seed: int = 0):
    """``n_eps`` synthetic ``(feat, q)`` episodes of random length in
    ``[min_t, max_t]`` -- ``q`` sampled inside ``robot``'s joint limits so
    :meth:`KeypointTrainer.build_target`'s FK call never sees an
    out-of-range angle."""
    g = torch.Generator().manual_seed(seed)
    lo, hi = robot.q_lo, robot.q_hi
    episodes = []
    for _i in range(n_eps):
        t = int(torch.randint(min_t, max_t + 1, (1,), generator=g))
        feat = torch.randn(t, n_tokens, embed_dim, generator=g)
        u = torch.rand(t, n_joints, generator=g)
        q = lo + u * (hi - lo)
        episodes.append((feat, q))
    return episodes


class TestKeypointTrainer:
    def test_head_out_width_is_3k_for_franka(self):
        robot = get_robot("franka_panda", device="cpu")
        k = len(robot.keypoint_links)
        assert KeypointTrainer.head_out_width(robot) == 3 * k

    def test_validate_head_rejects_mismatched_n_out(self):
        robot = get_robot("franka_panda", device="cpu")
        bad_head = ReadoutV2Head(in_dim=8, d_model=16, n_heads=2, temporal_nhead=2,
                                 ff=16, n_temporal_layers=1, t_max=8,
                                 n_out=KeypointTrainer.head_out_width(robot) + 1)
        try:
            KeypointTrainer(bad_head, robot)
        except ValueError as exc:
            assert "n_out" in str(exc)
        else:
            raise AssertionError("expected a ValueError on n_out mismatch")

    def test_loss_trends_down_over_a_few_steps(self):
        torch.manual_seed(0)
        robot = get_robot("franka_panda", device="cpu")
        n_tokens, embed_dim = 6, 8
        n_out = KeypointTrainer.head_out_width(robot)

        head = ReadoutV2Head(in_dim=embed_dim, d_model=16, n_heads=2,
                             temporal_nhead=2, ff=16, n_temporal_layers=1,
                             t_max=8, n_out=n_out)
        cfg = TrainConfigPose(steps=40, phase_a=15, batch_size=4, window_size=4,
                              lr=5e-3, eval_every=0, seed=0, log_every=0,
                              device="cpu")
        trainer = KeypointTrainer(head, robot, cfg)
        assert trainer.head.n_out == n_out

        raw_episodes = _synthetic_episodes(6, 3, 10, n_tokens, embed_dim,
                                           robot.n_joints, robot, seed=1)
        train_episodes = [(feat, trainer.build_target(q)) for feat, q in raw_episodes]

        result = trainer.fit(train_episodes=train_episodes)

        assert result.steps == 40
        assert len(result.loss_history) == 40
        assert all(torch.isfinite(torch.tensor(v)) for v in result.loss_history)

        first_10 = sum(result.loss_history[:10]) / 10
        last_10 = sum(result.loss_history[-10:]) / 10
        assert last_10 < first_10, (
            f"loss did not decrease: first10={first_10:.5f} last10={last_10:.5f}")

        assert result.train_keypoint_mm == result.train_keypoint_mm  # not NaN
        assert result.train_keypoint_mm >= 0.0

    def test_validation_split_tracks_a_best_checkpoint(self):
        torch.manual_seed(0)
        robot = get_robot("franka_panda", device="cpu")
        n_tokens, embed_dim = 4, 8
        n_out = KeypointTrainer.head_out_width(robot)

        head = ReadoutV2Head(in_dim=embed_dim, d_model=16, n_heads=2,
                             temporal_nhead=2, ff=16, n_temporal_layers=1,
                             t_max=8, n_out=n_out)
        cfg = TrainConfigPose(steps=8, phase_a=4, batch_size=4, window_size=4,
                              lr=5e-3, eval_every=4, seed=0, log_every=0,
                              device="cpu")
        trainer = KeypointTrainer(head, robot, cfg)

        raw_train = _synthetic_episodes(4, 5, 8, n_tokens, embed_dim,
                                        robot.n_joints, robot, seed=2)
        raw_val = _synthetic_episodes(2, 5, 8, n_tokens, embed_dim,
                                      robot.n_joints, robot, seed=3)
        train_episodes = [(f, trainer.build_target(q)) for f, q in raw_train]
        val_episodes = [(f, trainer.build_target(q)) for f, q in raw_val]

        result = trainer.fit(train_episodes=train_episodes, val_episodes=val_episodes)

        assert result.val_keypoint_mm is not None
        assert result.val_keypoint_mm >= 0.0
        assert result.best_val_keypoint_mm is not None
        assert result.best_step > 0
        assert result.best_state_dict  # non-empty

        fresh = ReadoutV2Head(in_dim=embed_dim, d_model=16, n_heads=2,
                              temporal_nhead=2, ff=16, n_temporal_layers=1,
                              t_max=8, n_out=n_out)
        fresh.load_state_dict(result.best_state_dict)
        ev = trainer.eval_val(fresh, val_episodes)
        assert abs(ev["keypoint_mm"] - result.best_val_keypoint_mm) < 1e-3


class TestJointTrainerOnSharedLoop:
    def test_head_out_width_is_n_joints_and_a_few_steps_run(self):
        torch.manual_seed(0)
        robot = get_robot("franka_panda", device="cpu")
        assert JointTrainer.head_out_width(robot) == robot.n_joints

        n_tokens, embed_dim = 4, 8
        head = ReadoutV2Head(in_dim=embed_dim, d_model=16, n_heads=2,
                             temporal_nhead=2, ff=16, n_temporal_layers=1,
                             t_max=8, n_out=robot.n_joints)
        cfg = TrainConfigPose(steps=10, phase_a=4, batch_size=4, window_size=4,
                              lr=5e-3, eval_every=0, seed=0, log_every=0,
                              device="cpu")
        trainer = JointTrainer(head, robot, cfg)

        raw_episodes = _synthetic_episodes(4, 5, 8, n_tokens, embed_dim,
                                           robot.n_joints, robot, seed=4)
        train_episodes = [(f, trainer.build_target(q)) for f, q in raw_episodes]

        result = trainer.fit(train_episodes=train_episodes)
        assert result.steps == 10
        assert len(result.loss_history) == 10
        assert all(torch.isfinite(torch.tensor(v)) for v in result.loss_history)
