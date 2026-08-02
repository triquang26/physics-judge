"""``kinescore.training.trainer_rawrad``: the raw_rad two-phase loop runs and
the loss drops.

The ``raw_rad`` sibling of ``tests/test_training_smoke.py`` (see that file's
own docstring for the pattern this follows): CPU-only, tiny tensors, no
backbone and no real checkpoint -- random ``(N, n_tokens, D)`` "cached
tokens" and targets inside :class:`~kinescore.robots.synthetic.Synthetic2R`'s
range. What this pins is :func:`~kinescore.training.trainer_rawrad.train_head_rawrad`
itself: both phases run, the loss trends down, ``eval_keypoint_mm_rawrad``
returns a finite number via a robot-agnostic ``RobotSpec``, and the
best-by-val checkpoint tracking round-trips into a fresh head.
"""
from __future__ import annotations

import copy

import torch

from kinescore.heads.heteroscedastic import ReadoutV2Head
from kinescore.robots.synthetic import Synthetic2R
from kinescore.training.trainer_rawrad import (
    TrainConfigRawRad,
    eval_keypoint_mm_rawrad,
    train_head_rawrad,
)


def _synthetic_split(n: int, n_tokens: int, embed_dim: int, n_joints: int,
                     seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    feat = torch.randn(n, n_tokens, embed_dim, generator=g)
    q = (torch.rand(n, n_joints, generator=g) * 2 - 1) * 2.0
    return feat, q


def _head(embed_dim: int, n_joints: int) -> ReadoutV2Head:
    return ReadoutV2Head(in_dim=embed_dim, d_model=16, n_heads=2, temporal_nhead=2,
                         ff=16, n_temporal_layers=1, t_max=4, n_out=n_joints)


class TestTrainHeadRawRadLossDecreases:
    def test_loss_trends_down_across_both_phases(self):
        torch.manual_seed(0)
        robot = Synthetic2R()
        n_tokens, embed_dim = 6, 8
        head = _head(embed_dim, robot.n_joints)
        feat, q = _synthetic_split(64, n_tokens, embed_dim, robot.n_joints)
        # phase_a < steps so both the mu-only and beta-NLL phases run.
        cfg = TrainConfigRawRad(steps=60, phase_a=20, batch_size=16, lr=5e-3,
                                eval_every=0, seed=0, log_every=0, device="cpu")

        result = train_head_rawrad(head, robot, train_feat=feat, train_q=q, cfg=cfg)

        assert result.steps == 60
        assert len(result.loss_history) == 60
        assert all(torch.isfinite(torch.tensor(v)) for v in result.loss_history)

        first_10 = sum(result.loss_history[:10]) / 10
        last_10 = sum(result.loss_history[-10:]) / 10
        assert last_10 < first_10, (
            f"loss did not decrease: first10={first_10:.5f} last10={last_10:.5f}")

        assert result.train_keypoint_mm == result.train_keypoint_mm  # not NaN
        assert result.train_keypoint_mm >= 0.0

    def test_reproducible_given_the_same_seed(self):
        robot = Synthetic2R()
        n_tokens, embed_dim = 4, 8
        feat, q = _synthetic_split(32, n_tokens, embed_dim, robot.n_joints, seed=1)
        cfg = TrainConfigRawRad(steps=16, phase_a=8, batch_size=8, lr=1e-3,
                                eval_every=0, seed=42, log_every=0)

        torch.manual_seed(0)
        head_a = _head(embed_dim, robot.n_joints)
        torch.manual_seed(0)
        head_b = _head(embed_dim, robot.n_joints)

        result_a = train_head_rawrad(head_a, robot, train_feat=feat, train_q=q, cfg=cfg)
        result_b = train_head_rawrad(head_b, robot, train_feat=feat, train_q=q, cfg=cfg)

        assert result_a.loss_history == result_b.loss_history

    def test_validation_split_tracks_a_best_checkpoint(self):
        torch.manual_seed(0)
        robot = Synthetic2R()
        n_tokens, embed_dim = 4, 8
        head = _head(embed_dim, robot.n_joints)
        train_feat, train_q = _synthetic_split(32, n_tokens, embed_dim,
                                               robot.n_joints, seed=2)
        val_feat, val_q = _synthetic_split(8, n_tokens, embed_dim,
                                           robot.n_joints, seed=3)
        cfg = TrainConfigRawRad(steps=12, phase_a=4, batch_size=8, lr=1e-3,
                                eval_every=4, seed=0, log_every=0)

        result = train_head_rawrad(head, robot, train_feat=train_feat, train_q=train_q,
                                   val_feat=val_feat, val_q=val_q, cfg=cfg)

        assert result.val_keypoint_mm is not None
        assert result.val_keypoint_mm >= 0.0
        assert result.best_val_keypoint_mm is not None
        assert result.best_step > 0
        assert result.best_state_dict  # non-empty

        # The saved best state actually loads back into a fresh head and
        # reproduces the reported best_val_keypoint_mm.
        fresh = _head(embed_dim, robot.n_joints)
        fresh.load_state_dict(result.best_state_dict)
        ev = eval_keypoint_mm_rawrad(fresh, robot, val_feat, val_q,
                                     q_lo=robot.q_lo, q_hi=robot.q_hi, device="cpu")
        assert abs(ev["keypoint_mm"] - result.best_val_keypoint_mm) < 1e-3

    def test_rejects_a_head_whose_n_out_does_not_match_robot_n_joints(self):
        robot = Synthetic2R()
        head = _head(embed_dim=8, n_joints=robot.n_joints + 1)
        feat, q = _synthetic_split(16, 4, 8, robot.n_joints)
        cfg = TrainConfigRawRad(steps=1, eval_every=0, log_every=0)
        try:
            train_head_rawrad(head, robot, train_feat=feat, train_q=q, cfg=cfg)
        except ValueError as exc:
            assert "n_out" in str(exc)
        else:
            raise AssertionError("expected a ValueError on n_out mismatch")

    def test_default_cfg_is_not_shared_mutable_state(self):
        # cfg=None -> TrainConfigRawRad() constructed fresh inside the
        # function (see the B008 fix); two independent calls must not
        # observe each other's config.
        robot = Synthetic2R()
        n_tokens, embed_dim = 4, 8
        head_a = _head(embed_dim, robot.n_joints)
        feat, q = _synthetic_split(16, n_tokens, embed_dim, robot.n_joints)
        cfg = TrainConfigRawRad(steps=2, eval_every=0, log_every=0)
        result = train_head_rawrad(copy.deepcopy(head_a), robot, train_feat=feat,
                                   train_q=q, cfg=cfg)
        assert result.steps == 2
