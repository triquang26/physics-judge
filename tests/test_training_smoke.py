"""``kinescore.training.trainer``: the head-only loop runs and the loss drops.

CPU-only, tiny tensors, no backbone and no real checkpoint: features are
random ``(N, n_tokens, D)`` "cached tokens" and targets are random joint
angles inside :class:`~kinescore.robots.synthetic.Synthetic2R`'s range --
exactly the seam :mod:`kinescore.training.cache`/``.datasets`` would
otherwise sit behind (see their own tests). What this test pins is
:func:`~kinescore.training.trainer.train_head` itself: it runs the declared
number of steps, the loss trends down, and :func:`eval_keypoint_mm` (the FK
eval metric) returns a finite number via a robot-agnostic ``RobotSpec``, not
one hardcoded to Franka.
"""
from __future__ import annotations

import torch

from kinescore.heads.attentive import AttentivePoseHead
from kinescore.robots.synthetic import Synthetic2R
from kinescore.training.trainer import TrainConfig, eval_keypoint_mm, train_head


def _synthetic_split(n: int, n_tokens: int, embed_dim: int, n_joints: int,
                     seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    feat = torch.randn(n, n_tokens, embed_dim, generator=g)
    q = (torch.rand(n, n_joints, generator=g) * 2 - 1) * 2.0  # in (-2, 2) subset of (-pi, pi)
    gripper = torch.rand(n, 1, generator=g)
    return feat, q, gripper


class TestTrainHeadLossDecreases:
    def test_loss_trends_down_over_training(self):
        torch.manual_seed(0)
        robot = Synthetic2R()
        n_tokens, embed_dim = 6, 8
        head = AttentivePoseHead(in_dim=embed_dim, hidden=16, n_joints=robot.n_joints,
                                 n_heads=2, n_cams=1)

        feat, q, gripper = _synthetic_split(64, n_tokens, embed_dim, robot.n_joints)
        cfg = TrainConfig(steps=150, batch_size=16, lr=5e-3, seed=0, log_every=0,
                          device="cpu")

        result = train_head(head, robot, train_feat=feat, train_q=q,
                            train_gripper=gripper, cfg=cfg)

        assert result.steps == 150
        assert len(result.loss_history) == 150
        assert all(torch.isfinite(torch.tensor(v)) for v in result.loss_history)

        first_10 = sum(result.loss_history[:10]) / 10
        last_10 = sum(result.loss_history[-10:]) / 10
        assert last_10 < first_10, (
            f"loss did not decrease: first10={first_10:.5f} last10={last_10:.5f}")

        assert result.train_keypoint_mm == result.train_keypoint_mm  # not NaN
        assert result.train_keypoint_mm >= 0.0
        assert result.train_grip_mae is not None
        assert result.train_grip_mae >= 0.0

    def test_reproducible_given_the_same_seed(self):
        torch.manual_seed(1)
        robot = Synthetic2R()
        n_tokens, embed_dim = 4, 8
        feat, q, gripper = _synthetic_split(32, n_tokens, embed_dim, robot.n_joints, seed=1)
        cfg = TrainConfig(steps=20, batch_size=8, lr=1e-3, seed=42, log_every=0)

        torch.manual_seed(0)
        head_a = AttentivePoseHead(in_dim=embed_dim, hidden=16, n_joints=robot.n_joints,
                                   n_heads=2, n_cams=1)
        torch.manual_seed(0)
        head_b = AttentivePoseHead(in_dim=embed_dim, hidden=16, n_joints=robot.n_joints,
                                   n_heads=2, n_cams=1)

        result_a = train_head(head_a, robot, train_feat=feat, train_q=q,
                              train_gripper=gripper, cfg=cfg)
        result_b = train_head(head_b, robot, train_feat=feat, train_q=q,
                              train_gripper=gripper, cfg=cfg)

        assert result_a.loss_history == result_b.loss_history

    def test_validation_split_is_evaluated_when_given(self):
        torch.manual_seed(0)
        robot = Synthetic2R()
        n_tokens, embed_dim = 4, 8
        head = AttentivePoseHead(in_dim=embed_dim, hidden=16, n_joints=robot.n_joints,
                                 n_heads=2, n_cams=1)
        train_feat, train_q, train_gripper = _synthetic_split(32, n_tokens, embed_dim,
                                                               robot.n_joints, seed=2)
        val_feat, val_q, val_gripper = _synthetic_split(8, n_tokens, embed_dim,
                                                         robot.n_joints, seed=3)
        cfg = TrainConfig(steps=10, batch_size=8, lr=1e-3, seed=0, log_every=0)

        result = train_head(head, robot, train_feat=train_feat, train_q=train_q,
                            train_gripper=train_gripper, val_feat=val_feat,
                            val_q=val_q, val_gripper=val_gripper, cfg=cfg)

        assert result.val_keypoint_mm is not None
        assert result.val_keypoint_mm >= 0.0
        assert result.val_grip_mae is not None

    def test_no_gripper_head_trains_without_a_gripper_target(self):
        # A head that emits exactly n_joints (no trailing gripper column) --
        # has_gripper is inferred from train_gripper being None, and the two
        # must agree in shape or the squash raises immediately.
        import torch.nn as nn

        class JointsOnlyHead(nn.Module):
            def __init__(self, in_dim, n_joints):
                super().__init__()
                self.proj = nn.Linear(in_dim, n_joints)

            def forward(self, feat):  # (B,T,N,D) -> (B,T,n_joints)
                return self.proj(feat.mean(dim=2))

        robot = Synthetic2R()
        n_tokens, embed_dim = 4, 8
        head = JointsOnlyHead(embed_dim, robot.n_joints)
        feat, q, _ = _synthetic_split(32, n_tokens, embed_dim, robot.n_joints)
        cfg = TrainConfig(steps=20, batch_size=8, lr=1e-2, seed=0, log_every=0)

        result = train_head(head, robot, train_feat=feat, train_q=q,
                            train_gripper=None, cfg=cfg)
        assert len(result.loss_history) == 20
        assert result.train_grip_mae is None


class TestEvalKeypointMm:
    def test_finite_and_matches_keypoint_count(self):
        torch.manual_seed(0)
        robot = Synthetic2R()
        n_tokens, embed_dim = 4, 8
        head = AttentivePoseHead(in_dim=embed_dim, hidden=16, n_joints=robot.n_joints,
                                 n_heads=2, n_cams=1)
        feat, q, gripper = _synthetic_split(16, n_tokens, embed_dim, robot.n_joints)

        out = eval_keypoint_mm(head, robot, feat, q, gripper, q_lo=robot.q_lo,
                               q_hi=robot.q_hi, device="cpu")
        assert out["keypoint_mm"] >= 0.0
        assert out["per_keypoint_mm"].shape == (len(robot.keypoint_links),)
        assert out["grip_mae"] is not None and out["grip_mae"] >= 0.0
