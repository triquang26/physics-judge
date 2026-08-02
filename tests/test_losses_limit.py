"""``kinescore.training.losses.loss_limit``: soft joint-limit hinge.

Pins the verbatim-ported numerics (``relu(q-hi) + relu(lo-q)``, mean over all
elements) and the property that actually matters for the ``raw_rad`` path
(defect D7, see the module docstring): unlike
:func:`~kinescore.heads.ranges.squash_to_limits`, this penalty does not
prevent an out-of-range prediction -- it only costs the optimiser something
for producing one.
"""
from __future__ import annotations

import torch

from kinescore.training.losses import loss_limit


def _limits(lo: float, hi: float, n: int) -> torch.Tensor:
    return torch.stack([torch.full((n,), lo), torch.full((n,), hi)], dim=-1)


class TestLossLimit:
    def test_zero_inside_limits(self):
        limits = _limits(-1.0, 1.0, 3)
        q = torch.tensor([[0.0, 0.5, -0.9]])
        assert float(loss_limit(q, limits)) == 0.0

    def test_zero_exactly_at_boundary(self):
        limits = _limits(-1.0, 1.0, 2)
        q = torch.tensor([[-1.0, 1.0]])
        assert float(loss_limit(q, limits)) == 0.0

    def test_positive_and_matches_hand_computation_above_upper(self):
        limits = _limits(-1.0, 1.0, 2)
        q = torch.tensor([[1.5, 0.0]])  # joint 0 overshoots hi by 0.5
        expected = 0.5 / 2  # mean over 2 elements
        assert float(loss_limit(q, limits)) == expected

    def test_positive_and_matches_hand_computation_below_lower(self):
        limits = _limits(-1.0, 1.0, 2)
        q = torch.tensor([[0.0, -1.7]])  # joint 1 undershoots lo by 0.7
        expected = 0.7 / 2
        assert abs(float(loss_limit(q, limits)) - expected) < 1e-6

    def test_never_squashes_the_prediction_itself(self):
        # The defining difference from squash_to_limits (D7): this is a
        # penalty computed FROM q, not a transform OF q -- an out-of-range
        # q_hat is passed through unchanged; only the loss value reflects it.
        limits = _limits(-1.0, 1.0, 1)
        q = torch.tensor([[5.0]])
        loss_limit(q, limits)
        assert float(q[0, 0]) == 5.0

    def test_broadcasts_over_leading_batch_and_time_dims(self):
        limits = _limits(-1.0, 1.0, 2)
        q = torch.zeros(3, 4, 2)
        q[0, 0, 0] = 2.0  # one overshoot of 1.0 rad among 24 elements
        expected = 1.0 / (3 * 4 * 2)
        assert abs(float(loss_limit(q, limits)) - expected) < 1e-6

    def test_differentiable_and_pushes_gradient_toward_the_feasible_set(self):
        limits = _limits(-1.0, 1.0, 1)
        q = torch.tensor([[2.0]], requires_grad=True)
        loss = loss_limit(q, limits)
        loss.backward()
        # Over the upper limit -> positive gradient -> a gradient step
        # (param -= lr * grad) moves q back down toward hi.
        assert q.grad is not None
        assert float(q.grad[0, 0]) > 0.0
