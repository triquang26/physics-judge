"""A1XEESpec: rigid by construction, protocol-complete, registered."""
from __future__ import annotations

import torch

from kinescore.core.robot import RobotSpec
from kinescore.robots import available_robots, get_robot
from kinescore.robots.a1x_ee.spec import EE_OFFSETS_M, A1XEESpec


def _random_q(b: int = 2, t: int = 7, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q = torch.empty(b, t, 6)
    q[..., :3] = torch.rand(b, t, 3, generator=g) * 0.8 - 0.4
    q[..., 3:] = (torch.rand(b, t, 3, generator=g) - 0.5) * 2 * torch.pi
    return q


class TestProtocol:
    def test_registered_and_protocol_complete(self):
        assert "a1x_ee" in available_robots()
        spec = get_robot("a1x_ee")
        assert isinstance(spec, RobotSpec)
        assert spec.n_joints == 6
        assert len(spec.keypoint_links) == len(EE_OFFSETS_M)

    def test_fk_shapes(self):
        spec = A1XEESpec()
        q = _random_q()
        P = spec.forward_kinematics(q)
        assert P.shape == (2, 7, 4, 3)
        P2, R = spec.forward_transforms(q)
        assert torch.equal(P, P2)
        assert R.shape == (2, 7, 4, 3, 3)

    def test_rejects_wrong_q_shape(self):
        spec = A1XEESpec()
        try:
            spec.forward_kinematics(torch.zeros(2, 7, 5))
        except ValueError:
            return
        raise AssertionError("expected ValueError for q with 5 columns")


class TestRigidBody:
    def test_bone_lengths_hold_under_any_pose(self):
        spec = A1XEESpec()
        P = spec.forward_kinematics(_random_q(b=3, t=11, seed=1))
        d = (P[..., spec.rigid_bone_pairs[:, 0], :]
             - P[..., spec.rigid_bone_pairs[:, 1], :]).norm(dim=-1)
        assert torch.allclose(d, spec.rigid_bone_lengths.expand_as(d),
                              atol=1e-5)

    def test_nothing_is_degenerate(self):
        spec = A1XEESpec()
        assert torch.equal(spec.bone_pairs, spec.rigid_bone_pairs)
        assert (spec.rigid_bone_lengths > 1e-3).all()

    def test_pure_translation_moves_every_keypoint_equally(self):
        spec = A1XEESpec()
        q = _random_q(b=1, t=1)
        shifted = q.clone()
        shifted[..., :3] += torch.tensor([0.1, -0.2, 0.3])
        delta = (spec.forward_kinematics(shifted)
                 - spec.forward_kinematics(q))
        assert torch.allclose(delta, delta[..., :1, :].expand_as(delta),
                              atol=1e-6)

    def test_zero_rotation_places_offsets_verbatim(self):
        spec = A1XEESpec()
        q = torch.zeros(1, 1, 6)
        q[..., :3] = torch.tensor([0.2, -0.1, 0.15])
        P = spec.forward_kinematics(q)[0, 0]
        want = q[0, 0, :3] + torch.tensor(EE_OFFSETS_M)
        assert torch.allclose(P, want, atol=1e-6)

    def test_rotations_are_orthonormal(self):
        spec = A1XEESpec()
        _, R = spec.forward_transforms(_random_q())
        eye = torch.eye(3).expand_as(R)
        assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-5)
        assert torch.allclose(torch.linalg.det(R), torch.ones(R.shape[:-2]),
                              atol=1e-5)

    def test_aux_is_ignored(self):
        spec = A1XEESpec()
        q = _random_q(b=1, t=3)
        assert torch.equal(spec.forward_kinematics(q, None),
                           spec.forward_kinematics(q, torch.rand(1, 3, 1)))
