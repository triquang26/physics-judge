"""CPU-only tests for :class:`kinescore.robots.synthetic.Synthetic2R`.

No marks: this is the one robot test file that must run everywhere with zero
optional dependencies -- see ``kinescore/robots/synthetic.py``'s module
docstring for why (``Synthetic2R`` is what the metric-layer test suite
exercises FK-consuming metrics against).
"""
import math

import torch

from kinescore.core.robot import Capability
from kinescore.robots.synthetic import Synthetic2R


def _spec(link1=0.4, link2=0.3):
    return Synthetic2R(link1_m=link1, link2_m=link2)


def test_protocol_attributes():
    spec = _spec()
    assert spec.name == "synthetic_2r"
    assert spec.n_joints == 2
    assert spec.keypoint_links == ("base", "elbow", "tip")
    assert spec.capabilities == frozenset({Capability.ROTATIONS})
    assert spec.urdf_sha256 is None
    assert spec.vel_limits is None
    assert spec.effort_limits is None
    assert spec.ee_sites() == (2,)


def test_bones_are_link_lengths_and_nothing_dropped():
    spec = _spec(link1=0.4, link2=0.3)
    assert torch.equal(spec.bone_pairs, torch.tensor([[0, 1], [1, 2]]))
    assert torch.allclose(spec.bone_lengths, torch.tensor([0.4, 0.3]))
    # No degenerate bone on a synthetic arm with positive link lengths, so
    # nothing is dropped -- unlike the Franka gripper.
    assert torch.equal(spec.rigid_bone_pairs, spec.bone_pairs)
    assert torch.allclose(spec.rigid_bone_lengths, spec.bone_lengths)


def test_fk_rest_pose_zero_angles():
    spec = _spec(link1=0.4, link2=0.3)
    q = torch.zeros(1, 1, 2)
    P = spec.forward_kinematics(q)
    assert P.shape == (1, 1, 3, 3)
    expected = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.7, 0.0, 0.0]])
    assert torch.allclose(P[0, 0], expected, atol=1e-6)


def test_fk_quarter_turn_first_joint():
    spec = _spec(link1=0.4, link2=0.3)
    q = torch.tensor([[[math.pi / 2, 0.0]]])
    P = spec.forward_kinematics(q)
    expected = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.7, 0.0]])
    assert torch.allclose(P[0, 0], expected, atol=1e-6)


def test_fk_second_joint_folds_perpendicular():
    spec = _spec(link1=0.4, link2=0.3)
    q = torch.tensor([[[0.0, math.pi / 2]]])
    P = spec.forward_kinematics(q)
    expected = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.4, 0.3, 0.0]])
    assert torch.allclose(P[0, 0], expected, atol=1e-6)


def test_fk_preserves_bone_lengths_under_rotation():
    """Rigidity sanity: for ANY q, base->elbow and elbow->tip distances equal
    the nominal link lengths -- this is the property rigidity metrics check."""
    spec = _spec(link1=0.4, link2=0.3)
    torch.manual_seed(0)
    q = (torch.rand(4, 5, 2) * 2 - 1) * math.pi
    P = spec.forward_kinematics(q)
    d01 = (P[..., 1, :] - P[..., 0, :]).norm(dim=-1)
    d12 = (P[..., 2, :] - P[..., 1, :]).norm(dim=-1)
    assert torch.allclose(d01, torch.full_like(d01, 0.4), atol=1e-5)
    assert torch.allclose(d12, torch.full_like(d12, 0.3), atol=1e-5)


def test_forward_transforms_rotations():
    spec = _spec()
    q = torch.tensor([[[math.pi / 2, math.pi / 4]]])
    P, R = spec.forward_transforms(q)
    assert P.shape == (1, 1, 3, 3)
    assert R.shape == (1, 1, 3, 3, 3)
    assert torch.allclose(R[0, 0, 0], torch.eye(3), atol=1e-6)

    t1 = math.pi / 2
    expected_r1 = torch.tensor([[math.cos(t1), -math.sin(t1), 0.0],
                                 [math.sin(t1), math.cos(t1), 0.0],
                                 [0.0, 0.0, 1.0]])
    assert torch.allclose(R[0, 0, 1], expected_r1, atol=1e-6)

    t12 = math.pi / 2 + math.pi / 4
    expected_r2 = torch.tensor([[math.cos(t12), -math.sin(t12), 0.0],
                                 [math.sin(t12), math.cos(t12), 0.0],
                                 [0.0, 0.0, 1.0]])
    assert torch.allclose(R[0, 0, 2], expected_r2, atol=1e-6)


def test_forward_kinematics_matches_forward_transforms_positions():
    spec = _spec()
    q = torch.randn(2, 3, 2)
    P = spec.forward_kinematics(q)
    P2, _ = spec.forward_transforms(q)
    assert torch.allclose(P, P2)


def test_registry_returns_synthetic_without_heavy_deps():
    """``kinescore.robots.get_robot('synthetic_2r')`` must work with zero
    heavy (pytorch_kinematics/robot_descriptions) dependencies imported."""
    from kinescore.robots import available_robots, get_robot

    assert "synthetic_2r" in available_robots()
    spec = get_robot("synthetic_2r")
    assert isinstance(spec, Synthetic2R)
