"""FrankaSpec: registered, protocol-complete, rigid where it claims to be."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_kinematics")
pytest.importorskip("robot_descriptions")

from kinescore.core.robot import RobotSpec
from kinescore.robots import available_robots, get_robot


@pytest.fixture(scope="module")
def spec():
    return get_robot("franka_panda")


def test_registered_and_protocol_complete(spec):
    assert "franka_panda" in available_robots()
    assert isinstance(spec, RobotSpec)
    assert spec.n_joints == 7
    assert len(spec.keypoint_links) == 8


def test_fk_shapes(spec):
    q = torch.zeros(2, 3, 7)
    P = spec.forward_kinematics(q)
    assert P.shape == (2, 3, 8, 3)


def test_rest_pose_matches_declared_bone_lengths(spec):
    P = spec.forward_kinematics(torch.zeros(1, 1, 7))
    d = (P[..., spec.rigid_bone_pairs[:, 0], :]
         - P[..., spec.rigid_bone_pairs[:, 1], :]).norm(dim=-1)
    assert torch.allclose(d, spec.rigid_bone_lengths.expand_as(d), atol=1e-4)
