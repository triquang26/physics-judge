"""ALOHA bimanual rigid-bone filtering (D9 checklist item, see docs/ARCHITECTURE.md#adding-a-robot).

Mirrors the Franka gripper case (three bones per arm dropped because an
endpoint is a non-predicted, gripper-actuated link), doubled for two arms --
see ``constants.py``'s ``ACTUATED_LINKS`` docstring and
``legacy_docs/ADDING_ALOHA_NOTES.md``'s "Keypoint / D9 plan" section for the
worked-out reasoning this implements.

Requires ``pytorch_kinematics`` + a resolvable ALOHA URDF; skipped entirely
when unavailable.
"""
import pytest

pytest.importorskip("pytorch_kinematics")


def _aloha_spec():
    from kinescore.robots.aloha.spec import AlohaSpec
    try:
        return AlohaSpec()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"ALOHA URDF / KINESCORE_ASSETS unavailable: {exc}")


def test_full_bone_set_has_sixteen_bones_six_gripper_actuated():
    spec = _aloha_spec()
    assert spec.bone_pairs.shape == (16, 2)
    assert spec.bone_lengths.shape == (16,)
    # left arm: indices 5,6,7 touch a finger link (gripper-actuated);
    # right arm: same pattern offset by 8.
    for i in (5, 6, 7, 13, 14, 15):
        assert spec.bone_lengths[i].item() > 0.0  # non-degenerate by LENGTH --
        # excluded structurally (D9 rule 1), not by the length safety net.


def test_rigid_bone_pairs_drops_six_gripper_actuated_bones():
    spec = _aloha_spec()
    assert spec.rigid_bone_pairs.shape == (10, 2)
    assert spec.rigid_bone_lengths.shape == (10,)
    for length in spec.rigid_bone_lengths.tolist():
        assert length > 1e-3

    kept_names = [
        (spec.keypoint_links[i], spec.keypoint_links[j])
        for i, j in spec.rigid_bone_pairs.tolist()
    ]
    assert kept_names == [
        ("left/shoulder_link", "left/upper_arm_link"),
        ("left/upper_arm_link", "left/upper_forearm_link"),
        ("left/upper_forearm_link", "left/lower_forearm_link"),
        ("left/lower_forearm_link", "left/wrist_link"),
        ("left/wrist_link", "left/gripper_link"),
        ("right/shoulder_link", "right/upper_arm_link"),
        ("right/upper_arm_link", "right/upper_forearm_link"),
        ("right/upper_forearm_link", "right/lower_forearm_link"),
        ("right/lower_forearm_link", "right/wrist_link"),
        ("right/wrist_link", "right/gripper_link"),
    ]


def test_dropped_bones_are_exactly_the_gripper_actuated_ones():
    spec = _aloha_spec()
    dropped_names = {
        (spec.keypoint_links[i], spec.keypoint_links[j])
        for k, (i, j) in enumerate(spec.bone_pairs.tolist())
        if k not in {0, 1, 2, 3, 4, 8, 9, 10, 11, 12}
    }
    assert dropped_names == {
        ("left/gripper_link", "left/left_finger_link"),
        ("left/left_finger_link", "left/right_finger_link"),
        ("left/right_finger_link", "left/ee_gripper_link"),
        ("right/gripper_link", "right/left_finger_link"),
        ("right/left_finger_link", "right/right_finger_link"),
        ("right/right_finger_link", "right/ee_gripper_link"),
    }


def test_effort_limits_capability_is_backed_by_real_urdf_values():
    """ALOHA's URDF (like Airbot MMK2's, unlike GR-1's) carries
    ``<limit effort=...>`` values for every predicted arm joint, so
    ``EFFORT_LIMITS`` is declared and ``effort_limits`` is not ``None``.
    """
    from kinescore.core.robot import Capability
    spec = _aloha_spec()
    assert Capability.EFFORT_LIMITS in spec.capabilities
    assert spec.effort_limits is not None
    assert spec.effort_limits.shape == (spec.n_joints,)
    assert (spec.effort_limits > 0).all()


def test_no_support_polygon_no_colliders():
    """No mesh geometry (kinematics-only URDF), no legs/feet (bolted to a
    table, like Franka) -- both capabilities must be absent, not silently
    defaulted. See spec.py's module docstring.
    """
    from kinescore.core.robot import Capability
    spec = _aloha_spec()
    assert Capability.SUPPORT_POLYGON not in spec.capabilities
    assert Capability.COLLIDERS not in spec.capabilities


def test_registered_under_aloha_bimanual_not_bare_aloha():
    """The registry key is ``"aloha_bimanual"`` -- the old, descoped configs'
    bare ``"aloha"`` key is not registered (see spec.py's module docstring
    and legacy_docs/ADDING_ALOHA_NOTES.md)."""
    from kinescore.robots import available_robots
    assert "aloha_bimanual" in available_robots()
    assert "aloha" not in available_robots()


def test_gripper_aux_moves_only_finger_and_tcp_keypoints():
    """Opening both grippers must move the finger/TCP keypoints (6,7,8 per
    arm) and leave every arm-structure keypoint (0-4) exactly where it was --
    the whole point of routing the gripper through ``aux`` rather than ``q``.
    """
    import torch
    spec = _aloha_spec()
    q0 = torch.zeros(1, 1, spec.n_joints)
    closed = spec.forward_kinematics(q0, torch.zeros(1, 1, 2))
    open_ = spec.forward_kinematics(q0, torch.ones(1, 1, 2))

    arm_idx = list(range(0, 5)) + list(range(9, 14))
    finger_idx = [6, 7, 15, 16]
    assert torch.allclose(closed[0, 0, arm_idx], open_[0, 0, arm_idx], atol=1e-6)
    assert (closed[0, 0, finger_idx] - open_[0, 0, finger_idx]).abs().max() > 1e-3
