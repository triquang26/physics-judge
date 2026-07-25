"""Franka rigid-bone filtering: dropping bones the gripper (not the arm) drives.

See ``kinescore.robots.franka.constants.RIGID_BONE_MIN_M`` for the full
reasoning: 3 of the 7 consecutive-keypoint bones touch ``panda_leftfinger`` /
``panda_rightfinger``, whose positions track the gripper's own prismatic
joint rather than the rigid 7-DOF arm, so ``FrankaSpec`` excludes all three
from ``rigid_bone_pairs`` (not just the one with an exactly-zero rest
length). This test pins the resulting shapes/values and asserts the warning
that names what was dropped.

Requires ``pytorch_kinematics`` + a resolvable ``robot_descriptions`` Panda
URDF; skipped entirely when unavailable.
"""
import warnings

import pytest
import torch

pytest.importorskip("pytorch_kinematics")


def _franka_spec_with_warnings():
    from kinescore.robots.franka.spec import FrankaSpec
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spec = FrankaSpec()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Franka URDF / robot_descriptions unavailable: {exc}")
    return spec, caught


def test_full_bone_set_has_seven_bones_bone5_degenerate():
    spec, _ = _franka_spec_with_warnings()
    assert spec.bone_pairs.shape == (7, 2)
    assert spec.bone_lengths.shape == (7,)
    assert spec.bone_lengths[5].item() == pytest.approx(0.0, abs=1e-9)
    i, j = spec.bone_pairs[5].tolist()
    assert spec.keypoint_links[i] == "panda_leftfinger"
    assert spec.keypoint_links[j] == "panda_rightfinger"


def test_rigid_bone_pairs_drops_three_finger_touching_bones():
    spec, _ = _franka_spec_with_warnings()
    assert spec.rigid_bone_pairs.shape == (4, 2)
    assert spec.rigid_bone_lengths.shape == (4,)

    kept_names = [
        (spec.keypoint_links[i], spec.keypoint_links[j])
        for i, j in spec.rigid_bone_pairs.tolist()
    ]
    for a, b in kept_names:
        assert "finger" not in a and "finger" not in b, (a, b)
    assert kept_names == [
        ("panda_link1", "panda_link3"),
        ("panda_link3", "panda_link5"),
        ("panda_link5", "panda_link7"),
        ("panda_link7", "panda_hand"),
    ]
    assert torch.allclose(
        spec.rigid_bone_lengths,
        torch.tensor([0.316, 0.384, 0.088, 0.107]), atol=5e-4)


def test_dropping_emits_a_warning_naming_the_bones():
    spec, caught = _franka_spec_with_warnings()
    messages = [str(w.message) for w in caught]
    relevant = [m for m in messages if "rigid_bone_pairs" in m]
    assert relevant, f"expected a rigid_bone_pairs warning, got: {messages}"
    warning_text = relevant[0]
    for name in ("panda_leftfinger", "panda_rightfinger", "panda_hand",
                 "panda_hand_tcp"):
        assert name in warning_text, (name, warning_text)
