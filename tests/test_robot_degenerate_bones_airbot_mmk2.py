"""Airbot MMK2 rigid-bone filtering (D9 checklist item, see docs/ARCHITECTURE.md#adding-a-robot).

Unlike the Franka gripper case (three bones dropped because an endpoint is a
*non-predicted, gripper-actuated* link) and unlike GR-1 (no degenerate bones
at all), Airbot MMK2 has two degenerate bones per arm
(``link1``->``link2``, ``link4``->``link5``) whose endpoints are BOTH driven
by predicted joints -- a genuine zero-offset joint pair on the physical
AIRBOT Play arm (see ``constants.py``'s module docstring). This is exactly
the "second, independent safety net" case the checklist describes:
``core.robot.rigid_bone_mask``'s rest-length threshold does the exclusion
here, not a structural actuated-link rule (there is no
``ACTUATED_LINKS``-equivalent set for this robot -- every link is arm
structure).

Requires ``pytorch_kinematics`` + a resolvable Airbot MMK2 URDF; skipped
entirely when unavailable.
"""
import pytest

pytest.importorskip("pytorch_kinematics")


def _airbot_spec():
    from kinescore.robots.airbot_mmk2.spec import AirbotMMK2Spec
    try:
        return AirbotMMK2Spec()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Airbot MMK2 URDF / KINESCORE_ASSETS unavailable: {exc}")


def test_full_bone_set_has_ten_bones_two_degenerate_per_arm():
    spec = _airbot_spec()
    assert spec.bone_pairs.shape == (10, 2)
    assert spec.bone_lengths.shape == (10,)
    # left: index 0 (link1->link2) and index 3 (link4->link5) are degenerate
    assert spec.bone_lengths[0].item() == pytest.approx(0.0, abs=1e-9)
    assert spec.bone_lengths[3].item() == pytest.approx(0.0, abs=1e-9)
    # right: same pattern offset by 5
    assert spec.bone_lengths[5].item() == pytest.approx(0.0, abs=1e-9)
    assert spec.bone_lengths[8].item() == pytest.approx(0.0, abs=1e-9)


def test_rigid_bone_pairs_drops_four_degenerate_bones():
    spec = _airbot_spec()
    assert spec.rigid_bone_pairs.shape == (6, 2)
    assert spec.rigid_bone_lengths.shape == (6,)
    for length in spec.rigid_bone_lengths.tolist():
        assert length > 1e-3

    kept_names = [
        (spec.keypoint_links[i], spec.keypoint_links[j])
        for i, j in spec.rigid_bone_pairs.tolist()
    ]
    assert kept_names == [
        ("left_link2", "left_link3"),
        ("left_link3", "left_link4"),
        ("left_link5", "left_link6"),
        ("right_link2", "right_link3"),
        ("right_link3", "right_link4"),
        ("right_link5", "right_link6"),
    ]


def test_effort_limits_capability_is_backed_by_real_urdf_values():
    """Airbot MMK2's URDF (unlike GR-1's) carries <limit effort=...> values,
    so EFFORT_LIMITS is declared and effort_limits is not None -- the
    opposite situation from GR1Spec, worth pinning explicitly so a future
    edit cannot silently regress one robot's honesty check into the other's.
    """
    from kinescore.core.robot import Capability
    spec = _airbot_spec()
    assert Capability.EFFORT_LIMITS in spec.capabilities
    assert spec.effort_limits is not None
    assert spec.effort_limits.shape == (spec.n_joints,)
    assert (spec.effort_limits > 0).all()


def test_no_support_polygon_no_colliders():
    """No mesh geometry, no legs/feet in this composite URDF -- both
    capabilities must be absent, not silently defaulted to a fabricated
    value. See spec.py's module docstring.
    """
    from kinescore.core.robot import Capability
    spec = _airbot_spec()
    assert Capability.SUPPORT_POLYGON not in spec.capabilities
    assert Capability.COLLIDERS not in spec.capabilities
