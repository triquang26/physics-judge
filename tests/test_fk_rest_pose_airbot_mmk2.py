"""Golden rest-pose values for the Airbot MMK2 arm FK.

Same rationale as ``test_fk_rest_pose.py`` (Franka): pinning literal bone
lengths turns "the composite URDF changed" from a silent regenerate-and-move-
on reflex into something a reviewer has to consciously approve. Here that
matters even more than for Franka/GR-1, because
``airbot_mmk2_bimanual_arms.urdf`` is not a single upstream file but a
hand-assembled composite (see ``build_airbot_mmk2_urdf.py`` and
``KINESCORE_ASSETS/airbot_mmk2/urdf/MANIFEST.json``) -- a value drifting here
could mean either an upstream DISCOVERSE change *or* a mistake in the
composition script, and this test cannot tell those apart, but it can at
least make either one loud.

Requires ``pytorch_kinematics`` + ``KINESCORE_ASSETS/airbot_mmk2/urdf/
airbot_mmk2_bimanual_arms.urdf``; skipped entirely when either is unavailable.
"""
import pytest
import torch

pytest.importorskip("pytorch_kinematics")

from kinescore.robots.airbot_mmk2.constants import (  # noqa: E402
    KEYPOINTS_LEFT,
    KEYPOINTS_RIGHT,
)

# VERIFIED (see this module's docstring, and the module docstring of
# build_airbot_mmk2_urdf.py for how the source URDF fragments were combined):
# the 5 consecutive-keypoint bone rest lengths per arm, zero joint angles.
# link1->link2 and link4->link5 are genuinely coincident at rest (a real
# zero-offset joint pair on the physical AIRBOT Play arm -- see
# airbot_play_v3_gripper_fixed.urdf's joint2/joint5 <origin xyz="0 0 0">),
# not a modelling artefact.
GOLDEN_BONE_LENGTHS_LEFT_M = [0.0, 0.2701, 0.2902, 0.0, 0.2365]
GOLDEN_BONE_LENGTHS_RIGHT_M = [0.0, 0.2701, 0.2902, 0.0, 0.2365]

GOLDEN_KEYPOINTS_LEFT = (
    "left_link1", "left_link2", "left_link3", "left_link4", "left_link5", "left_link6",
)
GOLDEN_KEYPOINTS_RIGHT = (
    "right_link1", "right_link2", "right_link3", "right_link4", "right_link5", "right_link6",
)


def _airbot_spec():
    from kinescore.robots.airbot_mmk2.spec import AirbotMMK2Spec
    try:
        return AirbotMMK2Spec()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Airbot MMK2 URDF / KINESCORE_ASSETS unavailable: {exc}")


def test_keypoint_link_tuples():
    assert KEYPOINTS_LEFT == GOLDEN_KEYPOINTS_LEFT
    assert KEYPOINTS_RIGHT == GOLDEN_KEYPOINTS_RIGHT
    spec = _airbot_spec()
    assert spec.keypoint_links == GOLDEN_KEYPOINTS_LEFT + GOLDEN_KEYPOINTS_RIGHT


def test_literal_bone_lengths():
    spec = _airbot_spec()
    assert spec.bone_lengths.shape == (10,)
    got = spec.bone_lengths.tolist()
    golden = GOLDEN_BONE_LENGTHS_LEFT_M + GOLDEN_BONE_LENGTHS_RIGHT_M
    for i, (g, want) in enumerate(zip(got, golden, strict=True)):
        assert g == pytest.approx(want, abs=5e-4), (
            f"bone {i} ({spec.keypoint_links[spec.bone_pairs[i][0]]} -> "
            f"{spec.keypoint_links[spec.bone_pairs[i][1]]}): got {g}, golden "
            f"{want}. If this fails because the composite URDF changed, read "
            f"this module's docstring before touching GOLDEN_BONE_LENGTHS_*.")


def test_bone_pairs_are_consecutive_keypoints_per_arm():
    spec = _airbot_spec()
    expected = torch.tensor(
        [[i, i + 1] for i in range(5)] + [[i, i + 1] for i in range(6, 11)],
        dtype=torch.long)
    assert torch.equal(spec.bone_pairs, expected)


def test_left_right_symmetry_at_rest():
    """Left/right arm keypoints must mirror across x=0 at the zero pose.

    This is the cheapest possible check that the hand-assembled left/right
    mount transforms (taken from DISCOVERSE's mmk2_s_g2.urdf) were applied
    correctly -- a bimanual robot whose two arms do not mirror at rest has a
    composition bug, not a URDF-content difference.
    """
    spec = _airbot_spec()
    q0 = torch.zeros(1, 1, spec.n_joints)
    P = spec.forward_kinematics(q0)[0, 0]  # (12,3)
    left = P[:6]
    right = P[6:]
    mirror = right * torch.tensor([-1.0, 1.0, -1.0])
    assert torch.allclose(left, mirror, atol=1e-4), (left - mirror).abs().max()
