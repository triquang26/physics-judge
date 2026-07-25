"""Golden rest-pose values for the Franka Panda FK.

Why literal numbers instead of a recomputation
------------------------------------------------
If ``robot_descriptions`` or ``pytorch_kinematics`` ever changes the Panda
URDF (a version bump, a mesh/inertial correction, a differently-tuned
collision model), every keypoint-based value in this repo drifts at once, and
the natural reflex when a test like this fails is "the URDF changed, let's
regenerate the golden numbers" -- which would silently fold the exact defect
this benchmark exists to catch (rigidity_residual_mm = 15.37 on a rigid,
motionless arm that merely opens its gripper; see
``kinescore.robots.franka.constants.RIGID_BONE_MIN_M`` and
``docs/PROVENANCE.md``, D9) into the new baseline without anyone deciding to
accept that. Pinning the literal values here, with a diff in version control
as the only way to change them, turns "regenerate the golden file" from a
silent reflex into something a reviewer has to consciously approve.

Requires ``pytorch_kinematics`` + a resolvable ``robot_descriptions`` Panda
URDF (network-cached on first use); skipped entirely when either is
unavailable rather than failing the default test run.
"""
import pytest
import torch

pytest.importorskip("pytorch_kinematics")

from kinescore.robots.franka.constants import KEYPOINT_LINKS  # noqa: E402

# VERIFIED (see this module's docstring): the 7 consecutive-keypoint bone
# rest lengths for the robot_descriptions Panda URDF, closed gripper ("aux"
# omitted), zero arm angles.
GOLDEN_BONE_LENGTHS_M = [0.316, 0.384, 0.088, 0.107, 0.0584, 0.0, 0.045]

GOLDEN_KEYPOINT_LINKS = (
    "panda_link1", "panda_link3", "panda_link5", "panda_link7",
    "panda_hand", "panda_leftfinger", "panda_rightfinger", "panda_hand_tcp",
)


def _franka_spec():
    from kinescore.robots.franka.spec import FrankaSpec
    try:
        return FrankaSpec()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Franka URDF / robot_descriptions unavailable: {exc}")


def test_keypoint_link_tuple():
    assert KEYPOINT_LINKS == GOLDEN_KEYPOINT_LINKS
    spec = _franka_spec()
    assert spec.keypoint_links == GOLDEN_KEYPOINT_LINKS


def test_literal_bone_lengths():
    spec = _franka_spec()
    assert spec.bone_lengths.shape == (7,)
    got = spec.bone_lengths.tolist()
    for i, (g, want) in enumerate(zip(got, GOLDEN_BONE_LENGTHS_M, strict=True)):
        assert g == pytest.approx(want, abs=5e-4), (
            f"bone {i} ({spec.keypoint_links[i]} -> {spec.keypoint_links[i + 1]}): "
            f"got {g}, golden {want}. If this fails because robot_descriptions "
            f"or pytorch_kinematics changed the Panda URDF, read this module's "
            f"docstring before touching GOLDEN_BONE_LENGTHS_M.")


def test_bone_pairs_are_consecutive_keypoints():
    spec = _franka_spec()
    expected = torch.tensor([[i, i + 1] for i in range(7)], dtype=torch.long)
    assert torch.equal(spec.bone_pairs, expected)
