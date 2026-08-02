"""Golden rest-pose values for the ALOHA bimanual FK.

Same rationale as ``test_fk_rest_pose_airbot_mmk2.py`` (Airbot MMK2) and
``test_fk_rest_pose.py`` (Franka): pinning the LITERAL rest-pose bone lengths
turns "the composite URDF changed" from a silent regenerate-and-move-on
reflex into something a reviewer has to consciously approve -- a symbolic
``length > 0`` assertion would miss a keypoint-link list edited into a
different (but still positive-length) chain.

"Rest pose" here means BOTH arms at ``q = 0`` (all 12 predicted joints) AND
BOTH grippers fully closed (``aux = None`` -> ``gripper2 = [0, 0]``) -- the
same "closed gripper" rest convention
:class:`~kinescore.robots.franka.fk.FrankaFK._compute_rest_bones` uses, for
the same reason: a gripper has no canonical "rest" angle the way an arm joint
at 0 rad does, so this package picks closed, consistently, everywhere a
robot has one.

Requires ``pytorch_kinematics`` + ``KINESCORE_ASSETS/aloha/urdf/
aloha_bimanual.urdf``; skipped entirely when either is unavailable.
"""
import pytest
import torch

pytest.importorskip("pytorch_kinematics")

from kinescore.robots.aloha.constants import (  # noqa: E402
    KEYPOINTS_LEFT,
    KEYPOINTS_RIGHT,
)

# VERIFIED (computed from aloha_bimanual.urdf via AlohaSpec -- see this
# module's docstring for the "rest pose" convention and
# legacy_docs/ADDING_ALOHA_NOTES.md / KINESCORE_ASSETS/MANIFEST.json for the URDF's
# own provenance). Per arm, the 8 consecutive-keypoint bone rest lengths:
# indices 0-4 are pure arm-structure links (shoulder->...->gripper_link);
# indices 5-7 all have an endpoint on a gripper-actuated finger link (see
# ACTUATED_LINKS in constants.py) and are dropped from rigid_bone_pairs, but
# their rest lengths are pinned here too since bone_lengths (the full,
# legacy-reproducible set) still reports them.
GOLDEN_BONE_LENGTHS_LEFT_M = [
    0.04805, 0.30585, 0.20000, 0.10000, 0.06974,  # arm structure (kept)
    0.07184, 0.04200, 0.04385,                     # gripper-actuated (dropped)
]
GOLDEN_BONE_LENGTHS_RIGHT_M = list(GOLDEN_BONE_LENGTHS_LEFT_M)  # symmetric build

GOLDEN_KEYPOINTS_LEFT = (
    "left/shoulder_link", "left/upper_arm_link", "left/upper_forearm_link",
    "left/lower_forearm_link", "left/wrist_link", "left/gripper_link",
    "left/left_finger_link", "left/right_finger_link", "left/ee_gripper_link",
)
GOLDEN_KEYPOINTS_RIGHT = (
    "right/shoulder_link", "right/upper_arm_link", "right/upper_forearm_link",
    "right/lower_forearm_link", "right/wrist_link", "right/gripper_link",
    "right/left_finger_link", "right/right_finger_link", "right/ee_gripper_link",
)


def _aloha_spec():
    from kinescore.robots.aloha.spec import AlohaSpec
    try:
        return AlohaSpec()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"ALOHA URDF / KINESCORE_ASSETS unavailable: {exc}")


def test_keypoint_link_tuples():
    assert KEYPOINTS_LEFT == GOLDEN_KEYPOINTS_LEFT
    assert KEYPOINTS_RIGHT == GOLDEN_KEYPOINTS_RIGHT
    spec = _aloha_spec()
    assert spec.keypoint_links == GOLDEN_KEYPOINTS_LEFT + GOLDEN_KEYPOINTS_RIGHT


def test_literal_bone_lengths():
    spec = _aloha_spec()
    assert spec.bone_lengths.shape == (16,)
    got = spec.bone_lengths.tolist()
    golden = GOLDEN_BONE_LENGTHS_LEFT_M + GOLDEN_BONE_LENGTHS_RIGHT_M
    for i, (g, want) in enumerate(zip(got, golden, strict=True)):
        assert g == pytest.approx(want, abs=5e-4), (
            f"bone {i} ({spec.keypoint_links[spec.bone_pairs[i][0]]} -> "
            f"{spec.keypoint_links[spec.bone_pairs[i][1]]}): got {g}, golden "
            f"{want}. If this fails because the composite URDF changed, read "
            f"this module's docstring before touching GOLDEN_BONE_LENGTHS_*.")


def test_bone_pairs_are_consecutive_keypoints_per_arm():
    spec = _aloha_spec()
    expected = torch.tensor(
        [[i, i + 1] for i in range(8)] + [[i, i + 1] for i in range(9, 17)],
        dtype=torch.long)
    assert torch.equal(spec.bone_pairs, expected)


def test_left_right_symmetry_at_rest():
    """Left/right arm keypoints must mirror across x=0 at the zero pose.

    Cheapest possible check that the hand-assembled left/right mount
    transforms (Menagerie ALOHA's ``aloha.xml`` mount poses, see
    ``legacy_docs/ADDING_ALOHA_NOTES.md``) were applied correctly. Unlike Airbot
    MMK2's mirror (a uniform ``[-1,1,-1]`` scale), ALOHA's right arm is
    mounted at a 180-degree yaw (not a mirror reflection) composed with the
    x-translation split, so the two GRIPPER FINGER keypoints (indices 6, 7 --
    ``left_finger_link``/``right_finger_link``) swap identity between arms:
    left's ``left_finger_link`` mirrors right's ``right_finger_link``, not
    right's ``left_finger_link`` -- verified numerically (both fingers are
    the same distance from the wrist centreline, so the swap is a real
    kinematic fact about the merged URDF, not a bug to paper over).
    """
    spec = _aloha_spec()
    q0 = torch.zeros(1, 1, spec.n_joints)
    P = spec.forward_kinematics(q0)[0, 0]  # (18,3)
    left, right = P[:9], P[9:]
    mirror = right * torch.tensor([-1.0, 1.0, 1.0])

    # indices 0-5, 8 line up directly; 6 and 7 (the two fingers) swap.
    straight = [0, 1, 2, 3, 4, 5, 8]
    assert torch.allclose(left[straight], mirror[straight], atol=1e-4), (
        (left[straight] - mirror[straight]).abs().max())
    assert torch.allclose(left[6], mirror[7], atol=1e-4), (left[6] - mirror[7]).abs().max()
    assert torch.allclose(left[7], mirror[6], atol=1e-4), (left[7] - mirror[6]).abs().max()
