"""URDF joint/link names for the Airbot MMK2 arms."""
from __future__ import annotations

__all__ = [
    "LEFT_ARM_JOINTS", "RIGHT_ARM_JOINTS",
    "KEYPOINTS_LEFT", "KEYPOINTS_RIGHT", "EE_LINK",
]

#: URDF joint names (`airbot_mmk2_bimanual_arms.urdf`), matching the parquet's
#: `{left,right}_arm_joint_{1..6}_rad` order (unit suffix dropped).
LEFT_ARM_JOINTS: tuple[str, ...] = tuple(f"left_arm_joint_{i}" for i in range(1, 7))
RIGHT_ARM_JOINTS: tuple[str, ...] = tuple(f"right_arm_joint_{i}" for i in range(1, 7))

#: Per-arm keypoint links (shoulder -> ... -> wrist/flange), K=6. `link1`/
#: `link2` and `link4`/`link5` coincide at rest (real zero-offset joints, not
#: a modelling artefact) -- dropped from `rigid_bone_pairs` by the
#: degenerate-length safety net.
KEYPOINTS_LEFT: tuple[str, ...] = tuple(f"left_link{i}" for i in range(1, 7))
KEYPOINTS_RIGHT: tuple[str, ...] = tuple(f"right_link{i}" for i in range(1, 7))

#: No dedicated end-effector link exists (see `build_airbot_mmk2_urdf.py`) --
#: link6 (the flange) doubles as the end-effector site for each arm.
EE_LINK: dict[str, str] = {"left": "left_link6", "right": "right_link6"}
