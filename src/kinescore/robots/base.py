"""Shared implementation for URDF-driven robots, so a new one is mostly declaration.

Every ``pytorch_kinematics``-backed :class:`~kinescore.core.robot.RobotSpec`
repeats the same mechanical steps around the chain
:mod:`kinescore.robots.urdf` builds: read per-joint ``<limit>`` values into
canonical-order arrays, scatter a predicted joint subset into the chain's full
DOF vector, and derive rest-pose consecutive-keypoint bone geometry from one
zero-ish FK call. Those live here so a robot's ``fk.py`` writes only the
geometry specific to it: keypoint names, predicted-joint order, mount
transforms.

:mod:`kinescore.robots.urdf` owns URDF resolution (asset tree vs.
``robot_descriptions``) and hashing; this module does not duplicate either.
"""
from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

import torch

from kinescore.core.robot import DEGENERATE_BONE_M, rigid_bone_mask

__all__ = [
    "read_joint_limit_arrays",
    "consecutive_bone_pairs",
    "rest_pose_bone_lengths",
    "build_pred_chain_index",
    "scatter_predicted",
    "assert_keypoints_in_urdf",
    "structural_rigid_bone_mask",
    "warn_dropped_bones",
]


# ---------------------------------------------------------------------------
# joint limits
# ---------------------------------------------------------------------------

def read_joint_limit_arrays(urdf_path: str | Path, joint_names: Sequence[str], *,
                            default_velocity: float = 1.0e3
                            ) -> tuple[list[float], list[float], list[float],
                                       list[float | None]]:
    """Read ``<limit>`` lower/upper/velocity/effort for ``joint_names``, in order.

    Generalises the near-identical private ``_read_urdf_limits`` static
    methods :class:`~kinescore.robots.gr1.fk.GR1FK` and
    :class:`~kinescore.robots.airbot_mmk2.fk.AirbotMMK2FK` each carried
    (GR-1's without an effort column, Airbot's with one) into a single
    four-column reader every URDF-backed robot can call.

    Parameters
    ----------
    urdf_path:
        Path to the URDF file.
    joint_names:
        Revolute/prismatic joint names to read, in the order the caller wants
        the returned lists in (the robot's own canonical predicted-joint order,
        not URDF declaration order).
    default_velocity:
        Value used for a joint whose ``<limit>`` omits ``velocity`` -- a large
        cap (never a false violation), matching both source loops' own
        fallback. Effort has no equivalent fallback: a joint with no declared
        effort gets ``None`` in the returned list, since fabricating a rating
        would misrepresent the URDF (see
        :func:`kinescore.robots.urdf.parse_joint_limits`'s
        :class:`~kinescore.robots.urdf.JointLimits` docstring for the same
        principle applied there).

    Returns
    -------
    (lower, upper, velocity, effort):
        Four ``len(joint_names)``-long lists, ``float`` except ``effort``
        which is ``float | None`` per joint.

    Raises
    ------
    ValueError
        If any ``joint_names`` entry has no revolute/prismatic ``<limit>`` in
        the file -- a typo'd joint name, surfaced at construction time rather
        than as an ``IndexError``/``KeyError`` mismatch later.
    """
    root = ET.parse(str(urdf_path)).getroot()
    lim: dict[str, tuple[float, float, float, float | None]] = {}
    for j in root.findall("joint"):
        limit_el = j.find("limit")
        if limit_el is not None and j.get("type") in ("revolute", "prismatic"):
            v = limit_el.get("velocity")
            e = limit_el.get("effort")
            lim[j.get("name")] = (
                float(limit_el.get("lower")), float(limit_el.get("upper")),
                float(v) if v is not None else default_velocity,
                float(e) if e is not None else None,
            )
    missing = [n for n in joint_names if n not in lim]
    if missing:
        raise ValueError(
            f"joint(s) {missing} have no revolute/prismatic <limit> in {urdf_path}")
    lo = [lim[n][0] for n in joint_names]
    hi = [lim[n][1] for n in joint_names]
    vel = [lim[n][2] for n in joint_names]
    eff = [lim[n][3] for n in joint_names]
    return lo, hi, vel, eff


# ---------------------------------------------------------------------------
# rest-pose consecutive-keypoint bones
# ---------------------------------------------------------------------------

def consecutive_bone_pairs(k: int) -> torch.Tensor:
    """``(k-1, 2)`` long tensor ``[[0,1],[1,2],...,[k-2,k-1]]``.

    The "each keypoint bones to the next one in the list" convention every
    robot in this package uses (see ``core/robot.py``'s module docstring for
    why that convention -- not a full URDF parent/child graph -- is what
    ``bone_pairs`` means here).
    """
    return torch.tensor([[i, i + 1] for i in range(k - 1)], dtype=torch.long)


def rest_pose_bone_lengths(rest_positions: torch.Tensor,
                           bone_pairs: torch.Tensor) -> torch.Tensor:
    """Consecutive-keypoint distances at one FK-evaluated pose.

    Parameters
    ----------
    rest_positions:
        ``(K, 3)`` keypoint positions from a single FK call at whatever pose
        the caller considers "rest" (zero predicted joints for GR-1/Airbot;
        closed-gripper aux for a robot with a gripper -- see
        :class:`~kinescore.robots.franka.fk.FrankaFK._compute_rest_bones` for
        why "closed gripper" is the Franka convention this mirrors).
    bone_pairs:
        ``(K-1, 2)`` long, typically :func:`consecutive_bone_pairs`.

    Returns
    -------
    torch.Tensor
        ``(K-1,)`` fp32 rest lengths in metres, detached (this is a static
        geometry constant computed once at construction, never part of an
        autograd graph).
    """
    diffs = rest_positions[bone_pairs[:, 1]] - rest_positions[bone_pairs[:, 0]]
    return diffs.norm(dim=-1).detach().to(torch.float32).cpu()


# ---------------------------------------------------------------------------
# predicted-joint scatter into a full pytorch_kinematics chain input
# ---------------------------------------------------------------------------

def build_pred_chain_index(chain_joint_names: Sequence[str],
                           pred_joint_names: Sequence[str]) -> torch.Tensor:
    """``LongTensor`` mapping each predicted joint to its slot in the chain.

    Computed once at construction (see
    :class:`~kinescore.robots.franka.spec.FrankaSpec`'s own docstring, point
    3, for why a one-time index buffer beats a per-call ``list.index`` loop
    on the scoring hot path) and registered as a buffer by the caller so it
    follows the module across ``.to(device)``.

    Raises
    ------
    ValueError
        If a predicted joint name is not one of the chain's joint parameter
        names -- a typo'd/renamed joint, caught here rather than surfacing as
        a silent all-zero column downstream.
    """
    names = list(chain_joint_names)
    missing = [n for n in pred_joint_names if n not in names]
    if missing:
        raise ValueError(
            f"predicted joint(s) {missing} not in chain joint names {names}")
    return torch.tensor([names.index(n) for n in pred_joint_names], dtype=torch.long)


def scatter_predicted(q_flat: torch.Tensor, pred_chain_idx: torch.Tensor,
                      n_chain_joints: int) -> torch.Tensor:
    """``(N, n_pred) -> (N, n_chain_joints)``, zero everywhere else.

    The scatter every ``_full_theta``/``_theta`` FK entry point in this
    package performs: predicted joints land at their
    :func:`build_pred_chain_index` slot, every non-predicted chain joint
    (legs, neck, an unmodelled hand -- whatever the robot does not predict)
    stays at ``0.0``, i.e. the URDF's rest pose for that joint.
    """
    idx = pred_chain_idx.to(q_flat.device)
    th = q_flat.new_zeros(q_flat.shape[0], n_chain_joints)
    th[:, idx] = q_flat
    return th


# ---------------------------------------------------------------------------
# keypoint-presence validation
# ---------------------------------------------------------------------------

def assert_keypoints_in_urdf(available: set[str],
                             keypoints: dict[str, Sequence[str]], *,
                             ee_link: dict[str, str] | None = None) -> None:
    """Raise if any per-side keypoint (or EE link) is absent from the chain.

    Generalises the identical validation loop
    :class:`~kinescore.robots.gr1.fk.GR1FK` and
    :class:`~kinescore.robots.airbot_mmk2.fk.AirbotMMK2FK` each ran by hand.

    Parameters
    ----------
    available:
        ``chain.get_frame_names(exclude_fixed=False)`` as a set.
    keypoints:
        ``{side: keypoint_link_names}``, e.g. ``{"left": KEYPOINTS_LEFT,
        "right": KEYPOINTS_RIGHT}``.
    ee_link:
        Optional ``{side: link_name}`` end-effector links to check alongside
        the keypoint lists.
    """
    for side, kps in keypoints.items():
        missing = [k for k in kps if k not in available]
        if missing:
            raise ValueError(f"{side} keypoint links absent from URDF: {missing}")
        if ee_link is not None and ee_link[side] not in available:
            raise ValueError(f"EE link {ee_link[side]} absent from URDF")


# ---------------------------------------------------------------------------
# D9: structural + degenerate-length rigid-bone mask
# ---------------------------------------------------------------------------

def structural_rigid_bone_mask(keypoint_links: Sequence[str],
                               bone_pairs: torch.Tensor,
                               bone_lengths: torch.Tensor,
                               actuated_links: frozenset[str], *,
                               min_length_m: float = DEGENERATE_BONE_M
                               ) -> torch.Tensor:
    """Generalises :meth:`FrankaSpec._rigid_bone_mask`'s two-rule pattern (D9).

    A bone survives into ``rigid_bone_pairs`` only if BOTH hold:

    1. **Structural** -- neither endpoint link is in ``actuated_links`` (a
       link whose pose depends on a joint outside the robot's predicted
       state -- a gripper finger being the recurring example). Read off the
       keypoint chain, transfers to any robot; this is the primary rule.
    2. **Degenerate-length** -- the rest length exceeds ``min_length_m``
       (:data:`~kinescore.core.robot.DEGENERATE_BONE_M` by default). A second,
       independent safety net: a bone can be non-degenerate by length yet
       still actuation-tracking (rule 1 catches it), or degenerate by length
       with both endpoints genuinely arm-structure (rule 2 catches it, e.g.
       Airbot MMK2's zero-offset joint pairs, which have no
       ``actuated_links`` set at all). See ``docs/ARCHITECTURE.md#adding-a-robot``'s D9
       checklist item for why neither rule alone suffices.

    Parameters
    ----------
    keypoint_links:
        Ordered link names indexed by ``bone_pairs``.
    bone_pairs:
        ``(n_bones, 2)`` long.
    bone_lengths:
        ``(n_bones,)`` rest lengths in metres.
    actuated_links:
        Link names driven by a non-predicted actuator. Empty for a robot with
        no such link (rule 1 becomes a no-op and rule 2 alone applies, e.g.
        GR-1's arms or Airbot MMK2).

    Returns
    -------
    torch.Tensor
        ``(n_bones,)`` bool, ``True`` for bones to keep in ``rigid_bone_pairs``.
    """
    actuated = torch.tensor(
        [keypoint_links[i] in actuated_links or keypoint_links[j] in actuated_links
         for i, j in bone_pairs.tolist()], dtype=torch.bool)
    long_enough = rigid_bone_mask(bone_lengths, min_length_m=min_length_m)
    return (~actuated) & long_enough


def warn_dropped_bones(robot_name: str, keypoint_links: Sequence[str],
                       bone_pairs: torch.Tensor, bone_lengths: torch.Tensor,
                       mask: torch.Tensor) -> None:
    """Warn once, by name, for every bone :func:`structural_rigid_bone_mask` drops.

    Generalises :meth:`FrankaSpec._warn_dropped_bones` -- silent filtering
    here is exactly the "why is my rigidity number different" class of bug
    that costs an afternoon instead of one warning at construction time.
    """
    dropped = [
        f"{keypoint_links[i]}->{keypoint_links[j]} "
        f"(rest length {bone_lengths[k].item():.4f} m)"
        for k, (i, j) in enumerate(bone_pairs.tolist()) if not mask[k]
    ]
    if dropped:
        warnings.warn(
            f"{robot_name}: excluded from rigid_bone_pairs (endpoint is a "
            f"non-predicted-actuator link, or the bone is degenerate at rest "
            f"-- either way its length tracks actuation, not arm rigidity): "
            + "; ".join(dropped),
            stacklevel=3,
        )
