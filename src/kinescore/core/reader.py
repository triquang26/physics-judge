"""Pose reader contract: pixels -> joint angles.

A ``PoseReader`` wraps a frozen vision backbone and a trained head, and turns
video frames into the robot's joint configuration. It is deliberately the *only*
learned component in kinescore -- everything downstream (FK, physics residuals)
is analytic, so the benchmark cannot drift with a model.

``limit_semantics`` is the load-bearing field
---------------------------------------------
Historically two head families existed here, and they differed in a way that
silently changed what a metric means: a ``"squashed"`` head emitted
``q = lo + (hi-lo) * sigmoid(raw)``, so predicted joints were inside the URDF
limits *by construction* -- joint-limit violation was then exactly ``0.0``
for every clip ever scored, which read as "perfect" but measured nothing
about the video (defect D7). That family (the ``squash_to_limits`` function,
its head, and its reader) has been removed entirely -- see
``legacy_docs/PROVENANCE.md``'s D7 addendum -- so ``"raw_rad"`` is now the only
value: the head emits unconstrained radians; FK runs on a clamped copy and
the clamp magnitude **is** the violation signal.

``limit_semantics`` stays a declared field rather than being deleted outright
because a reader should still say what it is instead of every caller
assuming ``"raw_rad"`` by convention -- and because the general
``unobservable_when`` mechanism it fed (see ``core/metric.py``) remains
available to a future head family that is structurally blind to some other
metric, for whatever reason that turns out to be.

A third value now exists for a reason unrelated to D7: ``"keypoints"``
(:class:`~kinescore.readers.direct_keypoint.DirectKeypointPoseReader`) skips
joint angles and FK entirely, reading 3-D keypoints straight off the frames
into :attr:`Readout.P`. It has no clamp and no limits to be raw or squashed
about, so it is not a third flavour of ``"raw_rad"`` -- it is a different
axis (pixels -> points, not pixels -> joints) that happens to share this
same reader contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import torch

from kinescore.core.clip import ViewLayout

__all__ = ["Readout", "PoseReader", "LimitSemantics"]

#: ``"keypoints"`` is a second, structurally different reader family: a head
#: that regresses 3-D keypoints directly, with no forward-kinematics and no
#: joint-limit clamp anywhere in the path (see
#: ``readers/direct_keypoint.py``). It carries no joint-limit semantics at
#: all -- there is no ``q``/``clamp_for_fk`` to be "raw_rad" or otherwise
#: about -- so it gets its own value rather than overloading ``"raw_rad"``
#: for a reader that has no limits to violate.
LimitSemantics = Literal["raw_rad", "keypoints"]


@dataclass
class Readout:
    """What a pose reader produces for one clip.

    Attributes
    ----------
    q:
        ``(B,T,n_joints)`` joint angles in radians, guaranteed inside the URDF
        limits and therefore safe to feed to FK. ``None`` for a reader that
        has no joints at all (``limit_semantics="keypoints"``) -- such a
        reader sets :attr:`P` instead.
    q_raw:
        ``(B,T,n_joints)`` unclamped angles, or ``None`` for a reader that
        does not expose them. ``q_raw - clamp(q_raw)`` is the joint-limit
        excess when present.
    sigma:
        ``(B,T,n_joints)`` or ``(B,T,K)`` per-output aleatoric std, for
        heteroscedastic heads. ``None`` otherwise. Used to gate low-confidence
        frames out of scoring rather than to score them badly.
    aux:
        Robot-specific extras needed by FK -- gripper opening, hand DoF.
    extras:
        Anything else a metric may consume, e.g. ``P_free`` and ``gamma`` for
        the FK-projection anomaly, or ensemble disagreement.
    P:
        ``(B,T,K,3)`` 3-D keypoints in the robot-base frame, metres. Set by a
        :class:`~kinescore.readers.direct_keypoint.DirectKeypointPoseReader`
        (whose ``q``/``q_raw`` are ``None`` -- it never runs FK or a joint
        clamp); ``None`` for every joint-angle reader.
    """

    q: torch.Tensor | None
    q_raw: torch.Tensor | None = None
    sigma: torch.Tensor | None = None
    aux: Any | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    P: torch.Tensor | None = None

    @property
    def n_frames(self) -> int:
        if self.q is not None:
            return int(self.q.shape[1])
        if self.P is not None:
            return int(self.P.shape[1])
        raise ValueError(
            "Readout has neither q nor P set -- cannot determine n_frames")


@runtime_checkable
class PoseReader(Protocol):
    """Frozen backbone + trained head that reads joint angles from frames.

    Attributes
    ----------
    limit_semantics:
        See the module docstring. Propagated into
        :attr:`~kinescore.core.metric.MetricContext.flags` so metrics can gate
        on it.
    view_layout:
        Camera packing the reader was trained for. Checked against the clip's
        layout before scoring, so a 3-view checkpoint cannot silently consume
        single-view frames.
    robot_name:
        Which :class:`~kinescore.core.robot.RobotSpec` this reader targets.
    reader_id:
        Stable identity for the output record, e.g.
        ``"attentive/dinov3_vitl16@1024/p2/3view"``.
    """

    limit_semantics: LimitSemantics
    view_layout: ViewLayout
    robot_name: str
    reader_id: str

    def read(self, frames: torch.Tensor) -> Readout:
        """``(T,H,W,3) uint8`` or ``(B,T,3,H,W)`` float in ``[0,1]`` -> Readout."""
        ...
