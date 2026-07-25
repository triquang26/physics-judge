"""Pose reader contract: pixels -> joint angles.

A ``PoseReader`` wraps a frozen vision backbone and a trained head, and turns
video frames into the robot's joint configuration. It is deliberately the *only*
learned component in kinescore -- everything downstream (FK, physics residuals)
is analytic, so the benchmark cannot drift with a model.

``limit_semantics`` is the load-bearing field
---------------------------------------------
Two head families exist, and they differ in a way that silently changes what a
metric means:

* ``"squashed"`` -- the head emits ``q = lo + (hi-lo) * sigmoid(raw)``, so
  predicted joints are inside the URDF limits *by construction*. Joint-limit
  violation is then exactly ``0.0`` for every clip ever scored, which reads as
  "perfect" but measures nothing about the video (defect D7).
* ``"raw_rad"`` -- the head emits unconstrained radians; FK runs on a clamped
  copy and the clamp magnitude **is** the violation signal.

Rather than let this be an accident of which checkpoint was loaded, readers
declare it. Metrics that are unobservable under a squashed head return ``NaN``
with the reason ``unobservable:limit_semantics=squashed`` instead of ``0``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Protocol, runtime_checkable

import torch

from kinescore.core.clip import ViewLayout

__all__ = ["Readout", "PoseReader", "LimitSemantics"]

LimitSemantics = Literal["squashed", "raw_rad"]


@dataclass
class Readout:
    """What a pose reader produces for one clip.

    Attributes
    ----------
    q:
        ``(B,T,n_joints)`` joint angles in radians, guaranteed inside the URDF
        limits and therefore safe to feed to FK.
    q_raw:
        ``(B,T,n_joints)`` unsquashed angles, or ``None`` for a squashed head.
        When present, ``q_raw - clamp(q_raw)`` is the joint-limit excess.
    sigma:
        ``(B,T,n_joints)`` per-joint aleatoric std in radians, for
        heteroscedastic heads. ``None`` otherwise. Used to gate low-confidence
        frames out of scoring rather than to score them badly.
    aux:
        Robot-specific extras needed by FK -- gripper opening, hand DoF.
    extras:
        Anything else a metric may consume, e.g. ``P_free`` and ``gamma`` for
        the FK-projection anomaly, or ensemble disagreement.
    """

    q: torch.Tensor
    q_raw: Optional[torch.Tensor] = None
    sigma: Optional[torch.Tensor] = None
    aux: Optional[Any] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.q.shape[1])


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
