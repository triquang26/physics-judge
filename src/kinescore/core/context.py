"""What a violation detector reads, assembled once per clip."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from kinescore.core.clip import validate_dt

__all__ = ["ClipContext"]


@dataclass
class ClipContext:
    """One clip's predicted keypoints plus the robot they belong to.

    ``dt`` has no default: every detector with a time dimension scales with it,
    so a caller must state the timebase rather than inherit a guess.

    Attributes
    ----------
    dt:
        Seconds between frames.
    P:
        ``(B, T, K, 3)`` keypoints in the robot-base frame, metres.
    robot:
        :class:`~kinescore.core.robot.RobotSpec`; read by the rigidity
        detector for bone pairs and rest lengths.
    flags:
        Free-form provenance carried into the output record.
    aux:
        Extras a detector may consume.
    """

    dt: float
    P: torch.Tensor
    robot: Any = None
    flags: dict[str, str] = field(default_factory=dict)
    aux: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_dt(self.dt)

    @property
    def n_frames(self) -> int:
        return int(self.P.shape[1])
