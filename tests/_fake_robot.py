"""A minimal, hand-built ``RobotSpec`` for metric-layer tests that don't need FK.

Not a test module itself (no ``test_*`` prefix, so pytest never collects it),
just shared support for the metric tests this agent owns
(``test_metric_registry_conformance.py``, ``test_rigidity*.py``,
``test_analytic_physics.py``, ``test_limit_semantics.py``).

Why this exists alongside ``kinescore.robots.synthetic.Synthetic2R``
-----------------------------------------------------------------------
``Synthetic2R`` is the right fixture for anything that needs real FK (its
``forward_kinematics`` is exercised by ``kinescore.robots`` tests directly).
But it deliberately has **no degenerate bone** ("unlike the Franka gripper",
per its own docstring) and **no joint-limit-violation-relevant q_raw
scenario builder** -- it is a geometry fixture, not a scenario fixture. A lot
of what this agent needs to test (bone-set contamination, joint-limit
gating, vel/effort-limit gating) is about feeding a metric a *hand-picked*
``P`` / ``q`` / ``q_raw`` array and a *hand-picked* robot geometry chosen to
land exactly on a boundary condition -- which calls for a robot object with
no FK at all, just the tensors :class:`kinescore.core.robot.RobotSpec`
declares. :class:`FakeRobot` is exactly that: every field a metric might
read, all overridable, structurally satisfying the ``RobotSpec`` Protocol
(``isinstance(FakeRobot(), RobotSpec)`` is ``True`` since it is
``@runtime_checkable`` and only checks attribute presence).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = ["FakeRobot"]


@dataclass
class FakeRobot:
    """Hand-built ``RobotSpec``. Every field has a small, inspectable default.

    Default geometry: two bones over three keypoints (``base``, ``mid``,
    ``tip``), rest lengths ``[1.0, 1.0]`` m -- deliberately not degenerate, so
    tests that *want* a degenerate bone (gripper-contamination reproduction)
    pass ``bone_lengths=torch.tensor([1.0, 0.0])`` explicitly rather than
    relying on a hidden default. ``forward_kinematics`` raises
    ``NotImplementedError`` by default: this fixture is for tests that
    construct ``P``/``q``/``R`` by hand and only need the *geometry/limits*
    side of ``RobotSpec``, not FK. Pass ``fk_fn`` for the rare test that needs
    both.
    """

    name: str = "fake_robot"
    n_joints: int = 2
    keypoint_links: tuple[str, ...] = ("base", "mid", "tip")

    bone_pairs: torch.Tensor = field(
        default_factory=lambda: torch.tensor([[0, 1], [1, 2]], dtype=torch.long))
    bone_lengths: torch.Tensor = field(
        default_factory=lambda: torch.tensor([1.0, 1.0], dtype=torch.float32))
    rigid_bone_pairs: torch.Tensor | None = None
    rigid_bone_lengths: torch.Tensor | None = None

    q_lo: torch.Tensor = field(
        default_factory=lambda: torch.tensor([-3.0, -3.0], dtype=torch.float32))
    q_hi: torch.Tensor = field(
        default_factory=lambda: torch.tensor([3.0, 3.0], dtype=torch.float32))
    vel_limits: torch.Tensor | None = None
    effort_limits: torch.Tensor | None = None

    capabilities: frozenset[str] = field(default_factory=frozenset)
    urdf_sha256: str | None = None

    _ee_sites: tuple[int, ...] = (2,)
    fk_fn: Any | None = None

    def __post_init__(self) -> None:
        # Default rigid_bone_pairs/_lengths == bone_pairs/_lengths (nothing
        # dropped) unless a test explicitly wants them to differ (the
        # gripper-contamination scenario).
        if self.rigid_bone_pairs is None:
            self.rigid_bone_pairs = self.bone_pairs.clone()
        if self.rigid_bone_lengths is None:
            self.rigid_bone_lengths = self.bone_lengths.clone()

    def forward_kinematics(self, q: torch.Tensor,
                           aux: Any | None = None) -> torch.Tensor:
        if self.fk_fn is None:
            raise NotImplementedError(
                "FakeRobot has no fk_fn; construct P by hand for this test "
                "instead of calling forward_kinematics.")
        return self.fk_fn(q, aux)[0]

    def forward_transforms(self, q: torch.Tensor, aux: Any | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fk_fn is None:
            raise NotImplementedError(
                "FakeRobot has no fk_fn; construct P/R by hand for this "
                "test instead of calling forward_transforms.")
        return self.fk_fn(q, aux)

    def ee_sites(self) -> tuple[int, ...]:
        return self._ee_sites


def with_capability(*caps: str) -> frozenset[str]:
    """Convenience: ``with_capability(Capability.COLLIDERS)`` -> a frozenset."""
    return frozenset(caps)
