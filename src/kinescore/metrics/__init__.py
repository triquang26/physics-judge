"""Metric implementations: the decomposed ``PhysicsConsistency``.

Importing this package is what populates
:data:`kinescore.core.metric.REGISTRY` -- each submodule calls
:func:`kinescore.core.metric.register` at import time, so simply importing
``kinescore.metrics`` (directly, or transitively via
``kinescore.metrics.suites``) is enough to make every metric below available
through :func:`kinescore.core.metric.get_metric` / :func:`.all_metrics`.

Submodule -> source method map
-------------------------------
* :mod:`.ops` -- the shared ``_fd`` finite-difference primitive (verbatim).
* :mod:`.rigidity` -- ``KinematicConsistency.rigidity_residual_mm`` /
  ``.rigidity_wobble_mm``.
* :mod:`.temporal` -- ``PhysicsConsistency.velocity/acceleration/jerk``,
  ``mean_jerk``, ``accel_violation_frac``.
* :mod:`.angular` -- ``PhysicsConsistency.angular_velocity`` / ``angular_stats``.
* :mod:`.energy` -- ``PhysicsConsistency.kinetic_energy`` / ``energy`` /
  ``momentum_continuity``.
* :mod:`.joint_limits` -- ``PhysicsConsistency.limit_violation`` (retargeted
  at ``q_raw``, see the module for why), plus the new
  ``limit_headroom_rad``.
* :mod:`.joint_dynamics` -- ``PhysicsConsistency.joint_feasibility``.
* :mod:`.smoothness` -- ``models/physics/smoothness.py``'s ``sparc`` /
  ``dimensionless_jerk`` / ``log_dimensionless_jerk`` (verbatim numpy).
* :mod:`.feasibility` -- ``models/physics/feasibility.py``'s
  ``MechanicalFeasibility.self_collision`` / ``com_balance`` / ``no_teleport``.
* :mod:`.suites` -- ``INVARIANT_V1``, the successor to
  ``PhysicsConsistency.INVARIANT_KEYS``. Not imported here (see its
  docstring for why) -- import it explicitly:
  ``from kinescore.metrics.suites import INVARIANT_V1``.
"""
from __future__ import annotations

# Side-effecting imports: each submodule registers its Metric instances with
# kinescore.core.metric.REGISTRY at import time. Order does not matter --
# there are no cross-module dependencies between metric definitions -- but is
# kept alphabetical for readability.
from kinescore.metrics import (
                               angular,
                               energy,
                               feasibility,
                               joint_dynamics,
                               joint_limits,
                               rigidity,
                               smoothness,
                               temporal,
                               torque,
)
from kinescore.metrics.angular import (
                               MaxAngularSpeed,
                               MeanAngularAccel,
                               MeanAngularSpeed,
)
from kinescore.metrics.energy import KineticEnergyTStd, MomentumDpMean, TotalEnergyTStd
from kinescore.metrics.feasibility import (
                               CoMBalance,
                               NoTeleport,
                               Penetration,
                               SelfCollision,
)
from kinescore.metrics.joint_dynamics import (
                               EffortProxy,
                               MeanQddot,
                               MeanQdot,
                               VelViolationFrac,
)
from kinescore.metrics.joint_limits import (
                               LimitExcessRad,
                               LimitHeadroomRad,
                               LimitViolationFrac,
)
from kinescore.metrics.rigidity import RigidityResidual, RigidityWobble
from kinescore.metrics.smoothness import LogDimensionlessJerk, Sparc
from kinescore.metrics.temporal import (
                               AccelViolationFrac,
                               MaxAccel,
                               MeanAccel,
                               MeanJerk,
                               MeanSpeed,
)
from kinescore.metrics.torque import TorqueFracRated

__all__ = [
    "angular", "energy", "feasibility", "joint_dynamics", "joint_limits",
    "rigidity", "smoothness", "temporal", "torque",
    "RigidityResidual", "RigidityWobble",
    "MeanSpeed", "MeanAccel", "MaxAccel", "MeanJerk", "AccelViolationFrac",
    "MeanAngularSpeed", "MaxAngularSpeed", "MeanAngularAccel",
    "KineticEnergyTStd", "TotalEnergyTStd", "MomentumDpMean",
    "LimitViolationFrac", "LimitExcessRad", "LimitHeadroomRad",
    "MeanQdot", "MeanQddot", "VelViolationFrac", "EffortProxy",
    "Sparc", "LogDimensionlessJerk",
    "Penetration", "SelfCollision", "CoMBalance", "NoTeleport",
    "TorqueFracRated",
]
