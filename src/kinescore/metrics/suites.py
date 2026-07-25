"""The shared, robot-agnostic metric suite.

``INVARIANT_V1`` is the direct successor to the source's
``PhysicsConsistency.INVARIANT_KEYS`` / ``QUANTITY_KEYS`` split, restructured
to satisfy ``core/suite.py``'s fix for defect D3 (a variable term set that
silently produced non-comparable aggregates): every metric this package
registers appears in ``output_keys`` (so every clip's output row has the same
static schema, ``NaN`` + reason standing in for whatever a given robot or
reader cannot supply), and ``invariant_keys`` is an **explicit**, hand-picked
subset -- not "whatever happened to be computable" -- that forms the
normalised aggregate (PIS).

``invariant_keys`` intentionally mirrors the source's ``INVARIANT_KEYS``
exactly (ten keys), for two reasons:

1. It was already the right list -- every one of those ten is a task-
   *invariant* residual (should be small/bounded for any real robot motion,
   regardless of what the robot is doing), as opposed to a task-*dependent*
   descriptive magnitude like ``mean_speed_mps`` (a fast wipe and a slow
   insertion are both "physically plausible" at very different speeds, so
   raw speed cannot be part of a pass/fail residual).
2. Keeping it identical to the source's list means a PIS score computed here
   is the direct, nameable successor of the source's own aggregate --
   useful for provenance comparison -- rather than a new, differently-shaped
   number that happens to reuse the name.

Everything this package registers that is *not* in ``invariant_keys`` (raw
speed/accel magnitudes, the smoothness pair, the joint-space magnitudes, the
squashed-head-observable ``limit_headroom_rad``, the feasibility checks) is
still part of ``output_keys`` -- available in every result row for reporting,
distributional (W1) comparison via ``quantity_keys``, and debugging, just not
folded into the single invariant-residual aggregate.

Robot-specific suites
----------------------
No robot-specific suite is defined here yet: the metric *set* in this module
is already robot-agnostic by construction (every metric that needs a
capability the robot lacks resolves to ``NaN`` + reason rather than crashing
or silently vanishing, per each metric's own docstring), so a Franka-only or
GR-1-only suite would today be the same term list under a different
``suite_id``. Once ``kinescore.robots`` ships robot-specific extras (e.g. a
GR-1 suite that wants ``self_collision_frac``/``com_margin_m`` promoted into
its own ``invariant_keys`` because that robot can actually stand on its own
feet), add them here rather than in ``kinescore.robots`` -- suite composition
is this package's responsibility, robot capability is theirs.
"""
from __future__ import annotations

# Importing kinescore.metrics (not just its submodules) is what populates the
# registry these keys are looked up in -- see metrics/__init__.py.
import kinescore.metrics  # noqa: F401  (side effect: registers every metric)
from kinescore.core.suite import MetricSuite

__all__ = ["INVARIANT_V1"]

#: Every metric this package registers, in a fixed, documented order --
#: the suite's static output schema.
_ALL_METRIC_KEYS = (
    # geometry (dt_exponent=0)
    "rigidity_residual_mm", "rigidity_wobble_mm",
    # linear dynamics
    "mean_speed_mps", "mean_accel_mps2", "max_accel_mps2",
    "mean_jerk_mps3", "accel_violation_frac",
    # angular dynamics (needs R)
    "mean_angvel_radps", "max_angvel_radps", "mean_angacc_radps2",
    # energy / momentum
    "kinetic_energy_tstd", "total_energy_tstd", "momentum_dp_mean",
    # joint limits (needs q / q_raw)
    "limit_violation_frac", "limit_excess_rad", "limit_headroom_rad",
    # joint-space dynamics (needs q)
    "mean_qdot_radps", "mean_qddot_radps2", "vel_violation_frac", "effort_proxy",
    # dimensionless smoothness (needs robot.ee_sites())
    "sparc", "log_dimensionless_jerk",
    # mechanical feasibility (capability-gated; NaN, never 0.0, when absent)
    "penetration_mm", "self_collision_frac", "com_margin_m", "no_teleport_frac",
)

#: Task-invariant residual keys, verbatim from the source's
#: ``PhysicsConsistency.INVARIANT_KEYS`` -- see the module docstring for why
#: this list is not simply "every metric".
_INVARIANT_KEYS = (
    "rigidity_residual_mm", "rigidity_wobble_mm", "mean_jerk_mps3",
    "accel_violation_frac", "mean_angacc_radps2", "total_energy_tstd",
    "momentum_dp_mean", "limit_violation_frac", "vel_violation_frac",
    "effort_proxy",
)

#: The shared, robot-agnostic suite. Construction validates that every key in
#: ``_ALL_METRIC_KEYS`` / ``_INVARIANT_KEYS`` is actually registered and that
#: there are no duplicates (``MetricSuite.__init__``, frozen).
INVARIANT_V1 = MetricSuite(
    name="invariant_v1", metrics=list(_ALL_METRIC_KEYS),
    invariant_keys=_INVARIANT_KEYS)
