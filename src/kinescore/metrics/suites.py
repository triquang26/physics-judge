"""The shared, robot-agnostic metric suites: ``INVARIANT_V1``, ``RATE_FREE``, ``ALL_METRICS``.

Every metric this package registers appears in every suite's ``output_keys``
(``NaN`` + reason standing in for whatever a robot/reader can't supply, per
``core/suite.py``'s fix for defect D3) — but only ``invariant_keys``, an
explicit hand-picked subset, folds into the normalised PIS aggregate.
``RATE_FREE`` restricts to metrics whose ``dt_exponent`` is exactly ``0`` (see
``docs/BENCHMARKING.md`` layer 3 for why, and the `sparc`-is-excluded /
`log_dimensionless_jerk`-is-included distinction). See ``legacy_docs/DECISIONS.md``
D-E for the suite table, `invariant_keys` selection rationale, and why no
robot-specific suite exists yet.
"""
from __future__ import annotations

# Importing kinescore.metrics (not just its submodules) is what populates the
# registry these keys are looked up in -- see metrics/__init__.py.
import kinescore.metrics  # noqa: F401  (side effect: registers every metric)
from kinescore.core.metric import get_metric
from kinescore.core.suite import MetricSuite

__all__ = ["INVARIANT_V1", "RATE_FREE"]

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
#: ``PhysicsConsistency.INVARIANT_KEYS`` -- see legacy_docs/DECISIONS.md D-E.
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

#: Every ``INVARIANT_V1`` key with ``dt_exponent == 0``, derived from the
#: live registry (not hand-copied) -- see docs/BENCHMARKING.md layer 3.
_RATE_FREE_KEYS = tuple(
    key for key in _ALL_METRIC_KEYS if get_metric(key).spec.dt_exponent == 0)

#: The cross-frame-rate-safe suite. No composite score -- legacy_docs/DECISIONS.md D-E.
RATE_FREE = MetricSuite(
    name="rate_free", metrics=list(_RATE_FREE_KEYS), invariant_keys=())

#: ``INVARIANT_V1`` + ``torque_frac_rated`` + ``rigidity_worst_bone_mm``,
#: the suite to score a fresh run with. Kept separate from ``_ALL_METRIC_KEYS``
#: rather than extending ``INVARIANT_V1`` in place, since ``suite_id`` is a
#: hash of the term set and mutating it would invalidate every golden fixture.
#: See legacy_docs/DECISIONS.md D-E (why this suite, torque's physical ceiling) and
#: D-A (why ``rigidity_worst_bone_mm`` over ``rigidity_residual_mm``).
_ALL_METRICS_KEYS = _ALL_METRIC_KEYS + ("torque_frac_rated", "rigidity_worst_bone_mm")

ALL_METRICS = MetricSuite(
    name="all_metrics", metrics=list(_ALL_METRICS_KEYS),
    invariant_keys=())
