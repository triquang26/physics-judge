"""Metric contract: every metric declares what it needs and how it scales.

The source benchmarks computed metrics as methods on one big
``PhysicsConsistency`` object, which produced two structural problems that this
module exists to prevent:

1. **A varying term set.** ``invariant_residuals()`` emitted 6 keys if only
   keypoints were available, 7 with rotations, and 9-10 with joint angles. The
   aggregate score then divided by ``len(whatever_was_present)``, so two clips
   scored on different inputs produced numbers that were silently not
   comparable (defect D3).
2. **A hidden timebase.** Nothing recorded how a quantity scaled with ``dt``, so
   a wrong frame interval was undetectable by inspection (defect D1).

Here a metric is an object with a :class:`MetricSpec` that declares its units,
its ``dt`` exponent, what inputs it requires, and when it is *unobservable*.
A metric that cannot be computed returns ``NaN`` **with a reason** -- never
``0.0``, which would read as "perfect". The declared ``dt_exponent`` is verified
numerically by ``tests/test_metric_registry_conformance.py``, so a wrong
declaration fails CI rather than silently corrupting a benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import torch

from kinescore.core.clip import validate_dt

__all__ = [
    "MetricSpec", "MetricContext", "MetricValue", "Metric",
    "REGISTRY", "register", "get_metric", "all_metrics",
    "Requires", "Direction",
]

Direction = Literal["lower_better", "higher_better"]

#: Input names a metric may declare in ``MetricSpec.requires``.
#:
#: ``P``  -- keypoint positions ``(B,T,K,3)`` in metres, robot base frame
#: ``R``  -- keypoint rotations ``(B,T,K,3,3)``
#: ``q``  -- joint angles used for FK ``(B,T,n_joints)`` in radians
#: ``q_raw`` -- *unsquashed* joint angles, present only for readers whose
#:            ``limit_semantics == "raw_rad"``. This is what makes joint-limit
#:            violation observable at all (defect D7).
#: ``colliders`` / ``support_polygon`` -- robot capabilities, see RobotSpec.
Requires = Literal["P", "R", "q", "q_raw", "colliders", "support_polygon"]


@dataclass(frozen=True)
class MetricSpec:
    """Static declaration of a metric's contract.

    Parameters
    ----------
    key:
        Output column name. Must be unique across the registry.
    units:
        Physical units, e.g. ``"mm"``, ``"m/s^3"``, ``"fraction"``, ``"rad"``.
        ``"fraction"`` marks a value already in ``[0,1]``, which the reference
        normaliser treats with an absolute rather than relative tolerance.
    dt_exponent:
        Power of ``1/dt`` in the metric's value. A speed is ``1``, an
        acceleration ``2``, a jerk ``3``; a purely geometric quantity such as a
        bone length is ``0``. Verified numerically -- if you declare 2 and the
        metric actually scales as ``1/dt^3``, the conformance test fails.

        Metrics that are not homogeneous in ``dt`` (``total_energy_*`` mixes a
        ``dt^-2`` kinetic term with a ``dt^0`` potential one) must declare
        ``dt_exponent=None`` and are excluded from the conformance check, with
        the reason recorded in ``docs/METRICS.md``.
    direction:
        Whether lower or higher values indicate more physically plausible motion.
    requires:
        Inputs that must be present for the metric to be computable.
    min_frames:
        Minimum ``T``. Below this the metric yields ``NaN`` with a reason rather
        than a degenerate number -- e.g. an unbiased temporal std over a single
        frame is ``NaN`` and would poison any mean it fed (defect D7 in the
        rigidity family).
    unobservable_when:
        Conditions under which this metric is structurally incapable of firing,
        as ``"attribute=value"`` strings checked against
        :attr:`MetricContext.flags`. Historically this gated
        ``limit_violation_frac``/``limit_excess_rad`` on
        ``"limit_semantics=squashed"`` (a head that sigmoid-squashes joint
        angles into the URDF limits can never exceed them, so joint-limit
        violation would be exactly zero for a reason that has nothing to do
        with the video being scored) -- that head family is gone (see
        ``legacy_docs/PROVENANCE.md``'s D7 addendum) and no metric currently declares
        this, but the mechanism stays available for the next metric that is
        structurally blind under some head/robot combination rather than
        merely missing an input.
    perframe:
        Whether the metric also emits a per-frame array into the sidecar NPZ.
    """

    key: str
    units: str
    dt_exponent: int | None
    direction: Direction
    requires: frozenset[str]
    min_frames: int = 1
    unobservable_when: tuple[str, ...] = ()
    perframe: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if not self.key or not self.key.replace("_", "").isalnum():
            raise ValueError(f"metric key {self.key!r} must be alphanumeric/underscore")
        bad = set(self.requires) - {"P", "R", "q", "q_raw", "colliders",
                                    "support_polygon"}
        if bad:
            raise ValueError(f"{self.key}: unknown requires {sorted(bad)}")
        if self.min_frames < 1:
            raise ValueError(f"{self.key}: min_frames must be >= 1")
        for cond in self.unobservable_when:
            if "=" not in cond:
                raise ValueError(
                    f"{self.key}: unobservable_when entry {cond!r} must be "
                    f"'flag=value'")


@dataclass
class MetricContext:
    """Everything a metric may read, assembled once per clip.

    ``dt`` is a required field with no default. Constructing a context without
    one is a ``TypeError`` at the call site, which is the point: the source's
    ``dt: float = 0.2`` default is exactly how the timebase bug survived.
    """

    dt: float
    P: torch.Tensor | None = None            # (B,T,K,3) metres
    R: torch.Tensor | None = None            # (B,T,K,3,3)
    q: torch.Tensor | None = None            # (B,T,n_joints) radians
    q_raw: torch.Tensor | None = None        # (B,T,n_joints) unsquashed
    robot: Any = None                           # RobotSpec
    flags: dict[str, str] = field(default_factory=dict)
    aux: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_dt(self.dt)

    @property
    def n_frames(self) -> int:
        for t in (self.P, self.q, self.R):
            if t is not None:
                return int(t.shape[1])
        return 0

    def available(self) -> frozenset[str]:
        """Input names actually present, for matching against ``requires``."""
        got = {name for name in ("P", "R", "q", "q_raw")
               if getattr(self, name) is not None}
        caps = getattr(self.robot, "capabilities", frozenset())
        if "COLLIDERS" in caps:
            got.add("colliders")
        if "SUPPORT_POLYGON" in caps:
            got.add("support_polygon")
        return frozenset(got)


@dataclass(frozen=True)
class MetricValue:
    """A metric's result: a scalar, plus why it is missing when it is.

    ``value`` is ``NaN`` exactly when ``reason`` is set. Downstream code must
    treat ``NaN`` as "not measured" and never coerce it to zero; the JSON writer
    emits ``null`` and the parquet column is nullable so aggregates skip it
    instead of being dragged toward a fabricated zero.
    """

    key: str
    value: float
    reason: str | None = None
    perframe: Any | None = None              # np.ndarray when spec.perframe

    @property
    def available(self) -> bool:
        return self.reason is None

    @classmethod
    def unavailable(cls, key: str, reason: str) -> MetricValue:
        return cls(key=key, value=float("nan"), reason=reason)


@runtime_checkable
class Metric(Protocol):
    """A single scalar measurement over a clip.

    Implementations subclass :class:`BaseMetric` rather than implementing this
    directly; the protocol exists so the registry and the suite can be typed.
    """

    spec: MetricSpec

    def compute(self, ctx: MetricContext) -> MetricValue: ...


class BaseMetric:
    """Convenience base handling availability, frame count and NaN reasons.

    Subclasses implement :meth:`_compute` and can assume every input named in
    ``spec.requires`` is present and that ``T >= spec.min_frames``.
    """

    spec: MetricSpec

    def compute(self, ctx: MetricContext) -> MetricValue:
        reason = self.unavailable_reason(ctx)
        if reason is not None:
            return MetricValue.unavailable(self.spec.key, reason)
        return self._compute(ctx)

    def unavailable_reason(self, ctx: MetricContext) -> str | None:
        """Why this metric cannot be computed here, or ``None`` if it can."""
        for cond in self.spec.unobservable_when:
            flag, _, want = cond.partition("=")
            if ctx.flags.get(flag) == want:
                return f"unobservable:{cond}"
        missing = set(self.spec.requires) - set(ctx.available())
        if missing:
            return f"missing_input:{','.join(sorted(missing))}"
        if ctx.n_frames < self.spec.min_frames:
            return f"too_few_frames:{ctx.n_frames}<{self.spec.min_frames}"
        return None

    def _compute(self, ctx: MetricContext) -> MetricValue:  # pragma: no cover
        raise NotImplementedError

    def _ok(self, value: float, perframe: Any = None) -> MetricValue:
        return MetricValue(self.spec.key, float(value), None, perframe)


# ── registry ────────────────────────────────────────────────────────────────
REGISTRY: dict[str, Metric] = {}


def register(metric: Metric) -> Metric:
    """Register a metric instance under its ``spec.key``.

    Registration is what subjects a metric to
    ``tests/test_metric_registry_conformance.py``, which numerically verifies
    the declared ``dt_exponent``. A metric that is not registered is not part of
    any suite and is not covered by that guarantee.
    """
    key = metric.spec.key
    if key in REGISTRY:
        raise ValueError(f"metric {key!r} is already registered")
    REGISTRY[key] = metric
    return metric


def get_metric(key: str) -> Metric:
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"unknown metric {key!r}; registered: {sorted(REGISTRY)}") from None


def all_metrics() -> dict[str, Metric]:
    """Shallow copy of the registry, for tests and suite construction."""
    return dict(REGISTRY)
