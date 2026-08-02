"""Core contracts. Import these; do not redefine them.

Every subpackage codes against this module. The layering is deliberate:

* :mod:`kinescore.core.clip` -- ``ClipSpec`` owns ``dt``. torch-free.
* :mod:`kinescore.core.robot` -- ``RobotSpec``: FK, limits, bones, capabilities.
* :mod:`kinescore.core.reader` -- ``PoseReader``: pixels -> joint angles.
* :mod:`kinescore.core.metric` -- ``Metric``/``MetricSpec``: one measurement,
  self-describing (units, ``dt`` exponent, required inputs).
* :mod:`kinescore.core.suite` -- ``MetricSuite``: a fixed, hashed term set.
* :mod:`kinescore.core.scorer` -- ``Scorer``: the facade that composes them.
* :mod:`kinescore.core.contracts` -- every extension-point interface above,
  re-exported from one door, plus ``ClipSource``/``DataLayout`` for
  ``bench``.
* :mod:`kinescore.core.registry` -- the one ``Registry[T]`` every extension
  axis (robots, suites, ``bench`` sources/layouts) can share instead of
  hand-rolling its own ``dict[str, Callable]``.
"""
from kinescore.core.clip import (
    ClipSpec,
    DtSource,
    TimebaseError,
    ViewLayout,
    validate_dt,
)
from kinescore.core.contracts import ClipSource, DataLayout
from kinescore.core.metric import (
    REGISTRY,
    BaseMetric,
    Metric,
    MetricContext,
    MetricSpec,
    MetricValue,
    all_metrics,
    get_metric,
    register,
)
from kinescore.core.reader import LimitSemantics, PoseReader, Readout
from kinescore.core.registry import Registry
from kinescore.core.robot import (
    DEGENERATE_BONE_M,
    Capability,
    RobotSpec,
    rigid_bone_mask,
)
from kinescore.core.scorer import ScoredClip, Scorer
from kinescore.core.suite import MetricSuite, SuiteResult

__all__ = [
    "ClipSpec", "ViewLayout", "DtSource", "TimebaseError", "validate_dt",
    "RobotSpec", "Capability", "DEGENERATE_BONE_M", "rigid_bone_mask",
    "PoseReader", "Readout", "LimitSemantics",
    "Metric", "MetricSpec", "MetricContext", "MetricValue", "BaseMetric",
    "REGISTRY", "register", "get_metric", "all_metrics",
    "MetricSuite", "SuiteResult", "Scorer", "ScoredClip",
    "ClipSource", "DataLayout", "Registry",
]
