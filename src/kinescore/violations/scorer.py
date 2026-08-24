"""``ViolationScorer``: calibrates every ``Detector`` on GT, scores clips.

One clip scored through :class:`ViolationScorer` produces one report per
error type, each carrying its own threshold and its own interval list -- see
:mod:`kinescore.violations.detectors` for why that per-type separation is the
whole point of this package.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from kinescore.core.context import ClipContext
from kinescore.violations.detectors import (
    Detector,
    JerkDetector,
    JointLimitDetector,
    RigidityDetector,
    SelfCollisionDetector,
    TeleportDetector,
)

__all__ = ["DETECTORS", "HEADLINE", "ViolationScorer"]

#: Per-detector calibration floor (same units as the detector), keyed by
#: ``Detector.name``. A near-zero GT spread on rigidity/joint_limit otherwise
#: calibrates a threshold so tight it flags real, violation-free motion.
#: ``self_collision`` is ``higher_is_worse=False`` (lower distance = worse), so
#: a floor there would loosen exactly the bound calibration tightens -- it has
#: none. See ``Detector.calibrate``.
_CALIBRATION_FLOOR = {"rigidity": 18.0, "joint_limit": 3.0}


def _default_detectors() -> list[Detector]:
    """One instance of each of the five built-in detectors, default-parameterised."""
    return [
        RigidityDetector(),
        JerkDetector(),
        TeleportDetector(),
        JointLimitDetector(),
        SelfCollisionDetector(),
    ]


#: Reference list of the five built-in detector *types*, one instance each,
#: default-parameterised -- e.g. ``[d.name for d in DETECTORS]``. Not the
#: list a :class:`ViolationScorer` actually calibrates against: each scorer
#: builds its own fresh instances (see ``ViolationScorer.__init__``) so
#: calibrating one scorer never mutates another's thresholds through a
#: shared ``Detector`` object.
DETECTORS: list[Detector] = _default_detectors()

#: The detectors a benchmark number is read off. Every detector in
#: ``DETECTORS`` is still computed and written to ``results.jsonl``; this is
#: what ``report`` and ``render`` show unless asked for more.
HEADLINE: tuple[str, ...] = ("rigidity", "jerk")


class ViolationScorer:
    """Runs every detector; calibrates each on GT; scores clips into per-type reports.

    Parameters
    ----------
    detectors:
        Detector instances to run. Defaults to one of each of the five
        built-in types with default parameterisation (fresh instances, not
        :data:`DETECTORS` itself). Pass your own list to use robot-specific
        parameterisation (e.g. ``RigidityDetector(rigid_idx=(0, 2, 3))`` for
        a Franka) or to score a subset of error types.
    """

    def __init__(self, detectors: Sequence[Detector] | None = None) -> None:
        self.detectors = list(detectors) if detectors is not None else _default_detectors()

    def calibrate(self, gt_contexts: Sequence[ClipContext], pct: float = 95.0) -> None:
        """Fit + threshold every detector against pooled GT per-frame scores.

        Calls ``det.fit(gt_contexts)`` first (a no-op for detectors that
        don't need it), then pools ``det.per_frame(c)`` over every GT clip
        and calibrates the threshold at ``pct``, using this detector's own
        floor from :data:`_CALIBRATION_FLOOR` (0.0 if it has none).
        """
        for det in self.detectors:
            det.fit(gt_contexts)
            scores = (np.concatenate([det.per_frame(c) for c in gt_contexts])
                      if gt_contexts else np.array([0.0]))
            floor = _CALIBRATION_FLOOR.get(det.name, 0.0)
            det.calibrate(scores, pct=pct, floor=floor)

    def thresholds(self) -> dict:
        """``{detector_name: {"units": ..., "threshold": ...}}`` for every detector."""
        return {
            d.name: {"units": d.units, "threshold": round(float(d.threshold), 2)}
            for d in self.detectors
        }

    def score(self, ctx: ClipContext) -> dict:
        """Score one clip: ``{detector_name: report}``, each with its own intervals."""
        return {det.name: det.report(ctx) for det in self.detectors}
