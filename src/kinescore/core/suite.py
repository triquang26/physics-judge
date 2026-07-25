"""A fixed, identified set of metrics -- the unit of comparability.

Why a suite has an identity
---------------------------
The source aggregated its headline score by averaging over "whatever metric keys
happened to be present", so a clip scored with joint angles produced a mean over
9-10 terms while one scored from keypoints alone produced a mean over 6. Both
were reported as the same number and compared to each other (defect D3).

A ``MetricSuite`` fixes the term set up front and hashes it into a ``suite_id``.
Every result row carries that id, and the aggregator refuses to pool rows whose
ids differ. Two numbers are comparable if and only if they were produced by the
same suite -- and that is now checkable rather than assumed.

Consequently the output schema is **static**: every clip emits every declared
key, using ``NaN`` plus a reason for metrics that could not be computed. A
metric never simply vanishes from a row.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from kinescore.core.metric import (Metric, MetricContext, MetricValue,
                                   get_metric)

__all__ = ["MetricSuite", "SuiteResult"]


@dataclass(frozen=True)
class SuiteResult:
    """All metric values for one clip, plus the suite that produced them."""

    suite_id: str
    suite_name: str
    values: Dict[str, MetricValue]

    def scalars(self) -> Dict[str, Optional[float]]:
        """``{key: value or None}`` -- ``None`` where unavailable.

        ``None`` rather than ``NaN`` so JSON emits ``null`` and parquet gets a
        true null: an aggregate then *skips* the clip instead of averaging in a
        fabricated zero.
        """
        return {k: (None if not v.available else v.value)
                for k, v in self.values.items()}

    def reasons(self) -> Dict[str, str]:
        """``{key: reason}`` for every unavailable metric."""
        return {k: v.reason for k, v in self.values.items()
                if v.reason is not None}

    def perframe(self) -> Dict[str, object]:
        return {k: v.perframe for k, v in self.values.items()
                if v.perframe is not None}


class MetricSuite:
    """An ordered, hashed collection of metrics evaluated together.

    Parameters
    ----------
    name:
        Human-readable suite name, e.g. ``"invariant_v1"``.
    metrics:
        Metric instances, or registry keys to look up.
    invariant_keys:
        Subset of keys forming the normalised aggregate (PIS). Declaring this
        explicitly -- rather than inferring it from what was computable -- is
        the fix for defect D3. Defaults to every key in the suite.
    """

    def __init__(self, name: str, metrics: Sequence[Metric | str],
                 invariant_keys: Optional[Iterable[str]] = None) -> None:
        self.name = name
        self.metrics: List[Metric] = [
            get_metric(m) if isinstance(m, str) else m for m in metrics]
        keys = [m.spec.key for m in self.metrics]
        dupes = {k for k in keys if keys.count(k) > 1}
        if dupes:
            raise ValueError(f"duplicate metric keys in suite: {sorted(dupes)}")
        self.output_keys: tuple[str, ...] = tuple(keys)

        inv = tuple(invariant_keys) if invariant_keys is not None else tuple(keys)
        unknown = set(inv) - set(keys)
        if unknown:
            raise ValueError(
                f"invariant_keys {sorted(unknown)} are not in suite {name!r}")
        self.invariant_keys: tuple[str, ...] = inv

    @property
    def suite_id(self) -> str:
        """Stable hash over the declared term set.

        Adding, removing or renaming a metric changes this id, which invalidates
        comparison against previously-scored runs. That is intentional: the
        alternative is silently comparing two different benchmarks.
        """
        payload = "|".join([
            self.name,
            ",".join(self.output_keys),
            "inv:" + ",".join(sorted(self.invariant_keys)),
        ])
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def quantity_keys(self) -> tuple[str, ...]:
        """Keys whose per-frame samples feed distributional comparison (W1/KFD)."""
        return tuple(m.spec.key for m in self.metrics if m.spec.perframe)

    def spec_of(self, key: str):
        for m in self.metrics:
            if m.spec.key == key:
                return m.spec
        raise KeyError(f"{key!r} not in suite {self.name!r}")

    def evaluate(self, ctx: MetricContext) -> SuiteResult:
        """Run every metric, filling unavailable ones with ``NaN`` + reason.

        The returned key set is exactly :attr:`output_keys`, for every clip,
        regardless of what inputs were available -- the static-schema guarantee.
        """
        values: Dict[str, MetricValue] = {}
        for metric in self.metrics:
            key = metric.spec.key
            try:
                values[key] = metric.compute(ctx)
            except Exception as exc:                      # noqa: BLE001
                values[key] = MetricValue.unavailable(
                    key, f"error:{type(exc).__name__}:{exc}")
        return SuiteResult(self.suite_id, self.name, values)

    def describe(self) -> List[dict]:
        """Machine-readable suite description, used by ``kinescore describe``."""
        return [{
            "key": m.spec.key, "units": m.spec.units,
            "dt_exponent": m.spec.dt_exponent, "direction": m.spec.direction,
            "requires": sorted(m.spec.requires), "min_frames": m.spec.min_frames,
            "unobservable_when": list(m.spec.unobservable_when),
            "in_pis": m.spec.key in self.invariant_keys,
            "description": m.spec.description,
        } for m in self.metrics]

    def __repr__(self) -> str:
        return (f"MetricSuite(name={self.name!r}, n_metrics={len(self.metrics)}, "
                f"suite_id={self.suite_id})")
