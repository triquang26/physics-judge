"""Per-(method, metric) separation: does this ruler tell real from fake.

The project page draws a hard line between two different questions a number
can answer, and refuses to collapse them into one:

    "a **magnitude** in physical units ... says *how large* a violation is;
    a **separation** score (0.5-1.0) says *how reliably* a ruler tells real
    from fake ... We keep each in its natural form rather than squashing
    everything into one 0-1 number."

``bench/stats.py`` already implements every statistic this module needs
(``paired_deltas``, ``wilcoxon_signed``, ``bootstrap_ci``, ``holm``,
``auroc``, ``cliffs_delta``, ``noise_units``, ``second_difference``) -- this
module is wiring, not new statistics: it decides *which* two groups go into
``auroc`` and in *which order*, reads a metric's declared ``units``/
``direction``/``dt_exponent`` from the registry instead of hardcoding them,
and shapes the result into one frozen, always-present row per
(method, metric) so a report can render every ruler -- including the ones
that score ~0.50 -- without ad hoc branching.

Two different real-vs-generated comparisons, on purpose
---------------------------------------------------------
:func:`compute_separation` computes two logically distinct things from the
same joined DataFrame, and does not confuse them:

* **paired** -- per-episode delta = phi(pred) - phi(gt), the same quantity
  ``bench.stats.aggregate`` reports. Answers "how much extra physics tax did
  this episode's generated clip carry, against its own ground truth" --
  ``delta_median``, its BCa CI, Wilcoxon p, and ``frac_worse`` (the page's
  "in 99% of those episodes the generated clip was the jerkier one") all come
  from this paired view, because they are inherently about *matched pairs*.
* **unpaired** -- the full distribution of real values against the full
  distribution of generated values, real and generated pooled across
  whatever episodes each side happens to have. Answers "if you handed me one
  clip with its identity hidden, could this ruler alone tell you which pile
  it came from" -- this is what ``separation`` (AUROC) and ``cliffs_delta``
  measure, and it is a *classification* question, not a *pairing* question,
  so it deliberately does not require gt/pred counts to match.

Direction matters for AUROC, and the page states the invariant plainly:
a ruler's separation must be oriented so that ``1.0`` always means "perfectly
separates, and generated is the worse one" -- never "perfectly separates,
in whichever direction the raw AUROC happened to come out." Getting this
backwards for a ``higher_better`` metric (``limit_headroom_rad``,
``com_margin_m``) silently reports a strong ruler as a useless one and vice
versa. See :func:`compute_separation`'s body for the orientation rule; see
``tests/test_separation.py`` for the regression test that catches a flipped
sign.

Never ``0.0`` for unavailable
------------------------------
A row that cannot be computed (too few episodes after pairing, the metric
column missing from the joined results entirely, or an unregistered metric
key) returns a :class:`SeparationRow` with every numeric field ``None`` and
``reason`` set -- never a fabricated ``0.0``/``0.5``. This mirrors the same
invariant ``legacy_docs/SCHEMA.md`` states for ``results.jsonl`` itself: unavailable
means ``null`` plus a reason, and the row is still returned (never dropped),
so a report can show *why* a cell is empty instead of the cell silently not
existing.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from kinescore.bench import stats
from kinescore.core.metric import get_metric

__all__ = [
    "DEFAULT_MIN_EPISODES", "SeparationRow", "CacheRankRow", "ExtraCostRow",
    "compute_separation", "separation_table", "rank_caches",
    "extra_cost_vs_baseline",
]

#: Below this many paired episodes a delta/CI/AUROC is too noisy to report as
#: a number -- the row still exists, carrying ``n`` and a ``reason`` instead.
DEFAULT_MIN_EPISODES = 5

#: Separation (AUROC) -> verdict wording, exactly the bands the project page
#: uses. Anything below the "clear signal" floor -- including a value that
#: happens to fall *below* 0.5, i.e. the ruler ranking real clips as worse
#: than generated ones -- reads as "can't tell them apart": the page's own
#: point is that this is not a failure to hide, it is itself a finding (a
#: ruler that is a specific diagnostic rather than a blanket "generated=bad"
#: alarm should score ~0.50 on axes it has no business detecting).
_STRONG_SIGNAL = 0.80
_CLEAR_SIGNAL = 0.60


def _verdict(separation: float | None) -> str | None:
    if separation is None or not math.isfinite(separation):
        return None
    if separation >= _STRONG_SIGNAL:
        return "a strong signal"
    if separation >= _CLEAR_SIGNAL:
        return "a clear signal"
    return "can't tell them apart"


def _finite_or_none(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


@dataclass(frozen=True)
class SeparationRow:
    """One (method, metric) row: paired physics tax + unpaired separation.

    ``method`` names whatever identity column ``bench.stats.load_scores``
    joined in (today: the manifest's ``method`` column -- a cache/generator
    combination once ``bench/matrix.py`` ships a multi-axis cell identity,
    the same field carries that instead without this module changing).

    All fields except ``method``/``metric``/``n``/``reason`` are ``None``
    when ``reason`` is set -- see the module docstring's "never 0.0" section.
    """

    method: str
    metric: str
    units: str
    dt_exponent: int | None
    direction: str
    #: ``dt_exponent is not None`` -- read from the registry, never a
    #: hardcoded metric-name list (a `rate_free` suite/policy is landing
    #: separately; this flag is what lets a report carry it without knowing
    #: which metrics currently qualify).
    rate_comparable: bool
    #: Paired episodes, after dropping any episode missing either role or
    #: with a non-finite value on either side -- the unit of analysis, not
    #: frames (see ``bench/stats.py``'s module docstring).
    n: int
    real_median: float | None
    gen_median: float | None
    delta_median: float | None
    ci_lo: float | None
    ci_hi: float | None
    p: float | None
    #: Holm-adjusted across the metrics tested for this ``method`` in the
    #: same :func:`separation_table` call; ``None`` if this row was built
    #: standalone via :func:`compute_separation`.
    p_holm: float | None
    cliffs_delta: float | None
    #: Fraction of paired episodes where the generated clip was worse than
    #: its own ground truth, oriented by ``direction`` -- the page's "in 99%
    #: of those episodes the generated clip was the jerkier one".
    frac_worse: float | None
    #: AUROC of real-vs-generated, oriented so 1.0 = perfectly separates and
    #: generated is worse; < 0.5 = the ruler ranks real as worse (a finding,
    #: never filtered out -- see module docstring).
    separation: float | None
    noise_floor: float | None
    above_noise: bool | None
    verdict: str | None
    reason: str | None = None


def _unavailable_row(method: str, metric_key: str, *, units: str = "",
                     dt_exponent: int | None = None,
                     direction: str = "lower_better", n: int = 0,
                     noise_floor: float | None = None,
                     reason: str) -> SeparationRow:
    return SeparationRow(
        method=method, metric=metric_key, units=units, dt_exponent=dt_exponent,
        direction=direction, rate_comparable=dt_exponent is not None, n=n,
        real_median=None, gen_median=None, delta_median=None, ci_lo=None,
        ci_hi=None, p=None, p_holm=None, cliffs_delta=None, frac_worse=None,
        separation=None, noise_floor=noise_floor, above_noise=None,
        verdict=None, reason=reason)


def _resolve_noise_floor(
    noise_floor: float | dict | None, metric_key: str,
) -> float | None:
    if noise_floor is None:
        return None
    if isinstance(noise_floor, dict):
        val = noise_floor.get(metric_key)
        return None if val is None else float(val)
    return float(noise_floor)


def compute_separation(
    df, method: str, metric_key: str, *,
    min_episodes: int = DEFAULT_MIN_EPISODES,
    noise_floor: float | dict | None = None,
    B: int = 10000, seed: int = 0,
) -> SeparationRow:
    """One (method, metric) row -- paired tax + unpaired separation.

    Parameters
    ----------
    df:
        Output of :func:`kinescore.bench.stats.load_scores`.
    method:
        Value of the joined ``method`` column to select.
    metric_key:
        Bare registry key (e.g. ``"mean_jerk_mps3"``, not
        ``"metrics.mean_jerk_mps3"``) -- looked up via
        :func:`kinescore.core.metric.get_metric` for ``units``/
        ``dt_exponent``/``direction``. An unregistered key produces a row
        with ``reason`` set (the same "never drop the row" invariant as
        every other unavailability case here), not a raised exception --
        this function's job is to describe the requested cell, including
        when the request itself was invalid.
    min_episodes:
        Below this many paired episodes, the row reports only ``n`` plus a
        ``reason`` (see :data:`DEFAULT_MIN_EPISODES`).
    noise_floor:
        Either a single float applied to every metric, a ``{metric_key:
        float}`` mapping, or ``None`` (no noise-floor comparison --
        ``above_noise`` stays ``None``, meaning "unknown", not "below").
    B, seed:
        Passed to :func:`kinescore.bench.stats.bootstrap_ci`.
    """
    try:
        spec = get_metric(metric_key).spec
    except KeyError as exc:
        return _unavailable_row(method, metric_key, reason=str(exc))

    nf = _resolve_noise_floor(noise_floor, metric_key)
    col = f"metrics.{metric_key}"
    if col not in df.columns:
        return _unavailable_row(
            method, metric_key, units=spec.units, dt_exponent=spec.dt_exponent,
            direction=spec.direction, noise_floor=nf,
            reason=f"metric_unavailable:{metric_key} not in joined results")

    _, delta = stats.paired_deltas(df, method, col)
    n = int(delta.size)
    if n < min_episodes:
        return _unavailable_row(
            method, metric_key, units=spec.units, dt_exponent=spec.dt_exponent,
            direction=spec.direction, n=n, noise_floor=nf,
            reason=f"too_few_episodes:{n}<{min_episodes}")

    sub = df[df["method"].astype(str) == str(method)]
    gt_vals = (sub.loc[sub["role"].astype(str) == "gt", col]
              .dropna().astype(float).to_numpy())
    pred_vals = (sub.loc[sub["role"].astype(str) == "pred", col]
                .dropna().astype(float).to_numpy())

    w = stats.wilcoxon_signed(delta)
    ci = stats.bootstrap_ci(delta, B=B, seed=seed)

    lower_better = spec.direction == "lower_better"
    # Orient so `pos` is the group that scoring HIGHER means "worse" --
    # for a lower_better metric that is literally the raw value (pred vs
    # gt); for a higher_better metric (limit_headroom_rad, com_margin_m)
    # "worse" means a LOWER value, so the pos/neg roles of gt/pred swap.
    # auroc(neg, pos) = P(pos scores higher) -- see bench/stats.py.
    neg, pos = (gt_vals, pred_vals) if lower_better else (pred_vals, gt_vals)
    separation = stats.auroc(neg, pos)
    cliffs = stats.cliffs_delta(pos, neg)

    worse_mask = (delta > 0) if lower_better else (delta < 0)
    frac_worse = float(np.mean(worse_mask)) if n else None

    above_noise = None
    if nf is not None and math.isfinite(nf) and nf != 0 and math.isfinite(w["median"]):
        above_noise = bool(stats.noise_units(w["median"], nf) > 1.0)

    sep_finite = _finite_or_none(separation)
    return SeparationRow(
        method=method, metric=metric_key, units=spec.units,
        dt_exponent=spec.dt_exponent, direction=spec.direction,
        rate_comparable=spec.dt_exponent is not None, n=n,
        real_median=(_finite_or_none(np.median(gt_vals)) if gt_vals.size else None),
        gen_median=(_finite_or_none(np.median(pred_vals)) if pred_vals.size else None),
        delta_median=_finite_or_none(w["median"]),
        ci_lo=_finite_or_none(ci["lo"]), ci_hi=_finite_or_none(ci["hi"]),
        p=_finite_or_none(w["p"]), p_holm=None,
        cliffs_delta=_finite_or_none(cliffs), frac_worse=frac_worse,
        separation=sep_finite, noise_floor=nf, above_noise=above_noise,
        verdict=_verdict(sep_finite), reason=None,
    )


def separation_table(
    df, methods: Sequence[str], metric_keys: Sequence[str], *,
    min_episodes: int = DEFAULT_MIN_EPISODES,
    noise_floor: float | dict | None = None,
    B: int = 10000, seed: int = 0,
) -> list[SeparationRow]:
    """:func:`compute_separation` for every (method, metric) pair.

    Holm-adjusts ``p`` across the metrics tested *for each method* (the
    multiple-comparisons family is "all the rulers checked on this one
    method/cell", mirroring how ``kinescore aggregate`` already loops
    methods x metrics) via :func:`kinescore.bench.stats.holm`, filling
    ``p_holm`` on the returned rows in the same order as
    ``methods x metric_keys``.
    """
    out: list[SeparationRow] = []
    for method in methods:
        rows = [
            compute_separation(df, method, key, min_episodes=min_episodes,
                               noise_floor=noise_floor, B=B, seed=seed)
            for key in metric_keys
        ]
        adj = stats.holm([r.p if r.p is not None else float("nan") for r in rows])
        for row, a in zip(rows, adj, strict=True):
            p_holm = _finite_or_none(a) if row.reason is None else None
            out.append(
                SeparationRow(
                    method=row.method, metric=row.metric, units=row.units,
                    dt_exponent=row.dt_exponent, direction=row.direction,
                    rate_comparable=row.rate_comparable, n=row.n,
                    real_median=row.real_median, gen_median=row.gen_median,
                    delta_median=row.delta_median, ci_lo=row.ci_lo,
                    ci_hi=row.ci_hi, p=row.p, p_holm=p_holm,
                    cliffs_delta=row.cliffs_delta, frac_worse=row.frac_worse,
                    separation=row.separation, noise_floor=row.noise_floor,
                    above_noise=row.above_noise, verdict=row.verdict,
                    reason=row.reason))
    return out


@dataclass(frozen=True)
class ExtraCostRow:
    """One cache method's extra physics tax over a baseline (e.g. ``dense``).

    Thin metadata wrapper around
    :func:`kinescore.bench.stats.second_difference` -- reproduces the page's
    per-cache lines ("dicache +5.4 [4.8,6.0]") by attaching
    ``units``/``direction``/``dt_exponent`` from the metric registry to the
    bare dict that function already returns, rather than recomputing
    anything.
    """

    method: str
    baseline: str
    metric: str
    units: str
    dt_exponent: int | None
    direction: str
    rate_comparable: bool
    n: int
    median: float | None
    ci_lo: float | None
    ci_hi: float | None
    p: float | None
    paired: bool
    fallback: str | None
    reason: str | None = None


def extra_cost_vs_baseline(
    df, method: str, baseline: str, metric_key: str, *,
    min_episodes: int = 1, B: int = 10000, seed: int = 0,
) -> ExtraCostRow:
    """``delta(method) - delta(baseline)`` on the same generator's output.

    ``min_episodes`` defaults to ``1`` (not
    :data:`DEFAULT_MIN_EPISODES`) because
    :func:`~kinescore.bench.stats.second_difference` already has its own
    unpaired Mann-Whitney fallback for the "no shared episode ids" case
    (``fallback="mannwhitney"`` on the returned row) -- refusing a small-n
    paired intersection here would just hide that fallback ever ran.
    """
    try:
        spec = get_metric(metric_key).spec
    except KeyError as exc:
        return ExtraCostRow(
            method=method, baseline=baseline, metric=metric_key, units="",
            dt_exponent=None, direction="lower_better", rate_comparable=False,
            n=0, median=None, ci_lo=None, ci_hi=None, p=None, paired=False,
            fallback=None, reason=str(exc))

    col = f"metrics.{metric_key}"
    if col not in df.columns:
        return ExtraCostRow(
            method=method, baseline=baseline, metric=metric_key,
            units=spec.units, dt_exponent=spec.dt_exponent,
            direction=spec.direction, rate_comparable=spec.dt_exponent is not None,
            n=0, median=None, ci_lo=None, ci_hi=None, p=None, paired=False,
            fallback=None,
            reason=f"metric_unavailable:{metric_key} not in joined results")

    sd = stats.second_difference(df, method, baseline, col, B=B, seed=seed)
    n = sd["n"]
    if n < min_episodes:
        return ExtraCostRow(
            method=method, baseline=baseline, metric=metric_key,
            units=spec.units, dt_exponent=spec.dt_exponent,
            direction=spec.direction, rate_comparable=spec.dt_exponent is not None,
            n=n, median=None, ci_lo=None, ci_hi=None, p=None, paired=False,
            fallback=None, reason=f"too_few_episodes:{n}<{min_episodes}")

    return ExtraCostRow(
        method=method, baseline=baseline, metric=metric_key, units=spec.units,
        dt_exponent=spec.dt_exponent, direction=spec.direction,
        rate_comparable=spec.dt_exponent is not None, n=int(n),
        median=_finite_or_none(sd["median"]),
        ci_lo=_finite_or_none(sd["ci"]["lo"]), ci_hi=_finite_or_none(sd["ci"]["hi"]),
        p=_finite_or_none(sd["p"]), paired=bool(sd["paired"]),
        fallback=sd["fallback"], reason=None,
    )


@dataclass(frozen=True)
class CacheRankRow:
    """One cache method's rank on each physical axis, plus its mean rank.

    Reproduces the page's Exp8 table: "extra jerk from caching" generalised
    to any set of physical axes (jerk / balance margin / joint limits, or
    whatever :class:`SeparationRow`\\ s were passed in). Rank ``1`` = best on
    that axis.
    """

    method: str
    #: ``{metric_key: rank}``, 1 = best on that axis. A metric absent here
    #: means fewer than two methods had a computable ``delta_median`` for it
    #: (nothing to rank against), not that this method scored worst.
    axis_ranks: dict
    #: Mean of ``axis_ranks.values()``; ``None`` if no axis was rankable.
    mean_rank: float | None
    n_axes: int
    #: ``{metric_key: oriented_tax(method) - oriented_tax(baseline)}`` point
    #: estimate (no CI -- use :func:`extra_cost_vs_baseline` for the CI'd
    #: version); positive means "more physics tax than the baseline on this
    #: axis". ``None`` per-axis if either side's ``delta_median`` was
    #: unavailable, or for the baseline's own row (always 0 by
    #: construction, reported as ``None`` rather than a misleading exact
    #: zero from floating-point subtraction).
    axis_extra_cost: dict


def _oriented_tax(delta_median: float, direction: str) -> float:
    """A cache's physics tax, oriented so *higher = worse* for every metric.

    ``delta_median`` (pred - gt) already means "more tax = worse" for a
    ``lower_better`` metric. For a ``higher_better`` metric (``com_margin_m``,
    ``limit_headroom_rad``) the sign flips: a *negative* delta (generated
    clip's margin is lower than ground truth's) is the tax, so this negates
    the value to keep "higher = worse" uniform across both metric families,
    which is what lets a single ascending sort double as "best first".
    """
    return delta_median if direction == "lower_better" else -delta_median


def rank_caches(
    rows: Sequence[SeparationRow], *, baseline: str = "dense",
) -> list[CacheRankRow]:
    """Rank methods (cache variants) by mean rank across physical axes.

    For each metric present across ``rows``, methods with a computable
    ``delta_median`` on that metric are ranked 1..k (1 = least physics tax,
    oriented per :func:`_oriented_tax`); ties break on method name for
    determinism. A method's ``mean_rank`` is the mean of its per-axis ranks
    over however many axes it was rankable on (``n_axes``) -- a method
    missing an axis entirely (e.g. a metric ``null`` for its robot) is
    averaged over fewer axes rather than penalised with a fabricated worst
    rank.

    Subtracting a baseline's oriented tax from every method's on the same
    axis shifts every candidate's value by the *same* constant, so it never
    changes the rank *order* -- ``baseline`` therefore only affects
    ``axis_extra_cost`` (an informational point estimate), never
    ``axis_ranks``/``mean_rank``. Returned sorted by ``mean_rank`` ascending
    (best first); rows with no rankable axis sort last.
    """
    by_metric: dict = {}
    for r in rows:
        if r.delta_median is not None and math.isfinite(r.delta_median):
            by_metric.setdefault(r.metric, {})[r.method] = r

    methods = sorted({r.method for r in rows})
    axis_ranks: dict = {m: {} for m in methods}
    axis_extra_cost: dict = {m: {} for m in methods}

    for metric, per_method in by_metric.items():
        if len(per_method) < 2:
            continue
        direction = next(iter(per_method.values())).direction
        items = sorted(
            per_method.items(),
            key=lambda kv: (_oriented_tax(kv[1].delta_median, direction), kv[0]))
        for i, (method, _row) in enumerate(items, start=1):
            axis_ranks[method][metric] = float(i)

        base_row = per_method.get(baseline)
        if base_row is not None:
            base_tax = _oriented_tax(base_row.delta_median, direction)
            for method, row in per_method.items():
                if method == baseline:
                    continue
                axis_extra_cost[method][metric] = (
                    _oriented_tax(row.delta_median, direction) - base_tax)

    out = []
    for method in methods:
        ranks = axis_ranks[method]
        vals = list(ranks.values())
        mean_rank = float(np.mean(vals)) if vals else None
        out.append(CacheRankRow(
            method=method, axis_ranks=dict(ranks), mean_rank=mean_rank,
            n_axes=len(vals), axis_extra_cost=dict(axis_extra_cost[method])))

    out.sort(key=lambda r: (r.mean_rank is None,
                            r.mean_rank if r.mean_rank is not None else 0.0,
                            r.method))
    return out
