"""Paired episode-level statistics -- ported near-verbatim from the source.

Pure, side-effect-free functions (no CLI, no file writes). They implement the
*honest* statistical spine of the benchmark, unchanged from
``Marionette-fkjepa/eval/bench/stats.py``:

* unit of analysis = **episode / clip** (frames are autocorrelated; never pool
  frames as independent samples);
* paired design: for one method, per episode compute delta = phi(pred) -
  phi(gt); the method's physics tax is a *distribution of per-episode delta*;
* a headline second difference delta - delta_baseline isolates one method's
  extra tax from a shared baseline's tax on the same generator's output.

``paired_deltas``, ``wilcoxon_signed``, ``bootstrap_ci``, ``second_difference``,
``holm``, ``auroc``, ``cliffs_delta`` and ``noise_units`` are unchanged logic
(numpy/pandas/scipy), carried over as-is.

**Only :func:`load_scores` is rewritten.** The source read one JSON file per
clip from ``outputs_bench/scores/<method>/<episode>__<role>.json``, and each
file already carried ``method``/``episode``/``role``/``pair_key`` inline
(``scoring.py``'s ``clip_dict``). kinescore's canonical record
(``ScoredClip.to_record()`` plus ``kinescore.bench.store``'s bookkeeping)
carries none of that: a :class:`~kinescore.core.clip.ClipSpec` only knows a
clip's *path*, deliberately -- which benchmark episode or role produced a clip
is manifest identity, not something scoring should need to know or be able to
get wrong. So :func:`load_scores` here joins ``results.jsonl`` (scores, keyed
by path) against the manifest (identity, keyed by path) instead of reading
identity out of the score file itself. Everything below operates on the
resulting flattened metric columns, e.g. ``metrics.mean_jerk_mps3`` in place
of the source's bare ``jerk``.

Two functions are new, not ported: :func:`aggregate` (a convenience wrapper
tying ``paired_deltas`` + ``wilcoxon_signed`` + ``bootstrap_ci`` together for
one method/metric, absent from the source) and its suite-mixing guard --
``aggregate`` refuses to pool rows whose
``run.suite_id`` (:class:`~kinescore.core.suite.MetricSuite.suite_id`)
differs, because two different suite ids are two different benchmarks: see
that class's docstring for why (defect D3, generalised from "varying term
set within one clip" to "varying term set across a benchmark run").
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "load_scores", "paired_deltas", "wilcoxon_signed", "bootstrap_ci",
    "second_difference", "aggregate", "holm", "auroc", "cliffs_delta",
    "noise_units",
]

_IDENTITY_COLUMNS = ("method", "family", "episode", "role", "pair_key")


# ── loading ────────────────────────────────────────────────────────────────
def load_scores(results_path: str, manifest) -> pd.DataFrame:  # noqa: F821
    """Join ``results.jsonl`` with the manifest into one tidy DataFrame.

    Parameters
    ----------
    results_path:
        Path to a ``results.jsonl`` written by
        :class:`kinescore.bench.store.ResultsStore`.
    manifest:
        Either a path to a manifest file (``.parquet`` or ``.json``, read via
        :func:`kinescore.bench.manifest.load_manifest`) or an already-loaded
        list of manifest row dicts.

    Returns
    -------
    pandas.DataFrame
        One row per clip, inner-joined on ``path``: the manifest's
        ``method``/``family``/``episode``/``role``/``pair_key`` columns plus
        every flattened score column (``clip.*``, ``run.*``, ``coverage.*``,
        ``metrics.*``, ``metrics_unavailable.*``, ``status``). Clips present
        in only one of the two inputs are silently dropped (inner join) --
        that is the manifest/results-store boundary working as intended: a
        clip that was never scored has no metrics to analyse, and a scored
        path missing from the manifest has no episode/role identity to pair
        it by.

        Empty (no columns) if ``results_path`` has no rows.
    """
    import pandas as pd

    from kinescore.bench.manifest import load_manifest
    from kinescore.bench.store import flatten, iter_records

    recs = [flatten(r) for r in iter_records(results_path)]
    if not recs:
        return pd.DataFrame()
    res_df = pd.DataFrame(recs).rename(columns={"clip.path": "path"})

    rows = load_manifest(manifest) if isinstance(manifest, str) else list(manifest)
    man_df = pd.DataFrame(rows)[["path", *_IDENTITY_COLUMNS]]

    df = pd.merge(man_df, res_df, on="path", how="inner")
    for k in _IDENTITY_COLUMNS:
        if k in df.columns:
            df[k] = df[k].astype("string")
    return df


# ── paired deltas ──────────────────────────────────────────────────────────
def paired_deltas(df, method: str, metric: str):
    """Per-episode delta = phi(pred) - phi(gt) for one method/metric, gt<->pred paired.

    Proves the *within-episode* effect: by inner-joining gt and pred on the
    episode id we cancel the episode-level (task/scene) variance, so delta
    measures only what the world-model+cache did to that clip. Returns
    ``(episodes: np.ndarray[str], delta: np.ndarray[float])`` sorted by
    episode, dropping episodes lacking either role or with a NaN metric on
    either side.
    """
    import pandas as pd

    sub = df[df["method"].astype(str) == str(method)]
    gt = sub[sub["role"].astype(str) == "gt"][["episode", metric]]
    pred = sub[sub["role"].astype(str) == "pred"][["episode", metric]]
    if gt.empty or pred.empty:
        return np.array([], dtype=object), np.array([], dtype=float)
    gt = gt.rename(columns={metric: "gt"})
    pred = pred.rename(columns={metric: "pred"})
    m = pd.merge(gt, pred, on="episode", how="inner").dropna(subset=["gt", "pred"])
    m = m.sort_values("episode")
    ep = m["episode"].astype(str).to_numpy()
    delta = (m["pred"].astype(float) - m["gt"].astype(float)).to_numpy()
    return ep, delta


# ── Wilcoxon signed-rank ───────────────────────────────────────────────────
def wilcoxon_signed(delta: Sequence[float]) -> dict:
    """Paired Wilcoxon signed-rank test that the per-episode delta median is != 0.

    Proves that the physics tax is *systematic across episodes*, not a couple
    of outliers -- a distribution-free companion to the median CI. Uses
    scipy's ``wilcoxon`` with ``zero_method="pratt"`` (keeps zero-delta pairs
    in the ranking, the conservative choice). Also returns the
    **rank-biserial** effect size ``r = (W+ - W-)/(W+ + W-)`` in [-1, 1] (sign
    follows the median delta).

    Returns ``{stat, p, n, median, r}``; all NaN if n < 1 or all-zero delta.
    """
    d = np.asarray(delta, dtype=float)
    d = d[np.isfinite(d)]
    n = int(d.size)
    out = {"stat": float("nan"), "p": float("nan"), "n": n,
           "median": float(np.median(d)) if n else float("nan"),
           "r": float("nan")}
    if n < 1 or np.allclose(d, 0.0):
        return out
    from scipy.stats import rankdata, wilcoxon
    absd = np.abs(d)
    ranks = rankdata(absd)
    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    denom = w_pos + w_neg
    r = float((w_pos - w_neg) / denom) if denom > 0 else float("nan")
    try:
        res = wilcoxon(d, zero_method="pratt", alternative="two-sided")
        out["stat"], out["p"] = float(res.statistic), float(res.pvalue)
    except ValueError:
        # e.g. all differences zero after pratt -- leave NaN
        pass
    out["r"] = r
    return out


# ── bootstrap CI on the median ─────────────────────────────────────────────
def bootstrap_ci(delta: Sequence[float], B: int = 10000, seed: int = 0,
                 alpha: float = 0.05, method: str = "bca") -> dict:
    """Bootstrap confidence interval for the **median** per-episode delta.

    Proves the effect's precision without a normality assumption.
    Deterministic: a fixed ``np.random.default_rng(seed)`` -- never any
    global RNG state. ``method="bca"`` gives the bias-corrected-accelerated
    interval (falls back to percentile on degeneracy); ``method="percentile"``
    forces the simpler one.

    Returns ``{point, lo, hi, B, seed, method}`` where ``point`` is the
    observed median and ``[lo, hi]`` the ``1-alpha`` CI. Empty input -> all NaN.
    """
    d = np.asarray(delta, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    out = {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
           "B": int(B), "seed": int(seed), "method": method}
    if n == 0:
        return out
    point = float(np.median(d))
    out["point"] = point
    if n == 1:
        out["lo"] = out["hi"] = point
        return out
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boot = np.median(d[idx], axis=1)
    lo_p, hi_p = 100 * alpha / 2, 100 * (1 - alpha / 2)
    if method == "bca":
        from scipy.stats import norm
        prop = np.mean(boot < point)
        prop = min(max(prop, 1.0 / (B + 1)), 1.0 - 1.0 / (B + 1))
        z0 = norm.ppf(prop)
        jack = np.empty(n)
        full = d.copy()
        for i in range(n):
            jack[i] = np.median(np.delete(full, i))
        jbar = jack.mean()
        num = np.sum((jbar - jack) ** 3)
        den = 6.0 * (np.sum((jbar - jack) ** 2) ** 1.5)
        a = num / den if den != 0 else 0.0
        zl, zu = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)

        def adj(z):
            return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        p_lo, p_hi = 100 * adj(zl), 100 * adj(zu)
        if not (np.isfinite(p_lo) and np.isfinite(p_hi)):
            p_lo, p_hi = lo_p, hi_p
        out["lo"] = float(np.percentile(boot, p_lo))
        out["hi"] = float(np.percentile(boot, p_hi))
    else:
        out["method"] = "percentile"
        out["lo"] = float(np.percentile(boot, lo_p))
        out["hi"] = float(np.percentile(boot, hi_p))
    return out


# ── second difference: delta_method - delta_baseline ──────────────────────
def second_difference(df, method: str, baseline: str, metric: str,
                      B: int = 10000, seed: int = 0) -> dict:
    """delta - delta_baseline: one method's tax net of a shared baseline's tax.

    delta_method(ep) and delta_baseline(ep) both measure pred-gt on the SAME
    generator's output; their episode-wise difference on the id INTERSECTION
    isolates what ``method`` added on top of ``baseline``.

    If the two methods share no episode ids, we cannot pair, so we fall back
    to an UNPAIRED Mann-Whitney U on the two delta distributions and set
    ``paired=False`` / ``fallback="mannwhitney"`` so the caller can flag it.

    Returns a dict with ``paired`` bool, ``n``, the median second difference,
    its bootstrap CI, the test p-value, and (paired branch) a Wilcoxon block.
    """
    ep_m, d_m = paired_deltas(df, method, metric)
    ep_b, d_b = paired_deltas(df, baseline, metric)
    map_m = dict(zip(ep_m.tolist(), d_m.tolist(), strict=True))
    map_b = dict(zip(ep_b.tolist(), d_b.tolist(), strict=True))
    common = sorted(set(map_m) & set(map_b))
    if common:
        dd = np.array([map_m[e] - map_b[e] for e in common], dtype=float)
        w = wilcoxon_signed(dd)
        ci = bootstrap_ci(dd, B=B, seed=seed)
        return {"paired": True, "fallback": None, "n": int(dd.size),
                "median": float(np.median(dd)), "ci": ci,
                "p": w["p"], "wilcoxon": w,
                "delta_method_median": float(np.median(d_m)) if d_m.size else float("nan"),
                "delta_baseline_median": float(np.median(d_b)) if d_b.size else float("nan")}
    from scipy.stats import mannwhitneyu
    a, b = d_m[np.isfinite(d_m)], d_b[np.isfinite(d_b)]
    p = float("nan")
    if a.size and b.size:
        try:
            p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        except ValueError:
            pass
    med = (float(np.median(a)) if a.size else float("nan")) - \
          (float(np.median(b)) if b.size else float("nan"))
    return {"paired": False, "fallback": "mannwhitney",
            "n": int(min(a.size, b.size)), "median": med,
            "ci": {"point": med, "lo": float("nan"), "hi": float("nan"),
                   "B": B, "seed": seed, "method": "none"},
            "p": p, "wilcoxon": None,
            "delta_method_median": float(np.median(a)) if a.size else float("nan"),
            "delta_baseline_median": float(np.median(b)) if b.size else float("nan")}


# ── aggregate: one method/metric summary, suite-id guarded ────────────────
def aggregate(df, method: str, metric: str, *, baseline: str | None = None,
             allow_mixed_suites: bool = False, B: int = 10000,
             seed: int = 0) -> dict:
    """One method's physics tax on one metric: paired delta + Wilcoxon + CI.

    New in kinescore (absent from the source): refuses to pool rows whose
    ``run.suite_id`` differs for ``method``, unless
    ``allow_mixed_suites=True``. ``suite_id`` is
    :attr:`~kinescore.core.suite.MetricSuite.suite_id`, a hash of the suite's
    declared metric term set -- two different ids mean two different
    benchmarks, possibly computed from different inputs or a changed metric
    definition. Silently averaging across them is defect D3 recurring one
    layer up: "varying term set within one clip" generalised to "varying term
    set across a run". When overridden, the returned dict carries a
    non-``None`` ``"warning"`` so the caller cannot miss that the numbers are
    mixed-provenance.

    Parameters
    ----------
    df:
        Output of :func:`load_scores`.
    method, metric:
        As in :func:`paired_deltas` (``metric`` is a flattened column, e.g.
        ``"metrics.mean_jerk_mps3"``).
    baseline:
        If given, also computes :func:`second_difference` against it and
        stores it under ``"second_difference_vs_baseline"``.

    Returns
    -------
    dict
        ``{"method", "metric", "n", "median", "ci", "wilcoxon", "warning",
        ["second_difference_vs_baseline"]}``.
    """
    warning = None
    if "run.suite_id" in df.columns:
        ids = (df.loc[df["method"].astype(str) == str(method), "run.suite_id"]
              .dropna().unique())
        if len(ids) > 1:
            msg = (f"aggregate(method={method!r}, metric={metric!r}): rows span "
                  f"suite_ids {sorted(map(str, ids))}; these are different "
                  f"benchmarks (see MetricSuite.suite_id)")
            if not allow_mixed_suites:
                raise ValueError(msg + ". Pass allow_mixed_suites=True to override.")
            warning = "MIXED SUITES POOLED: " + msg

    _, delta = paired_deltas(df, method, metric)
    w = wilcoxon_signed(delta)
    ci = bootstrap_ci(delta, B=B, seed=seed)
    out = {"method": method, "metric": metric, "n": int(delta.size),
           "median": w["median"], "ci": ci, "wilcoxon": w, "warning": warning}
    if baseline is not None:
        out["second_difference_vs_baseline"] = second_difference(
            df, method, baseline, metric, B=B, seed=seed)
    return out


# ── Holm-Bonferroni ────────────────────────────────────────────────────────
def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (family-wise error control).

    Proves the paired findings survive multiplicity across several tested
    methods: testing them as one family means a single lucky p<0.05 is not
    enough. Preserves input order; NaNs pass through as NaN and do not
    consume a rank. Returns adjusted p-values clipped to [0, 1].
    """
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.where(np.isfinite(p))[0]
    if finite.size == 0:
        return out
    order = finite[np.argsort(p[finite])]
    m = finite.size
    running = 0.0
    for rank, i in enumerate(order):
        adj = (m - rank) * p[i]
        running = max(running, adj)  # enforce monotonicity
        out[i] = min(running, 1.0)
    return out


# ── AUROC (Mann-Whitney, tie-safe) ─────────────────────────────────────────
def auroc(neg, pos) -> float:
    """Mann-Whitney AUROC that ``pos`` scores higher than ``neg`` (tie-safe).

    Average-rank tie handling means an all-ties axis returns 0.5 -- an honest
    "cannot separate", not a spurious 1.0.
    """
    neg, pos = np.asarray(neg, dtype=float), np.asarray(pos, dtype=float)
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    try:
        from scipy.stats import rankdata
        ranks = rankdata(np.concatenate([neg, pos]))
    except Exception:
        allv = np.concatenate([neg, pos])
        order = allv.argsort()
        ranks = np.empty(len(allv))
        ranks[order] = np.arange(1, len(allv) + 1)
    return float((ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg) + 1e-9))


# ── Cliff's delta ──────────────────────────────────────────────────────────
def cliffs_delta(a, b) -> float:
    """Cliff's delta = P(a>b) - P(a<b) in [-1, 1] -- non-parametric effect size.

    Derived from the same rank sum as :func:`auroc` (delta = 2*AUROC - 1) with
    average-rank ties.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return 2.0 * auroc(b, a) - 1.0  # a as 'pos' vs b as 'neg'


# ── noise units ────────────────────────────────────────────────────────────
def noise_units(median_delta: float, meas_err_p95: float) -> float:
    """Effect expressed in *measurement-ruler* units: |median delta| / meas_err_p95.

    Proves the ruler is finer than the tax: if |median delta| is several
    times the p95 of the readout error, the effect cannot be a readout
    artefact. > 1 means the tax exceeds the 95th-percentile measurement
    noise. Returns NaN if the denominator is 0/NaN.
    """
    if meas_err_p95 is None or not np.isfinite(meas_err_p95) or meas_err_p95 == 0:
        return float("nan")
    return float(abs(median_delta) / meas_err_p95)
