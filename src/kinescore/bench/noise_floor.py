"""The paired re-encode noise floor -- the right ruler for a paired claim.

Ported from ``Marionette-fkjepa/scripts/bench/33_noise_floor.py`` (module
docstring, source lines 1-16). See ``legacy_docs/PROVENANCE.md`` for the full
source->destination record.

Why this is not an absolute readout-error bound
---------------------------------------------------
A naive measurement-error anchor (the source's stage-15c
``measurement_error.json``, ~21.6 for jerk on that repo's data) is an
ABSOLUTE video->joint readout error: how far a reader's estimate of a
quantity is from the true value on ONE clip. But this benchmark's headline
claims are PAIRED -- ``delta_e = phi(pred_e) - phi(gt_e)`` per episode -- and
the reader's own per-clip jitter is *common-mode* between ``pred`` and
``gt``: whatever the reader gets systematically wrong about clip ``e``, it
gets wrong the same way on both the prediction and the ground truth for that
same episode, so it cancels in the subtraction. Comparing a paired
``delta`` against an *absolute* per-clip error bound is therefore comparing
two different questions.

The honest ruler for a paired claim is the null-delta you get from
re-measuring the SAME content through a DIFFERENT lossy encode::

    null_delta_e = phi(gt_reencoded_e) - phi(gt_e)

where ``gt_reencoded`` is the episode's ground-truth clip re-encoded with a
DIFFERENT, deterministic CRF (seeded per episode: ``crf_base + episode_num %
crf_mod``, :func:`episode_crf`). ``p95(|null_delta|)`` over episodes
(:func:`summarize_null_deltas`) is the true paired measurement floor: if an
observed physics-tax delta sits far above this floor, it is a real signal,
not a re-encode artefact. This is the same distinction ``legacy_docs/PROVENANCE.md``
records for D1/D2 (a hidden, wrong-by-construction unit is invisible in the
numbers alone) -- here the unit is "how big a paired delta needs to be
before it means anything", and the floor makes that unit explicit and
checkable rather than assumed.

Scope of this module
------------------------
This module owns the re-encode-and-diff *mechanism*
(:func:`reencode_crf`, :func:`episode_crf`), the pure statistics
(:func:`summarize_null_deltas`), and the orchestration
(:func:`build_noise_floor`) that ties them together against a
caller-supplied ``score_fn`` -- deliberately not a hardcoded
:class:`kinescore.core.scorer.Scorer` call, so this module stays testable
and importable without torch/a trained reader (mirrors the source's own
``score_clip(scorer, row, sigma_tau=tau)`` indirection, generalised to a
plain callable). It does **not** ship a CLI subcommand of its own -- the
caller (whichever command builds a real ``Scorer`` and iterates a manifest)
wires this in; see :func:`below_floor` for how a report should read the
result.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_METRICS",
    "episode_crf",
    "reencode_crf",
    "summarize_null_deltas",
    "build_noise_floor",
    "below_floor",
]

#: Temporal metrics a paired null-delta can be formed for -- verbatim
#: ``_METRICS`` (33_noise_floor.py:46). Any caller-supplied ``metrics``
#: sequence to :func:`build_noise_floor` overrides this; these are just a
#: sensible default matching the source's own scope (the metrics whose
#: names are common to this package's ``mean_speed_mps`` /
#: ``mean_accel_mps2`` /``mean_jerk_mps3`` family after flattening -- see
#: :func:`build_noise_floor`'s docstring for the exact keys a caller's
#: ``score_fn`` should return).
DEFAULT_METRICS: tuple[str, ...] = ("mean_jerk_mps3", "mean_accel_mps2", "mean_speed_mps")

#: Deterministic per-episode CRF schedule -- verbatim source defaults
#: (33_noise_floor.py: ``--crf_base 23 --crf_mod 12``).
_DEFAULT_CRF_BASE = 23
_DEFAULT_CRF_MOD = 12

_EP_NUM_RE = re.compile(r"(\d+)")


def episode_crf(episode: str, *, base: int = _DEFAULT_CRF_BASE,
                 mod: int = _DEFAULT_CRF_MOD) -> int:
    """Deterministic, seeded-by-episode CRF for the null re-encode.

    Verbatim port of the source's ``_ep_num()`` (33_noise_floor.py:52-54)
    plus its inline ``crf = args.crf_base + (_ep_num(ep) % args.crf_mod)``
    (:120) -- a different CRF per episode (not one fixed CRF for every clip)
    so the null floor is not itself an artefact of one particular
    compression level. Episodes with no digit in their id get ``crf=base``
    (matching the source's ``_ep_num`` returning ``0`` when its regex finds
    nothing).
    """
    m = _EP_NUM_RE.search(str(episode))
    n = int(m.group(1)) if m else 0
    return base + (n % mod)


def reencode_crf(src: str, dst: str, crf: int, pix_fmt: str = "yuv420p") -> None:
    """Re-encode ``src`` -> ``dst``, same resolution/fps, only the CRF changes.

    Verbatim port of ``reencode_crf()`` (33_noise_floor.py:57-61) -- exactly
    the ffmpeg invocation the source uses, so a caller's re-encoded clip is
    the same lossy re-measurement the recorded ``noise_floor.json`` numbers
    were built from, not a similar-but-different one.

    Raises
    ------
    subprocess.CalledProcessError
        If ``ffmpeg`` fails (missing binary, unreadable/corrupt ``src``).
    """
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
         "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", pix_fmt, "-an", dst],
        check=True,
    )


def summarize_null_deltas(deltas: Sequence[float]) -> dict[str, float]:
    """``{n, null_mean, null_median, null_std, null_p95, null_abs_median}``.

    Verbatim port of the per-metric summary block in the source's ``main()``
    (33_noise_floor.py:162-173). ``null_p95`` (``p95(|null_delta|)``) is the
    floor value the module docstring describes; the other fields are carried
    over for parity with the source's ``noise_floor.json`` shape. Non-finite
    deltas are dropped before any statistic is computed (mirrors the
    source's ``a[np.isfinite(a)]``); an empty/all-non-finite input returns
    every field as ``NaN`` with ``n=0`` rather than raising.
    """
    a = np.asarray([d for d in deltas if d is not None], dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "null_mean": float("nan"), "null_median": float("nan"),
                "null_std": float("nan"), "null_p95": float("nan"),
                "null_abs_median": float("nan")}
    return {
        "n": int(a.size),
        "null_mean": float(a.mean()),
        "null_median": float(np.median(a)),
        "null_std": float(a.std()),
        "null_p95": float(np.percentile(np.abs(a), 95)),
        "null_abs_median": float(np.median(np.abs(a))),
    }


def build_noise_floor(
    gt_rows: Sequence[Mapping[str, Any]],
    score_fn: Callable[[str, float], Mapping[str, float]],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    crf_base: int = _DEFAULT_CRF_BASE,
    crf_mod: int = _DEFAULT_CRF_MOD,
    scratch_dir: str,
    reenc_cache: MutableMapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Build the paired-re-encode noise floor for one set of ground-truth clips.

    Orchestrates the source's ``main()`` (33_noise_floor.py:77-192): for
    each row, re-encode its ground-truth video at a deterministic
    per-episode CRF (:func:`episode_crf`), score both the original and the
    re-encoded clip via ``score_fn``, and accumulate the per-metric null
    delta ``reencoded - original``.

    Parameters
    ----------
    gt_rows:
        One mapping per ground-truth clip, each needing at least
        ``"episode"`` (any stringifiable id), ``"path"`` (video file), and
        ``"dt"`` (seconds/frame). An optional ``"scores"`` key -- a
        ``{metric: value}`` mapping already computed for the ORIGINAL clip
        -- lets a caller reuse an existing score instead of paying to
        recompute it (mirrors the source reusing stage-30's cached
        ``<ep>__gt.json``, 33_noise_floor.py:113-118); when absent,
        ``score_fn(path, dt)`` is called on the original path too.
    score_fn:
        ``(path, dt) -> {metric_key: value}``. Deliberately a plain
        callable rather than a :class:`kinescore.core.scorer.Scorer`
        reference, so this module has no hard torch/reader dependency and a
        test can pass a fake. A real caller typically wraps
        ``Scorer.score(load_rgb(clip), clip).result.scalars()`` (see
        ``cli/_scoring.py``/``cli/cmd_score.py`` for that composition) with
        the metric keys this package actually registers, e.g.
        ``"mean_jerk_mps3"``.
    metrics:
        Which score keys to form a null delta for. Default
        :data:`DEFAULT_METRICS` (jerk/accel/speed, matching the source's
        scope); any metric key ``score_fn`` returns can be requested.
    crf_base, crf_mod:
        Passed to :func:`episode_crf`.
    scratch_dir:
        Directory temporary re-encoded ``.mp4`` files are written to and
        deleted from -- created if missing. Mirrors the source keeping
        re-encoded video OUT of any permanent scores directory
        (33_noise_floor.py:20-21): only the *scores*, not the re-encoded
        bytes, are worth keeping.
    reenc_cache:
        Optional mutable mapping used to skip re-encoding+re-scoring a
        ``(path, crf)`` pair already computed -- keyed internally as
        ``f"{path}::crf{crf}"`` -> the re-encoded clip's score mapping.
        Mirrors the source's on-disk ``_nf_cache/`` (33_noise_floor.py:93,
        121-124), generalised to an injectable mapping (a caller wanting
        the on-disk behaviour can back this with a small
        JSON-file-backed ``MutableMapping``); ``None`` (default) disables
        caching -- every row is freshly re-encoded and scored.

    Returns
    -------
    dict
        ``{"crf_rule", "n_clips", "summary": {metric: summarize_null_deltas(...)},
        "pairs": [...]}`` -- the same shape as the source's
        ``noise_floor.json`` (33_noise_floor.py:176-184), minus the
        source's ``"method"``/``"dt"`` top-level fields (those are
        properties of the caller's manifest selection, not of this
        function).
    """
    os.makedirs(scratch_dir, exist_ok=True)
    null: dict[str, list[float]] = {m: [] for m in metrics}
    pairs: list[dict[str, Any]] = []

    for row in gt_rows:
        episode = str(row["episode"])
        path = str(row["path"])
        dt = float(row["dt"])

        orig = row.get("scores")
        if orig is None:
            orig = score_fn(path, dt)

        crf = episode_crf(episode, base=crf_base, mod=crf_mod)
        cache_key = f"{path}::crf{crf}"
        reenc = reenc_cache.get(cache_key) if reenc_cache is not None else None
        if reenc is None:
            tmp_path = os.path.join(scratch_dir, f"nf_{episode}_crf{crf}.mp4")
            try:
                reencode_crf(path, tmp_path, crf)
                reenc = score_fn(tmp_path, dt)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            if reenc_cache is not None:
                reenc_cache[cache_key] = reenc

        rec: dict[str, Any] = {"episode": episode, "crf": crf}
        for m in metrics:
            b = float(orig.get(m, float("nan")))
            a = float(reenc.get(m, float("nan")))
            d = a - b
            rec[f"orig_{m}"] = b
            rec[f"reenc_{m}"] = a
            rec[f"nulldelta_{m}"] = d
            if math.isfinite(d):
                null[m].append(d)
        pairs.append(rec)

    summary = {m: summarize_null_deltas(null[m]) for m in metrics}
    return {
        "crf_rule": f"{crf_base} + ep%{crf_mod}",
        "n_clips": len(pairs),
        "summary": summary,
        "pairs": pairs,
    }


def below_floor(observed_delta: float, floor: Mapping[str, float] | float) -> bool:
    """``True`` iff ``observed_delta`` does not clear the noise floor.

    Thin wrapper around the already-existing, already-tested
    :func:`kinescore.bench.stats.noise_units` (``|delta| / floor``): a delta
    is *inconclusive* -- indistinguishable from paired re-encode noise --
    when that ratio is below 1. This is the "flag any delta smaller than the
    floor as not conclusive" behaviour a report should apply on top of
    :func:`build_noise_floor`'s output; it deliberately does not reimplement
    the ratio itself, since ``bench/stats.py`` already owns it.

    Parameters
    ----------
    observed_delta:
        The paired delta being checked (e.g. a method's median
        ``pred - gt`` on some metric).
    floor:
        Either the floor value directly, or a
        :func:`summarize_null_deltas`-shaped mapping (``floor["null_p95"]``
        is read from it).
    """
    from kinescore.bench.stats import noise_units

    floor_val = float(floor) if isinstance(floor, (int, float)) else float(floor["null_p95"])
    units = noise_units(observed_delta, floor_val)
    return not (math.isfinite(units) and units >= 1.0)
