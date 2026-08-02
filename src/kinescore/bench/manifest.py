"""Enumerate clips to score, and verify paired gt/pred clips are comparable.

Ported from ``Marionette-fkjepa/eval/bench/manifest.py`` (the harness
backbone this module keeps): its :func:`~kinescore.video.probe.ffprobe` is
good and lives on unchanged in ``kinescore.video.probe``; its row schema is
kept, extended (see :data:`ROW_KEYS`); its ``verify_manifest`` pairing check
is kept and generalised. What does **not** survive the port:

* **Defect D3** -- the source's ``_row`` probed the file, then overwrote the
  probed ``fps`` with a hardcoded per-family config-table lookup, discarding
  the measurement. Row construction here goes through
  :func:`kinescore.video.probe.resolve_timebase`, which always probes and
  raises if a table value disagrees with the probe -- see that module's
  docstring for the full defect writeup.
* **Hardcoded discovery.** The source's ``build_manifest`` was three
  near-identical ``if family == "dreamdojo" / elif ... / else`` blocks baked
  into one function; adding a fourth data source meant editing shared code
  and risking the other two. Discovery here is a list of **plugins** -- plain
  zero-argument callables yielding :class:`DiscoveredClip` -- so a new data
  source is a new, independently testable callable, not a new branch.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from kinescore.core.clip import ViewLayout
from kinescore.video.probe import ffprobe, resolve_timebase

__all__ = [
    "ROW_KEYS", "DiscoveredClip", "SourcePlugin", "build_manifest",
    "verify_manifest", "save_manifest", "load_manifest", "autodetect_manifest",
]

#: Manifest row column order (also the parquet/JSON column order).
#:
#: The first eleven keys are the source's exact schema
#: (``{"method","family","episode","role","path","n_frames","fps","w","h",
#: "dt","pair_key"}``); everything after ``pair_key`` is new:
#:
#: * ``fps_probed`` -- what ffprobe measured, kept *even when a table/CLI
#:   value won* (``fps`` above), so a resolved-but-suspicious row is still
#:   auditable after the fact instead of the probe's answer being thrown away
#:   exactly like the D3 defect this module fixes.
#: * ``dt_source`` -- provenance of ``fps``/``dt``, from
#:   :class:`~kinescore.core.clip.ClipSpec.dt_source`.
#: * ``codec`` -- so :func:`verify_manifest` doesn't need to re-probe pairs to
#:   compare codecs (the source's ``verify_manifest`` did, because its row
#:   didn't carry codec).
#: * ``sha1`` -- optional content hash, for detecting a clip silently
#:   replaced on disk between manifest build and scoring.
#: * ``view_layout`` -- the camera-packing key (see
#:   :class:`~kinescore.core.clip.ViewLayout.key`).
ROW_KEYS = (
    "method", "family", "episode", "role", "path",
    "n_frames", "fps", "w", "h", "dt", "pair_key",
    "fps_probed", "dt_source", "codec", "sha1", "view_layout",
)


@dataclass(frozen=True)
class DiscoveredClip:
    """One clip as reported by a source plugin, before probing.

    A plugin only needs to know *where* a clip lives and its benchmark
    identity; probing (dimensions, frame count, and the D3 timebase
    cross-check) is centralised in :func:`build_manifest` so every plugin gets
    the same protection for free instead of reimplementing ffprobe calls
    per family, as the source's hardcoded cascade did.

    Parameters
    ----------
    method, family, episode, role, path, pair_key:
        Same meaning as the source's row schema -- see module docstring.
    fps_hint:
        This entry's config-table fps, if the plugin has one (``None`` if the
        plugin wants the probed rate trusted outright). Passed straight
        through to :func:`~kinescore.video.probe.resolve_timebase` as
        ``fps_table``.
    view_layout:
        Camera packing for this clip.
    """

    method: str
    family: str
    episode: str
    role: str
    path: str
    pair_key: str
    fps_hint: float | None = None
    view_layout: ViewLayout = field(default_factory=ViewLayout)


#: A source plugin enumerates the clips it knows about. Zero-argument so a
#: plugin closes over whatever configuration (root dir, glob pattern, fps
#: table) it needs at construction time -- ``build_manifest`` never sees that
#: configuration, only the resulting entries.
SourcePlugin = Callable[[], Iterable[DiscoveredClip]]


def _sha1_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _row(entry: DiscoveredClip, compute_sha1: bool) -> dict:
    clip = resolve_timebase(entry.path, fps_table=entry.fps_hint,
                            view_layout=entry.view_layout)
    probed = ffprobe(entry.path)
    return {
        "method": entry.method, "family": entry.family, "episode": entry.episode,
        "role": entry.role, "path": clip.path,
        "n_frames": clip.n_frames, "fps": clip.fps,
        "w": clip.width, "h": clip.height, "dt": clip.dt,
        "pair_key": entry.pair_key,
        "fps_probed": float(probed["fps"]), "dt_source": clip.dt_source,
        "codec": probed["codec"],
        "sha1": _sha1_file(entry.path) if compute_sha1 else None,
        "view_layout": clip.view_layout.key,
    }


def build_manifest(plugins: Sequence[SourcePlugin], *,
                   episode_cap: int | None = None,
                   compute_sha1: bool = False,
                   on_error: str = "skip") -> list[dict]:
    """Enumerate clips from ``plugins`` into manifest rows.

    Parameters
    ----------
    plugins:
        Zero-arg callables, each yielding :class:`DiscoveredClip` for one
        data source (dreamdojo, dreamgen, a fresh dataset, ...). Called in
        order; rows are appended in the order plugins and their entries are
        yielded.
    episode_cap:
        If given, stop pulling entries from a given ``family`` once that many
        have been *accepted* into the manifest from it (mirrors the source's
        ``episode_cap``, generalised from "per dreamdojo method" to "per
        family" since plugins no longer share one function to coordinate a
        finer-grained cap).
    compute_sha1:
        Hash every clip's bytes (slow for large files; off by default).
    on_error:
        ``"skip"`` (default) logs and drops a clip that fails to probe/resolve
        (unreadable file, TimebaseError, ...), matching the source's
        behaviour of not letting one corrupt pair abort the whole build.
        ``"raise"`` propagates the exception instead, for callers building a
        manifest where every entry is expected to be valid.

    Returns
    -------
    list of dict
        Rows with exactly the keys in :data:`ROW_KEYS`.
    """
    if on_error not in ("skip", "raise"):
        raise ValueError(f"on_error must be 'skip' or 'raise', got {on_error!r}")

    rows: list[dict] = []
    n_by_family: dict[str, int] = {}
    for plugin in plugins:
        for entry in plugin():
            if episode_cap is not None and n_by_family.get(entry.family, 0) >= episode_cap:
                continue
            try:
                row = _row(entry, compute_sha1)
            except Exception as exc:  # noqa: BLE001
                if on_error == "raise":
                    raise
                print(f"[manifest] skip {entry.pair_key} ({entry.path}): {exc}",
                     flush=True)
                continue
            rows.append(row)
            n_by_family[entry.family] = n_by_family.get(entry.family, 0) + 1
    return rows


def verify_manifest(rows: list[dict], *, dt_rel_tol: float = 0.01) -> dict:
    """Assert paired gt/pred clips share width, height, codec and ``dt``.

    Generalised from the source, which only checked pairs in the
    ``"dreamdojo"`` family by name. Any ``pair_key`` group containing both a
    ``"gt"`` and a ``"pred"`` role is checked here, regardless of family --
    pairing is a property of the roles present, not a hardcoded family string.

    Also simpler than the source in one respect: codec comparison uses the
    ``codec`` column the manifest row already carries (see :data:`ROW_KEYS`),
    instead of re-probing both files as the source had to (its row didn't
    carry codec).

    ``dt`` is checked alongside ``w``/``h``/``codec`` for the same reason
    those are checked: the whole point of a paired gt/pred comparison is that
    frame rate cancels *within the pair* (see ``docs/BENCHMARKING.md``,
    layer 1, "paired") -- a pair whose two members were probed at different
    frame rates is not a valid instance of that cancellation, and every
    derivative metric scored on it (speed, accel, jerk, ...) would be
    comparing two different physical quantities. This is therefore a **hard**
    mismatch, exactly like a width/height/codec disagreement, not a warning:
    it flips ``ok`` to ``False`` and is reported in ``mismatches``, in the
    same shape as the existing fields (``dt_ok``, ``gt_dt``, ``pred_dt``,
    mirroring ``wh_ok``/``gt_wh``/``pred_wh`` and ``codec_ok``/``gt_codec``/
    ``pred_codec``).

    Parameters
    ----------
    dt_rel_tol:
        Relative tolerance for the ``dt`` comparison (``math.isclose``'s
        ``rel_tol``). ``0.01`` (1%) by default, matching
        ``video/probe.py::resolve_timebase``'s ``probe_tolerance`` default --
        enough to absorb container fps rounding (e.g. 29.97 vs 30 both
        resolving to "the same" nominal rate) without passing a genuine
        cross-rate pair (e.g. 10 fps vs 16 fps, a 60% relative difference).

    Returns
    -------
    dict
        ``{"n_rows", "n_rows_by_family", "n_pairs", "pairs_per_method",
        "n_mismatches", "mismatches", "ok"}``.
    """
    by_pair: dict[str, list[dict]] = {}
    for r in rows:
        by_pair.setdefault(r["pair_key"], []).append(r)

    per_method: dict[str, int] = {}
    mismatches: list[dict] = []
    n_pairs = 0
    for pair_key, rs in sorted(by_pair.items()):
        roles = {r["role"] for r in rs}
        if not ({"gt", "pred"} <= roles):
            continue
        n_pairs += 1
        method = rs[0]["method"]
        per_method[method] = per_method.get(method, 0) + 1
        gt = next(r for r in rs if r["role"] == "gt")
        pred = next(r for r in rs if r["role"] == "pred")
        wh_ok = (gt["w"], gt["h"]) == (pred["w"], pred["h"])
        codec_ok = gt.get("codec") == pred.get("codec")
        dt_ok = math.isclose(gt["dt"], pred["dt"], rel_tol=dt_rel_tol)
        if not (wh_ok and codec_ok and dt_ok):
            mismatches.append({
                "pair_key": pair_key,
                "gt_wh": [gt["w"], gt["h"]], "pred_wh": [pred["w"], pred["h"]],
                "gt_codec": gt.get("codec"), "pred_codec": pred.get("codec"),
                "gt_dt": gt["dt"], "pred_dt": pred["dt"],
                "wh_ok": wh_ok, "codec_ok": codec_ok, "dt_ok": dt_ok,
            })

    n_rows_by_family: dict[str, int] = {}
    for r in rows:
        n_rows_by_family[r["family"]] = n_rows_by_family.get(r["family"], 0) + 1

    return {
        "n_rows": len(rows),
        "n_rows_by_family": n_rows_by_family,
        "n_pairs": n_pairs,
        "pairs_per_method": per_method,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches,
        "ok": len(mismatches) == 0,
    }


def save_manifest(rows: list[dict], outdir: str,
                  pairing_report: dict | None = None) -> dict[str, str]:
    """Write the manifest to ``outdir``, parquet if available, JSON otherwise.

    ``pandas``/``pyarrow`` are the ``bench`` extra and are lazily imported so
    this still works (via the JSON fallback) in the numpy-only CPU test tier
    that doesn't install them. The two formats are interchangeable to
    :func:`load_manifest`.
    """
    os.makedirs(outdir, exist_ok=True)
    paths: dict[str, str] = {}
    try:
        import pandas as pd
    except ImportError:
        jp = os.path.join(outdir, "bench_manifest.json")
        with open(jp, "w") as f:
            json.dump(rows, f, indent=2)
        paths["manifest"] = jp
    else:
        df = pd.DataFrame(rows, columns=list(ROW_KEYS))
        pq = os.path.join(outdir, "bench_manifest.parquet")
        df.to_parquet(pq, index=False)
        paths["manifest"] = pq

    if pairing_report is not None:
        rp = os.path.join(outdir, "pairing_report.json")
        with open(rp, "w") as f:
            json.dump(pairing_report, f, indent=2)
        paths["pairing_report"] = rp
    return paths


def load_manifest(path: str) -> list[dict]:
    """Read a manifest written by :func:`save_manifest` back into row dicts.

    Dispatches on extension: ``.parquet`` needs pandas/pyarrow (lazily
    imported), ``.json`` needs neither.
    """
    if path.endswith(".parquet"):
        import pandas as pd
        return pd.read_parquet(path).to_dict("records")
    with open(path) as f:
        return json.load(f)


def autodetect_manifest(dir_: str, *, required: bool = True) -> str | None:
    """Find ``bench_manifest.parquet``/``.json`` (or the first ``*manifest*``) under ``dir_``.

    One function, two contracts, chosen by ``required`` -- this used to be
    three near-identical copies (``cli/cmd_aggregate.py``, ``cli/cmd_rank.py``
    byte-identical and raising; ``cli/cmd_export.py``'s returning ``None``
    instead). The right contract depends on what the caller can do without a
    manifest:

    * ``required=True`` (default) -- raises :class:`FileNotFoundError`. Use
      this when the manifest is load-bearing (``kinescore aggregate``/
      ``kinescore rank`` immediately join it against ``results.jsonl``; there
      is nothing useful to do without it).
    * ``required=False`` -- returns ``None``. Use this when the caller has a
      documented degraded path (``kinescore export`` writes CSVs with blank
      episode/role identity and prints a warning rather than failing outright
      -- see :mod:`kinescore.bench.csv_export`).

    Making the caller pass ``required=`` explicitly is the point: the
    divergence is a one-word decision at the call site, not something a
    reader has to diff three copies of the same function to discover.
    """
    import glob

    for name in ("bench_manifest.parquet", "bench_manifest.json"):
        candidate = os.path.join(dir_, name)
        if os.path.exists(candidate):
            return candidate
    matches = sorted(glob.glob(os.path.join(dir_, "*manifest*")))
    if matches:
        return matches[0]
    if required:
        raise FileNotFoundError(
            f"no bench_manifest.parquet/.json found under {dir_!r}; pass "
            f"--manifest explicitly")
    return None
