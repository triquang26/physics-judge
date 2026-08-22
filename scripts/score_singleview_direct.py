#!/usr/bin/env python3
# What:   Score the eight SINGLEVIEW dreamdojo/dreamgen cells (fourier_gr1
#         humanoid + aloha_bimanual bimanual, makovian + non_makovian) with
#         the direct-keypoint readers, and emit the same artifact set the
#         three ctrlworld multiview cells already published to the
#         `twanghcmut/result-video-bench` bucket under `direct_keypoint/`:
#         per-cell `scored_direct.jsonl` + `RESULTS.md`, plus a run-level
#         `frame_scores.csv` and `thresholds.json`.
# Why:    STATUS.md lists these four generator x embodiment cells as "reader
#         ready, not scored". The readers exist
#         (`fourier_gr1_singleview_kp.pt` 45.31mm,
#         `aloha_bimanual_singleview_kp.pt` 73.69mm) and every detector in
#         `kinescore.violations` runs off keypoints alone, so nothing but a
#         driver was missing -- the ctrlworld run's driver was ad-hoc and was
#         never committed. Thresholds are calibrated per (robot, view) on
#         that reader's OWN real-teleop val split, never on the generated
#         clips being judged, so a cell cannot calibrate its own violations
#         away. `rigid_idx` is auto-resolved per robot by STABILITY: a bone
#         is scored only if its real-motion length varies by less than
#         --rigid-tol-mm (MAD around its own median), so a bone spanning a
#         moving joint does not manufacture a warp out of ordinary
#         articulation. Deliberately NOT the distance-from-URDF-rest rule in
#         `direct_keypoint/README.md` -- a biased-but-rigid reader passes the
#         stability test and fails that one; see auto_rigid_idx.
# Input:  $KINESCORE_DATA_ROOT/video_gen_physics/dense/<emb>/output/singleview/
#           <gen>/<horizon>/<pinned iter>/**  (predictions; ground truth
#           scored too wherever the export shape carries it)
#         $KINESCORE_DATA_ROOT/train/<domain>/videos/val/*.mp4  (calibration)
#         $KINESCORE_CKPT_DIR/{fourier_gr1,aloha_bimanual}_singleview_kp.pt
# Output: <out-root>/dense/<emb>/output/singleview/<gen>/<horizon>/
#           {scored_direct.jsonl,RESULTS.md}
#         <out-root>/{frame_scores.csv,thresholds.json,run_manifest.json}
"""Direct-keypoint violation scoring for the singleview benchmark cells.

The five detectors, their thresholds and their per-type interval lists come
straight from :mod:`kinescore.violations`; this module only resolves *which*
clips make up a cell, decodes them, turns each into a one-clip
:class:`~kinescore.core.metric.MetricContext`, and writes the published
artifact shapes. No metric logic lives here.

Two decode-side facts the ctrlworld cells never hit, both handled here:

* **Long clips.** ``ReadoutV2Head``'s temporal encoder has a fixed
  positional table (``t_max``, 64 for these checkpoints) and *raises* past
  it. Real singleview episodes run into the hundreds of frames, so
  :func:`read_keypoints` slides a ``t_max`` window with 50% overlap and
  keeps, for every frame, the prediction from the window in which that
  frame sits most centrally. Non-overlapping windows would put a seam --
  a fake teleport/jerk spike -- at every window boundary.
* **Mixed export shapes in one directory.** dreamdojo's
  ``fourier_gr1/makovian`` iter packs a flat ``NNNN_pred.mp4`` export and a
  nested ``episode_*/full_pred.mp4`` export into the same directory; see
  :data:`CELLS` for the per-cell resolution and
  ``configs/benchmark.yaml``'s source-pin comment for the survey behind it.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kinescore.core.clip import ClipSpec, ViewLayout
from kinescore.core.metric import MetricContext
from kinescore.readers import load_reader
from kinescore.robots import get_robot
from kinescore.video.probe import ffprobe
from kinescore.video.reader import load_rgb
from kinescore.violations import (
    JerkDetector,
    JointLimitDetector,
    RigidityDetector,
    SelfCollisionDetector,
    TeleportDetector,
    ViolationScorer,
)

SINGLE_VIEW = ViewLayout(n_views=1)

#: Detector order used in every emitted artifact (jsonl keys, CSV columns,
#: RESULTS.md columns) -- fixed so a consumer can rely on column order.
DETECTOR_ORDER = ("rigidity", "jerk", "teleport", "joint_limit", "self_collision")


@dataclass(frozen=True)
class Domain:
    """One (robot, view) reader and the real split its thresholds come from."""

    robot: str
    checkpoint: str
    train_domain: str          # $KINESCORE_DATA_ROOT/train/<train_domain>
    val_mm: float


DOMAINS = {
    "fourier_gr1_singleview": Domain(
        robot="fourier_gr1", checkpoint="fourier_gr1_singleview_kp.pt",
        train_domain="fourier_gr1_singleview", val_mm=45.31),
    "aloha_bimanual_singleview": Domain(
        robot="aloha_bimanual", checkpoint="aloha_bimanual_singleview_kp.pt",
        train_domain="aloha_bimanual_singleview", val_mm=73.69),
}


@dataclass(frozen=True)
class Cell:
    """One benchmark cell: where its clips are and which reader reads them.

    ``shapes`` lists the export shapes to look for, in order; every shape
    that resolves to at least one prediction contributes. A shape is
    ``(pred_glob, gt_suffix_or_None)`` relative to the pinned iter
    directory. ``gt`` is resolved by substituting ``pred`` -> ``gt`` in the
    matched prediction's own filename, which is how both dreamdojo exports
    pair (``0000_pred.mp4``/``0000_gt.mp4``,
    ``episode_x/full_pred.mp4``/``episode_x/full_gt.mp4``).
    """

    embodiment: str
    generator: str
    horizon: str
    iter_dir: str
    domain: str
    shapes: tuple[tuple[str, bool], ...]
    #: The LITERAL on-disk view directory. `singleview` and `single_view`
    #: both exist under `dense/bimanual/output/` and hold DIFFERENT episode
    #: sets -- `singleview` is pred-only, `single_view` carries real
    #: `full_gt.mp4` pairs (see configs/benchmark.yaml's sources comment).
    #: They must stay separate cells; globbing across the two spellings
    #: would silently merge two different exports.
    view_dir: str = "singleview"

    @property
    def rel(self) -> str:
        return (f"dense/{self.embodiment}/output/{self.view_dir}/"
                f"{self.generator}/{self.horizon}")

    @property
    def key(self) -> str:
        return f"{self.embodiment}/{self.view_dir}/{self.generator}/{self.horizon}"


#: iter pins are `configs/benchmark.yaml`'s, verified there against both the
#: HF tree and the local disk (see its `sources` header comment).
CELLS: tuple[Cell, ...] = (
    Cell("humanoid", "dreamdojo", "makovian", "iter_000050000",
         "fourier_gr1_singleview",
         (("*_pred.mp4", True), ("episode_*/full_pred.mp4", True))),
    Cell("humanoid", "dreamdojo", "non_makovian", "iter_000050000",
         "fourier_gr1_singleview",
         (("episode_*/full_pred.mp4", True),)),
    Cell("humanoid", "dreamgen", "makovian", "iter_000090000",
         "fourier_gr1_singleview", (("episode_*.mp4", False),)),
    Cell("humanoid", "dreamgen", "non_makovian", "iter_000090000",
         "fourier_gr1_singleview", (("episode_*.mp4", False),)),
    Cell("bimanual", "dreamdojo", "makovian", "iter_000100000",
         "aloha_bimanual_singleview", (("*/full_pred.mp4", True),)),
    Cell("bimanual", "dreamdojo", "non_makovian", "iter_000100000",
         "aloha_bimanual_singleview", (("*/full_pred.mp4", True),)),
    Cell("bimanual", "dreamgen", "makovian", "iter_000110000_bimanual16fps",
         "aloha_bimanual_singleview", (("*.mp4", False),)),
    Cell("bimanual", "dreamgen", "non_makovian", "iter_000110000_bimanual16fps",
         "aloha_bimanual_singleview", (("*.mp4", False),)),
    # The `single_view` (underscored) bimanual dreamdojo tree: FEWER episodes
    # than `singleview` (103/85 vs 150/150) but it is the only ALOHA
    # singleview export that carries real `full_gt.mp4` pairs, so it is the
    # only one that can say how much of a flagged frame is the generator and
    # how much is the reader's own 73.69 mm error. Different iter per
    # horizon, unlike every other dreamdojo cell -- both pins from
    # configs/benchmark.yaml.
    Cell("bimanual", "dreamdojo", "makovian", "iter_000100000",
         "aloha_bimanual_singleview", (("*/full_pred.mp4", True),),
         view_dir="single_view"),
    Cell("bimanual", "dreamdojo", "non_makovian", "iter_000070000",
         "aloha_bimanual_singleview", (("*/full_pred.mp4", True),),
         view_dir="single_view"),
)


# --------------------------------------------------------------------------
# decode + read
# --------------------------------------------------------------------------

def clip_spec(path: Path) -> ClipSpec:
    """Probe ``path`` into a single-view :class:`ClipSpec` with its own dt."""
    info = ffprobe(str(path))
    fps = float(info["fps"])                      # type: ignore[arg-type]
    return ClipSpec(
        path=str(path), fps=fps, dt=1.0 / fps,
        n_frames=int(info["n_frames"]),           # type: ignore[arg-type]
        width=int(info["w"]),                     # type: ignore[arg-type]
        height=int(info["h"]),                    # type: ignore[arg-type]
        view_layout=SINGLE_VIEW,
        codec=info.get("codec"))                  # type: ignore[arg-type]


def _windows(n: int, size: int) -> list[tuple[int, int]]:
    """Half-overlapping ``[start, end)`` windows covering ``range(n)``."""
    if n <= size:
        return [(0, n)]
    step = max(1, size // 2)
    out = []
    start = 0
    while start + size < n:
        out.append((start, start + size))
        start += step
    out.append((n - size, n))
    return out


@torch.no_grad()
def read_keypoints(reader: Any, frames: torch.Tensor, t_max: int,
                   encode_chunk: int = 64) -> torch.Tensor:
    """``(T,3,H,W)`` -> ``(T,K,3)`` keypoints, windowed past ``t_max``.

    The backbone is frozen and per-frame, so it runs ONCE over the whole
    clip in ``encode_chunk``-frame blocks; only the head's temporal encoder
    is windowed. Doing the whole ``reader.read`` per window instead would
    re-encode every overlapped frame (~2x the GPU cost, which dominates).

    ``ReadoutV2Head``'s positional table raises past ``t_max``, so windows
    are half-overlapping and each frame takes its prediction from the
    window in which it sits most centrally -- non-overlapping windows would
    put a fake teleport/jerk spike at every seam.
    """
    n = int(frames.shape[0])
    feats = []
    for s in range(0, n, encode_chunk):
        f = reader.backbone.encode(frames[s:s + encode_chunk])   # (b,V,P,D)
        _, V, P_, D = f.shape
        feats.append(f.reshape(-1, V * P_, D).float())
    feat = torch.cat(feats, dim=0)                               # (T, VP, D)

    def head_on(sl: slice) -> torch.Tensor:
        out = reader.head(feat[sl].unsqueeze(0), use_context=reader.use_context)
        return out["mu"].reshape(-1, reader.n_keypoints, 3)

    wins = _windows(n, t_max)
    if len(wins) == 1:
        return head_on(slice(0, n))

    out: torch.Tensor | None = None
    best = np.full(n, np.inf)
    for start, end in wins:
        P = head_on(slice(start, end))
        if out is None:
            out = P.new_zeros((n,) + tuple(P.shape[1:]))
        centre = (start + end - 1) / 2.0
        for local, frame in enumerate(range(start, end)):
            d = abs(frame - centre)
            if d < best[frame]:
                best[frame] = d
                out[frame] = P[local]
    assert out is not None
    return out


def read_clip(reader: Any, path: Path, max_frames: int) -> tuple[torch.Tensor, ClipSpec]:
    """Decode + read one clip into ``(T,K,3)`` CPU keypoints and its spec."""
    spec = clip_spec(path)
    frames = load_rgb(spec, max_frames=max_frames)
    device = next(reader.head.parameters()).device
    P = read_keypoints(reader, frames.to(device), t_max=int(reader.head.t_max))
    return P.float().cpu(), spec


def context_of(P: torch.Tensor, robot: Any, dt: float, stride: int = 1) -> MetricContext:
    """One-clip context from ``(T,K,3)`` keypoints, optionally decimated.

    ``stride`` is how the real-motion calibration split is brought onto a
    generated cell's frame rate: ``jerk`` (mm/frame³) and ``teleport``
    (mm/frame) are PER-FRAME quantities, so a threshold fitted at 30 fps
    means something different at 10 fps. Decimating the already-read
    keypoints costs nothing (the backbone pass is what is expensive) and is
    exactly ``ClipSpec.subsample``'s contract, one level up.
    """
    Q = P[::stride] if stride > 1 else P
    return MetricContext(dt=dt * stride, P=Q.unsqueeze(0), robot=robot,
                         flags={"limit_semantics": "keypoints"})


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def auto_rigid_idx(robot: Any, gt_contexts: list[MetricContext],
                   tol_mm: float) -> dict[str, Any]:
    """Keep only bones whose REAL length is STABLE across poses.

    "Rigid" means the length does not change as the arm moves. It does NOT
    mean the length equals URDF rest: a reader can be biased -- read a bone
    consistently 3x too long -- and still be perfectly rigid, and the
    threshold, being the 95th percentile of |L - rest| on REAL motion,
    absorbs a constant bias completely. What it cannot absorb is a bone whose
    length tracks actuation (one spanning a moving joint), because then
    ordinary articulation reads as a warp -- the Franka bone-1 case the
    ctrlworld run dropped.

    So selection is on the MEDIAN ABSOLUTE DEVIATION of the bone's length
    around its own median, not on its distance from URDF rest. An earlier
    version of this function used the distance-from-rest rule stated in
    `direct_keypoint/README.md`; on `fourier_gr1_singleview_kp.pt` that
    dropped all ten bones and made rigidity unavailable for every GR-1 cell,
    purely because that reader is biased -- its bones are stable to 19-35 mm
    MAD while sitting 18-148 mm from URDF rest. Selecting on bias threw away
    a working detector. Both numbers are reported below either way.
    """
    pairs = [(int(a), int(b)) for a, b in robot.rigid_bone_pairs]
    rest = robot.rigid_bone_lengths.cpu().float() * 1000.0
    devs, lens = [], []
    for ctx in gt_contexts:
        P = ctx.P[0]
        L = torch.stack([(P[:, a] - P[:, b]).norm(dim=-1) for a, b in pairs],
                        dim=1) * 1000.0
        devs.append((L - rest[None]).abs())
        lens.append(L)
    dev = torch.cat(devs, dim=0).median(dim=0).values
    L_all = torch.cat(lens, dim=0)
    length = L_all.median(dim=0).values
    mad = (L_all - length[None]).abs().median(dim=0).values
    keep = [i for i, m in enumerate(mad.tolist()) if m <= tol_mm]
    return {"stability_mad_mm": [round(float(m), 1) for m in mad.tolist()],
            "dev_from_rest_mm": [round(float(d), 1) for d in dev.tolist()],
            "rest_mm": [round(float(x), 1) for x in rest.tolist()],
            "predicted_mm": [round(float(x), 1) for x in length.tolist()],
            "pred_over_rest": [round(float(a / b), 2)
                               for a, b in zip(length.tolist(), rest.tolist(),
                                               strict=True)],
            "rigid_idx": keep,
            "tol_mm": tol_mm}


def build_domain(name: str, domain: Domain, data_root: Path, ckpt_dir: Path,
                 device: str, max_frames: int, cal_cap: int = 0,
                 cal_max_frames: int = 0) -> dict[str, Any]:
    """Load the reader and read this domain's real-teleop val split once.

    The keypoints are kept (not the video): every per-cell calibration below
    is a re-thresholding of these same arrays at a different stride, which
    costs nothing.
    """
    robot = get_robot(domain.robot)
    reader = load_reader(str(ckpt_dir / domain.checkpoint), robot=robot,
                         view_layout=SINGLE_VIEW, device=device)
    val_dir = data_root / "train" / domain.train_domain / "videos" / "val"
    val_clips = sorted(val_dir.glob("*.mp4"))
    if cal_cap:
        val_clips = val_clips[:cal_cap]
    if not val_clips:
        raise SystemExit(f"{name}: no calibration clips under {val_dir}")

    print(f"[{name}] reading {len(val_clips)} real val clips from {val_dir}",
          flush=True)
    gt_P, gt_fps = [], []
    t0 = time.time()
    for i, p in enumerate(val_clips, 1):
        P, spec = read_clip(reader, p, cal_max_frames or max_frames)
        gt_P.append(P)
        gt_fps.append(spec.fps)
        if i % 5 == 0 or i == len(val_clips):
            print(f"[{name}]   {i}/{len(val_clips)} "
                  f"({time.time()-t0:.0f}s, {sum(len(x) for x in gt_P)} frames)",
                  flush=True)
    return {"name": name, "domain": domain, "robot": robot, "reader": reader,
            "gt_P": gt_P, "gt_fps": gt_fps, "val_dir": val_dir,
            "cal_max_frames": cal_max_frames,
            "fps_cal": float(np.median(gt_fps)), "_cache": {}}


#: Largest relative gap between the strided calibration rate and a cell's own
#: rate that is still treated as rate-matched. Beyond it the cell's RESULTS.md
#: carries an explicit frame-rate caveat instead of a silent mismatch.
RATE_TOL = 0.15


def scorer_for(built: dict[str, Any], target_fps: float, rigid_tol_mm: float,
               pct: float) -> dict[str, Any]:
    """Calibrate the five detectors onto ``target_fps``; cached per stride.

    Picks the integer decimation of the real split whose resulting rate is
    closest to ``target_fps``. Only integer strides are considered: a
    fractional resample would have to interpolate, which smooths exactly the
    3rd-difference the jerk detector measures and would fit a threshold to
    motion no real camera ever recorded.
    """
    fps_cal = built["fps_cal"]
    stride = max(1, int(round(fps_cal / target_fps))) if target_fps > 0 else 1
    if stride in built["_cache"]:
        return built["_cache"][stride]

    robot = built["robot"]
    dt = 1.0 / fps_cal
    ctxs = [context_of(P, robot, dt, stride) for P in built["gt_P"]]
    rigid = auto_rigid_idx(robot, ctxs, rigid_tol_mm)
    # A reader whose keypoints do not reproduce ANY of this robot's URDF bone
    # lengths on REAL motion cannot have a rigidity threshold: every mm of
    # that gap is the reader's own error, so calibrating through it would
    # charge the generator for the reader. Drop the detector and say so,
    # rather than widen the tolerance until something passes. The other four
    # detectors are pure keypoint kinematics -- they never touch the URDF --
    # and stay valid.
    dets: list[Any] = []
    if rigid["rigid_idx"]:
        dets.append(RigidityDetector(rigid_idx=rigid["rigid_idx"]))
    else:
        rigid["excluded_reason"] = (
            f"no bone's real-motion length is stable to within "
            f"{rigid_tol_mm:g} mm (MAD around its own median = "
            f"{rigid['stability_mad_mm']} mm). Every bone's length tracks the "
            f"pose, so ordinary articulation would read as a warp. rigidity is "
            f"reported UNAVAILABLE for this reader, not zero and not widened.")
        print(f"[{built['name']}] rigidity EXCLUDED: {rigid['excluded_reason']}",
              flush=True)
    dets += [JerkDetector(), TeleportDetector(),
             JointLimitDetector(), SelfCollisionDetector()]
    scorer = ViolationScorer(dets)
    scorer.calibrate(ctxs, pct=pct)
    eff = fps_cal / stride
    entry = {
        "scorer": scorer, "rigid": rigid,
        "calibration": {
            "n_clips": len(ctxs), "source": str(built["val_dir"]),
            "fps_real": round(fps_cal, 3), "stride": stride,
            "fps_effective": round(eff, 3), "fps_target": round(target_fps, 3),
            "rate_matched": abs(eff - target_fps) <= RATE_TOL * target_fps,
            "percentile": pct,
            "n_frames": int(sum(c.P.shape[1] for c in ctxs)),
            "clip_frame_cap": built.get("cal_max_frames") or None,
        },
    }
    print(f"[{built['name']}] calibrated @stride {stride} "
          f"({fps_cal:g}->{eff:g} fps, target {target_fps:g}): "
          f"{scorer.thresholds()} rigid_idx={rigid['rigid_idx']}", flush=True)
    built["_cache"][stride] = entry
    return entry


# --------------------------------------------------------------------------
# cell resolution + scoring
# --------------------------------------------------------------------------

def resolve_clips(cell: Cell, dense_root: Path, cap: int) -> list[dict[str, Any]]:
    """Predictions (and their ground truth where the export carries it)."""
    root = dense_root / cell.embodiment / "output" / cell.view_dir / \
        cell.generator / cell.horizon / cell.iter_dir
    found: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for glob, has_gt in cell.shapes:
        for pred in sorted(root.glob(glob)):
            if pred in seen or not pred.is_file():
                continue
            seen.add(pred)
            gt = pred.with_name(pred.name.replace("pred", "gt")) if has_gt else None
            found.append({"pred": pred,
                          "gt": gt if (gt is not None and gt.exists()) else None,
                          "episode": (pred.parent.name if pred.name.startswith("full_")
                                      else pred.stem.replace("_pred", ""))})
    return found[:cap] if cap else found


def score_cell(cell: Cell, built: dict[str, Any], dense_root: Path,
               data_root: Path, out_root: Path, cap: int, max_frames: int,
               rigid_tol_mm: float, pct: float) -> dict[str, Any]:
    """Score one cell; write its jsonl + RESULTS.md; return its summary."""
    reader, robot = built["reader"], built["robot"]
    clips = resolve_clips(cell, dense_root, cap)
    out_dir = out_root / cell.rel
    out_dir.mkdir(parents=True, exist_ok=True)

    if not clips:
        summary = {"cell": cell.key, "status": "no_clips", "n_pred": 0,
                   "reason": f"no predictions matched {cell.shapes} under "
                             f"{cell.iter_dir}"}
        (out_dir / "scored_direct.jsonl").write_text("")
        return summary

    cell_fps = clip_spec(clips[0]["pred"]).fps
    entry = scorer_for(built, cell_fps, rigid_tol_mm, pct)
    scorer, cal = entry["scorer"], entry["calibration"]

    print(f"[{cell.key}] {len(clips)} predictions "
          f"({sum(1 for c in clips if c['gt'])} with gt) from {cell.iter_dir} "
          f"@{cell_fps:g} fps", flush=True)

    rows: list[dict[str, Any]] = []
    frame_rows: list[list[Any]] = []
    fps_seen: list[float] = []
    t0 = time.time()
    for i, item in enumerate(clips, 1):
        for role in ("pred", "gt"):
            path = item[role]
            if path is None:
                continue
            try:
                P, spec = read_clip(reader, path, max_frames)
                ctx = context_of(P, robot, spec.dt)
            except Exception as exc:                      # noqa: BLE001
                print(f"[{cell.key}] FAILED {role} {path.name}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                continue
            fps_seen.append(spec.fps)
            violations = scorer.score(ctx)
            rel_video = str(path.relative_to(data_root / "video_gen_physics"))
            rows.append({"video": rel_video, "role": role,
                         "episode": item["episode"],
                         "n_frames": int(ctx.P.shape[1]),
                         "fps": round(spec.fps, 3),
                         "violations": violations})
            for f in range(int(ctx.P.shape[1])):
                row: list[Any] = [rel_video, f, role]
                for name in DETECTOR_ORDER:
                    rep = violations.get(name)
                    if rep is None:              # detector excluded for this reader
                        row += ["", ""]
                        continue
                    s = rep["per_frame"][f]
                    hi = name != "self_collision"
                    flag = int(s > rep["threshold"] if hi else s < rep["threshold"])
                    row += [s, flag]
                frame_rows.append(row)
        if i % 10 == 0 or i == len(clips):
            print(f"[{cell.key}]   {i}/{len(clips)} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    with (out_dir / "scored_direct.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    pred_rows = [r for r in rows if r["role"] == "pred"]
    gt_rows = [r for r in rows if r["role"] == "gt"]

    def mean_frac(rs: list[dict[str, Any]]) -> dict[str, float]:
        return {n: round(float(np.mean([r["violations"][n]["fraction"]
                                        for r in rs])), 4)
                for n in DETECTOR_ORDER
                if rs and n in rs[0]["violations"]} if rs else {}

    summary = {
        "cell": cell.key,
        "status": "scored",
        "rel": cell.rel,
        "iter": cell.iter_dir,
        "reader": built["domain"].checkpoint,
        "robot": built["domain"].robot,
        "n_pred": len(pred_rows),
        "n_gt": len(gt_rows),
        "n_frames": sum(r["n_frames"] for r in rows),
        "fps_pred_median": round(float(np.median(fps_seen)), 3) if fps_seen else None,
        "calibration": cal,
        "clip_frame_cap": max_frames or None,
        "rigid_idx": entry["rigid"]["rigid_idx"],
        "thresholds": scorer.thresholds(),
        "mean_fraction_pred": mean_frac(pred_rows),
        "mean_fraction_gt": mean_frac(gt_rows),
    }
    write_results_md(cell, built, rows, out_dir, summary, entry)
    (out_dir / "cell_summary.json").write_text(json.dumps(summary, indent=1))
    return summary | {"_frame_rows": frame_rows}


def write_results_md(cell: Cell, built: dict[str, Any], rows: list[dict[str, Any]],
                     out_dir: Path, summary: dict[str, Any],
                     entry: dict[str, Any]) -> None:
    """Per-cell table, worst clip first -- the published RESULTS.md shape."""
    pred = [r for r in rows if r["role"] == "pred"]
    thr = entry["scorer"].thresholds()
    cal = entry["calibration"]
    rate_note = ""
    if not cal["rate_matched"]:
        rate_note = (
            f"\n**Frame-rate caveat — read before comparing jerk/teleport.** "
            f"The real split is {cal['fps_real']:g} fps and these clips are "
            f"{cal['fps_target']:g} fps. The closest available integer decimation "
            f"of the real split is stride {cal['stride']} → "
            f"{cal['fps_effective']:g} fps, still "
            f"{abs(cal['fps_effective'] - cal['fps_target']) / cal['fps_target'] * 100:.0f}% "
            f"off. `jerk` (mm/frame³) and `teleport` (mm/frame) are per-FRAME "
            f"quantities, so that gap biases them: a calibration rate FASTER than "
            f"the clips' own moves less per frame, making the threshold too tight "
            f"and over-flagging. `rigidity`, `joint_limit` and `self_collision` are "
            f"per-frame geometry and are unaffected. No fractional resample was "
            f"used — interpolating would smooth the very 3rd difference `jerk` "
            f"measures.\n")
    elif cal["stride"] > 1:
        rate_note = (
            f"\nReal split decimated stride {cal['stride']} "
            f"({cal['fps_real']:g} → {cal['fps_effective']:g} fps) to match these "
            f"clips' {cal['fps_target']:g} fps before thresholding, so the "
            f"per-frame `jerk`/`teleport` boundaries mean the same thing on both "
            f"sides.\n")

    def sev(r: dict[str, Any]) -> float:
        return max(v.get("severity_ratio_median", 0.0)
                   for v in r["violations"].values())

    pred.sort(key=sev, reverse=True)
    gt_line = ""
    if summary["mean_fraction_gt"]:
        gt_line = ("\nMean flagged fraction, pred vs the real conditioning video "
                   "scored through the same thresholds:\n\n"
                   "| | " + " | ".join(DETECTOR_ORDER) + " |\n"
                   "|---" * (len(DETECTOR_ORDER) + 1) + "|\n"
                   "| pred | " + " | ".join(
                       f"{summary['mean_fraction_pred'][n]*100:.1f}%"
                       if n in summary["mean_fraction_pred"] else "n/a"
                       for n in DETECTOR_ORDER) + " |\n"
                   "| gt | " + " | ".join(
                       f"{summary['mean_fraction_gt'][n]*100:.1f}%"
                       if n in summary["mean_fraction_gt"] else "n/a"
                       for n in DETECTOR_ORDER) + " |\n")

    lines = [
        f"# {built['domain'].robot} {cell.generator} / {cell.horizon} "
        f"(singleview) — direct-keypoint violations",
        "",
        f"reader `{built['domain'].checkpoint}` (val {built['domain'].val_mm} mm) · "
        f"iter `{cell.iter_dir}` · n_pred={len(pred)} · "
        f"n_gt={summary['n_gt']} · "
        f"rigid_idx={entry['rigid']['rigid_idx']}",
        "",
        "thresholds · " + " · ".join(
            f"{n} {thr[n]['threshold']:g} {thr[n]['units']}" for n in thr),
        "",
        f"Calibrated on {cal['n_clips']} real val clips, {cal['n_frames']} frames "
        f"({cal['percentile']:g}th pct; 5th for self_collision) from "
        f"`train/{built['domain'].train_domain}/videos/val` — never on the "
        f"generated clips below.",
        rate_note,
        (f"\n**Only the first {summary['clip_frame_cap']} frames of each clip were "
         f"scored** (`--max-frames`), so a violation later in a longer episode is "
         f"not in these numbers. Frame counts per clip are in "
         f"`scored_direct.jsonl`.\n" if summary["clip_frame_cap"] else ""),
        gt_line,
        "Ranked by `severity` = max over the five detectors of "
        "`severity_ratio_median` (per-frame score ÷ threshold, inverted for "
        "self_collision). 1.0 = a typical frame sits exactly on the boundary "
        "only 5% of real motion crosses. The flagged-frame %s saturate near "
        "100% on the worst clips and stop ranking; severity does not.",
        "",
        "| # | episode | severity | rigidity% | jerk% | teleport% | joint_limit% | self_collision% |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(pred, 1):
        v = r["violations"]
        lines.append(
            f"| {i} | {r['episode']} | {sev(r):.2f}x | " + " | ".join(
                f"{round(v[n]['fraction']*100):d}%" if n in v else "n/a"
                for n in DETECTOR_ORDER) + " |")
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap", type=int, default=150,
                    help="max predictions per cell (configs/benchmark.yaml's "
                         "caps.episodes_per_cell); 0 = no cap")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="hard per-clip frame cap (0 = whole clip)")
    ap.add_argument("--rigid-tol-mm", type=float, default=40.0,
                    help="a bone is scored for rigidity only if its REAL-motion "
                         "length is STABLE to within this (MAD around its own "
                         "median) -- see auto_rigid_idx for why stability, not "
                         "distance from URDF rest, is the right test. 40 mm "
                         "keeps all ten ALOHA bones (MAD 4.4-10.1) and eight "
                         "of ten GR-1 bones, dropping only right-arm bones 5 "
                         "and 6 (MAD 87.5 / 63.7 -- genuinely pose-dependent "
                         "in that reader). Every bone's MAD and its distance "
                         "from URDF rest are both recorded in thresholds.json.")
    ap.add_argument("--pct", type=float, default=95.0)
    ap.add_argument("--cal-max-frames", type=int, default=0,
                    help="per-clip frame cap for CALIBRATION clips only. The "
                         "real ALOHA val episodes run to 3000 frames; a 95th "
                         "percentile does not need all of them, and the "
                         "backbone pass over them is the run's single biggest "
                         "cost. 0 = whole clip. The cap actually used is "
                         "recorded in thresholds.json.")
    ap.add_argument("--cal-cap", type=int, default=0,
                    help="limit calibration clips per domain (smoke tests only; "
                         "0 = the whole val split, which is what a real run uses)")
    ap.add_argument("--only", default="",
                    help="comma-separated cell keys (emb/gen/horizon) to run")
    args = ap.parse_args(argv)

    cells = list(CELLS)
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        cells = [c for c in cells if c.key in want]
        if not cells:
            raise SystemExit(f"--only matched no cell; known: "
                             f"{[c.key for c in CELLS]}")

    dense_root = args.data_root / "video_gen_physics" / "dense"
    args.out_root.mkdir(parents=True, exist_ok=True)

    built: dict[str, dict[str, Any]] = {}
    for name in sorted({c.domain for c in cells}):
        built[name] = build_domain(name, DOMAINS[name], args.data_root,
                                   args.ckpt_dir, args.device, args.max_frames,
                                   args.cal_cap, args.cal_max_frames)

    header = ["video", "frame", "role"]
    for n in DETECTOR_ORDER:
        header += [f"{n}_score", f"{n}_flag"]
    csv_path = args.out_root / "frame_scores.csv"
    with csv_path.open("w", newline="") as fh:
        csv.writer(fh).writerow(header)

    summaries = []
    for cell in cells:
        s = score_cell(cell, built[cell.domain], dense_root, args.data_root,
                       args.out_root, args.cap, args.max_frames,
                       args.rigid_tol_mm, args.pct)
        # Appended per cell, not accumulated: the full run is ~10^6 rows and
        # holding them all costs more memory than every keypoint array
        # combined.
        with csv_path.open("a", newline="") as fh:
            csv.writer(fh).writerows(s.pop("_frame_rows", []))
        summaries.append(s)
        print(f"[{cell.key}] DONE {json.dumps({k: v for k, v in s.items() if k != 'thresholds'})}",
              flush=True)
        (args.out_root / "run_manifest.json").write_text(json.dumps({
            "cells": summaries, "cap_per_cell": args.cap,
            "readers": {n: {"checkpoint": b["domain"].checkpoint,
                            "robot": b["domain"].robot,
                            "val_mm": b["domain"].val_mm}
                        for n, b in built.items()},
        }, indent=1))

    (args.out_root / "thresholds.json").write_text(json.dumps({
        "thresholds": {f"{n}@stride{s}": e["scorer"].thresholds()
                       for n, b in built.items() for s, e in b["_cache"].items()},
        "rigid_idx": {f"{n}@stride{s}": e["rigid"]
                      for n, b in built.items() for s, e in b["_cache"].items()},
        "calibration": {f"{n}@stride{s}": e["calibration"]
                        for n, b in built.items() for s, e in b["_cache"].items()},
    }, indent=1))
    print("WROTE", args.out_root, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
