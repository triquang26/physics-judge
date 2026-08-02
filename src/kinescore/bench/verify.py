"""``kinescore data verify``'s engine: ffprobe every ingested clip against ``data_spec.yaml``.

Two kinds of problem, both reported with the SPECIFIC clip path (never just
"cell X has a problem" -- a failure three directories deep is useless
without the exact file to go look at):

* **Format mismatches** -- width/height always a hard error (a resolution
  drift silently changes what every downstream metric measures); fps a hard
  error UNLESS :attr:`~kinescore.bench.data_spec.GeneratorSpec.fps_tolerant`
  is set (ctrlworld only -- see that generator's real, verified mixed-rate
  minority in ``kinescore.bench.sources.ctrlworld``'s docstring); a
  pred/gt PAIR whose frame counts disagree is also a hard error -- the
  dataset's own per-generator contract (see ``configs/data_spec.yaml``'s
  header) gives width/height/fps as absolute constants but not an absolute
  frame count (ctrlworld's own docstring notes fps is not even uniform
  within a cell, so neither is frame count at a fixed fps); a mismatched
  PAIR is the checkable invariant instead, mirroring
  ``kinescore.bench.manifest.verify_manifest``'s existing wh/codec/dt pairing
  check, extended here to ``n_frames``.
* **Broken symlinks** -- a canonical episode whose ``pred.mp4``/``gt.mp4``
  points at nothing (the raw file moved or was deleted after ingestion).
  Detected via :meth:`~kinescore.bench.layout.CanonicalLayout.validate`,
  folded into the same report so a caller reads one list, not two.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from kinescore.bench.data_spec import DataSpec
from kinescore.bench.layout import GT_NAME, PRED_NAME, CanonicalLayout
from kinescore.video.probe import ffprobe

__all__ = ["ClipProblem", "VerifyReport", "verify_layout"]


@dataclass(frozen=True)
class ClipProblem:
    path: str
    reason: str


@dataclass
class VerifyReport:
    n_cells: int = 0
    n_episodes: int = 0
    n_clips_checked: int = 0
    problems: list[ClipProblem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _check_clip(path: str, *, gspec, problems: list[ClipProblem]) -> dict | None:
    try:
        probed = ffprobe(path)
    except Exception as exc:  # noqa: BLE001 -- an unreadable file is itself the finding
        problems.append(ClipProblem(path, f"unreadable / ffprobe failed: {exc}"))
        return None
    if probed["w"] != gspec.width or probed["h"] != gspec.height:
        problems.append(ClipProblem(
            path, f"resolution {probed['w']}x{probed['h']} != declared "
                 f"{gspec.width}x{gspec.height}"))
    return probed


def verify_layout(canonical: CanonicalLayout, data_spec: DataSpec, *,
                  fps_rel_tol: float = 0.01) -> VerifyReport:
    """ffprobe every clip :meth:`CanonicalLayout.cells` finds, against ``data_spec``.

    Parameters
    ----------
    fps_rel_tol:
        Relative tolerance for the fps comparison (mirrors
        ``kinescore.bench.manifest.verify_manifest``'s ``dt_rel_tol``
        default) -- absorbs container fps rounding, not a real rate mismatch.
    """
    report = VerifyReport()

    for msg in canonical.validate():
        report.problems.append(ClipProblem(canonical.root, msg))

    for cell in canonical.cells():
        report.n_cells += 1
        gspec = data_spec.generators.get(cell.generator)
        if gspec is None:
            report.problems.append(ClipProblem(
                canonical.cell_dir(cell),
                f"generator {cell.generator!r} not in data_spec.yaml"))
            continue
        cell_dir = canonical.cell_dir(cell)
        if not os.path.isdir(cell_dir):
            continue
        expected_fps = None if gspec.fps_tolerant else gspec.resolve_fps(robot=cell.robot)

        for name in sorted(os.listdir(cell_dir)):
            episode_dir = os.path.join(cell_dir, name)
            if not name.startswith("episode_") or not os.path.isdir(episode_dir):
                continue
            report.n_episodes += 1

            pred_path = os.path.join(episode_dir, PRED_NAME)
            gt_path = os.path.join(episode_dir, GT_NAME)
            has_pred = os.path.exists(pred_path)
            has_gt = os.path.exists(gt_path)

            if not has_pred:
                report.problems.append(ClipProblem(pred_path, "missing pred.mp4"))
            if gspec.has_ground_truth and not has_gt:
                report.problems.append(ClipProblem(gt_path, "missing gt.mp4 (generator "
                                                             "is supposed to have one)"))

            pred_probed = gt_probed = None
            if has_pred:
                report.n_clips_checked += 1
                pred_probed = _check_clip(pred_path, gspec=gspec, problems=report.problems)
                if (pred_probed is not None and expected_fps is not None
                        and not math.isclose(pred_probed["fps"], expected_fps,
                                             rel_tol=fps_rel_tol)):
                    report.problems.append(ClipProblem(
                        pred_path, f"fps {pred_probed['fps']} != declared {expected_fps}"))
            if has_gt:
                report.n_clips_checked += 1
                gt_probed = _check_clip(gt_path, gspec=gspec, problems=report.problems)

            if (pred_probed is not None and gt_probed is not None
                    and pred_probed["n_frames"] != gt_probed["n_frames"]):
                report.problems.append(ClipProblem(
                    episode_dir, f"pred/gt frame count mismatch: "
                                f"{pred_probed['n_frames']} vs {gt_probed['n_frames']}"))

    return report
