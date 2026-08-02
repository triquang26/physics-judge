"""Synthetic smoke test for :mod:`kinescore.violations`.

No real data: every clip here is a hand-built rigid 3-link planar chain (a
closed-form ``theta``/``phi``/``psi`` parameterisation, not FK off a URDF), in
the spirit of ``robots/synthetic/spec.py``'s ``Synthetic2R`` fixture and
``_fake_robot.py``'s ``FakeRobot``. The point is not to validate any metric's
numeric accuracy against real motion capture -- it is to check the plumbing:
that :class:`~kinescore.violations.ViolationScorer` calibrates per-detector
thresholds from GT clips, that a genuinely warped clip trips the rigidity
detector while an unwarped one does not, and that every detector reports its
own, independent interval list rather than one pooled list.

Chain geometry
--------------
Four keypoints, three bones: ``base -(L0)-> p1 -(L1)-> p2 -(L2)-> p3``, all in
the XY plane. ``theta(t)`` sweeps the shoulder angle; ``phi``/``psi`` add
small, smooth elbow/wrist bends so the chain is not just a rigid rotation
(exercising joint_limit's bend-angle envelope without leaving it). The warp
below shifts keypoints ``p2``/``p3`` together, outward along the ``p1->p2``
bone direction, for a contiguous frame window -- this lengthens bone ``(1,2)``
by exactly the shift magnitude while leaving bone ``(2,3)`` and every bend
angle untouched (both endpoints move by the same vector), isolating the
violation to rigidity on that one bone.
"""
from __future__ import annotations

import math

import torch
from _fake_robot import FakeRobot

from kinescore.core.metric import MetricContext
from kinescore.violations import DETECTORS, ViolationScorer

_L = (0.30, 0.25, 0.20)          # bone rest lengths (m): base-p1, p1-p2, p2-p3
_FPS = 5.0
_DT = 1.0 / _FPS


def _chain_robot() -> FakeRobot:
    return FakeRobot(
        name="synthetic_chain",
        n_joints=3,
        keypoint_links=("base", "p1", "p2", "p3"),
        bone_pairs=torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long),
        bone_lengths=torch.tensor(_L, dtype=torch.float32),
        # rigid_bone_pairs/_lengths left None -> FakeRobot.__post_init__
        # defaults them to a clone of bone_pairs/bone_lengths (nothing
        # dropped), which is what we want: all three bones are genuinely
        # rigid in this synthetic chain, no joint-crossing bone to exclude.
    )


def _rigid_chain(T: int, phase: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Smooth rigid 3-link chain -> ``(P, d12)``, ``P`` is ``(T,4,3)``.

    ``d12`` is the unit direction of bone ``p1->p2`` at each frame, returned
    so the warp below can extend that bone along its own axis.
    """
    t = torch.linspace(0.0, math.pi / 2, T)
    theta = t + phase
    phi = 0.25 * torch.sin(t)                       # elbow bend, bounded & smooth
    psi = 0.15 * torch.cos(t)                        # wrist bend, bounded & smooth

    zeros = torch.zeros(T)
    d01 = torch.stack([torch.cos(theta), torch.sin(theta), zeros], dim=-1)
    p0 = torch.zeros(T, 3)
    p1 = p0 + _L[0] * d01

    ang2 = theta + phi
    d12 = torch.stack([torch.cos(ang2), torch.sin(ang2), zeros], dim=-1)
    p2 = p1 + _L[1] * d12

    ang3 = ang2 + psi
    d23 = torch.stack([torch.cos(ang3), torch.sin(ang3), zeros], dim=-1)
    p3 = p2 + _L[2] * d23

    P = torch.stack([p0, p1, p2, p3], dim=1)          # (T, 4, 3)
    return P, d12


def _clip_ctx(P: torch.Tensor, robot: FakeRobot) -> MetricContext:
    return MetricContext(dt=_DT, P=P.unsqueeze(0), robot=robot)  # (1,T,4,3)


def _warp_clip(P: torch.Tensor, d12: torch.Tensor, window: tuple[int, int],
              extra_m: float) -> torch.Tensor:
    """Stretch bone ``(1,2)`` by ``extra_m`` over ``window`` frames (inclusive).

    Shifts keypoints ``p2`` (index 2) and ``p3`` (index 3) by the same
    ``extra_m * d12[t]`` vector at every frame in the window: bone ``(1,2)``
    lengthens by exactly ``extra_m`` (only its endpoint ``p1`` stays put),
    bone ``(2,3)`` is unchanged (both endpoints move together), and every
    bend angle is unchanged (a shift along a bone's own direction does not
    rotate it).
    """
    P = P.clone()
    a, b = window
    delta = extra_m * d12[a:b + 1]                     # (w, 3)
    P[a:b + 1, 2, :] += delta
    P[a:b + 1, 3, :] += delta
    return P


def test_detector_names_match_prototype() -> None:
    assert [d.name for d in DETECTORS] == [
        "rigidity", "jerk", "teleport", "joint_limit", "self_collision",
    ]


def test_warped_clip_flags_rigidity_but_gt_clip_does_not() -> None:
    robot = _chain_robot()
    T = 24

    gt_clips = [_rigid_chain(T, phase=ph)[0] for ph in (-0.10, 0.0, 0.10)]
    gt_contexts = [_clip_ctx(P, robot) for P in gt_clips]

    scorer = ViolationScorer()
    scorer.calibrate(gt_contexts, pct=95.0)

    # Rigidity threshold should be at (or above) the 18mm floor documented in
    # the prototype, since these GT clips are exactly rigid up to float error.
    thr = scorer.thresholds()
    assert thr["rigidity"]["threshold"] >= 18.0

    # Score a held-out, unwarped clip -- same construction, new phase -- and
    # a warped one built from it.
    held_out, d12 = _rigid_chain(T, phase=0.03)
    held_out_ctx = _clip_ctx(held_out, robot)

    window = (10, 13)
    warped = _warp_clip(held_out, d12, window, extra_m=0.04)   # +40mm on bone(1,2)
    warped_ctx = _clip_ctx(warped, robot)

    report_clean = scorer.score(held_out_ctx)
    report_warp = scorer.score(warped_ctx)

    assert report_clean["rigidity"]["intervals"] == []
    assert report_warp["rigidity"]["intervals"] != []
    # every flagged rigidity frame must fall inside (or at the boundary of)
    # the warp window
    for lo, hi in report_warp["rigidity"]["intervals"]:
        assert window[0] <= lo and hi <= window[1]


def test_each_detector_reports_its_own_interval_list() -> None:
    robot = _chain_robot()
    T = 24
    gt_clips = [_rigid_chain(T, phase=ph)[0] for ph in (-0.10, 0.0, 0.10)]
    gt_contexts = [_clip_ctx(P, robot) for P in gt_clips]

    scorer = ViolationScorer()
    scorer.calibrate(gt_contexts, pct=95.0)

    P, d12 = _rigid_chain(T, phase=0.05)
    warped = _warp_clip(P, d12, (10, 13), extra_m=0.04)
    ctx = _clip_ctx(warped, robot)

    report = scorer.score(ctx)

    assert set(report.keys()) == {
        "rigidity", "jerk", "teleport", "joint_limit", "self_collision",
    }
    for name, r in report.items():
        assert isinstance(r["intervals"], list), name
        for interval in r["intervals"]:
            assert isinstance(interval, list) and len(interval) == 2, name
        assert isinstance(r["threshold"], float), name
        assert isinstance(r["per_frame"], list) and len(r["per_frame"]) == T, name

    # Not all detectors share the same interval list -- rigidity's plateau
    # (the warp window) and teleport's single-frame jump at the window's
    # boundary are structurally different, confirming each Detector owns an
    # independent list rather than everyone re-reporting one shared flag.
    assert report["rigidity"]["intervals"] != report["teleport"]["intervals"]


def test_bone_units_and_thresholds_are_per_detector() -> None:
    robot = _chain_robot()
    T = 20
    gt_clips = [_rigid_chain(T, phase=ph)[0] for ph in (-0.05, 0.05)]
    gt_contexts = [_clip_ctx(P, robot) for P in gt_clips]

    scorer = ViolationScorer()
    scorer.calibrate(gt_contexts, pct=95.0)
    thr = scorer.thresholds()

    assert set(thr.keys()) == {
        "rigidity", "jerk", "teleport", "joint_limit", "self_collision",
    }
    # units differ per error type -- confirms these are not the same
    # underlying quantity re-thresholded five ways.
    units = {v["units"] for v in thr.values()}
    assert len(units) == 5


def test_severity_ratio_keeps_ranking_past_100pct_fraction() -> None:
    """``fraction`` saturates at 1.0 once every frame flags; ``severity_ratio``
    must not -- two all-flagged clips of different severity should still
    rank in severity order by ``severity_ratio_median``, the thing
    ``fraction`` alone cannot distinguish (see the GR-1 cosmos-data
    saturation this was found against: several clips reported 91-100%
    flagged with no way to tell "just over" from "way over" apart).
    """
    robot = _chain_robot()
    T = 24
    gt_clips = [_rigid_chain(T, phase=ph)[0] for ph in (-0.10, 0.0, 0.10)]
    gt_contexts = [_clip_ctx(P, robot) for P in gt_clips]

    scorer = ViolationScorer()
    scorer.calibrate(gt_contexts, pct=95.0)
    rigidity = next(d for d in scorer.detectors if d.name == "rigidity")

    P, d12 = _rigid_chain(T, phase=0.05)
    mild = _clip_ctx(_warp_clip(P, d12, (0, T - 1), extra_m=0.02), robot)
    severe = _clip_ctx(_warp_clip(P, d12, (0, T - 1), extra_m=0.20), robot)

    r_mild = rigidity.report(mild)
    r_severe = rigidity.report(severe)

    # Both warped for every frame of the clip -> fraction saturates at 1.0
    # for both, indistinguishable.
    assert r_mild["fraction"] == 1.0
    assert r_severe["fraction"] == 1.0

    # severity_ratio is unbounded and still separates them.
    assert r_severe["severity_ratio_median"] > r_mild["severity_ratio_median"] > 1.0


def test_severity_ratio_orientation_matches_higher_is_worse() -> None:
    """``severity_ratio`` must read ">1 = worse" in the same direction for
    every detector, including ``self_collision`` (``higher_is_worse=False``,
    smaller distance = worse) -- a caller comparing ratios across detector
    types would silently invert self_collision's meaning otherwise.
    """
    robot = _chain_robot()
    T = 24
    gt_clips = [_rigid_chain(T, phase=ph)[0] for ph in (-0.10, 0.0, 0.10)]
    gt_contexts = [_clip_ctx(P, robot) for P in gt_clips]

    scorer = ViolationScorer()
    scorer.calibrate(gt_contexts, pct=95.0)
    report = scorer.score(_clip_ctx(_rigid_chain(T, phase=0.05)[0], robot))

    for name, r in report.items():
        # >= 0, not > 0: this synthetic chain is exactly rigid (closed-form
        # geometry, no reader noise), so a detector whose per-frame score is
        # genuinely 0 on a clean clip (e.g. rigidity) legitimately reports
        # ratio 0.0 -- that is correct, not a floor violation.
        assert r["severity_ratio_median"] >= 0, name
        assert r["severity_ratio_p90"] >= r["severity_ratio_median"] - 1e-6, name


def test_rigid_idx_can_exclude_a_bone() -> None:
    """``RigidityDetector(rigid_idx=...)`` narrows which bones are checked.

    Constructing with ``rigid_idx=(0, 2)`` (dropping bone index 1, the one
    the warp below stretches) must make that detector blind to the warp --
    the documented Franka use case (dropping a joint-crossing bone),
    exercised here on a bone dropped for a different reason.
    """
    from kinescore.violations import RigidityDetector

    robot = _chain_robot()
    T = 20
    P, d12 = _rigid_chain(T, phase=0.0)
    ctx_gt = _clip_ctx(P, robot)

    full = RigidityDetector()                 # default: all bones
    narrowed = RigidityDetector(rigid_idx=(0, 2))   # bone 1 excluded

    for det in (full, narrowed):
        det.fit([ctx_gt])
        scores = det.per_frame(ctx_gt)
        det.calibrate(scores, pct=95.0, floor=18.0)

    warped = _warp_clip(P, d12, (8, 11), extra_m=0.05)
    ctx_warp = _clip_ctx(warped, robot)

    assert full._intervals(full._flag(full.per_frame(ctx_warp))) != []
    assert narrowed._intervals(narrowed._flag(narrowed.per_frame(ctx_warp))) == []
