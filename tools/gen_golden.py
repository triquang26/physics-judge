#!/usr/bin/env python
"""Freeze the OLD judge's numeric BEHAVIOUR into ``tests/golden/*.npz``.

Why this runs before anyone touches ``src/kinescore``
-------------------------------------------------------
``source_snapshot.py`` proves which *bytes* the port started from. This
script proves which *numbers* those bytes produced -- for a fixed seed, a
fixed input, this exact FK chain and this exact head, here is the output,
to fp32 precision, forever. Once ``src/kinescore`` exists and starts
diverging from a straight port (bug fixes, refactors, degenerate-bone
handling -- see ``src/kinescore/core/robot.py``'s D9 note), there is no other
way to tell "we fixed a defect" from "we introduced one" than diffing against
what the defective original actually did. You cannot fix a bug you cannot
reproduce.

Determinism contract
---------------------
* ``torch.manual_seed(0)`` before every random draw (each generator reseeds,
  so fixture order in :data:`GENERATORS` cannot change any fixture's values).
* fp32 throughout -- no autocast, no fp16 (the source's own DINO encode path
  uses fp16, but every fixture here starts from the FK / head layer, which
  the source itself always runs outside autocast; see ``judge/fk.py``'s
  ``forward``).
* ``torch.use_deterministic_algorithms(True)`` where the ops involved support
  it (matmul/FK on CPU do; a handful of reduction ops warn instead of erroring
  and are left alone rather than papering over a warning as a hard failure).
* CPU only. No network, no GPU, no live HF Hub.

Layout
------
Each ``golden_*`` function returns a flat ``dict[str, np.ndarray]`` (nested
dicts/tuples from the source functions are flattened with ``__`` as the
namespace separator -- see :func:`_flatten`) plus the list of source
qualnames it exercised, for the manifest. :func:`main` drives all of them,
enforces the total fixture-size budget, and writes ``MANIFEST.json``.

``--diff`` mode re-runs each fixture's *inputs* through ``src/kinescore``
(not the source repos) and prints a legacy-vs-new table. It is expected to
report "unavailable" for every fixture until another agent lands the matching
``kinescore`` code -- see :data:`_DIFF_ADAPTERS`.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # torch is imported per-function at runtime; this is for annotations only
    import torch

__all__ = ["GENERATORS", "generate_all"]

# --------------------------------------------------------------------------
# sys.path: SOURCE trees only, never $R/src. Order matters -- see NOTE below.
# --------------------------------------------------------------------------
#
# A (Marionette-ciasc) has its own top-level `models/` directory (no
# `__init__.py` -- ctrl_world.py etc, unrelated to the judge). B
# (Marionette-fkjepa) also ships `models/` (WITH `__init__.py`, and this is
# the one `judge.pixel_judge` and `motion_reference.py` need). Verified
# empirically (see task notes): CPython's path-based finder keeps scanning
# sys.path for a *regular* package (has __init__.py) even after finding a
# directory without one, and a regular package always wins over an
# accumulated namespace-package portion regardless of order. So `import
# models` resolves to B's `models/` correctly whether A or B is inserted
# first. We insert B last (index 0) purely so the common case --
# `judge.*` from A -- doesn't have to compete with anything, not because
# order is load-bearing for the `models` case.


def _install_source_paths(source_a: Path, source_b: Path) -> None:
    for p in (source_a, source_b):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def _seed_everything(seed: int = 0) -> None:
    import torch
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        # Some CPU ops (e.g. certain scatter variants) only WARN when
        # nondeterministic under this flag rather than support strict mode.
        # We still want the flag on for everything that does honour it.
        pass


# --------------------------------------------------------------------------
# flatten / save helpers
# --------------------------------------------------------------------------

def _flatten(prefix: str, value, out: dict[str, np.ndarray]) -> None:
    """Recursively flatten dict/tuple/list/Tensor/scalar into ``out``.

    ``__`` is the namespace separator (not ``.``) so keys stay unambiguous
    and grep-able: ``report__mean_jerk_mps3``, ``case2__scores__vel_violation_frac``.
    """
    import torch

    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}__{k}" if prefix else str(k), v, out)
    elif isinstance(value, (tuple, list)) and not isinstance(value, str):
        for i, v in enumerate(value):
            _flatten(f"{prefix}__{i}", v, out)
    elif isinstance(value, torch.Tensor):
        out[prefix] = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        out[prefix] = value
    elif isinstance(value, (int, float, bool)):
        out[prefix] = np.array(value)
    elif isinstance(value, str):
        out[prefix] = np.array(value)
    else:
        raise TypeError(f"cannot flatten {prefix}: unsupported type {type(value)}")


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# golden_fk.npz
# --------------------------------------------------------------------------

def golden_fk(source_a: Path, source_b: Path) -> tuple[dict, list[str]]:
    import torch
    from judge.fk import FrankaFK

    _seed_everything(0)
    fk = FrankaFK(ee_link="panda_link8", device="cpu", dtype=torch.float32)

    lo = fk.joint_limits[:, 0]  # (7,)
    hi = fk.joint_limits[:, 1]  # (7,)

    # random q within joint limits, (4,16,7)
    u = torch.rand(4, 16, 7, dtype=torch.float32)
    q_random = lo + (hi - lo) * u

    # 3 handwritten poses: zeros, "home" (Franka's conventional ready pose --
    # not defined anywhere in the source; picked once here and pinned by
    # this fixture, documented so nobody re-derives a different "home"),
    # limit-corner (alternating lower/upper bound per joint).
    zeros_pose = torch.zeros(7, dtype=torch.float32)
    home_pose = torch.tensor(
        [0.0, -0.785398163, 0.0, -2.356194490, 0.0, 1.570796327, 0.785398163],
        dtype=torch.float32,
    )
    limit_corner = torch.stack(
        [lo[i] if i % 2 == 0 else hi[i] for i in range(7)]
    )
    q_handwritten = torch.stack([zeros_pose, home_pose, limit_corner])  # (3,7)

    gripper_values = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)

    P_random, R_random = [], []
    P_hw, R_hw = [], []
    for g in gripper_values:
        grip_r = g.expand(4, 16, 1)
        p, r = fk.forward_transforms(q_random, grip_r)
        P_random.append(p)
        R_random.append(r)

        grip_hw = g.expand(1, 3, 1)
        p2, r2 = fk.forward_transforms(q_handwritten[None], grip_hw)
        P_hw.append(p2[0])
        R_hw.append(r2[0])

    ee_pos_random, ee_rotvec_random = fk.ee_pose(q_random)
    ee_pos_hw, ee_rotvec_hw = fk.ee_pose(q_handwritten[None])

    out = {}
    P_random_stacked = torch.stack(P_random)                  # (3,4,16,8,3)
    _flatten("P_random", P_random_stacked, out)
    _flatten("R_random", torch.stack(R_random), out)          # (3,4,16,8,3,3)
    _flatten("P_handwritten", torch.stack(P_hw), out)         # (3,3,8,3)
    _flatten("R_handwritten", torch.stack(R_hw), out)         # (3,3,8,3,3)
    _flatten("ee_pos_random", ee_pos_random, out)
    _flatten("ee_rotvec_random", ee_rotvec_random, out)
    _flatten("ee_pos_handwritten", ee_pos_hw[0], out)
    _flatten("ee_rotvec_handwritten", ee_rotvec_hw[0], out)
    _flatten("bone_pairs", fk.bone_pairs, out)
    _flatten("bone_lengths", fk.bone_lengths, out)
    _flatten("joint_limits", fk.joint_limits, out)
    _flatten("joint_vel_limits", fk.joint_vel_limits, out)
    _flatten("joint_effort_limits", fk.joint_effort_limits, out)
    # inputs, for --diff and for anyone re-deriving by hand
    _flatten("q_random", q_random, out)
    _flatten("q_handwritten", q_handwritten, out)
    _flatten("gripper_values", gripper_values, out)
    out["pose_names"] = np.array(["zeros", "home", "limit_corner"], dtype="<U20")

    # Plain `q`/`P` aliases (closed gripper, i.e. gripper_values[0] == 0.0):
    # a bare "one q, one P, same leading shape" pair with no gripper axis to
    # unpack. Pure duplication of data already above under q_random/
    # P_random[0] -- added because it's the natural, obvious shape for a
    # first-touch consumer to guess at (verified against
    # tests/test_fk_parity_franka.py, owned by another agent, which looks for
    # exactly this pair and gracefully skips without it). Costs ~6KB.
    _flatten("q", q_random, out)
    _flatten("P", P_random_stacked[0], out)

    qualnames = ["FrankaFK.forward", "FrankaFK.forward_transforms", "FrankaFK.ee_pose"]
    return out, qualnames


# --------------------------------------------------------------------------
# golden_physics.npz
# --------------------------------------------------------------------------

def _q_archetype(name: str, t: torch.Tensor, lo: torch.Tensor,
                 hi: torch.Tensor) -> torch.Tensor:
    """One (T,7) joint trajectory archetype. ``t`` is the real time axis (s)."""
    import torch
    mid = (lo + hi) / 2.0
    half_span = (hi - lo) / 2.0 * 0.6  # stay inside limits with margin

    if name == "constant":
        return mid.expand(t.shape[0], 7).clone()

    if name == "smooth_sinusoid":
        freqs = torch.tensor([0.2, 0.25, 0.3, 0.15, 0.35, 0.4, 0.1])
        phases = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        phase_arg = 2 * torch.pi * freqs[None, :] * t[:, None] + phases[None, :]
        return mid[None, :] + half_span[None, :] * torch.sin(phase_arg)

    if name == "teleport_step":
        # piecewise-constant, jumps to a new within-limit corner every T//4
        # samples -- a popping/teleporting arm.
        T = t.shape[0]
        corners = torch.stack([
            lo, hi, mid, (lo + mid) / 2, (hi + mid) / 2,
        ])
        seg = max(T // 4, 1)
        idx = (torch.arange(T) // seg) % corners.shape[0]
        return corners[idx]

    if name == "high_freq_jitter":
        base = mid[None, :] + half_span[None, :] * 0.5 * torch.sin(
            2 * torch.pi * 0.1 * t[:, None]
        )
        g = torch.Generator().manual_seed(0)
        noise = torch.randn(t.shape[0], 7, generator=g) * (half_span[None, :] * 0.05)
        return base + noise

    raise ValueError(name)


ARCHETYPES = ("smooth_sinusoid", "teleport_step", "high_freq_jitter", "constant")
DTS = (0.05, 0.1, 0.2, 0.4)


def golden_physics(source_a: Path, source_b: Path) -> tuple[dict, list[str]]:
    import torch
    from judge.fk import FrankaFK
    from judge.physics import KinematicConsistency, PhysicsConsistency

    _seed_everything(0)
    fk = FrankaFK(ee_link="panda_link8", device="cpu", dtype=torch.float32)
    lo, hi = fk.joint_limits[:, 0], fk.joint_limits[:, 1]

    out: dict[str, np.ndarray] = {}
    T = 64
    trajectories: dict[str, dict[float, dict]] = {}

    for arch in ARCHETYPES:
        for dt in DTS:
            t = torch.arange(T, dtype=torch.float32) * dt
            q = _q_archetype(arch, t, lo, hi)[None]           # (1,T,7)
            gripper = torch.zeros(1, T, 1)
            P, R = fk.forward_transforms(q, gripper)          # (1,T,8,3),(1,T,8,3,3)
            trajectories.setdefault(arch, {})[dt] = {"q": q, "P": P, "R": R}

    for arch in ARCHETYPES:
        for dt in DTS:
            key = f"{arch}__dt{dt}"
            q, P, R = (trajectories[arch][dt][k] for k in ("q", "P", "R"))

            phys = PhysicsConsistency(
                fk.bone_pairs, fk.bone_lengths, dt=dt, accel_bound=5.0,
                q_lo=lo, q_hi=hi, vel_limits=fk.joint_vel_limits,
                effort_limits=fk.joint_effort_limits,
            )
            kc = KinematicConsistency(fk.bone_pairs, fk.bone_lengths)

            # a deterministic "commanded" pose distinct from the realised one,
            # so command_consistency_mm / KinematicConsistency.report(P_cmd=...)
            # exercise a non-degenerate path.
            offset = 0.01 * torch.sin(torch.arange(P.shape[1])[None, :, None, None].float())
            P_cmd = P + offset

            _flatten(f"{key}__report", phys.report(P, q=q, R=R), out)
            _flatten(f"{key}__invariant_residuals", phys.invariant_residuals(P, q=q, R=R), out)
            _flatten(f"{key}__motion_samples", phys.motion_samples(P, q=q, R=R), out)
            _flatten(f"{key}__joint_feasibility", phys.joint_feasibility(q), out)
            _flatten(f"{key}__limit_violation", phys.limit_violation(q), out)
            _flatten(f"{key}__angular_stats", phys.angular_stats(R), out)
            _flatten(f"{key}__energy", phys.energy(P), out)
            _flatten(f"{key}__momentum_continuity", phys.momentum_continuity(P), out)
            _flatten(f"{key}__kinetic_energy", phys.kinetic_energy(P), out)
            _flatten(f"{key}__speed_profile", PhysicsConsistency.speed_profile(P, dt), out)
            _flatten(f"{key}__kc_report_no_cmd", kc.report(P), out)
            _flatten(f"{key}__kc_report_with_cmd", kc.report(P, P_cmd), out)
            _flatten(f"{key}__kc_rigidity_residual_mm", kc.rigidity_residual_mm(P), out)
            _flatten(f"{key}__kc_rigidity_wobble_mm", kc.rigidity_wobble_mm(P), out)
            _flatten(f"{key}__kc_command_consistency_mm",
                      KinematicConsistency.command_consistency_mm(P, P_cmd), out)
            # inputs, for reproducibility
            _flatten(f"{key}__q", q, out)

    # profile_w1: one representative distributional comparison (smooth vs
    # teleport, same dt) plus the trivial self-distance sanity check.
    dt0 = DTS[0]
    speed_smooth = PhysicsConsistency.speed_profile(trajectories["smooth_sinusoid"][dt0]["P"], dt0)
    speed_teleport = PhysicsConsistency.speed_profile(trajectories["teleport_step"][dt0]["P"], dt0)
    out["profile_w1__smooth_vs_teleport"] = np.array(
        PhysicsConsistency.profile_w1(speed_smooth, speed_teleport)
    )
    out["profile_w1__self_zero"] = np.array(
        PhysicsConsistency.profile_w1(speed_smooth, speed_smooth)
    )

    qualnames = [
        "KinematicConsistency.rigidity_residual_mm", "KinematicConsistency.rigidity_wobble_mm",
        "KinematicConsistency.command_consistency_mm", "KinematicConsistency.report",
        "PhysicsConsistency.velocity", "PhysicsConsistency.acceleration", "PhysicsConsistency.jerk",
        "PhysicsConsistency.speed_stats", "PhysicsConsistency.accel_stats", "PhysicsConsistency.mean_jerk",
        "PhysicsConsistency.accel_violation_frac", "PhysicsConsistency.kinetic_energy",
        "PhysicsConsistency.limit_violation", "PhysicsConsistency.angular_velocity",
        "PhysicsConsistency.angular_stats", "PhysicsConsistency.joint_feasibility",
        "PhysicsConsistency.energy", "PhysicsConsistency.momentum_continuity",
        "PhysicsConsistency.invariant_residuals", "PhysicsConsistency.motion_samples",
        "PhysicsConsistency.report", "PhysicsConsistency.speed_profile", "PhysicsConsistency.profile_w1",
    ]
    return out, qualnames


# --------------------------------------------------------------------------
# golden_head.npz
# --------------------------------------------------------------------------

def golden_head(source_a: Path, source_b: Path) -> tuple[dict, list[str]]:
    import torch
    from judge.pixel_judge import AttentivePoseHead, DinoPoseHead, pool_patch_tokens

    out: dict[str, np.ndarray] = {}

    # -- pool_patch_tokens: square grid so k=2 pooling is well-defined -----
    _seed_everything(0)
    tokens = torch.randn(2, 5, 16, 32)  # (...,P=16=4x4,D=32)
    _flatten("pool__tokens_in", tokens, out)
    _flatten("pool__k1_noop", pool_patch_tokens(tokens, 1), out)
    _flatten("pool__k2_pooled", pool_patch_tokens(tokens, 2), out)

    # -- DinoPoseHead --------------------------------------------------------
    _seed_everything(0)
    dino_head = DinoPoseHead(in_dim=32, hidden=16, n_joints=7, dropout=0.1)
    dino_head.eval()
    feat_dino = torch.randn(2, 5, 32)
    with torch.no_grad():
        out_dino = dino_head(feat_dino)
    _flatten("dino_head__state_dict", dict(dino_head.state_dict()), out)
    _flatten("dino_head__feat_in", feat_dino, out)
    _flatten("dino_head__output", out_dino, out)

    # -- AttentivePoseHead, n_cams=1 -----------------------------------------
    _seed_everything(0)
    head1 = AttentivePoseHead(in_dim=32, hidden=16, n_joints=7, dropout=0.1,
                              n_heads=2, n_cams=1)
    head1.eval()
    feat_attn = torch.randn(2, 5, 27, 32)
    with torch.no_grad():
        out1, w1 = head1(feat_attn, return_attn=True)
    _flatten("attn_ncam1__state_dict", dict(head1.state_dict()), out)
    _flatten("attn_ncam1__feat_in", feat_attn, out)
    _flatten("attn_ncam1__output", out1, out)
    _flatten("attn_ncam1__w", w1, out)

    # -- AttentivePoseHead, n_cams=3 (N=27 -> P=9 per camera) ----------------
    _seed_everything(0)
    head3 = AttentivePoseHead(in_dim=32, hidden=16, n_joints=7, dropout=0.1,
                              n_heads=2, n_cams=3)
    head3.eval()
    with torch.no_grad():
        out3, w3 = head3(feat_attn, return_attn=True)
    _flatten("attn_ncam3__state_dict", dict(head3.state_dict()), out)
    _flatten("attn_ncam3__output", out3, out)
    _flatten("attn_ncam3__w", w3, out)

    qualnames = ["pool_patch_tokens", "DinoPoseHead.forward", "AttentivePoseHead.forward"]
    return out, qualnames


# --------------------------------------------------------------------------
# golden_predict_pose.npz -- RETIRED
# --------------------------------------------------------------------------
# This generator used to freeze `PixelPhysicsJudge.predict_pose`'s sigmoid
# squash (raw -> q = lo + (hi-lo)*sigmoid(raw)) so `tests/test_head_golden.py`
# could check `kinescore.heads.ranges.squash_to_limits` against it. Both the
# squash function and the squashed pose-reader it validated
# (readers/squashed.py::SquashedPoseReader) have been removed from
# src/kinescore entirely -- see docs/PROVENANCE.md's D7 addendum -- so there
# is nothing left in this package for the fixture to validate. The generator
# function and its `tests/golden/golden_predict_pose.npz` output are both
# retired, not merely unused: re-adding either would resurrect a fixture for
# code that no longer exists. See git history for the removed implementation.


# --------------------------------------------------------------------------
# golden_reference.npz
# --------------------------------------------------------------------------

def golden_reference(source_a: Path, source_b: Path) -> tuple[dict, list[str]]:
    import torch
    from judge.fk import FrankaFK
    from judge.physics import PhysicsConsistency
    from models.evaluation.motion_reference import RealMotionReference

    _seed_everything(0)
    fk = FrankaFK(ee_link="panda_link8", device="cpu", dtype=torch.float32)
    lo, hi = fk.joint_limits[:, 0], fk.joint_limits[:, 1]
    dt = 0.1
    T = 32
    t = torch.arange(T, dtype=torch.float32) * dt

    phys = PhysicsConsistency(
        fk.bone_pairs, fk.bone_lengths, dt=dt, accel_bound=5.0,
        q_lo=lo, q_hi=hi, vel_limits=fk.joint_vel_limits,
        effort_limits=fk.joint_effort_limits,
    )

    def _pose(q: torch.Tensor):
        P, R = fk.forward_transforms(q, torch.zeros(1, T, 1))
        res = phys.invariant_residuals(P, q=q, R=R)
        samp = phys.motion_samples(P, q=q, R=R)
        return res, samp

    # 12 "real" rollouts: smooth sinusoids with per-rollout jittered
    # frequency/phase/amplitude (deterministic via a per-rollout torch
    # Generator, so this is stable across reruns and independent of any
    # earlier global RNG draws in this function).
    residual_list, samples_list = [], []
    for i in range(12):
        g = torch.Generator().manual_seed(1000 + i)
        freqs = 0.15 + 0.25 * torch.rand(7, generator=g)
        phases = 2 * torch.pi * torch.rand(7, generator=g)
        amp = ((hi - lo) / 2.0) * (0.3 + 0.2 * torch.rand(7, generator=g))
        mid = (lo + hi) / 2.0
        q = (mid[None, :] + amp[None, :] *
             torch.sin(2 * torch.pi * freqs[None, :] * t[:, None] + phases[None, :]))[None]
        res, samp = _pose(q)
        residual_list.append(res)
        samples_list.append(samp)

    reference = RealMotionReference.build(residual_list, samples_list, n_q=100)

    out: dict[str, np.ndarray] = {}
    _flatten("reference__inv_baseline", reference.inv_baseline, out)
    _flatten("reference__quantiles", reference.quantiles, out)
    _flatten("reference__feat_mu", reference.feat_mu, out)
    _flatten("reference__feat_cov", reference.feat_cov, out)
    out["reference__quantity_keys"] = np.array(list(reference.quantity_keys), dtype="<U20")

    # 5 test cases -----------------------------------------------------------
    cases: dict[str, dict] = {}

    # 1. near-zero baseline: the real median itself, barely perturbed -> pis~0
    baseline_res = {k: v * 1.0001 for k, v in reference.inv_baseline.items()}
    q_mid = ((lo + hi) / 2.0)[None, :].expand(T, 7)[None]  # (1,T,7), held constant
    _, baseline_samp = _pose(q_mid)
    cases["case1_near_zero_baseline"] = (baseline_res, baseline_samp)

    # 2. missing-q: invariant_residuals()/motion_samples() called WITHOUT q,
    # so q-dependent keys (limit_violation_frac, vel_violation_frac,
    # effort_proxy, qdot, qddot) are simply absent -- tests the "graceful
    # with fewer keys" path in invariance_score/w1.
    g = torch.Generator().manual_seed(2000)
    freqs = 0.2 + 0.1 * torch.rand(7, generator=g)
    q_missing = (((lo + hi) / 2.0)[None, :] +
                 ((hi - lo) / 4.0)[None, :] *
                 torch.sin(2 * torch.pi * freqs[None, :] * t[:, None]))[None]
    P_missing, R_missing = fk.forward_transforms(q_missing, torch.zeros(1, T, 1))
    res_missing = phys.invariant_residuals(P_missing, q=None, R=R_missing)
    samp_missing = phys.motion_samples(P_missing, q=None, R=R_missing)
    cases["case2_missing_q"] = (res_missing, samp_missing)

    # 3. severe violation: teleporting joints -> huge jerk/accel -> pis -> 1
    corners = torch.stack([lo, hi, (lo + hi) / 2, lo, hi])
    seg = max(T // 4, 1)
    idx = (torch.arange(T) // seg) % corners.shape[0]
    q_teleport = corners[idx][None]
    res_teleport, samp_teleport = _pose(q_teleport)
    cases["case3_severe_teleport"] = (res_teleport, samp_teleport)

    # 4. high-frequency jitter: bounded velocity, explosive jerk
    g = torch.Generator().manual_seed(3000)
    base = ((lo + hi) / 2.0)[None, :] + ((hi - lo) / 8.0)[None, :] * torch.sin(
        2 * torch.pi * 0.1 * t[:, None]
    )
    noise = torch.randn(T, 7, generator=g) * ((hi - lo)[None, :] * 0.02)
    q_jitter = (base + noise)[None]
    res_jitter, samp_jitter = _pose(q_jitter)
    cases["case4_high_freq_jitter"] = (res_jitter, samp_jitter)

    # 5. moderate/mixed: smooth but larger-amplitude, faster than the "real"
    # reference set (partial violation, not saturating)
    g = torch.Generator().manual_seed(4000)
    freqs = 0.5 + 0.2 * torch.rand(7, generator=g)
    q_moderate = (((lo + hi) / 2.0)[None, :] + ((hi - lo) / 2.0)[None, :] * 0.5 *
                  torch.sin(2 * torch.pi * freqs[None, :] * t[:, None]))[None]
    res_moderate, samp_moderate = _pose(q_moderate)
    cases["case5_moderate_mixed"] = (res_moderate, samp_moderate)

    # NOTE (discovered while generating this fixture, not asserted a priori):
    # RealMotionReference.kfd is LESS forgiving than .w1 about missing keys --
    # .w1 skips any reference quantity absent from a sample dict (`if k not in
    # samples: continue`), but .kfd's `_feat_vector` indexes `samples[k]` for
    # every one of `self.quantity_keys` unconditionally and raises KeyError if
    # one is missing. case2_missing_q (built with q=None, so it has no
    # 'qdot'/'qddot') would crash kfd() if included in its input list. This
    # asymmetry is exactly the kind of behaviour a straight port must
    # preserve -- silently "fixing" kfd to also skip missing keys would be a
    # semantic change, not a refactor. `kfd_eligible` records which cases
    # this fixture actually fed to kfd() so a future port test can assert the
    # same asymmetry (e.g. via pytest.raises on the excluded case).
    case_samples_for_kfd = []
    for name, (res, samp) in cases.items():
        pis, scores = reference.invariance_score(res)
        w1 = reference.w1(samp)
        eligible = set(reference.quantity_keys) <= set(samp)
        _flatten(f"{name}__pis", pis, out)
        _flatten(f"{name}__n_terms", len(scores), out)
        _flatten(f"{name}__scores", scores, out)
        _flatten(f"{name}__w1", w1, out)
        _flatten(f"{name}__residuals", res, out)
        _flatten(f"{name}__kfd_eligible", eligible, out)
        if eligible:
            case_samples_for_kfd.append(samp)

    out["kfd"] = np.array(reference.kfd(case_samples_for_kfd))
    out["kfd_n_cases"] = np.array(len(case_samples_for_kfd))

    qualnames = [
        "RealMotionReference.build", "RealMotionReference.invariance_score",
        "RealMotionReference.w1", "RealMotionReference.kfd",
    ]
    return out, qualnames


# --------------------------------------------------------------------------
# golden_gr1_fk.npz
# --------------------------------------------------------------------------

def golden_gr1_fk(source_a: Path, source_b: Path) -> tuple[dict, list[str]]:
    import torch
    from models.kinematics.gr1_fk import GR1FK

    urdf_path = source_b / "assets/grx/GRX/GR1/gr1t1/urdf/gr1t1_fourier_hand_6dof.urdf"
    if not urdf_path.is_file():
        raise FileNotFoundError(f"GR1 URDF not found at {urdf_path}")

    _seed_everything(0)
    fk = GR1FK(str(urdf_path), device="cpu")

    u = torch.rand(2, 10, 17, dtype=torch.float32)
    q17 = fk.q_lo[None, None, :] + (fk.q_hi - fk.q_lo)[None, None, :] * u

    P_left, R_left = fk.forward_transforms(q17, "left")
    P_right, R_right = fk.forward_transforms(q17, "right")
    ee_pos_left, ee_rotvec_left = fk.ee_pose(q17, "left")
    ee_pos_right, ee_rotvec_right = fk.ee_pose(q17, "right")

    out: dict[str, np.ndarray] = {}
    _flatten("q17", q17, out)
    _flatten("P_left", P_left, out)
    _flatten("R_left", R_left, out)
    _flatten("P_right", P_right, out)
    _flatten("R_right", R_right, out)
    _flatten("ee_pos_left", ee_pos_left, out)
    _flatten("ee_rotvec_left", ee_rotvec_left, out)
    _flatten("ee_pos_right", ee_pos_right, out)
    _flatten("ee_rotvec_right", ee_rotvec_right, out)
    _flatten("q_lo", fk.q_lo, out)
    _flatten("q_hi", fk.q_hi, out)
    _flatten("q_vel_max", fk.q_vel_max, out)
    _flatten("bone_pairs_left", fk.bone_pairs_left, out)
    _flatten("bone_lengths_left", fk.bone_lengths_left, out)
    _flatten("bone_pairs_right", fk.bone_pairs_right, out)
    _flatten("bone_lengths_right", fk.bone_lengths_right, out)

    qualnames = ["GR1FK.forward_transforms", "GR1FK.ee_pose", "GR1FK.keypoints_fk"]
    return out, qualnames


# --------------------------------------------------------------------------
# golden_ckpt_head.npz
# --------------------------------------------------------------------------

_CKPTS = {
    "judge_v3l": ("model_ckpt/judge_v3l/judge.pt", 1, 49),
    "judge_v3l_mv": ("model_ckpt/judge_v3l_mv/judge.pt", 3, 147),
}


def golden_ckpt_head(source_a: Path, source_b: Path) -> tuple[dict, list[str]]:
    """REAL checkpoint weights, but NOT the checkpoint bytes themselves.

    Only the tiny (1,8,8) output + the checkpoint's own SHA-256 + cfg are
    stored (~100s of bytes each) -- the checkpoints stay ~10MB each and are
    explicitly out of budget for this fixture file. This preserves the
    *behaviour* (what did this exact set of trained weights output for this
    exact seeded input) even if ``model_ckpt/`` is later deleted; it does not
    let you regenerate the input feature tensor without the seed+shape
    recorded here AND a torch version whose ``randn`` stream matches.
    """
    import torch
    from judge.pixel_judge import AttentivePoseHead

    out: dict[str, np.ndarray] = {}
    qualnames = ["AttentivePoseHead.forward"]

    for name, (rel, _n_cams, n_tokens) in _CKPTS.items():
        ckpt_path = source_a / rel
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        sha = _sha256_of(ckpt_path)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]

        head = AttentivePoseHead(
            in_dim=cfg["embed_dim"], hidden=512, n_joints=7, dropout=0.1,
            n_heads=cfg["n_heads"], n_cams=cfg.get("n_cams", 1),
        )
        head.load_state_dict(ck["head"], strict=True)
        head.eval()

        seed = 0
        g = torch.Generator().manual_seed(seed)
        feat = torch.randn(1, 8, n_tokens, cfg["embed_dim"], generator=g)
        with torch.no_grad():
            output = head(feat)

        out[f"{name}__output"] = output.detach().cpu().numpy()
        out[f"{name}__ckpt_sha256"] = np.array(sha)
        out[f"{name}__cfg_json"] = np.array(json.dumps(cfg, sort_keys=True))
        out[f"{name}__seed"] = np.array(seed)
        out[f"{name}__feat_shape"] = np.array([1, 8, n_tokens, cfg["embed_dim"]])

    return out, qualnames


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[Path, Path], tuple[dict, list[str]]]] = {
    "golden_fk": golden_fk,
    "golden_physics": golden_physics,
    "golden_head": golden_head,
    "golden_reference": golden_reference,
    "golden_gr1_fk": golden_gr1_fk,
    "golden_ckpt_head": golden_ckpt_head,
}

#: fixture name -> source-relative paths it actually exercises, each
#: prefixed "A/" or "B/" for which root to resolve against. Used only to
#: attach source SHA-256s to MANIFEST.json -- the single source of truth for
#: "which bytes produced this fixture" is still ``provenance/sources.sha256``
#: (source_snapshot.py); this is a convenience cross-reference so the
#: manifest is self-contained without requiring the reader to also have run
#: that script.
FIXTURE_SOURCE_FILES: dict[str, tuple[str, ...]] = {
    "golden_fk": ("A/judge/fk.py",),
    "golden_physics": ("A/judge/fk.py", "A/judge/physics.py"),
    "golden_head": ("A/judge/pixel_judge.py",),
    "golden_reference": (
        "A/judge/fk.py", "A/judge/physics.py",
        "B/models/evaluation/motion_reference.py",
    ),
    "golden_gr1_fk": ("B/models/kinematics/gr1_fk.py",),
    "golden_ckpt_head": ("A/judge/pixel_judge.py",),
}

#: fixture name -> the RNG seed(s) its generator function seeds with, for
#: the manifest. Every generator in this file calls ``_seed_everything(0)``
#: (or a per-item ``torch.Generator().manual_seed(N)`` documented inline in
#: that generator) -- recorded here rather than introspected, since several
#: generators mix a global seed with local per-item generators.
FIXTURE_SEEDS: dict[str, str] = {
    "golden_fk": "torch.manual_seed(0)",
    "golden_physics": "torch.manual_seed(0); high_freq_jitter uses "
                       "torch.Generator().manual_seed(0) per (archetype,dt)",
    "golden_head": "torch.manual_seed(0), reseeded before each of the 4 heads",
    "golden_reference": "12 real rollouts: torch.Generator().manual_seed(1000+i) "
                         "for i in range(12); 5 test cases: seeds 2000/3000/4000 "
                         "where randomness is used",
    "golden_gr1_fk": "torch.manual_seed(0)",
    "golden_ckpt_head": "torch.Generator().manual_seed(0) per checkpoint",
}

#: Total budget across every fixture file (see task spec). Enforced in
#: :func:`generate_all` AFTER all fixtures are written, so a single oversized
#: fixture is easy to spot from the per-file report.
BUDGET_BYTES = 4 * 1024 * 1024


def _package_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for mod_name, pkg_name in (
        ("torch", "torch"), ("numpy", "numpy"),
        ("pytorch_kinematics", "pytorch-kinematics"),
        ("robot_descriptions", "robot-descriptions"),
    ):
        try:
            import importlib.metadata as im
            versions[pkg_name] = im.version(pkg_name)
        except Exception:
            try:
                versions[pkg_name] = importlib.import_module(mod_name).__version__
            except Exception as exc:
                versions[pkg_name] = f"unknown ({exc})"
    return versions


def _urdf_sha256() -> tuple[str, str]:
    from robot_descriptions import panda_description
    p = Path(panda_description.URDF_PATH)
    return str(p), _sha256_of(p)


def generate_all(source_a: Path, source_b: Path, out_dir: Path,
                 only: list[str] | None = None) -> dict:
    """Run every (or ``only``-selected) generator, write ``.npz`` + MANIFEST.

    Returns the manifest dict for the caller to print/inspect.
    """
    _install_source_paths(source_a, source_b)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = only or list(GENERATORS)
    manifest: dict = {
        "generated_at": None,  # filled below, after we know how long it took
        "generator_sha256": _sha256_of(Path(__file__)),
        "source_a": str(source_a),
        "source_b": str(source_b),
        "versions": _package_versions(),
        "seed": 0,
        "fixtures": {},
    }
    try:
        urdf_path, urdf_sha = _urdf_sha256()
        manifest["panda_urdf_path"] = urdf_path
        manifest["panda_urdf_sha256"] = urdf_sha
    except Exception as exc:  # pragma: no cover - environment-dependent
        manifest["panda_urdf_error"] = str(exc)

    t0 = time.time()
    sizes = {}
    for name in names:
        gen = GENERATORS[name]
        data, qualnames = gen(source_a, source_b)
        npz_path = out_dir / f"{name}.npz"
        np.savez(npz_path, **data)
        size = npz_path.stat().st_size
        sizes[name] = size
        source_sha256 = {}
        for tagged_rel in FIXTURE_SOURCE_FILES.get(name, ()):
            root_tag, rel = tagged_rel.split("/", 1)
            root = source_a if root_tag == "A" else source_b
            source_sha256[tagged_rel] = _sha256_of(root / rel)
        manifest["fixtures"][name] = {
            "file": npz_path.name,
            "size_bytes": size,
            "n_arrays": len(data),
            "source_qualnames": qualnames,
            "source_sha256": source_sha256,
            "rng_seeds": FIXTURE_SEEDS.get(name, ""),
        }
        print(f"  wrote {npz_path.name}: {len(data)} arrays, {size:,} bytes")

    manifest["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["wall_time_s"] = round(time.time() - t0, 2)
    manifest["total_bytes"] = sum(sizes.values())
    manifest["budget_bytes"] = BUDGET_BYTES

    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if only is None:
        # only assert the whole-suite budget when generating everything --
        # a `--only` partial run isn't meant to prove the aggregate.
        assert manifest["total_bytes"] < BUDGET_BYTES, (
            f"golden fixtures total {manifest['total_bytes']:,} bytes, "
            f"over the {BUDGET_BYTES:,}-byte budget. Trim a generator before "
            f"committing (see BUDGET_BYTES in gen_golden.py)."
        )

    return manifest


# --------------------------------------------------------------------------
# --diff mode: legacy (golden) vs new ($R/src/kinescore), degrades gracefully
# --------------------------------------------------------------------------

#: Fixture name -> callable that takes (golden npz dict) and returns a
#: dict[str, np.ndarray] computed from CURRENT `kinescore` code, or raises
#: (any exception is caught by `run_diff` and reported as "unavailable").
#: Empty until another agent's `src/kinescore` code exists to call -- every
#: robots/heads/reference submodule is a 0-byte stub as of this writing, so
#: every fixture is expected to report unavailable right now. That is the
#: intended degrade-gracefully behaviour, not a bug in this script.
_DIFF_ADAPTERS: dict[str, Callable[[dict], dict]] = {}


def run_diff(out_dir: Path) -> int:
    """For each fixture with a registered adapter, print a legacy/new table.

    Returns 0 always (this is a report, not a gate) unless golden fixtures
    are simply missing, which is a usage error.
    """
    manifest_path = out_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        print(f"error: {manifest_path} not found -- run gen_golden.py without "
              f"--diff first", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text())
    for name in manifest["fixtures"]:
        npz_path = out_dir / f"{name}.npz"
        golden = dict(np.load(npz_path, allow_pickle=False))
        adapter = _DIFF_ADAPTERS.get(name)
        if adapter is None:
            print(f"{name}: SKIP (no --diff adapter registered yet -- "
                  f"src/kinescore has no landed implementation to compare)")
            continue
        try:
            new = adapter(golden)
        except Exception as exc:
            print(f"{name}: SKIP (adapter raised {type(exc).__name__}: {exc})")
            continue
        print(f"{name}:")
        print(f"  {'key':<40} {'legacy':>14} {'new':>14} {'abs':>12} {'rel':>10}")
        for key in sorted(set(golden) & set(new)):
            g, n = float(np.asarray(golden[key]).mean()), float(np.asarray(new[key]).mean())
            abs_diff = abs(g - n)
            rel_diff = abs_diff / (abs(g) + 1e-12)
            print(f"  {key:<40} {g:>14.6g} {n:>14.6g} {abs_diff:>12.3g} {rel_diff:>10.3g}")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source-a", required=True, type=Path,
                    help="path to the Marionette-ciasc checkout")
    p.add_argument("--source-b", required=True, type=Path,
                    help="path to the Marionette-fkjepa checkout")
    p.add_argument("--out-dir", default=Path("tests/golden"), type=Path)
    p.add_argument("--only", nargs="*", choices=list(GENERATORS),
                    help="regenerate only these fixtures (default: all)")
    p.add_argument("--diff", action="store_true",
                    help="instead of generating, diff existing golden "
                         "fixtures against src/kinescore (degrades "
                         "gracefully per-fixture; see _DIFF_ADAPTERS)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if not args.source_a.is_dir():
        print(f"error: --source-a {args.source_a} is not a directory", file=sys.stderr)
        return 2
    if not args.source_b.is_dir():
        print(f"error: --source-b {args.source_b} is not a directory", file=sys.stderr)
        return 2

    if args.diff:
        return run_diff(args.out_dir)

    manifest = generate_all(args.source_a, args.source_b, args.out_dir, args.only)
    print(f"\n{len(manifest['fixtures'])} fixtures, "
          f"{manifest['total_bytes']:,} / {manifest['budget_bytes']:,} bytes "
          f"({manifest['wall_time_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
