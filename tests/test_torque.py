"""``torque_frac_rated`` -- static-gravity ground truth, ordering pinning,
availability discipline, and (when reachable) a numeric regression against
``Marionette-fkjepa``'s recorded ``torque_summary.json``.

Default tier is CPU-only, network-free and asset-free: the static-gravity
closed-form check, the ``clip_peak_pct`` ordering pin, the
:mod:`kinescore.robots.inertia` NaN-not-fabricated checks (synthetic URDFs
built inline), and the registry/availability-gating checks. The one test
that reproduces the recorded real-numbers-from-the-paper
(``test_regression_against_recorded_torque_summary``) needs a real GR-1 URDF
(``$KINESCORE_ASSETS``) and a real ``Marionette-fkjepa`` checkout's cached
pose trajectories (``$KINESCORE_FKJEPA_ROOT``) -- it self-skips with an
actionable message when either is unset, the same pattern
``tests/test_rigidity_gripper_contamination.py`` / ``tests/test_fk_rest_pose.py``
use for the Franka URDF.
"""
from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from _fake_robot import FakeRobot

from kinescore.core.metric import MetricContext
from kinescore.metrics.torque import (
    TorqueFracRated,
    clip_peak_pct,
    joint_torques,
    smooth_frames,
)
from kinescore.robots.inertia import (
    ChainDynamics,
    build_chain_dynamics,
    parse_link_inertials,
)

FKJEPA_ROOT_ENV = "KINESCORE_FKJEPA_ROOT"


# ===========================================================================
# static-gravity ground truth (requirement 2)
# ===========================================================================

def test_static_gravity_hold_matches_closed_form():
    """A stationary single-mass, single-joint chain's torque equals the
    textbook gravity-compensation formula, computed independently by hand.

    Mirrors ``54_torque_feasibility.py``'s own sanity check (source
    ``main()``:238-240: "static gravity torque only (a=w=al=0) -> holding
    torque", printed there as "<<limits" but not asserted against a
    closed-form value). Here the closed form IS asserted: a point mass ``m``
    offset by ``r`` along the chain's local x-axis, held motionless, with the
    joint axis along y, needs a holding torque of exactly
    ``tau = axis . (r x F) = -(m*g*r_x)`` (F = m*(0-g) = (0,0,m*g) points
    +z; r x F = (0, -r_x*m*g, 0); dotted with axis=(0,1,0) gives -r_x*m*g).
    With no rotation, zero angular velocity/acceleration, and a perfectly
    static (constant) pose, every finite-difference term in
    :func:`joint_torques` (``a``, ``w``, ``al``) is EXACTLY zero at EVERY
    frame -- not approximately, since central-differencing a constant
    sequence has no truncation error, and (unlike a genuinely moving clip)
    the endpoint frames are not merely "zeroed by construction" here, they
    are correctly zero because the true motion really is static there too.
    So this is an exact equality check across the whole clip, not just the
    interior, and not a tolerance-loosened one. (See
    :func:`kinescore.metrics.torque.joint_torques`'s own docstring for why
    endpoints do NOT generally match the interior on a non-static clip --
    the gravity term in ``F=m(a-g)`` survives even where ``a`` was never
    actually computed.)
    """
    m, r_x, g = 2.0, 0.3, 9.81
    chain = ChainDynamics(
        joint_names=("j0",), links=("link0",),
        axis=np.array([[0.0, 1.0, 0.0]]),
        effort=np.array([100.0]),
        mass=np.array([m]),
        com=np.array([[r_x, 0.0, 0.0]]),
        inertia=np.zeros((1, 3, 3)),
        n_joint=1,
    )
    T = 5
    frames = torch.eye(4, dtype=torch.float64).view(1, 1, 1, 4, 4).repeat(1, T, 1, 1, 1)

    tau = joint_torques(frames, chain, dt=0.1, g=g)

    expected = -(m * g * r_x)
    assert torch.allclose(tau[0, :, 0], torch.full((T,), expected, dtype=torch.float64),
                          atol=1e-9)


def test_static_gravity_hold_independent_of_dt():
    """The static-hold torque above does not depend on which ``dt`` is passed.

    A second, cheap way the "independently computed" requirement is
    satisfied: the closed-form answer has no ``dt`` in it at all, so if
    :func:`joint_torques` is correct, changing ``dt`` on a truly static
    sequence must not move the answer (every derivative it feeds is zero
    regardless of what it's divided by).
    """
    chain = ChainDynamics(
        joint_names=("j0",), links=("link0",),
        axis=np.array([[0.0, 1.0, 0.0]]),
        effort=np.array([100.0]), mass=np.array([1.5]),
        com=np.array([[0.2, 0.0, 0.0]]), inertia=np.zeros((1, 3, 3)), n_joint=1)
    frames = torch.eye(4, dtype=torch.float64).view(1, 1, 1, 4, 4).repeat(1, 6, 1, 1, 1)
    tau_a = joint_torques(frames, chain, dt=0.05)
    tau_b = joint_torques(frames, chain, dt=0.3)
    assert torch.allclose(tau_a[:, 1:-1], tau_b[:, 1:-1], atol=1e-9)


# ===========================================================================
# clip_peak_pct: per-frame envelope, THEN percentile (requirement 3)
# ===========================================================================

def test_clip_peak_pct_pins_envelope_then_percentile_order():
    """The canonical statistic is max-over-joints-per-frame, then a
    percentile over TIME -- not a percentile per joint over time, then a
    max over joints. A synthetic case where the two orders give visibly
    different numbers, so the ordering is actually pinned (not just
    documented).

    Two joints, 200 frames, baseline ratio 0.1. Joint A spikes to 0.99 at 3
    frames, joint B spikes to 0.99 at 3 DIFFERENT frames (never together).
    The per-frame envelope (max over joints) is therefore >=0.99 at 6/200 =
    3% of frames -- comfortably above the 2%-from-the-top p98 threshold, so
    the correct (envelope-then-percentile) statistic is ~0.99. But EACH
    joint individually only spikes at 3/200 = 1.5% of ITS OWN frames --
    below the 2% cutoff -- so that joint's own p98 is still 0.1, and the
    naive (percentile-per-joint-then-max) statistic is ~0.1. The two must
    disagree by roughly an order of magnitude for this test to mean
    anything; asserted with headroom below.
    """
    T = 200
    a = np.full(T, 0.1)
    b = np.full(T, 0.1)
    idx_a, idx_b = [10, 50, 90], [20, 60, 100]
    a[idx_a] = 0.99
    b[idx_b] = 0.99
    ratio = torch.tensor(np.stack([a, b], axis=-1))[None]  # (1,T,2)

    correct = clip_peak_pct(ratio, 98.0).item()
    naive = max(float(np.percentile(a, 98)), float(np.percentile(b, 98)))

    assert correct == pytest.approx(0.99, abs=1e-6)
    assert naive == pytest.approx(0.1, abs=1e-6)
    assert correct - naive > 0.5, (
        f"ordering must be pinned by a large margin: correct={correct} "
        f"naive={naive}")


def test_clip_peak_pct_constant_envelope_is_that_constant():
    """Degenerate sanity check: a flat envelope's p-anything is the constant."""
    ratio = torch.full((2, 10, 3), 0.42, dtype=torch.float64)
    out = clip_peak_pct(ratio, 98.0)
    assert out.shape == (2,)
    assert torch.allclose(out, torch.full((2,), 0.42, dtype=torch.float64))


# ===========================================================================
# robots.inertia: NaN, never a fabricated fallback (the D11 fix)
# ===========================================================================

_URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="synthetic_arm">
  <link name="base_link"/>
  <link name="link_a">
    <inertial>
      <origin xyz="0.1 0 0" rpy="0 0 0"/>
      <mass value="1.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <link name="link_b"/>
  <joint name="joint_a" type="revolute">
    <parent link="base_link"/>
    <child link="link_a"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="{effort_b}" velocity="1"/>
  </joint>
  <joint name="joint_b" type="revolute">
    <parent link="link_a"/>
    <child link="link_b"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="10.0" velocity="1"/>
  </joint>
</robot>
"""


def _write_urdf(tmp_path: Path, *, effort_b: str = "") -> Path:
    # NOTE: joint_a's own limit's effort is templated (see callers); joint_b
    # always has a real effort. link_b (joint_b's child) never has <inertial>.
    text = _URDF_TEMPLATE.format(effort_b=effort_b)
    path = tmp_path / "synthetic_arm.urdf"
    path.write_text(text)
    return path


def test_parse_link_inertials_omits_massless_links_not_zero_fills(tmp_path):
    urdf = _write_urdf(tmp_path, effort_b="5.0")
    inertials = parse_link_inertials(urdf)
    assert "link_a" in inertials
    assert inertials["link_a"].mass == pytest.approx(1.5)
    assert "link_b" not in inertials  # no <inertial> at all -> omitted, not mass=0
    assert "base_link" not in inertials


def test_build_chain_dynamics_missing_effort_is_nan_not_fabricated_infinity(tmp_path):
    """A joint whose <limit> omits ``effort`` gets ``NaN``, matching this
    package's "NaN with reason, never a fabricated benign value" discipline
    -- NOT the source's ``e = 1e9`` fallback (see robots/inertia.py's module
    docstring for why that fallback is itself a defect: it makes "never
    declared" silently read as "practically unconstrained").
    """
    urdf = _write_urdf(tmp_path, effort_b="")  # joint_a's <limit> has no effort=
    chain = build_chain_dynamics(str(urdf), ("joint_a", "joint_b"))
    assert math.isnan(chain.effort[0])          # joint_a: no effort attribute
    assert chain.effort[1] == pytest.approx(10.0)  # joint_b: declared normally


def test_build_chain_dynamics_missing_inertial_is_nan_not_zero(tmp_path):
    """A joint's child link with no <inertial> gets ``mass=NaN``, not
    ``mass=0.0`` -- the source's ``or (0.0, zeros(3), zeros(3,3))`` fallback
    would silently make that link contribute nothing to the dynamics sum
    while looking like "measured and found massless".
    """
    urdf = _write_urdf(tmp_path, effort_b="5.0")
    chain = build_chain_dynamics(str(urdf), ("joint_a", "joint_b"))
    assert chain.links == ("link_a", "link_b")
    assert not math.isnan(chain.mass[0])   # link_a: has <inertial>
    assert math.isnan(chain.mass[1])       # link_b: no <inertial> at all


def test_build_chain_dynamics_unknown_joint_raises(tmp_path):
    urdf = _write_urdf(tmp_path, effort_b="5.0")
    with pytest.raises(ValueError, match="not found"):
        build_chain_dynamics(str(urdf), ("nonexistent_joint",))


# ===========================================================================
# TorqueFracRated: availability discipline (never 0.0, always NaN+reason)
# ===========================================================================

def test_unsupported_robot_is_nan_with_reason_not_zero():
    metric = TorqueFracRated()
    robot = FakeRobot(name="some_other_robot")
    ctx = MetricContext(dt=0.1, q=torch.zeros(1, 5, 2), robot=robot)
    val = metric.compute(ctx)
    assert not val.available
    assert math.isnan(val.value)
    assert val.reason == "unsupported_robot:some_other_robot"


def test_no_robot_is_nan_with_reason():
    metric = TorqueFracRated()
    ctx = MetricContext(dt=0.1, q=torch.zeros(1, 5, 2), robot=None)
    val = metric.compute(ctx)
    assert not val.available
    assert val.reason == "missing_input:robot"


def test_too_few_frames_is_gated_by_min_frames():
    metric = TorqueFracRated()
    assert metric.spec.min_frames == 3
    robot = FakeRobot(name="fourier_gr1")  # name matches, but no .fk -> still unavailable
    ctx = MetricContext(dt=0.1, q=torch.zeros(1, 2, 17), robot=robot)
    val = metric.compute(ctx)
    assert not val.available
    assert val.reason.startswith("too_few_frames:")


def test_gr1_name_without_fk_attribute_is_nan_not_a_crash():
    """A robot that happens to be named 'fourier_gr1' but has no `.fk`
    (e.g. a hand-built test double) must not crash -- it's unavailable."""
    metric = TorqueFracRated()
    robot = FakeRobot(name="fourier_gr1")
    ctx = MetricContext(dt=0.1, q=torch.zeros(1, 5, 17), robot=robot)
    val = metric.compute(ctx)
    assert not val.available
    assert val.reason == "missing_input:fk_link_frames"


def test_metric_registered_with_declared_dt_exponent_none():
    assert TorqueFracRated().spec.dt_exponent is None
    assert TorqueFracRated().spec.key == "torque_frac_rated"
    assert TorqueFracRated().spec.units == "percent"


# ===========================================================================
# smooth_frames: basic sanity (used by TorqueFracRated before differentiation)
# ===========================================================================

def test_smooth_frames_sigma_zero_is_identity():
    q = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
    out = smooth_frames(q, sigma=0.0)
    assert np.array_equal(out, q)


def test_smooth_frames_preserves_shape_and_constant_signal():
    q = np.full((20, 4), 3.5, dtype=np.float32)
    out = smooth_frames(q, sigma=1.0)
    assert out.shape == q.shape
    assert np.allclose(out, 3.5, atol=1e-5)  # smoothing a constant changes nothing


# ===========================================================================
# regression against the recorded torque_summary.json (requirement 4)
# ===========================================================================

def _fkjepa_cache_dir() -> Path | None:
    root = os.environ.get(FKJEPA_ROOT_ENV)
    if not root:
        return None
    d = Path(root) / "outputs_gr1" / "highlow_violation_gallery" / "cache"
    return d if d.is_dir() else None


def _fkjepa_summary_json() -> Path | None:
    root = os.environ.get(FKJEPA_ROOT_ENV)
    if not root:
        return None
    p = Path(root) / "outputs_gr1" / "torque" / "torque_summary.json"
    return p if p.is_file() else None


def test_regression_against_recorded_torque_summary():
    """Reproduce the ``gen_high``/``gen_low`` per-clip torque_frac_rated
    numbers recorded in ``Marionette-fkjepa/outputs_gr1/torque/torque_summary.json``
    (fps=30, sigma=1.0, pct=98) from that repo's own cached ``q17_raw``
    trajectories, via THIS package's ported ``TorqueFracRated``.

    Reproduces: gen_high mean ~12.01%, gen_low mean ~14.24%, worst clip
    ~27.2% (all from the task brief, matching the source repo's own printed
    numbers). The ``real`` teleop numbers (~8.77%) in that same file are
    **not** reproduced here: the real-teleop dataset directory
    (``dataset_meta_info/dino_gr1_v3l_in768_p2``) is empty on every host this
    was ported on (verified: 0 files) -- only the *generated* cache
    (``outputs_gr1/highlow_violation_gallery/cache/*.npz``, which carries
    ``q17_raw`` directly) is reachable, so this test is scoped to what is
    actually checkable rather than asserting a number that cannot be
    independently verified here.

    Self-skips (actionable message) unless both ``$KINESCORE_ASSETS`` (a
    GR-1 URDF tree) and ``$KINESCORE_FKJEPA_ROOT`` (a Marionette-fkjepa
    checkout with its ``outputs_gr1`` cache still on disk) are set --
    neither is available in a fresh clone / default CI, matching every other
    real-URDF/real-checkpoint-gated test in this suite.
    """
    cache_dir = _fkjepa_cache_dir()
    summary_path = _fkjepa_summary_json()
    if cache_dir is None or summary_path is None:
        pytest.skip(
            f"${FKJEPA_ROOT_ENV} not set, or its outputs_gr1 cache/summary "
            f"are missing; cannot reproduce the recorded torque_summary.json "
            f"numbers on this host. Set ${FKJEPA_ROOT_ENV} to a "
            f"Marionette-fkjepa checkout that still has "
            f"outputs_gr1/highlow_violation_gallery/cache/*.npz and "
            f"outputs_gr1/torque/torque_summary.json on disk.")

    from kinescore.paths import MissingPathError
    try:
        from kinescore.robots.gr1.spec import GR1Spec
        robot = GR1Spec()
    except MissingPathError as exc:
        pytest.skip(f"$KINESCORE_ASSETS unavailable: {exc}")

    recorded = json.loads(summary_path.read_text())
    metric = TorqueFracRated(sigma=float(recorded["sigma"]), pct=float(recorded["pct"]))
    fps = float(recorded["fps"])

    files = sorted(glob.glob(str(cache_dir / "*.npz")))
    n_cap = 40
    gen: dict[str, list[float]] = {"high": [], "low": []}
    for f in files:
        base = os.path.basename(f)
        if "__fields" in base:
            continue
        tier = "high" if base.startswith("high__") else (
            "low" if base.startswith("low__") else None)
        if tier is None or len(gen[tier]) >= n_cap:
            continue
        d = np.load(f, allow_pickle=True)
        if "q17_raw" not in d.files:
            continue
        q17 = d["q17_raw"]
        if q17.shape[0] < 8:
            continue
        q = torch.tensor(q17, dtype=torch.float32)[None]
        ctx = MetricContext(dt=1.0 / fps, q=q, robot=robot)
        val = metric.compute(ctx)
        assert val.available, f"{f}: unexpectedly unavailable ({val.reason})"
        gen[tier].append(val.value)

    assert len(gen["high"]) == len(recorded["torque_pct"]["gen_high"])
    assert len(gen["low"]) == len(recorded["torque_pct"]["gen_low"])

    got_high = np.array(gen["high"])
    got_low = np.array(gen["low"])
    want_high = np.array(recorded["torque_pct"]["gen_high"])
    want_low = np.array(recorded["torque_pct"]["gen_low"])

    # per-clip agreement to well within fp32-vs-fp64-accumulation noise
    np.testing.assert_allclose(got_high, want_high, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(got_low, want_low, rtol=1e-3, atol=1e-3)

    assert got_high.mean() == pytest.approx(12.01, abs=0.05)
    assert got_low.mean() == pytest.approx(14.24, abs=0.05)
    worst = float(max(got_high.max(), got_low.max()))
    assert worst == pytest.approx(27.2, abs=0.05)
