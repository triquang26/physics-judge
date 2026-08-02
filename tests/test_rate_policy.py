"""Rate policy: the ``RATE_FREE`` suite, ``verify_manifest``'s ``dt`` check,
``core/resample.py``, and ``core/scorer.py::Scorer``'s ``rate_policy`` wiring.

See ``docs/BENCHMARKING.md`` for the full argument this file enforces:
comparing two clips scored at different frame rates is only valid through one
of three code paths (paired-within-episode, the rate-free suite, or explicit
opt-in resampling), and each path has a specific, checkable contract. This
file is the contract check for the pieces this change owns: the metric-level
``paired``/``rate_free`` machinery and ``core/resample.py``. The fourth layer
(re-encoding the real anchor to the generator's rate) is ported separately
and is out of scope here -- see ``docs/BENCHMARKING.md``'s layer 2.
"""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest
import torch
from _fake_robot import FakeRobot

import kinescore.metrics  # noqa: F401  (side effect: populates the registry)
from kinescore.core.clip import ClipSpec, ViewLayout
from kinescore.core.metric import MetricContext, get_metric
from kinescore.core.reader import Readout
from kinescore.core.resample import (
    ResamplePlan,
    UpsampleRefusedError,
    parse_rate_policy,
    plan_resample,
    resample_clip,
    resample_readout,
    resample_series,
)
from kinescore.core.scorer import Scorer
from kinescore.metrics.suites import INVARIANT_V1, RATE_FREE

# ===========================================================================
# 1. RATE_FREE suite membership: derived from the registry, not hand-copied
# ===========================================================================

#: The set the design doc names explicitly. This is a REGRESSION check, not
#: the source of truth -- the source of truth is "dt_exponent == 0, read from
#: the registry" (see the parametrized test below), which is what actually
#: guards against a future metric silently joining or leaving this suite.
_EXPECTED_RATE_FREE_KEYS = frozenset({
    "rigidity_residual_mm", "rigidity_wobble_mm",
    "limit_violation_frac", "limit_excess_rad", "limit_headroom_rad",
    "penetration_mm", "self_collision_frac", "com_margin_m",
    "log_dimensionless_jerk",
})


class TestRateFreeSuiteMembership:
    def test_matches_the_expected_key_set(self):
        assert set(RATE_FREE.output_keys) == _EXPECTED_RATE_FREE_KEYS

    @pytest.mark.parametrize("key", sorted(_EXPECTED_RATE_FREE_KEYS))
    def test_every_member_has_dt_exponent_zero_per_the_live_registry(self, key):
        # The membership guarantee: read straight off the registered metric's
        # spec, not off any list (including the one two lines above) -- a
        # metric added to RATE_FREE by hand with the wrong exponent fails
        # exactly here.
        assert get_metric(key).spec.dt_exponent == 0

    def test_no_member_has_a_non_zero_or_none_dt_exponent(self):
        for key in RATE_FREE.output_keys:
            spec = get_metric(key).spec
            assert spec.dt_exponent == 0, (
                f"{key}: RATE_FREE membership requires dt_exponent==0, "
                f"got {spec.dt_exponent!r}")

    def test_sparc_is_excluded_despite_being_scale_free(self):
        # "scale-free" (page) == amplitude/duration-invariant, NOT frame-rate
        # invariant -- sparc's own spec says dt_exponent=None (fixed 10 Hz
        # cutoff), which is exactly why it must not be in this suite. See
        # docs/BENCHMARKING.md's dedicated section on this distinction.
        assert "sparc" not in RATE_FREE.output_keys
        assert get_metric("sparc").spec.dt_exponent is None

    def test_log_dimensionless_jerk_is_included(self):
        assert "log_dimensionless_jerk" in RATE_FREE.output_keys
        assert get_metric("log_dimensionless_jerk").spec.dt_exponent == 0

    def test_dt_dependent_metrics_are_excluded(self):
        for key in ("mean_speed_mps", "mean_accel_mps2", "mean_jerk_mps3",
                   "mean_angvel_radps", "mean_angacc_radps2",
                   "kinetic_energy_tstd", "momentum_dp_mean",
                   "mean_qdot_radps", "mean_qddot_radps2", "effort_proxy"):
            assert key not in RATE_FREE.output_keys

    def test_none_exponent_threshold_metrics_are_excluded(self):
        # Threshold-against-a-fixed-constant metrics: not comparable across
        # rates even though some are spatial-flavoured; dt_exponent=None
        # means "excluded from the power-law claim entirely", not "0".
        for key in ("accel_violation_frac", "vel_violation_frac",
                   "no_teleport_frac", "total_energy_tstd"):
            assert key not in RATE_FREE.output_keys

    def test_legacy_all_bone_set_variants_are_excluded(self):
        # rigidity_residual_all_mm / rigidity_wobble_all_mm also have
        # dt_exponent=0 in the raw registry, but they exist only to
        # reproduce legacy gripper-contaminated numbers (docs/METRICS.md:
        # "never register this in a new suite meant to score grasping
        # clips") -- RATE_FREE filters through INVARIANT_V1's own declared
        # term set (which already excludes them), not the raw registry, so
        # they must not leak in here either.
        assert "rigidity_residual_all_mm" not in RATE_FREE.output_keys
        assert "rigidity_wobble_all_mm" not in RATE_FREE.output_keys
        assert get_metric("rigidity_residual_all_mm").spec.dt_exponent == 0

    def test_has_no_composite_invariant_keys(self):
        # RATE_FREE reports every member individually -- no PIS-style
        # aggregate for a cross-rate suite.
        assert RATE_FREE.invariant_keys == ()

    def test_suite_id_differs_from_invariant_v1(self):
        assert RATE_FREE.suite_id != INVARIANT_V1.suite_id


class TestInvariantV1Unaffected:
    """Adding RATE_FREE must not touch INVARIANT_V1 -- golden-fixture safety."""

    def test_invariant_v1_output_key_count_unchanged(self):
        # 26, not the registry's full 28 -- INVARIANT_V1's own term set
        # (_ALL_METRIC_KEYS) excludes the two legacy "_all_mm" bone-set
        # variants (see RATE_FREE's docstring in metrics/suites.py).
        assert len(INVARIANT_V1.output_keys) == 26

    def test_invariant_v1_invariant_keys_unchanged(self):
        assert INVARIANT_V1.invariant_keys == (
            "rigidity_residual_mm", "rigidity_wobble_mm", "mean_jerk_mps3",
            "accel_violation_frac", "mean_angacc_radps2", "total_energy_tstd",
            "momentum_dp_mean", "limit_violation_frac", "vel_violation_frac",
            "effort_proxy",
        )

    def test_invariant_v1_name_unchanged(self):
        assert INVARIANT_V1.name == "invariant_v1"


# ===========================================================================
# 2. parse_rate_policy
# ===========================================================================

class TestParseRatePolicy:
    def test_paired(self):
        p = parse_rate_policy("paired")
        assert p.kind == "paired"
        assert p.target_fps is None

    def test_rate_free(self):
        p = parse_rate_policy("rate_free")
        assert p.kind == "rate_free"
        assert p.target_fps is None

    def test_resample_parses_hz(self):
        p = parse_rate_policy("resample:10")
        assert p.kind == "resample"
        assert p.target_fps == pytest.approx(10.0)
        assert p.allow_upsample is False

    def test_resample_parses_float_hz(self):
        p = parse_rate_policy("resample:16.5")
        assert p.target_fps == pytest.approx(16.5)

    def test_resample_forwards_allow_upsample(self):
        p = parse_rate_policy("resample:30", allow_upsample=True)
        assert p.allow_upsample is True

    def test_resample_rejects_non_numeric_hz(self):
        with pytest.raises(ValueError, match="not a number"):
            parse_rate_policy("resample:fast")

    def test_resample_rejects_nonpositive_hz(self):
        with pytest.raises(ValueError):
            parse_rate_policy("resample:0")
        with pytest.raises(ValueError):
            parse_rate_policy("resample:-5")

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError, match="unknown rate policy"):
            parse_rate_policy("bogus")


# ===========================================================================
# 3. Scorer(rate_policy=...) construction-time validation
# ===========================================================================

def _fake_reader(robot_name: str = "fake_robot") -> SimpleNamespace:
    return SimpleNamespace(robot_name=robot_name, limit_semantics="raw_rad",
                           view_layout=ViewLayout(), reader_id="fake/reader")


class TestScorerRatePolicy:
    def test_paired_is_the_default_and_requires_no_validation(self):
        robot = FakeRobot()
        suite = INVARIANT_V1  # full, rate-dependent suite
        scorer = Scorer(robot, _fake_reader(), suite)
        assert scorer.rate_policy.kind == "paired"

    def test_rate_free_rejects_a_suite_with_dt_dependent_metrics(self):
        robot = FakeRobot()
        with pytest.raises(ValueError, match="dt_exponent==0"):
            Scorer(robot, _fake_reader(), INVARIANT_V1, rate_policy="rate_free")

    def test_rate_free_accepts_the_rate_free_suite(self):
        robot = FakeRobot()
        scorer = Scorer(robot, _fake_reader(), RATE_FREE, rate_policy="rate_free")
        assert scorer.rate_policy.kind == "rate_free"

    def test_resample_policy_is_parsed_and_stored(self):
        robot = FakeRobot()
        scorer = Scorer(robot, _fake_reader(), RATE_FREE,
                        rate_policy="resample:10")
        assert scorer.rate_policy.kind == "resample"
        assert scorer.rate_policy.target_fps == pytest.approx(10.0)

    def test_unknown_rate_policy_string_raises_at_construction(self):
        robot = FakeRobot()
        with pytest.raises(ValueError):
            Scorer(robot, _fake_reader(), RATE_FREE, rate_policy="nonsense")


# ===========================================================================
# 4. plan_resample / resample_clip -- ClipSpec-level rate matching
# ===========================================================================

def _native_clip(fps=20.0, n_frames=41) -> ClipSpec:
    return ClipSpec.from_fps(path="/tmp/native.mp4", fps=fps, n_frames=n_frames,
                             width=64, height=64, dt_source="synthetic")


class TestPlanResample:
    def test_noop_when_target_matches_native(self):
        clip = _native_clip()
        plan = plan_resample(clip, 20.0)
        assert plan.method == "noop"
        assert plan.clip is clip

    def test_integer_decimation_delegates_to_subsample(self):
        clip = _native_clip(fps=20.0, n_frames=41)
        plan = plan_resample(clip, 10.0)
        assert plan.method == "decimate"
        assert plan.stride == 2
        expected = clip.subsample(2)
        assert plan.clip.fps == pytest.approx(expected.fps)
        assert plan.clip.dt == pytest.approx(expected.dt)
        assert plan.clip.n_frames == expected.n_frames
        assert plan.clip.stride == expected.stride

    def test_integer_decimation_tags_dt_source_as_resampled(self):
        clip = _native_clip(fps=20.0, n_frames=41)
        plan = plan_resample(clip, 10.0)
        assert plan.clip.dt_source == "resampled"
        assert plan.clip.dt_source != clip.dt_source

    def test_non_integer_ratio_uses_interpolation(self):
        clip = _native_clip(fps=20.0, n_frames=41)
        plan = plan_resample(clip, 7.0)
        assert plan.method == "interpolate"
        assert plan.stride is None
        assert plan.clip.fps == pytest.approx(7.0)
        assert plan.clip.dt == pytest.approx(1.0 / 7.0)
        assert plan.clip.dt_source == "resampled"

    def test_interpolation_never_extrapolates_past_native_duration(self):
        clip = _native_clip(fps=20.0, n_frames=41)  # duration = 2.0s
        plan = plan_resample(clip, 7.0)
        last_t = (plan.clip.n_frames - 1) * plan.clip.dt
        duration = (clip.n_frames - 1) * clip.dt
        assert last_t <= duration + 1e-9

    def test_upsample_refused_by_default(self):
        clip = _native_clip(fps=10.0, n_frames=21)
        with pytest.raises(UpsampleRefusedError):
            plan_resample(clip, 20.0)

    def test_upsample_allowed_with_flag(self):
        clip = _native_clip(fps=10.0, n_frames=21)
        plan = plan_resample(clip, 20.0, allow_upsample=True)
        assert plan.clip.fps == pytest.approx(20.0)
        assert plan.clip.dt_source == "resampled"

    def test_resample_clip_returns_just_the_spec(self):
        clip = _native_clip(fps=20.0, n_frames=41)
        new_clip = resample_clip(clip, 10.0)
        assert isinstance(new_clip, ClipSpec)
        assert new_clip.fps == pytest.approx(10.0)


# ===========================================================================
# 5. resample_series -- the PCHIP trajectory interpolant
# ===========================================================================

class TestResampleSeries:
    def test_recovers_native_samples_on_an_aligned_grid(self):
        # k=2 decimation: every target timestamp coincides exactly with a
        # native one, so PCHIP (which interpolates exactly through its own
        # nodes) should reproduce x[:, ::2] near machine precision.
        t = torch.linspace(0, 2 * torch.pi, 41).double()
        x = torch.stack([torch.sin(t), torch.cos(1.7 * t)], dim=-1).unsqueeze(0)
        dt = float(t[1] - t[0])
        got = resample_series(x, dt, dt * 2)
        want = x[:, ::2]
        assert got.shape == want.shape
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)

    def test_rejects_fewer_than_two_frames(self):
        x = torch.zeros(1, 1, 3)
        with pytest.raises(ValueError, match="at least 2 frames"):
            resample_series(x, 0.1, 0.2)

    def test_rejects_nonpositive_native_dt(self):
        x = torch.zeros(1, 5, 3)
        with pytest.raises(ValueError, match="native_dt"):
            resample_series(x, 0.0, 0.1)

    def test_rejects_nonpositive_target_dt(self):
        x = torch.zeros(1, 5, 3)
        with pytest.raises(ValueError, match="target_dt"):
            resample_series(x, 0.1, -0.1)

    def test_output_dtype_and_device_match_input(self):
        x = torch.randn(1, 10, 2, dtype=torch.float32)
        got = resample_series(x, 0.1, 0.15)
        assert got.dtype == torch.float32
        assert got.device == x.device

    def test_never_extrapolates(self):
        x = torch.randn(1, 10, 2)
        native_dt = 0.1
        target_dt = 0.13
        got = resample_series(x, native_dt, target_dt)
        duration = (x.shape[1] - 1) * native_dt
        last_t = (got.shape[1] - 1) * target_dt
        assert last_t <= duration + 1e-9


# ===========================================================================
# 6. resample_readout -- the intended entry point, and the DoD invariance test
# ===========================================================================

def _smooth_q(t: torch.Tensor) -> torch.Tensor:
    """``(N,) -> (N,2)``: well inside FakeRobot's default [-3,3] limits."""
    q1 = 0.5 * torch.sin(1.3 * t) + 0.1 * torch.cos(2.1 * t)
    q2 = 0.4 * torch.cos(0.9 * t) + 0.15 * torch.sin(1.7 * t + 0.3)
    return torch.stack([q1, q2], dim=-1)


def _smooth_P(t: torch.Tensor) -> torch.Tensor:
    """``(N,) -> (N,3,3)``: base/mid/tip keypoints, bones oscillating near 1 m."""
    zero = torch.zeros_like(t)
    base = torch.stack([zero, zero, zero], dim=-1)
    mid = torch.stack([
        1.0 + 0.05 * torch.sin(1.3 * t),
        0.05 * torch.cos(0.9 * t),
        0.05 * torch.sin(1.7 * t + 0.4),
    ], dim=-1)
    tip = mid + torch.stack([
        0.05 * torch.sin(1.1 * t + 0.2),
        1.0 + 0.05 * torch.cos(1.9 * t),
        0.05 * torch.cos(2.3 * t),
    ], dim=-1)
    return torch.stack([base, mid, tip], dim=1)


def _assert_rate_free_close(a: float, b: float, rtol: float, key: str,
                            atol: float = 1e-6) -> None:
    if abs(a) <= atol and abs(b) <= atol:
        return
    rel = abs(a - b) / (abs(a) + atol)
    assert rel < rtol, (
        f"{key}: native={a!r} vs resampled={b!r} differ by {rel:.4f} "
        f"(rtol={rtol})")


@pytest.fixture(scope="module")
def robot() -> FakeRobot:
    return FakeRobot()  # default q_lo/hi=[-3,3], bone lengths [1,1], ee=(2,)


@pytest.fixture(scope="module")
def native():
    n = 41
    t = torch.arange(n, dtype=torch.float64) * 0.05  # 20 Hz, 0..2.0s
    q = _smooth_q(t).unsqueeze(0).float()
    P = _smooth_P(t).unsqueeze(0).float()
    clip = _native_clip(fps=20.0, n_frames=n)
    readout = Readout(q=q, q_raw=q.clone())
    return clip, readout, P


@pytest.fixture(scope="module")
def resampled(native):
    clip, readout, P = native
    new_readout, new_clip = resample_readout(readout, clip, 10.0)
    new_P = resample_series(P, clip.dt, new_clip.dt)
    return new_clip, new_readout, new_P


class TestResampleReadoutRateFreeInvariance:
    """DoD #4: 20 Hz -> 10 Hz leaves the rate-free metrics ~unchanged, and
    the resampled ClipSpec.dt_source records the resample, not a native rate.
    """

    def test_dt_source_records_the_resample_not_a_native_rate(self, native, resampled):
        clip, _, _ = native
        new_clip, _, _ = resampled
        assert new_clip.dt_source == "resampled"
        assert new_clip.dt_source not in ("ffprobe", "fps_arg", "dt_arg", "table")
        assert new_clip.fps == pytest.approx(10.0)
        assert new_clip.dt == pytest.approx(0.1)

    def test_decimation_path_is_used_for_this_exact_ratio(self, native, resampled):
        # 20/10 == 2 exactly -> this must be the ClipSpec.subsample path, not
        # PCHIP -- exercised separately by the non-integer test below.
        clip, _, _ = native
        new_clip, _, _ = resampled
        assert new_clip.n_frames == clip.subsample(2).n_frames
        assert new_clip.stride == clip.subsample(2).stride

    @pytest.mark.parametrize("key", sorted(_EXPECTED_RATE_FREE_KEYS))
    def test_rate_free_metric_is_unchanged_within_tolerance(self, robot, native,
                                                             resampled, key):
        clip, readout, P = native
        new_clip, new_readout, new_P = resampled

        ctx_native = MetricContext(
            dt=clip.dt, P=P, q=readout.q, q_raw=readout.q_raw, robot=robot,
            flags={"limit_semantics": "raw_rad"})
        ctx_resampled = MetricContext(
            dt=new_clip.dt, P=new_P, q=new_readout.q, q_raw=new_readout.q_raw,
            robot=robot, flags={"limit_semantics": "raw_rad"})

        metric = get_metric(key)
        v_native = metric.compute(ctx_native)
        v_resampled = metric.compute(ctx_resampled)

        if not v_native.available or not v_resampled.available:
            # penetration_mm / self_collision_frac / com_margin_m need
            # collider/support-polygon geometry (robot.body_collider_spheres,
            # robot.world_com, robot.support_polygon) that FakeRobot -- a
            # geometry-only test double, see tests/_fake_robot.py -- does not
            # implement; they report NaN+reason (never a fabricated 0.0),
            # exactly like tests/test_metric_registry_conformance.py's own
            # "unavailable on the shared fixture" skip for the same reason.
            pytest.skip(f"{key}: unavailable on FakeRobot "
                       f"(native.reason={v_native.reason!r}, "
                       f"resampled.reason={v_resampled.reason!r})")

        # log_dimensionless_jerk is a finite-difference-based (3rd
        # derivative) quantity -- real truncation error enters under
        # decimation, same reasoning as tests/test_dt_invariance.py's
        # k=2 rtol=0.05. The purely static/geometric members (rigidity,
        # limit_*) have no derivative at all and only differ because the
        # *set* of averaged frames changed, so a tighter bound holds; use
        # one shared, generous bound for all of them to keep this simple
        # and non-flaky.
        _assert_rate_free_close(v_native.value, v_resampled.value,
                                rtol=0.15, key=key)


class TestResampleReadoutAux:
    def test_per_frame_aux_is_resampled_consistently_with_q(self):
        n = 21
        t = torch.arange(n, dtype=torch.float64) * 0.1  # 10 Hz
        q = _smooth_q(t).unsqueeze(0).float()
        aux = torch.linspace(0, 1, n).view(1, n, 1).float()  # e.g. gripper opening
        clip = _native_clip(fps=10.0, n_frames=n)
        readout = Readout(q=q, q_raw=q.clone(), aux=aux)

        new_readout, new_clip = resample_readout(readout, clip, 5.0)  # k=2

        assert new_readout.aux is not None
        assert new_readout.aux.shape[:2] == new_readout.q.shape[:2]
        torch.testing.assert_close(new_readout.aux, aux[:, ::2])
        assert new_clip.n_frames == new_readout.q.shape[1]

    def test_non_per_frame_aux_passes_through_unchanged(self):
        n = 21
        t = torch.arange(n, dtype=torch.float64) * 0.1
        q = _smooth_q(t).unsqueeze(0).float()
        clip = _native_clip(fps=10.0, n_frames=n)
        readout = Readout(q=q, q_raw=q.clone(), aux="not-a-tensor")

        new_readout, _ = resample_readout(readout, clip, 5.0)
        assert new_readout.aux == "not-a-tensor"

    def test_none_aux_stays_none(self):
        n = 21
        t = torch.arange(n, dtype=torch.float64) * 0.1
        q = _smooth_q(t).unsqueeze(0).float()
        clip = _native_clip(fps=10.0, n_frames=n)
        readout = Readout(q=q, q_raw=q.clone(), aux=None)

        new_readout, _ = resample_readout(readout, clip, 5.0)
        assert new_readout.aux is None


class TestResampleReadoutUpsampleRefusal:
    """DoD #5: upsampling is refused unless explicitly allowed."""

    def _readout_and_clip(self, fps=10.0, n=21):
        t = torch.arange(n, dtype=torch.float64) * (1.0 / fps)
        q = _smooth_q(t).unsqueeze(0).float()
        clip = _native_clip(fps=fps, n_frames=n)
        return Readout(q=q, q_raw=q.clone()), clip

    def test_upsample_raises_by_default(self):
        readout, clip = self._readout_and_clip(fps=10.0, n=21)
        with pytest.raises(UpsampleRefusedError):
            resample_readout(readout, clip, 20.0)

    def test_upsample_succeeds_with_allow_upsample(self):
        readout, clip = self._readout_and_clip(fps=10.0, n=21)
        new_readout, new_clip = resample_readout(readout, clip, 20.0,
                                                  allow_upsample=True)
        assert new_clip.fps == pytest.approx(20.0)
        assert new_clip.dt_source == "resampled"
        assert new_readout.q.shape[1] == new_clip.n_frames

    def test_downsample_never_needs_the_flag(self):
        readout, clip = self._readout_and_clip(fps=10.0, n=21)
        # Must not raise.
        resample_readout(readout, clip, 5.0)


class TestResampleReadoutNoiseWarning:
    def test_interpolation_path_warns_about_the_noise_spectrum(self):
        n = 41
        t = torch.arange(n, dtype=torch.float64) * 0.05  # 20 Hz
        q = _smooth_q(t).unsqueeze(0).float()
        clip = _native_clip(fps=20.0, n_frames=n)
        readout = Readout(q=q, q_raw=q.clone())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resample_readout(readout, clip, 7.0)  # non-integer -> interpolate
        assert any("noise" in str(w.message).lower() for w in caught)

    def test_decimation_path_does_not_warn(self):
        n = 41
        t = torch.arange(n, dtype=torch.float64) * 0.05
        q = _smooth_q(t).unsqueeze(0).float()
        clip = _native_clip(fps=20.0, n_frames=n)
        readout = Readout(q=q, q_raw=q.clone())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resample_readout(readout, clip, 10.0)  # exact k=2
        assert not any("noise" in str(w.message).lower() for w in caught)

    def test_noop_does_not_warn(self):
        n = 21
        t = torch.arange(n, dtype=torch.float64) * 0.1
        q = _smooth_q(t).unsqueeze(0).float()
        clip = _native_clip(fps=10.0, n_frames=n)
        readout = Readout(q=q, q_raw=q.clone())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resample_readout(readout, clip, 10.0)
        assert len(caught) == 0


# ===========================================================================
# 7. ResamplePlan / RatePolicy dataclass sanity
# ===========================================================================

class TestRatePolicyDataclass:
    def test_resample_kind_requires_target_fps(self):
        with pytest.raises(ValueError):
            from kinescore.core.resample import RatePolicy
            RatePolicy(kind="resample")

    def test_non_resample_kind_rejects_target_fps(self):
        with pytest.raises(ValueError):
            from kinescore.core.resample import RatePolicy
            RatePolicy(kind="paired", target_fps=10.0)


class TestResamplePlanDataclass:
    def test_fields(self):
        clip = _native_clip()
        plan = ResamplePlan(clip=clip, method="noop")
        assert plan.method == "noop"
        assert plan.stride is None


# ===========================================================================
# 8. Scorer.score_readout / Scorer.score end-to-end -- the actual wiring
# ===========================================================================

def _trivial_fk(q: torch.Tensor, aux):
    """``(B,T,2) -> P (B,T,3,3), R (B,T,3,3,3)`` -- just enough FK to run
    the metric layer on FakeRobot's 3-keypoint (base/mid/tip) geometry.
    """
    b, t = q.shape[0], q.shape[1]
    base = torch.zeros(b, t, 3)
    mid = torch.stack([1.0 + 0.1 * q[..., 0], 0.1 * q[..., 1],
                       torch.zeros(b, t)], dim=-1)
    tip = mid + torch.stack([0.1 * q[..., 1], 1.0 + 0.1 * q[..., 0],
                             torch.zeros(b, t)], dim=-1)
    P = torch.stack([base, mid, tip], dim=2)
    R = torch.eye(3).view(1, 1, 1, 3, 3).expand(b, t, 3, 3, 3).clone()
    return P, R


class TestScorerScoreReadoutIntegration:
    """End-to-end: Scorer actually resamples before FK, and the returned
    ScoredClip's provenance reflects it -- not just the isolated resample.py
    functions or the construction-time rate_policy validation tested above.
    """

    def _reader(self, robot_name="fake_robot", read=None):
        return SimpleNamespace(robot_name=robot_name, limit_semantics="raw_rad",
                               view_layout=ViewLayout(), reader_id="fake/reader",
                               read=read)

    def test_paired_policy_scores_the_native_clip_unchanged(self):
        robot = FakeRobot(fk_fn=_trivial_fk)
        scorer = Scorer(robot, self._reader(), RATE_FREE)  # default: paired
        n = 21
        t = torch.arange(n, dtype=torch.float64) * 0.1
        q = _smooth_q(t).unsqueeze(0).float()
        clip = _native_clip(fps=10.0, n_frames=n)
        readout = Readout(q=q, q_raw=q.clone())

        scored = scorer.score_readout(readout, clip)

        assert scored.clip.fps == pytest.approx(10.0)
        assert scored.clip.dt_source == clip.dt_source  # untouched
        assert scored.n_frames_scored == n
        assert scored.result.suite_id == RATE_FREE.suite_id

    def test_resample_policy_resamples_before_scoring(self):
        robot = FakeRobot(fk_fn=_trivial_fk)
        scorer = Scorer(robot, self._reader(), RATE_FREE, rate_policy="resample:10")
        n = 41
        t = torch.arange(n, dtype=torch.float64) * 0.05  # 20 Hz native
        q = _smooth_q(t).unsqueeze(0).float()
        clip = _native_clip(fps=20.0, n_frames=n)
        readout = Readout(q=q, q_raw=q.clone())

        scored = scorer.score_readout(readout, clip)

        # The returned clip is the RESAMPLED one -- 10 fps, tagged, fewer
        # frames -- never the native 20 fps clip that was passed in.
        assert scored.clip.fps == pytest.approx(10.0)
        assert scored.clip.dt_source == "resampled"
        assert scored.clip.n_frames == clip.subsample(2).n_frames
        assert scored.n_frames_scored == scored.clip.n_frames
        # Scoring actually ran on the resampled trajectory (not skipped):
        assert scored.result.values["limit_headroom_rad"].available

    def test_resample_policy_refuses_upsample_at_score_time(self):
        robot = FakeRobot(fk_fn=_trivial_fk)
        scorer = Scorer(robot, self._reader(), RATE_FREE, rate_policy="resample:20")
        n = 21
        t = torch.arange(n, dtype=torch.float64) * 0.1  # 10 Hz native
        q = _smooth_q(t).unsqueeze(0).float()
        clip = _native_clip(fps=10.0, n_frames=n)
        readout = Readout(q=q, q_raw=q.clone())

        with pytest.raises(UpsampleRefusedError):
            scorer.score_readout(readout, clip)

    def test_score_runs_check_layout_then_reader_then_score_readout(self):
        robot = FakeRobot(fk_fn=_trivial_fk)
        n = 21
        t = torch.arange(n, dtype=torch.float64) * 0.1
        q = _smooth_q(t).unsqueeze(0).float()
        fixed_readout = Readout(q=q, q_raw=q.clone())
        reader = self._reader(read=lambda frames: fixed_readout)
        scorer = Scorer(robot, reader, RATE_FREE)
        clip = _native_clip(fps=10.0, n_frames=n)

        scored = scorer.score(torch.zeros(1, n, 3, 8, 8), clip)

        assert scored.n_frames_scored == n
        assert scored.clip.dt_source == clip.dt_source

    def test_score_rejects_a_view_layout_mismatch(self):
        robot = FakeRobot(fk_fn=_trivial_fk)
        mismatched_reader = SimpleNamespace(
            robot_name="fake_robot", limit_semantics="raw_rad",
            view_layout=ViewLayout(n_views=3), reader_id="fake/reader",
            read=lambda frames: None)
        scorer = Scorer(robot, mismatched_reader, RATE_FREE)
        clip = _native_clip(fps=10.0, n_frames=21)  # ViewLayout(n_views=1)

        with pytest.raises(ValueError, match="defect D4"):
            scorer.score(torch.zeros(1), clip)
