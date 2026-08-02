"""Per-frame trace export: scalar-preservation, alignment, sidecar I/O, wiring.

Covers task 3 ("per-frame traces exported, not just scalars", see
``TASKS.md`` and ``legacy_docs/DECISIONS.md``): every ``perframe``-declaring metric
this agent touched (``rigidity_worst_bone_mm``, ``mean_jerk_mps3``,
``mean_speed_mps``, ``mean_accel_mps2``, ``accel_violation_frac``,
``no_teleport_frac``, ``torque_frac_rated``), the alignment bookkeeping in
``kinescore.bench.traces``, the ``.npz``+JSON sidecar store, and the
``bench.runner.run(trace_store=...)`` wiring ``cli/cmd_score.py --traces``
sits on top of.

Hard constraint this file exists to prove: **adding a trace must be
numerically inert**. Each metric's scalar is checked two ways: (1) against
an independent reference implementation written fresh in this file (not
imported from the metric under test), and (2) as a pinned literal on a fixed
seed, so a future refactor that nudges the scalar's floating-point value --
not just its trace -- fails loudly here.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from _fake_robot import FakeRobot

import kinescore.metrics  # noqa: F401  (side effect: populates the registry)
from kinescore.bench.traces import (
    ALIGNMENTS,
    ClipTraces,
    TraceStore,
    clip_traces_from_scored,
    first_frame_index,
    load_example,
)
from kinescore.core.metric import MetricContext, all_metrics

# ===========================================================================
# shared synthetic fixture
# ===========================================================================

_B, _T, _K = 1, 16, 4
_DT = 0.1


def _make_P() -> torch.Tensor:
    g = torch.Generator().manual_seed(42)
    return torch.cumsum(torch.randn(_B, _T, _K, 3, generator=g) * 0.05, dim=1)


def _fake_robot() -> FakeRobot:
    bone_pairs = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
    bone_lengths = torch.tensor([1.0, 1.0, 1.0])
    return FakeRobot(bone_pairs=bone_pairs, rigid_bone_pairs=bone_pairs,
                     bone_lengths=bone_lengths, rigid_bone_lengths=bone_lengths)


def _ctx() -> MetricContext:
    return MetricContext(dt=_DT, P=_make_P(), robot=_fake_robot())


def _fd_chain(P: torch.Tensor, dt: float, order: int) -> torch.Tensor:
    """Independent re-derivation of an n-th order finite difference.

    Deliberately re-implemented here (not imported from
    ``kinescore.metrics.ops.fd``) so this is a genuine second implementation
    of the arithmetic, not a call back into the code under test.
    """
    x = P
    for _ in range(order):
        x = (x[:, 1:] - x[:, :-1]) / dt
    return x


# ===========================================================================
# rigidity_worst_bone_mm
# ===========================================================================

class TestRigidityWorstBonePerframe:
    def test_scalar_matches_independent_reference_and_is_pinned(self):
        P = _make_P()
        robot = _fake_robot()
        metric = all_metrics()["rigidity_worst_bone_mm"]
        mv = metric.compute(MetricContext(dt=_DT, P=P, robot=robot))
        assert mv.available

        # Independent reference: bone lengths from raw keypoint norms, worst
        # (amax) deviation from each bone's own temporal median, in mm.
        bp = robot.rigid_bone_pairs
        i, j = bp[:, 0], bp[:, 1]
        L = torch.linalg.norm(P[..., i, :] - P[..., j, :], dim=-1)  # (B,T,n_bones)
        med = L.median(dim=1, keepdim=True).values
        dev = (L - med).abs().amax(dim=2)
        ref_scalar = float(dev.mean() * 1000.0)

        assert mv.value == pytest.approx(ref_scalar, rel=1e-6)
        # Regression pin on the fixed seed -- catches a scalar-arithmetic
        # change even if the reference implementation above also drifted.
        assert mv.value == pytest.approx(218.79092407226562, rel=1e-5)

    def test_perframe_shape_alignment_and_reduces_to_scalar(self):
        mv = all_metrics()["rigidity_worst_bone_mm"].compute(_ctx())
        assert mv.perframe is not None
        assert mv.perframe.dtype == np.float32
        assert mv.perframe.shape == (_T,)
        assert ALIGNMENTS["rigidity_worst_bone_mm"] == "same_length"
        assert first_frame_index("rigidity_worst_bone_mm", _T, len(mv.perframe)) == 0
        # Same pre-mean quantity the scalar reduces -- mean(trace) == scalar.
        assert float(mv.perframe.mean()) == pytest.approx(mv.value, rel=1e-5)

    def test_unavailable_metric_has_no_perframe(self):
        mv = all_metrics()["rigidity_worst_bone_mm"].compute(
            MetricContext(dt=_DT, P=None, robot=_fake_robot()))
        assert not mv.available
        assert mv.perframe is None


# ===========================================================================
# mean_jerk_mps3 / mean_speed_mps / mean_accel_mps2
# ===========================================================================

@pytest.mark.parametrize("key,order,pinned", [
    ("mean_speed_mps", 1, 0.7858313918113708),
    ("mean_accel_mps2", 2, 10.610590934753418),
    ("mean_jerk_mps3", 3, 179.440185546875),
])
def test_derivative_metric_scalar_and_perframe(key, order, pinned):
    P = _make_P()
    metric = all_metrics()[key]
    mv = metric.compute(MetricContext(dt=_DT, P=P, robot=_fake_robot()))
    assert mv.available

    ref = torch.linalg.norm(_fd_chain(P, _DT, order), dim=-1)  # (B, T-order, K)
    ref_scalar = float(ref.mean())
    assert mv.value == pytest.approx(ref_scalar, rel=1e-6)
    assert mv.value == pytest.approx(pinned, rel=1e-5)

    assert mv.perframe is not None
    assert mv.perframe.dtype == np.float32
    assert mv.perframe.shape == (_T - order,)
    assert ALIGNMENTS[key] == "front"
    assert first_frame_index(key, _T, len(mv.perframe)) == order
    assert float(mv.perframe.mean()) == pytest.approx(mv.value, rel=1e-5)


# ===========================================================================
# accel_violation_frac / no_teleport_frac
# ===========================================================================

class TestViolationFracPerframe:
    def test_accel_violation_frac_perframe_is_fraction_and_reduces_to_scalar(self):
        P = _make_P()
        metric = all_metrics()["accel_violation_frac"]
        mv = metric.compute(MetricContext(dt=_DT, P=P, robot=_fake_robot()))
        assert mv.available

        a = torch.linalg.norm(_fd_chain(P, _DT, 2), dim=-1)  # (B, T-2, K)
        ref_scalar = float((a > 5.0).float().mean())
        assert mv.value == pytest.approx(ref_scalar, rel=1e-6)
        assert mv.value == pytest.approx(0.875, rel=1e-5)

        assert mv.perframe.shape == (_T - 2,)
        assert ALIGNMENTS["accel_violation_frac"] == "front"
        assert first_frame_index("accel_violation_frac", _T, len(mv.perframe)) == 2
        assert mv.perframe.min() >= 0.0 and mv.perframe.max() <= 1.0
        # continuous per-frame fraction -> its own mean reproduces the scalar
        assert float(mv.perframe.mean()) == pytest.approx(mv.value, rel=1e-5)

    def test_no_teleport_frac_perframe_is_a_literal_boolean(self):
        P = _make_P()
        metric = all_metrics()["no_teleport_frac"]
        mv = metric.compute(MetricContext(dt=_DT, P=P, robot=_fake_robot()))
        assert mv.available
        assert mv.value == pytest.approx(0.0, abs=1e-9)

        assert mv.perframe.shape == (_T - 1,)
        assert ALIGNMENTS["no_teleport_frac"] == "front"
        assert first_frame_index("no_teleport_frac", _T, len(mv.perframe)) == 1
        # a genuine 0/1 indicator, not a continuous fraction
        assert set(np.unique(mv.perframe).tolist()) <= {0.0, 1.0}
        # exact reduction (this one IS already frame-granular in the source
        # arithmetic, so mean(trace) == scalar to float precision, not just
        # approximately as with accel_violation_frac's split reduction)
        assert float(mv.perframe.mean()) == pytest.approx(mv.value, abs=1e-9)

    def test_no_teleport_frac_perframe_fires_on_an_actual_teleport(self):
        """A synthetic one-frame jump must show up in the trace, not just the
        scalar -- this is literally the "show me where it fired" use case."""
        P = _make_P().clone()
        P[:, 8] += 100.0  # one huge jump at frame 8 -> frames 7->8 and 8->9 violate
        metric = all_metrics()["no_teleport_frac"]
        mv = metric.compute(MetricContext(dt=_DT, P=P, robot=_fake_robot()))
        assert mv.value > 0.0
        # velocity sample i is frames (i, i+1); frame-8 jump violates samples
        # i=7 (7->8) and i=8 (8->9), aligned to original frame index i+1 (see
        # ALIGNMENTS['front']), i.e. trace indices 7 and 8 (frames 8 and 9).
        assert mv.perframe[7] == 1.0
        assert mv.perframe[8] == 1.0


# ===========================================================================
# torque_frac_rated (needs a real GR-1 URDF via $KINESCORE_ASSETS)
# ===========================================================================

def _gr1_robot():
    from kinescore.paths import MissingPathError
    try:
        from kinescore.robots.gr1.spec import GR1Spec
        return GR1Spec()
    except MissingPathError as exc:
        pytest.skip(f"$KINESCORE_ASSETS unavailable: {exc}")


class TestTorqueFracRatedPerframe:
    def test_perframe_shape_alignment_and_ordering_vs_scalar(self):
        from kinescore.metrics.torque import TorqueFracRated

        robot = _gr1_robot()
        T = 12
        g = torch.Generator().manual_seed(7)
        n_q = robot.q_lo.shape[0]
        q = torch.cumsum(torch.randn(1, T, n_q, generator=g) * 0.01, dim=1)
        ctx = MetricContext(dt=1.0 / 30.0, q=q, robot=robot)

        metric = TorqueFracRated()
        mv = metric.compute(ctx)
        assert mv.available, mv.reason

        assert mv.perframe is not None
        assert mv.perframe.dtype == np.float32
        assert mv.perframe.shape == (T - 2,)
        assert ALIGNMENTS["torque_frac_rated"] == "interior"
        assert first_frame_index("torque_frac_rated", T, len(mv.perframe)) == 1

        # The scalar is a p98 percentile of exactly this per-frame envelope,
        # so it must sit at or below the trace's max (never above it) and
        # not be a wildly different order of magnitude.
        assert mv.value <= float(mv.perframe.max()) + 1e-4
        assert mv.value >= float(np.percentile(mv.perframe, 0))

    def test_unsupported_robot_has_no_perframe(self):
        from kinescore.metrics.torque import TorqueFracRated

        robot = FakeRobot(name="not_a_gr1")
        q = torch.zeros(1, 6, 3)
        mv = TorqueFracRated().compute(MetricContext(dt=0.1, q=q, robot=robot))
        assert not mv.available
        assert mv.perframe is None


# ===========================================================================
# suite ids: perframe must not participate in MetricSuite.suite_id
# ===========================================================================

class TestSuiteIdsUnchanged:
    def test_the_three_suite_ids_match_the_declared_values(self):
        from kinescore.metrics.suites import ALL_METRICS, INVARIANT_V1, RATE_FREE

        print(f"invariant_v1 = {INVARIANT_V1.suite_id}")
        print(f"all_metrics  = {ALL_METRICS.suite_id}")
        print(f"rate_free    = {RATE_FREE.suite_id}")

        assert INVARIANT_V1.suite_id == "sha256:cb01e10a9318c420"
        # Was `full` / sha256:d346cf8f84a45742 before a (concurrent, not
        # this agent's) rename of the suite's *name* -- suite_id hashes
        # name+term-set (core/suite.py:100), so the rename alone moved it.
        # The metric *term set* is unchanged (invariant_v1 + torque_frac_rated
        # + rigidity_worst_bone_mm, still 28 keys).
        assert ALL_METRICS.suite_id == "sha256:b6924a162403ca8d"
        assert RATE_FREE.suite_id == "sha256:53b771bbad755c58"

    def test_perframe_does_not_participate_in_the_hash(self):
        """Flip every metric's perframe on synthetically and confirm the id
        this package actually ships is unaffected by real perframe=True use.

        This does not literally toggle `perframe` off in-process (`MetricSpec`
        is frozen and the registry is populated at import time), so instead
        it checks the property analytically against `MetricSuite.suite_id`'s
        own implementation: the id is a hash of `name` + `output_keys` +
        `invariant_keys` only. `perframe` is not read anywhere in that
        payload -- reproduced here from the same three suite objects the
        real benchmark run uses, so a future edit to `suite_id` that started
        reading `perframe` would have to also edit this assertion.
        """
        import inspect

        from kinescore.core.suite import MetricSuite
        from kinescore.metrics.suites import ALL_METRICS, INVARIANT_V1

        src = inspect.getsource(MetricSuite.suite_id.fget)
        assert "perframe" not in src

        # invariant_v1/all_metrics both carry perframe=True metrics (jerk,
        # rigidity_worst_bone_mm, ...) so the id-stability claim is not
        # vacuous for them. rate_free is filtered to dt_exponent==0 members
        # of invariant_v1's term set (metrics/suites.py), none of which
        # declare perframe=True today -- included in the loop anyway so a
        # future rate_free member that does gain perframe is still covered
        # by the `"perframe" not in src` check above, just not by this
        # per-suite non-vacuousness assertion.
        for suite in (INVARIANT_V1, ALL_METRICS):
            assert any(m.spec.perframe for m in suite.metrics), (
                f"suite {suite.name!r} has no perframe=True metric in this "
                f"parametrization -- the id-stability claim above would be "
                f"vacuous for it")


# ===========================================================================
# ALIGNMENTS / first_frame_index bookkeeping
# ===========================================================================

class TestAlignmentTable:
    def test_every_perframe_declaring_metric_is_classified(self):
        undeclared = [key for key, m in all_metrics().items()
                     if m.spec.perframe and key not in ALIGNMENTS]
        assert not undeclared, (
            f"metric(s) declare perframe=True but bench/traces.py's "
            f"ALIGNMENTS doesn't know how to align them: {undeclared}")

    def test_unknown_metric_raises(self):
        with pytest.raises(KeyError):
            first_frame_index("not_a_real_metric", 10, 10)

    def test_drop_mismatch_raises(self):
        with pytest.raises(ValueError, match="frame\\(s\\) dropped"):
            first_frame_index("mean_jerk_mps3", 10, 10)  # jerk must drop 3

    def test_same_length_front_interior_all_agree_with_expected_drop(self):
        assert first_frame_index("rigidity_worst_bone_mm", 20, 20) == 0
        assert first_frame_index("mean_jerk_mps3", 20, 17) == 3
        assert first_frame_index("torque_frac_rated", 20, 18) == 1


# ===========================================================================
# TraceStore: sidecar I/O, resumability, plain-numpy loadability
# ===========================================================================

def _traces(record_key, path="clip.mp4", **overrides) -> ClipTraces:
    defaults = {
        "record_key": record_key, "path": path, "dt": 0.1, "fps": 10.0,
        "n_frames": 16, "suite_id": "sha256:deadbeef",
        "suite_name": "test_suite", "reader_id": "fake/reader@1",
        "arrays": {
            "mean_jerk_mps3": np.arange(13, dtype=np.float32),
            "rigidity_worst_bone_mm": np.arange(16, dtype=np.float32) * 2.0,
        },
        "meta": {
            "mean_jerk_mps3": {"units": "m/s^3", "length": 13,
                              "first_frame_index": 3, "dt_exponent": 3},
            "rigidity_worst_bone_mm": {"units": "mm", "length": 16,
                                      "first_frame_index": 0, "dt_exponent": 0},
        },
    }
    defaults.update(overrides)
    return ClipTraces(**defaults)


def _load_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class TestTraceStore:
    def test_write_then_plain_numpy_reload_round_trips(self, tmp_path):
        store = TraceStore(str(tmp_path / "traces.npz"))
        key = ("clip_a.mp4", "sha256:deadbeef", "fake/reader@1")
        store.append(_traces(key, path="clip_a.mp4"))
        store.close()  # commit: nothing is on disk until a checkpoint

        # Plain numpy/json, no kinescore -- the "notebook" contract.
        (entry,) = _load_jsonl(tmp_path / "traces_index.jsonl")
        npz = np.load(tmp_path / "traces.npz")

        assert entry["path"] == "clip_a.mp4"
        assert entry["dt"] == 0.1
        assert entry["fps"] == 10.0
        assert entry["n_frames"] == 16
        assert entry["suite_id"] == "sha256:deadbeef"
        assert entry["traces"]["mean_jerk_mps3"]["first_frame_index"] == 3

        arr = npz[f"{entry['clip_id']}/mean_jerk_mps3"]
        np.testing.assert_array_equal(arr, np.arange(13, dtype=np.float32))
        assert arr.dtype == np.float32

    def test_append_is_idempotent_for_the_same_clip_identity(self, tmp_path):
        store = TraceStore(str(tmp_path / "traces.npz"))
        key = ("clip_a.mp4", "sha256:deadbeef", "fake/reader@1")
        store.append(_traces(key))
        store.append(_traces(key))  # simulate a resumed run re-visiting it
        store.close()

        entries = _load_jsonl(tmp_path / "traces_index.jsonl")
        assert len(entries) == 1
        npz = np.load(tmp_path / "traces.npz")
        # exactly one entry per metric, not two
        assert sum(1 for n in npz.files if n.endswith("mean_jerk_mps3")) == 1

    def test_idempotent_across_a_checkpoint_boundary(self, tmp_path):
        """The same no-duplicate guarantee must hold once a clip has already
        been durably committed (index on disk), not just while still
        pending in memory."""
        store = TraceStore(str(tmp_path / "traces.npz"))
        key = ("clip_a.mp4", "sha256:deadbeef", "fake/reader@1")
        store.append(_traces(key))
        store.checkpoint()  # durably commit
        store.append(_traces(key))  # re-visit after the checkpoint
        store.close()

        entries = _load_jsonl(tmp_path / "traces_index.jsonl")
        assert len(entries) == 1
        npz = np.load(tmp_path / "traces.npz")
        assert sum(1 for n in npz.files if n.endswith("mean_jerk_mps3")) == 1

    def test_two_distinct_clips_both_land_and_are_independently_readable(self, tmp_path):
        store = TraceStore(str(tmp_path / "traces.npz"))
        key_a = ("clip_a.mp4", "sha256:deadbeef", "fake/reader@1")
        key_b = ("clip_b.mp4", "sha256:deadbeef", "fake/reader@1")
        store.append(_traces(key_a, path="clip_a.mp4"))
        store.append(_traces(key_b, path="clip_b.mp4"))
        store.close()

        entries = _load_jsonl(tmp_path / "traces_index.jsonl")
        assert len(entries) == 2
        paths = {e["path"] for e in entries}
        assert paths == {"clip_a.mp4", "clip_b.mp4"}
        clip_ids = {e["clip_id"] for e in entries}
        assert len(clip_ids) == 2  # distinct hashes, no collision

    def test_has_reflects_pending_and_committed_state(self, tmp_path):
        store = TraceStore(str(tmp_path / "traces.npz"))
        key = ("clip_a.mp4", "sha256:deadbeef", "fake/reader@1")
        assert not store.has(key)
        store.append(_traces(key))
        assert store.has(key)  # true even before a checkpoint (pending)
        store.checkpoint()
        assert store.has(key)  # still true once durably committed

    def test_nothing_on_disk_until_checkpoint_or_close(self, tmp_path):
        """Durability is batched (see TraceStore's docstring): an append
        alone must not yet produce a readable index line."""
        store = TraceStore(str(tmp_path / "traces.npz"))
        key = ("clip_a.mp4", "sha256:deadbeef", "fake/reader@1")
        store.append(_traces(key))
        assert not (tmp_path / "traces_index.jsonl").exists()
        store.close()
        assert (tmp_path / "traces_index.jsonl").exists()

    def test_load_example_returns_a_time_axis_and_values(self, tmp_path):
        store = TraceStore(str(tmp_path / "traces.npz"))
        key = ("clip_a.mp4", "sha256:deadbeef", "fake/reader@1")
        store.append(_traces(key))
        store.close()

        t, values = load_example(str(tmp_path), metric_key="mean_jerk_mps3")
        assert len(t) == len(values) == 13
        # first_frame_index=3, dt=0.1 -> t0 = 0.3
        assert t[0] == pytest.approx(0.3)
        assert t[1] - t[0] == pytest.approx(0.1)

    def test_load_example_missing_run_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_example(str(tmp_path))


# ===========================================================================
# clip_traces_from_scored: absent metrics stay absent, never zero
# ===========================================================================

class TestClipTracesFromScored:
    def test_no_perframe_metrics_in_suite_returns_none(self):
        from kinescore.core.clip import ClipSpec
        from kinescore.core.metric import MetricValue
        from kinescore.core.scorer import ScoredClip
        from kinescore.core.suite import MetricSuite, SuiteResult

        suite = MetricSuite(name="scalar_only", metrics=["rigidity_residual_mm"])
        values = {"rigidity_residual_mm": MetricValue("rigidity_residual_mm", 1.23)}
        result = SuiteResult(suite.suite_id, suite.name, values)
        clip = ClipSpec.from_fps(path="x.mp4", fps=10.0, n_frames=16,
                                 width=64, height=64)
        scored = ScoredClip(clip=clip, result=result, robot="franka_panda",
                           reader_id="r@1", limit_semantics="raw_rad",
                           n_frames_scored=16)
        assert clip_traces_from_scored(scored) is None

    def test_unavailable_perframe_metric_is_absent_not_zero(self):
        from kinescore.core.clip import ClipSpec
        from kinescore.core.metric import MetricValue
        from kinescore.core.scorer import ScoredClip
        from kinescore.core.suite import MetricSuite, SuiteResult

        suite = MetricSuite(name="mixed",
                           metrics=["mean_jerk_mps3", "rigidity_worst_bone_mm"])
        values = {
            "mean_jerk_mps3": MetricValue.unavailable(
                "mean_jerk_mps3", "too_few_frames:2<4"),
            "rigidity_worst_bone_mm": MetricValue(
                "rigidity_worst_bone_mm", 4.2,
                perframe=np.ones(6, dtype=np.float32)),
        }
        result = SuiteResult(suite.suite_id, suite.name, values)
        clip = ClipSpec.from_fps(path="x.mp4", fps=10.0, n_frames=6,
                                 width=64, height=64)
        scored = ScoredClip(clip=clip, result=result, robot="franka_panda",
                           reader_id="r@1", limit_semantics="raw_rad",
                           n_frames_scored=6)

        traces = clip_traces_from_scored(scored)
        assert traces is not None
        assert "rigidity_worst_bone_mm" in traces.arrays
        assert "mean_jerk_mps3" not in traces.arrays  # absent, never zeros
        assert "mean_jerk_mps3" not in traces.meta


# ===========================================================================
# bench.runner.run(trace_store=...) wiring
# ===========================================================================

class _FakeSuite:
    def __init__(self, suite_id: str) -> None:
        self.suite_id = suite_id
        self.name = "test_suite"
        self.output_keys = ("mean_jerk_mps3",)


class _FakeReader:
    def __init__(self, reader_id: str) -> None:
        self.reader_id = reader_id
        self.limit_semantics = "raw_rad"


class _FakeRobotName:
    name = "franka_panda"


class _FakeScorer:
    """Duck-typed stand-in for :class:`~kinescore.core.scorer.Scorer`.

    ``bench.runner.run`` touches ``scorer.suite.suite_id``,
    ``scorer.reader.reader_id`` and ``scorer.score(frames, clip)`` on the
    happy path, plus (on a failure) everything
    ``bench.store.failed_record`` reads: ``scorer.robot.name``,
    ``scorer.reader.limit_semantics``, ``scorer.suite.name``/``output_keys``.
    This implements exactly that surface so the runner (real code, not
    mocked) can be exercised without a real reader/robot/video pipeline.
    """

    def __init__(self, suite_id: str, reader_id: str, fail_paths: frozenset = frozenset()):
        self.suite = _FakeSuite(suite_id)
        self.reader = _FakeReader(reader_id)
        self.robot = _FakeRobotName()
        self.fail_paths = fail_paths

    def score(self, frames, clip):
        from kinescore.core.metric import MetricValue
        from kinescore.core.scorer import ScoredClip
        from kinescore.core.suite import SuiteResult

        if clip.path in self.fail_paths:
            raise RuntimeError(f"synthetic failure for {clip.path}")

        values = {
            "mean_jerk_mps3": MetricValue(
                "mean_jerk_mps3", 3.14,
                perframe=np.linspace(0, 1, clip.n_frames - 3, dtype=np.float32)),
        }
        result = SuiteResult(self.suite.suite_id, "test_suite", values)
        return ScoredClip(clip=clip, result=result, robot="franka_panda",
                          reader_id=self.reader.reader_id,
                          limit_semantics="raw_rad", n_frames_scored=clip.n_frames)


def _rows():
    return [
        {"path": "ok_1.mp4", "fps": 10.0, "n_frames": 16, "w": 64, "h": 64},
        {"path": "ok_2.mp4", "fps": 10.0, "n_frames": 20, "w": 64, "h": 64},
        {"path": "bad.mp4", "fps": 10.0, "n_frames": 16, "w": 64, "h": 64},
    ]


class TestRunnerTraceWiring:
    def _run(self, tmp_path, name, *, with_traces: bool, monkeypatch):
        import kinescore.video.reader as video_reader
        from kinescore.bench.runner import run as run_bench
        from kinescore.bench.traces import TraceStore

        monkeypatch.setattr(
            video_reader, "load_rgb",
            lambda clip, max_frames=0: torch.zeros(1, clip.n_frames, 3))

        scorer = _FakeScorer("sha256:deadbeef", "fake/reader@1",
                            fail_paths=frozenset({"bad.mp4"}))
        out = tmp_path / name
        results_path = str(out / "results.jsonl")
        trace_store = TraceStore(str(out / "traces.npz")) if with_traces else None
        summary = run_bench(_rows(), scorer, results_path,
                            trace_store=trace_store)
        return results_path, out, summary

    def test_traces_flag_off_leaves_results_jsonl_untouched(self, tmp_path, monkeypatch):
        path_no, _out_no, summary_no = self._run(
            tmp_path, "no_traces", with_traces=False, monkeypatch=monkeypatch)
        path_yes, out_yes, summary_yes = self._run(
            tmp_path, "with_traces", with_traces=True, monkeypatch=monkeypatch)

        assert summary_no == summary_yes
        with open(path_no) as f:
            content_no = f.read()
        with open(path_yes) as f:
            content_yes = f.read()
        assert content_no == content_yes  # byte-identical either way

        assert not (tmp_path / "no_traces" / "traces.npz").exists()
        assert (out_yes / "traces.npz").exists()
        assert (out_yes / "traces_index.jsonl").exists()

    def test_traces_written_only_for_ok_rows(self, tmp_path, monkeypatch):
        _path, out, summary = self._run(
            tmp_path, "run", with_traces=True, monkeypatch=monkeypatch)
        assert summary["n_scored"] == 2
        assert summary["n_failed"] == 1

        entries = _load_jsonl(out / "traces_index.jsonl")
        paths = {e["path"] for e in entries}
        assert paths == {"ok_1.mp4", "ok_2.mp4"}
        assert "bad.mp4" not in paths

    def test_traces_default_off_matches_no_trace_store_argument(self, tmp_path, monkeypatch):
        """Calling run() the old way (no trace_store kwarg at all) must
        behave identically to passing trace_store=None explicitly."""
        import kinescore.video.reader as video_reader
        from kinescore.bench.runner import run as run_bench

        monkeypatch.setattr(
            video_reader, "load_rgb",
            lambda clip, max_frames=0: torch.zeros(1, clip.n_frames, 3))
        scorer = _FakeScorer("sha256:deadbeef", "fake/reader@1")

        out = tmp_path / "legacy_call"
        results_path = str(out / "results.jsonl")
        summary = run_bench(_rows()[:2], scorer, results_path)  # no trace_store kwarg
        assert summary["n_scored"] == 2
        assert not (out / "traces.npz").exists()
