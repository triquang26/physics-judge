"""ClipSpec is the sole owner of dt -- these tests pin that contract down.

Covers defect D1 (a wrong dt silently corrupting every derivative metric):
:meth:`ClipSpec.subsample` must be the only way frame count and dt move
together, :meth:`ClipSpec.from_fps` must derive dt correctly, and
:func:`validate_dt` must catch the classic fps-passed-as-dt mistake before it
reaches a metric.
"""
from __future__ import annotations

import pytest

from kinescore.core.clip import ClipSpec, TimebaseError, ViewLayout, validate_dt


def _spec(**overrides) -> ClipSpec:
    kwargs = {"path": "/tmp/clip.mp4", "fps": 10.0, "dt": 0.1, "n_frames": 30,
             "width": 224, "height": 224}
    kwargs.update(overrides)
    return ClipSpec(**kwargs)


class TestSubsample:
    def test_scales_dt_and_n_frames(self):
        spec = _spec(n_frames=30)
        sub = spec.subsample(2)
        assert sub.dt == pytest.approx(0.2)
        assert sub.fps == pytest.approx(5.0)
        assert sub.n_frames == 15
        assert sub.stride == 2

    def test_n_frames_rounds_up_on_uneven_division(self):
        # 31 frames at stride 3 -> ceil(31/3) = 11, not floor
        spec = _spec(n_frames=31)
        sub = spec.subsample(3)
        assert sub.n_frames == 11

    def test_stride_accumulates_across_calls(self):
        spec = _spec(n_frames=100)
        sub = spec.subsample(2).subsample(2)
        assert sub.stride == 4
        assert sub.dt == pytest.approx(0.4)
        assert sub.fps == pytest.approx(2.5)

    def test_factor_one_is_identity(self):
        spec = _spec()
        assert spec.subsample(1) is spec

    def test_rejects_factor_below_one(self):
        spec = _spec()
        with pytest.raises(ValueError):
            spec.subsample(0)

    def test_dt_and_fps_stay_consistent_after_subsample(self):
        # __post_init__ re-validates dt*fps == 1.0 on the replaced instance --
        # subsample must update both fields together, never just one.
        spec = _spec(fps=30.0, dt=1.0 / 30.0, n_frames=90)
        sub = spec.subsample(3)
        assert sub.dt * sub.fps == pytest.approx(1.0)


class TestFromFps:
    def test_derives_dt_as_reciprocal_of_fps(self):
        spec = ClipSpec.from_fps(path="/tmp/c.mp4", fps=25.0, n_frames=50,
                                 width=64, height=64)
        assert spec.dt == pytest.approx(1.0 / 25.0)
        assert spec.dt_source == "ffprobe"  # default

    def test_records_dt_source(self):
        spec = ClipSpec.from_fps(path="/tmp/c.mp4", fps=25.0, n_frames=50,
                                 width=64, height=64, dt_source="table")
        assert spec.dt_source == "table"

    def test_rejects_nonpositive_fps(self):
        with pytest.raises(TimebaseError):
            ClipSpec.from_fps(path="/tmp/c.mp4", fps=0.0, n_frames=1,
                              width=1, height=1)

    def test_rejects_nonfinite_fps(self):
        with pytest.raises(TimebaseError):
            ClipSpec.from_fps(path="/tmp/c.mp4", fps=float("inf"), n_frames=1,
                              width=1, height=1)


class TestValidateDt:
    def test_accepts_a_normal_dt(self):
        assert validate_dt(0.1) == pytest.approx(0.1)

    def test_rejects_fps_passed_as_dt(self):
        # The canonical mistake this guard exists to catch: dt=30 instead of
        # dt=1/30. 10s is comfortably above any plausible real dt and
        # comfortably below any plausible fps used as a dt by mistake.
        with pytest.raises(ValueError, match="fps"):
            validate_dt(30.0)

    def test_rejects_zero(self):
        with pytest.raises(ValueError):
            validate_dt(0.0)

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            validate_dt(-0.1)

    def test_rejects_nonfinite(self):
        with pytest.raises(ValueError):
            validate_dt(float("nan"))

    def test_rejects_bool(self):
        # bool is a subclass of int in Python; must not silently pass as a dt.
        with pytest.raises(TypeError):
            validate_dt(True)

    def test_rejects_non_numeric(self):
        with pytest.raises(TypeError):
            validate_dt("0.1")


class TestClipSpecConsistency:
    def test_post_init_rejects_dt_fps_mismatch(self):
        with pytest.raises(TimebaseError):
            ClipSpec(path="/tmp/c.mp4", fps=10.0, dt=0.2, n_frames=1,
                     width=1, height=1)

    def test_as_row_round_trips_dt(self):
        spec = _spec()
        row = spec.as_row()
        assert row["dt"] == spec.dt
        assert row["fps"] == spec.fps
        assert row["dt_source"] == spec.dt_source

    def test_duration_s(self):
        spec = _spec(fps=10.0, dt=0.1, n_frames=30)
        assert spec.duration_s == pytest.approx(3.0)


class TestViewLayout:
    def test_view_height_divides_evenly(self):
        layout = ViewLayout(n_views=3)
        assert layout.view_height(576) == 192

    def test_view_height_rejects_indivisible(self):
        layout = ViewLayout(n_views=3)
        with pytest.raises(ValueError):
            layout.view_height(577)
