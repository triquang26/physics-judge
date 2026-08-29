"""ClipSpec is the sole owner of dt -- these tests pin that contract down.

A wrong ``dt`` corrupts every derivative quantity silently, so:
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


class TestViewLayoutKeyStability:
    """``key`` is stored in checkpoints and manifest rows and compared for
    equality -- these pin the exact strings the real corpus already has on
    disk (verified against ``kinescore_runtime/out/**/bench_manifest.parquet``:
    every existing row's ``view_layout`` column is one of these two values).
    Extending ``ViewLayout`` with ``packing``/``n_panels``/``panels`` must not
    change either.
    """

    def test_1view_height_stack_key_unchanged(self):
        assert ViewLayout(n_views=1).key == "1x?:unnamed"

    def test_3view_height_stack_key_unchanged(self):
        layout = ViewLayout(n_views=3, order=("exterior_1", "exterior_2", "wrist"))
        assert layout.key == "3x?:exterior_1+exterior_2+wrist"

    def test_tokens_per_view_still_appears_in_the_key(self):
        assert ViewLayout(n_views=1, tokens_per_view=64).key == "1x64:unnamed"

    def test_non_height_packing_gets_a_different_key(self):
        # A different physical packing is different geometry -- it must not
        # collide with an old-format key for the same n_views/order.
        plain = ViewLayout(n_views=2, order=("a", "b"))
        width = ViewLayout(n_views=2, order=("a", "b"), packing="width")
        assert plain.key != width.key
        assert plain.key == "2x?:a+b"
        assert width.key == "2x?:a+b:width:2p:0,1"

    def test_panel_subset_gets_a_different_key(self):
        plain = ViewLayout(n_views=2, order=("a", "b"))
        subset = ViewLayout(n_views=2, order=("a", "b"), packing="width",
                            n_panels=3, panels=(0, 1))
        assert plain.key != subset.key


class TestViewLayoutPacking:
    def test_default_packing_is_height_for_backward_compatibility(self):
        assert ViewLayout(n_views=3).packing == "height"

    def test_width_stack_crops_columns(self):
        layout = ViewLayout(n_views=3, packing="width")
        assert layout.view_crops(frame_width=960, frame_height=192) == [
            (0, 192, 0, 320), (0, 192, 320, 640), (0, 192, 640, 960)]

    def test_height_stack_crops_rows(self):
        layout = ViewLayout(n_views=3)
        assert layout.view_crops(frame_width=320, frame_height=576) == [
            (0, 192, 0, 320), (192, 384, 0, 320), (384, 576, 0, 320)]

    def test_grid2x2_crops_quadrants(self):
        layout = ViewLayout(n_views=4, packing="grid2x2", n_panels=4)
        assert layout.view_crops(frame_width=768, frame_height=432) == [
            (0, 216, 0, 384), (0, 216, 384, 768),
            (216, 432, 0, 384), (216, 432, 384, 768)]

    def test_grid2x2_requires_exactly_4_panels(self):
        with pytest.raises(ValueError, match="4 physical panels"):
            ViewLayout(n_views=3, packing="grid2x2", n_panels=3)

    def test_unknown_packing_rejected(self):
        with pytest.raises(ValueError):
            ViewLayout(n_views=1, packing="depth")  # type: ignore[arg-type]


class TestViewLayoutPanelSubset:
    """Selecting a subset of panels is a first-class layout property, not
    caller-side slicing -- the ctrlworld case (columns 0:320 and 320:640,
    wrist panel dropped)."""

    def test_ctrlworld_style_subset_drops_the_third_panel(self):
        layout = ViewLayout(n_views=2, order=("exterior_1", "exterior_2"),
                            packing="width", n_panels=3, panels=(0, 1))
        assert layout.panel_indices == (0, 1)
        assert layout.is_subset
        assert layout.view_crops(frame_width=960, frame_height=192) == [
            (0, 192, 0, 320), (0, 192, 320, 640)]

    def test_no_subset_is_not_flagged_as_one(self):
        assert not ViewLayout(n_views=3).is_subset
        assert not ViewLayout(n_views=1).is_subset

    def test_panels_must_have_one_entry_per_view(self):
        with pytest.raises(ValueError, match="panels has"):
            ViewLayout(n_views=2, n_panels=3, panels=(0, 1, 2))

    def test_panels_must_be_distinct(self):
        with pytest.raises(ValueError, match="distinct"):
            ViewLayout(n_views=2, n_panels=3, panels=(0, 0))

    def test_panel_index_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            ViewLayout(n_views=2, n_panels=3, panels=(0, 5))

    def test_mismatched_panel_and_view_count_without_explicit_panels_rejected(self):
        # n_panels != n_views but no `panels` given -- must say explicitly
        # which panels map to which views, never guess an implicit mapping.
        with pytest.raises(ValueError, match="no `panels` subset"):
            ViewLayout(n_views=2, n_panels=3)


class TestViewLayoutRefusesRatherThanGuesses:
    """The check that would have caught the real ctrlworld 960x192 bug: a
    3-view HEIGHT layout fed a WIDTH-stacked frame divides evenly
    (192 % 3 == 0) but produces implausible 960x64 panels -- this must raise,
    never silently slice three meaningless bands."""

    def test_width_stacked_960x192_declared_as_height_stack_raises(self):
        layout = ViewLayout(n_views=3, order=("exterior_1", "exterior_2", "wrist"))
        with pytest.raises(ValueError, match="aspect"):
            layout.view_crops(frame_width=960, frame_height=192)

    def test_same_frame_with_the_correct_width_packing_does_not_raise(self):
        layout = ViewLayout(n_views=3, order=("exterior_1", "exterior_2", "wrist"),
                            packing="width")
        crops = layout.view_crops(frame_width=960, frame_height=192)
        assert len(crops) == 3

    def test_indivisible_height_stack_still_raises(self):
        layout = ViewLayout(n_views=3)
        with pytest.raises(ValueError):
            layout.view_crops(frame_width=320, frame_height=577)

    def test_indivisible_width_stack_still_raises(self):
        layout = ViewLayout(n_views=3, packing="width")
        with pytest.raises(ValueError):
            layout.view_crops(frame_width=961, frame_height=192)

    def test_single_view_is_never_subject_to_the_aspect_guard(self):
        # n_panels=1 -- there's no packing ambiguity to guard against; a
        # single camera's own frame can legitimately be any aspect ratio.
        layout = ViewLayout(n_views=1)
        assert layout.view_crops(frame_width=960, frame_height=64) == [(0, 64, 0, 960)]


class TestViewLayoutAssertTokensAllModes:
    """``assert_tokens`` only depends on n_views/tokens_per_view -- pin that
    it keeps working for every packing mode, not just the default."""

    @pytest.mark.parametrize("packing,n_panels,panels", [
        ("height", None, ()),
        ("width", None, ()),
        ("grid2x2", 4, ()),
    ])
    def test_assert_tokens_unaffected_by_packing(self, packing, n_panels, panels):
        n_views = 4 if packing == "grid2x2" else 3
        layout = ViewLayout(n_views=n_views, tokens_per_view=5, packing=packing,
                            n_panels=n_panels, panels=panels)
        layout.assert_tokens(n_views * 5)  # does not raise
        with pytest.raises(ValueError):
            layout.assert_tokens(n_views * 5 + 1)
