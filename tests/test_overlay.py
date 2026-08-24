"""The rendered timeline must say what the scorer said, in the scorer's sense."""
from __future__ import annotations

import numpy as np
import pytest

from kinescore.video import overlay

pytestmark = pytest.mark.unit


def _detector(per_frame, threshold):
    return {"threshold": threshold, "per_frame": list(per_frame)}


def _row(length=16, **over):
    """A clip no detector faults, plus whatever ``over`` replaces.

    A passing value differs by direction: below the threshold for a
    higher-is-worse detector, above it for one thresholded from below.
    """
    from kinescore.violations import segments

    base = {name: _detector([1.0 if higher_is_worse else 100.0] * 4, 10.0)
            for name, higher_is_worse in overlay.detector_order()}
    base.update(over)
    names = [n for n, _ in overlay.detector_order()]
    return {"id": "00001", "role": "dense", "aug_tag": None, "task": "t",
            "violations": base,
            "segments": segments.report(base, names, length=length)}


class TestSeverity:
    def test_a_higher_is_worse_reading_scales_with_the_value(self):
        assert overlay._severity(5.0, 10.0, True) == pytest.approx(0.5)

    def test_a_lower_is_worse_reading_is_the_reciprocal(self):
        assert overlay._severity(200.0, 100.0, False) == pytest.approx(0.5)

    def test_a_lower_is_worse_reading_at_the_bound_reads_as_one(self):
        assert overlay._severity(100.0, 100.0, False) == pytest.approx(1.0)

    def test_a_zero_threshold_cannot_divide(self):
        assert overlay._severity(3.0, 0.0, True) == 0.0


class TestOrder:
    def test_self_collision_is_the_one_thresholded_from_below(self):
        senses = dict(overlay.detector_order())
        assert senses["self_collision"] is False
        assert all(v for k, v in senses.items() if k != "self_collision")

    def test_a_name_list_restricts_and_orders_the_rows(self):
        assert overlay.detector_order(["jerk", "rigidity"]) == (
            ("jerk", True), ("rigidity", True))


class TestMeasure:
    def test_a_large_reading_stays_inside_its_column(self):
        assert len(overlay._measure(2774313.06)) <= 7

    def test_a_small_reading_stays_readable(self):
        assert overlay._measure(76.54) == "77"


class TestViolatedMask:
    def test_a_violated_segment_covers_its_whole_frame_range(self):
        mask = overlay._violated_mask(
            [{"start": 1, "end": 3, "violated": True}], 6)
        assert list(mask) == [False, True, True, True, False, False]

    def test_a_segment_that_passed_marks_nothing(self):
        mask = overlay._violated_mask(
            [{"start": 0, "end": 3, "violated": False}], 4)
        assert not mask.any()

    def test_verdicts_are_read_off_the_row(self):
        row = _row(length=2, rigidity=_detector([1.0, 1.0, 99.0, 99.0], 10.0))
        got = overlay._verdicts(row, "rigidity")
        assert [v["violated"] for v in got] == [False, True]


class TestRenderClip:
    def test_the_panel_grows_by_one_row_per_detector(self):
        frames = np.zeros((4, 32, 640, 3), dtype=np.uint8)
        out = overlay.render_clip(frames, _row())
        rows = len(overlay.detector_order())
        assert out.shape == (4, 32 + overlay._HEADER_H + overlay._ROW_H * rows,
                             640, 3)

    def test_only_the_named_detectors_get_a_row(self):
        frames = np.zeros((4, 32, 640, 3), dtype=np.uint8)
        out = overlay.render_clip(frames, _row(), ["rigidity", "jerk"])
        assert out.shape[1] == 32 + overlay._HEADER_H + overlay._ROW_H * 2

    def test_a_detector_left_out_cannot_outline_a_frame(self):
        violating = _detector([1.0, 1.0, 99.0, 99.0], 10.0)
        frames = np.zeros((4, 16, 320, 3), dtype=np.uint8)
        out = overlay.render_clip(frames, _row(length=2, teleport=violating),
                                  ["rigidity", "jerk"])
        assert not out[2, overlay._HEADER_H, 160].any()

    def test_the_source_frame_is_carried_through_unchanged(self):
        frames = np.random.randint(0, 255, (2, 16, 320, 3), dtype=np.uint8)
        out = overlay.render_clip(frames, _row())
        top = overlay._HEADER_H
        assert np.array_equal(out[0, top:top + 16], frames[0])

    def test_a_violated_segment_outlines_its_frames_and_a_clean_one_does_not(self):
        rigid = _detector([1.0, 1.0, 99.0, 99.0], 10.0)
        out = overlay.render_clip(
            np.zeros((4, 16, 320, 3), dtype=np.uint8),
            _row(length=2, rigidity=rigid))
        top = overlay._HEADER_H
        assert out[2, top, 160].any()
        assert not out[0, top, 160].any()


class TestTrace:
    def test_a_clip_names_the_path_it_came_from(self):
        row = _row()
        row["source_path"] = "augment/bimanual/output/ep_000/pred.mp4"
        assert overlay._trace(row, 1280) == row["source_path"]

    def test_an_over_long_path_keeps_its_tail(self):
        row = _row()
        row["source_path"] = "a/" * 200 + "episode_000123/pred.mp4"
        out = overlay._trace(row, 640)
        assert out.startswith("...")
        assert out.endswith("episode_000123/pred.mp4")

    def test_a_clip_with_no_source_falls_back_to_its_bench_id(self):
        assert overlay._trace(_row(), 1280) == "clips/00001.mp4"
