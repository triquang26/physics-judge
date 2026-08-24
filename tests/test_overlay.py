"""The rendered timeline must say what the scorer said, in the scorer's sense."""
from __future__ import annotations

import numpy as np
import pytest

from kinescore.video import overlay

pytestmark = pytest.mark.unit


def _detector(per_frame, threshold, intervals=()):
    return {"threshold": threshold, "per_frame": list(per_frame),
            "intervals": [list(i) for i in intervals]}


def _row(**over):
    base = {name: _detector([1.0] * 4, 10.0) for name, _ in
            overlay.detector_order()}
    base.update(over)
    return {"id": "00001", "role": "dense", "aug_tag": None, "task": "t",
            "violations": base}


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


class TestFlaggedMask:
    def test_interval_endpoints_are_inclusive(self):
        mask = overlay._flagged_mask(np.zeros(6), [[1, 3]])
        assert list(mask) == [False, True, True, True, False, False]

    def test_no_interval_flags_nothing(self):
        assert not overlay._flagged_mask(np.zeros(4), []).any()


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
        flagged = _detector([1.0, 99.0, 1.0, 1.0], 10.0, [[1, 1]])
        frames = np.zeros((4, 16, 320, 3), dtype=np.uint8)
        out = overlay.render_clip(frames, _row(teleport=flagged),
                                  ["rigidity", "jerk"])
        assert not out[1, overlay._HEADER_H, 160].any()

    def test_the_source_frame_is_carried_through_unchanged(self):
        frames = np.random.randint(0, 255, (2, 16, 320, 3), dtype=np.uint8)
        out = overlay.render_clip(frames, _row())
        top = overlay._HEADER_H
        assert np.array_equal(out[0, top:top + 16], frames[0])

    def test_a_flagged_frame_is_outlined_and_a_clean_one_is_not(self):
        rigid = _detector([1.0, 99.0, 1.0, 1.0], 10.0, [[1, 1]])
        out = overlay.render_clip(
            np.zeros((4, 16, 320, 3), dtype=np.uint8), _row(rigidity=rigid))
        top = overlay._HEADER_H
        assert out[1, top, 160].any()
        assert not out[0, top, 160].any()
