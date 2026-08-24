"""Segment verdicts: fixed windows, each detector reduced its own way."""
from __future__ import annotations

import pytest

from kinescore.violations import DETECTORS, segments

pytestmark = pytest.mark.unit

_NAMES = ("rigidity", "jerk")


def _violations(**series):
    thresholds = {"rigidity": 76.0, "jerk": 2.0e6, "teleport": 1000.0,
                  "joint_limit": 4.0, "self_collision": 190.0}
    return {n: {"threshold": thresholds[n], "per_frame": list(v)}
            for n, v in series.items()}


class TestBounds:
    def test_an_even_clip_splits_into_whole_windows(self):
        assert segments.bounds(32, 16) == [(0, 15), (16, 31)]

    def test_the_last_window_is_kept_short_rather_than_dropped(self):
        assert segments.bounds(21, 16) == [(0, 15), (16, 20)]

    def test_a_clip_shorter_than_one_window_is_one_segment(self):
        assert segments.bounds(9, 16) == [(0, 8)]

    def test_an_empty_clip_has_no_segments(self):
        assert segments.bounds(0, 16) == []


class TestReduction:
    def test_rigidity_is_judged_on_its_typical_frame(self):
        assert next(d for d in DETECTORS
                    if d.name == "rigidity").segment_reduce == "median"

    def test_jerk_is_judged_on_its_worst_frame(self):
        assert next(d for d in DETECTORS
                    if d.name == "jerk").segment_reduce == "worst"

    def test_one_stretched_frame_does_not_violate_rigidity(self):
        v = _violations(rigidity=[10.0] * 15 + [500.0])
        seg = segments.report(v, ["rigidity"], length=16)[0]
        assert seg["detectors"]["rigidity"]["violated"] is False
        assert seg["detectors"]["rigidity"]["value"] == 10.0

    def test_a_segment_stretched_throughout_violates_rigidity(self):
        v = _violations(rigidity=[90.0] * 16)
        seg = segments.report(v, ["rigidity"], length=16)[0]
        assert seg["detectors"]["rigidity"]["violated"] is True

    def test_one_jerk_spike_violates_the_segment(self):
        v = _violations(jerk=[1.0] * 15 + [3.0e6])
        seg = segments.report(v, ["jerk"], length=16)[0]
        assert seg["detectors"]["jerk"]["violated"] is True
        assert seg["detectors"]["jerk"]["value"] == 3.0e6

    def test_an_even_length_window_takes_the_midpoint_of_the_two_middles(self):
        v = _violations(rigidity=[10.0, 20.0, 30.0, 40.0])
        seg = segments.report(v, ["rigidity"], length=4)[0]
        assert seg["detectors"]["rigidity"]["value"] == pytest.approx(25.0)


class TestReport:
    def test_each_segment_carries_its_own_frame_range(self):
        v = _violations(rigidity=[10.0] * 20, jerk=[1.0] * 20)
        got = segments.report(v, _NAMES, length=16)
        assert [(s["start"], s["end"]) for s in got] == [(0, 15), (16, 19)]

    def test_every_named_detector_gets_a_verdict_in_every_segment(self):
        v = _violations(rigidity=[10.0] * 20, jerk=[1.0] * 20)
        got = segments.report(v, _NAMES, length=16)
        assert all(set(s["detectors"]) == set(_NAMES) for s in got)

    def test_the_threshold_judged_against_is_recorded(self):
        v = _violations(rigidity=[10.0] * 4)
        seg = segments.report(v, ["rigidity"], length=16)[0]
        assert seg["detectors"]["rigidity"]["threshold"] == 76.0

    def test_a_detector_with_no_series_is_left_out(self):
        got = segments.report(_violations(rigidity=[10.0] * 4),
                              ["rigidity", "jerk"], length=16)
        assert set(got[0]["detectors"]) == {"rigidity"}

    def test_self_collision_violates_from_below(self):
        v = {"self_collision": {"threshold": 190.0, "per_frame": [100.0] * 4}}
        seg = segments.report(v, ["self_collision"], length=16)[0]
        assert seg["detectors"]["self_collision"]["violated"] is True
