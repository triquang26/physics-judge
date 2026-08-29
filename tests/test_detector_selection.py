"""``--detectors`` picks what is reported; scoring still runs all of them."""
from __future__ import annotations

import pytest

from kinescore.cli._shared import resolve_detectors
from kinescore.violations import DETECTORS, HEADLINE

pytestmark = pytest.mark.unit

_ALL = {d.name for d in DETECTORS}


class TestHeadline:
    def test_the_headline_set_is_a_subset_of_what_is_scored(self):
        assert set(HEADLINE) <= _ALL

    def test_no_argument_falls_back_to_the_headline_set(self):
        assert resolve_detectors(None, _ALL) == list(HEADLINE)


class TestResolve:
    def test_all_expands_to_every_scored_detector(self):
        assert set(resolve_detectors(["all"], _ALL)) == _ALL

    def test_an_explicit_list_is_kept_in_the_order_given(self):
        assert resolve_detectors(["jerk", "rigidity"], _ALL) == \
            ["jerk", "rigidity"]

    def test_a_detector_absent_from_the_results_is_refused(self):
        with pytest.raises(SystemExit, match="teleport"):
            resolve_detectors(["teleport"], {"rigidity", "jerk"})

    def test_a_headline_absent_from_the_results_is_refused(self):
        with pytest.raises(SystemExit):
            resolve_detectors(None, {"self_collision"})
