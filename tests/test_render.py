"""Rendering a scored cell: what gets a file, and what the reel holds."""
from __future__ import annotations

import json

import imageio.v3 as iio
import numpy as np
import pytest

from kinescore.video import render

pytestmark = pytest.mark.unit


def _clip(tmp_path, name, n=3):
    path = tmp_path / f"{name}.mp4"
    frames = np.random.randint(0, 255, (n, 16, 64, 3), dtype=np.uint8)
    iio.imwrite(path, frames, fps=5, codec="libx264", macro_block_size=1)
    return str(path), n


_ALL = ("rigidity", "jerk", "teleport", "joint_limit", "self_collision")


def _row(tmp_path, clip_id, role="dense", violating=(), n=3):
    from kinescore.violations import segments

    path, n = _clip(tmp_path, clip_id, n)
    detectors = {name: {"threshold": 10.0, "per_frame": [1.0] * n}
                 for name in _ALL}
    for name in violating:
        detectors[name]["per_frame"] = [99.0] * n
    return {"path": path, "id": clip_id, "role": role, "aug_tag": None,
            "task": "t", "violations": detectors,
            "segments": segments.report(detectors, _ALL)}


class TestReadResults:
    def test_a_missing_results_file_names_the_command_that_writes_it(
            self, tmp_path):
        with pytest.raises(SystemExit, match="kinescore score"):
            render.read_results(tmp_path / "results.jsonl")

    def test_rows_keep_the_order_they_were_scored_in(self, tmp_path):
        path = tmp_path / "results.jsonl"
        path.write_text("\n".join(json.dumps({"id": i}) for i in "bac"))
        assert [r["id"] for r in render.read_results(path)] == ["b", "a", "c"]

    def test_blank_lines_are_not_rows(self, tmp_path):
        path = tmp_path / "results.jsonl"
        path.write_text('{"id": "a"}\n\n')
        assert len(render.read_results(path)) == 1


class TestIsFlagged:
    def test_a_detector_outside_the_reported_set_does_not_count(self, tmp_path):
        row = _row(tmp_path, "a", violating=["teleport"])
        assert render.is_flagged(row, ["rigidity", "jerk"]) is False

    def test_a_reported_detector_counts(self, tmp_path):
        row = _row(tmp_path, "a", violating=["rigidity"])
        assert render.is_flagged(row, ["rigidity", "jerk"]) is True


class TestRenderResults:
    def test_every_row_gets_a_file_named_by_clip_and_role(self, tmp_path):
        rows = [_row(tmp_path, "00001", "dense"),
                _row(tmp_path, "00002", "fast")]
        out = render.render_results(rows, tmp_path / "out", ["rigidity"],
                                    log=lambda _: None)
        assert {p.name for p in out.glob("*.mp4")} == {
            "00001_dense.mp4", "00002_fast.mp4", "reel.mp4"}

    def test_the_reel_is_every_clip_end_to_end(self, tmp_path):
        rows = [_row(tmp_path, "00001", n=3), _row(tmp_path, "00002", n=4)]
        out = render.render_results(rows, tmp_path / "out", ["rigidity"],
                                    log=lambda _: None)
        assert len(iio.imread(out / "reel.mp4")) == 7

    def test_no_reel_leaves_the_per_clip_files_alone(self, tmp_path):
        rows = [_row(tmp_path, "00001")]
        out = render.render_results(rows, tmp_path / "out", ["rigidity"],
                                    reel=False, log=lambda _: None)
        assert not (out / "reel.mp4").exists()
        assert (out / "00001_dense.mp4").exists()

    def test_the_drawn_panel_matches_the_reported_detectors(self, tmp_path):
        rows = [_row(tmp_path, "00001")]
        out = render.render_results(rows, tmp_path / "out",
                                    ["rigidity", "jerk"], log=lambda _: None)
        from kinescore.video.overlay import _HEADER_H, _ROW_H

        assert iio.imread(out / "00001_dense.mp4").shape[1] == \
            16 + _HEADER_H + _ROW_H * 2
