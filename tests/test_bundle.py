"""Exporting a scored cell as numbered clips + one segments.json."""
from __future__ import annotations

import json

import imageio.v3 as iio
import numpy as np
import pytest

from kinescore.video.bundle import clip_entry, write_bundle

pytestmark = pytest.mark.unit

_ALL = ("rigidity", "jerk", "teleport", "joint_limit", "self_collision")
HEADLINE = ["rigidity", "jerk"]


def _clip(path, n=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.random.randint(0, 255, (n, 16, 64, 3), dtype=np.uint8)
    iio.imwrite(path, frames, fps=5, codec="libx264", macro_block_size=1)
    return str(path)


def _row(tmp_path, rel, violating=(), n=3):
    from kinescore.violations import segments

    detectors = {name: {"units": "mm", "threshold": 10.0,
                        "per_frame": [1.0] * n} for name in _ALL}
    for name in violating:
        detectors[name]["per_frame"] = [99.0] * n
    return {"path": _clip(tmp_path / "videos" / rel, n), "id": rel,
            "violations": detectors,
            "segments": segments.report(detectors, _ALL)}


class TestClipEntry:
    def test_each_segment_carries_value_ratio_and_verdict(self, tmp_path):
        video, detectors = clip_entry(
            _row(tmp_path, "a/pred.mp4", violating=["jerk"]), HEADLINE)

        (seg,) = video["segments"]
        assert seg["start_frame"] == 0 and seg["end_frame"] == 2
        assert seg["n_frames"] == 3
        assert seg["rigidity"] == {"value": 1.0, "ratio": 0.1,
                                   "violated": False}
        assert seg["jerk"]["ratio"] == 9.9 and seg["jerk"]["violated"]
        assert video["n_violated"] == 1

    def test_the_calibration_sits_beside_not_inside_the_segments(
            self, tmp_path):
        video, detectors = clip_entry(_row(tmp_path, "a/pred.mp4"), HEADLINE)

        assert detectors["rigidity"] == {"threshold": 10.0, "units": "mm",
                                         "reduce": "median"}
        assert detectors["jerk"]["reduce"] == "max"
        assert "teleport" not in detectors
        assert "threshold" not in video["segments"][0]["rigidity"]

    def test_source_is_the_path_below_the_videos_root(self, tmp_path):
        row = _row(tmp_path, "makovian/e0/pred.mp4")
        video, _ = clip_entry(row, HEADLINE, tmp_path / "videos")
        assert video["source"] == "makovian/e0/pred.mp4"


class TestWriteBundle:
    def test_clips_are_numbered_in_scoring_order_beside_one_json(
            self, tmp_path):
        rows = [_row(tmp_path, f"m/e{i}/pred.mp4") for i in range(3)]

        out = write_bundle(rows, tmp_path / "web" / "bundle", HEADLINE,
                           log=lambda m: None)

        assert sorted(p.name for p in out.iterdir()) == [
            "1.mp4", "2.mp4", "3.mp4", "segments.json"]
        doc = json.loads((out / "segments.json").read_text())
        assert list(doc["videos"]) == ["1", "2", "3"]
        video = doc["videos"]["2"]
        # source is the path below the deepest common parent of the rows
        assert video["source"] == "e1/pred.mp4"
        assert video["n_frames"] == 3 and video["fps"] == 5.0

    def test_the_json_counts_violations_per_clip_and_keeps_provenance(
            self, tmp_path):
        rows = [_row(tmp_path, "m/e0/pred.mp4", violating=["rigidity"]),
                _row(tmp_path, "m/e1/pred.mp4")]

        out = write_bundle(rows, tmp_path / "web" / "bundle", HEADLINE,
                           summary={"cell_id": "c", "reader_id": "r",
                                    "checkpoint_sha256": "abc"},
                           log=lambda m: None)

        doc = json.loads((out / "segments.json").read_text())
        assert doc["provenance"] == {"cell": "c", "reader": "r",
                                     "checkpoint": "abc"}
        assert doc["n_clips"] == 2
        assert doc["detectors"]["rigidity"]["threshold"] == 10.0
        assert doc["videos"]["1"]["n_violated"] == 1
        assert doc["videos"]["2"]["n_violated"] == 0

    def test_a_failed_row_is_skipped_and_counted_not_renumbered_around(
            self, tmp_path):
        good = _row(tmp_path, "m/e1/pred.mp4")
        failed = {"path": str(tmp_path / "videos" / "m/e0/pred.mp4"),
                  "id": "m/e0/pred", "error": "boom"}

        out = write_bundle([failed, good], tmp_path / "web" / "bundle",
                           HEADLINE, log=lambda m: None)

        doc = json.loads((out / "segments.json").read_text())
        assert doc["n_clips"] == 1 and doc["n_skipped"] == 1
        # the failed row still anchors the common root
        assert doc["videos"]["1"]["source"] == "e1/pred.mp4"


class TestPushJobs:
    def test_each_artifact_lands_at_its_bucket_path(self):
        import argparse

        from kinescore.cli.cmd_push import BUCKET, jobs

        args = argparse.Namespace(
            reader=["r1.c.v"], scores=["out/cell.v.m"], web=["out/web/b"],
            bucket=BUCKET)
        assert jobs(args, "/ckpt") == [
            ("/ckpt/r1.c.v.diff", f"{BUCKET}/train/r1.c.v/diffusion"),
            ("out/cell.v.m", f"{BUCKET}/scores/cell.v.m/diffusion"),
            ("out/web/b", f"{BUCKET}/web/b"),
        ]


class TestVideosIds:
    def test_ids_mirror_the_tree_so_shared_filenames_stay_distinct(self):
        from kinescore.cli.cmd_score import _videos_id

        assert _videos_id("/r/m/e0/pred.mp4", "/r") == "m__e0__pred"
        assert _videos_id("/r/m/e1/pred.mp4", "/r") == "m__e1__pred"
        assert _videos_id("/r/flat.mp4", "/r") == "flat"

    def test_pattern_filters_gt_and_merged_out(self, tmp_path):
        from kinescore.cli.cmd_score import _clips

        for name in ("full_pred", "full_gt", "full_merged"):
            _clip(tmp_path / "e0" / f"{name}.mp4")

        assert _clips(str(tmp_path), 0, "full_pred.mp4") == [
            str(tmp_path / "e0" / "full_pred.mp4")]
        assert len(_clips(str(tmp_path), 0)) == 3
