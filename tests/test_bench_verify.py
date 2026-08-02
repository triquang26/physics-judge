"""kinescore.bench.verify.verify_layout: ffprobe every canonical clip against data_spec.yaml.

Marked ``@pytest.mark.ffmpeg`` throughout -- every check here needs a real
``ffprobe`` on ``PATH`` (auto-skipped otherwise -- see ``tests/conftest.py``).
"""
from __future__ import annotations

import json
import os
import subprocess

import pytest

from kinescore.bench.data_spec import parse_data_spec
from kinescore.bench.layout import CELL_CARD_NAME, PRED_NAME, CanonicalLayout
from kinescore.bench.verify import verify_layout

pytestmark = pytest.mark.ffmpeg


def _write_mp4(path, *, w=8, h=8, fps=5, dur=0.4) -> None:
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"color=c=black:s={w}x{h}:d={dur}:r={fps}", "-pix_fmt", "yuv420p", str(path)],
        check=True)


def _data_spec(**overrides):
    raw = {
        "generators": {
            "ctrlworld": {
                "shape": "episode_dir", "pred_filename": "pred_all_views.mp4",
                "gt_filename": "gt_all_views.mp4", "has_iter_level": False,
                "width": 8, "height": 8, "n_views": 1, "has_ground_truth": True,
                "fps": 5.0, "fps_tolerant": False,
            },
        },
        "exclude_globs": [], "robots": {},
    }
    raw.update(overrides)
    return parse_data_spec(raw)


def _write_cell(root, *, cache="dense", robot="airbot_mmk2", view="multiview",
                generator="ctrlworld", horizon="makovian"):
    cell_dir = os.path.join(str(root), cache, robot, view, generator, horizon)
    os.makedirs(cell_dir, exist_ok=True)
    card = {"cache": cache, "robot": robot, "view": view, "generator": generator,
           "horizon": horizon, "embodiment": "humanoid", "iter": None}
    with open(os.path.join(cell_dir, CELL_CARD_NAME), "w") as f:
        json.dump(card, f)
    return cell_dir


class TestVerifyLayout:
    def test_correctly_formatted_clips_pass(self, tmp_path):
        cell_dir = _write_cell(tmp_path)
        ep = os.path.join(cell_dir, "episode_0000")
        _write_mp4(os.path.join(ep, "pred.mp4"), w=8, h=8, fps=5)
        _write_mp4(os.path.join(ep, "gt.mp4"), w=8, h=8, fps=5)

        canon = CanonicalLayout(str(tmp_path))
        report = verify_layout(canon, _data_spec())
        assert report.ok
        assert report.n_episodes == 1
        assert report.n_clips_checked == 2

    def test_wrong_resolution_is_a_hard_error_naming_the_clip(self, tmp_path):
        cell_dir = _write_cell(tmp_path)
        ep = os.path.join(cell_dir, "episode_0000")
        _write_mp4(os.path.join(ep, "pred.mp4"), w=16, h=16, fps=5)  # declared: 8x8
        _write_mp4(os.path.join(ep, "gt.mp4"), w=16, h=16, fps=5)

        canon = CanonicalLayout(str(tmp_path))
        report = verify_layout(canon, _data_spec())
        assert not report.ok
        bad = [p for p in report.problems if "resolution" in p.reason]
        assert bad and bad[0].path.endswith("pred.mp4")

    def test_wrong_fps_is_a_hard_error_unless_fps_tolerant(self, tmp_path):
        cell_dir = _write_cell(tmp_path)
        ep = os.path.join(cell_dir, "episode_0000")
        _write_mp4(os.path.join(ep, "pred.mp4"), w=8, h=8, fps=30)  # declared: 5.0
        _write_mp4(os.path.join(ep, "gt.mp4"), w=8, h=8, fps=30)

        canon = CanonicalLayout(str(tmp_path))
        strict_report = verify_layout(canon, _data_spec())
        assert not strict_report.ok
        assert any("fps" in p.reason for p in strict_report.problems)

        tolerant_spec = _data_spec()
        tolerant_spec.generators["ctrlworld"] = tolerant_spec.generators["ctrlworld"].__class__(
            **{**tolerant_spec.generators["ctrlworld"].__dict__, "fps_tolerant": True})
        tolerant_report = verify_layout(canon, tolerant_spec)
        assert not any("fps" in p.reason for p in tolerant_report.problems)

    def test_mismatched_pred_gt_frame_count_is_a_hard_error(self, tmp_path):
        cell_dir = _write_cell(tmp_path)
        ep = os.path.join(cell_dir, "episode_0000")
        _write_mp4(os.path.join(ep, "pred.mp4"), w=8, h=8, fps=5, dur=0.4)
        _write_mp4(os.path.join(ep, "gt.mp4"), w=8, h=8, fps=5, dur=1.0)  # more frames

        canon = CanonicalLayout(str(tmp_path))
        report = verify_layout(canon, _data_spec())
        assert not report.ok
        assert any("frame count mismatch" in p.reason for p in report.problems)

    def test_missing_gt_when_generator_expects_one_is_a_hard_error(self, tmp_path):
        cell_dir = _write_cell(tmp_path)
        ep = os.path.join(cell_dir, "episode_0000")
        _write_mp4(os.path.join(ep, "pred.mp4"), w=8, h=8, fps=5)  # no gt.mp4

        canon = CanonicalLayout(str(tmp_path))
        report = verify_layout(canon, _data_spec())
        assert not report.ok
        assert any(p.path.endswith("gt.mp4") for p in report.problems)

    def test_broken_symlink_is_detected(self, tmp_path):
        cell_dir = _write_cell(tmp_path)
        ep = os.path.join(cell_dir, "episode_0000")
        os.makedirs(ep, exist_ok=True)
        os.symlink("/nonexistent/pred.mp4", os.path.join(ep, PRED_NAME))
        os.makedirs(ep, exist_ok=True)  # gt required, but pred already broken -> report both

        canon = CanonicalLayout(str(tmp_path))
        report = verify_layout(canon, _data_spec())
        assert not report.ok
        assert any("broken symlink" in p.reason for p in report.problems)

    def test_unreadable_file_reports_ffprobe_failure_not_a_crash(self, tmp_path):
        cell_dir = _write_cell(tmp_path)
        ep = os.path.join(cell_dir, "episode_0000")
        os.makedirs(ep, exist_ok=True)
        with open(os.path.join(ep, "pred.mp4"), "wb") as f:
            f.write(b"not a real video file")
        _write_mp4(os.path.join(ep, "gt.mp4"), w=8, h=8, fps=5)

        canon = CanonicalLayout(str(tmp_path))
        report = verify_layout(canon, _data_spec())
        assert not report.ok
        assert any("unreadable" in p.reason for p in report.problems)
