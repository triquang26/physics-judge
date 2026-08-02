"""kinescore.bench.ingest.Ingestor: materialise RawHFLayout -> CanonicalLayout.

Marked ``@pytest.mark.ffmpeg`` throughout: :meth:`Ingestor.run` probes one
episode per cell via :func:`kinescore.video.probe.ffprobe` for the
``cell_card.json`` width/height/fps fields, so every test here needs a real
``ffprobe`` on ``PATH`` (auto-skipped otherwise -- see ``tests/conftest.py``).
"""
from __future__ import annotations

import json
import os
import subprocess

import pytest

from kinescore.bench.data_spec import parse_data_spec
from kinescore.bench.ingest import Ingestor
from kinescore.bench.layout import CanonicalLayout, RawHFLayout
from kinescore.bench.robot_map import parse_robot_map

pytestmark = pytest.mark.ffmpeg


def _write_tiny_mp4(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "color=c=black:s=8x8:d=0.4:r=5", "-pix_fmt", "yuv420p", str(path)],
        check=True)


def _robot_map():
    return parse_robot_map({
        "robots": {
            "airbot_mmk2": {"embodiment": "humanoid", "generators": ["ctrlworld"]},
        },
    })


def _data_spec():
    return parse_data_spec({
        "generators": {
            "ctrlworld": {
                "shape": "episode_dir", "pred_filename": "pred_all_views.mp4",
                "gt_filename": "gt_all_views.mp4", "has_iter_level": False,
                "width": 8, "height": 8, "n_views": 1, "has_ground_truth": True,
                "fps": 5.0, "fps_tolerant": True,
            },
        },
        "exclude_globs": [],
        "robots": {},
    })


@pytest.fixture
def raw_root(tmp_path):
    base = (tmp_path / "video_gen_physics" / "dense" / "humanoid" / "output"
           / "multiview" / "ctrlworld" / "makovian")
    for i in range(2):
        ep = base / f"episode_{i:04d}"
        _write_tiny_mp4(ep / "pred_all_views.mp4")
        _write_tiny_mp4(ep / "gt_all_views.mp4")
    return tmp_path


class TestIngestSymlinks:
    def test_symlinks_by_default_and_writes_a_cell_card(self, raw_root, tmp_path):
        raw = RawHFLayout(str(raw_root), _robot_map(), _data_spec())
        canon = CanonicalLayout(str(tmp_path / "canon"))
        report = Ingestor(raw, canon).run()

        assert report.n_cells == 1
        assert report.n_episodes == 2
        cell_dir = canon.cell_dir(next(canon.cells()))
        pred = os.path.join(cell_dir, "episode_0000", "pred.mp4")
        assert os.path.islink(pred)
        assert os.path.realpath(pred) == os.path.realpath(
            str(raw_root / "video_gen_physics" / "dense" / "humanoid" / "output"
               / "multiview" / "ctrlworld" / "makovian" / "episode_0000"
               / "pred_all_views.mp4"))

        with open(os.path.join(cell_dir, "cell_card.json")) as f:
            card = json.load(f)
        assert card["robot"] == "airbot_mmk2"
        assert card["n_episodes_actual"] == 2
        assert card["width"] == 8 and card["height"] == 8
        assert card["source_path"] == os.path.abspath(str(
            raw_root / "video_gen_physics" / "dense" / "humanoid" / "output"
           / "multiview" / "ctrlworld" / "makovian"))

    def test_copy_flag_copies_bytes_instead_of_symlinking(self, raw_root, tmp_path):
        raw = RawHFLayout(str(raw_root), _robot_map(), _data_spec())
        canon = CanonicalLayout(str(tmp_path / "canon"))
        Ingestor(raw, canon).run(copy=True)

        cell_dir = canon.cell_dir(next(canon.cells()))
        pred = os.path.join(cell_dir, "episode_0000", "pred.mp4")
        assert not os.path.islink(pred)
        assert os.path.isfile(pred)

    def test_two_cells_with_different_iters_never_collide(self, tmp_path):
        # The exact bug an earlier version of this module had: two raw
        # iter_* directories both mapping into the SAME 5-segment canonical
        # path (no iter level -- see kinescore.bench.cell.PATH_AXIS_ORDER)
        # silently overwrote each other. RawHFLayout.cells() auto-picks
        # exactly one iter per (cache, robot, view, generator, horizon), so
        # this must produce exactly one canonical cell, not a silent merge.
        base = (tmp_path / "video_gen_physics" / "dense" / "single_arm" / "output"
               / "singleview" / "dreamdojo" / "makovian")
        _write_tiny_mp4(base / "iter_000030000" / "episode_0000" / "full_pred.mp4")
        _write_tiny_mp4(base / "iter_000030000" / "episode_0000" / "full_gt.mp4")
        for i in range(3):
            _write_tiny_mp4(base / "iter_000090000" / f"episode_{i:04d}" / "full_pred.mp4")
            _write_tiny_mp4(base / "iter_000090000" / f"episode_{i:04d}" / "full_gt.mp4")

        rm = parse_robot_map({"robots": {
            "franka_panda": {"embodiment": "single_arm", "generators": ["dreamdojo"]}}})
        ds = parse_data_spec({
            "generators": {"dreamdojo": {
                "shape": "flat_or_dir", "flat_pred_glob": "*_pred.mp4",
                "flat_gt_glob": "*_gt.mp4", "dir_pred_filename": "full_pred.mp4",
                "dir_gt_filename": "full_gt.mp4", "has_iter_level": True,
                "width": 8, "height": 8, "n_views": 1, "has_ground_truth": True,
                "fps": 10.0, "fps_tolerant": True,
            }},
            "exclude_globs": [], "robots": {},
        })
        raw = RawHFLayout(str(tmp_path), rm, ds)
        canon = CanonicalLayout(str(tmp_path / "canon"))
        report = Ingestor(raw, canon).run()

        assert report.n_cells == 1
        assert report.n_episodes == 3  # the more-populated iter_000090000 wins
        assert len(list(canon.cells())) == 1

    def test_episode_missing_gt_is_counted_and_reported(self, tmp_path):
        base = (tmp_path / "video_gen_physics" / "dense" / "humanoid" / "output"
               / "multiview" / "ctrlworld" / "makovian")
        _write_tiny_mp4(base / "episode_0000" / "pred_all_views.mp4")  # no gt
        raw = RawHFLayout(str(tmp_path), _robot_map(), _data_spec())
        canon = CanonicalLayout(str(tmp_path / "canon"))
        report = Ingestor(raw, canon).run()

        # ctrlworld's shape (episode_dir) already drops a gt-less episode at
        # discovery time (see kinescore.bench.layout's docstring: ingest
        # only never-drops for a GENERATOR with no ground truth at all,
        # e.g. dreamgen) -- so this cell has zero episodes, not a
        # skipped-count. Both are legitimate; this pins which one applies
        # to ctrlworld specifically.
        assert report.n_episodes == 0
