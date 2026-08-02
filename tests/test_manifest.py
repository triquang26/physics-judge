"""bench.manifest: pluggable discovery, D3-safe row building, pairing checks.

``verify_manifest`` operates on plain row dicts and needs no real media, so
those tests run in the CPU tier. ``build_manifest``/``save_manifest`` need a
real probeable file (they call into ``kinescore.video.probe``), so they are
marked ``ffmpeg`` and synthesize a tiny mp4 with imageio, mirroring
``tests/test_video_probe.py``.
"""
from __future__ import annotations

import json

import pytest

from kinescore.bench.manifest import (
    ROW_KEYS,
    DiscoveredClip,
    build_manifest,
    load_manifest,
    save_manifest,
    verify_manifest,
)
from kinescore.core.clip import TimebaseError


def _row(pair_key, role, w=64, h=64, method="m", family="f", codec="h264",
        dt=0.1):
    return {"method": method, "family": family, "episode": pair_key.split("/")[-1],
           "role": role, "path": f"/tmp/{pair_key}_{role}.mp4",
           "n_frames": 10, "fps": 1.0 / dt, "w": w, "h": h, "dt": dt,
           "pair_key": pair_key, "fps_probed": 1.0 / dt, "dt_source": "ffprobe",
           "codec": codec, "sha1": None, "view_layout": "1x?:unnamed"}


class TestVerifyManifest:
    def test_matching_pair_has_no_mismatch(self):
        rows = [_row("m/ep0", "gt"), _row("m/ep0", "pred")]
        report = verify_manifest(rows)
        assert report["ok"] is True
        assert report["n_pairs"] == 1
        assert report["n_mismatches"] == 0
        assert report["pairs_per_method"] == {"m": 1}

    def test_wh_mismatch_is_reported(self):
        rows = [_row("m/ep0", "gt", w=64, h=64), _row("m/ep0", "pred", w=32, h=32)]
        report = verify_manifest(rows)
        assert report["ok"] is False
        assert report["n_mismatches"] == 1
        assert report["mismatches"][0]["wh_ok"] is False

    def test_codec_mismatch_is_reported(self):
        rows = [_row("m/ep0", "gt", codec="h264"), _row("m/ep0", "pred", codec="vp9")]
        report = verify_manifest(rows)
        assert report["ok"] is False
        assert report["mismatches"][0]["codec_ok"] is False

    def test_dt_mismatch_is_reported(self):
        # RATE_POLICY layer 1 ("paired"): dt cancels only when the pair
        # genuinely shares a timebase. A gt probed at 10 fps (dt=0.1) paired
        # with a pred probed at 16 fps (dt=0.0625) -- the literal
        # dreamdojo-vs-dreamgen rate mismatch this check exists to catch --
        # must be a hard mismatch, not silently accepted.
        rows = [_row("m/ep0", "gt", dt=0.1), _row("m/ep0", "pred", dt=1.0 / 16.0)]
        report = verify_manifest(rows)
        assert report["ok"] is False
        assert report["n_mismatches"] == 1
        mismatch = report["mismatches"][0]
        assert mismatch["dt_ok"] is False
        assert mismatch["gt_dt"] == pytest.approx(0.1)
        assert mismatch["pred_dt"] == pytest.approx(1.0 / 16.0)
        # wh/codec were fine -- only dt is the reported cause.
        assert mismatch["wh_ok"] is True
        assert mismatch["codec_ok"] is True

    def test_matching_dt_within_tolerance_is_not_a_mismatch(self):
        # Container fps rounding (e.g. 29.97 vs 30) must not false-positive.
        rows = [_row("m/ep0", "gt", dt=1.0 / 30.0),
               _row("m/ep0", "pred", dt=1.0 / 29.97)]
        report = verify_manifest(rows)
        assert report["ok"] is True
        assert report["n_mismatches"] == 0

    def test_dt_tolerance_is_configurable(self):
        rows = [_row("m/ep0", "gt", dt=0.100), _row("m/ep0", "pred", dt=0.102)]
        # 2% relative difference: passes a loose tolerance, fails a tight one.
        assert verify_manifest(rows, dt_rel_tol=0.05)["ok"] is True
        assert verify_manifest(rows, dt_rel_tol=0.01)["ok"] is False

    def test_dt_and_wh_can_both_be_reported_in_one_mismatch(self):
        rows = [_row("m/ep0", "gt", w=64, h=64, dt=0.1),
               _row("m/ep0", "pred", w=32, h=32, dt=1.0 / 16.0)]
        report = verify_manifest(rows)
        assert report["ok"] is False
        mismatch = report["mismatches"][0]
        assert mismatch["wh_ok"] is False
        assert mismatch["dt_ok"] is False

    def test_unpaired_role_is_not_checked_or_counted_as_a_pair(self):
        # A lone "real" role clip (no gt/pred counterpart) isn't a pair at
        # all -- must not show up as a mismatch or inflate n_pairs.
        rows = [_row("m/ep0", "real")]
        report = verify_manifest(rows)
        assert report["n_pairs"] == 0
        assert report["n_mismatches"] == 0

    def test_pairing_is_not_hardcoded_to_the_dreamdojo_family(self):
        # Generalisation over the source: any family with gt+pred roles is
        # checked, not just one hardcoded family name.
        rows = [_row("x/ep0", "gt", family="brand_new_family"),
               _row("x/ep0", "pred", family="brand_new_family")]
        report = verify_manifest(rows)
        assert report["n_pairs"] == 1
        assert report["ok"] is True

    def test_n_rows_by_family(self):
        rows = [_row("m/ep0", "gt"), _row("m/ep0", "pred"), _row("m/ep1", "real")]
        report = verify_manifest(rows)
        assert report["n_rows"] == 3
        assert report["n_rows_by_family"] == {"f": 3}


class TestSaveAndLoadManifestJsonFallback:
    def test_json_round_trip(self, tmp_path, monkeypatch):
        rows = [_row("m/ep0", "gt"), _row("m/ep0", "pred")]

        # Force the JSON fallback path even though pandas/pyarrow are
        # installed in this environment, to prove the CPU-tier path works
        # without them.
        real_import = __import__

        def fake_import(name, *a, **k):
            if name == "pandas":
                raise ImportError("pandas intentionally unavailable in test")
            return real_import(name, *a, **k)

        monkeypatch.setattr("builtins.__import__", fake_import)
        paths = save_manifest(rows, str(tmp_path))
        assert paths["manifest"].endswith(".json")
        monkeypatch.undo()

        loaded = json.loads((tmp_path / "bench_manifest.json").read_text())
        assert loaded == rows


@pytest.mark.ffmpeg
class TestBuildManifestWithRealMedia:
    @pytest.fixture()
    def two_clip_pair(self, tmp_path):
        iio = pytest.importorskip("imageio.v3")
        import numpy as np

        def write(path, n=4, fps=10.0):
            frames = np.zeros((n, 16, 16, 3), dtype="uint8")
            iio.imwrite(str(path), frames, fps=fps, codec="libx264")

        gt = tmp_path / "gt.mp4"
        pred = tmp_path / "pred.mp4"
        write(gt)
        write(pred)
        return str(gt), str(pred)

    def test_plugin_discovers_a_pair_and_probes_correctly(self, two_clip_pair):
        gt_path, pred_path = two_clip_pair

        def plugin():
            yield DiscoveredClip(method="demo", family="demo_family",
                                 episode="ep0", role="gt", path=gt_path,
                                 pair_key="demo/ep0")
            yield DiscoveredClip(method="demo", family="demo_family",
                                 episode="ep0", role="pred", path=pred_path,
                                 pair_key="demo/ep0")

        rows = build_manifest([plugin])
        assert len(rows) == 2
        assert set(rows[0]) == set(ROW_KEYS)
        for row in rows:
            assert row["fps"] == pytest.approx(10.0, abs=0.05)
            assert row["dt_source"] == "ffprobe"

        report = verify_manifest(rows)
        assert report["ok"] is True

    def test_bad_fps_hint_is_skipped_not_fatal_by_default(self, two_clip_pair):
        gt_path, _ = two_clip_pair

        def plugin():
            yield DiscoveredClip(method="demo", family="demo_family",
                                 episode="ep0", role="gt", path=gt_path,
                                 pair_key="demo/ep0", fps_hint=999.0)

        rows = build_manifest([plugin])  # default on_error="skip"
        assert rows == []

    def test_bad_fps_hint_raises_with_on_error_raise(self, two_clip_pair):
        gt_path, _ = two_clip_pair

        def plugin():
            yield DiscoveredClip(method="demo", family="demo_family",
                                 episode="ep0", role="gt", path=gt_path,
                                 pair_key="demo/ep0", fps_hint=999.0)

        with pytest.raises(TimebaseError):
            build_manifest([plugin], on_error="raise")

    def test_save_and_load_manifest_parquet_round_trip(self, two_clip_pair, tmp_path):
        pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")
        gt_path, pred_path = two_clip_pair

        def plugin():
            yield DiscoveredClip(method="demo", family="demo_family",
                                 episode="ep0", role="gt", path=gt_path,
                                 pair_key="demo/ep0")

        rows = build_manifest([plugin])
        paths = save_manifest(rows, str(tmp_path))
        assert paths["manifest"].endswith(".parquet")
        loaded = load_manifest(paths["manifest"])
        assert len(loaded) == len(rows)
        assert loaded[0]["path"] == rows[0]["path"]
