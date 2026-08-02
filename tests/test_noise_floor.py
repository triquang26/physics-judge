"""``kinescore.bench.noise_floor``: the paired re-encode null-delta ruler.

Default tier is CPU-only, network-free, ffmpeg-free: the deterministic CRF
schedule, the summary statistics, the ``build_noise_floor`` orchestration
against a fake ``score_fn`` (no ffmpeg, no torch), and the ``below_floor``
flagging helper. Anything that actually shells out to ``ffmpeg``
(:func:`kinescore.bench.noise_floor.reencode_crf`, and
:func:`build_noise_floor` used end-to-end against a real video) is marked
``@pytest.mark.ffmpeg``.
"""
from __future__ import annotations

import math
import os
import subprocess

import numpy as np
import pytest

from kinescore.bench.noise_floor import (
    below_floor,
    build_noise_floor,
    episode_crf,
    reencode_crf,
    summarize_null_deltas,
)

# ===========================================================================
# episode_crf: deterministic per-episode schedule
# ===========================================================================

def test_episode_crf_matches_source_formula():
    # verbatim source: crf = base + (ep_num % mod), ep_num from the first
    # run of digits found in the episode id (33_noise_floor.py:52-54, :120)
    assert episode_crf("episode_000007", base=23, mod=12) == 23 + (7 % 12)
    assert episode_crf("episode_000013", base=23, mod=12) == 23 + (13 % 12)
    assert episode_crf("13", base=23, mod=12) == 23 + (13 % 12)


def test_episode_crf_no_digits_falls_back_to_base():
    assert episode_crf("no_digits_here", base=23, mod=12) == 23


def test_episode_crf_is_deterministic_and_pure():
    a = [episode_crf(f"episode_{i:06d}") for i in range(30)]
    b = [episode_crf(f"episode_{i:06d}") for i in range(30)]
    assert a == b


def test_episode_crf_default_matches_source_defaults():
    # source main()'s argparse defaults: --crf_base 23 --crf_mod 12
    assert episode_crf("episode_000000") == 23
    assert episode_crf("episode_000012") == 23  # 12 % 12 == 0


# ===========================================================================
# summarize_null_deltas: pure statistics
# ===========================================================================

def test_summarize_null_deltas_matches_hand_computation():
    deltas = [1.0, -2.0, 3.0, -0.5, 2.5]
    got = summarize_null_deltas(deltas)
    a = np.asarray(deltas)
    assert got["n"] == 5
    assert got["null_mean"] == pytest.approx(float(a.mean()))
    assert got["null_median"] == pytest.approx(float(np.median(a)))
    assert got["null_std"] == pytest.approx(float(a.std()))
    assert got["null_p95"] == pytest.approx(float(np.percentile(np.abs(a), 95)))
    assert got["null_abs_median"] == pytest.approx(float(np.median(np.abs(a))))


def test_summarize_null_deltas_drops_non_finite():
    got = summarize_null_deltas([1.0, float("nan"), 3.0, float("inf"), -1.0])
    assert got["n"] == 3  # only 1.0, 3.0, -1.0 survive


def test_summarize_null_deltas_empty_is_all_nan_n_zero():
    got = summarize_null_deltas([])
    assert got["n"] == 0
    for k in ("null_mean", "null_median", "null_std", "null_p95", "null_abs_median"):
        assert math.isnan(got[k])


def test_summarize_null_deltas_all_non_finite_is_also_n_zero():
    got = summarize_null_deltas([float("nan"), float("inf"), float("-inf")])
    assert got["n"] == 0


# ===========================================================================
# build_noise_floor: orchestration against a fake score_fn (no ffmpeg)
# ===========================================================================

class _FakeEncoder:
    """Stands in for ffmpeg: 'encodes' by recording (src, crf) pairs and
    touching an empty file at dst, so build_noise_floor's file-exists /
    cleanup logic is exercised without a real video or subprocess."""

    def __init__(self):
        self.calls: list[tuple[str, str, int]] = []

    def __call__(self, src: str, dst: str, crf: int, pix_fmt: str = "yuv420p") -> None:
        self.calls.append((src, dst, crf))
        with open(dst, "wb"):
            pass


def _fake_score_fn(deterministic_bias: dict[str, float]):
    """score_fn(path, dt) -> {metric: value}; value depends only on whether
    'reenc' is in the path, so re-scoring the same original path twice
    (once as orig via row['scores'] absence, once again by mistake) would be
    visible as a test failure if it ever happened."""

    def _fn(path: str, dt: float) -> dict[str, float]:
        is_reenc = "crf" in os.path.basename(path)
        out = {}
        for k, bias in deterministic_bias.items():
            out[k] = 10.0 + (bias if is_reenc else 0.0)
        return out

    return _fn


def test_build_noise_floor_computes_null_deltas_against_fake_encoder(
    tmp_path, monkeypatch
):
    encoder = _FakeEncoder()
    monkeypatch.setattr("kinescore.bench.noise_floor.reencode_crf", encoder)

    rows = [
        {"episode": f"episode_{i:06d}", "path": f"/data/ep{i}.mp4", "dt": 0.1}
        for i in range(5)
    ]
    score_fn = _fake_score_fn({"mean_jerk_mps3": 2.0})

    result = build_noise_floor(
        rows, score_fn, metrics=("mean_jerk_mps3",),
        scratch_dir=str(tmp_path))

    assert result["n_clips"] == 5
    assert result["crf_rule"] == "23 + ep%12"
    assert len(encoder.calls) == 5  # one re-encode per row, no cache
    summary = result["summary"]["mean_jerk_mps3"]
    assert summary["n"] == 5
    # every null delta is exactly +2.0 by construction (the fake score_fn's bias)
    assert summary["null_mean"] == pytest.approx(2.0)
    assert summary["null_p95"] == pytest.approx(2.0)
    for pair in result["pairs"]:
        assert pair["nulldelta_mean_jerk_mps3"] == pytest.approx(2.0)
    # scratch files were cleaned up (build_noise_floor removes them after scoring)
    assert list(tmp_path.iterdir()) == []


def test_build_noise_floor_reuses_precomputed_original_scores(tmp_path, monkeypatch):
    """When a row carries `scores`, score_fn must NOT be called on the
    original path -- only on the re-encoded temp file."""
    encoder = _FakeEncoder()
    monkeypatch.setattr("kinescore.bench.noise_floor.reencode_crf", encoder)

    calls: list[str] = []

    def score_fn(path, dt):
        calls.append(path)
        return {"mean_jerk_mps3": 10.0 if "crf" not in path else 11.0}

    rows = [{"episode": "episode_000001", "path": "/data/ep1.mp4", "dt": 0.1,
            "scores": {"mean_jerk_mps3": 9.0}}]
    result = build_noise_floor(rows, score_fn, metrics=("mean_jerk_mps3",),
                               scratch_dir=str(tmp_path))
    assert "/data/ep1.mp4" not in calls  # original never re-scored
    assert len(calls) == 1               # only the re-encoded temp file
    assert result["pairs"][0]["orig_mean_jerk_mps3"] == pytest.approx(9.0)


def test_build_noise_floor_uses_reenc_cache_to_skip_repeat_work(tmp_path, monkeypatch):
    encoder = _FakeEncoder()
    monkeypatch.setattr("kinescore.bench.noise_floor.reencode_crf", encoder)
    score_fn = _fake_score_fn({"mean_jerk_mps3": 3.0})
    cache: dict = {}

    rows = [{"episode": "episode_000005", "path": "/data/same.mp4", "dt": 0.1}]
    build_noise_floor(rows, score_fn, metrics=("mean_jerk_mps3",),
                      scratch_dir=str(tmp_path), reenc_cache=cache)
    assert len(encoder.calls) == 1
    assert len(cache) == 1

    # Same path+episode again -> same crf -> cache hit, no second encode call.
    build_noise_floor(rows, score_fn, metrics=("mean_jerk_mps3",),
                      scratch_dir=str(tmp_path), reenc_cache=cache)
    assert len(encoder.calls) == 1  # unchanged


def test_build_noise_floor_creates_scratch_dir(tmp_path, monkeypatch):
    encoder = _FakeEncoder()
    monkeypatch.setattr("kinescore.bench.noise_floor.reencode_crf", encoder)
    scratch = tmp_path / "nested" / "scratch"
    assert not scratch.exists()
    build_noise_floor(
        [{"episode": "episode_000000", "path": "/data/e.mp4", "dt": 0.1}],
        _fake_score_fn({"mean_jerk_mps3": 1.0}),
        metrics=("mean_jerk_mps3",), scratch_dir=str(scratch))
    assert scratch.is_dir()


# ===========================================================================
# below_floor: "not conclusive" flagging, built on the existing noise_units
# ===========================================================================

def test_below_floor_flags_a_delta_smaller_than_the_floor():
    assert below_floor(1.0, floor=3.7) is True     # matches the page's ~3.7 floor
    assert below_floor(-1.0, floor=3.7) is True     # sign doesn't matter
    assert below_floor(5.5, floor=3.7) is False     # the page's own +5.5 headline tax


def test_below_floor_accepts_a_summarize_null_deltas_mapping():
    floor = summarize_null_deltas([1.0, 2.0, 3.7, -3.6, 0.5])
    assert below_floor(0.1, floor=floor) is True


def test_below_floor_exactly_at_floor_is_conclusive():
    # noise_units == 1.0 exactly -> not "below" (the ratio is >= 1, matching
    # noise_units's own ">1 means the tax exceeds the p95 noise" framing).
    assert below_floor(3.7, floor=3.7) is False


def test_below_floor_nan_floor_is_inconclusive():
    assert below_floor(5.5, floor=float("nan")) is True


# ===========================================================================
# ffmpeg-gated: the real reencode_crf shells out correctly
# ===========================================================================

@pytest.mark.ffmpeg
def test_reencode_crf_produces_a_playable_file(tmp_path):
    src = tmp_path / "src.mp4"
    dst = tmp_path / "dst.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
         "testsrc=size=64x64:rate=10:duration=1", "-pix_fmt", "yuv420p", str(src)],
        check=True)
    reencode_crf(str(src), str(dst), crf=30)
    assert dst.is_file() and dst.stat().st_size > 0

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name", "-of", "default=nw=1:nk=1", str(dst)],
        capture_output=True, text=True, check=True)
    assert probe.stdout.strip() == "h264"
