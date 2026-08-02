"""``video/viewpoint.py`` -- the exterior-vs-wrist camera-visibility classifier.

``flow_features``/``TestClassifyViewpointLogic`` run on synthetic numpy
frames (CPU-only, no video I/O). ``TestClassifyViewpointEndToEnd`` decodes a
real synthesized mp4 through the full pipeline (``cv2.VideoCapture`` +
ffmpeg), matching ``tests/test_video_probe.py``'s pattern for that.
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import kinescore.video.viewpoint as vp_mod  # noqa: E402
from kinescore.video.viewpoint import (  # noqa: E402
    WRIST_MOVING_FRAC_THRESHOLD,
    classify_viewpoint,
    flow_features,
)


def _textured_frame(size=(96, 128), seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=size, dtype=np.uint8)


class TestFlowFeatures:
    def test_zero_frames_returns_nones(self):
        assert flow_features([]) == (None, None, 0)

    def test_one_frame_returns_nones(self):
        assert flow_features([_textured_frame()]) == (None, None, 0)

    def test_static_background_with_a_small_moving_patch_has_low_moving_frac(self):
        # Simulates a fixed exterior camera: most of the frame (background)
        # never changes; only a small "arm" patch moves each step.
        base = _textured_frame()
        frames = []
        for i in range(5):
            f = base.copy()
            f[40:50, 40 + i:50 + i] = 255
            frames.append(f)
        mean_mag, moving_frac, n_pairs = flow_features(frames)
        assert n_pairs == 4
        assert mean_mag is not None
        assert moving_frac < WRIST_MOVING_FRAC_THRESHOLD

    def test_whole_frame_translation_has_high_moving_frac(self):
        # Simulates a wrist/ego camera: the ENTIRE frame (incl. background)
        # shifts every step.
        base = _textured_frame()
        frames = [np.roll(base, shift=4 * i, axis=1) for i in range(5)]
        mean_mag, moving_frac, n_pairs = flow_features(frames)
        assert n_pairs == 4
        assert moving_frac >= WRIST_MOVING_FRAC_THRESHOLD


class TestClassifyViewpointLogic:
    """Exercises the threshold/verdict logic in isolation, without decoding
    a real file -- ``sample_gray_frames``/``flow_features`` are swapped for
    deterministic stand-ins."""

    def test_no_decodable_frames_gives_verdict_none_not_a_guess(self, monkeypatch):
        monkeypatch.setattr(vp_mod, "sample_gray_frames", lambda *a, **k: [])
        v = classify_viewpoint("nonexistent.mp4")
        assert v.verdict is None
        assert v.moving_frac is None
        assert v.mean_mag is None
        assert v.n_pairs == 0
        assert v.path == "nonexistent.mp4"

    def test_moving_frac_at_or_above_threshold_is_wrist(self, monkeypatch):
        monkeypatch.setattr(vp_mod, "sample_gray_frames", lambda *a, **k: [0, 1])
        monkeypatch.setattr(vp_mod, "flow_features", lambda frames, **k: (1.0, 0.42, 1))
        v = classify_viewpoint("clip.mp4", threshold=0.42)
        assert v.verdict == "wrist"
        assert v.moving_frac == pytest.approx(0.42)

    def test_moving_frac_below_threshold_is_exterior(self, monkeypatch):
        monkeypatch.setattr(vp_mod, "sample_gray_frames", lambda *a, **k: [0, 1])
        monkeypatch.setattr(vp_mod, "flow_features", lambda frames, **k: (1.0, 0.1, 1))
        v = classify_viewpoint("clip.mp4", threshold=0.42)
        assert v.verdict == "exterior"

    def test_default_threshold_is_the_validated_constant(self, monkeypatch):
        monkeypatch.setattr(vp_mod, "sample_gray_frames", lambda *a, **k: [0, 1])
        monkeypatch.setattr(
            vp_mod, "flow_features",
            lambda frames, **k: (1.0, WRIST_MOVING_FRAC_THRESHOLD - 0.001, 1))
        v = classify_viewpoint("clip.mp4")
        assert v.verdict == "exterior"


@pytest.mark.ffmpeg
class TestClassifyViewpointEndToEnd:
    """Full pipeline against a real, synthesized mp4 -- requires ffmpeg."""

    def _write(self, path, frames) -> None:
        iio = pytest.importorskip("imageio.v3")
        rgb = np.stack([np.stack([f, f, f], axis=-1) for f in frames])
        iio.imwrite(str(path), rgb, fps=10.0, codec="libx264",
                   plugin="pyav" if self._has_pyav() else None)

    @staticmethod
    def _has_pyav() -> bool:
        try:
            import av  # noqa: F401
            return True
        except ImportError:
            return False

    def test_static_exterior_style_clip_classifies_exterior(self, tmp_path):
        base = _textured_frame(size=(96, 128), seed=1)
        frames = []
        for i in range(10):
            f = base.copy()
            f[40:50, 40 + i:50 + i] = 255  # small moving "arm" patch only
            frames.append(f)
        path = tmp_path / "exterior.mp4"
        self._write(path, frames)

        v = classify_viewpoint(str(path))
        assert v.n_pairs > 0
        assert v.verdict == "exterior"

    def test_globally_translating_wrist_style_clip_classifies_wrist(self, tmp_path):
        base = _textured_frame(size=(96, 128), seed=2)
        frames = [np.roll(base, shift=5 * i, axis=1) for i in range(10)]
        path = tmp_path / "wrist.mp4"
        self._write(path, frames)

        v = classify_viewpoint(str(path))
        assert v.n_pairs > 0
        assert v.verdict == "wrist"
