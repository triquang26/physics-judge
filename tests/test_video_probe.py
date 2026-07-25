"""ffprobe + resolve_timebase against a real, synthesized mp4.

Requires the ``ffmpeg``/``ffprobe`` binaries (marker ``ffmpeg``) and the
``video`` extra (``imageio``) to synthesize the fixture; both are skipped for
via ``pytest.importorskip``/marker rather than failing the CPU tier.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.ffmpeg

iio = pytest.importorskip("imageio.v3")

from kinescore.core.clip import TimebaseError  # noqa: E402
from kinescore.video.probe import ffprobe, resolve_timebase  # noqa: E402


def _write_mp4(path, n_frames: int, fps: float, size=(32, 32)) -> None:
    h, w = size
    frames = np.zeros((n_frames, h, w, 3), dtype=np.uint8)
    for i in range(n_frames):
        frames[i] = (i * 40) % 256  # distinct-ish frames
    iio.imwrite(str(path), frames, fps=fps, codec="libx264",
               plugin="pyav" if _has_pyav() else None)


def _has_pyav() -> bool:
    try:
        import av  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture()
def three_frame_10fps_mp4(tmp_path):
    path = tmp_path / "clip.mp4"
    _write_mp4(path, n_frames=3, fps=10.0)
    return str(path)


class TestFfprobe:
    def test_probes_fps_dims_and_frame_count(self, three_frame_10fps_mp4):
        p = ffprobe(three_frame_10fps_mp4)
        assert p["fps"] == pytest.approx(10.0, abs=0.05)
        assert p["n_frames"] == 3
        assert p["w"] == 32
        assert p["h"] == 32
        assert p["codec"]

    def test_falls_back_to_count_frames_when_nb_frames_missing(
            self, three_frame_10fps_mp4, monkeypatch):
        # Simulate a container whose nb_frames tag is absent/N/A by forcing
        # the fast-path branch to look like it returned nothing, so ffprobe()
        # must take the "-count_frames" exact-decode fallback instead.
        import subprocess

        real_run = subprocess.run
        calls = []

        def fake_run(args, **kwargs):
            calls.append(args)
            result = real_run(args, **kwargs)
            if "-count_frames" not in args and "nb_frames" in " ".join(args):
                # Blank out nb_frames in the fast-path output to force fallback.
                text = result.stdout
                lines = [ln for ln in text.splitlines()
                        if not ln.startswith("nb_frames=")]
                lines.append("nb_frames=N/A")
                result.stdout = "\n".join(lines)
            return result

        monkeypatch.setattr(subprocess, "run", fake_run)
        p = ffprobe(three_frame_10fps_mp4)
        assert p["n_frames"] == 3
        assert any("-count_frames" in c for c in calls)


class TestResolveTimebase:
    def test_no_override_trusts_the_probe(self, three_frame_10fps_mp4):
        clip = resolve_timebase(three_frame_10fps_mp4)
        assert clip.fps == pytest.approx(10.0, abs=0.05)
        assert clip.dt == pytest.approx(0.1, abs=0.001)
        assert clip.n_frames == 3
        assert clip.dt_source == "ffprobe"

    def test_matching_table_value_is_accepted_and_recorded(
            self, three_frame_10fps_mp4):
        clip = resolve_timebase(three_frame_10fps_mp4, fps_table=10.0)
        assert clip.dt_source == "table"
        assert clip.fps == pytest.approx(10.0)

    def test_raises_when_table_disagrees_with_probe(self, three_frame_10fps_mp4):
        # This is the D3 regression test: a stale config table claiming
        # 16fps for a probed-10fps file must be a hard error, not a silent
        # override in either direction.
        with pytest.raises(TimebaseError, match="10|16"):
            resolve_timebase(three_frame_10fps_mp4, fps_table=16.0)

    def test_fps_and_dt_args_are_mutually_exclusive(self, three_frame_10fps_mp4):
        with pytest.raises(ValueError):
            resolve_timebase(three_frame_10fps_mp4, fps_arg=10.0, dt_arg=0.1)

    def test_dt_arg_within_tolerance_is_accepted(self, three_frame_10fps_mp4):
        clip = resolve_timebase(three_frame_10fps_mp4, dt_arg=0.1)
        assert clip.dt_source == "dt_arg"
        assert clip.fps == pytest.approx(10.0)
