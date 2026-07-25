"""``kinescore score``'s timebase cross-check: --fps/--dt mutual exclusion,
and refusing to let either silently override a probe that disagrees.

``kinescore.cli._scoring`` is the shared logic ``score`` and
``reference build`` both use (see that module's docstring); testing it
directly here is testing exactly what both commands rely on, without needing
a full manifest/checkpoint/robot round trip.
"""
from __future__ import annotations

import pytest

from kinescore.core.clip import TimebaseError

pytestmark = pytest.mark.ffmpeg

iio = pytest.importorskip("imageio.v3")


def _write_mp4(path, n_frames: int, fps: float, size=(32, 32)) -> None:
    import numpy as np

    h, w = size
    frames = np.zeros((n_frames, h, w, 3), dtype="uint8")
    for i in range(n_frames):
        frames[i] = (i * 40) % 256
    iio.imwrite(str(path), frames, fps=fps, codec="libx264")


@pytest.fixture()
def ten_fps_clip(tmp_path):
    path = tmp_path / "clip.mp4"
    _write_mp4(path, n_frames=5, fps=10.0)
    return str(path)


class TestMutualExclusion:
    def test_fps_and_dt_together_raises_before_touching_the_file(self):
        from kinescore.cli._scoring import resolve_row_clip
        from kinescore.core.clip import ViewLayout

        # A path that does not exist: if this raised for any reason OTHER
        # than the mutual-exclusion check, it would be a probe error, not a
        # ValueError -- proving the check runs before ffprobe is even called.
        row = {"path": "/nonexistent/path/does/not/matter.mp4", "fps": 10.0}
        with pytest.raises(ValueError, match="mutually exclusive"):
            resolve_row_clip(row, fps=10.0, dt=0.1, view_layout=ViewLayout())

    def test_score_cli_rejects_fps_and_dt_together(self):
        from kinescore.cli.main import main

        with pytest.raises(SystemExit) as exc:
            main(["score", "--manifest", "m.parquet", "--robot", "synthetic_2r",
                 "--reader", "r.pt", "--out", "out", "--fps", "10", "--dt", "0.1"])
        assert exc.value.code != 0


class TestProbeContradiction:
    def test_fps_contradicting_the_probe_raises_timebase_error(self, ten_fps_clip):
        from kinescore.cli._scoring import resolve_row_clip
        from kinescore.core.clip import ViewLayout

        row = {"path": ten_fps_clip, "fps": 10.0}
        with pytest.raises(TimebaseError):
            resolve_row_clip(row, fps=30.0, dt=None, view_layout=ViewLayout())

    def test_dt_contradicting_the_probe_raises_timebase_error(self, ten_fps_clip):
        from kinescore.cli._scoring import resolve_row_clip
        from kinescore.core.clip import ViewLayout

        row = {"path": ten_fps_clip, "fps": 10.0}
        with pytest.raises(TimebaseError):
            resolve_row_clip(row, fps=None, dt=1.0, view_layout=ViewLayout())

    def test_fps_matching_the_probe_within_tolerance_succeeds(self, ten_fps_clip):
        from kinescore.cli._scoring import resolve_row_clip
        from kinescore.core.clip import ViewLayout

        row = {"path": ten_fps_clip, "fps": 10.0}
        clip = resolve_row_clip(row, fps=10.0, dt=None, view_layout=ViewLayout())
        assert clip.fps == pytest.approx(10.0, abs=0.05)
        assert clip.dt_source == "fps_arg"

    def test_no_override_trusts_the_probe(self, ten_fps_clip):
        from kinescore.cli._scoring import resolve_row_clip
        from kinescore.core.clip import ViewLayout

        row = {"path": ten_fps_clip, "fps": 10.0}
        clip = resolve_row_clip(row, fps=None, dt=None, view_layout=ViewLayout())
        assert clip.fps == pytest.approx(10.0, abs=0.05)
        assert clip.dt_source == "table"


class TestApplyResolvedTimebase:
    def test_rewrites_every_row_and_does_not_mutate_the_input(self, ten_fps_clip):
        from kinescore.cli._scoring import apply_resolved_timebase
        from kinescore.core.clip import ViewLayout

        rows = [{"path": ten_fps_clip, "fps": 10.0, "n_frames": 5, "w": 32, "h": 32}]
        out = apply_resolved_timebase(rows, fps=None, dt=None, view_layout=ViewLayout())
        assert out[0]["dt"] == pytest.approx(0.1, abs=1e-3)
        assert out[0] is not rows[0]
        assert "dt" not in rows[0]

    def test_contradicting_override_raises_for_the_whole_batch(self, ten_fps_clip):
        from kinescore.cli._scoring import apply_resolved_timebase
        from kinescore.core.clip import ViewLayout

        rows = [{"path": ten_fps_clip, "fps": 10.0, "n_frames": 5, "w": 32, "h": 32}]
        with pytest.raises(TimebaseError):
            apply_resolved_timebase(rows, fps=100.0, dt=None, view_layout=ViewLayout())
