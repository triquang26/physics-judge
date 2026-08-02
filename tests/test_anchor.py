"""``kinescore anchor build``: real-footage frame-rate/resolution matching.

The library logic (:func:`build_anchor`, :func:`probe_crf_context`,
:func:`reencode_anchor_clip`) lives in :mod:`kinescore.video.anchor`, tested
directly here without going through argparse at all -- that is the whole
point of ``cli/cmd_anchor.py`` being a thin shell around it. The CLI-only
concerns (argument parsing/dispatch, validation, provenance-JSON wiring) are
tested against ``cli.cmd_anchor`` separately, further down.

Default tier is CPU-only, network-free, ffmpeg-free: argument parsing/
dispatch, :func:`build_anchor`'s resume/skip/failure bookkeeping against a
fake encoder, and the CLI's own argument validation. Anything that shells
out to ``ffmpeg``/``ffprobe`` for real (:func:`probe_crf_context`,
:func:`reencode_anchor_clip`, and an end-to-end ``anchor build`` run) is
marked ``@pytest.mark.ffmpeg``.
"""
from __future__ import annotations

import json
import subprocess

import pytest

from kinescore.cli import cmd_anchor
from kinescore.video import anchor as video_anchor
from kinescore.video.anchor import (
    build_anchor,
    probe_crf_context,
    reencode_anchor_clip,
)

# ===========================================================================
# argparse wiring: the standard HELP / add_arguments / run trio
# ===========================================================================

def test_has_the_standard_cli_trio():
    assert isinstance(cmd_anchor.HELP, str) and cmd_anchor.HELP
    assert callable(cmd_anchor.add_arguments)
    assert callable(cmd_anchor.run)


def test_add_arguments_registers_a_build_action():
    import argparse

    parser = argparse.ArgumentParser()
    cmd_anchor.add_arguments(parser)
    args = parser.parse_args(
        ["build", "--real-glob", "*.mp4", "--out-dir", "/tmp/out",
         "--fps", "10", "--width", "640", "--height", "480"])
    assert args.anchor_action == "build"
    assert args.real_glob == "*.mp4"
    assert args.fps == 10.0
    assert args.crf == 23          # source default for both real_dm/real_dm16


def test_run_with_no_action_prints_usage_and_returns_2(capsys):
    import argparse

    parser = argparse.ArgumentParser()
    cmd_anchor.add_arguments(parser)
    args = parser.parse_args([])
    rc = cmd_anchor.run(args)
    assert rc == 2
    assert "usage" in capsys.readouterr().err


def test_build_action_requires_probe_clip_or_fps_width_height(tmp_path, capsys):
    import argparse

    parser = argparse.ArgumentParser()
    cmd_anchor.add_arguments(parser)
    args = parser.parse_args(
        ["build", "--real-glob", str(tmp_path / "*.mp4"), "--out-dir", str(tmp_path)])
    rc = cmd_anchor.run(args)
    assert rc == 2
    assert "probe-clip" in capsys.readouterr().err


def test_build_action_reports_when_glob_matches_nothing(tmp_path, capsys):
    import argparse

    parser = argparse.ArgumentParser()
    cmd_anchor.add_arguments(parser)
    args = parser.parse_args(
        ["build", "--real-glob", str(tmp_path / "nothing_here" / "*.mp4"),
         "--out-dir", str(tmp_path / "out"), "--fps", "10", "--width", "640",
         "--height", "480"])
    rc = cmd_anchor.run(args)
    assert rc == 1
    assert "no files matched" in capsys.readouterr().err


# ===========================================================================
# build_anchor: resume/skip/failure bookkeeping against a fake encoder
# ===========================================================================

def _touch(p):
    p.write_bytes(b"not really a video")


def test_build_anchor_encodes_every_source_once(tmp_path, monkeypatch):
    calls = []

    def fake_encode(src, dst, *, fps, width, height, pix_fmt, crf):
        calls.append((src, dst, fps, width, height, pix_fmt, crf))
        with open(dst, "wb"):
            pass

    monkeypatch.setattr(video_anchor, "reencode_anchor_clip", fake_encode)

    srcs = []
    for i in range(3):
        p = tmp_path / f"episode_{i:06d}.mp4"
        _touch(p)
        srcs.append(str(p))

    out_dir = tmp_path / "out"
    result = build_anchor(srcs, str(out_dir), fps=10.0, width=640, height=480, crf=23)

    assert result["n_built"] == 3
    assert result["n_skipped"] == 0
    assert result["n_failed"] == 0
    assert len(calls) == 3
    for _src, _dst, fps, width, height, pix_fmt, crf in calls:
        assert (fps, width, height, pix_fmt, crf) == (10.0, 640, 480, "yuv420p", 23)


def test_build_anchor_is_resume_safe_skips_existing_dst(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(video_anchor, "reencode_anchor_clip",
                        lambda *a, **k: calls.append((a, k)))

    src = tmp_path / "episode_000000.mp4"
    _touch(src)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "episode_000000.mp4").write_bytes(b"already built")

    result = build_anchor([str(src)], str(out_dir), fps=10.0, width=640, height=480, crf=23)
    assert result["n_skipped"] == 1
    assert result["n_built"] == 0
    assert len(calls) == 0  # never re-encoded


def test_build_anchor_one_failure_does_not_abort_the_batch(tmp_path, monkeypatch):
    def flaky_encode(src, dst, **kwargs):
        if "bad" in src:
            raise subprocess.CalledProcessError(1, ["ffmpeg"])
        with open(dst, "wb"):
            pass

    monkeypatch.setattr(video_anchor, "reencode_anchor_clip", flaky_encode)

    good = tmp_path / "good.mp4"
    bad = tmp_path / "bad.mp4"
    _touch(good)
    _touch(bad)
    out_dir = tmp_path / "out"

    result = build_anchor([str(good), str(bad)], str(out_dir), fps=10.0,
                          width=640, height=480, crf=23)
    assert result["n_built"] == 1
    assert result["n_failed"] == 1
    assert len(result["failures"]) == 1
    assert "bad.mp4" in result["failures"][0]["src"]


def test_build_anchor_creates_out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(video_anchor, "reencode_anchor_clip",
                        lambda src, dst, **k: open(dst, "wb").close())
    src = tmp_path / "e.mp4"
    _touch(src)
    out_dir = tmp_path / "nested" / "out"
    assert not out_dir.exists()
    build_anchor([str(src)], str(out_dir), fps=10.0, width=640, height=480, crf=23)
    assert out_dir.is_dir()


# ===========================================================================
# full CLI run against a fake encoder: provenance JSON is written
# ===========================================================================

def test_run_build_writes_provenance_json(tmp_path, monkeypatch):
    import argparse

    monkeypatch.setattr(video_anchor, "reencode_anchor_clip",
                        lambda src, dst, **k: open(dst, "wb").close())

    src = tmp_path / "episode_000000.mp4"
    _touch(src)
    out_dir = tmp_path / "out"

    parser = argparse.ArgumentParser()
    cmd_anchor.add_arguments(parser)
    args = parser.parse_args(
        ["build", "--real-glob", str(tmp_path / "*.mp4"), "--out-dir", str(out_dir),
         "--fps", "16", "--width", "768", "--height", "432", "--crf", "23"])
    rc = cmd_anchor.run(args)
    assert rc == 0

    prov_path = out_dir / "anchor_provenance.json"
    assert prov_path.is_file()
    prov = json.loads(prov_path.read_text())
    assert prov["fps"] == 16.0
    assert prov["width"] == 768
    assert prov["height"] == 432
    assert prov["n_built"] == 1
    assert "kinescore_version" in prov  # from cli._provenance.provenance_block


# ===========================================================================
# ffmpeg-gated: real probing and encoding
# ===========================================================================

def _make_testsrc(path, *, width=64, height=64, fps=10, duration=1):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
         f"testsrc=size={width}x{height}:rate={fps}:duration={duration}",
         "-pix_fmt", "yuv420p", str(path)], check=True)


@pytest.mark.ffmpeg
def test_probe_crf_context_reads_real_video_properties(tmp_path):
    clip = tmp_path / "clip.mp4"
    _make_testsrc(clip, width=128, height=96, fps=15, duration=1)
    ctx = probe_crf_context(str(clip))
    assert ctx["width"] == 128
    assert ctx["height"] == 96
    assert ctx["fps"] == pytest.approx(15.0, abs=0.01)
    assert ctx["codec"] == "h264"
    assert ctx["pix_fmt"]


@pytest.mark.ffmpeg
def test_probe_crf_context_raises_on_unreadable_file(tmp_path):
    bogus = tmp_path / "not_a_video.mp4"
    bogus.write_bytes(b"definitely not a video")
    with pytest.raises(subprocess.CalledProcessError):
        probe_crf_context(str(bogus))


@pytest.mark.ffmpeg
def test_reencode_anchor_clip_matches_requested_fps_and_resolution(tmp_path):
    src = tmp_path / "src.mp4"
    dst = tmp_path / "dst.mp4"
    _make_testsrc(src, width=320, height=240, fps=30, duration=1)

    reencode_anchor_clip(str(src), str(dst), fps=10.0, width=160, height=120,
                         pix_fmt="yuv420p", crf=28)
    assert dst.is_file()
    ctx = probe_crf_context(str(dst))
    assert ctx["width"] == 160
    assert ctx["height"] == 120
    assert ctx["fps"] == pytest.approx(10.0, abs=0.01)


@pytest.mark.ffmpeg
def test_end_to_end_anchor_build_with_probe_clip(tmp_path, monkeypatch):
    """Real ffmpeg/ffprobe, full command path: probe a 'generated' clip's
    context, then re-encode a batch of 'real' clips to match it."""
    import argparse

    monkeypatch.chdir(tmp_path)
    generated = tmp_path / "generated.mp4"
    _make_testsrc(generated, width=200, height=150, fps=12, duration=1)

    real_dir = tmp_path / "real"
    real_dir.mkdir()
    for i in range(2):
        _make_testsrc(real_dir / f"episode_{i:06d}.mp4", width=320, height=240,
                      fps=24, duration=1)

    out_dir = tmp_path / "anchor_out"
    parser = argparse.ArgumentParser()
    cmd_anchor.add_arguments(parser)
    args = parser.parse_args(
        ["build", "--real-glob", str(real_dir / "*.mp4"), "--out-dir", str(out_dir),
         "--probe-clip", str(generated)])
    rc = cmd_anchor.run(args)
    assert rc == 0

    built = sorted(out_dir.glob("episode_*.mp4"))
    assert len(built) == 2
    for clip in built:
        ctx = probe_crf_context(str(clip))
        assert ctx["width"] == 200
        assert ctx["height"] == 150
        assert ctx["fps"] == pytest.approx(12.0, abs=0.01)
