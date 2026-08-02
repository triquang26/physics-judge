"""Direct, argparse-free tests for :mod:`kinescore.bench.doctor`.

``kinescore.cli.cmd_doctor`` is a thin shell around ``build_report``/
``render_human``/``render_markdown`` here; ``tests/test_cli_smoke.py``
already exercises the CLI end to end (``--json``/``--markdown``, the
torch-missing degradation). This file is the library-level coverage the
move from ``cli/cmd_doctor.py`` is supposed to buy: every function callable
and testable without building an ``argparse.Namespace`` at all.
"""
from __future__ import annotations

import sys

from kinescore.bench import doctor


class TestBuildReport:
    def test_has_the_expected_top_level_keys(self):
        report = doctor.build_report()
        for key in ("kinescore_version", "python_version", "platform", "torch",
                    "pytorch_kinematics", "transformers", "pandas", "scipy",
                    "ffprobe", "env_vars", "asset_hashes"):
            assert key in report

    def test_degrades_gracefully_without_torch(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        report = doctor.build_report()
        assert report["torch"]["available"] is False
        assert "reason" in report["torch"]

    def test_ffprobe_missing_from_path_is_reported_not_raised(self, monkeypatch):
        monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
        info = doctor._ffprobe_info()
        assert info["available"] is False
        assert "not on PATH" in info["reason"]

    def test_cached_panda_urdf_is_none_without_a_cache_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path / "nope"))
        assert doctor._cached_panda_urdf() is None


class TestRenderHuman:
    def test_includes_version_and_status_rows(self):
        report = doctor.build_report()
        text = doctor.render_human(report)
        assert report["kinescore_version"] in text
        assert "torch" in text
        assert "env vars:" in text
        assert "assets:" in text


class TestRenderMarkdown:
    def test_is_a_component_status_table(self):
        report = doctor.build_report()
        text = doctor.render_markdown(report)
        assert text.startswith("| Component | Status |")
        assert "| Python |" in text
        assert "| torch |" in text

    def test_reports_cuda_na_when_torch_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        report = doctor.build_report()
        text = doctor.render_markdown(report)
        assert "| CUDA | n/a |" in text
        assert "| torch | not installed |" in text
