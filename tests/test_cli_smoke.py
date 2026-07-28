"""CLI smoke tests: every subcommand's ``--help`` works, ``doctor`` runs and
reports, ``describe --json`` emits valid JSON matching ``MetricSuite.describe()``.

These deliberately do not exercise the heavy, I/O-bound paths (``score``,
``cache``, ``train``, ...) end-to-end -- that needs real checkpoints/video and
is out of scope for the CPU-only, network-free default tier. What they pin is
the contract every subcommand must satisfy regardless: it parses, it has a
``run``/``add_arguments`` pair main.py can wire up, and the two commands with
no other test coverage in this suite (``doctor``, ``describe``) actually work.
"""
from __future__ import annotations

import json
import sys

import pytest

from kinescore.cli.main import _SUBCOMMANDS, build_parser, main


class TestHelp:
    def test_top_level_help_lists_every_subcommand(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        for name, _ in _SUBCOMMANDS:
            assert name in out

    @pytest.mark.parametrize("name", [n for n, _ in _SUBCOMMANDS])
    def test_subcommand_help_exits_zero(self, name, capsys):
        with pytest.raises(SystemExit) as exc:
            main([name, "--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "usage:" in out

    def test_reference_build_help_exits_zero(self, capsys):
        # `reference` has a nested subparser (`build`); its own --help must
        # also work, not just the outer `reference --help`.
        with pytest.raises(SystemExit) as exc:
            main(["reference", "build", "--help"])
        assert exc.value.code == 0
        assert "usage:" in capsys.readouterr().out

    def test_no_command_prints_help_and_returns_nonzero(self, capsys):
        rc = main([])
        assert rc != 0
        assert "usage:" in capsys.readouterr().out

    def test_build_parser_is_import_light(self):
        # Constructing the whole parser tree must not require torch -- see
        # kinescore.cli's module docstring. If any cmd_*.py imports torch at
        # module scope, this either raises or (if torch happens to already be
        # importable in the test env) at least proves parser construction
        # itself doesn't need it by never referencing torch here.
        parser = build_parser()
        assert parser.prog == "kinescore"


class TestDoctor:
    def test_doctor_runs_and_returns_zero(self, capsys):
        rc = main(["doctor"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "kinescore" in out

    def test_doctor_json_is_valid_and_has_expected_top_level_keys(self, capsys):
        rc = main(["doctor", "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        report = json.loads(out)
        for key in ("kinescore_version", "python_version", "torch", "ffprobe",
                   "env_vars", "asset_hashes"):
            assert key in report

    def test_doctor_markdown_emits_a_table(self, capsys):
        rc = main(["doctor", "--markdown"])
        assert rc == 0
        out = capsys.readouterr().out
        assert out.strip().startswith("| Component | Status |")

    def test_doctor_degrades_gracefully_without_torch(self, capsys, monkeypatch):
        """doctor must not crash when torch is unimportable (see its docstring).

        ``sys.modules["torch"] = None`` (rather than patching
        ``builtins.__import__``) is the standard way to simulate "module not
        installed": both a plain ``import torch`` statement and
        ``importlib.import_module("torch")`` check ``sys.modules`` first and
        raise ``ImportError`` when an entry is explicitly ``None``, so both
        of ``cmd_doctor``'s two import paths (``_check_import`` and
        ``_torch_info``'s own ``import torch``) see a consistently "missing"
        torch, unlike patching ``__import__`` (which ``importlib`` bypasses).
        """
        monkeypatch.setitem(sys.modules, "torch", None)
        rc = main(["doctor", "--json"])
        assert rc == 0
        report = json.loads(capsys.readouterr().out)
        assert report["torch"]["available"] is False
        assert "reason" in report["torch"]


class TestDescribe:
    def test_describe_json_is_valid_and_matches_metric_suite_describe(self, capsys):
        from kinescore.metrics.suites import INVARIANT_V1

        rc = main(["describe", "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)

        assert payload["suite_id"] == INVARIANT_V1.suite_id
        assert payload["suite_name"] == INVARIANT_V1.name

        want = INVARIANT_V1.describe()
        assert payload["metrics"] == want
        # Keys of every per-metric dict match MetricSuite.describe()'s shape.
        want_keys = set(want[0].keys())
        for entry in payload["metrics"]:
            assert set(entry.keys()) == want_keys

    def test_describe_human_readable_runs(self, capsys):
        rc = main(["describe"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "rigidity_residual_mm" in out

    def test_describe_unknown_suite_raises(self):
        with pytest.raises(KeyError):
            main(["describe", "--suite", "not_a_real_suite"])

    def test_describe_reader_json_surfaces_readout_v2_observability(self, capsys,
                                                                     tmp_path):
        """``--reader`` on a ReadoutV2 (heteroscedastic) checkpoint must
        surface that limit_violation becomes observable under raw_rad --
        see docs/PROVENANCE.md D7 and cli/cmd_describe.py's docstring."""
        import torch

        from kinescore.core.clip import ViewLayout
        from kinescore.heads.heteroscedastic import ReadoutV2Head
        from kinescore.readers import checkpoint_v2

        torch.manual_seed(0)
        head = ReadoutV2Head(in_dim=8, d_model=16, n_heads=2, temporal_nhead=2,
                             ff=16, n_temporal_layers=1, t_max=8, n_out=4)
        path = str(tmp_path / "rv2.pt")
        checkpoint_v2.save(path, head, view_layout=ViewLayout(), sigma_scale=1.9375)

        rc = main(["describe", "--reader", path, "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["reader"]["limit_semantics"] == "raw_rad"
        assert payload["reader"]["head_family"].startswith("readout_v2")
        assert "OBSERVABLE" in payload["reader"]["note"]
        assert payload["reader"]["sigma_scale"] == pytest.approx(1.9375)

    def test_describe_reader_json_surfaces_squashed_unobservability(self, capsys,
                                                                     tmp_path):
        from kinescore.core.clip import ViewLayout
        from kinescore.heads.attentive import AttentivePoseHead
        from kinescore.readers import checkpoint as ckpt_mod

        head = AttentivePoseHead(in_dim=8, hidden=8, n_joints=5, n_heads=2, n_cams=1)
        path = str(tmp_path / "attentive.pt")
        ckpt_mod.save(path, head, view_layout=ViewLayout(), robot_name="franka_panda")

        rc = main(["describe", "--reader", path, "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["reader"]["limit_semantics"] == "squashed"
        assert "UNOBSERVABLE" in payload["reader"]["note"]

    def test_describe_without_reader_has_no_reader_key(self, capsys):
        rc = main(["describe", "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "reader" not in payload

    def test_describe_reader_human_readable_prints_a_note(self, capsys, tmp_path):
        import torch

        from kinescore.core.clip import ViewLayout
        from kinescore.heads.heteroscedastic import ReadoutV2Head
        from kinescore.readers import checkpoint_v2

        torch.manual_seed(0)
        head = ReadoutV2Head(in_dim=8, d_model=16, n_heads=2, temporal_nhead=2,
                             ff=16, n_temporal_layers=1, t_max=8, n_out=4)
        path = str(tmp_path / "rv2.pt")
        checkpoint_v2.save(path, head, view_layout=ViewLayout())

        rc = main(["describe", "--reader", path])
        assert rc == 0
        out = capsys.readouterr().out
        assert "reader:" in out
        assert "raw_rad" in out
