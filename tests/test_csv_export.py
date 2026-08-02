"""Tests for ``kinescore.bench.csv_export`` and ``kinescore export`` (``cli/cmd_export.py``).

Synthesises ``results.jsonl`` + a manifest by hand (mirroring
``tests/test_rank.py``'s approach) rather than running any real scoring --
CPU-only and network-free, per the repo's test tiers
(``tests/conftest.py``). Two record shapes are built deliberately, matching
``legacy_docs/SCHEMA.md``'s documented divergence:

* an **``ok``** record's ``"clip"`` block is ``ClipSpec.as_row()`` shape
  (``width``/``height``, no ``episode``/``role`` at all) -- so these tests
  exercise the actual join this module exists for, not a shortcut around it.
* a **``failed``** record's ``"clip"`` block is the raw manifest row verbatim
  (``method``/``family``/``episode``/``role``/... plus ``w``/``h``) --
  ``bench/store.py::failed_record``'s documented shape.
"""
from __future__ import annotations

import csv
import json
import os

import pytest

pytest.importorskip("pandas")

import kinescore.metrics  # noqa: F401  (side effect: populates the metric registry)
from kinescore.bench.csv_export import (
    ClipRow,
    build_clip_rows,
    export_csvs,
    group_key_for_path,
    relative_to_data_root,
    sort_group,
    summarize_group,
    write_clip_csv,
    write_summary_csv,
)
from kinescore.bench.store import ResultsStore

DATA_ROOT = "/data/kinescore_data"


def _dreamgen_path(episode: str, horizon: str = "makovian") -> str:
    return (f"{DATA_ROOT}/video_gen_physics/dense/humanoid/output/singleview/"
           f"dreamgen/{horizon}/iter_000090000/{episode}.mp4")


def _dreamdojo_path(episode: str, role: str, horizon: str = "makovian") -> str:
    return (f"{DATA_ROOT}/video_gen_physics/dense/humanoid/output/singleview/"
           f"dreamdojo/{horizon}/iter_000050000/{episode}_{role}.mp4")


def _ok_record(path: str, *, width=8, height=8, n_frames=4, dt=0.0625,
               metrics=None, unavailable=None) -> dict:
    return {
        "clip": {"path": path, "fps": 16.0, "dt": dt, "n_frames": n_frames,
                 "width": width, "height": height, "dt_source": "table",
                 "view_layout": "1x?:unnamed", "stride": 1, "codec": "h264",
                 "sha1": None},
        "run": {"robot": "fourier_gr1", "reader_id": "humanoid.pt",
                "limit_semantics": "raw_rad", "suite_id": "sha256:test",
                "suite_name": "invariant_v1"},
        "coverage": {"n_frames_scored": n_frames, "gate_coverage": 1.0},
        "metrics": dict(metrics or {}),
        "metrics_unavailable": dict(unavailable or {}),
        "status": "ok",
    }


def _failed_record(path: str, manifest_row: dict, metric_keys, reason: str) -> dict:
    return {
        "clip": dict(manifest_row),
        "run": {"robot": "fourier_gr1", "reader_id": "humanoid.pt",
                "limit_semantics": "raw_rad", "suite_id": "sha256:test",
                "suite_name": "invariant_v1"},
        "coverage": {"n_frames_scored": 0, "gate_coverage": 0.0},
        "metrics": dict.fromkeys(metric_keys),
        "metrics_unavailable": dict.fromkeys(metric_keys, reason),
        "status": "failed",
    }


def _manifest_row(path: str, *, episode: str, role: str, fps_probed=16.0,
                  n_frames=4, w=8, h=8, dt=0.0625, codec="h264") -> dict:
    return {"method": "dreamgen", "family": "f", "episode": episode,
           "role": role, "path": path, "pair_key": f"dreamgen/{episode}",
           "fps_probed": fps_probed, "n_frames": n_frames, "w": w, "h": h,
           "dt": dt, "codec": codec, "dt_source": "table", "sha1": None,
           "view_layout": "1x?:unnamed"}


# ── group_key_for_path ─────────────────────────────────────────────────────

class TestGroupKeyForPath:
    def test_output_tree_layout_parses(self):
        rel = ("video_gen_physics/dense/humanoid/output/singleview/dreamgen/"
              "makovian/iter_000090000/episode_000200.mp4")
        assert group_key_for_path(rel) == (
            "dense", "humanoid", "singleview", "dreamgen", "makovian")

    def test_input_tree_layout_also_parses(self):
        # ground truth (when it exists) lives under input/, not output/ --
        # same 5 identity segments either way.
        rel = ("video_gen_physics/dense/single_arm/input/multiview/ctrlworld/"
              "non_makovian/episode_0001.mp4")
        assert group_key_for_path(rel) == (
            "dense", "single_arm", "multiview", "ctrlworld", "non_makovian")

    def test_unrecognised_layout_falls_back_to_unmatched_not_dropped(self):
        rel = "cosmos_synthetic_data/high/clip_0001.mp4"
        group = group_key_for_path(rel)
        assert group[0] == "_unmatched"
        assert group == ("_unmatched", "cosmos_synthetic_data", "high")


class TestRelativeToDataRoot:
    def test_relativizes_under_root(self):
        rel = relative_to_data_root(f"{DATA_ROOT}/video_gen_physics/a.mp4", DATA_ROOT)
        assert rel == "video_gen_physics/a.mp4"
        assert not rel.startswith("/")

    def test_outside_root_falls_back_to_absolute_rather_than_raising(self):
        rel = relative_to_data_root("/somewhere/else/a.mp4", DATA_ROOT)
        assert rel == "/somewhere/else/a.mp4"


# ── build_clip_rows: the join ───────────────────────────────────────────────

class TestBuildClipRowsJoin:
    def test_ok_record_gets_episode_and_role_from_manifest(self):
        path = _dreamgen_path("episode_000200")
        records = [_ok_record(path, metrics={"mean_jerk_mps3": 1.5})]
        manifest = [_manifest_row(path, episode="episode_000200", role="pred")]

        [row] = build_clip_rows(records, manifest, DATA_ROOT)
        assert row.episode == "episode_000200"
        assert row.role == "pred"
        assert row.path == "video_gen_physics/dense/humanoid/output/singleview/dreamgen/makovian/iter_000090000/episode_000200.mp4"
        assert not row.path.startswith("/")  # never absolute
        assert row.fps_probed == pytest.approx(16.0)  # only the manifest carries this

    def test_ok_record_clip_block_wins_over_manifest_on_overlapping_fields(self):
        # score-time rescoring may update dt/n_frames/w/h relative to the
        # manifest's original probe; the record's own clip block must win.
        path = _dreamgen_path("episode_000200")
        records = [_ok_record(path, width=99, height=11, n_frames=7, dt=0.5,
                              metrics={"mean_jerk_mps3": 1.0})]
        manifest = [_manifest_row(path, episode="episode_000200", role="pred",
                                  w=8, h=8, n_frames=4, dt=0.0625)]
        [row] = build_clip_rows(records, manifest, DATA_ROOT)
        assert row.width == 99 and row.height == 11
        assert row.n_frames == 7
        assert row.dt == pytest.approx(0.5)

    def test_failed_record_carries_its_own_identity_no_manifest_needed(self):
        path = _dreamdojo_path("0000", "gt")
        manifest_row = _manifest_row(path, episode="flat:0000", role="gt")
        records = [_failed_record(path, manifest_row, ["mean_jerk_mps3"],
                                  "error:RuntimeError:boom")]
        [row] = build_clip_rows(records, [], DATA_ROOT)  # no manifest passed in!
        assert row.episode == "flat:0000"
        assert row.role == "gt"
        assert row.status == "failed"
        assert row.failure_reason == "error:RuntimeError:boom"
        assert row.metrics["mean_jerk_mps3"] is None

    def test_unavailable_metric_is_none_with_a_reason_never_zero(self):
        path = _dreamgen_path("episode_000200")
        records = [_ok_record(
            path, metrics={"mean_jerk_mps3": None, "sparc": 0.0},
            unavailable={"mean_jerk_mps3": "missing_input:q"})]
        manifest = [_manifest_row(path, episode="episode_000200", role="pred")]
        [row] = build_clip_rows(records, manifest, DATA_ROOT)
        assert row.metrics["mean_jerk_mps3"] is None
        assert row.metrics_unavailable["mean_jerk_mps3"] == "missing_input:q"
        assert row.metrics["sparc"] == 0.0  # a real zero must survive intact

    def test_unscored_reason_none_omits_never_scored_manifest_rows(self):
        scored_path = _dreamgen_path("episode_000200")
        never_scored_path = _dreamgen_path("episode_999999")
        records = [_ok_record(scored_path, metrics={"mean_jerk_mps3": 1.0})]
        manifest = [
            _manifest_row(scored_path, episode="episode_000200", role="pred"),
            _manifest_row(never_scored_path, episode="episode_999999", role="pred"),
        ]
        rows = build_clip_rows(records, manifest, DATA_ROOT, unscored_reason=None)
        assert len(rows) == 1

    def test_unscored_reason_set_emits_skipped_row_for_never_scored_clip(self):
        never_scored_path = _dreamgen_path("episode_999999")
        manifest = [_manifest_row(never_scored_path, episode="episode_999999",
                                  role="pred")]
        rows = build_clip_rows([], manifest, DATA_ROOT,
                               unscored_reason="not scored: no reader")
        [row] = rows
        assert row.status == "skipped"
        assert row.failure_reason == "not scored: no reader"
        assert row.episode == "episode_999999"
        assert row.metrics == {}


# ── sort_group ───────────────────────────────────────────────────────────

class TestSortGroup:
    def _rows(self, values, key="mean_jerk_mps3"):
        return [ClipRow(group=("g",), episode=str(i), path=f"p{i}.mp4",
                        role="pred", fps_probed=16.0, dt=0.0625, n_frames=4,
                        width=8, height=8, codec="h264", robot="r",
                        reader_id="rd", suite_id="s", suite_name="test_suite", limit_semantics="raw_rad",
                        metrics={key: v}, status="ok")
               for i, v in enumerate(values)]

    def test_lower_better_worst_first_is_highest_value(self):
        rows = self._rows([1.0, 9.0, 3.0])
        out = sort_group(rows, "mean_jerk_mps3", "lower_better")
        assert [r.metrics["mean_jerk_mps3"] for r in out] == [9.0, 3.0, 1.0]

    def test_higher_better_worst_first_is_lowest_value(self):
        rows = self._rows([0.5, 0.05, 0.4], key="limit_headroom_rad")
        out = sort_group(rows, "limit_headroom_rad", "higher_better")
        assert [r.metrics["limit_headroom_rad"] for r in out] == [0.05, 0.4, 0.5]

    def test_none_values_sort_last(self):
        rows = self._rows([5.0, None, 2.0])
        out = sort_group(rows, "mean_jerk_mps3", "lower_better")
        assert out[-1].metrics["mean_jerk_mps3"] is None
        assert [r.metrics["mean_jerk_mps3"] for r in out[:2]] == [5.0, 2.0]


# ── summarize_group / write_summary_csv ─────────────────────────────────────

class TestSummarizeGroup:
    def test_median_skips_none_never_treats_as_zero(self):
        rows = [
            ClipRow(group=("g",), episode="0", path="p0", role="pred",
                    fps_probed=16.0, dt=0.0625, n_frames=4, width=8, height=8,
                    codec="h264", robot="r", reader_id="rd", suite_id="s", suite_name="test_suite",
                    limit_semantics="raw_rad",
                    metrics={"mean_jerk_mps3": 2.0}, status="ok"),
            ClipRow(group=("g",), episode="1", path="p1", role="pred",
                    fps_probed=16.0, dt=0.0625, n_frames=4, width=8, height=8,
                    codec="h264", robot="r", reader_id="rd", suite_id="s", suite_name="test_suite",
                    limit_semantics="raw_rad",
                    metrics={"mean_jerk_mps3": None},
                    metrics_unavailable={"mean_jerk_mps3": "missing_input:q"},
                    status="ok"),
            ClipRow(group=("g",), episode="2", path="p2", role="pred",
                    fps_probed=16.0, dt=0.0625, n_frames=4, width=8, height=8,
                    codec="h264", robot="r", reader_id="rd", suite_id="s", suite_name="test_suite",
                    limit_semantics="raw_rad",
                    metrics={"mean_jerk_mps3": 4.0}, status="ok"),
        ]
        summary = summarize_group(rows, ["mean_jerk_mps3"])
        assert summary.medians["mean_jerk_mps3"] == pytest.approx(3.0)
        assert summary.n_clips == 3 and summary.n_ok == 3

    def test_all_unavailable_median_is_none_not_zero(self):
        rows = [ClipRow(group=("g",), episode="0", path="p0", role="pred",
                        fps_probed=16.0, dt=0.0625, n_frames=4, width=8,
                        height=8, codec="h264", robot="r", reader_id="rd",
                        suite_id="s", suite_name="test_suite", limit_semantics="raw_rad",
                        metrics={"mean_jerk_mps3": None}, status="ok")]
        summary = summarize_group(rows, ["mean_jerk_mps3"])
        assert summary.medians["mean_jerk_mps3"] is None


# ── CSV rendering ────────────────────────────────────────────────────────

class TestWriteClipCsv:
    def test_records_sort_key_in_a_leading_comment_and_stays_parseable(self, tmp_path):
        rows = [ClipRow(group=("g",), episode="0", path="p0.mp4", role="pred",
                        fps_probed=16.0, dt=0.0625, n_frames=4, width=8,
                        height=8, codec="h264", robot="r", reader_id="rd",
                        suite_id="s", suite_name="test_suite", limit_semantics="raw_rad",
                        metrics={"mean_jerk_mps3": 5.0},
                        status="ok")]
        out = tmp_path / "clips.csv"
        write_clip_csv(str(out), rows, ["mean_jerk_mps3"],
                       sort_by="mean_jerk_mps3", direction="lower_better")
        text = out.read_text()
        first_line = text.splitlines()[0]
        assert first_line.startswith("#")
        assert "sort_by=mean_jerk_mps3" in first_line
        assert "direction=lower_better" in first_line
        # a suite name/id is "which rulers", not "which clips" -- the
        # latter is decided by the config axes baked into this file's own
        # path, never by the suite (see SUITE_MEANING_NOTE).
        assert "suite=which rulers were computed" in first_line
        assert "not by the suite" in first_line

        # csv.DictReader would treat the comment as the header if not
        # skipped by the caller; a spreadsheet/pandas user passes
        # comment='#' -- verify the *real* header appears on line 2.
        lines = text.splitlines()
        header = lines[1].split(",")
        assert header[:3] == ["episode", "path", "role"]
        assert "mean_jerk_mps3" in header
        assert "mean_jerk_mps3_reason" in header
        assert header[-2:] == ["status", "failure_reason"]

    def test_unavailable_metric_cell_is_empty_not_zero(self, tmp_path):
        rows = [ClipRow(group=("g",), episode="0", path="p0.mp4", role="pred",
                        fps_probed=16.0, dt=0.0625, n_frames=4, width=8,
                        height=8, codec="h264", robot="r", reader_id="rd",
                        suite_id="s", suite_name="test_suite", limit_semantics="raw_rad",
                        metrics={"mean_jerk_mps3": None},
                        metrics_unavailable={"mean_jerk_mps3": "missing_input:q"},
                        status="ok")]
        out = tmp_path / "clips.csv"
        write_clip_csv(str(out), rows, ["mean_jerk_mps3"],
                       sort_by="mean_jerk_mps3", direction="lower_better")
        lines = out.read_text().splitlines()
        reader = csv.DictReader(lines[1:])
        [record] = list(reader)
        assert record["mean_jerk_mps3"] == ""
        assert record["mean_jerk_mps3_reason"] == "missing_input:q"

    def test_real_zero_metric_value_is_written_as_zero(self, tmp_path):
        rows = [ClipRow(group=("g",), episode="0", path="p0.mp4", role="pred",
                        fps_probed=16.0, dt=0.0625, n_frames=4, width=8,
                        height=8, codec="h264", robot="r", reader_id="rd",
                        suite_id="s", suite_name="test_suite", limit_semantics="raw_rad",
                        metrics={"sparc": 0.0}, status="ok")]
        out = tmp_path / "clips.csv"
        write_clip_csv(str(out), rows, ["sparc"], sort_by="sparc",
                       direction="higher_better")
        lines = out.read_text().splitlines()
        [record] = list(csv.DictReader(lines[1:]))
        assert record["sparc"] == "0"


class TestWriteSummaryCsv:
    def test_group_column_is_slash_joined(self, tmp_path):
        from kinescore.bench.csv_export import GroupSummary

        summary = GroupSummary(
            group=("dense", "humanoid", "singleview", "dreamgen", "makovian"),
            n_clips=2, n_ok=2, n_failed=0, n_skipped=0, robot="fourier_gr1",
            reader_id="humanoid.pt", suite_id="sha256:test", suite_name="invariant_v1", fps=16.0,
            medians={"mean_jerk_mps3": 3.0})
        out = tmp_path / "SUMMARY.csv"
        write_summary_csv(str(out), [summary], ["mean_jerk_mps3"])
        lines = out.read_text().splitlines()
        assert lines[0].startswith("#")  # self-describing leading comment
        assert "suite=which rulers were computed" in lines[0]
        [row] = list(csv.DictReader(lines[1:]))
        assert row["group"] == "dense/humanoid/singleview/dreamgen/makovian"
        assert row["median_mean_jerk_mps3"] == "3"
        assert row["n_clips"] == "2"
        assert row["suite_id"] == "sha256:test"
        assert row["suite_name"] == "invariant_v1"

    def test_headline_rigidity_note_present_when_metric_in_suite(self, tmp_path):
        from kinescore.bench.csv_export import HEADLINE_RIGIDITY_METRIC, GroupSummary

        summary = GroupSummary(
            group=("dense", "humanoid", "singleview", "dreamgen", "makovian"),
            n_clips=1, n_ok=1, n_failed=0, n_skipped=0, robot="fourier_gr1",
            reader_id="humanoid.pt", suite_id="sha256:full", suite_name="full",
            fps=16.0, medians={HEADLINE_RIGIDITY_METRIC: 5.0})
        out = tmp_path / "SUMMARY.csv"
        write_summary_csv(str(out), [summary], [HEADLINE_RIGIDITY_METRIC])
        first_line = out.read_text().splitlines()[0]
        assert f"headline_rigidity={HEADLINE_RIGIDITY_METRIC}" in first_line

    def test_headline_rigidity_note_absent_when_metric_not_in_suite(self, tmp_path):
        from kinescore.bench.csv_export import GroupSummary

        summary = GroupSummary(
            group=("dense", "humanoid", "singleview", "dreamgen", "makovian"),
            n_clips=1, n_ok=1, n_failed=0, n_skipped=0, robot="fourier_gr1",
            reader_id="humanoid.pt", suite_id="sha256:inv", suite_name="invariant_v1",
            fps=16.0, medians={"mean_jerk_mps3": 5.0})
        out = tmp_path / "SUMMARY.csv"
        write_summary_csv(str(out), [summary], ["mean_jerk_mps3"])
        first_line = out.read_text().splitlines()[0]
        assert "headline_rigidity" not in first_line

    def test_fps_caveat_fires_when_groups_have_different_fps(self, tmp_path):
        from kinescore.bench.csv_export import GroupSummary

        summaries = [
            GroupSummary(group=("dense", "humanoid", "singleview", "dreamgen", "makovian"),
                        n_clips=1, n_ok=1, n_failed=0, n_skipped=0, robot="r",
                        reader_id="rd", suite_id="s", suite_name="full",
                        fps=16.0, medians={"mean_jerk_mps3": 10.0}),
            GroupSummary(group=("dense", "humanoid", "singleview", "dreamdojo", "makovian"),
                        n_clips=1, n_ok=1, n_failed=0, n_skipped=0, robot="r",
                        reader_id="rd", suite_id="s", suite_name="full",
                        fps=10.0, medians={"mean_jerk_mps3": 3.0}),
        ]
        out = tmp_path / "SUMMARY.csv"
        write_summary_csv(str(out), summaries, ["mean_jerk_mps3"])
        first_line = out.read_text().splitlines()[0]
        assert "WARNING fps varies" in first_line
        assert "10.0" in first_line and "16.0" in first_line

    def test_fps_caveat_silent_when_all_groups_share_fps(self, tmp_path):
        from kinescore.bench.csv_export import GroupSummary

        summaries = [
            GroupSummary(group=("dense", "humanoid", "singleview", "dreamdojo", "makovian"),
                        n_clips=1, n_ok=1, n_failed=0, n_skipped=0, robot="r",
                        reader_id="rd", suite_id="s", suite_name="full",
                        fps=10.0, medians={"mean_jerk_mps3": 3.0}),
            GroupSummary(group=("dense", "humanoid", "singleview", "dreamdojo", "non_makovian"),
                        n_clips=1, n_ok=1, n_failed=0, n_skipped=0, robot="r",
                        reader_id="rd", suite_id="s", suite_name="full",
                        fps=10.0, medians={"mean_jerk_mps3": 4.0}),
        ]
        out = tmp_path / "SUMMARY.csv"
        write_summary_csv(str(out), summaries, ["mean_jerk_mps3"])
        first_line = out.read_text().splitlines()[0]
        assert "WARNING" not in first_line


# ── export_csvs end-to-end ───────────────────────────────────────────────

class TestExportCsvsEndToEnd:
    def _write_run(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        store = ResultsStore(str(run_dir / "results.jsonl"))

        manifest = []
        # dreamgen/makovian: 2 pred-only clips (no gt -- matches the real
        # dataset's dreamgen gap), one ok one failed.
        ok_path = _dreamgen_path("episode_000200", "makovian")
        store.append(_ok_record(ok_path, metrics={"mean_jerk_mps3": 2.0,
                                                   "sparc": -1.5}))
        manifest.append(_manifest_row(ok_path, episode="episode_000200",
                                      role="pred"))

        failed_path = _dreamgen_path("episode_000829", "makovian")
        failed_manifest_row = _manifest_row(failed_path, episode="episode_000829",
                                           role="pred")
        store.append(_failed_record(failed_path, failed_manifest_row,
                                   ["mean_jerk_mps3", "sparc"],
                                   "error:RuntimeError:decode failed"))
        manifest.append(failed_manifest_row)

        # dreamdojo/makovian: a paired gt/pred clip, both ok, different cell.
        gt_path = _dreamdojo_path("0000", "gt", "makovian")
        pred_path = _dreamdojo_path("0000", "pred", "makovian")
        store.append(_ok_record(gt_path, dt=0.1,
                                metrics={"mean_jerk_mps3": 1.0, "sparc": -1.0}))
        store.append(_ok_record(pred_path, dt=0.1,
                                metrics={"mean_jerk_mps3": 8.0, "sparc": -2.0}))
        manifest.append(_manifest_row(gt_path, episode="flat:0000", role="gt",
                                      fps_probed=10.0, dt=0.1))
        manifest.append(_manifest_row(pred_path, episode="flat:0000", role="pred",
                                      fps_probed=10.0, dt=0.1))

        manifest_path = run_dir / "bench_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        return str(run_dir / "results.jsonl"), str(manifest_path)

    def test_writes_mirrored_tree_and_summary(self, tmp_path):
        results_path, manifest_path = self._write_run(tmp_path)
        out_dir = tmp_path / "out"

        result = export_csvs(results_path, str(out_dir), data_root=DATA_ROOT,
                             manifest_path=manifest_path,
                             sort_by="mean_jerk_mps3")

        dreamgen_csv = out_dir / "dense" / "humanoid" / "singleview" / "dreamgen" / "makovian" / "clips.csv"
        dreamdojo_csv = out_dir / "dense" / "humanoid" / "singleview" / "dreamdojo" / "makovian" / "clips.csv"
        assert dreamgen_csv.is_file()
        assert dreamdojo_csv.is_file()
        assert (out_dir / "SUMMARY.csv").is_file()
        assert result.n_rows == 4
        assert result.n_groups == 2

        dojo_lines = dreamdojo_csv.read_text().splitlines()
        rows = list(csv.DictReader(dojo_lines[1:]))
        assert len(rows) == 2
        # worst (pred, jerk=8.0) first
        assert rows[0]["role"] == "pred"
        assert rows[0]["mean_jerk_mps3"] == "8"
        assert rows[1]["role"] == "gt"

        summary_lines = (out_dir / "SUMMARY.csv").read_text().splitlines()
        assert summary_lines[0].startswith("#")
        summary_rows = list(csv.DictReader(summary_lines[1:]))
        by_group = {r["group"]: r for r in summary_rows}
        dojo_summary = by_group["dense/humanoid/singleview/dreamdojo/makovian"]
        assert dojo_summary["n_ok"] == "2"
        assert dojo_summary["median_mean_jerk_mps3"] == "4.5"  # (1+8)/2
        assert dojo_summary["suite_id"] == "sha256:test"
        assert dojo_summary["suite_name"] == "invariant_v1"

        dreamgen_summary = by_group["dense/humanoid/singleview/dreamgen/makovian"]
        assert dreamgen_summary["n_ok"] == "1"
        assert dreamgen_summary["n_failed"] == "1"
        # only 1 of 2 clips has a value -> median is that one value, not
        # dragged toward zero by the failed clip.
        assert dreamgen_summary["median_mean_jerk_mps3"] == "2"

    def test_paths_in_csv_are_relative_never_absolute(self, tmp_path):
        results_path, manifest_path = self._write_run(tmp_path)
        out_dir = tmp_path / "out"
        export_csvs(results_path, str(out_dir), data_root=DATA_ROOT,
                   manifest_path=manifest_path)
        for csv_path in (out_dir / "dense" / "humanoid" / "singleview"
                        / "dreamgen" / "makovian" / "clips.csv",):
            lines = csv_path.read_text().splitlines()
            rows = list(csv.DictReader(lines[1:]))
            for row in rows:
                assert not row["path"].startswith("/")
                assert not os.path.isabs(row["path"])

    def test_rerun_is_idempotent(self, tmp_path):
        results_path, manifest_path = self._write_run(tmp_path)
        out_dir = tmp_path / "out"
        r1 = export_csvs(results_path, str(out_dir), data_root=DATA_ROOT,
                         manifest_path=manifest_path)
        r2 = export_csvs(results_path, str(out_dir), data_root=DATA_ROOT,
                         manifest_path=manifest_path)
        assert r1.n_rows == r2.n_rows == 4
        assert r1.csv_paths == r2.csv_paths

    def test_unknown_sort_by_raises(self, tmp_path):
        results_path, manifest_path = self._write_run(tmp_path)
        with pytest.raises(ValueError, match="not among this run's metrics"):
            export_csvs(results_path, str(tmp_path / "out"), data_root=DATA_ROOT,
                       manifest_path=manifest_path, sort_by="not_a_real_metric")

    def test_unscored_reason_produces_skip_only_cell(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        # No results.jsonl rows at all -- an unsupported-robot cell, e.g.
        # ctrlworld/humanoid/multiview (Airbot MMK2, no reader).
        results_path = run_dir / "results.jsonl"
        results_path.touch()
        never_scored_path = (f"{DATA_ROOT}/video_gen_physics/dense/humanoid/"
                            "output/multiview/ctrlworld/makovian/"
                            "episode_AIRBOT_MMK2_000000/full_pred.mp4")
        manifest = [_manifest_row(never_scored_path, episode="AIRBOT_MMK2_000000",
                                  role="pred", fps_probed=30.0)]
        manifest_path = run_dir / "bench_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        out_dir = tmp_path / "out"
        result = export_csvs(
            str(results_path), str(out_dir), data_root=DATA_ROOT,
            manifest_path=str(manifest_path),
            unscored_reason="not scored: robot is Airbot MMK2, no reader")

        assert result.n_rows == 1
        csv_path = (out_dir / "dense" / "humanoid" / "multiview" / "ctrlworld"
                   / "makovian" / "clips.csv")
        assert csv_path.is_file()
        [row] = list(csv.DictReader(csv_path.read_text().splitlines()[1:]))
        assert row["status"] == "skipped"
        assert row["failure_reason"] == "not scored: robot is Airbot MMK2, no reader"
        assert row["mean_jerk_mps3"] == ""  # empty, never fabricated


# ── CLI wiring (cli/cmd_export.py) ──────────────────────────────────────────

class TestCliWiring:
    def _parse_args(self, argv):
        import argparse

        from kinescore.cli import cmd_export
        parser = argparse.ArgumentParser()
        cmd_export.add_arguments(parser)
        return parser.parse_args(argv)

    def test_help_is_a_string(self):
        from kinescore.cli import cmd_export
        assert isinstance(cmd_export.HELP, str) and cmd_export.HELP

    def test_missing_results_file_is_an_error(self, tmp_path, capsys):
        from kinescore.cli import cmd_export
        empty = tmp_path / "empty"
        empty.mkdir()
        args = self._parse_args(["--results", str(empty), "--out",
                                 str(tmp_path / "out")])
        rc = cmd_export.run(args)
        assert rc == 1
        assert "results.jsonl" in capsys.readouterr().err

    def test_end_to_end_cli_run_writes_provenance(self, tmp_path, capsys):
        from kinescore.cli import cmd_export

        results_path, manifest_path = TestExportCsvsEndToEnd()._write_run(tmp_path)
        run_dir = os.path.dirname(results_path)
        out_dir = tmp_path / "out"
        args = self._parse_args([
            "--results", run_dir, "--out", str(out_dir),
            "--data-root", DATA_ROOT, "--manifest", manifest_path,
        ])
        rc = cmd_export.run(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "SUMMARY.csv" in out
        prov_path = out_dir / "export_provenance.json"
        assert prov_path.is_file()
        prov = json.loads(prov_path.read_text())
        assert prov["sort_by"] == "mean_jerk_mps3"
        assert prov["n_rows"] == 4

    def test_manifest_autodetected_next_to_results(self, tmp_path, capsys):
        from kinescore.cli import cmd_export

        results_path, manifest_path = TestExportCsvsEndToEnd()._write_run(tmp_path)
        run_dir = os.path.dirname(results_path)
        # bench_manifest.json already sits next to results.jsonl (written by
        # _write_run) -- don't pass --manifest, rely on auto-detection.
        args = self._parse_args(["--results", run_dir, "--out",
                                 str(tmp_path / "out"), "--data-root", DATA_ROOT])
        rc = cmd_export.run(args)
        assert rc == 0
