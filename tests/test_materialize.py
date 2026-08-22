"""``kinescore.registry.materialize``: adapter output -> the canonical tree.

Uses a fake adapter over ffmpeg-generated clips, so the writer is exercised
end to end -- packing, geometry check, disjoint split, annotations, card --
without any corpus on disk.
"""
from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from kinescore.adapters.base import RawEpisode, SkippedEpisode, register_adapter
from kinescore.registry.cells import ReaderSpec, TrainSource
from kinescore.registry.materialize import (
    GRIPPER_KEY,
    JOINT_KEY,
    materialize_train_tree,
)
from kinescore.registry.views import ViewSpec

pytestmark = pytest.mark.ffmpeg

VIEW = ViewSpec(view_id="mv2_row", n_views=2, packing="width", n_panels=2,
                panel=(32, 24), order=("left", "right"))
SV1 = ViewSpec(view_id="sv1", n_views=1, packing="none", panel=(32, 24))


def _clip(path, width=32, height=24, n_frames=8, fps=10) -> str:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc=size={width}x{height}:rate={fps}:duration="
               f"{n_frames / fps}", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True)
    return str(path)


def _episodes(tmp_path, n_scenes=4, per_scene=3, packed=False, n_frames=8):
    out = []
    for s in range(n_scenes):
        for i in range(per_scene):
            eid = f"task{s}__{i}"
            joints = np.zeros((n_frames, 7), dtype=np.float32)
            kwargs = {}
            if packed:
                kwargs["packed"] = _clip(tmp_path / f"{eid}.mp4", width=64)
            else:
                kwargs["views"] = {
                    "a_left": _clip(tmp_path / f"{eid}_l.mp4"),
                    "b_right": _clip(tmp_path / f"{eid}_r.mp4"),
                }
            out.append(RawEpisode(
                episode_id=eid, joints=joints, fps=10.0, scene_key=f"task{s}",
                source_path=f"/corpus/{eid}", **kwargs))
    return out


def _reader(tmp_path, view, adapter_id, scene_key="prefix") -> ReaderSpec:
    return ReaderSpec(
        reader_id="franka_panda.mv2_row", robot="franka_panda", view=view,
        train=TrainSource(adapter=adapter_id, root=str(tmp_path),
                          scene_key=scene_key))


def _register(episodes, adapter_id):
    class _Fake:
        SOURCE_ID = adapter_id

        def episodes(self, source):
            yield from episodes

    register_adapter(adapter_id, _Fake)


@pytest.fixture(autouse=True)
def _paths(tmp_path, monkeypatch):
    for key in ("KINESCORE_DATA_ROOT", "KINESCORE_CACHE_DIR",
                "KINESCORE_CKPT_DIR", "KINESCORE_ASSETS"):
        monkeypatch.setenv(key, str(tmp_path / key.lower()))


class TestTreeLayout:
    def test_packs_views_and_writes_both_splits(self, tmp_path, request):
        adapter_id = f"fake_{request.node.name}"
        _register(_episodes(tmp_path), adapter_id)
        reader = _reader(tmp_path, VIEW, adapter_id)

        report = materialize_train_tree(reader, val_ratio=0.25, seed=0)

        assert report.n_train and report.n_val
        assert report.n_written == 12
        tree = reader.train_tree
        for split in ("train", "val"):
            videos = sorted((tree / "videos" / split).glob("*.mp4"))
            labels = sorted((tree / "annotation" / split).glob("*.json"))
            assert [p.stem for p in videos] == [p.stem for p in labels]

    def test_split_is_scene_disjoint(self, tmp_path, request):
        adapter_id = f"fake_{request.node.name}"
        _register(_episodes(tmp_path), adapter_id)
        reader = _reader(tmp_path, VIEW, adapter_id)
        materialize_train_tree(reader, val_ratio=0.25, seed=0)

        tree = reader.train_tree

        def scenes(split):
            return {p.stem.split("__")[0]
                    for p in (tree / "videos" / split).glob("*.mp4")}

        assert not (scenes("train") & scenes("val"))

    def test_annotation_carries_real_joints(self, tmp_path, request):
        adapter_id = f"fake_{request.node.name}"
        episodes = _episodes(tmp_path, n_scenes=2, per_scene=2)
        episodes = [RawEpisode(**{**e.__dict__,
                                  "gripper": np.zeros(8, dtype=np.float32)})
                    for e in episodes]
        _register(episodes, adapter_id)
        reader = _reader(tmp_path, VIEW, adapter_id)
        materialize_train_tree(reader, val_ratio=0.25, seed=0)

        label = json.loads(next(
            (reader.train_tree / "annotation" / "train").glob("*.json")).read_text())
        assert label["joint_source"] == "real"
        assert np.asarray(label[JOINT_KEY]).shape[1] == 7
        assert GRIPPER_KEY in label
        assert label["fps"] == 10.0

    def test_card_records_geometry_and_skips(self, tmp_path, request):
        adapter_id = f"fake_{request.node.name}"
        _register(_episodes(tmp_path), adapter_id)
        reader = _reader(tmp_path, VIEW, adapter_id)
        materialize_train_tree(reader, val_ratio=0.25, seed=0)

        card = json.loads((reader.train_tree / "dataset_card.json").read_text())
        assert card["reader_id"] == reader.reader_id
        assert card["view"]["view_id"] == "mv2_row"
        assert card["n_train"] + card["n_val"] == 12

    def test_already_packed_clips_pass_through(self, tmp_path, request):
        adapter_id = f"fake_{request.node.name}"
        _register(_episodes(tmp_path, n_scenes=2, per_scene=2, packed=True),
                  adapter_id)
        reader = _reader(tmp_path, VIEW, adapter_id)
        report = materialize_train_tree(reader, val_ratio=0.25, seed=0)
        assert report.n_written == 4


class TestRefusals:
    def test_wrong_frame_geometry_is_skipped_not_written(self, tmp_path, request):
        adapter_id = f"fake_{request.node.name}"
        episodes = _episodes(tmp_path, n_scenes=2, per_scene=2, packed=True)
        _register(episodes, adapter_id)
        reader = _reader(tmp_path, SV1, adapter_id)  # sv1 expects 32x24

        report = materialize_train_tree(reader, val_ratio=0.25, seed=0)

        assert report.n_written == 0
        assert len(report.skipped) == 4
        assert all("expects" in reason for _, reason in report.skipped)

    def test_adapter_skips_are_reported(self, tmp_path, request):
        adapter_id = f"fake_{request.node.name}"
        episodes = _episodes(tmp_path, n_scenes=2, per_scene=2)
        episodes.append(SkippedEpisode(
            episode_id="task9__0", reason="no metadata.json",
            source_path="/corpus/task9__0"))
        _register(episodes, adapter_id)
        reader = _reader(tmp_path, VIEW, adapter_id)

        report = materialize_train_tree(reader, val_ratio=0.25, seed=0)

        assert ("task9__0", "no metadata.json") in report.skipped

    def test_blocked_reader_is_refused(self, tmp_path, request):
        adapter_id = f"fake_{request.node.name}"
        _register(_episodes(tmp_path, n_scenes=1, per_scene=2), adapter_id)
        reader = ReaderSpec(
            reader_id="franka_panda.mv2_row", robot="franka_panda", view=VIEW,
            train=TrainSource(adapter=adapter_id, root=str(tmp_path)),
            status="blocked: this corpus logs no state")
        with pytest.raises(ValueError, match="not trainable"):
            materialize_train_tree(reader)

    def test_reader_without_a_source_is_refused(self, tmp_path):
        reader = ReaderSpec(reader_id="franka_panda.mv2_row",
                            robot="franka_panda", view=VIEW)
        with pytest.raises(ValueError, match="declares no train source"):
            materialize_train_tree(reader)


class TestSceneKeyMode:
    """A corpus with two huge scenes, split each way.

    ``val_ratio=0.15`` of 20 episodes targets 3 val episodes, but a scene is
    10 and scenes move whole, so keying on the scene sends a third of the
    corpus past the target in one step.
    """

    def _run(self, tmp_path, adapter_id, scene_key):
        _register(_episodes(tmp_path, n_scenes=2, per_scene=10, packed=True),
                  adapter_id)
        reader = _reader(tmp_path, VIEW, adapter_id, scene_key=scene_key)
        return materialize_train_tree(reader, val_ratio=0.15, seed=0)

    def test_two_scene_corpus_overshoots_when_keyed_on_the_scene(
            self, tmp_path, request):
        report = self._run(tmp_path, f"fake_{request.node.name}", "prefix")

        assert (report.n_train, report.n_val) == (10, 10)

    def test_per_episode_key_hits_the_requested_ratio(self, tmp_path, request):
        report = self._run(tmp_path, f"fake_{request.node.name}", "episode")

        assert report.n_val == 3
        assert report.n_train == 17

    def test_per_episode_key_still_writes_both_sides_whole(
            self, tmp_path, request):
        adapter_id = f"fake_{request.node.name}"
        report = self._run(tmp_path, adapter_id, "episode")
        reader = _reader(tmp_path, VIEW, adapter_id, scene_key="episode")

        tree = reader.train_tree
        written = sorted(
            p.stem
            for split in ("train", "val")
            for p in (tree / "videos" / split).glob("*.mp4"))
        assert len(written) == report.n_written == 20
        assert len(set(written)) == 20


class TestRewrite:
    """A second run leaves the tree describing only that run."""

    def test_an_episode_that_changes_side_is_not_left_in_both(
            self, tmp_path, request):
        adapter_id = f"fake_{request.node.name}"
        _register(_episodes(tmp_path, n_scenes=8, per_scene=2, packed=True),
                  adapter_id)
        reader = _reader(tmp_path, VIEW, adapter_id)
        tree = reader.train_tree

        materialize_train_tree(reader, val_ratio=0.5, seed=0)
        wide_val = {p.stem for p in (tree / "videos" / "val").glob("*.mp4")}
        materialize_train_tree(reader, val_ratio=0.125, seed=0)
        narrow_val = {p.stem for p in (tree / "videos" / "val").glob("*.mp4")}

        moved = wide_val - narrow_val
        assert moved, "the two ratios must disagree for this to test anything"
        train_now = {p.stem for p in (tree / "videos" / "train").glob("*.mp4")}
        assert moved <= train_now
        assert not (train_now & narrow_val)

    def test_an_episode_the_corpus_drops_is_removed(self, tmp_path, request):
        adapter_id = f"fake_{request.node.name}"
        pool = _episodes(tmp_path, n_scenes=4, per_scene=2, packed=True)
        n = len(pool)
        _register(pool, adapter_id)  # the fake yields whatever `pool` holds
        reader = _reader(tmp_path, VIEW, adapter_id)
        tree = reader.train_tree

        materialize_train_tree(reader, val_ratio=0.25, seed=0)
        dropped = pool.pop().episode_id
        materialize_train_tree(reader, val_ratio=0.25, seed=0)

        left = {p.stem
                for split in ("train", "val")
                for p in (tree / "videos" / split).glob("*.mp4")}
        assert dropped not in left
        assert len(left) == n - 1

    def test_annotations_are_rewritten_with_the_videos(self, tmp_path, request):
        adapter_id = f"fake_{request.node.name}"
        _register(_episodes(tmp_path, n_scenes=8, per_scene=2, packed=True),
                  adapter_id)
        reader = _reader(tmp_path, VIEW, adapter_id)
        tree = reader.train_tree

        materialize_train_tree(reader, val_ratio=0.5, seed=0)
        materialize_train_tree(reader, val_ratio=0.125, seed=0)

        for split in ("train", "val"):
            videos = {p.stem for p in (tree / "videos" / split).glob("*.mp4")}
            labels = {p.stem
                      for p in (tree / "annotation" / split).glob("*.json")}
            assert videos == labels
