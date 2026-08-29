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

CORPUS = "test_corpus"


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
        reader_id=f"airbot_mmk2.{CORPUS}.{view.view_id}", robot="airbot_mmk2",
        view=view,
        train=TrainSource(corpus=CORPUS, adapter=adapter_id, root=str(tmp_path),
                          cameras=("a_left", "b_right"), scene_key=scene_key))


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
            reader_id=f"airbot_mmk2.{CORPUS}.mv2_row", robot="airbot_mmk2",
            view=VIEW,
            train=TrainSource(corpus=CORPUS, adapter=adapter_id,
                              root=str(tmp_path), cameras=("a_left", "b_right")),
            status="blocked: this corpus logs no state")
        with pytest.raises(ValueError, match="not trainable"):
            materialize_train_tree(reader)

    def test_reader_without_a_source_is_refused(self, tmp_path):
        reader = ReaderSpec(reader_id=f"airbot_mmk2.{CORPUS}.mv2_row",
                            robot="airbot_mmk2", view=VIEW)
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


def _solid(path, color, width=32, height=24, n_frames=8, fps=10) -> str:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"color=c={color}:size={width}x{height}:rate={fps}:duration="
               f"{n_frames / fps}", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True)
    return str(path)


def _written(reader) -> list:
    return [p for split in ("train", "val")
            for p in (reader.train_tree / "videos" / split).glob("*.mp4")]


def _first_frame(path, width, height) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(height, width, 3)


class TestPanelOrder:
    """Panel order is the declared camera order, not the alphabetical one."""

    def _episode(self, tmp_path, cameras):
        views = {"red": _solid(tmp_path / "red.mp4", "red"),
                 "blue": _solid(tmp_path / "blue.mp4", "blue")}
        return RawEpisode(
            episode_id="task0__0", joints=np.zeros((8, 7), dtype=np.float32),
            fps=10.0, scene_key="task0", source_path="/corpus/task0__0",
            views={c: views[c] for c in cameras})

    def _left_panel(self, tmp_path, adapter_id, cameras):
        _register([self._episode(tmp_path, cameras)], adapter_id)
        reader = ReaderSpec(
            reader_id=f"airbot_mmk2.{CORPUS}.mv2_row", robot="airbot_mmk2",
            view=VIEW,
            train=TrainSource(corpus=CORPUS, adapter=adapter_id,
                              root=str(tmp_path), cameras=cameras))
        materialize_train_tree(reader, val_ratio=0.25, seed=0)
        (video,) = _written(reader)
        return _first_frame(video, 64, 24)[12, 16]

    def test_the_first_declared_camera_is_the_first_panel(self, tmp_path,
                                                          request):
        left = self._left_panel(tmp_path, f"fake_{request.node.name}",
                                ("red", "blue"))
        assert left[0] > left[2]

    def test_reversing_the_declaration_reverses_the_panels(self, tmp_path,
                                                           request):
        left = self._left_panel(tmp_path, f"fake_{request.node.name}",
                                ("blue", "red"))
        assert left[2] > left[0]


class TestSinglePanel:
    """A ``none`` packing still rescales: corpus and bench differ in size."""

    def test_one_camera_is_rescaled_to_the_declared_panel(self, tmp_path,
                                                          request):
        adapter_id = f"fake_{request.node.name}"
        episode = RawEpisode(
            episode_id="task0__0", joints=np.zeros((8, 7), dtype=np.float32),
            fps=10.0, scene_key="task0", source_path="/corpus/task0__0",
            views={"global": _clip(tmp_path / "big.mp4", width=64, height=48)})
        _register([episode], adapter_id)
        reader = ReaderSpec(
            reader_id=f"airbot_mmk2.{CORPUS}.sv1", robot="airbot_mmk2",
            view=SV1,
            train=TrainSource(corpus=CORPUS, adapter=adapter_id,
                              root=str(tmp_path), cameras=("global",)))

        report = materialize_train_tree(reader, val_ratio=0.25, seed=0)

        assert report.n_written == 1
        (video,) = _written(reader)
        assert _first_frame(video, 32, 24).shape == (24, 32, 3)

    def test_two_cameras_into_one_panel_is_refused(self, tmp_path, request):
        adapter_id = f"fake_{request.node.name}"
        _register(_episodes(tmp_path, n_scenes=1, per_scene=1), adapter_id)
        reader = ReaderSpec(
            reader_id=f"airbot_mmk2.{CORPUS}.sv1", robot="airbot_mmk2",
            view=ViewSpec(view_id="sv1", n_views=1, packing="none",
                          n_panels=2, panel=(32, 24)),
            train=TrainSource(corpus=CORPUS, adapter=adapter_id,
                              root=str(tmp_path),
                              cameras=("a_left", "b_right")))

        report = materialize_train_tree(reader, val_ratio=0.25, seed=0)

        assert report.n_written == 0
        assert all("single panel" in reason for _, reason in report.skipped)


GRID_BR_BLANK = ViewSpec(view_id="mv4_grid_br_blank", n_views=3,
                         packing="grid2x2", n_panels=4, panels=(0, 1, 2),
                         panel=(32, 24))


class TestGridPacking:
    """A 2x2 grid whose fourth cell the corpus does not fill."""

    def _reader(self, tmp_path, adapter_id, view, cameras):
        return ReaderSpec(
            reader_id=f"airbot_mmk2.{CORPUS}.{view.view_id}",
            robot="airbot_mmk2", view=view,
            train=TrainSource(corpus=CORPUS, adapter=adapter_id,
                              root=str(tmp_path), cameras=cameras))

    def test_three_panels_still_fill_the_declared_frame(self, tmp_path,
                                                        request):
        adapter_id = f"fake_{request.node.name}"
        episode = RawEpisode(
            episode_id="task0__0", joints=np.zeros((8, 7), dtype=np.float32),
            fps=10.0, scene_key="task0", source_path="/corpus/task0__0",
            views={c: _solid(tmp_path / f"{c}.mp4", "red")
                   for c in ("a", "b", "c")})
        _register([episode], adapter_id)
        reader = self._reader(tmp_path, adapter_id, GRID_BR_BLANK,
                              ("a", "b", "c"))

        report = materialize_train_tree(reader, val_ratio=0.25, seed=0)

        assert report.n_written == 1, report.skipped
        (video,) = _written(reader)
        frame = _first_frame(video, 64, 48)
        assert frame[12, 16, 0] > frame[12, 16, 2]
        assert frame[36, 48].tolist() == [0, 0, 0]


def _row_subset(panels: tuple[int, ...]) -> ViewSpec:
    return ViewSpec(view_id="mv4_row_subset", n_views=len(panels),
                    packing="width", n_panels=4, panels=panels, panel=(32, 24))


class TestWidthSubset:
    """A row packing whose corpus fills only some of the panels."""

    def _frame(self, tmp_path, adapter_id, view, cameras):
        episode = RawEpisode(
            episode_id="task0__0", joints=np.zeros((8, 7), dtype=np.float32),
            fps=10.0, scene_key="task0", source_path="/corpus/task0__0",
            views={c: _solid(tmp_path / f"{c}.mp4", c) for c in cameras})
        _register([episode], adapter_id)
        reader = ReaderSpec(
            reader_id=f"airbot_mmk2.{CORPUS}.{view.view_id}",
            robot="airbot_mmk2", view=view,
            train=TrainSource(corpus=CORPUS, adapter=adapter_id,
                              root=str(tmp_path), cameras=cameras))

        report = materialize_train_tree(reader, val_ratio=0.25, seed=0)

        assert report.n_written == 1, report.skipped
        (video,) = _written(reader)
        return _first_frame(video, 128, 24)

    def test_the_dropped_panels_are_black_and_the_frame_keeps_its_width(
            self, tmp_path, request):
        frame = self._frame(tmp_path, f"fake_{request.node.name}",
                            _row_subset((0, 1)),
                            ("red", "blue", "green", "white"))

        assert frame[12, 16, 0] > frame[12, 16, 2]
        assert frame[12, 48, 2] > frame[12, 48, 0]
        assert frame[12, 80].tolist() == [0, 0, 0]
        assert frame[12, 112].tolist() == [0, 0, 0]

    def test_a_panel_lands_at_the_index_it_is_declared_at(self, tmp_path,
                                                          request):
        frame = self._frame(tmp_path, f"fake_{request.node.name}",
                            _row_subset((1, 3)),
                            ("red", "blue", "green", "white"))

        assert frame[12, 16].tolist() == [0, 0, 0]
        assert frame[12, 48, 2] > frame[12, 48, 0]
        assert frame[12, 80].tolist() == [0, 0, 0]
        assert frame[12, 112].min() > 200
