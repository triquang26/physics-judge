"""``load_split_stratified``: one pool directory -> scene-respecting (train, val).

CPU-only, tiny synthetic cache/annotation files (no real backbone, no real
robot) -- exercises the actual glob -> stratify -> two-pass-load integration,
not just :mod:`kinescore.training.splits` in isolation
(``tests/test_training_splits.py`` already covers the splitter's own logic).
"""
import json
import os

import torch

from kinescore.training.cache import CacheHeader, write_cache
from kinescore.training.datasets import load_split, load_split_stratified


def _write_episode(cache_dir: str, ann_dir: str, ep_id: str, *, n_frames: int,
                   n_joints: int = 3, n_tokens: int = 2, embed_dim: int = 4) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    feat = torch.randn(n_frames, n_tokens, embed_dim, dtype=torch.float16)
    header = CacheHeader(
        schema=1, view_layout_key="1cam", n_views=1, tokens_per_view=n_tokens,
        backbone_id="fake@224:p1", source_path=f"{ep_id}.mp4",
        n_frames=n_frames, embed_dim=embed_dim)
    write_cache(os.path.join(cache_dir, f"{ep_id}.pt"), feat, header)

    label = {
        "joint_source": "real",
        "observation.state.joint_position": [[0.1 * i] * n_joints
                                             for i in range(n_frames)],
    }
    with open(os.path.join(ann_dir, f"{ep_id}.json"), "w") as f:
        json.dump(label, f)


def _pool(tmp_path, task_sizes: dict[str, int], n_frames: int = 4):
    cache_dir = str(tmp_path / "cache" / "all")
    ann_dir = str(tmp_path / "ann" / "all")
    ids = []
    for task, n in task_sizes.items():
        for i in range(n):
            ep_id = f"{task}_{i:03d}"
            _write_episode(cache_dir, ann_dir, ep_id, n_frames=n_frames)
            ids.append(ep_id)
    return str(tmp_path / "cache"), str(tmp_path / "ann"), ids


class TestLoadSplitStratified:
    def test_train_val_disjoint_and_scene_respecting(self, tmp_path):
        cache_root, annotation_root, ids = _pool(
            tmp_path, {"task_a": 6, "task_b": 4, "task_c": 5})

        train, val = load_split_stratified(
            cache_root, annotation_root, "all", down_sample=1,
            val_ratio=0.3, seed=0)

        assert set(train.episode_ids) | set(val.episode_ids) == set(ids)
        assert not (set(train.episode_ids) & set(val.episode_ids))

        def scene(ep):
            return ep.rsplit("_", 1)[0]

        train_scenes = {scene(e) for e in train.episode_ids}
        val_scenes = {scene(e) for e in val.episode_ids}
        assert not (train_scenes & val_scenes)

    def test_reproducible_given_the_same_seed(self, tmp_path):
        cache_root, annotation_root, _ = _pool(
            tmp_path, {"task_a": 5, "task_b": 5, "task_c": 5})
        train1, val1 = load_split_stratified(
            cache_root, annotation_root, "all", down_sample=1, seed=3)
        train2, val2 = load_split_stratified(
            cache_root, annotation_root, "all", down_sample=1, seed=3)
        assert train1.episode_ids == train2.episode_ids
        assert val1.episode_ids == val2.episode_ids

    def test_frames_and_joints_flatten_correctly(self, tmp_path):
        cache_root, annotation_root, ids = _pool(
            tmp_path, {"task_a": 4, "task_b": 4}, n_frames=3)
        train, val = load_split_stratified(
            cache_root, annotation_root, "all", down_sample=1, val_ratio=0.4, seed=1)
        assert train.n_frames == train.q.shape[0] == train.feats.shape[0]
        assert val.n_frames == val.q.shape[0] == val.feats.shape[0]
        assert train.n_frames + val.n_frames == 3 * len(ids)
        assert train.n_joints == 3

    def test_directory_based_load_split_still_works_unchanged(self, tmp_path):
        # existing behaviour: separate train/ and val/ directories, no
        # stratification involved at all.
        cache_dir_train = str(tmp_path / "cache" / "train")
        cache_dir_val = str(tmp_path / "cache" / "val")
        ann_dir_train = str(tmp_path / "ann" / "train")
        ann_dir_val = str(tmp_path / "ann" / "val")
        _write_episode(cache_dir_train, ann_dir_train, "ep0", n_frames=4)
        _write_episode(cache_dir_train, ann_dir_train, "ep1", n_frames=4)
        _write_episode(cache_dir_val, ann_dir_val, "ep2", n_frames=4)

        train = load_split(str(tmp_path / "cache"), str(tmp_path / "ann"),
                           "train", down_sample=1)
        val = load_split(str(tmp_path / "cache"), str(tmp_path / "ann"),
                         "val", down_sample=1)
        assert set(train.episode_ids) == {"ep0", "ep1"}
        assert set(val.episode_ids) == {"ep2"}
