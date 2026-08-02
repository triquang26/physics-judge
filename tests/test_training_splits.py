"""Scene-stratified episode splitting (see kinescore/training/splits.py).

Pins the defect this module fixes -- a val set with no ratio, no seed, no
stratification -- and the specific behaviours the docstring promises:
scene-disjointness, reproducibility, graceful behaviour on a scene-less
(purely numeric) id pool, and a close-to-target ratio when scenes are
fine-grained relative to it.
"""
import pytest

from kinescore.training.splits import default_scene_key, stratified_episode_split


def _task_ids(task: str, n: int, start: int = 0) -> list[str]:
    return [f"{task}_{i:03d}" for i in range(start, start + n)]


class TestDefaultSceneKey:
    def test_strips_trailing_numeric_suffix(self):
        assert default_scene_key("close_cardboard_box_000026") == "close_cardboard_box"
        assert default_scene_key("hit_mark-7") == "hit_mark"

    def test_purely_numeric_id_falls_back_to_itself(self):
        # DROID-style bare integer ids carry no scene signal to strip --
        # falling back to the id (not "") keeps every such episode its own
        # one-episode scene instead of silently collapsing the whole pool.
        assert default_scene_key("0") == "0"
        assert default_scene_key("100") == "100"


class TestStratifiedEpisodeSplit:
    def test_rejects_empty_pool(self):
        with pytest.raises(ValueError):
            stratified_episode_split([])

    @pytest.mark.parametrize("bad_ratio", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_out_of_range_ratio(self, bad_ratio):
        with pytest.raises(ValueError):
            stratified_episode_split(["a_1", "b_1"], val_ratio=bad_ratio)

    def test_train_val_partition_the_pool_exactly(self):
        ids = _task_ids("close_box", 10) + _task_ids("hit_mark", 5) + _task_ids("insert_gear", 8)
        train, val = stratified_episode_split(ids, val_ratio=0.15, seed=0)
        assert set(train) | set(val) == set(ids)
        assert set(train) & set(val) == set()

    def test_no_scene_split_across_train_and_val(self):
        ids = _task_ids("close_box", 10) + _task_ids("hit_mark", 5) + _task_ids("insert_gear", 8)
        train, val = stratified_episode_split(ids, val_ratio=0.15, seed=0)
        scenes_train = {default_scene_key(x) for x in train}
        scenes_val = {default_scene_key(x) for x in val}
        assert not (scenes_train & scenes_val)
        assert val  # a non-trivial pool must not silently produce an empty val

    def test_reproducible_given_the_same_seed(self):
        ids = _task_ids("close_box", 10) + _task_ids("hit_mark", 5) + _task_ids("insert_gear", 8)
        train_a, val_a = stratified_episode_split(ids, val_ratio=0.15, seed=7)
        train_b, val_b = stratified_episode_split(ids, val_ratio=0.15, seed=7)
        assert train_a == train_b
        assert val_a == val_b

    def test_different_seeds_can_choose_different_scenes(self):
        ids = _task_ids("a", 4) + _task_ids("b", 4) + _task_ids("c", 4) + _task_ids("d", 4)
        splits = {stratified_episode_split(ids, val_ratio=0.25, seed=s)[1] for s in range(8)}
        assert len(splits) > 1, "every seed picked the same val scene -- seed is a no-op"

    def test_achieved_ratio_tracks_target_for_fine_grained_scenes(self):
        # 20 scenes of 5 episodes each -- fine-grained relative to a 0.2
        # target, so the achieved ratio should land exactly on target.
        ids = [f"s{g}_{i}" for g in range(20) for i in range(5)]
        train, val = stratified_episode_split(ids, val_ratio=0.2, seed=3)
        assert len(val) / len(ids) == pytest.approx(0.2, abs=1e-9)

    def test_numeric_ids_still_produce_a_nonempty_val(self):
        # Each bare-integer id is its own scene (see default_scene_key), so
        # this degrades to an ordinary (still seeded, still reproducible)
        # per-episode split -- must still respect the ratio reasonably and
        # never silently return an empty val for a large enough pool.
        ids = [str(i) for i in range(50)]
        train, val = stratified_episode_split(ids, val_ratio=0.15, seed=0)
        assert set(train) | set(val) == set(ids)
        assert 0 < len(val) < len(ids)

    def test_small_pool_below_one_episode_of_val_is_honestly_empty(self):
        # val_ratio=0.15 on 3 episodes rounds to a target of 0 -- returning
        # an empty val here is the requested ratio, not a bug to paper over.
        train, val = stratified_episode_split(["a_1", "a_2", "a_3"], val_ratio=0.15, seed=0)
        assert val == ()
        assert set(train) == {"a_1", "a_2", "a_3"}

    def test_custom_scene_key_fn_is_respected(self):
        ids = ["ep0", "ep1", "ep2", "ep3", "ep4", "ep5"]
        # group by parity instead of the default trailing-digit strip
        key_fn = lambda ep: "even" if int(ep[2:]) % 2 == 0 else "odd"  # noqa: E731
        train, val = stratified_episode_split(ids, val_ratio=0.4, seed=0, scene_key_fn=key_fn)
        groups_train = {key_fn(x) for x in train}
        groups_val = {key_fn(x) for x in val}
        assert not (groups_train & groups_val)

    def test_directory_based_split_is_unaffected(self):
        """This module adds an alternative to, not a replacement for, the
        existing directory-based `{cache_root}/{train,val}/*.pt` layout --
        see kinescore.training.datasets.load_split, which does not call
        anything in this module at all."""
        import inspect

        from kinescore.training import datasets
        assert "stratified" not in inspect.getsource(datasets.load_split)
