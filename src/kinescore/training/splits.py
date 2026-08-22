"""Scene-stratified train/val episode splitting.

:func:`stratified_episode_split` groups a pool of episode ids by a scene/task
key and keeps every scene entirely on one side of the split. Two episodes of
the same scene are far more alike than two episodes of different tasks, so
splitting them across train and val leaks scene-specific texture (lighting,
background, object identity) into what is supposed to measure
generalisation. See :func:`default_scene_key` for how the key is derived when
the caller does not supply one.
"""
from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence

__all__ = ["default_scene_key", "stratified_episode_split", "DEFAULT_VAL_RATIO"]

#: Default val fraction (of episode COUNT, not scene count -- see
#: :func:`stratified_episode_split`'s docstring for why the target is
#: computed against episodes even though whole scenes move together).
DEFAULT_VAL_RATIO = 0.15

_TRAILING_INDEX_RE = re.compile(r"[-_]?(\d+)$")


def default_scene_key(episode_id: str) -> str:
    """Best-effort scene/task key derived from an episode id string.

    Strips a trailing run of digits (optionally preceded by ``-``/``_``) --
    the common ``<task>_<index>`` / ``<task>-<index>`` naming convention this
    benchmark's own data uses once an episode is identified by a single
    string (e.g. a ``close_cardboard_box`` task directory's
    ``episode_000026`` -> id ``"close_cardboard_box_000026"`` -> scene key
    ``"close_cardboard_box"``).

    A **purely numeric** episode id (DROID's own convention -- plain
    integers like ``"0"``, ``"1"``, ``"100"``, see ``docs/DATA_PREP.md``)
    strips to an empty string; falling through to that empty key would
    silently collapse every episode in the pool into ONE scene, which
    defeats stratification in the opposite direction (a val set that is
    either empty or the entire pool, chosen by an accident of the ratio
    arithmetic) -- worse than doing nothing. This function instead falls
    back to the untouched id in that case: every such episode becomes its
    own one-episode "scene", which is an honest admission that a bare
    integer carries no recoverable scene information, not a fabricated
    grouping. A caller whose episode ids are bare integers but who DOES have
    real scene/task structure (e.g. a ``task`` field in each episode's own
    annotation JSON) should pass their own ``scene_key_fn`` to
    :func:`stratified_episode_split` instead of relying on this default.
    """
    stripped = _TRAILING_INDEX_RE.sub("", episode_id)
    return stripped if stripped else episode_id


def stratified_episode_split(
    episode_ids: Sequence[str], *, val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = 0, scene_key_fn: Callable[[str], str] = default_scene_key,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split ``episode_ids`` into ``(train, val)``, keeping every scene together.

    Parameters
    ----------
    episode_ids:
        Every episode id in the pool being split (order does not matter --
        the split is keyed by scene, not position).
    val_ratio:
        Target val fraction of the total EPISODE count, in ``(0, 1)``. Exact
        only when scene sizes divide evenly; otherwise the achieved ratio is
        whatever whole-scene assignment gets closest without exceeding the
        target (see algorithm below).
    seed:
        Shuffles the scene order before greedily assigning scenes to val, so
        two different seeds can produce two different (still
        scene-respecting) splits of the same pool, and the same seed always
        reproduces the same split.
    scene_key_fn:
        ``episode_id -> scene key``; episodes with the same key always land
        on the same side. Defaults to :func:`default_scene_key`.

    Algorithm
    ---------
    Group episodes by key, shuffle the group order with ``random.Random(seed)``
    (so ties among equal-size scenes are broken reproducibly per seed, not by
    dict/insertion order), then sort ascending by GROUP SIZE (stable, so the
    shuffle still decides ties) and greedily add whole groups to val, smallest
    first, stopping just before the next addition would push the running
    count over ``round(len(episode_ids) * val_ratio)``. Smallest-first is what
    keeps the achieved ratio close to the target when scene sizes are small
    relative to it. When even the single smallest scene exceeds the target,
    that scene is still added -- an empty val set is a worse failure than an
    over-sized one -- and the achieved ratio runs high rather than reporting
    the requested ratio as if it had been achieved. First-fit-ascending
    greedy bin-packing, O(n log n), not an optimal partition.

    Returns
    -------
    (train_ids, val_ids):
        Two disjoint tuples whose union is exactly ``set(episode_ids)``
        (each id appears in exactly one, in ``episode_ids``' original
        relative order within each tuple).

    Raises
    ------
    ValueError
        If ``val_ratio`` is not strictly between 0 and 1, or ``episode_ids``
        is empty.
    """
    if not episode_ids:
        raise ValueError("episode_ids is empty -- nothing to split")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    groups: dict[str, list[str]] = {}
    for ep in episode_ids:
        groups.setdefault(scene_key_fn(ep), []).append(ep)

    scene_keys = sorted(groups)  # deterministic pre-shuffle order
    rng = random.Random(seed)
    rng.shuffle(scene_keys)
    scene_keys.sort(key=lambda k: len(groups[k]))  # stable: shuffle breaks ties

    target_val = round(len(episode_ids) * val_ratio)
    val_scene_keys: set[str] = set()
    val_count = 0
    for key in scene_keys:
        if val_count >= target_val:
            break
        size = len(groups[key])
        if val_count > 0 and val_count + size > target_val:
            break  # would overshoot and we already have SOME val episodes
        val_scene_keys.add(key)
        val_count += size

    train_ids = tuple(ep for ep in episode_ids if scene_key_fn(ep) not in val_scene_keys)
    val_ids = tuple(ep for ep in episode_ids if scene_key_fn(ep) in val_scene_keys)
    return train_ids, val_ids
