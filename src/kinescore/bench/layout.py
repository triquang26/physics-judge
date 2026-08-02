"""``DataLayout``: where a benchmark cell's clips live, and how to enumerate them.

Two implementations, one abstraction
--------------------------------------
:class:`RawHFLayout` reads the dataset exactly as the Hugging Face mirror
ships it (``video_gen_physics/<cache>/<embodiment>/output/<view_dir>/
<generator>/<horizon>/[iter_*/]...`` -- see ``kinescore.bench.sources.*``'s
docstrings for the full per-generator shape). :class:`CanonicalLayout` is the
target shape ``kinescore data ingest`` materialises everything into:
``bench/<cache>/<robot>/<view>/<generator>/<horizon>/episode_XXXX/
{pred.mp4,gt.mp4}`` plus a ``cell_card.json`` per cell. Both share the same
:class:`DataLayout` interface (``cells()``/``cell_dir()``/``validate()``) so
``kinescore.bench.ingest.Ingestor`` (and ``kinescore data verify``) can treat
"where do I read from" and "where do I write to" symmetrically.

``kinescore.core.contracts`` was checked for an existing ``DataLayout``
definition (see ``bench.sources.base``'s docstring for the identical check
against ``ClipSource``): it now defines one, matching this module's shape
exactly (``cells()``/``cell_dir(cell)``/``validate() -> list[str]``) -- so
:class:`DataLayout` below is imported from there instead of a second,
identical definition living here.

Why RawHFLayout does NOT reuse ``kinescore.bench.sources.ClipSource``
------------------------------------------------------------------------
The ``ClipSource`` plugins (``CtrlWorldSource``/``DreamDojoSource``/
``DreamGenSource``) are deliberately GT-pairing-STRICT: for scoring, an
unpaired prediction is useless and every one of them either requires a
matching ground-truth file or drops the episode (see e.g.
``DreamDojoSource``'s pred-only "dir, orphan" drop). ``kinescore data
ingest``'s job is archival, not scoring -- a dreamgen prediction with no
ground truth (which is EVERY dreamgen prediction; see the module docstring
of ``kinescore.bench.sources.dreamgen``) is still real data worth keeping in
the canonical tree. Reusing ``ClipSource`` here would materialise ZERO
dreamgen episodes, silently. So :meth:`RawHFLayout.episodes` re-implements
per-shape discovery driven by ``configs/data_spec.yaml`` (see
:mod:`kinescore.bench.data_spec`) instead: a prediction is always kept; its
ground truth is attached if present, and an episode is only DROPPED for a
missing ground truth when ``GeneratorSpec.has_ground_truth`` says this
generator is supposed to have one (ctrlworld, dreamdojo) -- never for
dreamgen, which has none by design.

Exclusion globs
-----------------
:func:`_glob_excluded` is a small ``**``-aware matcher (``**`` = zero or more
whole path segments, ``*``/``?`` = fnmatch within one segment) applied to
every candidate path relative to ``$KINESCORE_DATA_ROOT`` against
``configs/data_spec.yaml``'s ``exclude_globs`` -- see that file's header
comment for the exact list and what each entry guards against (ablations,
frame-rate experiments, a malformed HF download-link artifact, ...).

Resolved tension, worth keeping the history: an earlier ``configs/
data_spec.yaml`` also listed ``**/*_static16fps/**`` and
``**/*bimanual16fps/**`` as exclusions. Both suffixes turned out to name
real, populated dreamgen ``iter_*`` directories that are the pinned, scored
iteration for several cells (``iter_000090000_static16fps`` for franka_panda,
120 episodes; ``iter_000110000_bimanual16fps`` for aloha_bimanual, 300
episodes -- the ONLY iter present for bimanual/singleview/dreamgen -- see
``configs/benchmark.yaml``/``configs/benchmark_bimanual.yaml``'s comments and
``kinescore.bench.sources.dreamgen``'s docstring). Applied literally (as
this module always does -- it never silently carves out an ``iter_*``
exception), both globs matched those real directories too, and there was no
compensating benefit: :mod:`kinescore.bench.sources` (the plugins
``kinescore bench run`` actually reads through) never consult
``exclude_globs`` at all -- they resolve ``sources.<generator>.iter`` from
``configs/benchmark*.yaml`` as an explicit whitelist, so a directory that is
never pinned there is never read regardless of what ``exclude_globs`` says.
This module (via :meth:`RawHFLayout.cells`'s auto-pick-most-populated-iter
policy) was the ONLY place either glob had teeth, and there it was actively
harmful -- silently disqualifying the pinned directory from ever being the
auto-picked one. Both entries were removed from ``configs/data_spec.yaml``;
see that file's header comment for the fuller writeup and
``tests/test_bench_layout.py::TestExcludeGlobsMatchRealDataSpec`` for the
pinned outcome (the real ``configs/data_spec.yaml`` is loaded and checked
against both a real pinned path and an unpinned ablation).
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass

from kinescore.bench.cell import PATH_AXIS_ORDER, Cell, raw_tree_segments
from kinescore.bench.data_spec import DataSpec, GeneratorSpec
from kinescore.bench.robot_map import RobotMap
from kinescore.core.clip import ViewLayout
from kinescore.core.contracts import DataLayout

__all__ = [
    "DataLayout", "EpisodeFiles", "RawHFLayout", "CanonicalLayout",
    "sanitize_episode", "cell_card_path",
]

#: ``view`` (canonical axis value) -> camera packing, mirroring
#: ``kinescore.bench.matrix._VIEW_LAYOUTS``. Duplicated as a small local
#: constant rather than imported: ``bench.matrix`` additionally needs
#: ``BenchConfig``, which a raw disk walk (this module's whole point) must
#: not require -- ``kinescore data ingest`` has to work before any scoring
#: run config is chosen.
_VIEW_LAYOUTS: dict[str, ViewLayout] = {
    "singleview": ViewLayout(n_views=1),
    "single_view": ViewLayout(n_views=1),
    "multiview": ViewLayout(n_views=3, order=("exterior_1", "exterior_2", "wrist")),
}

_HORIZON_DIRS = frozenset({"makovian", "non_makovian"})
_ITER_RE = re.compile(r"^iter_\d+")

#: Canonical filenames every ingested episode uses, regardless of what the
#: raw generator called its files -- the whole point of ingestion is that a
#: downstream reader of the canonical tree never needs to know
#: ``pred_all_views.mp4`` vs ``full_pred.mp4`` vs ``episode_000012.mp4`` is
#: the same kind of thing.
PRED_NAME = "pred.mp4"
GT_NAME = "gt.mp4"
CELL_CARD_NAME = "cell_card.json"


def _glob_to_regex(pattern: str) -> re.Pattern:
    """Translate a ``**``-aware glob into a compiled regex over a ``/``-joined path.

    A small hand-rolled scanner rather than :func:`fnmatch.translate` (whose
    ``*`` already crosses ``/`` with no way to special-case ``**``): a
    leading/embedded ``**/`` becomes "zero or more characters ending in a
    slash" and a trailing ``/**`` becomes "optionally, a slash followed by
    anything" -- together these let e.g. ``**/tmp/**`` match ``tmp`` itself
    (a bare directory, no children yet) as well as anything nested under it,
    which is what every exclusion in ``configs/data_spec.yaml`` needs: a
    caller checks a directory against this BEFORE descending into it. A bare
    ``*``/``?`` matches within one segment only (never crosses ``/``); a
    ``[...]`` character class is passed through verbatim (used for the one
    literal ``?`` in the malformed-download-link exclusion).
    """
    i, n = 0, len(pattern)
    out: list[str] = []
    while i < n:
        if pattern[i:i + 3] == "**/":
            out.append(r"(?:.*/)?")
            i += 3
            continue
        if pattern[i:i + 3] == "/**" and i + 3 == n:
            out.append(r"(?:/.*)?")
            i += 3
            continue
        if pattern[i:i + 2] == "**":
            out.append(r".*")
            i += 2
            continue
        c = pattern[i]
        if c == "*":
            out.append(r"[^/]*")
            i += 1
        elif c == "?":
            out.append(r"[^/]")
            i += 1
        elif c == "[":
            j = pattern.find("]", i + 1)
            if j == -1:
                out.append(re.escape(c))
                i += 1
            else:
                out.append(pattern[i:j + 1])
                i = j + 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def _glob_excluded(rel_path: str, exclude_globs: tuple[str, ...]) -> bool:
    """``True`` if ``rel_path`` (``/``-joined, relative to the data root) matches any exclusion."""
    rel_path = rel_path.replace(os.sep, "/")
    return any(_glob_to_regex(pat).match(rel_path) for pat in exclude_globs)


def sanitize_episode(episode: str) -> str:
    """A raw episode id (may contain ``/``/``:``, e.g. ``"flat:0000"``,
    ``"task_a/episode_000012"``) -> a safe, single directory-name component.
    """
    safe = episode.replace("/", "__").replace(":", "_")
    return safe if safe.startswith("episode_") else f"episode_{safe}"


def cell_card_path(cell_dir: str) -> str:
    return os.path.join(cell_dir, CELL_CARD_NAME)


@dataclass(frozen=True)
class EpisodeFiles:
    """One discovered episode: a prediction, and its ground truth if any."""

    episode: str
    pred_path: str
    gt_path: str | None


class RawHFLayout(DataLayout[Cell]):
    """Reads the dataset in its as-shipped Hugging Face mirror shape.

    Parameters
    ----------
    data_root:
        ``$KINESCORE_DATA_ROOT`` (or an override) -- the directory containing
        ``video_gen_physics/``.
    robot_map:
        Resolves an on-disk ``embodiment`` + ``generator`` pair to a robot
        (see ``kinescore.bench.robot_map``); an (embodiment, generator)
        combination it does not claim is skipped, not an error (a real
        outcome -- e.g. a ``cache``/``embodiment`` combination with a
        generator directory this robot table has never heard of).
    data_spec:
        The per-generator file contract (see :mod:`kinescore.bench.data_spec`)
        -- drives :meth:`episodes`' per-``shape`` discovery and the exclusion
        globs both :meth:`cells` and :meth:`episodes` apply.
    """

    def __init__(self, data_root: str, robot_map: RobotMap, data_spec: DataSpec) -> None:
        self.data_root = str(data_root)
        self.robot_map = robot_map
        self.data_spec = data_spec
        self._root = os.path.join(self.data_root, "video_gen_physics")

    def _rel(self, *parts: str) -> str:
        return os.path.relpath(os.path.join(self._root, *parts), self.data_root)

    def _excluded(self, *parts: str) -> bool:
        return _glob_excluded(self._rel(*parts), self.data_spec.exclude_globs)

    def _listdir(self, path: str) -> list[str]:
        try:
            return sorted(os.listdir(path))
        except (FileNotFoundError, NotADirectoryError):
            return []

    def cells(self) -> Iterator[Cell]:
        """Walk the raw tree; yield one :class:`Cell` per (cache, robot, view,
        generator, horizon) combination actually present on disk.

        ``cell.view`` here is the LITERAL on-disk directory name
        (``multiview``/``singleview``/``single_view``) -- see the module
        docstring's note on why ``single_view``/``singleview`` must stay two
        distinct cells, never merged.

        For a generator with an ``iter_*`` level (dreamdojo, dreamgen), the
        CANONICAL layout has no ``iter`` path segment (see
        ``kinescore.bench.cell.PATH_AXIS_ORDER`` -- five segments, no sixth
        for iter), so exactly one iter must be chosen per (cache, robot,
        view, generator, horizon) rather than yielding one cell per iter
        found. The policy here is auto-pick-the-best-populated: every
        candidate ``iter_*`` directory is counted (via the same per-shape
        discovery :meth:`episodes` uses) and the one with the most episodes
        wins, ties broken toward the lexicographically later iter name (for
        ``iter_NNNNNN``, the higher/more recent checkpoint number). This is
        DELIBERATELY independent of any ``configs/benchmark.yaml`` scoring
        pin -- ingestion has to work before a scoring run config is chosen,
        and its job is a complete, honest archival snapshot, not
        reproducing a human-verified scoring decision. The auto-picked
        ``iter`` is recorded in ``cell_card.json`` (see
        ``kinescore.bench.ingest``) precisely so it is auditable, not a
        silent guess.
        """
        for cache in self._listdir(self._root):
            if self._excluded(cache):
                continue
            for embodiment in self._listdir(os.path.join(self._root, cache)):
                if self._excluded(cache, embodiment):
                    continue
                output_dir = os.path.join(self._root, cache, embodiment, "output")
                for view_dir in self._listdir(output_dir):
                    if self._excluded(cache, embodiment, "output", view_dir):
                        continue
                    if view_dir not in _VIEW_LAYOUTS:
                        continue
                    gen_root = os.path.join(output_dir, view_dir)
                    for generator in self._listdir(gen_root):
                        if generator not in self.data_spec.generators:
                            continue
                        if self._excluded(cache, embodiment, "output", view_dir, generator):
                            continue
                        robot = self.robot_map.resolve(embodiment=embodiment, generator=generator)
                        if robot is None:
                            continue
                        gspec = self.data_spec.generators[generator]
                        horizon_root = os.path.join(gen_root, generator)
                        for horizon in self._listdir(horizon_root):
                            if horizon not in _HORIZON_DIRS:
                                continue  # e.g. dreamgen's camera-named siblings trap
                            if self._excluded(cache, embodiment, "output", view_dir,
                                              generator, horizon):
                                continue
                            horizon_dir = os.path.join(horizon_root, horizon)
                            best_iter = None
                            if gspec.has_iter_level:
                                candidates = [
                                    n for n in self._listdir(horizon_dir)
                                    if _ITER_RE.match(n) and not self._excluded(
                                        cache, embodiment, "output", view_dir,
                                        generator, horizon, n)]
                                if not candidates:
                                    continue
                                counts = {n: sum(1 for _ in self._episodes_for_shape(
                                              os.path.join(horizon_dir, n), gspec))
                                         for n in candidates}
                                best_iter = max(candidates, key=lambda n: (counts[n], n))
                            yield Cell(cache=cache, robot=robot, view=view_dir,
                                      generator=generator, horizon=horizon,
                                      embodiment=embodiment,
                                      view_layout=_VIEW_LAYOUTS[view_dir], iter=best_iter)

    def cell_dir(self, cell: Cell) -> str:
        """The raw directory ``cell``'s episodes live directly under."""
        segments = raw_tree_segments(cache=cell.cache, embodiment=cell.embodiment,
                                     view_dir=cell.view, generator=cell.generator,
                                     horizon=cell.horizon, iter=cell.iter)
        return os.path.join(self.data_root, *segments)

    def episodes(self, cell: Cell) -> Iterator[EpisodeFiles]:
        """Discover ``cell``'s episodes, per its generator's ``shape``.

        A prediction is ALWAYS kept. Its ground truth is attached if the
        matching file exists; an episode is only DROPPED for a missing
        ground truth when :attr:`~kinescore.bench.data_spec.GeneratorSpec.has_ground_truth`
        is true for this generator (ctrlworld, dreamdojo) -- never for
        dreamgen, which has none, ever, by design (see the module docstring's
        "Why RawHFLayout does NOT reuse ClipSource" section).
        """
        gspec = self.data_spec.generators[cell.generator]
        yield from self._episodes_for_shape(self.cell_dir(cell), gspec)

    def _episodes_for_shape(self, cell_dir: str, gspec: GeneratorSpec) -> Iterator[EpisodeFiles]:
        """Dispatch on ``gspec.shape`` -- shared by :meth:`episodes` (given a
        full cell's directory) and :meth:`cells` (counting candidate iter
        directories before one is picked -- see that method's docstring).
        """
        if gspec.shape == "episode_dir":
            yield from self._episodes_episode_dir(cell_dir, gspec)
        elif gspec.shape == "task_episode":
            yield from self._episodes_task_episode(cell_dir, gspec)
        elif gspec.shape == "flat_or_dir":
            yield from self._episodes_flat_or_dir(cell_dir, gspec)
        else:
            raise AssertionError(f"unhandled shape {gspec.shape!r}")  # data_spec validates this

    def _episodes_episode_dir(self, cell_dir: str, gspec: GeneratorSpec) -> Iterator[EpisodeFiles]:
        for name in self._listdir(cell_dir):
            episode_dir = os.path.join(cell_dir, name)
            if not name.startswith("episode_") or not os.path.isdir(episode_dir):
                continue
            pred_path = os.path.join(episode_dir, gspec.pred_filename)
            if not os.path.isfile(pred_path):
                continue
            gt_path = os.path.join(episode_dir, gspec.gt_filename) if gspec.gt_filename else None
            if gt_path is not None and not os.path.isfile(gt_path):
                gt_path = None
            if gspec.has_ground_truth and gt_path is None:
                continue  # supposed to have one; drop and let the caller count it
            yield EpisodeFiles(episode=name, pred_path=pred_path, gt_path=gt_path)

    def _episodes_task_episode(self, cell_dir: str, gspec: GeneratorSpec) -> Iterator[EpisodeFiles]:
        """dreamgen's ``<task>/episode_*.mp4`` shape -- confirmed NOT uniform.

        Some robots' iters wrap every episode in a task directory
        (single_arm: a single ``global/``); others have episodes directly
        under the iter directory with no task level at all (humanoid:
        ``iter_000090000/episode_000200.mp4``, no wrapper). Both are real,
        verified against the downloaded tree -- see
        ``kinescore.bench.sources.dreamgen``'s docstring for the same finding
        at the ClipSource layer. Both are checked; a video file directly
        under ``cell_dir`` is treated as a taskless episode (``episode`` has
        no ``/``), never silently missed by only looking one level down.
        """
        for pred_path in sorted(_glob_join(cell_dir, gspec.pred_glob)):
            episode = os.path.splitext(os.path.basename(pred_path))[0]
            yield EpisodeFiles(episode=episode, pred_path=pred_path, gt_path=None)
        for task in self._listdir(cell_dir):
            task_dir = os.path.join(cell_dir, task)
            if task.startswith("_") or not os.path.isdir(task_dir):
                continue
            for pred_path in sorted(_glob_join(task_dir, gspec.pred_glob)):
                episode = os.path.splitext(os.path.basename(pred_path))[0]
                yield EpisodeFiles(episode=f"{task}/{episode}", pred_path=pred_path, gt_path=None)

    def _episodes_flat_or_dir(self, cell_dir: str, gspec: GeneratorSpec) -> Iterator[EpisodeFiles]:
        for pred_path in sorted(_glob_join(cell_dir, gspec.flat_pred_glob)):
            name = os.path.basename(pred_path)
            suffix = gspec.flat_pred_glob.lstrip("*")  # "_pred.mp4"
            idx = name[: -len(suffix)] if suffix and name.endswith(suffix) else name
            gt_path = os.path.join(cell_dir, f"{idx}{gspec.flat_gt_glob.lstrip('*')}")
            if not os.path.isfile(gt_path):
                if gspec.has_ground_truth:
                    continue
                gt_path = None
            yield EpisodeFiles(episode=f"flat:{idx}", pred_path=pred_path, gt_path=gt_path)

        for name in self._listdir(cell_dir):
            episode_dir = os.path.join(cell_dir, name)
            if not name.startswith("episode_") or not os.path.isdir(episode_dir):
                continue
            idx = name[len("episode_"):]
            pred_path = os.path.join(episode_dir, gspec.dir_pred_filename)
            if not os.path.isfile(pred_path):
                continue
            gt_path = os.path.join(episode_dir, gspec.dir_gt_filename)
            if not os.path.isfile(gt_path):
                if gspec.has_ground_truth:
                    continue  # pred-only "dir, orphan" -- unscoreable, dropped (see dreamdojo docstring)
                gt_path = None
            yield EpisodeFiles(episode=f"dir:{idx}", pred_path=pred_path, gt_path=gt_path)

    def validate(self) -> list[str]:
        """Problems found walking the raw tree: unresolvable (embodiment, generator)
        pairs and generator directories not in ``data_spec.generators`` -- both
        reported (not raised) so a caller can decide whether they matter.
        """
        problems: list[str] = []
        for cache in self._listdir(self._root):
            for embodiment in self._listdir(os.path.join(self._root, cache)):
                output_dir = os.path.join(self._root, cache, embodiment, "output")
                for view_dir in self._listdir(output_dir):
                    if view_dir not in _VIEW_LAYOUTS:
                        problems.append(
                            f"{cache}/{embodiment}/output/{view_dir}: unrecognised view "
                            f"directory name")
                        continue
                    for generator in self._listdir(os.path.join(output_dir, view_dir)):
                        if generator not in self.data_spec.generators:
                            continue  # not a benchmark generator dir -- silently not ours
                        robot = self.robot_map.resolve(embodiment=embodiment, generator=generator)
                        if robot is None:
                            problems.append(
                                f"{cache}/{embodiment}/output/{view_dir}/{generator}: no "
                                f"robot in robot_map.yaml claims (embodiment={embodiment!r}, "
                                f"generator={generator!r})")
        return problems


def _glob_join(base: str, pattern: str) -> list[str]:
    import glob as _glob
    return _glob.glob(os.path.join(base, pattern))


class CanonicalLayout(DataLayout[Cell]):
    """The target ingested shape: ``<root>/<cache>/<robot>/<view>/<generator>/
    <horizon>/episode_XXXX/{pred.mp4,gt.mp4}`` + one ``cell_card.json`` per cell.

    Path-segment order is :data:`~kinescore.bench.cell.PATH_AXIS_ORDER` --
    the one place that ordering is declared; this class (and
    ``kinescore.bench.ingest.Ingestor``) never hand-writes the segment order
    itself.
    """

    def __init__(self, root: str) -> None:
        self.root = str(root)

    def cell_dir(self, cell: Cell) -> str:
        return os.path.join(self.root, *(getattr(cell, axis) for axis in PATH_AXIS_ORDER))

    def episode_dir(self, cell: Cell, episode: str) -> str:
        return os.path.join(self.cell_dir(cell), sanitize_episode(episode))

    def cells(self) -> Iterator[Cell]:
        """Reconstruct cells from ``cell_card.json`` sidecars already on disk.

        Read back from the card (not re-derived from the path segments) so
        this is exactly what :meth:`~kinescore.bench.ingest.Ingestor.run`
        wrote, including ``iter``/``embodiment``, which are not fully
        recoverable from :data:`~kinescore.bench.cell.PATH_AXIS_ORDER` alone.
        """
        if not os.path.isdir(self.root):
            return
        for dirpath, _dirnames, filenames in os.walk(self.root):
            if CELL_CARD_NAME not in filenames:
                continue
            with open(os.path.join(dirpath, CELL_CARD_NAME)) as f:
                card = json.load(f)
            yield Cell(cache=card["cache"], robot=card["robot"], view=card["view"],
                      generator=card["generator"], horizon=card["horizon"],
                      embodiment=card["embodiment"],
                      view_layout=_VIEW_LAYOUTS.get(card["view"], ViewLayout(n_views=1)),
                      iter=card.get("iter"))

    def validate(self) -> list[str]:
        """Broken symlinks and cell dirs whose ``cell_card.json`` is missing/malformed."""
        problems: list[str] = []
        if not os.path.isdir(self.root):
            return problems
        for dirpath, _dirnames, filenames in os.walk(self.root):
            for name in filenames:
                p = os.path.join(dirpath, name)
                if os.path.islink(p) and not os.path.exists(p):
                    problems.append(f"broken symlink: {p} -> {os.readlink(p)}")
            if CELL_CARD_NAME in filenames:
                card_path = os.path.join(dirpath, CELL_CARD_NAME)
                try:
                    with open(card_path) as f:
                        card = json.load(f)
                    missing = [k for k in ("cache", "robot", "view", "generator", "horizon",
                                           "embodiment") if k not in card]
                    if missing:
                        problems.append(f"{card_path}: missing key(s) {missing}")
                except (OSError, json.JSONDecodeError) as exc:
                    problems.append(f"{card_path}: unreadable ({exc})")
        return problems
