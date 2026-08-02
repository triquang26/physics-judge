"""dreamgen plugin: predictions under a pinned iter_ dir, gt from the sibling input/ tree.

Layout (verified against the HF API, see ``bench.sources``'s docstring)::

    {data_root}/video_gen_physics/{cache}/{embodiment}/output/{view_dir}/
        dreamgen/{horizon}/iter_XXXXXX/<task>/episode_XXXXXX.mp4 (+ .txt caption)

``{embodiment}`` is :attr:`~kinescore.bench.cell.Cell.embodiment` (derived
from ``cell.robot`` -- see ``bench.cell``'s module docstring): dreamgen
under ``embodiment=humanoid`` is Fourier GR-1 (``robot=fourier_gr1``), never
Airbot MMK2 -- ``robot_map.yaml`` only lists ``ctrlworld`` as airbot_mmk2's
generator, so ``kinescore.bench.matrix`` never builds an airbot_mmk2 x
dreamgen cell.

dreamgen never writes a ground-truth file next to its prediction; the
``gt_from: input`` design (below) was meant to read it from the sibling
``input/`` tree instead.

.. warning::
   **Confirmed, not just unverified: NO ground-truth video exists anywhere
   in dreamgen's ``input/`` tree, for either embodiment, under either
   ``view_dir``.** A previous version of this docstring called the
   ``<task>/episode_XXXXXX.mp4`` shape below "inferred, not directly
   confirmed" -- it has since been checked directly against the downloaded
   tree on disk (both ``humanoid`` and ``single_arm``, ``singleview`` AND
   ``multiview``) and the assumption does not hold: every ``input/.../
   dreamgen/`` subtree contains only ``batch_input_*.json`` manifests and a
   ``first_frames/episode_XXXXXX.png`` directory (ONE conditioning frame per
   episode, not a video) -- never a per-episode ground-truth video, at any
   nesting depth. Separately, ``output/.../dreamgen/{horizon}/iter_XXXXXX/``
   itself is not reliably ``<task>/episode_XXXXXX.mp4`` either: humanoid's
   iters have episodes directly under the iter directory with NO task level
   at all (``iter_000090000/episode_000200.mp4``); single_arm's happen to
   still glob correctly only because its export wraps every episode in a
   single ``global/`` directory that incidentally satisfies the "one
   directory per task" shape this plugin was written for.
   Net effect: **every dreamgen cell currently discovers zero manifest
   rows**, regardless of which iter is pinned -- not because of a wrong
   pin (the pins in ``configs/benchmark.yaml`` are independently verified
   correct against the predictions that DO exist), but because the
   gt-matching design this plugin implements has no ground truth to find.
   This is a real, separate, pre-existing gap -- fixing it means deciding
   where dreamgen's ground truth should actually come from (a different
   tree entirely? no paired GT at all, only unpaired absolute-magnitude
   scoring?), which is new design work, not a discovery-layout bug fix, and
   is intentionally NOT attempted here. See the top-level task report for
   the full investigation.

Assumed (and now known partially wrong, see above) layout::

    {data_root}/video_gen_physics/{baseline_cache}/{embodiment}/input/{view_dir}/
        dreamgen/{horizon}/<task>/episode_XXXXXX.mp4

dreamgen has several ``iter_*`` directories on disk, and -- confirmed against
the live inventory -- the RIGHT one differs per robot and per horizon
(fourier_gr1: ``iter_000090000`` at 130/130 episodes both horizons;
franka_panda makovian: ``iter_000090000_static16fps`` at 120; franka_panda
has no ``non_makovian`` directory for dreamgen at all -- an N/A cell, not a
missing pin). ``sources.dreamgen.iter`` (see
:meth:`~kinescore.bench.config.SourceConfig.resolve_iter`) resolves per cell
accordingly, and this plugin never globs ``iter_*``.

Two more traps confirmed against the real HF file listing for ``single_arm``,
both handled explicitly below rather than by assuming a fixed directory
shape:

* **Not every entry under ``dreamgen/`` is a horizon.** ``dense/single_arm/
  input/singleview/dreamgen/`` has view-named siblings of the horizon
  directories (``exterior_1_left``, ``exterior_2_left``, ``global``) next to
  ``makovian``/``non_makovian``. This plugin never lists that directory and
  treats an entry positionally -- it goes straight to
  ``.../dreamgen/{cell.horizon}/`` using the horizon *value already
  validated against* :data:`~kinescore.bench.config.AXIS_VALUES` at config
  load time (see :data:`_KNOWN_HORIZONS`, asserted again here as a
  belt-and-braces check against a :class:`~kinescore.bench.cell.Cell` built
  by hand rather than through :func:`kinescore.bench.matrix.expand`). The
  camera-named siblings are simply never touched.
* **``_``-prefixed directories are not episodes.** ``dense/single_arm/
  output/singleview/dreamgen/`` contains a ``_fps_compare`` directory (a
  frame-rate experiment, not benchmark data). Any path segment encountered
  while globbing that starts with ``_`` is skipped, and the count of
  skipped entries is logged -- silently including it would inject an
  unrelated experiment into the benchmark numbers.
"""
from __future__ import annotations

import glob
import os
import sys
from collections.abc import Iterator
from typing import TYPE_CHECKING

from kinescore.bench.manifest import DiscoveredClip, SourcePlugin
from kinescore.bench.sources.base import ClipSource
from kinescore.core.clip import ViewLayout

if TYPE_CHECKING:
    from kinescore.bench.cell import Cell
    from kinescore.bench.config import BenchConfig

__all__ = ["DreamGenSource"]

#: dreamgen's real multiview packing (measured at 768x432 across 257 files):
#: a 2x2 grid, not the dataset-wide height-stack convention
#: ``cell.view_layout`` would otherwise assign. Unlike ctrlworld, no panel
#: subset is declared here: a single visual sample showed 3 populated
#: quadrants and 1 solid-black one, which is *suggestive* of a droppable
#: panel but is one sample, not a validated rule across the corpus -- all 4
#: panels are kept, unnamed, until that is actually checked. This cell is
#: currently declared N/A in ``configs/benchmark.yaml`` (dreamgen's
#: ``gt_from: input`` finds no ground truth for either robot -- see the
#: module docstring's warning), so this layout is not yet exercised by a
#: real run; it exists so the plugin is correct if/when that gap closes. See
#: ``legacy_docs/DECISIONS.md`` D-G.
_MULTIVIEW_LAYOUT = ViewLayout(n_views=4, packing="grid2x2", n_panels=4)

#: The only horizon labels this plugin will ever treat as a horizon
#: directory. Mirrors ``kinescore.bench.config.AXIS_VALUES["horizon"]`` --
#: duplicated as a local constant (rather than imported) so this module has
#: no import-time dependency on ``bench.config`` beyond the ``TYPE_CHECKING``
#: block, and so the assertion below reads as "this plugin's own contract"
#: rather than reaching into another module's validation.
_KNOWN_HORIZONS = frozenset({"makovian", "non_makovian"})


def _is_skippable(name: str) -> bool:
    """``True`` for a directory entry that is never an episode/task.

    Currently just the ``_``-prefix convention (``_fps_compare`` and any
    future sibling like it); a single predicate so every glob loop in this
    module applies the same rule instead of repeating the check inline.
    """
    return name.startswith("_")


class DreamGenSource(ClipSource):
    """Discovers dreamgen's ``<task>/episode_XXXXXX.mp4`` predictions. See module docstring."""

    GENERATOR = "dreamgen"

    def make_plugin(self, cell: Cell, data_root: str, config: BenchConfig) -> SourcePlugin:
        """Build the zero-arg plugin discovering ``cell``'s dreamgen episodes.

        Raises
        ------
        ValueError
            If ``cell.generator`` is not ``"dreamgen"``, if
            ``sources.dreamgen.iter`` resolves to nothing for this cell's
            (robot, horizon) (several ``iter_*`` directories exist on disk;
            an unresolved pin is a config bug, not a "discover nothing"
            situation), or if ``cell.horizon`` is not one of
            :data:`_KNOWN_HORIZONS` (see the module docstring's first trap --
            this fires loudly instead of quietly treating a camera-named
            directory as a horizon).
        """
        if cell.generator != self.GENERATOR:
            raise ValueError(
                f"DreamGenSource given cell.generator={cell.generator!r}, "
                f"expected {self.GENERATOR!r}")
        if cell.horizon not in _KNOWN_HORIZONS:
            raise ValueError(
                f"DreamGenSource given cell.horizon={cell.horizon!r}, not one of "
                f"{sorted(_KNOWN_HORIZONS)}. dreamgen's input/ tree has view-named "
                f"directories (e.g. exterior_1_left, global) alongside the horizon "
                f"ones for some embodiments -- refusing to treat an unrecognised "
                f"segment as a horizon rather than silently mis-labelling clips.")
        source = config.sources[self.GENERATOR]
        resolved_iter = source.resolve_iter(robot=cell.robot, horizon=cell.horizon)
        if not resolved_iter:
            raise ValueError(
                f"sources.dreamgen.iter has no pin for robot="
                f"{cell.robot!r} horizon={cell.horizon!r} (dreamgen has "
                f"several iter_* directories on disk; a config must name one "
                f"per cell). Got iter={source.iter!r}.")
        fps_hint = config.resolve_fps(generator=self.GENERATOR, robot=cell.robot)
        # Override the dataset-wide height-stack convention with dreamgen's real
        # packing for multiview cells -- see :data:`_MULTIVIEW_LAYOUT`.
        view_layout = _MULTIVIEW_LAYOUT if cell.view == "multiview" else cell.view_layout
        data_root = str(data_root)
        output_iter_dir = os.path.join(
            data_root, "video_gen_physics", cell.cache, cell.embodiment, "output",
            source.resolve_view_dir(robot=cell.robot), self.GENERATOR,
            cell.horizon, resolved_iter)
        input_root = os.path.join(
            data_root, "video_gen_physics", config.baseline_cache, cell.embodiment,
            "input", source.resolve_view_dir(robot=cell.robot), self.GENERATOR,
            cell.horizon)

        def _plugin() -> Iterator[DiscoveredClip]:
            n_skipped_dirs = 0
            for task_dir in sorted(glob.glob(os.path.join(output_iter_dir, "*"))):
                if not os.path.isdir(task_dir):
                    continue
                task = os.path.basename(task_dir)
                if _is_skippable(task):
                    n_skipped_dirs += 1
                    continue
                for pred_path in sorted(glob.glob(os.path.join(task_dir, "episode_*.mp4"))):
                    episode = os.path.splitext(os.path.basename(pred_path))[0]
                    gt_path = os.path.join(input_root, task, f"{episode}.mp4")
                    if not os.path.isfile(gt_path):
                        continue
                    pair_key = f"{cell.family}/{resolved_iter}/{task}/{episode}"
                    yield DiscoveredClip(method=self.GENERATOR, family=cell.family,
                                         episode=f"{task}/{episode}", role="pred",
                                         path=pred_path, pair_key=pair_key,
                                         fps_hint=fps_hint, view_layout=view_layout)
                    yield DiscoveredClip(method=self.GENERATOR, family=cell.family,
                                         episode=f"{task}/{episode}", role="gt",
                                         path=gt_path, pair_key=pair_key,
                                         fps_hint=fps_hint, view_layout=view_layout)
            if n_skipped_dirs:
                print(f"[dreamgen] skipped {n_skipped_dirs} non-episode "
                     f"'_'-prefixed director{'y' if n_skipped_dirs == 1 else 'ies'} "
                     f"under {output_iter_dir!r}", file=sys.stderr)

        return _plugin
