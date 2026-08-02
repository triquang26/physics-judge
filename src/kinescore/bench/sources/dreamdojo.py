"""dreamdojo plugin: THREE coexisting export shapes in one pinned iter_ dir.

Layout (verified against the HF API and, for humanoid, against real
downloaded bytes -- see below)::

    {data_root}/video_gen_physics/{cache}/{embodiment}/output/{view_dir}/
        dreamdojo/{horizon}/iter_XXXXXX/
            0000_pred.mp4  0000_gt.mp4  0000_actions.npy  0000_metrics.json  0000_merged.mp4
            0001_pred.mp4  0001_gt.mp4  ...                                    # "flat" shape
            episode_0000/full_pred.mp4  episode_0000/full_gt.mp4  ...          # "dir" shape, complete
            episode_000200/full_pred.mp4                                      # "dir" shape, pred-only

``{embodiment}`` is :attr:`~kinescore.bench.cell.Cell.embodiment` (derived
from ``cell.robot`` -- see ``bench.cell``'s module docstring): dreamdojo
under ``embodiment=humanoid`` is Fourier GR-1 (``robot=fourier_gr1``);
Airbot MMK2 (``robot=airbot_mmk2``) never appears here -- ``robot_map.yaml``
lists ``ctrlworld`` as airbot_mmk2's only generator, so
``kinescore.bench.matrix`` never builds a cell pairing airbot_mmk2 with this
source in the first place.

dreamdojo has several ``iter_*`` directories on disk (different training
checkpoints); ``sources.dreamdojo.iter`` (see
:meth:`~kinescore.bench.config.SourceConfig.resolve_iter`) pins exactly one
per (robot, horizon), and this plugin never globs ``iter_*`` -- mixing
checkpoints into one number is the bug ``SourceConfig.iter`` (a string or a
per-cell mapping, never a list) exists to prevent.

Three shapes, not two -- and NOT duplicates of each other
------------------------------------------------------------
A first pass over ``humanoid/makovian/iter_000050000`` treated this as one
flat-vs-dir dedupe question and found a single matching pair
(``0000_pred.mp4`` == ``episode_0000/full_pred.mp4`` by size). That was a
coincidence, not the pattern. Comparing LFS ``sha256`` for every
overlapping index via the HF API (no download needed) across the full
118-pair overlap between "flat" and the 4-digit "dir" episodes found only
1/118 identical pred hashes and 1/118 identical gt hashes -- everything else
differs. **Same nominal index, different content.** Downloading a few
``metrics.json`` sidecars confirmed why: the 4-digit dir episodes carry a
``trajectory_id`` field (e.g. ``episode_0000`` -> ``trajectory_id: 200``)
that maps onto a THIRD group -- 6-digit dir episodes named after that same
trajectory id (``episode_000200``) -- which exist ONLY as ``full_pred.mp4``,
with no ``full_gt.mp4``, ``full_merged.mp4`` or ``metrics.json`` sibling
ever, in any cell checked, and no ``input/`` tree for dreamdojo at all to
source a ground truth from elsewhere. So the real shape is:

* **flat** (``NNNN_pred.mp4``/``NNNN_gt.mp4``) -- an independent,
  much larger export batch (658 episodes for the cell above); cross-checking
  its full hash SET against the dir shapes' found only 4/118 accidental
  content matches (not the whole set) -- overwhelming evidence flat is a
  distinct population, not a subset or a dupe of the dir shapes.
* **dir, complete** (``episode_<id>/full_pred.mp4`` + a sibling
  ``full_gt.mp4``) -- usable, kept. ``<id>`` may be 4 or 6 digits depending
  on the export batch (both occur; single_arm's pinned iters only ever
  produce the 6-digit form, WITH a matching ``full_gt.mp4`` every time --
  unlike humanoid's 6-digit group, single_arm's dir episodes are never
  pred-only. Digit count alone is therefore not a reliable proxy for
  "usable"; GT presence is checked directly instead).
* **dir, orphan** (``episode_<id>/full_pred.mp4`` with NO ``full_gt.mp4``
  sibling) -- cannot be scored (no ground truth exists anywhere for it) and
  is dropped, with the drop count logged so ``n`` in a report is never
  silently smaller than what was on disk without an audit trail.

Both usable shapes are kept when both are present (verified NOT duplicates,
above), each stamped with the shape baked into ``episode``/``pair_key``
(``"flat:0000"`` vs ``"dir:0000"``) so a flat and a dir episode that happen
to share a bare numeric index -- confirmed to occur, see above -- can never
collide into one ``pair_key``. Counting both is not double-counting the
same scene; it is reporting two real, differently-sourced populations that
happen to sit in the same ``iter_*`` directory.
"""
from __future__ import annotations

import glob
import os
import re
import sys
from collections.abc import Iterator
from typing import TYPE_CHECKING

from kinescore.bench.manifest import DiscoveredClip, SourcePlugin
from kinescore.bench.sources.base import ClipSource

if TYPE_CHECKING:
    from kinescore.bench.cell import Cell
    from kinescore.bench.config import BenchConfig

__all__ = ["DreamDojoSource"]

_FLAT_PRED_RE = re.compile(r"^(?P<episode>\d+)_pred\.mp4$")
_DIR_RE = re.compile(r"^episode_(?P<episode>\d+)$")
_DIR_PRED_NAME = "full_pred.mp4"
_DIR_GT_NAME = "full_gt.mp4"


class DreamDojoSource(ClipSource):
    """Discovers dreamdojo's flat + dir (complete/orphan) shapes. See module docstring."""

    GENERATOR = "dreamdojo"

    def make_plugin(self, cell: Cell, data_root: str, config: BenchConfig) -> SourcePlugin:
        """Build the zero-arg plugin discovering ``cell``'s dreamdojo episodes.

        Raises
        ------
        ValueError
            If ``cell.generator`` is not ``"dreamdojo"``, or if
            ``sources.dreamdojo.iter`` resolves to nothing for this cell's
            (robot, horizon) -- dreamdojo always has several ``iter_*``
            directories on disk; an unresolved pin would either glob all of
            them or discover nothing, neither of which is a valid run.
        """
        if cell.generator != self.GENERATOR:
            raise ValueError(
                f"DreamDojoSource given cell.generator={cell.generator!r}, "
                f"expected {self.GENERATOR!r}")
        source = config.sources[self.GENERATOR]
        resolved_iter = source.resolve_iter(robot=cell.robot, horizon=cell.horizon)
        if not resolved_iter:
            raise ValueError(
                f"sources.dreamdojo.iter has no pin for robot="
                f"{cell.robot!r} horizon={cell.horizon!r} (dreamdojo "
                f"always has several iter_* directories on disk; a config must "
                f"name one per cell). Got iter={source.iter!r}.")
        fps_hint = config.resolve_fps(generator=self.GENERATOR, robot=cell.robot)
        iter_dir = os.path.join(
            str(data_root), "video_gen_physics", cell.cache, cell.embodiment,
            "output", source.resolve_view_dir(robot=cell.robot), self.GENERATOR,
            cell.horizon, resolved_iter)

        def _plugin() -> Iterator[DiscoveredClip]:
            n_flat = n_dir = n_dir_orphans = 0

            for pred_path in sorted(glob.glob(os.path.join(iter_dir, "*_pred.mp4"))):
                m = _FLAT_PRED_RE.match(os.path.basename(pred_path))
                if not m:
                    continue
                idx = m.group("episode")
                gt_path = os.path.join(iter_dir, f"{idx}_gt.mp4")
                if not os.path.isfile(gt_path):
                    continue
                episode = f"flat:{idx}"
                pair_key = f"{cell.family}/{resolved_iter}/{episode}"
                n_flat += 1
                yield DiscoveredClip(method=self.GENERATOR, family=cell.family,
                                     episode=episode, role="pred", path=pred_path,
                                     pair_key=pair_key, fps_hint=fps_hint,
                                     view_layout=cell.view_layout)
                yield DiscoveredClip(method=self.GENERATOR, family=cell.family,
                                     episode=episode, role="gt", path=gt_path,
                                     pair_key=pair_key, fps_hint=fps_hint,
                                     view_layout=cell.view_layout)

            for ep_dir in sorted(glob.glob(os.path.join(iter_dir, "episode_*"))):
                if not os.path.isdir(ep_dir):
                    continue
                m = _DIR_RE.match(os.path.basename(ep_dir))
                if not m:
                    continue
                idx = m.group("episode")
                pred_path = os.path.join(ep_dir, _DIR_PRED_NAME)
                gt_path = os.path.join(ep_dir, _DIR_GT_NAME)
                if not os.path.isfile(pred_path):
                    continue
                if not os.path.isfile(gt_path):
                    # pred-only export: no ground truth exists for it anywhere
                    # (dreamdojo has no input/ tree to fall back to) -- cannot
                    # be scored, so it is dropped, not silently included.
                    n_dir_orphans += 1
                    continue
                episode = f"dir:{idx}"
                pair_key = f"{cell.family}/{resolved_iter}/{episode}"
                n_dir += 1
                yield DiscoveredClip(method=self.GENERATOR, family=cell.family,
                                     episode=episode, role="pred", path=pred_path,
                                     pair_key=pair_key, fps_hint=fps_hint,
                                     view_layout=cell.view_layout)
                yield DiscoveredClip(method=self.GENERATOR, family=cell.family,
                                     episode=episode, role="gt", path=gt_path,
                                     pair_key=pair_key, fps_hint=fps_hint,
                                     view_layout=cell.view_layout)

            print(f"[dreamdojo] {iter_dir}: {n_flat} flat episode(s), {n_dir} dir "
                 f"episode(s) with gt, {n_dir_orphans} dir episode(s) dropped "
                 f"(pred with no matching full_gt.mp4 -- unscoreable)",
                 file=sys.stderr)

        return _plugin
