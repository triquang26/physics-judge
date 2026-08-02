"""ctrlworld plugin: one multiview grid pair per episode directory.

Layout (verified against the HF API and, for humanoid, every downloaded
episode directory -- see below)::

    {data_root}/video_gen_physics/{cache}/{embodiment}/output/{view_dir}/
        ctrlworld/{horizon}/episode_<task>__<id>/
            full_pred.mp4        # NOT discovered here -- see below
            pred_all_views.mp4   # role="pred"
            gt_all_views.mp4     # role="gt"

``{embodiment}`` is :attr:`~kinescore.bench.cell.Cell.embodiment` (the
on-disk directory, derived from ``cell.robot`` via ``configs/robot_map.yaml``
-- see ``bench.cell``'s module docstring) -- ctrlworld under ``embodiment=
humanoid`` is ALWAYS Airbot MMK2 (``robot=airbot_mmk2``; episode directories
are literally named ``episode_AIRBOT_MMK2_*``/``episode_Airbot_MMK2_*``),
never Fourier GR-1.

ctrlworld only ever has ``view=multiview`` cells in practice (ctrlworld x
singleview is a declared N/A cell in ``configs/benchmark.yaml``), so this
plugin always discovers the packed multiview pair, never ``full_pred.mp4``.

Why ``gt_all_views.mp4`` and not the per-view ``input/`` files
------------------------------------------------------------------
A SEPARATE ``input/`` tree (sibling of ``output/``, used by dreamgen's
``gt_from: input`` -- see :mod:`kinescore.bench.sources.dreamgen`) also
carries this generator's ground truth, in a different shape:
``full_gt.mp4`` (a 3-camera horizontal grid, same 960x192/37-frame shape as
``output/``'s ``gt_all_views.mp4``) plus three untruncated per-view files
``view_0.mp4``/``view_1.mp4``/``view_2.mp4`` (320x192, 39 frames -- 2 frames
longer than ``full_gt.mp4``). Pixel-diffing confirmed the relationship is a
clean HEAD-aligned truncation: ``full_gt.mp4 == concat(view_0, view_1,
view_2)[:len(full_gt)]`` (head-aligned diff 2.1-3.0 vs tail-aligned 6.3-18.7,
head-aligned winning 15/15 spot checks, no mid-clip spike). So if
``gt_all_views.mp4`` were ever missing for a discovered ``pred_all_views.mp4``,
the documented, safe fallback is to read the three per-view files (the
untruncated, more primitive source) and head-truncate to the prediction's
frame count -- NOT to reach for ``input/full_gt.mp4`` directly, since that is
just this same truncation already applied for a length that might not match
THIS cell's prediction.

That fallback is deliberately **not wired in below**: every discovered
``pred_all_views.mp4`` across every embodiment checked (humanoid, single_arm,
bimanual -- 805 episodes total via the HF API) has a co-located
``gt_all_views.mp4``, 100% coverage, zero gaps. Adding an untested fallback
path for a gap that has never been observed would be exactly the kind of
speculative branch this codebase's plugins avoid elsewhere (see
``dreamgen``'s refusal to guess at an unverified ``iter_*`` layout). If a
future data drop introduces a real gap, this docstring is the pointer to the
already-verified formula for closing it.

Real pixel packing: width-stacked, wrist panel dropped
-------------------------------------------------------
``cell.view_layout`` (from ``bench.matrix``'s dataset-wide ``multiview`` ->
height-stack convention) does NOT match what ctrlworld actually writes:
measured at 960x192 across 771 files, a 3-panel stack on the **width** axis
(``exterior_1 | exterior_2 | wrist``), not height -- ``192 % 3 == 0`` too, so
the height-stack default would silently slice three meaningless 960x64 bands
with no error (now a hard error instead, see
:meth:`~kinescore.core.clip.ViewLayout._panel_size`). This plugin overrides
``cell.view_layout`` with :data:`_VIEW_LAYOUT` -- 2 exposed views (the two
exterior panels, columns ``0:320`` and ``320:640``), the wrist panel dropped
entirely rather than exposed and ignored downstream. See
``legacy_docs/DECISIONS.md`` D-G.

fps is not trustworthy as a single per-family number
----------------------------------------------------
ctrlworld has no ``fps_expected`` table entry (see
``configs/benchmark.yaml``) and must not gain one: probing showed fps is NOT
constant even within one ctrlworld cell (two episodes measured at 30 fps
against 394 at 5 fps in one probed sample). A per-generator fps TABLE value
would either reject the minority as "wrong" or silently paper over a real
mixed-rate cell; ``kinescore.video.probe.resolve_timebase`` always trusts the
per-clip probed rate for ctrlworld instead, and that probe can only ever
DISAGREE with a table (raising), never be overridden by one -- which is
exactly why no table entry is added here.
"""
from __future__ import annotations

import glob
import os
from collections.abc import Iterator
from typing import TYPE_CHECKING

from kinescore.bench.manifest import DiscoveredClip, SourcePlugin
from kinescore.bench.sources.base import ClipSource
from kinescore.core.clip import ViewLayout

if TYPE_CHECKING:
    from kinescore.bench.cell import Cell
    from kinescore.bench.config import BenchConfig

__all__ = ["CtrlWorldSource"]

_PRED_NAME = "pred_all_views.mp4"
_GT_NAME = "gt_all_views.mp4"

#: ctrlworld's real packing -- see the module docstring's "Real pixel
#: packing" section. 3 physical panels stacked on WIDTH; only the two
#: exterior panels (0, 1) are exposed as views, the wrist panel (2) dropped.
_VIEW_LAYOUT = ViewLayout(n_views=2, order=("exterior_1", "exterior_2"),
                          packing="width", n_panels=3, panels=(0, 1))


class CtrlWorldSource(ClipSource):
    """Discovers ctrlworld's ``{episode}/{pred,gt}_all_views.mp4`` pairs. See module docstring."""

    GENERATOR = "ctrlworld"

    def make_plugin(self, cell: Cell, data_root: str, config: BenchConfig) -> SourcePlugin:
        """Build the zero-arg plugin discovering ``cell``'s ctrlworld episodes.

        Raises
        ------
        ValueError
            If ``cell.generator`` is not ``"ctrlworld"`` -- this source only
            knows how to discover the generator it is named after; a
            mismatch here means the caller wired the wrong source to the
            wrong cell.
        """
        if cell.generator != self.GENERATOR:
            raise ValueError(
                f"CtrlWorldSource given cell.generator={cell.generator!r}, "
                f"expected {self.GENERATOR!r}")
        source = config.sources[self.GENERATOR]
        fps_hint = config.resolve_fps(generator=self.GENERATOR, robot=cell.robot)
        episodes_root = os.path.join(
            str(data_root), "video_gen_physics", cell.cache, cell.embodiment,
            "output", source.resolve_view_dir(robot=cell.robot), self.GENERATOR,
            cell.horizon)

        def _plugin() -> Iterator[DiscoveredClip]:
            for episode_dir in sorted(glob.glob(os.path.join(episodes_root, "episode_*"))):
                if not os.path.isdir(episode_dir):
                    continue
                episode = os.path.basename(episode_dir)
                pred_path = os.path.join(episode_dir, _PRED_NAME)
                gt_path = os.path.join(episode_dir, _GT_NAME)
                if not (os.path.isfile(pred_path) and os.path.isfile(gt_path)):
                    continue
                pair_key = f"{cell.family}/{episode}"
                yield DiscoveredClip(method=self.GENERATOR, family=cell.family,
                                     episode=episode, role="pred", path=pred_path,
                                     pair_key=pair_key, fps_hint=fps_hint,
                                     view_layout=_VIEW_LAYOUT)
                yield DiscoveredClip(method=self.GENERATOR, family=cell.family,
                                     episode=episode, role="gt", path=gt_path,
                                     pair_key=pair_key, fps_hint=fps_hint,
                                     view_layout=_VIEW_LAYOUT)

        return _plugin
