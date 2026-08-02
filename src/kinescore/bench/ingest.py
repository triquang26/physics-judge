"""``Ingestor``: materialise :class:`~kinescore.bench.layout.RawHFLayout` into
:class:`~kinescore.bench.layout.CanonicalLayout`.

Symlinks by default -- never copy
------------------------------------
The dataset is ~13 GB; copying it into a second canonical tree would double
that for no benefit (the raw bytes never change once downloaded). Every
episode file is a symlink into the raw tree by default. ``--copy`` (wired
via ``kinescore data ingest``) is the fallback for a filesystem that cannot
hold a symlink (some network mounts, some container overlay setups) --
:meth:`Ingestor.run` takes ``copy: bool = False`` and switches
``os.symlink`` for :func:`shutil.copy2` when set, with everything else
(what gets materialised, the ``cell_card.json`` written) identical either
way.

``cell_card.json`` schema
----------------------------
One per cell directory (see :data:`kinescore.bench.layout.CELL_CARD_NAME`),
recording exactly what :func:`Ingestor.run`'s docstring promises: ``robot``,
``generator``, ``view``, ``horizon``, ``cache``, ``embodiment``, the pinned
``iter`` (``null`` for a generator with no ``iter_*`` level), episode counts
(``n_episodes_actual`` vs ``n_episodes_declared`` -- see
:func:`_declared_episode_count` for why the declared number is read from the
source's own ``info.json``/``all_summary.json`` and NOT trusted, only
recorded, when it disagrees), clip format (``width``/``height``/``fps``/
``n_views``), the pred/gt filenames used, and ``source_path`` -- the
ABSOLUTE raw directory this cell was materialised from, so a caller can tell
a moved/re-pulled ``raw/`` tree apart from a stale canonical one.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field

from kinescore.bench.layout import (
    GT_NAME,
    PRED_NAME,
    CanonicalLayout,
    RawHFLayout,
    cell_card_path,
)
from kinescore.video.probe import ffprobe

__all__ = ["IngestReport", "CellReport", "Ingestor"]


@dataclass
class CellReport:
    cell_id: str
    n_episodes_actual: int
    n_episodes_declared: int | None
    n_skipped_missing_gt: int
    cell_card: str


@dataclass
class IngestReport:
    cells: list[CellReport] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def n_episodes(self) -> int:
        return sum(c.n_episodes_actual for c in self.cells)


def _declared_episode_count(raw_cell_dir: str) -> int | None:
    """Read a template episode count from the source's own metadata, if any.

    Checked two ways: ``info.json``'s ``total_episodes`` (LeRobot-v2 style)
    and ``all_summary.json``'s ``num_episodes`` (the real key, confirmed
    against ``dense/humanoid/output/multiview/ctrlworld/makovian/
    all_summary.json`` on disk -- ``{"mean_psnr": ..., "num_episodes": 98}``).
    These values are TEMPLATE values copied from the generation
    run config and routinely disagree with what actually landed on disk
    (see ``kinescore.bench.config._parse_fps_expected``'s docstring and
    ``cli.cmd_data``'s module docstring for the same defect class on fps/
    file-count metadata) -- this function only RECORDS the declared number
    for ``cell_card.json``'s ``n_episodes_declared``; it is never compared
    against, trusted, or used to decide what to ingest.
    """
    for name, keys in (("info.json", ("total_episodes",)),
                       ("all_summary.json", ("num_episodes", "n_episodes", "total_episodes"))):
        # info.json is a sibling of the horizon dir's parent (LeRobot meta/
        # convention) in some exports; all_summary.json is a sibling of the
        # episode dirs, i.e. directly inside the horizon dir. Both are
        # checked at raw_cell_dir's own level and one level up, since the
        # exact placement is not uniform across generators.
        for base in (raw_cell_dir, os.path.dirname(raw_cell_dir)):
            path = os.path.join(base, name)
            if not os.path.isfile(path):
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            for key in keys:
                if isinstance(data.get(key), int):
                    return data[key]
    return None


class Ingestor:
    """Walks ``raw``, materialises ``canonical`` via symlinks (or copies).

    Parameters
    ----------
    raw:
        The source layout to read from.
    canonical:
        The target layout to write into.
    """

    def __init__(self, raw: RawHFLayout, canonical: CanonicalLayout) -> None:
        self.raw = raw
        self.canonical = canonical

    def run(self, *, copy: bool = False) -> IngestReport:
        """Materialise every cell :attr:`raw` has, into :attr:`canonical`.

        For each cell: creates ``<canonical_cell_dir>/episode_XXXX/`` per
        discovered episode (see
        :meth:`~kinescore.bench.layout.RawHFLayout.episodes`), symlinking (or
        copying, if ``copy=True``) its prediction to ``pred.mp4`` and its
        ground truth (if any) to ``gt.mp4``, then writes one
        ``cell_card.json`` per cell (see module docstring for the schema).
        Probes exactly ONE episode per cell (the first) via
        :func:`kinescore.video.probe.ffprobe` for the card's
        width/height/fps/codec -- not every episode -- since the per-cell
        format contract is meant to be uniform (``configs/data_spec.yaml``);
        ``kinescore data verify`` is what checks every clip individually.
        """
        report = IngestReport()
        for cell in self.raw.cells():
            raw_cell_dir = self.raw.cell_dir(cell)
            episodes = list(self.raw.episodes(cell))
            gspec = self.raw.data_spec.generators[cell.generator]
            n_skipped_missing_gt = 0

            cell_dir = self.canonical.cell_dir(cell)
            os.makedirs(cell_dir, exist_ok=True)

            probed: dict | None = None
            for ep in episodes:
                episode_dir = self.canonical.episode_dir(cell, ep.episode)
                os.makedirs(episode_dir, exist_ok=True)
                self._place(ep.pred_path, os.path.join(episode_dir, PRED_NAME), copy=copy)
                if ep.gt_path is not None:
                    self._place(ep.gt_path, os.path.join(episode_dir, GT_NAME), copy=copy)
                elif gspec.has_ground_truth:
                    n_skipped_missing_gt += 1
                if probed is None:
                    try:
                        probed = ffprobe(ep.pred_path)
                    except Exception:  # noqa: BLE001 -- a card is still worth writing without it
                        probed = {}

            card = {
                "cache": cell.cache, "robot": cell.robot, "generator": cell.generator,
                "view": cell.view, "horizon": cell.horizon, "embodiment": cell.embodiment,
                "iter": cell.iter,
                "n_episodes_actual": len(episodes),
                "n_episodes_declared": _declared_episode_count(raw_cell_dir),
                # ffprobe (kinescore.video.probe) returns "w"/"h", not
                # "width"/"height" -- falls back to the declared data_spec
                # value when probing failed (unreadable file) or was skipped
                # (cell had zero episodes).
                "width": probed.get("w") if probed else gspec.width,
                "height": probed.get("h") if probed else gspec.height,
                "fps": probed.get("fps") if probed else gspec.resolve_fps(robot=cell.robot),
                "n_views": gspec.n_views,
                "pred_filename": PRED_NAME, "gt_filename": GT_NAME if gspec.has_ground_truth else None,
                "source_path": os.path.abspath(raw_cell_dir),
            }
            card_path = cell_card_path(cell_dir)
            with open(card_path, "w") as f:
                json.dump(card, f, indent=2, sort_keys=True)

            report.cells.append(CellReport(
                cell_id=cell.cell_id, n_episodes_actual=len(episodes),
                n_episodes_declared=card["n_episodes_declared"],
                n_skipped_missing_gt=n_skipped_missing_gt, cell_card=card_path))

        report.unresolved = self.raw.validate()
        return report

    @staticmethod
    def _place(src: str, dst: str, *, copy: bool) -> None:
        if os.path.lexists(dst):
            os.remove(dst)
        if copy:
            shutil.copy2(src, dst)
        else:
            os.symlink(os.path.abspath(src), dst)
