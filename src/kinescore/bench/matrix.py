"""Expand a validated :class:`~kinescore.bench.config.BenchConfig` into cells.

This is the ONLY module in the package that knows the five-axis grid exists.
``bench.sources`` plugins, ``cli.cmd_bench`` and (later) the scorer/aggregator
all receive a :class:`~kinescore.bench.cell.Cell` and never touch
``axes``/``na_cells`` directly -- so a change to how the grid is built (a
sixth axis, a different N/A rule) touches this one file instead of every
consumer re-deriving the same Cartesian product and inevitably disagreeing
about the N/A cases at the edges.

A cell's ``view_layout`` follows a fixed, dataset-wide convention (see
:data:`_VIEW_LAYOUTS`): ``singleview``/``single_view`` is one camera,
``multiview`` is the three-camera exterior/exterior/wrist stack already used
as the worked example in ``legacy_docs/SCHEMA.md`` (``"3x49:exterior_1+exterior_2+
wrist"``). Every generator plugin must use the SAME mapping -- it is defined
once, here, and forwarded onto every :class:`~kinescore.bench.manifest.DiscoveredClip`
a plugin yields, rather than each plugin guessing its own camera count.

Robot resolution (the (embodiment, generator) -> robot table)
-----------------------------------------------------------------
:class:`~kinescore.bench.config.BenchConfig` validates ``axes.robot`` against
the live robot registry but does not know which on-disk ``embodiment``
directory a robot's clips live under -- that is
``configs/robot_map.yaml``/:mod:`kinescore.bench.robot_map`'s job, kept
deliberately separate (see that module's docstring). This module is where
the two meet: every :func:`expand`/:func:`na_cells` call takes a
:class:`~kinescore.bench.robot_map.RobotMap` alongside the config, and uses
:meth:`~kinescore.bench.robot_map.RobotMap.embodiment_of` to fill in each
cell's derived ``embodiment`` and :meth:`~kinescore.bench.robot_map.RobotMap.generators_of`
to N/A any (robot, generator) combination the table does not claim (e.g.
robot=airbot_mmk2, generator=dreamgen -- airbot_mmk2 only ever appears via
ctrlworld) -- automatically, without every ``configs/benchmark.yaml`` having
to spell out every such combination by hand in ``na_cells``.

Iter resolution and hard-early validation
------------------------------------------
Each cell also resolves ``sources.<generator>.iter`` for its own
(robot, horizon) via :meth:`~kinescore.bench.config.SourceConfig.resolve_iter`
(see that method's docstring for the three shapes ``iter`` may take) into
:attr:`~kinescore.bench.cell.Cell.iter` -- so a caller printing the cell table
(or a source plugin resolving its own iter independently, the same way)
always agrees on which iteration a cell means.

:func:`expand` takes an **optional** ``data_root``. Left out (the default,
and what every existing caller -- ``bench.sources.*``'s tests,
``cli.cmd_data``'s ``allow_patterns`` path, which must work before any data
has been pulled -- already does), it stays a pure function of ``config``
(plus ``robot_map``, itself pure once loaded): no filesystem access, so it
cannot fail because data has not been downloaded yet. Passed a
``data_root``, it additionally hard-validates, for every non-N/A cell whose
generator pins an iter, that the resolved iter directory actually exists on
disk -- naming the iters that DO exist there in the error otherwise. This is
the fix for the exact failure the plan calls out: pinning a single global
``iter_000113000`` that exists for neither robot produced zero rows
silently; resolving per-cell AND validating at expand time turns that into a
loud, actionable :class:`~kinescore.bench.config.ConfigError` before any
plugin ever globs a directory.
"""
from __future__ import annotations

import os
from itertools import product

from kinescore.bench.cell import Cell, raw_tree_segments
from kinescore.bench.config import BenchConfig, ConfigError
from kinescore.bench.robot_map import RobotMap
from kinescore.core.clip import ViewLayout

__all__ = ["Cell", "expand", "na_cells", "allow_patterns", "cell_row",
          "parse_only_filters", "matches_only"]

#: ``view`` axis value -> camera packing. See the module docstring for why
#: this exact mapping (and not e.g. a per-generator one) is the shared
#: convention. ``single_view`` (the alternate on-disk spelling; see
#: ``kinescore.bench.config.VIEW_DIR_VALUES``) is not an axis value -- only
#: ``singleview``/``multiview`` ever reach this mapping, keyed by axis value,
#: not by the literal ``view_dir`` directory name.
_VIEW_LAYOUTS: dict[str, ViewLayout] = {
    "singleview": ViewLayout(n_views=1),
    "multiview": ViewLayout(n_views=3, order=("exterior_1", "exterior_2", "wrist")),
}


def _is_na(config: BenchConfig, robot_map: RobotMap, *, robot: str, view: str,
          horizon: str, cache: str, generator: str) -> bool:
    if generator not in robot_map.generators_of(robot):
        return True  # this robot's clips are never produced by this generator
    return any(rule.matches(robot=robot, view=view, horizon=horizon,
                            cache=cache, generator=generator)
              for rule in config.na_cells)


def _make_cell(config: BenchConfig, robot_map: RobotMap, *, robot: str, view: str,
              horizon: str, cache: str, generator: str) -> Cell:
    source = config.sources[generator]
    resolved_iter = source.resolve_iter(robot=robot, horizon=horizon)
    embodiment = robot_map.embodiment_of(robot)
    return Cell(cache=cache, robot=robot, view=view, generator=generator,
               horizon=horizon, embodiment=embodiment,
               view_layout=_VIEW_LAYOUTS[view], iter=resolved_iter)


def _all_combinations(config: BenchConfig):
    axes = config.axes
    yield from product(axes.robot, axes.view, axes.horizon, axes.cache,
                       axes.generator)


def _validate_iter_on_disk(config: BenchConfig, cell: Cell,
                           data_root: str | os.PathLike) -> None:
    """Hard-fail ``cell`` if its resolved iter is unset or missing on disk.

    Two distinct failures, both named precisely rather than left to surface
    later as "0 rows discovered":

    * The generator pins an iter (``source.iter is not None``) but
      resolution came back empty for this (robot, horizon) -- either
      the nested mapping never covered this cell, or it did and the
      directory it names simply is not there. Either way this raises before
      any plugin runs.
    """
    source = config.sources[cell.generator]
    if source.iter is None:
        return  # generator has no iter_* level at all (ctrlworld) -- nothing to check
    horizon_dir = os.path.join(str(data_root), *raw_tree_segments(
        cache=cell.cache, embodiment=cell.embodiment,
        view_dir=source.resolve_view_dir(robot=cell.robot),
        generator=cell.generator, horizon=cell.horizon))
    if cell.iter is None:
        existing = _list_iter_dirs(horizon_dir)
        raise ConfigError(
            f"{cell.cell_id}: sources.{cell.generator}.iter has no pin for "
            f"robot={cell.robot!r} horizon={cell.horizon!r}. "
            f"Iters that exist on disk at {horizon_dir!r}: {existing!r}. "
            f"Pin one explicitly in the config, or add this cell to "
            f"na_cells if it should not be scored.")
    iter_dir = os.path.join(horizon_dir, cell.iter)
    if not os.path.isdir(iter_dir):
        existing = _list_iter_dirs(horizon_dir)
        raise ConfigError(
            f"{cell.cell_id}: sources.{cell.generator}.iter={cell.iter!r} "
            f"does not exist at {iter_dir!r}. Iters that exist on disk "
            f"there: {existing!r}.")


def _list_iter_dirs(horizon_dir: str) -> list[str]:
    """Sibling directory names under ``horizon_dir``, for an error message.

    Never raises: an absent ``horizon_dir`` itself is reported as an empty
    list (with the caller's message already naming the path), not a second
    exception masking the first.
    """
    if not os.path.isdir(horizon_dir):
        return []
    return sorted(name for name in os.listdir(horizon_dir)
                 if os.path.isdir(os.path.join(horizon_dir, name)))


def expand(config: BenchConfig, robot_map: RobotMap, *,
          data_root: str | os.PathLike | None = None) -> list[Cell]:
    """The Cartesian product of ``config.axes``, minus every N/A cell.

    This is what every downstream consumer (manifest building, scoring,
    reporting) iterates -- an N/A cell never reaches them as a cell with zero
    discovered clips (which would be indistinguishable from a real "no data
    found" bug); it is simply absent from this list. Use :func:`na_cells` to
    report the excluded cells explicitly.

    A cell is N/A either because ``config.na_cells`` says so, OR because
    ``robot_map`` does not claim this (robot, generator) pair at all (see the
    module docstring) -- both are checked, and a caller cannot tell which
    fired from this function alone (use :func:`na_cells` for that).

    Parameters
    ----------
    data_root:
        If given, additionally hard-validates every non-N/A cell's resolved
        iter directory exists on disk (see the module docstring's "Iter
        resolution and hard-early validation" section) -- raising
        :class:`~kinescore.bench.config.ConfigError` immediately, naming the
        iters that DO exist, rather than letting a typo'd pin silently
        expand into a cell that later discovers zero clips. Left as
        ``None`` (the default), this stays a pure function of ``config``
        with no filesystem access -- required for callers that must work
        before data exists (``kinescore data pull``'s dry run) and for tests
        that never touch a real data root.
    """
    cells = []
    for robot, view, horizon, cache, generator in _all_combinations(config):
        if _is_na(config, robot_map, robot=robot, view=view, horizon=horizon,
                 cache=cache, generator=generator):
            continue
        cell = _make_cell(config, robot_map, robot=robot, view=view,
                          horizon=horizon, cache=cache, generator=generator)
        if data_root is not None:
            _validate_iter_on_disk(config, cell, data_root)
        cells.append(cell)
    return cells


def na_cells(config: BenchConfig, robot_map: RobotMap) -> list[Cell]:
    """The cells excluded from :func:`expand` -- for reporting them as N/A.

    Same resolution logic as :func:`expand` (embodiment, view layout) so a
    report can print an N/A row with the same shape as a real one, just with
    no data behind it.
    """
    cells = []
    for robot, view, horizon, cache, generator in _all_combinations(config):
        if _is_na(config, robot_map, robot=robot, view=view, horizon=horizon,
                 cache=cache, generator=generator):
            cells.append(_make_cell(config, robot_map, robot=robot, view=view,
                                    horizon=horizon, cache=cache, generator=generator))
    return cells


def allow_patterns(config: BenchConfig, robot_map: RobotMap) -> list[str]:
    """HF ``snapshot_download(allow_patterns=...)`` globs derived from the matrix.

    Exactly the cache/embodiment/view_dir/generator/horizon combinations that
    :func:`expand` actually resolves to are included -- so a config that asks
    for only ``cache: [dense]`` can never produce a pattern mentioning
    ``dicache``/``fastercache``/etc, because no cell in ``expand(config,
    robot_map)`` has any cache value other than ``dense``. This is what stops
    a `data pull` from downloading the other eight (hundreds of GB) cache
    methods when only one was requested.

    dreamgen is special-cased: its ground truth lives under
    ``config.baseline_cache``'s ``input/`` tree (see
    ``kinescore.bench.config.SourceConfig.gt_from``), which may not be
    ``cell.cache`` if a run asks for a non-baseline cache -- that pattern is
    added in addition to the prediction tree's, not instead of it.
    """
    patterns: set[str] = set()
    for cell in expand(config, robot_map):
        source = config.sources[cell.generator]
        view_dir = source.resolve_view_dir(robot=cell.robot)
        patterns.add("/".join(raw_tree_segments(
            cache=cell.cache, embodiment=cell.embodiment, view_dir=view_dir,
            generator=cell.generator, horizon=cell.horizon)) + "/**")
        if source.gt_from == "input":
            patterns.add("/".join(raw_tree_segments(
                cache=config.baseline_cache, embodiment=cell.embodiment,
                view_dir=view_dir, generator=cell.generator,
                horizon=cell.horizon, stage="input")) + "/**")
    return sorted(patterns)


def cell_row(cell: Cell, config: BenchConfig, *, status: str,
            n_rows: int | None = None) -> dict:
    """One JSON-able row for ``kinescore bench run --cells-out``'s cell table.

    Used for both real and N/A cells (``status="pending"``/``"na"``, or
    ``"scored"`` once a manifest has been built for it) -- callers only vary
    ``status``/``n_rows``, the rest of the row is read straight off ``cell``
    and the matching ``config.robots`` entry.
    """
    robot_cfg = config.robots.get(cell.robot)
    row = {
        "robot": cell.robot, "embodiment": cell.embodiment, "view": cell.view,
        "horizon": cell.horizon, "cache": cell.cache, "generator": cell.generator,
        "robot_spec": robot_cfg.spec if robot_cfg else None,
        "reader": robot_cfg.reader if robot_cfg else None,
        "source": cell.generator, "view_layout": cell.view_layout.key,
        "family": cell.family, "cell_id": cell.cell_id, "status": status,
    }
    if n_rows is not None:
        row["n_rows"] = n_rows
    return row


def parse_only_filters(only: list[str] | None,
                       axis_values: dict) -> list[tuple[str, str]]:
    """Parse ``kinescore bench run``'s repeatable ``--only AXIS=VALUE`` into filters.

    Validates both the axis name and the value against ``axis_values``
    (``kinescore.bench.config.AXIS_VALUES``) so a typo'd ``--only`` fails
    fast with the valid options listed, rather than silently matching zero
    cells.

    Raises
    ------
    ValueError
        If an entry is not ``AXIS=VALUE``, or names an unknown axis/value.
    """
    if not only:
        return []
    filters = []
    for entry in only:
        axis, sep, value = entry.partition("=")
        if not sep:
            raise ValueError(f"--only {entry!r} must be AXIS=VALUE")
        if axis not in axis_values:
            raise ValueError(
                f"--only {entry!r}: unknown axis {axis!r}; valid axes are "
                f"{sorted(axis_values)}")
        if value not in axis_values[axis]:
            raise ValueError(
                f"--only {entry!r}: unknown {axis} value {value!r}; valid "
                f"values are {sorted(axis_values[axis])}")
        filters.append((axis, value))
    return filters


def matches_only(cell: Cell, filters: list[tuple[str, str]]) -> bool:
    """``True`` iff ``cell`` matches every ``(axis, value)`` from :func:`parse_only_filters`."""
    return all(getattr(cell, axis) == value for axis, value in filters)
