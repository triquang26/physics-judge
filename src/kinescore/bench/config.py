"""Load and validate ``benchmark.yaml`` -- the reproducibility contract.

A benchmark run is described by exactly one YAML file (see
``configs/benchmark.yaml`` for the concrete first run). This module is the
only place that parses it, and it validates **hard and early**: an unknown
axis value, a missing per-generator source entry, a malformed fps table entry
or a config that references an unset environment variable all raise here,
before ``bench.matrix`` ever expands a cell or a source plugin ever globs a
directory. The alternative -- discovering a typo'd axis value three hours
into a scoring run because the offending cell silently produced zero rows --
is exactly the failure mode this module exists to prevent.

Everything is a frozen dataclass, never a bare dict passed around: a
``BenchConfig`` cannot be mutated after ``load_config`` returns it, so a
downstream module (``bench.matrix``, a ``bench.sources`` plugin) cannot
accidentally drift the config it was handed away from what was written to
disk and hashed into the run's provenance.

Each dataclass validates its own shape via its own ``from_dict`` classmethod,
right next to the fields it constrains -- ``RobotConfig.from_dict``,
``SourceConfig.from_dict``, ``NaCellRule.from_dict``, ``AxesConfig.from_dict``,
``BenchConfig.from_dict``. This exists because the rule and the field used to
live hundreds of lines apart (five dataclasses up top, a parallel set of
``_parse_*`` free functions validating them far below), which is exactly
where three real defects came from: ``from_dict`` silently accepting a
misspelled top-level key, ``robots.<robot>.spec`` accepting a value that
matched no registered robot, and a config naming a reader checkpoint that
does not exist on disk. A collection field (``robots``, ``sources``,
``na_cells``) still has a small module-level ``_parse_*`` orchestrator --
looping over entries and checking the collection against ``axes`` is a
cross-object concern no single dataclass owns -- but per-entry validation is
always the dataclass's own ``from_dict``.

Robot, not embodiment, is the primary axis
-------------------------------------------
``axes.robot`` replaces the old ``axes.embodiment``: see
``kinescore.bench.cell``'s module docstring for why one embodiment directory
(``humanoid``) covers two different physical robots depending on generator,
which an embodiment-keyed axis cannot express. A robot value here is always a
:func:`kinescore.robots.available_robots` registry key -- never a free
string -- so a typo'd or not-yet-implemented robot name (the ``spec: aloha``
bug this module used to have) fails validation here instead of silently
scoring nothing. Mapping a robot to the embodiment directory its clips
actually live under is deliberately NOT this module's job -- that is
``configs/robot_map.yaml`` / :mod:`kinescore.bench.robot_map`, loaded
separately by :mod:`kinescore.bench.matrix` -- so a ``BenchConfig`` stays
parseable and testable with zero filesystem access beyond the YAML file
itself.

Environment variables
----------------------
Any string value in the YAML may contain ``${VAR_NAME}`` and is expanded
against the process environment before validation, via
:func:`kinescore.bench.env_expand.expand_env` (split into its own module --
see that module's docstring -- because environment-variable expansion has
nothing to do with the ``BenchConfig`` schema itself).
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kinescore.bench.env_expand import expand_env

__all__ = [
    "ConfigError", "AXIS_VALUES", "VIEW_DIR_VALUES", "READER_STATUS_VALUES",
    "RATE_POLICY_RE", "IterMap", "RobotConfig", "SourceConfig", "NaCellRule",
    "AxesConfig", "BenchConfig", "load_config", "from_dict",
]


class ConfigError(ValueError):
    """A ``benchmark.yaml`` failed validation.

    Deliberately a plain ``ValueError`` subclass (not a new exception
    hierarchy) so callers that already catch ``ValueError`` around config
    loading keep working; the subclass exists so a caller that wants to
    distinguish "this config is invalid" from "some other ValueError
    happened downstream" can do so precisely.
    """


def _robot_axis_values() -> frozenset[str]:
    """Valid ``axes.robot`` values: exactly the robot registry's keys.

    A local import (not a module-level one) so importing ``bench.config``
    never has a hard import-order dependency on ``kinescore.robots`` --
    matches this module's own no-filesystem-access promise (see module
    docstring) and ``kinescore.robots``'s own "importing this module must
    never require pytorch_kinematics" contract, which this call relies on
    staying true.
    """
    from kinescore.robots import available_robots
    return frozenset(available_robots())


#: Canonical values for each of the five matrix axes. An axis value outside
#: this set is always a config error -- these are the only cells the dataset
#: (and every downstream reader/robot spec) is defined for. ``robot`` is
#: intentionally NOT a frozen module-level literal like the other four: it is
#: computed once, at import time, from the live robot registry (see
#: :func:`_robot_axis_values`) so a robot only becomes a valid axis value once
#: ``kinescore.robots`` actually knows how to build it -- adding a robot here
#: without a working spec/reader is impossible by construction.
AXIS_VALUES: dict[str, frozenset[str]] = {
    "robot": _robot_axis_values(),
    "view": frozenset({"multiview", "singleview"}),
    "horizon": frozenset({"makovian", "non_makovian"}),
    "cache": frozenset({
        "dense", "dicache", "fastercache", "itm", "pisa", "radial",
        "sito", "svg1", "worldcache",
    }),
    "generator": frozenset({"ctrlworld", "dreamgen", "dreamdojo"}),
}

#: Literal on-disk directory names a ``sources.<generator>.view_dir`` may
#: name. Both spellings of "single view" exist as DIFFERENT directories with
#: different file sets (see the module docstring of
#: ``kinescore.bench.sources`` for the counts) -- a config must pin one
#: explicitly, never glob across them, so this set intentionally has THREE
#: members even though the ``view`` axis itself has only two.
VIEW_DIR_VALUES: frozenset[str] = frozenset({"multiview", "singleview", "single_view"})

#: ``rate_policy`` is either one of two fixed words or ``resample:<hz>`` with
#: a positive number.
RATE_POLICY_RE = re.compile(r"^(paired|rate_free)$|^resample:(?P<hz>[0-9]+(\.[0-9]+)?)$")

#: Every key ``BenchConfig.from_dict`` will read from the config root. A key
#: present in the raw YAML but absent here is always a typo -- see
#: :meth:`BenchConfig.from_dict`'s "reject unknown top-level keys" check,
#: which is what makes e.g. ``rate_polciy: paired`` a hard load-time error
#: instead of a silently ignored key whose real value (``rate_policy``)
#: quietly falls back to its default.
_TOP_LEVEL_KEYS = frozenset({
    "run_id", "seed", "axes", "na_cells", "robots", "sources",
    "fps_expected", "rate_policy", "suite", "baseline_cache", "caps",
})


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


#: ``robots.<robot>.reader_status`` -- explicit, schema-level marking of a
#: reader checkpoint's acceptance state. Added after a config once pointed
#: ``franka_panda``/``aloha_bimanual`` at checkpoint filenames that did not
#: exist on disk (``single_arm.pt``, ``bimanual.pt``) with nothing to stop it
#: from loading and shipping numbers whose reader was never actually there --
#: a phantom path is a load-time crash at best and a silent wrong-reader
#: substitution at worst, neither of which a comment above the YAML key can
#: prevent (a comment is not read by ``load_reader``). ``tests/test_bench_config.py``
#: (and the ``robots.*.reader`` resolvability test alongside it) enforce that
#: every ``RobotConfig`` here is honest about which of these three states it
#: is in -- see :meth:`RobotConfig.from_dict`.
#:
#: * ``"accepted"`` (default) -- ``reader`` names a checkpoint file that
#:   exists and passed its own acceptance gate; used at face value.
#: * ``"failing_gate"`` -- ``reader`` names a checkpoint that exists and
#:   loads, but whose own acceptance-gate run failed (see ``reader_note`` for
#:   the numbers). Kept in the config as an explicit, visibly-flagged
#:   control/comparison point -- never silently presented as if it had
#:   passed.
#: * ``"untrained"`` -- no checkpoint has been trained for this robot yet.
#:   ``reader`` MUST be ``None`` for this status (see
#:   :meth:`RobotConfig.from_dict`) -- naming a filename that does not exist
#:   on disk is exactly the phantom path this field exists to make
#:   impossible.
READER_STATUS_VALUES = frozenset({"accepted", "failing_gate", "untrained"})

#: Keys :meth:`RobotConfig.from_dict` will read from one ``robots.<robot>``
#: entry. A key present in the raw YAML but absent here is a typo, rejected
#: the same way :meth:`BenchConfig.from_dict`'s top-level key check rejects
#: one (see that method's docstring) -- ``reader_status``/``reader_note`` are
#: new enough that a typo'd ``reader_stauts`` silently keeping the
#: ``"accepted"`` default would be exactly the bug this whole mechanism
#: exists to close.
_ROBOT_ENTRY_KEYS = frozenset({"spec", "reader", "assets", "reader_status", "reader_note"})


@dataclass(frozen=True)
class RobotConfig:
    """One robot's reader checkpoint filename, its acceptance state, and asset dir.

    ``spec`` is the :func:`kinescore.robots.available_robots` registry key
    used to build this robot's :class:`~kinescore.core.robot.RobotSpec` (via
    ``kinescore.robots.get_robot(spec)``). It is validated in
    :meth:`from_dict` against the SAME live registry ``axes.robot`` itself is
    validated against (see :data:`AXIS_VALUES`'s ``"robot"`` entry) --
    previously this field accepted any string (``spec: aloha``, a robot
    package that has never existed, passed silently), which is the bug this
    validation exists to close. Usually equal to the ``robots.<key>``
    mapping key itself, but kept as its own field rather than collapsed into
    the key so a future indirection (two run-config entries sharing one
    robot spec under different names) stays possible without a schema change.

    ``reader`` is ``None`` exactly when ``reader_status == "untrained"`` --
    see :data:`READER_STATUS_VALUES` -- never a filename that happens not to
    exist on disk. ``reader_note`` is free text (e.g. a failed val-mm number
    against its acceptance band) for ``"failing_gate"``/``"untrained"``;
    ``None`` when there is nothing more to say than the status itself.
    """

    spec: str
    reader: str | None
    assets: str
    reader_status: str = "accepted"
    reader_note: str | None = None

    @classmethod
    def from_dict(cls, robot: str, entry: Any) -> RobotConfig:
        """Validate one ``robots.<robot>`` entry (``robot`` is the mapping key)."""
        _require(robot in AXIS_VALUES["robot"],
                 f"robots.{robot}: not a registered robot; registered robots "
                 f"are {sorted(AXIS_VALUES['robot'])} (kinescore.robots."
                 f"available_robots())")
        _require(isinstance(entry, dict) and {"spec", "reader", "assets"} <= set(entry),
                 f"robots.{robot}: must have keys spec, reader, assets, "
                 f"got {sorted(entry) if isinstance(entry, dict) else entry!r}")
        unknown = set(entry) - _ROBOT_ENTRY_KEYS
        _require(not unknown,
                 f"robots.{robot}: unknown key(s) {sorted(unknown)}; valid "
                 f"keys are {sorted(_ROBOT_ENTRY_KEYS)}")
        spec = str(entry["spec"])
        _require(spec in AXIS_VALUES["robot"],
                 f"robots.{robot}.spec={spec!r} is not a registered robot; "
                 f"registered robots are {sorted(AXIS_VALUES['robot'])} "
                 f"(kinescore.robots.available_robots()) -- this used to "
                 f"accept any string (e.g. 'aloha', which has never been a "
                 f"registered robot) and silently score nothing")

        reader_status = entry.get("reader_status", "accepted")
        _require(reader_status in READER_STATUS_VALUES,
                 f"robots.{robot}.reader_status={reader_status!r} must be "
                 f"one of {sorted(READER_STATUS_VALUES)}")

        reader_raw = entry.get("reader")
        if reader_status == "untrained":
            _require(reader_raw is None,
                     f"robots.{robot}.reader must be omitted or null when "
                     f"reader_status='untrained' (no checkpoint has been "
                     f"trained for this robot yet -- naming a filename here "
                     f"would be exactly the phantom-checkpoint-path bug "
                     f"this field exists to prevent), got {reader_raw!r}")
            reader: str | None = None
        else:
            _require(isinstance(reader_raw, str) and reader_raw,
                     f"robots.{robot}.reader must be a non-empty checkpoint "
                     f"filename when reader_status={reader_status!r}, got "
                     f"{reader_raw!r} -- if no checkpoint has been trained "
                     f"yet, use reader_status: untrained and reader: null "
                     f"instead")
            reader = reader_raw

        reader_note = entry.get("reader_note")
        _require(reader_note is None or isinstance(reader_note, str),
                 f"robots.{robot}.reader_note must be a string if given, "
                 f"got {reader_note!r}")

        return cls(spec=spec, reader=reader, assets=str(entry["assets"]),
                   reader_status=reader_status, reader_note=reader_note)


#: An ``iter`` pin nested per robot then horizon -- see :class:`SourceConfig`'s
#: ``iter`` docstring for why a single string is not enough: dreamgen and
#: dreamdojo's live inventory has a genuinely different right checkpoint per
#: (robot, horizon), not one value that covers the whole matrix. Keyed by
#: ROBOT (not embodiment) for the same reason ``axes.robot`` replaced
#: ``axes.embodiment`` -- ``humanoid`` alone would be ambiguous between
#: Airbot MMK2 and Fourier GR-1.
IterMap = Mapping[str, Mapping[str, str]]

#: A ``view_dir`` pin nested per robot, with a mandatory ``"default"`` entry
#: -- see :class:`SourceConfig`'s ``view_dir`` docstring for why a single
#: string is not always enough: dreamdojo's bimanual tree genuinely has a
#: different populated view_dir (``single_view``, 103/85 paired episodes)
#: than the other three robots (``singleview``). Unlike :data:`IterMap`,
#: this is flat (robot -> literal directory name, no horizon level) because
#: view_dir does not vary by horizon in any known cell.
ViewDirMap = Mapping[str, str]

#: The one key every :data:`ViewDirMap` must carry -- the view_dir any robot
#: NOT otherwise listed in the mapping resolves to. Required (not merely
#: allowed to be absent) so :meth:`SourceConfig.resolve_view_dir` can never
#: raise for an unlisted robot the way an unresolved ``iter`` pin can -- a
#: source plugin needs a view_dir unconditionally, even for a cell nobody
#: has pinned an override for yet, to build the path it globs.
VIEW_DIR_MAP_DEFAULT_KEY = "default"


@dataclass(frozen=True)
class SourceConfig:
    """Per-generator discovery pins: the literal on-disk directory names.

    Parameters
    ----------
    view_dir:
        Literal directory name under the generator's tree -- one of
        :data:`VIEW_DIR_VALUES`. Never a glob: this is exactly the trap
        ``singleview``/``single_view`` exists to prevent (see
        :data:`VIEW_DIR_VALUES`'s docstring).

        Two shapes, both validated in :meth:`from_dict` (via
        :meth:`_parse_view_dir`) and resolved via :meth:`resolve_view_dir`:

        * A single string -- the same view_dir applies to every robot this
          generator appears in (ctrlworld, dreamgen today).
        * A :data:`ViewDirMap` (``{"default": ..., robot: ...}``) -- a live
          HuggingFace-tree crawl of ``dense/bimanual/output/`` found
          dreamdojo has TWO differently-populated view_dir subtrees for
          aloha_bimanual: ``singleview`` (150 episodes, pred-only, no
          ``full_gt.mp4`` anywhere -- unpairable) and ``single_view``
          (103/85 episodes across makovian/non_makovian, WITH
          ``full_gt.mp4`` -- pairable). Every other robot's dreamdojo cell
          stays on ``singleview`` (its populated, paired tree); only
          aloha_bimanual needs the underscored spelling. A flat string
          cannot express "this one robot differs"; forcing the whole
          generator onto ``single_view`` would silently break
          fourier_gr1/franka_panda's already-verified pins. The mandatory
          ``"default"`` key is what every OTHER robot resolves to -- see
          :data:`VIEW_DIR_MAP_DEFAULT_KEY`.
    iter:
        Pinned training-checkpoint iteration directory (e.g.
        ``"iter_000090000"``), for generators whose output tree has several.
        ``None`` for a generator with no ``iter_*`` level (ctrlworld).

        Three shapes, all validated in :meth:`from_dict` (via
        :meth:`_parse_iter`):

        * ``None`` -- no ``iter_*`` level exists for this generator.
        * A single string -- one iteration applies to every (robot, horizon)
          cell this generator appears in.
        * A mapping ``{robot: {horizon: "iter_..."}}`` -- a live inventory of
          dreamgen/dreamdojo showed the right checkpoint is genuinely
          different **per robot and per horizon** (e.g. dreamgen's
          fourier_gr1 uses ``iter_000090000`` at 130/130 episodes, while
          dreamgen's franka_panda makovian cell only has a well-populated
          iteration under a differently-named ``_static16fps`` directory, and
          franka_panda has no ``non_makovian`` directory for dreamgen at all
          -- that cell is declared N/A instead of guessing an iter for it). A
          cell whose (robot, horizon) has no entry in this mapping resolves
          to ``None`` -- see :meth:`resolve_iter`.

        Never a **list**, at either the flat or the nested-per-cell level --
        a list is exactly "two iter_ values silently averaged/globbed
        together", the bug this field exists to prevent. This guard is
        checked at both nesting depths in :meth:`_parse_iter`.
    gt_from:
        ``"input"`` when this generator has no ground-truth file next to its
        prediction and the ground truth must instead come from the sibling
        ``input/`` tree under the baseline cache (dreamgen). ``None`` when
        ground truth is co-located with the prediction (ctrlworld, dreamdojo).
    """

    view_dir: str | ViewDirMap
    iter: str | IterMap | None = None
    gt_from: str | None = None

    @classmethod
    def from_dict(cls, generator: str, entry: Any) -> SourceConfig:
        """Validate one ``sources.<generator>`` entry."""
        _require(isinstance(entry, dict), f"sources.{generator}: must be a mapping")
        view_dir = cls._parse_view_dir(generator, entry.get("view_dir"))
        it = cls._parse_iter(generator, entry.get("iter"))
        gt_from = entry.get("gt_from")
        _require(gt_from is None or gt_from == "input",
                 f"sources.{generator}.gt_from must be omitted or \"input\", "
                 f"got {gt_from!r}")
        return cls(view_dir=view_dir, iter=it, gt_from=gt_from)

    @staticmethod
    def _parse_view_dir(generator: str, vd: Any) -> str | ViewDirMap:
        """Validate ``sources.<generator>.view_dir`` in either of its two shapes.

        See the class docstring's ``view_dir`` section for what each shape
        means. A bare string must be one of :data:`VIEW_DIR_VALUES`; a
        mapping must carry the mandatory ``"default"`` key (see
        :data:`VIEW_DIR_MAP_DEFAULT_KEY`) plus zero or more registered-robot
        overrides, every value again one of :data:`VIEW_DIR_VALUES`.
        """
        if isinstance(vd, str):
            _require(vd in VIEW_DIR_VALUES,
                     f"sources.{generator}.view_dir must be one of "
                     f"{sorted(VIEW_DIR_VALUES)} (an explicit literal directory "
                     f"name -- never a glob), got {vd!r}")
            return vd
        if isinstance(vd, dict):
            _require(VIEW_DIR_MAP_DEFAULT_KEY in vd,
                     f"sources.{generator}.view_dir, given as a mapping, must "
                     f"carry a {VIEW_DIR_MAP_DEFAULT_KEY!r} key -- the view_dir "
                     f"every robot NOT otherwise listed resolves to; got keys "
                     f"{sorted(vd)}")
            out: dict[str, str] = {}
            for robot, value in vd.items():
                _require(robot == VIEW_DIR_MAP_DEFAULT_KEY or robot in AXIS_VALUES["robot"],
                         f"sources.{generator}.view_dir.{robot}: not "
                         f"{VIEW_DIR_MAP_DEFAULT_KEY!r} and not a registered "
                         f"robot; registered robots are "
                         f"{sorted(AXIS_VALUES['robot'])}")
                _require(isinstance(value, str) and value in VIEW_DIR_VALUES,
                         f"sources.{generator}.view_dir.{robot} must be one of "
                         f"{sorted(VIEW_DIR_VALUES)}, got {value!r}")
                out[robot] = value
            return out
        _require(False,
                 f"sources.{generator}.view_dir must be a single string (one "
                 f"literal directory name) or a mapping with a "
                 f"{VIEW_DIR_MAP_DEFAULT_KEY!r} key plus per-robot overrides, "
                 f"got {vd!r} of type {type(vd).__name__}")
        raise AssertionError("unreachable")  # _require(False, ...) always raises

    @staticmethod
    def _parse_iter(generator: str, it: Any) -> str | dict[str, dict[str, str]] | None:
        """Validate ``sources.<generator>.iter`` in either of its two shapes.

        See the class docstring's ``iter`` section for what each shape means.
        Raises on a THIRD shape too: an unrecognised type (e.g. a number) is
        rejected with the same "single string" message a list gets, since
        both are "not one of the two documented shapes."
        """
        if it is None or isinstance(it, str):
            return it
        if isinstance(it, dict):
            out: dict[str, dict[str, str]] = {}
            for robot, horizon_map in it.items():
                _require(robot in AXIS_VALUES["robot"],
                         f"sources.{generator}.iter.{robot}: not a registered "
                         f"robot; registered robots are "
                         f"{sorted(AXIS_VALUES['robot'])}")
                _require(isinstance(horizon_map, dict) and horizon_map,
                         f"sources.{generator}.iter.{robot} must be a "
                         f"non-empty mapping of horizon -> iter string, got "
                         f"{horizon_map!r}")
                inner: dict[str, str] = {}
                for horizon, value in horizon_map.items():
                    _require(horizon in AXIS_VALUES["horizon"],
                             f"sources.{generator}.iter.{robot}.{horizon}: "
                             f"not a valid horizon; valid values are "
                             f"{sorted(AXIS_VALUES['horizon'])}")
                    _require(isinstance(value, str) and value,
                             f"sources.{generator}.iter.{robot}.{horizon} "
                             f"must be a single string (one pinned iter_* "
                             f"directory), got {value!r} of type "
                             f"{type(value).__name__} -- a list means more than "
                             f"one checkpoint iteration would be mixed into one "
                             f"number, which is exactly the bug this field "
                             f"prevents")
                    inner[horizon] = value
                out[robot] = inner
            return out
        _require(False,
                 f"sources.{generator}.iter must be a single string (one pinned "
                 f"iter_* directory) or a mapping of robot -> horizon -> "
                 f"string, got {it!r} of type {type(it).__name__} -- a list "
                 f"means more than one checkpoint iteration would be mixed into "
                 f"one number, which is exactly the bug this field prevents")
        raise AssertionError("unreachable")  # _require(False, ...) always raises

    def resolve_view_dir(self, *, robot: str) -> str:
        """This source's pinned ``view_dir`` for one robot.

        * ``self.view_dir`` is a string -- returned unchanged regardless of
          ``robot`` (every non-dreamdojo generator today).
        * ``self.view_dir`` is a :data:`ViewDirMap` -- ``self.view_dir[robot]``
          if present, else ``self.view_dir["default"]`` (guaranteed present
          by :meth:`_parse_view_dir`, so this never raises ``KeyError``).

        Unlike :meth:`resolve_iter`, this never returns ``None`` -- a source
        plugin needs a literal directory name unconditionally to build the
        path it globs, even for a cell no one has pinned an explicit
        override for.
        """
        if isinstance(self.view_dir, str):
            return self.view_dir
        return self.view_dir.get(robot, self.view_dir[VIEW_DIR_MAP_DEFAULT_KEY])

    def resolve_iter(self, *, robot: str, horizon: str) -> str | None:
        """This source's pinned iter for one (robot, horizon) cell.

        * ``self.iter is None`` -- generator has no ``iter_*`` level at all
          (ctrlworld); always resolves to ``None``.
        * ``self.iter`` is a string -- the same pin applies to every cell;
          returned unchanged regardless of ``robot``/``horizon``.
        * ``self.iter`` is a mapping -- looks up
          ``self.iter[robot][horizon]``, returning ``None`` (not a
          ``KeyError``) if either level has no entry. A cell whose iter
          resolves to ``None`` here is a config bug UNLESS that cell is also
          listed in ``na_cells`` -- ``bench.matrix.expand`` is what turns a
          ``None`` resolution on a real (non-N/A) cell into a hard,
          early ``ConfigError``, not this method (this method has no access
          to ``na_cells`` and must stay a pure, always-succeeding lookup).
        """
        if self.iter is None or isinstance(self.iter, str):
            return self.iter
        return self.iter.get(robot, {}).get(horizon)


@dataclass(frozen=True)
class NaCellRule:
    """A partial axis->value match; every cell it matches is N/A.

    ``{"generator": "dreamdojo", "view": "multiview"}`` marks every cell with
    that generator and that view N/A regardless of the other three axes --
    the match is on the KEYS PRESENT in ``axes`` only, not on all five.
    """

    axes: Mapping[str, str]

    @classmethod
    def from_dict(cls, entry: Any, *, index: int) -> NaCellRule:
        """Validate one ``na_cells[index]`` entry."""
        _require(isinstance(entry, dict) and entry,
                 f"na_cells[{index}]: must be a non-empty mapping, got {entry!r}")
        bad_keys = set(entry) - set(AXIS_VALUES)
        _require(not bad_keys,
                 f"na_cells[{index}]: unknown axis key(s) {sorted(bad_keys)}; "
                 f"valid axis keys are {sorted(AXIS_VALUES)}")
        for axis, value in entry.items():
            _require(value in AXIS_VALUES[axis],
                     f"na_cells[{index}].{axis}={value!r} is not a valid {axis} "
                     f"value; valid values are {sorted(AXIS_VALUES[axis])}")
        return cls(axes=dict(entry))

    def matches(self, *, robot: str, view: str, horizon: str,
                cache: str, generator: str) -> bool:
        values = {"robot": robot, "view": view, "horizon": horizon,
                 "cache": cache, "generator": generator}
        return all(values.get(k) == v for k, v in self.axes.items())


@dataclass(frozen=True)
class AxesConfig:
    """The five matrix axes, each a tuple of the values to include."""

    robot: tuple[str, ...]
    view: tuple[str, ...]
    horizon: tuple[str, ...]
    cache: tuple[str, ...]
    generator: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: Any) -> AxesConfig:
        """Validate the ``axes`` mapping: exactly the five known keys, each
        a non-empty list of values from :data:`AXIS_VALUES`.
        """
        _require(isinstance(raw, dict) and set(raw) == set(AXIS_VALUES),
                 f"axes: must declare exactly the keys {sorted(AXIS_VALUES)}, "
                 f"got {sorted(raw) if isinstance(raw, dict) else type(raw).__name__}")
        kwargs: dict[str, tuple[str, ...]] = {}
        for axis, valid in AXIS_VALUES.items():
            values = raw[axis]
            _require(isinstance(values, list) and values,
                     f"axes.{axis}: must be a non-empty list, got {values!r}")
            bad = [v for v in values if v not in valid]
            _require(not bad,
                     f"axes.{axis}: unknown value(s) {bad!r}; valid values are "
                     f"{sorted(valid)}")
            kwargs[axis] = tuple(values)
        return cls(**kwargs)


def _parse_na_cells(raw: Any) -> tuple[NaCellRule, ...]:
    if raw is None:
        return ()
    _require(isinstance(raw, list), f"na_cells: must be a list, got {type(raw).__name__}")
    return tuple(NaCellRule.from_dict(entry, index=i) for i, entry in enumerate(raw))


def _parse_robots(raw: Any, axes: AxesConfig) -> dict[str, RobotConfig]:
    _require(isinstance(raw, dict) and raw,
             f"robots: must be a non-empty mapping, got {type(raw).__name__}")
    out = {robot: RobotConfig.from_dict(robot, entry) for robot, entry in raw.items()}
    missing = [r for r in axes.robot if r not in out]
    _require(not missing,
             f"robots: no entry for robot(s) {missing} that appear in "
             f"axes.robot; every robot in the matrix needs a robot "
             f"spec/reader/assets triple")
    return out


def _parse_sources(raw: Any, axes: AxesConfig) -> dict[str, SourceConfig]:
    _require(isinstance(raw, dict) and raw,
             f"sources: must be a non-empty mapping, got {type(raw).__name__}")
    out: dict[str, SourceConfig] = {}
    for generator, entry in raw.items():
        _require(generator in AXIS_VALUES["generator"],
                 f"sources.{generator}: not a valid generator; valid values "
                 f"are {sorted(AXIS_VALUES['generator'])}")
        out[generator] = SourceConfig.from_dict(generator, entry)
    missing = [g for g in axes.generator if g not in out]
    _require(not missing,
             f"sources: no entry for generator(s) {missing} that appear in "
             f"axes.generator")
    return out


def _parse_fps_expected(raw: Any) -> dict[str, float | dict[str, float]]:
    """``{generator: fps}`` or ``{generator: {robot: fps}}``.

    The nested form exists because **one generator can emit different frame
    rates for different robots**, and a flat table cannot say so. Measured:
    ``dreamdojo`` is 10 fps for ``fourier_gr1`` but **15 fps** for
    ``franka_panda``. With the flat table, every one of 480
    ``franka_panda``/dreamdojo episodes was silently dropped to zero rows --
    ``video/probe.py::resolve_timebase`` correctly refused to reconcile a
    declared 10 against a probed 15 (defect D3 protection working exactly as
    intended), and the result was an empty cell that looked like missing data
    rather than a wrong config.

    That is the third distinct frame-rate mismatch found in this corpus, after
    5-vs-30 fps *within* one ctrlworld cell and 15-vs-16 between a generator and
    its own ground truth. The lesson each time is the same: a declared rate is a
    **cross-check that may only fail**, never a value that wins, and the table
    must be allowed to be as fine-grained as reality is.
    """
    if raw is None:
        return {}
    _require(isinstance(raw, dict),
             f"fps_expected: must be a mapping, got {type(raw).__name__}")

    def _positive(value: Any, where: str) -> float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
        _require(ok, f"{where} must be a positive float, got {value!r}")
        return float(value)

    out: dict[str, float | dict[str, float]] = {}
    for generator, value in raw.items():
        _require(generator in AXIS_VALUES["generator"],
                 f"fps_expected.{generator}: not a valid generator; valid "
                 f"values are {sorted(AXIS_VALUES['generator'])}")
        if isinstance(value, dict):
            per_robot: dict[str, float] = {}
            for robot, fps in value.items():
                _require(robot in AXIS_VALUES["robot"],
                         f"fps_expected.{generator}.{robot}: not a "
                         f"registered robot; registered robots are "
                         f"{sorted(AXIS_VALUES['robot'])}")
                per_robot[robot] = _positive(
                    fps, f"fps_expected.{generator}.{robot}")
            out[generator] = per_robot
        else:
            out[generator] = _positive(value, f"fps_expected.{generator}")
    return out


def _parse_caps(raw: Any) -> dict[str, int]:
    if raw is None:
        return {}
    _require(isinstance(raw, dict), f"caps: must be a mapping, got {type(raw).__name__}")
    out: dict[str, int] = {}
    for key, value in raw.items():
        ok = isinstance(value, int) and not isinstance(value, bool) and value > 0
        _require(ok, f"caps.{key} must be a positive integer, got {value!r}")
        out[key] = int(value)
    return out


@dataclass(frozen=True)
class BenchConfig:
    """A fully validated ``benchmark.yaml``. See the module docstring."""

    run_id: str
    seed: int
    axes: AxesConfig
    na_cells: tuple[NaCellRule, ...]
    robots: Mapping[str, RobotConfig]
    sources: Mapping[str, SourceConfig]
    fps_expected: Mapping[str, float | Mapping[str, float]]
    rate_policy: str
    suite: str
    baseline_cache: str
    caps: Mapping[str, int]
    path: str | None = field(default=None, compare=False)

    @classmethod
    def from_dict(cls, raw: dict, *, path: str | None = None) -> BenchConfig:
        """Validate an already-parsed config mapping into a :class:`BenchConfig`.

        Split out from :func:`load_config` so tests can exercise validation on
        a plain dict without writing a YAML file to disk.

        Rejects any top-level key not in :data:`_TOP_LEVEL_KEYS` -- this
        module's own docstring (see above) promises hard-and-early
        validation, but every field here used to be read via
        ``raw.get(...)`` with no check that ``set(raw)`` was actually
        covered, so a typo like ``rate_polciy`` was silently ignored and
        ``rate_policy`` silently kept its default instead of the typo'd
        value ever being read. This is that check.
        """
        _require(isinstance(raw, dict), f"config root must be a mapping, got {type(raw).__name__}")
        unknown = set(raw) - _TOP_LEVEL_KEYS
        _require(not unknown,
                 f"config has unknown top-level key(s) {sorted(unknown)}; valid "
                 f"keys are {sorted(_TOP_LEVEL_KEYS)} -- if this is a typo (e.g. "
                 f"'rate_polciy'), the real key silently kept its default with "
                 f"no error before this check existed")
        raw = expand_env(raw)

        run_id = raw.get("run_id")
        _require(isinstance(run_id, str) and run_id, "run_id: must be a non-empty string")

        seed = raw.get("seed", 0)
        _require(isinstance(seed, int) and not isinstance(seed, bool), "seed: must be an integer")

        axes = AxesConfig.from_dict(raw.get("axes", {}))
        na_cells = _parse_na_cells(raw.get("na_cells"))
        robots = _parse_robots(raw.get("robots", {}), axes)
        sources = _parse_sources(raw.get("sources", {}), axes)
        fps_expected = _parse_fps_expected(raw.get("fps_expected"))

        rate_policy = raw.get("rate_policy", "paired")
        _require(isinstance(rate_policy, str) and RATE_POLICY_RE.match(rate_policy),
                 f"rate_policy must be 'paired', 'rate_free' or 'resample:<hz>', "
                 f"got {rate_policy!r}")

        suite = raw.get("suite")
        _require(isinstance(suite, str) and suite, "suite: must be a non-empty string")

        baseline_cache = raw.get("baseline_cache")
        _require(isinstance(baseline_cache, str) and baseline_cache in axes.cache,
                 f"baseline_cache={baseline_cache!r} must be one of axes.cache "
                 f"={list(axes.cache)}")

        caps = _parse_caps(raw.get("caps"))

        return cls(run_id=run_id, seed=seed, axes=axes, na_cells=na_cells,
                   robots=robots, sources=sources, fps_expected=fps_expected,
                   rate_policy=rate_policy, suite=suite,
                   baseline_cache=baseline_cache, caps=caps, path=path)

    def resolve_fps(self, *, generator: str, robot: str) -> float | None:
        """Declared fps for one (generator, robot), or ``None`` if undeclared.

        ``None`` means "trust the probe outright" and is the safe default --
        never a guess. A declared value is only ever a **cross-check**:
        ``video/probe.py::resolve_timebase`` raises if it disagrees with what
        ffprobe measures, in either direction. It can fail a run; it can never
        override a measurement.

        Prefer omitting an entry to guessing one. An absent entry costs nothing;
        a wrong one silently empties a whole cell -- see
        :func:`_parse_fps_expected` for the 480-episode case that motivated the
        nested form.
        """
        entry = self.fps_expected.get(generator)
        if entry is None:
            return None
        if isinstance(entry, Mapping):
            return entry.get(robot)
        return float(entry)


def from_dict(raw: dict, *, path: str | None = None) -> BenchConfig:
    """Validate an already-parsed config mapping into a :class:`BenchConfig`.

    A thin module-level alias for :meth:`BenchConfig.from_dict` -- kept so
    ``from kinescore.bench.config import from_dict`` (every existing caller
    and test) keeps working unchanged.
    """
    return BenchConfig.from_dict(raw, path=path)


def load_config(path: str | Path) -> BenchConfig:
    """Read and validate a ``benchmark.yaml`` file.

    Raises
    ------
    ConfigError
        The file is structurally invalid (see the individual ``from_dict``
        methods/``_parse_*`` helpers for exact messages).
    kinescore.paths.MissingPathError
        The file references ``${SOME_KINESCORE_VAR}`` and that variable is
        not set in the environment.
    """
    import yaml

    path = str(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    return BenchConfig.from_dict(raw, path=path)
