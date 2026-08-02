"""``Cell``: the benchmark's unit of discovery, keyed by ROBOT, not embodiment.

Why this replaces ``embodiment`` as the identity
--------------------------------------------------
The on-disk dataset's ``embodiment`` directory (``humanoid``/``single_arm``/
``bimanual``) is a coarse physical-form label, not a robot. ``humanoid`` alone
covers TWO physically different robots depending on which generator wrote the
clip: Airbot MMK2 (ctrlworld -- episode directories are literally named
``episode_AIRBOT_MMK2_*``) and Fourier GR-1 (dreamgen/dreamdojo). Scoring
either against a reader trained for the other robot would silently compare
the wrong kinematics. ``embodiment`` cannot express this because it is a
directory name, not a (robot, generator) function -- so it is demoted here to
a *derived, recorded* field (see :attr:`Cell.embodiment`), and ``robot`` (a
:mod:`kinescore.robots` registry key) becomes the primary key every other
module keys off.

The (embodiment, generator) -> robot mapping itself lives in
``configs/robot_map.yaml`` (see :mod:`kinescore.bench.robot_map`), not here --
this module only defines the identity a resolved mapping produces, so a
config edit (a fifth robot, a generator moving to a different robot) never
requires touching this dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass

from kinescore.core.clip import ViewLayout

__all__ = ["Cell", "PATH_AXIS_ORDER", "RAW_TREE_ROOT", "raw_tree_segments"]

#: The ONE place path-segment ordering is declared. Every module that builds
#: an on-disk path from a cell's axes (``bench.layout``, ``bench.ingest``,
#: ``bench.sources.*``) imports this instead of hand-writing
#: ``os.path.join(cache, robot, view, generator, horizon)`` -- so reordering
#: the canonical layout (the plan's target shape,
#: ``bench/<cache>/<robot>/<view>/<generator>/<horizon>/``) is a one-line
#: change here, not a grep-and-fix across every consumer.
PATH_AXIS_ORDER: tuple[str, ...] = ("cache", "robot", "view", "generator", "horizon")

#: The literal top-level directory name every raw Hugging-Face-mirror path
#: starts under -- shared by :func:`raw_tree_segments` so the three places
#: that used to hand-write ``"video_gen_physics"`` plus the same six-segment
#: shape independently (``bench.matrix``'s on-disk iter validation and
#: ``allow_patterns``, ``bench.layout.RawHFLayout.cell_dir``) cannot drift
#: from each other. NOT the same shape as :data:`PATH_AXIS_ORDER`: this is
#: the layout kinescore READS from the HF mirror as shipped (keyed by
#: ``embodiment`` + the literal on-disk ``view_dir``), ``PATH_AXIS_ORDER`` is
#: the layout kinescore WRITES to (``kinescore data ingest``'s canonical
#: tree, keyed by ``robot`` + the ``view`` axis value) -- see
#: ``bench.layout``'s module docstring for the two-layouts split.
RAW_TREE_ROOT = "video_gen_physics"


def raw_tree_segments(*, cache: str, embodiment: str, view_dir: str, generator: str,
                      horizon: str, stage: str = "output",
                      iter: str | None = None) -> tuple[str, ...]:
    """The raw HF-mirror path segments for one cell, in the shipped order.

    ``stage`` is ``"output"`` (a generator's own predictions/ground truth,
    the default) or ``"input"`` (dreamgen's separate ground-truth tree, see
    ``kinescore.bench.config.SourceConfig.gt_from``). ``iter``, if given, is
    appended as a final segment (a generator's ``iter_*`` pin).

    Returns segments only -- never a joined string or path -- so a caller
    building a real filesystem path prepends its own root via
    ``os.path.join(root, *segments)`` and a caller building an HF
    ``allow_patterns`` glob joins with ``"/"`` instead; both need exactly the
    same segment order, which is the point of sharing this function rather
    than each hand-writing it.
    """
    segments = (RAW_TREE_ROOT, cache, embodiment, stage, view_dir, generator, horizon)
    if iter is not None:
        segments += (iter,)
    return segments


@dataclass(frozen=True)
class Cell:
    """One benchmark cell: cache x robot x view x generator x horizon.

    Parameters
    ----------
    cache:
        Diffusion-acceleration cache method (``dense``, ``dicache``, ...).
    robot:
        A :func:`kinescore.robots.available_robots` registry key -- the
        PRIMARY key. See the module docstring for why this, not
        ``embodiment``, is the identity.
    view:
        ``multiview`` / ``singleview`` (the dataset-wide axis value; see
        ``kinescore.bench.config.VIEW_DIR_VALUES`` for the literal on-disk
        directory name, which may differ, e.g. ``single_view``).
    generator:
        ``ctrlworld`` / ``dreamgen`` / ``dreamdojo``.
    horizon:
        ``makovian`` / ``non_makovian`` (verbatim dataset spelling -- see
        ``configs/data_spec.yaml``'s header comment).
    embodiment:
        The on-disk embodiment directory name this cell's data actually
        lives under (``humanoid``/``single_arm``/``bimanual``). DERIVED --
        never independently chosen -- from ``robot`` via
        ``configs/robot_map.yaml``, and carried here only because every
        path-building consumer (the raw HF layout) still needs it: the
        dataset's directory names are embodiment-shaped, robot identity is
        a kinescore-side overlay on top of them.
    view_layout:
        Camera packing convention for this cell (see
        :class:`~kinescore.core.clip.ViewLayout`). A generator plugin may
        still override this with its own measured packing (see
        ``bench.sources.ctrlworld``'s docstring for why ctrlworld must).
    iter:
        Resolved ``sources.<generator>.iter`` pin for this cell, or ``None``
        for a generator with no ``iter_*`` level. See
        ``kinescore.bench.config.SourceConfig.resolve_iter``.
    """

    cache: str
    robot: str
    view: str
    generator: str
    horizon: str
    embodiment: str
    view_layout: ViewLayout
    iter: str | None = None

    @property
    def family(self) -> str:
        """Stable, parseable key encoding every axis (``key=value`` pairs).

        ``embodiment`` is included at the end, after the five primary axes,
        so existing consumers that split on ``|`` and look up ``robot=``/
        ``cache=``/etc. by name are unaffected by its presence, while a
        report can still recover which on-disk directory a row came from.
        """
        return (f"cache={self.cache}|robot={self.robot}|view={self.view}|"
               f"generator={self.generator}|horizon={self.horizon}|"
               f"embodiment={self.embodiment}")

    @property
    def cell_id(self) -> str:
        """Short slash-joined identity, in :data:`PATH_AXIS_ORDER`, for filenames/logs."""
        return "/".join(getattr(self, axis) for axis in PATH_AXIS_ORDER)
