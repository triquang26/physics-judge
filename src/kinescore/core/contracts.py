"""Every extension-point interface, in one place.

This module does not define new contracts for the axes that already have
one -- :class:`~kinescore.core.robot.RobotSpec`,
:class:`~kinescore.core.reader.PoseReader`,
:class:`~kinescore.core.metric.Metric` are each still owned by their home
module (``core/robot.py``, ``core/reader.py``, ``core/metric.py``
respectively); this module only re-exports them so a caller wiring up a new
robot/reader/metric/source/layout has one door to knock on instead of five.

Two contracts are new here, formalising shapes ``bench`` already implements
concretely as of this writing:

* :class:`ClipSource` -- mirrors ``bench/sources/base.py::ClipSource``
  exactly (``GENERATOR`` class var, ``make_plugin(cell, data_root, config) ->
  SourcePlugin``, a ``__call__`` alias). That module's own docstring already
  anticipated this: "``kinescore.core.contracts`` was checked for an
  existing ``ClipSource`` definition before this one was written [...] if a
  future ``kinescore.core.contracts.ClipSource`` appears, importing from
  there instead of here is the fix, not maintaining two definitions." This
  is that future definition; ``bench`` owns making the swap.
* :class:`DataLayout` -- mirrors ``bench/layout.py::DataLayout`` exactly
  (``cells()``/``cell_dir(cell)``/``validate() -> list[str]``), which left
  the identical note. ``bench`` owns making the swap there too.

Import-light on purpose: no ``torch`` at module scope of *this* module
(``RobotSpec``'s own module needs it for tensor-typed attributes, so
re-exporting ``RobotSpec`` still pulls torch in transitively -- unavoidable,
not something this module adds).

No ``bench`` import, not even under ``TYPE_CHECKING``
--------------------------------------------------------
``core`` is lower-level than ``bench`` in this package's layering (see
``core/__init__.py``'s docstring): ``bench`` depends on ``core``, never the
reverse. An annotations-only ``TYPE_CHECKING`` import of ``bench.cell.Cell``/
``bench.config.BenchConfig``/``bench.manifest.SourcePlugin`` used to sit here
-- it broke nothing at runtime (``from __future__ import annotations`` defers
every annotation to a string), but it meant this file, read on its own,
could not be understood without also knowing ``bench``'s shapes, and a
"``core`` never imports ``bench``" check could not tell an inert
type-only import apart from a real one without a special case for this file.

:data:`CellT` and :data:`ConfigT` below replace that import. Neither
``ClipSource`` nor ``DataLayout`` actually reads a field off a cell or a
config anywhere in this file -- ``DataLayout.cell_dir`` just needs to receive
back, unchanged, whatever :meth:`DataLayout.cells` handed out, and
``ClipSource.make_plugin`` just threads ``cell``/``config`` straight through
to its returned closure. That is an identity relationship a bound
:class:`~typing.TypeVar` expresses honestly (this parameter is *some* type,
consistent across one layout/source's calls, but ``core`` does not care
which); a :class:`~typing.Protocol` would have to invent attributes this
module never uses just to have something to declare, which would be less
honest, not more. ``bench.cell.Cell``/``bench.config.BenchConfig`` remain
the only types anything actually instantiates here -- see
``bench/layout.py``'s and ``bench/sources/*.py``'s ``DataLayout[Cell]``/
``ClipSource[Cell, BenchConfig]`` subclasses, which is where the concrete
types get bound back in.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from typing import Any, ClassVar, Generic, TypeVar

from kinescore.core.metric import Metric
from kinescore.core.reader import PoseReader
from kinescore.core.robot import RobotSpec

__all__ = ["RobotSpec", "PoseReader", "Metric", "ClipSource", "DataLayout",
          "CellT", "ConfigT"]

#: Structural stand-in for a benchmark "cell" (``bench.cell.Cell`` is the one
#: concrete implementation as of this writing). See the module docstring's
#: "No ``bench`` import" section for why a ``TypeVar``, not the concrete
#: type or an invented ``Protocol``.
CellT = TypeVar("CellT")

#: Structural stand-in for a validated benchmark config
#: (``bench.config.BenchConfig`` is the one concrete implementation as of
#: this writing) -- same reasoning as :data:`CellT`.
ConfigT = TypeVar("ConfigT")


class ClipSource(ABC, Generic[CellT, ConfigT]):
    """One generator's discovery plugin (ctrlworld, dreamdojo, dreamgen, ...).

    A concrete subclass sets :attr:`GENERATOR` to the axis value it
    implements, binds :data:`CellT`/:data:`ConfigT` to its real cell/config
    types (e.g. ``ClipSource[Cell, BenchConfig]``), and implements
    :meth:`make_plugin`, which validates everything it can from
    ``cell``/``config`` alone and raises immediately (never inside the
    returned closure), so a caller building plugins for many cells fails
    fast on the first bad one rather than after globbing every other cell
    first. The returned zero-argument closure (shaped like
    ``bench.manifest.SourcePlugin`` -- a callable ``core`` cannot name
    exactly either, see the module docstring) is what actually globs the
    filesystem, and only when ``bench.manifest.build_manifest`` calls it.
    """

    #: The ``generator`` axis value this source implements.
    GENERATOR: ClassVar[str]

    @abstractmethod
    def make_plugin(self, cell: CellT, data_root: str, config: ConfigT
                    ) -> Callable[[], Iterable[Any]]:
        """Build the zero-arg plugin discovering ``cell``'s episodes."""
        ...

    def __call__(self, cell: CellT, data_root: str, config: ConfigT
                 ) -> Callable[[], Iterable[Any]]:
        """Convenience alias for :meth:`make_plugin` -- a source is callable."""
        return self.make_plugin(cell, data_root, config)


class DataLayout(ABC, Generic[CellT]):
    """Where a benchmark's clips live, and how to enumerate/locate them.

    A benchmark scores results per matrix cell (robot x view x horizon x
    cache x generator, see ``bench/cell.py::Cell``). A ``DataLayout``
    resolves cell identity to on-disk location and checks that the layout is
    well-formed, so store/runner code depends on this contract instead of
    reaching into filesystem paths ad hoc per caller. A concrete subclass
    binds :data:`CellT` to its real cell type (e.g. ``DataLayout[Cell]``).
    """

    @abstractmethod
    def cells(self) -> Iterator[CellT]:
        """Every cell this layout has (or expects to have) data for."""
        ...

    @abstractmethod
    def cell_dir(self, cell: CellT) -> str:
        """Filesystem directory holding ``cell``'s clips."""
        ...

    @abstractmethod
    def validate(self) -> list[str]:
        """Human-readable problems found, empty if the layout is internally consistent."""
        ...
