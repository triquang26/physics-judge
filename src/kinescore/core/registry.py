"""One generic registry, reused by every extension axis (robots, metrics,
suites, and now benchmark sources / data layouts via :mod:`kinescore.core.contracts`).

Before this module each axis grew its own hand-rolled ``dict[str, Callable]``
plus a ``get_x``/``available_x`` pair -- see ``kinescore.robots._FACTORIES``,
which is the model this generalises rather than replaces (it stays
untouched: robots' lazy-import discipline is exactly what :class:`Registry`
preserves below, and there is no benefit to a mechanical rewrite of code that
already gets this right).

The one property every caller depends on: **factories are lazy**. A registry
stores ``name -> zero-arg factory``, never a constructed instance and never
an eagerly-imported class. Registering ``"franka_panda"`` must not import
``pytorch_kinematics``; only *calling* :meth:`Registry.get` may. This is what
keeps a CPU-only, network-free test run fast and keeps a caller who only
ever asks for ``"synthetic_2r"`` from ever triggering a heavy, optional-in-
spirit dependency it doesn't need.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

__all__ = ["Registry"]

T = TypeVar("T")


class Registry(Generic[T]):
    """``name -> lazy factory -> T``, for one extension axis.

    Parameters
    ----------
    kind:
        Human-readable name for what this registry holds (``"robot"``,
        ``"metric suite"``, ...) -- used only to make error messages name
        the right thing when a project has several registries in play.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._factories: dict[str, Callable[[], T]] = {}

    def register(self, name: str, factory: Callable[[], T]) -> None:
        """Add ``name -> factory``. ``factory`` is stored, not called.

        Raises
        ------
        ValueError
            If ``name`` is already registered -- a silent overwrite would let
            a second, differently-behaved implementation shadow the first
            with no signal at the call site.
        """
        if name in self._factories:
            raise ValueError(f"{self._kind} {name!r} is already registered")
        self._factories[name] = factory

    def get(self, name: str, **kwargs) -> T:
        """Call and return the factory registered under ``name``.

        Parameters
        ----------
        **kwargs:
            Forwarded to the factory (e.g. ``device``/``dtype``/``urdf_path``
            overrides some factories accept). Factories that take no
            arguments simply ignore an empty ``kwargs``, so this is
            backward-compatible with every existing zero-arg registration.

        Raises
        ------
        ValueError
            If ``name`` is not registered -- lists every valid name, so a
            typo fails at the call site instead of surfacing deep inside
            whatever consumed the missing ``T``.
        """
        try:
            factory = self._factories[name]
        except KeyError:
            raise ValueError(
                f"unknown {self._kind} {name!r}; available: "
                f"{list(self.available())}") from None
        return factory(**kwargs)

    def available(self) -> tuple[str, ...]:
        """Registered names, in a stable (sorted) order."""
        return tuple(sorted(self._factories))
