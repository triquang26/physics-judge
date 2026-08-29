"""What an adapter yields, and the registry that finds one."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from kinescore.core.registry import Registry

if TYPE_CHECKING:
    from kinescore.registry.cells import TrainSource

__all__ = [
    "RawEpisode", "SkippedEpisode", "DatasetAdapter", "register_adapter",
    "get_adapter", "available_adapters",
]


@dataclass(frozen=True)
class RawEpisode:
    """One episode of real teleop, as its source stores it.

    Attributes
    ----------
    episode_id:
        Unique within the corpus; becomes the filename.
    views:
        Camera name -> mp4 path, in the order the target view expects.
    packed:
        A single mp4 already packed to the target view.
    joints:
        ``[T, J]`` in the robot's canonical joint order -- the adapter has
        already selected and reordered the source's columns.
    gripper:
        ``[T]`` gripper opening, or ``None``.
    fps:
        Frames per second, from the source.
    scene_key:
        Task or scene identity. Whole scenes move together across the
        train/val split, so a validation number measures generalisation.
    source_path:
        Where this episode was read from.
    """

    episode_id: str
    joints: np.ndarray
    fps: float
    scene_key: str
    source_path: str
    views: dict[str, str] = field(default_factory=dict)
    packed: str | None = None
    gripper: np.ndarray | None = None

    def __post_init__(self) -> None:
        if bool(self.views) == (self.packed is not None):
            raise ValueError(
                f"episode {self.episode_id!r} must carry either per-view "
                f"files or one packed file, not both and not neither "
                f"(views={sorted(self.views)}, packed={self.packed!r})")
        if self.joints.ndim != 2:
            raise ValueError(
                f"episode {self.episode_id!r}: joints must be [T, J], got "
                f"shape {self.joints.shape}")


@dataclass(frozen=True)
class SkippedEpisode:
    """An episode the adapter could not use, and why."""

    episode_id: str
    reason: str
    source_path: str


@runtime_checkable
class DatasetAdapter(Protocol):
    """Reads one corpus shape.

    Attributes
    ----------
    SOURCE_ID:
        Name used in ``cells.yaml``'s ``train.adapter``.
    """

    SOURCE_ID: str

    def episodes(self, source: TrainSource
                 ) -> Iterator[RawEpisode | SkippedEpisode]:
        """Walk ``source.root``, yielding one entry per episode found."""
        ...


_REGISTRY: Registry[DatasetAdapter] = Registry(kind="dataset adapter")


def register_adapter(name: str, factory: Callable[[], DatasetAdapter]) -> None:
    """Register ``name -> factory``; the factory is stored, not called."""
    _REGISTRY.register(name, factory)


def get_adapter(name: str) -> DatasetAdapter:
    """Construct the named adapter, listing every name on a miss."""
    import kinescore.adapters  # noqa: F401  -- populates the registry

    return _REGISTRY.get(name)


def available_adapters() -> tuple[str, ...]:
    """Registered adapter names, sorted."""
    import kinescore.adapters  # noqa: F401

    return _REGISTRY.available()
