"""One module per corpus shape; importing this package registers them all."""
from kinescore.adapters.base import (
    DatasetAdapter,
    RawEpisode,
    SkippedEpisode,
    available_adapters,
    get_adapter,
    register_adapter,
)
from kinescore.adapters.canonical import CanonicalTreeAdapter
from kinescore.adapters.ctrlworld import CtrlWorldTeleopAdapter

__all__ = [
    "RawEpisode", "SkippedEpisode", "DatasetAdapter", "CtrlWorldTeleopAdapter",
    "CanonicalTreeAdapter", "get_adapter", "available_adapters",
    "register_adapter",
]
