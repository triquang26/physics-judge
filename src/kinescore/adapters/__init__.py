"""One module per corpus shape; importing this package registers them all."""
from kinescore.adapters.base import (
    DatasetAdapter,
    RawEpisode,
    SkippedEpisode,
    available_adapters,
    get_adapter,
    register_adapter,
)
from kinescore.adapters.lerobot import LeRobotAdapter

__all__ = [
    "RawEpisode", "SkippedEpisode", "DatasetAdapter", "LeRobotAdapter",
    "get_adapter", "available_adapters", "register_adapter",
]
