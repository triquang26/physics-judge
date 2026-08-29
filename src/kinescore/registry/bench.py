"""The scored clip set, read from ``bench/manifest.json``."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from kinescore.paths import env_path

__all__ = ["BenchItem", "bench_root", "load_bench", "select"]

_FIELDS = ("embodiment", "view", "model", "split", "task", "role", "aug_tag",
           "source_path")


@dataclass(frozen=True)
class BenchItem:
    """One clip and the coordinates it is reported under."""

    id: str
    path: str
    embodiment: str
    view: str
    model: str
    split: str
    method: str
    role: str
    task: str
    aug_tag: str
    #: Where the clip sits in the dataset it was drawn from, e.g.
    #: ``augment/bimanual/output/multiview/.../pred_all_views.mp4``. The bench
    #: flattens every clip to ``clips/<id>.mp4``, so this is what ties a scored
    #: row back to the tree it was drawn from.
    source_path: str

    def coords(self) -> dict[str, str]:
        return {"id": self.id, "method": self.method,
                **{f: getattr(self, f) for f in _FIELDS}}


def bench_root() -> Path:
    return env_path("KINESCORE_DATA_ROOT") / "bench"


def load_bench(root: Path | None = None) -> list[BenchItem]:
    """Every clip the manifest declares, checked against what is on disk."""
    root = root or bench_root()
    manifest = root / "manifest.json"
    if not manifest.exists():
        raise SystemExit(
            f"no manifest at {manifest} -- run `kinescore pull --what bench`")
    doc = json.loads(manifest.read_text())
    items = []
    for raw in doc.get("items", []):
        clip = root / "clips" / f"{raw['id']}.mp4"
        if not clip.exists():
            continue
        items.append(BenchItem(
            id=str(raw["id"]), path=str(clip),
            embodiment=str(raw.get("embodiment", "")),
            view=str(raw.get("view", "")),
            model=str(raw.get("model", "")),
            split=str(raw.get("split", "")),
            method=str(raw.get("source_method", "")),
            role=str(raw.get("role", "")),
            task=str(raw.get("task", "")),
            aug_tag=str(raw.get("aug_tag") or ""),
            source_path=str(raw.get("source_path") or ""),
        ))
    return items


def select(items: list[BenchItem], criteria: dict[str, str]) -> list[BenchItem]:
    """Items matching every key of ``criteria``."""
    return [i for i in items
            if all(getattr(i, k, None) == v for k, v in criteria.items())]
