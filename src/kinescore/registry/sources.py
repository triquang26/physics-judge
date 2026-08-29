"""``configs/sources.yaml`` -> what to download, and what was downloaded."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kinescore.paths import env_path

__all__ = [
    "SourceSpec", "load_sources", "DEFAULT_SOURCES_PATH", "REVISIONS_FILE",
    "read_revisions", "record_revision",
]

DEFAULT_SOURCES_PATH = Path(__file__).resolve().parents[3] / "configs" / "sources.yaml"
REVISIONS_FILE = "REVISIONS.json"

_KEYS = {"repo", "include", "dest", "strip_prefix", "repo_type"}


@dataclass(frozen=True)
class SourceSpec:
    """One remote repository and the subset of it this benchmark uses."""

    name: str
    repo: str
    include: tuple[str, ...]
    dest: str
    strip_prefix: str | None = None
    repo_type: str = "dataset"

    @property
    def local_dir(self) -> Path:
        return env_path("KINESCORE_DATA_ROOT") / self.dest


def _from_entry(name: str, entry: dict[str, Any]) -> SourceSpec:
    unknown = set(entry) - _KEYS
    if unknown:
        raise ValueError(f"source {name!r}: unknown key(s) {sorted(unknown)}")
    include = entry.get("include") or []
    if not include:
        raise ValueError(
            f"source {name!r} lists no `include` globs; an empty whitelist "
            f"would fetch the whole repository")
    return SourceSpec(
        name=name,
        repo=str(entry["repo"]),
        include=tuple(str(i) for i in include),
        dest=str(entry["dest"]),
        strip_prefix=(None if entry.get("strip_prefix") is None
                      else str(entry["strip_prefix"])),
        repo_type=str(entry.get("repo_type", "dataset")),
    )


def load_sources(path: str | Path = DEFAULT_SOURCES_PATH) -> dict[str, SourceSpec]:
    """Every declared source, keyed by name."""
    doc = yaml.safe_load(Path(path).read_text()) or {}
    entries = doc.get("sources") or {}
    if not isinstance(entries, dict):
        raise ValueError(f"{path}: `sources` must be a mapping")
    return {name: _from_entry(name, entry) for name, entry in entries.items()}


def _revisions_path() -> Path:
    return env_path("KINESCORE_DATA_ROOT") / REVISIONS_FILE


def read_revisions() -> dict[str, Any]:
    """What is on disk, per source."""
    path = _revisions_path()
    return json.loads(path.read_text()) if path.exists() else {}


def record_revision(source: SourceSpec, revision: str, *, pulled_at: str,
                    n_files: int) -> None:
    """Record one completed pull, leaving other sources' entries alone."""
    path = _revisions_path()
    doc = read_revisions()
    doc[source.name] = {
        "repo": source.repo,
        "repo_type": source.repo_type,
        "revision": revision,
        "include": list(source.include),
        "dest": source.dest,
        "n_files": n_files,
        "pulled_at": pulled_at,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
