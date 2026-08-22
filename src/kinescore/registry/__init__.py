"""What a run is: which packing, which robot, which corpus, which head."""
from kinescore.registry.cells import (
    CellSpec,
    ReaderSpec,
    Registry,
    TrainSource,
    load_registry,
)
from kinescore.registry.materialize import materialize_train_tree
from kinescore.registry.provenance import run_manifest, write_run_manifest
from kinescore.registry.views import ViewSpec, load_views

__all__ = [
    "Registry", "CellSpec", "ReaderSpec", "TrainSource", "ViewSpec",
    "load_registry", "load_views", "materialize_train_tree", "run_manifest",
    "write_run_manifest",
]
