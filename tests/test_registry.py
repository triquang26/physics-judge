"""The shipped registry loads, and every cross-reference in it holds.

``configs/{views,robots,cells}.yaml`` are the only place a packing, a robot or
a scored unit is declared. These tests read the real files -- a config edit
that breaks an id, a view or an embodiment agreement fails here, not halfway
through a training run.
"""
from __future__ import annotations

import pytest
import yaml

from kinescore.registry.cells import (
    DEFAULT_ROBOTS_PATH,
    SCENE_KEY_MODES,
    TrainSource,
    load_registry,
)
from kinescore.registry.views import DEFAULT_VIEWS_PATH, load_views


@pytest.fixture(autouse=True)
def _paths(tmp_path, monkeypatch):
    """Path variables the registry expands. Nothing here is read from disk."""
    for key in ("KINESCORE_DATA_ROOT", "KINESCORE_CACHE_DIR",
                "KINESCORE_CKPT_DIR", "KINESCORE_ASSETS"):
        monkeypatch.setenv(key, str(tmp_path / key.lower()))


@pytest.fixture
def registry():
    return load_registry()


class TestShippedConfigs:
    def test_registry_loads(self, registry):
        assert registry.readers
        assert registry.cells
        assert registry.sources

    def test_reader_ids_are_robot_dot_view(self, registry):
        for reader_id, reader in registry.readers.items():
            assert reader_id == f"{reader.robot}.{reader.view.view_id}"

    def test_cell_ids_are_embodiment_dot_view_dot_model(self, registry):
        for cell_id, cell in registry.cells.items():
            assert cell_id == f"{cell.embodiment}.{cell.view_id}.{cell.model}"

    def test_every_cell_reader_is_declared(self, registry):
        for cell in registry.cells.values():
            assert cell.reader.reader_id in registry.readers

    def test_cell_and_reader_agree_on_packing(self, registry):
        for cell in registry.cells.values():
            assert cell.view.view_id == cell.view_id

    def test_a_blocked_reader_blocks_its_cells(self, registry):
        for cell in registry.cells.values():
            if cell.reader.status:
                assert not cell.scorable

    def test_trainable_readers_declare_a_corpus(self, registry):
        for reader in registry.readers.values():
            if reader.trainable:
                assert reader.train is not None and reader.train.root

    def test_lookup_miss_lists_what_exists(self, registry):
        with pytest.raises(KeyError, match="declared cells"):
            registry.cell("no.such.cell")
        with pytest.raises(KeyError, match="declared readers"):
            registry.reader("no.such_reader")

    def test_cells_for_reader_round_trips(self, registry):
        for cell in registry.cells.values():
            assert cell in registry.cells_for_reader(cell.reader.reader_id)


class TestViews:
    def test_shipped_views_load(self):
        views = load_views(DEFAULT_VIEWS_PATH)
        assert "sv1" in views and "mv3_row" in views

    def test_panel_geometry_matches_the_declared_frame(self):
        views = load_views(DEFAULT_VIEWS_PATH)
        mv3 = views["mv3_row"]
        mv3.check_frame_size(960, 192)
        with pytest.raises(ValueError):
            mv3.check_frame_size(640, 480)

    def test_layout_exposes_the_declared_view_count(self):
        views = load_views(DEFAULT_VIEWS_PATH)
        assert views["mv3_row"].layout(4).n_views == 3
        assert views["mv4_grid_br_blank"].layout(4).n_views == 3
        assert views["mv4_grid_br_blank"].layout(4).panel_count == 4


def _load_yaml(tmp_path, cells: dict, robots: dict | None = None):
    """Load a registry from ``cells`` written to ``tmp_path``."""
    cells_path = tmp_path / "cells.yaml"
    cells_path.write_text(yaml.safe_dump(cells))
    robots_path = tmp_path / "robots.yaml"
    robots_path.write_text(yaml.safe_dump(
        robots or yaml.safe_load(DEFAULT_ROBOTS_PATH.read_text())))
    return load_registry(cells_path, robots_path, DEFAULT_VIEWS_PATH)


class TestRejections:
    def _load(self, tmp_path, cells, robots=None):
        return _load_yaml(tmp_path, cells, robots)

    def test_reader_id_must_match_robot_and_view(self, tmp_path):
        with pytest.raises(ValueError, match="must be <robot>.<view_id>"):
            self._load(tmp_path, {
                "readers": {"wrong.sv1": {"robot": "franka_panda",
                                          "view": "sv1"}},
                "cells": {},
            })

    def test_unknown_view_is_named(self, tmp_path):
        with pytest.raises(ValueError, match="views.yaml"):
            self._load(tmp_path, {
                "readers": {"franka_panda.nope": {"robot": "franka_panda",
                                                  "view": "nope"}},
                "cells": {},
            })

    def test_cell_id_needs_three_parts(self, tmp_path):
        with pytest.raises(ValueError, match="<embodiment>.<view_id>.<model>"):
            self._load(tmp_path, {
                "readers": {"franka_panda.sv1": {"robot": "franka_panda",
                                                 "view": "sv1"}},
                "cells": {"single_arm.sv1": {"reader": "franka_panda.sv1"}},
            })

    def test_cell_packing_must_match_its_reader(self, tmp_path):
        with pytest.raises(ValueError, match="but its reader"):
            self._load(tmp_path, {
                "readers": {"franka_panda.sv1": {"robot": "franka_panda",
                                                 "view": "sv1"}},
                "cells": {"single_arm.mv3_row.ctrlworld": {
                    "reader": "franka_panda.sv1"}},
            })

    def test_cell_embodiment_must_match_its_robot(self, tmp_path):
        with pytest.raises(ValueError, match="is declared"):
            self._load(tmp_path, {
                "readers": {"franka_panda.sv1": {"robot": "franka_panda",
                                                 "view": "sv1"}},
                "cells": {"bimanual.sv1.dreamgen": {
                    "reader": "franka_panda.sv1"}},
            })

    def test_unknown_key_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unknown key"):
            self._load(tmp_path, {
                "readers": {"franka_panda.sv1": {"robot": "franka_panda",
                                                 "view": "sv1",
                                                 "epochs": 3}},
                "cells": {},
            })


class TestSceneKeyDeclaration:
    """``scene_key`` selects how the train/val split groups episodes."""

    def test_default_is_the_id_prefix(self):
        assert TrainSource(adapter="canonical", root="/x").scene_key == "prefix"

    def test_only_declared_modes_are_accepted(self):
        with pytest.raises(ValueError, match="scene_key must be one of"):
            TrainSource(adapter="canonical", root="/x", scene_key="task")

    def test_modes_are_prefix_and_episode(self):
        assert SCENE_KEY_MODES == {"prefix", "episode"}

    def test_yaml_declares_it(self, tmp_path):
        registry = _load_yaml(tmp_path, {
            "readers": {"franka_panda.sv1": {
                "robot": "franka_panda", "view": "sv1",
                "train": {"adapter": "canonical", "root": "/x",
                          "scene_key": "episode"}}},
            "cells": {},
        })

        assert registry.readers["franka_panda.sv1"].train.scene_key == "episode"

    def test_yaml_rejects_an_undeclared_mode(self, tmp_path):
        with pytest.raises(ValueError, match="scene_key must be one of"):
            _load_yaml(tmp_path, {
                "readers": {"franka_panda.sv1": {
                    "robot": "franka_panda", "view": "sv1",
                    "train": {"adapter": "canonical", "root": "/x",
                              "scene_key": "task"}}},
                "cells": {},
            })
