"""kinescore.bench.cell: the Cell identity -- robot primary, embodiment derived."""
from __future__ import annotations

import pytest

from kinescore.bench.cell import PATH_AXIS_ORDER, Cell
from kinescore.core.clip import ViewLayout


def _cell(**overrides) -> Cell:
    kwargs = {"cache": "dense", "robot": "franka_panda", "view": "multiview",
             "generator": "ctrlworld", "horizon": "makovian", "embodiment": "single_arm",
             "view_layout": ViewLayout(n_views=1)}
    kwargs.update(overrides)
    return Cell(**kwargs)


class TestIdentity:
    def test_robot_is_the_primary_key_embodiment_is_just_carried(self):
        cell = _cell()
        assert cell.robot == "franka_panda"
        assert cell.embodiment == "single_arm"

    def test_frozen(self):
        cell = _cell()
        with pytest.raises(Exception):  # noqa: B017 -- dataclasses.FrozenInstanceError
            cell.robot = "other"  # type: ignore[misc]

    def test_iter_defaults_to_none(self):
        cell = _cell()
        assert cell.iter is None

    def test_iter_may_be_set(self):
        cell = _cell(iter="iter_000050000")
        assert cell.iter == "iter_000050000"


class TestCellId:
    def test_cell_id_follows_path_axis_order(self):
        cell = _cell()
        assert PATH_AXIS_ORDER == ("cache", "robot", "view", "generator", "horizon")
        assert cell.cell_id == "dense/franka_panda/multiview/ctrlworld/makovian"

    def test_cell_id_never_includes_iter_or_embodiment(self):
        # cell_id is the CANONICAL layout's directory identity -- neither
        # `iter` (auto-picked per kinescore.bench.layout.RawHFLayout, not a
        # canonical path segment) nor `embodiment` (derived, not primary)
        # belongs in it.
        cell = _cell(iter="iter_000050000")
        assert "iter_000050000" not in cell.cell_id
        assert "single_arm" not in cell.cell_id


class TestFamily:
    def test_family_encodes_every_axis_plus_embodiment(self):
        cell = _cell()
        parsed = dict(kv.split("=") for kv in cell.family.split("|"))
        assert parsed == {
            "cache": "dense", "robot": "franka_panda", "view": "multiview",
            "generator": "ctrlworld", "horizon": "makovian", "embodiment": "single_arm",
        }

    def test_two_cells_differing_only_in_embodiment_have_different_families(self):
        # Even though embodiment is derived, two Cells built by hand with
        # the same primary axes but different embodiments must not collide
        # in `family` -- this is what makes a raw-layout Cell (embodiment
        # from disk) traceable back to its source directory.
        a = _cell(embodiment="single_arm")
        b = _cell(embodiment="humanoid")
        assert a.family != b.family
