"""CLI wiring: ``--reader`` auto-routes to the right reader family.

``kinescore.cli._scoring.build_scorer`` used to raise ``NotImplementedError``
for anything that wasn't a legacy ``AttentivePoseHead``-format checkpoint --
that head class has since been deleted as dead code (its one reader,
``SquashedPoseReader``, was removed; see ``legacy_docs/PROVENANCE.md``'s D7
addendum), so it is now the OTHER way around: only a ``ReadoutV2Head`` cfg
routes to a working reader, everything else raises. These tests exercise
``build_scorer`` itself -- not just the lower-level
``kinescore.readers.loader.load_reader`` it now delegates to (that is
covered file-by-file in ``tests/test_checkpoint_v2.py``) -- with a real
``--robot franka_panda`` (needs the cached Panda URDF the rest of this test
suite already depends on; see ``tests/conftest.py::panda_urdf_path``) so this
is as close to the real CLI path as a network-free test can get:
``FeatureBackbone`` construction never touches the network (weights load
lazily on first ``.encode()``, which none of these tests call -- see
``backbones/dino.py``'s docstring), only routing is under test.
"""
from __future__ import annotations

import argparse

import pytest
import torch

from kinescore.core.clip import ViewLayout
from kinescore.heads.heteroscedastic import ReadoutV2Head
from kinescore.readers import checkpoint_v2


def _args(**kw) -> argparse.Namespace:
    defaults = {"robot": "franka_panda", "suite": "invariant_v1", "device": "cpu"}
    defaults.update(kw)
    return argparse.Namespace(**defaults)


@pytest.fixture
def _requires_franka_urdf(panda_urdf_path):
    # panda_urdf_path is a SESSION fixture that skips (with an actionable
    # message) when no cached Panda URDF / network is available -- see
    # tests/conftest.py. Depending on it here means this file degrades the
    # same way the rest of the FK-dependent suite does, instead of a
    # confusing robot_descriptions error.
    return panda_urdf_path


def test_build_scorer_routes_readout_v2_checkpoint(tmp_path, _requires_franka_urdf):
    from kinescore.cli._scoring import build_scorer

    torch.manual_seed(0)
    head = ReadoutV2Head(in_dim=32, d_model=16, n_heads=2, temporal_nhead=2,
                         ff=16, n_temporal_layers=1, t_max=8, n_out=7)
    head.eval()
    path = str(tmp_path / "rv2_franka7.pt")
    layout = ViewLayout(n_views=1)
    checkpoint_v2.save(path, head, view_layout=layout, robot_name="franka_panda")

    args = _args(reader=path)
    scorer = build_scorer(args, layout)

    assert isinstance(scorer.reader, checkpoint_v2.ReadoutV2PoseReader)
    assert scorer.reader.limit_semantics == "raw_rad"
    assert scorer.reader.robot_name == "franka_panda"
    assert scorer.robot.name == "franka_panda"


def test_build_scorer_raises_for_legacy_attentive_checkpoint(tmp_path, _requires_franka_urdf):
    """The squashed pose-reader path was removed (see legacy_docs/PROVENANCE.md's D7
    addendum) -- a legacy AttentivePoseHead-format checkpoint no longer has a
    reader to route to, so `build_scorer` must fail loudly rather than build
    a SquashedPoseReader (or anything else) for it.

    Only the checkpoint's cfg SHAPE matters for routing (checked before any
    model construction) -- so this writes a legacy-format ``{"head", "cfg",
    "meta"}`` file directly rather than via the now-deleted
    ``AttentivePoseHead``/``readers/checkpoint.py::save``."""
    from kinescore.cli._scoring import build_scorer

    layout = ViewLayout(n_views=1)
    path = str(tmp_path / "attentive7.pt")
    cfg = {"hidden": 16, "n_heads": 2, "n_cams": 1, "embed_dim": 32,
          "dropout": 0.1, "robot_name": "franka_panda",
          "view_layout_key": layout.key, "limit_semantics": "squashed",
          "backbone": {"embed_dim": 32}}
    torch.save({"head": {}, "cfg": cfg, "meta": {}}, path)

    args = _args(reader=path)
    with pytest.raises(NotImplementedError, match="AttentivePoseHead"):
        build_scorer(args, layout)


def test_build_scorer_raises_on_joint_count_mismatch_for_readout_v2(
        tmp_path, _requires_franka_urdf):
    from kinescore.cli._scoring import build_scorer

    torch.manual_seed(0)
    head = ReadoutV2Head(in_dim=32, d_model=16, n_heads=2, temporal_nhead=2,
                         ff=16, n_temporal_layers=1, t_max=8, n_out=3)
    head.eval()
    path = str(tmp_path / "rv2_too_small.pt")
    layout = ViewLayout(n_views=1)
    checkpoint_v2.save(path, head, view_layout=layout, robot_name="franka_panda")

    args = _args(reader=path)
    with pytest.raises(ValueError, match="only predicts"):
        build_scorer(args, layout)
