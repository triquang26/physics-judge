"""CLI wiring: ``--reader`` auto-routes to the right reader family.

``kinescore.cli._scoring.build_scorer`` used to raise ``NotImplementedError``
for anything that wasn't an :class:`~kinescore.heads.attentive.AttentivePoseHead`
checkpoint (see the removed docstring this replaced). These tests exercise
``build_scorer`` itself -- not just the lower-level
``kinescore.readers.checkpoint.load_reader`` it now delegates to (that is
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
from kinescore.heads.attentive import AttentivePoseHead
from kinescore.heads.heteroscedastic import ReadoutV2Head
from kinescore.readers import checkpoint as ckpt_mod
from kinescore.readers import checkpoint_v2
from kinescore.readers.squashed import SquashedPoseReader


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


def test_build_scorer_still_routes_attentive_checkpoint(tmp_path, _requires_franka_urdf):
    """Regression: the pre-existing squashed/Franka path must be unaffected
    by adding the ReadoutV2 branch -- same reader type, same limit_semantics
    as before this change."""
    from kinescore.cli._scoring import build_scorer

    head = AttentivePoseHead(in_dim=32, hidden=16, n_joints=7, n_heads=2, n_cams=1)
    head.eval()
    path = str(tmp_path / "attentive7.pt")
    layout = ViewLayout(n_views=1)
    ckpt_mod.save(path, head, view_layout=layout, robot_name="franka_panda",
                 limit_semantics="squashed", backbone_cfg={"embed_dim": 32})

    args = _args(reader=path)
    scorer = build_scorer(args, layout)

    assert isinstance(scorer.reader, SquashedPoseReader)
    assert scorer.reader.limit_semantics == "squashed"


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
