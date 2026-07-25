"""``readers/checkpoint.py`` save -> load round-trip.

Saves a head with a non-default shape in every shape-bearing dimension
(``hidden=256`` != the 512 default, ``dropout=0.3`` != the 0.1 default,
``n_heads=2`` != the 4 default, ``n_views=3`` multiview), reloads it, and
checks both the raw state dict and the actual forward output are identical --
not just "loads without error". This is the D5 regression test: the source
would have hardcoded ``hidden=512`` on load and either produced a shape
mismatch (state_dict load failure) or, worse, silently loaded the wrong
shape via a lenient loader.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from kinescore.core.clip import ViewLayout
from kinescore.heads.attentive import AttentivePoseHead
from kinescore.readers import checkpoint

IN_DIM = 24

# Real production checkpoints this package targets compatibility with (see
# the task's Sources section). Located via $KINESCORE_CIASC_ROOT (a
# Marionette-ciasc checkout containing model_ckpt/*/judge.pt), the same env
# var conftest.py's `ckpt` marker probe uses -- so this file never hardcodes
# anyone's absolute path, and the test is auto-skipped by
# `pytest_collection_modifyitems` (with an actionable message) whenever the
# marker probe already determined no checkpoint is available.
_CKPT_ROOT_ENV = "KINESCORE_CIASC_ROOT"
_REAL_CKPTS = {
    "judge_v3l": dict(n_cams=1, hidden=512, n_heads=4, embed_dim=1024, n_joints=7),
    "judge_v3l_mv": dict(n_cams=3, hidden=512, n_heads=4, embed_dim=1024, n_joints=7),
    "judge_reward": dict(n_cams=1, hidden=512, n_heads=4, embed_dim=1024, n_joints=7),
}


def _make_head() -> AttentivePoseHead:
    torch.manual_seed(0)
    head = AttentivePoseHead(in_dim=IN_DIM, hidden=256, n_joints=6, dropout=0.3,
                             n_heads=2, n_cams=3)
    head.eval()
    return head


def test_roundtrip_state_dict_identical(tmp_path):
    head = _make_head()
    path = str(tmp_path / "head.pt")
    view_layout = ViewLayout(n_views=3, order=("ext1", "ext2", "wrist"),
                             tokens_per_view=16)
    checkpoint.save(path, head, view_layout=view_layout, robot_name="franka_panda",
                    limit_semantics="squashed", meta={"note": "roundtrip test"})

    loaded = checkpoint.load(path)
    sd1, sd2 = head.state_dict(), loaded.head.state_dict()
    assert set(sd1.keys()) == set(sd2.keys())
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), f"state_dict mismatch at {k!r}"


def test_roundtrip_forward_output_identical(tmp_path):
    head = _make_head()
    path = str(tmp_path / "head.pt")
    view_layout = ViewLayout(n_views=3, tokens_per_view=16)
    checkpoint.save(path, head, view_layout=view_layout, robot_name="franka_panda",
                    limit_semantics="squashed")
    loaded = checkpoint.load(path)

    feat = torch.randn(2, 4, 3 * 16, IN_DIM)  # (B,T,N=n_cams*P,D)
    with torch.no_grad():
        out1 = head(feat)
        out2 = loaded.head(feat)
    assert torch.equal(out1, out2)


def test_roundtrip_cfg_shape_fields(tmp_path):
    head = _make_head()
    path = str(tmp_path / "head.pt")
    view_layout = ViewLayout(n_views=3, tokens_per_view=16)
    checkpoint.save(path, head, view_layout=view_layout, robot_name="franka_panda",
                    limit_semantics="squashed")
    loaded = checkpoint.load(path)

    assert loaded.hidden == 256
    assert loaded.n_heads == 2
    assert loaded.n_cams == 3
    assert loaded.embed_dim == IN_DIM
    assert loaded.n_joints == 6
    assert abs(loaded.dropout - 0.3) < 1e-8
    assert loaded.robot_name == "franka_panda"
    assert loaded.limit_semantics == "squashed"
    assert loaded.view_layout_key == view_layout.key


def test_roundtrip_cross_check_raises_on_corrupted_cfg(tmp_path):
    """A hand-edited cfg that disagrees with the weights must fail loudly
    (naming both the declared and inferred values), not silently construct a
    head with the wrong shape."""
    head = _make_head()
    path = str(tmp_path / "head.pt")
    view_layout = ViewLayout(n_views=3, tokens_per_view=16)
    checkpoint.save(path, head, view_layout=view_layout, robot_name="franka_panda",
                    limit_semantics="squashed")

    ck = torch.load(path, map_location="cpu")
    ck["cfg"]["hidden"] = 999  # disagrees with mlp.0.bias.shape[0]==256
    torch.save(ck, path)

    try:
        checkpoint.load(path)
        assert False, "expected a ValueError on cfg/state_dict shape mismatch"
    except ValueError as e:
        msg = str(e)
        assert "999" in msg and "256" in msg


@pytest.mark.ckpt
@pytest.mark.parametrize("name", sorted(_REAL_CKPTS))
def test_real_production_checkpoints_load(name):
    """Loads the three real ``judge.pt`` files this package targets
    compatibility with, and checks every shape-bearing field matches what
    was verified by hand (see docstrings in ``readers/checkpoint.py`` and
    ``tests/test_checkpoint_legacy_cfg.py``). Skips if the checkpoint
    directory isn't mounted on this host."""
    root = os.environ.get(_CKPT_ROOT_ENV)
    if not root:
        pytest.skip(f"${_CKPT_ROOT_ENV} not set")
    path = os.path.join(root, "model_ckpt", name, "judge.pt")
    if not os.path.exists(path):
        pytest.skip(f"{path} not present under ${_CKPT_ROOT_ENV}")
    expected = _REAL_CKPTS[name]
    loaded = checkpoint.load(path)
    assert loaded.n_cams == expected["n_cams"]
    assert loaded.hidden == expected["hidden"]
    assert loaded.n_heads == expected["n_heads"]
    assert loaded.embed_dim == expected["embed_dim"]
    assert loaded.n_joints == expected["n_joints"]
    assert hasattr(loaded.head, "cam_emb") == (expected["n_cams"] > 1)

    # A forward pass on a correctly-shaped random input should run cleanly.
    n = expected["n_cams"] * 49  # matches real patch_pool=2 @ dino_input=224
    feat = torch.randn(1, 2, n, expected["embed_dim"])
    with torch.no_grad():
        out = loaded.head(feat)
    assert out.shape == (1, 2, expected["n_joints"] + 1)


_GOLDEN_CKPT_HEAD = os.path.join(os.path.dirname(__file__), "golden",
                                 "golden_ckpt_head.npz")


@pytest.mark.ckpt
@pytest.mark.skipif(not os.path.exists(_GOLDEN_CKPT_HEAD),
                    reason="tests/golden/golden_ckpt_head.npz not present")
@pytest.mark.parametrize("name", ["judge_v3l", "judge_v3l_mv"])
def test_real_checkpoint_forward_matches_golden(name):
    """End-to-end: ``readers.checkpoint.load`` on the real, on-disk
    ``judge.pt`` -> ``AttentivePoseHead.forward`` on a seeded input ->
    compared against ``golden_ckpt_head.npz`` (recorded straight from the
    Marionette source's ``AttentivePoseHead.forward``). Also checks the
    on-disk file's sha256 against the golden's recorded ``ckpt_sha256``, so
    a stale or swapped checkpoint on this host is caught before the numeric
    comparison would otherwise just look like a port bug."""
    import hashlib
    import json

    root = os.environ.get(_CKPT_ROOT_ENV)
    if not root:
        pytest.skip(f"${_CKPT_ROOT_ENV} not set")
    path = os.path.join(root, "model_ckpt", name, "judge.pt")
    if not os.path.exists(path):
        pytest.skip(f"{path} not present under ${_CKPT_ROOT_ENV}")

    data = np.load(_GOLDEN_CKPT_HEAD, allow_pickle=True)
    file_sha256 = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert file_sha256 == str(data[f"{name}__ckpt_sha256"]), (
        f"{path} does not match the checkpoint golden_ckpt_head.npz was "
        f"recorded from (sha256 mismatch) -- results below would not be "
        f"comparable")

    loaded = checkpoint.load(path)
    loaded.head.eval()
    cfg = json.loads(str(data[f"{name}__cfg_json"]))
    assert loaded.n_cams == cfg.get("n_cams", 1)
    assert loaded.n_heads == cfg["n_heads"]
    assert loaded.embed_dim == cfg["embed_dim"]

    shape = tuple(int(x) for x in data[f"{name}__feat_shape"])
    gen = torch.Generator().manual_seed(int(data[f"{name}__seed"]))
    feat = torch.randn(*shape, generator=gen)
    with torch.no_grad():
        out = loaded.head(feat)
    expected = torch.from_numpy(np.asarray(data[f"{name}__output"])).float()
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)
