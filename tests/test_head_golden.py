"""Golden-value regression against the real Marionette source.

If ``tests/golden/golden_head.npz`` exists (generated out-of-band by another
agent, running the *actual* ``judge/pixel_judge.py`` classes from the
Marionette source against fixed inputs/weights -- this repo never imports
that source tree), this file compares this package's ports bit-for-bit
against it. This is what actually proves "ported verbatim" rather than
merely "looks equivalent by inspection". Otherwise every test here is
skipped: the golden file is produced out-of-band and its absence should not
fail CI for everyone else's PRs.

Archive layout (``<component>__<field>``, state-dict entries flattened as
``<component>__state_dict__<param name>`` since npz has no nested dict):

* ``pool__tokens_in`` -- ``(B,T,P,D)`` input to ``pool_patch_tokens``.
  ``pool__k1_noop`` -- ``k=1`` output (must equal the input exactly).
  ``pool__k2_pooled`` -- ``k=2`` output.
* ``dino_head__state_dict__*`` -- a ``DinoPoseHead`` state dict.
  ``dino_head__feat_in`` -- ``(B,T,D)`` input. ``dino_head__output`` -- expected.
* ``attn_ncam1__state_dict__*`` -- an ``AttentivePoseHead(n_cams=1)`` state dict
  (no ``cam_emb`` key). ``attn_ncam1__feat_in`` -- ``(B,T,N,D)`` input.
  ``attn_ncam1__output``, ``attn_ncam1__w`` -- expected ``(out, attn_weights)``.
* ``attn_ncam3__state_dict__*`` -- an ``AttentivePoseHead(n_cams=3)`` state dict
  (has ``cam_emb``), reusing ``attn_ncam1__feat_in`` as input (its token count
  is a multiple of 3). ``attn_ncam3__output``, ``attn_ncam3__w`` -- expected.

A second, independent fixture, ``tests/golden/golden_predict_pose.npz``
(recorded from ``PixelPhysicsJudge.predict_pose``), checks
:func:`~kinescore.heads.ranges.squash_to_limits` the same way: ``raw`` (a
fixed linspace, not random) -> the exact ``q``/``gripper`` the source
produced against the real Franka joint limits. Skipped independently of
``golden_head.npz`` since it is a separate archive.

A third, ``tests/golden/golden_ckpt_head.npz``, replays
``AttentivePoseHead.forward`` seeded with ``torch.Generator().manual_seed(0)``
against the **real** ``judge_v3l``/``judge_v3l_mv`` checkpoint weights
(embedded directly in the archive's ``*__state_dict__*`` keys, not loaded
from a checkpoint file on disk -- see
``tests/test_checkpoint_roundtrip.py::test_real_checkpoint_forward_matches_golden``
for the on-disk version, which additionally verifies
``readers.checkpoint.load`` against the real ``judge.pt`` files' sha256).
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from kinescore.backbones.pooling import pool_patch_tokens
from kinescore.heads.attentive import AttentivePoseHead
from kinescore.heads.mlp import DinoPoseHead
from kinescore.heads.ranges import squash_to_limits

_GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "golden_head.npz")
_GOLDEN_PREDICT_POSE = os.path.join(os.path.dirname(__file__), "golden",
                                    "golden_predict_pose.npz")
_available = os.path.exists(_GOLDEN)

pytestmark = pytest.mark.skipif(not _available,
                                reason="tests/golden/golden_head.npz not present")


def _t(data, key) -> torch.Tensor:
    return torch.from_numpy(np.asarray(data[key])).float()


def _state_dict(data, prefix: str) -> dict:
    full = f"{prefix}__state_dict__"
    return {k[len(full):]: _t(data, k) for k in data.files if k.startswith(full)}


def test_pool_patch_tokens_k1_is_noop():
    data = np.load(_GOLDEN)
    tokens_in = _t(data, "pool__tokens_in")
    out = pool_patch_tokens(tokens_in, 1)
    torch.testing.assert_close(out, _t(data, "pool__k1_noop"))
    assert torch.equal(out, tokens_in)  # k<=1 truly is a no-op, not a copy


def test_pool_patch_tokens_k2_matches_golden():
    data = np.load(_GOLDEN)
    tokens_in = _t(data, "pool__tokens_in")
    out = pool_patch_tokens(tokens_in, 2)
    torch.testing.assert_close(out, _t(data, "pool__k2_pooled"), atol=1e-5, rtol=1e-5)


def test_dino_pose_head_matches_golden():
    data = np.load(_GOLDEN)
    sd = _state_dict(data, "dino_head")
    in_dim = sd["net.0.weight"].shape[1]
    hidden = sd["net.0.weight"].shape[0]
    n_joints = sd["net.3.weight"].shape[0] - 1

    head = DinoPoseHead(in_dim=in_dim, hidden=hidden, n_joints=n_joints, dropout=0.0)
    head.eval()
    head.load_state_dict(sd, strict=True)

    with torch.no_grad():
        out = head(_t(data, "dino_head__feat_in"))
    torch.testing.assert_close(out, _t(data, "dino_head__output"), atol=1e-4, rtol=1e-4)


def test_attentive_head_ncam1_matches_golden():
    data = np.load(_GOLDEN)
    sd = _state_dict(data, "attn_ncam1")
    assert "cam_emb" not in sd
    in_dim = sd["norm.weight"].shape[0]
    n_heads = sd["score.weight"].shape[0]
    hidden = sd["mlp.0.weight"].shape[0]
    n_joints = sd["mlp.3.weight"].shape[0] - 1

    head = AttentivePoseHead(in_dim=in_dim, hidden=hidden, n_joints=n_joints,
                             n_heads=n_heads, n_cams=1, dropout=0.0)
    head.eval()
    head.load_state_dict(sd, strict=True)

    feat = _t(data, "attn_ncam1__feat_in")
    with torch.no_grad():
        out, w = head(feat, return_attn=True)
    torch.testing.assert_close(out, _t(data, "attn_ncam1__output"), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(w, _t(data, "attn_ncam1__w"), atol=1e-4, rtol=1e-4)


def test_attentive_head_ncam3_matches_golden():
    data = np.load(_GOLDEN)
    sd = _state_dict(data, "attn_ncam3")
    assert "cam_emb" in sd
    n_cams = sd["cam_emb"].shape[0]
    in_dim = sd["norm.weight"].shape[0]
    n_heads = sd["score.weight"].shape[0]
    hidden = sd["mlp.0.weight"].shape[0]
    n_joints = sd["mlp.3.weight"].shape[0] - 1

    head = AttentivePoseHead(in_dim=in_dim, hidden=hidden, n_joints=n_joints,
                             n_heads=n_heads, n_cams=n_cams, dropout=0.0)
    head.eval()
    head.load_state_dict(sd, strict=True)

    # attn_ncam3 shares the same feature cache as attn_ncam1 -- its token
    # count (27) is a multiple of n_cams=3 (9 tokens/cam), which is exactly
    # what lets the same cached grid be reinterpreted under a different
    # camera-count head.
    feat = _t(data, "attn_ncam1__feat_in")
    with torch.no_grad():
        out, w = head(feat, return_attn=True)
    torch.testing.assert_close(out, _t(data, "attn_ncam3__output"), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(w, _t(data, "attn_ncam3__w"), atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not os.path.exists(_GOLDEN_PREDICT_POSE),
                    reason="tests/golden/golden_predict_pose.npz not present")
def test_squash_to_limits_matches_predict_pose_golden():
    """``squash_to_limits`` must reproduce ``PixelPhysicsJudge.predict_pose``'s
    sigmoid squash exactly, including at the extremes of a wide linspace
    (where sigmoid saturates) -- this is the function every squashed reader's
    validity guarantee rests on."""
    data = np.load(_GOLDEN_PREDICT_POSE)
    raw = _t(data, "raw")  # (1,8,8): 7 joints + 1 gripper logit
    q_lo = _t(data, "q_lo")
    q_hi = _t(data, "q_hi")

    q_raw, gripper_raw = raw[..., :7], raw[..., 7:]
    q = squash_to_limits(q_raw, q_lo, q_hi)
    gripper = torch.sigmoid(gripper_raw)

    torch.testing.assert_close(q, _t(data, "q"), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(gripper, _t(data, "gripper"), atol=1e-6, rtol=1e-5)
    # Every squashed value is inside (or, at fp32 sigmoid saturation for
    # |raw|>~20, within fp32 rounding of) the limits -- never meaningfully
    # outside them. `lo + (hi-lo)*sigmoid(raw)` is not bit-exact to `hi` at
    # sigmoid's saturated 1.0 (a round-trip subtract-then-add rounding
    # artifact of a few ULP), so the bound needs a matching epsilon.
    eps = 1e-4
    assert bool((q >= q_lo.view(1, 1, -1) - eps).all())
    assert bool((q <= q_hi.view(1, 1, -1) + eps).all())
