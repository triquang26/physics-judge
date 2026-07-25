"""Multiview token-count guard (defect D4).

The source's only guard was ``N % n_cams != 0``, and it lived *inside* the
``if n_cams > 1`` branch of ``AttentivePoseHead.forward`` -- so a checkpoint
trained with ``n_cams=1`` had *no* guard at all, and would silently accept a
147-token 3-camera feature grid (``147 % 1 == 0``) for its entire training
run. These tests exercise both ``ViewEmbedding`` (the new, standalone,
unconditional guard -- see ``heads/views.py``) and
``AttentivePoseHead`` itself (which gets the same fix a different way, via an
internal ``ViewLayout`` asserted before the ``n_cams > 1`` branch -- see
``heads/attentive.py``), in both directions: a too-few-tokens and a
too-many-tokens mismatch.
"""
from __future__ import annotations

import pytest
import torch

from kinescore.core.clip import ViewLayout
from kinescore.heads.attentive import AttentivePoseHead
from kinescore.heads.views import ViewEmbedding

D = 16  # tiny embed width, CPU-fast


def test_view_embedding_3view_head_fed_wrong_token_count_raises():
    # 3 views x 7 tokens/view expected = 21; feed 7 (as if only one view's
    # worth of tokens arrived) -- must raise, not silently broadcast.
    ve = ViewEmbedding(in_dim=D, view_layout=ViewLayout(n_views=3, tokens_per_view=7))
    tokens = torch.randn(2, 4, 7, D)  # (B,T,N=7,D)
    with pytest.raises(ValueError):
        ve(tokens)


def test_view_embedding_1view_head_fed_147_tokens_raises():
    # This is the exact source bug: a 1-camera embedding (tokens_per_view=49)
    # fed a 3-camera-worth grid (147 = 3*49). 147 % 1 == 0, so the source's
    # weak divisibility check would have accepted this silently.
    ve = ViewEmbedding(in_dim=D, view_layout=ViewLayout(n_views=1, tokens_per_view=49))
    tokens = torch.randn(1, 2, 147, D)
    with pytest.raises(ValueError):
        ve(tokens)


def test_view_embedding_v1_is_bit_identical_to_no_embedding():
    ve = ViewEmbedding(in_dim=D, view_layout=ViewLayout(n_views=1, tokens_per_view=13))
    tokens = torch.randn(3, 5, 13, D)
    out = ve(tokens)
    assert torch.equal(out, tokens), (
        "a zero-init, single-view ViewEmbedding must add exactly zero bias")


def test_view_embedding_multiview_bias_is_zero_only_at_init():
    ve = ViewEmbedding(in_dim=D, view_layout=ViewLayout(n_views=2, tokens_per_view=5))
    tokens = torch.randn(1, 1, 10, D)
    out_before = ve(tokens)
    assert torch.equal(out_before, tokens), "zero-init cam_emb must be a no-op"
    with torch.no_grad():
        ve.cam_emb.add_(1.0)
    out_after = ve(tokens)
    assert not torch.equal(out_after, tokens)


def test_attentive_head_3view_fed_49_tokens_raises():
    head = AttentivePoseHead(in_dim=D, hidden=8, n_joints=4, n_heads=2,
                             n_cams=3, tokens_per_view=49)
    feat = torch.randn(1, 2, 49, D)  # only one view's worth, expected 147
    with pytest.raises(ValueError):
        head(feat)


def test_attentive_head_1view_fed_147_tokens_raises():
    head = AttentivePoseHead(in_dim=D, hidden=8, n_joints=4, n_heads=2,
                             n_cams=1, tokens_per_view=49)
    feat = torch.randn(1, 2, 147, D)  # a 3-camera grid fed to a 1-cam head
    with pytest.raises(ValueError):
        head(feat)


def test_attentive_head_v1_output_bit_identical_with_and_without_zero_cam_emb():
    """A single-view head has NO cam_emb parameter at all (source-verbatim
    conditional creation, for checkpoint strict-load compat). Its output for
    n_cams=1 must therefore already equal what an explicit zero-bias would
    produce -- i.e. adding a zero ViewEmbedding changes nothing."""
    torch.manual_seed(0)
    head = AttentivePoseHead(in_dim=D, hidden=8, n_joints=4, n_heads=2, n_cams=1)
    head.eval()  # dropout must be off for a deterministic bit-identity check
    assert not hasattr(head, "cam_emb")
    feat = torch.randn(2, 3, 11, D)
    out_direct = head(feat)

    ve = ViewEmbedding(in_dim=D, view_layout=ViewLayout(n_views=1, tokens_per_view=11))
    out_via_embedding = head(ve(feat))
    assert torch.equal(out_direct, out_via_embedding)


def test_attentive_head_multicam_creates_cam_emb_matching_checkpoint_shape():
    head = AttentivePoseHead(in_dim=D, hidden=8, n_joints=4, n_heads=2, n_cams=3)
    assert hasattr(head, "cam_emb")
    assert tuple(head.cam_emb.shape) == (3, D)
    assert torch.equal(head.cam_emb, torch.zeros(3, D))  # zero-init


def test_view_layout_still_permissive_when_tokens_per_view_unknown():
    """Without tokens_per_view the guard falls back to the weaker (but still
    correct-for-what-it-knows) divisibility check -- documents the residual
    gap for legacy checkpoints that never persisted a token count."""
    layout = ViewLayout(n_views=1)  # tokens_per_view=None
    layout.assert_tokens(147)  # does not raise: 147 % 1 == 0
