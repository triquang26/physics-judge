"""Multiview token-count guard (defect D4).

The source's only guard was ``N % n_cams != 0``, and it lived *inside* the
``if n_cams > 1`` branch of the (since-removed) ``AttentivePoseHead.forward``
-- so a checkpoint trained with ``n_cams=1`` had *no* guard at all, and would
silently accept a 147-token 3-camera feature grid (``147 % 1 == 0``) for its
entire training run. ``ViewEmbedding`` (see ``heads/views.py``) is the
standalone, unconditional fix: the bias is always constructed (zero-init) and
validated against a full ``ViewLayout``, in both directions -- a too-few-
tokens and a too-many-tokens mismatch. ``ViewEmbedding`` is what
``readers/heteroscedastic.py`` composes in front of
:class:`~kinescore.heads.heteroscedastic.ReadoutV2Head`, the one live head
family, to gain multiview support (that head has none natively).
"""
from __future__ import annotations

import pytest
import torch

from kinescore.core.clip import ViewLayout
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


def test_view_layout_still_permissive_when_tokens_per_view_unknown():
    """Without tokens_per_view the guard falls back to the weaker (but still
    correct-for-what-it-knows) divisibility check -- documents the residual
    gap for legacy checkpoints that never persisted a token count."""
    layout = ViewLayout(n_views=1)  # tokens_per_view=None
    layout.assert_tokens(147)  # does not raise: 147 % 1 == 0
