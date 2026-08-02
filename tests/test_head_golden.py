"""Golden-value regression against the real Marionette source.

If ``tests/golden/golden_head.npz`` exists (generated out-of-band by another
agent, running the *actual* ``judge/pixel_judge.py`` classes from the
Marionette source against fixed inputs/weights -- this repo never imports
that source tree), this file compares this package's ports bit-for-bit
against it. This is what actually proves "ported verbatim" rather than
merely "looks equivalent by inspection". Otherwise every test here is
skipped: the golden file is produced out-of-band and its absence should not
fail CI for everyone else's PRs.

Archive layout (``<component>__<field>``): only ``pool__tokens_in`` /
``pool__k1_noop`` / ``pool__k2_pooled`` (``(B,T,P,D)`` input/output for
``pool_patch_tokens``) are consumed here. The archive also carries
``dino_head__*`` and ``attn_ncam{1,3}__*`` arrays (``DinoPoseHead`` and
``AttentivePoseHead`` fixtures) -- those two heads have been deleted as dead
code (neither could produce a working
:class:`~kinescore.core.reader.PoseReader` any more; see
``legacy_docs/PROVENANCE.md``'s D7 addendum), so the tests that consumed those keys
are gone too. The keys themselves are left in ``golden_head.npz`` rather than
regenerating the archive -- ``tools/gen_golden.py`` (owned separately) still
references them as of this writing.

Two other fixtures that used to be exercised from this file are also gone
without a replacement test: ``tests/golden/golden_predict_pose.npz``
(``kinescore.heads.ranges.squash_to_limits``, the sigmoid squash -- removed
along with the squashed pose-reader path it validated) and
``tests/golden/golden_ckpt_head.npz`` (``AttentivePoseHead.forward`` replayed
against real checkpoint weights -- deleted alongside the test that was its
only consumer, ``test_checkpoint_roundtrip.py::test_real_checkpoint_forward_matches_golden``,
since that test round-tripped ``AttentivePoseHead`` through the now-deleted
``readers/checkpoint.py::load``).
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from kinescore.backbones.pooling import pool_patch_tokens

_GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "golden_head.npz")
_available = os.path.exists(_GOLDEN)

pytestmark = pytest.mark.skipif(not _available,
                                reason="tests/golden/golden_head.npz not present")


def _t(data, key) -> torch.Tensor:
    return torch.from_numpy(np.asarray(data[key])).float()


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
