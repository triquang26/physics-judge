"""``readers/checkpoint_v2.py`` against the REAL production checkpoint.

Loads the actual ``Marionette-fkjepa/model_ckpt/readout_v2_gr1.pt`` through
``kinescore.readers.checkpoint_v2.load_head`` and independently through the
*source* ``models.posendf.readout_v2.ReadoutV2Head.load`` (imported live from
a Marionette-fkjepa checkout via ``sys.path``, mirroring
``tools/gen_golden.py``'s approach -- see its module docstring), then checks
both reproduce the same ``mu``/``sigma`` on a fixed random feature input.
This is the "strict-load + numeric parity" verification the checkpoint
loader is judged against: ``load_state_dict(strict=True)`` succeeding proves
the shapes match; this test proves the *numbers* match too, including the
``sigma_scale`` post-hoc calibration (``ReadoutV2Scorer._resolve_sigma_scale``)
that a bare ``load_state_dict`` would silently skip.

Gated behind ``@pytest.mark.ckpt`` (skipped by default -- see
``pyproject.toml``'s ``addopts``) and additionally skipped unless
``KINESCORE_FKJEPA_ROOT`` points at a Marionette-fkjepa checkout containing
``model_ckpt/readout_v2_gr1.pt`` -- deliberately a *different* env var from
``tests/conftest.py``'s ``KINESCORE_CIASC_ROOT`` (the Franka/AttentivePoseHead
checkpoints live in a Marionette-ciasc checkout; this is a different source
repo entirely, see ``legacy_docs/PROVENANCE.md``'s A/B split), so this file never
hardcodes anyone's absolute path.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

from kinescore.readers import checkpoint_v2

_FKJEPA_ROOT_ENV = "KINESCORE_FKJEPA_ROOT"


def _resolve_ckpt_path() -> str:
    root = os.environ.get(_FKJEPA_ROOT_ENV)
    if not root:
        pytest.skip(f"${_FKJEPA_ROOT_ENV} not set (Marionette-fkjepa checkout)")
    path = os.path.join(root, "model_ckpt", "readout_v2_gr1.pt")
    if not os.path.exists(path):
        pytest.skip(f"{path} not present under ${_FKJEPA_ROOT_ENV}")
    return path


@pytest.fixture
def source_readoutv2head_cls():
    """Import the SOURCE ``ReadoutV2Head`` live, from ``$KINESCORE_FKJEPA_ROOT``.

    ``models.posendf.readout_v2`` only needs torch/numpy at import time (its
    package ``__init__`` chain does not pull in ``pytorch_kinematics`` or
    ``transformers`` -- verified by reading ``models/posendf/{__init__,
    robot_spec}.py``), so this stays a cheap, offline import. ``sys.path``/
    ``sys.modules`` are restored afterward so this test cannot leak a stale
    ``models`` package into any test that runs after it.
    """
    root = os.environ.get(_FKJEPA_ROOT_ENV)
    if not root:
        pytest.skip(f"${_FKJEPA_ROOT_ENV} not set (Marionette-fkjepa checkout)")

    root = os.path.abspath(root)
    inserted = root not in sys.path
    if inserted:
        sys.path.insert(0, root)
    pre_existing_models_modules = {
        k: v for k, v in sys.modules.items()
        if k == "models" or k.startswith("models.")
    }
    for k in pre_existing_models_modules:
        del sys.modules[k]

    try:
        from models.posendf.readout_v2 import ReadoutV2Head as SourceReadoutV2Head
        yield SourceReadoutV2Head
    finally:
        for k in list(sys.modules):
            if k == "models" or k.startswith("models."):
                del sys.modules[k]
        sys.modules.update(pre_existing_models_modules)
        if inserted:
            sys.path.remove(root)


@pytest.mark.ckpt
def test_real_checkpoint_loads_strict(source_readoutv2head_cls):
    """``load_state_dict(strict=True)`` -- 0 missing / 0 unexpected keys --
    against the real ``readout_v2_gr1.pt``, and the shape-bearing cfg fields
    match what was hand-verified before this task started (see the task
    brief's VERIFIED section)."""
    path = _resolve_ckpt_path()
    loaded = checkpoint_v2.load_head(path)

    assert loaded.head.in_dim == 1024
    assert loaded.head.d_model == 512
    assert loaded.head.n_heads == 4
    assert loaded.head.temporal_nhead == 8
    assert loaded.head.ff == 1024
    assert loaded.head.n_temporal_layers == 2
    assert loaded.head.t_max == 64
    assert abs(loaded.head.dropout - 0.1) < 1e-8
    assert loaded.head.logvar_min == -10.0
    assert loaded.head.logvar_max == 4.0
    assert loaded.head.n_out == 29  # 17 GR-1 joints + 12 hand DoF
    assert tuple(loaded.head.mu_head.weight.shape) == (29, 512)
    assert tuple(loaded.head.logvar_head.weight.shape) == (29, 512)
    assert abs(float(loaded.sigma_scale) - 1.9375) < 1e-3
    assert loaded.limit_semantics == "raw_rad"


@pytest.mark.ckpt
def test_real_checkpoint_reproduces_source_mu_sigma(source_readoutv2head_cls):
    """The numeric-parity check: kinescore's loaded head, on a fixed random
    feature input, reproduces the SOURCE ``ReadoutV2Scorer``'s ``mu``/``sigma``
    (single-head case: mean-over-members == mu itself, epistemic == 0, so
    ``sigma == sqrt(sigma_scale**2 * exp(logvar))``) to ``atol=1e-4``.
    """
    path = _resolve_ckpt_path()
    SourceReadoutV2Head = source_readoutv2head_cls

    kinescore_loaded = checkpoint_v2.load_head(path)
    source_head = SourceReadoutV2Head.load(path, device="cpu")

    torch.manual_seed(0)
    feat = torch.randn(1, 8, 64, 1024)  # (B,T,P,D) -- P is arbitrary, pool-invariant

    with torch.no_grad():
        out_new = kinescore_loaded.head(feat, use_context=True)
        out_src = source_head(feat, use_context=True)

    torch.testing.assert_close(out_new["mu"], out_src["mu"], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out_new["logvar"], out_src["logvar"],
                               atol=1e-4, rtol=1e-4)

    # sigma: source's ReadoutV2Scorer.readout() calibrates the ALEATORIC
    # VARIANCE by sigma_scale**2 (single head -> epistemic==0, mean-over-
    # members == mu/logvar themselves) -- see readout_v2_scorer.py::readout.
    scale = kinescore_loaded.sigma_scale
    al_src = torch.exp(out_src["logvar"]) * (scale.to(out_src["logvar"].dtype) ** 2)
    sigma_src = al_src.clamp_min(0).sqrt()
    sigma_new = torch.exp(0.5 * out_new["logvar"]) * scale.to(out_new["logvar"].dtype)

    torch.testing.assert_close(sigma_new, sigma_src, atol=1e-4, rtol=1e-4)


@pytest.mark.ckpt
def test_real_checkpoint_end_to_end_through_readoutv2_pose_reader(
        source_readoutv2head_cls):
    """Same numeric-parity check, but through the actual composed reader
    (:class:`~kinescore.readers.checkpoint_v2.ReadoutV2PoseReader`) rather
    than the bare head -- proves ``load_reader`` (what the CLI actually
    calls) applies the same calibration, using a fake backbone so this stays
    offline (no DINOv3 weights needed)."""
    path = _resolve_ckpt_path()
    SourceReadoutV2Head = source_readoutv2head_cls
    source_head = SourceReadoutV2Head.load(path, device="cpu")

    class _FakeRobot:
        name = "fourier_gr1"
        n_joints = 17
        q_lo = torch.full((17,), -3.2)
        q_hi = torch.full((17,), 3.2)

    class _FakeBackbone:
        def __init__(self, view_layout):
            self.view_layout = view_layout

        def encode(self, rgb):
            n = rgb.shape[0]
            torch.manual_seed(0)
            return torch.randn(n, self.view_layout.n_views, 64, 1024)

        def to(self, device):
            return self

        def eval(self):
            return self

    from kinescore.core.clip import ViewLayout

    layout = ViewLayout(n_views=1, tokens_per_view=64)
    reader = checkpoint_v2.load_reader(path, robot=_FakeRobot(), view_layout=layout,
                                       backbone=_FakeBackbone(layout))

    frames = torch.randint(0, 255, (8, 32, 32, 3), dtype=torch.uint8)
    out = reader.read(frames)

    torch.manual_seed(0)
    feat = torch.randn(1, 8, 64, 1024)
    with torch.no_grad():
        out_src = source_head(feat, use_context=True)
    mu17_src, mu12_src = out_src["mu"][..., :17], out_src["mu"][..., 17:]

    # q_raw (unclamped) on the FK slice must equal the source's q17 exactly;
    # aux["hand_raw"] must equal the source's hand12 exactly.
    torch.testing.assert_close(out.q_raw, mu17_src, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out.aux["hand_raw"], mu12_src, atol=1e-4, rtol=1e-4)
