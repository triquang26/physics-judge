"""Reproducibility fix: every corruption operator takes an explicit generator.

The source drew randomness from the global RNG, so a "reproducible" severity
sweep was only reproducible by accident of process history. These tests pin
down the fix: same seed -> identical output, different seed -> different
output, for the two operators that actually draw random numbers
(``gaussian_noise``, ``temporal_break``); the two deterministic operators
(``occlusion``, ``blur``) are checked for output stability and shape/range
invariants instead.
"""
from __future__ import annotations

import torch

from kinescore.video.corruptions import VideoCorruptions as VC


def _rgb(T=6, H=16, W=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(T, 3, H, W, generator=g)


class TestGaussianNoise:
    def test_zero_sigma_is_identity(self):
        rgb = _rgb()
        out = VC.gaussian_noise(rgb, 0.0, generator=torch.Generator().manual_seed(1))
        assert torch.equal(out, rgb)

    def test_same_seed_gives_identical_output(self):
        rgb = _rgb()
        out1 = VC.gaussian_noise(rgb, 0.2, generator=torch.Generator().manual_seed(42))
        out2 = VC.gaussian_noise(rgb, 0.2, generator=torch.Generator().manual_seed(42))
        assert torch.equal(out1, out2)

    def test_different_seed_gives_different_output(self):
        rgb = _rgb()
        out1 = VC.gaussian_noise(rgb, 0.2, generator=torch.Generator().manual_seed(1))
        out2 = VC.gaussian_noise(rgb, 0.2, generator=torch.Generator().manual_seed(2))
        assert not torch.equal(out1, out2)

    def test_output_stays_in_unit_range(self):
        rgb = _rgb()
        out = VC.gaussian_noise(rgb, 5.0, generator=torch.Generator().manual_seed(0))
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_none_generator_falls_back_to_global_rng_without_erroring(self):
        rgb = _rgb()
        out = VC.gaussian_noise(rgb, 0.1, generator=None)
        assert out.shape == rgb.shape


class TestTemporalBreak:
    def test_zero_frac_is_identity(self):
        rgb = _rgb()
        out = VC.temporal_break(rgb, 0.0, generator=torch.Generator().manual_seed(1))
        assert torch.equal(out, rgb)

    def test_same_seed_gives_identical_output(self):
        rgb = _rgb()
        out1 = VC.temporal_break(rgb, 0.5, generator=torch.Generator().manual_seed(7))
        out2 = VC.temporal_break(rgb, 0.5, generator=torch.Generator().manual_seed(7))
        assert torch.equal(out1, out2)

    def test_different_seed_gives_different_output(self):
        # Not a mathematical guarantee (a replacement frame can coincide with
        # its source, see the module docstring) but overwhelmingly likely at
        # T=32 with frac=0.8 for two distinct seeds.
        rgb = _rgb(T=32)
        out1 = VC.temporal_break(rgb, 0.8, generator=torch.Generator().manual_seed(1))
        out2 = VC.temporal_break(rgb, 0.8, generator=torch.Generator().manual_seed(2))
        assert not torch.equal(out1, out2)

    def test_preserves_shape(self):
        rgb = _rgb()
        out = VC.temporal_break(rgb, 0.5, generator=torch.Generator().manual_seed(0))
        assert out.shape == rgb.shape

    def test_every_output_frame_came_from_the_original_clip(self):
        # Regardless of which frames got swapped, no new pixel values should
        # appear -- every output frame is some original frame.
        rgb = _rgb(T=8)
        out = VC.temporal_break(rgb, 0.5, generator=torch.Generator().manual_seed(3))
        for t in range(out.shape[0]):
            assert any(torch.equal(out[t], rgb[s]) for s in range(rgb.shape[0]))


class TestOcclusion:
    def test_zero_frac_is_identity(self):
        rgb = _rgb()
        out = VC.occlusion(rgb, 0.0)
        assert torch.equal(out, rgb)

    def test_blacks_out_a_centred_square(self):
        rgb = torch.ones(4, 3, 16, 16)
        out = VC.occlusion(rgb, 0.25)
        assert out.min() == 0.0
        # corners untouched
        assert out[0, 0, 0, 0] == 1.0

    def test_deterministic_given_shape_and_severity(self):
        rgb = _rgb()
        out1 = VC.occlusion(rgb, 0.3, generator=torch.Generator().manual_seed(1))
        out2 = VC.occlusion(rgb, 0.3, generator=torch.Generator().manual_seed(2))
        assert torch.equal(out1, out2)


class TestBlur:
    def test_zero_sigma_is_identity(self):
        rgb = _rgb()
        out = VC.blur(rgb, 0.0)
        assert torch.equal(out, rgb)

    def test_output_stays_in_unit_range(self):
        rgb = _rgb()
        out = VC.blur(rgb, 2.0)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_deterministic_given_shape_and_severity(self):
        rgb = _rgb()
        out1 = VC.blur(rgb, 2.0, generator=torch.Generator().manual_seed(1))
        out2 = VC.blur(rgb, 2.0, generator=torch.Generator().manual_seed(2))
        assert torch.equal(out1, out2)


class TestApplyAndSweeps:
    def test_apply_dispatches_by_name(self):
        rgb = _rgb()
        g = torch.Generator().manual_seed(0)
        out = VC.apply("gaussian_noise", rgb, 0.1, generator=g)
        assert out.shape == rgb.shape

    def test_sweeps_start_at_zero_severity(self):
        for sweep in VC.SWEEPS.values():
            assert sweep[0] == 0.0

    def test_all_sweep_names_are_valid_operators(self):
        for name in VC.SWEEPS:
            assert hasattr(VC, name)
