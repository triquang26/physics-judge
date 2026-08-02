"""``kinescore.training.calibrate``: post-hoc sigma-temperature fitting.

Was 0% covered -- nothing in this module needs a GPU, the network, or a
checkpoint (it is pure tensor math on already-in-memory ``mu``/``logvar``/
``target``), so the gap was a plain oversight, not a legitimately-untestable
path. Pins: the moment-matching identity ``T == std(z)``, the ``eps`` floor
on a degenerate zero-variance split, the per-joint vs. global axis choice,
and that ``apply_temperature`` broadcasts a per-joint fit onto matching
``sigma`` shapes.
"""
from __future__ import annotations

import math

import torch

from kinescore.training.calibrate import (
    CalibrationResult,
    apply_temperature,
    calibrate_sigma,
    fit_sigma_temperature,
    standardized_residual,
)


class TestStandardizedResidual:
    def test_matches_hand_computation(self):
        mu = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([0.0, 2.0, 5.0])
        sigma = torch.tensor([1.0, 0.5, 2.0])
        z = standardized_residual(mu, target, sigma)
        assert torch.allclose(z, torch.tensor([1.0, 0.0, -1.0]))


class TestFitSigmaTemperature:
    def test_global_temperature_equals_std(self):
        torch.manual_seed(0)
        z = torch.randn(1000) * 2.5  # std ~2.5
        T = fit_sigma_temperature(z)
        assert T.shape == ()
        assert abs(float(T) - float(z.std(unbiased=True))) < 1e-6

    def test_per_joint_temperature_is_per_last_axis(self):
        torch.manual_seed(0)
        z = torch.stack([torch.randn(500) * 1.0, torch.randn(500) * 3.0], dim=-1)
        T = fit_sigma_temperature(z, per_joint=True)
        assert T.shape == (2,)
        expected = z.std(dim=0, unbiased=True)
        assert torch.allclose(T, expected, atol=1e-6)

    def test_eps_floor_on_degenerate_zero_variance_split(self):
        z = torch.ones(10)  # std == 0 exactly
        T = fit_sigma_temperature(z, eps=1e-8)
        assert math.isclose(float(T), 1e-8, rel_tol=1e-3)

    def test_eps_floor_does_not_clip_a_healthy_std(self):
        torch.manual_seed(1)
        z = torch.randn(200)
        T = fit_sigma_temperature(z, eps=1e-8)
        assert float(T) > 1e-6

    def test_result_is_float32(self):
        z = torch.randn(50, dtype=torch.float64)
        T = fit_sigma_temperature(z)
        assert T.dtype == torch.float32


class TestCalibrateSigma:
    def test_recovers_known_overconfidence_factor(self):
        # sigma_true = 2 * sigma_reported -> standardized residual has std 2.
        torch.manual_seed(2)
        n = 2000
        target = torch.zeros(n)
        true_sigma = 2.0
        mu = torch.randn(n) * true_sigma
        reported_sigma = torch.ones(n)  # head thinks sigma == 1 (overconfident)
        logvar = torch.log(reported_sigma ** 2)
        result = calibrate_sigma(mu, logvar, target)
        assert isinstance(result, CalibrationResult)
        assert result.per_joint is False
        assert result.n_samples == n
        # T should land near true_sigma / reported_sigma == 2.0.
        assert abs(float(result.temperature) - true_sigma) < 0.2

    def test_per_joint_flag_is_threaded_through(self):
        mu = torch.zeros(10, 3)
        logvar = torch.zeros(10, 3)
        target = torch.zeros(10, 3)
        result = calibrate_sigma(mu, logvar, target, per_joint=True)
        assert result.per_joint is True
        assert result.temperature.shape == (3,)
        assert result.n_samples == 30

    def test_n_samples_counts_every_element_not_just_frames(self):
        mu = torch.zeros(4, 7)
        logvar = torch.zeros(4, 7)
        target = torch.zeros(4, 7)
        result = calibrate_sigma(mu, logvar, target)
        assert result.n_samples == 28


class TestApplyTemperature:
    def test_scalar_temperature_scales_uniformly(self):
        sigma = torch.tensor([1.0, 2.0, 3.0])
        calib = CalibrationResult(temperature=torch.tensor(2.0), per_joint=False,
                                  n_samples=100)
        out = apply_temperature(sigma, calib)
        assert torch.allclose(out, sigma * 2.0)

    def test_per_joint_temperature_broadcasts_over_leading_dims(self):
        sigma = torch.ones(5, 3)  # (frames, joints)
        calib = CalibrationResult(temperature=torch.tensor([1.0, 2.0, 3.0]),
                                  per_joint=True, n_samples=15)
        out = apply_temperature(sigma, calib)
        expected = torch.tensor([1.0, 2.0, 3.0]).expand(5, 3)
        assert torch.allclose(out, expected)

    def test_preserves_input_device_and_dtype(self):
        sigma = torch.ones(4, dtype=torch.float64)
        calib = CalibrationResult(temperature=torch.tensor(1.5, dtype=torch.float32),
                                  per_joint=False, n_samples=4)
        out = apply_temperature(sigma, calib)
        assert out.dtype == torch.float64

    def test_round_trip_with_calibrate_sigma_on_synthetic_data(self):
        torch.manual_seed(3)
        n = 500
        target = torch.zeros(n)
        mu = torch.randn(n) * 3.0
        reported_sigma = torch.ones(n)
        logvar = torch.log(reported_sigma ** 2)
        calib = calibrate_sigma(mu, logvar, target)
        calibrated_sigma = apply_temperature(reported_sigma, calib)
        z_cal = standardized_residual(mu, target, calibrated_sigma)
        # After calibration, std(z_cal) should be ~1 (that's the whole point).
        assert math.isclose(float(z_cal.std(unbiased=True)), 1.0, abs_tol=0.1)
