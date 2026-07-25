"""D2: a reference pins its ``dt``, serializes it, and refuses a mismatched rate.

Covers:
* schema-2 ``save``/``load`` round-trips ``dt`` (and ``suite_id``/``term_keys``/``floors``).
* a legacy (schema-less) file RAISES on ``load(path)`` without an explicit ``dt=``.
* ``load(path, dt=...)`` recovers a legacy file, with a warning.
* scoring against a mismatched rate raises ``RateMismatchError``;
  ``allow_rate_mismatch=True`` downgrades to a warning and reports the mismatch.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from kinescore.core.metric import MetricSpec
from kinescore.core.suite import MetricSuite
from kinescore.reference.fingerprint import RateMismatchError, RealMotionReference


class _M:
    """Minimal Metric-protocol stand-in: only `.spec` is needed to build a suite."""

    def __init__(self, key, units="mm", perframe=False):
        self.spec = MetricSpec(key=key, units=units, dt_exponent=0,
                               direction="lower_better", requires=frozenset(),
                               perframe=perframe)

    def compute(self, ctx):  # pragma: no cover - never called in these tests
        raise NotImplementedError


def _build_suite():
    return MetricSuite("dt_roundtrip_suite",
                       [_M("rigidity_wobble_mm"), _M("mean_jerk_mps3"), _M("qty1", perframe=True)],
                       invariant_keys=["rigidity_wobble_mm", "mean_jerk_mps3"])


def _build_reference(dt=0.2):
    suite = _build_suite()
    rng = np.random.default_rng(0)
    residual_list = [
        {"rigidity_wobble_mm": float(0.3 + 0.05 * rng.normal()),
         "mean_jerk_mps3": float(0.01 + 0.002 * rng.normal())}
        for _ in range(8)
    ]
    samples_list = [{"qty1": rng.normal(size=50)} for _ in range(8)]
    return RealMotionReference.build(suite, residual_list, samples_list, dt=dt)


def test_schema2_roundtrips_dt(tmp_path):
    ref = _build_reference(dt=0.2)
    path = tmp_path / "ref_v2.pt"
    ref.save(str(path))

    loaded = RealMotionReference.load(str(path))
    assert loaded.schema == 2
    assert loaded.dt == pytest.approx(0.2)
    assert loaded.suite_id == ref.suite_id
    assert loaded.term_keys == ref.term_keys
    assert loaded.floors == pytest.approx(ref.floors)
    assert loaded.inv_baseline == pytest.approx(ref.inv_baseline)


def test_schema2_load_rejects_disagreeing_dt(tmp_path):
    ref = _build_reference(dt=0.2)
    path = tmp_path / "ref_v2.pt"
    ref.save(str(path))
    with pytest.raises(ValueError, match="disagrees"):
        RealMotionReference.load(str(path), dt=0.1)


def _write_legacy_file(path, dt_used_but_not_saved=0.2):
    """Mimic the source's schema-less save(): no 'schema', no 'dt' key at all."""
    suite = _build_suite()
    ref = _build_reference(dt=dt_used_but_not_saved)
    legacy_payload = {
        "inv_baseline": dict(ref.inv_baseline),
        "quantiles": {k: np.asarray(v) for k, v in ref.quantiles.items()},
        "feat_mu": ref.feat_mu,
        "feat_cov": ref.feat_cov,
        "quantity_keys": list(ref.quantity_keys),
        "n_q": ref.n_q,
    }
    torch.save(legacy_payload, path)
    return suite


def test_legacy_file_without_dt_raises(tmp_path):
    path = tmp_path / "legacy_ref.pt"
    _write_legacy_file(str(path))
    with pytest.raises(ValueError, match="legacy reference.*v1.*records no dt"):
        RealMotionReference.load(str(path))


def test_legacy_file_with_explicit_dt_loads_and_warns(tmp_path):
    path = tmp_path / "legacy_ref.pt"
    _write_legacy_file(str(path), dt_used_but_not_saved=0.2)
    with pytest.warns(UserWarning, match="legacy"):
        loaded = RealMotionReference.load(str(path), dt=0.2)
    assert loaded.schema == 1
    assert loaded.dt == pytest.approx(0.2)
    assert loaded.suite_id == "legacy:v1"
    # term_keys recovered from whatever was in the file -- not suite-pinned.
    assert set(loaded.term_keys) == {"rigidity_wobble_mm", "mean_jerk_mps3"}


def test_cross_rate_scoring_raises_rate_mismatch(tmp_path):
    ref = _build_reference(dt=0.2)
    # A run scored at 2x the reference's rate (defect D1's exact scenario).
    with pytest.raises(RateMismatchError):
        ref.check_rate(0.1)


def test_allow_rate_mismatch_warns_and_flags(tmp_path):
    ref = _build_reference(dt=0.2)
    with pytest.warns(UserWarning, match="rescales"):
        mismatch = ref.check_rate(0.1, allow_rate_mismatch=True)
    assert mismatch is True

    # Matching rate: no warning, no mismatch.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ref.check_rate(0.2) is False


def test_invariance_score_dt_param_records_rate_mismatch():
    from kinescore.reference.normalize import invariance_score

    ref = _build_reference(dt=0.2)
    residuals = {"rigidity_wobble_mm": 0.3, "mean_jerk_mps3": 0.01}

    with pytest.raises(RateMismatchError):
        invariance_score(residuals, ref, dt=0.1)

    with pytest.warns(UserWarning):
        result = invariance_score(residuals, ref, dt=0.1, allow_rate_mismatch=True)
    assert result.rate_mismatch is True

    result_ok = invariance_score(residuals, ref, dt=0.2)
    assert result_ok.rate_mismatch is False
