"""Closed-form checks with no fixture, no golden file, no other subpackage.

Every other test in this package compares against *something else* --
another dt, another bone set, a measured provenance figure. These checks
compare against elementary calculus / statistics facts that a golden fixture
could not accidentally canonise a bug into: a constant-velocity trajectory
has, by the definition of the third derivative, exactly zero jerk; two
points moving together as a rigid body have, by the definition of a rigid
body, an exactly constant separation; the identity/translation properties of
1-Wasserstein distance are definitional. If any of these fail, the bug is in
the formula itself, not in a comparison baseline -- which is exactly the
"a golden regression test would have canonised the bug instead of catching
it" failure mode this file exists to rule out.
"""
from __future__ import annotations

import torch
from _fake_robot import FakeRobot

from kinescore.core.metric import MetricContext
from kinescore.metrics.rigidity import RigidityWobble
from kinescore.metrics.temporal import MeanJerk

# profile_w1's canonical implementation lives in kinescore.reference (owned
# by that subpackage, already covered by its own tests/test_kfd.py); reused
# here rather than re-defined, so there is exactly one W1 implementation in
# the codebase. See that module's docstring for the seeded-subsampling
# behaviour above kinescore's DEFAULT_SUBSAMPLE_CAP -- irrelevant at the
# small sample sizes used below.
from kinescore.reference.distances import profile_w1

# ===========================================================================
# constant velocity => mean_jerk == 0
# ===========================================================================

#: This "== 0" claim is exact in the reals (a first difference of an affine
#: sequence is a constant, a first difference of a constant is exactly
#: zero, for ANY dt), but a naive float32 computation of ``t = arange(T)*dt``
#: accumulates a different rounding error at each ``i``, so the sequence is
#: only affine to within ~1e-7 relative error, not bit-exact -- that shows up
#: as a spurious ``mean_jerk ~= 1e-3`` instead of the true 0, an artifact of
#: float32 rounding, not of the metric's formula. Computing in float64
#: (~1e-16 relative precision) shrinks that artifact by ~9 orders of
#: magnitude, to comfortably below any threshold a real defect could hide
#: under -- which is what "closed-form, not empirical" means in practice.
_ZERO_TOL = 1e-9


def test_constant_velocity_has_zero_jerk():
    """A keypoint moving at exactly constant velocity has zero acceleration,
    hence zero jerk, at every finite-difference order -- this holds exactly
    (not approximately) because a first difference of an affine sequence is
    constant, and a first difference of a constant sequence is exactly zero,
    regardless of ``dt``. See :data:`_ZERO_TOL` for why float64 + a tight
    (not bit-exact) tolerance is the correct way to check this."""
    T = 12
    dt = 0.15
    velocity = torch.tensor([0.3, -0.2, 0.05], dtype=torch.float64)
    t = torch.arange(T, dtype=torch.float64).view(1, T, 1) * dt
    P = (t * velocity.view(1, 1, 3)).unsqueeze(2)                  # (1,T,1,3)

    ctx = MetricContext(dt=dt, P=P)
    value = MeanJerk().compute(ctx)
    assert value.available, value.reason
    assert abs(value.value) < _ZERO_TOL, value.value


def test_constant_velocity_zero_jerk_holds_at_any_dt():
    """The same fact, at several unrelated dt values -- confirms it is not a
    coincidence of one particular dt."""
    T = 10
    velocity = torch.tensor([1.0, 0.0, -0.5], dtype=torch.float64)
    for dt in (0.01, 0.1, 0.33, 2.5):
        t = torch.arange(T, dtype=torch.float64).view(1, T, 1) * dt
        P = (t * velocity.view(1, 1, 3)).unsqueeze(2)
        ctx = MetricContext(dt=dt, P=P)
        value = MeanJerk().compute(ctx)
        assert abs(value.value) < _ZERO_TOL, (dt, value.value)


# ===========================================================================
# a bone whose endpoints move rigidly => wobble == 0
# ===========================================================================

def test_rigid_body_bone_has_zero_wobble():
    """Two points rotating together as a rigid body (fixed separation)
    produce an exactly-constant realised bone length, so its temporal std
    (rigidity_wobble_mm) is exactly zero -- this is the definition of "rigid
    body motion", not an approximation that happens to be small."""
    T = 20
    radius = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64)    # bone half-length-ish offset
    theta = torch.linspace(0.0, 3.0, T, dtype=torch.float64)       # radians, arbitrary rotation
    c, s = torch.cos(theta), torch.sin(theta)
    # point A traces a circle of radius |radius|; point B is A's rigid
    # antipode about the origin -- separation is constant (= 2*|radius|)
    # for ANY rotation, by construction (both are rigidly attached to the
    # same rotating frame).
    offset = torch.stack(
        [radius[0] * c, radius[0] * s, torch.zeros(T, dtype=torch.float64)], dim=-1)
    point_a = offset
    point_b = -offset
    P = torch.stack([point_a, point_b], dim=1).unsqueeze(0)        # (1,T,2,3)

    robot = FakeRobot(
        bone_pairs=torch.tensor([[0, 1]], dtype=torch.long),
        bone_lengths=torch.tensor([1.0], dtype=torch.float32))
    ctx = MetricContext(dt=0.1, P=P, robot=robot)
    value = RigidityWobble(bone_set="rigid").compute(ctx)
    assert value.available, value.reason
    # mm units: 1e-9 tol here is ~1e-12 m, far below float64 rounding noise
    # in the trig functions above but far above any real defect.
    assert abs(value.value) < _ZERO_TOL, value.value


def test_rigid_body_bone_zero_wobble_survives_translation():
    """Rigid rotation PLUS a rigid-body translation of the whole pair (a
    moving, spinning object) still has exactly constant bone length -- the
    metric's own docstring claims robustness to a constant localisation
    bias; this checks robustness to a *moving* rigid-body frame, a stronger
    property in the same direction."""
    T = 15
    theta = torch.linspace(0.0, 5.0, T)
    c, s = torch.cos(theta), torch.sin(theta)
    offset = torch.stack([0.3 * c, 0.3 * s, torch.zeros(T)], dim=-1)
    drift = torch.stack([torch.linspace(0, 2.0, T),
                         torch.linspace(0, -1.0, T),
                         torch.zeros(T)], dim=-1)
    point_a = drift + offset
    point_b = drift - offset
    P = torch.stack([point_a, point_b], dim=1).unsqueeze(0)

    robot = FakeRobot(
        bone_pairs=torch.tensor([[0, 1]], dtype=torch.long),
        bone_lengths=torch.tensor([0.6], dtype=torch.float32))
    ctx = MetricContext(dt=0.05, P=P, robot=robot)
    value = RigidityWobble(bone_set="rigid").compute(ctx)
    assert value.available, value.reason
    assert abs(value.value) < 1e-4, value.value


# ===========================================================================
# profile_w1: identity and pure-shift properties
# ===========================================================================

def test_profile_w1_identity_is_zero():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(500, generator=g)
    assert profile_w1(x, x) == 0.0


def test_profile_w1_pure_shift_is_the_shift_magnitude():
    """W1 between a distribution and a pure translate of itself by ``c`` is
    exactly ``|c|`` -- every quantile shifts by exactly ``c``, so the mean
    absolute quantile gap is exactly ``|c|``. Definitional, not empirical."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(500, generator=g)
    for c in (0.0, 1.0, -2.5, 10.0):
        got = profile_w1(x, x + c)
        assert abs(got - abs(c)) < 1e-4, (c, got)
