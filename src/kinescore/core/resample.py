"""Match frame rates across clips -- the opt-in third layer of the rate policy.

kinescore's benchmark spans clips recorded (or generated) at genuinely
different frame rates. The default answer to "how do I compare them" is:
don't -- score every generated clip against *its own* ground truth (the
``paired`` policy, see ``docs/BENCHMARKING.md``), because fps cancels within a
pair and every derivative metric in this package is exactly homogeneous in
``dt`` (see ``core/metric.py``'s ``dt_exponent``). The second answer is: use
only metrics that don't care about ``dt`` in the first place (the
``rate_free`` suite in ``metrics/suites.py``).

This module is the third, explicit-opt-in answer: actually resample a
trajectory onto a common grid so the *full* metric suite can be compared
across rates. It is deliberately not the default path for either policy above
and should not become one -- resampling is a real transformation of the
signal, not a bookkeeping change, and the caveat below is why.

Why this operates on ``q(t)``, not on video frames
----------------------------------------------------
``kinescore.video.reader.load_rgb`` documents (see its module docstring) why
the source's ``np.linspace`` uniform *frame* resampling was rejected: it
silently changes which pixels are decoded without updating any metadata, so
the interval between "kept" frames drifts out from under the timebase that
scores them (defect D1). Resampling pixels is also not defensible on its own
merits -- there is no honest way to invent an intermediate video frame.

A joint-angle trajectory is different: ``q(t)`` is a physical signal (a
robot's actual configuration over time), and interpolating a *smooth,
slowly-varying* signal between real samples is a well-posed, standard
operation, provided the interpolant does not invent structure the samples
don't support. That is why this module resamples ``q``/``q_raw``/``sigma``
(and any per-frame ``aux`` a robot's FK needs) *after* the pose reader has
run and *before* forward kinematics -- the only point in the pipeline where
"resample" has an honest meaning. Resampling raw pixels, or resampling
already-computed metric values, would not.

Why PCHIP, not a plain cubic spline
--------------------------------------
A natural/not-a-knot cubic spline is smoother (C2) but can *overshoot*
between samples -- it is fit to minimize curvature globally, not to respect
local monotonicity, so a spline through a joint trajectory that briefly
plateaus can ring above/below the plateau. That overshoot would manufacture
exactly the kind of extra jerk this benchmark exists to detect, as an
artifact of the resampler rather than of the video being scored -- an
unacceptable failure mode for a physics-plausibility tool. PCHIP (Piecewise
Cubic Hermite Interpolating Polynomial, Fritsch & Carlson 1980, via
``scipy.interpolate.PchipInterpolator``) is only C1 but is *shape-preserving*:
it never overshoots the data and reproduces local monotonicity, at the cost
of a small amount of smoothness the plain spline would have had. For a signal
whose second derivative (acceleration) already matters to the metrics reading
it, "never invents an extremum" is the right trade.

The honest caveat: resampling changes the noise spectrum
------------------------------------------------------------
A pose reader's frame-to-frame error is not white noise in general, and even
if it were, PCHIP interpolation is a nonlinear operator that reshapes
whatever spectrum it is given -- the interpolated trajectory does not have
the same noise characteristics as a trajectory that was natively sampled at
the target rate. Concretely: **a jerk (or any derivative-based metric)
computed on a resampled clip is not comparable to one computed on a natively
sampled clip at the same rate**, in either direction. This module resamples
*only* when explicitly asked (``rate_policy="resample:<hz>"`` in
``core/scorer.py``), and every result it produces is tagged so this caveat
survives into the provenance rather than being silently forgotten -- see
"the ``dt_source`` gap" below.

Downsampling is the default direction; upsampling is refused
------------------------------------------------------------------
Interpolating a coarse trajectory onto a *finer* grid invents frames between
real samples. Since those invented frames did not come from the reader, they
are necessarily smoother than reality could guarantee (PCHIP's whole point is
to avoid inventing extrema, which is exactly what makes an upsampled clip
read as unrealistically smooth) -- upsampling systematically flatters a
generator's apparent smoothness. Downsampling to the lowest rate in the
compared set has the opposite, safe property: it can only throw information
away, never invent it. :func:`resample_series`, :func:`resample_clip` and
:func:`resample_readout` therefore all raise :class:`UpsampleRefusedError`
when the requested target rate is higher than the native rate, unless the
caller passes ``allow_upsample=True`` explicitly.

Integer decimation delegates to ``ClipSpec.subsample``
------------------------------------------------------------
When the native-to-target ratio is (within floating-point tolerance) a whole
number ``k``, no interpolation is needed or wanted: every ``k``-th frame is a
*real* sample, not an invented one, and this is exactly what
:meth:`~kinescore.core.clip.ClipSpec.subsample` already does, correctly, with
tests. This module never reimplements that arithmetic -- the integer path
calls ``clip.subsample(k)`` and slices the trajectory tensors with the same
``[:, ::k]`` stride, matching :mod:`kinescore.video.reader`'s contract for
what "decimating a clip" means.

The ``dt_source`` gap (read before wiring this into a report)
-------------------------------------------------------------------
:class:`~kinescore.core.clip.ClipSpec.dt_source` is typed as the closed
``Literal`` ``DtSource`` in ``core/clip.py`` (``"ffprobe" | "fps_arg" |
"dt_arg" | "table" | "synthetic"``) -- none of which mean "this dt came from
resampling a different native rate", and ``core/clip.py`` is out of scope for
this change (see the module's own docstring for why it is the sole owner of
the timebase and worth treating carefully). Every :class:`~kinescore.core.
clip.ClipSpec` returned by this module therefore carries
``dt_source="resampled"`` -- a value that is **not** one of the five
``DtSource`` literal members. Nothing in this codebase validates ``dt_source``
against that ``Literal`` at runtime (it is a plain ``str`` field with no
``__post_init__`` check), so this works correctly today and is the closest
honest answer available without touching a file outside this change's scope:
it plainly does *not* claim any of the four native-probe provenances, which
is the property the design explicitly asked for ("the provenance never claims
the clip was natively at that rate"). A static type checker would flag it;
this repository runs no such check in CI (only ``ruff``, which does not
type-check). Whoever owns ``core/clip.py`` next should add ``"resampled"`` to
``DtSource`` to close this gap formally.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import torch

from kinescore.core.clip import ClipSpec
from kinescore.core.reader import Readout

__all__ = [
    "UpsampleRefusedError", "RatePolicy", "ResamplePlan",
    "parse_rate_policy", "plan_resample",
    "resample_series", "resample_clip", "resample_readout",
]

#: Relative tolerance for "is this ratio an integer" / "are these two rates
#: the same rate" -- generous enough to absorb container fps rounding (e.g.
#: 29.97 vs 30), matching ``video/probe.py``'s ``probe_tolerance`` default in
#: spirit though not in call site (that one guards a table-vs-probe
#: cross-check; this one guards a decimation-factor-is-a-whole-number check).
_RATE_TOL = 1e-6


class UpsampleRefusedError(ValueError):
    """Raised when a resample would invent frames (upsample) without opt-in.

    See the module docstring's "downsampling is the default direction"
    section for why this is refused by default rather than merely a
    documented caveat.
    """


RatePolicyKind = Literal["paired", "rate_free", "resample"]


@dataclass(frozen=True)
class RatePolicy:
    """One of the three layers ``core/scorer.py::Scorer`` understands.

    ``"paired"`` and ``"rate_free"`` carry no ``target_fps`` -- they change
    *which suite is allowed*, not the timebase itself, and are enforced in
    ``Scorer.__init__`` (structurally, against the suite's own declared
    ``dt_exponent``\\ s -- no suite name needs to be known here). Only
    ``"resample"`` carries a ``target_fps``, and ``Scorer`` is what actually
    calls into this module's resampling functions for it.
    """

    kind: RatePolicyKind
    target_fps: float | None = None
    allow_upsample: bool = False

    def __post_init__(self) -> None:
        if self.kind == "resample":
            if self.target_fps is None or self.target_fps <= 0:
                raise ValueError(
                    f"RatePolicy(kind='resample') needs target_fps > 0, got "
                    f"{self.target_fps!r}")
        elif self.target_fps is not None:
            raise ValueError(
                f"RatePolicy(kind={self.kind!r}) must not carry a target_fps "
                f"(got {self.target_fps!r}); only 'resample' does")


def parse_rate_policy(spec: str, *, allow_upsample: bool = False) -> RatePolicy:
    """Parse a ``--rate-policy`` string into a :class:`RatePolicy`.

    Accepts exactly ``"paired"``, ``"rate_free"``, or ``"resample:<hz>"``
    (e.g. ``"resample:10"``, ``"resample:16.0"``) -- the three policies named
    in ``docs/BENCHMARKING.md``. Anything else is a ``ValueError`` naming the
    valid forms, so a typo in a config file fails at load time rather than
    silently falling back to ``paired``.
    """
    if spec == "paired":
        return RatePolicy("paired")
    if spec == "rate_free":
        return RatePolicy("rate_free")
    if spec.startswith("resample:"):
        hz_text = spec[len("resample:"):]
        try:
            hz = float(hz_text)
        except ValueError:
            raise ValueError(
                f"invalid rate policy {spec!r}: {hz_text!r} is not a "
                f"number; expected 'resample:<hz>', e.g. 'resample:10'"
            ) from None
        if hz <= 0 or not math.isfinite(hz):
            raise ValueError(
                f"invalid rate policy {spec!r}: target rate must be finite "
                f"and > 0, got {hz}")
        return RatePolicy("resample", target_fps=hz, allow_upsample=allow_upsample)
    raise ValueError(
        f"unknown rate policy {spec!r}; expected one of 'paired', "
        f"'rate_free', or 'resample:<hz>' (e.g. 'resample:10')")


# ===========================================================================
# trajectory-level resampling
# ===========================================================================

def resample_series(x: torch.Tensor, native_dt: float, target_dt: float, *,
                     time_dim: int = 1) -> torch.Tensor:
    """PCHIP-interpolate ``x`` from a uniform ``native_dt`` grid to ``target_dt``.

    Parameters
    ----------
    x:
        A tensor whose ``time_dim`` axis is the frame axis, sampled uniformly
        at ``native_dt``. Any other axes (batch, joints, ...) are carried
        through unchanged -- ``scipy``'s ``PchipInterpolator`` interpolates
        independently along every axis except ``axis``.
    native_dt, target_dt:
        Seconds between frames, before and after. Must both be positive and
        finite (this function does not itself enforce the "no upsampling"
        policy -- see :func:`resample_clip` / :func:`resample_readout`, which
        do, and which are the intended entry points; call this directly only
        if you have already made that call).
    time_dim:
        Which axis of ``x`` is time. Defaults to ``1`` (``(B,T,...)``, the
        convention every tensor in this package uses -- see
        ``core/metric.py``'s ``Requires`` docstring).

    Returns
    -------
    torch.Tensor
        Same dtype and device as ``x``, same shape except ``time_dim``, whose
        length becomes the number of ``target_dt``-spaced samples that fit
        inside ``x``'s original time span **without extrapolating** (the
        last returned sample is at or before the last native sample's
        timestamp -- this function never invents a value past the edge of
        the data PCHIP was actually fit to).

    Raises
    ------
    ValueError
        If ``x`` has fewer than 2 frames along ``time_dim`` (PCHIP needs at
        least 2 points to define an interpolant) or if either ``dt`` is not
        finite and positive.
    """
    if not (math.isfinite(native_dt) and native_dt > 0):
        raise ValueError(f"native_dt must be finite and > 0, got {native_dt}")
    if not (math.isfinite(target_dt) and target_dt > 0):
        raise ValueError(f"target_dt must be finite and > 0, got {target_dt}")

    from scipy.interpolate import PchipInterpolator

    n = x.shape[time_dim]
    if n < 2:
        raise ValueError(
            f"resample_series: need at least 2 frames along time_dim="
            f"{time_dim} to interpolate, got {n}")

    t_native = np.arange(n, dtype=np.float64) * native_dt
    duration = t_native[-1]
    n_target = int(math.floor(duration / target_dt + 1e-9)) + 1
    t_target = np.arange(n_target, dtype=np.float64) * target_dt
    # Guard against a float-rounding edge case pushing the last target sample
    # a hair past the last native one -- clip rather than let PCHIP extrapolate.
    t_target = np.minimum(t_target, duration)

    x_np = x.detach().cpu().double().numpy()
    interp = PchipInterpolator(t_native, x_np, axis=time_dim, extrapolate=False)
    y_np = interp(t_target)
    if np.isnan(y_np).any():  # pragma: no cover -- defensive; see comment below
        # Should not happen given the clipping above; fail loudly rather than
        # silently return a NaN-poisoned trajectory if it ever does.
        raise ValueError(
            "resample_series: PCHIP produced NaN despite the extrapolation "
            "guard -- t_target exceeded the native time span; this is a bug "
            "in resample_series, please report it")
    return torch.as_tensor(y_np, dtype=x.dtype, device=x.device)


# ===========================================================================
# ClipSpec-level resampling (delegates decimation to ClipSpec.subsample)
# ===========================================================================

@dataclass(frozen=True)
class ResamplePlan:
    """What :func:`plan_resample` decided, and how to carry it out.

    ``method``:

    * ``"noop"`` -- ``target_fps`` already matches the clip's native rate
      (within :data:`_RATE_TOL`); ``clip`` is the input clip, unchanged.
    * ``"decimate"`` -- the native/target ratio is a whole number ``stride``;
      ``clip`` is ``native_clip.subsample(stride)`` with ``dt_source``
      overridden to ``"resampled"``. Trajectories should be sliced
      ``[:, ::stride]``, matching :meth:`~kinescore.core.clip.ClipSpec.
      subsample`'s own contract.
    * ``"interpolate"`` -- non-integer ratio; ``clip`` carries the new
      ``fps``/``dt``/``n_frames``. Trajectories should go through
      :func:`resample_series`.
    """

    clip: ClipSpec
    method: Literal["noop", "decimate", "interpolate"]
    stride: int | None = None


def _integer_decimation_factor(native_fps: float, target_fps: float,
                               tol: float = _RATE_TOL) -> int | None:
    """``k`` such that ``native_fps / k == target_fps``, or ``None``."""
    ratio = native_fps / target_fps
    k = round(ratio)
    if k >= 1 and abs(ratio - k) <= tol * max(1.0, k):
        return int(k)
    return None


def plan_resample(clip: ClipSpec, target_fps: float, *,
                  allow_upsample: bool = False) -> ResamplePlan:
    """Decide how to move ``clip`` from its native rate to ``target_fps``.

    Refuses to upsample (``target_fps > clip.fps``) unless ``allow_upsample``
    is ``True`` -- see the module docstring. Never extrapolates and never
    duplicates :meth:`~kinescore.core.clip.ClipSpec.subsample`'s arithmetic;
    see :class:`ResamplePlan` for what each ``method`` means.
    """
    if math.isclose(target_fps, clip.fps, rel_tol=_RATE_TOL):
        return ResamplePlan(clip=clip, method="noop")

    is_upsample = target_fps > clip.fps
    if is_upsample and not allow_upsample:
        raise UpsampleRefusedError(
            f"target_fps={target_fps} > native fps={clip.fps} would "
            f"upsample {clip.path!r}, inventing frames between real "
            f"samples and inflating apparent smoothness. Refused by "
            f"default -- pass allow_upsample=True if this is deliberate "
            f"(the resulting ClipSpec still records dt_source='resampled', "
            f"never a native rate).")

    if not is_upsample:
        k = _integer_decimation_factor(clip.fps, target_fps)
        if k is not None:
            new_clip = replace(clip.subsample(k), dt_source="resampled")
            return ResamplePlan(clip=new_clip, method="decimate", stride=k)

    target_dt = 1.0 / target_fps
    duration = (clip.n_frames - 1) * clip.dt
    n_target = int(math.floor(duration / target_dt + 1e-9)) + 1
    new_clip = replace(clip, fps=target_fps, dt=target_dt, n_frames=n_target,
                       dt_source="resampled")
    return ResamplePlan(clip=new_clip, method="interpolate")


def resample_clip(clip: ClipSpec, target_fps: float, *,
                  allow_upsample: bool = False) -> ClipSpec:
    """The :class:`ClipSpec` half of :func:`plan_resample` -- see that function.

    This resamples metadata only. To also resample a trajectory consistently
    with the returned spec, use :func:`resample_readout` (or replicate
    :func:`plan_resample`'s ``method``/``stride`` yourself -- do not
    reimplement the rate arithmetic).
    """
    return plan_resample(clip, target_fps, allow_upsample=allow_upsample).clip


# ===========================================================================
# Readout-level resampling (the actual entry point for scoring)
# ===========================================================================

def _looks_per_frame(x: object, b: int, t: int) -> bool:
    return (isinstance(x, torch.Tensor) and x.ndim >= 2
            and x.shape[0] == b and x.shape[1] == t)


def resample_readout(readout: Readout, clip: ClipSpec, target_fps: float, *,
                     allow_upsample: bool = False
                     ) -> tuple[Readout, ClipSpec]:
    """Resample a whole :class:`~kinescore.core.reader.Readout` to ``target_fps``.

    This is the intended call site: ``core/scorer.py::Scorer`` invokes it
    (for ``rate_policy="resample:<hz>"``) after the pose reader has run and
    before forward kinematics -- see the module docstring for why that is the
    only defensible place to interpolate.

    ``readout.q``, ``readout.q_raw`` and ``readout.sigma`` (whichever are not
    ``None``) are resampled identically, so they stay frame-aligned with each
    other. ``readout.aux`` is resampled too **iff** it is a tensor whose
    first two dimensions match ``readout.q``'s ``(B,T)`` -- this is the case
    for e.g. the Franka gripper opening (``FrankaSpec._gripper_bt1`` requires
    ``aux`` to broadcast to ``(B,T,1)``), where leaving it at the native
    frame count would desynchronise gripper state from joint angles and
    either crash forward kinematics on a shape mismatch or, worse, silently
    pair the wrong gripper sample with the wrong joint sample. ``aux`` that
    is ``None``, not a tensor, or not shaped ``(B,T,...)`` (e.g. a fixed
    per-clip scalar) is passed through unchanged. ``readout.extras`` is never
    touched -- it is documented as "anything else a metric may consume" with
    no per-frame contract of its own; a caller relying on a per-frame
    ``extras`` entry surviving a resample should resample it explicitly.

    A real (non-``"noop"``) interpolation emits a :class:`UserWarning`
    reiterating the noise-spectrum caveat from the module docstring, so it is
    visible at the call site even if nobody reads this docstring.

    Raises
    ------
    UpsampleRefusedError
        See :func:`plan_resample`.
    """
    plan = plan_resample(clip, target_fps, allow_upsample=allow_upsample)
    if plan.method == "noop":
        return readout, plan.clip

    b, t = readout.q.shape[0], readout.q.shape[1]

    if plan.method == "decimate":
        k = plan.stride
        assert k is not None  # method=="decimate" always sets stride
        new_q = readout.q[:, ::k]
        new_q_raw = readout.q_raw[:, ::k] if readout.q_raw is not None else None
        new_sigma = readout.sigma[:, ::k] if readout.sigma is not None else None
        new_aux = (readout.aux[:, ::k] if _looks_per_frame(readout.aux, b, t)
                  else readout.aux)
    else:
        new_q = resample_series(readout.q, clip.dt, plan.clip.dt)
        new_q_raw = (resample_series(readout.q_raw, clip.dt, plan.clip.dt)
                    if readout.q_raw is not None else None)
        new_sigma = (resample_series(readout.sigma, clip.dt, plan.clip.dt)
                    if readout.sigma is not None else None)
        new_aux = (resample_series(readout.aux, clip.dt, plan.clip.dt)
                  if _looks_per_frame(readout.aux, b, t) else readout.aux)
        warnings.warn(
            f"resample_readout: interpolated q(t) from dt={clip.dt:.6f}s "
            f"({clip.fps:.3f} fps) to dt={plan.clip.dt:.6f}s "
            f"({plan.clip.fps:.3f} fps) via PCHIP. The reader's noise "
            f"spectrum has changed -- derivative metrics (speed/accel/jerk/"
            f"energy/momentum/...) computed on this trajectory are NOT "
            f"comparable to ones computed on a natively-sampled clip at "
            f"either rate. See docs/BENCHMARKING.md.",
            stacklevel=2)

    new_readout = replace(readout, q=new_q, q_raw=new_q_raw, sigma=new_sigma,
                          aux=new_aux)
    return new_readout, plan.clip
