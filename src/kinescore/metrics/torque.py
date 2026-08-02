"""``torque_frac_rated``: inverse-dynamics torque, as a percent of rated motor effort.

Ports ``Marionette-fkjepa/scripts/gr1/54_torque_feasibility.py`` -- the one
DYNAMICS ruler in this benchmark (every other metric is kinematic or
arbitrary-unit smoothness). A hand-rolled Newton-Euler pass over each arm's
links via virtual work -- **not** pinocchio, **not** a true recursive RNEA
(no inter-arm Coriolis coupling, no floating base), same as the source is
upfront about being. Wired for ``fourier_gr1`` only; any other robot gets
``NaN`` + ``unsupported_robot:<name>``, never a fabricated number. See
``docs/METRICS.md``'s torque section for the full formula, the
``dt_exponent=None`` proof, and scope notes.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from kinescore.core.metric import MetricContext, MetricSpec, MetricValue, register
from kinescore.heads.ranges import clamp_for_fk
from kinescore.metrics._base import SafeMetric
from kinescore.metrics.ops import vee
from kinescore.robots.inertia import ChainDynamics, build_chain_dynamics

__all__ = [
    "TorqueFracRated",
    "smooth_frames",
    "joint_torques",
    "clip_peak_pct",
]

_DEFAULT_PCT = 98.0    # source constant PCT (54_torque_feasibility.py:62)
_DEFAULT_SIGMA = 1.0   # smoothing width in FRAMES, not seconds -- source --sigma default
_DEFAULT_G = 9.81      # m/s^2, source constant G

#: Per-robot chain configuration: robot name -> ((label, joint_names, ee_link), ...).
#: Only ``fourier_gr1`` is wired -- see module docstring.
_ROBOT_CHAINS: dict[str, tuple] = {}


def _register_gr1_chains() -> None:
    """Populate :data:`_ROBOT_CHAINS` for ``fourier_gr1``, importing GR-1 lazily.

    A function, not a module-level import, so importing ``kinescore.metrics``
    never requires ``pytorch_kinematics`` unless a caller actually scores a
    GR-1 clip -- same lazy-import discipline as ``kinescore.robots.__init__``.
    """
    if "fourier_gr1" in _ROBOT_CHAINS:
        return
    from kinescore.robots.gr1.fk import EE_LINK, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS

    _ROBOT_CHAINS["fourier_gr1"] = (
        ("left", LEFT_ARM_JOINTS, EE_LINK["left"]),
        ("right", RIGHT_ARM_JOINTS, EE_LINK["right"]),
    )


def smooth_frames(q: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing along the time axis, per joint. ``(T,n) -> (T,n)``.

    Verbatim port of ``_smooth()`` (54_torque_feasibility.py:127-136):
    edge-padded, ``sigma`` in FRAMES (not seconds), applied identically to
    every clip (real or generated).
    """
    if sigma <= 0:
        return q
    rad = max(1, int(3 * sigma))
    x = np.arange(-rad, rad + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    pad = np.pad(q, ((rad, rad), (0, 0)), mode="edge")
    return np.stack(
        [np.convolve(pad[:, c], k, "valid") for c in range(q.shape[1])], axis=1)


def _omega_bt(R: torch.Tensor, dt: float) -> torch.Tensor:
    """``(B,T,L,3,3) -> (B,T,L,3)`` angular velocity via central-difference ``Rdot R^T``.

    Algebraically identical to the source's ``_omega()``
    (54_torque_feasibility.py:118-124), re-expressed via
    :func:`kinescore.metrics.ops.vee` applied to the *antisymmetric
    projection* ``0.5*(W - W^T)`` of ``W = Rdot R^T``, rather than reading
    ``W``'s lower-triangular entries directly. Same formula element for
    element (``vee(0.5*(W-W^T))[0] == 0.5*(W[2,1]-W[1,2])``, the source's own
    expression); the projection matters because ``W`` is built from a
    finite-difference ``Rdot``, so it is skew-symmetric only up to
    truncation error, and a bare ``vee(W)`` would silently pick up that
    error's symmetric part. Generalises the source's per-link Python loop to
    a batch of links/clips at once. Endpoints (``t=0``, ``t=T-1``) are
    exactly zero, not a small-but-nonzero approximation -- no neighbour exists.
    """
    b, t, ln = R.shape[0], R.shape[1], R.shape[2]
    w = torch.zeros(b, t, ln, 3, dtype=R.dtype, device=R.device)
    if t < 3:
        return w
    rdot = 0.5 * (R[:, 2:] - R[:, :-2])                      # (B,T-2,L,3,3)
    rt = R[:, 1:-1]                                           # (B,T-2,L,3,3)
    W = torch.einsum("btlij,btlkj->btlik", rdot, rt)          # Rdot @ Rt^T
    skew = 0.5 * (W - W.transpose(-1, -2))
    w[:, 1:-1] = vee(skew) / dt
    return w


def joint_torques(frames: torch.Tensor, chain: ChainDynamics, dt: float,
                   g: float = _DEFAULT_G) -> torch.Tensor:
    """``(B,T,L,4,4)`` posed links -> ``(B,T,n_joint)`` joint torques (N.m).

    Generalises the source's ``joint_torques()`` (54_torque_feasibility.py
    :139-164) from a single-clip numpy loop to a batched torch computation;
    the Newton-Euler-via-virtual-work arithmetic (:154-160) is unchanged.
    ``frames`` must be ``robot.fk.link_frames(q, chain.links)`` -- see
    ``docs/METRICS.md``'s torque section for the physics.

    Only interior frames (``1..T-2``) are physically meaningful. ``a``/``w``/
    ``al`` are exactly zero at the two endpoints (no neighbour for the
    central difference), but ``F = m*(a-g)`` still includes gravity there --
    so ``tau`` is a static-hold approximation at the endpoints, not zero, and
    a caller must slice interior frames off rather than average endpoints in
    (exactly why the source's own ``clip_peak()`` does ``tau[1:-1]``).
    """
    R = frames[..., :3, :3]
    p = frames[..., :3, 3]
    b, t, ln = R.shape[0], R.shape[1], R.shape[2]
    device, dtype = R.device, R.dtype

    com_local = torch.as_tensor(chain.com, dtype=dtype, device=device)   # (L,3)
    mass = torch.as_tensor(chain.mass, dtype=dtype, device=device)       # (L,)
    ib = torch.as_tensor(chain.inertia, dtype=dtype, device=device)      # (L,3,3)
    axis = torch.as_tensor(chain.axis, dtype=dtype, device=device)       # (L,3)
    gvec = torch.tensor([0.0, 0.0, -float(g)], dtype=dtype, device=device)

    com_w = p + torch.einsum("btlij,lj->btli", R, com_local)             # (B,T,L,3)

    a = torch.zeros_like(com_w)
    if t >= 3:
        a[:, 1:-1] = (com_w[:, 2:] - 2 * com_w[:, 1:-1] + com_w[:, :-2]) / dt ** 2

    w = _omega_bt(R, dt)                                                  # (B,T,L,3)
    al = torch.zeros_like(w)
    if t >= 3:
        al[:, 1:-1] = (w[:, 2:] - w[:, :-2]) / (2 * dt)

    iw = torch.einsum("btlij,ljk,btlmk->btlim", R, ib, R)                 # (B,T,L,3,3)
    F = mass.view(1, 1, -1, 1) * (a - gvec.view(1, 1, 1, 3))              # (B,T,L,3)
    N = (torch.einsum("btlij,btlj->btli", iw, al)
         + torch.cross(w, torch.einsum("btlij,btlj->btli", iw, w), dim=-1))

    n_joint = chain.n_joint
    tau = torch.zeros(b, t, n_joint, dtype=dtype, device=device)
    for j in range(n_joint):
        zj = torch.einsum("btij,j->bti", R[:, :, j], axis[j])             # (B,T,3)
        pj = p[:, :, j]                                                   # (B,T,3)
        acc = torch.zeros(b, t, dtype=dtype, device=device)
        for i in range(j, ln):
            lever = torch.cross(com_w[:, :, i] - pj, F[:, :, i], dim=-1)
            acc = acc + torch.einsum("bti,bti->bt", zj, lever)
            acc = acc + torch.einsum("bti,bti->bt", zj, N[:, :, i])
        tau[:, :, j] = acc
    return tau


def clip_peak_pct(ratio: torch.Tensor, pct: float) -> torch.Tensor:
    """``(B,T,J) -> (B,)``: per-frame binding-joint envelope, then a percentile.

    At each frame, take the most-loaded joint's ratio (``max`` over joints --
    the binding constraint for that instant), then a percentile of that
    envelope *over time*. Order matters: percentile-per-joint-then-max is a
    different, wrong statistic that dilutes a briefly-spiking joint -- pinned
    by ``test_clip_peak_pct_pins_envelope_then_percentile_order``. Matches
    the source's ``clip_peak()`` (54_torque_feasibility.py:167-178), minus
    the smoothing/clamping/FK steps :class:`TorqueFracRated` does separately
    (kept apart so this reduction is testable without a robot at all).
    """
    env = ratio.amax(dim=-1)                                              # (B,T)
    return torch.quantile(env.double(), pct / 100.0, dim=1)


class TorqueFracRated(SafeMetric):
    """Peak inverse-dynamics torque, as a percent of each joint's rated effort.

    See the module docstring for the physics, the ``dt_exponent=None``
    justification, and this metric's honest ``fourier_gr1``-only scope.
    """

    def __init__(self, sigma: float = _DEFAULT_SIGMA, pct: float = _DEFAULT_PCT,
                 g: float = _DEFAULT_G) -> None:
        self.sigma = float(sigma)
        self.pct = float(pct)
        self.g = float(g)
        self.spec = MetricSpec(
            key="torque_frac_rated", units="percent", dt_exponent=None,
            direction="lower_better", requires=frozenset({"q"}), min_frames=3,
            perframe=True,
            description=(
                "Per-frame envelope max_j(|tau_j|/effort_j) (effort_j = "
                f"URDF-rated N.m), clip statistic = 100*p{pct:g} over interior "
                "frames; tau_j from hand-rolled Newton-Euler inverse "
                "dynamics over the robot's arm links (NOT a full recursive "
                "RNEA, no external dynamics library -- see module "
                "docstring). q is smoothed (Gaussian, sigma frames) and "
                "clamp_for_fk-clamped before differentiation, identically "
                "for every clip. dt_exponent=None: tau sums a dt^-2 "
                "inertial-force/moment term with a dt^0 gravity-force term "
                "(F=m(a-g)), the same non-homogeneous-sum shape as "
                "total_energy_tstd (kinetic dt^-2 + potential dt^0), not a "
                "threshold crossing. Wired for fourier_gr1 only; NaN with an "
                "explicit reason for any other robot, or for a URDF that "
                "omits an inertial/effort declaration a wired chain needs -- "
                "never a fabricated near-infinite/zero fallback. "
                "perframe=True: the per-frame binding-joint envelope "
                f"100*max_j(|tau_j|/effort_j) BEFORE the p{pct:g} reduction "
                "(same env=ratio.amax(dim=-1) `clip_peak_pct` computes "
                "internally, recomputed here so the scalar's own call is "
                "untouched) -- this is the quantity `torque_frac_rated` is a "
                "percentile over. Only interior frames are physically "
                "meaningful (endpoints are a static-hold approximation, see "
                "`joint_torques`'s docstring): trace length == n_frames - 2, "
                "trace[0] is frame 1 (one dropped from each end)."
            ))

    def _compute(self, ctx: MetricContext) -> MetricValue:
        key = self.spec.key
        if ctx.robot is None:
            return MetricValue.unavailable(key, "missing_input:robot")

        robot_name = getattr(ctx.robot, "name", None)
        _register_gr1_chains()
        chains_cfg = _ROBOT_CHAINS.get(robot_name)
        if chains_cfg is None:
            return MetricValue.unavailable(key, f"unsupported_robot:{robot_name}")

        fk = getattr(ctx.robot, "fk", None)
        urdf_path = getattr(fk, "urdf_path", None)
        if fk is None or urdf_path is None or not hasattr(fk, "link_frames"):
            return MetricValue.unavailable(key, "missing_input:fk_link_frames")

        try:
            chains = [build_chain_dynamics(urdf_path, joints, ee_link=ee)
                      for _, joints, ee in chains_cfg]
        except Exception as exc:  # noqa: BLE001 -- a malformed URDF is "unavailable"
            return MetricValue.unavailable(
                key, f"missing_input:urdf_dynamics:{type(exc).__name__}")

        bad_effort = [f"{cname}:{jn}"
                      for (cname, _, _), cd in zip(chains_cfg, chains, strict=True)
                      for jn, e in zip(cd.joint_names, cd.effort, strict=True)
                      if not math.isfinite(e)]
        if bad_effort:
            return MetricValue.unavailable(
                key, "missing_input:effort_limits:" + ",".join(bad_effort))

        bad_mass = [f"{cname}:{lk}"
                    for (cname, _, _), cd in zip(chains_cfg, chains, strict=True)
                    for lk, m in zip(cd.links, cd.mass, strict=True) if math.isnan(m)]
        if bad_mass:
            return MetricValue.unavailable(
                key, "missing_input:inertial:" + ",".join(bad_mass))

        q = ctx.q
        b = q.shape[0]
        q_np = q.detach().cpu().numpy().astype(np.float32)
        smoothed = np.stack(
            [smooth_frames(q_np[i], self.sigma) for i in range(b)], axis=0)
        q_smoothed = torch.as_tensor(smoothed, dtype=q.dtype, device=q.device)
        q_clamped, _ = clamp_for_fk(q_smoothed, fk.q_lo, fk.q_hi)

        taus, efforts = [], []
        for (_cname, _joints, _ee), cd in zip(chains_cfg, chains, strict=True):
            frames = fk.link_frames(q_clamped, cd.links)                  # (B,T,L,4,4)
            tau = joint_torques(frames, cd, ctx.dt, g=self.g)             # (B,T,n_joint)
            taus.append(tau)
            efforts.append(torch.as_tensor(cd.effort, dtype=tau.dtype,
                                           device=tau.device))
        tau_all = torch.cat(taus, dim=-1)                                 # (B,T,J)
        effort_all = torch.cat(efforts, dim=0)                            # (J,)
        ratio = tau_all.abs() / effort_all.view(1, 1, -1)

        interior = ratio[:, 1:-1]
        if interior.shape[1] == 0:
            return MetricValue.unavailable(key, f"too_few_frames:{ctx.n_frames}<3")

        peak_pct = 100.0 * clip_peak_pct(interior, self.pct)              # (B,)
        # perframe: same envelope clip_peak_pct takes its percentile over,
        # recomputed independently so the scalar call above stays untouched.
        env_pct = (100.0 * interior.amax(dim=-1)).reshape(-1).detach().cpu().numpy()
        return self._ok(peak_pct.mean(), perframe=env_pct)


register(TorqueFracRated())
