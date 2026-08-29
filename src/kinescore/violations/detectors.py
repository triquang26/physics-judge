"""Per-frame violation detectors: one ERROR TYPE = one ``Detector``."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from kinescore.core.context import ClipContext

__all__ = [
    "Detector",
    "RigidityDetector",
    "JerkDetector",
    "TeleportDetector",
    "JointLimitDetector",
    "SelfCollisionDetector",
]


def _single_clip_P(ctx: ClipContext) -> torch.Tensor:
    """``ctx.P`` squeezed to ``(T,K,3)``, enforcing the single-clip contract.
    """
    if ctx.P is None:
        raise ValueError("Detector requires ctx.P")
    if ctx.P.shape[0] != 1:
        raise ValueError(
            f"Detector scores one clip per call; got batch size "
            f"{ctx.P.shape[0]}. Build one ClipContext per clip and loop.")
    return ctx.P[0]


class Detector:
    """Base class: one error type, one GT-calibrated threshold, one interval list.
    """

    name: str = "base"
    units: str = ""
    #: ``True``: a frame is flagged when its score is *above* the threshold
    #: (most detectors). ``False``: flagged *below* -- e.g. self-collision's
    #: minimum inter-keypoint distance, where smaller is worse.
    higher_is_worse: bool = True
    #: How a segment's frames reduce to the one number judged against the
    #: threshold. ``"worst"`` takes the extreme in this detector's own
    #: direction -- a single bad frame is the violation. ``"median"`` takes the
    #: typical frame instead, so a violation means the segment was bad
    #: throughout, not once.
    segment_reduce: str = "worst"

    def __init__(self) -> None:
        self.threshold: float | None = None

    def per_frame(self, ctx: ClipContext) -> np.ndarray:
        """Per-frame score ``(T,)`` for one clip. Must be overridden."""
        raise NotImplementedError

    def fit(self, gt_contexts: Sequence[ClipContext]) -> None:
        """Learn detector-specific state from GT clips, before :meth:`calibrate`.
        """

    def calibrate(self, gt_scores: np.ndarray, pct: float = 95.0,
                  floor: float = 0.0) -> None:
        """Set ``self.threshold`` from a pooled GT per-frame score array."""
        if self.higher_is_worse:
            self.threshold = float(max(floor, np.percentile(gt_scores, pct)))
        else:
            self.threshold = float(np.percentile(gt_scores, 100.0 - pct))

    def _flag(self, s: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError(
                f"{self.name}: calibrate() must run before scoring")
        return s > self.threshold if self.higher_is_worse else s < self.threshold

    def _intervals(self, flag: np.ndarray) -> list[list[int]]:
        """Contiguous-run-length encode a boolean flag array into ``[start,end]`` pairs."""
        out: list[list[int]] = []
        i, n = 0, len(flag)
        while i < n:
            if flag[i]:
                j = i
                while j + 1 < n and flag[j + 1]:
                    j += 1
                out.append([int(i), int(j)])
                i = j + 1
            else:
                i += 1
        return out

    def report(self, ctx: ClipContext) -> dict:
        """Score one clip: threshold, flagged fraction, severity, and this detector's own intervals.
        """
        s = self.per_frame(ctx)
        flag = self._flag(s)
        ratio = (s / self.threshold) if self.higher_is_worse else (self.threshold / np.maximum(s, 1e-9))
        return {
            "units": self.units,
            "threshold": round(float(self.threshold), 2),
            "fraction": round(float(flag.mean()), 3),
            "n_flagged": int(flag.sum()),
            "severity_ratio_median": round(float(np.median(ratio)), 3),
            "severity_ratio_p90": round(float(np.percentile(ratio, 90)), 3),
            "intervals": self._intervals(flag),           # <-- per-type interval list
            "per_frame": [round(float(x), 1) for x in s],
        }


class RigidityDetector(Detector):
    """Warp: a rigid arm link stretches/shrinks from its true URDF length.

    Parameters
    ----------
    rigid_idx:
        Which indices *into* ``robot.rigid_bone_pairs`` this detector treats
        as truly rigid. Default (``None``): every index, i.e.
        ``range(len(robot.rigid_bone_pairs))``.

        A robot may need to narrow this further than ``rigid_bone_pairs``
        already does. That set only drops *degenerate* bones (near-zero rest
        length); it says nothing about a bone that spans a moving joint --
        such a bone has a well-defined rest length at any single pose, but
        that length is not constant across poses, so treating it as "rigid"
        manufactures a rigidity violation out of ordinary motion. The Franka
        needs exactly this: its bone index 1 spans a rotating joint, so only
        ``rigid_idx=(0, 2, 3)`` are genuinely rigid. That is a per-robot
        URDF-topology fact this
        detector cannot infer from ``rigid_bone_pairs`` alone, so it is a
        constructor argument the caller resolves once per robot, not an
        auto-detected default.
    """

    name = "rigidity"
    #: A rigid link cannot stretch, so a lone stretched frame is measurement
    #: noise; a segment whose typical frame is stretched is not.
    segment_reduce = "median"
    units = "mm"

    def __init__(self, rigid_idx: Sequence[int] | None = None) -> None:
        super().__init__()
        self.rigid_idx = list(rigid_idx) if rigid_idx is not None else None

    def _resolve_idx(self, n_bones: int) -> list[int]:
        return self.rigid_idx if self.rigid_idx is not None else list(range(n_bones))

    def per_frame(self, ctx: ClipContext) -> np.ndarray:
        P = _single_clip_P(ctx)
        robot = ctx.robot
        bone_pairs = [(int(a), int(b)) for a, b in robot.rigid_bone_pairs]
        rest = robot.rigid_bone_lengths.cpu().float() * 1000.0
        idx = self._resolve_idx(len(bone_pairs))
        L = torch.stack(
            [(P[:, a] - P[:, b]).norm(dim=-1) for a, b in bone_pairs], dim=1,
        ) * 1000.0                                        # (T, n_bones) mm
        dev = (L[:, idx] - rest[idx][None]).abs().amax(dim=1)
        return dev.numpy()


class JerkDetector(Detector):
    """Jitter: 3rd derivative of keypoint position."""

    name = "jerk"
    units = "mm/s^3"

    def per_frame(self, ctx: ClipContext) -> np.ndarray:
        P = _single_clip_P(ctx)
        T = len(P)
        out = np.zeros(T)
        if T >= 4:
            j = P[3:] - 3 * P[2:-1] + 3 * P[1:-2] - P[:-3]
            jm = j.norm(dim=-1).amax(1)
            out[2:2 + len(jm)] = (jm * 1000.0).numpy() / (ctx.dt ** 3)
        return out


class TeleportDetector(Detector):
    """Teleport: worst keypoint speed. Per second, as :class:`JerkDetector`."""

    name = "teleport"
    units = "mm/s"

    def per_frame(self, ctx: ClipContext) -> np.ndarray:
        P = _single_clip_P(ctx)
        T = len(P)
        out = np.zeros(T)
        if T >= 2:
            step = (P[1:] - P[:-1]).norm(dim=-1).amax(1) * 1000.0
            out[1:] = step.numpy() / ctx.dt
        return out


class JointLimitDetector(Detector):
    """Impossible pose: a joint's bend angle leaves the range real motion ever shows.

    Parameters
    ----------
    lo_q, hi_q:
        Quantiles (in ``[0, 1]``) of the GT bend-angle distribution that
        define the per-joint envelope in :meth:`fit`. Trimming the 1% tails
        keeps one noisy GT frame from blowing the envelope open.
    """

    name = "joint_limit"
    units = "deg"
    higher_is_worse = True

    def __init__(self, lo_q: float = 0.01, hi_q: float = 0.99) -> None:
        super().__init__()
        self.lo_q = float(lo_q)
        self.hi_q = float(hi_q)
        self.lo: torch.Tensor | None = None                    # per-joint GT envelope
        self.hi: torch.Tensor | None = None

    @staticmethod
    def _bend(P: torch.Tensor) -> torch.Tensor:                # (T,K,3) -> (T,K-2) deg
        b1 = P[:, 1:-1] - P[:, :-2]
        b2 = P[:, 2:] - P[:, 1:-1]
        cos = (b1 * b2).sum(-1) / (b1.norm(dim=-1) * b2.norm(dim=-1) + 1e-6)
        return torch.rad2deg(torch.acos(cos.clamp(-1, 1)))

    def fit(self, gt_contexts: Sequence[ClipContext]) -> None:
        ang = torch.cat([self._bend(_single_clip_P(c)) for c in gt_contexts], dim=0)
        self.lo = torch.quantile(ang, self.lo_q, dim=0)
        self.hi = torch.quantile(ang, self.hi_q, dim=0)

    def per_frame(self, ctx: ClipContext) -> np.ndarray:
        P = _single_clip_P(ctx)
        if self.lo is None:
            return np.zeros(len(P))
        a = self._bend(P)                                       # (T,K-2)
        over = torch.relu(a - self.hi) + torch.relu(self.lo - a)  # deg outside GT range
        return over.amax(1).numpy()


class SelfCollisionDetector(Detector):
    """Arm folds into itself: two non-adjacent keypoints get closer than real ever gets.

    Parameters
    ----------
    non_adjacent_gap:
        Minimum keypoint-index gap for a pair to be scored: at the default
        2, pairs ``(i, j)`` with ``j >= i + 2``, which excludes only
        bone-connected neighbours. A robot whose keypoint
        chain has short, non-bone-connected branches close together (e.g.
        two gripper fingertips one index apart) may need a larger gap so
        those pairs -- close by construction, not by fault -- do not
        dominate the minimum distance.
    """

    name = "self_collision"
    units = "mm(min-dist)"
    higher_is_worse = False                                     # lower dist = worse

    def __init__(self, non_adjacent_gap: int = 2) -> None:
        super().__init__()
        self.non_adjacent_gap = int(non_adjacent_gap)

    def per_frame(self, ctx: ClipContext) -> np.ndarray:
        P = _single_clip_P(ctx)
        K = P.shape[1]
        pairs = [(i, j) for i in range(K) for j in range(i + self.non_adjacent_gap, K)]
        if not pairs:
            return np.full(len(P), np.inf)
        d = torch.stack(
            [(P[:, i] - P[:, j]).norm(dim=-1) for i, j in pairs], dim=1,
        ) * 1000.0
        return d.amin(dim=1).numpy()                            # (T,) closest non-adjacent pair mm
