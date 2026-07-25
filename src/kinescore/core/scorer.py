"""The façade: clip + reader + robot + suite -> one result record.

``Scorer`` is the single place the pieces meet, and the single place the
cross-checks live:

* the clip's camera layout must match the reader's,
* the reader's target robot must match the robot spec,
* ``dt`` comes from the clip and is passed explicitly to every metric,
* the reader's ``limit_semantics`` is published as a flag so metrics that are
  structurally unable to fire under a squashed head report ``NaN`` with a reason
  rather than a misleading ``0``.

Nothing here decodes latents. The input contract is frames, which is what makes
the benchmark model-agnostic: anything that can write an mp4 can be scored.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from kinescore.core.clip import ClipSpec
from kinescore.core.metric import MetricContext
from kinescore.core.reader import PoseReader, Readout
from kinescore.core.robot import Capability, RobotSpec
from kinescore.core.suite import MetricSuite, SuiteResult

__all__ = ["Scorer", "ScoredClip"]


@dataclass(frozen=True)
class ScoredClip:
    """One clip's complete result: metrics plus enough provenance to trust them."""

    clip: ClipSpec
    result: SuiteResult
    robot: str
    reader_id: str
    limit_semantics: str
    n_frames_scored: int
    gate_coverage: float = 1.0

    def to_record(self) -> dict:
        """Flat, JSON-safe record. See ``docs/SCHEMA.md``."""
        return {
            "clip": self.clip.as_row(),
            "run": {
                "robot": self.robot,
                "reader_id": self.reader_id,
                "limit_semantics": self.limit_semantics,
                "suite_id": self.result.suite_id,
                "suite_name": self.result.suite_name,
            },
            "coverage": {
                "n_frames_scored": self.n_frames_scored,
                "gate_coverage": self.gate_coverage,
            },
            "metrics": self.result.scalars(),
            "metrics_unavailable": self.result.reasons(),
        }


class Scorer:
    """Compose a robot, a pose reader and a metric suite into a scorer.

    Parameters
    ----------
    robot:
        Embodiment providing FK and limits.
    reader:
        Pixels -> joint angles.
    suite:
        The fixed metric set; its ``suite_id`` stamps every result.
    gate:
        Optional confidence gate (heteroscedastic readers only). Frames whose
        predicted sigma exceeds the threshold are dropped before metrics run,
        and the surviving fraction is reported as ``gate_coverage`` so a clip
        scored on 30% of its frames is visibly different from one scored on all.
    """

    def __init__(self, robot: RobotSpec, reader: PoseReader,
                 suite: MetricSuite, gate: Optional[Any] = None) -> None:
        if reader.robot_name != robot.name:
            raise ValueError(
                f"reader targets robot {reader.robot_name!r} but robot spec is "
                f"{robot.name!r}")
        self.robot = robot
        self.reader = reader
        self.suite = suite
        self.gate = gate

    def score(self, frames: torch.Tensor, clip: ClipSpec) -> ScoredClip:
        """Score decoded frames against their :class:`ClipSpec`.

        ``dt`` is taken from ``clip`` -- there is no ``dt`` parameter to forget.
        """
        self._check_layout(clip)
        readout = self.reader.read(frames)
        return self.score_readout(readout, clip)

    def score_readout(self, readout: Readout, clip: ClipSpec) -> ScoredClip:
        """Score an already-computed readout (lets callers cache the backbone)."""
        n_total = readout.n_frames
        coverage = 1.0
        if self.gate is not None and readout.sigma is not None:
            readout, coverage = self.gate.apply(readout)

        P, R = self.robot.forward_transforms(readout.q, readout.aux)
        has_rot = Capability.ROTATIONS in self.robot.capabilities

        ctx = MetricContext(
            dt=clip.dt,
            P=P,
            R=R if has_rot else None,
            q=readout.q,
            q_raw=readout.q_raw,
            robot=self.robot,
            flags={"limit_semantics": self.reader.limit_semantics},
            aux=dict(readout.extras),
        )
        result = self.suite.evaluate(ctx)
        return ScoredClip(
            clip=clip, result=result, robot=self.robot.name,
            reader_id=self.reader.reader_id,
            limit_semantics=self.reader.limit_semantics,
            n_frames_scored=readout.n_frames,
            gate_coverage=coverage,
        )

    def _check_layout(self, clip: ClipSpec) -> None:
        want, got = self.reader.view_layout, clip.view_layout
        if want.n_views != got.n_views:
            raise ValueError(
                f"reader expects {want.n_views} view(s) ({want.key}) but clip "
                f"declares {got.n_views} ({got.key}). A multiview checkpoint "
                f"silently consuming single-view frames is defect D4.")
        clip.view_layout.view_height(clip.height)
