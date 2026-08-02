"""The façade: clip + reader + robot + suite -> one result record.

``Scorer`` is the single place the pieces meet, and the single place the
cross-checks live:

* the clip's camera layout must match the reader's,
* the reader's target robot must match the robot spec,
* ``dt`` comes from the clip and is passed explicitly to every metric,
* the reader's ``limit_semantics`` is published as a flag so a metric that is
  structurally incapable of firing under some head family can report ``NaN``
  with a reason rather than a misleading ``0`` (see ``core/metric.py``'s
  ``unobservable_when``) -- the head family this originally guarded against
  (a sigmoid-squashed reader, always exactly in-limits) has since been
  removed (``legacy_docs/PROVENANCE.md`` D7), so the flag is currently always
  ``"raw_rad"``,
* the ``rate_policy`` (``"paired"`` / ``"rate_free"`` / ``"resample:<hz>"``,
  see ``docs/BENCHMARKING.md``) gates which suite is legal and, for
  ``"resample:<hz>"``, actually resamples the trajectory -- see
  :meth:`Scorer.__init__` and :meth:`Scorer.score_readout`.

Nothing here decodes latents. The input contract is frames, which is what makes
the benchmark model-agnostic: anything that can write an mp4 can be scored.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from kinescore.core.clip import ClipSpec
from kinescore.core.metric import MetricContext
from kinescore.core.reader import PoseReader, Readout
from kinescore.core.resample import RatePolicy, parse_rate_policy, resample_readout
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
        """Flat, JSON-safe record. See ``legacy_docs/SCHEMA.md``."""
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
    rate_policy:
        One of ``"paired"`` (default), ``"rate_free"``, or
        ``"resample:<hz>"`` -- see ``docs/BENCHMARKING.md`` for the full
        argument. Parsed once, at construction, via
        :func:`~kinescore.core.resample.parse_rate_policy`:

        * ``"paired"`` changes nothing here -- it is the default because fps
          cancels within a gt/pred pair scored at the same rate, which
          ``bench/manifest.py::verify_manifest`` now enforces (its ``dt``
          check) rather than anything in this class.
        * ``"rate_free"`` is a **structural check at construction time, not a
          suite override**: every metric in ``suite`` must have
          ``dt_exponent == 0`` (checked against the metric objects the suite
          was actually built with, so this needs no import of
          ``kinescore.metrics.suites`` and works for any suite, not just
          ``RATE_FREE``), or construction raises ``ValueError`` naming the
          offending keys. This is "suite đầy đủ mà so chéo nhịp -> lỗi" from
          the design doc: a full (rate-dependent) suite scored under a
          cross-generator, cross-rate claim is refused up front rather than
          producing a silently-invalid number.
        * ``"resample:<hz>"`` defers to scoring time: :meth:`score_readout`
          resamples the readout (and the clip's timebase) to the target rate
          via :func:`~kinescore.core.resample.resample_readout` *before*
          forward kinematics, then scores the resampled trajectory. Upsampling
          is refused unless ``allow_upsample=True`` is also passed -- see
          ``core/resample.py``'s module docstring for why.
    allow_upsample:
        Forwarded to :func:`~kinescore.core.resample.resample_readout` when
        ``rate_policy`` is ``"resample:<hz>"``. Ignored otherwise. Default
        ``False`` -- upsampling invents frames and is opt-in only.
    """

    def __init__(self, robot: RobotSpec, reader: PoseReader,
                 suite: MetricSuite, gate: Any | None = None,
                 rate_policy: str = "paired",
                 allow_upsample: bool = False) -> None:
        if reader.robot_name != robot.name:
            raise ValueError(
                f"reader targets robot {reader.robot_name!r} but robot spec is "
                f"{robot.name!r}")
        self.robot = robot
        self.reader = reader
        self.suite = suite
        self.gate = gate
        self.rate_policy: RatePolicy = parse_rate_policy(
            rate_policy, allow_upsample=allow_upsample)
        if self.rate_policy.kind == "rate_free":
            not_rate_free = sorted(m.spec.key for m in suite.metrics
                                   if m.spec.dt_exponent != 0)
            if not_rate_free:
                raise ValueError(
                    f"rate_policy='rate_free' requires every metric in "
                    f"suite {suite.name!r} to have dt_exponent==0, but "
                    f"{len(not_rate_free)} do not: {not_rate_free}. Pass "
                    f"kinescore.metrics.suites.RATE_FREE as the suite, or "
                    f"use rate_policy='paired'/'resample:<hz>' instead.")

    @staticmethod
    def _reader_device(reader: PoseReader) -> torch.device | None:
        """Device the reader's weights live on, or ``None`` if undiscoverable.

        ``PoseReader`` is a protocol, not an ``nn.Module``, so there is no
        ``.device`` to ask for. Every concrete reader in this package does hold
        its head (and backbone) as attributes, so the device is recoverable
        from the first parameter found. Returning ``None`` rather than guessing
        ``"cpu"`` keeps a reader that genuinely has no parameters (a stub or
        test double) on the caller's original tensor, unmoved.
        """
        # `inner` first: the checkpoint-v2 loader wraps the real reader in
        # `ReadoutV2PoseReader`, whose own `head`/`backbone` attributes are
        # None -- looking only at the outer object silently finds nothing.
        candidates = (reader, getattr(reader, "inner", None))
        for base in candidates:
            if base is None:
                continue
            for attr in (None, "head", "backbone"):
                obj = base if attr is None else getattr(base, attr, None)
                if isinstance(obj, torch.nn.Module):
                    for p in obj.parameters():
                        return p.device
        return None

    def score(self, frames: torch.Tensor, clip: ClipSpec) -> ScoredClip:
        """Score decoded frames against their :class:`ClipSpec`.

        ``dt`` is taken from ``clip`` -- there is no ``dt`` parameter to forget.

        Frames are moved to the reader's own device first.
        :func:`kinescore.video.reader.load_rgb` decodes to CPU (it goes through
        ``imageio``/numpy, which have no notion of the reader's device), while
        ``--device cuda`` puts the reader's weights on the GPU. Nothing joined
        those two halves until the first end-to-end scoring run, so every clip
        failed with ``Expected all tensors to be on the same device``. This is
        the same class of gap as the frame-shape mismatch fixed in
        ``readers/_frames.py``, and it belongs here for the same reason: this
        method is already the single place that owns the decoder-to-reader
        handoff, alongside ``dt`` and the layout cross-check.
        """
        self._check_layout(clip)
        device = self._reader_device(self.reader)
        if device is not None and frames.device != device:
            frames = frames.to(device)
        readout = self.reader.read(frames)
        return self.score_readout(readout, clip)

    def score_readout(self, readout: Readout, clip: ClipSpec) -> ScoredClip:
        """Score an already-computed readout (lets callers cache the backbone).

        When ``rate_policy`` is ``"resample:<hz>"``, ``readout`` and ``clip``
        are both resampled to the target rate first (see
        :func:`~kinescore.core.resample.resample_readout`) -- forward
        kinematics and every metric then run on the resampled trajectory, and
        the returned :class:`ScoredClip` carries the resampled ``clip``
        (``dt_source="resampled"``), never the native one, so a reader of the
        output cannot mistake it for a natively-sampled clip.
        """
        coverage = 1.0
        if self.gate is not None and readout.sigma is not None:
            readout, coverage = self.gate.apply(readout)

        if self.rate_policy.kind == "resample":
            readout, clip = resample_readout(
                readout, clip, self.rate_policy.target_fps,
                allow_upsample=self.rate_policy.allow_upsample)

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
        # Raises if clip.view_layout's packing is inconsistent with the
        # probed frame (wrong divisibility, or an implausible panel aspect --
        # see ViewLayout._panel_size / legacy_docs/DECISIONS.md D-G) instead of
        # letting a mismatched clip reach the backbone at all.
        clip.view_layout.view_crops(frame_width=clip.width, frame_height=clip.height)
