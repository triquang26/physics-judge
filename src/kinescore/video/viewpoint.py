"""Camera-visibility classifier: does a clip's camera actually show the arm?

A pose reader cannot read joint angles out of a frame the arm isn't in, and
if asked to it returns plausible numbers that sort, plot and mean nothing --
see ``legacy_docs/DECISIONS.md`` D-D. That question reduces to *is this a static
exterior camera or a wrist/ego camera that moves with the arm*, which is
answerable cheaply from pixels alone, without a trained model: a wrist/ego
camera translates+rotates with the arm every step, so the WHOLE frame
(including background) moves; a fixed exterior camera's background stays put
even while the arm itself moves quickly. :func:`classify_viewpoint` measures
exactly that with a whole-frame Farneback optical-flow discriminator.

Provenance of :data:`WRIST_MOVING_FRAC_THRESHOLD`: validated against 40
hand-labelled clips across 13 groups from ``dense/single_arm/singleview`` --
100% agreement with the visual label at this threshold (see ``TASKS.md``'s
"Camera visibility" section). That is a real, useful validation; it is not
exhaustive. Re-validate on a sample before trusting it on data outside that
population (a different embodiment, a much shakier or slower camera, etc).

``cv2`` (``opencv-python-headless``, the ``video`` extra) is imported lazily
inside the functions that need it, not at module scope, so
``import kinescore.video.viewpoint`` never requires it -- matching
``backbones/dino.py``'s lazy ``transformers`` import.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "Verdict", "ViewpointVerdict", "WRIST_MOVING_FRAC_THRESHOLD",
    "sample_gray_frames", "flow_features", "classify_viewpoint",
]

Verdict = Literal["exterior", "wrist"]

#: moving_frac at/above this is classified "wrist" (moves with the arm),
#: below it "exterior" (fixed background). See the module docstring for
#: validation provenance.
WRIST_MOVING_FRAC_THRESHOLD = 0.42

#: Per-pixel flow magnitude counted as "moving" when computing moving_frac.
_FLOW_MAG_THRESHOLD = 0.5
_DEFAULT_N_SAMPLES = 6
_DEFAULT_SIZE = (128, 96)  # (width, height), downsized before flow for speed


@dataclass(frozen=True)
class ViewpointVerdict:
    """One clip's camera-visibility classification.

    ``verdict`` and the two flow statistics are all ``None`` together when
    fewer than 2 frames could be sampled and decoded (nothing to compute flow
    between) -- surfaced explicitly rather than defaulting to a guess.
    ``moving_frac`` is exposed alongside ``verdict`` specifically so a caller
    can see how close to :data:`WRIST_MOVING_FRAC_THRESHOLD` a borderline
    clip landed, rather than only a bare boolean.
    """

    path: str
    verdict: Verdict | None
    moving_frac: float | None
    mean_mag: float | None
    n_pairs: int


def _sample_frame_indices(n_frames: int, n_samples: int) -> list[int]:
    if n_frames <= 0:
        return []
    n_samples = max(1, min(n_samples, n_frames))
    if n_samples == 1:
        return [0]
    step = (n_frames - 1) / (n_samples - 1)
    return sorted({round(i * step) for i in range(n_samples)})


def sample_gray_frames(path: str, *, n_samples: int = _DEFAULT_N_SAMPLES,
                       size: tuple[int, int] = _DEFAULT_SIZE) -> list:
    """Decode ``n_samples`` frames evenly spaced across ``path``, grayscale + resized.

    Returns a plain ``list`` of ``(size[1], size[0])`` uint8 arrays (possibly
    fewer than ``n_samples`` if some frame reads fail), never raises on a
    decode failure for an individual frame -- an unreadable file simply
    yields an empty or short list, which :func:`flow_features` and
    :func:`classify_viewpoint` handle by reporting ``n_pairs=0`` /
    ``verdict=None`` rather than crashing a batch classification run.
    """
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for idx in _sample_frame_indices(n_frames, n_samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
            frames.append(gray)
        return frames
    finally:
        cap.release()


def flow_features(frames: list, *, flow_mag_threshold: float = _FLOW_MAG_THRESHOLD
                  ) -> tuple[float | None, float | None, int]:
    """Dense optical flow between consecutive ``frames`` -> ``(mean_mag, moving_frac, n_pairs)``.

    ``mean_mag`` is the mean flow magnitude over the whole frame, averaged
    over consecutive pairs; ``moving_frac`` is the mean fraction of pixels
    per pair with flow magnitude above ``flow_mag_threshold``. Returns
    ``(None, None, 0)`` for fewer than 2 frames -- there is nothing to
    compute flow between.
    """
    import cv2
    import numpy as np

    if len(frames) < 2:
        return None, None, 0
    mags, moving = [], []
    for a, b in zip(frames[:-1], frames[1:], strict=True):
        flow = cv2.calcOpticalFlowFarneback(
            a, b, None, pyr_scale=0.5, levels=2, winsize=15,
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mags.append(float(mag.mean()))
        moving.append(float((mag > flow_mag_threshold).mean()))
    return float(np.mean(mags)), float(np.mean(moving)), len(mags)


def classify_viewpoint(path: str, *, n_samples: int = _DEFAULT_N_SAMPLES,
                       size: tuple[int, int] = _DEFAULT_SIZE,
                       threshold: float = WRIST_MOVING_FRAC_THRESHOLD
                      ) -> ViewpointVerdict:
    """Classify ``path`` as an ``"exterior"`` or ``"wrist"`` camera clip.

    Samples ``n_samples`` frames evenly across the clip (see
    :func:`sample_gray_frames`), computes whole-frame Farneback flow between
    consecutive samples (see :func:`flow_features`), and thresholds
    ``moving_frac`` at ``threshold`` (default :data:`WRIST_MOVING_FRAC_THRESHOLD`,
    see the module docstring for its validation).
    """
    frames = sample_gray_frames(path, n_samples=n_samples, size=size)
    mean_mag, moving_frac, n_pairs = flow_features(frames)
    verdict: Verdict | None = None
    if moving_frac is not None:
        verdict = "wrist" if moving_frac >= threshold else "exterior"
    return ViewpointVerdict(path=path, verdict=verdict, moving_frac=moving_frac,
                            mean_mag=mean_mag, n_pairs=n_pairs)
