"""Video I/O: probing, decoding, controlled corruption, and camera-visibility.

Everything here is torch/imageio/cv2-facing and stays outside ``kinescore.core``
on purpose (see ``kinescore.core``'s layering docstring) -- the core contracts
must stay importable without a video decoding stack installed. Heavy imports
(``imageio``, ``cv2``) are deferred into the functions that need them so
``kinescore.video.probe`` (subprocess-only) stays lightweight even when the
``video`` extra isn't installed.
"""
from kinescore.video.anchor import build_anchor, probe_crf_context, reencode_anchor_clip
from kinescore.video.corruptions import VideoCorruptions
from kinescore.video.probe import ffprobe, resolve_timebase
from kinescore.video.reader import load_rgb
from kinescore.video.viewpoint import (
    WRIST_MOVING_FRAC_THRESHOLD,
    ViewpointVerdict,
    classify_viewpoint,
)

__all__ = [
    "ffprobe", "resolve_timebase", "load_rgb", "VideoCorruptions",
    "classify_viewpoint", "ViewpointVerdict", "WRIST_MOVING_FRAC_THRESHOLD",
    "build_anchor", "probe_crf_context", "reencode_anchor_clip",
]
