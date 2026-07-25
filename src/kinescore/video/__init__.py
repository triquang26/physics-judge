"""Video I/O: probing, decoding, and controlled corruption.

Everything here is torch/imageio-facing and stays outside ``kinescore.core``
on purpose (see ``kinescore.core``'s layering docstring) -- the core contracts
must stay importable without a video decoding stack installed. Heavy imports
(``imageio``) are deferred into the functions that need them so
``kinescore.video.probe`` (subprocess-only) stays lightweight even when the
``video`` extra isn't installed.
"""
from kinescore.video.corruptions import VideoCorruptions
from kinescore.video.probe import ffprobe, resolve_timebase
from kinescore.video.reader import load_rgb

__all__ = ["ffprobe", "resolve_timebase", "load_rgb", "VideoCorruptions"]
