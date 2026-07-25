"""Clip identity and the timebase.

``ClipSpec`` is the **sole owner of ``dt``** in kinescore. Nothing else stores a
frame interval, and no metric carries a default one.

Why this is a whole module instead of a float argument
------------------------------------------------------
In the source benchmarks ``dt`` defaulted to ``0.2`` in three places and was
simply never passed at two call sites, while the training loader decimated
frames with a random stride of 1 or 2. Half the clips were therefore scored at a
``dt`` that was wrong by 2x, which inflates speed 2x, acceleration 4x and jerk
**8x** (measured; see ``docs/PROVENANCE.md`` D1). Ratio metrics partly hide it,
absolute ones do not.

The structural fix is that ``dt`` never travels as a bare float. It is a field of
this frozen dataclass, and :meth:`ClipSpec.subsample` is the *only* supported way
to decimate frames -- it scales ``dt`` for you. A loader cannot drop every other
frame without going through a code path that fixes up the timebase.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal, Optional

__all__ = ["ViewLayout", "ClipSpec", "DtSource", "TimebaseError"]

DtSource = Literal["ffprobe", "fps_arg", "dt_arg", "table", "synthetic"]


class TimebaseError(ValueError):
    """Raised when a clip's frame rate cannot be established or is inconsistent."""


@dataclass(frozen=True)
class ViewLayout:
    """How multiple camera views are packed into one frame / token sequence.

    kinescore treats multiview as a first-class property rather than an
    ``if n_cams > 1`` branch. A single-view clip is ``ViewLayout(n_views=1)``,
    which is the same code path as three views, not a special case.

    Two packings coexist in the sources and must stay consistent:

    * **pixel space** -- views stacked on the image *height* axis, so a 3-view
      192px-per-view clip is a 576px-tall frame. ``view_height`` is derived from
      the probed frame height, never declared.
    * **token space** -- per-view patch grids concatenated on the token axis, so
      ``n_tokens == n_views * tokens_per_view``.

    They agreed only by convention in the source, with no assertion; a 3-view
    cache fed to a 1-view head trained silently. :meth:`assert_tokens` is the
    check that turns that into an immediate error.

    Parameters
    ----------
    n_views:
        Number of camera views packed together (>= 1).
    order:
        Human-readable view names in packing order, e.g.
        ``("exterior_1", "exterior_2", "wrist_left")``. Length must equal
        ``n_views`` when given. Used for reporting and to catch reordering.
    tokens_per_view:
        Patch tokens contributed by one view after backbone pooling, when known.
        ``None`` for pixel-space-only layouts.
    """

    n_views: int = 1
    order: tuple[str, ...] = ()
    tokens_per_view: Optional[int] = None

    def __post_init__(self) -> None:
        if self.n_views < 1:
            raise ValueError(f"n_views must be >= 1, got {self.n_views}")
        if self.order and len(self.order) != self.n_views:
            raise ValueError(
                f"order has {len(self.order)} names but n_views={self.n_views}")
        if self.tokens_per_view is not None and self.tokens_per_view < 1:
            raise ValueError("tokens_per_view must be >= 1 when given")

    @property
    def key(self) -> str:
        """Stable identity string, stored in checkpoints and cache headers."""
        names = "+".join(self.order) if self.order else "unnamed"
        tpv = self.tokens_per_view if self.tokens_per_view is not None else "?"
        return f"{self.n_views}x{tpv}:{names}"

    def view_height(self, frame_height: int) -> int:
        """Per-view pixel height, derived from the actual frame height.

        Raises
        ------
        ValueError
            If ``frame_height`` is not divisible by ``n_views`` -- i.e. the clip
            is not the stack this layout claims it is.
        """
        if frame_height % self.n_views:
            raise ValueError(
                f"frame height {frame_height} is not divisible by n_views="
                f"{self.n_views}; the clip is not a {self.n_views}-view stack")
        return frame_height // self.n_views

    def assert_tokens(self, n_tokens: int) -> None:
        """Assert a token count matches this layout exactly.

        This is the check the source lacked. It catches both directions: a
        3-view feature fed to a 1-view head *and* a 1-view feature fed to a
        3-view head, instead of only the divisibility case.
        """
        if self.tokens_per_view is None:
            if n_tokens % self.n_views:
                raise ValueError(
                    f"token count {n_tokens} is not divisible by n_views="
                    f"{self.n_views}")
            return
        expected = self.n_views * self.tokens_per_view
        if n_tokens != expected:
            raise ValueError(
                f"token count {n_tokens} != n_views*tokens_per_view = "
                f"{self.n_views}*{self.tokens_per_view} = {expected}. "
                f"The features and the head disagree about the camera layout "
                f"(layout key {self.key!r}).")


@dataclass(frozen=True)
class ClipSpec:
    """One video clip to score, with its timebase attached.

    Parameters
    ----------
    path:
        Absolute path to the clip (mp4, or a directory of frames).
    fps:
        Frames per second actually used for scoring.
    dt:
        Seconds between consecutive scored frames. Always ``1.0 / fps``; kept
        as an explicit field because it is the quantity every metric consumes.
    n_frames, width, height:
        Geometry as probed.
    dt_source:
        Provenance of the timebase, recorded in every output row so a wrong
        rate is diagnosable after the fact rather than invisible.
    view_layout:
        Camera packing (see :class:`ViewLayout`).
    stride:
        Accumulated decimation factor relative to the source file. ``1`` means
        every frame; ``2`` means every other frame, with ``dt`` already doubled.
    codec, sha1:
        Optional provenance of the media itself.
    """

    path: str
    fps: float
    dt: float
    n_frames: int
    width: int
    height: int
    dt_source: DtSource = "ffprobe"
    view_layout: ViewLayout = ViewLayout()
    stride: int = 1
    codec: Optional[str] = None
    sha1: Optional[str] = None

    def __post_init__(self) -> None:
        validate_dt(self.dt)
        if self.fps <= 0 or not math.isfinite(self.fps):
            raise TimebaseError(f"fps must be finite and > 0, got {self.fps}")
        if abs(self.dt * self.fps - 1.0) > 1e-6:
            raise TimebaseError(
                f"dt={self.dt} and fps={self.fps} disagree "
                f"(dt*fps={self.dt * self.fps:.6f}, expected 1.0)")
        if self.n_frames < 0:
            raise ValueError(f"n_frames must be >= 0, got {self.n_frames}")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")

    @classmethod
    def from_fps(cls, path: str, fps: float, n_frames: int, width: int,
                 height: int, *, dt_source: DtSource = "ffprobe",
                 view_layout: ViewLayout = ViewLayout(),
                 codec: Optional[str] = None,
                 sha1: Optional[str] = None) -> "ClipSpec":
        """Build a spec from a frame rate, deriving ``dt = 1/fps``."""
        if fps <= 0 or not math.isfinite(fps):
            raise TimebaseError(f"fps must be finite and > 0, got {fps}")
        return cls(path=path, fps=fps, dt=1.0 / fps, n_frames=n_frames,
                   width=width, height=height, dt_source=dt_source,
                   view_layout=view_layout, codec=codec, sha1=sha1)

    def subsample(self, k: int) -> "ClipSpec":
        """Return the spec for every ``k``-th frame, with ``dt`` scaled by ``k``.

        **This is the only supported way to decimate a clip.** Frame-dropping
        anywhere else -- a slice in a loader, a stride in a dataset -- reproduces
        defect D1, because the timebase silently stops matching the frames.
        """
        if k < 1:
            raise ValueError(f"subsample factor must be >= 1, got {k}")
        if k == 1:
            return self
        return replace(self, fps=self.fps / k, dt=self.dt * k,
                       n_frames=(self.n_frames + k - 1) // k,
                       stride=self.stride * k)

    @property
    def duration_s(self) -> float:
        """Wall-clock duration of the scored frames."""
        return self.n_frames * self.dt

    def as_row(self) -> dict:
        """Flat dict for the manifest / result record (``clip.*`` columns)."""
        return {
            "path": self.path, "fps": self.fps, "dt": self.dt,
            "n_frames": self.n_frames, "width": self.width,
            "height": self.height, "dt_source": self.dt_source,
            "view_layout": self.view_layout.key, "stride": self.stride,
            "codec": self.codec, "sha1": self.sha1,
        }


def validate_dt(dt: float) -> float:
    """Validate a frame interval in seconds, returning it unchanged.

    Rejects non-finite, non-positive, and absurd values. The upper bound of 10s
    exists to catch the common mistake of passing an *fps* where a ``dt`` is
    expected (e.g. ``dt=30``), which would otherwise silently divide every
    derivative by 30 instead of multiplying.
    """
    if not isinstance(dt, (int, float)) or isinstance(dt, bool):
        raise TypeError(f"dt must be a float, got {type(dt).__name__}")
    dt = float(dt)
    if not math.isfinite(dt):
        raise ValueError(f"dt must be finite, got {dt}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if dt > 10.0:
        raise ValueError(
            f"dt={dt}s is implausibly large; did you pass fps instead of "
            f"1/fps? Use ClipSpec.from_fps(...) to derive dt from a frame rate.")
    return dt
