"""Clip identity and the timebase.

:class:`ClipSpec` is the sole owner of ``dt``: nothing else stores a frame
interval and no detector carries a default one. ``dt`` never travels as a bare
float, and :meth:`ClipSpec.subsample` is the only supported way to decimate
frames -- it scales ``dt`` with the stride, so a loader cannot drop frames
without fixing up the timebase.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

__all__ = ["ViewLayout", "ClipSpec", "DtSource", "TimebaseError", "PackingMode"]

DtSource = Literal["ffprobe", "fps_arg", "dt_arg", "table", "synthetic"]

#: How the physical camera panels are arranged inside one packed frame.
PackingMode = Literal["none", "height", "width", "grid2x2"]
_PACKINGS: tuple[PackingMode, ...] = ("none", "height", "width", "grid2x2")

#: A packed panel this far outside a plausible single-camera aspect ratio is
#: almost certainly the wrong packing axis, not a real crop. A 960x192
#: width-stacked frame sliced as 3 height bands would be 960x64 (aspect 15.0);
#: its real panels are 320x192 width slices (aspect 1.67).
_MIN_PANEL_ASPECT = 0.2
_MAX_PANEL_ASPECT = 5.0


class TimebaseError(ValueError):
    """Raised when a clip's frame rate cannot be established or is inconsistent."""


@dataclass(frozen=True)
class ViewLayout:
    """How multiple camera views are packed into one frame / token sequence.

    A single-view clip is ``ViewLayout(n_views=1)`` -- the same code path as
    three views, not a special case.

    Two packings coexist and must stay consistent:

    * **pixel space** -- ``n_panels`` camera panels arranged per ``packing``
      (stacked on height or width, or a 2x2 grid); see :meth:`view_crops`.
    * **token space** -- per-view patch grids concatenated on the token axis,
      so ``n_tokens == n_views * tokens_per_view``; see :meth:`assert_tokens`.

    ``n_views`` is the number of views this layout *exposes* to a caller,
    which need not equal ``n_panels`` (the number of physical panels in the
    packed frame) -- see ``panels`` below for selecting a subset.

    Parameters
    ----------
    n_views:
        Number of camera views this layout exposes (>= 1).
    order:
        Human-readable view names in exposed order, e.g.
        ``("exterior_1", "exterior_2")``. Length must equal ``n_views`` when
        given.
    tokens_per_view:
        Patch tokens contributed by one view after backbone pooling, when known.
        ``None`` for pixel-space-only layouts.
    packing:
        Physical arrangement of panels in the packed frame.
    n_panels:
        Physical panel count in the packed frame, if different from
        ``n_views`` (a subset selection -- see ``panels``). ``None`` (the
        default) means "no subset": ``n_panels == n_views``.
    panels:
        Which physical panel indices (0-indexed in packing order) this
        layout's views map to, in order. Required, one entry per view, when
        ``n_panels != n_views``; empty (the default) means "every panel, in
        order" and is only valid when ``n_panels == n_views``.
    """

    n_views: int = 1
    order: tuple[str, ...] = ()
    tokens_per_view: int | None = None
    packing: PackingMode = "height"
    n_panels: int | None = None
    panels: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.n_views < 1:
            raise ValueError(f"n_views must be >= 1, got {self.n_views}")
        if self.order and len(self.order) != self.n_views:
            raise ValueError(
                f"order has {len(self.order)} names but n_views={self.n_views}")
        if self.tokens_per_view is not None and self.tokens_per_view < 1:
            raise ValueError("tokens_per_view must be >= 1 when given")
        if self.packing not in _PACKINGS:
            raise ValueError(f"packing must be one of {_PACKINGS}, got {self.packing!r}")
        if self.n_panels is not None and self.n_panels < 1:
            raise ValueError(f"n_panels must be >= 1, got {self.n_panels}")
        panel_count = self.panel_count
        if self.packing == "none" and panel_count != 1:
            raise ValueError(
                f"packing='none' means the frame is one whole panel, so it "
                f"has exactly 1 panel, got n_panels={panel_count}")
        if self.packing == "grid2x2" and panel_count != 4:
            raise ValueError(
                f"grid2x2 packing always has exactly 4 physical panels, "
                f"got n_panels={panel_count}")
        if self.panels:
            if len(self.panels) != self.n_views:
                raise ValueError(
                    f"panels has {len(self.panels)} entries but n_views="
                    f"{self.n_views} -- panels selects exactly the views "
                    f"this layout exposes, one panel index per view")
            if len(set(self.panels)) != len(self.panels):
                raise ValueError(f"panels must be distinct, got {self.panels}")
            for p in self.panels:
                if not (0 <= p < panel_count):
                    raise ValueError(
                        f"panel index {p} out of range for n_panels={panel_count}")
        elif panel_count != self.n_views:
            raise ValueError(
                f"n_panels={panel_count} != n_views={self.n_views} but no "
                f"`panels` subset was given -- a layout exposing fewer views "
                f"than physical panels must say explicitly which panels map "
                f"to which views")

    @property
    def panel_count(self) -> int:
        """Physical panel count in the packed frame (``n_panels`` or ``n_views``)."""
        return self.n_panels if self.n_panels is not None else self.n_views

    @property
    def panel_indices(self) -> tuple[int, ...]:
        """This layout's views, as panel indices into the packed frame, in order."""
        return self.panels if self.panels else tuple(range(self.panel_count))

    @property
    def is_subset(self) -> bool:
        """``True`` if this layout exposes only some of the packed frame's panels."""
        return bool(self.panels) or self.panel_count != self.n_views

    @property
    def key(self) -> str:
        """Stable identity string, stored in checkpoints and cache headers.

        A plain height stack keys as ``{n_views}x{tokens_per_view}:{names}``;
        any other packing or an explicit panel subset appends enough to
        disambiguate, since those describe genuinely different geometry.
        """
        names = "+".join(self.order) if self.order else "unnamed"
        tpv = self.tokens_per_view if self.tokens_per_view is not None else "?"
        base = f"{self.n_views}x{tpv}:{names}"
        if self.packing == "height" and self.n_panels is None and not self.panels:
            return base
        idx = ",".join(str(p) for p in self.panel_indices)
        return f"{base}:{self.packing}:{self.panel_count}p:{idx}"

    def _panel_size(self, frame_width: int, frame_height: int) -> tuple[int, int]:
        """Validate ``packing`` against the probed frame, return ``(h, w)`` per panel.

        Refuses rather than guesses: raises on a divisibility mismatch (the
        clip isn't the stack/grid this layout claims) *and* on an implausible
        panel aspect ratio when there is more than one panel. Divisibility
        alone is not enough -- a 960x192 width-stacked frame divides evenly
        into three height bands too, and those bands are meaningless.
        """
        n = self.panel_count
        if self.packing == "none":
            ph, pw = frame_height, frame_width
        elif self.packing == "height":
            if frame_height % n:
                raise ValueError(
                    f"frame height {frame_height} is not divisible by "
                    f"n_panels={n} for packing='height'")
            ph, pw = frame_height // n, frame_width
        elif self.packing == "width":
            if frame_width % n:
                raise ValueError(
                    f"frame width {frame_width} is not divisible by "
                    f"n_panels={n} for packing='width'")
            ph, pw = frame_height, frame_width // n
        else:  # grid2x2, n == 4 (enforced in __post_init__)
            if frame_height % 2 or frame_width % 2:
                raise ValueError(
                    f"{frame_width}x{frame_height} frame is not evenly "
                    f"divisible into a 2x2 grid")
            ph, pw = frame_height // 2, frame_width // 2
        if n > 1:
            aspect = pw / ph if ph else float("inf")
            if not (_MIN_PANEL_ASPECT <= aspect <= _MAX_PANEL_ASPECT):
                raise ValueError(
                    f"packing={self.packing!r} with n_panels={n} on a "
                    f"{frame_width}x{frame_height} frame implies a "
                    f"{pw}x{ph} panel (aspect {aspect:.2f}), outside the "
                    f"plausible single-camera range "
                    f"[{_MIN_PANEL_ASPECT}, {_MAX_PANEL_ASPECT}]. This is "
                    f"very likely the wrong packing axis declared for this "
                    f"frame, not a real camera crop -- refusing to slice it "
                    f"rather than guess.")
        return ph, pw

    def panel_box(self, panel_index: int, frame_width: int, frame_height: int
                 ) -> tuple[int, int, int, int]:
        """Pixel box ``(top, bottom, left, right)`` for one physical panel.

        ``panel_index`` is 0-indexed into the packed frame (not into this
        layout's possibly-subset ``order``) -- use :meth:`view_crops` to get
        the boxes for this layout's exposed views directly.
        """
        n = self.panel_count
        if not (0 <= panel_index < n):
            raise ValueError(f"panel_index {panel_index} out of range for n_panels={n}")
        ph, pw = self._panel_size(frame_width, frame_height)
        if self.packing == "height":
            top = panel_index * ph
            return (top, top + ph, 0, frame_width)
        if self.packing == "width":
            left = panel_index * pw
            return (0, frame_height, left, left + pw)
        row, col = divmod(panel_index, 2)  # grid2x2
        return (row * ph, (row + 1) * ph, col * pw, (col + 1) * pw)

    def view_crops(self, frame_width: int, frame_height: int
                   ) -> list[tuple[int, int, int, int]]:
        """Pixel boxes ``(top, bottom, left, right)``, one per exposed view, in order.

        The single place crop geometry is computed: callers never re-derive
        it, and a panel subset is applied here rather than by caller-side
        slicing. Raises via :meth:`_panel_size` if ``packing`` is inconsistent
        with the probed frame instead of silently dividing.
        """
        return [self.panel_box(p, frame_width, frame_height) for p in self.panel_indices]

    def assert_tokens(self, n_tokens: int) -> None:
        """Assert a token count matches this layout exactly.

        Catches both directions: a 3-view feature fed to a 1-view head and a
        1-view feature fed to a 3-view head. Token-space arithmetic depends
        only on ``n_views``/``tokens_per_view``, so this is independent of
        ``packing``.
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


#: ``ViewLayout`` is frozen, so one shared instance is a safe default.
_DEFAULT_VIEW_LAYOUT = ViewLayout()


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
    view_layout: ViewLayout = _DEFAULT_VIEW_LAYOUT
    stride: int = 1
    codec: str | None = None
    sha1: str | None = None

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
                 view_layout: ViewLayout = _DEFAULT_VIEW_LAYOUT,
                 codec: str | None = None,
                 sha1: str | None = None) -> ClipSpec:
        """Build a spec from a frame rate, deriving ``dt = 1/fps``."""
        if fps <= 0 or not math.isfinite(fps):
            raise TimebaseError(f"fps must be finite and > 0, got {fps}")
        return cls(path=path, fps=fps, dt=1.0 / fps, n_frames=n_frames,
                   width=width, height=height, dt_source=dt_source,
                   view_layout=view_layout, codec=codec, sha1=sha1)

    def subsample(self, k: int) -> ClipSpec:
        """Return the spec for every ``k``-th frame, with ``dt`` scaled by ``k``.

        The only supported way to decimate a clip. Frame-dropping anywhere
        else -- a slice in a loader, a stride in a dataset -- leaves the
        timebase describing frames that are no longer being scored.
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
