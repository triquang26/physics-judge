"""``view_id -> panel geometry``, read from ``configs/views.yaml``."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kinescore.core.clip import ViewLayout

__all__ = ["ViewSpec", "load_views", "DEFAULT_VIEWS_PATH"]

#: ``configs/views.yaml``, relative to the repository root.
DEFAULT_VIEWS_PATH = Path(__file__).resolve().parents[3] / "configs" / "views.yaml"


@dataclass(frozen=True)
class ViewSpec:
    """One packing: how many cameras, arranged how, at what panel size.

    Attributes
    ----------
    view_id:
        Name used in a ``cell_id`` and everywhere downstream.
    n_views:
        Cameras exposed to the model. Below ``n_panels`` when the packed
        frame carries a panel this view drops.
    packing:
        ``none`` (one whole frame), ``width``, ``height`` or ``grid2x2``.
    n_panels:
        Physical panels in the packed frame.
    panels:
        Which panel indices are exposed, in order. Empty means all of them.
    panel:
        Measured ``(width, height)`` of one panel in pixels, or ``None`` when
        the corpus is not fixed-size. Checked against the decoded frame.
    order:
        Camera names per exposed view, for readable attention/score output.
    """

    view_id: str
    n_views: int
    packing: str = "none"
    n_panels: int | None = None
    panels: tuple[int, ...] = ()
    panel: tuple[int, int] | None = None
    order: tuple[str, ...] = ()

    @property
    def panel_count(self) -> int:
        """Physical panels in the packed frame."""
        return self.n_panels if self.n_panels is not None else self.n_views

    @property
    def panel_indices(self) -> tuple[int, ...]:
        """Exposed views as panel indices into the packed frame, in order."""
        return self.panels if self.panels else tuple(range(self.panel_count))

    def layout(self, tokens_per_view: int | None = None) -> ViewLayout:
        """The :class:`~kinescore.core.clip.ViewLayout` this view describes."""
        return ViewLayout(
            n_views=self.n_views, order=self.order,
            tokens_per_view=tokens_per_view, packing=self.packing,
            n_panels=self.n_panels, panels=self.panels)

    def check_frame_size(self, width: int, height: int) -> None:
        """Raise if a decoded frame does not match the measured panel size.

        Raises
        ------
        ValueError
            If this view declares a panel size and the frame is not
            ``n_panels`` of them.
        """
        if self.panel is None:
            return
        pw, ph = self.panel
        n = self.panel_count
        expected = {
            "none": (pw, ph),
            "width": (pw * n, ph),
            "height": (pw, ph * n),
            "grid2x2": (pw * 2, ph * 2),
        }[self.packing]
        if (width, height) != expected:
            raise ValueError(
                f"view {self.view_id!r} expects a {expected[0]}x{expected[1]} "
                f"frame ({n} panels of {pw}x{ph}, packing={self.packing}), got "
                f"{width}x{height}")


def _view_from_entry(view_id: str, entry: dict[str, Any]) -> ViewSpec:
    unknown = set(entry) - {"n_views", "packing", "n_panels", "panels",
                            "panel", "order"}
    if unknown:
        raise ValueError(
            f"view {view_id!r} has unknown key(s) {sorted(unknown)}")
    panel = entry.get("panel")
    return ViewSpec(
        view_id=view_id,
        n_views=int(entry["n_views"]),
        packing=str(entry.get("packing", "none")),
        n_panels=None if entry.get("n_panels") is None else int(entry["n_panels"]),
        panels=tuple(int(p) for p in entry.get("panels", ())),
        panel=None if panel is None else (int(panel[0]), int(panel[1])),
        order=tuple(str(v) for v in entry.get("order", ())),
    )


def load_views(path: str | Path = DEFAULT_VIEWS_PATH) -> dict[str, ViewSpec]:
    """Read ``views.yaml`` into ``{view_id: ViewSpec}``."""
    doc = yaml.safe_load(Path(path).read_text()) or {}
    views = doc.get("views") or {}
    if not isinstance(views, dict):
        raise ValueError(f"{path}: `views` must be a mapping of view_id -> entry")
    out = {vid: _view_from_entry(vid, entry) for vid, entry in views.items()}
    for spec in out.values():
        spec.layout()
    return out
