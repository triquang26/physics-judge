"""Precompute frozen-backbone patch tokens to disk, self-describing."""
from __future__ import annotations

import glob
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch

from kinescore.backbones.default import backbone_id
from kinescore.core.clip import ViewLayout

__all__ = [
    "CACHE_SCHEMA_VERSION", "CacheHeader", "assert_real_joint_source",
    "encode_clip", "write_cache", "load_cache", "CacheBuilder",
]

#: Bumped when the on-disk header shape changes.
CACHE_SCHEMA_VERSION = 2

#: ``progress(message)`` for line-oriented status; ``None`` means silent.
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class CacheHeader:
    """Everything :func:`load_cache` checks before handing over tokens.

    Parameters
    ----------
    schema:
        On-disk schema version.
    reader_id:
        ``<robot>.<view_id>`` this cache was built for. A cache built for one
        reader cannot feed a head being trained for another, even when the
        two happen to share a token count.
    view_layout_key:
        Layout the tokens were encoded with.
    n_views:
        Redundant with ``view_layout_key``, kept separate so a header can be
        checked without parsing the key.
    tokens_per_view:
        Patch tokens one view contributes, or ``None`` when pooling leaves
        that unfixed.
    backbone_id:
        Identity of the backbone that produced the tokens.
    source_path:
        Clip this file was encoded from.
    n_frames, embed_dim:
        Shape of the tensor beside this header.
    """

    schema: int
    reader_id: str
    view_layout_key: str
    n_views: int
    tokens_per_view: int | None
    backbone_id: str
    source_path: str
    n_frames: int
    embed_dim: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CacheHeader:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def assert_matches(self, *, path: str = "", reader_id: str | None = None,
                       view_layout: ViewLayout | None = None,
                       backbone: str | None = None) -> None:
        """Raise if this header disagrees with what the caller expects.

        Raises
        ------
        ValueError
            Naming the file and both sides of the disagreement.
        """
        where = f" ({path})" if path else ""
        if reader_id is not None and self.reader_id != reader_id:
            raise ValueError(
                f"cache{where} was built for reader {self.reader_id!r}, but "
                f"{reader_id!r} is being trained")
        if view_layout is not None and self.n_views != view_layout.n_views:
            raise ValueError(
                f"cache{where} holds {self.n_views} view(s) "
                f"({self.view_layout_key!r}) but the caller expects "
                f"{view_layout.n_views} ({view_layout.key!r})")
        if backbone is not None and self.backbone_id != backbone:
            raise ValueError(
                f"cache{where} was encoded by {self.backbone_id!r} but the "
                f"caller reads through {backbone!r}")


def assert_real_joint_source(annotation_path: str) -> dict:
    """Load an annotation and require ``joint_source == "real"``.

    Raises
    ------
    ValueError
        If ``joint_source`` is anything but ``"real"``, or the joint array is
        missing.
    """
    with open(annotation_path) as f:
        label = json.load(f)
    source = label.get("joint_source")
    if source != "real":
        raise ValueError(
            f"{annotation_path}: joint_source={source!r}, expected 'real'")
    if "observation.state.joint_position" not in label:
        raise ValueError(
            f"{annotation_path}: missing observation.state.joint_position")
    return label


def encode_clip(backbone: Any, clip_path: str, *, view_layout: ViewLayout,
                max_frames: int = 0, device: str = "cpu",
                fps_arg: float | None = None, dt_arg: float | None = None,
                frame_chunk: int = 0):
    """Decode ``clip_path`` and encode it -> ``(T, n_tokens, D)`` fp16 on CPU.

    Parameters
    ----------
    backbone:
        Anything with ``encode(rgb) -> (N, V, P, D)``.
    clip_path:
        mp4 path or frame directory.
    view_layout:
        Camera packing of the clip.
    max_frames:
        Cap on decoded frames (``0`` = all).
    device:
        Where the frames are encoded.
    fps_arg, dt_arg:
        Mutually exclusive timebase overrides.
    frame_chunk:
        Encode at most this many frames per call, and move only that many to
        ``device``. Attention is quadratic in tokens per frame, so a long
        episode at full resolution can exhaust a large GPU in one batch.
        Frames are encoded independently, so chunking changes memory and time,
        never the numbers. Device memory is then bounded by this rather than
        by the clip's length.

    Returns
    -------
    (torch.Tensor, kinescore.core.clip.ClipSpec)
    """
    from kinescore.video.probe import resolve_timebase
    from kinescore.video.reader import load_rgb_u8

    clip = resolve_timebase(clip_path, fps_arg=fps_arg, dt_arg=dt_arg,
                            view_layout=view_layout)
    frames = load_rgb_u8(clip, max_frames=max_frames)
    n = frames.shape[0]
    step = n if frame_chunk <= 0 else frame_chunk
    feat = None
    with torch.no_grad():
        for i in range(0, n, step):
            batch = frames[i:i + step].to(device).float().div_(255.0)
            out = backbone.encode(batch).half().cpu()
            if feat is None:
                feat = torch.empty((n, *out.shape[1:]), dtype=out.dtype)
            feat[i:i + out.shape[0]] = out
    t, v, p, d = feat.shape
    return feat.reshape(t, v * p, d), clip


def write_cache(out_path: str, feat: torch.Tensor, header: CacheHeader) -> None:
    """Write one episode's tokens and header, creating parent directories."""
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save({"feat": feat, "header": header.as_dict()}, out_path)


def load_cache(path: str, *, reader_id: str | None = None,
               view_layout: ViewLayout | None = None,
               backbone: str | None = None, mmap: bool = False
               ) -> tuple[torch.Tensor, CacheHeader]:
    """Read a cache file, checking its header against what the caller expects.

    Parameters
    ----------
    mmap:
        Memory-map the token tensor instead of reading it into anonymous
        memory. One three-panel episode is a few hundred megabytes, so a
        caller holding a whole split resident cannot fit an ordinary
        allocation; mapped pages are page cache, which the kernel reclaims
        under pressure rather than the allocation being killed. The returned
        tensor is read-only, so slice it before writing.

    Raises
    ------
    ValueError
        If the file carries no header, or the header disagrees.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False,
                         mmap=mmap)
    if (not isinstance(payload, dict) or "header" not in payload
            or "feat" not in payload):
        raise ValueError(
            f"{path!r} is not a kinescore cache file: it has no "
            f"'header'/'feat' pair")
    header = CacheHeader.from_dict(payload["header"])
    header.assert_matches(path=path, reader_id=reader_id,
                          view_layout=view_layout, backbone=backbone)
    return payload["feat"], header


class CacheBuilder:
    """Encode every real-joint episode of a reader's train tree.

    Parameters
    ----------
    backbone:
        Anything with ``encode(rgb) -> (N, V, P, D)``.
    view_layout:
        Camera packing shared by every clip this builder reads.
    reader_id:
        Stamped into every header.
    """

    def __init__(self, backbone: Any, view_layout: ViewLayout,
                 reader_id: str) -> None:
        self.backbone = backbone
        self.view_layout = view_layout
        self.reader_id = reader_id
        self._backbone_id = backbone_id(backbone)

    def build_split(self, *, video_root: str, annotation_root: str,
                    out_root: str, split: str, pattern: str = "*.mp4",
                    limit: int = 0, device: str = "cpu",
                    overwrite: bool = False, max_frames: int = 0,
                    frame_chunk: int = 0,
                    progress: ProgressCallback | None = None) -> dict[str, int]:
        """Cache one split into ``out_root/split/{episode}.pt``.

        Returns
        -------
        dict
            ``{"n_done", "n_skipped_existing", "n_skipped_no_annotation"}``.
        """
        video_dir = os.path.join(video_root, split)
        files = sorted(glob.glob(os.path.join(video_dir, pattern)))
        if limit > 0:
            files = files[:limit]

        n_done = n_skip_existing = n_skip_no_ann = 0
        for clip_path in files:
            ep = os.path.splitext(os.path.basename(clip_path))[0]
            out_path = os.path.join(out_root, split, f"{ep}.pt")
            if os.path.exists(out_path) and not overwrite:
                n_skip_existing += 1
                continue

            ann_path = os.path.join(annotation_root, split, f"{ep}.json")
            if not os.path.exists(ann_path):
                n_skip_no_ann += 1
                continue
            assert_real_joint_source(ann_path)

            feat, _clip = encode_clip(
                self.backbone, clip_path, view_layout=self.view_layout,
                max_frames=max_frames, device=device, frame_chunk=frame_chunk)
            header = CacheHeader(
                schema=CACHE_SCHEMA_VERSION, reader_id=self.reader_id,
                view_layout_key=self.view_layout.key,
                n_views=self.view_layout.n_views,
                tokens_per_view=self.view_layout.tokens_per_view,
                backbone_id=self._backbone_id, source_path=clip_path,
                n_frames=int(feat.shape[0]), embed_dim=int(feat.shape[-1]))
            write_cache(out_path, feat, header)
            n_done += 1
            if progress and n_done % 25 == 0:
                progress(f"[{split}] {n_done}/{len(files)} encoded")

        if progress:
            progress(f"[{split}] {n_done} written, {n_skip_existing} already "
                     f"cached, {n_skip_no_ann} without annotation -> "
                     f"{os.path.join(out_root, split)}")
        return {"n_done": n_done, "n_skipped_existing": n_skip_existing,
                "n_skipped_no_annotation": n_skip_no_ann}
