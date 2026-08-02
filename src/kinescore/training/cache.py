"""Precompute frozen-backbone patch-token features to disk, self-describing.

Adapted from ``Marionette-ciasc/scripts/precompute_dino_cache.py``, with
three deliberate departures from that source:

1. **No diffusers, no latent decode.** The source decoded a Stable-Video-
   Diffusion VAE latent to pixels before running the frozen DINO backbone
   (``LatentDinoCache`` there wraps ``AutoencoderKLTemporalDecoder``). That
   coupled a feature-cache builder -- something that should work for *any*
   video, generated or real -- to one specific generator's latent layout.
   This module reads mp4 frames directly via
   :func:`kinescore.video.reader.load_rgb`, exactly like every other
   frame-consuming path in this package (see
   ``kinescore.backbones.dino``'s module docstring for the same call on the
   backbone side).
2. **Explicit arguments instead of a shared config object.** The source
   resolved six fields off one repo-wide ``config_loader.load_config()``
   object across its two scripts: ``svd_model_path`` (the VAE -- gone
   entirely now that there is no latent decode), ``dataset_root_path``,
   ``annotation_name``, ``dino_repo_dir``, ``down_sample`` and
   ``fk_ee_link``. None of those exist here; every one of the remaining five
   is a plain, explicit parameter on :func:`precompute_cache` /
   :class:`~kinescore.cli.cmd_cache`'s flags (``fk_ee_link`` is not this
   module's concern at all -- FK end-effector choice lives on the
   :class:`~kinescore.core.robot.RobotSpec`, not the cache builder).
3. **Self-describing cache files (fixes D4).** The source's
   ``torch.save(emb.cpu(), out_path)`` wrote a *bare tensor* -- no view
   count, no backbone identity, nothing. A 3-view multiview cache
   (``(T, 3*P, D)``) and a 1-view cache (``(T, P, D)``) are both just "a
   tensor of some shape" on disk; nothing stopped (and nothing here
   reproduces) a 1-view head silently consuming a 3-view cache's tokens by
   flattening across a dimension boundary it never checked. Every file this
   module writes carries a :class:`CacheHeader` alongside the tensor
   (``view_layout.key``, ``backbone_id``, ``tokens_per_view``, ``n_views``,
   the source clip path); :func:`load_cache` refuses (``ValueError``) to
   hand a caller tokens whose header disagrees with the
   :class:`~kinescore.core.clip.ViewLayout` that caller declares it expects.
"""
from __future__ import annotations

import glob
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch

from kinescore.core.clip import ViewLayout

__all__ = [
    "CACHE_SCHEMA_VERSION", "CacheHeader", "backbone_id", "build_backbone",
    "assert_real_joint_source", "encode_clip", "write_cache", "load_cache",
    "CacheBuilder", "precompute_cache",
]

#: Bumped whenever the on-disk header shape changes -- see
#: ``kinescore.bench.store.SCHEMA_VERSION`` for why every persisted format in
#: this codebase carries one of these.
CACHE_SCHEMA_VERSION = 1

#: Called as ``progress(message: str)`` for line-oriented status; ``None`` (the
#: default everywhere below) means silent.
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class CacheHeader:
    """Everything :func:`load_cache` needs to check a cache file before use.

    Parameters
    ----------
    schema:
        On-disk schema version, see :data:`CACHE_SCHEMA_VERSION`.
    view_layout_key:
        :attr:`~kinescore.core.clip.ViewLayout.key` of the layout the tokens
        were encoded with -- the single field the source's cache format
        lacked entirely (defect D4, see module docstring).
    n_views:
        Redundant with ``view_layout_key`` (which embeds it) but kept as its
        own field so a header can be sanity-checked without re-parsing the
        key string.
    tokens_per_view:
        Patch tokens contributed by one view, or ``None`` if the backbone's
        pooling makes that not fixed ahead of time. ``feat.shape[-2] ==
        n_views * tokens_per_view`` when both are known.
    backbone_id:
        Stable identity string for the backbone config that produced the
        tokens (see :func:`backbone_id`) -- so a cache built with
        ``patch_pool=1`` cannot be silently fed to a head trained expecting
        ``patch_pool=2``'s (differently pooled, differently shaped) tokens.
    source_path:
        The clip this cache file was encoded from, for provenance.
    n_frames:
        Frame count the cache actually holds (``feat.shape[0]``).
    embed_dim:
        Backbone embedding width (``feat.shape[-1]``).
    """

    schema: int
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

    def assert_matches_layout(self, view_layout: ViewLayout, *, path: str = "") -> None:
        """Raise if this header's view layout disagrees with ``view_layout``.

        This is the D4 guard: a 3-view cache handed to code that declares a
        1-view :class:`~kinescore.core.clip.ViewLayout` (or vice versa) fails
        here, loudly, naming both layouts and the file -- instead of the two
        silently disagreeing the way a bare tensor always let them.
        """
        if self.n_views != view_layout.n_views:
            where = f" ({path})" if path else ""
            raise ValueError(
                f"cache{where} was written with {self.n_views} view(s) "
                f"({self.view_layout_key!r}) but the caller expects "
                f"{view_layout.n_views} ({view_layout.key!r}) -- a multiview "
                f"cache silently feeding a single-view head is defect D4.")


def backbone_id(backbone: Any) -> str:
    """Stable identity string for a :class:`~kinescore.backbones.dino.FeatureBackbone` config.

    ``"{dino_model}@{dino_input}:p{patch_pool}"``, e.g.
    ``"dinov3_vitl16@224:p2"``. Two backbones with the same id produce
    token-compatible output (same patch grid, same pooling); this is what a
    cache header's ``backbone_id`` is compared against when a caller wants to
    verify a cache matches the backbone it is about to (or did) train a head
    against.
    """
    return f"{backbone.dino_model}@{backbone.dino_input}:p{backbone.patch_pool}"


def build_backbone(*, dino_model: str = "dinov3_vitl16", embed_dim: int = 1024,
                   dino_input: int = 224, patch_pool: int = 2,
                   dino_repo_dir: str = "", view_layout: ViewLayout | None = None,
                   device: str = "cpu"):
    """Construct a frozen :class:`~kinescore.backbones.dino.FeatureBackbone`, eval mode.

    A thin, explicit-argument wrapper (no config object -- see module
    docstring point 2) so :func:`precompute_cache` and
    ``kinescore.cli.cmd_cache`` share one construction path.
    """
    from kinescore.backbones.dino import FeatureBackbone

    if view_layout is None:
        view_layout = ViewLayout()  # frozen dataclass -- no mutable-default hazard
    backbone = FeatureBackbone(
        dino_model=dino_model, dino_repo_dir=dino_repo_dir, embed_dim=embed_dim,
        dino_input=dino_input, patch_pool=patch_pool, view_layout=view_layout)
    return backbone.to(device).eval()


def assert_real_joint_source(annotation_path: str) -> dict:
    """Load an episode's annotation JSON and assert ``joint_source == "real"``.

    Ported from ``LatentDinoCache._assert_real_joints``: a synthetic-joint
    episode must never enter the pose-head's training cache, because the
    head is trained (and evaluated) against logged real joints -- caching a
    synthetic one would silently contaminate that signal, not just the cache
    format. Unlike the source (which returned ``False`` to skip a missing
    file), this only handles the "found but wrong" case; call-site discovery
    (:func:`precompute_cache`) is what decides a missing annotation means
    "skip this episode."

    Raises
    ------
    ValueError
        If ``joint_source`` is present but not ``"real"``, or the joint
        position field is missing entirely.
    """
    with open(annotation_path) as f:
        label = json.load(f)
    source = label.get("joint_source")
    if source != "real":
        raise ValueError(
            f"{annotation_path}: joint_source={source!r} (expected 'real'); "
            f"the pose head is trained on real joints only.")
    if "observation.state.joint_position" not in label:
        raise ValueError(
            f"{annotation_path}: missing observation.state.joint_position")
    return label


def encode_clip(backbone: Any, clip_path: str, *, view_layout: ViewLayout,
                max_frames: int = 0, device: str = "cpu",
                fps_arg: float | None = None, dt_arg: float | None = None,
                frame_chunk: int = 0):
    """Decode ``clip_path`` and run the frozen backbone -> ``(T, n_tokens, D)`` fp16.

    Parameters
    ----------
    backbone:
        A :class:`~kinescore.backbones.dino.FeatureBackbone` (or anything
        with the same ``.encode(rgb) -> (N,V,P,D)`` contract).
    clip_path:
        mp4 path or frame directory (see
        :func:`kinescore.video.reader.load_rgb`).
    view_layout:
        Camera packing of ``clip_path`` -- forwarded to
        :func:`~kinescore.video.probe.resolve_timebase` (which validates the
        frame height is divisible into this many views) and to the backbone.
    fps_arg, dt_arg:
        Optional cross-checked timebase override, forwarded to
        :func:`~kinescore.video.probe.resolve_timebase` -- mutually
        exclusive; see that function.
    frame_chunk:
        Split the episode's frames into chunks of at most this many before
        calling ``backbone.encode`` on each, concatenating the results.
        ``0`` (the default) passes every frame through in one call, exactly
        the prior (unchunked) behaviour. This exists because the transformer
        backbone's self-attention is quadratic in *tokens per frame*
        (``dino_input=768`` gives ~2300 tokens/frame for DINOv3-L, versus
        ~200 at the historical default of 224), so a long episode (DROID
        clips run up to several hundred frames) encoded in one batch at high
        resolution can exhaust even an 80GB GPU that is otherwise nearly
        idle -- observed OOMing a shared GPU at ``dino_input=768`` on a
        424-frame droid_std episode. Each frame is encoded independently (no
        batch-norm or other cross-sample coupling anywhere in
        :class:`~kinescore.backbones.dino.FeatureBackbone`), so chunking
        changes only memory/time, never the numeric result -- callers that
        never hit this ceiling (the default, and every existing caller) see
        no behaviour change at all.

    Returns
    -------
    (torch.Tensor, kinescore.core.clip.ClipSpec)
        The flattened token tensor ``(T, n_views*P, D)`` fp16 on CPU, and the
        :class:`~kinescore.core.clip.ClipSpec` it was decoded with (for the
        caller to read ``n_frames``/``dt``/provenance off of).
    """
    from kinescore.video.probe import resolve_timebase
    from kinescore.video.reader import load_rgb

    clip = resolve_timebase(clip_path, fps_arg=fps_arg, dt_arg=dt_arg,
                            view_layout=view_layout)
    frames = load_rgb(clip, max_frames=max_frames).to(device)
    with torch.no_grad():
        if frame_chunk and frame_chunk > 0 and frames.shape[0] > frame_chunk:
            chunks = [backbone.encode(frames[i:i + frame_chunk])
                     for i in range(0, frames.shape[0], frame_chunk)]
            feat = torch.cat(chunks, dim=0)  # (T, V, P, D)
        else:
            feat = backbone.encode(frames)  # (T, V, P, D)
    T, V, P, D = feat.shape
    return feat.reshape(T, V * P, D).half().cpu(), clip


def write_cache(out_path: str, feat: torch.Tensor, header: CacheHeader) -> None:
    """Write one episode's tokens + :class:`CacheHeader` to ``out_path``.

    ``torch.save`` of a ``{"feat", "header"}`` dict (not a bare tensor -- see
    module docstring point 3). Parent directories are created.
    """
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save({"feat": feat, "header": header.as_dict()}, out_path)


def load_cache(path: str, *, expected_view_layout: ViewLayout | None = None
              ) -> tuple[torch.Tensor, CacheHeader]:
    """Read a cache file written by :func:`write_cache`.

    Parameters
    ----------
    path:
        A ``.pt`` file written by :func:`write_cache`.
    expected_view_layout:
        If given, checked against the loaded header via
        :meth:`CacheHeader.assert_matches_layout` before returning -- this is
        the D4 guard's call site, e.g. a training loop passing the
        :class:`~kinescore.core.clip.ViewLayout` its head was constructed
        with.

    Raises
    ------
    ValueError
        If ``path`` has no ``"header"`` at all (not a file this module
        wrote -- e.g. the source's bare-tensor cache format, which is exactly
        the format that made D4 possible), or if ``expected_view_layout`` is
        given and disagrees with the header.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "header" not in payload or "feat" not in payload:
        raise ValueError(
            f"{path!r} is not a self-describing kinescore cache file (missing "
            f"'header'/'feat'); it may be a bare-tensor cache from the legacy "
            f"format this module's docstring (defect D4) replaces.")
    header = CacheHeader.from_dict(payload["header"])
    feat = payload["feat"]
    if expected_view_layout is not None:
        header.assert_matches_layout(expected_view_layout, path=path)
    return feat, header


class CacheBuilder:
    """Precompute one frozen backbone's tokens for every real-joint episode.

    Holds the two things every split of one caching run shares --
    ``backbone`` and ``view_layout`` -- as instance state, so
    :meth:`build_split` (called once per split: val, train, ...) does not
    repeat them at every call site. This is the class-based counterpart of
    the free-function :func:`precompute_cache`, which now just constructs
    one of these and calls :meth:`build_split` -- see that function's
    docstring for why it still exists (``kinescore.cli.cmd_cache`` and
    other pre-existing callers import the function directly).

    Parameters
    ----------
    backbone:
        A :class:`~kinescore.backbones.dino.FeatureBackbone` (or anything
        with the same ``.encode(rgb) -> (N,V,P,D)`` contract), typically
        built via :func:`build_backbone`.
    view_layout:
        Camera packing shared by every clip this builder will read -- see
        :func:`encode_clip`.
    """

    def __init__(self, backbone: Any, view_layout: ViewLayout) -> None:
        self.backbone = backbone
        self.view_layout = view_layout
        self._backbone_id = backbone_id(backbone)

    def build_split(self, *, video_root: str, annotation_root: str, out_root: str,
                    split: str, pattern: str = "*.mp4", limit: int = 0,
                    device: str = "cpu", overwrite: bool = False,
                    max_frames: int = 0, frame_chunk: int = 0,
                    progress: ProgressCallback | None = None) -> dict[str, int]:
        """Cache every real-joint episode of one split -> ``out_root/split/{ep}.pt``.

        Layout (mirrors the source's ``{cache_root}/{split}/{ep}.pt``,
        generalised to explicit roots -- see module docstring point 2):

        * clips:       ``video_root/split/{ep}.mp4`` (matched by ``pattern``)
        * annotations: ``annotation_root/split/{ep}.json``, with
          ``joint_source == "real"`` (see :func:`assert_real_joint_source`)

        An episode is skipped (not an error) when its annotation is missing
        entirely -- mirroring the source's "no counterpart, nothing to align
        against" behaviour -- but **raises** when the annotation exists with
        the wrong ``joint_source``, since that is caching-the-wrong-thing,
        not an absent pairing.

        Parameters
        ----------
        frame_chunk:
            Forwarded to :func:`encode_clip` -- ``0`` (default) is the
            original unchunked behaviour; see that function's docstring for
            why a long episode at a high ``dino_input`` may need this.

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
            assert_real_joint_source(ann_path)  # raises on a wrong joint_source

            feat, clip = encode_clip(self.backbone, clip_path,
                                     view_layout=self.view_layout,
                                     max_frames=max_frames, device=device,
                                     frame_chunk=frame_chunk)
            header = CacheHeader(
                schema=CACHE_SCHEMA_VERSION, view_layout_key=self.view_layout.key,
                n_views=self.view_layout.n_views,
                tokens_per_view=self.view_layout.tokens_per_view,
                backbone_id=self._backbone_id, source_path=clip_path,
                n_frames=int(feat.shape[0]), embed_dim=int(feat.shape[-1]))
            write_cache(out_path, feat, header)
            n_done += 1
            if progress and n_done % 50 == 0:
                progress(f"[{split}] {n_done} done, {n_skip_existing} skipped "
                         f"(existing), {n_skip_no_ann} skipped (no annotation)")

        if progress:
            progress(f"[{split}] cache done: {n_done} written, "
                     f"{n_skip_existing} already cached, "
                     f"{n_skip_no_ann} had no annotation -> "
                     f"{os.path.join(out_root, split)}")
        return {"n_done": n_done, "n_skipped_existing": n_skip_existing,
               "n_skipped_no_annotation": n_skip_no_ann}


def precompute_cache(*, video_root: str, annotation_root: str, out_root: str,
                     split: str, backbone: Any, view_layout: ViewLayout,
                     pattern: str = "*.mp4", limit: int = 0, device: str = "cpu",
                     overwrite: bool = False, max_frames: int = 0,
                     frame_chunk: int = 0,
                     progress: ProgressCallback | None = None) -> dict[str, int]:
    """Thin function wrapper around ``CacheBuilder(backbone, view_layout).build_split(...)``.

    Kept for existing callers (``kinescore.cli.cmd_cache``) that import this
    function directly; behaviour and signature are unchanged from before
    :class:`CacheBuilder` existed. New code -- especially anything caching
    more than one split against the same backbone -- should construct a
    :class:`CacheBuilder` once and call :meth:`~CacheBuilder.build_split` per
    split instead, so ``backbone_id(backbone)`` is computed once rather than
    once per split.
    """
    return CacheBuilder(backbone, view_layout).build_split(
        video_root=video_root, annotation_root=annotation_root, out_root=out_root,
        split=split, pattern=pattern, limit=limit, device=device,
        overwrite=overwrite, max_frames=max_frames, frame_chunk=frame_chunk,
        progress=progress)
