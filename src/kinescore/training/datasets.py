"""RAM-flattening loader: every cached episode -> one big tensor pair.

Ported from ``Marionette-ciasc/scripts/train_judge.py``'s ``load_split``,
kept deliberately close to the original two-pass strategy (it is the fast
path -- see that function's own docstring, reproduced below) and its
``joint_source == "real"`` assertion. What changes:

* Reads :mod:`kinescore.training.cache`'s self-describing ``.pt`` files
  (``{"feat", "header"}``) instead of a bare tensor, and cross-checks every
  episode's :class:`~kinescore.training.cache.CacheHeader` against the rest
  of the split (same view layout, same token count, same embed width) before
  it ever reaches the flattened tensor -- a second, split-wide layer of the
  D4 guard on top of the one :func:`~kinescore.training.cache.load_cache`
  already applies per file.
* ``n_joints`` is read off the annotation's own array width instead of being
  hardcoded to Franka's 7 -- this loader is robot-agnostic; whatever the
  logged ``observation.state.joint_position`` width is becomes ``n_joints``.
* Gripper is optional: a split whose annotations carry no
  ``observation.state.gripper_position`` (a robot with no gripper, or
  logged separately) loads fine with ``gripper=None``.

Original docstring, preserved for the memory/IO reasoning it documents:

    Memory: the multiview cache (n_cams*P tokens) is ~3x the single-cam one,
    so a naive ``torch.cat`` of the per-episode list would momentarily hold
    BOTH the list and the concatenated output (~2x peak) and blow the SLURM
    cgroup cap. Instead we do a cheap mmap pass-1 to sum the total frame
    count, preallocate ONE output tensor, then pass-2 copies each episode in
    and frees it -- peak stays ~1x (the output), not 2x.
"""
from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from kinescore.training.cache import CacheHeader, assert_real_joint_source

__all__ = ["SplitData", "load_split"]

#: Default annotation keys, matching the source's DROID/Franka annotations.
DEFAULT_JOINT_KEY = "observation.state.joint_position"
DEFAULT_GRIPPER_KEY = "observation.state.gripper_position"


@dataclass(frozen=True)
class SplitData:
    """One split, fully flattened into RAM.

    Parameters
    ----------
    feats:
        ``(N, n_tokens, D)`` fp16 patch tokens, frames from every episode in
        the split concatenated on axis 0.
    q:
        ``(N, n_joints)`` fp32 joint-angle targets, aligned to ``feats`` via
        ``down_sample`` (frame ``t``'s target is joint row ``t*down_sample``,
        clipped to the logged range).
    gripper:
        ``(N, 1)`` fp32 gripper-opening targets, or ``None`` if no episode's
        annotation carried a gripper key.
    n_joints, view_layout_key, n_episodes, n_frames:
        Provenance the caller (typically
        :func:`kinescore.training.trainer.train_head`) uses to size/validate
        the head it constructs against this data.
    episode_ids:
        Episode identifiers in load order, for debugging a bad batch back to
        its source file.
    """

    feats: torch.Tensor
    q: torch.Tensor
    gripper: Optional[torch.Tensor]
    n_joints: int
    view_layout_key: str
    n_episodes: int
    n_frames: int
    episode_ids: tuple[str, ...]

    @property
    def n_tokens(self) -> int:
        return int(self.feats.shape[1])

    @property
    def embed_dim(self) -> int:
        return int(self.feats.shape[2])

    @property
    def has_gripper(self) -> bool:
        return self.gripper is not None


def _episode_sort_key(path: str):
    stem = os.path.splitext(os.path.basename(path))[0]
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def load_split(cache_root: str, annotation_root: str, split: str,
               down_sample: int, *, limit: int = 0,
               joint_key: str = DEFAULT_JOINT_KEY,
               gripper_key: str = DEFAULT_GRIPPER_KEY,
               expected_view_layout=None, progress=None) -> SplitData:
    """Load every cached, real-joint episode of one split into RAM.

    Parameters
    ----------
    cache_root:
        Root containing ``{split}/{ep}.pt`` cache files (see
        :func:`kinescore.training.cache.precompute_cache`).
    annotation_root:
        Root containing ``{split}/{ep}.json`` annotation files.
    split:
        Split name (``"train"``, ``"val"``, ...).
    down_sample:
        Joint logs run at ``down_sample`` times the video's frame rate, so
        video frame ``t``'s matching joint row is ``t * down_sample``
        (clipped to the logged range) -- e.g. ``down_sample=3`` for
        15 Hz-logged joints against 5 Hz video.
    limit:
        Cap the number of episodes loaded (0 = all), applied after sorting.
    joint_key, gripper_key:
        Annotation JSON keys for the joint-angle and gripper-opening arrays.
        Gripper is optional (see :class:`SplitData`); joint is not.
    expected_view_layout:
        If given, every episode's cache header is checked against it (see
        :meth:`~kinescore.training.cache.CacheHeader.assert_matches_layout`).
    progress:
        ``progress(message: str)`` for line-oriented status, or ``None``.

    Raises
    ------
    RuntimeError
        No cache file in the split has a matching annotation.
    ValueError
        Any episode's ``joint_source`` is not ``"real"`` (see
        :func:`~kinescore.training.cache.assert_real_joint_source`), or its
        cache header (view layout, token count, embed width) or joint-array
        width disagrees with the rest of the split, or gripper presence is
        inconsistent across episodes (a split cannot be partially flattened
        into one gripper tensor).
    """
    split_dir = os.path.join(cache_root, split)
    files = sorted(glob.glob(os.path.join(split_dir, "*.pt")), key=_episode_sort_key)
    if limit > 0:
        files = files[:limit]
    t0 = time.time()

    # ---- pass 1: mmap headers (no full-tensor RAM) -> total frames + labels --
    metas: list[tuple[str, int, np.ndarray, Optional[np.ndarray]]] = []
    total = 0
    n_joints: Optional[int] = None
    embed_dim: Optional[int] = None
    view_layout_key: Optional[str] = None
    has_gripper: Optional[bool] = None
    for fp in files:
        ep = os.path.splitext(os.path.basename(fp))[0]
        ap = os.path.join(annotation_root, split, f"{ep}.json")
        if not os.path.exists(ap):
            continue
        label = assert_real_joint_source(ap)  # raises on joint_source != "real"

        payload = torch.load(fp, map_location="cpu", mmap=True)
        header = CacheHeader.from_dict(payload["header"])
        if expected_view_layout is not None:
            header.assert_matches_layout(expected_view_layout, path=fp)
        if view_layout_key is None:
            view_layout_key = header.view_layout_key
        elif header.view_layout_key != view_layout_key:
            raise ValueError(
                f"episode {ep!r}: cache view_layout {header.view_layout_key!r} "
                f"!= this split's {view_layout_key!r} -- a split's caches must "
                f"share one camera layout (defect D4, one layer up).")

        feat = payload["feat"]
        L = int(feat.shape[0])
        if embed_dim is None:
            embed_dim = int(feat.shape[-1])
        elif int(feat.shape[-1]) != embed_dim:
            raise ValueError(
                f"episode {ep!r}: embed_dim {feat.shape[-1]} != split's "
                f"{embed_dim} -- caches were built with different backbones.")

        jp = np.asarray(label[joint_key], dtype=np.float32)  # (R, n_joints)
        if n_joints is None:
            n_joints = int(jp.shape[-1])
        elif int(jp.shape[-1]) != n_joints:
            raise ValueError(
                f"episode {ep!r}: joint_position width {jp.shape[-1]} != "
                f"split's {n_joints}")
        idx = np.clip(np.arange(L) * down_sample, 0, len(jp) - 1)
        jp_i = jp[idx]

        gp_i = None
        if gripper_key in label:
            gp = np.asarray(label[gripper_key], dtype=np.float32)
            gp_i = gp[idx]
        if has_gripper is None:
            has_gripper = gp_i is not None
        elif has_gripper != (gp_i is not None):
            raise ValueError(
                f"episode {ep!r}: gripper_position present={gp_i is not None} "
                f"disagrees with earlier episodes in this split "
                f"(has_gripper={has_gripper}) -- cannot flatten a partially-"
                f"gripper split into one tensor.")

        metas.append((fp, L, jp_i, gp_i))
        total += L
        del payload, feat

    if not metas:
        raise RuntimeError(
            f"no cached episode under {split_dir!r} has a matching annotation "
            f"under {os.path.join(annotation_root, split)!r}")

    # ---- pass 2: preallocate once, fill, free each episode -------------------
    feats: Optional[torch.Tensor] = None
    n_tokens: Optional[int] = None
    q = torch.empty((total, n_joints), dtype=torch.float32)
    gripper = torch.empty((total, 1), dtype=torch.float32) if has_gripper else None
    episode_ids: list[str] = []
    o = 0
    for i, (fp, L, jp_i, gp_i) in enumerate(metas):
        payload = torch.load(fp, map_location="cpu")  # real load
        t = payload["feat"]
        if feats is None:
            n_tokens = int(t.shape[1])
            feats = torch.empty((total, n_tokens, embed_dim), dtype=torch.float16)
        elif int(t.shape[1]) != n_tokens:
            raise ValueError(
                f"{fp}: token count {t.shape[1]} != split's {n_tokens} -- "
                f"inconsistent token grid across episodes despite matching "
                f"view_layout_key.")
        feats[o:o + L] = t
        q[o:o + L] = torch.from_numpy(jp_i)
        if gripper is not None:
            gripper[o:o + L] = torch.from_numpy(gp_i)[:, None]
        episode_ids.append(os.path.splitext(os.path.basename(fp))[0])
        o += L
        del payload, t
        if progress and (i + 1) % 200 == 0:
            progress(f"[{split}] filled {i + 1}/{len(metas)} eps "
                     f"({time.time() - t0:.0f}s)")

    if progress:
        progress(f"[{split}] {feats.shape[0]} frames  feat={tuple(feats.shape)}  "
                 f"({time.time() - t0:.0f}s)")

    return SplitData(feats=feats, q=q, gripper=gripper, n_joints=n_joints,
                     view_layout_key=view_layout_key or "", n_episodes=len(metas),
                     n_frames=total, episode_ids=tuple(episode_ids))
