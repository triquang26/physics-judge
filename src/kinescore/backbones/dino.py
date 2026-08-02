"""Frozen DINOv2/DINOv3 patch-token backbone.

Adapted from ``judge.pixel_judge.PixelPhysicsJudge._load_dino`` /
``_encode_one`` / ``_encode_pixels``. Two changes from the source, both
deliberate:

1. **No latent decode.** The source's ``encode_latent`` /
   ``encode_latent_grad`` took a duck-typed Stable-Video-Diffusion ``vae`` and
   decoded chunks of a *latent* to pixels before running DINO. That was the
   only place a diffusers-shaped object touched this codebase, and it made the
   judge implicitly coupled to one generator's latent layout (SVD's
   ``(T,4,24,40)``). kinescore's contract is frames in, joints out (see
   ``core/reader.py``), so those two methods are **deleted**, not adapted --
   any model that can render an mp4 can be scored, and this module never
   imports diffusers.
2. **View splitting goes through :class:`~kinescore.core.clip.ViewLayout`.**
   The source split a multiview frame stack with a bespoke
   ``H % self.n_cams`` check inside ``_encode_pixels``. That duplicated (and
   could silently disagree with) the same arithmetic done by the head and by
   the cache writer. Here ``encode`` calls ``view_layout.view_crops`` instead
   -- one implementation of "which pixels belong to which camera", covering
   every packing (height/width stack, 2x2 grid) and panel subset, shared by
   every caller.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from kinescore.backbones.pooling import pool_patch_tokens
from kinescore.core.clip import ViewLayout

__all__ = ["FeatureBackbone"]

# DINOv2/v3 normalise with ImageNet statistics and need a side length that is
# a multiple of their patch size (DINOv2: 14 -> 224=16*14; DINOv3: 16 ->
# 224=14*16).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_DINO_INPUT = 224

# DINOv3 is loaded via HuggingFace transformers (its torch.hub weights are
# 403-blocked); weights are cached under ~/.cache/huggingface/hub. DINOv3
# prepends 1 CLS + ``n_register`` register tokens before the patches.
_DINOV3_HF = {
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
}
_DINOV3_DEFAULTS = {"patch_size": 16, "n_register": 4}
_DINOV2_PATCH = 14

#: ``ViewLayout`` is a frozen dataclass, so one shared instance is a safe
#: default -- see ``kinescore.core.clip``'s ``_DEFAULT_VIEW_LAYOUT`` for why
#: this is a module-level singleton rather than a bare ``ViewLayout()`` call
#: in the signature below.
_DEFAULT_VIEW_LAYOUT = ViewLayout()


class FeatureBackbone(nn.Module):
    """Frozen DINOv2/DINOv3 patch-token encoder, lazily loaded.

    Parameters mirror the fields real checkpoints (``judge_v3l``,
    ``judge_v3l_mv``, ``judge_reward``) persist in their ``cfg`` dict, so a
    backbone can be reconstructed directly from a loaded checkpoint's cfg
    (see ``readers/checkpoint.py``).

    Parameters
    ----------
    dino_model:
        Hub entrypoint (``dinov2_vits14`` | ``dinov2_vitb14`` |
        ``dinov2_vitl14``) or an HF DINOv3 short name (``dinov3_vitl16`` |
        ``dinov3_vits16``).
    dino_repo_dir:
        Local clone of ``facebookresearch/dinov2`` (loaded with
        ``source='local'`` so no network is needed for DINOv2).
    embed_dim:
        Backbone embedding width. Must match ``dino_model``.
    dino_input:
        Square side length frames are resized to before the backbone.
    patch_pool:
        ``k`` for :func:`~kinescore.backbones.pooling.pool_patch_tokens`.
    view_layout:
        Camera packing of the *input* frame (pixel space). ``encode`` uses
        ``view_layout.view_crops`` to crop each camera before encoding it,
        instead of a hardcoded ``H % n_cams``.
    """

    def __init__(
        self,
        dino_model: str = "dinov2_vitb14",
        dino_repo_dir: str = "",
        embed_dim: int = 768,
        dino_input: int = _DINO_INPUT,
        patch_pool: int = 1,
        hf_model_id: str = "",
        patch_size: int = 0,
        n_register: int = 0,
        view_layout: ViewLayout = _DEFAULT_VIEW_LAYOUT,
    ) -> None:
        super().__init__()
        self.dino_model = dino_model
        self.dino_repo_dir = dino_repo_dir
        self.embed_dim = int(embed_dim)
        self.dino_input = int(dino_input)
        self.patch_pool = int(patch_pool)
        self.view_layout = view_layout

        self.is_dinov3 = dino_model.startswith("dinov3")
        if self.is_dinov3:
            self.hf_model_id = hf_model_id or _DINOV3_HF.get(dino_model, "")
            if not self.hf_model_id:
                raise ValueError(
                    f"unknown DINOv3 model '{dino_model}'; pass hf_model_id")
            self.patch_size = patch_size or _DINOV3_DEFAULTS["patch_size"]
            self.n_register = n_register or _DINOV3_DEFAULTS["n_register"]
        else:
            self.hf_model_id = ""
            self.patch_size = patch_size or _DINOV2_PATCH
            self.n_register = 0

        self.register_buffer(
            "_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer(
            "_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

        self._dino: nn.Module | None = None  # lazy (heavy)

    # ── frozen backbone (lazy) ────────────────────────────────────────────

    def _load_dino(self, device: str) -> nn.Module:
        """Load the frozen backbone (offline), in eval mode.

        DINOv3 -> HuggingFace ``transformers.AutoModel`` from the HF cache.
        DINOv2 -> local torch.hub clone + cached pretrain weights.

        ``transformers`` is imported here, not at module scope, so
        ``import kinescore.backbones`` (and everything that transitively
        imports it -- ``heads``, ``readers``) never requires ``transformers``
        to be installed. Only code paths that actually load a DINOv3 backbone
        pay that cost.
        """
        if self._dino is not None:
            return self._dino
        if self.is_dinov3:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from transformers import AutoModel

            model = AutoModel.from_pretrained(self.hf_model_id)
        else:
            repo = self.dino_repo_dir or os.path.expanduser(
                "~/.cache/torch/hub/facebookresearch_dinov2_main")
            try:
                model = torch.hub.load(repo, self.dino_model, source="local",
                                       pretrained=True, trust_repo=True)
            except Exception:
                # Offline fallback: build skeleton, load cached weights.
                model = torch.hub.load(repo, self.dino_model, source="local",
                                       pretrained=False, trust_repo=True)
                ckpt = os.path.expanduser(
                    f"~/.cache/torch/hub/checkpoints/{self.dino_model}_pretrain.pth")
                model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._dino = model
        return model

    def _encode_one(self, rgb: torch.Tensor) -> torch.Tensor:
        """Encode ONE camera crop ``(N,3,H,W)`` -> patch tokens ``(N,P,D)``.

        Resizes to the backbone input grid, applies ImageNet normalisation,
        runs the frozen backbone, strips CLS/register tokens (DINOv3 only)
        and patch-pools. fp16 output, matching the source (the backbone is
        frozen and only ever feeds an eval-mode head).
        """
        device = rgb.device
        dino = self._load_dino(str(device))
        x = F.interpolate(rgb, size=(self.dino_input, self.dino_input),
                          mode="bilinear", align_corners=False)
        x = (x - self._mean.to(device)) / self._std.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            if self.is_dinov3:
                hs = dino(pixel_values=x).last_hidden_state  # (N,1+reg+P,D)
                feat = hs[:, 1 + self.n_register:]
            else:
                feat = dino.forward_features(x)["x_norm_patchtokens"]
            feat = pool_patch_tokens(feat, self.patch_pool)
        return feat.half()

    @torch.no_grad()
    def encode(self, rgb: torch.Tensor) -> torch.Tensor:
        """``(N,3,H,W)`` in ``[0,1]`` -> patch tokens ``(N,V,P,D)`` fp16.

        ``H``/``W`` are the *packed* frame dims (``view_layout.n_views``
        cameras arranged per ``view_layout.packing`` -- height/width stack or
        a 2x2 grid, possibly a subset of more physical panels; see
        :class:`~kinescore.core.clip.ViewLayout`). Each camera crop is
        resized and encoded independently; the per-camera token grids come
        back stacked on a new axis ``V`` rather than concatenated into the
        token axis, so the caller (typically :class:`~kinescore.heads.views.
        ViewEmbedding` or a reader) decides how to merge ``V`` and ``P`` and
        can run :meth:`~kinescore.core.clip.ViewLayout.assert_tokens` on the
        result -- the source concatenated immediately and never checked.
        """
        H, W = rgb.shape[2], rgb.shape[3]
        # Raises if packing is inconsistent with the probed frame instead of
        # silently slicing the wrong axis.
        crops = self.view_layout.view_crops(frame_width=W, frame_height=H)
        feats = [self._encode_one(rgb[:, :, top:bottom, left:right])
                 for top, bottom, left, right in crops]
        return torch.stack(feats, dim=1)  # (N, V, P, D)
