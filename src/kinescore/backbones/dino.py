"""Frozen DINOv3 patch-token backbone."""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from kinescore.backbones.pooling import pool_patch_tokens
from kinescore.core.clip import ViewLayout

__all__ = ["FeatureBackbone"]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

#: Short name -> HuggingFace repo. Weights come from the HF cache; the
#: torch.hub mirror for DINOv3 is not publicly readable.
_DINOV3_HF = {
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
}
_PATCH_SIZE = 16
_N_REGISTER = 4

#: ``ViewLayout`` is frozen, so one shared instance is a safe default.
_DEFAULT_VIEW_LAYOUT = ViewLayout()


class FeatureBackbone(nn.Module):
    """Frozen DINOv3 patch-token encoder, lazily loaded.

    Parameters
    ----------
    dino_model:
        A key of :data:`_DINOV3_HF`.
    hf_model_id:
        Overrides the repo the short name maps to.
    embed_dim:
        Backbone embedding width. Must match ``dino_model``.
    dino_input:
        Square side length each camera crop is resized to.
    patch_pool:
        ``k`` for :func:`~kinescore.backbones.pooling.pool_patch_tokens`.
    view_layout:
        Camera packing of the *input* frame, in pixel space. ``encode`` crops
        each camera out of the packed frame before encoding it.
    """

    def __init__(
        self,
        dino_model: str = "dinov3_vitl16",
        embed_dim: int = 1024,
        dino_input: int = 768,
        patch_pool: int = 2,
        hf_model_id: str = "",
        patch_size: int = 0,
        n_register: int = 0,
        view_layout: ViewLayout = _DEFAULT_VIEW_LAYOUT,
    ) -> None:
        super().__init__()
        self.dino_model = dino_model
        self.embed_dim = int(embed_dim)
        self.dino_input = int(dino_input)
        self.patch_pool = int(patch_pool)
        self.view_layout = view_layout

        self.hf_model_id = hf_model_id or _DINOV3_HF.get(dino_model, "")
        if not self.hf_model_id:
            raise ValueError(
                f"unknown backbone {dino_model!r}; known: "
                f"{sorted(_DINOV3_HF)} (or pass hf_model_id)")
        self.patch_size = patch_size or _PATCH_SIZE
        self.n_register = n_register or _N_REGISTER

        self.register_buffer(
            "_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer(
            "_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

        self._dino: nn.Module | None = None

    def _load_dino(self, device: str) -> nn.Module:
        """Load the frozen backbone from the HF cache, in eval mode."""
        if self._dino is not None:
            return self._dino
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import AutoModel

        model = AutoModel.from_pretrained(self.hf_model_id).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._dino = model
        return model

    def _encode_one(self, rgb: torch.Tensor) -> torch.Tensor:
        """Encode one camera crop ``(N,3,H,W)`` into patch tokens ``(N,P,D)``.
        """
        device = rgb.device
        dino = self._load_dino(str(device))
        x = F.interpolate(rgb, size=(self.dino_input, self.dino_input),
                          mode="bilinear", align_corners=False)
        x = (x - self._mean.to(device)) / self._std.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            hs = dino(pixel_values=x).last_hidden_state  # (N, 1+reg+P, D)
            feat = pool_patch_tokens(hs[:, 1 + self.n_register:],
                                     self.patch_pool)
        return feat.half()

    @torch.no_grad()
    def encode(self, rgb: torch.Tensor) -> torch.Tensor:
        """``(N,3,H,W)`` in ``[0,1]`` -> patch tokens ``(N,V,P,D)`` fp16."""
        H, W = rgb.shape[2], rgb.shape[3]
        crops = self.view_layout.view_crops(frame_width=W, frame_height=H)
        feats = [self._encode_one(rgb[:, :, top:bottom, left:right])
                 for top, bottom, left, right in crops]
        return torch.stack(feats, dim=1)  # (N, V, P, D)
