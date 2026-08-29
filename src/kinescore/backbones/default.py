"""The one frozen backbone this benchmark reads through."""
from __future__ import annotations

from typing import Any

from kinescore.core.clip import ViewLayout

__all__ = ["BACKBONE_CFG", "build_backbone", "backbone_id"]

#: DINOv3 ViT-L/16 at 768 px with 2x2 patch pooling.
BACKBONE_CFG: dict[str, Any] = {
    "dino_model": "dinov3_vitl16",
    "hf_model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dino_input": 768,
    "patch_pool": 2,
    "embed_dim": 1024,
    "n_register": 4,
}


def build_backbone(view_layout: ViewLayout, *, device: str = "cpu",
                   overrides: dict[str, Any] | None = None):
    """Construct the frozen backbone in eval mode.

    Parameters
    ----------
    view_layout:
        Camera packing of the frames this backbone will read.
    device:
        Where the module lives.
    overrides:
        Fields to change from :data:`BACKBONE_CFG`. Anything overridden shows
        up in :func:`backbone_id`, so a cache built with it cannot be mistaken
        for a default one.
    """
    from kinescore.backbones.dino import FeatureBackbone

    cfg = {**BACKBONE_CFG, **(overrides or {})}
    return FeatureBackbone(view_layout=view_layout, **cfg).to(device).eval()


def backbone_id(backbone: Any) -> str:
    """``"{dino_model}@{dino_input}:p{patch_pool}"``.
    """
    return f"{backbone.dino_model}@{backbone.dino_input}:p{backbone.patch_pool}"
