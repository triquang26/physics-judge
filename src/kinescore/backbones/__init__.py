"""Frozen vision backbones: frames -> patch tokens.

Importing this package never requires ``transformers`` -- see
:mod:`kinescore.backbones.dino` -- only calling
:meth:`~kinescore.backbones.dino.FeatureBackbone.encode` on a DINOv3 model
does.
"""
from kinescore.backbones.dino import FeatureBackbone
from kinescore.backbones.pooling import pool_patch_tokens

__all__ = ["FeatureBackbone", "pool_patch_tokens"]
