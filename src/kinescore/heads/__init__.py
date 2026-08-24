"""The trained head: DINO patch tokens -> 3-D keypoints."""
from kinescore.heads.diffusion import DiffusionKeypointHead, WorkspaceNormalizer
from kinescore.heads.keypoint import (
    KeypointHead,
    KeypointQueryDecoder,
    TemporalEncoder,
)

#: Either head: both read ``(B, T, P, D)`` tokens as ``(B, T, K, 3)`` metres.
AnyKeypointHead = KeypointHead | DiffusionKeypointHead

__all__ = ["AnyKeypointHead", "DiffusionKeypointHead", "KeypointHead",
           "KeypointQueryDecoder", "TemporalEncoder", "WorkspaceNormalizer"]
