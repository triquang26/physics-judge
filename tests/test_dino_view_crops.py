"""``backbones.dino.FeatureBackbone.encode`` -- crop geometry per ``ViewLayout``.

CPU-only, offline: ``_encode_one`` (the only method that touches the real
frozen DINO weights) is monkeypatched to record what it was called with and
return a small deterministic tensor, so these tests exercise exactly the
crop arithmetic :meth:`~kinescore.backbones.dino.FeatureBackbone.encode`
delegates to :meth:`~kinescore.core.clip.ViewLayout.view_crops` -- no
``dino_model``, no network, no GPU.
"""
from __future__ import annotations

import pytest
import torch

from kinescore.backbones.dino import FeatureBackbone
from kinescore.core.clip import ViewLayout

D = 4


def _backbone(view_layout: ViewLayout) -> tuple[FeatureBackbone, list[torch.Tensor]]:
    bb = FeatureBackbone(view_layout=view_layout)
    seen: list[torch.Tensor] = []

    def _fake_encode_one(rgb: torch.Tensor) -> torch.Tensor:
        seen.append(rgb)
        n = rgb.shape[0]
        return torch.zeros(n, 1, D)  # (N, P=1, D)

    bb._encode_one = _fake_encode_one  # type: ignore[method-assign]
    return bb, seen


class TestSingleView:
    def test_whole_frame_is_the_one_crop(self):
        bb, seen = _backbone(ViewLayout(n_views=1))
        rgb = torch.rand(2, 3, 48, 32)
        out = bb.encode(rgb)
        assert out.shape == (2, 1, 1, D)
        assert len(seen) == 1
        assert torch.equal(seen[0], rgb)


class TestHeightStack:
    def test_three_views_split_evenly_on_height(self):
        bb, seen = _backbone(ViewLayout(n_views=3, order=("a", "b", "c")))
        rgb = torch.rand(1, 3, 576, 320)
        out = bb.encode(rgb)
        assert out.shape == (1, 3, 1, D)
        assert len(seen) == 3
        for i, crop in enumerate(seen):
            assert crop.shape == (1, 3, 192, 320)
            assert torch.equal(crop, rgb[:, :, i * 192:(i + 1) * 192, :])


class TestCtrlworldWidthStackSubset:
    """The regression this generalization exists for: a 960x192 clip is a
    3-panel WIDTH stack (exterior_1 | exterior_2 | wrist), not a height
    stack. See ``kinescore.bench.sources.ctrlworld`` and legacy_docs/DECISIONS.md D-G.
    """

    LAYOUT = ViewLayout(n_views=2, order=("exterior_1", "exterior_2"),
                        packing="width", n_panels=3, panels=(0, 1))

    def test_crops_the_two_exterior_columns_and_drops_the_wrist_panel(self):
        bb, seen = _backbone(self.LAYOUT)
        rgb = torch.rand(1, 3, 192, 960)
        out = bb.encode(rgb)
        assert out.shape == (1, 2, 1, D)
        assert len(seen) == 2
        assert torch.equal(seen[0], rgb[:, :, :, 0:320])     # exterior_1
        assert torch.equal(seen[1], rgb[:, :, :, 320:640])   # exterior_2
        # The wrist panel (640:960) is never passed to _encode_one at all.
        for crop in seen:
            assert not torch.equal(crop, rgb[:, :, :, 640:960])

    def test_misdeclared_as_a_plain_height_stack_raises_not_silently_slices(self):
        # This is the exact defect: 192 % 3 == 0, so a bare divisibility
        # check does not catch a 3-view HEIGHT layout fed this WIDTH-stacked
        # 960x192 frame -- it would have silently produced three 960x64
        # bands. The aspect-plausibility guard in
        # ViewLayout._panel_size must raise instead.
        bb, _ = _backbone(ViewLayout(n_views=3, order=("exterior_1", "exterior_2", "wrist")))
        rgb = torch.rand(1, 3, 192, 960)
        with pytest.raises(ValueError, match="aspect"):
            bb.encode(rgb)


class TestGrid2x2:
    def test_four_quadrants_in_row_major_order(self):
        bb, seen = _backbone(ViewLayout(n_views=4, packing="grid2x2", n_panels=4))
        rgb = torch.rand(1, 3, 432, 768)
        out = bb.encode(rgb)
        assert out.shape == (1, 4, 1, D)
        assert len(seen) == 4
        assert torch.equal(seen[0], rgb[:, :, 0:216, 0:384])      # top-left
        assert torch.equal(seen[1], rgb[:, :, 0:216, 384:768])    # top-right
        assert torch.equal(seen[2], rgb[:, :, 216:432, 0:384])    # bottom-left
        assert torch.equal(seen[3], rgb[:, :, 216:432, 384:768])  # bottom-right
