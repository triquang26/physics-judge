"""The denoising head: shapes, the workspace box, sampling and the checkpoint.

CPU-only and backbone-free -- the head consumes patch tokens, so a test can
hand it random ones, and every tensor here is small enough to denoise in a
handful of milliseconds.
"""
from __future__ import annotations

import pytest
import torch

from kinescore.core.clip import ViewLayout
from kinescore.heads.diffusion import DiffusionKeypointHead, WorkspaceNormalizer
from kinescore.readers.checkpoint import HEAD_CTOR_KEYS, load_reader, save_reader

D = 32
TOKENS_PER_VIEW = 4
N_VIEWS = 3
LAYOUT = ViewLayout(n_views=N_VIEWS, tokens_per_view=TOKENS_PER_VIEW,
                    packing="width", order=("exterior_1", "exterior_2", "wrist"))
LO = (-0.4, -0.3, 0.0)
HI = (0.4, 0.3, 0.5)


def _head(n_keypoints=5, calibrated=True, **kwargs) -> DiffusionKeypointHead:
    torch.manual_seed(0)
    spec = {"in_dim": D, "n_views": N_VIEWS,
            "tokens_per_view": TOKENS_PER_VIEW, "d_model": 16,
            "decoder_nhead": 2, "n_decoder_layers": 2, "temporal_nhead": 2,
            "ff": 32, "n_temporal_layers": 1, "t_max": 8,
            "n_coord_freqs": 3, "sample_steps": 3, "n_samples": 2, **kwargs}
    head = DiffusionKeypointHead(n_keypoints=n_keypoints, **spec)
    if calibrated:
        head.calibrate([torch.tensor([LO, HI])], margin=0.0)
    return head.eval()


def _tokens(b=2, t=3) -> torch.Tensor:
    return torch.randn(b, t, N_VIEWS * TOKENS_PER_VIEW, D)


class _Robot:
    name = "airbot_mmk2"


class _StubBackbone:
    def encode(self, rgb):  # pragma: no cover - never called here
        raise AssertionError("the backbone must not be built to load a head")


class TestShape:
    def test_output_is_points_not_a_flat_vector(self):
        out = _head(n_keypoints=5)(_tokens(b=2, t=3))
        assert out.shape == (2, 3, 5, 3)

    def test_n_out_is_three_per_keypoint(self):
        assert _head(n_keypoints=8).n_out == 24

    def test_points_land_inside_the_workspace_box(self):
        out = _head()(_tokens())
        lo = torch.tensor(LO)
        hi = torch.tensor(HI)
        assert bool((out >= lo - 1e-6).all()) and bool((out <= hi + 1e-6).all())

    def test_a_token_count_the_head_was_not_built_for_is_refused(self):
        with pytest.raises(ValueError, match="built for 12 tokens"):
            _head()(torch.randn(1, 2, 64, D))

    def test_a_non_square_patch_grid_is_refused(self):
        with pytest.raises(ValueError, match="square patch grid"):
            DiffusionKeypointHead(in_dim=D, n_keypoints=2, n_views=1,
                                  tokens_per_view=5, d_model=16,
                                  decoder_nhead=2)


class TestTrainingLoss:
    def _batch(self, head, b=2, t=3):
        target = torch.rand(b, t, head.n_keypoints, 3) * 0.4 - 0.2
        return _tokens(b, t), target, torch.ones(b, t)

    def test_loss_is_a_finite_scalar(self):
        head = _head()
        loss = head.training_loss(*self._batch(head), beta=0.05)
        assert loss.shape == () and torch.isfinite(loss)

    def test_gradients_reach_every_parameter(self):
        head = _head()
        head.training_loss(*self._batch(head), beta=0.05).backward()
        missing = [n for n, p in head.named_parameters() if p.grad is None]
        assert not missing

    def test_padded_frames_do_not_enter_the_loss(self):
        head = _head()
        feat, target, mask = self._batch(head)
        loss = head.training_loss(feat, target, mask * 0.0, beta=0.05)
        assert float(loss.detach()) == 0.0


class TestDenoiser:
    def test_an_untrained_head_returns_what_it_was_handed(self):
        head = _head()
        feat = _tokens(b=1, t=2)
        x_t = torch.randn(1, 2, head.n_keypoints, 3)
        k, v = head.decoder.keys_values(feat)
        with torch.no_grad():
            x0_hat = head._denoise(x_t, torch.tensor([0.4]), k, v)
        assert torch.equal(x0_hat, x_t)


class TestWorkspace:
    def test_encode_and_decode_round_trip(self):
        norm = WorkspaceNormalizer()
        norm.fit(torch.tensor(LO), torch.tensor(HI))
        points = torch.rand(4, 3) * 0.3 - 0.1
        assert torch.allclose(norm.decode(norm.encode(points)), points,
                              atol=1e-6)

    def test_the_corners_map_to_the_unit_box(self):
        norm = WorkspaceNormalizer()
        norm.fit(torch.tensor(LO), torch.tensor(HI))
        corners = norm.encode(torch.tensor([LO, HI]))
        assert torch.allclose(corners, torch.tensor([[-1.0] * 3, [1.0] * 3]))

    def test_a_box_with_no_volume_is_refused(self):
        with pytest.raises(ValueError, match="above lo on every axis"):
            WorkspaceNormalizer().fit(torch.tensor(LO), torch.tensor(LO))

    def test_calibrate_needs_at_least_one_episode(self):
        with pytest.raises(ValueError, match="at least one episode"):
            _head(calibrated=False).calibrate([])

    def test_calibrate_spans_every_episode(self):
        head = _head(calibrated=False)
        head.calibrate([torch.tensor([LO]), torch.tensor([HI])], margin=0.0)
        assert torch.allclose(head.workspace.lo, torch.tensor(LO))
        assert torch.allclose(head.workspace.hi, torch.tensor(HI))

    def test_calibrate_leaves_a_margin_around_the_targets(self):
        head = _head(calibrated=False)
        head.calibrate([torch.tensor([LO, HI])], margin=0.1)
        assert float(head.workspace.lo[0]) < LO[0]
        assert float(head.workspace.hi[0]) > HI[0]

    def test_reading_before_the_box_is_fitted_is_refused(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            _head(calibrated=False)(_tokens())

    def test_training_before_the_box_is_fitted_is_refused(self):
        head = _head(calibrated=False)
        with pytest.raises(RuntimeError, match="not fitted"):
            head.training_loss(_tokens(), torch.zeros(2, 3, 5, 3),
                               torch.ones(2, 3), beta=0.05)


class TestSampling:
    def test_a_seed_reproduces_a_read(self):
        head = _head()
        feat = _tokens()
        torch.manual_seed(1)
        first = head.sample(feat, n_steps=2, n_samples=1)
        torch.manual_seed(1)
        again = head.sample(feat, n_steps=2, n_samples=1)
        assert torch.equal(first, again)

    def _spread(self, head, feat, n_samples: int) -> float:
        reads = []
        for seed in range(6):
            torch.manual_seed(seed)
            reads.append(head.sample(feat, n_steps=2, n_samples=n_samples))
        return float(torch.stack(reads).std(dim=0).mean())

    def test_averaging_samples_narrows_the_read(self):
        head = _head()
        feat = _tokens()
        assert (self._spread(head, feat, 8) < self._spread(head, feat, 1))

    def test_a_read_can_start_from_an_earlier_one(self):
        head = _head()
        feat = _tokens()
        init = head.sample(feat, n_steps=2, n_samples=1)
        refined = head.sample(feat, n_steps=2, n_samples=1, init_m=init,
                              init_t=0.3)
        assert refined.shape == init.shape

    @pytest.mark.parametrize("kwargs", [{"n_steps": 0}, {"n_samples": 0}])
    def test_a_read_that_would_do_nothing_is_refused(self, kwargs):
        with pytest.raises(ValueError, match="must be >= 1"):
            _head().sample(_tokens(), **kwargs)


class TestCheckpoint:
    def _save(self, tmp_path, head) -> str:
        path = str(tmp_path / "reader.pt")
        save_reader(path, head, cell_id="single_arm.mv3_row.ctrlworld",
                    robot="airbot_mmk2", view_id="mv3_row", view_layout=LAYOUT)
        return path

    def test_the_head_class_survives(self, tmp_path):
        head = _head(n_keypoints=8)
        reader = load_reader(self._save(tmp_path, head), robot=_Robot(),
                             view_layout=LAYOUT, backbone=_StubBackbone())
        assert isinstance(reader.head, DiffusionKeypointHead)
        assert reader.head.n_keypoints == 8

    def test_every_constructor_keyword_survives(self, tmp_path):
        head = _head(sample_steps=4, n_samples=3, init_t=0.2, n_coord_freqs=2)
        reader = load_reader(self._save(tmp_path, head), robot=_Robot(),
                             view_layout=LAYOUT, backbone=_StubBackbone())
        for key in HEAD_CTOR_KEYS:
            assert getattr(reader.head, key) == getattr(head, key), key

    def test_the_workspace_box_survives(self, tmp_path):
        head = _head()
        reader = load_reader(self._save(tmp_path, head), robot=_Robot(),
                             view_layout=LAYOUT, backbone=_StubBackbone())
        assert bool(reader.head.workspace.fitted)
        assert torch.equal(reader.head.workspace.lo, head.workspace.lo)
        assert torch.equal(reader.head.workspace.hi, head.workspace.hi)

    def test_the_reloaded_head_reads_the_same_points(self, tmp_path):
        head = _head()
        feat = _tokens()
        reader = load_reader(self._save(tmp_path, head), robot=_Robot(),
                             view_layout=LAYOUT, backbone=_StubBackbone())
        torch.manual_seed(3)
        want = head.sample(feat, n_steps=2, n_samples=1)
        torch.manual_seed(3)
        got = reader.head.sample(feat, n_steps=2, n_samples=1)
        assert torch.allclose(got, want, atol=1e-6)
