"""Train the keypoint head against forward-kinematics targets.

No corpus logs 3-D keypoints, so the target is manufactured: a real-teleop
episode logs joint angles, and the robot's forward kinematics turns those into
the ``(T, K, 3)`` points the head learns to read off pixels. Forward kinematics
therefore runs once per episode at load time and never again -- the head that
ships has no kinematic chain in it.

Episodes stay whole rather than being flattened into a frame pool: the head's
temporal stage attends across a window, so it needs frames that really are
contiguous within one episode. Each step samples one window per batch slot,
zero-padding an episode shorter than the window and masking the padding out of
the loss.

Only the target is held in memory. Tokens stay on disk: a three-panel episode
is a few hundred megabytes and a split runs to hundreds of episodes, far past
any ordinary allocation, so a window is memory-mapped, copied, and released at
the point it is sampled. Resident token memory is therefore one batch of
windows regardless of how large the split is.
"""
from __future__ import annotations

import copy
import glob
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from kinescore.core.clip import ViewLayout
from kinescore.core.robot import RobotSpec
from kinescore.heads.keypoint import KeypointHead
from kinescore.training.cache import assert_real_joint_source, load_cache

__all__ = [
    "DEFAULT_JOINT_KEY", "Episode", "TrainConfig", "TrainResult",
    "KeypointTrainer",
]

#: Annotation key holding the logged joint array.
DEFAULT_JOINT_KEY = "observation.state.joint_position"

#: One loaded episode: ``(cache_path, target (T, K, 3) fp32)``. Tokens are read
#: from ``cache_path`` a window at a time; see :meth:`KeypointTrainer.read_window`.
Episode = tuple[str, torch.Tensor]


def _episode_sort_key(path: str) -> tuple[int, object]:
    """Numeric-first sort, so ``10.pt`` follows ``2.pt``."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return (0, int(stem)) if stem.isdigit() else (1, stem)


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for :meth:`KeypointTrainer.fit`.

    Parameters
    ----------
    steps:
        Total optimisation steps.
    batch_size:
        Windows per step, each sampled from its own randomly chosen episode.
    window_size:
        Frames per window.
    lr, lr_late, lr_step_at:
        Learning rate; it drops to ``lr_late`` at step ``lr_step_at`` and the
        cosine schedule restarts over the remaining steps. ``lr_step_at`` is
        clamped to ``steps // 2``, so the late rate always runs at least half
        the schedule.
    weight_decay:
        AdamW decay.
    huber_beta:
        Smooth-L1 transition point, metres. Below it the loss is quadratic, so
        sub-centimetre error is not swamped by the occasional gross miss.
    seed:
        Seeds both the optimiser init and the window sampler.
    eval_every, log_every:
        Cadences in steps; ``0`` disables.
    device:
        Where the head and every batch live.
    """

    steps: int = 6000
    batch_size: int = 32
    window_size: int = 16
    lr: float = 1e-3
    lr_late: float = 5e-4
    lr_step_at: int = 1500
    weight_decay: float = 1e-4
    huber_beta: float = 0.05
    seed: int = 0
    read_workers: int = 4
    eval_every: int = 500
    log_every: int = 200
    device: str = "cpu"


@dataclass(frozen=True)
class TrainResult:
    """What :meth:`KeypointTrainer.fit` reports.

    Parameters
    ----------
    steps:
        Steps run.
    loss_history:
        Training loss per step, in order.
    best_step:
        Step whose state is in ``best_state_dict``.
    best_val_mm, train_mm, val_mm:
        Keypoint error, millimetres. ``val_mm`` scores the best state; the
        ``val_*`` fields are ``None`` when no validation split was given.
    best_state_dict:
        Head weights at ``best_step`` -- what a caller persists.
    """

    steps: int
    loss_history: list[float] = field(default_factory=list)
    best_step: int = 0
    best_val_mm: float | None = None
    train_mm: float = float("nan")
    val_mm: float | None = None
    best_state_dict: dict = field(default_factory=dict)


class KeypointTrainer:
    """Fit a :class:`~kinescore.heads.keypoint.KeypointHead` on cached tokens.

    Parameters
    ----------
    head:
        The head to train, in place. Its ``n_keypoints`` must equal the
        robot's, which :meth:`n_keypoints` reads off one dry FK call.
    robot:
        Supplies the forward kinematics that build the target.
    reader_id, view_layout:
        Which reader is being trained. Every cache file loaded is checked
        against them, so a cache built for another packing cannot enter this
        run.
    cfg:
        See :class:`TrainConfig`.

    Raises
    ------
    ValueError
        If ``head.n_keypoints`` disagrees with the robot's keypoint count.
    """

    def __init__(self, head: KeypointHead, robot: RobotSpec, *, reader_id: str,
                 view_layout: ViewLayout, cfg: TrainConfig | None = None
                 ) -> None:
        expected = self.n_keypoints(robot)
        if head.n_keypoints != expected:
            raise ValueError(
                f"head predicts {head.n_keypoints} keypoints but "
                f"{robot.name} forward kinematics produces {expected}; build "
                f"the head with n_keypoints={expected}")
        self.head = head
        self.robot = robot
        self.reader_id = reader_id
        self.view_layout = view_layout
        self.cfg = cfg or TrainConfig()
        self._readers = ThreadPoolExecutor(
            max_workers=max(1, self.cfg.read_workers))

    @staticmethod
    def n_keypoints(robot: RobotSpec) -> int:
        """``K`` for ``robot``, from one forward-kinematics call at zero pose."""
        device = robot.q_lo.device
        with torch.no_grad():
            q0 = torch.zeros(1, 1, robot.n_joints, device=device)
            p = robot.forward_kinematics(q0, None)
        return int(p.shape[-2])

    def build_target(self, q: torch.Tensor) -> torch.Tensor:
        """``(T, n_joints)`` logged joints -> ``(T, K, 3)`` keypoints, metres."""
        device = self.robot.q_lo.device
        with torch.no_grad():
            p = self.robot.forward_kinematics(q[:, None].to(device), None)
        return p[:, 0].detach().cpu().float()

    def load_episodes(self, cache_root: str, annotation_root: str, split: str, *,
                      joint_key: str = DEFAULT_JOINT_KEY, limit: int = 0,
                      progress=None) -> list[Episode]:
        """Load ``{cache_root}/{split}/*.pt`` with their annotations.

        An episode is kept when it has a matching annotation whose
        ``joint_source`` is ``"real"``. The target is built once, here, and
        truncated to the frame count the tokens and joints share; the tokens
        themselves are left on disk and read per window during the run.

        Parameters
        ----------
        cache_root, annotation_root:
            Roots holding ``{split}/{ep}.pt`` and ``{split}/{ep}.json``.
        split:
            Split name.
        joint_key:
            Annotation key for the joint array.
        limit:
            Cap on episodes loaded, after sorting. ``0`` means all.
        progress:
            ``progress(message)`` for status lines, or ``None``.

        Raises
        ------
        RuntimeError
            If no cached episode in the split has a real-joint annotation.
        """
        split_dir = os.path.join(cache_root, split)
        files = sorted(glob.glob(os.path.join(split_dir, "*.pt")),
                       key=_episode_sort_key)
        if limit > 0:
            files = files[:limit]
        annotation_dir = os.path.join(annotation_root, split)

        episodes: list[Episode] = []
        for fp in files:
            ep = os.path.splitext(os.path.basename(fp))[0]
            ap = os.path.join(annotation_dir, f"{ep}.json")
            if not os.path.exists(ap):
                continue
            label = assert_real_joint_source(ap)
            _feat, header = load_cache(fp, reader_id=self.reader_id,
                                       view_layout=self.view_layout, mmap=True)
            q = torch.tensor(np.asarray(label[joint_key], dtype=np.float32))
            t = min(int(header.n_frames), int(q.shape[0]))
            episodes.append((fp, self.build_target(q[:t])))
            if progress and len(episodes) % 200 == 0:
                progress(f"[{split}] loaded {len(episodes)}/{len(files)} episodes")

        if not episodes:
            raise RuntimeError(
                f"no cached episode under {split_dir!r} has a matching "
                f"real-joint annotation under {annotation_dir!r}")
        if progress:
            progress(f"[{split}] {len(episodes)} episodes loaded")
        return episodes

    def read_window(self, path: str, start: int, size: int) -> torch.Tensor:
        """``(size, n_tokens, D)`` tokens from ``path``, starting at ``start``.

        The file is mapped, the window copied out, and the mapping dropped, so
        resident token memory never grows past the windows currently in hand.
        A window running past the end of the episode comes back short; the
        caller pads it and masks the padding out of the loss.
        """
        feat, _header = load_cache(path, reader_id=self.reader_id,
                                   view_layout=self.view_layout, mmap=True)
        return feat[start:start + size].clone()

    def _sample_windows(self, episodes: list[Episode], *, gen: torch.Generator,
                        device: torch.device
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``cfg.batch_size`` windows -> ``(feat, target, mask)`` on ``device``.

        ``feat`` is ``(B, W, n_tokens, D)``, ``target`` ``(B, W, K, 3)``, and
        ``mask`` ``(B, W)`` with ``1`` on real frames.

        Which windows to take is decided first, on this thread and from
        ``gen``, so a seed reproduces a run whatever the reads do; only the
        reads themselves are spread across ``cfg.read_workers``.

        Windows stay in the cache's own half precision until they reach the
        device, and are widened there. A batch of three-panel windows is
        gigabytes, so casting on the host would double both the copy held in
        memory and the bytes crossing the bus, for a conversion that is exact
        either way.
        """
        window = self.cfg.window_size
        picks = []
        for _ in range(self.cfg.batch_size):
            ei = int(torch.randint(0, len(episodes), (1,), generator=gen))
            path, target = episodes[ei]
            t = int(target.shape[0])
            w = min(window, t)
            s = (0 if t <= window
                 else int(torch.randint(0, t - window, (1,), generator=gen)))
            picks.append((path, target, s, w))

        reads = list(self._readers.map(
            lambda p: self.read_window(p[0], p[2], p[3]), picks))

        fb, yb, mb = [], [], []
        for f, (_path, target, s, w) in zip(reads, picks, strict=True):
            y = target[s:s + w]
            if w < window:
                f = torch.cat([f, f.new_zeros(window - w, *f.shape[1:])])
                y = torch.cat([y, y.new_zeros(window - w, *y.shape[1:])])
            fb.append(f)
            yb.append(y)
            mb.append(torch.cat([torch.ones(w), torch.zeros(window - w)]))
        return (torch.stack(fb).to(device).float(), torch.stack(yb).to(device),
                torch.stack(mb).to(device))

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
        """Smooth-L1 over real frames only.

        ``pred``/``target`` are ``(B, W, K, 3)`` and ``mask`` is ``(B, W)``:
        padded frames are excluded from the loss itself, not merely from the
        reported metric.
        """
        per_frame = F.smooth_l1_loss(
            pred, target, beta=self.cfg.huber_beta, reduction="none",
        ).mean(dim=(-1, -2))
        return (per_frame * mask).sum() / mask.sum().clamp_min(1e-8)

    @torch.no_grad()
    def evaluate(self, head: KeypointHead, episodes: list[Episode]) -> dict:
        """Root-mean-square per-keypoint 3-D error, millimetres.

        Each episode is scored in chunks of ``head.t_max``, so a clip longer
        than the positional table still scores and only one chunk of tokens is
        resident at a time.
        """
        was_training = head.training
        head.eval()
        device = next(head.parameters()).device
        chunk = max(1, head.t_max)
        se, n = 0.0, 0
        for path, target in episodes:
            t = int(target.shape[0])
            pred = torch.cat([
                head(self.read_window(path, i, chunk)
                     .to(device)[None].float())[0].cpu()
                for i in range(0, t, chunk)
            ])[:t]
            e = (pred - target).norm(dim=-1)
            se += float((e ** 2).sum())
            n += e.numel()
        if was_training:
            head.train()
        return {"keypoint_mm": (se / max(1, n)) ** 0.5 * 1000.0}

    def fit(self, *, train_episodes: list[Episode],
            val_episodes: list[Episode] | None = None,
            progress=None) -> TrainResult:
        """Run the loop, mutating ``self.head``.

        Parameters
        ----------
        train_episodes, val_episodes:
            From :meth:`load_episodes`. Without a validation split there is no
            best-by-val selection and the final state is returned.
        progress:
            ``progress(step, loss)``, or ``None``.
        """
        cfg = self.cfg
        device = torch.device(cfg.device)
        head = self.head.to(device)
        self.head = head

        torch.manual_seed(cfg.seed)
        opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(cfg.steps, 1))
        gen = torch.Generator().manual_seed(cfg.seed)

        step_at = min(cfg.lr_step_at, max(1, cfg.steps // 2))
        loss_history: list[float] = []
        best_val = float("inf")
        best_step = 0
        best_state: dict | None = None

        head.train()
        for step in range(1, cfg.steps + 1):
            f, y, m = self._sample_windows(train_episodes, gen=gen, device=device)
            loss = self.compute_loss(head(f), y, m)

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            loss_history.append(float(loss.item()))

            if step == step_at:
                for g in opt.param_groups:
                    g["lr"] = cfg.lr_late
                    g["initial_lr"] = cfg.lr_late
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(1, cfg.steps - step_at))

            if progress and cfg.log_every and step % cfg.log_every == 0:
                progress(step, loss_history[-1])

            if (val_episodes is not None and cfg.eval_every
                    and (step % cfg.eval_every == 0 or step == cfg.steps)):
                mm = self.evaluate(head, val_episodes)["keypoint_mm"]
                if mm < best_val:
                    best_val, best_step = mm, step
                    best_state = copy.deepcopy(head.state_dict())

        train_mm = self.evaluate(head, train_episodes)["keypoint_mm"]

        val_mm = None
        if val_episodes is not None:
            scored = head
            if best_state is not None:
                scored = copy.deepcopy(head)
                scored.load_state_dict(best_state)
            val_mm = self.evaluate(scored, val_episodes)["keypoint_mm"]

        if best_state is None:
            best_state = copy.deepcopy(head.state_dict())
            best_step = cfg.steps

        if progress:
            progress(cfg.steps, loss_history[-1] if loss_history else float("nan"))

        return TrainResult(
            steps=cfg.steps, loss_history=loss_history, best_step=best_step,
            best_val_mm=None if val_episodes is None else best_val,
            train_mm=train_mm, val_mm=val_mm, best_state_dict=best_state,
        )
