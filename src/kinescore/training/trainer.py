"""Train the keypoint head against forward-kinematics targets."""
from __future__ import annotations

import copy
import glob
import os
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from kinescore.core.clip import ViewLayout
from kinescore.core.robot import RobotSpec
from kinescore.heads import DiffusionKeypointHead
from kinescore.heads.blocks import masked_smooth_l1
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

#: One materialised window: ``(feat, target, mask)``, padded to ``window_size``.
_Window = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _meminfo_available() -> int:
    """Bytes the host reports free, from ``MemAvailable``."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo has no MemAvailable")


#: ``memory.stat`` keys the kernel cannot reclaim under pressure.
_UNRECLAIMABLE = ("anon", "slab", "unevictable")


def _cgroup_headroom() -> int:
    """Bytes an allocation may still take in this cgroup, or ``sys.maxsize``.
    """
    try:
        with open("/proc/self/cgroup") as f:
            rel = f.read().split(":")[-1].strip()
    except OSError:
        return sys.maxsize
    node = Path("/sys/fs/cgroup") / rel.lstrip("/")
    headroom = sys.maxsize
    while node != Path("/sys/fs/cgroup") and node.is_dir():
        try:
            limit = (node / "memory.max").read_text().strip()
            stat = dict(line.split(maxsplit=1)
                        for line in (node / "memory.stat").read_text()
                        .splitlines())
        except (OSError, ValueError):
            node = node.parent
            continue
        if limit != "max":
            pinned = sum(int(stat.get(k, 0)) for k in _UNRECLAIMABLE)
            headroom = min(headroom, int(limit) - pinned)
        node = node.parent
    return max(0, headroom)


def _available_memory() -> int:
    """Bytes this process may actually take, host and cgroup alike."""
    return min(_meminfo_available(), _cgroup_headroom())


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
    workspace_margin:
        Headroom, as a fraction of each axis' span, left around the training
        targets when a head is calibrated against them.
    seed:
        Seeds both the optimiser init and the window sampler.
    read_workers:
        Threads reading token windows. Reads are latency-bound on network
        storage, so more threads raise throughput until the link saturates.
    preload:
        Hold every episode's tokens in RAM instead of reading windows from
        storage each step.
    buffer_bytes:
        Memory for a resident pool of sampled windows (``0`` = read every
        batch from storage). A step then reads ``buffer_refresh`` windows
        rather than ``batch_size``.
    buffer_refresh:
        Windows replaced per step while a buffer is in use. Higher mixes the
        pool faster and reads more.
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
    workspace_margin: float = 0.05
    seed: int = 0
    read_workers: int = 16
    preload: bool = False
    buffer_bytes: int = 0
    buffer_refresh: int = 4
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
    """Fit a :class:`~kinescore.heads.diffusion.DiffusionKeypointHead` on cached tokens.

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

    def __init__(self, head: DiffusionKeypointHead, robot: RobotSpec, *,
                 reader_id: str, view_layout: ViewLayout,
                 cfg: TrainConfig | None = None) -> None:
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
        self._resident: dict[str, torch.Tensor] = {}
        self._buffer: list[_Window] | None = None
        self._pending: list[tuple[int, Future]] = []

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

    def preload(self, episodes: list[Episode], *, progress=None) -> int:
        """Read ``episodes``' tokens into RAM and return the bytes held.

        Raises
        ------
        MemoryError
            If the split does not fit in available memory. Half of what is
            free is the ceiling: the trainer still needs room for batches, and
            a split that only fits by swapping reads slower than the disk it
            replaced.
        """
        want = sum(os.path.getsize(p) for p, _ in episodes
                   if p not in self._resident)
        free = _available_memory()
        if want > free // 2:
            raise MemoryError(
                f"preloading {len(episodes)} episodes needs {want / 2**30:.0f} "
                f"GiB but only {free / 2**30:.0f} GiB is available; train "
                f"without --preload, or with fewer episodes (--limit)")
        held = 0
        for i, (path, _target) in enumerate(episodes, 1):
            if path not in self._resident:
                feat, _header = load_cache(path, reader_id=self.reader_id,
                                           view_layout=self.view_layout)
                self._resident[path] = feat
            held += self._resident[path].numel() * self._resident[path].element_size()
            if progress and i % 50 == 0:
                progress(f"preloaded {i}/{len(episodes)} episodes, "
                         f"{held / 2**30:.1f} GiB")
        if progress:
            progress(f"{len(episodes)} episodes resident, {held / 2**30:.1f} GiB")
        return held

    def read_window(self, path: str, start: int, size: int) -> torch.Tensor:
        """``(size, n_tokens, D)`` tokens from ``path``, starting at ``start``.
        """
        feat = self._resident.get(path)
        if feat is not None:
            return feat[start:start + size]
        feat, _header = load_cache(path, reader_id=self.reader_id,
                                   view_layout=self.view_layout, mmap=True)
        return feat[start:start + size].clone()

    def _pick(self, episodes: list[Episode], *, gen: torch.Generator
              ) -> tuple[str, torch.Tensor, int, int]:
        """One ``(path, target, start, length)`` window to read."""
        window = self.cfg.window_size
        ei = int(torch.randint(0, len(episodes), (1,), generator=gen))
        path, target = episodes[ei]
        t = int(target.shape[0])
        w = min(window, t)
        s = (0 if t <= window
             else int(torch.randint(0, t - window, (1,), generator=gen)))
        return path, target, s, w

    def _materialize(self, pick: tuple[str, torch.Tensor, int, int]
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read one pick into ``(feat, target, mask)``, zero-padded to size."""
        path, target, s, w = pick
        window = self.cfg.window_size
        f = self.read_window(path, s, w)
        y = target[s:s + w]
        if w < window:
            f = torch.cat([f, f.new_zeros(window - w, *f.shape[1:])])
            y = torch.cat([y, y.new_zeros(window - w, *y.shape[1:])])
        return f, y, torch.cat([torch.ones(w), torch.zeros(window - w)])

    def fill_buffer(self, episodes: list[Episode], *, gen: torch.Generator,
                    progress=None) -> int:
        """Fill the sampling buffer and return the windows it holds.

        Raises
        ------
        MemoryError
            If ``cfg.buffer_bytes`` exceeds what this process may take.
        """
        budget = self.cfg.buffer_bytes
        free = _available_memory()
        if budget > free // 2:
            raise MemoryError(
                f"a {budget / 2**30:.1f} GiB window buffer exceeds the "
                f"{free / 2**30:.1f} GiB available to this process")
        first = self._materialize(self._pick(episodes, gen=gen))
        per_window = first[0].numel() * first[0].element_size()
        n = max(self.cfg.batch_size, budget // per_window)
        rest = [self._pick(episodes, gen=gen) for _ in range(n - 1)]
        self._buffer = [first, *self._readers.map(self._materialize, rest)]
        if progress:
            progress(f"buffer holds {n} windows, "
                     f"{n * per_window / 2**30:.1f} GiB")
        return n

    def _drain(self) -> None:
        """Move completed refills into the buffer."""
        for slot, future in self._pending:
            self._buffer[slot] = future.result()
        self._pending = []

    def _refill(self, episodes: list[Episode], *, gen: torch.Generator) -> None:
        """Submit this step's replacements, collected at the next."""
        slots = torch.randint(0, len(self._buffer), (self.cfg.buffer_refresh,),
                              generator=gen)
        self._pending = [
            (int(slot), self._readers.submit(self._materialize,
                                             self._pick(episodes, gen=gen)))
            for slot in slots]

    def _sample_windows(self, episodes: list[Episode], *, gen: torch.Generator,
                        device: torch.device
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``cfg.batch_size`` windows -> ``(feat, target, mask)`` on ``device``.
        """
        if self._buffer is None:
            picks = [self._pick(episodes, gen=gen)
                     for _ in range(self.cfg.batch_size)]
            items = list(self._readers.map(self._materialize, picks))
        else:
            self._drain()
            slots = torch.randint(0, len(self._buffer),
                                  (self.cfg.batch_size,), generator=gen)
            items = [self._buffer[int(i)] for i in slots]
            self._refill(episodes, gen=gen)

        fb, yb, mb = zip(*items, strict=True)
        return (torch.stack(fb).to(device).float(), torch.stack(yb).to(device),
                torch.stack(mb).to(device))

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
        """Smooth-L1 over real frames only."""
        return masked_smooth_l1(pred, target, mask, beta=self.cfg.huber_beta)

    @torch.no_grad()
    def evaluate(self, head: DiffusionKeypointHead, episodes: list[Episode]) -> dict:
        """Root-mean-square per-keypoint 3-D error, millimetres."""
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
            progress=None, on_eval=None) -> TrainResult:
        """Run the loop, mutating ``self.head``.

        Parameters
        ----------
        train_episodes, val_episodes:
            From :meth:`load_episodes`. Without a validation split there is no
            best-by-val selection and the final state is returned. The head is
            calibrated against the training targets before the first step.
        progress:
            ``progress(step, loss)``, or ``None``.
        on_eval:
            ``on_eval(step, keypoint_mm)`` each time the validation split is
            scored, or ``None``. Validation is what says whether a run is
            working; without this a long run reports only its training loss
            until it ends.
        """
        cfg = self.cfg
        device = torch.device(cfg.device)
        head = self.head.to(device)
        self.head = head
        head.calibrate([target for _path, target in train_episodes],
                       margin=cfg.workspace_margin)

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
            loss = head.training_loss(f, y, m, beta=cfg.huber_beta)

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
                if on_eval:
                    on_eval(step, mm)
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
