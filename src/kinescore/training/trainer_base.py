"""Shared per-episode, temporal-window pose-head training loop, and its two
concrete recipes: :class:`KeypointTrainer` (direct 3-D keypoint regression,
no FK at eval time -- the package's PRIMARY reader recipe) and
:class:`JointTrainer` (the legacy joint-angle + FK recipe, kept for
provenance).

Consolidates the validated ``train_kp_head_v3.py`` prototype (per-episode
temporal-window training, ``W=16``/``B=32``, smooth-L1 against FK keypoints,
23.76 mm val on Franka ctrlworld) into a reusable class, alongside the
existing joint-angle recipe (:mod:`kinescore.training.trainer_rawrad`) --
:class:`PoseTrainerBase` owns everything the two share (episode loading,
windowed batching with a padding mask, AdamW + two-phase cosine LR,
best-by-val checkpointing); the two subclasses differ only in what they
regress and how they score it.

Why per-episode, not :func:`~kinescore.training.datasets.load_split`
----------------------------------------------------------------------
:func:`~kinescore.training.datasets.load_split` flattens every cached
episode into one ``(N, n_tokens, D)`` tensor with no preserved episode
boundaries (its own docstring: the fast RAM path for
:class:`~kinescore.training.trainer_rawrad.RawRadTrainer`'s per-frame,
``use_context=False`` recipe). :class:`ReadoutV2Head`'s temporal context
(:class:`~kinescore.heads.heteroscedastic.TemporalEncoder`) needs real
contiguous windows *within one episode* to have anything to attend over, so
this module reads each episode's cache file directly via
:func:`~kinescore.training.cache.load_cache`, joined with the annotation
JSON's joint-position array -- exactly what the prototype's ``load_eps`` did
-- and keeps every episode's frames together until a training step samples a
fixed-length window from one of them (:meth:`PoseTrainerBase._sample_window_batch`).

Two-phase LR, one recipe each for the loss
-------------------------------------------
Both subclasses keep :class:`~kinescore.training.trainer_rawrad.RawRadTrainer`'s
``phase_a``/``phase_b`` LR schedule (warmup at ``cfg.lr``, then a drop to
``cfg.lr_phase_b`` with the cosine schedule re-initialised over the remaining
steps). :class:`KeypointTrainer`'s loss does not actually change shape across
the boundary (smooth-L1 throughout -- there is no variance head to warm up
separately), so for it ``phase_a`` only ever affects the LR curve; for
:class:`JointTrainer` it also switches the loss formula (mse -> beta-NLL),
identically to :class:`~kinescore.training.trainer_rawrad.RawRadTrainer`.
"""
from __future__ import annotations

import copy
import glob
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from kinescore.core.robot import RobotSpec
from kinescore.heads.heteroscedastic import ReadoutV2Head
from kinescore.training.cache import assert_real_joint_source, load_cache
from kinescore.training.losses import beta_nll_loss, loss_limit
from kinescore.training.trainer_rawrad import (
    TrainConfigRawRad,
    TrainResultRawRad,
    eval_keypoint_mm_rawrad,
)

__all__ = [
    "DEFAULT_JOINT_KEY", "TrainConfigPose", "PoseTrainerBase",
    "KeypointTrainer", "JointTrainer",
]

#: Same default the rest of :mod:`kinescore.training` uses (see
#: :data:`kinescore.training.datasets.DEFAULT_JOINT_KEY`).
DEFAULT_JOINT_KEY = "observation.state.joint_position"

#: One loaded episode: ``(feat (T, n_tokens, D) fp16, target (T, out_dim)
#: fp32)`` -- ``target`` is already :meth:`PoseTrainerBase.build_target`'s
#: output, not the raw joint array (see that method).
Episode = tuple[torch.Tensor, torch.Tensor]


def _episode_sort_key(path: str):
    """Numeric-first sort so ``"10.pt"`` sorts after ``"2.pt"`` -- same
    convention as :func:`kinescore.training.datasets._episode_sort_key`
    (duplicated rather than imported: it is three lines and this module
    should not depend on that module's private helpers).
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    return (0, int(stem)) if stem.isdigit() else (1, stem)


@dataclass(frozen=True)
class TrainConfigPose(TrainConfigRawRad):
    """:class:`TrainConfigRawRad` plus the one field the windowed loop needs.

    Every other field (``steps``, ``phase_a``, ``lr``/``lr_phase_b``,
    ``weight_decay``, ``beta``, ``limit_weight``, ``seed``, ``eval_every``,
    ``log_every``, ``device``) means exactly what it means there.
    ``batch_size`` here counts *episode-windows per step* (the prototype's
    ``B=32``), not flattened frames -- a windowed batch is ``(batch_size,
    window_size, ...)``, not ``(batch_size, ...)``.

    Parameters
    ----------
    window_size:
        Frames per sampled window (the prototype's ``W=16``). An episode
        shorter than this is zero-padded and masked (see
        :meth:`PoseTrainerBase._sample_window_batch`); an episode at least
        this long has a random start offset sampled each step.
    """

    #: Windowed batching reinterprets ``batch_size`` as episode-windows per
    #: step, so the inherited flattened-frame default (2048) would allocate
    #: ``2048 * window_size`` frames and OOM. Override to the prototype's B=32.
    batch_size: int = 32
    window_size: int = 16


class PoseTrainerBase:
    """Shared per-episode windowed training loop for a :class:`ReadoutV2Head`.

    Owns episode loading (:meth:`load_episodes`), windowed batch sampling
    with a padding mask (:meth:`_sample_window_batch`), and the optimisation
    loop itself (:meth:`fit`: AdamW + two-phase cosine LR, best-by-val
    checkpointing) -- the parts every pose-head recipe in this module shares.
    A subclass supplies only *what* is regressed and *how* it is scored, via
    five hooks:

    * :meth:`build_target` -- raw logged ``q (T, n_joints)`` -> per-episode
      training target ``(T, out_dim)``, computed once at load time.
    * :meth:`head_out_width` -- ``robot -> out_dim``, so a caller can size
      the head *before* constructing one (classmethod: no trainer instance
      needed yet, mirrors the prototype's ``head = ReadoutV2Head(...,
      n_out=head_out_width(robot))`` construction order).
    * :meth:`validate_head` -- raises if an already-built head's ``n_out``
      disagrees with :meth:`head_out_width`; called once at construction
      (classmethod, same reason).
    * :meth:`compute_loss` -- ``(mu, logvar, target, mask, step, phase_a) ->
      scalar``. Takes the padding ``mask`` explicitly (unlike
      :class:`~kinescore.training.trainer_rawrad.RawRadTrainer`'s per-frame
      loop, which never pads) because a window straddling an episode's end
      always has padded frames that must not contribute to the loss.
    * :meth:`eval_val` -- ``(head, episodes) -> {"keypoint_mm": float, ...}``;
      the ``"keypoint_mm"`` key is what best-by-val checkpointing tracks
      (lower is better), matching
      :class:`~kinescore.training.trainer_rawrad.TrainResultRawRad`'s own
      field of that name.

    Parameters
    ----------
    head:
        An untrained (or warm-started) :class:`ReadoutV2Head` with
        ``head.n_out == type(self).head_out_width(robot)`` -- checked at
        construction via :meth:`validate_head`.
    robot:
        Supplies FK (:class:`KeypointTrainer`'s target) or joint limits
        (:class:`JointTrainer`'s loss/eval); robot-agnostic via
        :class:`~kinescore.core.robot.RobotSpec`.
    cfg:
        See :class:`TrainConfigPose`; ``None`` constructs the default.

    Raises
    ------
    ValueError
        At construction, if ``head.n_out`` disagrees with
        ``type(self).head_out_width(robot)`` (see :meth:`validate_head`).
    """

    def __init__(self, head: ReadoutV2Head, robot: RobotSpec,
                cfg: TrainConfigPose | None = None) -> None:
        self.validate_head(head, robot)
        self.head = head
        self.robot = robot
        self.cfg = cfg or TrainConfigPose()

    # ------------------------------------------------------------------
    # Abstract hooks -- a subclass must implement every one of these.
    # ------------------------------------------------------------------
    def build_target(self, q: torch.Tensor) -> torch.Tensor:
        """``(T, n_joints) -> (T, out_dim)`` training target for one episode.

        Called once per episode, at load time (:meth:`load_episodes`), never
        inside the step loop -- so an FK-based target (see
        :class:`KeypointTrainer`) costs one forward pass per episode, not one
        per step.
        """
        raise NotImplementedError

    @classmethod
    def head_out_width(cls, robot: RobotSpec) -> int:
        """``robot -> out_dim``, the ``n_out`` a head must have for this recipe."""
        raise NotImplementedError

    @classmethod
    def validate_head(cls, head: ReadoutV2Head, robot: RobotSpec) -> None:
        """Raise ``ValueError`` if ``head.n_out != cls.head_out_width(robot)``.

        The default implementation (both subclasses use it as-is) is the
        windowed-loop analogue of
        :class:`~kinescore.training.trainer_rawrad.RawRadTrainer`'s
        constructor check.
        """
        expected = cls.head_out_width(robot)
        if head.n_out != expected:
            raise ValueError(
                f"head.n_out={head.n_out} != {cls.__name__}.head_out_width(robot)="
                f"{expected}; a head trained by {cls.__name__} must emit "
                f"exactly the target width this recipe builds (see "
                f"build_target/head_out_width).")

    def compute_loss(self, mu: torch.Tensor, logvar: torch.Tensor,
                     target: torch.Tensor, mask: torch.Tensor, step: int,
                     phase_a: bool) -> torch.Tensor:
        """``(mu, logvar, target, mask (B,W), step, phase_a) -> scalar loss``.

        ``mu``/``logvar``/``target`` are all ``(B, W, out_dim)``; ``mask`` is
        ``(B, W)``, ``1`` for a real frame and ``0`` for a zero-padded one
        (see :meth:`_sample_window_batch`) -- every implementation must
        exclude padded frames from the loss, not just from the metric.
        """
        raise NotImplementedError

    def eval_val(self, head: ReadoutV2Head, episodes: list[Episode]) -> dict:
        """Score ``head`` against ``episodes`` -> ``{"keypoint_mm": float, ...}``.

        ``"keypoint_mm"`` is the only key :meth:`fit` reads (for best-by-val
        tracking); a subclass may return additional diagnostic keys.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared: per-episode loading.
    # ------------------------------------------------------------------
    def load_episodes(self, cache_root: str, annotation_root: str, split: str, *,
                      joint_key: str = DEFAULT_JOINT_KEY, limit: int = 0,
                      progress=None) -> list[Episode]:
        """Load one split's episodes as ``(feat, target)`` pairs, per-episode.

        For every ``{cache_root}/{split}/{ep}.pt`` with a matching
        ``{annotation_root}/{split}/{ep}.json`` (``joint_source == "real"``,
        see :func:`~kinescore.training.cache.assert_real_joint_source`):
        loads the cached tokens (:func:`~kinescore.training.cache.load_cache`),
        reads the logged joint array, truncates both to their common frame
        count, and calls :meth:`build_target` once to produce that episode's
        training target. Unlike
        :func:`~kinescore.training.datasets.load_split`, nothing is
        concatenated across episodes -- each stays its own ``(feat, target)``
        pair so :meth:`_sample_window_batch` can sample a real temporal
        window from within one.

        Parameters
        ----------
        cache_root, annotation_root:
            Roots containing ``{split}/{ep}.pt`` / ``{split}/{ep}.json``.
        split:
            Split name (``"train"``, ``"val"``, ...).
        joint_key:
            Annotation JSON key for the joint-angle array.
        limit:
            Cap the number of episodes loaded (0 = all), applied after
            sorting.
        progress:
            ``progress(message: str)`` for line-oriented status, or ``None``.

        Raises
        ------
        RuntimeError
            No cache file in the split has a matching, real-joint annotation.
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
            label = assert_real_joint_source(ap)  # raises on joint_source != "real"
            feat, _header = load_cache(fp)
            q = torch.tensor(np.asarray(label[joint_key], dtype=np.float32))
            t = min(int(feat.shape[0]), int(q.shape[0]))
            feat, q = feat[:t], q[:t]
            target = self.build_target(q)
            episodes.append((feat, target))
            if progress and len(episodes) % 200 == 0:
                progress(f"[{split}] loaded {len(episodes)}/{len(files)} episodes")

        if not episodes:
            raise RuntimeError(
                f"no cached episode under {split_dir!r} has a matching, "
                f"real-joint annotation under {annotation_dir!r}")
        if progress:
            progress(f"[{split}] {len(episodes)} episodes loaded")
        return episodes

    # ------------------------------------------------------------------
    # Shared: windowed batch sampling.
    # ------------------------------------------------------------------
    def _sample_window_batch(self, episodes: list[Episode], *, out_dim: int,
                             gen: torch.Generator, device: torch.device
                             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample ``cfg.batch_size`` random windows of ``cfg.window_size``
        frames each, one per randomly-chosen episode.

        An episode shorter than ``window_size`` contributes its full length
        plus zero padding; one at least that long gets a random start offset.
        Ported from the prototype's ``batch()`` (same semantics: random
        episode, random start, zero-pad short windows, and a ``0``/``1``
        mask marking which frames are real).

        Returns
        -------
        (feat, target, mask)
            ``feat``: ``(B, W, n_tokens, D)`` fp32 on ``device``.
            ``target``: ``(B, W, out_dim)`` fp32 on ``device``.
            ``mask``: ``(B, W)`` fp32 on ``device``, ``1`` for real frames.
        """
        window = self.cfg.window_size
        fb, yb, mb = [], [], []
        for _ in range(self.cfg.batch_size):
            ei = int(torch.randint(0, len(episodes), (1,), generator=gen))
            feat, target = episodes[ei]
            t = feat.shape[0]
            w = min(window, t)
            s = (0 if t <= window
                 else int(torch.randint(0, t - window, (1,), generator=gen)))
            f = feat[s:s + w].float()
            y = target[s:s + w]
            if w < window:
                f = torch.cat([f, f.new_zeros(window - w, *f.shape[1:])])
                y = torch.cat([y, y.new_zeros(window - w, out_dim)])
            fb.append(f)
            yb.append(y)
            mb.append(torch.cat([torch.ones(w), torch.zeros(window - w)]))
        return (torch.stack(fb).to(device), torch.stack(yb).to(device),
                torch.stack(mb).to(device))

    # ------------------------------------------------------------------
    # Shared: the optimisation loop.
    # ------------------------------------------------------------------
    def fit(self, *, train_episodes: list[Episode],
           val_episodes: list[Episode] | None = None,
           progress=None) -> TrainResultRawRad:
        """Run the two-phase windowed loop on ``self.head`` (mutated in place).

        Same phase-A/phase-B LR handling as
        :meth:`~kinescore.training.trainer_rawrad.RawRadTrainer.fit` (LR
        drops to ``cfg.lr_phase_b`` and the cosine schedule re-initialises at
        ``step == phase_a``); what happens to the *loss* at that boundary is
        entirely up to :meth:`compute_loss`'s ``phase_a`` argument.

        Parameters
        ----------
        train_episodes, val_episodes:
            From :meth:`load_episodes`. ``val_episodes=None`` disables
            periodic and final validation (the "best" state is then just the
            final one, mirroring
            :class:`~kinescore.training.trainer_rawrad.RawRadTrainer`).
        progress:
            ``progress(step, loss)`` callback, or ``None``.

        Returns
        -------
        TrainResultRawRad
        """
        cfg = self.cfg
        head, robot = self.head, self.robot

        device = torch.device(cfg.device)
        head = head.to(device)
        self.head = head
        out_dim = type(self).head_out_width(robot)

        torch.manual_seed(cfg.seed)
        opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.steps, 1))
        gen = torch.Generator().manual_seed(cfg.seed)

        phase_a = min(cfg.phase_a, max(1, cfg.steps // 2))
        loss_history: list[float] = []
        best_val = float("inf")
        best_step = 0
        best_state: dict | None = None
        transitioned = False

        head.train()
        for step in range(1, cfg.steps + 1):
            f, y, m = self._sample_window_batch(
                train_episodes, out_dim=out_dim, gen=gen, device=device)

            out = head(f)
            mu, logvar = out["mu"], out["logvar"]
            loss = self.compute_loss(mu, logvar, y, m, step, step <= phase_a)

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            loss_val = float(loss.item())
            loss_history.append(loss_val)

            if step == phase_a and not transitioned:
                transitioned = True
                for g in opt.param_groups:
                    g["lr"] = cfg.lr_phase_b
                    g["initial_lr"] = cfg.lr_phase_b
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(1, cfg.steps - phase_a))

            if progress and cfg.log_every and step % cfg.log_every == 0:
                progress(step, loss_val)

            if (val_episodes is not None and cfg.eval_every
                    and (step % cfg.eval_every == 0 or step == cfg.steps)):
                ev = self.eval_val(head, val_episodes)
                head.train()
                if ev["keypoint_mm"] < best_val:
                    best_val = ev["keypoint_mm"]
                    best_step = step
                    best_state = copy.deepcopy(head.state_dict())

        train_ev = self.eval_val(head, train_episodes)
        head.train()

        val_keypoint_mm = None
        if val_episodes is not None:
            if best_state is not None:
                eval_head = copy.deepcopy(head)
                eval_head.load_state_dict(best_state)
                val_ev = self.eval_val(eval_head, val_episodes)
            else:
                val_ev = self.eval_val(head, val_episodes)
                head.train()
            val_keypoint_mm = val_ev["keypoint_mm"]

        if best_state is None:
            best_state = copy.deepcopy(head.state_dict())
            best_step = cfg.steps
            best_val = train_ev["keypoint_mm"] if val_episodes is None else best_val

        if progress:
            progress(cfg.steps, loss_history[-1] if loss_history else float("nan"))

        return TrainResultRawRad(
            steps=cfg.steps, loss_history=loss_history, best_step=best_step,
            best_val_keypoint_mm=None if val_episodes is None else best_val,
            train_keypoint_mm=train_ev["keypoint_mm"],
            val_keypoint_mm=val_keypoint_mm, best_state_dict=best_state,
        )


class KeypointTrainer(PoseTrainerBase):
    """Direct 3-D keypoint regression -- the package's PRIMARY reader recipe.

    Regresses FK keypoints directly (``mu (T, K*3)``), never runs FK at eval
    time (eval is a plain 3-D norm between predicted and target keypoints).
    This is :class:`PoseTrainerBase` wired up exactly like the validated
    ``train_kp_head_v3.py`` prototype (module docstring): smooth-L1 loss
    (``beta=0.05``) with the padding mask, no ``loss_limit`` term (a
    keypoint has no joint-limit analogue), no beta-NLL phase (``logvar`` is
    computed by the head but never trained against -- this recipe emits no
    uncertainty).
    """

    def build_target(self, q: torch.Tensor) -> torch.Tensor:
        """``(T, n_joints) -> (T, K*3)``: FK keypoints, flattened.

        Computed once per episode (see :meth:`PoseTrainerBase.load_episodes`)
        on ``self.robot.q_lo``'s device -- never re-run inside the step loop.
        """
        device = self.robot.q_lo.device
        with torch.no_grad():
            p = self.robot.forward_kinematics(q[:, None].to(device), None)  # (T,1,K,3)
        t = q.shape[0]
        return p[:, 0].reshape(t, -1).detach().cpu().float()

    @classmethod
    def head_out_width(cls, robot: RobotSpec) -> int:
        """``3 * K``, ``K`` found via one dry FK call on ``robot``."""
        device = robot.q_lo.device
        with torch.no_grad():
            q0 = torch.zeros(1, 1, robot.n_joints, device=device)
            p = robot.forward_kinematics(q0, None)
        return 3 * int(p.shape[-2])

    def compute_loss(self, mu: torch.Tensor, logvar: torch.Tensor,
                     target: torch.Tensor, mask: torch.Tensor, step: int,
                     phase_a: bool) -> torch.Tensor:
        """Masked smooth-L1 (``beta=0.05``); ``logvar``/``step``/``phase_a``
        are unused -- see class docstring."""
        del logvar, step, phase_a
        per_frame = F.smooth_l1_loss(mu, target, beta=0.05, reduction="none").mean(-1)
        return (per_frame * mask).sum() / mask.sum().clamp_min(1e-8)

    @torch.no_grad()
    def eval_val(self, head: ReadoutV2Head, episodes: list[Episode]) -> dict:
        """Root-mean-square per-keypoint 3-D error (mm), no FK involved.

        Each episode is forwarded whole, chunked by ``head.t_max`` (the
        temporal encoder's positional-table limit) so an episode longer than
        that still runs -- exactly the prototype's ``val_mm``.
        """
        head.eval()
        device = next(head.parameters()).device
        chunk = max(1, head.t_max)
        se, n = 0.0, 0
        for feat, target in episodes:
            f = feat.float().to(device)[None]  # (1,T,P,D)
            mu = torch.cat([
                head(f[:, i:i + chunk])["mu"][0].cpu()
                for i in range(0, f.shape[1], chunk)
            ])
            k = target.shape[-1] // 3
            e = (mu - target).reshape(-1, k, 3).norm(dim=-1)
            se += float((e ** 2).sum())
            n += e.numel()
        head.train()
        return {"keypoint_mm": (se / max(1, n)) ** 0.5 * 1000.0}


class JointTrainer(PoseTrainerBase):
    """Legacy joint-angle + FK recipe, kept for provenance.

    Same loss/eval recipe as
    :class:`~kinescore.training.trainer_rawrad.RawRadTrainer` (mse+limit in
    phase A, beta-NLL+limit in phase B; FK-based ``keypoint_mm`` eval),
    reimplemented against :class:`PoseTrainerBase`'s shared per-episode
    windowed loop so both recipes share one training loop end to end.
    :meth:`eval_val` delegates to
    :func:`~kinescore.training.trainer_rawrad.eval_keypoint_mm_rawrad`
    directly (flattening the episode list back to one tensor pair, which
    that function expects) rather than reimplementing FK-based scoring here.
    """

    def build_target(self, q: torch.Tensor) -> torch.Tensor:
        """Identity: the target *is* the logged joint angles."""
        return q.float()

    @classmethod
    def head_out_width(cls, robot: RobotSpec) -> int:
        return int(robot.n_joints)

    def compute_loss(self, mu: torch.Tensor, logvar: torch.Tensor,
                     target: torch.Tensor, mask: torch.Tensor, step: int,
                     phase_a: bool) -> torch.Tensor:
        """Phase A: masked MSE + ``limit_weight * loss_limit``. Phase B:
        masked beta-NLL + ``limit_weight * loss_limit`` -- same constants as
        :meth:`~kinescore.training.trainer_rawrad.RawRadTrainer.fit`."""
        del step
        cfg = self.cfg
        limits = torch.stack([self.robot.q_lo, self.robot.q_hi], dim=-1).to(mu.device)
        lim = loss_limit(mu, limits)
        w = mask[..., None]  # (B,W,1), broadcasts over out_dim
        if phase_a:
            per_elem = F.mse_loss(mu, target, reduction="none")
            mse = (per_elem * w).sum() / w.expand_as(per_elem).sum().clamp_min(1e-8)
            return mse + cfg.limit_weight * lim
        return beta_nll_loss(mu, logvar, target, beta=cfg.beta, weight=w) \
            + cfg.limit_weight * lim

    def eval_val(self, head: ReadoutV2Head, episodes: list[Episode]) -> dict:
        """Flattens ``episodes`` back into one ``(feat, q)`` pair and calls
        :func:`~kinescore.training.trainer_rawrad.eval_keypoint_mm_rawrad`
        (which processes every frame independently at ``T=1``, so
        concatenating episode boundaries away changes nothing numerically)."""
        device = next(head.parameters()).device
        feat = torch.cat([f for f, _ in episodes], dim=0)
        q = torch.cat([t for _, t in episodes], dim=0)
        return eval_keypoint_mm_rawrad(
            head, self.robot, feat, q, q_lo=self.robot.q_lo, q_hi=self.robot.q_hi,
            device=str(device))
