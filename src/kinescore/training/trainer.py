"""The head-only training loop: AdamW + cosine schedule, MSE in joint space.

Ported from ``Marionette-ciasc/scripts/train_judge.py``'s training loop and
``eval_mm``, generalised from that script's one hardcoded embodiment (Franka,
7 joints, ``judge.fk``) to any :class:`~kinescore.core.robot.RobotSpec` --
the eval metric calls ``robot.forward_kinematics`` rather than a
Franka-specific FK method, so the same function trains a head for the
synthetic 2-link arm in a CPU-only unit test and, unmodified, a real Franka
or GR-1 head.

Everything here trains one ``nn.Module`` (the head) against cached backbone
tokens (see :mod:`kinescore.training.cache`/``.datasets``); the frozen
backbone itself never appears -- there is nothing to train in it, and by the
time this module runs the tokens are already on disk.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from kinescore.core.robot import RobotSpec
from kinescore.heads.ranges import squash_to_limits

__all__ = ["TrainConfig", "TrainResult", "train_head", "eval_keypoint_mm"]

#: Called as ``progress(step: int, loss: float)`` every ``cfg.log_every``
#: steps, or ``None`` (the default) for silent training.
ProgressCallback = Callable[[int, float], None]


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for :func:`train_head`.

    Parameters
    ----------
    steps:
        Optimisation steps (source default: 6000 for a full DROID run; the
        CPU smoke test uses a handful).
    batch_size:
        Frames per minibatch, sampled with replacement from the flattened
        split (matches the source's ``torch.randint`` sampling -- an
        "epoch" is not a concept this loop tracks, since the RAM-flattened
        data has no natural episode boundary to respect per step).
    lr, weight_decay:
        AdamW hyperparameters.
    seed:
        Seeds a dedicated ``torch.Generator`` for minibatch sampling *and*
        torch's global RNG (via ``torch.manual_seed``) at the start of
        :func:`train_head`, so two runs with the same config, data and
        initial weights are bit-for-bit reproducible -- the global reseed
        matters because dropout inside the head draws from the global
        generator, not a per-call one; minibatch sampling gets its own
        generator anyway so it stays reproducible independent of how many
        dropout draws a given architecture happens to consume.
    log_every:
        How often :attr:`ProgressCallback` fires (0 disables all but the
        final call).
    device:
        Where the head and every minibatch are moved to during training.
    """

    steps: int = 6000
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    log_every: int = 200
    device: str = "cpu"


@dataclass(frozen=True)
class TrainResult:
    """What :func:`train_head` reports when it finishes.

    Parameters
    ----------
    steps:
        Steps actually run (== ``cfg.steps``).
    loss_history:
        Training loss at every step, in order -- what
        ``tests/test_training_smoke.py`` checks decreases.
    train_keypoint_mm, val_keypoint_mm:
        Mean per-keypoint FK error in millimetres (see
        :func:`eval_keypoint_mm`) on the training data and, if a validation
        split was given, on it. ``val_keypoint_mm`` is ``None`` when no
        validation data was passed to :func:`train_head`.
    train_grip_mae, val_grip_mae:
        Gripper-opening mean absolute error in ``[0,1]``, or ``None`` when
        the split carries no gripper target.
    """

    steps: int
    loss_history: list = field(default_factory=list)
    train_keypoint_mm: float = float("nan")
    val_keypoint_mm: Optional[float] = None
    train_grip_mae: Optional[float] = None
    val_grip_mae: Optional[float] = None


def _split_head_output(raw: torch.Tensor, has_gripper: bool):
    """``(..., n_joints[+1]) -> (q_raw (..., n_joints), gripper_raw (..., 1) | None)``."""
    if has_gripper:
        return raw[..., :-1], raw[..., -1:]
    return raw, None


@torch.no_grad()
def eval_keypoint_mm(head: nn.Module, robot: RobotSpec, feat: torch.Tensor,
                     q_target: torch.Tensor, gripper_target: Optional[torch.Tensor] = None,
                     *, q_lo: torch.Tensor, q_hi: torch.Tensor, device: str = "cpu",
                     batch_size: int = 4096) -> dict:
    """Mean per-keypoint FK error (mm) of ``head`` vs the logged joint targets.

    This is the eval metric ``train_head`` reports: rather than a joint-space
    MSE (whose units are radians, and whose per-joint weighting is arbitrary),
    keypoint error runs *both* the predicted and target joint angles through
    ``robot.forward_kinematics`` and measures the resulting keypoint distance
    in the robot's actual metric geometry -- the same quantity a human
    inspecting the pose would judge accuracy by, and unit-comparable across
    robots with different joint counts/ranges.

    Parameters
    ----------
    head:
        A head whose ``forward(feat[:, None])`` returns ``(B, 1, n_out)``
        (see :func:`train_head`'s docstring for the same per-frame calling
        convention).
    robot:
        Supplies ``forward_kinematics`` and (via ``q_lo``/``q_hi`` below) the
        squash range.
    feat:
        ``(N, n_tokens, D)`` cached tokens.
    q_target:
        ``(N, n_joints)`` radians.
    gripper_target:
        ``(N, 1)`` in ``[0,1]``, or ``None``.
    q_lo, q_hi:
        ``(n_joints,)`` radians, forwarded to
        :func:`~kinescore.heads.ranges.squash_to_limits`.
    batch_size:
        Chunk size for the FK pass (keeps peak memory bounded on a large
        validation split).

    Returns
    -------
    dict
        ``{"keypoint_mm": float, "per_keypoint_mm": np.ndarray, "grip_mae": float | None}``.
    """
    head.eval()
    has_gripper = gripper_target is not None
    errs = []
    grip_errs = []
    n = feat.shape[0]
    for i in range(0, n, batch_size):
        f = feat[i:i + batch_size].to(device).float()
        raw = head(f[:, None])[:, 0]  # (b, n_joints[+1])
        q_raw, grip_raw = _split_head_output(raw, has_gripper)
        q_pred = squash_to_limits(q_raw, q_lo.to(device), q_hi.to(device))

        aux_pred = torch.sigmoid(grip_raw) if grip_raw is not None else None
        aux_true = gripper_target[i:i + batch_size].to(device) if has_gripper else None

        # FK always runs on CPU here, independent of cfg.device: it is a
        # diagnostic metric (not on the backward path), and decoupling it
        # from the training device sidesteps any GPU-residency assumption a
        # given RobotSpec's wrapped FK chain (e.g. pytorch_kinematics) may or
        # may not support well.
        P_pred = robot.forward_kinematics(q_pred[:, None].cpu(),
                                          aux_pred.cpu() if aux_pred is not None else None)
        P_true = robot.forward_kinematics(
            q_target[i:i + batch_size][:, None].cpu(), aux_true.cpu() if aux_true is not None else None)
        e = torch.linalg.norm(P_pred - P_true, dim=-1) * 1000.0  # (b,1,K) mm
        errs.append(e.reshape(-1, e.shape[-1]))
        if has_gripper:
            grip_errs.append((aux_pred.cpu()[:, 0] - aux_true.cpu()).abs().reshape(-1))

    E = torch.cat(errs, dim=0)
    grip_mae = float(torch.cat(grip_errs, dim=0).mean()) if grip_errs else None
    return {"keypoint_mm": float(E.mean()),
           "per_keypoint_mm": E.mean(dim=0).numpy(), "grip_mae": grip_mae}


def train_head(head: nn.Module, robot: RobotSpec, *, train_feat: torch.Tensor,
               train_q: torch.Tensor, train_gripper: Optional[torch.Tensor] = None,
               val_feat: Optional[torch.Tensor] = None,
               val_q: Optional[torch.Tensor] = None,
               val_gripper: Optional[torch.Tensor] = None,
               cfg: TrainConfig = TrainConfig(),
               progress: Optional[ProgressCallback] = None) -> TrainResult:
    """Train ``head`` on cached tokens with plain minibatch AdamW + cosine LR.

    ``head`` is called per-frame (``feat[:, None]``, i.e. ``T=1``) exactly as
    the source's ``judge.predict_pose`` did -- the cached tokens have no
    temporal structure to exploit (they were pooled per-frame by
    :mod:`kinescore.backbones.dino`), so there is nothing a windowed call
    would buy over a flat batch of independent frames. A head that *does*
    want temporal context (e.g.
    :class:`~kinescore.heads.heteroscedastic.ReadoutV2Head`) is trained by a
    caller that batches its own frame windows before calling this loop's
    lower-level pieces directly -- this function's per-frame convention
    matches :class:`~kinescore.heads.attentive.AttentivePoseHead`, the head
    ``kinescore.readers.checkpoint`` actually persists.

    Loss: MSE between the squashed prediction and the logged target, in joint
    space (radians), plus (when ``train_gripper`` is given) MSE between
    ``sigmoid(gripper_raw)`` and the logged gripper opening. Whether ``head``
    emits a trailing gripper channel is inferred from whether
    ``train_gripper`` is given -- not a separate config flag -- so a
    mismatched (head shape, gripper argument) pair fails immediately with a
    shape error from the loss, rather than silently training a phantom
    channel.

    Parameters
    ----------
    head:
        An un-trained (or warm-started) head, e.g.
        :class:`~kinescore.heads.attentive.AttentivePoseHead`, already
        ``.to(cfg.device)``-able. Only ``head.parameters()`` are optimised --
        nothing else in this call graph has learnable weights.
    robot:
        Supplies ``q_lo``/``q_hi`` (for the squash) and ``forward_kinematics``
        (for the eval metric). Any :class:`~kinescore.core.robot.RobotSpec`
        works, including :class:`~kinescore.robots.synthetic.Synthetic2R` for
        a torch-kinematics-free unit test.
    train_feat, train_q, train_gripper:
        See :class:`~kinescore.training.datasets.SplitData` -- typically
        ``split.feats``/``split.q``/``split.gripper``.
    val_feat, val_q, val_gripper:
        Optional held-out split, evaluated once at the end (not during
        training -- this loop has no early stopping).
    cfg:
        See :class:`TrainConfig`.
    progress:
        See :data:`ProgressCallback`.

    Returns
    -------
    TrainResult
    """
    device = torch.device(cfg.device)
    head = head.to(device)
    q_lo, q_hi = robot.q_lo.to(device), robot.q_hi.to(device)
    has_gripper = train_gripper is not None

    # Reseed the global RNG (dropout inside `head` draws from it) in addition
    # to the dedicated Generator below (minibatch sampling) -- see
    # TrainConfig.seed's docstring for why both are needed for reproducibility.
    torch.manual_seed(cfg.seed)

    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.steps, 1))
    gen = torch.Generator().manual_seed(cfg.seed)

    n = train_feat.shape[0]
    loss_history: list[float] = []
    head.train()
    for step in range(1, cfg.steps + 1):
        idx = torch.randint(0, n, (cfg.batch_size,), generator=gen)
        f = train_feat[idx].to(device).float()
        raw = head(f[:, None])[:, 0]  # (bs, n_joints[+1])
        q_raw, grip_raw = _split_head_output(raw, has_gripper)
        q_pred = squash_to_limits(q_raw, q_lo, q_hi)

        loss = F.mse_loss(q_pred, train_q[idx].to(device))
        if has_gripper:
            grip_pred = torch.sigmoid(grip_raw)
            loss = loss + F.mse_loss(grip_pred, train_gripper[idx].to(device))

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        loss_val = float(loss.item())
        loss_history.append(loss_val)
        if progress and cfg.log_every and step % cfg.log_every == 0:
            progress(step, loss_val)

    train_eval = eval_keypoint_mm(head, robot, train_feat, train_q, train_gripper,
                                  q_lo=q_lo, q_hi=q_hi, device=cfg.device)
    val_keypoint_mm = val_grip_mae = None
    if val_feat is not None and val_q is not None:
        val_eval = eval_keypoint_mm(head, robot, val_feat, val_q, val_gripper,
                                    q_lo=q_lo, q_hi=q_hi, device=cfg.device)
        val_keypoint_mm = val_eval["keypoint_mm"]
        val_grip_mae = val_eval["grip_mae"]

    if progress:
        progress(cfg.steps, loss_history[-1] if loss_history else float("nan"))

    return TrainResult(
        steps=cfg.steps, loss_history=loss_history,
        train_keypoint_mm=train_eval["keypoint_mm"],
        val_keypoint_mm=val_keypoint_mm,
        train_grip_mae=train_eval["grip_mae"], val_grip_mae=val_grip_mae)
