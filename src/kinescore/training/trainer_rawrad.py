"""The ``raw_rad`` head-only training loop: two-phase AdamW + cosine, for
:class:`~kinescore.heads.heteroscedastic.ReadoutV2Head`.

Ported from ``scripts/gr1/36_train_readout_v2.py`` (Marionette-fkjepa), the
recipe used to train the one GR-1 checkpoint this package already loads
(``readout_v2_gr1.pt`` -- see :mod:`kinescore.readers.checkpoint_v2`). This
module is the same recipe generalised to any :class:`~kinescore.core.robot.RobotSpec`
and adapted to the frame-flattened data this package's
:func:`kinescore.training.datasets.load_split` produces.

This module used to exist **alongside** ``kinescore.training.trainer`` (the
squashed head-only loop -- it trained
:class:`~kinescore.heads.attentive.AttentivePoseHead` with the since-removed
``squash_to_limits``, and every real ``judge_v3l``/``judge_v3l_mv``/
``judge_reward`` checkpoint loaded through it), not in place of it. That
module, its checkpoint format's reader (``SquashedPoseReader``), and the
``kinescore train`` subcommand built on it have since been removed entirely
-- see ``legacy_docs/PROVENANCE.md``'s D7 addendum -- so this is now the package's
only head-training loop. It trains
:class:`~kinescore.heads.heteroscedastic.ReadoutV2Head` with
:func:`~kinescore.heads.ranges.clamp_for_fk`, which is what makes
joint-limit violation an observable quantity at inference time (defect D7 --
see ``heads/ranges.py`` and ``kinescore.training.losses``).

Two deliberate departures from the source script
--------------------------------------------------
1. **No temporal windowing.** The source trains on ``(B, T_win, ...)``
   windows sampled within one episode, with context-dropout (``p=0.25``)
   toggling ``use_context``. :func:`~kinescore.training.datasets.load_split`
   flattens every cached episode into one ``(N, n_tokens, D)`` tensor with no
   preserved episode boundaries (by design -- see that module's docstring:
   it is the fast path for data that fits in RAM), so there is no windowed
   sequence to sample a context batch from without a second data-loading
   pipeline this task was explicitly scoped to avoid ("do not write a new
   feature-caching pipeline"). Every step here is therefore called with
   ``T=1`` and ``use_context=False`` -- exactly
   :class:`~kinescore.heads.heteroscedastic.ReadoutV2Head`'s documented
   per-frame mode, the same per-frame convention
   :func:`kinescore.training.trainer.train_head` uses for
   :class:`~kinescore.heads.attentive.AttentivePoseHead`. The residual
   wiring in :class:`~kinescore.heads.heteroscedastic.TemporalEncoder` means
   this is a strict subset of the source's capability (no temporal context),
   not a different model.
2. **``weighted_mse``'s weight is always 1 (no ``vis`` masking).** The source's
   per-element weight down-weights left/right arm dims by an annotation
   ``vis`` field GR-1's dataset carries (documented there as "all-ones in v1
   caches -- a no-op today"); this package's annotations carry no analogous
   field for Franka, so phase A here is plain
   ``F.mse_loss(mu, target) + limit_weight * loss_limit(mu, limits)`` --
   numerically identical to the source's own no-op case.

Everything else -- the phase-A/phase-B split, the ``lr`` step-down with a
cosine re-init at the boundary, ``_LIMIT_W = 0.05``, ``beta=0.5`` -- is the
same recipe, same constants.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from kinescore.core.robot import RobotSpec
from kinescore.heads.heteroscedastic import ReadoutV2Head
from kinescore.heads.ranges import clamp_for_fk
from kinescore.training.losses import beta_nll_loss, loss_limit

__all__ = [
    "TrainConfigRawRad", "TrainResultRawRad", "RawRadTrainer",
    "train_head_rawrad", "eval_keypoint_mm_rawrad",
]

#: Soft joint-limit weight -- see ``kinescore.training.losses.loss_limit``'s
#: docstring for why this is deliberately soft. Matches the source script's
#: ``_LIMIT_W`` exactly.
DEFAULT_LIMIT_WEIGHT = 0.05


@dataclass(frozen=True)
class TrainConfigRawRad:
    """Hyperparameters for :func:`train_head_rawrad`.

    Defaults are the source script's own (``scripts/gr1/36_train_readout_v2.py``):
    ``steps=6000``, ``bs=2048`` (this package's own DROID-scale default, see
    :class:`kinescore.training.trainer.TrainConfig`, not the source script's
    smaller GR-1 window batch), ``lr=1e-3`` dropping to ``lr_phase_b=5e-4`` at
    ``phase_a``, ``beta=0.5``, ``limit_weight=0.05``.

    Parameters
    ----------
    steps:
        Total optimisation steps.
    phase_a:
        Length of the mu-only warmup phase (clamped to ``steps // 2`` inside
        :func:`train_head_rawrad`, so phase B always runs at least half the
        steps -- mirrors the source's own clamp).
    batch_size, lr, lr_phase_b, weight_decay, beta, limit_weight, seed:
        See module docstring / the individual loss functions.
    eval_every:
        Validation cadence in steps (0 disables periodic eval -- only the
        final train/val eval still runs).
    log_every:
        :data:`~kinescore.training.trainer.ProgressCallback` cadence.
    device:
        Where the head and every minibatch are moved to during training.
    """

    steps: int = 6000
    phase_a: int = 1500
    batch_size: int = 2048
    lr: float = 1e-3
    lr_phase_b: float = 5e-4
    weight_decay: float = 1e-4
    beta: float = 0.5
    limit_weight: float = DEFAULT_LIMIT_WEIGHT
    seed: int = 0
    eval_every: int = 500
    log_every: int = 200
    device: str = "cpu"


@dataclass(frozen=True)
class TrainResultRawRad:
    """What :func:`train_head_rawrad` reports when it finishes.

    Parameters
    ----------
    steps:
        Steps actually run (== ``cfg.steps``).
    loss_history:
        Training loss at every step, in order.
    best_step:
        Step at which ``best_val_keypoint_mm`` was recorded (or ``cfg.steps``
        if no validation split was given -- the "best" state is then just the
        final one).
    best_val_keypoint_mm:
        Lowest validation keypoint-mm seen during training, or ``None`` if no
        validation split was given.
    train_keypoint_mm, val_keypoint_mm:
        Final-epoch keypoint mm on train / val (val uses the *best* saved
        state when one exists, matching the source script's "best-by-val"
        checkpointing; train always uses the final state).
    best_state_dict:
        ``head.state_dict()`` at ``best_step`` (a plain copy, safe to
        ``load_state_dict`` into a fresh head) -- what a caller should
        actually persist, not necessarily the head object's live weights
        after :func:`train_head_rawrad` returns.
    """

    steps: int
    loss_history: list = field(default_factory=list)
    best_step: int = 0
    best_val_keypoint_mm: float | None = None
    train_keypoint_mm: float = float("nan")
    val_keypoint_mm: float | None = None
    best_state_dict: dict = field(default_factory=dict)


@torch.no_grad()
def eval_keypoint_mm_rawrad(head: ReadoutV2Head, robot: RobotSpec,
                            feat: torch.Tensor, q_target: torch.Tensor, *,
                            q_lo: torch.Tensor, q_hi: torch.Tensor,
                            device: str = "cpu", batch_size: int = 4096
                            ) -> dict:
    """Mean per-keypoint FK error (mm) of ``head`` vs the logged joint targets.

    The ``raw_rad`` analogue of the now-removed ``kinescore.training.trainer.eval_keypoint_mm``
    (see ``legacy_docs/PROVENANCE.md``'s D7 addendum): uses
    :func:`~kinescore.heads.ranges.clamp_for_fk` (clamp + report overshoot)
    where that function used the since-removed ``squash_to_limits``, and
    calls ``head`` with ``use_context=False`` (see module docstring, point 1).

    ``aux`` (gripper) is held at ``None`` -- i.e. FK's default closed-gripper
    pose, see :class:`~kinescore.robots.franka.spec.FrankaSpec`'s ``aux``
    contract -- for **both** the predicted and the target forward-kinematics
    call. This head does not predict a gripper channel at all (see
    ``n_out=robot.n_joints`` at the call site -- no trailing gripper dim to
    split off), so there is no head-predicted gripper value to feed FK with,
    the way the squashed baseline's jointly-trained gripper channel does.
    Using the *same* fixed aux on both sides rather than, say, the logged GT
    gripper is a deliberate choice: it makes every millimetre of the reported
    distance attributable to the ``n_joints`` arm-radian prediction this head
    exists to train, with zero contribution from gripper opening (any
    residual dependence of a keypoint's position on gripper state is
    identical on both sides and cancels in the difference). This is *not*
    bit-identical to how the squashed baseline's own keypoint-mm number was
    produced (that number includes its own gripper-channel error); see the
    training report for the direct discussion of this deviation.

    Returns
    -------
    dict
        ``{"keypoint_mm": float, "per_keypoint_mm": np.ndarray,
        "rmse_rad": float, "sigma_mean": float}``.
    """
    head.eval()
    errs = []
    se_q, n_q = 0.0, 0
    sig_sum, sig_cnt = 0.0, 0
    n = feat.shape[0]
    for i in range(0, n, batch_size):
        f = feat[i:i + batch_size].to(device).float()[:, None]  # (b,1,P,D)
        out = head(f, use_context=False)
        mu = out["mu"][:, 0].float().cpu()          # (b, n_joints)
        logvar = out["logvar"][:, 0].float().cpu()
        tgt = q_target[i:i + batch_size]

        se_q += float(((mu - tgt) ** 2).sum())
        n_q += mu.numel()
        sig_sum += float(torch.exp(0.5 * logvar).sum())
        sig_cnt += logvar.numel()

        q_pred, _clamp_rad = clamp_for_fk(mu, q_lo.cpu(), q_hi.cpu())
        P_pred = robot.forward_kinematics(q_pred[:, None], None)
        P_true = robot.forward_kinematics(tgt[:, None], None)
        e = torch.linalg.norm(P_pred - P_true, dim=-1) * 1000.0  # (b,1,K) mm
        errs.append(e.reshape(-1, e.shape[-1]))

    E = torch.cat(errs, dim=0)
    return {
        "keypoint_mm": float(E.mean()),
        "per_keypoint_mm": E.mean(dim=0).numpy(),
        "rmse_rad": (se_q / max(1, n_q)) ** 0.5,
        "sigma_mean": sig_sum / max(1, sig_cnt),
    }


class RawRadTrainer:
    """Two-phase ``raw_rad`` head trainer (see module docstring for the recipe).

    Holds the three things one training run shares -- ``head``, ``robot``,
    ``cfg`` -- as instance state, so :meth:`fit` (the actual optimisation
    loop) reads as "run the recipe on the data" rather than repeating them.
    This is the class-based counterpart of the free function
    :func:`train_head_rawrad`, which now just constructs one of these and
    calls :meth:`fit` -- see that function's docstring for why it still
    exists (``kinescore.cli.cmd_train_rawrad`` and the existing test suite
    import the function directly).

    The training recipe itself (phase-A/phase-B split, ``beta_nll_loss(mu,
    logvar, target, beta=0.5) + 0.05 * loss_limit(mu, limits)`` with the
    first 1500 steps using weighted MSE and ``logvar`` frozen) is settled and
    identical for every robot -- this class does not parameterise it beyond
    what :class:`TrainConfigRawRad` already exposes.

    Parameters
    ----------
    head:
        An untrained (or warm-started) :class:`ReadoutV2Head`, with
        ``head.n_out == robot.n_joints`` (this loop -- like
        :meth:`eval_keypoint_mm` -- treats every head output as an FK joint;
        a head with auxiliary non-FK outputs needs its own loop, the way
        GR-1's hand DoF are handled downstream in
        :mod:`kinescore.readers.checkpoint_v2` at *inference* time, not here).
    robot:
        Supplies ``q_lo``/``q_hi`` (for :func:`~kinescore.training.losses.loss_limit`
        and the eval clamp) and ``forward_kinematics`` (for the eval metric).
        Robot-agnostic via the :class:`~kinescore.core.robot.RobotSpec`
        protocol -- the same trainer runs unmodified for Franka, GR-1,
        Airbot MMK2 or ALOHA, as long as ``head.n_out`` matches
        ``robot.n_joints``.
    cfg:
        See :class:`TrainConfigRawRad`; ``None`` constructs the default.

    Raises
    ------
    ValueError
        At construction, if ``head.n_out != robot.n_joints``.
    """

    #: Static eval helper, exposed on the class so a caller who only wants
    #: to score a checkpoint (no training loop) doesn't need an instance --
    #: identical to the free function :func:`eval_keypoint_mm_rawrad`.
    eval_keypoint_mm = staticmethod(eval_keypoint_mm_rawrad)

    def __init__(self, head: ReadoutV2Head, robot: RobotSpec,
                cfg: TrainConfigRawRad | None = None) -> None:
        if head.n_out != robot.n_joints:
            raise ValueError(
                f"head.n_out={head.n_out} != robot.n_joints={robot.n_joints}; "
                f"RawRadTrainer treats every head output as an FK joint (see "
                f"class docstring) -- a head with auxiliary non-FK outputs "
                f"(e.g. a gripper channel) needs a different loop.")
        self.head = head
        self.robot = robot
        self.cfg = cfg or TrainConfigRawRad()

    def fit(self, *, train_feat: torch.Tensor, train_q: torch.Tensor,
           val_feat: torch.Tensor | None = None,
           val_q: torch.Tensor | None = None, progress=None) -> TrainResultRawRad:
        """Run the two-phase loop on ``self.head`` (mutated in place).

        Phase A (``step <= phase_a``): ``F.mse_loss(mu, target) + limit_weight
        * loss_limit(mu, limits)`` -- ``logvar`` never enters the loss, so
        that head gets no gradient (stays at init) until phase B, exactly
        like the source: it stops the model declaring itself uncertain
        before ``mu`` is any good.

        Phase B (``phase_a < step <= steps``): ``beta_nll_loss(mu, logvar,
        target, beta=cfg.beta) + limit_weight * loss_limit(mu, limits)``,
        with ``lr`` dropped to ``cfg.lr_phase_b`` and the cosine schedule
        re-initialised over the remaining steps -- both at the exact
        transition step, matching the source script's ``step == phase_a``
        handling.

        Parameters
        ----------
        train_feat, train_q:
            See :class:`~kinescore.training.datasets.SplitData` --
            ``split.feats``/``split.q``. Gripper (``split.gripper``) is not
            consumed here at all (see class docstring's ``n_out`` point).
        val_feat, val_q:
            Optional held-out split, evaluated every ``cfg.eval_every`` steps
            (best-by-``keypoint_mm`` state is what
            :attr:`TrainResultRawRad.best_state_dict` holds) and once more at
            the end.
        progress:
            See :data:`kinescore.training.trainer_rawrad.ProgressCallback`-shaped
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
        q_lo, q_hi = robot.q_lo.to(device), robot.q_hi.to(device)
        limits = torch.stack([q_lo, q_hi], dim=-1)  # (n_joints, 2)

        torch.manual_seed(cfg.seed)
        opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.steps, 1))
        gen = torch.Generator().manual_seed(cfg.seed)

        phase_a = min(cfg.phase_a, max(1, cfg.steps // 2))
        n = train_feat.shape[0]
        loss_history: list[float] = []
        best_val = float("inf")
        best_step = 0
        best_state: dict | None = None
        transitioned = False

        head.train()
        for step in range(1, cfg.steps + 1):
            idx = torch.randint(0, n, (cfg.batch_size,), generator=gen)
            f = train_feat[idx].to(device).float()[:, None]      # (bs,1,P,D)
            q_t = train_q[idx].to(device)[:, None]                # (bs,1,n_joints)

            out = head(f, use_context=False)
            mu, logvar = out["mu"], out["logvar"]
            lim = loss_limit(mu, limits)

            if step <= phase_a:
                loss = F.mse_loss(mu, q_t) + cfg.limit_weight * lim
            else:
                loss = beta_nll_loss(mu, logvar, q_t, beta=cfg.beta) + cfg.limit_weight * lim

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

            if (val_feat is not None and cfg.eval_every
                    and (step % cfg.eval_every == 0 or step == cfg.steps)):
                ev = eval_keypoint_mm_rawrad(head, robot, val_feat, val_q,
                                             q_lo=q_lo, q_hi=q_hi, device=cfg.device)
                head.train()
                if ev["keypoint_mm"] < best_val:
                    best_val = ev["keypoint_mm"]
                    best_step = step
                    best_state = copy.deepcopy(head.state_dict())

        train_eval = eval_keypoint_mm_rawrad(head, robot, train_feat, train_q,
                                             q_lo=q_lo, q_hi=q_hi, device=cfg.device)
        head.train()

        val_keypoint_mm = None
        if val_feat is not None and val_q is not None:
            if best_state is not None:
                eval_head = copy.deepcopy(head)
                eval_head.load_state_dict(best_state)
                val_eval = eval_keypoint_mm_rawrad(eval_head, robot, val_feat, val_q,
                                                   q_lo=q_lo, q_hi=q_hi, device=cfg.device)
            else:
                val_eval = eval_keypoint_mm_rawrad(head, robot, val_feat, val_q,
                                                   q_lo=q_lo, q_hi=q_hi, device=cfg.device)
                head.train()
            val_keypoint_mm = val_eval["keypoint_mm"]

        if best_state is None:
            best_state = copy.deepcopy(head.state_dict())
            best_step = cfg.steps
            best_val = train_eval["keypoint_mm"] if val_feat is None else best_val

        if progress:
            progress(cfg.steps, loss_history[-1] if loss_history else float("nan"))

        return TrainResultRawRad(
            steps=cfg.steps, loss_history=loss_history, best_step=best_step,
            best_val_keypoint_mm=None if val_feat is None else best_val,
            train_keypoint_mm=train_eval["keypoint_mm"],
            val_keypoint_mm=val_keypoint_mm, best_state_dict=best_state,
        )


def train_head_rawrad(head: ReadoutV2Head, robot: RobotSpec, *,
                      train_feat: torch.Tensor, train_q: torch.Tensor,
                      val_feat: torch.Tensor | None = None,
                      val_q: torch.Tensor | None = None,
                      cfg: TrainConfigRawRad | None = None,
                      progress=None) -> TrainResultRawRad:
    """Thin function wrapper around ``RawRadTrainer(head, robot, cfg).fit(...)``.

    Kept for existing callers (``kinescore.cli.cmd_train_rawrad``, and the
    pre-existing test suite in ``tests/test_training_rawrad_smoke.py``) that
    import this function directly; behaviour and signature are unchanged
    from before :class:`RawRadTrainer` existed -- see that class's docstring
    for the actual recipe and the ``head``/``robot`` contract.
    """
    return RawRadTrainer(head, robot, cfg).fit(
        train_feat=train_feat, train_q=train_q, val_feat=val_feat, val_q=val_q,
        progress=progress)
