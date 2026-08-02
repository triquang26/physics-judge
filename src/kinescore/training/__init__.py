"""Head-only training: precompute a feature cache, train, calibrate sigma.

Everything in this package trains exactly one thing -- a small pose-readout
head sitting on top of a *frozen* backbone (see
:mod:`kinescore.backbones.dino`) -- never the backbone itself. That is why a
feature cache (:mod:`.cache`) is worth having at all: the backbone's output
for a given clip never changes across an entire training run, so it is
computed once and read from disk thousands of times instead of re-run through
the (comparatively enormous) frozen network on every minibatch.

Submodules
----------
:mod:`.cache`
    Precompute backbone patch tokens to a self-describing ``.pt`` file per
    episode (:class:`~kinescore.training.cache.CacheBuilder`, and the thin
    :func:`~kinescore.training.cache.precompute_cache` function wrapper kept
    for existing callers). See its module docstring for the defect (D4) this
    format fixes relative to its source.
:mod:`.datasets`
    RAM-flattening loader: every cached episode's tokens + aligned real joint
    labels, concatenated into one big tensor pair. The fast path for a corpus
    that fits in host RAM (a few GB of fp16 tokens for a DROID-scale run).
    :func:`~kinescore.training.datasets.load_split` for a pre-split
    ``{train,val}/`` directory layout; :func:`~kinescore.training.datasets.load_split_stratified`
    for a single, not-yet-split pool directory (see :mod:`.splits`).
:mod:`.splits`
    Scene-stratified train/val episode partitioning
    (:func:`~kinescore.training.splits.stratified_episode_split`) -- keeps an
    entire scene/task on one side of the split so a val metric measures
    generalisation rather than memorisation. See that module's docstring for
    the defect it fixes (a val set with no ratio, no seed, no stratification
    at all).
:mod:`.trainer_rawrad`
    The two-phase AdamW/cosine ``raw_rad`` head-only training loop
    (:class:`~kinescore.training.trainer_rawrad.RawRadTrainer`, and the thin
    :func:`~kinescore.training.trainer_rawrad.train_head_rawrad` function
    wrapper kept for existing callers), robot-agnostic via
    ``RobotSpec.forward_kinematics`` for its ``keypoint_mm`` eval metric.
    Trains against :func:`~kinescore.training.datasets.load_split`'s
    RAM-flattened, per-frame data -- no temporal context.
:mod:`.trainer_base`
    The per-episode, temporal-window sibling of ``.trainer_rawrad``:
    :class:`~kinescore.training.trainer_base.PoseTrainerBase` owns a shared
    loop (episode loading, windowed batching with a padding mask, two-phase
    AdamW/cosine, best-by-val checkpointing) that two recipes plug into --
    :class:`~kinescore.training.trainer_base.KeypointTrainer` (direct 3-D
    keypoint regression, no FK at eval time -- the package's PRIMARY reader
    recipe) and :class:`~kinescore.training.trainer_base.JointTrainer` (the
    legacy joint-angle + FK recipe, reimplemented on the shared loop for
    provenance; delegates its eval to ``.trainer_rawrad``'s
    ``eval_keypoint_mm_rawrad``). Neither loop wraps the other -- pick
    whichever matches your data: flattened, per-frame (``.trainer_rawrad``)
    or per-episode, temporal-window (``.trainer_base``).
:mod:`.losses`
    :func:`~kinescore.training.losses.beta_nll_loss` for the heteroscedastic
    head (:class:`~kinescore.heads.heteroscedastic.ReadoutV2Head`).
:mod:`.calibrate`
    Post-hoc sigma temperature fitting for that same head.

Nothing in this package's top level imports torch -- construct/import the
submodule you need directly, the same convention ``kinescore.robots`` and
``kinescore.readers`` use.
"""
from __future__ import annotations
