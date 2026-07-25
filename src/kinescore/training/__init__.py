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
    episode. See its module docstring for the defect (D4) this format fixes
    relative to its source.
:mod:`.datasets`
    RAM-flattening loader: every cached episode's tokens + aligned real joint
    labels, concatenated into one big tensor pair. The fast path for a corpus
    that fits in host RAM (a few GB of fp16 tokens for a DROID-scale run).
:mod:`.trainer`
    The AdamW/cosine head-only training loop, robot-agnostic via
    ``RobotSpec.forward_kinematics`` for its ``keypoint_mm`` eval metric.
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
