"""The single checkpoint-to-reader load path: :func:`load_reader` / :func:`save_reader`.

Before this module there were two places a caller might reasonably look for
"load a checkpoint into a reader": ``readers/checkpoint.py`` (which carried
both the auto-routing dispatcher below AND a full save/load round-trip for
:class:`~kinescore.heads.attentive.AttentivePoseHead` -- a head that, once
its one reader (``readers/squashed.py::SquashedPoseReader``) was removed
(``legacy_docs/PROVENANCE.md`` D7 addendum), could never again produce anything
:func:`load_reader` would route to) and ``readers/checkpoint_v2.py`` (the
``ReadoutV2Head``-specific loader that dispatcher actually delegates to).
That split was never live parallelism -- there has only ever been one working
reader family -- it was one module doing two unrelated jobs. The
``AttentivePoseHead`` round-trip is deleted outright (dead code: nothing
could load its output into a working reader); this module keeps the one job
that *is* live -- routing a ``.pt`` file to the right reader by inspecting
its own ``cfg`` -- as its own, clearly-named thing.

``readers/checkpoint.py`` still exists as a backward-compatible re-export of
:func:`load_reader` (some callers import the submodule by that name); prefer
importing from here or from ``kinescore.readers`` directly in new code.
"""
from __future__ import annotations

from typing import Any

import torch

from kinescore.core.clip import ViewLayout
from kinescore.core.reader import PoseReader

__all__ = ["load_reader", "save_reader"]


def load_reader(path: str, *, robot: Any, view_layout: ViewLayout,
                device: str = "cpu", reader_id: str | None = None,
                backbone: Any | None = None,
                sigma_scale: Any | None = None) -> PoseReader:
    """Load ``path`` into a ready-to-score :class:`PoseReader`, auto-routed
    by the checkpoint's own cfg shape.

    Peeks at ``path``'s saved ``cfg`` dict (``torch.load`` only -- no model
    construction yet) and checks, in order:

    1. :func:`~kinescore.readers.checkpoint_v2.is_direct_keypoint_cfg` --
       ``cfg["head_target"] == "keypoints"`` -- routing to
       :func:`~kinescore.readers.checkpoint_v2.load_direct_keypoint_reader`
       for a :class:`~kinescore.readers.direct_keypoint.DirectKeypointPoseReader`.
       Checked *first* because a minimal keypoint cfg may carry none of the
       next check's own markers.
    2. :func:`~kinescore.readers.checkpoint_v2.is_readout_v2_cfg` -- looks
       like a saved :class:`~kinescore.heads.heteroscedastic.ReadoutV2Head`
       (checks for the ``d_model``/``temporal_nhead`` keys that format alone
       has) -- routing to
       :func:`~kinescore.readers.checkpoint_v2.load_reader` for a
       :class:`~kinescore.readers.checkpoint_v2.ReadoutV2PoseReader`.

    Those are currently the only two checkpoint families this function can
    build a working reader for.

    Parameters
    ----------
    robot:
        A constructed :class:`~kinescore.core.robot.RobotSpec`. Its
        ``.name`` must equal the checkpoint's declared ``robot_name`` in
        spirit (not enforced here -- :class:`~kinescore.core.scorer.Scorer`
        checks ``reader.robot_name == robot.name`` at construction, which is
        the actual enforcement point); ``.n_joints``/``.q_lo``/``.q_hi``
        size the returned reader.
    backbone:
        Optional pre-built backbone (skips constructing a new
        :class:`~kinescore.backbones.dino.FeatureBackbone`).
    sigma_scale:
        Forwarded to
        :func:`~kinescore.readers.checkpoint_v2.resolve_sigma_scale` as an
        override.

    Raises
    ------
    NotImplementedError
        If ``path``'s cfg is not a ``ReadoutV2Head`` one -- i.e. it is a
        legacy ``AttentivePoseHead``-format checkpoint (``judge_v3l``,
        ``judge_v3l_mv``, ``judge_reward``, or anything ever written by the
        now-removed ``readers/checkpoint.py::save``). That format's only
        reader, ``SquashedPoseReader``, was removed (see
        ``legacy_docs/PROVENANCE.md``'s D7 addendum); there is no shim and none is
        planned -- retrain the checkpoint with ``kinescore train-rawrad``
        instead.
    """
    ck = torch.load(path, map_location="cpu")
    cfg: dict = dict(ck.get("cfg", {}))

    from kinescore.readers import checkpoint_v2

    if checkpoint_v2.is_direct_keypoint_cfg(cfg):
        # No sigma_scale here -- that calibration path is specific to
        # ReadoutV2PoseReader's joint-limit family (see
        # checkpoint_v2.resolve_sigma_scale); a direct-keypoint checkpoint
        # has nothing for it to scale.
        return checkpoint_v2.load_direct_keypoint_reader(
            path, robot=robot, view_layout=view_layout, device=device,
            reader_id=reader_id, backbone=backbone)

    if checkpoint_v2.is_readout_v2_cfg(cfg):
        return checkpoint_v2.load_reader(
            path, robot=robot, view_layout=view_layout, device=device,
            reader_id=reader_id, backbone=backbone, sigma_scale=sigma_scale)

    raise NotImplementedError(
        f"{path!r} is an AttentivePoseHead-format checkpoint (cfg keys "
        f"{sorted(cfg)}), not a ReadoutV2Head one. The squashed pose-reader "
        f"path this format used to load through (SquashedPoseReader) has "
        f"been removed -- see legacy_docs/PROVENANCE.md's D7 addendum -- and there "
        f"is no replacement reader for it. Retrain via `kinescore "
        f"train-rawrad` to get a raw_rad (ReadoutV2Head) checkpoint instead.")


def save_reader(path: str, head: Any, *, view_layout: ViewLayout,
                robot_name: str | None = None,
                sigma_scale: Any | None = None,
                urdf_sha256: str | None = None,
                meta: dict | None = None) -> None:
    """Persist a :class:`~kinescore.heads.heteroscedastic.ReadoutV2Head` in
    the one checkpoint format :func:`load_reader` can load back.

    Thin forwarder to
    :func:`~kinescore.readers.checkpoint_v2.save` -- exists so a caller has
    one save entry point symmetric with :func:`load_reader`, without needing
    to know ``checkpoint_v2`` is where the ``ReadoutV2Head``-specific
    machinery actually lives. ``robot_name`` defaults to
    :data:`~kinescore.readers.checkpoint_v2.DEFAULT_ROBOT_NAME` when omitted,
    matching :func:`~kinescore.readers.checkpoint_v2.save`'s own default.
    """
    from kinescore.readers import checkpoint_v2

    kwargs: dict[str, Any] = {
        "view_layout": view_layout, "sigma_scale": sigma_scale,
        "urdf_sha256": urdf_sha256, "meta": meta,
    }
    if robot_name is not None:
        kwargs["robot_name"] = robot_name
    checkpoint_v2.save(path, head, **kwargs)
