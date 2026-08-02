"""Shared plumbing between ``kinescore score`` and ``kinescore reference build``.

Both commands need the same three things: a camera :class:`ViewLayout` built
from ``--n-views``/``--view-order``, a per-clip timebase cross-check against
``--fps``/``--dt`` (never silently overriding the probe -- see
``kinescore.video.probe.resolve_timebase``), and a composed
:class:`~kinescore.core.scorer.Scorer` from ``--robot``/``--reader``/
``--suite``. Duplicating the argparse *flags* across the two commands' own
``add_arguments`` is fine (they read naturally as "this command's options");
duplicating the *logic* behind them is how the two commands would quietly
drift apart on what "the same --fps" means. This module is that logic, kept
in one place.

Top-level imports here stay light (only ``argparse`` typing); every function
that touches ``torch``/``kinescore.core``/``kinescore.robots``/
``kinescore.readers`` imports those inside itself, so importing this module
(which ``kinescore.cli.main`` does transitively while building ``--help``)
never requires torch.
"""
from __future__ import annotations

import argparse

__all__ = [
    "add_common_arguments", "view_layout_from_args", "resolve_row_clip",
    "apply_resolved_timebase", "build_scorer",
]


def add_common_arguments(parser: argparse.ArgumentParser, *,
                         require_reader: bool = True) -> None:
    """Add the ``--robot``/``--reader``/``--suite``/timebase/view-layout flags.

    Parameters
    ----------
    parser:
        The subcommand's argument parser (or a nested one, e.g. ``reference
        build``'s).
    require_reader:
        ``score`` always needs a trained reader; a future reference-building
        mode that reads real joint labels straight out of annotation JSON
        rather than a pose reader would not -- kept as a parameter rather
        than hardcoding ``required=True`` so that mode can reuse this
        function instead of re-declaring every other flag.
    """
    parser.add_argument("--robot", required=True,
                        help="registered robot name (see `kinescore describe`"
                             " for available metrics; robot names are listed"
                             " by kinescore.robots.available_robots())")
    parser.add_argument("--reader", required=require_reader,
                        help="pose-reader checkpoint path, as written by "
                             "`kinescore train-rawrad` / "
                             "kinescore.readers.checkpoint_v2.save")
    parser.add_argument("--suite", default="invariant_v1",
                        help="metric suite name (default: invariant_v1)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-views", type=int, default=1,
                        help="camera views packed into every clip's frame "
                             "(must match the reader's checkpoint). Assumes "
                             "a plain HEIGHT stack -- there is no CLI flag "
                             "yet for a non-default ViewLayout.packing "
                             "(width stack / 2x2 grid / a panel subset, see "
                             "kinescore.core.clip.ViewLayout); scoring a clip "
                             "packed another way from the CLI needs those "
                             "passed programmatically instead")
    parser.add_argument("--view-order", default=None,
                        help="comma-separated view names, e.g. "
                             "exterior_1,exterior_2,wrist_left (length must "
                             "equal --n-views if given)")
    rate = parser.add_mutually_exclusive_group()
    rate.add_argument("--fps", type=float, default=None,
                      help="override every clip's fps, cross-checked against "
                           "ffprobe (mutually exclusive with --dt)")
    rate.add_argument("--dt", type=float, default=None,
                      help="override every clip's dt in seconds, cross-checked "
                           "against ffprobe (mutually exclusive with --fps)")


def view_layout_from_args(args: argparse.Namespace):
    """Build the :class:`~kinescore.core.clip.ViewLayout` ``--n-views``/``--view-order`` declare."""
    from kinescore.core.clip import ViewLayout

    order = tuple(args.view_order.split(",")) if args.view_order else ()
    return ViewLayout(n_views=args.n_views, order=order)


def resolve_row_clip(row: dict, *, fps: float | None, dt: float | None, view_layout):
    """Re-resolve one manifest row's timebase, cross-checked against the probe.

    The manifest row's own ``fps`` (already cross-checked once, at
    ``kinescore manifest`` time) becomes ``fps_table`` here -- so this is a
    *second*, independent cross-check against a fresh probe of the file on
    disk, not a re-trust of the manifest. A stale manifest or a file swapped
    out from under it therefore cannot silently change what timebase a run is
    scored at.

    Raises
    ------
    ValueError
        If both ``fps`` and ``dt`` are given (see
        :func:`~kinescore.video.probe.resolve_timebase`).
    kinescore.core.clip.TimebaseError
        If ``fps``/``dt``/the manifest's own fps disagrees with what
        ``ffprobe`` measures on the file today.
    """
    from kinescore.video.probe import resolve_timebase

    return resolve_timebase(row["path"], fps_arg=fps, dt_arg=dt,
                            fps_table=row.get("fps"), view_layout=view_layout)


def apply_resolved_timebase(rows: list[dict], *, fps: float | None,
                            dt: float | None, view_layout) -> list[dict]:
    """Re-probe every row and fold the cross-checked timebase back into it.

    Returns new row dicts (the input list is not mutated) with
    ``fps``/``dt``/``dt_source``/``n_frames``/``w``/``h``/``codec`` replaced
    by what :func:`resolve_row_clip` just established, so a downstream
    :class:`~kinescore.core.clip.ClipSpec` built from the row (e.g.
    ``kinescore.bench.runner.run``) agrees with what was actually
    cross-checked here instead of a stale manifest value.
    """
    out = []
    for row in rows:
        clip = resolve_row_clip(row, fps=fps, dt=dt, view_layout=view_layout)
        r = dict(row)
        r.update(fps=clip.fps, dt=clip.dt, dt_source=clip.dt_source,
                 n_frames=clip.n_frames, w=clip.width, h=clip.height,
                 codec=clip.codec)
        out.append(r)
    return out


def build_scorer(args: argparse.Namespace, view_layout):
    """Compose robot + checkpoint reader + suite into a :class:`~kinescore.core.scorer.Scorer`.

    ``--reader`` auto-routes via :func:`kinescore.readers.checkpoint.load_reader`,
    which inspects the checkpoint's ``cfg`` dict (no model construction yet):
    a :class:`~kinescore.heads.heteroscedastic.ReadoutV2Head` cfg (the GR-1
    production checkpoint, ``readout_v2_gr1.pt``, or any checkpoint written by
    ``kinescore train-rawrad``) routes to a
    :class:`~kinescore.readers.checkpoint_v2.ReadoutV2PoseReader` (sigma
    calibration + the FK/aux dimension split -- see that module's docstring
    for why GR-1's 29-dim head output cannot feed
    ``GR1Spec.forward_transforms`` directly). Anything else -- notably a
    legacy :class:`~kinescore.heads.attentive.AttentivePoseHead` cfg
    (``judge_v3l``/``judge_v3l_mv``/``judge_reward``) -- raises
    ``NotImplementedError``: that format's only reader
    (``SquashedPoseReader``) was removed (see ``legacy_docs/PROVENANCE.md``'s D7
    addendum), so there is no reader left to build for it.

    See ``readers/checkpoint.py::load_reader``'s own docstring for the
    routing rule itself.
    """
    from kinescore.bench.suites import get_suite
    from kinescore.core.scorer import Scorer
    from kinescore.readers import checkpoint as ckpt_mod
    from kinescore.robots import get_robot

    robot = get_robot(args.robot, device=args.device)
    reader = ckpt_mod.load_reader(args.reader, robot=robot, view_layout=view_layout,
                                  device=args.device)

    suite = get_suite(args.suite)
    return Scorer(robot=robot, reader=reader, suite=suite)
