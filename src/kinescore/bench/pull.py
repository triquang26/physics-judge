"""``kinescore data pull``'s engine: pull the benchmark's video datasets from Hugging Face.

Used by ``kinescore.cli.cmd_data`` (the argparse shell for ``data
pull``/``ingest``/``verify``); everything below is plain functions over
primitives (``str``/``dict``/``BenchConfig``/``RobotMap``), reachable and
unit-testable from Python without building an ``argparse.Namespace`` --
``data ingest``/``data verify`` already delegated to
:mod:`kinescore.bench.ingest`/:mod:`kinescore.bench.verify` and stayed thin
CLI wrappers; this module is the same move for ``data pull``, which had no
library home of its own yet.

Three Hugging Face dataset repos back the matrix described by
``benchmark.yaml`` (see :mod:`kinescore.bench.config`):

* ``doanh25032004/video_gen_physics`` -- generated clips, one top-level
  directory per diffusion-acceleration *cache* method (``dense``,
  ``dicache``, ``fastercache``, ...). 56k files total; pulling the whole
  repo when a run only needs ``dense`` would download hundreds of GB for
  nothing.
* ``doanh25032004/video_gen_physics_real_video`` -- the real-motion
  reference distribution *and* the pose-reader training data, in LeRobot v2
  layout (``meta/info.json`` + ``data/chunk-*/*.parquet`` joint states +
  ``videos/chunk-*/observation.images.<cam>/*.mp4``).

  **DO NOT pull ``humanoid/`` or ``single_arm/`` from this repo.**
  Coordinator directive (2026-07-29): both are replaced by real, local,
  already-verified trees referenced via ``.env`` --
  ``$KINESCORE_DROID_STD_DIR`` (single_arm/Franka, see below) and
  ``$KINESCORE_TELEOP_GR1_DIR`` (humanoid/GR-1). Two separate findings drove
  this, both worth carrying in code, not just conversation history:

  1. **HfApi().dataset_info()'s ``siblings`` listing is unreliable for this
     repo.** For most of one work session it consistently reported only
     97,565 total files with no top-level ``humanoid/``/``single_arm/`` dir
     at all -- every check against it (repeated ``dataset_info(files_metadata=
     True/False)`` calls, at different points, all agreeing) said 0 files
     under either prefix. A live ``snapshot_download`` with
     ``allow_patterns=["humanoid/**"]`` proved otherwise: it pulled real,
     valid files. ``HfApi().list_repo_files()`` then reported 188,304 total
     files and a ``humanoid/`` directory with 52,021 entries alone. **Do not
     trust ``dataset_info().siblings`` alone to decide "this prefix has zero
     files" on a large/actively-growing repo** -- corroborate with
     ``list_repo_files()`` or an actual ``snapshot_download(dry_run=True)``
     before concluding a pattern matches nothing.
  2. **The ``humanoid/`` prefix that does exist is not GR-1.** Every
     ``meta/info.json`` sampled under it (6 task dirs) reports
     ``robot_type: "Airbot_MMK2"`` / ``"discover_robotics_aitbot_mmk2"`` --
     confirmed independently on the generated side too: ``dense/humanoid/
     output/multiview/ctrlworld`` episodes are named ``AIRBOT_MMK2_*`` and
     their ``cam_high``/wrist frames show no GR-1 in shot at all (bare table
     / gloved hand). The ``humanoid`` axis label in this dataset covers TWO
     different physical robots depending on generator: Airbot MMK2 for
     ctrlworld (and this repo's real tree), genuine GR-1 for the
     ``singleview`` dreamgen/dreamdojo clips (GR-1 ego-view, confirmed
     visually). Scoring the GR-1 reader against this repo's ``humanoid/``
     real data would silently compare the wrong robot. ~130 GB extrapolated
     size (52,021 files x ~2.54 MB/file measured average) made this doubly
     not worth pulling even before the robot-identity problem surfaced.
     **The 2.8 GB of it that did land locally during investigation is real
     Airbot MMK2 footage, kept on disk as-is -- it is NOT a GR-1 reference,
     do not treat it as one.**

  Separately, ``single_arm/multiview/makovian/meta/info.json`` in this repo
  declares ``robot_type: Franka, total_episodes: 95600, fps: 15`` against a
  subtree that ``list_repo_files()`` shows has only 2,583 files total --
  nowhere near enough for 95,600 episodes. This is the same defect class as
  D3 (see :mod:`kinescore.video.probe`'s docstring): metadata copied from a
  source dataset and never updated for the actually-exported subset. It is
  exactly why :func:`kinescore.video.probe.resolve_timebase` treats a
  declared fps as a cross-check that can fail, never a trusted value --
  ``fps: 15`` here is not supported by any clip this project has actually
  probed. Reader: do not take a LeRobot ``meta/info.json`` field at face
  value for a subset repo without checking the file count agrees with it.

  This repo also has no top-level ``droid_1.0.1_20chunks`` replacement
  needed on the Franka side: that 44.25 GB tree is likewise skipped in favor
  of ``$KINESCORE_DROID_STD_DIR`` (446 MB, already local, at the SAME
  320x192/5fps as the ctrlworld generated clips -- no anchor re-encode
  needed, real joint/gripper/cartesian ground truth). Net effect: this
  module's fallback still *generates* ``humanoid/**``/``single_arm/**``
  patterns for this repo from ``config.axes`` (see
  :func:`_fallback_allow_patterns` -- it is a general-purpose builder and
  isn't hardcoded to skip them), but no run should actually invoke
  ``data pull --repo video_gen_physics_real_video`` for the humanoid/
  single_arm axes anymore; ``bimanual/**`` remains a legitimate pull for a
  later round.
* ``doanh25032004/cosmos_synthetic_data`` -- small (under 200 MB), pulled
  in full; used for the human-label separation check (E1), not the matrix.

``resolve_allow_patterns`` never guesses which subset to fetch: it reads
``--config``'s ``axes`` (via :func:`kinescore.bench.config.load_config`) and
turns them into ``huggingface_hub.snapshot_download``'s ``allow_patterns``,
so pulling a run that only exercises ``axes.cache: [dense]`` cannot
accidentally also pull ``dicache``/``fastercache``/etc. ``kinescore data pull
--dry-run`` prints the resolved repos and patterns without touching the
network or disk -- the guard against exactly that "typo in the config,
downloaded 400 GB before anyone noticed" failure mode.

Translating the five matrix axes into on-disk directory names is properly
:mod:`kinescore.bench.matrix`'s job (it already owns cell expansion and
knows about ``sources.<generator>.view_dir``/``iter``/``gt_from`` pins); this
module imports its ``allow_patterns(config, robot_map)`` lazily and only
falls back to the small local builder below when that function does not
exist yet. See :func:`_fallback_allow_patterns` for exactly what the
fallback does and does not know. ``axes.robot`` (not ``axes.embodiment`` --
see ``kinescore.bench.cell``'s module docstring for why robot replaced
embodiment as the primary axis) is turned into the on-disk ``embodiment``
directory set via ``configs/robot_map.yaml``/:mod:`kinescore.bench.robot_map`
(``--robot-map``, default next to ``--config``) BEFORE any pattern is built,
here exactly like in :mod:`kinescore.cli.cmd_bench`.
"""
from __future__ import annotations

import itertools
import os
import sys
from typing import Any

__all__ = ["HF_REPOS", "resolve_allow_patterns", "resolve_data_root", "pull_one"]

#: The three source repos this benchmark reads from, keyed by the name used
#: everywhere else in this package (CLI ``--repo``, the on-disk subdirectory
#: under ``$KINESCORE_DATA_ROOT``, and the provenance sidecar). Verified live
#: against the Hugging Face API before this module was written -- see
#: ``dense/humanoid/**`` (7,533 files / 7.53 GB) + ``dense/single_arm/**``
#: (6,597 files / 4.78 GB), and the ``video_gen_physics_real_video`` note
#: above for a layout surprise (no top-level ``humanoid/``/``single_arm/``,
#: and its ``droid_1.0.1_20chunks`` is intentionally never pulled here).
HF_REPOS: dict[str, str] = {
    "video_gen_physics": "doanh25032004/video_gen_physics",
    "video_gen_physics_real_video": "doanh25032004/video_gen_physics_real_video",
    "cosmos_synthetic_data": "doanh25032004/cosmos_synthetic_data",
}


def _fallback_allow_patterns(config: Any, robot_map: Any) -> dict[str, list[str]]:
    """Build ``{dataset_key: [glob, ...]}`` directly from ``config.axes``.

    TODO(bench.matrix): the ``video_gen_physics`` entry this returns is a
    coarse backstop for when ``kinescore.bench.matrix.allow_patterns``
    cannot be imported at all -- that module now exists and, when available,
    :func:`resolve_allow_patterns` uses its patterns for ``video_gen_physics``
    instead (it is source-aware: ``sources.<generator>.view_dir``/``iter``/
    ``gt_from``, ``na_cells``, all honoured; this function only knows
    top-level directory names). ``video_gen_physics_real_video`` and
    ``cosmos_synthetic_data`` are **not** covered by ``bench.matrix`` at all
    (see ``kinescore.bench.sources.lerobot``'s docstring: real video "is not
    indexed by the cache/generator axes"), so those two entries here are the
    only builder for them, matrix.py present or not.

    This fallback only knows the three repos' *top-level* shape, confirmed
    live against the HF API:

    * ``video_gen_physics`` -- ``{cache}/{embodiment}/**`` (cache and
      embodiment are the only two axes that appear as directory names at
      that repo's top two levels; ``view``/``horizon``/``generator`` are
      nested inside and are not needed to avoid over-fetching). ``embodiment``
      here is derived from ``axes.robot`` via ``robot_map`` (see
      :meth:`~kinescore.bench.robot_map.RobotMap.embodiment_of`) -- NOT an
      axis on ``config`` itself, since ``axes.robot`` replaced
      ``axes.embodiment`` (see ``kinescore.bench.cell``'s module docstring).
    * ``video_gen_physics_real_video`` -- ``{embodiment}/**`` for every
      embodiment implied by the axes. **Deliberately does not** add
      ``droid_1.0.1_20chunks/**`` for ``single_arm`` -- that 44 GB tree is
      redundant with ``$KINESCORE_DROID_STD_DIR`` (see module docstring and
      ``.env``), which is real, local, and already at the ctrlworld clips'
      native 320x192/5fps. A pattern for an embodiment that has no matching
      top-level directory (at write time: ``humanoid``, ``single_arm``
      literally) simply matches zero files; not an error here, but worth
      noticing in the pull's reported file count.
    * ``cosmos_synthetic_data`` -- always pulled whole (``["**"]``); it is
      the E1 human-label dataset, not part of the axis matrix, and is small
      (under 200 MB / 93 files).
    """
    axes = config.axes
    embodiments = sorted({robot_map.embodiment_of(r) for r in axes.robot})
    video_gen_physics = sorted(
        f"{cache}/{embodiment}/**"
        for cache, embodiment in itertools.product(axes.cache, embodiments))
    real_video = sorted(f"{embodiment}/**" for embodiment in embodiments)
    return {
        "video_gen_physics": video_gen_physics,
        "video_gen_physics_real_video": real_video,
        "cosmos_synthetic_data": ["**"],
    }


#: ``bench.matrix.allow_patterns`` returns globs relative to
#: ``$KINESCORE_DATA_ROOT`` (it mirrors the on-disk layout every
#: ``bench.sources`` plugin reads, e.g.
#: ``video_gen_physics/dense/humanoid/output/multiview/ctrlworld/makovian/**``
#: -- see ``kinescore.bench.sources.ctrlworld``'s docstring for the matching
#: on-disk path). That is one path segment more than
#: ``snapshot_download(repo_id="doanh25032004/video_gen_physics", ...)``
#: wants: HF's own ``allow_patterns`` is relative to the *repo root*, which
#: has no ``video_gen_physics/`` segment (files there are named
#: ``dense/humanoid/...`` directly -- verified live against the API). This
#: prefix is what :func:`resolve_allow_patterns` strips off.
_MATRIX_REPO_PREFIX = "video_gen_physics/"


def resolve_allow_patterns(config: Any, robot_map: Any) -> dict[str, list[str]]:
    """``{dataset_key: [glob, ...]}`` for every repo in :data:`HF_REPOS`.

    Starts from :func:`_fallback_allow_patterns` (which is the only builder
    ``video_gen_physics_real_video``/``cosmos_synthetic_data`` ever get --
    ``bench.matrix`` does not index either, see that function's docstring),
    then overrides the ``video_gen_physics`` entry with
    ``kinescore.bench.matrix.allow_patterns(config, robot_map)`` when that
    module can be imported: it is source-aware (``na_cells``, per-generator
    ``view_dir``/``iter``/``gt_from``) in a way the local fallback
    deliberately is not.
    """
    patterns = _fallback_allow_patterns(config, robot_map)

    try:
        from kinescore.bench.matrix import allow_patterns as _matrix_allow_patterns
    except ImportError:
        return patterns

    raw = _matrix_allow_patterns(config, robot_map)
    unexpected = [p for p in raw if not p.startswith(_MATRIX_REPO_PREFIX)]
    if unexpected:
        raise ValueError(
            f"kinescore.bench.matrix.allow_patterns returned pattern(s) "
            f"outside the {_MATRIX_REPO_PREFIX!r} prefix cmd_data knows how "
            f"to map to the video_gen_physics HF repo: {unexpected!r}")
    stripped = sorted({p[len(_MATRIX_REPO_PREFIX):] for p in raw})
    if stripped:
        patterns["video_gen_physics"] = stripped
    return patterns


def resolve_data_root(data_root: str | None, *, dry_run: bool) -> str | None:
    """``data_root`` if given, else ``$KINESCORE_DATA_ROOT``.

    On a missing env var: raises ``MissingPathError`` normally, but during
    ``--dry-run`` (``dry_run=True``) prints a note to stderr and returns
    ``None`` instead -- a dry run's whole point is to work without a real
    data root configured yet.
    """
    if data_root:
        return data_root
    from kinescore.paths import MissingPathError, env_path

    try:
        return str(env_path("KINESCORE_DATA_ROOT"))
    except MissingPathError as exc:
        if dry_run:
            print(f"[data] note: {exc}", file=sys.stderr)
            return None
        raise


def _count_local(root: str) -> tuple[int, int]:
    """``(n_files, total_bytes)`` under ``root``, skipping HF's ``.cache``.

    ``snapshot_download(local_dir=...)`` writes a ``.cache/huggingface``
    bookkeeping tree alongside the real files (visible after Task A's pulls
    landed); counting it would inflate both numbers with metadata files that
    are not clips.
    """
    n_files = 0
    total_bytes = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".cache"]
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                total_bytes += os.path.getsize(path)
            except OSError:
                continue
            n_files += 1
    return n_files, total_bytes


def pull_one(dataset_key: str, *, patterns: list[str], data_root: str,
            max_workers: int) -> dict:
    """Download one :data:`HF_REPOS` entry's ``allow_patterns`` subset.

    Deliberately does not write ``provenance.json`` itself -- unlike
    ``data ingest``/``data verify``, whose provenance a single call
    (``--robot-map``/``--data-spec``/etc.) fully describes, one ``data pull``
    invocation writes ONE ``provenance.json`` per dataset but shares
    ``run_id``/``config`` across all of them; composing that per-dataset
    provenance block is CLI-layer bookkeeping
    (:mod:`kinescore.cli._provenance`), and ``kinescore.bench`` must not
    import ``kinescore.cli`` (see ``tests/test_import_layering.py``). The
    caller (``kinescore.cli.cmd_data``) builds and writes the sidecar from
    this function's return value instead.

    Returns
    -------
    dict
        ``{"repo_id", "revision", "n_files", "total_bytes", "local_dir"}``.
    """
    from huggingface_hub import HfApi, snapshot_download

    repo_id = HF_REPOS[dataset_key]
    local_dir = os.path.join(data_root, dataset_key)
    print(f"[data] pulling {repo_id} -> {local_dir} "
         f"(allow_patterns={patterns!r})")

    info = HfApi().dataset_info(repo_id)
    revision = info.sha

    snapshot_download(
        repo_id=repo_id, repo_type="dataset", revision=revision,
        local_dir=local_dir, allow_patterns=patterns, max_workers=max_workers)

    n_files, total_bytes = _count_local(local_dir)
    print(f"[data] {dataset_key}: {n_files} file(s), "
         f"{total_bytes / 1e9:.2f} GB -> {local_dir}")
    return {"repo_id": repo_id, "revision": revision, "n_files": n_files,
            "total_bytes": total_bytes, "local_dir": local_dir}
