#!/usr/bin/env python3
# What:   Convert the Ctrl-World "video_gen_physics/dense/<embodiment>/input/
#         multiview/ctrlworld/{makovian,non_makovian}/episode_*/" trees (real
#         teleop joint trajectories used to condition Ctrl-World rollouts, at
#         320x192 @ 5fps -- the SAME resolution/fps/cameras as the generated
#         clips kinescore benchmarks) into kinescore's training contract:
#         annotation/{train,val}/<ep>.json + videos/{train,val}/<ep>.mp4,
#         one robot at a time (single_arm/franka_panda, bimanual/
#         aloha_bimanual, humanoid/airbot_mmk2 -- the 12 arm joints only,
#         hand channels dropped).
# Why:    This is training data in the exact domain being scored. The
#         existing Franka reader (trained on droid_std -- different cameras,
#         concatenated single view) scored train 20.14mm / val 165.35mm: it
#         memorises and does not generalise. This converter tests the
#         domain-match hypothesis by producing a same-resolution,
#         same-fps, same-camera training set for all three robots kinescore
#         reads (franka_panda, aloha_bimanual, airbot_mmk2).
# Input:  $KINESCORE_DATA_ROOT/video_gen_physics/dense/{single_arm,bimanual,
#         humanoid}/input/multiview/ctrlworld/{makovian,non_makovian}/
#         episode_*/{view_0,view_1,view_2,full_gt}.mp4 + metadata.json
#         (instruction, num_frames, fps, states, and for single_arm only, a
#         second `joints` array -- see ROBOTS below for which field this
#         script actually reads per robot and why).
# Output: $KINESCORE_DATA_ROOT/train/<kinescore_robot>_ctrlworld/
#           annotation/{train,val}/<episode_id>.json
#           videos/{train,val}/<episode_id>.mp4
#           dataset_card.json
"""See the header above for the What/Why/Input/Output contract.

Two load-bearing discoveries this script's logic depends on -- both verified
against real data on disk before writing a line of conversion code, not
assumed from the coordinator's brief (see the module-level DISCOVERIES
comment block below for the numbers):

1. ``states`` vs ``joints`` for single_arm.  ``metadata.json["states"]`` is a
   7-D **Cartesian** EE pose (xyz + 3 rotation channels + gripper in [0,1]) --
   its 4th-6th channels range to +/-pi, and its first 3 channels are exactly
   the workspace-box position range droid_std's own
   ``observation.state.cartesian_position`` uses. It is NOT joint radians,
   despite the coordinator's brief table saying "states shape (T,7)
   radians". The actual 7 Franka joint angles + gripper live in the
   sibling ``metadata.json["joints"]`` array (T, 8): per-channel min/max
   across 30 sampled episodes falls inside ``get_robot("franka_panda")``'s
   ``q_lo``/``q_hi`` almost exactly on every channel (including joint4's
   asymmetric [-3.0718,-0.0698] range, which ``states`` cannot match, being
   a pose not an angle). ``joints`` is present, 8-wide and length-matched to
   ``states`` on all 257/257 single_arm episodes (both makovian and
   non_makovian) -- never missing, so gripper presence is uniform across
   the whole robot. This script therefore reads ``joints[:, :7]`` as
   ``observation.state.joint_position`` and ``joints[:, 7]`` as
   ``observation.state.gripper_position`` for single_arm, NOT ``states``.

2. ``view_N.mp4`` vs ``full_gt.mp4`` frame counts.  The per-camera
   ``view_0/1/2.mp4`` files frequently do NOT match ``metadata.json``'s
   ``num_frames``/``len(states)`` -- confirmed on random 40-episode samples
   per robot (thread-pooled ffprobe ``-count_frames``, not the container's
   possibly-stale ``nb_frames`` tag): view0 matches only 6/40 (single_arm),
   1/40 (bimanual), 11/40 (humanoid). The mismatch is not a rounding
   artifact -- single_arm's view_0.mp4 is frequently LESS THAN HALF the
   claimed episode length (e.g. episode_000255: num_frames=101,
   len(states)=101, but view_0.mp4 decodes to exactly 35 frames / 7.0s).
   ``full_gt.mp4``, in contrast, matches ``num_frames`` on 120/120 sampled
   episodes across all three robots, and pixel-diff confirms it really is
   the 3 views width-stacked at x-offsets {0,320,640} (mean abs diff ~4/255
   against the matching ``view_N.mp4`` when both exist and agree -- pure
   compression noise, not a different render). So this script reads ONLY
   ``full_gt.mp4`` (never ``view_N.mp4``) and crops the 3 panels from it,
   after an ffprobe-verified frame-count match against
   ``num_frames``/``len(states)`` -- an episode where even ``full_gt.mp4``
   disagrees is skipped and listed, never truncated or padded (see
   ``_verify_and_probe``).

A THIRD deviation from the coordinator's literal output-path brief, on the
video CONTAINER shape (not skipped silently -- flagged here and in every
run's dataset_card.json ``layout_note``): the brief's ASCII tree writes
``videos/{train,val}/<episode>/{0,1,2}.mp4`` (three separate per-view files
in a per-episode directory) and says to symlink them from the source. That
layout is NOT what ``kinescore cache``/``kinescore.training.cache`` actually
reads: ``cmd_cache.py``'s ``--video-root`` is glob-matched as
``{video_root}/{split}/*.mp4`` (flat, one packed multiview file per
episode -- see ``training/cache.py::CacheBuilder.build_split``/
``precompute_cache``), and ``--n-views``/``--view-order`` slice ONE packed
frame via :class:`kinescore.core.clip.ViewLayout` (default
``packing="height"``, i.e. vstacked panels, matching
``convert_lerobot_to_train.py``'s proven ``_pack_views``) -- there is no
CLI flag to instead read a directory of separate per-view files, and no
code path in ``training/cache.py`` that would find one if there were. The
franka droid_std reference tree this task's brief compares against
(``$KINESCORE_DATA_ROOT/droid_std_franka_rawrad/videos/train/*.mp4``)
confirms this empirically: it is flat, one symlinked file per episode
(single-view only, matching its ``--n-views 1`` training command), not a
nested per-episode directory. This script therefore writes ONE packed,
height-stacked (vstack) mp4 per episode at ``videos/{split}/<ep>.mp4`` --
directly consumable by ``kinescore cache --n-views 3`` with zero extra
packing step -- rather than three separate symlinked files. Because
``full_gt.mp4`` must be read and re-encoded (crop x3 + vstack) regardless
(discovery 2 above), "symlink, do not copy" for the per-camera files is
moot; what IS preserved is that the SOURCE tree is never written to, only
read, and the output is a small derived artifact (a few dozen frames at
320x192 per view, tens of MB per robot, not the ~1GB the brief was
guarding against when it assumed direct symlinks of already-correct files).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kinescore.paths import env_path  # noqa: E402
from kinescore.training.splits import (  # noqa: E402
    default_scene_key,
    stratified_episode_split,
)

VIEW_WIDTH = 320
VIEW_HEIGHT = 192
N_VIEWS = 3
FPS = 5


@dataclass(frozen=True)
class RobotConfig:
    kinescore_robot: str
    input_subdir: str          # under video_gen_physics/dense/
    out_name: str              # $KINESCORE_DATA_ROOT/train/<out_name>/
    raw_field: str             # "joints" or "states"
    joint_indices: tuple[int, ...]
    gripper_index: int | None
    n_raw_channels: int        # expected width of metadata[raw_field] rows
    scene_key_mode: str        # "instruction" | "episode_dir"
    note: str


ROBOTS: dict[str, RobotConfig] = {
    "single_arm": RobotConfig(
        kinescore_robot="franka_panda",
        input_subdir="single_arm",
        out_name="franka_panda_ctrlworld",
        raw_field="joints",
        joint_indices=tuple(range(7)),
        gripper_index=7,
        n_raw_channels=8,
        scene_key_mode="instruction",
        note=("metadata['joints'][:, :7] = 7 real Franka joint angles (rad); "
              "metadata['states'] is a 7-D CARTESIAN pose (xyz+rot+gripper), "
              "NOT joint radians -- verified against get_robot('franka_panda')"
              ".q_lo/.q_hi (per-channel min/max of 'joints' matches the URDF "
              "limits almost exactly, incl. joint4's asymmetric range; "
              "'states' does not). gripper = metadata['joints'][:, 7]."),
    ),
    "bimanual": RobotConfig(
        kinescore_robot="aloha_bimanual",
        input_subdir="bimanual",
        out_name="aloha_bimanual_ctrlworld",
        raw_field="states",
        joint_indices=(0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12),
        gripper_index=None,
        n_raw_channels=14,
        scene_key_mode="episode_dir",
        note=("metadata['states'] (T,14) = qpos [left(7), right(7)], each "
              "[waist,shoulder,elbow,forearm_roll,wrist_angle,wrist_rotate,"
              "gripper]. Indices 6 and 13 (per-side gripper) verified in-"
              "range [~-0.06, ~1.0] (single-sided, unlike the symmetric arm "
              "channels) and dropped; the remaining 12 verified inside "
              "get_robot('aloha_bimanual').q_lo/.q_hi per-channel. No "
              "gripper_position written: AlohaSpec's gripper is a 2-value "
              "[left,right] aux, not load_split's single scalar -- same "
              "choice convert_lerobot_to_train.py's aloha_bimanual makes."),
    ),
    "humanoid": RobotConfig(
        kinescore_robot="airbot_mmk2",
        input_subdir="humanoid",
        out_name="airbot_mmk2_ctrlworld",
        raw_field="states",
        joint_indices=tuple(range(12)),
        gripper_index=None,
        n_raw_channels=36,
        scene_key_mode="episode_dir",
        note=("metadata['states'][:, :12] = 12 arm joints (rad, max ~1.7); "
              "verified inside get_robot('airbot_mmk2').q_lo/.q_hi per-"
              "channel. states[:, 12:36] are 24 hand-joint channels in a "
              "different unit (max ~85, likely degrees or percent) that "
              "kinescore's spec does not model -- dropped, matching "
              "convert_lerobot_to_train.py's airbot_mmk2 mapping."),
    ),
}

_EP_PREFIX_RE = re.compile(r"^episode_")


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _ffprobe_frames(path: str) -> int | None:
    """Actual decoded frame count (the stream, not a container tag)."""
    try:
        out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-count_frames", "-show_entries", "stream=nb_read_frames",
                    "-of", "csv=p=0", path], timeout=60, check=True)
        return int(out.stdout.strip())
    except Exception:
        return None


def _ffprobe_dims(path: str) -> tuple[int, int] | None:
    try:
        out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=width,height",
                    "-of", "csv=p=0", path], timeout=30, check=True)
        w, h = out.stdout.strip().split(",")
        return int(w), int(h)
    except Exception:
        return None


@dataclass
class EpisodeResult:
    episode_id: str
    ok: bool
    reason: str = ""
    scene_key: str = ""
    q: list[list[float]] | None = None
    gripper: list[float] | None = None
    instruction: str = ""
    bucket: str = ""
    source_dir: str = ""
    n_frames: int = 0


def _discover_episodes(input_root: str) -> list[tuple[str, str]]:
    """[(bucket, episode_dir), ...] across makovian/non_makovian, sorted."""
    out = []
    for bucket in ("makovian", "non_makovian"):
        for d in sorted(glob.glob(os.path.join(input_root, bucket, "episode_*"))):
            out.append((bucket, d))
    return out


def _episode_id_for(robot_key: str, bucket: str, ep_dir: str, meta: dict) -> str:
    dirname = os.path.basename(ep_dir)
    stem = _EP_PREFIX_RE.sub("", dirname)
    if robot_key == "single_arm":
        # dir/metadata id is bare-numeric; bucket isn't embedded -- add it
        # ourselves so makovian/non_makovian episodes can't collide (they
        # don't in this snapshot, but nothing guarantees that in general).
        return f"{bucket}_{stem}"
    # bimanual/humanoid: metadata's own episode_id already carries the
    # bucket prefix (e.g. "makovian_close_cardboard_box__000026") -- prefer
    # it, but fall back to the directory-derived id if it's ever absent so
    # a single malformed metadata.json can't silently collide two episodes.
    mid = meta.get("episode_id")
    if isinstance(mid, str) and mid:
        return mid
    return f"{bucket}_{stem}"


def _verify_and_probe(ep_dir: str, meta: dict, cfg: RobotConfig
                       ) -> tuple[int, str] | tuple[None, str]:
    """Return (n_frames, "") on success or (None, reason) on any disagreement.

    Checks, in order: metadata num_frames == len(states/joints array) ==
    ffprobe-decoded full_gt.mp4 frame count == 3x view-panel width. Never
    truncates or pads -- any disagreement is a skip.
    """
    n_meta = meta.get("num_frames")
    raw = meta.get(cfg.raw_field)
    if raw is None:
        return None, f"metadata missing '{cfg.raw_field}'"
    n_states = len(raw)
    if not isinstance(n_meta, int) or n_meta != n_states:
        return None, f"num_frames={n_meta} != len(metadata['{cfg.raw_field}'])={n_states}"
    if n_states == 0:
        return None, "empty episode (0 frames)"
    width0 = len(raw[0]) if raw and hasattr(raw[0], "__len__") else None
    if width0 != cfg.n_raw_channels:
        return None, (f"metadata['{cfg.raw_field}'] row width {width0} != "
                      f"expected {cfg.n_raw_channels}")

    full_gt = os.path.join(ep_dir, "full_gt.mp4")
    if not os.path.exists(full_gt):
        return None, "full_gt.mp4 missing"
    dims = _ffprobe_dims(full_gt)
    if dims is None:
        return None, "ffprobe failed to read full_gt.mp4 dimensions"
    w, h = dims
    if h != VIEW_HEIGHT or w != VIEW_WIDTH * N_VIEWS:
        return None, f"full_gt.mp4 is {w}x{h}, expected {VIEW_WIDTH * N_VIEWS}x{VIEW_HEIGHT}"

    n_video = _ffprobe_frames(full_gt)
    if n_video is None:
        return None, "ffprobe failed to count full_gt.mp4 frames"
    if n_video != n_meta:
        return None, (f"full_gt.mp4 decodes to {n_video} frames != "
                      f"num_frames/len(states)={n_meta}")
    return n_meta, ""


def _pack_video(full_gt_path: str, out_path: str) -> None:
    """Crop the 3 width-stacked panels out of ``full_gt_path`` and vstack them
    into ONE height-packed mp4 at ``out_path`` -- the layout
    ``ViewLayout(n_views=3)`` (default ``packing="height"``) expects.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    crops = ";".join(
        f"[0:v]crop={VIEW_WIDTH}:{VIEW_HEIGHT}:{i * VIEW_WIDTH}:0[v{i}]"
        for i in range(N_VIEWS))
    stack_inputs = "".join(f"[v{i}]" for i in range(N_VIEWS))
    filt = f"{crops};{stack_inputs}vstack=inputs={N_VIEWS}[v]"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", full_gt_path,
          "-filter_complex", filt, "-map", "[v]",
          "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path]
    subprocess.run(cmd, check=True, timeout=120)


def _extract_q_gripper(raw: list[list[float]], cfg: RobotConfig
                       ) -> tuple[list[list[float]], list[float] | None]:
    q = [[float(row[i]) for i in cfg.joint_indices] for row in raw]
    gripper = None
    if cfg.gripper_index is not None:
        gripper = [float(row[cfg.gripper_index]) for row in raw]
    return q, gripper


def _instruction_scene_key(instruction: str) -> str:
    return re.sub(r"\s+", " ", instruction.strip().lower())


def convert_robot(robot_key: str, *, data_root: str, out_root: str,
                  val_ratio: float, seed: int, workers: int,
                  limit: int, overwrite: bool) -> dict:
    cfg = ROBOTS[robot_key]
    input_root = os.path.join(data_root, "video_gen_physics", "dense",
                              cfg.input_subdir, "input", "multiview", "ctrlworld")
    out_dir = os.path.join(out_root, cfg.out_name)
    ann_dir = os.path.join(out_dir, "annotation")
    vid_dir = os.path.join(out_dir, "videos")
    for split in ("train", "val"):
        os.makedirs(os.path.join(ann_dir, split), exist_ok=True)
        os.makedirs(os.path.join(vid_dir, split), exist_ok=True)

    episodes = _discover_episodes(input_root)
    if limit:
        episodes = episodes[:limit]
    print(f"[{robot_key}] {len(episodes)} candidate episodes under {input_root}")

    def _process_one(item: tuple[str, str]) -> EpisodeResult:
        bucket, ep_dir = item
        meta_path = os.path.join(ep_dir, "metadata.json")
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception as e:
            return EpisodeResult(episode_id=os.path.basename(ep_dir), ok=False,
                                 reason=f"metadata.json unreadable: {e}")

        ep_id = _episode_id_for(robot_key, bucket, ep_dir, meta)
        n_frames, reason = _verify_and_probe(ep_dir, meta, cfg)
        if n_frames is None:
            return EpisodeResult(episode_id=ep_id, ok=False, reason=reason,
                                 bucket=bucket, source_dir=ep_dir)

        raw = meta[cfg.raw_field]
        q, gripper = _extract_q_gripper(raw, cfg)
        instruction = str(meta.get("instruction", ""))
        scene_key = (_instruction_scene_key(instruction)
                    if cfg.scene_key_mode == "instruction"
                    else default_scene_key(ep_id))
        return EpisodeResult(episode_id=ep_id, ok=True, scene_key=scene_key,
                             q=q, gripper=gripper, instruction=instruction,
                             bucket=bucket, source_dir=ep_dir, n_frames=n_frames)

    results: list[EpisodeResult] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_process_one, episodes):
            results.append(r)

    ok = [r for r in results if r.ok]
    skipped = [r for r in results if not r.ok]

    # duplicate-id guard: two source dirs mapping to the same episode_id
    # would silently overwrite one another's annotation/video -- catch it.
    seen: dict[str, EpisodeResult] = {}
    deduped_ok: list[EpisodeResult] = []
    for r in ok:
        if r.episode_id in seen:
            skipped.append(EpisodeResult(
                episode_id=r.episode_id, ok=False,
                reason=f"duplicate episode_id, also produced by "
                       f"{seen[r.episode_id].source_dir}",
                bucket=r.bucket, source_dir=r.source_dir))
            continue
        seen[r.episode_id] = r
        deduped_ok.append(r)
    ok = deduped_ok

    if not ok:
        raise RuntimeError(f"[{robot_key}] 0 usable episodes out of "
                           f"{len(episodes)} -- nothing to write")

    episode_ids = tuple(r.episode_id for r in ok)
    scene_key_by_id = {r.episode_id: r.scene_key for r in ok}
    _, val_ids = stratified_episode_split(
        episode_ids, val_ratio=val_ratio, seed=seed,
        scene_key_fn=lambda eid: scene_key_by_id[eid])
    val_id_set = set(val_ids)

    # ---- write annotation + packed video for every kept episode (train),
    # and symlink the SAME packed video (+ a duplicate small annotation
    # json) into val/ for the stratified val subset. Per the coordinator's
    # explicit split policy: train gets ALL episodes; val is a diagnostic
    # slice that also appears in train (the benchmark only ever scores
    # generated clips the reader never sees, so this overlap is not
    # circular) -- see this script's module docstring / the task brief.
    def _write_one(r: EpisodeResult) -> None:
        ann = {
            "joint_source": "real",
            "observation.state.joint_position": r.q,
            "provenance": {
                "robot": cfg.kinescore_robot, "source_dir": r.source_dir,
                "bucket": r.bucket, "instruction": r.instruction,
                "n_frames": r.n_frames, "raw_field": cfg.raw_field,
                "joint_indices": list(cfg.joint_indices), "note": cfg.note,
            },
        }
        if r.gripper is not None:
            ann["observation.state.gripper_position"] = r.gripper

        train_ann_path = os.path.join(ann_dir, "train", f"{r.episode_id}.json")
        train_vid_path = os.path.join(vid_dir, "train", f"{r.episode_id}.mp4")
        if overwrite or not os.path.exists(train_ann_path):
            with open(train_ann_path, "w") as f:
                json.dump(ann, f)
        if overwrite or not os.path.exists(train_vid_path):
            full_gt = os.path.join(r.source_dir, "full_gt.mp4")
            _pack_video(full_gt, train_vid_path)

        if r.episode_id in val_id_set:
            val_ann_path = os.path.join(ann_dir, "val", f"{r.episode_id}.json")
            val_vid_path = os.path.join(vid_dir, "val", f"{r.episode_id}.mp4")
            if overwrite or not os.path.exists(val_ann_path):
                with open(val_ann_path, "w") as f:
                    json.dump(ann, f)
            if overwrite or not (os.path.exists(val_vid_path)
                                 or os.path.islink(val_vid_path)):
                os.symlink(os.path.realpath(train_vid_path), val_vid_path)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_write_one, ok))

    # ---- gripper-presence consistency guard (load_split requires it
    # uniform per split; by construction it's uniform per robot here, but
    # verify rather than assume).
    has_gripper = {r.episode_id: (r.gripper is not None) for r in ok}
    if len(set(has_gripper.values())) > 1:
        raise RuntimeError(
            f"[{robot_key}] gripper presence is inconsistent across kept "
            f"episodes -- load_split cannot flatten a partially-gripper "
            f"split into one tensor.")

    # ---- probe actual output geometry/fps for the dataset card
    probed = None
    if ok:
        sample_vid = os.path.join(vid_dir, "train", f"{ok[0].episode_id}.mp4")
        probed = _ffprobe_dims(sample_vid)

    reasons: dict[str, int] = {}
    for r in skipped:
        # bucket the free-text reason into a stable category prefix for the
        # card's summary; keep full text in the per-episode listing.
        key = r.reason.split(":")[0].split("=")[0].strip()
        reasons[key] = reasons.get(key, 0) + 1

    card = {
        "robot": cfg.kinescore_robot,
        "source_input_key": robot_key,
        "source_path": input_root,
        "n_episodes_written": len(ok),
        "n_episodes_val": len(val_id_set),
        "n_episodes_skipped": len(skipped),
        "skip_reasons_summary": reasons,
        "skip_examples": [{"episode_id": r.episode_id, "reason": r.reason,
                           "source_dir": r.source_dir} for r in skipped],
        "joint_channel_indices": list(cfg.joint_indices),
        "joint_field_used": cfg.raw_field,
        "gripper_channel_index": cfg.gripper_index,
        "channel_selection_note": cfg.note,
        "view_width": VIEW_WIDTH, "view_height": VIEW_HEIGHT,
        "n_views": N_VIEWS, "fps": FPS,
        "packed_video_dims_probed": (f"{probed[0]}x{probed[1]}" if probed else None),
        "packing": "height (vstack), matching ViewLayout default",
        "down_sample": 1,
        "val_ratio": val_ratio, "split_seed": seed,
        "scene_key_mode": cfg.scene_key_mode,
        "split_policy": ("train = ALL written episodes; val = a "
                         f"{val_ratio:.0%} scene-stratified diagnostic "
                         "subset ALSO present in train (no circularity vs "
                         "the benchmark, which only scores generated clips "
                         "the reader never sees) -- see module docstring."),
        "video_source_note": ("Read exclusively from full_gt.mp4 (cropped "
                              "x3 + vstacked); view_0/1/2.mp4 were found "
                              "unreliable against metadata num_frames/"
                              "states -- see this script's module "
                              "docstring, discovery 2."),
    }
    card_path = os.path.join(out_dir, "dataset_card.json")
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2)

    print(f"[{robot_key}] written={len(ok)} val={len(val_id_set)} "
         f"skipped={len(skipped)} -> {out_dir}")
    print(f"[{robot_key}] skip reasons: {reasons}")
    return card


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", action="append", dest="robots",
                    choices=sorted(ROBOTS), default=None,
                    help="which robot(s) to convert (repeatable; default: all)")
    ap.add_argument("--data-root", default=None,
                    help="override $KINESCORE_DATA_ROOT")
    ap.add_argument("--out-root", default=None,
                    help="override output root (default: "
                         "$KINESCORE_DATA_ROOT/train)")
    ap.add_argument("--val-ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap candidate episodes per robot (0 = all); smoke test")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    data_root = args.data_root or str(env_path("KINESCORE_DATA_ROOT"))
    out_root = args.out_root or os.path.join(data_root, "train")
    robots = args.robots or sorted(ROBOTS)

    for robot_key in robots:
        convert_robot(robot_key, data_root=data_root, out_root=out_root,
                     val_ratio=args.val_ratio, seed=args.seed,
                     workers=args.workers, limit=args.limit,
                     overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
