#!/usr/bin/env python3
# What:   Convert one or more LeRobot v1/v2 task directories (HF-style
#         data/chunk-*/episode_*.parquet + videos/chunk-*/observation.images.
#         <cam>/episode_*.mp4) into kinescore's `cache`/`train-rawrad` input
#         contract: annotation/{train,val}/*.json + video/{train,val}/*.mp4.
# Why:    Every existing conversion (cache/airbot_mmk2_train_input/
#         convert_airbot_real_to_kinescore.py) was a one-off, per-dataset
#         script hardcoding its own task list, camera choice and joint
#         slice. training/datasets.py::load_split's contract (joint_source
#         == "real", observation.state.joint_position at the array width
#         RobotSpec.n_joints expects, one mp4 per episode already packed to
#         the ViewLayout the cache step will read) is the same for every
#         robot; only the per-robot state-slicing and camera set differ.
#         This script is that one general converter, with the three
#         real per-robot mappings this task verified against actual data on
#         disk (see ROBOTS below and each entry's `note`).
# Input:  --task-dir PATH (repeatable), each a LeRobot task directory
#         containing meta/info.json, data/chunk-*/episode_*.parquet,
#         videos/chunk-*/observation.images.<cam>/episode_*.mp4
# Output: --out-video/{train,val}/<episode_id>.mp4
#         --out-annotation/{train,val}/<episode_id>.json
"""See the module docstring above (the What/Why/Input/Output header) for the
contract this produces. Run ``--help`` for the full flag list; ``--robot``
selects one of the three mappings in ROBOTS, each verified against a real
LeRobot dump as part of the kinescore docs/scripts audit this script was
written for (see each entry's ``note``):

fourier_gr1
    LeRobot v2 (PhysicalAI GR-1 Teleop), robot_type "GR1T1", one camera
    (``ego_view_freq20``), 44-D ``observation.state`` with
    ``meta/modality.json`` declaring ``left_arm[0:7] left_hand[7:13]
    left_leg[13:19] neck[19:22] right_arm[22:29] right_hand[29:35]
    right_leg[35:41] waist[41:44]``. ``GR1Spec.n_joints == 17`` is
    ``[left_arm(7), right_arm(7), waist(3)]`` (GR1FK's module docstring) --
    the legs, hands and neck are logged but never predicted, so this script
    slices state[0:7] + state[22:29] + state[41:44] and drops the rest.

airbot_mmk2
    LeRobot v2, one video per camera under
    ``observation.images.<cam>_rgb`` / ``observation.images.cam_third_view``
    (4 cams total: cam_high_rgb, cam_left_wrist_rgb, cam_right_wrist_rgb,
    cam_third_view), 36-D ``observation.state`` = [left_arm_joint_1..6,
    right_arm_joint_1..6, 24 hand dims]. ``AirbotMMK2FK.N_Q == 12``; the
    hand dims are dropped because their claimed `_rad` units don't check out
    numerically (see robots/airbot_mmk2/constants.py, docs/DECISIONS.md D-H)
    -- this script slices state[0:12] and drops [12:36].

aloha_bimanual
    LeRobot v1 (note: v1 and v2 use the identical data/videos parquet+mp4
    layout on disk; the ``codebase_version`` string in meta/info.json is the
    only thing that differs, so no v1-specific parsing is needed here), 4
    cams (cam_high, cam_left_wrist, cam_low, cam_right_wrist), 42-D
    ``observation.state`` with ``meta/modality.json`` declaring
    ``qpos[0:14] qvel[14:28] effort[28:42]``. ``AlohaFK.N_Q == 12``: the
    14-slot qpos block is `[waist, shoulder, elbow, forearm_roll,
    wrist_angle, wrist_rotate, gripper]` per side (left then right), and
    only the 12 non-gripper slots are predicted -- indices 6 and 13 (the
    per-side gripper) are dropped, matching
    robots/aloha/constants.py::LEFT_ARM_JOINTS/RIGHT_ARM_JOINTS. No
    ``gripper_position`` is written for this robot: AlohaSpec's gripper
    is a 2-value ``aux`` (``[left, right]`` opening, see aloha/fk.py), not
    the single scalar ``observation.state.gripper_position`` load_split
    supports -- writing one placeholder value for two independent grippers
    would repeat the exact "channel 13 reads as fabricated data" problem
    airbot_mmk2_NOTICE.txt CAVEAT 2 documents. Gripper is optional in
    load_split's contract, so omitting the key is the honest choice.

Multi-camera packing
---------------------
When a robot has more than one camera, the cameras named in ``--cams`` (in
that order) are vstacked into ONE mp4 per episode via ffmpeg's ``vstack``
filter, matching ``kinescore.core.clip.ViewLayout``'s default
``packing="height"`` -- the SAME packing `kinescore cache --n-views N`
(no ``--view-order`` override) assumes when it slices a packed frame back
into per-view crops. Pass ``--n-views`` matching ``len(--cams)`` (or the
robot's default cam count) to the later ``kinescore cache`` call.

``--down-sample`` is NOT a flag on this script
------------------------------------------------
The annotation JSON this script writes carries the joint log at its OWN
native rate (one row per parquet row) -- exactly as
``convert_airbot_real_to_kinescore.py`` already did. ``--down-sample`` is a
property of the (video fps, joint-log rate) PAIR and is consumed later, by
``kinescore cache``/``train-rawrad`` (see docs/TRAINING.md) -- baking a
guess into this script would silently re-introduce the mis-paired-label risk
that flag exists to prevent. What this script DOES do is probe each
episode's video frame count against its parquet row count (via ffprobe) and
print the ratio, so the operator can read off the down_sample value
directly instead of guessing -- see ``_probe_frame_count``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RobotMap:
    """One robot's LeRobot-state -> kinescore-joint-target mapping."""

    n_joints: int
    default_cams: tuple[str, ...]
    slice_fn: Callable[[Sequence[float]], list[float]]
    note: str
    gripper_fn: Callable[[Sequence[float]], float] | None = None


def _gr1_slice(state44: Sequence[float]) -> list[float]:
    # left_arm[0:7] + right_arm[22:29] + waist[41:44] -> 17, per
    # meta/modality.json (verified against a real gr1_teleop info.json).
    s = list(state44)
    if len(s) != 44:
        raise ValueError(f"expected 44-D GR-1 observation.state, got {len(s)}")
    return s[0:7] + s[22:29] + s[41:44]


def _airbot_slice(state36: Sequence[float]) -> list[float]:
    s = list(state36)
    if len(s) != 36:
        raise ValueError(f"expected 36-D Airbot MMK2 observation.state, got {len(s)}")
    return s[0:12]


def _aloha_slice(state42: Sequence[float]) -> list[float]:
    # qpos block is state[0:14] = [left(7), right(7)], each
    # [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper].
    # Drop the two gripper slots (indices 6, 13) -> 12 predicted arm joints.
    s = list(state42)
    if len(s) != 42:
        raise ValueError(f"expected 42-D ALOHA observation.state, got {len(s)}")
    qpos = s[0:14]
    return qpos[0:6] + qpos[7:13]


ROBOTS: dict[str, RobotMap] = {
    "fourier_gr1": RobotMap(
        n_joints=17, default_cams=("ego_view_freq20",), slice_fn=_gr1_slice,
        note="state[0:7]+state[22:29]+state[41:44] per meta/modality.json"),
    "airbot_mmk2": RobotMap(
        n_joints=12,
        default_cams=("cam_high_rgb", "cam_left_wrist_rgb",
                      "cam_right_wrist_rgb", "cam_third_view"),
        slice_fn=_airbot_slice,
        note="state[0:12]; hand dims [12:36] dropped, unverified units"),
    "aloha_bimanual": RobotMap(
        n_joints=12,
        default_cams=("cam_high", "cam_left_wrist", "cam_low", "cam_right_wrist"),
        slice_fn=_aloha_slice,
        note="qpos=state[0:14]; grippers (idx 6,13) dropped, see aux contract"),
}


def _episode_ids(task_dir: str) -> list[str]:
    parqs = sorted(glob.glob(os.path.join(task_dir, "data", "chunk-*", "episode_*.parquet")))
    return [os.path.splitext(os.path.basename(p))[0].replace("episode_", "") for p in parqs]


def _load_task_instructions(task_dir: str) -> dict[str, str]:
    """``episode id -> first instruction string``, from ``meta/episodes.jsonl``.

    Used as the scene key for ``--stratified-split`` (see that flag's help):
    grouping by the actual task/instruction text, not the episode index, so a
    val split never separates near-duplicate reruns of the same instruction
    from their train-side siblings. Returns ``{}`` (not an error) if the file
    is absent -- callers fall back to the episode id itself, an honest
    admission that this task dir carries no recoverable scene signal (same
    reasoning as ``kinescore.training.splits.default_scene_key`` for bare
    integer ids).
    """
    path = os.path.join(task_dir, "meta", "episodes.jsonl")
    if not os.path.exists(path):
        return {}
    out: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ep = f"{int(d['episode_index']):06d}"
            tasks = d.get("tasks") or []
            out[ep] = tasks[0] if tasks else ""
    return out


def _parquet_path(task_dir: str, ep: str) -> str:
    hits = glob.glob(os.path.join(task_dir, "data", "chunk-*", f"episode_{ep}.parquet"))
    if not hits:
        raise FileNotFoundError(f"no parquet for episode {ep} under {task_dir}")
    return hits[0]


def _video_path(task_dir: str, cam: str, ep: str) -> str:
    hits = glob.glob(os.path.join(
        task_dir, "videos", "chunk-*", f"observation.images.{cam}", f"episode_{ep}.mp4"))
    if not hits:
        raise FileNotFoundError(f"no video for cam={cam!r} episode {ep} under {task_dir}")
    return hits[0]


def _probe_frame_count(path: str) -> int | None:
    """``ffprobe`` frame count via the container's own frame index, no full decode.

    Returns ``None`` (rather than raising) on any ffprobe failure -- this is
    a diagnostic aid, not a correctness gate; a missing count degrades to
    "no ratio printed for this episode", not a hard failure.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_packets", "-show_entries", "stream=nb_read_packets",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30, check=True)
        return int(out.stdout.strip())
    except Exception:
        return None


def _pack_views(cam_paths: list[str], out_path: str) -> None:
    """vstack ``cam_paths`` (in order) into one mp4 -- ``ViewLayout(packing='height')``."""
    if len(cam_paths) == 1:
        shutil.copyfile(cam_paths[0], out_path) if not os.path.islink(cam_paths[0]) \
            else os.symlink(os.path.realpath(cam_paths[0]), out_path)
        return
    inputs = []
    for p in cam_paths:
        inputs += ["-i", p]
    filt = "".join(f"[{i}:v]" for i in range(len(cam_paths))) + \
        f"vstack=inputs={len(cam_paths)}[v]"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs,
          "-filter_complex", filt, "-map", "[v]",
          "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path]
    subprocess.run(cmd, check=True)


def convert(*, robot: str, task_dirs: list[str], out_video: str, out_annotation: str,
           cams: list[str], val_ratio: float, limit: int, overwrite: bool,
           probe: bool, stratified_split: bool = False, seed: int = 0) -> None:
    """See the module docstring for the general contract. ``stratified_split``
    switches the train/val assignment strategy (see ``--stratified-split``'s
    CLI help): instead of each ``task_dir`` independently sending its own
    last ``val_ratio`` fraction of episodes (in on-disk order) to val, every
    episode across ALL ``task_dirs`` is pooled and partitioned in one pass by
    :func:`kinescore.training.splits.stratified_episode_split`, keyed by each
    episode's own instruction text (:func:`_load_task_instructions`) rather
    than its numeric id or task-dir name -- this is what lets two horizon
    directories (e.g. ``makovian``/``non_makovian``) merge into one training
    pool without a near-duplicate instruction landing on both sides of the
    split.
    """
    rmap = ROBOTS[robot]
    for split in ("train", "val"):
        os.makedirs(os.path.join(out_video, split), exist_ok=True)
        os.makedirs(os.path.join(out_annotation, split), exist_ok=True)

    totals = {"train": 0, "val": 0}
    skipped: list[tuple[str, str]] = []
    ratio_report: list[str] = []

    # ---- resolve per-episode (task_dir, ep, ep_out_id, split, codebase_version) ----
    plan: list[tuple[str, str, str, str, str | None]] = []
    if stratified_split:
        from kinescore.training.splits import stratified_episode_split

        pool_scene_key: dict[str, str] = {}
        pool_meta: dict[str, tuple[str, str, str | None]] = {}  # out_id -> (task_dir, ep, cbv)
        for task_dir in task_dirs:
            task = os.path.basename(os.path.normpath(task_dir))
            info_path = os.path.join(task_dir, "meta", "info.json")
            codebase_version = None
            if os.path.exists(info_path):
                with open(info_path) as f:
                    codebase_version = json.load(f).get("codebase_version")
            instructions = _load_task_instructions(task_dir)

            ep_ids = _episode_ids(task_dir)
            if limit:
                ep_ids = ep_ids[:limit]
            for ep in ep_ids:
                ep_out_id = f"{task}__{ep}"
                instr = instructions.get(ep, "")
                pool_scene_key[ep_out_id] = instr if instr else ep_out_id
                pool_meta[ep_out_id] = (task_dir, ep, codebase_version)

        train_ids, val_ids = stratified_episode_split(
            tuple(pool_scene_key), val_ratio=val_ratio, seed=seed,
            scene_key_fn=lambda out_id: pool_scene_key[out_id])
        print(f"stratified split: {len(train_ids)} train / {len(val_ids)} val "
             f"(pool {len(pool_scene_key)}, target val_ratio={val_ratio}, seed={seed})")

        for split, ids in (("train", train_ids), ("val", val_ids)):
            for ep_out_id in ids:
                task_dir, ep, cbv = pool_meta[ep_out_id]
                plan.append((task_dir, ep, ep_out_id, split, cbv))
    else:
        for task_dir in task_dirs:
            task = os.path.basename(os.path.normpath(task_dir))
            info_path = os.path.join(task_dir, "meta", "info.json")
            codebase_version = None
            if os.path.exists(info_path):
                with open(info_path) as f:
                    codebase_version = json.load(f).get("codebase_version")

            ep_ids = _episode_ids(task_dir)
            if limit:
                ep_ids = ep_ids[:limit]
            n = len(ep_ids)
            n_val = max(1, round(n * val_ratio)) if n > 1 else 0

            for i, ep in enumerate(ep_ids):
                ep_out_id = f"{task}__{ep}"
                split = "val" if i >= (n - n_val) else "train"
                plan.append((task_dir, ep, ep_out_id, split, codebase_version))

    for task_dir, ep, ep_out_id, split, codebase_version in plan:
        task = os.path.basename(os.path.normpath(task_dir))
        try:
            cam_paths = [_video_path(task_dir, c, ep) for c in cams]
        except FileNotFoundError as e:
            skipped.append((ep_out_id, str(e)))
            continue

        pq_path = _parquet_path(task_dir, ep)
        df = pd.read_parquet(pq_path)
        state = df["observation.state"].to_numpy()
        try:
            q = [[float(x) for x in rmap.slice_fn(list(row))] for row in state]
        except ValueError as e:
            skipped.append((ep_out_id, str(e)))
            continue

        ann = {
            "joint_source": "real",
            "observation.state.joint_position": q,
            "provenance": {
                "robot": robot, "task": task, "task_dir": task,
                "cams": cams, "source_episode": ep,
                "codebase_version": codebase_version,
                "note": rmap.note,
            },
        }
        ann_path = os.path.join(out_annotation, split, f"{ep_out_id}.json")
        if overwrite or not os.path.exists(ann_path):
            with open(ann_path, "w") as f:
                json.dump(ann, f)

        vid_path = os.path.join(out_video, split, f"{ep_out_id}.mp4")
        if overwrite or not os.path.exists(vid_path):
            _pack_views(cam_paths, vid_path)

        if probe:
            nf = _probe_frame_count(cam_paths[0])
            if nf is not None and nf > 0:
                ratio_report.append(
                    f"{ep_out_id}: {len(state)} joint rows / {nf} video "
                    f"frames = down_sample~{len(state) / nf:.3f}")

        totals[split] += 1

    print("totals:", totals)
    print("skipped:", len(skipped))
    for ep_id, reason in skipped[:20]:
        print(f"  skip {ep_id}: {reason}")
    if len(skipped) > 20:
        print(f"  ... and {len(skipped) - 20} more")
    if ratio_report:
        print("frame/row ratios (first 10, verify --down-sample against these):")
        for line in ratio_report[:10]:
            print(f"  {line}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True, choices=sorted(ROBOTS),
                    help="which RobotMap in ROBOTS to apply")
    ap.add_argument("--task-dir", action="append", required=True, dest="task_dirs",
                    help="LeRobot task directory (repeatable); each must contain "
                         "meta/, data/chunk-*/, videos/chunk-*/")
    ap.add_argument("--out-video", required=True)
    ap.add_argument("--out-annotation", required=True)
    ap.add_argument("--cams", default=None,
                    help="comma-separated camera keys, in vstack (top-to-bottom) "
                         "order; default is the robot's ROBOTS[robot].default_cams")
    ap.add_argument("--val-ratio", type=float, default=0.10,
                    help="val fraction. Without --stratified-split: deterministic "
                         "last-N%% of EACH task dir's episodes, in on-disk order "
                         "(matches convert_airbot_real_to_kinescore.py's split). "
                         "With --stratified-split: target fraction of the WHOLE "
                         "pooled episode count passed to "
                         "kinescore.training.splits.stratified_episode_split.")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap episodes per task dir (0 = all); for a smoke test")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the ffprobe frame/row ratio report (faster)")
    ap.add_argument("--stratified-split", action="store_true",
                    help="Pool episodes across ALL --task-dir values (e.g. two "
                         "horizon directories of the same robot) and assign "
                         "train/val with kinescore.training.splits."
                         "stratified_episode_split, keyed by each episode's own "
                         "instruction text from meta/episodes.jsonl (falling back "
                         "to the output episode id when episodes.jsonl is absent "
                         "or an episode has no instruction) -- NOT the per-task-dir "
                         "last-N%% split this script otherwise uses. Use this "
                         "when task dirs are two horizons of the SAME visual "
                         "domain that should merge into one training pool rather "
                         "than being scored/split independently.")
    ap.add_argument("--seed", type=int, default=0,
                    help="shuffle seed for --stratified-split's scene ordering "
                         "(kinescore.training.splits.stratified_episode_split)")
    args = ap.parse_args()

    cams = args.cams.split(",") if args.cams else list(ROBOTS[args.robot].default_cams)

    convert(robot=args.robot, task_dirs=args.task_dirs, out_video=args.out_video,
           out_annotation=args.out_annotation, cams=cams, val_ratio=args.val_ratio,
           limit=args.limit, overwrite=args.overwrite, probe=not args.no_probe,
           stratified_split=args.stratified_split, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
