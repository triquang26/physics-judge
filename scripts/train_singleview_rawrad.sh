#!/usr/bin/env bash
# What:   Cache DINO tokens and train a 1-view raw_rad fourier_gr1 reader from
#         the real GR1T2 singleview domain (humanoid/singleview/{makovian,
#         non_makovian} under $KINESCORE_DATA_ROOT/video_gen_physics_real_video),
#         converted+merged by scripts/convert_lerobot_to_train.py --stratified-split
#         into $KINESCORE_DATA_ROOT/train/fourier_gr1_singleview.
# Why:    The singleview dreamgen/dreamdojo cells (fourier_gr1, franka_panda)
#         have no reader trained on their own visual domain -- every existing
#         checkpoint is either 3-view (airbot_mmk2/aloha_bimanual/franka_panda
#         ctrlworld readers) or trained on GR1T1 GRO0T-Teleop, not this GR1T2
#         export (humanoid.pt, val 19.19mm, domain never verified against this
#         tree -- see the coordinator brief's Task 3 for the in-domain
#         comparison this script's output feeds). 260 episodes (130 makovian +
#         130 non_makovian, merged: horizon is a benchmark label, not a visual
#         domain) with frame-aligned joint ground truth (ffprobe-verified
#         down_sample=1.0 on every sampled episode of both horizons).
#         --frame-chunk 256 on the cache step: GR1T2 singleview episodes run
#         up to 2840 frames (median 208, 9 episodes over 1000) -- the default
#         frame_chunk=0 (whole episode through the backbone in one call)
#         OOMed an otherwise-idle 80GB H100 on this tree (escalating
#         18GB/26GB/35GB allocation failures, then the GPU pinned near-full
#         at 80/81GB), the exact failure mode encode_clip's own docstring
#         documents for long episodes at default frame_chunk=0. Chunking is
#         numerically a no-op (each frame is encoded independently, see that
#         docstring) -- this only bounds memory.
#         franka_panda's matching tree (single_arm/singleview, RoboChallenge,
#         7-DOF) is DELIBERATELY NOT included here: its observation.state
#         per-channel range ([0.14,0.53]/[-0.30,0.29]/[-0.01,0.45] m-scale for
#         dims 0-2, [0,0.087] for dim 6) does not correspond to
#         get_robot("franka_panda").q_lo/.q_hi (Panda joint radians, e.g.
#         dim3 always in [-3.07,-0.07]) at all -- it reads as a 6-DoF
#         Cartesian EE pose + gripper opening (meta/modality.json's "state"
#         key is unlabeled, dtype float32 shape [7], no rotation_type/units),
#         not 7 joint angles. Training a Franka FK reader against it would
#         silently fit garbage. See this run's final report for the full
#         per-channel evidence; flagged to the coordinator rather than trained.
#         Runs sequentially: one GPU, and caching is the expensive half.
# Input:  $KINESCORE_DATA_ROOT/train/fourier_gr1_singleview/{annotation,videos}
# Output: $KINESCORE_CACHE_DIR/fourier_gr1_singleview_tokens
#         $KINESCORE_CKPT_DIR/fourier_gr1_singleview_rawrad.pt (+ .provenance.json)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a; . "$REPO/.env"; set +a
export MKL_THREADING_LAYER=GNU
PY="$REPO/.venv/bin/python"

ROBOT=fourier_gr1
TRAIN="$KINESCORE_DATA_ROOT/train/${ROBOT}_singleview"
CACHE="$KINESCORE_CACHE_DIR/${ROBOT}_singleview_tokens"
CKPT="$KINESCORE_CKPT_DIR/${ROBOT}_singleview_rawrad.pt"

echo "=============== $ROBOT singleview : cache ($(date +%H:%M:%S)) ==============="
"$PY" -m kinescore.cli.main cache \
  --video-root      "$TRAIN/videos" \
  --annotation-root "$TRAIN/annotation" \
  --out             "$CACHE" \
  --split train --split val \
  --n-views 1 --device cuda --frame-chunk 256

echo "=============== $ROBOT singleview : train ($(date +%H:%M:%S)) ==============="
"$PY" -m kinescore.cli.main train-rawrad \
  --cache-root      "$CACHE" \
  --annotation-root "$TRAIN/annotation" \
  --robot "$ROBOT" \
  --down-sample 1 --n-views 1 \
  --steps 6000 --phase-a 1500 --bs 2048 --lr 1e-3 \
  --beta 0.5 --limit-weight 0.05 \
  --device cuda --seed 0 \
  --out "$CKPT"
echo "=============== $ROBOT singleview : DONE ($(date +%H:%M:%S)) ==============="
