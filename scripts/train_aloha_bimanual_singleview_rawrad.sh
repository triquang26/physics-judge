#!/usr/bin/env bash
# What:   Cache DINO tokens and train a 1-view raw_rad aloha_bimanual reader
#         from the real ALOHA singleview domain (bimanual/singleview/
#         {makovian,non_makovian} under $KINESCORE_DATA_ROOT/
#         video_gen_physics_real_video), converted+merged by
#         scripts/convert_lerobot_to_train.py --stratified-split into
#         $KINESCORE_DATA_ROOT/train/aloha_bimanual_singleview.
# Why:    This is the last reader missing for full benchmark coverage: the
#         four aloha_bimanual singleview cells (dreamdojo/dreamgen, both
#         horizons) have generated clips and no reader that can read them --
#         the existing aloha_bimanual_ctrlworld_rawrad.pt is 3-view. 300
#         episodes (150 makovian + 150 non_makovian, merged: horizon is a
#         benchmark label, not a visual domain) with frame-aligned joint
#         ground truth -- ffprobe-verified down_sample=1.0 on ALL 300
#         episodes (not a sample), one camera (cam_high). --frame-chunk 256
#         on the cache step: episodes run up to 3000 frames (median 1000, 25
#         of 300 over 1000, 12 over 1500) -- the same long-episode profile
#         that made the fourier_gr1_singleview cache OOM at the default
#         frame_chunk=0 (see that script's header), so chunking is carried
#         over here too; it is numerically a no-op (each frame is encoded
#         independently), it only bounds memory. The 12-D slice (qpos minus
#         the two per-side gripper channels, see
#         convert_lerobot_to_train.py::_aloha_slice) was checked against
#         get_robot("aloha_bimanual").q_lo/.q_hi over all 300 episodes:
#         0.111% of frames have any out-of-limit channel (313/280929),
#         confined to channel 8 (right shoulder) in exactly 2 episodes --
#         normal teleop noise, not a wrong slice.
#         Runs sequentially: one GPU, and caching is the expensive half.
# Input:  $KINESCORE_DATA_ROOT/train/aloha_bimanual_singleview/{annotation,videos}
# Output: $KINESCORE_CACHE_DIR/aloha_bimanual_singleview_tokens
#         $KINESCORE_CKPT_DIR/aloha_bimanual_singleview_rawrad.pt (+ .provenance.json)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a; . "$REPO/.env"; set +a
export MKL_THREADING_LAYER=GNU
PY="$REPO/.venv/bin/python"

ROBOT=aloha_bimanual
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
