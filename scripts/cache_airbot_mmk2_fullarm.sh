#!/usr/bin/env bash
# What:   Precompute the frozen-DINOv3 patch-token cache for the Airbot MMK2
#         "full-arm" training set: all 6 real tasks, one wide camera per task
#         (cam_third_view or cam_front_rgb, whichever shows the full bimanual
#         arm chain -- see cache/airbot_mmk2_train_input/
#         convert_airbot_real_to_kinescore.py's per-task camera table).
# Why:    `kinescore train-rawrad` reads precomputed tokens, not video -- this
#         is the `kinescore cache` pass that must run once before
#         train_airbot_mmk2_fullarm_rawrad.sh (the run that produced the
#         ACCEPTED airbot_mmk2_rawrad.pt, val 19.52mm) can train anything.
#         "Full-arm" was the FIRST camera choice tried, because it shows the
#         whole kinematic chain the reader is scored against; see
#         cache_airbot_mmk2_camhigh_subsetA.sh's header for why a second,
#         differently-camera'd attempt exists at all.
# Input:  $KINESCORE_CACHE_DIR/airbot_mmk2_train_input/{video,annotation}/{train,val}/
#         (produced by convert_airbot_real_to_kinescore.py; the 12-D joint
#         target is [left_arm_joint_1..6, right_arm_joint_1..6] rad)
# Output: $KINESCORE_CACHE_DIR/airbot_mmk2_tokens/{train,val}/*.pt
#         + $KINESCORE_CACHE_DIR/airbot_mmk2_tokens/cache_provenance.json
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
set -a; [ -f .env ] && . ./.env; set +a
export HF_HUB_DISABLE_XET=1

: "${KINESCORE_CACHE_DIR:?set KINESCORE_CACHE_DIR in .env, see docs/DATA_PREP.md}"

python3 -m kinescore.cli.main cache \
  --video-root      "$KINESCORE_CACHE_DIR/airbot_mmk2_train_input/video" \
  --annotation-root "$KINESCORE_CACHE_DIR/airbot_mmk2_train_input/annotation" \
  --out             "$KINESCORE_CACHE_DIR/airbot_mmk2_tokens" \
  --split train --split val \
  --n-views 1 \
  --device cuda
