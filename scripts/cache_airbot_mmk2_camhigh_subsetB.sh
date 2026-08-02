#!/usr/bin/env bash
# What:   Precompute the frozen-DINOv3 patch-token cache for Airbot MMK2
#         "camhigh subset B": 2 tasks (boxs_storage, mobile_car) recorded
#         under the LeRobot dataset's all-caps "AIRBOT_MMK2_*" task
#         directories, whose high camera is logged under the observation key
#         `cam_high_rgb`.
# Why:    Companion to cache_airbot_mmk2_camhigh_subsetA.sh -- see that
#         script's header for the full "why two subsets" explanation
#         (`cam_head_rgb` under mixed-case task dirs vs `cam_high_rgb` under
#         all-caps task dirs, the same conceptual camera spelled two ways,
#         hence two `kinescore cache` passes).
# STATUS: UNRUN as of this repo's docs/scripts audit. Unlike subset A (which
#         has a completed cache_camhigh_A.log and a token cache on disk),
#         there is no cache_camhigh_B.log and no
#         airbot_mmk2_camhigh_B_tokens/ directory -- only the converted
#         video/annotation input exists. This script is included for
#         symmetry with subset A and has not been executed or verified.
# Input:  $KINESCORE_CACHE_DIR/airbot_mmk2_camhigh_B_train_input/{video,annotation}/{train,val}/
#         (each annotation JSON's provenance.camera == "cam_high_rgb",
#         provenance.subset == "B")
# Output: $KINESCORE_CACHE_DIR/airbot_mmk2_camhigh_B_tokens/{train,val}/*.pt
#         + $KINESCORE_CACHE_DIR/airbot_mmk2_camhigh_B_tokens/cache_provenance.json
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
set -a; [ -f .env ] && . ./.env; set +a
export HF_HUB_DISABLE_XET=1

: "${KINESCORE_CACHE_DIR:?set KINESCORE_CACHE_DIR in .env, see docs/DATA_PREP.md}"

python3 -m kinescore.cli.main cache \
  --video-root      "$KINESCORE_CACHE_DIR/airbot_mmk2_camhigh_B_train_input/video" \
  --annotation-root "$KINESCORE_CACHE_DIR/airbot_mmk2_camhigh_B_train_input/annotation" \
  --out             "$KINESCORE_CACHE_DIR/airbot_mmk2_camhigh_B_tokens" \
  --split train --split val \
  --n-views 1 \
  --device cuda
