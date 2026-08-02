#!/usr/bin/env bash
# What:   Precompute the frozen-DINOv3 patch-token cache for Airbot MMK2
#         "camhigh subset A": 4 tasks (doodled_line, move_pan,
#         storage_bottle_part, storage_mango -- 142 episodes) recorded under
#         the LeRobot dataset's mixed-case "Airbot_MMK2_*" task directories,
#         whose high camera is logged under the observation key
#         `cam_head_rgb`.
# Why:    The ACCEPTED reader (airbot_mmk2_rawrad.pt, see
#         train_airbot_mmk2_fullarm_rawrad.sh) was trained on cam_third_view/
#         cam_front_rgb -- CAVEAT 3 of $KINESCORE_CKPT_DIR/
#         airbot_mmk2_NOTICE.txt flags that the generated ctrlworld eval
#         cells this reader ultimately scores only render cam_high,
#         cam_left_wrist and cam_right_wrist, NONE of which is the camera the
#         reader was trained on. This cache (and subsetB's) exist to test
#         retraining on the camera the eval cells actually have.
#         Subset A/B is NOT two different physical cameras -- it is that the
#         source LeRobot dump spells the "high" camera two different ways
#         across two task-naming conventions: `cam_head_rgb` under the
#         mixed-case "Airbot_MMK2_*" dirs (this script, subset A) vs
#         `cam_high_rgb` under the all-caps "AIRBOT_MMK2_*" dirs (subset B,
#         see cache_airbot_mmk2_camhigh_subsetB.sh) -- so two separate
#         `kinescore cache` passes were needed, one per camera-key spelling.
# Input:  $KINESCORE_CACHE_DIR/airbot_mmk2_camhigh_A_train_input/{video,annotation}/{train,val}/
#         (each annotation JSON's provenance.camera == "cam_head_rgb",
#         provenance.subset == "A")
# Output: $KINESCORE_CACHE_DIR/airbot_mmk2_camhigh_A_tokens/{train,val}/*.pt
#         + $KINESCORE_CACHE_DIR/airbot_mmk2_camhigh_A_tokens/cache_provenance.json
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
set -a; [ -f .env ] && . ./.env; set +a
export HF_HUB_DISABLE_XET=1

: "${KINESCORE_CACHE_DIR:?set KINESCORE_CACHE_DIR in .env, see docs/DATA_PREP.md}"

python3 -m kinescore.cli.main cache \
  --video-root      "$KINESCORE_CACHE_DIR/airbot_mmk2_camhigh_A_train_input/video" \
  --annotation-root "$KINESCORE_CACHE_DIR/airbot_mmk2_camhigh_A_train_input/annotation" \
  --out             "$KINESCORE_CACHE_DIR/airbot_mmk2_camhigh_A_tokens" \
  --split train --split val \
  --n-views 1 \
  --device cuda
