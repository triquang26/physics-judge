#!/usr/bin/env bash
# What:   Train the raw_rad-head Airbot MMK2 reader against the camhigh
#         subset-A token cache (cam_head_rgb episodes only: doodled_line,
#         move_pan, storage_bottle_part, storage_mango -- 142 episodes).
# Why:    Test whether training on the camera the ctrlworld eval cells
#         actually render (a cam_high-equivalent view) fixes CAVEAT 3 of
#         $KINESCORE_CKPT_DIR/airbot_mmk2_NOTICE.txt: the ACCEPTED
#         airbot_mmk2_rawrad.pt was trained on cam_third_view/cam_front_rgb,
#         a view the eval cells don't render at all. See
#         cache_airbot_mmk2_camhigh_subsetA.sh for why subset A/B exist as
#         two separate caches.
# STATUS: INCOMPLETE / UNVERIFIED. logs/train_airbot_camhighA.log (as found
#         alongside this script) contains only a startup UserWarning, no
#         `[train]` progress lines and no final val_keypoint_mm -- this run
#         did not reach a checkpoint this audit could confirm. Treat any
#         airbot_mmk2_camhighA.pt found on disk as unvalidated; it is NOT the
#         accepted reader (that is airbot_mmk2_rawrad.pt, see
#         train_airbot_mmk2_fullarm_rawrad.sh). There is no equivalent
#         subset-B training script because subset B's cache was never run
#         either (see cache_airbot_mmk2_camhigh_subsetB.sh).
# Input:  $KINESCORE_CACHE_DIR/airbot_mmk2_camhigh_A_tokens/{train,val}/*.pt
#         $KINESCORE_CACHE_DIR/airbot_mmk2_camhigh_A_train_input/annotation/{train,val}/*.json
# Output: $KINESCORE_CKPT_DIR/airbot_mmk2_camhighA.pt + .provenance.json  (unverified)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
set -a; [ -f .env ] && . ./.env; set +a

: "${KINESCORE_CKPT_DIR:?set KINESCORE_CKPT_DIR in .env, see docs/DATA_PREP.md}"
: "${KINESCORE_CACHE_DIR:?set KINESCORE_CACHE_DIR in .env, see docs/DATA_PREP.md}"

python -m kinescore.cli.main train-rawrad \
  --cache-root      "$KINESCORE_CACHE_DIR/airbot_mmk2_camhigh_A_tokens" \
  --annotation-root "$KINESCORE_CACHE_DIR/airbot_mmk2_camhigh_A_train_input/annotation" \
  --robot airbot_mmk2 --down-sample 1 --n-views 1 \
  --steps 6000 --phase-a 1500 --bs 2048 --lr 1e-3 --beta 0.5 --limit-weight 0.05 \
  --device cuda --seed 0 \
  --out "$KINESCORE_CKPT_DIR/airbot_mmk2_camhighA.pt"
echo "EXIT=$?"
