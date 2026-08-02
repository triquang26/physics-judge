#!/usr/bin/env bash
# What:   Train the raw_rad-head (ReadoutV2Head) Airbot MMK2 reader --
#         `kinescore train-rawrad` -- against the full-arm token cache.
#         This is the run that produced the checkpoint that got ACCEPTED.
# Why:    The squashed loop (train_airbot_mmk2_fullarm_squashed.sh) could not
#         make limit_violation_frac/limit_excess_rad observable at all (D7).
#         Rerunning the same full-arm cache through train-rawrad instead
#         fixes that, and clears the reader-acceptance bar: val_keypoint_mm =
#         19.5194 at best_step 5500 (train 8.40), against an untrained
#         baseline of 359.93 and Franka's REJECTED 162.10 -- 19.52 is well
#         inside the accepted band.
# Input:  $KINESCORE_CACHE_DIR/airbot_mmk2_tokens/{train,val}/*.pt
#         $KINESCORE_CACHE_DIR/airbot_mmk2_train_input/annotation/{train,val}/*.json
# Output: $KINESCORE_CKPT_DIR/airbot_mmk2_rawrad.pt + .provenance.json  (ACCEPTED)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
set -a; [ -f .env ] && . ./.env; set +a

: "${KINESCORE_CKPT_DIR:?set KINESCORE_CKPT_DIR in .env, see docs/DATA_PREP.md}"
: "${KINESCORE_CACHE_DIR:?set KINESCORE_CACHE_DIR in .env, see docs/DATA_PREP.md}"

python -m kinescore.cli.main train-rawrad \
  --cache-root      "$KINESCORE_CACHE_DIR/airbot_mmk2_tokens" \
  --annotation-root "$KINESCORE_CACHE_DIR/airbot_mmk2_train_input/annotation" \
  --robot airbot_mmk2 --down-sample 1 --n-views 1 \
  --steps 6000 --phase-a 1500 --bs 2048 --lr 1e-3 --beta 0.5 --limit-weight 0.05 \
  --device cuda --seed 0 \
  --out "$KINESCORE_CKPT_DIR/airbot_mmk2_rawrad.pt"
echo "EXIT=$?"
