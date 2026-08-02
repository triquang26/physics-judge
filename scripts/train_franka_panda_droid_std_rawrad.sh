#!/usr/bin/env bash
# What:   Train the franka_panda raw_rad pose reader on the droid_std cache.
# Why:    The shipped single_arm_rawrad.pt fails its gate at 162.10 mm, but its
#         provenance records n_val_episodes=4 -- that number was measured on a
#         4-episode validation set. The cache on disk has since been split
#         340/60, and the two splits share only one non-empty instruction, so
#         val is effectively scene-independent. Retraining against the real 60
#         tells us whether 162 was a split artefact or a genuine data-volume
#         problem, before anyone spends hours converting more DROID.
# Input:  $KINESCORE_CACHE_DIR/droid_std_franka_rawrad_singleview (44 GB, 400 .pt)
#         $KINESCORE_DATA_ROOT/droid_std_franka_rawrad/annotation/{train,val}
# Output: $KINESCORE_CKPT_DIR/franka_panda_droid_std_rawrad.pt (+ .provenance.json)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a; . "$REPO/.env"; set +a
export MKL_THREADING_LAYER=GNU

# down_sample=3: droid_std logs 167 joint states against 56 video frames.
"$REPO/.venv/bin/python" -m kinescore.cli.main train-rawrad \
  --cache-root      "$KINESCORE_CACHE_DIR/droid_std_franka_rawrad_singleview" \
  --annotation-root "$KINESCORE_DATA_ROOT/droid_std_franka_rawrad/annotation" \
  --robot franka_panda \
  --down-sample 3 --n-views 1 \
  --steps 6000 --phase-a 1500 --bs 2048 --lr 1e-3 \
  --beta 0.5 --limit-weight 0.05 \
  --device cuda --seed 0 \
  --out "$KINESCORE_CKPT_DIR/franka_panda_droid_std_rawrad.pt"
