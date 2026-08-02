#!/usr/bin/env bash
# What:   Cache DINO tokens and train a 3-view raw_rad reader for franka_panda,
#         aloha_bimanual and airbot_mmk2, from the ctrlworld-domain training
#         trees built by scripts/convert_ctrlworld_input_to_train.py.
# Why:    Every reader shipped so far is 1-view, so the six multiview ctrlworld
#         cells cannot be scored at all. And the Franka reader trained on
#         droid_std scores train 20.14 mm / val 165.35 mm -- it memorises and
#         does not generalise, because droid_std is a different domain (other
#         cameras, one concatenated view). The ctrlworld input footage is the
#         exact domain being scored: same 320x192 panels, same 5 fps, same
#         cameras, and its metadata.json carries frame-aligned joint ground
#         truth, so --down-sample is 1 rather than a guess.
#         Runs sequentially: one GPU, and caching is the expensive half.
# Input:  $KINESCORE_DATA_ROOT/train/<robot>_ctrlworld/{annotation,videos}
# Output: $KINESCORE_CACHE_DIR/<robot>_ctrlworld_tokens  (~86 GB total)
#         $KINESCORE_CKPT_DIR/<robot>_ctrlworld_rawrad.pt (+ .provenance.json)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a; . "$REPO/.env"; set +a
export MKL_THREADING_LAYER=GNU
PY="$REPO/.venv/bin/python"

for ROBOT in franka_panda aloha_bimanual airbot_mmk2; do
  TRAIN="$KINESCORE_DATA_ROOT/train/${ROBOT}_ctrlworld"
  CACHE="$KINESCORE_CACHE_DIR/${ROBOT}_ctrlworld_tokens"
  CKPT="$KINESCORE_CKPT_DIR/${ROBOT}_ctrlworld_rawrad.pt"

  echo "=============== $ROBOT : cache ($(date +%H:%M:%S)) ==============="
  "$PY" -m kinescore.cli.main cache \
    --video-root      "$TRAIN/videos" \
    --annotation-root "$TRAIN/annotation" \
    --out             "$CACHE" \
    --split train --split val \
    --n-views 3 --device cuda

  echo "=============== $ROBOT : train ($(date +%H:%M:%S)) ==============="
  "$PY" -m kinescore.cli.main train-rawrad \
    --cache-root      "$CACHE" \
    --annotation-root "$TRAIN/annotation" \
    --robot "$ROBOT" \
    --down-sample 1 --n-views 3 \
    --steps 6000 --phase-a 1500 --bs 2048 --lr 1e-3 \
    --beta 0.5 --limit-weight 0.05 \
    --device cuda --seed 0 \
    --out "$CKPT"
  echo "=============== $ROBOT : DONE ($(date +%H:%M:%S)) ==============="
done
