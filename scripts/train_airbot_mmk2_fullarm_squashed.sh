#!/usr/bin/env bash
# What:   Train the squashed-head (AttentivePoseHead) Airbot MMK2 reader via
#         the (now-removed) `kinescore train` subcommand, against the
#         full-arm token cache (cache_airbot_mmk2_fullarm.sh's output).
# Why:    First Airbot MMK2 training attempt, before this repo's raw_rad-only
#         refactor. Produced airbot_mmk2.pt, val_keypoint_mm=18.31 -- but
#         `limit_semantics` is "squashed" (sigmoid-into-limits), not the
#         "raw_rad" the task this checkpoint was built for actually asked
#         for (see $KINESCORE_CKPT_DIR/airbot_mmk2_NOTICE.txt CAVEAT 1), so
#         limit_violation_frac/limit_excess_rad read null against it, never a
#         real number. Superseded by train_airbot_mmk2_fullarm_rawrad.sh, the
#         run that IS accepted; kept here as the historical record of why the
#         squashed attempt was not good enough.
# CAVEAT: `kinescore train` (the squashed-head loop, AttentivePoseHead) has
#         since been REMOVED from this codebase entirely -- see
#         src/kinescore/training/trainer_rawrad.py's module docstring ("D7
#         addendum"): training/trainer.py, heads/attentive.py, heads/mlp.py,
#         heads/disentangled.py and readers/ensemble.py are gone; only
#         `train-rawrad` (ReadoutV2Head) remains. This script WILL fail
#         against the current CLI -- it is preserved verbatim for provenance,
#         not as a runnable recipe. Use train_airbot_mmk2_fullarm_rawrad.sh.
# Input:  $KINESCORE_CACHE_DIR/airbot_mmk2_tokens/{train,val}/*.pt
#         $KINESCORE_CACHE_DIR/airbot_mmk2_train_input/annotation/{train,val}/*.json
# Output: $KINESCORE_CKPT_DIR/airbot_mmk2.pt + .provenance.json  (squashed, superseded)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
set -a; [ -f .env ] && . ./.env; set +a
export HF_HUB_DISABLE_XET=1

: "${KINESCORE_CKPT_DIR:?set KINESCORE_CKPT_DIR in .env, see docs/DATA_PREP.md}"
: "${KINESCORE_CACHE_DIR:?set KINESCORE_CACHE_DIR in .env, see docs/DATA_PREP.md}"
mkdir -p "$KINESCORE_CKPT_DIR"

python3 -m kinescore.cli.main train \
  --cache-root      "$KINESCORE_CACHE_DIR/airbot_mmk2_tokens" \
  --annotation-root "$KINESCORE_CACHE_DIR/airbot_mmk2_train_input/annotation" \
  --robot airbot_mmk2 \
  --down-sample 1 \
  --n-views 1 \
  --steps 6000 --bs 2048 --lr 1e-3 \
  --device cuda \
  --out "$KINESCORE_CKPT_DIR/airbot_mmk2.pt"
