#!/usr/bin/env bash
# What:   Download the three trees still missing for full dense coverage:
#         real_video/{humanoid,single_arm}/singleview (domain-matched real
#         footage with joint GT) and video_gen_physics/dense/bimanual/output
#         (the generated bimanual clips).
# Why:    Every singleview cell was reported unscoreable for lack of a
#         domain-matched reader, and every bimanual cell for lack of data.
#         Both were wrong: real_video carries GR1T2 ego-view footage (130 ep
#         per horizon, 44-d state -- the same camera humanoid.pt trains on)
#         and RoboChallenge 7-d footage at 15 fps, which is exactly the
#         dreamdojo single_arm rate. Episode counts match the generated side
#         one-for-one (130<->130, 120<->120), so these are the very episodes
#         the clips were generated from. bimanual/output simply was never
#         pulled.
#         meta/episodes_stats.jsonl is excluded: it is 965 MB per tree of
#         per-episode statistics nothing here reads.
# Input:  HF doanh25032004/{video_gen_physics_real_video,video_gen_physics}
# Output: $KINESCORE_DATA_ROOT/video_gen_physics_real_video/{humanoid,single_arm}/singleview/
#         $KINESCORE_DATA_ROOT/video_gen_physics/dense/bimanual/output/
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a; . "$REPO/.env"; set +a
# Measured earlier this session: disabling Xet took throughput 374 -> 2893 KB/s.
export HF_HUB_DISABLE_XET=1

# `huggingface-cli` is deprecated in this env and refuses to run; `hf` is the
# current entry point and takes --type rather than --repo-type.
hf_get() {  # repo, local_dir, patterns...
  local repo="$1" dest="$2"; shift 2
  local args=()
  for p in "$@"; do args+=(--include "$p"); done
  "$REPO/.venv/bin/hf" download "$repo" --type dataset \
    --local-dir "$dest" "${args[@]}" --exclude "**/episodes_stats.jsonl" \
    --max-workers 8
}

hf_get doanh25032004/video_gen_physics_real_video \
  "$KINESCORE_DATA_ROOT/video_gen_physics_real_video" \
  "humanoid/singleview/makovian/*" "humanoid/singleview/non_makovian/*" \
  "single_arm/singleview/makovian/*" "single_arm/singleview/non_makovian/*"

hf_get doanh25032004/video_gen_physics \
  "$KINESCORE_DATA_ROOT/video_gen_physics" \
  "dense/bimanual/output/*"

echo "PULL DONE $(date +%H:%M:%S)"
