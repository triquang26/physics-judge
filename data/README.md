# `data/` — drop your downloads here

This tree ships **empty**. The directories exist so you can see where each
download goes without reading anything else. Point `KINESCORE_DATA_ROOT` at
this folder (or at any folder with this shape) and nothing needs moving.

```bash
cp .env.example .env
# .env:  KINESCORE_DATA_ROOT=/absolute/path/to/this/repo/data
kinescore doctor          # tells you what is present and what is missing
```

Everything below `data/` is gitignored except these placeholders — the repo
never carries video. A prior extraction of this same code shipped 9 GB of
outputs inside its own git history, which cannot be undone without a rewrite
that breaks every clone. That is why `tools/check_repo_hygiene.py` runs before
a commit, not in review.

## What goes where

| directory | source | size |
|---|---|---|
| `video_gen_physics/` | HF `doanh25032004/video_gen_physics` — **`dense/` only** | ~13 GB |
| `video_gen_physics_real_video/` | HF `doanh25032004/video_gen_physics_real_video` | ~4 GB |
| `cosmos_synthetic_data/` | HF `doanh25032004/cosmos_synthetic_data` | ~144 MB |
| `train/franka_droid_std/` | DROID, converted: `annotation/{train,val}/*.json` + `videos/` | ~450 MB |
| `train/gr1_teleop/` | HF `nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1` (LeRobot v2), converted via `scripts/convert_lerobot_to_train.py --robot fourier_gr1` | ~1.3 GB |
| `train/airbot_mmk2/` | Airbot MMK2 real episodes (LeRobot v2), converted via `scripts/convert_lerobot_to_train.py --robot airbot_mmk2` | ~3 GB |
| `train/aloha_bimanual/` | ALOHA real episodes (LeRobot v1), converted via `scripts/convert_lerobot_to_train.py --robot aloha_bimanual` | not yet populated |

Subdirectory names match their HuggingFace repo names exactly, so a plain
`huggingface-cli download <repo> --local-dir data/<repo-name>` lands correctly
with nothing to rename. `kinescore data pull --config configs/benchmark.yaml`
does the same and additionally derives its `allow_patterns` from the benchmark
matrix — **use `--dry-run` first**: `video_gen_physics` has nine top-level
directories and `dense/` is one of them; pulling all nine is hundreds of GB.

The first three are what gets **scored**. `train/` is what pose readers are
**trained** on — one subdirectory per robot, because a reader is only valid for
the robot it was trained on and its own URDF. Each `train/<robot>/` directory
must satisfy the `annotation/{train,val}/*.json` + `videos/{train,val}/*.mp4`
contract regardless of the source dataset's own layout;
`scripts/convert_lerobot_to_train.py` is the general converter from a raw
LeRobot v1/v2 dump to that contract, verified against real GR-1/Airbot
MMK2/ALOHA data — see [`../docs/DATA_PREP.md`](../docs/DATA_PREP.md).

## The other roots are not here

`data/` is only `KINESCORE_DATA_ROOT`. Four more paths live outside the repo
because they differ in kind:

| variable | why not here |
|---|---|
| `KINESCORE_ASSETS` | URDF + meshes; the GR-1 tree alone is ~285 MB |
| `KINESCORE_CKPT_DIR` | trained weights, not source |
| `KINESCORE_CACHE_DIR` | derived backbone features — **~56 GB, delete freely**, `kinescore cache` rebuilds it |
| `KINESCORE_OUTPUT_DIR` | run outputs |

None has a fallback: an unset variable raises and names itself
(`src/kinescore/paths.py`) rather than silently resolving to a path that
happens to exist on someone else's machine.

Full detail, including the download flags that matter, the per-generator file
contract, and a warning about directory names in this dataset that do not mean
what they say, is in [`../docs/DATA_PREP.md`](../docs/DATA_PREP.md).
