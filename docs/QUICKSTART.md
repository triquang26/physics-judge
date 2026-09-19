# Quickstart

Checkpoint in, scores out. Every command resolves its paths from the process
environment (`KINESCORE_DATA_ROOT`, `KINESCORE_CACHE_DIR`,
`KINESCORE_CKPT_DIR`, `KINESCORE_ASSETS`, `KINESCORE_OUTPUT_DIR`), and a
missing variable raises at resolution time naming itself.

The CLI does not parse `.env` on its own. Export it once per shell first:

```bash
set -a; . .env; set +a
```

## 1. Get a checkpoint and score your own videos

This is the whole path from nothing to a table, if the reader you need is
already trained. Everything past this section is for building or extending
a reader yourself.

```bash
export KINESCORE_ASSETS=/shared/kinescore-store/assets
export KINESCORE_CKPT_DIR=/shared/kinescore-store/ckpt
export KINESCORE_OUTPUT_DIR=/shared/kinescore-store/out
export HF_TOKEN=<read token>

hf sync hf://buckets/twanghcmut/hallucinate-bench/asset $KINESCORE_ASSETS
hf buckets cp hf://buckets/twanghcmut/hallucinate-bench/train/fourier_gr1.humanoid_sv.sv1_16x9/diffusion/checkpoint.pt \
    $KINESCORE_CKPT_DIR/fourier_gr1.humanoid_sv.sv1_16x9.pt
```

```bash
kinescore score --cell humanoid.sv1_16x9.dreamgen --device cuda \
    --videos /path/to/your/clips \
    --out $KINESCORE_OUTPUT_DIR/myrun
```

`--videos` is searched recursively for `*.mp4`. The cell fixes the reader,
and the reader fixes the accepted frame size; a clip that does not match is
rejected by name before the backbone runs. `kinescore score --list` prints
every cell.

`--out` holds:

| file | one row per | holds |
|---|---|---|
| `summary.json` | run | cell, reader, checkpoint SHA-256, clip counts, the fitted thresholds |
| `segments.csv` | 16-frame segment | `<detector>_{reduce,value,threshold,violated}` |
| `metrics.csv` | clip | flagged fraction, worst segment, median severity per detector |
| `results.jsonl` | clip | the full per-frame series and flagged frame intervals |
| `run_manifest.json` | run | argv, git sha, host, config hashes |
| `render/` | clip | `<id>_clip.mp4`, keypoints and verdicts drawn on |

Before scoring anything, `score` fits its own thresholds on 24 real clips
from that reader's validation split, at the 95th percentile. There is no
separate calibration step, and no threshold is shared between robots.

Useful flags: `--limit 40` caps clips, `--pattern full_pred.mp4` picks one
basename out of a tree that stores prediction and ground truth side by side,
`--frame-chunk 16` lowers GPU memory on long clips.

```bash
kinescore report --by method    # one table over every cell scored so far
```

`report` only finds a cell's scores when they sit in that cell's own output
directory, `$KINESCORE_OUTPUT_DIR/<cell_id>`. A run written to a custom
`--out` shows as `not scored` there. Drop `--out`, or name the directory
after the cell.

## 2. Build a reader from a corpus

Needed only for a robot or packing with no checkpoint on the bucket yet.

### Download the corpus

`configs/sources.yaml` declares every download; each pull pins its revision in
`$KINESCORE_DATA_ROOT/REVISIONS.json`, and re-pulling fetches that same
revision until `--revision` names a new one.

```bash
kinescore pull --list                # declared sources and what is on disk
kinescore pull --what bench         # scored clips + manifest.json
kinescore pull --what train         # LeRobot v2 corpora
kinescore pull --what fastercache   # generated clips outside the manifest
```

```
$KINESCORE_DATA_ROOT/
    bench/      clips/<id>.mp4, manifest.json, dense|augment|worldcache/, fastercache/
    corpus/     {bimanual,humanoid}/{multiview,singleview}/
    trees/      <reader_id>/{videos,annotation}/{train,val}/
    REVISIONS.json
```

| corpus | robot | cameras | episodes |
|---|---|---|---|
| `bimanual_mv` | ALOHA (2×vx300s) | cam_high, cam_low, two wrists | 300 |
| `bimanual_sv` | ALOHA (2×vx300s) | cam_high | 300 |
| `humanoid_mv` | Airbot MMK2 | head, third-person, two wrists | 248 |
| `humanoid_sv` | Fourier GR-1 | ego view | 260 |

### Train

Three commands per reader, in order. Ids: `kinescore readers --ids`.

```bash
R=fourier_gr1.humanoid_sv.sv1_16x9
kinescore data  --reader $R                  # corpus → train tree (ffmpeg, CPU)
kinescore cache --reader $R --device cuda    # frozen backbone → token cache
kinescore train --reader $R --device cuda    # → $KINESCORE_CKPT_DIR/$R.pt
```

- `data` packs the declared cameras into the view's frame, splits
  scene-disjoint (`--val-ratio 0.1 --seed 0`), and writes
  `trees/<reader_id>/` whole. A re-run rewrites it.
- `cache` encodes once; already-cached episodes are skipped, so re-running
  finishes an interrupted cache. `--frame-chunk` bounds GPU memory.
- `train` fits the diffusion head (`--steps 6000`, `--batch-size 32`,
  `--window-size 16`) and reports RMS keypoint error in millimetres on the
  scene-disjoint val split. `val_mm` is the number to quote.

`train` refuses a cache shorter than the tree (`--allow-partial-cache`
overrides) and refuses a robot whose declared keypoint count disagrees with
its forward kinematics.

## 3. Score without `--videos`

```bash
kinescore score --list                                     # cells and status
kinescore score --cell humanoid.sv1_4x3.dreamdojo --device cuda
```

With no `--videos`, the cell scores exactly the clips `bench/manifest.json`
assigns to it. A directory outside the manifest:

```bash
kinescore score --cell humanoid.sv1_4x3.dreamdojo \
    --videos $KINESCORE_DATA_ROOT/bench/fastercache/humanoid/output/singleview \
    --pattern full_pred.mp4 \
    --out out/fastercache.sv1_4x3.dreamdojo
```

`--pattern` picks one basename out of a tree that stores prediction and ground
truth side by side. Clip ids mirror the tree, so clips sharing a filename keep
distinct ids.

Order of operations: resolve the cell → load the checkpoint under a
`ReaderExpectation` (robot, view, panel count; a mismatch fails before the
backbone is built) → **calibrate** every detector at `--percentile 95` on up
to `--calibration-clips 24` real clips from the reader's own `videos/val/` →
score each clip → render. Thresholds come from real motion the head was not
fitted on; scores are comparable within a cell, and across cells only when the
readers have comparable `val_mm`.

## 4. Outputs, in full

`out/<cell_id>/` per scored cell:

| file | holds |
|---|---|
| `results.jsonl` | one line per clip: coords, per-detector `violations` (threshold, per-frame series, flagged intervals), per-segment verdicts |
| `segments.csv` | one row per 16-frame segment: `<detector>_{reduce,value,threshold,violated}` |
| `metrics.csv` | one row per clip |
| `summary.json` | cell, reader, checkpoint + SHA-256, clip counts, percentile, fitted thresholds |
| `run_manifest.json` | argv, git sha, host, config hashes |
| `render/` | every clip redrawn with keypoints and verdicts, mirroring the source tree; `render/reel/<tree>.mp4` one reel per tree |

A clip that fails to read is written as `{"path", "error"}` and counted; the
run continues and exits non-zero. `kinescore render --cell <id>` redraws
without re-scoring.

```bash
kinescore report --by role      # fraction of clips flagged, per cell/partition
kinescore report --by method    # --by split is the third axis
```

## 5. Export for a rating UI

```bash
kinescore export --cell humanoid.sv1_4x3.dreamdojo --name dreamdojo_humanoid_sv
# or:  --results out/fastercache.sv1_4x3.dreamdojo
```

Writes `$KINESCORE_OUTPUT_DIR/web/<name>/`: clips renumbered `1.mp4 … N.mp4`
in scoring order, plus one `segments.json`:

```json
{"detectors": {"rigidity": {"threshold": 30.55, "units": "mm",
   "reduce": "median"},
  "jerk": {"threshold": 463094.2, "units": "mm/s^3", "reduce": "max"}},
 "provenance": {"cell": "...", "reader": "...", "checkpoint": "..."},
 "n_skipped": 0, "n_clips": 431,
 "videos": {"1": {"source": "makovian/.../full_pred.mp4",
   "fps": 10.0, "n_frames": 49, "n_violated": 1,
   "segments": [{"start_frame": 0, "end_frame": 15, "n_frames": 16,
     "rigidity": {"value": 19.6, "ratio": 0.643, "violated": false},
     "jerk": {"value": 47017.1, "ratio": 0.1015, "violated": false}}]}}}
```

`ratio = value / threshold`; `> 1` means violated for these detectors
(`self_collision` is the one min-distance detector where lower is worse).
`--detectors` selects others (default: rigidity + jerk).

## 6. Push to the bucket

```bash
export HF_TOKEN=<write token>   # env only, never a file
kinescore push --reader fourier_gr1.humanoid_sv.sv1_16x9 \
    --scores out/fastercache.sv1_4x3.dreamdojo \
    --web out/web/fastercache_humanoid_sv
```

Syncs are incremental (`hf sync`) into
`hf://buckets/twanghcmut/hallucinate-bench` (override with `--bucket`):
`train/<reader>/diffusion/`, `scores/<cell>/diffusion/`, `web/<bundle>/`.
The cell id is the `--scores` directory basename; verify a web push by
counting N mp4 + 1 `segments.json` on the bucket.
