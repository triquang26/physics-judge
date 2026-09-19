# Runbook

Commands to reproduce the full benchmark from an empty machine. README covers
install and `.env` setup; start here after `set -a; . .env; set +a`.

## 1. Pull

```bash
export HF_TOKEN=<read token>
hf sync hf://buckets/twanghcmut/hallucinate-bench/asset $KINESCORE_ASSETS

kinescore pull --what train
for W in radial itm sito dicache pisa svg1 svg2 worldcache fastercache dense cosmos; do
  kinescore pull --what $W
done
```

The A1X corpus needs a filtered symlink (the `non_makovian` split is Franka,
not A1X):

```bash
mkdir -p $KINESCORE_DATA_ROOT/corpus/single_arm/singleview_a1x
ln -sfn ../singleview/makovian \
   $KINESCORE_DATA_ROOT/corpus/single_arm/singleview_a1x/makovian
```

## 2. Train

```bash
for R in fourier_gr1.humanoid_sv.sv1_16x9 \
         aloha_bimanual.bimanual_sv.sv1_16x9 \
         a1x_ee.a1x_sv.sv1_4x3; do
  kinescore data  --reader $R
  kinescore cache --reader $R --device cuda
  kinescore train --reader $R --device cuda --steps 8000
done
```

Reference val error (RMS, scene-disjoint split):

| reader | K | val_mm |
|---|---|---|
| `fourier_gr1.humanoid_sv.sv1_16x9` | 22 | 51.2 |
| `aloha_bimanual.bimanual_sv.sv1_16x9` | 18 | 83.5 |
| `a1x_ee.a1x_sv.sv1_4x3` | 4 | 24.4 |

Compare val_mm only within a reader.

## 3. Score

Calibration runs automatically: 24 real val clips, 95th percentile, one
threshold set per cell. One GPU per cell (calibration drifts ~1% across GPUs).

### Manifest cells

```bash
kinescore score --cell humanoid.sv1_16x9.dreamgen --device cuda
kinescore score --cell bimanual.sv1_16x9.dreamgen --device cuda
```

### Clips outside the manifest (via --videos)

```bash
M=$KINESCORE_DATA_ROOT/bench/methods

kinescore score --cell humanoid.sv1_16x9.dreamgen --device cuda \
    --videos $M/itm/humanoid/output/singleview/dreamgen/makovian \
    --out $KINESCORE_OUTPUT_DIR/itm.humanoid.gr1

kinescore score --cell bimanual.sv1_16x9.radial_dreamgen --device cuda \
    --videos $KINESCORE_DATA_ROOT/bench/radial/bimanual/output/singleview \
    --out $KINESCORE_OUTPUT_DIR/radial.bimanual.sv

kinescore score --cell single_arm.sv1_4x3.radial_dreamgen --device cuda \
    --videos $KINESCORE_DATA_ROOT/bench/radial/single_arm/output/singleview/dreamgen \
    --out $KINESCORE_OUTPUT_DIR/radial.single_arm.sv
```

### Real-teleop control

Same reader, scored on its own train clips:

```bash
kinescore score --cell humanoid.sv1_16x9.dreamgen --device cuda --limit 30 \
    --videos $KINESCORE_DATA_ROOT/trees/fourier_gr1.humanoid_sv.sv1_16x9/videos/train \
    --out $KINESCORE_OUTPUT_DIR/real.humanoid.gr1
```

`kinescore score --list` prints every cell.

## 4. Report and export

```bash
kinescore report --by method
kinescore report --by role

kinescore export --cell humanoid.sv1_16x9.dreamgen --name dreamgen_humanoid_sv
kinescore export --results $KINESCORE_OUTPUT_DIR/radial.bimanual.sv --name radial_bimanual_sv
kinescore export --results $KINESCORE_OUTPUT_DIR/radial.single_arm.sv --name radial_single_arm_sv
```

Each export writes `$KINESCORE_OUTPUT_DIR/web/<name>/`: clips renumbered
`1.mp4` to `N.mp4`, plus one `segments.json` with per-segment values,
thresholds, and verdicts.

## 5. Push

```bash
export HF_TOKEN=<write token>   # env only, rotate after use
kinescore push --reader fourier_gr1.humanoid_sv.sv1_16x9 \
    --reader aloha_bimanual.bimanual_sv.sv1_16x9 \
    --reader a1x_ee.a1x_sv.sv1_4x3 \
    --scores $KINESCORE_OUTPUT_DIR/itm.humanoid.gr1 \
    --scores $KINESCORE_OUTPUT_DIR/radial.bimanual.sv \
    --scores $KINESCORE_OUTPUT_DIR/radial.single_arm.sv \
    --web out/web/dreamgen_humanoid_sv \
    --web out/web/radial_bimanual_sv \
    --web out/web/radial_single_arm_sv
```

Targets under `hf://buckets/twanghcmut/hallucinate-bench`:
`train/<reader>/diffusion/`, `scores/<cell>/diffusion/`, `web/<bundle>/`.

## Output files

| file | one row per | holds |
|---|---|---|
| `summary.json` | run | cell, reader, checkpoint SHA-256, thresholds |
| `segments.csv` | 16-frame segment | `<detector>_{value,threshold,violated}` |
| `metrics.csv` | clip | flagged fraction, severity per detector |
| `results.jsonl` | clip | full per-frame series, flagged intervals |
| `run_manifest.json` | run | argv, git sha, host, config hashes |
| `render/` | clip | keypoints and verdicts drawn on |

## Against human ratings

```bash
python scripts/score_agreement.py --ratings ratings.csv --key key.json --by-embodiment
```
