# Reproduce

Empty machine to rating batch. Every command reads `.env`; `docs/QUICKSTART.md`
documents the flags, this page is the exact sequence that produced the current
readers and the 900-label batch.

## 0. Storage

Corpora, token caches and score outputs go on the shared filesystem. A
full-length cache runs to 80–110 GB per reader, which no RAM disk holds, and
capping frames to fit one costs real accuracy: GR-1 went from 63.0 to 51.2 mm
val error on the single change of removing that cap.

```bash
KINESCORE_CKPT_DIR=<shared>/kinescore-store/ckpt
KINESCORE_CACHE_DIR=<shared>/kinescore-store/cache
KINESCORE_DATA_ROOT=<shared>/kinescore-store/data
KINESCORE_OUTPUT_DIR=<shared>/kinescore-store/out
KINESCORE_ASSETS=<fast local>/assets      # URDFs, ~400 MB
HF_HOME=<fast local>/hf                   # backbone weights, ~1.6 GB
```

Assets are never vendored. Fetch them once:

```bash
hf download --repo-type dataset ...        # or:
hf sync hf://buckets/twanghcmut/hallucinate-bench/asset $KINESCORE_ASSETS
```

## 1. Download

`configs/sources.yaml` is the only thing `pull` reads, and each pull pins its
revision in `REVISIONS.json` so a second machine gets the same bytes.

```bash
export HF_TOKEN=<read token>
kinescore pull --list                 # every source and what is on disk
kinescore pull --what train           # LeRobot corpora: the real teleop
kinescore pull --what radial          # one generator
kinescore pull --what itm             # ...and the rest, one flag each:
kinescore pull --what sito            # sito, dicache, pisa, svg1, svg2,
kinescore pull --what dicache         # worldcache, fastercache, dense, cosmos
```

Each generator ships the same `dreamgen/makovian` branch for all three
embodiments, so one pull serves every cell.

## 2. Train the three readers

```bash
for R in fourier_gr1.humanoid_sv.sv1_16x9 \
         aloha_bimanual.bimanual_sv.sv1_16x9 \
         a1x_ee.a1x_sv.sv1_4x3; do
  kinescore data  --reader $R                    # corpus → packed train tree
  kinescore cache --reader $R --device cuda      # no --max-frames: see §0
  kinescore train --reader $R --device cuda --steps 8000
done
```

A1X needs its corpus filtered first — only the makovian splits are that robot:

```bash
mkdir -p $KINESCORE_DATA_ROOT/corpus/single_arm/singleview_a1x
ln -sfn ../singleview/makovian \
   $KINESCORE_DATA_ROOT/corpus/single_arm/singleview_a1x/makovian
```

Reference val error, RMS over the scene-disjoint split:

| reader | K | val_mm |
|---|---|---|
| `fourier_gr1.humanoid_sv.sv1_16x9` | 22 | 51.2 (arm keypoints alone: 29.5) |
| `aloha_bimanual.bimanual_sv.sv1_16x9` | 18 | 83.5 |
| `a1x_ee.a1x_sv.sv1_4x3` | 4 | 24.4 |

Compare val_mm only within a reader: K differs, and a fingertip is a harder
target than a shoulder.

## 3. Score

Generated clips sit outside the bench manifest, so scoring points `--videos` at
the pulled tree. Thresholds are the 95th percentile of the same quantity
measured on 24 real clips from that reader's own val split — one calibration per
cell, never shared across robots.

```bash
M=$KINESCORE_DATA_ROOT/bench/methods
kinescore score --cell humanoid.sv1_16x9.dreamgen --device cuda \
    --videos $M/itm/humanoid/output/singleview/dreamgen/makovian \
    --out $KINESCORE_OUTPUT_DIR/itm.humanoid.gr1
```

Score the real teleop the same way, from the reader's own train tree — it is the
control the whole comparison rests on:

```bash
kinescore score --cell humanoid.sv1_16x9.dreamgen --device cuda --limit 30 \
    --videos $KINESCORE_DATA_ROOT/trees/fourier_gr1.humanoid_sv.sv1_16x9/videos/train \
    --out $KINESCORE_OUTPUT_DIR/real.humanoid.gr1
```

**One GPU per cell.** Calibration is not bitwise reproducible across GPUs; a
`--videos` tree split across two of them is judged by two thresholds.

## 4. What the scores say

Fraction of segments past threshold, real teleop included as the reference row.

| humanoid, peak | | bimanual, rigidity | | a1x, peak | |
|---|---|---|---|---|---|
| dicache | 2.2% | **real** | **0.0%** | **real** | **3.4%** |
| pisa | 3.0% | fastercache | 1.0% | dicache | 33% |
| fastercache | 6.2% | svg1 | 3.8% | pisa | 40% |
| **real** | **10.0%** | itm | 4.1% | itm | 59% |
| sito | 29.4% | dicache | 7.0% | fastercache | 61% |
| itm | 52.6% | sito | 7.9% | sito | 73% |

Two readings the numbers force:

- On humanoid, seven of nine generators score *cleaner* than real teleop.
  Generated video smooths fast motion, and jerk rewards smoothness, so only a
  badly broken generator clears the real-footage floor.
- On bimanual the ordering only holds for rigidity. Jerk there ranks real teleop
  worse than every generator (16.1% against 0.1%), which is why the rating batch
  samples each detector's extremes separately rather than their maximum.

## 5. Build the rating batch

`scripts/build_final.py` draws 100 clips per embodiment: 34 clean, 17 middle, 34
high, and 15 real-teleop controls held out of the strata. Each band takes half
its clips from the rigidity extreme and half from the jerk extreme, so both
detectors reach the raters with a range worth measuring.

```bash
python scripts/build_final.py          # → $KINESCORE_OUTPUT_DIR/web/rate900
kinescore push --web $KINESCORE_OUTPUT_DIR/web/rate900
```

Each clip is three 16-frame segments; raters judge the middle one, so 300 clips
carry 900 ratings. `key.json` holds the answers and must not reach a rater.

## 6. Read the ratings

```bash
python scripts/score_agreement.py --ratings ratings.csv \
       --key $KINESCORE_OUTPUT_DIR/.rate900_key.json --by-embodiment
```

Reports AUC of each detector's ratio against the human labels, per embodiment.
AUC reads the ranking only, so no threshold is fitted and none can be tuned
after the fact. Read it per embodiment: pooling across them mixes calibrations
and the pooled number means nothing.
