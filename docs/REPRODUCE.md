# Reproduce

Empty machine to benchmark table. Every command reads `.env`.

## 0. Environment

```bash
KINESCORE_CKPT_DIR=<shared>/kinescore-store/ckpt
KINESCORE_CACHE_DIR=<shared>/kinescore-store/cache
KINESCORE_DATA_ROOT=<shared>/kinescore-store/data
KINESCORE_OUTPUT_DIR=<shared>/kinescore-store/out
KINESCORE_ASSETS=<fast local>/assets
HF_HOME=<fast local>/hf
export HF_TOKEN=<read token>
```

```bash
pip install -e .
hf sync hf://buckets/twanghcmut/hallucinate-bench/asset $KINESCORE_ASSETS
```

## 1. Pull

```bash
kinescore pull --list
kinescore pull --what train
for W in radial itm sito dicache pisa svg1 svg2 worldcache fastercache dense cosmos; do
  kinescore pull --what $W
done
```

## 2. Train

```bash
mkdir -p $KINESCORE_DATA_ROOT/corpus/single_arm/singleview_a1x
ln -sfn ../singleview/makovian \
   $KINESCORE_DATA_ROOT/corpus/single_arm/singleview_a1x/makovian

for R in fourier_gr1.humanoid_sv.sv1_16x9 \
         aloha_bimanual.bimanual_sv.sv1_16x9 \
         a1x_ee.a1x_sv.sv1_4x3; do
  kinescore data  --reader $R
  kinescore cache --reader $R --device cuda
  kinescore train --reader $R --device cuda --steps 8000
done
```

`--loss mse` swaps smooth-L1 for MSE. `kinescore readers` lists the ids.

Reference val error, RMS over the scene-disjoint split:

| reader | K | val_mm |
|---|---|---|
| `fourier_gr1.humanoid_sv.sv1_16x9` | 22 | 51.2 |
| `aloha_bimanual.bimanual_sv.sv1_16x9` | 18 | 83.5 |
| `a1x_ee.a1x_sv.sv1_4x3` | 4 | 24.4 |

Compare val_mm only within a reader.

## 3. Score

`score` calibrates before it scores: 24 clips from the reader's own val split,
95th percentile, one threshold set per cell. No separate step, and no threshold
is carried between robots. One GPU per cell — a `--videos` tree split across two
is judged by two calibrations.

```bash
M=$KINESCORE_DATA_ROOT/bench/methods
B=dreamgen/makovian                    # or dreamgen/non_makovian

kinescore score --cell humanoid.sv1_16x9.dreamgen --device cuda \
    --videos $M/itm/humanoid/output/singleview/$B \
    --out $KINESCORE_OUTPUT_DIR/itm.humanoid.gr1
```

The real-teleop control, same reader, from its own train tree:

```bash
kinescore score --cell humanoid.sv1_16x9.dreamgen --device cuda --limit 30 \
    --videos $KINESCORE_DATA_ROOT/trees/fourier_gr1.humanoid_sv.sv1_16x9/videos/train \
    --out $KINESCORE_OUTPUT_DIR/real.humanoid.gr1
```

Cells: `kinescore score --list`.

## 4. Read out

```bash
kinescore report --by role
kinescore render --cell humanoid.sv1_16x9.dreamgen --flagged-only
kinescore push --scores $KINESCORE_OUTPUT_DIR/itm.humanoid.gr1
```

Against human ratings:

```bash
python scripts/score_agreement.py --ratings ratings.csv --key key.json \
       --by-embodiment
```
