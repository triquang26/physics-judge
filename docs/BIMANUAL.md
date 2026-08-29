# ALOHA bimanual, end to end

Every command from empty disk to bucket push, with the reference numbers one
full run produced (2026-08-25). General flag documentation lives in
[QUICKSTART.md](QUICKSTART.md); this page is the bimanual-specific run book.

## The embodiment

`aloha_bimanual` — an ALOHA rig: two Interbotix ViperX 300s arms bolted to a
table. 18 keypoints (9 per arm, shoulder → fingers → TCP), 12 predicted
joints (6 revolute per arm); each gripper travels through the FK `aux`
channel, not `q` (see `src/kinescore/robots/aloha/fk.py`).

`observation.state` in both corpora is 14-dim:
`[left_arm(6), left_gripper, right_arm(6), right_gripper]` — hence
`joint_columns: [0,1,2,3,4,5,7,8,9,10,11,12]`, `gripper_column: 6` in
`configs/cells.yaml`.

**Asset**: `$KINESCORE_ASSETS/aloha/urdf/aloha_bimanual.urdf` (29 links,
28 joints, kinematics-only, ~30 KB). Assets are never vendored in this repo;
mirror the assets tree, whose `MANIFEST.json` ("aloha" key) records full
provenance: the two vx300s xacro expansions
(Interbotix `interbotix_ros_manipulators` @ `0bb2b0e`), the left/right mount
transforms (Mujoco Menagerie `aloha.xml` @ `feadf76`), and the merge. The
pre-merge inputs are kept beside it under `aloha/urdf/vx300s_src/` so the
merge is reproducible.

## 1. Download

```bash
kinescore pull --what train     # vgp_datasets: bimanual/{multiview,singleview}, 300 episodes each
kinescore pull --what bench     # video_gen_physics: data_for_web clips + manifest.json
kinescore pull --what radial    # radial/bimanual/... generated clips (outside the manifest)
```

## 2. Train the readers

Four readers, one per (corpus, view). Each is `data → cache → train`; only
`cache`/`train` need the GPU.

```bash
for R in aloha_bimanual.bimanual_mv.mv4_row \
         aloha_bimanual.bimanual_mv.mv4_grid \
         aloha_bimanual.bimanual_sv.sv1_16x9 \
         aloha_bimanual.bimanual_sv.sv1_4x3; do
  kinescore data  --reader $R                        # 300 eps → 271 train / 29 val, scene-disjoint
  kinescore cache --reader $R --device cuda
  kinescore train --reader $R --device cuda          # → $KINESCORE_CKPT_DIR/$R.diff.pt
done
```

Reference: `aloha_bimanual.bimanual_sv.sv1_16x9` reached `val_mm ≈ 100` at
6000 steps. Bimanual val_mm runs higher than humanoid — one 768×432 frame
carries two full arms, so each keypoint gets fewer pixels; compare within the
embodiment, not across.

## 3. Score

Manifest cell (DreamGen singleview 16:9):

```bash
kinescore score --cell bimanual.sv1_16x9.dreamgen --device cuda
```

Radial clips are outside the manifest — same reader, `--videos`:

```bash
kinescore score --cell bimanual.sv1_16x9.radial_dreamgen --device cuda \
    --videos $KINESCORE_DATA_ROOT/bench/radial/bimanual/output/singleview \
    --out out/bimanual.sv1_16x9.radial_dreamgen
```

Reference thresholds (p95 on 24 val clips of the sv1_16x9 reader's own train
tree): rigidity 91.3 mm, jerk 1.89e6 mm/s³, teleport 779.6 mm/s, joint_limit
6.8°, self_collision 27.3 mm. Reference verdicts, 16-frame segments:
DreamGen non-makovian iter_000110000 leaf 1/384 rigidity + 2/384 jerk;
radial (150 clips, 8637 segments) 220 rigidity (2.5%) + 101 jerk (1.2%).

**One GPU per cell.** Calibration is not bitwise reproducible across GPUs
(~1% threshold drift on identical clips). If you split a big `--videos` tree
across GPUs for speed, the halves are judged by *different* thresholds:
re-judge every segment against one canonical `summary.json` before merging,
and say so in the merged `run_manifest.json`.

## 4. Export web bundles

Bundle naming: `<method>_<embodiment>_<view>`.

```bash
kinescore export --cell bimanual.sv1_16x9.dreamgen        --name augment_bimanual_sv
kinescore export --results out/bimanual.sv1_16x9.radial_dreamgen --name radial_bimanual_sv
```

Each writes `$KINESCORE_OUTPUT_DIR/web/<name>/`: `1.mp4 … N.mp4` plus one
`segments.json` (per-segment rigidity median + jerk max, `ratio =
value/threshold`, `violated`).

## 5. Push to the bucket

```bash
export HF_TOKEN=<write token>       # env only — never in a file, rotate after use
kinescore push --reader aloha_bimanual.bimanual_sv.sv1_16x9 \
    --scores out/bimanual.sv1_16x9.dreamgen \
    --scores out/bimanual.sv1_16x9.radial_dreamgen \
    --web out/web/augment_bimanual_sv \
    --web out/web/radial_bimanual_sv
```

The `--scores` directory basename is the cell id on the bucket; the `--web`
basename is the bundle name.

Targets under `hf://buckets/twanghcmut/hallucinate-bench`:
`train/<reader>/diffusion/`, `scores/<cell>/diffusion/`, `web/<bundle>/`.
Verify a web push by counting N mp4 + 1 `segments.json` on the bucket; a
scores push should show the five-file set (`summary.json`, `results.jsonl`,
`metrics.csv`, `segments.csv`, `run_manifest.json`) plus `render/` when
overlays were generated.
