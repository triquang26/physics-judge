# Humanoid multiview on the static cameras, end to end

Every command from empty disk to bucket push, with the reference numbers one
full run produced (2026-08-29). General flag documentation lives in
[QUICKSTART.md](QUICKSTART.md); this page is the run book for the
`mv4_grid_static` bench.

## The view

The multiview corpus and the DreamGen/fastercache clips pack four cameras into
one 768x432 frame as a 2x2 grid of 384x216 panels: top row head +
third-person, bottom row the two wrist cameras. A wrist camera rides the arm —
its frame shows the fingers, not where the arm is — so this bench reads the
two static panels only:

```yaml
mv4_grid_static:            # configs/views.yaml
  n_views: 2
  packing: grid2x2
  n_panels: 4
  panels: [0, 1]            # top-left = head, top-right = third-person
  panel: [384, 216]
```

The train tree is packed with the same geometry (wrist cells black, never
read), so the tree and the bench clips are one layout. Reader:
`airbot_mmk2.humanoid_mv.mv4_grid_static` (robot `airbot_mmk2`, corpus
`humanoid_mv` — same corpus blocks as the other multiview readers in
`configs/cells.yaml`).

## 1. Download

```bash
kinescore pull --what train        # vgp_datasets: humanoid/multiview, 248 episodes
kinescore pull --what fastercache  # video_gen_physics: fastercache/humanoid/output/{singleview,multiview}
```

The multiview bench lands at
`$KINESCORE_DATA_ROOT/bench/fastercache/humanoid/output/multiview/dreamgen/`:
248 episodes under `{makovian,non_makovian}/iter_000100000/<task>/episode_*/`,
each holding `full_gt.mp4`, `full_pred.mp4`, `full_merged.mp4`, `seg_*.mp4`
and `metrics.json`. Only `full_pred.mp4` is scored.

## 2. Train the reader

```bash
R=airbot_mmk2.humanoid_mv.mv4_grid_static
kinescore data  --reader $R              # 248 eps -> 227 train / 21 val, scene-disjoint
kinescore cache --reader $R --device cuda
kinescore train --reader $R --device cuda
mv $KINESCORE_CKPT_DIR/$R.pt $KINESCORE_CKPT_DIR/$R.diff.pt
mv $KINESCORE_CKPT_DIR/$R.train_log.jsonl $KINESCORE_CKPT_DIR/$R.diff.train_log.jsonl
```

Reference: `val_mm 104.0` at 6000 steps (best 103.9 at 5000, train 17.5).
Same order as bimanual (~100) and above the singleview humanoid readers —
a 384x216 panel gives each keypoint few pixels; compare within the view.

## 3. Score

```bash
kinescore score --cell humanoid.mv4_grid_static.dreamgen \
    --videos $KINESCORE_DATA_ROOT/bench/fastercache/humanoid/output/multiview/dreamgen \
    --pattern full_pred.mp4 \
    --checkpoint $KINESCORE_CKPT_DIR/$R.diff.pt \
    --out out/fastercache.mv4_grid_static.dreamgen
```

Reference thresholds (p95 on the 21 val clips): rigidity 55.4 mm, jerk 3.0e6
mm/s^3, teleport 1178.8 mm/s, joint_limit 3.83 deg, self_collision 186.9 mm.

Reference verdicts, 248 clips / 2700 segments, beside the reader's own
real-motion baseline (see METRICS.md "Segment-level baseline"):

| detector | dreamgen multiview | real baseline | reading |
|---|---|---|---|
| rigidity | 4.4% | 1.4% | +3.0 excess — mild warping |
| jerk | 0.0% | 33.4% | far below real — motion smoother than the robot |
| teleport | 2.6% | 39.2% | far below real — same |
| joint_limit | 89.1% | 51.4% | +37.7 excess — the strongest signal |
| self_collision | 10.0% | 16.6% | below real |

The generated motion is slower and smoother than the real corpus, so the
dynamics detectors under-fire by construction; the excess lives in
joint-limit violations and residual warping.

## 4. Export and push

```bash
kinescore export --results out/fastercache.mv4_grid_static.dreamgen \
    --name fastercache_humanoid_mv --detectors all
export HF_TOKEN=<write token>   # env only — never in a file
kinescore push --reader $R \
    --scores out/fastercache.mv4_grid_static.dreamgen \
    --web out/web/fastercache_humanoid_mv
```

Bucket targets: `train/<reader>/diffusion/`, 
`scores/fastercache.mv4_grid_static.dreamgen/diffusion/`,
`web/fastercache_humanoid_mv/` (248 mp4 + `segments.json`).

## Adapting to another robot or view

Three files, in order:

1. `configs/views.yaml` — measure one panel from the clips (seams stand an
   order of magnitude above the median column/row difference), declare
   `packing`, `n_panels`, `panel`; a static subset is `panels: [...]` by
   index into the packed frame.
2. `configs/cells.yaml` — a corpus block (`root`, `adapter`, `cameras` in
   panel order, `joint_field`, `joint_columns`), a reader
   (`<robot>.<corpus>.<view>`), and a cell naming that reader.
3. `src/kinescore/robots/<robot>/` + `configs/robots.yaml` — URDF under
   `$KINESCORE_ASSETS` (mirror: bucket `assets/`), an FK spec naming the
   keypoint links, registered in `robots/__init__.py`. `docs/BIMANUAL.md`
   documents the latest such port end to end.

Then the same five commands: `data → cache → train → score → export → push`.
