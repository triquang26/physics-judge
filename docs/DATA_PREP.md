# Data preparation

`kinescore data` turns a corpus into the canonical training tree a reader is
fitted on. It is the only stage that touches source layouts; everything after it
reads one tree shape.

```bash
kinescore data --list                              # readers and their status
kinescore data --reader franka_panda.mv3_row       # materialise one reader
kinescore data --reader aloha_bimanual.mv3_row --val-ratio 0.1 --seed 0
```

| flag | default | meaning |
|---|---|---|
| `--reader` | — | reader id to materialise |
| `--list` | — | print every declared reader and its status |
| `--val-ratio` | `0.1` | target val fraction, split by scene |
| `--seed` | `0` | split seed |
| `--limit` | `0` | cap episodes read (0 = all) |
| `--copy` | off | copy videos instead of symlinking |

## What it writes

```
$KINESCORE_DATA_ROOT/train/<reader_id>/
    videos/train/<episode_id>.mp4
    videos/val/<episode_id>.mp4
    annotation/train/<episode_id>.json
    annotation/val/<episode_id>.json
    dataset_card.json
    run_manifest.json
```

A run rewrites the tree whole: the four split directories are emptied before
anything is written, so the tree describes that run alone. Re-running with a
different `--val-ratio`, `--seed` or `scene_key` therefore moves episodes
cleanly rather than leaving one sitting in both `train` and `val`, and a
`--limit` run leaves a tree holding only the episodes it read.

Each annotation carries:

```json
{
  "joint_source": "real",
  "observation.state.joint_position": [[...], ...],
  "observation.state.gripper_position": [[...], ...],
  "fps": 30.0
}
```

`joint_source` is load-bearing. The cache stage asserts it is `"real"` before
encoding: a head fitted against interpolated or synthesised joints would report
millimetre errors that mean nothing, so the pipeline refuses rather than
producing an unfalsifiable number. `observation.state.gripper_position` is
present only for robots whose spec declares a gripper.

`dataset_card.json` records the reader, the view geometry, per-split counts, and
every skipped episode with its reason (up to 20 listed in full, the rest
counted).

## The split is scene-disjoint

`training/splits.py::stratified_episode_split` groups episodes by scene key and
assigns whole scenes to one side. A task never appears in both `train` and
`val`, so a validation millimetre number measures generalisation to unseen
scenes rather than recall of seen ones. Adapters supply the scene key; for
Ctrl-World-style corpora it is the task directory name.

Whole scenes move together, so the achieved val fraction is only as close to
`--val-ratio` as the scene sizes allow: the splitter adds whole scenes
smallest-first and stops before overshooting, but it always places at least
one scene in `val`. A corpus of a few large scenes therefore lands well past
the target -- two scenes of 130 episodes split 130/130 whatever ratio is asked
for.

A source whose episode ids carry no scene structure declares `scene_key:
episode` in its `train:` block, which makes every episode its own one-episode
scene. The split is then a plain stratified sample over episodes and hits the
requested ratio, and it claims no scene disjointness -- the corpus supplies
nothing to base one on. `fourier_gr1.sv1` uses it: that corpus names episodes
`<partition>__<index>` over two partitions and its provenance repeats the
partition as the task, so the id prefix is not a scene. The default,
`scene_key: prefix`, reads the id up to its last `__`.

## Adapters

An adapter reads one corpus layout and yields `RawEpisode`
(`adapters/base.py`): per-camera video paths, an optional pre-packed video, a
joint array, an optional gripper array, and a scene key. Episodes it cannot use
are yielded as `SkippedEpisode` with a reason instead of raising, so one broken
episode does not abort a corpus.

| adapter | reads |
|---|---|
| `ctrlworld` | metadata JSON + per-view or pre-packed `full_gt.mp4` trees |
| `canonical` | an already-materialised tree (re-split or re-pack it) |

`get_adapter(name)` raises `ValueError` listing the available adapters when the
name is unknown. The `canonical` adapter refuses any episode whose
`joint_source` is not `"real"`.

## Packing

Panels are composed into one frame by ffmpeg according to the reader's
`ViewSpec` (`configs/views.yaml`). Five packings are measured:

| view_id | panels | layout | panel size |
|---|---|---|---|
| `sv1` | 1 | single | source |
| `mv3_row` | 3 | horizontal row | 320×192 |
| `mv4_row` | 4 | horizontal row | 320×192 |
| `mv4_grid` | 4 | 2×2 | 384×216 |
| `mv4_grid_br_blank` | 3 of 4 | 2×2, bottom-right blank | 384×216 |

An episode whose source geometry does not match the view is skipped and reported
in the dataset card rather than silently letter-boxed. When the corpus already
ships a correctly packed video, it is passed through unchanged — no re-encode,
no generation loss.

`ViewSpec.panel_count` / `panel_indices` are what the packer reads, so a view
that exposes fewer panels than it has grid cells (`mv4_grid_br_blank`) packs
correctly.

## Environment

`.env` sets the roots every stage resolves through:

| variable | holds |
|---|---|
| `KINESCORE_DATA_ROOT` | canonical trees and score inputs |
| `KINESCORE_CACHE_DIR` | encoded token caches |
| `KINESCORE_CKPT_DIR` | reader checkpoints |
| `KINESCORE_ASSETS` | URDFs and robot assets |
| `KINESCORE_OUTPUT_DIR` | score reports |

A missing variable raises `MissingPathError` at resolution time.

## Next

[TRAINING.md](TRAINING.md) — encode the tree and fit the head.
