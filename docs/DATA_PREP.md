# Data preparation

Two commands: `kinescore pull` fetches the published data, `kinescore data`
turns a corpus into the canonical training tree a reader is fitted on. `data` is
the only stage that touches source layouts; everything after it reads one tree
shape.

## Sources

`configs/sources.yaml` declares every download. It is the only file `pull`
reads.

| source | repo | lands in | holds |
|---|---|---|---|
| `bench` | `doanh25032004/video_gen_physics` (`data_for_web/`) | `$KINESCORE_DATA_ROOT/bench` | 400 scored clips + `manifest.json` |
| `train` | `doanh25032004/vgp_datasets` | `$KINESCORE_DATA_ROOT/corpus` | six LeRobot v2 corpora |

```bash
kinescore pull --list                      # declared sources and what is on disk
kinescore pull --what bench                # 2.2 GB
kinescore pull --what train --workers 8    # 15 GB
kinescore pull --what train --revision <sha>
```

Each pull records repo, revision, include patterns and file count in
`$KINESCORE_DATA_ROOT/REVISIONS.json`, and a later pull of the same source
defaults to that revision rather than to `main`. The `train` whitelist covers
`bimanual/`, `humanoid/` and `single_arm/` only: the repo's `*_dreamgen*`
directories are `{metas, t5_xxl, videos}` with no robot state, so nothing in
them can supervise a keypoint head.

## The corpora

```
$KINESCORE_DATA_ROOT/corpus/<embodiment>/<view>/[<task>/]meta/info.json
                                                          data/chunk-NNN/episode_NNNNNN.parquet
                                                          videos/chunk-NNN/observation.images.<cam>/episode_NNNNNN.mp4
```

| corpus | robot | state read | episodes |
|---|---|---|---|
| `bimanual_mv` | ALOHA | `observation.state` cols 0-5, 7-12; gripper col 6 | 300 |
| `bimanual_sv` | ALOHA | same, one camera | 300 |
| `humanoid_mv` | Airbot MMK2 | `observation.state` cols 0-11 (both arms) | 248 |
| `humanoid_sv` | Fourier GR-1 | cols 0-6, 22-28, 41-43 (arms + waist) | 260 |
| `single_arm_mv` | Franka Panda | `observation.state.joint_position`, gripper from `observation.state.gripper_position` | 260 |

Each corpus is declared once in `configs/cells.yaml` as a YAML anchor and shared
by every reader trained on it. A camera entry may list alternatives as `a|b`:
`humanoid_mv` merges two collection runs that name the head and exterior cameras
differently, and the first name present is taken, so both runs pack into the
same panel order.

## `kinescore data`

```bash
kinescore data --list                                      # readers and status
kinescore data --reader franka_panda.single_arm_mv.mv3_row
kinescore data --reader aloha_bimanual.bimanual_mv.mv4_row --val-ratio 0.1 --seed 0
```

| flag | default | meaning |
|---|---|---|
| `--reader` | — | reader id to materialise |
| `--list` | — | print every declared reader and its status |
| `--val-ratio` | `0.1` | target val fraction, split by scene |
| `--seed` | `0` | split seed |
| `--limit` | `0` | cap episodes read (0 = all) |
| `--copy` | off | copy videos instead of symlinking |

### What it writes

```
$KINESCORE_DATA_ROOT/trees/<reader_id>/
    videos/{train,val}/<episode_id>.mp4
    annotation/{train,val}/<episode_id>.json
    dataset_card.json
    run_manifest.json
```

A run rewrites the tree whole: the four split directories are emptied before
anything is written, so the tree describes that run alone. Re-running with a
different `--val-ratio`, `--seed` or `scene_key` moves episodes cleanly rather
than leaving one sitting in both `train` and `val`, and a `--limit` run leaves a
tree holding only the episodes it read.

Each annotation carries:

```json
{
  "joint_source": "real",
  "observation.state.joint_position": [[...], ...],
  "observation.state.gripper_position": [...],
  "fps": 30.0,
  "scene_key": "makovian__close_toolbox",
  "source_path": "..."
}
```

`joint_source` is load-bearing. The cache stage asserts it is `"real"` before
encoding: a head fitted against interpolated or synthesised joints would report
millimetre errors that mean nothing, so the pipeline refuses rather than
producing an unfalsifiable number. `observation.state.gripper_position` is
present only for corpora that log one.

`dataset_card.json` records the reader, the view geometry, the corpus, the
cameras and columns read, per-split counts, and every skipped episode with its
reason (up to 20 listed in full, the rest counted).

## The split is scene-disjoint

`training/splits.py::stratified_episode_split` groups episodes by scene key and
assigns whole scenes to one side. A task never appears in both `train` and
`val`, so a validation millimetre number measures generalisation to unseen
scenes rather than recall of seen ones. The default `scene_key: prefix` takes
the episode id up to its last `__`, which for these corpora is
`<split>__<task>`.

Whole scenes move together, so the achieved val fraction is only as close to
`--val-ratio` as the scene sizes allow: the splitter adds whole scenes
smallest-first and stops before overshooting, but it always places at least one
scene in `val`. A corpus of a few large scenes therefore lands well past the
target.

A corpus whose episode ids carry no scene structure declares `scene_key:
episode`, which makes every episode its own one-episode scene. The split is then
a plain stratified sample and claims no scene disjointness — the corpus supplies
nothing to base one on. `humanoid_sv` and `single_arm_mv` use it: both hold two
partitions and repeat the partition as the task, so the id prefix is not a
scene.

## The adapter

An adapter reads one corpus layout and yields `RawEpisode`
(`adapters/base.py`): per-camera video paths, a joint array, an optional gripper
array, and a scene key. Episodes it cannot use are yielded as `SkippedEpisode`
with a reason instead of raising, so one broken episode does not abort a corpus.

`lerobot` is the only adapter. It walks every directory holding `meta/info.json`
— corpora both flat and grouped by task — validates the declared cameras against
`info.json`'s `observation.images.*` features, and slices `joint_columns` out of
`joint_field`. `get_adapter(name)` raises `ValueError` listing the available
adapters when the name is unknown.

## Packing

Cameras are stored one file each; `data` composes them into one frame with
ffmpeg, in the order `cameras:` declares. Panel order is never alphabetical and
never inferred.

| view_id | panels | layout | panel | frame |
|---|---|---|---|---|
| `sv1_4x3` | 1 | single | 640×480 | 640×480 |
| `sv1_16x9` | 1 | single | 768×432 | 768×432 |
| `mv3_row` | 3 | horizontal row | 320×192 | 960×192 |
| `mv4_row` | 4 | horizontal row | 320×192 | 1280×192 |
| `mv4_grid` | 4 | 2×2 | 384×216 | 768×432 |
| `mv4_grid_br_blank` | 3 of 4 | 2×2, bottom-right blank | 384×216 | 768×432 |

Every panel is scaled to the declared size, including the single-panel views:
the corpus camera and the clips a head scores differ in resolution, and the head
must see one geometry. An episode whose packed frame does not match the view is
skipped and reported in the dataset card rather than silently letter-boxed.

## Environment

`.env` sets the roots every stage resolves through:

| variable | holds |
|---|---|
| `KINESCORE_DATA_ROOT` | `bench/`, `corpus/`, `trees/`, `REVISIONS.json` |
| `KINESCORE_CACHE_DIR` | encoded token caches |
| `KINESCORE_CKPT_DIR` | reader checkpoints |
| `KINESCORE_ASSETS` | URDFs and robot assets |
| `KINESCORE_OUTPUT_DIR` | score reports |

A missing variable raises `MissingPathError` at resolution time.

## Next

[TRAINING.md](TRAINING.md) — encode the tree and fit the head.
