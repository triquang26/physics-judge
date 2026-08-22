# Target repo: the keypoint physics judge

Status: proposed, 2026-08-22. Nothing below is implemented.

One shipped path. A video goes in, per-frame physics-violation scores come out.
Everything that does not serve that path is deleted rather than deprecated.

## The pipeline

    HF dataset  ->  adapter  ->  canonical  ->  cache  ->  train  ->  score
                    per source   one format    DINO      keypoint   violations
                                               tokens    reader     on video

Five stages, one of which (`adapter`) is the only place a new corpus adds code.
Stages 3-6 never learn what dataset they came from.

## The model that ships

`DirectKeypointPoseReader`: DINO backbone -> `ViewEmbedding` -> `ReadoutV2Head`
emitting `n_out = 3K`, reshaped to `(B, T, K, 3)` — K points in the robot base
frame, metres, with a per-coordinate sigma. No forward kinematics, no joint
clamp, no joint limits in the forward path.

Supervision comes from FK applied to logged joint angles at data-prep time
(`P_true = robot.forward_kinematics(q_logged)`), so a robot still needs FK to
produce targets — what the reader drops is FK at *inference*. Loss is beta-NLL
over the 3K coordinates. There is no joint-limit penalty term.

Scoring is the five detectors in `kinescore.violations`: `rigidity`, `jerk`,
`teleport`, `joint_limit`, `self_collision`. Four run off predicted keypoints
alone; `rigidity` additionally reads `robot.rigid_bone_pairs` and
`rigid_bone_lengths`. Thresholds are calibrated per (robot, view) on that
reader's own real-teleop val split, never on the clips being judged.

## Data source of truth

`doanh25032004/video_gen_physics`, laid out as

    {method}/{embodiment}/output/{view}/{model}/{split}/{task_or_episode}/*.mp4

with a `catalog.json` indexing every clip:

```json
{"id": "00000",
 "video": "videos/augment/bimanual/output/multiview/ctrlworld_4view_grid/makovian/episode_insert_washer_shaft_both_hands__000055__v04_purple/pred_all_views.mp4",
 "embodiment": "bimanual", "view": "multiview",
 "model": "ctrlworld_4view_grid", "split": "makovian",
 "task": "insert_washer_shaft_both_hands"}
```

Four axes matter to the judge:

| axis | values |
|---|---|
| `method` | `dense`, `augment`, `worldcache` |
| `embodiment` | `bimanual`, `humanoid`, `single_arm` |
| `view` | `multiview`, `singleview` |
| `model` | `ctrlworld`, `ctrlworld_4view_grid`, `dreamgen`, `dreamdojo` |

`split` (`makovian` / `non_makovian`) partitions clips *inside* a cell: it
changes neither the reader nor the packing, so it is a sub-partition, not an
axis of the cell.

Note that **packing rides on `model`, not on `view`**: `ctrlworld` and
`ctrlworld_4view_grid` are both `multiview` and are packed differently. Any
design keyed on `view` alone is wrong on arrival. This is what the `view_id`
indirection below exists for.

## The unit: a cell

    cell_id = <method>.<embodiment>.<view_id>.<model>
    e.g.      dense.bimanual.mv4_grid.ctrlworld_4view_grid

A cell is one (corpus, packing, robot, reader) combination. Everything else
resolves from it. `cell_id` names the cache directory, the checkpoint, the score
output directory, and every manifest, so `grep -r <cell_id>` finds every
artifact that cell ever produced.

## Configuration: four files, manually written

    configs/views.yaml    view_id -> panel geometry. The only definition anywhere.
    configs/robots.yaml   robot   -> embodiment, K, FK spec key
    configs/cells.yaml    cell_id -> robot, adapter, source glob, reader, train tree, status
    configs/bench.yaml    which cells THIS RUN scores -- a selection, never a definition

Robot, view and training corpus are entered by hand. That is deliberate: the
packing of a new `model` must be *measured* before it is declared (see
`EXTENDING.md`), and auto-detection would turn a measurement into a guess.

`views.yaml` holds only packings that exist:

```yaml
views:
  sv1:               {n_views: 1, packing: none}
  mv3_row:           {n_views: 3, packing: width,   n_panels: 3, panel: [320, 192],
                      order: [exterior_1, exterior_2, wrist]}
  mv4_row:           {n_views: 4, packing: width,   n_panels: 4}
  mv4_grid:          {n_views: 4, packing: grid2x2, n_panels: 4, panel: [384, 216]}
  mv4_grid_br_blank: {n_views: 3, packing: grid2x2, n_panels: 4, panels: [0, 1, 2],
                      panel: [384, 216]}
```

A dead quadrant and a dropped wrist camera are the same operation — a panel
subset — so `mv4_grid_br_blank` needs no new code.

## Adapters

An adapter reads one source shape and yields episodes. It never writes a file.

```python
@dataclass(frozen=True)
class RawEpisode:
    episode_id: str
    views: dict[str, str]        # view name -> mp4 path, or one packed path
    joints: np.ndarray | None    # [T, J] in the source's own order
    joint_names: tuple[str, ...]
    gripper: np.ndarray | None
    fps: float
    scene_key: str               # task identity, for stratified splitting
    source_path: str


class DatasetAdapter(Protocol):
    SOURCE_ID: str
    def episodes(self, root: str) -> Iterator[RawEpisode | SkippedEpisode]: ...
```

`registry/materialize.py` owns the canonical format for every adapter: it packs
panels to the cell's `ViewSpec`, reindexes joints into the robot's canonical
order and fails on an unmappable joint name, assigns the train/val split, and
writes the annotations, the dataset card and the manifest.

Adding a corpus is one adapter class plus one `cells.yaml` entry. A corpus whose
on-disk shape already has an adapter is the YAML entry alone.

## Two canonical trees

**Score tree** — generated rollouts, no joint labels; the detectors run off what
the reader predicts.

    canonical/<cell_id>/<split>/<episode_id>/pred.mp4

**Train tree** — real teleop with logged state, the reader's supervision.

    train/<cell_id>/videos/{train,val}/<episode_id>.mp4
    train/<cell_id>/annotation/{train,val}/<episode_id>.json
    train/<cell_id>/dataset_card.json

Annotation JSON is `{"joint_source": "real", "observation.state.joint_position":
[[...J...], ...T], "observation.state.gripper_position": [...]}`, gripper
optional.

## Train and score cannot disagree

Both take `--cell <cell_id>` and resolve packing, robot, reader and paths from
the same `CellSpec`. There is no `--n-views` and no `--view-order`. Cache and
checkpoint paths derive from `cell_id` through one function, so a training run
and a scoring run cannot mean different things by the same name.

`CacheHeader` gains `cell_id` alongside the `view_layout_key` / `backbone_id` /
`tokens_per_view` fields it already carries, so a cache built for one cell
cannot feed a head being trained for another.

`load_reader` resolves the layout from the `CellSpec` and verifies the
checkpoint's own `cfg` (`robot`, `n_views`, `packing`, `n_out`) against it,
raising on mismatch. Today the layout comes from the caller and the checkpoint
cfg is never read, so a reader trained on three panels can be handed two and
nothing notices.

## Provenance

Every command — `data`, `cache`, `train`, `score` — writes a `run_manifest.json`:

```json
{"cell_id": "...", "command": "...", "argv": ["..."],
 "git_sha": "...", "git_dirty": false, "kinescore_version": "...",
 "views_sha256": "...", "cells_sha256": "...", "reader_sha256": "...",
 "started_at": "...", "host": "..."}
```

`reader_sha256` is the load-bearing field: `$KINESCORE_CKPT_DIR` currently holds
five `*_kp.pt` / `*_kp40.pt` pairs with identical `cfg` and no provenance file,
and no published number can be attributed to one rather than the other.

## What ships

    src/kinescore/
        registry/     views, cells, adapters, materialize, provenance
        adapters/     one module per source shape
        core/         clip, reader, metric, robot, contracts, resample, registry
        backbones/    dino
        heads/        heteroscedastic, views
        readers/      direct_keypoint, checkpoint
        robots/       <name>/{constants,fk,spec} + base
        training/     cache, datasets, splits, trainer, calibrate
        violations/   detectors, scorer
        video/        decode
        cli/          data, cache, train, score, report

## What is deleted

Verified by import graph, not by name. Every reference to `kinescore.metrics`
from `violations/detectors.py`, `readers/direct_keypoint.py` and
`core/scorer.py` is **docstring prose only** — the violations path is already
standalone, so the metric suite detaches cleanly.

| deleted | lines | why |
|---|---|---|
| `metrics/` | 2073 | the FK 31-metric suite; no shipped reader produces `q`, so every metric in it is unavailable by construction |
| `heads/ranges.py` | — | `clamp_for_fk` is the FK inference path |
| `readers/heteroscedastic.py`, `readers/checkpoint.py` | — | joint-angle reader and the v1 checkpoint format |
| `training/trainer_rawrad.py`, `losses.loss_limit` | 442 | joint-angle training |
| `cli/cmd_train_rawrad.py`, `cmd_aggregate`, `cmd_rank`, `cmd_reference`, `cmd_anchor`, `cmd_describe`, `cmd_export`, `cmd_manifest` | — | consumers of the deleted suite |
| `reference/` | 948 | reference-fingerprint ranking, used only by `cmd_rank` / `cmd_reference` |
| `robots/inertia.py` | 317 | torque metrics |
| `bench/` (most) | ~7035 | the cell matrix collapses into `registry/`; `csv_export`, `suites`, `traces`, `separation`, `noise_floor`, `stats`, `rank`, `report` are metric-suite reporting |
| `scripts/*rawrad*.sh`, `convert_*_to_train.py` | ~1000 | replaced by adapters |
| `legacy_docs/`, `TASKS.md`, `CHANGELOG.md` | — | history |

Deletion order is checked by the import graph at each step: a module leaves only
when nothing that ships imports it.

## The no-history rule

No code comment may describe what the code used to be. Specifically banned:
"ported from", "previously", "the bug this closes", defect identifiers, dated
notes, references to `legacy_docs/`, "kept for backward compatibility",
commented-out code, and prose recording a resolved disagreement.

The distinction that keeps this from destroying real content: **operative
rationale stays, history goes.** A constraint that still governs behaviour is
stated in the present tense as what the code requires —

    Bone 1 spans a rotating joint: its length tracks actuation, so scoring it
    reports ordinary articulation as a warp. Rigid indices are resolved by
    measured stability, not by distance from URDF rest.

— while the same fact told as a story ("the ctrlworld run had to drop this after
we found...") is history and does not ship. Findings worth keeping that have no
present-tense form go to `docs/`, not into the source.

Enforced by extending `tools/check_repo_hygiene.py` with a banned-phrase pass
over `src/`, run in the same gate as `ruff` and `mypy`.

## Phases

Each phase ends green on `pytest -q`, `ruff check src tests scripts`,
`mypy src`, and each is independently revertable.

**P1 — amputate.** Delete everything in the table above, in import-graph order.
Nothing is added. The repo shrinks to the keypoint path and stays green.

**P2 — registry.** `registry/` + `views.yaml` + `robots.yaml` + `cells.yaml`
describing the cells that exist. One definition of a packing; `AXIS_VALUES`
derived from YAML. Characterisation tests pin every existing cell's resolved
layout.

**P3 — adapters.** `DatasetAdapter` + `materialize.py`; one adapter for the HF
`{method}/{embodiment}/output/{view}/{model}/{split}/` shape driven by
`catalog.json`, one for ctrlworld's `input/` teleop trees. The two existing
converter scripts are deleted, not ported.

**P4 — train.** `kinescore train --cell` writing `head_target="keypoints"`. This
path does not exist in the repo today: all five shipped readers came from an
external driver. Validated by reproducing `airbot_mmk2_ctrlworld_kp.pt`'s
11.31 mm — the strongest reader and therefore the least forgiving target.

**P5 — score.** `kinescore score --cell --videos <dir>` folding
`score_singleview_direct.py` and the lost ctrlworld driver into one committed
command. Per-cell `scored.jsonl` + run-level `frame_scores.csv`,
`thresholds.json`, `run_manifest.json`.

**P6 — verification and provenance.** `cell_id` into `CacheHeader`, cfg
verification in `load_reader`, `run_manifest.json` from all four commands, and
the hygiene gate.

## Open

* `ctrlworld_4view_grid` is not on local disk; its packing (four panels in a row
  versus a 2x2 grid) must be measured before `views.yaml` names it. Both
  `mv4_row` and `mv4_grid` are listed above so whichever it turns out to be is
  already expressible.
* Local `ctrlworld` footage is 3 panels on width, confirmed by seam detection
  (boundaries at x=320 and x=640 stand 18-28x above the median column
  difference; nothing at 240/480/720). `single_arm` stays on this packing in the
  catalog while `bimanual` and `humanoid` move to `ctrlworld_4view_grid`.
* Three cells have data but no reader and no training corpus, because the
  generators that produced them log no state: `bimanual.mv4_grid.dreamgen`,
  `single_arm.mv4_grid_br_blank.dreamgen`, `single_arm.sv1.*`. They are blocked
  on a source of real joint logs for that (embodiment, view), not on code.
* Whether the published `ctrlworld` multiview numbers used two or three exposed
  views cannot be recovered: that scoring driver was never committed. They are
  re-run under P5 rather than re-labelled.
