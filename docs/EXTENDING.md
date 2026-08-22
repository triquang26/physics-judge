# Extending: new data, new robot, new view

Companion to `REFACTOR_PLAN.md`. That document defines the cell registry; this
one defines what you write when a new dataset, robot, or camera packing arrives.
Both describe the target design — see the plan's phase list for what exists.

## The two canonical formats

There are already **two** canonical trees in this repo, with different jobs.
Conflating them is the first mistake an adapter can make, so an adapter declares
which one it targets.

**(A) Bench tree** — `kinescore data ingest`, `CanonicalLayout`:

    bench/<cache>/<robot>/<view>/<generator>/<horizon>/episode_XXXX/{pred.mp4,gt.mp4}
    bench/.../cell_card.json

Archival and scoring. Holds *generated* rollouts. Carries no joint labels: the
violation detectors run off keypoints the reader predicts, so nothing here needs
ground-truth state. A prediction with no ground truth is still kept (this is why
`RawHFLayout` deliberately does not reuse the GT-strict `ClipSource` plugins).

**(B) Train tree** — `convert_*_to_train.py`:

    train/<domain>/videos/{train,val}/<episode_id>.mp4      # panels packed per the cell's ViewSpec
    train/<domain>/annotation/{train,val}/<episode_id>.json
    train/<domain>/dataset_card.json

Supervision for the reader. Requires **real** logged state — `datasets.py`
asserts `joint_source == "real"`. The JSON is:

```json
{"joint_source": "real",
 "observation.state.joint_position": [[...J floats...], ...T],
 "observation.state.gripper_position": [...]        // optional
}
```

Real teleop footage targets (B). Generator output targets (A). A dataset with
both (ctrlworld ships `input/` with `metadata.json` alongside `output/`)
produces one of each, from two adapter runs.

## Adapters: read only, never write format

Today `convert_ctrlworld_input_to_train.py` both *reads* ctrlworld's directory
shape and *decides* the output format — `N_VIEWS = 3`, panel size, `vstack`, the
JSON keys, the split policy are all module constants inside it. That is why a
second dataset means a second 500-line script, and why the two existing scripts
have already drifted apart.

Split it: **adapters own reading, one shared writer owns the canonical format.**

```python
@dataclass(frozen=True)
class RawEpisode:
    episode_id: str
    views: dict[str, str]              # view name -> mp4 path, OR one packed path + its ViewSpec
    joints: np.ndarray | None          # [T, J], in the SOURCE's own order
    joint_names: tuple[str, ...]       # what the source calls those J columns
    gripper: np.ndarray | None         # [T] or [T, 2]
    fps: float
    scene_key: str                     # task/scene identity, for stratified splitting
    source_path: str                   # provenance, one per episode


class DatasetAdapter(Protocol):
    SOURCE_ID: str                     # "ctrlworld_input", "lerobot", "robocoin", ...
    def episodes(self, root: str) -> Iterator[RawEpisode | SkippedEpisode]: ...
```

`registry/materialize.py` then does everything downstream, once, for every
adapter: pack panels to the target cell's `ViewSpec`, reindex joints into the
robot's canonical order, assign the train/val split, write the JSONs, write
`dataset_card.json` and `source_manifest.json`. An adapter never calls ffmpeg
and never decides a filename.

Three things this buys, each fixing something currently informal:

* **Joint order becomes checked, not commented.** `datasets.py` reads `n_joints`
  off the array width and is robot-agnostic — nothing verifies the *order*
  matches the robot spec. The aloha converter drops source indices 6 and 13 by
  hand, explained only in a prose note in its `dataset_card.json`. As
  `joint_names` on `RawEpisode`, the writer maps source order to
  `RobotSpec`-canonical order and fails on an unmappable name.
* **Skips become uniform.** The aloha converter already records
  `n_episodes_skipped` / `skip_reasons_summary` / `skip_examples`; the other
  does not. `SkippedEpisode` makes that the shared path, so "why is this corpus
  237 and not 239" is always answerable.
* **Packing stops being per-script.** The target packing comes from the cell's
  `ViewSpec`, so the same adapter can materialise a `mv3_row` tree and an `sv1`
  tree with no code change.

Adding a dataset = one adapter class (`episodes()` and nothing else) + one
`cells.yaml` entry. Adding a dataset that reuses an existing on-disk shape =
just the YAML entry.

## The loader needs nothing per dataset

Worth stating plainly, because it is the payoff of the above and it is easy to
over-engineer here. The full path is:

    raw corpus
      -> adapter          (per source: the only new code)
      -> train tree (B)   (one shared writer)
      -> kinescore cache  (DINO tokens + CacheHeader)
      -> load_split       (flatten to RAM)
      -> trainer

The cache stage normalises everything: after it, one episode is a `[T, n_views *
tokens_per_view, D]` tensor plus a `CacheHeader`, regardless of which corpus it
came from. So **no new DataLoader is written per dataset, ever** — the variation
is fully absorbed by the adapter.

Two changes to that pipeline, both small:

* **`cell_id` into `CacheHeader`.** It already guards `view_layout_key`,
  `backbone_id`, `tokens_per_view` — the D4 defect class. `cell_id` closes the
  remaining hole: a cache built for `bimanual.mv3_row.ctrlworld` cannot be fed
  to a head being trained for `bimanual.sv1.dreamdojo`.
* **Put `load_split` behind a `SplitSource` interface.** It flattens an entire
  split into one RAM tensor — a deliberate two-pass design that keeps peak at
  ~1x rather than 2x, and it is the validated fast path, so it stays the
  default. But the trainer should depend on the interface, not on `SplitData`,
  so a memmap implementation can land the day a corpus outgrows RAM. Do not
  build that implementation before then.

## What "keypoint-based" does and does not remove

Worth stating before the robot checklist, because the phrase invites a wrong
inference and the wrong inference changes what a new robot costs.

"Keypoint-based" describes the **head's output**, not the dataset.
`ReadoutV2Head` emits `n_out = 3K` numbers reshaped straight to `(B, T, K, 3)`
— K points in the robot base frame, metres. The joint-based variant emits J
angles and must then run FK and `clamp_for_fk` to get points. So what the
direct-keypoint reader removes is **FK at inference**: no robot, no URDF, no
joint limits in the forward path, and `q`/`q_raw` are `None`
(`readers/direct_keypoint.py`).

It does **not** remove FK from the project. No dataset here logs 3-D keypoints;
the annotations carry `observation.state.joint_position` `[T, J]`, and the
supervision target is manufactured from it — `trainer_rawrad.py:215-216`:

```python
P_pred = robot.forward_kinematics(q_pred[:, None], None)
P_true = robot.forward_kinematics(tgt[:, None], None)
```

FK moves from inference time to data-prep time. A new robot still needs it,
unless its corpus logs 3-D keypoints directly.

The loss is `beta_nll_loss` (Seitzer 2022 beta-NLL, diagonal Gaussian) over
`mu`/`logvar`; a keypoint head regresses 3K coordinates and carries **no**
`loss_limit` term, there being no joint angles to bound.

> The exact recipe is **not in version control.** `train-rawrad` has no
> `head_target` flag and all five `*_kp.pt` checkpoints came from an external
> driver, so the paragraph above is inferred from `n_out = 3K` plus the two
> losses that do ship — not read off the trainer that produced them. This is
> what phase P3 exists to fix.

What the five violation detectors each require, checked against
`violations/detectors.py`:

| detector | needs | notes |
|---|---|---|
| jerk | P only | third difference |
| teleport | P only | |
| self_collision | P only | min distance over non-adjacent keypoint pairs |
| joint_limit | P only | bend angle from `P[i-1], P[i], P[i+1]`; envelope fitted on GT |
| **rigidity** | P + `robot.rigid_bone_pairs` / `rigid_bone_lengths` | the **only** detector touching `ctx.robot` (line 215) |

Note that `violations.JointLimitDetector` scores bend angles derived from
keypoints and therefore works fine on a keypoint reader. This is a different
thing from the `limit_semantics = "keypoints"` note in `core/reader.py`, which
is about the *Readout* being unable to report a joint-limit violation. Do not
read the second as implying the first.

## New robot

    src/kinescore/robots/<name>/
        constants.py   joint names, keypoint link names, URDF path
        fk.py          forward kinematics
        spec.py        RobotSpec: name, n_joints, keypoint_links,
                       bone_pairs, bone_lengths, rigid_bone_pairs, rigid_bone_lengths
        __init__.py
    assets/<name>.urdf
    kinescore/robots/__init__.py   +1 lazy builder
    configs/robots.yaml            +1 entry (embodiment, K)
    an adapter                     producing the train tree

FK is required even though the reader never calls it: it turns logged joint
angles into the `[T, K, 3]` supervision target. A corpus that logs 3-D keypoints
directly is the one case that needs no FK.

`assert_keypoints_in_urdf` validates the keypoint link names against the URDF,
and `structural_rigid_bone_mask` derives which bones have enough rest length to
score. Both live in `robots/base.py`.

The rigidity detector needs bone pairs with rest lengths, but its dependency on
their exactness is weaker than it looks: the threshold is the 95th percentile of
`|L - rest|` over real motion, so a constant scale bias is absorbed completely —
a reader that reads one bone consistently 3x too long still passes. What no
calibration absorbs is a bone spanning a moving joint, whose length tracks
actuation and makes ordinary articulation read as a warp. Rigid indices are
therefore resolved by measured stability across real poses, not by distance from
URDF rest: a biased-but-rigid reader passes the first rule and fails the second.

## New view packing

One `views.yaml` entry. `ViewLayout` already covers height stacks, width stacks,
2x2 grids, and panel subsets, which spans every corpus measured so far; a fifth
packing mode should arrive with the data that needs it.

The subset field is the part people miss. A dead quadrant and a dropped wrist
camera are the same operation:

```yaml
mv4_grid_br_blank: {n_views: 3, packing: grid2x2, n_panels: 4, panels: [0, 1, 2]}
mv3_row:           {n_views: 3, packing: width,   n_panels: 3, panel: [320, 192]}
```

Before adding an entry, **measure the packing** rather than reading it off the
generator's documentation. Seam detection (mean absolute column-to-column
difference, averaged over ~8 frames) locates panel boundaries unambiguously: on
ctrlworld's 960x192 frames the boundaries at x=320 and x=640 stand 18-28x above
the median column difference, and no candidate boundary exists at 240/480/720 —
that is what settles 3 panels versus 4. For grid packings, per-quadrant standard
deviation across ~14 clips settles whether a quadrant is systematically blank
(single_arm's bottom-right: std 1.2, versus 61-64 for the other three).

One sample is not enough to make either call. The comment at
`bench/sources/dreamgen.py:115` was written from one visual sample and reads "3
populated quadrants and 1 solid-black one ... all 4 panels are kept, unnamed,
until that is actually checked"; measured across clips, the blank quadrant turns
out to be **robot-dependent** — present for single_arm, absent for bimanual.
A dataset-wide constant cannot express that. A per-cell `view_id` can.
