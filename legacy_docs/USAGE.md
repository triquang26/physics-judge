# Usage: what the code is, how to train a reader, how to benchmark

Three things happen here, in order. Everything else is support.

```
  train a pose reader          score clips                read the result
  (one per robot)      ──▶     against its robot   ──▶    CSV / rank / report
  kinescore train-rawrad       kinescore bench run        kinescore rank
```

A reader is only valid for **the robot it was trained on, against that robot's
own URDF**. There is no general reader. That constraint drives the whole layout.

---

## What the code is

```
src/kinescore/
  core/        the contracts: ClipSpec (timebase), RobotSpec, PoseReader,
               Metric/MetricSuite, Scorer. Everything else plugs into these.
  backbones/   frozen DINOv3 feature extraction
  heads/       the learned part: heteroscedastic (raw_rad, live) plus
               attentive/mlp/disentangled (legacy architectures, kept for
               checkpoint-format compatibility and provenance -- see D7 in
               PROVENANCE.md), per-camera view embedding
  readers/     head + backbone + limits, composed into a PoseReader;
               checkpoint load/save
  robots/      one package per robot: constants, FK, RobotSpec.
               franka · gr1 · airbot_mmk2 · synthetic (test fixture)
  metrics/     29 analytic rulers, grouped by what they measure
  training/    cache builder, datasets, losses, the raw_rad train loop
  bench/       matrix expansion, per-generator source plugins, manifest,
               runner, stats, separation, CSV export, traces
  video/       ffprobe timebase resolution, decoding, corruptions
  cli/         one module per subcommand, all torch-free at import
```

Two properties worth knowing before reading any of it:

- **`ClipSpec` is the only owner of `dt`.** There is no `dt` argument to forget
  and no default. It comes from `ffprobe`, always, and a config value that
  disagrees is a hard error rather than an override. See
  [RATE_POLICY.md](RATE_POLICY.md) for what this cost when it was not true.
- **A metric that cannot be computed returns `NaN` with a reason, never `0.0`.**
  `0.0` reads as "perfect". This invariant is tested
  (`tests/test_schema.py`) and is the reason several columns are empty rather
  than zero.

---

## Supported robots

| robot | `--robot` | joints | reader | `limit_semantics` | status |
|---|---|---|---|---|---|
| Fourier GR-1 | `fourier_gr1` | 17 + 12 hand | `readout_v2` | `raw_rad` | trained, val **19.19 mm** |
| Franka Panda | `franka_panda` | 7 + gripper | `readout_v2` | `raw_rad` | in training |
| Airbot MMK2 | `airbot_mmk2` | 12 (2 x 6) | `readout_v2` | `raw_rad` | trained, val **18.31 mm** |
| Synthetic 2R | `synthetic_2r` | 2 | — | — | test fixture, no reader |

`limit_semantics` is not cosmetic. Under a **squashed** head the prediction
cannot leave `[q_lo, q_hi]`, so `limit_violation_frac` is structurally `0`
regardless of the video — defect D7, reported as `null` rather than `0`. Under
`raw_rad` the head reports the raw angle even when impossible, and the clamp
overshoot *is* the violation signal. Every reader here is `raw_rad` -- the
squashed pose-reader path (and the legacy Franka checkpoint that used it)
was removed entirely, not kept as a control; see PROVENANCE.md's D7 addendum
for when and why.

Adding a robot is implementing one protocol — [ADDING_A_ROBOT.md](ADDING_A_ROBOT.md).
The URDF must be the real one for that robot; a fabricated kinematic chain
silently corrupts every number downstream.

---

## Training a reader

Two stages: cache the frozen backbone features once, then train the head on
them. The backbone never trains, so the cache is reusable across
hyperparameters and is the expensive part (tens of GB — and freely deletable).

```bash
# 1. features (GPU, slow, once per dataset+camera)
kinescore cache \
  --annotation-root $KINESCORE_DATA_ROOT/train/franka_droid_std/annotation \
  --video-root      $KINESCORE_DATA_ROOT/train/franka_droid_std/videos \
  --out             $KINESCORE_CACHE_DIR/franka_droid_std \
  --dino-model dinov3_vitl16 --patch-pool 2 --n-views 1

# 2. head (GPU, ~minutes)
kinescore train-rawrad \
  --cache-root      $KINESCORE_CACHE_DIR/franka_droid_std \
  --annotation-root $KINESCORE_DATA_ROOT/train/franka_droid_std/annotation \
  --robot franka_panda \
  --down-sample 3 \
  --steps 6000 --bs 2048 --lr 1e-3 \
  --out $KINESCORE_CKPT_DIR/single_arm_rawrad.pt
```

**`--down-sample` is required and has no safe default.** It aligns the joint log
to the video frames. For `droid_std`: 167 logged states against 56 video frames
gives 3. Derive it from the data — `video_length` and `raw_length` are in each
annotation JSON, and `fps` is in a LeRobot `meta/info.json` — and verify across
several episodes rather than one. Getting it wrong trains the head against
mis-paired labels and nothing downstream will tell you.

### What `train-rawrad` optimises

```python
lim  = loss_limit(mu, limits)                       # soft hinge from the URDF
loss = beta_nll_loss(mu, logvar, target, beta=0.5) + 0.05 * lim
```

Two phases: the first 1500 steps use `weighted_mse + 0.05*lim` with `logvar`
frozen, then it switches to beta-NLL. The warmup stops the head learning to
declare itself uncertain before `mu` is any good.

The limit weight of `0.05` is the design, not a tuning artefact: strong enough
to pull toward the feasible set when the image is ambiguous, far too weak to
gag the head when the image genuinely shows an impossible pose. A squash gags it
unconditionally — that single difference is what separates "violation measured
and found zero" from "violation unmeasurable".

**FK is not in the loss.** It appears only in evaluation, converting joint error
into a 3-D keypoint error in **millimetres** — the unit that is comparable
across robots with different joint counts and ranges, and the one a human can
judge. Note the consequence: the loss weights every joint equally while the
geometry does not, so a shoulder error costs far more in mm than a wrist error.
An FK-weighted loss is an obvious improvement and is not implemented.

### Accepting or rejecting a trained reader

Judge on held-out **keypoint mm**, not on loss. References: GR-1 19.19,
Airbot MMK2 18.31, legacy squashed Franka 57.38 (1-cam) / 49.01 (3-cam) --
that reader's path has since been removed (PROVENANCE.md's D7 addendum), the
number is kept only as a historical comparison point -- untrained baseline
359.93. A reader far outside that band should be reported as unusable rather
than used to produce numbers.

---

## Loading a reader

```python
from kinescore.readers.checkpoint import load_reader
from kinescore.robots import get_robot
from kinescore.core.clip import ViewLayout

robot  = get_robot("fourier_gr1")
reader = load_reader(f"{ckpt_dir}/humanoid.pt", robot=robot,
                     device="cuda", view_layout=ViewLayout(n_views=1))

readout = reader.read(frames)     # (T,H,W,3) | (T,3,H,W) | (B,T,3,H,W)
readout.q       # (B,T,n_joints)  clamped, safe for FK
readout.q_raw   # (B,T,n_joints)  RAW radians -- may exceed limits, that is the signal
readout.sigma   # (B,T,n_joints)  per-joint uncertainty (heteroscedastic heads)
```

`load_reader` dispatches on the checkpoint itself — the head family, view layout
and `limit_semantics` are read from the file, not passed in, so a checkpoint
cannot be loaded under the wrong reader family. `n_views` **must** match what
the checkpoint was trained with; a mismatch is caught by
`ViewLayout.assert_tokens` rather than silently consuming a wrong-sized grid
(defect D4).

Frames are moved to the reader's device automatically by `Scorer.score`; calling
`reader.read` directly means doing that yourself.

---

## Benchmarking

Two independent choices, which the CLI keeps separate:

| | decides | set in |
|---|---|---|
| **which clips** | data scope | `configs/*.yaml` -> `axes:` |
| **which rulers** | metric set | `--suite` |

```yaml
# configs/benchmark.yaml
axes:
  embodiment: [humanoid, single_arm]     # + bimanual once a reader exists
  view:       [multiview, singleview]
  horizon:    [makovian, non_makovian]
  cache:      [dense]                    # + dicache, fastercache, itm, pisa, ...
  generator:  [ctrlworld, dreamgen, dreamdojo]
na_cells:                                 # combinations that do not exist on disk
  - {generator: dreamdojo, view: multiview}
```

The axes are a Cartesian product minus `na_cells`. Cells that do not exist are
**declared** and reported as N/A — never as "0 episodes found", which is
indistinguishable from a broken glob.

```bash
kinescore data pull  --config configs/benchmark.yaml --dry-run   # patterns only
kinescore data pull  --config configs/benchmark.yaml
kinescore bench run  --config configs/benchmark.yaml --dry-run   # cell table
kinescore bench run  --config configs/benchmark.yaml --suite all_metrics
kinescore export     --results out/<run> --out out/<run>         # per-folder CSVs
kinescore rank       out/<run> --metric mean_jerk_mps3           # worst first
```

`data pull` derives its download patterns **from the axes**, so widening the
config and re-running pulls exactly the new cells. Always `--dry-run` first:
`video_gen_physics` has nine top-level directories and `dense/` is one of them.

### The three suites

| `--suite` | n | use |
|---|---|---|
| `all_metrics` | 28 | **score a fresh run with this** |
| `invariant_v1` | 26 | frozen; only to line up with previously published numbers |
| `rate_free` | 9 | the **only** valid basis for comparing clips at different frame rates |

Comparing a rate-dependent metric across different fps is not merely noisy, it
is wrong by orders of magnitude: jerk carries `dt_exponent = 3`, so 16 fps
against 10 fps inflates it by `(16/10)^3 = 4.1x` before any physics enters.
[RATE_POLICY.md](RATE_POLICY.md) has the measured cases.

### Output

```
out/<run_id>/
  bench_manifest.parquet
  results.jsonl                  one record per clip (docs/SCHEMA.md)
  dense/<emb>/<view>/<gen>/<horizon>/clips.csv     mirrors the data tree
  SUMMARY.csv                    one row per folder, carries suite_name + suite_id
  traces.npz                     per-frame arrays, plain np.load
  provenance.json
```

Every CSV carries `suite_id`, because rows scored under different suites have
different columns and are not poolable — `bench/stats.py::aggregate` refuses to
mix them.

---

## Before a long run

```bash
kinescore doctor        # torch, CUDA, ffprobe, env vars, asset hashes
pytest -q               # CPU-only, network-free, no checkpoints, < 60 s
```

`doctor` warms the `robot_descriptions` fetch that Franka's URDF needs, which is
otherwise a network dependency at first use.

Long runs: launch with `setsid nohup ... &` and poll a log. A foreground command
that exceeds a tool timeout gets SIGTERM'd and takes the job with it. And never
`pkill -f` a pattern that also matches your own shell's command line.
