# Training a pose reader

Per robot: cache frozen-backbone features once, train a `ReadoutV2Head`
against them, then measure held-out 3-D keypoint error in mm against the
robot's own URDF before trusting the checkpoint for anything. See
[DATA_PREP.md](DATA_PREP.md) for how the input (`annotation/{train,val}/*.json`
+ `videos/`) gets built in the first place, and
[ARCHITECTURE.md](ARCHITECTURE.md) for where `readers/`/`heads/`/`training/`
sit relative to everything else.

## There is one training loop now, not two

An earlier version of this codebase had two head/reader families: a squashed
head (`AttentivePoseHead` → sigmoid-into-limits, `training/trainer.py`, the
`kinescore train` subcommand) and a heteroscedastic raw_rad head
(`ReadoutV2Head`, `training/trainer_rawrad.py`, `kinescore train-rawrad`). The
squashed path was **removed entirely** — not merely deprecated — once every
real embodiment had (or was actively getting) a raw_rad reader, because a
squashed head structurally cannot report a joint-limit violation
(`q = lo + (hi-lo)*sigmoid(raw)` can never leave `[lo,hi]`, so
`limit_violation_frac`/`limit_excess_rad` are `0` by construction on any
squashed checkpoint, not by anything about the video). `readers/loader.py::
load_reader` still *detects* a squashed-format checkpoint's `cfg` shape — so
the error is legible — but raises `NotImplementedError` rather than loading
it. **`kinescore train-rawrad` is the only training command; there is no
`kinescore train`.** `scripts/train_airbot_mmk2_fullarm_squashed.sh` is kept
in this repo as the historical record of the squashed attempt that motivated
the retrain — it will not run against the current CLI.

## Step 1 — cache: `kinescore cache`

```bash
kinescore cache \
  --video-root      <video_root> \
  --annotation-root <annotation_root> \
  --out             <cache_root> \
  --split train --split val \
  --n-views <N> \
  --device cuda
```

Runs a frozen DINOv3 backbone (`--dino-model dinov3_vitl16` default,
`--dino-input`/`--patch-pool` control resolution/pooling) over every
`{split}/{episode}.mp4` and writes one self-describing `.pt` per episode to
`{out}/{split}/{episode}.pt` — self-describing meaning each carries a
`CacheHeader` (view layout key, backbone id, token shape), so a 3-view cache
can never silently feed a 1-view head's assumptions (or vice versa;
`ViewLayout.assert_matches_layout` is the runtime check). `--n-views` must
match the number of cameras packed into the video (see
[DATA_PREP.md](DATA_PREP.md#multi-camera-packing)); this repo's real caches
are all single-view (`--n-views 1`) as of this writing, even for the
multi-cam-capable robots — see the camhigh experiments below for why a
second camera choice was tried at all. `--frame-chunk` exists for a long
episode at high `--dino-input` that would otherwise OOM a shared GPU
(encodes at most that many frames per backbone call; numerically identical
result, just less peak memory).

## Step 2 — train: `kinescore train-rawrad`

```bash
kinescore train-rawrad \
  --cache-root      <cache_root> \
  --annotation-root <annotation_root> \
  --robot <robot_name> --down-sample <N> --n-views <N> \
  --steps 6000 --phase-a 1500 --bs 2048 --lr 1e-3 \
  --beta 0.5 --limit-weight 0.05 \
  --device cuda --seed 0 \
  --out <ckpt_path>.pt
```

`--down-sample` is **required, with no default** — see
[DATA_PREP.md](DATA_PREP.md) for what it means and why getting it wrong fails
silently (a worse-converging loss, not an error).

### The training recipe — identical for every robot

`training/trainer_rawrad.py::RawRadTrainer.fit`, two phases, `phase_a =
min(cfg.phase_a, max(1, cfg.steps // 2))` (so phase B always runs at least
half the steps regardless of what `--phase-a` asks for):

- **Phase A** (`step <= phase_a`, default 1500 of 6000 steps):
  `loss = F.mse_loss(mu, target) + 0.05 * loss_limit(mu, limits)`. `logvar`
  gets no gradient here — the head's uncertainty head stays at its random
  init through this phase.
- **Phase B** (`step > phase_a`): `loss = beta_nll_loss(mu, logvar, target,
  beta=0.5) + 0.05 * loss_limit(mu, limits)` — Seitzer et al. 2022's
  beta-NLL, `nll = 0.5*(logvar + (target-mu)^2 * exp(-logvar))` weighted by
  `var.detach()**beta`. At the exact phase boundary, `lr` drops from `--lr`
  (default 1e-3) to `--lr-phase-b` (default 5e-4) and the cosine schedule
  re-initialises over the remaining steps.

`loss_limit` (`training/losses.py`) is a **soft** hinge — `relu(q_hat - hi) +
relu(lo - q_hat)`, mean-reduced — deliberately soft so an occasional
near-limit prediction doesn't dominate the loss; it is a regulariser, not the
mechanism that makes limit violations observable (that's `q_raw` bypassing
`clamp_for_fk` entirely — see [ARCHITECTURE.md](ARCHITECTURE.md)).

**FK is not in the training loss anywhere.** It appears only at evaluation
(`eval_keypoint_mm_rawrad`), converting a joint-space error into the 3-D
keypoint-mm figure the acceptance gate below is measured in. Training is
pure joint-angle regression; FK is how the result gets checked.

## Step 3 — the mm acceptance gate

Every reader is judged by held-out 3-D keypoint error in mm — run the
val-split joint prediction through the robot's own `forward_kinematics` and
compare to the keypoints the *logged* (real) joints would produce, mean L2
over frames and keypoints. This is a floor-comparable number across robots
because it's measured in physical millimetres, not a robot-specific
joint-angle-error unit.

| robot | checkpoint | trained here? | val keypoint mm | verdict |
|---|---|---|---|---|
| Untrained baseline | — | — | **359.93** | — |
| Fourier GR-1 | `humanoid.pt` | **no** — copied from a sibling research checkout's `readout_v2_gr1.pt` | **19.19** | accepted |
| Airbot MMK2 | `airbot_mmk2_rawrad.pt` | yes (`scripts/train_airbot_mmk2_fullarm_rawrad.sh`) | **19.52** (best_step 5500, train 8.40) | accepted |
| Franka Panda | `single_arm_rawrad.pt` | yes | **162.10** | **rejected** — roughly 3x the accepted band; the reader's own error would swamp the effect a benchmark is trying to measure |

The band: 19.19/19.52 sit close together and close to GR-1's production
number; 162.10 does not — it is within striking distance of the *untrained*
baseline (359.93), which is the sign to read as "this reader cannot yet be
trusted to see what it's scoring," not "slightly worse."

### Why Franka failed, and what fixes it

Root cause was the training data, not the recipe. `droid_std` (the dataset
this reader trained on) holds 396 train / 4 val episodes — every hundredth
episode of a much larger DROID corpus — and a 4-episode val set cannot
estimate a mean over a corpus where every episode is a different scene.
Per-episode val error ranged 73–365mm, not uniformly bad: one held-out
episode landed at the untrained baseline and dominated the mean, but the
other three still sat 1.3–2.3x above a healthy band even excluding it. Train
error (18.74mm) was genuine (checked clean for train/val leak at both the
annotation-file and cache-file level) — the model **can** learn this
mapping, it just wasn't shown enough of the corpus's variation to generalise.
The fix in progress is a larger converted subset, split **stratified by
scene** rather than randomly (`training/splits.py::stratified_episode_split`
exists for exactly this — grouping by a scene key so a whole scene lands on
one side of the split, never leaking into both), on the unchanged training
recipe above.

### Why two Airbot MMK2 camera caches exist

`airbot_mmk2_rawrad.pt` (the accepted checkpoint, 19.52mm) was trained on
`cam_third_view`/`cam_front_rgb` — whichever camera per task shows the full
bimanual arm chain, the obvious choice for reading joint angles out of
pixels. But the generated `dense/humanoid/.../multiview/ctrlworld/` cells
this reader is meant to score only render `cam_high`, `cam_left_wrist`,
`cam_right_wrist` — **none of which is the training camera**. `cam_high`
alone shows at best a gripper-corner sliver for one task subset (106/248
episodes) or a partial forearm for the other (142/248); running the accepted
checkpoint against those cells would not error, it would produce
plausible-looking numbers for a robot the reader cannot actually see. See
`$KINESCORE_CKPT_DIR/airbot_mmk2_NOTICE.txt` for the full caveat (it also
documents a synthetic constant-0 gripper placeholder in this checkpoint's
13th output channel — treat that channel as absent, not as 0, in anything
that reads the head's raw output directly).

`scripts/cache_airbot_mmk2_camhigh_subsetA.sh` / `subsetB.sh` +
`scripts/train_airbot_mmk2_camhigh_subsetA_rawrad.sh` are the retraining
attempt on the camera the eval data actually has. Status as of this writing:
subset A's cache pass completed (142 episodes) and a training run was
started, but its log shows no completion (`[train]` progress lines or a
final `val_keypoint_mm`) — treat any `airbot_mmk2_camhighA.pt` on disk as
**unverified**, not a second accepted reader. Subset B's cache pass was
never run at all. Neither subset attempt supersedes `airbot_mmk2_rawrad.pt`
as the accepted checkpoint; they exist to eventually produce a reader that
can actually see what the eval cells render.

### `configs/benchmark.yaml`'s reader pins, and how they are marked

`configs/benchmark.yaml`'s `robots:` table now pins `airbot_mmk2_rawrad.pt`
(accepted) and `single_arm_rawrad.pt` for `franka_panda` — the two retired
squashed checkpoints (`single_arm.pt`, deleted along with the squashed path
itself; `airbot_mmk2.pt`, still on disk but squashed) are not referenced
anywhere in that file any more. `readers/loader.py::load_reader` would raise
`NotImplementedError` on either retired file; that failure mode is now
structurally unreachable from the shipped config, not just avoided by
convention.

Franka's retrain was **not** accepted (162.10mm, see the gate table above)
— pinning `single_arm_rawrad.pt` as if it were an ordinary `reader:` entry
would ship numbers whose own error dwarfs the effect this benchmark
measures, silently. Instead it carries `reader_status: failing_gate` (see
`kinescore.bench.config.RobotConfig.reader_status`) plus a `reader_note`
with the 162.10mm figure — a schema-level flag a downstream reader/tool
cannot miss the way a comment can be, not a hidden or omitted control.
`aloha_bimanual` carries `reader: null` / `reader_status: untrained` for the
same reason: no ALOHA checkpoint has ever been trained, so naming a
filename (the old `bimanual.pt`, which was never a real file) would be the
same phantom-checkpoint bug. `tests/test_bench_config.py::
TestShippedConfigsReadersAreResolvableOrHonestlyMarked` pins this: every
`robots.*.reader` in every shipped config either resolves to a real,
`load_reader`-routable file or is explicitly marked `failing_gate`/
`untrained` — see [BENCHMARKING.md](BENCHMARKING.md).

## Calibration: `kinescore calibrate`

A heteroscedastic head's raw `sigma` is not automatically well-calibrated —
`kinescore calibrate` fits a post-hoc temperature scale against a held-out
split (`--checkpoint`, `--cache-root`, `--annotation-root`, `--split val`,
`--down-sample` — same required-no-default rule as training) and writes the
fitted scale to `--out`. `readers/checkpoint_v2.py`'s loader applies it via
`resolve_sigma_scale` (explicit override > checkpoint `meta["sigma_scale"]` >
`1.0`) — the real GR-1 checkpoint carries `sigma_scale=1.9375`.
