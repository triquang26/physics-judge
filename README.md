# kinescore

**Physics-plausibility benchmark for AI-generated robot video.**

A world model that draws a robot must draw one that *moves like a robot*.
kinescore reads the robot's joint configuration out of the pixels with a frozen
vision backbone, projects it onto an **exact forward-kinematics model**, and
measures **analytic physics residuals** — no simulator, no VLM judge, no
learned critic downstream of the pose reader.

Two properties make the score trustworthy:

- **Non-circular.** The pose reader uses a frozen backbone the evaluated model
  never trained against, so a model cannot score well by matching the judge's
  features.
- **Valid by construction.** The reader regresses *joint angles*, not free 3-D
  points, and every reading is passed through real forward kinematics. Every
  pose it reports is a physically realisable robot configuration, which is what
  makes rigidity, joint-limit and feasibility residuals mean anything.

## Input contract

**Video files.** A clip is an mp4 (or a frame directory); the frame rate is
probed with `ffprobe` and `dt = 1/fps`. Anything that can write a video can be
scored — Cosmos, DreamGen, Ctrl-World, your own model — without kinescore
knowing how it generates. There is no `diffusers` dependency, not even optional.

## Install

```bash
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -e ".[dino,video,bench,dev]"
cp .env.example .env      # fill in KINESCORE_ASSETS, KINESCORE_DATA_ROOT, etc.
kinescore doctor          # what's present, what's missing, no network needed
```

`torch` (`>=2.5,<3`) is a base dependency and pulled in automatically; install
a CUDA build first (`uv pip install torch --index-url
https://download.pytorch.org/whl/<cuXXX>`) before the line above if you need
GPU support pinned to a specific CUDA version.

## Quickstart

```bash
kinescore manifest  --root  $KINESCORE_DATA_ROOT --out out/manifest.parquet
kinescore reference build   --manifest out/manifest.parquet --role gt --role real --out out/reference.pt
kinescore score     --manifest out/manifest.parquet --robot franka_panda --reader $CKPT --out out/
kinescore aggregate out/ && kinescore report out/stats.json
```

This is the shortest path from "clips on disk" to "an HTML report." For the
full pipeline (benchmark matrix config, ingest/verify, CSV export per cell,
per-frame traces, the frame-rate rules that make a comparison valid) see
[docs/DATA_PREP.md](docs/DATA_PREP.md) and
[docs/BENCHMARKING.md](docs/BENCHMARKING.md). To train a reader for a new
robot or dataset, see [docs/TRAINING.md](docs/TRAINING.md).

## Supported robots

| Robot | Registry key | DOF predicted | Cameras | Balance | Checkpoint | Status |
|---|---|---|---|---|---|---|
| Fourier GR-1 | `fourier_gr1` | 17 (arms + waist; legs/hands logged, not predicted) | 1 ego-view | yes — feet, CoM margin | `humanoid.pt` | **accepted**, val 19.19mm |
| Airbot MMK2 | `airbot_mmk2` | 12 (bimanual arms) | 1 (multi-cam data prepared, not yet cached) | n/a — bolted down | `airbot_mmk2_rawrad.pt` | **accepted**, val 19.52mm |
| Franka Panda | `franka_panda` | 7 + gripper (aux) | 1 or 3 (multiview) | n/a — bolted down | `single_arm_rawrad.pt` | **rejected**, val 162.10mm — retrain in progress, see [docs/TRAINING.md](docs/TRAINING.md) |
| ALOHA bimanual | `aloha_bimanual` | 12 (bimanual arms; grippers via `aux`) | 4 | n/a — table-mounted | none trained yet | robot registered, not yet scored |
| `Synthetic2R` | `synthetic_2r` | 2 | — | n/a | closed-form, no checkpoint | test/reference fixture only |

Every real robot uses the **same** reader family
(`ReadoutV2Head` → `HeteroscedasticPoseReader`, `limit_semantics="raw_rad"`) —
an earlier "squashed head" family that could never make joint-limit
violations observable was removed entirely, not kept as a second option; see
[legacy_docs/PROVENANCE.md](legacy_docs/PROVENANCE.md)'s "D7 addendum".
`--reader` auto-routes from the checkpoint's own `cfg`
(`kinescore.readers.loader.load_reader`). Adding a robot is implementing one
protocol — see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#adding-a-robot).

## What each metric does *not* detect

A benchmark is only as honest as its caveats. Full table in
[docs/METRICS.md](docs/METRICS.md); the traps worth knowing before you read a
number:

| Metric | Measures | Does **not** detect |
|---|---|---|
| Rigidity residual / wobble | link lengths staying constant | anything about *speed*; and under the full (unfiltered) bone set it can be contaminated by gripper actuation |
| Mean jerk | motion smoothness | a smoothly-executed but kinematically impossible motion |
| Joint-limit violation | joints driven past their stops | it needs a `raw_rad` reader (every checkpoint in the table above is one) — a hypothetical squashed reader would report a structural `0` here, and kinescore would report `null`, never a fabricated `0` |
| Accel/vel/no-teleport violation fraction | a fixed physical bound crossed | it thresholds a frame-rate-dependent quantity against an *absolute* constant, making it the metric group most sensitive to a wrong frame rate |
| Effort proxy | rough torque demand | real torque — there is no inertia model; comparative only, and `NaN` for a robot whose URDF declares no effort limits (GR-1) |
| PIS | aggregate deviation from real motion | comparability across different suites — only valid within one `suite_id` |
| KFD | distributional distance from real motion | anything across frame rates — it is not scale-invariant |

## Correctness posture

This code was merged from two research repositories. The numerical logic was
ported deliberately, but the sources carried defects that a benchmark cannot
ship with — a frame interval that silently defaulted to `0.2` on a corpus whose
true rate varied, an aggregate score that divided by a *varying* number of
terms, a rigidity metric contaminated by gripper opening, and a joint-limit
metric that was structurally incapable of firing.

Every one of those is fixed, and every fix is recorded in
[legacy_docs/PROVENANCE.md](legacy_docs/PROVENANCE.md) with the old behaviour,
the new behaviour, the rationale, and the test that pins it. Where a fix
changes a number, both the legacy and corrected values are kept as paired
fixtures, so a reviewer can see exactly which published numbers moved and why.

## Testing

```bash
pytest                # CPU-only, network-free, no checkpoints. < 60 s.
pytest -m ""          # everything, including GPU / network / checkpoint tiers
```

The default tier deliberately needs no GPU, no downloads and no trained
weights. The most valuable tests are the invariance properties — scoring a clip
at half the frame rate with a correspondingly doubled `dt` must leave every
derivative unchanged, and every registered metric must numerically match the
`dt` exponent it declares.

## Documentation

| | |
|---|---|
| [`docs/DATA_PREP.md`](docs/DATA_PREP.md) | where data goes, the mandatory format per cell, `ingest`/`verify`, training-input conversion |
| [`docs/TRAINING.md`](docs/TRAINING.md) | per robot: cache → train → the mm acceptance gate |
| [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) | config → run → CSV/traces → how to read the numbers, the frame-rate rules |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | the OOP map; how to add a robot / metric / data source |
| [`docs/METRICS.md`](docs/METRICS.md) | every metric: formula, units, `dt` exponent, what it misses |
| [`legacy_docs/`](legacy_docs/) | decision record — why things are the way they are, not how to use them today |

## Scope, honestly

This is an **evaluation methodology**, not a model. Absolute pose accuracy has a
floor (tens of millimetres), so kinescore is a **relative** referee: run the
same frozen judge on two systems and compare. Derivative-based metrics cancel
the reader's constant localisation bias, which is what makes that comparison
robust — but they do not cancel its *variance*, so a higher frame rate amplifies
reader noise into jerk. Score at a fixed, recorded frame rate.
