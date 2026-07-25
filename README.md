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

## Quickstart

```bash
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu126
uv pip install -e ".[dino,video,bench,dev]" -c constraints.txt
cp .env.example .env      # fill in KINESCORE_ASSETS etc.

kinescore manifest  --root  $KINESCORE_DATA_ROOT --out out/manifest.parquet
kinescore reference build   --manifest out/manifest.parquet --real-only --out out/reference.pt
kinescore score     --manifest out/manifest.parquet --robot franka --reader $CKPT --out out/
kinescore aggregate out/ && kinescore report out/
```

## Supported robots

| Robot | DOF | Cameras | Balance | Notes |
|---|---|---|---|---|
| Franka Panda | 7 + gripper | 1 or 3 (multiview) | n/a — bolted down | DROID-style exterior + wrist views |
| Fourier GR-1 | 17 + 12 hand | 1 ego-view | yes — feet, CoM margin | bimanual humanoid |

Adding a third is implementing one protocol: see [`docs/ADDING_A_ROBOT.md`](docs/ADDING_A_ROBOT.md).

## What each metric does *not* detect

A benchmark is only as honest as its caveats. Full table in
[`docs/METRICS.md`](docs/METRICS.md); the traps worth knowing before you read a
number:

| Metric | Measures | Does **not** detect |
|---|---|---|
| Rigidity residual / wobble | link lengths staying constant | anything about *speed*; and under the full bone set it is contaminated by gripper actuation (see D9) |
| Mean jerk | motion smoothness | a smoothly-executed but impossible motion |
| Joint-limit violation | joints driven past their stops | **nothing at all under a squashed-head reader** — it is structurally `0` (see D7), and kinescore reports `null`, never `0` |
| Accel violation fraction | accelerations over a fixed bound | it uses an *absolute* threshold, so it is the metric most sensitive to a wrong frame rate |
| Effort proxy | rough torque demand | real torque — there is no inertia model; comparative only |
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
[`docs/PROVENANCE.md`](docs/PROVENANCE.md) with the old behaviour, the new
behaviour, the rationale, and the test that pins it. Where a fix changes a
number, both the legacy and corrected values are kept as paired fixtures, so a
reviewer can see exactly which published numbers moved and why.

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
| [`docs/METRICS.md`](docs/METRICS.md) | every metric: formula, units, `dt` exponent, what it misses |
| [`docs/PROVENANCE.md`](docs/PROVENANCE.md) | where each file came from; every intentional behaviour change |
| [`docs/MODIFYING.md`](docs/MODIFYING.md) | common changes → which files, which tests, which docs |
| [`docs/ADDING_A_ROBOT.md`](docs/ADDING_A_ROBOT.md) | the `RobotSpec` protocol, walked through |
| [`docs/ADDING_A_METRIC.md`](docs/ADDING_A_METRIC.md) | the `Metric` protocol and its mandatory declarations |
| [`docs/SCHEMA.md`](docs/SCHEMA.md) | the canonical result record |

## Scope, honestly

This is an **evaluation methodology**, not a model. Absolute pose accuracy has a
floor (tens of millimetres), so kinescore is a **relative** referee: run the
same frozen judge on two systems and compare. Derivative-based metrics cancel
the reader's constant localisation bias, which is what makes that comparison
robust — but they do not cancel its *variance*, so a higher frame rate amplifies
reader noise into jerk. Score at a fixed, recorded frame rate.
