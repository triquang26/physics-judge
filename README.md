# kinescore

**Physics-plausibility benchmark for AI-generated robot video.**

A frozen DINOv3 backbone + a trained diffusion head read 3-D robot keypoints
out of the pixels; five analytic detectors measure physics violations on those
keypoints. No simulator, no VLM judge. Two properties make the score
trustworthy:

- **Non-circular** — the reader is frozen and the evaluated model never
  trained against it.
- **Calibrated on real motion** — every threshold is the 95th percentile of
  the same quantity measured on real teleop from the same robot and packing.

It is a **relative** referee: keypoint accuracy floors at tens of millimetres,
so compare systems through the same reader rather than reading absolutes.

## Pipeline

    HF ──pull──▶ bench clips + corpora
                    │ data     corpus → train tree
                    │ cache    frozen backbone → tokens
                    │ train    diffusion head ← forward-kinematics targets
                    │ score    generated clips → per-segment verdicts
                    │ report   one table over every cell
                    │ export   numbered clips + segments.json for a rating UI
                    └ push     hf sync to the bucket (train/ scores/ web/)

- **reader** = `<robot>.<corpus>.<view_id>` — one trained head.
- **cell** = `<embodiment>.<view_id>.<model>` — one scored unit.

Both are declared in `configs/cells.yaml`; panel geometry in
`configs/views.yaml` is measured, never inferred.

## Install

```bash
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env      # KINESCORE_ASSETS / CKPT_DIR / CACHE_DIR / DATA_ROOT / OUTPUT_DIR
```

`ffmpeg`/`ffprobe` must be on `PATH`. For a specific CUDA build install torch
first: `uv pip install torch --index-url https://download.pytorch.org/whl/<cuXXX>`.

## Quickstart

```bash
kinescore pull --what bench                       # scored clips + manifest
kinescore pull --what train                       # training corpora

R=fourier_gr1.humanoid_sv.sv1_4x3
kinescore data  --reader $R                       # corpus → train tree
kinescore cache --reader $R --device cuda         # backbone → token cache
kinescore train --reader $R --device cuda         # → $KINESCORE_CKPT_DIR/$R.pt

kinescore score --cell humanoid.sv1_4x3.dreamdojo --device cuda
kinescore report --by role --out out/report.json
kinescore export --cell humanoid.sv1_4x3.dreamdojo --name dreamdojo_humanoid_sv
```

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for every flag, output formats,
and scoring clips outside the bench manifest.

## Readers

| reader | robot | view | scores |
|---|---|---|---|
| `airbot_mmk2.humanoid_mv.mv4_row` | Airbot MMK2 | 4×(320×192) row | `humanoid.mv4_row.ctrlworld_4view_grid` |
| `airbot_mmk2.humanoid_mv.mv4_row_static` | Airbot MMK2 | panels 0–1 of `mv4_row` | `humanoid.mv4_row_static.ctrlworld_4view_grid` |
| `airbot_mmk2.humanoid_mv.mv4_grid` | Airbot MMK2 | 2×2 of 384×216 | `humanoid.mv4_grid.dreamgen` |
| `fourier_gr1.humanoid_sv.sv1_16x9` | Fourier GR-1 | 768×432 | `humanoid.sv1_16x9.dreamgen` |
| `fourier_gr1.humanoid_sv.sv1_4x3` | Fourier GR-1 | 640×480 | `humanoid.sv1_4x3.dreamdojo`, fastercache |
| `a1x_ee.a1x_sv.sv1_4x3` | Galaxea A1X (EE pose) | 640×480 | `single_arm.sv1_4x3.radial_dreamgen` (radial, via `--videos`) |

`Synthetic2R` (closed-form, no URDF) is the CPU-only test fixture.

## Documentation

| | |
|---|---|
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | data, training, scoring, outputs |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | modules, the diffusion head, adding a robot |
| [`docs/METRICS.md`](docs/METRICS.md) | detectors, segments, calibration |
