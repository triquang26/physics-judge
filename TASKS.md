# Task board

Live state of the benchmark build. Updated as things land — if something here
disagrees with the code, the code is right and this file is stale; say so.

Legend: **DONE** · **RUNNING** · **BLOCKED** · **TODO**

---

## 2026-08-06 — everything below this block predates the direct-keypoint switch

The board from "The three readers" down describes the **joint-angle
(`raw_rad`) era**. Those checkpoints no longer exist: `humanoid.pt`,
`airbot_mmk2_rawrad.pt`, `single_arm_rawrad.pt`, `airbot_mmk2.pt` and every
`*_ctrlworld_rawrad.pt` / `*_singleview_rawrad.pt` were deleted from
`$KINESCORE_CKPT_DIR`, so tasks 1-8 below cannot be re-run as written and
their val-mm numbers (19.19 / 19.52 / 162.10 / 56.66) describe files that are
gone. Kept as the decision record, not as instructions.

Current state — five direct-keypoint readers, all on disk, all uploaded to
the `twanghcmut/result-video-bench` bucket under `direct_keypoint/`:

| reader | K | val mm | cells scored |
|---|---|---|---|
| `airbot_mmk2_ctrlworld_kp.pt` | 12 | 11.31 | `humanoid/**/multiview/ctrlworld/{mak,non}` |
| `franka_panda_ctrlworld_kp.pt` | 8 | 23.22 | `single_arm/**/multiview/ctrlworld/{mak,non}` |
| `aloha_bimanual_ctrlworld_kp.pt` | 18 | 23.15 | `bimanual/**/multiview/ctrlworld/{mak,non}` |
| `fourier_gr1_singleview_kp.pt` | 12 | 45.31 | `humanoid/**/singleview/{dreamdojo,dreamgen}/{mak,non}` |
| `aloha_bimanual_singleview_kp.pt` | 18 | 73.69 | `bimanual/**/singleview/{dreamdojo,dreamgen}/{mak,non}` |

The singleview half is scored by `scripts/score_singleview_direct.py` (this
repo); the ctrlworld half was scored by an ad-hoc driver that was never
committed. Live status per cell lives in the bucket's
`direct_keypoint/STATUS.md`, not here.

Two things this switch left open, neither of them a code bug:

* `configs/benchmark.yaml`'s `robots:` block still names the deleted
  joint-angle checkpoints and therefore cannot load. It is NOT mechanically
  repinnable — the direct-keypoint checkpoints are keyed by (robot, domain)
  and that schema has one reader slot per robot with no view axis. See the
  warning block above `robots:` in that file.
* The FK-based 31-metric suite has no reader at all now: keypoint
  checkpoints carry `limit_semantics="keypoints"`, so `q`/`q_raw` are `None`
  and every joint-space metric is unavailable by construction. The five
  `kinescore.violations` detectors are the live scoring path.

---

## The three readers, and who trained what

The goal is one reader per robot, then score. Two of the three were **not**
trained here, which matters when reading any number they produce.

| robot | checkpoint | trained here? | training data | val keypoint mm |
|---|---|---|---|---|
| GR-1 | `humanoid.pt` (27 MB) | **no** — copied from `Marionette-fkjepa/model_ckpt/readout_v2_gr1.pt` | PhysicalAI GR-1 Teleop | **19.19** — accepted |
| Airbot MMK2 | `airbot_mmk2_rawrad.pt` (raw_rad; supersedes the squashed `airbot_mmk2.pt` row this table previously showed, val 18.31/train 6.55, same data) | **yes** — 6000 steps, `train-rawrad`, best_step 5500 | real LeRobot Airbot, 223 train / 25 val, full-arm camera (`cam_third_view`/`cam_front_rgb`) | **19.52** (train 8.40) — accepted. `configs/benchmark.yaml` still pins the old squashed `airbot_mmk2.pt`; update the reader pin before running a raw_rad-metrics benchmark against this robot, see `docs/TRAINING.md` |
| Franka | `single_arm_rawrad.pt` | trained, gate checked | `droid_std`, 396 train / 4 val | **162.10** — **rejected**, ~3x the accepted band (see "Franka reader: failed its gate, and why" below). `configs/benchmark.yaml` still pins the deleted squashed `single_arm.pt`, which no longer loads at all |

`single_arm.pt` / `single_arm_mv.pt` were the **squashed** `judge_v3l` copies.
**Deleted (2026-07-29)** from `$KINESCORE_CKPT_DIR`, along with the squashed
pose-reader path itself (`readers/squashed.py`, `heads/ranges.py::squash_to_limits`,
`training/trainer.py`, the `kinescore train` subcommand) -- see
`legacy_docs/PROVENANCE.md`'s D7 addendum. They are no longer loadable at all
(`readers/checkpoint.py::load_reader` now raises `NotImplementedError` for
that checkpoint format) -- this is intended, not a regression, and the same
backbone/checkpoint-mismatch failure noted below is now moot for them since
there is no reader left to attempt it.

Having three readers does **not** imply three sets of results — see Blocked.

## Now

| # | task | state | note |
|---|---|---|---|
| 1 | Score GR-1 singleview, `invariant_v1` | **DONE** | `out/dense_humanoid_singleview_run1/` — **2007 clips, 0 failed** |
| 2 | Same, all-metrics suite (28 metrics: `invariant_v1` + torque + worst-bone rigidity) | **DONE** | `.../full_suite/results.jsonl` — **2007/2007 GR-1 clips, 0 failed**, scored under the suite named `full` (`sha256:d346cf8f84a45742`) *before* the `full`→`all_metrics` rename landed. Cross-checked byte-for-byte against task 1's `invariant_v1` run: 2007 clips × 26 shared metrics = **52,182 comparisons, 0 mismatches**. Franka/single_arm (599 clips) scored under the *current* name `all_metrics` (`sha256:b6924a162403ca8d`) and **100% failed** — see "Bugs found, not fixed" below; this is a real reader bug, not a suite-naming artifact |
| 3 | Per-folder CSV tree mirroring the data layout (`kinescore export`, `bench/csv_export.py` + `cli/cmd_export.py`) | **DONE** | `out/dense_humanoid_singleview_run1/dense/<emb>/<view>/<gen>/<horizon>/clips.csv` (9 cells, 3102 rows) + `SUMMARY.csv`. Sorted worst-first by `mean_jerk_mps3`, empty cell + `<metric>_reason` for anything unavailable (never `0`), `suite_id`/`suite_name` columns, headline-rigidity + fps-comparability notes in the leading `#` comment. 34 tests in `tests/test_csv_export.py`. ctrlworld/Airbot-MMK2 cells (496 clips) shipped as explicit `status=skipped` rows rather than an absent file |
| 4 | **Per-frame traces exported, not just scalars** | **RUNNING** | needs one further scoring run to capture them — see below |
| 5 | Franka `raw_rad` reader (drop the squash) | **RUNNING** | `train-rawrad` live; 44 GB DINO cache built. Gate = val mm vs squashed 57.38 |
| 6 | Airbot MMK2 reader on `cam_high` | **RUNNING** | split A (142 ep, `cam_head_rgb`) cached + a train-rawrad run started, but its log shows no completion — treat as unverified, not a second accepted reader; split B (106 ep, `cam_high_rgb`) not even cached yet. Scripts moved to `scripts/cache_airbot_mmk2_camhigh_subset{A,B}.sh` / `scripts/train_airbot_mmk2_camhigh_subsetA_rawrad.sh`, see their headers |
| 7 | Camera-visibility gate per cell | **RUNNING** | must report a *proportion* per cell, not a verdict |

## Blocked — and the blocker is not the reader

Both remaining embodiments are blocked on **whether the camera shows the arm
at all**, not on model quality. A reader cannot read joint angles out of a
frame that does not contain the arm, and if asked to, it returns plausible
numbers that sort and plot and mean nothing.

| # | task | why |
|---|---|---|
| 8 | Score `dense/humanoid/**/multiview/ctrlworld/` | It is **Airbot MMK2**, not GR-1. Its rendered cameras are `cam_high` + 2 wrist; `cam_high` shows a bare table for one task subset and a partial forearm for the other. The 18.31 mm reader was trained on a wide view the generated data **never renders**. Waiting on task 6's per-subset val mm |
| 9 | Score `dense/single_arm/**/singleview/` | **MEASURED: only 52.4% is scoreable.** Of 1209 singleview clips, **633 exterior / 576 wrist (47.6%)** — nearly half have no arm in shot. Varies *per cell*, not per generator. Curated 633-clip manifest already built (see below) — do not redo |

Decision rule for both: if val mm on the camera the generated data *actually
has* lands in a usable range, score and record the viewpoint. If not, report
the cell as unscoreable **with the measured mm as the evidence**. A
well-evidenced "cannot be scored, here is why" is a complete result.

## Bugs found while scoring (out of my scope — reported, not fixed)

Both found while building the singleview GR-1 + Franka `all_metrics` run
above. Neither is in `bench/csv_export.py`/`cli/cmd_export.py`, so neither
was touched here; the CSV layer routed around both without editing the
files where they actually live.

1. **Franka/`single_arm.pt` reader: backbone/checkpoint mismatch, 100%
   failure.** All 599 `single_arm` clips (`franka_panda` + `single_arm.pt`)
   failed with `RuntimeError: Given normalized_shape=[1024], ... got input
   of size[1, T, 256, 768]`. The checkpoint's own `cfg` declares
   `dino_model: dinov3_vitl16` (`embed_dim: 1024`,
   `hf_model_id: facebook/dinov3-vitl16-pretrain-lvd1689m`), but the loaded
   backbone is DINOv2 ViT-B/14 (768-dim) — confirmed via
   `torch.hub` fetching `dinov2_vitb14_pretrain.pth` at load time, not a
   DINOv3 weight at all. This is in `readers/checkpoint.py`'s backbone
   routing (out of my scope, forbidden file). Reproduced directly:
   ```
   reader = checkpoint.load_reader("single_arm.pt", robot=franka, ...)
   reader.read(frames)  # -> RuntimeError at heads/attentive.py:108 self.norm(feat)
   ```
   The CSVs still ship real, honest rows for this cell — `status=failed`,
   `failure_reason` carrying this exact message, every metric cell empty —
   rather than an absent file or fabricated numbers.
2. **`configs/benchmark.yaml`'s `fps_expected.dreamdojo = 10.0` is stale for
   `single_arm`.** It is correct for GR-1 (dreamdojo really is 10fps there)
   but `single_arm`'s dreamdojo clips probe at **15fps** — every one of 480
   `single_arm`/dreamdojo episodes was silently dropped to 0 rows by the D3
   safety check (`kinescore.video.probe.resolve_timebase`) when built the
   normal way (`kinescore bench run --config configs/benchmark.yaml`).
   Worked around here by discovering `single_arm`/dreamdojo directly
   (`kinescore.bench.manifest.build_manifest` with a local plugin, no
   `fps_hint`) rather than editing the shared config — a one-line
   `fps_expected` fix or a per-embodiment table is the real fix, left to
   whoever owns `configs/`.

## Todo

| # | task | note |
|---|---|---|
| 9 | **Move training data under `KINESCORE_DATA_ROOT/train/`** | `.env` currently points `KINESCORE_DROID_STD_DIR` / `KINESCORE_TELEOP_GR1_DIR` **into sibling research checkouts** — a fresh clone has neither. This is the failure `provenance/never_copy.txt` already records: two dataset symlinks into a sibling that are now dangling, taking the original DROID feature cache with them. Fix is a `cp` of <2 GB, or run `scripts/convert_lerobot_to_train.py` directly against the sibling checkouts; see `docs/DATA_PREP.md`. **Asked, awaiting go-ahead** (touches `.env`) |
| 10 | `use_context=False` ablation | measures how much jerk the temporal transformer smooths away. One flag; `TemporalEncoder.forward` already supports it. Offered, not yet approved |
| 10 | Clear 271 pre-existing ruff errors | baseline debt (`UP006`/`UP035`), 220 auto-fixable. Not a regression. Deferred until agents release files |
| 11 | Bimanual / ALOHA | `aloha_bimanual` robot is now registered and wired into `configs/benchmark.yaml`'s `axes.robot` (was deferred; no longer). Still open: no trained `aloha_bimanual` reader checkpoint exists (config pins a nonexistent `bimanual.pt`), and `dense/bimanual/output/**` real generated data (~150 episodes, 5 branches) is not downloaded locally yet — see `docs/BENCHMARKING.md`. Verification handoff at `legacy_docs/ADDING_ALOHA_NOTES.md`; the confirmed (not hypothesised) joint layout is in `src/kinescore/robots/aloha/constants.py`'s own docstring |

---

## Why task 3 matters

`MetricValue.perframe` exists and `MetricSuite.perframe` collects it, but
**nothing persists it** — `core/scorer.py` and `bench/store.py` contain zero
references. The arrays are computed and discarded. Only `mean_jerk_mps3` even
declares `perframe=True`; every other ruler reduces to a scalar internally.

Consequence: the project page's own galleries — jerk over time, per-frame
rigidity deviation against a tolerance band — **cannot be reproduced from our
output**. A clip that is smooth for 40 frames then teleports once reports the
same mean as one that judders throughout.

Storage is not the obstacle: 2007 clips x ~50 frames x ~5 arrays x 4 bytes is
about **2 MB**.

---

## Done

| task | evidence |
|---|---|
| Benchmark matrix + YAML config + per-generator source plugins | 76 tests, N/A cells declared, `allow_patterns` from the matrix |
| Separation / report / `rank` CLI | AUROC orientated by each metric's declared `direction` |
| Rate policy + `RATE_FREE` suite | 9 metrics, membership derived from the registry not hand-listed |
| Full asset tree | `MANIFEST.json` with sha256 + license + upstream per robot |
| Torque inverse-dynamics port | reproduces published 12.0109% / 14.2441% to 5 s.f. |
| Data pulled | 16,163 files / 12.86 GB (`dense/` humanoid + single_arm) |
| Config pinned to real `iter_` dirs | `iter_000113000` did not exist anywhere; dreamdojo namespaces de-collided |
| `rigidity_worst_bone_mm` | ported ruler separated at 56.6% (chance); published one uses own-median + `amax`. `legacy_docs/DECISIONS.md` D-A |
| 3 integration bugs fixed | frame shape, device placement, `ViewEmbedding` device — scoring had never run end-to-end |
| GR-1 scored, `invariant_v1` | **2007 clips, 0 failed**; +27.4% jerk, generated worse in 89.8% of 659 paired episodes |
| GR-1 scored, 28-metric all-metrics suite + CSV tree (`out/dense_humanoid_singleview_run1/`) | medians (worst-first sort key `mean_jerk_mps3`), n=n_ok per cell:<br>`dreamdojo/makovian` n=1551 — jerk 17.14, rigidity_residual 1.61, rigidity_**worst_bone** 3.44, sparc −5.42, LDLJ −15.66, limit_violation 0, torque 7.36%<br>`dreamdojo/non_makovian` n=196 — jerk 18.39, rigidity_residual 1.85, rigidity_worst_bone 2.21, sparc −5.70, LDLJ −15.76, limit_violation 0, torque 7.07%<br>`dreamgen/makovian` n=130 — jerk 68.15, rigidity_residual 0.74, rigidity_worst_bone 1.18, sparc −3.26, LDLJ −15.90, limit_violation 0, torque 5.84%<br>`dreamgen/non_makovian` n=130 — jerk 77.52, rigidity_residual 1.44, rigidity_worst_bone 1.65, sparc −3.32, LDLJ −15.91, limit_violation 0, torque 7.26%<br>dreamgen's ~4x higher jerk vs dreamdojo is **fps, not physics**: `(16/10)**3 = 4.096` (`dt_exponent=3`) — corrected medians land at 16.6/18.9, back in dreamdojo's band. `limit_violation_frac=0` across every cell is a real, `raw_rad` reading (never squashed to 0 — see D7), not a metric that cannot fire |

## Reference

- `legacy_docs/DECISIONS.md` — every design choice with the measurement that settled it
- `legacy_docs/PROVENANCE.md` — what was ported, and defects D1-D12
- `docs/BENCHMARKING.md` — the frame-rate rules that make a comparison valid
- `docs/{DATA_PREP,TRAINING,BENCHMARKING,ARCHITECTURE,METRICS}.md` — the current, maintained docs; read these first
- `configs/benchmark.yaml` — the reproducibility contract for a run

---

## Camera visibility — measured, per cell

The gate "does the camera actually show the arm" is now answered with numbers
rather than judgement. A whole-frame Farneback optical-flow discriminator
(`moving_frac` = fraction of pixels with flow magnitude > 0.5 between
consecutive sampled frames; a wrist/ego camera translates the entire scene, a
fixed exterior one does not) was validated against **40 hand-eyeballed clips
across 13 groups — 100% agreement**, threshold `moving_frac >= 0.42`.

**`dense/single_arm/**/singleview/` — 1209 clips: 633 exterior (52.4%) / 576 wrist (47.6%).**

It varies by *cell*, not by generator, so no per-generator rule would have caught it:

| cell | verdict |
|---|---|
| `dreamdojo/makovian/iter_000060000` | 100% wrist |
| `dreamgen/makovian/iter_000070000` | 100% wrist |
| `dreamdojo/makovian/iter_000090000` | 246/250 wrist |
| `dreamdojo/non_makovian/iter_000100000` | 256/260 wrist |
| `dreamdojo/makovian/iter_000030000` | 240/240 exterior |
| `dreamdojo/non_makovian/iter_000030000` | 233/240 exterior |
| every `dreamgen/*_static16fps/global` | exterior |

**Multiview is not scoreable without panel cropping first:**

- `multiview/ctrlworld` — 771 files, 960x192, a 3-panel **width** stack
  (`exterior_1 | exterior_2 | wrist`), visually confirmed. `ViewLayout.view_height()`
  splits on **height**, and `192 % 3 == 0`, so `n_views=3` would silently slice
  three wrong 64px horizontal bands **with no error**. Crop columns `0:320` and
  `320:640` for the two exterior panels.
- `multiview/dreamgen` — 257 files, 768x432, a **2x2 grid**.

Prebuilt, reusable — do not redo this classification:
```
<scratchpad>/single_arm_scoreable_manifest.json/bench_manifest.parquet
    633 exterior singleview clips · 239 gt/pred pairs · 0 pairing mismatches
<scratchpad>/per_clip_verdicts.json
    per-clip exterior/wrist verdict
```

Caveat carried forward: the discriminator was validated on 40 clips over 13
groups. That is good, not exhaustive. Re-validate on a sample before trusting it
on cells outside `single_arm/singleview`.

## Franka reader: failed its gate, and why

```
train  18.74 mm        better than GR-1's production reader (19.19)
val   162.10 mm        2.8x the retired squashed baseline (57.38)
```

Per-episode, on the four held-out episodes:

| episode | frames | mm |
|---|---|---|
| 99 | 37 | **365.43** |
| 199 | 33 | 73.20 |
| 299 | 16 | 74.18 |
| 399 | 113 | 133.93 |

It is **not** "three fine, one broken". Episode 99 is at the untrained baseline
(359.93) and dominates the mean, **and** the other three sit 1.3–2.3x above
57.38 — excluding 99 entirely still leaves ~94 mm. Both statements are true;
neither cancels the other.

Train/val leak: **checked and clean** at both the annotation-file and cache-file
level. The 18.74 mm train figure is genuine.

Root cause is the data, not the recipe: `droid_std` holds **396 train / 4
val** — every hundredth episode — while **23,887 raw DROID episodes** sit on
this host under `oscar-droid/droid_raw/` (643 GB). We trained on **1.7%** of
what was available and then judged the reader on four episodes. Four cannot
estimate a mean over a corpus where every episode is a different kitchen or lab.

In progress: converting a larger subset with the existing resumable converter
(~360 episodes/hour measured), splitting ~85/15 **stratified by scene** (the
DROID directory prefix before `__` looks like a lab id — a random split leaks
the scene into training), then retraining on the unchanged GR-1 recipe.
