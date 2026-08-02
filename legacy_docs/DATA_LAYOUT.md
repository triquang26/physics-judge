# Data layout: download, drop in, run

The contract is: **whatever you download keeps its own name, and goes under one
root.** No renaming, no re-nesting, no moving files between trees. If a step
here asks you to reorganise a download, that is a bug in this document.

Everything resolves through `KINESCORE_*` environment variables, which have
**no fallbacks** by design (`src/kinescore/paths.py`) — an unset variable raises
and names itself rather than silently resolving to a path that exists on
someone else's machine. Copy `.env.example` to `.env` and fill it in.

---

## The five roots, and what each is for

| variable | holds | regenerable? | typical size |
|---|---|---|---|
| `KINESCORE_DATA_ROOT` | video to score, and video to train readers on | **no** — re-download | ~17 GB |
| `KINESCORE_ASSETS` | URDF + meshes per robot, `MANIFEST.json` | no — re-fetch from upstream | ~40 MB |
| `KINESCORE_CKPT_DIR` | trained pose readers + provenance | no — retrain or re-copy | ~60 MB |
| `KINESCORE_CACHE_DIR` | precomputed frozen-backbone features | **yes — delete freely** | **~56 GB** |
| `KINESCORE_OUTPUT_DIR` | manifests, `results.jsonl`, CSVs, traces | yes — rerun | small |

The cache is by far the largest and is **entirely derived**. Do not back it up,
do not copy it between machines, and delete it without hesitation when disk is
tight — `kinescore cache` rebuilds it. It is separated from everything else
precisely so that sentence can be true.

---

## `KINESCORE_DATA_ROOT`

Subdirectories are named **exactly** after their HuggingFace repos, so
`huggingface-cli download <repo> --local-dir $KINESCORE_DATA_ROOT/<repo-name>`
or `kinescore data pull` lands them in the right place with nothing to move.

```
$KINESCORE_DATA_ROOT/
├── video_gen_physics/               # doanh25032004/video_gen_physics        13 GB
│   └── dense/                       #   ONLY dense/ is pulled; the other 8
│       ├── humanoid/                #   top-level dirs are cache-accelerator
│       └── single_arm/              #   variants and are hundreds of GB
├── video_gen_physics_real_video/    # doanh25032004/video_gen_physics_real_video
│   └── humanoid/                    #   NOTE: this is Airbot MMK2, not GR-1.
│                                    #   NOTICE_NOT_GR1_REFERENCE.txt says so.
├── cosmos_synthetic_data/           # doanh25032004/cosmos_synthetic_data    144 MB
│   ├── high/                        #   human-labelled tiers -- the construct-
│   └── low/                         #   validity set for the score itself
├── train/                           # reader training sets, one dir per robot
│   ├── franka_droid_std/            #   annotation/{train,val}/*.json + videos/
│   ├── gr1_teleop/                  #   LeRobot v2 (PhysicalAI GR-1 Teleop)
│   └── airbot_mmk2/                 #   LeRobot v2
└── fps_probe.csv                    # generated: probed fps/resolution per clip
```

### Why `train/` is inside the data root

It is not, today. `KINESCORE_DROID_STD_DIR` and `KINESCORE_TELEOP_GR1_DIR`
currently point **into sibling research checkouts** on the machine this was
built on. That works here and works nowhere else: a fresh clone has neither
path, and nothing tells the newcomer what those trees should contain.

Worse, it is the failure this repo already documented once —
`provenance/never_copy.txt` records two dataset symlinks into a sibling
checkout that are now **dangling**, because that sibling moved. The original
DROID feature cache the Franka reader was first trained on is gone for exactly
this reason.

**Migration** (one-time, safe to run twice):

```bash
mkdir -p "$KINESCORE_DATA_ROOT/train"
cp -r "$KINESCORE_DROID_STD_DIR"/{annotation,videos} \
      "$KINESCORE_DATA_ROOT/train/franka_droid_std/"      # ~450 MB
cp -r "$KINESCORE_TELEOP_GR1_DIR"  "$KINESCORE_DATA_ROOT/train/gr1_teleop"   # ~1.3 GB
```

Copy, do not symlink — a symlink reintroduces exactly the dependency being
removed, and `tools/check_repo_hygiene.py` rejects symlinks in the tracked tree
for the same reason. After migrating, drop `KINESCORE_DROID_STD_DIR` and
`KINESCORE_TELEOP_GR1_DIR` from `.env`; they become derivable from
`KINESCORE_DATA_ROOT`.

Total cost is under 2 GB against a 17 GB data root, and it buys a tree that a
newcomer can populate from public sources alone.

---

## `KINESCORE_ASSETS`

```
$KINESCORE_ASSETS/
├── MANIFEST.json          # per robot: urdf relpath, sha256, license, upstream URL, date
├── README.md
├── gr1/urdf/ meshes/ LICENSE          # GPL-3.0 · FFTAI/Wiki-GRx-Models
├── franka/                            # BSD-3 · documentation copy only, see below
├── airbot_mmk2/urdf/ src/ LICENSE     # MIT · DISCOVERSE, assembled from 2 upstream URDFs
└── aloha/                             # prepared, not wired in
```

**Franka does not load from here.** It resolves through the `robot_descriptions`
pip package into `$ROBOT_DESCRIPTIONS_CACHE`
(`robots/urdf.py::resolve_robot_description_urdf`), because its URDF is a few MB
while GR-1's mesh tree is ~285 MB and would bloat every Franka-only clone. The
copy under `franka/` exists so the manifest can record its provenance.
`robot_descriptions` fetches on **first use** — a network dependency at runtime.
Warm it once (`kinescore doctor` does) before relying on an offline run.

Every `RobotSpec` records `urdf_sha256`, and that hash lands on every result
record — so "two runs disagree" becomes a `diff` rather than a mystery.

---

## `KINESCORE_CKPT_DIR`

One reader per robot, named after the robot, each with its provenance beside it.

```
$KINESCORE_CKPT_DIR/
├── humanoid.pt              + .provenance.json     GR-1,   raw_rad, val 19.19 mm
├── airbot_mmk2.pt           + .provenance.json     Airbot, val 18.31 mm  + _NOTICE.txt
├── single_arm_rawrad.pt     + .provenance.json     Franka, raw_rad
└── single_arm.pt            legacy judge_v3l, SQUASHED -- control only, not for scoring
```

Read the `_NOTICE.txt` files. They record things that are invisible in the
weights and would silently corrupt a downstream reading — e.g. `airbot_mmk2.pt`
has a 13th "gripper" output channel that is a **constant-0 placeholder**,
because that robot has no gripper channel. Anything treating channel 13 as a
real gripper opening reads "closed" on every frame.

---

## Getting the data

```bash
cp .env.example .env && $EDITOR .env          # fill in the five roots
kinescore doctor                              # what is present, what is missing
kinescore data pull --config configs/benchmark.yaml --dry-run   # patterns only
kinescore data pull --config configs/benchmark.yaml
```

`--dry-run` first, always. The download patterns are generated from the matrix in
the config, and the guard matters: `video_gen_physics` has nine top-level
directories, of which `dense/` is one. Pulling them all is hundreds of GB.

Set `HF_HUB_DISABLE_XET=1`. Measured on this host, the xet CDN throttled to
374 KB/s with repeated 429s from the xet-read-token endpoint; disabling it gave
**2893 KB/s**, 7.7x faster. `HF_HUB_ENABLE_HF_TRANSFER` is a **no-op** on
`huggingface_hub` 1.25.1 and warns as much — it is not the fix.

---

## A caution about the tree's own labels

Directory names in the downloaded data are not always what they say:

- `dense/humanoid/.../multiview/ctrlworld/` is **Airbot MMK2**, not GR-1.
- `singleview` and `single_view` both exist, with **different** episode sets.
- `_fps_compare/` sits where a horizon directory would; anything `_`-prefixed is
  not an episode.
- `single_arm/multiview/makovian/meta/info.json` declares `total_episodes: 95600`
  and `fps: 15` for a subtree of 2,583 files, and no clip probes at 15 fps.

Treat every declared value as a **cross-check that can only fail**, never as a
source of truth — that is why `resolve_timebase` always probes. See
[RATE_POLICY.md](RATE_POLICY.md) for the measured cost of getting this wrong.
