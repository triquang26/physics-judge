# Bringing kinescore up on a new machine

Everything except two items regenerates from a public link. Those two are what
has to travel with you.

## Carry these

| item | size | why it cannot be re-fetched by link |
|---|---|---|
| **HF token with DINOv3 access** | — | `facebook/dinov3-vitl16-pretrain-lvd1689m` is `gated: manual`. A token alone is not enough — the account behind it must have been **approved by hand**. Request access before you need the machine, not on the day. |
| **the asset tree** (`$KINESCORE_ASSETS`) | 69 MB | Hand-built and gitignored (`assets/`). `MANIFEST.json` records each robot's upstream URL, commit and URDF sha256, so it *can* be rebuilt from four separate repos — carrying the 40 MB is faster. A copy is in the bucket under `assets/`, with `gr1/`'s symlinks materialised so it lands intact on any filesystem. |

A third item is conditional: `$ROBOT_DESCRIPTIONS_CACHE`. Franka's spec resolves
its URDF through the `robot_descriptions` package, which **git-clones from GitHub
on first use** if that cache is empty. The readers on record are Airbot MMK2, whose
URDF is in the asset tree, so this only bites once a Franka cell is scored — and
then only as a one-time download, not a blocker.

Everything else — repo, data, checkpoints — comes from a link:

- code: <https://github.com/triquang26/physics-judge>
- data: `kinescore pull` (declared in `configs/sources.yaml`, no links to paste)
- checkpoints and scores: `hf://buckets/twanghcmut/video-bench-model`

## Machine requirements

- **~360 GB free disk per reader you intend to train.** The feature cache for
  `airbot_mmk2.humanoid_mv.mv4_row` alone is 337 GB (248 episodes). Data is
  another 19 GB (bench 2.1, corpus 15, trees 1.6).
- One CUDA GPU. Reference host is an H100 80 GB; the runs on record used a
  3g.40gb MIG slice of one. **One GPU process at a time** — the cache and train
  stages each expect the whole slice.
- Time: building that cache spanned **~22 h** on the reference host (first to
  last cache file). It is the dominant cost of the whole pipeline and is
  read-bound on NFS, not compute-bound.

## Steps

```bash
git clone https://github.com/triquang26/physics-judge.git kinescore && cd kinescore
python -m venv .venv && .venv/bin/pip install -e '.[dino,video,bench,dev]'

cp .env.example .env          # then fill in all five paths
.venv/bin/python -m pytest tests/ -q     # CPU-only, no network, no checkpoint

export HF_TOKEN=...           # the DINOv3-approved token; do not `hf auth login`
hf sync hf://buckets/twanghcmut/video-bench-model ./bundle
cp -r bundle/assets/*      "$KINESCORE_ASSETS"/
cp    bundle/checkpoints/* "$KINESCORE_CKPT_DIR"/

kinescore pull --what all               # bench + corpus, revisions pinned
kinescore data   --reader airbot_mmk2.humanoid_mv.mv4_row
kinescore cache  --reader airbot_mmk2.humanoid_mv.mv4_row   # the 22 h step
kinescore train  --reader airbot_mmk2.humanoid_mv.mv4_row --head diffusion
kinescore score  --cell humanoid.mv4_row.ctrlworld_4view_grid --device cuda --frame-chunk 16
kinescore ledger                        # what is built, trained, scored
```

To score only — the common case, since a checkpoint is in the bucket — skip
`data`/`cache`/`train`. Scoring reads bench clips directly and needs no feature
cache, so the 337 GB and the 22 h do not apply.

`pull` re-pulls at the revision already recorded in `REVISIONS.json` unless
`--revision` names another, so running it twice gives the same bytes twice.

`score` renders its own clips as its last step: `out/<cell>/render/reel.mp4` is
every scored clip with its violation timeline drawn underneath.

## Keeping the token out of shared config

The reference host is shared. `hf auth login` writes `$HF_HOME/token`, and
`git config --global credential.helper` writes `~/.gitconfig` — on a shared
account both collide with whoever else is using the machine. Pass credentials
per-invocation instead:

```bash
HF_TOKEN=hf_... hf sync ./bundle hf://buckets/twanghcmut/video-bench-model
git config --local credential.helper 'store --file=.git/credentials'   # chmod 600
```

## What is not in the bucket

The 337 GB feature cache and the 19 GB of pulled data. Both regenerate; neither
is worth moving.
