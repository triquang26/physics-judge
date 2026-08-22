# `data/`

Ships empty. Point `KINESCORE_DATA_ROOT` at this folder (or any folder with
this shape) and nothing needs moving:

```bash
cp .env.example .env
# .env:  KINESCORE_DATA_ROOT=/absolute/path/to/this/repo/data
```

Everything below `data/` is gitignored except the placeholder directories. The
repo never carries video.

## What goes where

    video_gen_physics/            the benchmark corpus, as published
      <method>/<embodiment>/input/<view>/<model>/<split>/episode_*/
                                  real teleop: metadata.json + full_gt.mp4
      <method>/<embodiment>/output/<view>/<model>/<split>/...
                                  generated rollouts, the clips being judged
      data_for_web/catalog.json   index of every published clip

    train/<reader_id>/            written by `kinescore data`, never by hand
      videos/{train,val}/*.mp4
      annotation/{train,val}/*.json
      dataset_card.json
      run_manifest.json

    canonical/<cell_id>/          clips to score, if you stage them here
                                  instead of passing `--videos`

`reader_id` is `<robot>.<view_id>` and `cell_id` is
`<embodiment>.<view_id>.<model>`; both are declared in `configs/cells.yaml`.
