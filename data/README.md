# `data/`

Ships empty. Point `KINESCORE_DATA_ROOT` at this folder (or any folder) and
`kinescore pull` fills it:

```bash
cp .env.example .env
# .env:  KINESCORE_DATA_ROOT=/absolute/path/to/this/repo/data
kinescore pull --what all
```

Everything below `data/` is gitignored except the placeholder directories. The
repo never carries video.

## What goes where

    bench/                        `kinescore pull --what bench`
      clips/<id>.mp4              the 400 scored clips
      manifest.json               what each clip is: embodiment, view, model,
                                  split, method, role, task
      {augment,dense,worldcache}/ the same clips in their source tree

    corpus/                       `kinescore pull --what train`
      <embodiment>/<view>/[<task>/]meta/info.json
                                  LeRobot v2, one directory per dataset
                                  data/chunk-NNN/episode_*.parquet
                                  videos/chunk-NNN/observation.images.<cam>/

    trees/<reader_id>/            written by `kinescore data`, never by hand
      videos/{train,val}/*.mp4
      annotation/{train,val}/*.json
      dataset_card.json
      run_manifest.json

    REVISIONS.json                repo, revision and file count per pull

`reader_id` is `<robot>.<corpus>.<view_id>` and `cell_id` is
`<embodiment>.<view_id>.<model>`; both are declared in `configs/cells.yaml`.
