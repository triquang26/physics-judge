"""Adapter output -> the canonical train tree."""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from kinescore.adapters.base import RawEpisode, SkippedEpisode, get_adapter
from kinescore.registry.cells import ReaderSpec
from kinescore.registry.views import ViewSpec
from kinescore.training.splits import stratified_episode_split
from kinescore.video.probe import ffprobe

__all__ = ["MaterializeReport", "materialize_train_tree"]

JOINT_KEY = "observation.state.joint_position"
GRIPPER_KEY = "observation.state.gripper_position"

#: How many skipped episodes the dataset card lists in full.
_MAX_SKIP_EXAMPLES = 20


@dataclass
class MaterializeReport:
    """What one materialisation wrote.

    Attributes
    ----------
    reader_id, tree:
        Which reader this supervises, and where the tree landed.
    n_train, n_val:
        Episodes written per split.
    skipped:
        ``(episode_id, reason)`` per episode the adapter could not use, plus
        any this writer rejected.
    """

    reader_id: str
    tree: str
    n_train: int = 0
    n_val: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def n_written(self) -> int:
        return self.n_train + self.n_val


def _pack_filter(view: ViewSpec, n: int) -> str:
    """ffmpeg ``filter_complex`` packing ``n`` scaled inputs into one frame."""
    scale = ""
    if view.panel is not None:
        w, h = view.panel
        scale = "".join(f"[{i}:v]scale={w}:{h}[s{i}];" for i in range(n))
    src = "".join(f"[s{i}]" if scale else f"[{i}:v]" for i in range(n))
    if view.packing == "width":
        if n == view.panel_count:
            return f"{scale}{src}hstack=inputs={n}[out]"
        if view.panel is None:
            raise ValueError(
                f"view {view.view_id!r} exposes {n} of {view.panel_count} "
                f"panels, which needs a `panel` size to place them")
        # A dropped panel is black here and never read: the crop takes each
        # exposed panel back out at the index the clips this view scores carry
        # it at, so the tree and the bench are one geometry.
        w, h = view.panel
        parts: list[str] = []
        pending = (f"[s0]pad={w * view.panel_count}:{h}:"
                   f"{w * view.panel_indices[0]}:0:black")
        for i, panel in enumerate(view.panel_indices[1:], 1):
            parts.append(f"{pending}[b{i}];")
            pending = f"[b{i}][s{i}]overlay=x={w * panel}:y=0"
        return f"{scale}{''.join(parts)}{pending}[out]"
    if view.packing == "height":
        return f"{scale}{src}vstack=inputs={n}[out]"
    if view.packing == "grid2x2":
        if view.panel is None:
            raise ValueError(
                f"view {view.view_id!r} packs a grid, which needs a `panel` "
                f"size to place the cells")
        w, h = view.panel
        if n == 4:
            return f"{scale}{src}xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[out]"
        if n == 3:
            # The fourth cell is blank in the clips this view scores. ffmpeg
            # 4.x has no `fill` on xstack, so the last row is padded instead.
            return (f"{scale}[s0][s1]hstack=inputs=2[top];"
                    f"[s2]pad={2 * w}:{h}:0:0:black[bottom];"
                    f"[top][bottom]vstack=inputs=2[out]")
        # A dropped cell is black here and never read, exactly as in the
        # width subset: each exposed panel sits at its own grid cell, so the
        # tree and the clips this view scores are one geometry.
        parts = []
        r0, c0 = divmod(view.panel_indices[0], 2)
        pending = f"[s0]pad={2 * w}:{2 * h}:{c0 * w}:{r0 * h}:black"
        for i, panel in enumerate(view.panel_indices[1:], 1):
            parts.append(f"{pending}[b{i}];")
            r, c = divmod(panel, 2)
            pending = f"[b{i}][s{i}]overlay=x={c * w}:y={r * h}"
        return f"{scale}{''.join(parts)}{pending}[out]"
    if view.packing == "none":
        # Nothing to stack, but the frame still has to reach the declared size:
        # the corpus camera and the clips this head scores differ in resolution.
        if n != 1:
            raise ValueError(
                f"view {view.view_id!r} is a single panel but {n} cameras "
                f"were given")
        if not scale:
            raise ValueError(
                f"view {view.view_id!r} declares no `panel` size, so a raw "
                f"camera cannot be resized to it")
        return f"{scale}[s0]null[out]"
    raise ValueError(
        f"view {view.view_id!r} has unknown packing {view.packing!r}")


def _pack(episode: RawEpisode, view: ViewSpec, dest: Path) -> None:
    """Write ``episode``'s per-camera files into one frame, packed per ``view``.

    Raises
    ------
    ValueError
        If the episode has fewer cameras than the view exposes.
    subprocess.CalledProcessError
        If ffmpeg fails.
    """
    # Insertion order is panel order, set by the adapter from `cameras:`.
    names = list(episode.views)
    wanted = list(view.panel_indices)
    if max(wanted) >= len(names):
        raise ValueError(
            f"view {view.view_id!r} exposes panel {max(wanted)} but episode "
            f"{episode.episode_id!r} has {len(names)} camera(s): {names}")
    inputs: list[str] = []
    for i in wanted:
        inputs += ["-i", episode.views[names[i]]]
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *inputs,
         "-filter_complex", _pack_filter(view, len(wanted)),
         "-map", "[out]", "-pix_fmt", "yuv420p", str(dest)],
        check=True, capture_output=True)


def _link_or_copy(src: str, dest: Path, copy: bool) -> None:
    """Place ``src`` at ``dest``, by symlink unless ``copy``."""
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if copy:
        dest.write_bytes(Path(src).read_bytes())
    else:
        dest.symlink_to(os.path.abspath(src))


def _write_annotation(path: Path, episode: RawEpisode, n_frames: int) -> None:
    """Write one episode's joint annotation, truncated to ``n_frames``."""
    label: dict[str, object] = {
        "joint_source": "real",
        JOINT_KEY: np.asarray(episode.joints[:n_frames], dtype=np.float32).tolist(),
        "fps": episode.fps,
        "scene_key": episode.scene_key,
        "source_path": episode.source_path,
    }
    if episode.gripper is not None:
        label[GRIPPER_KEY] = np.asarray(
            episode.gripper[:n_frames], dtype=np.float32).tolist()
    path.write_text(json.dumps(label))


def materialize_train_tree(reader: ReaderSpec, *, val_ratio: float = 0.1,
                           seed: int = 0, limit: int = 0, copy: bool = False,
                           progress=None) -> MaterializeReport:
    """Build ``reader``'s canonical train tree from its declared source.

    Parameters
    ----------
    reader:
        Must have a ``train`` source; a reader with a status is refused, since
        materialising a corpus that cannot supervise the head is not useful.
    val_ratio, seed:
        Scene-stratified split parameters.
    limit:
        Cap on episodes read (``0`` = all), useful for a smoke run. The tree is
        rewritten whole either way, so a capped run leaves a tree holding only
        the episodes it read -- point it at a throwaway ``KINESCORE_DATA_ROOT``
        rather than over a corpus that took hours to build.
    copy:
        Copy videos instead of symlinking them.
    progress:
        ``progress(message)``, or ``None``.

    Raises
    ------
    ValueError
        If the reader declares no source, or declares a blocking status.
    """
    if reader.train is None:
        raise ValueError(
            f"reader {reader.reader_id!r} declares no train source"
            + (f": {reader.status}" if reader.status else ""))
    if reader.status:
        raise ValueError(
            f"reader {reader.reader_id!r} is not trainable: {reader.status}")

    view = reader.view
    tree = reader.train_tree
    report = MaterializeReport(reader_id=reader.reader_id, tree=str(tree))

    adapter = get_adapter(reader.train.adapter)
    episodes: list[RawEpisode] = []
    for entry in adapter.episodes(reader.train):
        if isinstance(entry, SkippedEpisode):
            report.skipped.append((entry.episode_id, entry.reason))
            continue
        episodes.append(entry)
        if limit and len(episodes) >= limit:
            break
    if not episodes:
        raise ValueError(
            f"adapter {reader.train.adapter!r} found no usable episode under "
            f"{reader.train.root!r}")
    if progress:
        progress(f"{len(episodes)} episodes, {len(report.skipped)} skipped")

    scene = ({e.episode_id: e.episode_id for e in episodes}
             if reader.train.scene_key == "episode"
             else {e.episode_id: e.scene_key for e in episodes})
    train_ids, val_ids = stratified_episode_split(
        [e.episode_id for e in episodes], val_ratio=val_ratio, seed=seed,
        scene_key_fn=lambda eid: scene[eid])
    side = dict.fromkeys(train_ids, "train") | dict.fromkeys(val_ids, "val")

    for split in ("train", "val"):
        for kind, suffix in (("videos", "*.mp4"), ("annotation", "*.json")):
            d = tree / kind / split
            d.mkdir(parents=True, exist_ok=True)
            # This function is the tree's only writer and rewrites every
            # episode, so anything already here is from an earlier run. Left in
            # place, an episode the split now sends to the other side would sit
            # in both, and training would read its own validation set.
            for stale in d.glob(suffix):
                stale.unlink()

    for i, episode in enumerate(episodes, 1):
        split = side[episode.episode_id]
        video = tree / "videos" / split / f"{episode.episode_id}.mp4"
        try:
            if episode.packed is not None:
                _link_or_copy(episode.packed, video, copy)
            else:
                _pack(episode, view, video)
            probe = ffprobe(str(video))
            view.check_frame_size(int(probe["w"]), int(probe["h"]))
        except (ValueError, subprocess.CalledProcessError) as exc:
            if video.exists() or video.is_symlink():
                video.unlink()
            report.skipped.append((episode.episode_id, str(exc)))
            continue

        n_frames = min(int(probe["n_frames"]), int(episode.joints.shape[0]))
        if n_frames < 2:
            video.unlink()
            report.skipped.append(
                (episode.episode_id,
                 f"only {n_frames} usable frame(s): video has "
                 f"{probe['n_frames']}, joints {episode.joints.shape[0]}"))
            continue
        _write_annotation(
            tree / "annotation" / split / f"{episode.episode_id}.json",
            episode, n_frames)
        if split == "train":
            report.n_train += 1
        else:
            report.n_val += 1
        if progress and i % 50 == 0:
            progress(f"{i}/{len(episodes)} written")

    card = {
        "reader_id": reader.reader_id,
        "robot": reader.robot,
        "view_id": view.view_id,
        "view": asdict(view),
        "corpus": reader.train.corpus,
        "adapter": reader.train.adapter,
        "source_root": reader.train.root,
        "cameras": list(reader.train.cameras),
        "joint_field": reader.train.joint_field,
        "joint_columns": list(reader.train.joint_columns),
        "gripper_columns": list(reader.train.gripper_columns),
        "gripper_field": reader.train.gripper_field,
        "n_train": report.n_train,
        "n_val": report.n_val,
        "n_skipped": len(report.skipped),
        "val_ratio": val_ratio,
        "split_seed": seed,
        "video_placement": "copy" if copy else "symlink",
        "skipped": [{"episode_id": e, "reason": r}
                    for e, r in report.skipped[:_MAX_SKIP_EXAMPLES]],
    }
    (tree / "dataset_card.json").write_text(json.dumps(card, indent=2))
    if progress:
        progress(f"train {report.n_train} / val {report.n_val} -> {tree}")
    return report
