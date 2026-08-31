"""Build a segment-level rating batch for validating the detectors against human judgement.

Samples segments stratified by detector ratio so the full range is covered evenly,
cuts each as a three-segment clip (the rated segment plus one of context on each
side), and writes the answers to a key the raters never see.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import Counter
from pathlib import Path

BINS = ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 1e9))
PER_BIN = 25
CONTEXT = 1
DETECTORS = ("rigidity", "jerk")


def bundle_segments(path: Path, bundle: str) -> list[dict]:
    """Segments of one exported web bundle, each with its per-detector ratios."""
    out = []
    for key, video in json.loads(path.read_text())["videos"].items():
        for i, seg in enumerate(video["segments"]):
            ratios = {n: seg[n]["ratio"] for n in DETECTORS
                      if n in seg and seg[n]["ratio"] is not None}
            if not ratios:
                continue
            out.append({"bundle": bundle, "video": key, "index": i,
                        "n_segments": len(video["segments"]), "fps": video["fps"],
                        "ratios": ratios, "peak": max(ratios.values()),
                        "source": f"bucket:web/{bundle}/{key}.mp4"})
    return out


def scored_segments(path: Path, bundle: str) -> list[dict]:
    """Segments of one scored cell, read straight from its results.jsonl."""
    out = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if not row.get("segments"):
            continue
        for i, seg in enumerate(row["segments"]):
            ratios = {n: seg["detectors"][n]["value"] / seg["detectors"][n]["threshold"]
                      for n in DETECTORS if n in seg["detectors"]}
            out.append({"bundle": bundle, "video": row["id"], "index": i,
                        "n_segments": len(row["segments"]), "fps": None,
                        "ratios": ratios, "peak": max(ratios.values()),
                        "source": row["path"]})
    return out


def sample(pool: list[dict], rng: random.Random) -> list[dict]:
    """Equal-sized draws from each ratio bin, interior segments only.

    A segment at the very start or end of its video has no context on one side,
    so it is skipped: every clip must show the rated segment the same way.
    """
    picked = []
    for lo, hi in BINS:
        eligible = [s for s in pool
                    if lo < s["peak"] <= hi
                    and CONTEXT <= s["index"] < s["n_segments"] - CONTEXT]
        rng.shuffle(eligible)
        seen, chosen = Counter(), []
        for s in sorted(eligible, key=lambda s: seen[s["video"]]):
            if len(chosen) >= PER_BIN:
                break
            if seen[s["video"]] >= 2:
                continue
            seen[s["video"]] += 1
            chosen.append({**s, "bin": f"{lo}-{hi}"})
        picked += chosen
    return picked


def enforce_control(picks: list[dict], pool: list[dict], bundle: str, want: int,
                    rng: random.Random) -> list[dict]:
    """Swap picks until ``want`` of them come from ``bundle``, keeping the bin sizes.

    Each swap replaces a same-bin pick from another bundle, so the ratio strata the
    batch is built around stay exactly as sampled.
    """
    have = sum(1 for p in picks if p["bundle"] == bundle)
    if have >= want:
        return picks
    chosen = {(p["bundle"], p["video"], p["index"]) for p in picks}
    for _ in range(want - have):
        bins = Counter(p["bin"] for p in picks if p["bundle"] != bundle)
        for label, _count in bins.most_common():
            lo, hi = (float(x) for x in label.split("-"))
            spare = [s for s in pool
                     if s["bundle"] == bundle and lo < s["peak"] <= hi
                     and CONTEXT <= s["index"] < s["n_segments"] - CONTEXT
                     and (s["bundle"], s["video"], s["index"]) not in chosen]
            if not spare:
                continue
            new = rng.choice(spare)
            drop = next(i for i, p in enumerate(picks)
                        if p["bin"] == label and p["bundle"] != bundle)
            picks[drop] = {**new, "bin": label}
            chosen.add((new["bundle"], new["video"], new["index"]))
            break
        else:
            break
    return picks


def fetch(source: str, dest: Path, api) -> Path | None:
    """Local path of a sample's source video, downloading it when it lives on the bucket."""
    if not source.startswith("bucket:"):
        return Path(source) if Path(source).is_file() else None
    api.download_bucket_files("twanghcmut/hallucinate-bench",
                              [(source[len("bucket:"):], str(dest))])
    return dest if dest.is_file() else None


def scored_crop(view_id: str) -> str:
    """``crop=`` filter limiting a packed frame to the panels the reader reads.

    A view that exposes a subset of its panels leaves the rest in the clip, and a
    rater shown a panel the detector never read judges footage the score does not
    describe. Returns an empty string when the view exposes everything.
    """
    from kinescore.registry.views import load_views

    view = load_views().get(view_id)
    if view is None or view.panel is None:
        return ""
    exposed = set(view.panel_indices)
    if len(exposed) == view.panel_count:
        return ""
    width, height = view.panel
    if view.packing == "grid2x2":
        rows = {p // 2 for p in exposed}
        cols = {p % 2 for p in exposed}
        if exposed != {r * 2 + c for r in rows for c in cols}:
            raise ValueError(
                f"view {view_id!r} exposes panels {sorted(exposed)}, which do not "
                f"form a rectangle; a single crop cannot show exactly those")
        return (f"crop={len(cols) * width}:{len(rows) * height}:"
                f"{min(cols) * width}:{min(rows) * height}")
    if view.packing == "width":
        return f"crop={len(exposed) * width}:{height}:{min(exposed) * width}:0"
    if view.packing == "height":
        return f"crop={width}:{len(exposed) * height}:0:{min(exposed) * height}"
    raise ValueError(f"view {view_id!r} packs {view.packing!r}, which has no crop rule")


def cut(src: Path, dest: Path, first: int, last: int, crop: str = "") -> bool:
    """Write frames ``first..last`` of ``src`` to ``dest``, re-timed from zero."""
    chain = f"select='between(n,{first},{last})',setpts=N/FRAME_RATE/TB"
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf",
         f"{chain},{crop}" if crop else chain, "-an", str(dest)],
        capture_output=True)
    return proc.returncode == 0 and dest.is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="directory the batch is written to")
    parser.add_argument("--segment-frames", type=int, default=16)
    parser.add_argument("--overlap", type=int, default=20,
                        help="clips per embodiment flagged for multi-rater agreement")
    parser.add_argument("--control", action="append", default=[],
                        metavar="EMBODIMENT:BUNDLE:N",
                        help="guarantee N clips from BUNDLE, spread over the bins; "
                             "real footage rated Bad measures the rater floor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--view", action="append", default=[], metavar="BUNDLE:VIEW_ID",
                        help="crop a bundle's clips to the panels VIEW_ID exposes")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    prune = Path("/tmp/claude-1479/-pfss-mlde-workspaces-mlde-wsp-IAS-SAMMerge"
                 "/0ab3fba1-55e8-426a-870d-a993a0a7d0fc/scratchpad/prune")
    out_root = Path(args.out)
    scored = Path("/dev/shm/kinescore/out")
    pools = {
        "humanoid": (bundle_segments(prune / "fastercache_humanoid_sv.segments.json",
                                     "fastercache_humanoid_sv")
                     + bundle_segments(prune / "cosmos_humanoid_sv.segments.json",
                                       "cosmos_humanoid_sv")),
        "bimanual": (bundle_segments(prune / "radial_bimanual_sv.segments.json",
                                     "radial_bimanual_sv")
                     + bundle_segments(prune / "augment_bimanual_sv.segments.json",
                                       "augment_bimanual_sv")),
        "single_arm": (bundle_segments(prune / "radial_single_arm_sv.segments.json",
                                       "radial_single_arm_sv")
                       + scored_segments(scored / "radial.mv4_grid_static.dreamgen"
                                         / "results.jsonl", "radial_single_arm_mv")
                       + scored_segments(scored / "real.sv1_4x3.a1x" / "results.jsonl",
                                         "real_a1x_teleop")
                       + scored_segments(scored / "dense.sv1_4x3.dreamgen" / "results.jsonl",
                                         "dense_single_arm_sv")),
    }

    crops = {}
    for spec in args.view:
        bundle, view_id = spec.split(":")
        crops[bundle] = scored_crop(view_id)

    api = HfApi()
    rng = random.Random(args.seed)
    key: dict[str, dict] = {}
    counts: dict[str, Counter] = {}
    staging = out_root / ".src"
    staging.mkdir(parents=True, exist_ok=True)

    controls: dict[str, tuple[str, int]] = {}
    for spec in args.control:
        embodiment, bundle, count = spec.split(":")
        controls[embodiment] = (bundle, int(count))

    for embodiment, pool in pools.items():
        picks = sample(pool, rng)
        if embodiment in controls:
            picks = enforce_control(picks, pool, *controls[embodiment], rng)
        rng.shuffle(picks)
        directory = out_root / embodiment
        directory.mkdir(parents=True, exist_ok=True)
        counts[embodiment] = Counter()
        written = 0
        for item in picks:
            src = fetch(item["source"], staging / "src.mp4", api)
            if src is None:
                print(f"[skip] no source for {item['bundle']}/{item['video']}")
                continue
            first = (item["index"] - CONTEXT) * args.segment_frames
            last = (item["index"] + CONTEXT + 1) * args.segment_frames - 1
            written += 1
            dest = directory / f"{written}.mp4"
            crop = crops.get(item["bundle"], "")
            if not cut(src, dest, first, last, crop):
                print(f"[skip] cut failed for {item['bundle']}/{item['video']}")
                written -= 1
                continue
            key[f"{embodiment}/{written}.mp4"] = {
                "bundle": item["bundle"], "video": item["video"],
                "segment_index": item["index"], "bin": item["bin"],
                "ratios": {n: round(v, 4) for n, v in item["ratios"].items()},
                "peak_ratio": round(item["peak"], 4),
                "rated_segment_in_clip": [CONTEXT, CONTEXT],
                "overlap_set": written <= args.overlap,
                "control": item["bundle"] in ("real_a1x_teleop",),
                "cropped_to_scored_panels": bool(crop),
            }
            counts[embodiment][item["bin"]] += 1
            if src != staging / "src.mp4":
                continue
            src.unlink(missing_ok=True)
        print(f"[{embodiment}] {written} clips  {dict(counts[embodiment])}")

    shutil_targets = list(staging.glob("*"))
    for leftover in shutil_targets:
        leftover.unlink(missing_ok=True)
    staging.rmdir()

    (out_root / "key.json").write_text(json.dumps({
        "purpose": "detector-vs-human validation; ratios are hidden from raters",
        "segment_frames": args.segment_frames,
        "clip_layout": "three segments: one of context, the rated segment, one of context",
        "bins": [f"{lo}-{hi}" for lo, hi in BINS],
        "seed": args.seed,
        "clips": key,
    }, indent=1))
    (out_root / "README.md").write_text(
        "# Rating batch\n\n"
        "Each clip is three segments long. Rate **all three** on the Good / Medium / Bad\n"
        "scale from `rubric_vi.md`; the middle one is the segment this batch measures.\n\n"
        f"{len(key)} clips: {', '.join(f'{e} {sum(c.values())}' for e, c in counts.items())}.\n"
        "Clip order is shuffled and carries no information about severity.\n")
    print(f"[done] {len(key)} clips -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
