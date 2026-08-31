"""Compare human segment ratings against the detector ratios that selected them.

Reads a ratings CSV (`clip,rating` with Good/Medium/Bad) and the key written
beside the batch, then reports how well each detector's ratio ranks the clips
the way the rater did. AUC is the headline: it reads the ranking only, so no
threshold is fitted and nothing about it can be tuned after the fact.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ORDER = {"good": 0, "medium": 1, "bad": 2}


def read_ratings(path: Path) -> dict[str, str]:
    """``{clip id: rating}`` for every row that carries a rating."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            value = (row.get("rating") or "").strip().lower()
            if not value:
                continue
            if value not in ORDER:
                raise SystemExit(
                    f"{path}: clip {row['clip']!r} has rating {value!r}; "
                    f"expected one of {sorted(ORDER)}")
            out[str(row["clip"]).strip()] = value
    return out


def auc(scores: list[float], labels: list[int]) -> float | None:
    """Probability a positive outranks a negative, ties counting half.

    Computed by direct pair comparison rather than a trapezoid over a sampled
    curve, so a small batch with heavy ties still gets an exact answer.
    """
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation, averaging ranks over ties."""
    if len(xs) < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else None


def report(rated: list[tuple[dict, str]], detectors: list[str]) -> None:
    for name in detectors:
        rows = [(e["ratios"][name], r) for e, r in rated if name in e["ratios"]]
        if not rows:
            continue
        scores = [s for s, _ in rows]
        bad = [int(r == "bad") for _, r in rows]
        not_good = [int(r != "good") for _, r in rows]
        ordinal = [float(ORDER[r]) for _, r in rows]
        a_bad, a_ng = auc(scores, bad), auc(scores, not_good)
        rho = spearman(scores, ordinal)
        print(f"  {name:10s} n={len(rows):3d}  "
              f"AUC(Bad)={_fmt(a_bad)}  AUC(Bad+Medium)={_fmt(a_ng)}  rho={_fmt(rho)}")


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", required=True, help="CSV with clip,rating")
    parser.add_argument("--key", required=True, help="key.json written with the batch")
    parser.add_argument("--by-embodiment", action="store_true",
                        help="also break the report down per embodiment")
    args = parser.parse_args()

    ratings = read_ratings(Path(args.ratings))
    key = json.loads(Path(args.key).read_text())
    key = key.get("clips", key)
    rated = [(key[c], r) for c, r in ratings.items() if c in key]
    unknown = sorted(set(ratings) - set(key))
    if unknown:
        raise SystemExit(f"{len(unknown)} rated clip(s) are not in the key: {unknown[:5]}")
    if not rated:
        raise SystemExit("no rated clip appears in the key")

    detectors = sorted({n for e, _ in rated for n in e["ratios"]})
    counts = {r: sum(1 for _, x in rated if x == r) for r in ORDER}
    print(f"[ratings] {len(rated)} rated  {counts}")
    print("[all]")
    report(rated, detectors)
    peak = [(e["peak_ratio"], r) for e, r in rated]
    print(f"  {'peak':10s} n={len(peak):3d}  "
          f"AUC(Bad)={_fmt(auc([s for s, _ in peak], [int(r == 'bad') for _, r in peak]))}  "
          f"AUC(Bad+Medium)="
          f"{_fmt(auc([s for s, _ in peak], [int(r != 'good') for _, r in peak]))}  "
          f"rho={_fmt(spearman([s for s, _ in peak], [float(ORDER[r]) for _, r in peak]))}")

    controls = [(e, r) for e, r in rated if e.get("control")]
    if controls:
        bad = sum(1 for _, r in controls if r == "bad")
        print(f"[controls] real footage: {len(controls)} rated, "
              f"{bad} called Bad ({100 * bad / len(controls):.0f}% rater floor)")

    if args.by_embodiment:
        groups: dict[str, list] = {}
        for entry, rating in rated:
            source = entry.get("source_clip", "all/")
            groups.setdefault(source.split("/")[0], []).append((entry, rating))
        for name, rows in sorted(groups.items()):
            print(f"[{name}] n={len(rows)}")
            report(rows, detectors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
