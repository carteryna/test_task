#!/usr/bin/env python3
"""Profile teacher pseudo-label size vs confidence (clips A–D).

Reads data/labels/raw/boxes.csv (native-pixel xyxy + conf). Does not touch eval.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TIERS = (
    ("tiny", 0, 600),          # ~far-band candidates
    ("small", 600, 2000),
    ("medium_large", 2000, None),  # ~near-band
)


def tier_for(area: float) -> str:
    for name, lo, hi in TIERS:
        if hi is None:
            if area > lo:
                return name
        elif lo <= area < hi:
            return name
    return "tiny" if area < 600 else "medium_large"


def pct(n: int, total: int) -> float:
    return 100.0 * n / total if total else 0.0


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--boxes-csv",
        type=Path,
        default=REPO_ROOT / "data" / "labels" / "raw" / "boxes.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "splits" / "prediction_profile.json",
    )
    args = parser.parse_args()
    boxes_path = args.boxes_csv if args.boxes_csv.is_absolute() else REPO_ROOT / args.boxes_csv
    if not boxes_path.exists():
        raise FileNotFoundError(f"Missing {boxes_path}. Run auto_label.py first.")

    rows: list[dict] = []
    with boxes_path.open() as f:
        for r in csv.DictReader(f):
            if r["clip_id"] == "E":
                raise RuntimeError("Eval boxes found in teacher CSV — leak")
            x1, y1, x2, y2 = map(float, (r["x1"], r["y1"], r["x2"], r["y2"]))
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            area = w * h
            conf = float(r["conf"])
            rows.append(
                {
                    "clip_id": r["clip_id"],
                    "path": r["path"],
                    "conf": conf,
                    "width_px": w,
                    "height_px": h,
                    "area_px": area,
                    "tier": tier_for(area),
                }
            )

    if not rows:
        raise RuntimeError("No boxes to profile")

    by_tier: dict[str, list[dict]] = defaultdict(list)
    by_clip: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_tier[r["tier"]].append(r)
        by_clip[r["clip_id"]].append(r)

    total = len(rows)
    print(f"Teacher boxes profiled: {total} (native pixel xyxy from {boxes_path.relative_to(REPO_ROOT)})")
    print()
    print(f"{'tier':<14} {'count':>7} {'%':>7} {'mean_conf':>10} {'med_conf':>9} "
          f"{'mean_w':>8} {'mean_h':>8} {'med_area':>10}")
    tier_summary = {}
    for name, _lo, _hi in TIERS:
        xs = by_tier.get(name, [])
        confs = [x["conf"] for x in xs]
        areas = [x["area_px"] for x in xs]
        ws = [x["width_px"] for x in xs]
        hs = [x["height_px"] for x in xs]
        entry = {
            "count": len(xs),
            "pct": round(pct(len(xs), total), 2),
            "mean_conf": round(mean(confs), 4),
            "median_conf": round(median(confs), 4),
            "mean_width_px": round(mean(ws), 2),
            "mean_height_px": round(mean(hs), 2),
            "median_area_px": round(median(areas), 1),
        }
        tier_summary[name] = entry
        print(
            f"{name:<14} {entry['count']:7d} {entry['pct']:6.1f}% "
            f"{entry['mean_conf']:10.3f} {entry['median_conf']:9.3f} "
            f"{entry['mean_width_px']:8.1f} {entry['mean_height_px']:8.1f} "
            f"{entry['median_area_px']:10.0f}"
        )

    print()
    print(f"{'clip':<6} {'n':>6} {'mean_w':>8} {'med_w':>8} {'mean_h':>8} {'med_h':>8} "
          f"{'tiny%':>7} {'small%':>7} {'medlg%':>7} {'med_conf':>9}")
    clip_summary = {}
    for clip_id in sorted(by_clip):
        xs = by_clip[clip_id]
        ws = [x["width_px"] for x in xs]
        hs = [x["height_px"] for x in xs]
        confs = [x["conf"] for x in xs]
        n = len(xs)
        tiny_n = sum(1 for x in xs if x["tier"] == "tiny")
        small_n = sum(1 for x in xs if x["tier"] == "small")
        med_n = sum(1 for x in xs if x["tier"] == "medium_large")
        entry = {
            "count": n,
            "mean_width_px": round(mean(ws), 2),
            "median_width_px": round(median(ws), 2),
            "mean_height_px": round(mean(hs), 2),
            "median_height_px": round(median(hs), 2),
            "tiny_pct": round(pct(tiny_n, n), 2),
            "small_pct": round(pct(small_n, n), 2),
            "medium_large_pct": round(pct(med_n, n), 2),
            "median_conf": round(median(confs), 4),
        }
        clip_summary[clip_id] = entry
        print(
            f"{clip_id:<6} {n:6d} {entry['mean_width_px']:8.1f} {entry['median_width_px']:8.1f} "
            f"{entry['mean_height_px']:8.1f} {entry['median_height_px']:8.1f} "
            f"{entry['tiny_pct']:6.1f}% {entry['small_pct']:6.1f}% "
            f"{entry['medium_large_pct']:6.1f}% {entry['median_conf']:9.3f}"
        )

    # Conf histogram in the tiny tier — decides whether 0.15 already holds far boxes
    tiny = by_tier.get("tiny", [])
    tiny_conf_bins = {"<0.10": 0, "0.10-0.15": 0, "0.15-0.25": 0, ">=0.25": 0}
    for x in tiny:
        c = x["conf"]
        if c < 0.10:
            tiny_conf_bins["<0.10"] += 1
        elif c < 0.15:
            tiny_conf_bins["0.10-0.15"] += 1
        elif c < 0.25:
            tiny_conf_bins["0.15-0.25"] += 1
        else:
            tiny_conf_bins[">=0.25"] += 1

    tiny_pct = tier_summary["tiny"]["pct"]
    if tiny_pct >= 15.0:
        decision = (
            "scenario_A_healthy_far_band",
            "Tiny boxes are a solid share of predictions. Keep conf=0.15; "
            "do not lower the global threshold. Prefer cleanup / size-aware reject of large low-conf.",
        )
    elif tiny_pct < 5.0:
        decision = (
            "scenario_B_missing_far_band",
            "Tiny boxes are scarce. Consider re-running teacher at base conf=0.08 with "
            "dual thresholds (0.20 large / 0.08 small).",
        )
    else:
        decision = (
            "scenario_borderline",
            f"Tiny share is {tiny_pct:.1f}% (between 5 and 15). Inspect tiny conf bins; "
            "dual-threshold is optional, not mandatory.",
        )

    print()
    print("Tiny-tier confidence bins")
    for k, v in tiny_conf_bins.items():
        print(f"  {k:<10} {v:5d}  ({pct(v, len(tiny)):.1f}% of tiny)")

    print()
    print(f"Decision: {decision[0]}")
    print(f"  {decision[1]}")

    out = {
        "n_boxes": total,
        "source": str(boxes_path.relative_to(REPO_ROOT)),
        "tiers": tier_summary,
        "per_clip": clip_summary,
        "tiny_conf_bins": tiny_conf_bins,
        "decision": {"id": decision[0], "rationale": decision[1]},
    }
    out_path = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
