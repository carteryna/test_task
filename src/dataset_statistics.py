#!/usr/bin/env python3
"""Descriptive statistics for cleaned train/val YOLO labels (class 0 = vehicle)."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

ASPECT_BINS = (
    ("<1.2 (squarish)", 0.0, 1.2),
    ("1.2–2.0 (car-like)", 1.2, 2.0),
    ("2.0–3.5 (van/SUV)", 2.0, 3.5),
    ("3.5–6.0 (truck-like)", 3.5, 6.0),
    (">=6.0 (very long)", 6.0, None),
)


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def read_split_list(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def load_manifest_sizes(path: Path) -> dict[str, tuple[int, int]]:
    sizes: dict[str, tuple[int, int]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            sizes[row["path"]] = (int(row["width"]), int(row["height"]))
    return sizes


def load_boxes(txt_path: Path) -> list[tuple[float, float, float, float]]:
    if not txt_path.exists():
        return []
    out: list[tuple[float, float, float, float]] = []
    for line in txt_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _cls, x, y, w, h = parts
        out.append((float(x), float(y), float(w), float(h)))
    return out


def aspect_bin(ar: float) -> str:
    for name, lo, hi in ASPECT_BINS:
        if hi is None:
            if ar >= lo:
                return name
        elif lo <= ar < hi:
            return name
    return ASPECT_BINS[-1][0]


def pct(n: int, total: int) -> float:
    return 100.0 * n / total if total else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "data.yaml")
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument(
        "--allow-eval",
        action="store_true",
        help="Summarize cleaned hold-out labels under eval_gt.labels_dir.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Defaults to cleanup.clean_dir (or eval_gt.labels_dir with --allow-eval)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    cfg = load_config(cfg_path)

    if args.allow_eval:
        splits = args.splits or ["eval"]
        if splits != ["eval"]:
            raise RuntimeError("--allow-eval only accepts --splits eval")
        labels_dir = (
            REPO_ROOT / args.labels_dir
            if args.labels_dir
            else REPO_ROOT / cfg["eval_gt"]["labels_dir"]
        )
        out_default = REPO_ROOT / cfg["eval_gt"].get(
            "stats_path", "data/splits/eval_dataset_statistics.json"
        )
    else:
        splits = args.splits or ["train", "val"]
        if "eval" in splits:
            raise RuntimeError("Refusing to summarize eval here. Use --allow-eval.")
        labels_dir = (
            REPO_ROOT / args.labels_dir
            if args.labels_dir
            else REPO_ROOT / cfg["cleanup"]["clean_dir"]
        )
        out_default = REPO_ROOT / "data" / "splits" / "dataset_statistics.json"

    splits_dir = REPO_ROOT / cfg["paths"]["splits_dir"]
    sizes = load_manifest_sizes(splits_dir / "manifest.csv")

    image_rels: list[str] = []
    for split in splits:
        image_rels.extend(read_split_list(splits_dir / f"{split}.txt"))
    image_rels = list(dict.fromkeys(image_rels))

    per_frame_counts: list[int] = []
    widths_n: list[float] = []
    heights_n: list[float] = []
    widths_px: list[float] = []
    heights_px: list[float] = []
    aspects: list[float] = []
    by_clip_boxes: dict[str, int] = defaultdict(int)
    by_clip_frames: dict[str, int] = defaultdict(int)
    aspect_hist: dict[str, int] = defaultdict(int)

    eval_ids = {c["id"] for c in cfg["clips"]["eval"]}
    for rel in image_rels:
        clip = Path(rel).parent.name
        if clip in eval_ids and not args.allow_eval:
            raise RuntimeError(f"Eval path in statistics: {rel}")
        txt = labels_dir / clip / f"{Path(rel).stem}.txt"
        boxes = load_boxes(txt)
        per_frame_counts.append(len(boxes))
        by_clip_frames[clip] += 1
        by_clip_boxes[clip] += len(boxes)
        img_w, img_h = sizes.get(rel, (0, 0))
        for _x, _y, w, h in boxes:
            widths_n.append(w)
            heights_n.append(h)
            ar = max(w, h) / (min(w, h) + 1e-12)
            aspects.append(ar)
            aspect_hist[aspect_bin(ar)] += 1
            if img_w and img_h:
                widths_px.append(w * img_w)
                heights_px.append(h * img_h)

    n_frames = len(image_rels)
    n_boxes = sum(per_frame_counts)
    if n_frames == 0:
        raise RuntimeError("No frames found")

    print(f"Labels: {labels_dir.relative_to(REPO_ROOT)}")
    print(f"Splits: {splits}")
    print(f"Frames: {n_frames}")
    print(f"Boxes:  {n_boxes}  (class 0 = vehicle)")
    print(
        f"Boxes/frame: mean={statistics.fmean(per_frame_counts):.2f}  "
        f"min={min(per_frame_counts)}  max={max(per_frame_counts)}  "
        f"median={statistics.median(per_frame_counts):.1f}"
    )
    print()
    print("Normalized YOLO box size (w, h in 0–1)")
    print(
        f"  width:  mean={statistics.fmean(widths_n):.4f}  "
        f"min={min(widths_n):.4f}  max={max(widths_n):.4f}"
    )
    print(
        f"  height: mean={statistics.fmean(heights_n):.4f}  "
        f"min={min(heights_n):.4f}  max={max(heights_n):.4f}"
    )
    if widths_px:
        print("Pixel box size (from manifest width/height)")
        print(
            f"  width_px:  mean={statistics.fmean(widths_px):.1f}  "
            f"min={min(widths_px):.1f}  max={max(widths_px):.1f}"
        )
        print(
            f"  height_px: mean={statistics.fmean(heights_px):.1f}  "
            f"min={min(heights_px):.1f}  max={max(heights_px):.1f}"
        )
    print()
    print("Aspect ratio max(w,h)/min(w,h)")
    print(
        f"  mean={statistics.fmean(aspects):.2f}  "
        f"median={statistics.median(aspects):.2f}  "
        f"min={min(aspects):.2f}  max={max(aspects):.2f}"
    )
    print(f"{'bin':<22} {'count':>7} {'%':>7}")
    for name, _lo, _hi in ASPECT_BINS:
        c = aspect_hist.get(name, 0)
        print(f"{name:<22} {c:7d} {pct(c, n_boxes):6.1f}%")

    print()
    print(f"{'clip':<6} {'frames':>7} {'boxes':>7} {'mean/frame':>10}")
    for clip in sorted(by_clip_frames):
        fcount = by_clip_frames[clip]
        bcount = by_clip_boxes[clip]
        print(f"{clip:<6} {fcount:7d} {bcount:7d} {bcount / fcount:10.2f}")

    out = {
        "labels_dir": str(labels_dir.relative_to(REPO_ROOT)),
        "splits": splits,
        "n_frames": n_frames,
        "n_boxes": n_boxes,
        "boxes_per_frame": {
            "mean": round(statistics.fmean(per_frame_counts), 4),
            "min": min(per_frame_counts),
            "max": max(per_frame_counts),
            "median": statistics.median(per_frame_counts),
        },
        "normalized": {
            "width_mean": round(statistics.fmean(widths_n), 6),
            "width_min": round(min(widths_n), 6),
            "width_max": round(max(widths_n), 6),
            "height_mean": round(statistics.fmean(heights_n), 6),
            "height_min": round(min(heights_n), 6),
            "height_max": round(max(heights_n), 6),
        },
        "pixels": {
            "width_mean": round(statistics.fmean(widths_px), 2) if widths_px else None,
            "width_min": round(min(widths_px), 2) if widths_px else None,
            "width_max": round(max(widths_px), 2) if widths_px else None,
            "height_mean": round(statistics.fmean(heights_px), 2) if heights_px else None,
            "height_min": round(min(heights_px), 2) if heights_px else None,
            "height_max": round(max(heights_px), 2) if heights_px else None,
        },
        "aspect_ratio": {
            "mean": round(statistics.fmean(aspects), 4),
            "median": round(statistics.median(aspects), 4),
            "min": round(min(aspects), 4),
            "max": round(max(aspects), 4),
            "bins": {k: aspect_hist.get(k, 0) for k, _, _ in ASPECT_BINS},
        },
        "per_clip": {
            clip: {
                "frames": by_clip_frames[clip],
                "boxes": by_clip_boxes[clip],
                "mean_boxes_per_frame": round(by_clip_boxes[clip] / by_clip_frames[clip], 4),
            }
            for clip in sorted(by_clip_frames)
        },
    }
    if args.out is None:
        out_path = out_default
    else:
        out_path = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
