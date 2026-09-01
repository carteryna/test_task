#!/usr/bin/env python3
"""Pinhole distance bands for cleaned train/val boxes (class 0 = vehicle).

s_px = min(box_w, box_h) in pixels (short side ≈ physical width).
f_px = H / (2 * tan(FOV_v / 2))
distance_m = f_px * W_ref / s_px

Bands: near [0, 200), far [200, 400), beyond/failed separately.
Eval clip is refused.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def read_split_list(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def load_manifest(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            rows[row["path"]] = row
    return rows


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


def focal_px(img_h: int, fov_v_deg: float) -> float:
    return img_h / (2.0 * math.tan(math.radians(fov_v_deg) / 2.0))


def band_for(dist_m: float | None, near_max: float, far_max: float) -> str:
    if dist_m is None or not math.isfinite(dist_m) or dist_m <= 0:
        return "failed"
    if dist_m < near_max:
        return "near_0_200"
    if dist_m < far_max:
        return "far_200_400"
    return "beyond_400"


def summarize(xs: list[float]) -> dict | None:
    if not xs:
        return None
    xs_sorted = sorted(xs)
    n = len(xs_sorted)

    def pct(p: float) -> float:
        i = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
        return round(xs_sorted[i], 2)

    return {
        "n": n,
        "min": round(min(xs_sorted), 2),
        "p10": pct(10),
        "p25": pct(25),
        "median": round(statistics.median(xs_sorted), 2),
        "p75": pct(75),
        "p90": pct(90),
        "mean": round(statistics.fmean(xs_sorted), 2),
        "max": round(max(xs_sorted), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "data.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    cfg = load_config(cfg_path)

    if "eval" in args.splits:
        raise RuntimeError("Refusing distance tagging on eval here.")

    dist_cfg = cfg["distance"]
    w_ref = float(dist_cfg["w_ref_m"])
    fov_v = float(dist_cfg["fov_v_deg"])
    near_max = float(dist_cfg["near_max_m"])
    far_max = float(dist_cfg["far_max_m"])
    size_side = dist_cfg.get("size_side", "min")
    if size_side != "min":
        raise RuntimeError(f"Only size_side=min is supported (got {size_side})")

    splits_dir = REPO_ROOT / cfg["paths"]["splits_dir"]
    labels_dir = REPO_ROOT / cfg["cleanup"]["clean_dir"]
    manifest = load_manifest(splits_dir / "manifest.csv")

    image_rels: list[str] = []
    for split in args.splits:
        image_rels.extend(read_split_list(splits_dir / f"{split}.txt"))
    image_rels = list(dict.fromkeys(image_rels))

    band_counts: dict[str, int] = defaultdict(int)
    by_clip: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    box_rows: list[dict] = []
    distances: list[float] = []

    for rel in image_rels:
        clip = Path(rel).parent.name
        if clip == "E":
            raise RuntimeError(f"Eval path in distance tagging: {rel}")
        meta = manifest.get(rel)
        if meta is None:
            raise RuntimeError(f"Missing manifest row for {rel}")
        img_w, img_h = int(meta["width"]), int(meta["height"])
        f_px = focal_px(img_h, fov_v)
        txt = labels_dir / clip / f"{Path(rel).stem}.txt"
        for x, y, w, h in load_boxes(txt):
            w_px = w * img_w
            h_px = h * img_h
            s_px = min(w_px, h_px)
            if s_px <= 1e-6:
                dist_m = None
            else:
                dist_m = f_px * w_ref / s_px
            band = band_for(dist_m, near_max, far_max)
            band_counts[band] += 1
            by_clip[clip][band] += 1
            if dist_m is not None and math.isfinite(dist_m):
                distances.append(dist_m)
            box_rows.append(
                {
                    "path": rel,
                    "clip_id": clip,
                    "split": meta.get("split", ""),
                    "x": f"{x:.6f}",
                    "y": f"{y:.6f}",
                    "w": f"{w:.6f}",
                    "h": f"{h:.6f}",
                    "s_px": f"{s_px:.2f}",
                    "f_px": f"{f_px:.2f}",
                    "distance_m": f"{dist_m:.2f}" if dist_m is not None else "",
                    "band": band,
                }
            )

    n = len(box_rows)
    order = ["near_0_200", "far_200_400", "beyond_400", "failed"]
    s_pxs = [float(r["s_px"]) for r in box_rows]
    by_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_clip_fpx: dict[str, float] = {}
    for r in box_rows:
        by_split[r["split"]][r["band"]] += 1
        if r["clip_id"] not in by_clip_fpx:
            by_clip_fpx[r["clip_id"]] = float(r["f_px"])

    per_band: dict[str, dict] = {}
    for band in order:
        bd = [
            float(r["distance_m"])
            for r in box_rows
            if r["band"] == band and r["distance_m"]
        ]
        bs = [float(r["s_px"]) for r in box_rows if r["band"] == band]
        per_band[band] = {
            "count": len(bs),
            "share": round(len(bs) / n, 4) if n else 0.0,
            "distance_m": summarize(bd),
            "s_px": summarize(bs),
        }

    per_clip_out: dict[str, dict] = {}
    for clip in sorted(by_clip):
        near_c = by_clip[clip].get("near_0_200", 0)
        far_c = by_clip[clip].get("far_200_400", 0)
        total_c = sum(by_clip[clip].values())
        per_clip_out[clip] = {
            "near_0_200": near_c,
            "far_200_400": far_c,
            "beyond_400": by_clip[clip].get("beyond_400", 0),
            "failed": by_clip[clip].get("failed", 0),
            "n_boxes": total_c,
            "far_share": round(far_c / total_c, 4) if total_c else 0.0,
            "f_px": round(by_clip_fpx.get(clip, 0.0), 2),
        }

    print(f"Distance tagging: {n} boxes from {labels_dir.relative_to(REPO_ROOT)}")
    print(f"W_ref={w_ref} m  FOV_v={fov_v}°  s_px=min(w,h)  formula: f_px*W_ref/s_px")
    print()
    print(f"{'band':<14} {'count':>7} {'%':>7}")
    for band in order:
        c = band_counts.get(band, 0)
        print(f"{band:<14} {c:7d} {100.0 * c / n if n else 0:6.1f}%")
    dist_sum = summarize(distances)
    spx_sum = summarize(s_pxs)
    if dist_sum:
        print(
            f"\ndistance_m: min={dist_sum['min']}  median={dist_sum['median']}  "
            f"mean={dist_sum['mean']}  max={dist_sum['max']}"
        )
        print(
            f"s_px:       min={spx_sum['min']}  median={spx_sum['median']}  "
            f"mean={spx_sum['mean']}  max={spx_sum['max']}"
        )

    print()
    print(f"{'clip':<6} {'near':>7} {'far':>7} {'far%':>7} {'f_px':>8}")
    for clip, row in per_clip_out.items():
        print(
            f"{clip:<6} {row['near_0_200']:7d} {row['far_200_400']:7d} "
            f"{100.0 * row['far_share']:6.1f}% {row['f_px']:8.1f}"
        )

    print()
    print(f"{'split':<8} {'near':>7} {'far':>7} {'total':>7}")
    for split in ("train", "val"):
        near_s = by_split[split].get("near_0_200", 0)
        far_s = by_split[split].get("far_200_400", 0)
        print(f"{split:<8} {near_s:7d} {far_s:7d} {near_s + far_s:7d}")

    print()
    print(f"{'band':<14} {'s_px med':>9} {'s_px min':>9} {'dist med':>9}")
    for band in ("near_0_200", "far_200_400"):
        sp = per_band[band]["s_px"]
        dd = per_band[band]["distance_m"]
        if not sp or not dd:
            continue
        print(f"{band:<14} {sp['median']:9.1f} {sp['min']:9.1f} {dd['median']:9.1f}")

    # FOV sensitivity on band assignment (same s_px, alternate FOV)
    print("\nFOV sensitivity (band counts if FOV_v changes, same boxes)")
    print(f"{'FOV_v':>6} {'near':>7} {'far':>7} {'beyond':>8}")
    fov_sensitivity = {}
    for fov in (40.0, 70.0, 90.0):
        counts = defaultdict(int)
        for rel in image_rels:
            meta = manifest[rel]
            img_w, img_h = int(meta["width"]), int(meta["height"])
            f_px = focal_px(img_h, fov)
            clip = Path(rel).parent.name
            txt = labels_dir / clip / f"{Path(rel).stem}.txt"
            for _x, _y, w, h in load_boxes(txt):
                s_px = min(w * img_w, h * img_h)
                dist_m = f_px * w_ref / s_px if s_px > 1e-6 else None
                counts[band_for(dist_m, near_max, far_max)] += 1
        fov_sensitivity[str(fov)] = dict(counts)
        print(
            f"{fov:6.0f} {counts.get('near_0_200', 0):7d} "
            f"{counts.get('far_200_400', 0):7d} {counts.get('beyond_400', 0):8d}"
        )

    csv_path = REPO_ROOT / dist_cfg["boxes_csv"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path", "clip_id", "split", "x", "y", "w", "h",
                "s_px", "f_px", "distance_m", "band",
            ],
        )
        writer.writeheader()
        writer.writerows(box_rows)

    report = {
        "n_boxes": n,
        "n_frames": len(image_rels),
        "w_ref_m": w_ref,
        "fov_v_deg": fov_v,
        "size_side": size_side,
        "formula": "distance_m = f_px * W_ref / s_px; f_px = H / (2*tan(FOV_v/2))",
        "bands": {k: band_counts.get(k, 0) for k in order},
        "distance_m": dist_sum,
        "s_px": spx_sum,
        "per_band": per_band,
        "per_split": {s: dict(by_split[s]) for s in sorted(by_split)},
        "per_clip": per_clip_out,
        "fov_sensitivity": fov_sensitivity,
        "exclude_classes_note": dist_cfg.get("exclude_classes_note", ""),
        "boxes_csv": str(csv_path.relative_to(REPO_ROOT)),
    }
    report_path = REPO_ROOT / dist_cfg["report_path"]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {csv_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {report_path.relative_to(REPO_ROOT)}")
    near = band_counts.get("near_0_200", 0)
    far = band_counts.get("far_200_400", 0)
    if near == 0 or far == 0:
        print("WARNING: one training band is empty — check FOV / W_ref or add held-out eval coverage later.")
    else:
        print(f"OK: both bands populated (near={near}, far={far})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
