#!/usr/bin/env python3
"""Sample frames from train/eval clips and write leak-proof split lists.

Train clips are sampled at train_fps; the eval clip at eval_fps. Validation is
the last val_fraction of each train clip (time-ordered). The eval clip is never
written to train.txt or val.txt.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def repo_rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def extract_clip(
    video_path: Path,
    out_dir: Path,
    clip_id: str,
    sample_fps: float,
    jpeg_quality: int,
    ext: str,
) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if native_fps <= 0:
        cap.release()
        raise RuntimeError(f"Unknown FPS for {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    interval = 1.0 / sample_fps
    next_t = 0.0
    idx = 0
    rows: list[dict] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / native_fps
        if t + 1e-6 >= next_t:
            name = f"{idx:06d}.{ext}"
            out_path = out_dir / name
            ok_write = cv2.imwrite(
                str(out_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )
            if not ok_write:
                cap.release()
                raise RuntimeError(f"Failed to write {out_path}")
            h, w = frame.shape[:2]
            rows.append(
                {
                    "clip_id": clip_id,
                    "video_file": video_path.name,
                    "t_sec": round(t, 6),
                    "source_frame_index": idx,
                    "source_fps": round(native_fps, 6),
                    "sample_fps": sample_fps,
                    "path": repo_rel(out_path),
                    "width": w or width,
                    "height": h or height,
                }
            )
            next_t += interval
        idx += 1

    cap.release()
    if not rows:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return rows


def assign_splits(rows: list[dict], val_fraction: float) -> None:
    by_clip: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["role"] == "train_pool":
            by_clip[row["clip_id"]].append(row)
        else:
            row["split"] = "eval"

    for clip_id, clip_rows in by_clip.items():
        clip_rows.sort(key=lambda r: r["t_sec"])
        n = len(clip_rows)
        n_val = max(1, round(n * val_fraction))
        if n_val >= n:
            n_val = max(1, n - 1)
        split_at = n - n_val
        for i, row in enumerate(clip_rows):
            row["split"] = "val" if i >= split_at else "train"


def assert_leak_wall(rows: list[dict], eval_clip_ids: set[str]) -> None:
    train_paths = {r["path"] for r in rows if r["split"] == "train"}
    val_paths = {r["path"] for r in rows if r["split"] == "val"}
    eval_paths = {r["path"] for r in rows if r["split"] == "eval"}

    leaked = (train_paths | val_paths) & eval_paths
    if leaked:
        raise RuntimeError(f"Eval paths leaked into train/val: {sorted(leaked)[:5]}")

    for row in rows:
        if row["split"] in {"train", "val"} and row["clip_id"] in eval_clip_ids:
            raise RuntimeError(
                f"Eval clip {row['clip_id']} assigned to {row['split']}: {row['path']}"
            )
        if row["role"] == "eval" and row["split"] != "eval":
            raise RuntimeError(f"Eval-role row not in eval split: {row['path']}")
        if row["role"] == "train_pool" and row["split"] not in {"train", "val"}:
            raise RuntimeError(f"Train-pool row missing train/val split: {row['path']}")


def write_split_list(path: Path, rows: list[dict], split: str) -> int:
    selected = [r for r in rows if r["split"] == split]
    selected.sort(key=lambda r: (r["clip_id"], r["t_sec"]))
    path.write_text("".join(r["path"] + "\n" for r in selected))
    return len(selected)


def write_manifest(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "clip_id",
        "role",
        "split",
        "video_file",
        "t_sec",
        "source_frame_index",
        "source_fps",
        "sample_fps",
        "path",
        "width",
        "height",
    ]
    ordered = sorted(rows, key=lambda r: (r["clip_id"], r["t_sec"]))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)


def print_summary(rows: list[dict]) -> None:
    by_clip: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        by_clip[row["clip_id"]][row["split"]] += 1
    print("Frame counts by clip / split")
    for clip_id in sorted(by_clip):
        parts = [f"{split}={by_clip[clip_id][split]}" for split in ("train", "val", "eval") if by_clip[clip_id][split]]
        print(f"  {clip_id}: {', '.join(parts)}")
    n_train = sum(1 for r in rows if r["split"] == "train")
    n_val = sum(1 for r in rows if r["split"] == "val")
    n_eval = sum(1 for r in rows if r["split"] == "eval")
    print(f"  TOTAL: train={n_train} val={n_val} eval={n_eval}")
    print("OK: eval not in train/val")


def load_manifest_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--clips",
        nargs="+",
        default=None,
        help="Only re-extract these clip ids; keep other rows from existing manifest.csv",
    )
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    cfg = load_config(cfg_path)

    videos_dir = REPO_ROOT / cfg["paths"]["videos_dir"]
    frames_dir = REPO_ROOT / cfg["paths"]["frames_dir"]
    splits_dir = REPO_ROOT / cfg["paths"]["splits_dir"]
    sampling = cfg["sampling"]
    split_cfg = cfg["split"]
    ext = sampling.get("image_ext", "jpg")
    jpeg_quality = int(sampling.get("jpeg_quality", 95))

    if split_cfg.get("strategy") != "last_fraction_per_clip":
        raise RuntimeError(f"Unsupported split strategy: {split_cfg.get('strategy')}")

    only = set(args.clips) if args.clips else None
    if only is not None:
        known = {c["id"] for c in cfg["clips"]["train"]} | {c["id"] for c in cfg["clips"]["eval"]}
        unknown = only - known
        if unknown:
            raise RuntimeError(f"Unknown clip ids in --clips: {sorted(unknown)}")

    rows: list[dict] = []
    if only is not None:
        prev = load_manifest_rows(splits_dir / "manifest.csv")
        if not prev:
            raise RuntimeError("manifest.csv missing; run a full extract before --clips")
        for row in prev:
            if row["clip_id"] not in only:
                # Restore numeric fields from CSV strings.
                row["t_sec"] = float(row["t_sec"])
                row["source_frame_index"] = int(row["source_frame_index"])
                row["source_fps"] = float(row["source_fps"])
                row["sample_fps"] = float(row["sample_fps"])
                row["width"] = int(row["width"])
                row["height"] = int(row["height"])
                rows.append(row)
        print(f"Keeping {len(rows)} existing manifest rows for clips outside {sorted(only)}")

    for clip in cfg["clips"]["train"]:
        if only is not None and clip["id"] not in only:
            continue
        video_path = videos_dir / clip["file"]
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        print(f"Extracting train clip {clip['id']} at {sampling['train_fps']} fps from {video_path.name}")
        clip_rows = extract_clip(
            video_path=video_path,
            out_dir=frames_dir / clip["id"],
            clip_id=clip["id"],
            sample_fps=float(sampling["train_fps"]),
            jpeg_quality=jpeg_quality,
            ext=ext,
        )
        for row in clip_rows:
            row["role"] = "train_pool"
        rows.extend(clip_rows)

    eval_clip_ids: set[str] = {c["id"] for c in cfg["clips"]["eval"]}
    for clip in cfg["clips"]["eval"]:
        if only is not None and clip["id"] not in only:
            continue
        video_path = videos_dir / clip["file"]
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        print(f"Extracting eval clip {clip['id']} at {sampling['eval_fps']} fps from {video_path.name}")
        clip_rows = extract_clip(
            video_path=video_path,
            out_dir=frames_dir / clip["id"],
            clip_id=clip["id"],
            sample_fps=float(sampling["eval_fps"]),
            jpeg_quality=jpeg_quality,
            ext=ext,
        )
        for row in clip_rows:
            row["role"] = "eval"
        rows.extend(clip_rows)

    assign_splits(rows, float(split_cfg["val_fraction"]))
    assert_leak_wall(rows, eval_clip_ids)

    splits_dir.mkdir(parents=True, exist_ok=True)
    write_split_list(splits_dir / "train.txt", rows, "train")
    write_split_list(splits_dir / "val.txt", rows, "val")
    write_split_list(splits_dir / "eval.txt", rows, "eval")
    write_manifest(splits_dir / "manifest.csv", rows)

    print_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
