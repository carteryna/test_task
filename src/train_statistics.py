#!/usr/bin/env python3
"""Snapshot student-train metrics from gitignored runs/ into a committed JSON.

Reads Ultralytics results.csv + args.yaml for each run under runs/train/.
Eval clip is never involved; these numbers are val-split only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _float(v: str) -> float:
    return float(v.strip())


def count_boxes_from_distance_csv(csv_path: Path) -> dict:
    train_n = val_n = 0
    if not csv_path.exists():
        return {"train_boxes": None, "val_boxes": None}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row.get("split") == "train":
                train_n += 1
            elif row.get("split") == "val":
                val_n += 1
    return {"train_boxes": train_n, "val_boxes": val_n}


def parse_run(run_dir: Path) -> dict:
    results_csv = run_dir / "results.csv"
    args_yaml = run_dir / "args.yaml"
    if not results_csv.exists():
        raise FileNotFoundError(results_csv)

    epochs: list[dict] = []
    with results_csv.open() as f:
        for row in csv.DictReader(f):
            row = {k.strip(): v.strip() for k, v in row.items()}
            epochs.append(
                {
                    "epoch": int(float(row["epoch"])),
                    "time_s": round(_float(row["time"]), 1),
                    "train_box_loss": round(_float(row["train/box_loss"]), 4),
                    "train_cls_loss": round(_float(row["train/cls_loss"]), 4),
                    "train_dfl_loss": round(_float(row["train/dfl_loss"]), 4),
                    "precision": round(_float(row["metrics/precision(B)"]), 4),
                    "recall": round(_float(row["metrics/recall(B)"]), 4),
                    "map50": round(_float(row["metrics/mAP50(B)"]), 4),
                    "map50_95": round(_float(row["metrics/mAP50-95(B)"]), 4),
                }
            )

    best = max(epochs, key=lambda e: e["map50"]) if epochs else None
    last = epochs[-1] if epochs else None
    hparams: dict = {}
    if args_yaml.exists():
        with args_yaml.open() as f:
            raw = yaml.safe_load(f) or {}
        for key in (
            "model",
            "epochs",
            "batch",
            "imgsz",
            "freeze",
            "device",
            "seed",
            "patience",
        ):
            if key in raw:
                hparams[key] = raw[key]

    weights = run_dir / "weights" / "best.pt"
    return {
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "hparams": hparams,
        "n_epochs_logged": len(epochs),
        "epochs": epochs,
        "best_by_map50": best,
        "last": last,
        "best_weights": str(weights.relative_to(REPO_ROOT)) if weights.exists() else None,
        "note": "Val-only Ultralytics metrics. Not eval-clip scores. Smoke runs are not frozen students.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO_ROOT / "runs" / "train",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "splits" / "train_smoke.json",
    )
    args = parser.parse_args()
    runs_root = args.runs_root if args.runs_root.is_absolute() else REPO_ROOT / args.runs_root
    if not runs_root.exists():
        raise FileNotFoundError(f"No train runs at {runs_root}")

    runs = []
    for child in sorted(runs_root.iterdir()):
        if child.is_dir() and (child / "results.csv").exists():
            runs.append(parse_run(child))

    if not runs:
        raise RuntimeError(f"No results.csv under {runs_root}")

    yolo_root = REPO_ROOT / "data" / "yolo"
    train_dir = yolo_root / "images" / "train"
    val_dir = yolo_root / "images" / "val"
    n_train = len(list(train_dir.glob("*.jpg"))) if train_dir.exists() else None
    n_val = len(list(val_dir.glob("*.jpg"))) if val_dir.exists() else None
    box_counts = count_boxes_from_distance_csv(REPO_ROOT / "data" / "splits" / "distance_boxes.csv")
    payload = {
        "dataset": {
            "yolo_root": "data/yolo",
            "train_images": n_train,
            "val_images": n_val,
            "train_boxes": box_counts["train_boxes"],
            "val_boxes": box_counts["val_boxes"],
            "eval_used": False,
        },
        "runs": runs,
    }
    out_path = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Runs: {len(runs)}")
    for run in runs:
        name = Path(run["run_dir"]).name
        best = run["best_by_map50"] or {}
        print(
            f"  {name}: epochs={run['n_epochs_logged']}  "
            f"best mAP50={best.get('map50')} P={best.get('precision')} "
            f"R={best.get('recall')} (epoch {best.get('epoch')})"
        )
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
