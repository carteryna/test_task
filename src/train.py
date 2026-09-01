#!/usr/bin/env python3
"""Train student detectors on cleaned train/val labels (eval never used).

Builds an Ultralytics dataset from split lists + data/labels/clean, then fine-tunes
YOLOv8n and/or YOLO11n with the backbone frozen (freeze=N) and the head trainable.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def read_split_list(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def pick_device(requested: str) -> str:
    if requested and requested not in {"auto", "cpu"}:
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        # torchvision NMS lacks MPS on this stack — keep students on CPU unless forced.
        if requested == "cpu":
            return "cpu"
    except ImportError:
        pass
    return "cpu"


def prepare_yolo_dataset(
    *,
    data_cfg: dict,
    train_cfg: dict,
    splits_dir: Path,
) -> Path:
    """Symlink images + clean labels into data/yolo/{images,labels}/{train,val}."""
    root = REPO_ROOT / train_cfg["dataset"]["root"]
    labels_clean = REPO_ROOT / data_cfg["cleanup"]["clean_dir"]
    class_name = data_cfg["class_name"]
    class_id = int(data_cfg.get("class_id", 0))

    if root.exists():
        shutil.rmtree(root)

    for split in ("train", "val"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for rel in read_split_list(splits_dir / f"{split}.txt"):
            clip = Path(rel).parent.name
            if clip == "E" or "/E/" in rel:
                raise RuntimeError(f"Eval path in {split}: {rel}")
            src_img = REPO_ROOT / rel
            stem = Path(rel).stem
            src_lbl = labels_clean / clip / f"{stem}.txt"
            if not src_img.exists():
                raise FileNotFoundError(src_img)
            if not src_lbl.exists():
                raise FileNotFoundError(src_lbl)
            # Unique names across clips: A_000000.jpg
            name = f"{clip}_{stem}"
            dst_img = img_dir / f"{name}.jpg"
            dst_lbl = lbl_dir / f"{name}.txt"
            if dst_img.exists() or dst_img.is_symlink():
                dst_img.unlink()
            if dst_lbl.exists() or dst_lbl.is_symlink():
                dst_lbl.unlink()
            dst_img.symlink_to(src_img.resolve())
            dst_lbl.symlink_to(src_lbl.resolve())

    data_yaml = root / "dataset.yaml"
    payload = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {class_id: class_name},
    }
    data_yaml.write_text(yaml.safe_dump(payload, sort_keys=False))
    n_train = len(list((root / "images" / "train").glob("*.jpg")))
    n_val = len(list((root / "images" / "val").glob("*.jpg")))
    print(f"Prepared YOLO dataset at {root.relative_to(REPO_ROOT)}  train={n_train} val={n_val}")
    print("OK: eval not in training dataset")
    return data_yaml


def train_one(model_spec: dict, data_yaml: Path, train_hparams: dict, device: str) -> Path:
    from ultralytics import YOLO

    name = model_spec["name"]
    weights = model_spec["weights"]
    print(f"\n=== Training {name} from {weights} (freeze={train_hparams['freeze']}) device={device} ===")
    model = YOLO(weights)
    results = model.train(
        data=str(data_yaml),
        imgsz=int(train_hparams["imgsz"]),
        epochs=int(train_hparams["epochs"]),
        batch=int(train_hparams["batch"]),
        freeze=int(train_hparams["freeze"]),
        device=device,
        workers=int(train_hparams.get("workers", 0)),
        patience=int(train_hparams.get("patience", 15)),
        seed=int(train_hparams.get("seed", 0)),
        amp=bool(train_hparams.get("amp", True)),
        project=str(REPO_ROOT / train_hparams.get("project", "runs/train")),
        name=name,
        exist_ok=bool(train_hparams.get("exist_ok", True)),
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Best weights: {best}")
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "train.yaml")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model names, e.g. yolov8n yolo11n",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Build dataset.yaml and exit")
    parser.add_argument("--device", default=None, help="Override train.device (cpu/0/mps)")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs (e.g. 3 for smoke)")
    args = parser.parse_args()

    train_cfg_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    train_cfg = load_yaml(train_cfg_path)
    data_cfg_path = REPO_ROOT / train_cfg["data_config"]
    data_cfg = load_yaml(data_cfg_path)
    splits_dir = REPO_ROOT / data_cfg["paths"]["splits_dir"]

    data_yaml = prepare_yolo_dataset(
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        splits_dir=splits_dir,
    )
    if args.prepare_only:
        return 0

    hparams = dict(train_cfg["train"])
    if args.epochs is not None:
        hparams["epochs"] = args.epochs
        print(f"Overriding epochs -> {args.epochs}")

    device = pick_device(args.device or hparams.get("device", "cpu"))
    models = train_cfg["models"]
    if args.models:
        wanted = set(args.models)
        models = [m for m in models if m["name"] in wanted]
        if not models:
            raise RuntimeError(f"No models matched {args.models}")

    best_paths = []
    for spec in models:
        best_paths.append(train_one(spec, data_yaml, hparams, device))

    print("\nTraining complete (val-only model selection; eval clip unused).")
    for p in best_paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
