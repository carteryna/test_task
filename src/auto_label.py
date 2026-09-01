#!/usr/bin/env python3
"""Pseudo-label train/val frames with a general detector. Eval is refused."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def read_split_list(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def assert_train_pool_only(
    image_paths: list[str],
    eval_paths: list[str],
    eval_clip_ids: set[str],
) -> None:
    eval_set = set(eval_paths)
    leaked = [p for p in image_paths if p in eval_set]
    if leaked:
        raise RuntimeError(f"Eval paths passed to auto-label: {leaked[:5]}")
    for p in image_paths:
        parts = Path(p).parts
        for clip_id in eval_clip_ids:
            if clip_id in parts:
                raise RuntimeError(f"Eval clip folder in auto-label path: {p}")


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def nms_class_agnostic(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thr: float,
) -> list[int]:
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while len(order):
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = np.array([iou_xyxy(boxes[i], boxes[j]) for j in rest])
        order = rest[ious < iou_thr]
    return keep


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[float, float, float, float]:
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    return cx, cy, bw / w, bh / h


def pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        # Torchvision NMS has no MPS kernel on this stack (torch 2.2 / torchvision 0.17).
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("MPS is present but torchvision.nms is CPU-only here; using cpu")
    except ImportError:
        pass
    return "cpu"


def load_teacher(cfg: dict, device: str):
    """YOLO-World first; COCO YOLO if World weights or API fail."""
    from ultralytics import YOLO

    weights = cfg["weights"]
    prompts = list(cfg["prompts"])
    try:
        try:
            from ultralytics import YOLOWorld

            model = YOLOWorld(weights)
        except Exception:
            model = YOLO(weights)
        if hasattr(model, "set_classes"):
            model.set_classes(prompts)
        backend = "yolo_world"
        print(f"Teacher: YOLO-World {weights} prompts={prompts} device={device}")
        return model, backend, weights
    except Exception as exc:
        print(f"YOLO-World failed ({exc!r}); falling back to COCO {cfg['fallback_weights']}")

    model = YOLO(cfg["fallback_weights"])
    backend = "coco_yolo"
    print(f"Teacher: COCO YOLO {cfg['fallback_weights']} ids={cfg['coco_vehicle_ids']} device={device}")
    return model, backend, cfg["fallback_weights"]


def filter_boxes(
    xyxy: np.ndarray,
    confs: np.ndarray,
    classes: np.ndarray,
    *,
    backend: str,
    coco_ids: set[int],
    min_side: float,
    max_aspect: float,
    iou_thr: float,
    img_w: int,
    img_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    kept_boxes = []
    kept_scores = []
    for box, conf, cls in zip(xyxy, confs, classes):
        if backend == "coco_yolo" and int(cls) not in coco_ids:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        x1 = min(max(x1, 0.0), img_w)
        y1 = min(max(y1, 0.0), img_h)
        x2 = min(max(x2, 0.0), img_w)
        y2 = min(max(y2, 0.0), img_h)
        bw = x2 - x1
        bh = y2 - y1
        if bw < min_side or bh < min_side:
            continue
        aspect = max(bw / bh, bh / bw) if bw > 0 and bh > 0 else 999.0
        if aspect > max_aspect:
            continue
        kept_boxes.append([x1, y1, x2, y2])
        kept_scores.append(float(conf))
    if not kept_boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    boxes = np.asarray(kept_boxes, dtype=np.float32)
    scores = np.asarray(kept_scores, dtype=np.float32)
    keep = nms_class_agnostic(boxes, scores, iou_thr)
    return boxes[keep], scores[keep]


def label_path_for(image_rel: str, labels_dir: Path) -> Path:
    rel = Path(image_rel)
    return labels_dir / rel.parent.name / (rel.stem + ".txt")


def write_yolo_label(path: Path, boxes: np.ndarray, img_w: int, img_h: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for x1, y1, x2, y2 in boxes:
        cx, cy, nw, nh = xyxy_to_yolo(float(x1), float(y1), float(x2), float(y2), img_w, img_h)
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def draw_preview(image_bgr: np.ndarray, boxes: np.ndarray, scores: np.ndarray, out_path: Path) -> None:
    vis = image_bgr.copy()
    for (x1, y1, x2, y2), sc in zip(boxes, scores):
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(vis, p1, p2, (0, 220, 80), 2)
        cv2.putText(vis, f"{sc:.2f}", (p1[0], max(0, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 1)
    h, w = vis.shape[:2]
    max_side = 1280
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 85])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "data.yaml")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Which split lists to label. Default: train val. eval is rejected.",
    )
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    cfg = load_config(cfg_path)

    if "eval" in args.splits:
        raise RuntimeError("Refusing to auto-label eval. That happens after the student is frozen.")

    splits_dir = REPO_ROOT / cfg["paths"]["splits_dir"]
    eval_paths = read_split_list(splits_dir / "eval.txt")
    eval_clip_ids = {c["id"] for c in cfg["clips"]["eval"]}

    image_rels: list[str] = []
    for split in args.splits:
        image_rels.extend(read_split_list(splits_dir / f"{split}.txt"))
    image_rels = list(dict.fromkeys(image_rels))
    assert_train_pool_only(image_rels, eval_paths, eval_clip_ids)

    teacher_cfg = cfg["teacher"]
    device = pick_device()
    model, backend, weights_used = load_teacher(teacher_cfg, device)
    coco_ids = set(int(i) for i in teacher_cfg["coco_vehicle_ids"])

    labels_dir = REPO_ROOT / teacher_cfg["labels_dir"]
    preview_dir = REPO_ROOT / teacher_cfg["preview_dir"]
    boxes_csv_path = labels_dir / "boxes.csv"
    labels_dir.mkdir(parents=True, exist_ok=True)

    per_clip = defaultdict(lambda: {"images": 0, "boxes": 0, "empty": 0})
    best_preview: dict[str, tuple[int, str, np.ndarray, np.ndarray, np.ndarray]] = {}
    all_box_rows: list[dict] = []

    print(f"Labeling {len(image_rels)} images from splits={args.splits}; skipping {len(eval_paths)} eval frames")

    for i, rel in enumerate(image_rels, start=1):
        img_path = REPO_ROOT / rel
        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Failed to read {img_path}")
        h, w = image.shape[:2]
        result = model.predict(
            image,
            conf=float(teacher_cfg["conf"]),
            iou=float(teacher_cfg["iou"]),
            imgsz=int(teacher_cfg["imgsz"]),
            device=device,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            confs = np.zeros((0,), dtype=np.float32)
            clss = np.zeros((0,), dtype=np.float32)
        else:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()

        boxes, scores = filter_boxes(
            xyxy,
            confs,
            clss,
            backend=backend,
            coco_ids=coco_ids,
            min_side=float(teacher_cfg["min_side_px"]),
            max_aspect=float(teacher_cfg["max_aspect"]),
            iou_thr=float(teacher_cfg["iou"]),
            img_w=w,
            img_h=h,
        )
        out_lbl = label_path_for(rel, labels_dir)
        write_yolo_label(out_lbl, boxes, w, h)

        clip_id = Path(rel).parent.name
        per_clip[clip_id]["images"] += 1
        per_clip[clip_id]["boxes"] += int(len(boxes))
        if len(boxes) == 0:
            per_clip[clip_id]["empty"] += 1
        else:
            prev = best_preview.get(clip_id)
            if prev is None or len(boxes) > prev[0]:
                best_preview[clip_id] = (len(boxes), rel, image, boxes, scores)

        for box, sc in zip(boxes, scores):
            all_box_rows.append(
                {
                    "path": rel,
                    "clip_id": clip_id,
                    "conf": f"{float(sc):.4f}",
                    "x1": f"{box[0]:.1f}",
                    "y1": f"{box[1]:.1f}",
                    "x2": f"{box[2]:.1f}",
                    "y2": f"{box[3]:.1f}",
                }
            )

        if i == 1 or i % 20 == 0 or i == len(image_rels):
            print(f"  {i}/{len(image_rels)} {rel} boxes={len(boxes)}")

    for clip_id, payload in best_preview.items():
        _, _rel, image, boxes, scores = payload
        draw_preview(image, boxes, scores, preview_dir / f"{clip_id}.jpg")

    with boxes_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "clip_id", "conf", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(all_box_rows)

    n_images = sum(v["images"] for v in per_clip.values())
    n_boxes = sum(v["boxes"] for v in per_clip.values())
    n_empty = sum(v["empty"] for v in per_clip.values())
    summary = {
        "backend": backend,
        "weights": weights_used,
        "device": device,
        "imgsz": teacher_cfg["imgsz"],
        "conf": teacher_cfg["conf"],
        "splits": args.splits,
        "n_images": n_images,
        "n_boxes": n_boxes,
        "n_empty": n_empty,
        "n_eval_skipped": len(eval_paths),
        "per_clip": dict(per_clip),
        "tracking": "skipped — 2 fps is too sparse for ByteTrack on highway traffic",
    }
    summary_path = REPO_ROOT / teacher_cfg["summary_path"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("Boxes by clip")
    for clip_id in sorted(per_clip):
        s = per_clip[clip_id]
        print(f"  {clip_id}: images={s['images']} boxes={s['boxes']} empty={s['empty']}")
    print(f"  TOTAL: images={n_images} boxes={n_boxes} empty={n_empty}")
    print(f"OK: eval not labeled ({len(eval_paths)} frames skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
