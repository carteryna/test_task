#!/usr/bin/env python3
"""Band-aware student evaluation (val threshold freeze → one hold-out score).

Leak wall:
  --tune-val   uses val frames + clean labels only; refuses eval clips.
  --score-eval uses eval frames + eval GT only; loads frozen thresholds;
               never re-tunes on hold-out.

Matching: IoU >= iou_match, greedy one-pred-per-GT (preds sorted by conf).
Bands use the same pinhole prior as estimate_distance.py (GT and unmatched
preds both get a band from min(w,h)). FA/min = FP * 60 / N_frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

BANDS = ("near_0_200", "far_200_400")


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


def pick_device(requested: str | None) -> str:
    if requested and requested not in {"auto", "cpu"}:
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("MPS is present but torchvision.nms is CPU-only here; using cpu")
    except ImportError:
        pass
    return "cpu"


def load_yolo_boxes(txt_path: Path) -> list[tuple[float, float, float, float]]:
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


def yolo_to_xyxy(
    boxes: list[tuple[float, float, float, float]],
    img_w: int,
    img_h: int,
) -> np.ndarray:
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    rows = []
    for x, y, w, h in boxes:
        bw, bh = w * img_w, h * img_h
        cx, cy = x * img_w, y * img_h
        rows.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
    return np.asarray(rows, dtype=np.float32)


def focal_px(img_h: int, fov_v_deg: float) -> float:
    return img_h / (2.0 * math.tan(math.radians(fov_v_deg) / 2.0))


def band_for_box(
    xyxy: np.ndarray,
    img_h: int,
    *,
    w_ref: float,
    fov_v: float,
    near_max: float,
    far_max: float,
) -> str:
    w_px = float(xyxy[2] - xyxy[0])
    h_px = float(xyxy[3] - xyxy[1])
    s_px = min(w_px, h_px)
    if s_px <= 1e-6:
        return "failed"
    dist_m = focal_px(img_h, fov_v) * w_ref / s_px
    if dist_m < near_max:
        return "near_0_200"
    if dist_m < far_max:
        return "far_200_400"
    return "beyond_400"


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    out = np.zeros((len(a), len(b)), dtype=np.float32)
    for i, box in enumerate(a):
        ix1 = np.maximum(box[0], b[:, 0])
        iy1 = np.maximum(box[1], b[:, 1])
        ix2 = np.minimum(box[2], b[:, 2])
        iy2 = np.minimum(box[3], b[:, 3])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
        denom = area_a + area_b - inter
        out[i] = np.where(denom > 0, inter / denom, 0.0)
    return out


def greedy_match(
    gt_xyxy: np.ndarray,
    pred_xyxy: np.ndarray,
    pred_scores: np.ndarray,
    iou_thr: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Return (matches gt_i,pred_j), unmatched_gt, unmatched_pred."""
    if len(gt_xyxy) == 0:
        return [], [], list(range(len(pred_xyxy)))
    if len(pred_xyxy) == 0:
        return [], list(range(len(gt_xyxy))), []
    order = np.argsort(-pred_scores)
    ious = iou_matrix(gt_xyxy, pred_xyxy)
    gt_used = set()
    pred_used = set()
    matches: list[tuple[int, int]] = []
    for pj in order:
        pj = int(pj)
        best_i, best_iou = -1, iou_thr
        for gi in range(len(gt_xyxy)):
            if gi in gt_used:
                continue
            iou = float(ious[gi, pj])
            if iou >= best_iou:
                best_iou = iou
                best_i = gi
        if best_i >= 0:
            matches.append((best_i, pj))
            gt_used.add(best_i)
            pred_used.add(pj)
    unmatched_gt = [i for i in range(len(gt_xyxy)) if i not in gt_used]
    unmatched_pred = [j for j in range(len(pred_xyxy)) if j not in pred_used]
    return matches, unmatched_gt, unmatched_pred


def empty_band_counts() -> dict[str, dict[str, int]]:
    return {b: {"tp": 0, "fp": 0, "fn": 0} for b in BANDS}


def metrics_from_counts(counts: dict[str, dict[str, int]], n_frames: int) -> dict:
    out: dict = {}
    for band in BANDS:
        tp = counts[band]["tp"]
        fp = counts[band]["fp"]
        fn = counts[band]["fn"]
        det = tp / (tp + fn) if (tp + fn) else None
        prec = tp / (tp + fp) if (tp + fp) else None
        f1 = None
        if det is not None and prec is not None and (det + prec) > 0:
            f1 = 2 * det * prec / (det + prec)
        out[band] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "n_gt": tp + fn,
            "detection_rate": None if det is None else round(det, 4),
            "precision": None if prec is None else round(prec, 4),
            "f1": None if f1 is None else round(f1, 4),
            "false_alarms_per_min": round(fp * 60.0 / n_frames, 4) if n_frames else None,
        }
    return out


def selection_score(band_metrics: dict) -> float:
    f1s = []
    for band in BANDS:
        f1 = band_metrics[band]["f1"]
        if f1 is None:
            f1s.append(0.0)
        else:
            f1s.append(float(f1))
    return float(sum(f1s) / len(f1s))


def predict_cached(
    *,
    image_rels: list[str],
    weights: Path,
    imgsz: int,
    conf_floor: float,
    nms_iou: float,
    device: str,
) -> dict[str, dict]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    cache: dict[str, dict] = {}
    print(f"Predicting {len(image_rels)} images  weights={weights}  conf>={conf_floor}  imgsz={imgsz}")
    for i, rel in enumerate(image_rels, start=1):
        img_path = REPO_ROOT / rel
        result = model.predict(
            str(img_path),
            conf=conf_floor,
            iou=nms_iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
        else:
            xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            scores = result.boxes.conf.cpu().numpy().astype(np.float32)
        cache[rel] = {"xyxy": xyxy, "scores": scores}
        if i == 1 or i % 20 == 0 or i == len(image_rels):
            print(f"  {i}/{len(image_rels)} {rel} preds={len(scores)}")
    return cache


def score_split(
    *,
    image_rels: list[str],
    labels_dir: Path,
    manifest: dict[str, dict],
    pred_cache: dict[str, dict],
    conf: float,
    iou_match: float,
    dist_cfg: dict,
) -> tuple[dict, dict]:
    """Returns (report, per_frame detail for overlays / TTFD)."""
    w_ref = float(dist_cfg["w_ref_m"])
    fov_v = float(dist_cfg["fov_v_deg"])
    near_max = float(dist_cfg["near_max_m"])
    far_max = float(dist_cfg["far_max_m"])

    counts = empty_band_counts()
    first_tp_t: dict[str, dict[str, float]] = {b: {} for b in BANDS}
    per_frame: list[dict] = []

    for rel in image_rels:
        meta = manifest[rel]
        img_w, img_h = int(meta["width"]), int(meta["height"])
        t_sec = float(meta["t_sec"])
        clip = Path(rel).parent.name
        gt_yolo = load_yolo_boxes(labels_dir / clip / f"{Path(rel).stem}.txt")
        gt_xyxy = yolo_to_xyxy(gt_yolo, img_w, img_h)
        gt_bands = [
            band_for_box(box, img_h, w_ref=w_ref, fov_v=fov_v, near_max=near_max, far_max=far_max)
            for box in gt_xyxy
        ]

        raw = pred_cache[rel]
        keep = raw["scores"] >= conf
        pred_xyxy = raw["xyxy"][keep]
        pred_scores = raw["scores"][keep]
        pred_bands = [
            band_for_box(box, img_h, w_ref=w_ref, fov_v=fov_v, near_max=near_max, far_max=far_max)
            for box in pred_xyxy
        ]

        matches, unmatched_gt, unmatched_pred = greedy_match(
            gt_xyxy, pred_xyxy, pred_scores, iou_match
        )
        frame_tp = []
        frame_fn = []
        frame_fp = []
        for gi, pj in matches:
            band = gt_bands[gi]
            if band in counts:
                counts[band]["tp"] += 1
                if clip not in first_tp_t[band] or t_sec < first_tp_t[band][clip]:
                    first_tp_t[band][clip] = t_sec
            frame_tp.append((gi, pj))
        for gi in unmatched_gt:
            band = gt_bands[gi]
            if band in counts:
                counts[band]["fn"] += 1
            frame_fn.append(gi)
        for pj in unmatched_pred:
            band = pred_bands[pj]
            if band in counts:
                counts[band]["fp"] += 1
            frame_fp.append(pj)

        per_frame.append(
            {
                "path": rel,
                "clip_id": clip,
                "t_sec": t_sec,
                "img_w": img_w,
                "img_h": img_h,
                "gt_xyxy": gt_xyxy,
                "gt_bands": gt_bands,
                "pred_xyxy": pred_xyxy,
                "pred_scores": pred_scores,
                "pred_bands": pred_bands,
                "matches": frame_tp,
                "unmatched_gt": frame_fn,
                "unmatched_pred": frame_fp,
            }
        )

    n_frames = len(image_rels)
    band_metrics = metrics_from_counts(counts, n_frames)
    for band in BANDS:
        per_clip_t = first_tp_t[band]
        if per_clip_t:
            band_metrics[band]["time_to_first_s"] = round(min(per_clip_t.values()), 3)
            band_metrics[band]["time_to_first_per_clip_s"] = {
                k: round(v, 3) for k, v in sorted(per_clip_t.items())
            }
        else:
            band_metrics[band]["time_to_first_s"] = None
            band_metrics[band]["time_to_first_per_clip_s"] = {}

    report = {
        "n_frames": n_frames,
        "conf": conf,
        "iou_match": iou_match,
        "selection_score": round(selection_score(band_metrics), 4),
        "bands": band_metrics,
        "counts": counts,
    }
    return report, {"frames": per_frame, "first_tp_t": first_tp_t}


# BGR
COLOR_GT = (0, 220, 0)
COLOR_FN = (0, 255, 255)
COLOR_TP = (255, 160, 0)
COLOR_FP = (0, 0, 255)


def _resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def _box_style(img: np.ndarray) -> tuple[int, float, int]:
    """Scale line/font thickness with image size so boxes stay obvious to reviewers."""
    short = min(img.shape[:2])
    # Aggressive strokes: ~6 px on 720p panels, ~10 px on 1100–1400 display stills.
    thickness = max(6, int(round(short / 110.0)))
    font_scale = max(0.85, short / 750.0)
    font_thick = max(2, thickness // 2)
    return thickness, font_scale, font_thick


def _draw_boxes(
    img: np.ndarray,
    boxes: np.ndarray,
    *,
    color: tuple[int, int, int],
    labels: list[str] | None = None,
    thickness: int | None = None,
) -> None:
    if thickness is None:
        thickness, font_scale, font_thick = _box_style(img)
    else:
        _, font_scale, font_thick = _box_style(img)
    for i, box in enumerate(boxes):
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[2]), int(box[3]))
        # Dark outline under the colored stroke so thin boxes stay visible on bright asphalt.
        cv2.rectangle(img, p1, p2, (0, 0, 0), thickness + 2, lineType=cv2.LINE_AA)
        cv2.rectangle(img, p1, p2, color, thickness, lineType=cv2.LINE_AA)
        if labels is not None and i < len(labels) and labels[i]:
            tx, ty = p1[0], max(int(18 * font_scale), p1[1] - 6)
            cv2.putText(
                img,
                labels[i],
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                font_thick + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                labels[i],
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                font_thick,
                cv2.LINE_AA,
            )


def _banner(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    _, font_scale, font_thick = _box_style(out)
    bar_h = max(44, int(40 * font_scale))
    cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (12, int(bar_h * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 0.95,
        (255, 255, 255),
        max(2, font_thick),
        cv2.LINE_AA,
    )
    return out


def _legend(img: np.ndarray, lines: list[tuple[tuple[int, int, int], str]]) -> np.ndarray:
    out = img.copy()
    _, font_scale, font_thick = _box_style(out)
    row_h = max(28, int(26 * font_scale))
    swatch = max(18, int(16 * font_scale))
    pad = max(10, int(10 * font_scale))
    width = max(320, int(300 * font_scale))
    height = row_h * len(lines) + pad * 2
    x0, y0 = pad + 4, out.shape[0] - pad - height
    cv2.rectangle(out, (x0 - pad, y0 - pad), (x0 + width, y0 + height - pad), (0, 0, 0), -1)
    cv2.rectangle(out, (x0 - pad, y0 - pad), (x0 + width, y0 + height - pad), (220, 220, 220), 2)
    for i, (color, label) in enumerate(lines):
        y = y0 + pad + i * row_h + row_h // 2
        cv2.rectangle(out, (x0, y - swatch // 2), (x0 + swatch, y + swatch // 2), color, -1)
        cv2.rectangle(out, (x0, y - swatch // 2), (x0 + swatch, y + swatch // 2), (255, 255, 255), 1)
        cv2.putText(
            out,
            label,
            (x0 + swatch + 12, y + int(6 * font_scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.75,
            (240, 240, 240),
            max(2, font_thick - 1),
            cv2.LINE_AA,
        )
    return out


def _load_frame_scaled(fr: dict, max_side: int) -> tuple[np.ndarray, float]:
    """Load frame and scale boxes into the resized image space before drawing."""
    img = cv2.imread(str(REPO_ROOT / fr["path"]))
    if img is None:
        raise RuntimeError(f"Failed to read {fr['path']}")
    h0, w0 = img.shape[:2]
    img = _resize_max(img, max_side)
    h1, w1 = img.shape[:2]
    scale = h1 / h0  # uniform scale
    return img, scale


def _scaled_boxes(boxes: np.ndarray, scale: float) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    return boxes * scale


def render_gt_panel(fr: dict, *, max_side: int = 1280) -> np.ndarray:
    img, scale = _load_frame_scaled(fr, max_side)
    _draw_boxes(img, _scaled_boxes(fr["gt_xyxy"], scale), color=COLOR_GT)
    n_gt = len(fr["gt_xyxy"])
    n_far = sum(1 for b in fr["gt_bands"] if b == "far_200_400")
    img = _banner(img, f"GT  {fr['clip_id']}/{Path(fr['path']).stem}  n={n_gt}  far={n_far}")
    return _legend(img, [(COLOR_GT, "ground truth")])


def render_pred_panel(fr: dict, *, max_side: int = 1280) -> np.ndarray:
    img, scale = _load_frame_scaled(fr, max_side)
    labels = [f"{float(s):.2f}" for s in fr["pred_scores"]]
    _draw_boxes(img, _scaled_boxes(fr["pred_xyxy"], scale), color=COLOR_TP, labels=labels)
    n_pred = len(fr["pred_xyxy"])
    img = _banner(img, f"Pred  {fr['clip_id']}/{Path(fr['path']).stem}  n={n_pred}")
    return _legend(img, [(COLOR_TP, "student prediction")])


def render_combined_panel(fr: dict, *, max_side: int = 1280) -> np.ndarray:
    img, scale = _load_frame_scaled(fr, max_side)
    matched_gt = {gi for gi, _ in fr["matches"]}
    matched_pred = {pj for _, pj in fr["matches"]}
    gt = _scaled_boxes(fr["gt_xyxy"], scale)
    pred = _scaled_boxes(fr["pred_xyxy"], scale)
    for gi, box in enumerate(gt):
        color = COLOR_FN if gi not in matched_gt else COLOR_GT
        _draw_boxes(img, box.reshape(1, 4), color=color)
    for pj, box in enumerate(pred):
        color = COLOR_TP if pj in matched_pred else COLOR_FP
        _draw_boxes(
            img,
            box.reshape(1, 4),
            color=color,
            labels=[f"{float(fr['pred_scores'][pj]):.2f}"],
        )
    n_tp, n_fp, n_fn = len(fr["matches"]), len(fr["unmatched_pred"]), len(fr["unmatched_gt"])
    img = _banner(
        img,
        f"Match  {fr['clip_id']}/{Path(fr['path']).stem}  TP={n_tp} FP={n_fp} FN={n_fn}",
    )
    return _legend(
        img,
        [
            (COLOR_GT, "GT matched"),
            (COLOR_FN, "GT miss (FN)"),
            (COLOR_TP, "pred TP"),
            (COLOR_FP, "pred FP"),
        ],
    )


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # Panels are already drawn at display size; only pad to equal height.
    h = max(left.shape[0], right.shape[0])

    def pad(im: np.ndarray) -> np.ndarray:
        if im.shape[0] == h:
            return im
        top = (h - im.shape[0]) // 2
        bottom = h - im.shape[0] - top
        return cv2.copyMakeBorder(im, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    left, right = pad(left), pad(right)
    gap = np.zeros((h, 12, 3), dtype=np.uint8)
    return np.hstack([left, gap, right])


def frame_scores(fr: dict) -> dict[str, float | int]:
    n_tp = len(fr["matches"])
    n_fp = len(fr["unmatched_pred"])
    n_fn = len(fr["unmatched_gt"])
    n_gt = len(fr["gt_xyxy"])
    n_far_gt = sum(1 for b in fr["gt_bands"] if b == "far_200_400")
    n_far_tp = sum(
        1 for gi, _ in fr["matches"] if fr["gt_bands"][gi] == "far_200_400"
    )
    return {
        "n_tp": n_tp,
        "n_fp": n_fp,
        "n_fn": n_fn,
        "n_gt": n_gt,
        "n_far_gt": n_far_gt,
        "n_far_tp": n_far_tp,
        "n_err": n_fp + n_fn,
    }


def select_example_frames(detail: dict, per_clip: int) -> list[tuple[str, dict]]:
    """Pick a few review-friendly frames per clip (success, far, hard)."""
    by_clip: dict[str, list[dict]] = defaultdict(list)
    for fr in detail["frames"]:
        by_clip[fr["clip_id"]].append(fr)

    selected: list[tuple[str, dict]] = []
    for clip_id, frames in sorted(by_clip.items()):
        scored = [(frame_scores(fr), fr) for fr in frames]
        picks: list[tuple[str, dict]] = []
        used_paths: set[str] = set()

        def take(tag: str, fr: dict) -> None:
            if fr["path"] in used_paths or len(picks) >= per_clip:
                return
            used_paths.add(fr["path"])
            picks.append((tag, fr))

        # 1) Clean near success: many TPs, few errors
        success = sorted(
            scored,
            key=lambda t: (-(t[0]["n_tp"] - 0.5 * t[0]["n_err"]), -t[0]["n_tp"]),
        )
        if success:
            take("success", success[0][1])

        # 2) Best far-band coverage (TP preferred, else most far GT)
        far = sorted(
            scored,
            key=lambda t: (-t[0]["n_far_tp"], -t[0]["n_far_gt"], -t[0]["n_tp"]),
        )
        if far and (far[0][0]["n_far_tp"] > 0 or far[0][0]["n_far_gt"] > 0):
            take("far", far[0][1])

        # 3) Hard case: many FN/FP but still some GT
        hard = sorted(
            scored,
            key=lambda t: (-t[0]["n_err"], -t[0]["n_gt"]),
        )
        for _s, fr in hard:
            if _s["n_gt"] > 0 and _s["n_err"] > 0:
                take("hard", fr)
                break

        # Fill remaining with next-best success frames
        for _s, fr in success:
            take("extra", fr)
            if len(picks) >= per_clip:
                break

        selected.extend(picks)
    return selected


def draw_overlays(
    detail: dict,
    out_dir: Path,
    *,
    max_per_clip: int,
) -> list[str]:
    """Legacy debug overlays: combined match view, error-heavy frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_clip: dict[str, list[dict]] = defaultdict(list)
    for fr in detail["frames"]:
        s = frame_scores(fr)
        by_clip[fr["clip_id"]].append((s["n_err"], s["n_tp"], fr))

    written: list[str] = []
    for clip_id, items in sorted(by_clip.items()):
        items.sort(key=lambda t: (-t[0], -t[1]))
        for _n_err, _n_tp, fr in items[:max_per_clip]:
            img = render_combined_panel(fr, max_side=1280)
            stem = Path(fr["path"]).stem
            out_path = out_dir / f"{clip_id}_{stem}.jpg"
            cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            written.append(str(out_path.relative_to(REPO_ROOT)))
    return written


def export_submission_examples(
    detail: dict,
    out_dir: Path,
    *,
    per_clip: int,
    conf: float,
) -> list[dict]:
    """Curated GT|pred side-by-side + combined panels for the README / GitHub."""
    if out_dir.exists():
        for old in out_dir.glob("*"):
            if old.is_file():
                old.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = select_example_frames(detail, per_clip)
    manifest_rows: list[dict] = []
    for tag, fr in picks:
        clip = fr["clip_id"]
        stem = Path(fr["path"]).stem
        s = frame_scores(fr)
        # Draw after resize so stroke width stays thick for reviewers.
        panel_side = 1100
        gt_panel = render_gt_panel(fr, max_side=panel_side)
        pred_panel = render_pred_panel(fr, max_side=panel_side)
        both = render_combined_panel(fr, max_side=1400)
        sbs = side_by_side(gt_panel, pred_panel)

        sbs_name = f"{clip}_{stem}_{tag}_side_by_side.jpg"
        both_name = f"{clip}_{stem}_{tag}_combined.jpg"
        sbs_path = out_dir / sbs_name
        both_path = out_dir / both_name
        cv2.imwrite(str(sbs_path), sbs, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        cv2.imwrite(str(both_path), both, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        row = {
            "clip_id": clip,
            "tag": tag,
            "frame": fr["path"],
            "t_sec": fr["t_sec"],
            "conf": conf,
            "tp": s["n_tp"],
            "fp": s["n_fp"],
            "fn": s["n_fn"],
            "far_gt": s["n_far_gt"],
            "far_tp": s["n_far_tp"],
            "side_by_side": str(sbs_path.relative_to(REPO_ROOT)),
            "combined": str(both_path.relative_to(REPO_ROOT)),
        }
        manifest_rows.append(row)
        print(
            f"  example {clip}/{stem} [{tag}] "
            f"TP={s['n_tp']} FP={s['n_fp']} FN={s['n_fn']} far_tp={s['n_far_tp']}"
        )

    man_path = out_dir / "manifest.json"
    man_path.write_text(
        json.dumps(
            {
                "role": "submission_examples",
                "conf": conf,
                "n_examples": len(manifest_rows),
                "legend": {
                    "side_by_side": "Left = GT (green). Right = student preds (blue) + conf.",
                    "combined": "GT matched green, GT miss yellow, pred TP blue, pred FP red.",
                },
                "examples": manifest_rows,
            },
            indent=2,
        )
        + "\n"
    )
    return manifest_rows


def print_band_table(title: str, report: dict) -> None:
    print(f"\n{title}  conf={report['conf']}  frames={report['n_frames']}  "
          f"score={report['selection_score']}")
    print(f"{'band':<14} {'TP':>5} {'FP':>5} {'FN':>5} {'Det':>7} {'Prec':>7} "
          f"{'FA/min':>8} {'TTFD_s':>8}")
    for band in BANDS:
        m = report["bands"][band]
        det = "n/a" if m["detection_rate"] is None else f"{m['detection_rate']:.3f}"
        prec = "n/a" if m["precision"] is None else f"{m['precision']:.3f}"
        fa = "n/a" if m["false_alarms_per_min"] is None else f"{m['false_alarms_per_min']:.2f}"
        ttfd = "n/a" if m["time_to_first_s"] is None else f"{m['time_to_first_s']:.2f}"
        print(
            f"{band:<14} {m['tp']:5d} {m['fp']:5d} {m['fn']:5d} {det:>7} {prec:>7} "
            f"{fa:>8} {ttfd:>8}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "data.yaml")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--tune-val",
        action="store_true",
        help="Sweep conf on val only; write thresholds JSON.",
    )
    parser.add_argument(
        "--score-eval",
        action="store_true",
        help="Score hold-out once with frozen thresholds.",
    )
    parser.add_argument(
        "--export-examples",
        action="store_true",
        help="Write curated GT|pred examples under evaluation.examples_dir (implies --score-eval).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Override frozen conf when scoring (does not retune).",
    )
    parser.add_argument(
        "--thresholds-path",
        type=Path,
        default=None,
        help="Override evaluation.thresholds_path (keeps baseline JSON intact for A/B).",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=None,
        help="Override evaluation.metrics_path (keeps baseline JSON intact for A/B).",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=None,
        help="Override evaluation.overlay_dir",
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=None,
        help="Override evaluation.examples_dir",
    )
    args = parser.parse_args()
    if args.export_examples:
        args.score_eval = True
    if not args.tune_val and not args.score_eval:
        parser.error("Pass --tune-val and/or --score-eval / --export-examples")

    cfg_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    cfg = load_config(cfg_path)
    eval_cfg = cfg["evaluation"]
    dist_cfg = cfg["distance"]
    splits_dir = REPO_ROOT / cfg["paths"]["splits_dir"]
    manifest = load_manifest(splits_dir / "manifest.csv")
    eval_ids = {c["id"] for c in cfg["clips"]["eval"]}

    weights = Path(args.weights or eval_cfg["weights"])
    if not weights.is_absolute():
        weights = REPO_ROOT / weights
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights: {weights}")

    def _resolve(path_like: Path | str | None, default: str | Path) -> Path:
        p = Path(path_like if path_like is not None else default)
        return p if p.is_absolute() else REPO_ROOT / p

    device = pick_device(args.device)
    imgsz = int(eval_cfg["imgsz"])
    iou_match = float(eval_cfg["iou_match"])
    nms_iou = float(eval_cfg["nms_iou"])
    conf_floor = float(eval_cfg["predict_conf_floor"])
    conf_grid = [float(c) for c in eval_cfg["conf_grid"]]
    thresholds_path = _resolve(args.thresholds_path, eval_cfg["thresholds_path"])
    metrics_path = _resolve(args.metrics_path, eval_cfg["metrics_path"])
    overlay_dir = _resolve(args.overlay_dir, eval_cfg["overlay_dir"])
    overlay_max = int(eval_cfg.get("overlay_max_per_clip", 4))
    examples_dir = _resolve(args.examples_dir, eval_cfg.get("examples_dir", "outputs/examples"))
    examples_per_clip = int(eval_cfg.get("examples_per_clip", 3))

    frozen = None

    if args.tune_val:
        val_rels = read_split_list(splits_dir / "val.txt")
        leaked = [p for p in val_rels if Path(p).parent.name in eval_ids]
        if leaked:
            raise RuntimeError(f"Eval path in val tune list: {leaked[:3]}")
        labels_dir = REPO_ROOT / cfg["cleanup"]["clean_dir"]
        cache = predict_cached(
            image_rels=val_rels,
            weights=weights,
            imgsz=imgsz,
            conf_floor=conf_floor,
            nms_iou=nms_iou,
            device=device,
        )
        sweep = []
        best = None
        for conf in conf_grid:
            report, _ = score_split(
                image_rels=val_rels,
                labels_dir=labels_dir,
                manifest=manifest,
                pred_cache=cache,
                conf=conf,
                iou_match=iou_match,
                dist_cfg=dist_cfg,
            )
            sweep.append(
                {
                    "conf": conf,
                    "selection_score": report["selection_score"],
                    "bands": report["bands"],
                }
            )
            print_band_table(f"VAL conf={conf:.2f}", report)
            if best is None or report["selection_score"] > best["report"]["selection_score"]:
                best = {"conf": conf, "nms_iou": nms_iou, "report": report}

        assert best is not None
        frozen = {
            "role": "val_threshold_selection",
            "weights": str(weights.relative_to(REPO_ROOT)),
            "imgsz": imgsz,
            "device": device,
            "iou_match": iou_match,
            "nms_iou": nms_iou,
            "predict_conf_floor": conf_floor,
            "conf_grid": conf_grid,
            "selection": eval_cfg.get("selection", "mean_band_f1"),
            "chosen_conf": best["conf"],
            "chosen_nms_iou": best["nms_iou"],
            "val_selection_score": best["report"]["selection_score"],
            "val_bands_at_chosen": best["report"]["bands"],
            "sweep": sweep,
            "n_val_frames": len(val_rels),
            "eval_used": False,
            "note": "Thresholds frozen on val only; hold-out unused for selection.",
        }
        thresholds_path.parent.mkdir(parents=True, exist_ok=True)
        thresholds_path.write_text(json.dumps(frozen, indent=2) + "\n")
        print(f"\nOK: froze conf={best['conf']:.2f} nms_iou={best['nms_iou']} "
              f"(val score={best['report']['selection_score']:.4f})")
        print(f"Wrote {thresholds_path.relative_to(REPO_ROOT)}")

    if args.score_eval:
        if frozen is None:
            if not thresholds_path.exists():
                raise RuntimeError(
                    f"Missing {thresholds_path}; run --tune-val first or pass --conf."
                )
            frozen = json.loads(thresholds_path.read_text())
        conf = float(args.conf if args.conf is not None else frozen["chosen_conf"])
        nms_iou = float(frozen.get("chosen_nms_iou", nms_iou))

        eval_rels = read_split_list(splits_dir / "eval.txt")
        bad = [p for p in eval_rels if Path(p).parent.name not in eval_ids]
        if bad:
            raise RuntimeError(f"Non-eval path in eval score list: {bad[:3]}")
        labels_dir = REPO_ROOT / cfg["eval_gt"]["labels_dir"]
        cache = predict_cached(
            image_rels=eval_rels,
            weights=weights,
            imgsz=imgsz,
            conf_floor=min(conf_floor, conf),
            nms_iou=nms_iou,
            device=device,
        )
        report, detail = score_split(
            image_rels=eval_rels,
            labels_dir=labels_dir,
            manifest=manifest,
            pred_cache=cache,
            conf=conf,
            iou_match=iou_match,
            dist_cfg=dist_cfg,
        )
        print_band_table("EVAL (hold-out)", report)
        overlays = draw_overlays(detail, overlay_dir, max_per_clip=overlay_max)
        example_rows: list[dict] = []
        if args.export_examples:
            print(f"\nExporting submission examples → {examples_dir.relative_to(REPO_ROOT)}")
            example_rows = export_submission_examples(
                detail,
                examples_dir,
                per_clip=examples_per_clip,
                conf=conf,
            )

        # Per-clip breakdown
        by_clip_rels: dict[str, list[str]] = defaultdict(list)
        for rel in eval_rels:
            by_clip_rels[Path(rel).parent.name].append(rel)
        per_clip = {}
        for clip_id, rels in sorted(by_clip_rels.items()):
            sub_cache = {r: cache[r] for r in rels}
            clip_report, _ = score_split(
                image_rels=rels,
                labels_dir=labels_dir,
                manifest=manifest,
                pred_cache=sub_cache,
                conf=conf,
                iou_match=iou_match,
                dist_cfg=dist_cfg,
            )
            per_clip[clip_id] = clip_report
            print_band_table(f"EVAL clip {clip_id}", clip_report)

        payload = {
            "role": "holdout_eval",
            "weights": str(weights.relative_to(REPO_ROOT)),
            "imgsz": imgsz,
            "device": device,
            "conf": conf,
            "nms_iou": nms_iou,
            "iou_match": iou_match,
            "thresholds_path": str(thresholds_path.relative_to(REPO_ROOT)),
            "thresholds_source": {
                "chosen_conf": frozen.get("chosen_conf"),
                "val_selection_score": frozen.get("val_selection_score"),
                "eval_used_for_selection": frozen.get("eval_used", False),
            },
            "n_frames": report["n_frames"],
            "bands": report["bands"],
            "selection_score": report["selection_score"],
            "per_clip": {
                cid: {"n_frames": r["n_frames"], "bands": r["bands"]}
                for cid, r in per_clip.items()
            },
            "overlays": overlays,
            "examples": example_rows,
            "examples_dir": str(examples_dir.relative_to(REPO_ROOT)),
            "distance_priors": {
                "w_ref_m": dist_cfg["w_ref_m"],
                "fov_v_deg": dist_cfg["fov_v_deg"],
                "size_side": dist_cfg.get("size_side", "min"),
            },
            "note": (
                "IoU>=0.5 greedy match; TP/FN by GT band; FP by predicted-box band; "
                "FA/min = FP*60/N_frames; TTFD = min in-clip t_sec with a TP (per band)."
            ),
        }
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nWrote {metrics_path.relative_to(REPO_ROOT)}")
        print(f"Wrote {len(overlays)} overlays under {overlay_dir.relative_to(REPO_ROOT)}")
        print(f"Wrote {len(example_rows)} curated examples under {examples_dir.relative_to(REPO_ROOT)}")
        print("OK: hold-out scored once; thresholds were not fit on eval")
    return 0


if __name__ == "__main__":
    sys.exit(main())
