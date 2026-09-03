#!/usr/bin/env python3
"""Zero-human data factory: Grounding DINO -> SAM -> tight boxes -> rules -> YOLO labels.

Stage 1 (inference, cached under runs/cache/dino_sam):
  Grounding DINO runs the text prompt "vehicle." over 2x2 overlapping tiles plus
  the full frame, so far vehicles survive the ~800 px letterbox on 4K footage.
  Surviving boxes prompt SAM, and each mask collapses back to a tight box.

Stage 2 (rules, cheap to re-run):
  Schema enforcement (motorcycle aspect purge, articulated-truck merge, size
  guards) then a kinematic pass that drops static tracks -- the same failure
  modes the hold-out taxonomy found in the YOLO-World labels.

Train/val clips only; the eval hold-out is refused (see assert_train_pool_only).
With --allow-eval --splits eval the hold-out is labeled into a separate directory
as an audit of the teacher's proxy GT: it never overwrites data/labels/eval and
never reaches training (train.py builds from train.txt/val.txt and rejects eval).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import auto_label as al  # noqa: E402
import evaluate_custom as ev  # noqa: E402
from error_analysis import IoUTracker  # noqa: E402

REPO_ROOT = SRC_DIR.parent

DROP_COLORS = {
    "min_side": (120, 120, 120),
    "max_dim": (200, 0, 200),
    "max_aspect": (0, 0, 255),
    "suspected_motorcycle": (255, 0, 200),
    "static_track": (255, 200, 0),
    "short_track": (100, 100, 255),
}


# --------------------------------------------------------------------------- geometry


def tile_windows(
    width: int,
    height: int,
    *,
    rows: int,
    cols: int,
    overlap: float,
    full_frame: bool,
) -> list[tuple[int, int, int, int]]:
    """Overlapping xyxy windows over the frame, optionally plus the full frame."""
    windows: list[tuple[int, int, int, int]] = []
    step_x = width / cols
    step_y = height / rows
    pad_x = step_x * overlap
    pad_y = step_y * overlap
    for r in range(rows):
        for c in range(cols):
            x1 = int(max(0.0, c * step_x - pad_x))
            y1 = int(max(0.0, r * step_y - pad_y))
            x2 = int(min(float(width), (c + 1) * step_x + pad_x))
            y2 = int(min(float(height), (r + 1) * step_y + pad_y))
            if x2 - x1 >= 32 and y2 - y1 >= 32:
                windows.append((x1, y1, x2, y2))
    if full_frame or not windows:
        windows.append((0, 0, width, height))
    return windows


def rel_to_repo(path: Path) -> Path:
    try:
        return path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return path


def box_aspect(box) -> float:
    bw = float(box[2]) - float(box[0])
    bh = float(box[3]) - float(box[1])
    if bw <= 0 or bh <= 0:
        return 999.0
    return max(bw / bh, bh / bw)


def box_area(box) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def containment(a, b) -> float:
    """Intersection over the area of the smaller box."""
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    small = min(box_area(a), box_area(b))
    return float(inter / small) if small > 0 else 0.0


# --------------------------------------------------------------------------- models


def pick_torch_device(requested: str | None) -> str:
    import torch

    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    # SAM/DINO both hit unimplemented MPS kernels on torch 2.2; CPU is the safe path.
    return "cpu"


class DinoSam:
    """Grounding DINO (open-vocabulary boxes) chained into SAM (masks)."""

    def __init__(self, dino_cfg: dict, sam_cfg: dict, device: str) -> None:
        import torch
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            SamModel,
            SamProcessor,
        )

        self.torch = torch
        self.device = device
        self.dino_cfg = dino_cfg
        self.sam_cfg = sam_cfg

        print(f"Loading Grounding DINO {dino_cfg['model_id']} on {device}")
        self.dino_processor = AutoProcessor.from_pretrained(dino_cfg["model_id"])
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(dino_cfg["model_id"])
        self.dino.to(device).eval()

        print(f"Loading SAM {sam_cfg['model_id']} on {device}")
        self.sam_processor = SamProcessor.from_pretrained(sam_cfg["model_id"])
        self.sam = SamModel.from_pretrained(sam_cfg["model_id"])
        self.sam.to(device).eval()

    def detect(self, image_rgb: np.ndarray, windows: list[tuple[int, int, int, int]]) -> list[dict]:
        """Prompt DINO on every window; returns boxes in full-frame coordinates."""
        prompt = self.dino_cfg["prompt"]
        batch_size = int(self.dino_cfg.get("batch_size", 2))
        out: list[dict] = []
        for start in range(0, len(windows), batch_size):
            chunk = windows[start : start + batch_size]
            crops = [image_rgb[y1:y2, x1:x2] for (x1, y1, x2, y2) in chunk]
            inputs = self.dino_processor(
                images=crops,
                text=[prompt] * len(crops),
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            with self.torch.no_grad():
                outputs = self.dino(**inputs)
            results = self.dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=float(self.dino_cfg["box_threshold"]),
                text_threshold=float(self.dino_cfg["text_threshold"]),
                target_sizes=[c.shape[:2] for c in crops],
            )
            for (x1, y1, _x2, _y2), res in zip(chunk, results):
                boxes = res["boxes"].cpu().numpy()
                scores = res["scores"].cpu().numpy()
                for box, score in zip(boxes, scores):
                    out.append(
                        {
                            "xyxy": [
                                float(box[0]) + x1,
                                float(box[1]) + y1,
                                float(box[2]) + x1,
                                float(box[3]) + y1,
                            ],
                            "score": float(score),
                            "window": [int(v) for v in (x1, y1, _x2, _y2)],
                        }
                    )
        return out

    def segment(
        self,
        image_rgb: np.ndarray,
        window: tuple[int, int, int, int],
        boxes_global: np.ndarray,
    ) -> list[np.ndarray | None]:
        """One SAM image-encoder pass per window; box prompts share the embedding."""
        x1, y1, x2, y2 = window
        crop = image_rgb[y1:y2, x1:x2]
        local = boxes_global.copy()
        local[:, [0, 2]] -= x1
        local[:, [1, 3]] -= y1
        local[:, [0, 2]] = local[:, [0, 2]].clip(0, x2 - x1 - 1)
        local[:, [1, 3]] = local[:, [1, 3]].clip(0, y2 - y1 - 1)

        inputs = self.sam_processor(crop, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            embeddings = self.sam.get_image_embeddings(inputs["pixel_values"])

        masks: list[np.ndarray | None] = []
        chunk_size = int(self.sam_cfg.get("boxes_per_chunk", 16))
        for start in range(0, len(local), chunk_size):
            chunk = [[float(v) for v in b] for b in local[start : start + chunk_size]]
            prompt_inputs = self.sam_processor(
                crop, input_boxes=[chunk], return_tensors="pt"
            ).to(self.device)
            with self.torch.no_grad():
                outputs = self.sam(
                    input_boxes=prompt_inputs["input_boxes"],
                    image_embeddings=embeddings,
                    multimask_output=False,
                )
            processed = self.sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                prompt_inputs["original_sizes"].cpu(),
                prompt_inputs["reshaped_input_sizes"].cpu(),
            )[0]
            for m in processed:
                masks.append(m[0].numpy().astype(np.uint8))
        return masks


# --------------------------------------------------------------------------- mask -> box


def mask_to_box(
    mask: np.ndarray,
    prompt_local: np.ndarray,
    *,
    pad_frac: float,
    eps_frac: float,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Tight box + contour of the mask component that answers the prompt box."""
    h, w = mask.shape[:2]
    px1, py1, px2, py2 = [float(v) for v in prompt_local]
    pad_x = (px2 - px1) * pad_frac
    pad_y = (py2 - py1) * pad_frac
    wx1 = int(max(0, np.floor(px1 - pad_x)))
    wy1 = int(max(0, np.floor(py1 - pad_y)))
    wx2 = int(min(w, np.ceil(px2 + pad_x)))
    wy2 = int(min(h, np.ceil(py2 + pad_y)))
    if wx2 - wx1 < 2 or wy2 - wy1 < 2:
        return None

    # SAM happily bleeds into road or shadow; only the prompt neighbourhood counts.
    window = np.zeros_like(mask)
    window[wy1:wy2, wx1:wx2] = mask[wy1:wy2, wx1:wx2]
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(window, connectivity=8)
    if n_labels <= 1:
        return None

    prompt_box = np.array([px1, py1, px2, py2], dtype=np.float32)
    best = None
    for idx in range(1, n_labels):
        x, y, cw, ch, area = stats[idx]
        comp_box = np.array([x, y, x + cw, y + ch], dtype=np.float32)
        overlap = containment(comp_box, prompt_box)
        score = float(area) * (1.0 + overlap)
        if best is None or score > best[0]:
            best = (score, idx, comp_box, float(area))
    if best is None:
        return None
    _score, idx, comp_box, area = best

    component = (labels == idx).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = np.zeros((0, 2), dtype=np.int32)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        eps = eps_frac * cv2.arcLength(largest, True)
        contour = cv2.approxPolyDP(largest, eps, True).reshape(-1, 2)
    return comp_box, contour, area


def refine_with_sam(
    engine: DinoSam,
    image_rgb: np.ndarray,
    dets: list[dict],
    windows: list[tuple[int, int, int, int]],
    sam_cfg: dict,
) -> list[dict]:
    """Group boxes by the window that best contains them, then segment per window."""
    if not dets:
        return []
    full = (0, 0, image_rgb.shape[1], image_rgb.shape[0])
    groups: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for i, det in enumerate(dets):
        box = det["xyxy"]
        best_win = None
        best_margin = -1.0
        for win in windows:
            if box[0] >= win[0] and box[1] >= win[1] and box[2] <= win[2] and box[3] <= win[3]:
                margin = min(
                    box[0] - win[0], box[1] - win[1], win[2] - box[2], win[3] - box[3]
                )
                if margin > best_margin:
                    best_margin = margin
                    best_win = win
        groups[best_win or full].append(i)

    refined: list[dict] = [dict(d) for d in dets]
    for win, idxs in groups.items():
        boxes = np.array([dets[i]["xyxy"] for i in idxs], dtype=np.float32)
        masks = engine.segment(image_rgb, win, boxes)
        for slot, i in enumerate(idxs):
            det = refined[i]
            det["prompt_xyxy"] = list(det["xyxy"])
            det["source"] = "dino_fallback"
            det["mask_area"] = 0.0
            det["contour"] = []
            if slot >= len(masks) or masks[slot] is None:
                continue
            local = boxes[slot].copy()
            local[[0, 2]] -= win[0]
            local[[1, 3]] -= win[1]
            out = mask_to_box(
                masks[slot],
                local,
                pad_frac=float(sam_cfg["box_pad_frac"]),
                eps_frac=float(sam_cfg["contour_eps_frac"]),
            )
            if out is None:
                continue
            comp_box, contour, area = out
            cand = np.array(
                [
                    comp_box[0] + win[0],
                    comp_box[1] + win[1],
                    comp_box[2] + win[0],
                    comp_box[3] + win[1],
                ],
                dtype=np.float32,
            )
            prompt = np.array(det["prompt_xyxy"], dtype=np.float32)
            area_ratio = box_area(cand) / max(box_area(prompt), 1e-6)
            if not (
                float(sam_cfg["min_area_frac"]) <= area_ratio <= float(sam_cfg["max_area_frac"])
            ):
                continue
            if al.iou_xyxy(cand, prompt) < float(sam_cfg["min_iou"]):
                continue
            det["xyxy"] = [float(v) for v in cand]
            det["source"] = "sam"
            det["mask_area"] = float(area)
            det["contour"] = [[int(p[0]) + win[0], int(p[1]) + win[1]] for p in contour]
    return refined


# --------------------------------------------------------------------------- schema rules


def enforce_schema(
    dets: list[dict], width: int, height: int, cfg: dict
) -> tuple[list[dict], list[dict], int]:
    """Size/aspect purge + articulated-truck merge. Returns (kept, dropped)."""
    min_side = float(cfg["min_side_px"])
    max_dim = float(cfg["max_dim_frac"])
    max_aspect = float(cfg["max_aspect"])
    moto_aspect = float(cfg["motorcycle_aspect"])
    moto_side = float(cfg["motorcycle_max_side_px"])

    kept: list[dict] = []
    dropped: list[dict] = []
    for det in dets:
        x1, y1, x2, y2 = [float(v) for v in det["xyxy"]]
        x1 = min(max(x1, 0.0), width)
        y1 = min(max(y1, 0.0), height)
        x2 = min(max(x2, 0.0), width)
        y2 = min(max(y2, 0.0), height)
        det = dict(det, xyxy=[x1, y1, x2, y2])
        bw, bh = x2 - x1, y2 - y1
        aspect = box_aspect(det["xyxy"])
        if bw < min_side or bh < min_side:
            dropped.append(dict(det, drop_reason="min_side"))
            continue
        if bw > max_dim * width or bh > max_dim * height:
            dropped.append(dict(det, drop_reason="max_dim"))
            continue
        # Elongated *and* small: the motorcycle signature. Long trucks stay (big side).
        if aspect >= moto_aspect and min(bw, bh) <= moto_side:
            dropped.append(dict(det, drop_reason="suspected_motorcycle"))
            continue
        if aspect > max_aspect:
            dropped.append(dict(det, drop_reason="max_aspect"))
            continue
        kept.append(det)

    merged, n_merges = merge_fragments(kept, cfg)
    for det in merged:
        det.setdefault("merged_parts", 1)
    return merged, dropped, n_merges


def merge_fragments(dets: list[dict], cfg: dict) -> tuple[list[dict], int]:
    """Union cab/trailer fragments that overlap, while the union stays truck-shaped."""
    merge_iou = float(cfg["merge_iou"])
    merge_cont = float(cfg["merge_containment"])
    merge_aspect = float(cfg["merge_max_aspect"])
    max_growth = float(cfg["merge_max_growth"])
    min_aspect = float(cfg["merge_min_aspect"])
    min_side = float(cfg["merge_min_side_px"])
    items = [dict(d, merged_parts=d.get("merged_parts", 1)) for d in dets]
    n_merges = 0
    changed = True
    while changed and len(items) > 1:
        changed = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a = np.asarray(items[i]["xyxy"], dtype=np.float32)
                b = np.asarray(items[j]["xyxy"], dtype=np.float32)
                if al.iou_xyxy(a, b) < merge_iou and containment(a, b) < merge_cont:
                    continue
                union = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                # Only articulated-vehicle geometry qualifies: big and elongated.
                # Without this, overlapping FP clusters on gantries and poles merge too.
                if not (min_aspect <= box_aspect(union) <= merge_aspect):
                    continue
                sides = [min(a[2] - a[0], a[3] - a[1]), min(b[2] - b[0], b[3] - b[1])]
                if max(sides) < min_side:
                    continue
                # A fragment sits inside its parent, so the union barely grows.
                # Two neighbouring vehicles would balloon it: that is not a truck.
                if box_area(union) > max_growth * max(box_area(a), box_area(b)):
                    continue
                keep = items[i] if items[i]["score"] >= items[j]["score"] else items[j]
                merged = dict(keep)
                merged["xyxy"] = [float(v) for v in union]
                merged["score"] = max(items[i]["score"], items[j]["score"])
                merged["merged_parts"] = items[i]["merged_parts"] + items[j]["merged_parts"]
                merged["source"] = "merged"
                merged["contour"] = keep.get("contour", [])
                items = [it for k, it in enumerate(items) if k not in (i, j)] + [merged]
                n_merges += 1
                changed = True
                break
            if changed:
                break
    return items, n_merges


def purge_static_tracks(
    per_frame: list[dict],
    width: int,
    cfg: dict,
) -> tuple[dict[int, set[int]], dict[tuple[int, int], str], dict]:
    """Track one clip through time; flag boxes on stationary or flickering tracks."""
    tracker_cfg = cfg["tracker"]
    tracker = IoUTracker(
        high_conf=float(tracker_cfg["high_conf"]),
        iou_high=float(tracker_cfg["iou_high"]),
        iou_low=float(tracker_cfg["iou_low"]),
        max_age=int(tracker_cfg["max_age"]),
    )
    move_budget = float(cfg["static_move_frac"]) * width
    min_frames = int(cfg["static_min_frames"])
    min_hits = int(cfg["min_track_hits"])

    history: dict[int, list[tuple[int, int, tuple[float, float]]]] = defaultdict(list)
    for f_idx, frame in enumerate(per_frame):
        boxes = np.array([d["xyxy"] for d in frame["dets"]], dtype=np.float32).reshape(-1, 4)
        scores = np.array([d["score"] for d in frame["dets"]], dtype=np.float32)
        track_ids = tracker.update(f_idx, boxes, scores)
        for det_i, tid in enumerate(track_ids):
            cx = (boxes[det_i][0] + boxes[det_i][2]) / 2.0
            cy = (boxes[det_i][1] + boxes[det_i][3]) / 2.0
            history[tid].append((f_idx, det_i, (cx, cy)))

    drops: dict[int, set[int]] = defaultdict(set)
    reasons: dict[tuple[int, int], str] = {}
    n_static_tracks = 0
    n_short_tracks = 0
    for tid, entries in history.items():
        if len(entries) < min_hits:
            n_short_tracks += 1
            for f_idx, det_i, _c in entries:
                drops[f_idx].add(det_i)
                reasons[(f_idx, det_i)] = "short_track"
            continue
        if len(entries) < min_frames:
            continue
        xs = np.array([c[0] for _f, _d, c in entries])
        ys = np.array([c[1] for _f, _d, c in entries])
        drift = float(max(xs.max() - xs.min(), ys.max() - ys.min()))
        if drift < move_budget:
            n_static_tracks += 1
            for f_idx, det_i, _c in entries:
                drops[f_idx].add(det_i)
                reasons[(f_idx, det_i)] = "static_track"

    stats = {
        "tracks": len(history),
        "static_tracks": n_static_tracks,
        "short_tracks": n_short_tracks,
        "move_budget_px": round(move_budget, 2),
        "static_min_frames": min_frames,
    }
    return drops, reasons, stats


# --------------------------------------------------------------------------- cache


def cache_key(cfg: dict) -> dict:
    return {
        "dino": cfg["dino"],
        "tiling": cfg["tiling"],
        "sam": cfg["sam"],
        "version": 1,
    }


def cache_path(cache_dir: Path, image_rel: str) -> Path:
    rel = Path(image_rel)
    return cache_dir / rel.parent.name / (rel.stem + ".json")


def load_frame_cache(path: Path, key: dict) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if payload.get("key") != key:
        return None
    return payload


# --------------------------------------------------------------------------- previews


def draw_preview(image_bgr: np.ndarray, frame: dict, out_path: Path) -> None:
    vis = image_bgr.copy()
    h, w = vis.shape[:2]
    thickness = max(2, int(round(max(h, w) / 700)))
    font_scale = max(0.5, max(h, w) / 2200.0)

    for det in frame.get("dropped", []):
        color = DROP_COLORS.get(det.get("drop_reason", ""), (0, 0, 255))
        x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            vis,
            det.get("drop_reason", "drop"),
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    for det in frame["dets"]:
        prompt = det.get("prompt_xyxy")
        if prompt:
            p = [int(v) for v in prompt]
            cv2.rectangle(vis, (p[0], p[1]), (p[2], p[3]), (0, 200, 255), max(1, thickness - 1))
        contour = det.get("contour") or []
        if len(contour) >= 3:
            cv2.polylines(
                vis, [np.array(contour, dtype=np.int32)], True, (255, 255, 0), max(1, thickness - 1)
            )
        x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 0), thickness + 2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 230, 80), thickness)
        tag = f"{det['score']:.2f}"
        if det.get("merged_parts", 1) > 1:
            tag += f" merged x{det['merged_parts']}"
        cv2.putText(
            vis,
            tag,
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 230, 80),
            thickness,
            cv2.LINE_AA,
        )

    legend = [
        ("yellow = DINO prompt box", (0, 200, 255)),
        ("cyan = SAM mask contour", (255, 255, 0)),
        ("green = final label", (0, 230, 80)),
        ("red/magenta = dropped by rules", (0, 0, 255)),
    ]
    y = int(30 * font_scale) + 10
    for text, color in legend:
        cv2.putText(
            vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA
        )
        cv2.putText(
            vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA
        )
        y += int(36 * font_scale) + 8

    max_side = 1600
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])


# --------------------------------------------------------------------------- stages


def run_inference(
    image_rels: list[str],
    cfg: dict,
    *,
    cache_dir: Path,
    device: str,
    refresh: bool,
) -> tuple[list[dict], dict]:
    """Stage 1: DINO + SAM per frame, memoised on disk."""
    key = cache_key(cfg)
    frames: list[dict] = []
    todo = []
    for rel in image_rels:
        cached = None if refresh else load_frame_cache(cache_path(cache_dir, rel), key)
        if cached is None:
            todo.append(rel)
        frames.append({"path": rel, "cached": cached})

    timing = {"n_cached": len(image_rels) - len(todo), "n_computed": len(todo)}
    engine = None
    if todo:
        engine = DinoSam(cfg["dino"], cfg["sam"], device)

    t_start = time.time()
    for i, entry in enumerate(frames, start=1):
        rel = entry["path"]
        if entry["cached"] is not None:
            continue
        img_bgr = cv2.imread(str(REPO_ROOT / rel))
        if img_bgr is None:
            raise RuntimeError(f"Failed to read {rel}")
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        windows = tile_windows(
            w,
            h,
            rows=int(cfg["tiling"]["rows"]),
            cols=int(cfg["tiling"]["cols"]),
            overlap=float(cfg["tiling"]["overlap"]),
            full_frame=bool(cfg["tiling"]["full_frame_pass"]),
        )

        t0 = time.time()
        dets = engine.detect(img_rgb, windows)
        t_dino = time.time() - t0

        if dets:
            boxes = np.array([d["xyxy"] for d in dets], dtype=np.float32)
            scores = np.array([d["score"] for d in dets], dtype=np.float32)
            keep = al.nms_class_agnostic(boxes, scores, float(cfg["tiling"]["merge_iou"]))
            dets = [dets[k] for k in keep]

        t1 = time.time()
        refined = refine_with_sam(engine, img_rgb, dets, windows, cfg["sam"])
        t_sam = time.time() - t1

        payload = {
            "key": key,
            "path": rel,
            "width": w,
            "height": h,
            "n_windows": len(windows),
            "dets": [
                {
                    "xyxy": [round(float(v), 2) for v in d["xyxy"]],
                    "prompt_xyxy": [round(float(v), 2) for v in d["prompt_xyxy"]],
                    "score": round(float(d["score"]), 4),
                    "source": d["source"],
                    "mask_area": round(float(d["mask_area"]), 1),
                    "contour": d.get("contour", []),
                }
                for d in refined
            ],
            "timing": {"dino_s": round(t_dino, 2), "sam_s": round(t_sam, 2)},
        }
        out_path = cache_path(cache_dir, rel)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload) + "\n")
        entry["cached"] = payload
        n_sam = sum(1 for d in payload["dets"] if d["source"] == "sam")
        print(
            f"  [{i}/{len(frames)}] {rel} boxes={len(payload['dets'])} sam_refined={n_sam} "
            f"dino={t_dino:.1f}s sam={t_sam:.1f}s"
        )

    timing["inference_s"] = round(time.time() - t_start, 1)
    return [e["cached"] for e in frames], timing


def run_rules(frames: list[dict], cfg: dict) -> dict:
    """Stage 2: schema enforcement per frame, then the kinematic pass per clip."""
    drop_counts: dict[str, int] = defaultdict(int)
    n_merges_total = 0
    by_clip: dict[str, list[dict]] = defaultdict(list)

    for payload in frames:
        kept, dropped, n_merges = enforce_schema(
            payload["dets"], payload["width"], payload["height"], cfg["schema"]
        )
        n_merges_total += n_merges
        for d in dropped:
            drop_counts[d["drop_reason"]] += 1
        clip = Path(payload["path"]).parent.name
        by_clip[clip].append(
            {
                "path": payload["path"],
                "width": payload["width"],
                "height": payload["height"],
                "dets": kept,
                "dropped": dropped,
                "n_raw": len(payload["dets"]),
            }
        )

    track_stats = {}
    for clip, clip_frames in by_clip.items():
        clip_frames.sort(key=lambda f: f["path"])
        drops, reasons, stats = purge_static_tracks(
            clip_frames, clip_frames[0]["width"], cfg["kinematics"]
        )
        track_stats[clip] = stats
        for f_idx, frame in enumerate(clip_frames):
            dead = drops.get(f_idx, set())
            if not dead:
                continue
            keep = []
            for det_i, det in enumerate(frame["dets"]):
                if det_i in dead:
                    reason = reasons.get((f_idx, det_i), "static_track")
                    frame["dropped"].append(dict(det, drop_reason=reason))
                    drop_counts[reason] += 1
                else:
                    keep.append(det)
            frame["dets"] = keep

    return {
        "by_clip": by_clip,
        "drop_counts": dict(drop_counts),
        "n_merges": n_merges_total,
        "track_stats": track_stats,
    }


def write_outputs(
    rules: dict,
    cfg: dict,
    *,
    labels_dir: Path,
    preview_dir: Path,
    preview_per_clip: int,
    replace_clip_ids: set[str] | None,
) -> dict:
    labels_dir.mkdir(parents=True, exist_ok=True)
    per_clip: dict[str, dict] = {}
    box_rows: list[dict] = []

    for clip, frames in sorted(rules["by_clip"].items()):
        n_boxes = 0
        n_empty = 0
        n_sam = 0
        for frame in frames:
            boxes = np.array([d["xyxy"] for d in frame["dets"]], dtype=np.float32).reshape(-1, 4)
            al.write_yolo_label(
                al.label_path_for(frame["path"], labels_dir),
                boxes,
                frame["width"],
                frame["height"],
            )
            n_boxes += len(frame["dets"])
            n_empty += 1 if not frame["dets"] else 0
            n_sam += sum(1 for d in frame["dets"] if d["source"] == "sam")
            for det in frame["dets"]:
                box_rows.append(
                    {
                        "path": frame["path"],
                        "clip_id": clip,
                        "conf": f"{det['score']:.4f}",
                        "source": det["source"],
                        "merged_parts": det.get("merged_parts", 1),
                        "x1": f"{det['xyxy'][0]:.1f}",
                        "y1": f"{det['xyxy'][1]:.1f}",
                        "x2": f"{det['xyxy'][2]:.1f}",
                        "y2": f"{det['xyxy'][3]:.1f}",
                    }
                )
        per_clip[clip] = {
            "images": len(frames),
            "boxes": n_boxes,
            "empty": n_empty,
            "sam_refined_boxes": n_sam,
            "raw_boxes": sum(f["n_raw"] for f in frames),
        }

        for frame in sorted(frames, key=lambda f: -len(f["dets"]))[:preview_per_clip]:
            img = cv2.imread(str(REPO_ROOT / frame["path"]))
            if img is not None:
                stem = Path(frame["path"]).stem
                draw_preview(img, frame, preview_dir / f"{clip}_{stem}.jpg")

    boxes_csv = labels_dir / "boxes.csv"
    fieldnames = ["path", "clip_id", "conf", "source", "merged_parts", "x1", "y1", "x2", "y2"]
    rows = al.merge_boxes_csv(boxes_csv, box_rows, replace_clip_ids)
    with boxes_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return per_clip


def compare_to_reference(rules: dict, reference_dir: Path, iou_thr: float) -> dict:
    """Agreement with the human-cleaned teacher labels: recall, extras, tightness.

    The reference is itself pseudo-GT, so this measures divergence, not accuracy:
    'extra' boxes include both new false positives and vehicles the teacher missed.
    """
    per_clip: dict[str, dict] = {}
    ious: list[float] = []
    area_ratios: list[float] = []
    for clip, frames in sorted(rules["by_clip"].items()):
        n_ref = n_auto = n_match = 0
        for frame in frames:
            stem = Path(frame["path"]).stem
            ref_file = reference_dir / clip / f"{stem}.txt"
            if not ref_file.exists():
                continue
            rows = []
            for line in ref_file.read_text().splitlines():
                if line.strip():
                    _c, cx, cy, bw, bh = [float(v) for v in line.split()]
                    rows.append((cx, cy, bw, bh))
            ref = ev.yolo_to_xyxy(rows, frame["width"], frame["height"])
            auto = np.array([d["xyxy"] for d in frame["dets"]], dtype=np.float32).reshape(-1, 4)
            scores = np.array([d["score"] for d in frame["dets"]], dtype=np.float32)
            matches, _un_ref, _un_auto = ev.greedy_match(ref, auto, scores, iou_thr)
            n_ref += len(ref)
            n_auto += len(auto)
            n_match += len(matches)
            for ri, ai in matches:
                ious.append(float(ev.iou_matrix(ref[ri : ri + 1], auto[ai : ai + 1])[0, 0]))
                ref_area = box_area(ref[ri])
                if ref_area > 0:
                    area_ratios.append(box_area(auto[ai]) / ref_area)
        per_clip[clip] = {
            "reference_boxes": n_ref,
            "auto_boxes": n_auto,
            "matched": n_match,
            "recall_vs_reference": round(n_match / n_ref, 4) if n_ref else None,
            "extra_boxes": n_auto - n_match,
        }
    total_ref = sum(v["reference_boxes"] for v in per_clip.values())
    total_match = sum(v["matched"] for v in per_clip.values())
    total_auto = sum(v["auto_boxes"] for v in per_clip.values())
    return {
        "reference_dir": str(rel_to_repo(reference_dir)),
        "iou_thr": iou_thr,
        "recall_vs_reference": round(total_match / total_ref, 4) if total_ref else None,
        "extra_boxes": total_auto - total_match,
        "extra_per_frame": round(
            (total_auto - total_match) / max(sum(len(f) for f in rules["by_clip"].values()), 1), 2
        ),
        # <1 means the DINO+SAM box is tighter than the teacher box on the same vehicle.
        "median_matched_iou": round(float(np.median(ious)), 4) if ious else None,
        "median_area_ratio": round(float(np.median(area_ratios)), 4) if area_ratios else None,
        "per_clip": per_clip,
    }


def audit_student_fps(rules: dict, taxonomy_path: Path, iou_thr: float) -> dict | None:
    """Second opinion on the student's hold-out FPs.

    error_analysis.py flagged FPs as `gt_omission` (confident, no proxy GT) on the
    assumption they are labeling debt rather than model error. An independent
    labeler can test that: an FP overlapping a DINO+SAM box is a vehicle the proxy
    GT missed, so the student was right and the metric was wrong.
    """
    if not taxonomy_path.exists():
        return None
    taxonomy = json.loads(taxonomy_path.read_text())
    instances = taxonomy.get("fp_instances") or []
    if not instances:
        return None

    auto_by_path: dict[str, np.ndarray] = {}
    for frames in rules["by_clip"].values():
        for frame in frames:
            auto_by_path[frame["path"]] = np.array(
                [d["xyxy"] for d in frame["dets"]], dtype=np.float32
            ).reshape(-1, 4)

    buckets: dict[str, dict] = defaultdict(lambda: {"n": 0, "confirmed": 0, "no_frame": 0})
    for inst in instances:
        bucket = buckets[inst["error_type"]]
        auto = auto_by_path.get(inst["path"])
        if auto is None:
            bucket["no_frame"] += 1
            continue
        bucket["n"] += 1
        if len(auto) == 0:
            continue
        box = np.asarray(inst["xyxy"], dtype=np.float32).reshape(1, 4)
        if float(ev.iou_matrix(box, auto).max()) >= iou_thr:
            bucket["confirmed"] += 1

    out = {}
    for name, stats in buckets.items():
        if not stats["n"]:
            continue
        out[name] = {
            "sampled_fps": stats["n"],
            "confirmed_vehicle": stats["confirmed"],
            "confirmed_pct": round(100.0 * stats["confirmed"] / stats["n"], 1),
        }
    return {
        "source": str(rel_to_repo(taxonomy_path)),
        "iou_thr": iou_thr,
        # error_analysis.py stores a capped sample of instances, not every FP.
        "note": "share of sampled student FPs that overlap an independent DINO+SAM box",
        "buckets": out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "data.yaml")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Default: train val. With --allow-eval: eval.",
    )
    parser.add_argument(
        "--allow-eval",
        action="store_true",
        help="Audit the frozen hold-out. Writes eval.labels_dir; never training data.",
    )
    parser.add_argument("--clips", nargs="+", default=None, help="Restrict to these clip ids")
    parser.add_argument("--limit", type=int, default=None, help="First N frames (smoke runs)")
    parser.add_argument("--stride", type=int, default=1, help="Take every Nth frame")
    parser.add_argument(
        "--stage",
        choices=["all", "infer", "rules"],
        default="all",
        help="infer = DINO+SAM into the cache only; rules = cached boxes -> labels",
    )
    parser.add_argument("--refresh", action="store_true", help="Ignore the inference cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--labels-dir", type=Path, default=None, help="Override output label dir")
    parser.add_argument(
        "--compare-iou",
        type=float,
        default=0.4,
        help="IoU for agreement scoring against the reference labels",
    )
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    data_cfg = al.load_config(cfg_path)
    cfg = data_cfg["auto_label_dino_sam"]
    splits_dir = REPO_ROOT / data_cfg["paths"]["splits_dir"]
    eval_paths = al.read_split_list(splits_dir / "eval.txt")
    eval_clip_ids = {c["id"] for c in data_cfg["clips"]["eval"]}

    clip_ids = set(args.clips) if args.clips else None
    splits = args.splits or (["eval"] if args.allow_eval else ["train", "val"])

    if args.allow_eval:
        if splits != ["eval"]:
            raise RuntimeError("--allow-eval only accepts --splits eval")
        if clip_ids and clip_ids - eval_clip_ids:
            raise RuntimeError(f"--allow-eval --clips must be eval ids; got {sorted(clip_ids)}")
        out_cfg = cfg["eval"]
        reference_dir = REPO_ROOT / out_cfg["reference_dir"]
    else:
        if "eval" in splits:
            raise RuntimeError(
                "Refusing to auto-label the hold-out. Use --allow-eval --splits eval to audit it."
            )
        if clip_ids and clip_ids & eval_clip_ids:
            raise RuntimeError(f"Refusing eval clip ids: {sorted(clip_ids & eval_clip_ids)}")
        out_cfg = cfg
        reference_dir = REPO_ROOT / data_cfg["cleanup"]["clean_dir"]

    image_rels: list[str] = []
    for split in splits:
        image_rels.extend(al.read_split_list(splits_dir / f"{split}.txt"))
    image_rels = sorted(dict.fromkeys(image_rels))
    image_rels = al.filter_image_rels(image_rels, clip_ids)
    if args.allow_eval:
        al.assert_eval_only(image_rels, eval_paths, eval_clip_ids)
    else:
        al.assert_train_pool_only(image_rels, eval_paths, eval_clip_ids)
    if args.stride > 1:
        image_rels = image_rels[:: args.stride]
    if args.limit:
        image_rels = image_rels[: args.limit]

    labels_dir = args.labels_dir or REPO_ROOT / out_cfg["labels_dir"]
    if not Path(labels_dir).is_absolute():
        labels_dir = REPO_ROOT / labels_dir
    cache_dir = REPO_ROOT / cfg["cache_dir"]
    device = pick_torch_device(args.device)

    pool = "hold-out audit" if args.allow_eval else "train pool"
    print(
        f"DINO+SAM auto-label: {len(image_rels)} frames splits={splits} "
        f"clips={sorted(clip_ids) if clip_ids else f'all {pool}'} device={device}"
    )
    subsampled = args.stride > 1 or args.limit is not None
    if subsampled:
        print("  NOTE: subsampled run — tracks are broken, so the kinematic purge under-fires")
    frames, timing = run_inference(
        image_rels,
        cfg,
        cache_dir=cache_dir,
        device=device,
        refresh=args.refresh,
    )
    if args.stage == "infer":
        print(f"Cached inference for {len(frames)} frames in {cache_dir.relative_to(REPO_ROOT)}")
        return 0

    rules = run_rules(frames, cfg)
    per_clip = write_outputs(
        rules,
        cfg,
        labels_dir=Path(labels_dir),
        preview_dir=REPO_ROOT / out_cfg["preview_dir"],
        preview_per_clip=int(cfg["preview_per_clip"]),
        replace_clip_ids=clip_ids,
    )

    agreement = (
        compare_to_reference(rules, reference_dir, args.compare_iou)
        if reference_dir.exists()
        else None
    )
    fp_audit = (
        audit_student_fps(
            rules, REPO_ROOT / data_cfg["diagnostics"]["report_path"], args.compare_iou
        )
        if args.allow_eval
        else None
    )

    n_raw = sum(len(p["dets"]) for p in frames)
    n_final = sum(v["boxes"] for v in per_clip.values())
    n_sam = sum(v["sam_refined_boxes"] for v in per_clip.values())
    summary = {
        "pipeline": "grounding_dino -> sam -> tight_box -> schema+kinematics",
        "dino_model": cfg["dino"]["model_id"],
        "sam_model": cfg["sam"]["model_id"],
        "prompt": cfg["dino"]["prompt"],
        "device": device,
        "splits": splits,
        "role": "eval_proxy_audit" if args.allow_eval else "train_pool_labels",
        "student_frozen": True if args.allow_eval else None,
        "used_for_training": not args.allow_eval,
        "n_images": len(frames),
        "n_boxes_raw": n_raw,
        "n_boxes_final": n_final,
        "n_sam_refined": n_sam,
        "sam_refine_rate": round(n_sam / n_final, 4) if n_final else 0.0,
        "n_truck_merges": rules["n_merges"],
        "agreement_vs_reference": agreement,
        "student_fp_audit": fp_audit,
        "drops_by_reason": rules["drop_counts"],
        "track_stats": rules["track_stats"],
        "per_clip": per_clip,
        "timing": timing,
        "labels_dir": str(Path(labels_dir).relative_to(REPO_ROOT)),
        "config": {k: cfg[k] for k in ("dino", "tiling", "sam", "schema", "kinematics")},
        "eval_frames_skipped": len(eval_paths),
        "subsampled_run": subsampled,
    }
    summary_path = REPO_ROOT / out_cfg["summary_path"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("\nPer clip")
    for clip, stats in sorted(per_clip.items()):
        print(
            f"  {clip}: images={stats['images']} raw={stats['raw_boxes']} "
            f"final={stats['boxes']} sam={stats['sam_refined_boxes']} empty={stats['empty']}"
        )
    print(f"  TOTAL raw={n_raw} final={n_final} merges={rules['n_merges']}")
    print(f"  drops: {rules['drop_counts']}")
    if agreement:
        print(
            f"  vs {agreement['reference_dir']}: recall={agreement['recall_vs_reference']} "
            f"extra/frame={agreement['extra_per_frame']} "
            f"median IoU={agreement['median_matched_iou']} "
            f"median area ratio={agreement['median_area_ratio']}"
        )
    if fp_audit:
        print("  student FP audit (sampled, IoU >= %.1f):" % args.compare_iou)
        for name, stats in sorted(fp_audit["buckets"].items()):
            print(
                f"    {name}: {stats['confirmed_vehicle']}/{stats['sampled_fps']} "
                f"({stats['confirmed_pct']}%) confirmed as vehicles"
            )
    print(f"Wrote {Path(labels_dir).relative_to(REPO_ROOT)}")
    print(f"Wrote {summary_path.relative_to(REPO_ROOT)}")
    if args.allow_eval:
        print("OK: hold-out audit only — proxy GT untouched, labels never enter training")
    else:
        print(f"OK: eval not labeled ({len(eval_paths)} frames skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
