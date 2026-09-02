#!/usr/bin/env python3
"""Failure-mode taxonomy for the hold-out student run.

Diagnostic only: nothing here re-tunes the model. Thresholds come from
data/splits/eval_thresholds.json (frozen on val), same as evaluate_custom.py.

Predictions get persistent track IDs from a ByteTrack-style two-stage IoU
tracker (own implementation; the CPU stack here has no lap/scipy assignment
solver and full ByteTrack is overkill for 5 fps sampled frames). Errors are
then bucketed:

  Rule 1 gt_omission         confident FP (conf > gt_omission_conf), no GT
  Rule 2 suspected_motorcycle  FP with aspect ratio beyond the vehicle prior
  Rule 3 fractured_truck     large missed GT covered by >=2 contained FPs
  Rule 4 static_hallucination FP track with a stationary centroid over time

Outputs: taxonomy JSON, hard-negative crops, annotated 2x2 example grid.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import evaluate_custom as ev  # noqa: E402

REPO_ROOT = SRC_DIR.parent

ERROR_TYPES = (
    "gt_omission",
    "suspected_motorcycle",
    "fractured_truck",
    "static_hallucination",
)

# FP precedence when several rules fire on one box: most specific evidence wins.
FP_PRECEDENCE = (
    "fractured_truck",
    "static_hallucination",
    "suspected_motorcycle",
    "gt_omission",
)

TYPE_COLORS = {
    "gt_omission": (0, 165, 255),
    "suspected_motorcycle": (255, 0, 200),
    "fractured_truck": (0, 0, 255),
    "static_hallucination": (255, 200, 0),
}


class IoUTracker:
    """Two-stage IoU association (high-confidence first, then leftovers)."""

    def __init__(
        self,
        *,
        high_conf: float,
        iou_high: float,
        iou_low: float,
        max_age: int,
    ) -> None:
        self.high_conf = high_conf
        self.iou_high = iou_high
        self.iou_low = iou_low
        self.max_age = max_age
        self._next_id = 1
        self.tracks: dict[int, dict] = {}

    def _new_track(self, frame_idx: int, box: np.ndarray, score: float) -> int:
        tid = self._next_id
        self._next_id += 1
        self.tracks[tid] = {"xyxy": box.copy(), "last_frame": frame_idx, "hits": 1, "score": score}
        return tid

    def _associate(
        self,
        det_idx: list[int],
        boxes: np.ndarray,
        live_ids: list[int],
        iou_thr: float,
        assigned: dict[int, int],
        frame_idx: int,
    ) -> list[int]:
        """Greedy IoU match; returns detection indices left unmatched."""
        if not det_idx or not live_ids:
            return det_idx
        track_boxes = np.stack([self.tracks[tid]["xyxy"] for tid in live_ids])
        det_boxes = boxes[det_idx]
        ious = ev.iou_matrix(track_boxes, det_boxes)
        pairs = [
            (float(ious[ti, di]), ti, di)
            for ti in range(len(live_ids))
            for di in range(len(det_idx))
            if ious[ti, di] >= iou_thr
        ]
        pairs.sort(reverse=True)
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for _iou, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            tid = live_ids[ti]
            det = det_idx[di]
            assigned[det] = tid
            self.tracks[tid]["xyxy"] = boxes[det].copy()
            self.tracks[tid]["last_frame"] = frame_idx
            self.tracks[tid]["hits"] += 1
            used_tracks.add(ti)
            used_dets.add(di)
        return [d for k, d in enumerate(det_idx) if k not in used_dets]

    def update(self, frame_idx: int, boxes: np.ndarray, scores: np.ndarray) -> list[int]:
        """Returns one track id per detection, in detection order."""
        stale = [
            tid for tid, tr in self.tracks.items() if frame_idx - tr["last_frame"] > self.max_age
        ]
        for tid in stale:
            del self.tracks[tid]

        assigned: dict[int, int] = {}
        if len(boxes) == 0:
            return []

        high = [i for i in range(len(boxes)) if scores[i] >= self.high_conf]
        low = [i for i in range(len(boxes)) if scores[i] < self.high_conf]

        live_ids = sorted(self.tracks, key=lambda t: -self.tracks[t]["hits"])
        high_left = self._associate(high, boxes, live_ids, self.iou_high, assigned, frame_idx)

        live_ids = [tid for tid in live_ids if tid not in set(assigned.values())]
        low_left = self._associate(low, boxes, live_ids, self.iou_low, assigned, frame_idx)

        for det in high_left + low_left:
            assigned[det] = self._new_track(frame_idx, boxes[det], float(scores[det]))
        return [assigned[i] for i in range(len(boxes))]


def centroid(box: np.ndarray) -> tuple[float, float]:
    return (float(box[0] + box[2]) / 2.0, float(box[1] + box[3]) / 2.0)


def aspect_ratio(box: np.ndarray) -> float:
    w = max(1e-6, float(box[2] - box[0]))
    h = max(1e-6, float(box[3] - box[1]))
    return max(w / h, h / w)


def containment(inner: np.ndarray, outer: np.ndarray) -> float:
    """Share of `inner` area that falls inside `outer`."""
    ix1 = max(float(inner[0]), float(outer[0]))
    iy1 = max(float(inner[1]), float(outer[1]))
    ix2 = min(float(inner[2]), float(outer[2]))
    iy2 = min(float(inner[3]), float(outer[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1e-6, (float(inner[2]) - float(inner[0])) * (float(inner[3]) - float(inner[1])))
    return inter / area


def box_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def collect_frames(
    *,
    image_rels: list[str],
    labels_dir: Path,
    manifest: dict[str, dict],
    pred_cache: dict[str, dict],
    conf: float,
    iou_match: float,
    dist_cfg: dict,
    tracker_cfg: dict,
) -> list[dict]:
    """Per-frame GT/pred bookkeeping with track ids, ordered in time per clip."""
    by_clip: dict[str, list[str]] = defaultdict(list)
    for rel in image_rels:
        by_clip[Path(rel).parent.name].append(rel)

    frames: list[dict] = []
    for clip_id, rels in sorted(by_clip.items()):
        rels = sorted(rels, key=lambda r: float(manifest[r]["t_sec"]))
        tracker = IoUTracker(
            high_conf=float(tracker_cfg["high_conf"]),
            iou_high=float(tracker_cfg["iou_high"]),
            iou_low=float(tracker_cfg["iou_low"]),
            max_age=int(tracker_cfg["max_age"]),
        )
        for frame_idx, rel in enumerate(rels):
            meta = manifest[rel]
            img_w, img_h = int(meta["width"]), int(meta["height"])
            gt_xyxy = ev.yolo_to_xyxy(
                ev.load_yolo_boxes(labels_dir / clip_id / f"{Path(rel).stem}.txt"),
                img_w,
                img_h,
            )
            raw = pred_cache[rel]
            keep = raw["scores"] >= conf
            pred_xyxy = raw["xyxy"][keep]
            pred_scores = raw["scores"][keep]
            track_ids = tracker.update(frame_idx, pred_xyxy, pred_scores)

            matches, unmatched_gt, unmatched_pred = ev.greedy_match(
                gt_xyxy, pred_xyxy, pred_scores, iou_match
            )
            def band_of(box: np.ndarray) -> str:
                return ev.band_for_box(
                    box,
                    img_h,
                    w_ref=float(dist_cfg["w_ref_m"]),
                    fov_v=float(dist_cfg["fov_v_deg"]),
                    near_max=float(dist_cfg["near_max_m"]),
                    far_max=float(dist_cfg["far_max_m"]),
                )

            gt_bands = [band_of(b) for b in gt_xyxy]
            pred_bands = [band_of(b) for b in pred_xyxy]
            frames.append(
                {
                    "clip_id": clip_id,
                    "frame_idx": frame_idx,
                    "path": rel,
                    "t_sec": float(meta["t_sec"]),
                    "img_w": img_w,
                    "img_h": img_h,
                    "gt_xyxy": gt_xyxy,
                    "gt_bands": gt_bands,
                    "pred_bands": pred_bands,
                    "pred_xyxy": pred_xyxy,
                    "pred_scores": pred_scores,
                    "track_ids": track_ids,
                    "matches": matches,
                    "unmatched_gt": unmatched_gt,
                    "unmatched_pred": unmatched_pred,
                }
            )
    return frames


def classify_static_tracks(
    frames: list[dict],
    *,
    move_px: float,
    min_frames: int,
    relaxed_move_frac: float,
    relaxed_min_frames: int,
) -> tuple[dict[tuple[str, int], dict], dict[tuple[str, int], dict], list[dict]]:
    """Find FP tracks whose centroid barely moves. Returns (strict, relaxed, stats)."""
    per_track: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for fr in frames:
        fp_set = set(fr["unmatched_pred"])
        for pj, tid in enumerate(fr["track_ids"]):
            if pj not in fp_set:
                continue
            per_track[(fr["clip_id"], tid)].append(
                {
                    "frame_idx": fr["frame_idx"],
                    "path": fr["path"],
                    "xyxy": fr["pred_xyxy"][pj],
                    "conf": float(fr["pred_scores"][pj]),
                    "centroid": centroid(fr["pred_xyxy"][pj]),
                    "img_w": fr["img_w"],
                }
            )

    def longest_static_run(dets: list[dict], thr: float) -> list[dict]:
        best: list[dict] = []
        n = len(dets)
        for i in range(n):
            anchor = dets[i]["centroid"]
            run = [dets[i]]
            for j in range(i + 1, n):
                if dets[j]["frame_idx"] != dets[j - 1]["frame_idx"] + 1:
                    break
                dx = dets[j]["centroid"][0] - anchor[0]
                dy = dets[j]["centroid"][1] - anchor[1]
                if math.hypot(dx, dy) >= thr:
                    break
                run.append(dets[j])
            if len(run) > len(best):
                best = run
        return best

    strict: dict[tuple[str, int], dict] = {}
    relaxed: dict[tuple[str, int], dict] = {}
    stats: list[dict] = []
    for key, dets in per_track.items():
        dets.sort(key=lambda d: d["frame_idx"])
        strict_run = longest_static_run(dets, move_px)
        img_w = dets[0]["img_w"]
        relaxed_run = longest_static_run(dets, relaxed_move_frac * img_w)
        stats.append(
            {
                "clip_id": key[0],
                "track_id": key[1],
                "n_fp_frames": len(dets),
                "strict_run": len(strict_run),
                "relaxed_run": len(relaxed_run),
            }
        )
        if len(strict_run) > min_frames:
            strict[key] = {"run": strict_run, "len": len(strict_run)}
        elif len(relaxed_run) >= relaxed_min_frames:
            relaxed[key] = {"run": relaxed_run, "len": len(relaxed_run)}

    stats.sort(key=lambda s: (-s["strict_run"], -s["relaxed_run"]))
    return strict, relaxed, stats


def classify(frames: list[dict], rules: dict, tracker_cfg: dict) -> dict:
    """Apply the four rules; returns instances plus per-track static evidence."""
    gt_conf = float(rules["gt_omission_conf"])
    moto_aspect = float(rules["motorcycle_aspect"])
    min_parts = int(rules["fracture_min_parts"])
    containment_thr = float(rules["fracture_containment"])
    large_pct = float(rules["fracture_large_pct"])

    # "Large" GT is clip-relative: 4K F boxes and portrait E boxes differ a lot.
    areas_by_clip: dict[str, list[float]] = defaultdict(list)
    for fr in frames:
        for box in fr["gt_xyxy"]:
            areas_by_clip[fr["clip_id"]].append(box_area(box))
    large_thresh = {
        clip: float(np.percentile(areas, large_pct)) if areas else float("inf")
        for clip, areas in areas_by_clip.items()
    }

    strict_static, relaxed_static, static_stats = classify_static_tracks(
        frames,
        move_px=float(rules["static_move_px"]),
        min_frames=int(rules["static_min_frames"]),
        relaxed_move_frac=float(rules["relaxed_move_frac"]),
        relaxed_min_frames=int(rules["relaxed_min_frames"]),
    )
    static_frames: set[tuple[str, int, int]] = set()
    for (clip_id, tid), payload in strict_static.items():
        for det in payload["run"]:
            static_frames.add((clip_id, tid, det["frame_idx"]))

    fp_instances: list[dict] = []
    fn_instances: list[dict] = []
    totals = {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0}

    for fr in frames:
        clip_id = fr["clip_id"]
        totals["tp"] += len(fr["matches"])
        totals["fp"] += len(fr["unmatched_pred"])
        totals["fn"] += len(fr["unmatched_gt"])
        totals["gt"] += len(fr["gt_xyxy"])
        totals["pred"] += len(fr["pred_xyxy"])

        # Rule 3 first: it also tags the FP parts, not just the missed GT.
        fracture_parts: dict[int, int] = {}
        for gi in fr["unmatched_gt"]:
            gt_box = fr["gt_xyxy"][gi]
            if box_area(gt_box) < large_thresh.get(clip_id, float("inf")):
                continue
            parts = [
                pj
                for pj in fr["unmatched_pred"]
                if containment(fr["pred_xyxy"][pj], gt_box) >= containment_thr
            ]
            if len(parts) >= min_parts:
                for pj in parts:
                    fracture_parts[pj] = gi
                fn_instances.append(
                    {
                        "error_type": "fractured_truck",
                        "clip_id": clip_id,
                        "path": fr["path"],
                        "frame_idx": fr["frame_idx"],
                        "t_sec": fr["t_sec"],
                        "gt_index": gi,
                        "gt_xyxy": [round(float(v), 1) for v in gt_box],
                        "gt_band": fr["gt_bands"][gi],
                        "n_parts": len(parts),
                        "part_track_ids": [fr["track_ids"][pj] for pj in parts],
                        "part_xyxy": [
                            [round(float(v), 1) for v in fr["pred_xyxy"][pj]] for pj in parts
                        ],
                    }
                )

        for pj in fr["unmatched_pred"]:
            box = fr["pred_xyxy"][pj]
            conf = float(fr["pred_scores"][pj])
            tid = fr["track_ids"][pj]
            tags: list[str] = []
            if conf > gt_conf:
                tags.append("gt_omission")
            if aspect_ratio(box) > moto_aspect:
                tags.append("suspected_motorcycle")
            if pj in fracture_parts:
                tags.append("fractured_truck")
            if (clip_id, tid, fr["frame_idx"]) in static_frames:
                tags.append("static_hallucination")
            primary = next((t for t in FP_PRECEDENCE if t in tags), "unclassified")
            fp_instances.append(
                {
                    "error_type": primary,
                    "tags": tags,
                    "clip_id": clip_id,
                    "path": fr["path"],
                    "frame_idx": fr["frame_idx"],
                    "t_sec": fr["t_sec"],
                    "pred_index": pj,
                    "track_id": tid,
                    "conf": round(conf, 4),
                    "aspect": round(aspect_ratio(box), 2),
                    "band": fr["pred_bands"][pj],
                    "xyxy": [round(float(v), 1) for v in box],
                    "gt_index": fracture_parts.get(pj),
                }
            )

    return {
        "totals": totals,
        "fp_instances": fp_instances,
        "fn_instances": fn_instances,
        "strict_static": strict_static,
        "relaxed_static": relaxed_static,
        "static_stats": static_stats,
        "large_gt_area_threshold": {k: round(v, 1) for k, v in large_thresh.items()},
    }


def summarize(result: dict) -> dict:
    totals = result["totals"]
    n_fp = max(1, totals["fp"])
    n_fn = max(1, totals["fn"])

    fp_counts = defaultdict(int)
    fp_tag_counts = defaultdict(int)
    per_clip_fp = defaultdict(lambda: defaultdict(int))
    for inst in result["fp_instances"]:
        fp_counts[inst["error_type"]] += 1
        per_clip_fp[inst["clip_id"]][inst["error_type"]] += 1
        for tag in inst["tags"]:
            fp_tag_counts[tag] += 1

    fn_counts = defaultdict(int)
    per_clip_fn = defaultdict(lambda: defaultdict(int))
    fractured_gt = {
        (inst["path"], inst["gt_index"]) for inst in result["fn_instances"]
    }
    fn_counts["fractured_truck"] = len(fractured_gt)
    for inst in result["fn_instances"]:
        per_clip_fn[inst["clip_id"]]["fractured_truck"] += 1
    fn_counts["unclassified"] = totals["fn"] - fn_counts["fractured_truck"]

    fp_summary = {}
    for key in list(ERROR_TYPES) + ["unclassified"]:
        fp_summary[key] = {
            "count": fp_counts.get(key, 0),
            "pct_of_fp": round(100.0 * fp_counts.get(key, 0) / n_fp, 2),
        }
    fp_summary_tags = {
        key: {
            "count": fp_tag_counts.get(key, 0),
            "pct_of_fp": round(100.0 * fp_tag_counts.get(key, 0) / n_fp, 2),
        }
        for key in ERROR_TYPES
    }
    fn_summary = {
        key: {
            "count": fn_counts.get(key, 0),
            "pct_of_fn": round(100.0 * fn_counts.get(key, 0) / n_fn, 2),
        }
        for key in ("fractured_truck", "unclassified")
    }

    # The residual bucket is the biggest one, so describe it instead of leaving it opaque.
    residual = [i for i in result["fp_instances"] if i["error_type"] == "unclassified"]
    by_band = defaultdict(int)
    for inst in residual:
        by_band[inst["band"]] += 1
    confs = sorted(i["conf"] for i in residual)
    residual_profile = {
        "count": len(residual),
        "by_band": dict(sorted(by_band.items())),
        "conf_median": round(confs[len(confs) // 2], 4) if confs else None,
        "conf_p90": round(confs[int(0.9 * (len(confs) - 1))], 4) if confs else None,
        "note": "FPs no rule claimed: mostly sub-0.5 boxes on true-looking clutter.",
    }
    return {
        "residual_profile": residual_profile,
        "fp_primary": fp_summary,
        "fp_any_tag": fp_summary_tags,
        "fn": fn_summary,
        "per_clip_fp_primary": {c: dict(v) for c, v in sorted(per_clip_fp.items())},
        "per_clip_fn": {c: dict(v) for c, v in sorted(per_clip_fn.items())},
    }


def mine_hard_negatives(
    result: dict,
    out_dir: Path,
    *,
    max_crops: int,
    pad_px: int,
) -> dict:
    """Crop static-hallucination FPs for the next training round."""
    if out_dir.exists():
        for old in out_dir.glob("*.jpg"):
            old.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    sources: list[tuple[str, tuple[str, int], dict]] = []
    for key, payload in result["strict_static"].items():
        for det in payload["run"]:
            sources.append(("strict", key, det))
    for key, payload in result["relaxed_static"].items():
        for det in payload["run"]:
            sources.append(("relaxed", key, det))

    per_track_budget: dict[tuple[str, int], int] = defaultdict(int)
    rows: list[dict] = []
    cache: dict[str, np.ndarray] = {}
    for mode, key, det in sources:
        if len(rows) >= max_crops:
            break
        if per_track_budget[key] >= 3:
            continue
        img = cache.get(det["path"])
        if img is None:
            img = cv2.imread(str(REPO_ROOT / det["path"]))
            if img is None:
                continue
            cache[det["path"]] = img
        h, w = img.shape[:2]
        x1 = max(0, int(det["xyxy"][0]) - pad_px)
        y1 = max(0, int(det["xyxy"][1]) - pad_px)
        x2 = min(w, int(det["xyxy"][2]) + pad_px)
        y2 = min(h, int(det["xyxy"][3]) + pad_px)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        crop = img[y1:y2, x1:x2]
        clip_id, tid = key
        name = f"{clip_id}_t{tid:04d}_{Path(det['path']).stem}_{mode}.jpg"
        cv2.imwrite(str(out_dir / name), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        per_track_budget[key] += 1
        rows.append(
            {
                "file": str((out_dir / name).relative_to(REPO_ROOT)),
                "mode": mode,
                "clip_id": clip_id,
                "track_id": tid,
                "source_frame": det["path"],
                "xyxy": [round(float(v), 1) for v in det["xyxy"]],
                "conf": round(det["conf"], 4),
            }
        )

    manifest = {
        "role": "hard_negative_mining",
        "note": (
            "Static-centroid FP crops for the next training round. 'strict' meets "
            "Rule 4; 'relaxed' is a mining-only fallback for drone ego-motion."
        ),
        "n_crops": len(rows),
        "n_strict_tracks": len(result["strict_static"]),
        "n_relaxed_tracks": len(result["relaxed_static"]),
        "crops": rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def cache_paths(cache_dir: Path) -> tuple[Path, Path]:
    return cache_dir / "eval_preds.npz", cache_dir / "eval_preds.json"


def load_pred_cache(cache_dir: Path, key: dict, image_rels: list[str]) -> dict[str, dict] | None:
    npz_path, meta_path = cache_paths(cache_dir)
    if not npz_path.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    if meta.get("key") != key or set(meta.get("paths", [])) != set(image_rels):
        return None
    data = np.load(npz_path, allow_pickle=False)
    cache: dict[str, dict] = {}
    for i, rel in enumerate(meta["paths"]):
        cache[rel] = {"xyxy": data[f"b{i}"], "scores": data[f"s{i}"]}
    return cache


def save_pred_cache(cache_dir: Path, key: dict, cache: dict[str, dict]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path, meta_path = cache_paths(cache_dir)
    rels = list(cache)
    arrays = {}
    for i, rel in enumerate(rels):
        arrays[f"b{i}"] = cache[rel]["xyxy"]
        arrays[f"s{i}"] = cache[rel]["scores"]
    np.savez_compressed(npz_path, **arrays)
    meta_path.write_text(json.dumps({"key": key, "paths": rels}, indent=2) + "\n")


def pick_example_frames(result: dict, frames: list[dict]) -> dict[str, dict]:
    """One representative frame per error type (most instances of that type)."""
    by_frame: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for inst in result["fp_instances"]:
        # fractured_truck is illustrated from the missed GT side, not its fragments.
        if inst["error_type"] in ERROR_TYPES and inst["error_type"] != "fractured_truck":
            by_frame[inst["path"]][inst["error_type"]].append(inst)
    for inst in result["fn_instances"]:
        by_frame[inst["path"]]["fractured_truck"].append(inst)

    # Rule 4 rarely fires strictly under drone ego-motion; illustrate the
    # relaxed mining candidates instead of shipping an empty tile.
    relaxed_by_frame: dict[str, list[dict]] = defaultdict(list)
    for det_run in result["relaxed_static"].values():
        for det in det_run["run"]:
            relaxed_by_frame[det["path"]].append(
                {
                    "xyxy": [round(float(v), 1) for v in det["xyxy"]],
                    "conf": det["conf"],
                    "relaxed": True,
                }
            )

    frame_lookup = {fr["path"]: fr for fr in frames}
    picks: dict[str, dict] = {}
    for err in ERROR_TYPES:
        best_path, best_n = None, 0
        for path, buckets in by_frame.items():
            n = len(buckets.get(err, []))
            if n > best_n:
                best_path, best_n = path, n
        if best_path is None and err == "static_hallucination" and relaxed_by_frame:
            best_path = max(relaxed_by_frame, key=lambda p: len(relaxed_by_frame[p]))
            picks[err] = {
                "frame": frame_lookup[best_path],
                "instances": relaxed_by_frame[best_path],
                "n": len(relaxed_by_frame[best_path]),
                "relaxed": True,
            }
            continue
        if best_path is not None:
            picks[err] = {
                "frame": frame_lookup[best_path],
                "instances": by_frame[best_path][err],
                "n": best_n,
            }
    return picks


def render_error_tile(err: str, pick: dict | None, tile_w: int, tile_h: int) -> np.ndarray:
    if pick is None:
        tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        cv2.putText(
            tile,
            f"{err}: 0 instances",
            (24, tile_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (170, 170, 170),
            2,
            cv2.LINE_AA,
        )
        return tile

    fr = pick["frame"]
    img = cv2.imread(str(REPO_ROOT / fr["path"]))
    if img is None:
        return np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    h0 = img.shape[0]
    img = ev._resize_max(img, 1400)
    scale = img.shape[0] / h0
    color = TYPE_COLORS[err]

    # Context first: GT in green, then the flagged boxes on top in the type color.
    for box in fr["gt_xyxy"]:
        ev._draw_boxes(img, (box * scale).reshape(1, 4), color=(0, 200, 0), thickness=3)
    for inst in pick["instances"]:
        if err == "fractured_truck":
            gt_box = np.asarray(inst["gt_xyxy"], dtype=np.float32) * scale
            ev._draw_boxes(
                img,
                gt_box.reshape(1, 4),
                color=color,
                labels=[f"fractured_truck x{inst['n_parts']}"],
            )
            for part in inst["part_xyxy"]:
                part_box = np.asarray(part, dtype=np.float32) * scale
                ev._draw_boxes(img, part_box.reshape(1, 4), color=(0, 220, 255), thickness=4)
        else:
            box = np.asarray(inst["xyxy"], dtype=np.float32) * scale
            ev._draw_boxes(img, box.reshape(1, 4), color=color, labels=[f"{err} {inst['conf']:.2f}"])

    suffix = "  [relaxed mining rule]" if pick.get("relaxed") else ""
    img = ev._banner(
        img,
        f"{err}  {fr['clip_id']}/{Path(fr['path']).stem}  n={pick['n']}{suffix}",
    )
    legend = [(color, err), ((0, 200, 0), "ground truth")]
    if err == "fractured_truck":
        legend.insert(1, ((0, 220, 255), "FP fragments"))
    img = ev._legend(img, legend)

    canvas = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    s = min(tile_w / img.shape[1], tile_h / img.shape[0])
    resized = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))
    y0 = (tile_h - resized.shape[0]) // 2
    x0 = (tile_w - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    cv2.rectangle(canvas, (0, 0), (tile_w - 1, tile_h - 1), (60, 60, 60), 2)
    return canvas


def render_diagnostics(result: dict, frames: list[dict], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = pick_example_frames(result, frames)
    tile_w, tile_h = 900, 700
    tiles = []
    written: dict[str, str] = {}
    for err in ERROR_TYPES:
        tile = render_error_tile(err, picks.get(err), tile_w, tile_h)
        single = out_dir / f"{err}.jpg"
        cv2.imwrite(str(single), tile, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        written[err] = str(single.relative_to(REPO_ROOT))
        tiles.append(tile)

    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    grid_path = out_dir / "error_taxonomy_grid.jpg"
    cv2.imwrite(str(grid_path), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    written["grid"] = str(grid_path.relative_to(REPO_ROOT))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "data.yaml")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--conf", type=float, default=None, help="Override frozen conf.")
    parser.add_argument("--clips", nargs="+", default=None, help="Limit to these eval clips.")
    parser.add_argument("--no-crops", action="store_true", help="Skip hard-negative mining.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore the cached predictions under runs/cache and re-run inference.",
    )
    parser.add_argument("--max-instances", type=int, default=60, help="Per-type rows kept in JSON.")
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    cfg = ev.load_config(cfg_path)
    eval_cfg = cfg["evaluation"]
    diag_cfg = cfg["diagnostics"]
    dist_cfg = cfg["distance"]
    rules = diag_cfg["rules"]

    splits_dir = REPO_ROOT / cfg["paths"]["splits_dir"]
    manifest = ev.load_manifest(splits_dir / "manifest.csv")
    eval_ids = {c["id"] for c in cfg["clips"]["eval"]}

    weights = Path(args.weights or eval_cfg["weights"])
    if not weights.is_absolute():
        weights = REPO_ROOT / weights
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights: {weights}")

    thresholds_path = REPO_ROOT / eval_cfg["thresholds_path"]
    if args.conf is not None:
        conf = float(args.conf)
        thresholds_note = "conf overridden on the command line"
    else:
        if not thresholds_path.exists():
            raise RuntimeError(
                f"Missing {thresholds_path}; run evaluate_custom.py --tune-val first."
            )
        frozen = json.loads(thresholds_path.read_text())
        conf = float(frozen["chosen_conf"])
        thresholds_note = "conf frozen on val by evaluate_custom.py --tune-val"

    image_rels = ev.read_split_list(splits_dir / "eval.txt")
    bad = [p for p in image_rels if Path(p).parent.name not in eval_ids]
    if bad:
        raise RuntimeError(f"Non-eval path in diagnostics list: {bad[:3]}")
    if args.clips:
        wanted = set(args.clips)
        unknown = wanted - eval_ids
        if unknown:
            raise RuntimeError(f"Unknown eval clip ids: {sorted(unknown)}")
        image_rels = [p for p in image_rels if Path(p).parent.name in wanted]

    device = ev.pick_device(args.device)
    cache_dir = REPO_ROOT / "runs" / "cache"
    cache_key = {
        "weights": str(weights.relative_to(REPO_ROOT)),
        "weights_mtime": int(weights.stat().st_mtime),
        "imgsz": int(eval_cfg["imgsz"]),
        "conf_floor": conf,
        "nms_iou": float(eval_cfg["nms_iou"]),
    }
    cache = None if args.refresh else load_pred_cache(cache_dir, cache_key, image_rels)
    if cache is None:
        cache = ev.predict_cached(
            image_rels=image_rels,
            weights=weights,
            imgsz=int(eval_cfg["imgsz"]),
            conf_floor=conf,
            nms_iou=float(eval_cfg["nms_iou"]),
            device=device,
        )
        save_pred_cache(cache_dir, cache_key, cache)
    else:
        print(f"Using cached predictions for {len(image_rels)} frames (--refresh to redo)")
    frames = collect_frames(
        image_rels=image_rels,
        labels_dir=REPO_ROOT / cfg["eval_gt"]["labels_dir"],
        manifest=manifest,
        pred_cache=cache,
        conf=conf,
        iou_match=float(eval_cfg["iou_match"]),
        dist_cfg=dist_cfg,
        tracker_cfg=diag_cfg["tracker"],
    )
    result = classify(frames, rules, diag_cfg["tracker"])
    summary = summarize(result)

    n_tracks = len({(fr["clip_id"], tid) for fr in frames for tid in fr["track_ids"]})
    totals = result["totals"]

    print(f"\nFrames={len(frames)}  GT={totals['gt']}  preds={totals['pred']}  tracks={n_tracks}")
    print(f"TP={totals['tp']}  FP={totals['fp']}  FN={totals['fn']}  conf={conf}")
    print(f"\n{'FP error type':<24} {'count':>7} {'% of FP':>9}")
    for key in list(ERROR_TYPES) + ["unclassified"]:
        row = summary["fp_primary"][key]
        print(f"{key:<24} {row['count']:7d} {row['pct_of_fp']:8.2f}%")
    print(f"\n{'FN error type':<24} {'count':>7} {'% of FN':>9}")
    for key, row in summary["fn"].items():
        print(f"{key:<24} {row['count']:7d} {row['pct_of_fn']:8.2f}%")

    crops = {"n_crops": 0, "skipped": True}
    if not args.no_crops:
        crops = mine_hard_negatives(
            result,
            REPO_ROOT / diag_cfg["crops_dir"],
            max_crops=int(diag_cfg["max_crops"]),
            pad_px=int(diag_cfg["crop_pad_px"]),
        )
        print(
            f"\nHard negatives: {crops['n_crops']} crops "
            f"(strict tracks={crops['n_strict_tracks']}, relaxed={crops['n_relaxed_tracks']})"
        )

    visuals = render_diagnostics(result, frames, REPO_ROOT / diag_cfg["visuals_dir"])
    print(f"Diagnostics grid: {visuals['grid']}")

    def trim(rows: list[dict], key: str) -> list[dict]:
        out: list[dict] = []
        seen: dict[str, int] = defaultdict(int)
        for row in sorted(rows, key=lambda r: (-r.get("conf", 0.0))):
            t = row[key]
            if seen[t] >= args.max_instances:
                continue
            seen[t] += 1
            out.append(row)
        return out

    report = {
        "role": "error_taxonomy",
        "weights": str(weights.relative_to(REPO_ROOT)),
        "device": device,
        "conf": conf,
        "conf_source": thresholds_note,
        "iou_match": float(eval_cfg["iou_match"]),
        "clips": sorted({fr["clip_id"] for fr in frames}),
        "n_frames": len(frames),
        "n_tracks": n_tracks,
        "totals": totals,
        "fp_taxonomy": summary["fp_primary"],
        "fp_taxonomy_any_tag": summary["fp_any_tag"],
        "fp_unclassified_profile": summary["residual_profile"],
        "fn_taxonomy": summary["fn"],
        "per_clip_fp": summary["per_clip_fp_primary"],
        "per_clip_fn": summary["per_clip_fn"],
        "tracker": {
            "type": "bytetrack_style_two_stage_iou",
            **diag_cfg["tracker"],
            "note": "Own implementation; keeps the CPU-only dependency set unchanged.",
        },
        "rules": rules,
        "large_gt_area_threshold_px2": result["large_gt_area_threshold"],
        "static_track_stats_top": result["static_stats"][:10],
        "hard_negatives": {
            "dir": str(diag_cfg["crops_dir"]),
            "n_crops": crops.get("n_crops", 0),
            "n_strict_tracks": len(result["strict_static"]),
            "n_relaxed_tracks": len(result["relaxed_static"]),
        },
        "visuals": visuals,
        "fp_instances": trim(result["fp_instances"], "error_type"),
        "fn_instances": result["fn_instances"][: args.max_instances],
        "note": (
            "Diagnostic pass over frozen hold-out predictions. Precedence for FPs: "
            + " > ".join(FP_PRECEDENCE)
            + ". 'fp_taxonomy_any_tag' counts every rule that fired."
        ),
    }
    report_path = REPO_ROOT / diag_cfg["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {report_path.relative_to(REPO_ROOT)}")
    print("OK: diagnostics only; model and thresholds untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
