#!/usr/bin/env python3
"""Final differential diagnostics: baseline vs DINO+SAM students on hold-out.

Task 1 — Load error_taxonomy_baseline.json and error_taxonomy_dinosam.json;
          print absolute / percentage deltas for every FP/FN bucket, with
          explicit callouts for gt_omission (auto-labeler validation) and the
          unclassified FP residual (persistent domain shift).

Task 2 — Far-band (200–400 m) pixel-area ceiling on the DINO+SAM student:
          median w·h for TPs vs FNs, to bound the spatial resolution the
          YOLO11n head needs before features collapse.

Task 3 — Kinematic failure isolation: high frame-to-frame GT translation
          (drone pitch/yaw proxy) coinciding with sudden TP→FN flips or
          broken pred track IDs → kinematic_drift, justifying VIO/stabilization
          offload on the Pi 5.

Task 4 — Human audit pack: top-50 residual FPs by conf (unclassified, or
          gt_omission when the omission rule saturates at high frozen conf)
          and top-50 near-band FNs by area, cropped into
          outputs/audit/edge_cases/ with a blank audit_tags.csv.
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

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import error_analysis as ea  # noqa: E402
import evaluate_custom as ev  # noqa: E402

REPO_ROOT = SRC_DIR.parent

FP_KEYS = list(ea.ERROR_TYPES) + ["unclassified"]
FN_KEYS = ["fractured_truck", "unclassified"]


# --------------------------------------------------------------------------- Task 1


def load_taxonomy(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Generate with error_analysis.py "
            "(baseline copy or --report-path for the A/B student)."
        )
    return json.loads(path.read_text())


def taxonomy_delta(baseline: dict, dinosam: dict) -> dict:
    """Absolute and relative deltas for every FP/FN category."""
    out: dict = {
        "baseline": {
            "weights": baseline.get("weights"),
            "conf": baseline.get("conf"),
            "totals": baseline.get("totals"),
        },
        "dinosam": {
            "weights": dinosam.get("weights"),
            "conf": dinosam.get("conf"),
            "totals": dinosam.get("totals"),
        },
        "fp": {},
        "fn": {},
        "totals_delta": {},
    }
    for key in FP_KEYS:
        b = baseline["fp_taxonomy"].get(key, {"count": 0, "pct_of_fp": 0.0})
        d = dinosam["fp_taxonomy"].get(key, {"count": 0, "pct_of_fp": 0.0})
        abs_delta = int(d["count"]) - int(b["count"])
        pct_point = round(float(d["pct_of_fp"]) - float(b["pct_of_fp"]), 2)
        rel = round(100.0 * abs_delta / b["count"], 1) if b["count"] else None
        out["fp"][key] = {
            "baseline_count": b["count"],
            "dinosam_count": d["count"],
            "abs_delta": abs_delta,
            "pct_of_fp_baseline": b["pct_of_fp"],
            "pct_of_fp_dinosam": d["pct_of_fp"],
            "pct_point_delta": pct_point,
            "rel_pct_vs_baseline": rel,
        }
    for key in FN_KEYS:
        b = baseline["fn_taxonomy"].get(key, {"count": 0, "pct_of_fn": 0.0})
        d = dinosam["fn_taxonomy"].get(key, {"count": 0, "pct_of_fn": 0.0})
        abs_delta = int(d["count"]) - int(b["count"])
        out["fn"][key] = {
            "baseline_count": b["count"],
            "dinosam_count": d["count"],
            "abs_delta": abs_delta,
            "pct_of_fn_baseline": b["pct_of_fn"],
            "pct_of_fn_dinosam": d["pct_of_fn"],
            "pct_point_delta": round(float(d["pct_of_fn"]) - float(b["pct_of_fn"]), 2),
            "rel_pct_vs_baseline": (
                round(100.0 * abs_delta / b["count"], 1) if b["count"] else None
            ),
        }
    for key in ("tp", "fp", "fn", "gt", "pred"):
        bv = int(baseline["totals"].get(key, 0))
        dv = int(dinosam["totals"].get(key, 0))
        out["totals_delta"][key] = {"baseline": bv, "dinosam": dv, "abs_delta": dv - bv}

    # Residual domain-shift bucket: FPs no geometry/static rule claimed.
    b_res = int(out["fp"]["gt_omission"]["baseline_count"]) + int(
        out["fp"]["unclassified"]["baseline_count"]
    )
    d_res = int(out["fp"]["gt_omission"]["dinosam_count"]) + int(
        out["fp"]["unclassified"]["dinosam_count"]
    )
    out["residual_fp"] = {
        "baseline_count": b_res,
        "dinosam_count": d_res,
        "abs_delta": d_res - b_res,
        "note": "gt_omission ∪ unclassified — FPs that are not motorcycle/fracture/static",
    }
    dino_conf = dinosam.get("conf")
    out["gt_omission_saturation"] = (
        dino_conf is not None and float(dino_conf) >= 0.50 and out["fp"]["unclassified"]["dinosam_count"] == 0
    )
    return out


def _rel_str(rel: float | None) -> str:
    return f"{rel:+.1f}%" if rel is not None else "n/a"


def print_delta_summary(delta: dict) -> None:
    print("\n=== Task 1: Differential taxonomy (baseline → DINO+SAM) ===")
    print(
        f"baseline  weights={delta['baseline']['weights']}  conf={delta['baseline']['conf']}"
    )
    print(
        f"dinosam   weights={delta['dinosam']['weights']}  conf={delta['dinosam']['conf']}"
    )
    print(f"\n{'FP category':<24} {'base':>6} {'dino':>6} {'Δ':>6} {'base%':>7} {'dino%':>7} {'Δpp':>7}")
    for key in FP_KEYS:
        row = delta["fp"][key]
        print(
            f"{key:<24} {row['baseline_count']:6d} {row['dinosam_count']:6d} "
            f"{row['abs_delta']:+6d} {row['pct_of_fp_baseline']:6.1f}% "
            f"{row['pct_of_fp_dinosam']:6.1f}% {row['pct_point_delta']:+6.1f}"
        )
    go = delta["fp"]["gt_omission"]
    un = delta["fp"]["unclassified"]
    res = delta["residual_fp"]
    print(
        f"\n  gt_omission: {go['baseline_count']} → {go['dinosam_count']} "
        f"({go['abs_delta']:+d}, {_rel_str(go['rel_pct_vs_baseline'])} rel) "
        "— auto-labeler / proxy-GT signal"
    )
    print(
        f"  unclassified FP:  {un['baseline_count']} → {un['dinosam_count']} "
        f"({un['abs_delta']:+d}, {un['pct_point_delta']:+.1f} pp of FP) "
        "— residual domain shift"
    )
    print(
        f"  residual FP (gt_omission ∪ unclassified): "
        f"{res['baseline_count']} → {res['dinosam_count']} ({res['abs_delta']:+d})"
    )
    if delta.get("gt_omission_saturation"):
        print(
            "  note: dinosam frozen conf equals gt_omission_conf (0.50), so the "
            "omission rule saturates and unclassified empties — not a domain-shift win. "
            "Read residual FP + the independent hold-out DINO+SAM audit for auto-labeler validation."
        )
    print(f"\n{'FN category':<24} {'base':>6} {'dino':>6} {'Δ':>6}")
    for key in FN_KEYS:
        row = delta["fn"][key]
        print(
            f"{key:<24} {row['baseline_count']:6d} {row['dinosam_count']:6d} "
            f"{row['abs_delta']:+6d}"
        )
    td = delta["totals_delta"]
    print(
        f"\n  totals  TP {td['tp']['baseline']}→{td['tp']['dinosam']} "
        f"FP {td['fp']['baseline']}→{td['fp']['dinosam']} "
        f"FN {td['fn']['baseline']}→{td['fn']['dinosam']}"
    )


# --------------------------------------------------------------------------- Task 2


def far_band_pixel_ceiling(frames: list[dict]) -> dict:
    """Median pixel area of far-band GT boxes that are TP vs FN (DINO+SAM)."""
    tp_areas: list[float] = []
    fn_areas: list[float] = []
    tp_boxes: list[dict] = []
    fn_boxes: list[dict] = []
    for fr in frames:
        matched_gt = {gi for gi, _ in fr["matches"]}
        for gi, box in enumerate(fr["gt_xyxy"]):
            if fr["gt_bands"][gi] != "far_200_400":
                continue
            area = ea.box_area(box)
            row = {
                "path": fr["path"],
                "clip_id": fr["clip_id"],
                "t_sec": fr["t_sec"],
                "gt_index": gi,
                "xyxy": [round(float(v), 1) for v in box],
                "area_px2": round(area, 1),
                "w_px": round(float(box[2] - box[0]), 1),
                "h_px": round(float(box[3] - box[1]), 1),
            }
            if gi in matched_gt:
                tp_areas.append(area)
                tp_boxes.append(row)
            else:
                fn_areas.append(area)
                fn_boxes.append(row)

    def stats(areas: list[float]) -> dict:
        if not areas:
            return {
                "n": 0,
                "median_px2": None,
                "p25_px2": None,
                "p75_px2": None,
                "min_px2": None,
                "max_px2": None,
                "median_side_px": None,
            }
        arr = np.asarray(areas, dtype=np.float64)
        return {
            "n": int(len(arr)),
            "median_px2": round(float(np.median(arr)), 1),
            "p25_px2": round(float(np.percentile(arr, 25)), 1),
            "p75_px2": round(float(np.percentile(arr, 75)), 1),
            "min_px2": round(float(arr.min()), 1),
            "max_px2": round(float(arr.max()), 1),
            "median_side_px": round(float(math.sqrt(np.median(arr))), 1),
        }

    tp_s, fn_s = stats(tp_areas), stats(fn_areas)
    # Collapse threshold: FN median is the size where the head typically fails;
    # TP median is the size where it still fires.
    return {
        "band": "far_200_400",
        "tp": tp_s,
        "fn": fn_s,
        "collapse_threshold_median_px2": fn_s["median_px2"],
        "operating_median_px2": tp_s["median_px2"],
        "note": (
            "Median far-band GT area for FNs ≈ minimum spatial footprint the "
            "YOLO11n head needs before feature collapse; TPs sit above it."
        ),
        "tp_examples": sorted(tp_boxes, key=lambda r: r["area_px2"])[:5],
        "fn_examples": sorted(fn_boxes, key=lambda r: -r["area_px2"])[:5],
        "tp_areas_px2": [round(a, 1) for a in tp_areas],
        "fn_areas_px2": [round(a, 1) for a in fn_areas],
        "_tp_boxes": tp_boxes,
        "_fn_boxes": fn_boxes,
    }


def print_far_band_summary(ceil: dict) -> None:
    print("\n=== Task 2: Far-band pixel density ceiling (DINO+SAM) ===")
    tp, fn = ceil["tp"], ceil["fn"]

    def line(label: str, s: dict) -> None:
        if not s["n"]:
            print(f"  {label}: n=0")
            return
        print(
            f"  {label}: n={s['n']}  median area={s['median_px2']} px²  "
            f"(~{s['median_side_px']} px side)  IQR [{s['p25_px2']}, {s['p75_px2']}]"
        )

    line("far-band TP", tp)
    line("far-band FN", fn)
    if tp["median_px2"] is not None and fn["median_px2"] is not None:
        print(
            f"  collapse threshold ≈ FN median {fn['median_px2']} px² "
            f"(side ~{fn['median_side_px']} px); "
            f"TP operating point {tp['median_px2']} px²"
        )


# --------------------------------------------------------------------------- Task 3


def associate_gt_across_frames(
    prev_boxes: np.ndarray, cur_boxes: np.ndarray, iou_thr: float
) -> list[tuple[int, int, float]]:
    """Greedy IoU association of GT boxes between consecutive frames."""
    if len(prev_boxes) == 0 or len(cur_boxes) == 0:
        return []
    ious = ev.iou_matrix(prev_boxes, cur_boxes)
    pairs = [
        (float(ious[i, j]), i, j)
        for i in range(len(prev_boxes))
        for j in range(len(cur_boxes))
        if ious[i, j] >= iou_thr
    ]
    pairs.sort(reverse=True)
    used_i, used_j = set(), set()
    out: list[tuple[int, int, float]] = []
    for iou, i, j in pairs:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        out.append((i, j, iou))
    return out


def isolate_kinematic_failures(
    frames: list[dict],
    *,
    move_frac: float,
    gt_iou_thr: float,
) -> dict:
    """Flag sudden TP→FN / broken tracks under high frame-to-frame translation."""
    by_clip: dict[str, list[dict]] = defaultdict(list)
    for fr in frames:
        by_clip[fr["clip_id"]].append(fr)

    events: list[dict] = []
    motion_series: list[dict] = []
    high_motion_frames = 0
    sudden_fn = 0
    broken_tracks = 0

    for clip_id, clip_frames in sorted(by_clip.items()):
        clip_frames = sorted(clip_frames, key=lambda f: f["frame_idx"])
        for k in range(1, len(clip_frames)):
            prev, cur = clip_frames[k - 1], clip_frames[k]
            assoc = associate_gt_across_frames(prev["gt_xyxy"], cur["gt_xyxy"], gt_iou_thr)
            if not assoc:
                continue
            shifts = []
            for pi, ci, _iou in assoc:
                pc = ea.centroid(prev["gt_xyxy"][pi])
                cc = ea.centroid(cur["gt_xyxy"][ci])
                shifts.append(math.hypot(cc[0] - pc[0], cc[1] - pc[1]))
            median_shift = float(np.median(shifts))
            budget = move_frac * float(cur["img_w"])
            motion_series.append(
                {
                    "clip_id": clip_id,
                    "t_sec": round(float(cur["t_sec"]), 2),
                    "median_shift_px": round(median_shift, 1),
                    "shift_budget_px": round(budget, 1),
                    "high_motion": bool(median_shift >= budget),
                }
            )
            if median_shift < budget:
                continue
            high_motion_frames += 1

            prev_matched_gt = {gi for gi, _ in prev["matches"]}
            cur_matched_gt = {gi for gi, _ in cur["matches"]}
            prev_pred_of_gt = {gi: pj for gi, pj in prev["matches"]}

            for pi, ci, iou in assoc:
                # Sudden FN: was a TP last frame, missed now, under high ego-motion.
                if pi in prev_matched_gt and ci not in cur_matched_gt:
                    sudden_fn += 1
                    pc = ea.centroid(prev["gt_xyxy"][pi])
                    cc = ea.centroid(cur["gt_xyxy"][ci])
                    events.append(
                        {
                            "error_type": "kinematic_drift",
                            "subtype": "sudden_fn",
                            "clip_id": clip_id,
                            "path": cur["path"],
                            "prev_path": prev["path"],
                            "t_sec": cur["t_sec"],
                            "median_shift_px": round(median_shift, 1),
                            "shift_budget_px": round(budget, 1),
                            "gt_translation_px": round(
                                math.hypot(cc[0] - pc[0], cc[1] - pc[1]), 1
                            ),
                            "gt_xyxy": [round(float(v), 1) for v in cur["gt_xyxy"][ci]],
                            "prev_gt_xyxy": [round(float(v), 1) for v in prev["gt_xyxy"][pi]],
                            "gt_band": cur["gt_bands"][ci],
                            "gt_area_px2": round(ea.box_area(cur["gt_xyxy"][ci]), 1),
                        }
                    )

                # Broken track: GT still matched, but the pred track id changed.
                if pi in prev_matched_gt and ci in cur_matched_gt:
                    prev_pj = prev_pred_of_gt[pi]
                    cur_pj = next(pj for gi, pj in cur["matches"] if gi == ci)
                    prev_tid = prev["track_ids"][prev_pj]
                    cur_tid = cur["track_ids"][cur_pj]
                    if prev_tid != cur_tid:
                        broken_tracks += 1
                        events.append(
                            {
                                "error_type": "kinematic_drift",
                                "subtype": "broken_track",
                                "clip_id": clip_id,
                                "path": cur["path"],
                                "prev_path": prev["path"],
                                "t_sec": cur["t_sec"],
                                "median_shift_px": round(median_shift, 1),
                                "shift_budget_px": round(budget, 1),
                                "prev_track_id": int(prev_tid),
                                "cur_track_id": int(cur_tid),
                                "gt_xyxy": [round(float(v), 1) for v in cur["gt_xyxy"][ci]],
                                "gt_band": cur["gt_bands"][ci],
                            }
                        )

    return {
        "move_frac": move_frac,
        "gt_iou_thr": gt_iou_thr,
        "n_high_motion_frame_pairs": high_motion_frames,
        "n_sudden_fn": sudden_fn,
        "n_broken_tracks": broken_tracks,
        "n_events": len(events),
        "events": events[:200],
        "motion_series": motion_series,
        "note": (
            "High frame-to-frame GT centroid translation is a proxy for aggressive "
            "drone pitch/yaw. Sudden TP→FN or track-id breaks under that motion are "
            "logged as kinematic_drift — justify VIO / stabilization offload on Pi 5."
        ),
    }


def print_kinematic_summary(kin: dict) -> None:
    print("\n=== Task 3: Kinematic failure isolation ===")
    print(
        f"  high-motion frame pairs (median GT shift ≥ {kin['move_frac']*100:.1f}% img_w): "
        f"{kin['n_high_motion_frame_pairs']}"
    )
    print(f"  sudden_fn events:    {kin['n_sudden_fn']}")
    print(f"  broken_track events: {kin['n_broken_tracks']}")
    print(f"  kinematic_drift total logged: {kin['n_events']}")


# --------------------------------------------------------------------------- Task 4


def build_audit_lists(result: dict, frames: list[dict]) -> tuple[list[dict], list[dict]]:
    """Top residual FPs by conf; top near-band FNs by area.

    At a high frozen conf (≈ gt_omission threshold 0.50) the omission rule
    saturates and the unclassified bucket can empty. Audit the residual
    high-conf FPs: unclassified ∪ gt_omission (motorcycle/fracture/static stay out).
    """
    residual = {"unclassified", "gt_omission"}
    fps = []
    for inst in result["fp_instances"]:
        if inst["error_type"] not in residual:
            continue
        row = dict(inst)
        row["area_px2"] = round(ea.box_area(np.asarray(inst["xyxy"], dtype=np.float64)), 1)
        fps.append(row)
    fps.sort(key=lambda r: -float(r["conf"]))

    # Near-band FNs from frame bookkeeping (taxonomy only stores fractured FN rows).
    fractured = {(inst["path"], inst["gt_index"]) for inst in result["fn_instances"]}
    fns: list[dict] = []
    for fr in frames:
        matched = {gi for gi, _ in fr["matches"]}
        for gi in fr["unmatched_gt"]:
            if fr["gt_bands"][gi] != "near_0_200":
                continue
            if (fr["path"], gi) in fractured:
                continue
            box = fr["gt_xyxy"][gi]
            fns.append(
                {
                    "error_type": "near_band_fn",
                    "clip_id": fr["clip_id"],
                    "path": fr["path"],
                    "frame_idx": fr["frame_idx"],
                    "t_sec": fr["t_sec"],
                    "gt_index": gi,
                    "gt_band": fr["gt_bands"][gi],
                    "xyxy": [round(float(v), 1) for v in box],
                    "area_px2": round(ea.box_area(box), 1),
                    "conf": None,
                }
            )
    fns.sort(key=lambda r: -float(r["area_px2"]))
    return fps, fns


def crop_and_export(
    rows: list[dict],
    *,
    out_dir: Path,
    prefix: str,
    pad_px: int,
    limit: int,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, np.ndarray] = {}
    exported: list[dict] = []
    for i, row in enumerate(rows[:limit]):
        img = cache.get(row["path"])
        if img is None:
            img = cv2.imread(str(REPO_ROOT / row["path"]))
            if img is None:
                continue
            cache[row["path"]] = img
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in row["xyxy"]]
        cx1 = max(0, int(x1) - pad_px)
        cy1 = max(0, int(y1) - pad_px)
        cx2 = min(w, int(x2) + pad_px)
        cy2 = min(h, int(y2) + pad_px)
        if cx2 - cx1 < 4 or cy2 - cy1 < 4:
            continue
        crop = img[cy1:cy2, cx1:cx2].copy()
        # Draw the box relative to the crop for the auditor.
        cv2.rectangle(
            crop,
            (int(x1) - cx1, int(y1) - cy1),
            (int(x2) - cx1, int(y2) - cy1),
            (0, 255, 255) if prefix.startswith("fn") else (0, 0, 255),
            2,
        )
        stem = Path(row["path"]).stem
        conf_tag = f"_c{row['conf']:.2f}" if row.get("conf") is not None else ""
        area_tag = f"_a{int(row['area_px2'])}" if row.get("area_px2") is not None else ""
        fname = f"{prefix}_{i:02d}_{row['clip_id']}_{stem}{conf_tag}{area_tag}.jpg"
        out_path = out_dir / fname
        cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        pred_type = "FP" if prefix.startswith("fp") else "FN"
        exported.append(
            {
                "filename": fname,
                "prediction_type": pred_type,
                "semantic_cause": "",
                "clip_id": row["clip_id"],
                "path": row["path"],
                "t_sec": row.get("t_sec"),
                "conf": row.get("conf"),
                "area_px2": row.get("area_px2"),
                "band": row.get("band") or row.get("gt_band"),
                "error_type": row.get("error_type"),
                "xyxy": row["xyxy"],
            }
        )
    return exported


def write_audit_pack(
    fps: list[dict],
    fns: list[dict],
    *,
    out_dir: Path,
    pad_px: int,
    top_n: int,
) -> dict:
    if out_dir.exists():
        for old in out_dir.glob("*.jpg"):
            old.unlink()
        csv_path = out_dir / "audit_tags.csv"
        if csv_path.exists():
            csv_path.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    fp_rows = crop_and_export(fps, out_dir=out_dir, prefix="fp_residual", pad_px=pad_px, limit=top_n)
    fn_rows = crop_and_export(fns, out_dir=out_dir, prefix="fn_near", pad_px=pad_px, limit=top_n)

    # Blank 15-minute review sheet — exactly the three columns the auditor fills.
    csv_path = out_dir / "audit_tags.csv"
    fieldnames = ["filename", "prediction_type", "semantic_cause"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in fp_rows + fn_rows:
            writer.writerow(
                {
                    "filename": row["filename"],
                    "prediction_type": row["prediction_type"],
                    "semantic_cause": "",
                }
            )

    # Machine-readable companion with boxes (not required for the 15-min review).
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "n_fp": len(fp_rows),
                "n_fn": len(fn_rows),
                "fp": fp_rows,
                "fn": fn_rows,
                "audit_tags_csv": str(csv_path.relative_to(REPO_ROOT)),
                "note": "Fill semantic_cause during the 15-minute human audit.",
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "dir": str(out_dir.relative_to(REPO_ROOT)),
        "n_fp_crops": len(fp_rows),
        "n_fn_crops": len(fn_rows),
        "audit_tags_csv": str(csv_path.relative_to(REPO_ROOT)),
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
    }


def print_audit_summary(audit: dict) -> None:
    print("\n=== Task 4: Human audit extraction ===")
    print(
        f"  exported {audit['n_fp_crops']} residual FP crops (unclassified ∪ gt_omission, "
        f"highest conf) + {audit['n_fn_crops']} near-band FN crops (largest area) → {audit['dir']}"
    )
    print(f"  blank review sheet: {audit['audit_tags_csv']}")
    print("  columns: filename, prediction_type, semantic_cause")


# --------------------------------------------------------------------------- presentation figures


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "axes.axisbelow": True,
        }
    )
    return plt


def _read_bgr(rel: str) -> np.ndarray | None:
    img = cv2.imread(str(REPO_ROOT / rel))
    return img


def _crop_box(img: np.ndarray, xyxy: list, pad: int, color: tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    cx1 = max(0, int(x1) - pad)
    cy1 = max(0, int(y1) - pad)
    cx2 = min(w, int(x2) + pad)
    cy2 = min(h, int(y2) + pad)
    crop = img[cy1:cy2, cx1:cx2].copy()
    cv2.rectangle(
        crop,
        (int(x1) - cx1, int(y1) - cy1),
        (int(x2) - cx1, int(y2) - cy1),
        color,
        2,
    )
    return crop


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _nearest_box(boxes: list[dict], target_area: float | None) -> dict | None:
    if not boxes or target_area is None:
        return None
    return min(boxes, key=lambda r: abs(float(r["area_px2"]) - float(target_area)))


def render_task1_figure(delta: dict, out_path: Path) -> Path:
    plt = _mpl()
    labels = ["gt_omission", "unclassified", "motorcycle", "fracture", "static"]
    keys = [
        "gt_omission",
        "unclassified",
        "suspected_motorcycle",
        "fractured_truck",
        "static_hallucination",
    ]
    base = [delta["fp"][k]["baseline_count"] for k in keys]
    dino = [delta["fp"][k]["dinosam_count"] for k in keys]
    x = np.arange(len(labels))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    ax.bar(x - width / 2, base, width, label="Baseline (conf=0.20)", color="#4C78A8")
    ax.bar(x + width / 2, dino, width, label="DINO+SAM (conf=0.50)", color="#F58518")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Task 1 — Hold-out FP taxonomy")
    ax.legend(frameon=False, loc="upper left")

    ax2 = axes[1]
    t_labels = ["TP", "FP", "FN"]
    t_keys = ["tp", "fp", "fn"]
    t_base = [delta["totals_delta"][k]["baseline"] for k in t_keys]
    t_dino = [delta["totals_delta"][k]["dinosam"] for k in t_keys]
    xt = np.arange(len(t_labels))
    ax2.bar(xt - width / 2, t_base, width, color="#4C78A8")
    ax2.bar(xt + width / 2, t_dino, width, color="#F58518")
    ax2.set_xticks(xt, t_labels)
    ax2.set_ylabel("Count")
    ax2.set_title("Hold-out totals")
    for i, (b, d) in enumerate(zip(t_base, t_dino)):
        ax2.annotate(f"{d - b:+d}", (xt[i], max(b, d) + 8), ha="center", fontsize=9, color="#333")

    res = delta["residual_fp"]
    fig.suptitle(
        f"Residual FP (omission ∪ unclassified): {res['baseline_count']} → {res['dinosam_count']} "
        f"({res['abs_delta']:+d})   ·   omission jump is a conf=0.50 rule-saturation artifact",
        fontsize=10,
        y=0.02,
        color="#444",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_task2_figure(ceil: dict, out_path: Path) -> Path:
    plt = _mpl()
    tp_areas = np.asarray(ceil.get("tp_areas_px2") or [], dtype=np.float64)
    fn_areas = np.asarray(ceil.get("fn_areas_px2") or [], dtype=np.float64)
    tp_side = np.sqrt(tp_areas) if len(tp_areas) else tp_areas
    fn_side = np.sqrt(fn_areas) if len(fn_areas) else fn_areas

    fig = plt.figure(figsize=(12.4, 5.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1, 1], height_ratios=[1.15, 1], hspace=0.42, wspace=0.28)
    ax_hist = fig.add_subplot(gs[:, 0])
    bins = np.linspace(15, 55, 13)
    if len(fn_side):
        ax_hist.hist(fn_side, bins=bins, color="#E45756", alpha=0.55, label=f"FN  n={len(fn_side)}")
    if len(tp_side):
        ax_hist.hist(tp_side, bins=bins, color="#54A24B", alpha=0.75, label=f"TP  n={len(tp_side)}")
    if ceil["fn"]["median_side_px"] is not None:
        ax_hist.axvline(
            ceil["fn"]["median_side_px"],
            color="#E45756",
            linestyle="--",
            linewidth=1.4,
            label=f"FN median {ceil['fn']['median_side_px']} px",
        )
    if ceil["tp"]["median_side_px"] is not None:
        ax_hist.axvline(
            ceil["tp"]["median_side_px"],
            color="#54A24B",
            linestyle="--",
            linewidth=1.4,
            label=f"TP median {ceil['tp']['median_side_px']} px",
        )
    ax_hist.axvline(10 * 3840 / 1280, color="#9D755D", linestyle=":", linewidth=1.3, label="~10 px on 1280 tensor")
    ax_hist.set_xlabel("Equivalent side √(w·h)  (px, native 4K)")
    ax_hist.set_ylabel("Far-band GT boxes")
    ax_hist.set_title("Task 2 — Far-band pixel ceiling (200–400 m)")
    ax_hist.legend(frameon=False, loc="upper right")

    ax_box = fig.add_subplot(gs[0, 1:])
    data = [fn_areas, tp_areas] if len(fn_areas) or len(tp_areas) else [[], []]
    bp = ax_box.boxplot(
        data,
        tick_labels=["FN", "TP"],
        orientation="horizontal",
        widths=0.55,
        patch_artist=True,
        showfliers=True,
    )
    for patch, color in zip(bp["boxes"], ["#E45756", "#54A24B"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    ax_box.set_xlabel("GT box area  (px²)")
    ax_box.set_title("Same size band — detections are not the large tail")

    tp_box = _nearest_box(ceil.get("_tp_boxes") or [], ceil["tp"].get("median_px2"))
    fn_box = _nearest_box(ceil.get("_fn_boxes") or [], ceil["fn"].get("median_px2"))
    for col, row, title, color in (
        (1, fn_box, "Median-ish far FN", "#E45756"),
        (2, tp_box, "Median-ish far TP", "#54A24B"),
    ):
        ax = fig.add_subplot(gs[1, col])
        ax.set_title(title, color=color, fontsize=10)
        ax.axis("off")
        if row is None:
            continue
        img = _read_bgr(row["path"])
        if img is None:
            continue
        pad = max(24, int(0.8 * max(row["w_px"], row["h_px"])))
        crop = _bgr_to_rgb(_crop_box(img, row["xyxy"], pad))
        ax.imshow(crop)
        ax.set_xlabel(
            f"{row['clip_id']} t={row['t_sec']:.1f}s   {row['w_px']:.0f}×{row['h_px']:.0f} px   "
            f"{row['area_px2']:.0f} px²",
            fontsize=8,
        )

    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.08, hspace=0.42, wspace=0.28)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_task3_figure(kin: dict, out_path: Path) -> Path:
    plt = _mpl()
    series = kin.get("motion_series") or []
    events = kin.get("events") or []
    fig = plt.figure(figsize=(12.4, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.38, wspace=0.22)
    ax_e = fig.add_subplot(gs[0, 0])
    ax_f = fig.add_subplot(gs[0, 1], sharey=ax_e)

    for ax, clip_id in ((ax_e, "E"), (ax_f, "F")):
        rows = [r for r in series if r["clip_id"] == clip_id]
        if not rows:
            ax.set_title(f"Clip {clip_id} — no associated GT pairs")
            continue
        t = [r["t_sec"] for r in rows]
        s = [r["median_shift_px"] for r in rows]
        budget = rows[0]["shift_budget_px"]
        ax.fill_between(t, budget, max(s + [budget]) * 1.05, color="#F2CF5B", alpha=0.18, label="high motion")
        ax.plot(t, s, color="#4C78A8", linewidth=1.4, label="median GT shift")
        ax.axhline(budget, color="#9D755D", linestyle="--", linewidth=1.2, label=f"budget {budget:.0f} px")
        clip_events = [e for e in events if e["clip_id"] == clip_id]
        fn_t = [e["t_sec"] for e in clip_events if e["subtype"] == "sudden_fn"]
        fn_s = [e["median_shift_px"] for e in clip_events if e["subtype"] == "sudden_fn"]
        br_t = [e["t_sec"] for e in clip_events if e["subtype"] == "broken_track"]
        br_s = [e["median_shift_px"] for e in clip_events if e["subtype"] == "broken_track"]
        if fn_t:
            ax.scatter(fn_t, fn_s, color="#E45756", s=36, zorder=3, label="sudden FN")
        if br_t:
            ax.scatter(br_t, br_s, color="#B279A2", marker="D", s=32, zorder=3, label="broken track")
        ax.set_xlabel("t (s)")
        ax.set_title(f"Clip {clip_id}")
        ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax_e.set_ylabel("Median GT centroid shift (px)")
    fig.suptitle(
        f"Task 3 — Kinematic drift   high-motion pairs={kin['n_high_motion_frame_pairs']}   "
        f"sudden_fn={kin['n_sudden_fn']}   broken_track={kin['n_broken_tracks']}",
        fontsize=13,
        y=0.98,
    )

    example = next((e for e in events if e["subtype"] == "sudden_fn"), None)
    ax_prev = fig.add_subplot(gs[1, 0])
    ax_cur = fig.add_subplot(gs[1, 1])
    for ax, title in ((ax_prev, "t−Δ  (was TP)"), (ax_cur, "t  (sudden FN)")):
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    if example is not None:
        prev_img = _read_bgr(example["prev_path"])
        cur_img = _read_bgr(example["path"])
        xyxy = example["gt_xyxy"]
        pad = 80
        if prev_img is not None:
            prev_xyxy = example.get("prev_gt_xyxy") or xyxy
            crop = _bgr_to_rgb(_crop_box(prev_img, prev_xyxy, pad, color=(0, 220, 0)))
            ax_prev.imshow(crop)
            ax_prev.set_xlabel(
                f"{example['clip_id']}  {Path(example['prev_path']).name}  "
                f"shift={example['median_shift_px']} px",
                fontsize=8,
            )
        if cur_img is not None:
            crop = _bgr_to_rgb(_crop_box(cur_img, xyxy, pad, color=(0, 255, 255)))
            ax_cur.imshow(crop)
            ax_cur.set_xlabel(
                f"{example['clip_id']}  t={example['t_sec']:.1f}s  "
                f"translation={example.get('gt_translation_px')} px",
                fontsize=8,
            )

    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.08, hspace=0.35, wspace=0.22)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_presentation_figures(
    *,
    delta: dict,
    ceil: dict,
    kin: dict,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "task1": str(render_task1_figure(delta, out_dir / "task1_taxonomy_delta.png").relative_to(REPO_ROOT)),
        "task2": str(render_task2_figure(ceil, out_dir / "task2_far_band_ceiling.png").relative_to(REPO_ROOT)),
        "task3": str(render_task3_figure(kin, out_dir / "task3_kinematic_drift.png").relative_to(REPO_ROOT)),
    }
    print("\n=== Presentation figures ===")
    for key, rel in paths.items():
        print(f"  {key}: {rel}")
    return {"dir": str(out_dir.relative_to(REPO_ROOT)), **paths}


# --------------------------------------------------------------------------- pipeline helpers


def load_student_frames(
    *,
    weights: Path,
    thresholds_path: Path,
    cfg: dict,
    refresh: bool,
    device: str | None,
) -> tuple[list[dict], dict, float]:
    """Predict + classify one student; reuse error_analysis helpers."""
    eval_cfg = cfg["evaluation"]
    diag_cfg = cfg["diagnostics"]
    dist_cfg = cfg["distance"]
    splits_dir = REPO_ROOT / cfg["paths"]["splits_dir"]
    frozen = json.loads(thresholds_path.read_text())
    conf = float(frozen["chosen_conf"])
    image_rels = ev.read_split_list(splits_dir / "eval.txt")
    manifest = ev.load_manifest(splits_dir / "manifest.csv")
    device = ev.pick_device(device)

    cache_dir = REPO_ROOT / "runs" / "cache"
    cache_key = {
        "weights": str(weights.relative_to(REPO_ROOT)),
        "weights_mtime": int(weights.stat().st_mtime),
        "imgsz": int(eval_cfg["imgsz"]),
        "conf_floor": conf,
        "nms_iou": float(eval_cfg["nms_iou"]),
    }
    cache = None if refresh else ea.load_pred_cache(cache_dir, cache_key, image_rels)
    if cache is None:
        cache = ev.predict_cached(
            image_rels=image_rels,
            weights=weights,
            imgsz=int(eval_cfg["imgsz"]),
            conf_floor=conf,
            nms_iou=float(eval_cfg["nms_iou"]),
            device=device,
        )
        ea.save_pred_cache(cache_dir, cache_key, cache)
    else:
        print(f"Using cached predictions for {len(image_rels)} frames")

    frames = ea.collect_frames(
        image_rels=image_rels,
        labels_dir=REPO_ROOT / cfg["eval_gt"]["labels_dir"],
        manifest=manifest,
        pred_cache=cache,
        conf=conf,
        iou_match=float(eval_cfg["iou_match"]),
        dist_cfg=dist_cfg,
        tracker_cfg=diag_cfg["tracker"],
    )
    result = ea.classify(frames, diag_cfg["rules"], diag_cfg["tracker"])
    return frames, result, conf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "data.yaml")
    parser.add_argument(
        "--baseline-taxonomy",
        type=Path,
        default=REPO_ROOT / "data" / "splits" / "error_taxonomy_baseline.json",
    )
    parser.add_argument(
        "--dinosam-taxonomy",
        type=Path,
        default=REPO_ROOT / "data" / "splits" / "error_taxonomy_dinosam.json",
    )
    parser.add_argument(
        "--dinosam-weights",
        type=Path,
        default=None,
        help="Default: evaluation.dinosam.weights",
    )
    parser.add_argument(
        "--dinosam-thresholds",
        type=Path,
        default=None,
        help="Default: evaluation.dinosam.thresholds_path",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "splits" / "final_diagnostics.json",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "audit" / "edge_cases",
    )
    parser.add_argument(
        "--visuals-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "diagnostics_final",
    )
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--pad-px", type=int, default=16)
    parser.add_argument(
        "--kinematic-move-frac",
        type=float,
        default=0.02,
        help="Median GT centroid shift ≥ this fraction of image width ⇒ high motion",
    )
    parser.add_argument("--gt-assoc-iou", type=float, default=0.3)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Deltas + ceilings only; skip crop export",
    )
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Skip presentation figure export",
    )
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    cfg = ev.load_config(cfg_path)
    dinosam_cfg = (cfg.get("evaluation") or {}).get("dinosam") or {}

    def _resolve(p: Path | str | None, default: str | Path) -> Path:
        path = Path(p if p is not None else default)
        return path if path.is_absolute() else REPO_ROOT / path

    baseline_path = _resolve(args.baseline_taxonomy, "data/splits/error_taxonomy_baseline.json")
    dinosam_tax_path = _resolve(args.dinosam_taxonomy, "data/splits/error_taxonomy_dinosam.json")
    weights = _resolve(
        args.dinosam_weights,
        dinosam_cfg.get("weights", "runs/train/yolo11n_dinosam/weights/best.pt"),
    )
    thr_path = _resolve(
        args.dinosam_thresholds,
        dinosam_cfg.get("thresholds_path", "data/splits/eval_thresholds_dinosam.json"),
    )
    out_path = _resolve(args.out, "data/splits/final_diagnostics.json")
    audit_dir = _resolve(args.audit_dir, "outputs/audit/edge_cases")
    visuals_dir = _resolve(args.visuals_dir, "outputs/diagnostics_final")
    if not weights.exists():
        raise FileNotFoundError(f"Missing DINO+SAM weights: {weights}")
    if not thr_path.exists():
        raise FileNotFoundError(f"Missing DINO+SAM thresholds: {thr_path}")

    baseline = load_taxonomy(baseline_path)
    dinosam_tax = load_taxonomy(dinosam_tax_path)
    delta = taxonomy_delta(baseline, dinosam_tax)
    print_delta_summary(delta)

    print(
        f"\nLoading DINO+SAM frames for Tasks 2–4  weights={weights.relative_to(REPO_ROOT)}  "
        f"thresholds={thr_path.relative_to(REPO_ROOT)}"
    )
    frames, result, conf = load_student_frames(
        weights=weights,
        thresholds_path=thr_path,
        cfg=cfg,
        refresh=args.refresh,
        device=args.device,
    )

    ceil = far_band_pixel_ceiling(frames)
    print_far_band_summary(ceil)

    kin = isolate_kinematic_failures(
        frames,
        move_frac=float(args.kinematic_move_frac),
        gt_iou_thr=float(args.gt_assoc_iou),
    )
    print_kinematic_summary(kin)

    figures: dict = {"skipped": True}
    if not args.skip_visuals:
        figures = render_presentation_figures(
            delta=delta,
            ceil=ceil,
            kin=kin,
            out_dir=visuals_dir,
        )

    audit: dict = {"skipped": True}
    if not args.skip_audit:
        fps, fns = build_audit_lists(result, frames)
        audit = write_audit_pack(
            fps,
            fns,
            out_dir=audit_dir,
            pad_px=int(args.pad_px),
            top_n=int(args.top_n),
        )
        print_audit_summary(audit)

    ceil_public = {k: v for k, v in ceil.items() if not str(k).startswith("_")}
    payload = {
        "role": "final_differential_diagnostics",
        "baseline_taxonomy": str(baseline_path.relative_to(REPO_ROOT)),
        "dinosam_taxonomy": str(dinosam_tax_path.relative_to(REPO_ROOT)),
        "dinosam_weights": str(weights.relative_to(REPO_ROOT)),
        "dinosam_conf": conf,
        "taxonomy_delta": delta,
        "far_band_pixel_ceiling": ceil_public,
        "kinematic_drift": {
            k: kin[k]
            for k in (
                "move_frac",
                "gt_iou_thr",
                "n_high_motion_frame_pairs",
                "n_sudden_fn",
                "n_broken_tracks",
                "n_events",
                "note",
            )
        },
        "kinematic_drift_events": kin["events"],
        "kinematic_drift_motion": kin.get("motion_series") or [],
        "human_audit": audit,
        "presentation_figures": figures,
        "note": (
            "Task 1 uses the two taxonomy JSON files. Tasks 2–4 recompute on the "
            "DINO+SAM student with its frozen val conf. Fill semantic_cause in "
            "audit_tags.csv during the 15-minute review."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {out_path.relative_to(REPO_ROOT)}")
    print("OK: final diagnostics only; models and thresholds untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
