#!/usr/bin/env python3
"""Render a short annotated hold-out video for reviewer demos.

Loads a student checkpoint, freezes conf from a thresholds JSON (or --conf),
scores the eval split against the frozen proxy GT, draws the same match view
used in submission stills (GT matched / miss / pred TP / FP), and encodes an
H.264 MP4 via ffmpeg. Eval clips only; train/val are refused.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import evaluate_custom as ev  # noqa: E402

REPO_ROOT = SRC_DIR.parent


def encode_mp4(frame_paths: list[Path], out_mp4: Path, fps: float) -> None:
    if not frame_paths:
        raise RuntimeError("No frames to encode")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eval_vid_") as tmp:
        tmp_dir = Path(tmp)
        list_file = tmp_dir / "frames.txt"
        # Copy/symlink into a contiguous %06d sequence so ffmpeg concat is boring.
        for i, src in enumerate(frame_paths):
            dst = tmp_dir / f"{i:06d}.jpg"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())
        list_file.write_text(
            "".join(f"file '{tmp_dir / f'{i:06d}.jpg'}'\n" for i in range(len(frame_paths)))
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(tmp_dir / "%06d.jpg"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(out_mp4),
        ]
        subprocess.run(cmd, check=True)


def render_video_frame(fr: dict, *, conf: float, max_side: int) -> np.ndarray:
    """Match-view panel with a denser banner suited to video playback."""
    img = ev.render_combined_panel(fr, max_side=max_side)
    n_tp = len(fr["matches"])
    n_fp = len(fr["unmatched_pred"])
    n_fn = len(fr["unmatched_gt"])
    n_far = sum(1 for b in fr["gt_bands"] if b == "far_200_400")
    t_sec = fr.get("t_sec")
    t_txt = f"t={t_sec:.1f}s" if isinstance(t_sec, (int, float)) else ""
    banner = (
        f"{fr['clip_id']} {Path(fr['path']).stem}  {t_txt}  "
        f"conf>={conf:.2f}  TP={n_tp} FP={n_fp} FN={n_fn}  farGT={n_far}"
    )
    # Re-stamp the top banner with timing info (render_combined_panel already drew one).
    return ev._banner(img, banner)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "data.yaml")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Student weights (default: evaluation.dinosam.weights if set, else evaluation.weights)",
    )
    parser.add_argument(
        "--thresholds-path",
        type=Path,
        default=None,
        help="Frozen conf JSON (default: evaluation.dinosam.thresholds_path)",
    )
    parser.add_argument("--conf", type=float, default=None, help="Override frozen conf")
    parser.add_argument(
        "--clips",
        nargs="+",
        default=None,
        help="Eval clip ids only (default: all eval clips)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "videos",
    )
    parser.add_argument(
        "--tag",
        default="dinosam",
        help="Filename tag, e.g. dinosam or clean → eval_F_dinosam.mp4",
    )
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument(
        "--playback-fps",
        type=float,
        default=None,
        help="Encode fps (default: sampling.eval_fps from config, usually 5)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth sampled frame (1 = all eval samples)",
    )
    parser.add_argument(
        "--limit-per-clip",
        type=int,
        default=None,
        help="Cap frames per clip after stride (for a shorter cut)",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also write one MP4 concatenating all requested clips in order",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    cfg = ev.load_config(cfg_path)
    eval_cfg = cfg["evaluation"]
    dinosam_cfg = eval_cfg.get("dinosam") or {}
    dist_cfg = cfg["distance"]
    splits_dir = REPO_ROOT / cfg["paths"]["splits_dir"]
    eval_ids = {c["id"] for c in cfg["clips"]["eval"]}

    weights = Path(
        args.weights
        or dinosam_cfg.get("weights")
        or eval_cfg["weights"]
    )
    if not weights.is_absolute():
        weights = REPO_ROOT / weights
    if not weights.exists():
        raise FileNotFoundError(weights)

    thr_path = Path(
        args.thresholds_path
        or dinosam_cfg.get("thresholds_path")
        or eval_cfg["thresholds_path"]
    )
    if not thr_path.is_absolute():
        thr_path = REPO_ROOT / thr_path
    frozen = json.loads(thr_path.read_text()) if thr_path.exists() else {}
    conf = float(args.conf if args.conf is not None else frozen.get("chosen_conf", 0.25))
    nms_iou = float(frozen.get("chosen_nms_iou", eval_cfg["nms_iou"]))
    conf_floor = float(eval_cfg["predict_conf_floor"])
    imgsz = int(eval_cfg["imgsz"])
    iou_match = float(eval_cfg["iou_match"])
    device = ev.pick_device(args.device)
    playback_fps = float(args.playback_fps or cfg["sampling"]["eval_fps"])

    eval_rels = ev.read_split_list(splits_dir / "eval.txt")
    bad = [p for p in eval_rels if Path(p).parent.name not in eval_ids]
    if bad:
        raise RuntimeError(f"Non-eval path in eval list: {bad[:3]}")
    clip_ids = set(args.clips) if args.clips else None
    if clip_ids:
        unknown = clip_ids - eval_ids
        if unknown:
            raise RuntimeError(f"--clips must be eval ids; got {sorted(unknown)}")
        eval_rels = [p for p in eval_rels if Path(p).parent.name in clip_ids]
    if args.stride > 1:
        # Preserve clip order; stride inside each clip so both E and F stay represented.
        by_clip: dict[str, list[str]] = defaultdict(list)
        for rel in eval_rels:
            by_clip[Path(rel).parent.name].append(rel)
        eval_rels = []
        for clip_id in sorted(by_clip):
            eval_rels.extend(by_clip[clip_id][:: args.stride])
    if args.limit_per_clip:
        by_clip = defaultdict(list)
        for rel in eval_rels:
            by_clip[Path(rel).parent.name].append(rel)
        eval_rels = []
        for clip_id in sorted(by_clip):
            eval_rels.extend(by_clip[clip_id][: args.limit_per_clip])

    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    frames_dir = out_dir / f"frames_{args.tag}"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Eval video: {len(eval_rels)} frames  weights={weights.relative_to(REPO_ROOT)}  "
        f"conf>={conf:.2f}  playback={playback_fps} fps  device={device}"
    )
    cache = ev.predict_cached(
        image_rels=eval_rels,
        weights=weights,
        imgsz=imgsz,
        conf_floor=min(conf_floor, conf),
        nms_iou=nms_iou,
        device=device,
    )
    labels_dir = REPO_ROOT / cfg["eval_gt"]["labels_dir"]
    manifest = ev.load_manifest(splits_dir / "manifest.csv")
    _report, detail = ev.score_split(
        image_rels=eval_rels,
        labels_dir=labels_dir,
        manifest=manifest,
        pred_cache=cache,
        conf=conf,
        iou_match=iou_match,
        dist_cfg=dist_cfg,
    )

    # Attach t_sec from the manifest for the banner.
    for fr in detail["frames"]:
        row = manifest.get(fr["path"]) or {}
        try:
            fr["t_sec"] = float(row.get("t_sec", 0.0))
        except (TypeError, ValueError):
            fr["t_sec"] = None

    by_clip_frames: dict[str, list[dict]] = defaultdict(list)
    for fr in detail["frames"]:
        by_clip_frames[fr["clip_id"]].append(fr)

    written_videos: list[Path] = []
    all_jpg_paths: list[Path] = []
    for clip_id in sorted(by_clip_frames):
        clip_jpgs: list[Path] = []
        for fr in by_clip_frames[clip_id]:
            img = render_video_frame(fr, conf=conf, max_side=args.max_side)
            stem = Path(fr["path"]).stem
            out_jpg = frames_dir / f"{clip_id}_{stem}.jpg"
            cv2.imwrite(str(out_jpg), img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            clip_jpgs.append(out_jpg)
        mp4 = out_dir / f"eval_{clip_id}_{args.tag}.mp4"
        encode_mp4(clip_jpgs, mp4, playback_fps)
        written_videos.append(mp4)
        all_jpg_paths.extend(clip_jpgs)
        dur = len(clip_jpgs) / playback_fps
        print(f"  Wrote {mp4.relative_to(REPO_ROOT)}  ({len(clip_jpgs)} frames, {dur:.1f}s @ {playback_fps} fps)")

    if args.combined and all_jpg_paths:
        mp4 = out_dir / f"eval_all_{args.tag}.mp4"
        encode_mp4(all_jpg_paths, mp4, playback_fps)
        written_videos.append(mp4)
        print(
            f"  Wrote {mp4.relative_to(REPO_ROOT)}  "
            f"({len(all_jpg_paths)} frames, {len(all_jpg_paths) / playback_fps:.1f}s)"
        )

    manifest_path = out_dir / f"manifest_{args.tag}.json"
    payload = {
        "role": "eval_detector_video",
        "weights": str(weights.relative_to(REPO_ROOT)),
        "thresholds_path": str(thr_path.relative_to(REPO_ROOT)) if thr_path.exists() else None,
        "conf": conf,
        "nms_iou": nms_iou,
        "playback_fps": playback_fps,
        "max_side": args.max_side,
        "n_frames": len(all_jpg_paths),
        "clips": {
            clip_id: {
                "n_frames": len(frames),
                "video": str((out_dir / f"eval_{clip_id}_{args.tag}.mp4").relative_to(REPO_ROOT)),
            }
            for clip_id, frames in sorted(by_clip_frames.items())
        },
        "videos": [str(p.relative_to(REPO_ROOT)) for p in written_videos],
        "frames_dir": str(frames_dir.relative_to(REPO_ROOT)),
        "note": (
            "Match view: green=GT matched, yellow=GT miss, blue=pred TP, red=pred FP. "
            "Hold-out proxy GT; conf frozen on val."
        ),
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {manifest_path.relative_to(REPO_ROOT)}")
    print("OK: eval-only video; train/val unused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
