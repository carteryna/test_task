#!/usr/bin/env python3
"""Heuristic filter + OpenCV cleanup on train/val labels.

Raw teacher labels stay in data/labels/raw (backed up once). Cleaned labels
are written to data/labels/clean. Eval paths are refused.

Interactive keys:
  left-click       delete box under cursor
  shift+drag       draw a new vehicle box
  u                undo last edit on this frame
  d / a            save and next / previous
  q                save and quit
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
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
        raise RuntimeError(f"Eval paths passed to cleanup: {leaked[:5]}")
    for p in image_paths:
        parts = Path(p).parts
        for clip_id in eval_clip_ids:
            if clip_id in parts:
                raise RuntimeError(f"Eval clip folder in cleanup path: {p}")


def label_for(image_rel: str, labels_root: Path) -> Path:
    rel = Path(image_rel)
    return labels_root / rel.parent.name / f"{rel.stem}.txt"


def load_boxes(txt_path: Path) -> list[list[float]]:
    if not txt_path.exists():
        return []
    boxes: list[list[float]] = []
    for line in txt_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = parts
        boxes.append([float(cls), float(x), float(y), float(w), float(h)])
    return boxes


def save_boxes(txt_path: Path, boxes: list[list[float]]) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in boxes
    ]
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def clone_boxes(boxes: list[list[float]]) -> list[list[float]]:
    return [b[:] for b in boxes]


def auto_filter(
    boxes: list[list[float]],
    min_dim: float,
    max_dim: float,
    max_aspect: float,
) -> list[list[float]]:
    valid: list[list[float]] = []
    for cls, x, y, w, h in boxes:
        aspect = max(w, h) / (min(w, h) + 1e-6)
        if min_dim < w < max_dim and min_dim < h < max_dim and aspect < max_aspect:
            valid.append([cls, x, y, w, h])
    return valid


def count_boxes(labels_root: Path, image_rels: list[str]) -> int:
    return sum(len(load_boxes(label_for(rel, labels_root))) for rel in image_rels)


def build_clean_from_raw(
    image_rels: list[str],
    raw_dir: Path,
    clean_dir: Path,
    min_dim: float,
    max_dim: float,
    max_aspect: float,
) -> int:
    """Copy raw -> clean with heuristic filter. Returns auto-deleted count."""
    deleted = 0
    for rel in image_rels:
        raw_path = label_for(rel, raw_dir)
        clean_path = label_for(rel, clean_dir)
        raw_boxes = load_boxes(raw_path)
        filtered = auto_filter(raw_boxes, min_dim, max_dim, max_aspect)
        deleted += len(raw_boxes) - len(filtered)
        save_boxes(clean_path, filtered)
    return deleted


def interactive_review(
    image_rels: list[str],
    clean_dir: Path,
    display_max_side: int,
) -> tuple[int, int, int]:
    """Returns (n_reviewed, n_deleted_manual, n_added_manual)."""
    deleted_manual = 0
    added_manual = 0
    reviewed = 0
    idx = 0
    win = "Cleanup | click=del  shift+drag=add  u=undo  d=next  a=prev  q=quit"

    while 0 <= idx < len(image_rels):
        rel = image_rels[idx]
        img_path = REPO_ROOT / rel
        txt_path = label_for(rel, clean_dir)
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read {img_path}")
        img_h, img_w = img.shape[:2]
        boxes = load_boxes(txt_path)
        history = [clone_boxes(boxes)]
        reviewed = max(reviewed, idx + 1)

        scale = 1.0
        if max(img_h, img_w) > display_max_side:
            scale = display_max_side / max(img_h, img_w)
        disp_w, disp_h = int(img_w * scale), int(img_h * scale)

        drawing = False
        ix = iy = -1
        drag = None  # (x1, y1, x2, y2) in display pixels while dragging

        def push_history() -> None:
            history.append(clone_boxes(boxes))

        def on_mouse(event, mx, my, flags, _param) -> None:
            nonlocal drawing, ix, iy, drag
            shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)

            if event == cv2.EVENT_LBUTTONDOWN and shift:
                drawing = True
                ix, iy = mx, my
                drag = (mx, my, mx, my)
                return

            if event == cv2.EVENT_MOUSEMOVE and drawing:
                drag = (ix, iy, mx, my)
                return

            if event == cv2.EVENT_LBUTTONUP and drawing:
                drawing = False
                x1, y1 = min(ix, mx), min(iy, my)
                x2, y2 = max(ix, mx), max(iy, my)
                drag = None
                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                    ox1, oy1 = x1 / scale, y1 / scale
                    ox2, oy2 = x2 / scale, y2 / scale
                    bw = (ox2 - ox1) / img_w
                    bh = (oy2 - oy1) / img_h
                    bx = (ox1 + ox2) / (2.0 * img_w)
                    by = (oy1 + oy2) / (2.0 * img_h)
                    push_history()
                    boxes.append([0.0, bx, by, bw, bh])
                return

            if event == cv2.EVENT_LBUTTONDOWN and not shift:
                ox, oy = mx / scale, my / scale
                for i, (_cls, x, y, w, h) in enumerate(boxes):
                    x1 = (x - w / 2.0) * img_w
                    y1 = (y - h / 2.0) * img_h
                    x2 = (x + w / 2.0) * img_w
                    y2 = (y + h / 2.0) * img_h
                    if x1 <= ox <= x2 and y1 <= oy <= y2:
                        push_history()
                        boxes.pop(i)
                        break

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, disp_w, disp_h)
        cv2.setMouseCallback(win, on_mouse)

        while True:
            display = cv2.resize(img, (disp_w, disp_h)) if scale != 1.0 else img.copy()
            for _cls, x, y, w, h in boxes:
                x1 = int((x - w / 2.0) * img_w * scale)
                y1 = int((y - h / 2.0) * img_h * scale)
                x2 = int((x + w / 2.0) * img_w * scale)
                y2 = int((y + h / 2.0) * img_h * scale)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if drag is not None:
                x1, y1, x2, y2 = drag
                cv2.rectangle(
                    display,
                    (min(x1, x2), min(y1, y2)),
                    (max(x1, x2), max(y1, y2)),
                    (0, 255, 0),
                    2,
                )
            d_del, d_add = _edits_from_history(history, boxes)
            cv2.putText(
                display,
                f"{idx + 1}/{len(image_rels)} {rel} boxes={len(boxes)} "
                f"del={d_del} add={d_add}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            cv2.imshow(win, display)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("u") and len(history) > 1:
                history.pop()
                boxes[:] = clone_boxes(history[-1])
            elif key == ord("d"):
                d_del, d_add = _edits_from_history(history, boxes)
                deleted_manual += d_del
                added_manual += d_add
                save_boxes(txt_path, boxes)
                idx += 1
                break
            elif key == ord("a"):
                d_del, d_add = _edits_from_history(history, boxes)
                deleted_manual += d_del
                added_manual += d_add
                save_boxes(txt_path, boxes)
                idx = max(0, idx - 1)
                break
            elif key == ord("q"):
                d_del, d_add = _edits_from_history(history, boxes)
                deleted_manual += d_del
                added_manual += d_add
                save_boxes(txt_path, boxes)
                cv2.destroyAllWindows()
                return reviewed, deleted_manual, added_manual

    cv2.destroyAllWindows()
    return reviewed, deleted_manual, added_manual


def _edits_from_history(
    history: list[list[list[float]]],
    current: list[list[float]],
) -> tuple[int, int]:
    """Approximate deletes/adds vs the frame's starting box set."""
    start = history[0]
    # Match by IoU-free count delta only when one-sided; otherwise count both.
    start_n = len(start)
    cur_n = len(current)
    if cur_n == start_n:
        # Possible replace (delete+add). Treat as 0/0 unless sets differ.
        if current == start:
            return 0, 0
        # Cheap fingerprint: sorted tuples
        if sorted(map(tuple, current)) == sorted(map(tuple, start)):
            return 0, 0
        # Unknown mix — report net as deletes+adds using set difference on tuples
        s = set(map(tuple, start))
        c = set(map(tuple, current))
        return len(s - c), len(c - s)
    if cur_n < start_n:
        s = set(map(tuple, start))
        c = set(map(tuple, current))
        return len(s - c), len(c - s)
    s = set(map(tuple, start))
    c = set(map(tuple, current))
    return len(s - c), len(c - s)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "data.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--auto-only",
        action="store_true",
        help="Apply heuristic filter into clean/ and exit (no GUI).",
    )
    parser.add_argument(
        "--force-auto",
        action="store_true",
        help="Rebuild clean/ from raw even if clean/ already exists.",
    )
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    cfg = load_config(cfg_path)

    if "eval" in args.splits:
        raise RuntimeError("Refusing to clean eval labels here. Eval GT is a later step.")

    splits_dir = REPO_ROOT / cfg["paths"]["splits_dir"]
    eval_paths = read_split_list(splits_dir / "eval.txt")
    eval_clip_ids = {c["id"] for c in cfg["clips"]["eval"]}

    image_rels: list[str] = []
    for split in args.splits:
        image_rels.extend(read_split_list(splits_dir / f"{split}.txt"))
    image_rels = list(dict.fromkeys(image_rels))
    assert_train_pool_only(image_rels, eval_paths, eval_clip_ids)

    cleanup = cfg["cleanup"]
    raw_dir = REPO_ROOT / cleanup["raw_dir"]
    clean_dir = REPO_ROOT / cleanup["clean_dir"]
    backup_dir = REPO_ROOT / cleanup["backup_dir"]
    min_dim = float(cleanup["min_dim"])
    max_dim = float(cleanup["max_dim"])
    max_aspect = float(cleanup["max_aspect"])
    display_max_side = int(cleanup["display_max_side"])

    if not backup_dir.exists():
        shutil.copytree(raw_dir, backup_dir)
        print(f"Backed up raw labels to {backup_dir.relative_to(REPO_ROOT)}")

    n_before = count_boxes(raw_dir, image_rels)
    deleted_auto = 0
    need_build = args.force_auto or not clean_dir.exists()
    if need_build:
        if clean_dir.exists() and args.force_auto:
            shutil.rmtree(clean_dir)
        deleted_auto = build_clean_from_raw(
            image_rels, raw_dir, clean_dir, min_dim, max_dim, max_aspect
        )
        print(f"Auto-filter: deleted {deleted_auto} boxes into {clean_dir.relative_to(REPO_ROOT)}")
    else:
        print(f"Using existing {clean_dir.relative_to(REPO_ROOT)} (pass --force-auto to rebuild)")

    reviewed = 0
    deleted_manual = 0
    added_manual = 0
    if not args.auto_only:
        reviewed, deleted_manual, added_manual = interactive_review(
            image_rels, clean_dir, display_max_side
        )

    n_after = count_boxes(clean_dir, image_rels)
    log = {
        "splits": args.splits,
        "n_images": len(image_rels),
        "n_reviewed": reviewed if not args.auto_only else 0,
        "n_deleted_auto": deleted_auto if need_build else None,
        "n_deleted_manual": deleted_manual,
        "n_added_manual": added_manual,
        "n_boxes_raw": n_before,
        "n_boxes_clean": n_after,
        "auto_only": args.auto_only,
        "heuristics": {
            "min_dim": min_dim,
            "max_dim": max_dim,
            "max_aspect": max_aspect,
        },
    }
    log_path = REPO_ROOT / cleanup["log_path"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() and not need_build:
        prev = json.loads(log_path.read_text())
        if prev.get("n_deleted_auto") is not None and log["n_deleted_auto"] is None:
            log["n_deleted_auto"] = prev["n_deleted_auto"]
        if args.auto_only:
            # Keep prior manual stats when re-running auto-only.
            log["n_deleted_manual"] = prev.get("n_deleted_manual", 0)
            log["n_added_manual"] = prev.get("n_added_manual", 0)
            log["n_reviewed"] = prev.get("n_reviewed", 0)
    log_path.write_text(json.dumps(log, indent=2) + "\n")

    print(
        f"Cleanup log: reviewed={log['n_reviewed']} auto-deleted={log['n_deleted_auto']} "
        f"manually-deleted={log['n_deleted_manual']} manually-added={log['n_added_manual']} "
        f"boxes {n_before} -> {n_after}"
    )
    print(f"Wrote {log_path.relative_to(REPO_ROOT)}")
    print("OK: eval not cleaned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
