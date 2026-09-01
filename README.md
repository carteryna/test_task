# Aerial vehicle detection

End-to-end vehicle detection pipeline trained on pseudo-labeled aerial footage, evaluated across 0–200 m and 200–400 m distance bands.

## Overview & Strategy

I scoped this to one teacher, two nano students, one class (`vehicle`), and an 8-hour wall. The score is secondary; the labeling/eval path has to be repeatable and not leak.

Four Pexels clips (A–D) are the train pool. Clip E (`13722965`, portrait 2160×3840) is held out. The brief named eval as `32179597`; the file I have is `13722965`. I did not use E for training, thresholding, or model selection.

I sample train frames at 2 fps (147 train / 38 val) and eval at 5 fps (153 frames). Val is the last 20% of each train clip, time-ordered. Random 80/20 on frames would copy-paste near-duplicates across the split. Frames stay native resolution on disk; the student letterboxes to 1280 at train/infer.

I am not training on VisDrone, AU-AIR, or UAVDT. Pseudo-labels come from a general detector. I will train two nano students (YOLOv8n and YOLO11n) on the same boxes and score both on the four band metrics.

## Auto-Labeling & Data Pipeline

`src/extract_frames.py` reads `configs/data.yaml`, dumps JPEGs, writes `data/splits/{train,val,eval}.txt` and `manifest.csv`, and exits if any eval path lands in train or val.

| Clip | Scene | Train | Val | Eval |
|------|--------|------:|----:|-----:|
| A `8968356` | interchange, 1920×1080 | 31 | 8 | — |
| B `5382494` | rural highway, 4K | 50 | 13 | — |
| C `8457857` | top-down highway, 4K | 27 | 7 | — |
| D `3405804` | urban intersection, 4K | 39 | 10 | — |
| E `13722965` | city highway, portrait | — | — | 153 |

Teacher: YOLO-World `yolov8s-worldv2`, prompts `car, truck, bus, van, vehicle`, merged to class `0`. Clip C is nadir; a COCO-only YOLO often misses that view. Fallback if World fails: YOLO11s, keep `car/motorcycle/bus/truck`. Full-frame `imgsz=1280`, `conf=0.15`, CPU (MPS is present but `torchvision.nms` has no MPS kernel on this torch). I did not slice frames. On this machine a 4K image at 20% overlap is about six 640 tiles; 185 frames would have been hours. Low conf is cheaper: keep faint far boxes, delete extras in cleanup. I skipped ByteTrack — labels are 2 fps, and a highway car moves too far in 0.5 s for IoU tracks to mean much. Post-process: class-agnostic NMS, min side 8 px, max aspect 8. `src/auto_label.py` reads `train.txt`+`val.txt` only and refuses `--splits eval`.

Teacher run (train+val only; 153 eval frames skipped):

| Clip | Images | Boxes | Empty |
|------|-------:|------:|------:|
| A | 39 | 315 | 0 |
| B | 63 | 184 | 3 |
| C | 34 | 763 | 1 |
| D | 49 | 657 | 0 |
| **Total** | **185** | **1919** | **4** |

Clip C is dense and mostly vehicles. Clip B is sparse and has a few junk boxes on buildings/trees — cleanup targets. Previews: `results/auto_label/{A,B,C,D}.jpg`. Summary JSON: `data/splits/label_summary.json`.

To validate teacher model recall on small objects without guessing, I profiled all 1,919 generated bounding boxes by area tier (`src/profile_predictions.py`, native-pixel xyxy from `data/labels/raw/boxes.csv` — not a 1280×720 assumption). Tiny candidates (&lt; 600 px²) accounted for 16.0% of all labels with a median confidence of 0.239, confirming healthy far-band coverage at `conf=0.15` and eliminating the need for lower confidence re-runs. Full table: `data/splits/prediction_profile.json`.

| Tier | Area (px²) | Count | Share | Median conf |
|------|------------|------:|------:|------------:|
| Tiny | &lt; 600 | 307 | 16.0% | 0.239 |
| Small | 600–2000 | 426 | 22.2% | 0.334 |
| Medium/large | &gt; 2000 | 1186 | 61.8% | 0.289 |

Most tiny boxes sit in clip A. Remaining FPs go to cleanup, not another teacher pass.

Cleanup (`src/cleanup_labels.py`) on train/val only. Heuristic filter first, then a full OpenCV pass. Raw stays in `data/labels/raw`; edits live in `data/labels/clean`.

What I changed by hand:

- **Clip A (high / nadir-ish):** kept most YOLO-World boxes; added sparse far-band anchors; ignored sub-10 px blobs.
- **Clip B (occlusions):** dropped boxes with &gt;50% occlusion; for lighter occlusion, drew the full estimated footprint.
- **Clip C (articulated / clutter):** merged cab+trailer into one box; traffic cones overlapping a vehicle stayed inside the box.
- **Clip D — motorcycles out:** every two-wheeler deleted. Distance later uses the **short** box side as width (~1.8–2.5 m for cars/trucks). A 0.8 m bike would project into the wrong band under that prior.

Log (`data/splits/cleanup_log.json`): auto-deleted **9**, manually deleted **457**, manually added **754**. Clean set: **185** frames, **2207** boxes (**1919 → 2207**).

```bash
python src/dataset_statistics.py --config configs/data.yaml
```

| Stat | Value |
|------|------:|
| Frames | 185 |
| Boxes | 2207 |
| Boxes/frame (mean / min / max) | 11.93 / 2 / 34 |
| Mean box (px) | 96.4 × 77.7 |
| Aspect median (max/min) | 1.47 |

Aspect mix: ~29% squarish (&lt;1.2), ~50% car-like (1.2–2.0), ~18% van/SUV (2–3.5), ~3% truck-long (≥3.5). Per-clip and full JSON: `data/splits/dataset_statistics.json`. Eval GT cleanup is later, after the students freeze.

## Distance Estimation Model

Range is inferred from each GT box, not from telemetry. Pinhole, reference width `W_ref` subtending `s_px` on image height `H`:

```
range_m ≈ (W_ref * H) / (2 * s_px * tan(FOV_v / 2))
```

Assumptions I am using unless I change them and say so:

- `W_ref = 2.0 m` (highway vehicle width). `s_px = min(box_w, box_h)` — short side, so cars and long trucks share one prior. Motorcycles were removed in cleanup for this reason.
- Vertical FOV `70°`. The Pexels cameras are unknown.
- This is nadir-ish scale. Clips A/B/D are oblique; the number is a proxy, not a survey.

Bands: 0–200 m and 200–400 m. Boxes past 400 m or with a failed estimate stay in a third bucket in the log; I am not dropping them to dress the table. If eval has almost no GT in one band I will add a second held-out clip and record it here.

I will also publish a 3-row FOV sensitivity (40° / 70° / 90°) so band assignment is visible as an assumption, not a measurement.

## Evaluation & Metrics

IoU ≥ 0.5, greedy one-pred-per-GT. `conf` and NMS come from the train-only val split, then freeze. Eval is scored once.

False alarms / min is `FP × 60 / N_frames` as specified (per 60 frames, not fps-corrected). Time to first detection is seconds from eval start until the first TP in that band (`n/a` if none).

| Metric | YOLOv8n 0–200 m | YOLO11n 0–200 m | YOLOv8n 200–400 m | YOLO11n 200–400 m |
|--------|-----------------|-----------------|-------------------|-------------------|
| Detection rate TP / (TP + FN) | — | — | — | — |
| Precision TP / (TP + FP) | — | — | — | — |
| False alarms / min FP × 60 / N_frames | — | — | — | — |
| Time to first detection (s) | — | — | — | — |

mAP@0.5 across both bands if the eval script already has it. Overlay stills and a short eval demo go in `results/` after freeze. Weights will be a link, not a git blob.

## Quickstart

Python 3.10–3.12. Videos in `train_val_data/`.

```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python src/extract_frames.py --config configs/data.yaml
python src/auto_label.py --config configs/data.yaml
python src/profile_predictions.py
python src/cleanup_labels.py --config configs/data.yaml --auto-only
python src/cleanup_labels.py --config configs/data.yaml
python src/dataset_statistics.py --config configs/data.yaml
# later: python src/train.py && python src/eval.py
```

## Failure Modes & Trade-offs

2 fps on train undersamples fast motion and short occlusions; 5 fps eval is denser than the labels the student saw. I accepted that to stay inside the time box.

Distance from box size will mis-bin trucks, occluded cars, and oblique views. Far-band recall will likely be the weak number: 4K boxes become a few pixels after 1280 letterbox.

Train is landscape 1080p/4K; eval is portrait. That domain gap is real. Teacher boxes on B include a few non-vehicles at `conf=0.15`; that noise goes into the students unless cleanup removes it.

I am not claiming a Pi-ready detector. YOLO11n at 1280 is a training choice; onboard would be a smaller input, INT8, and a tracker, on a board I did not run here.
