# Aerial vehicle detection

End-to-end vehicle detection pipeline trained on pseudo-labeled aerial footage, evaluated across 0–200 m and 200–400 m distance bands.

## Overview & Strategy

I scoped this to one teacher, one student, one class (`vehicle`), and an 8-hour wall. The score is secondary; the labeling/eval path has to be repeatable and not leak.

Four Pexels clips (A–D) are the train pool. Clip E (`13722965`, portrait 2160×3840) is held out. The brief named eval as `32179597`; the file I have is `13722965`. I did not use E for training, thresholding, or model selection.

I sample train frames at 2 fps (147 train / 38 val) and eval at 5 fps (153 frames). Val is the last 20% of each train clip, time-ordered. Random 80/20 on frames would copy-paste near-duplicates across the split. Frames stay native resolution on disk; the student letterboxes to 1280 at train/infer.

I am not training on VisDrone, AU-AIR, or UAVDT. Pseudo-labels come from a general detector, then a YOLO11n/s student on those boxes only.

## Auto-Labeling & Data Pipeline

`src/extract_frames.py` reads `configs/data.yaml`, dumps JPEGs, writes `data/splits/{train,val,eval}.txt` and `manifest.csv`, and exits if any eval path lands in train or val.

| Clip | Scene | Train | Val | Eval |
|------|--------|------:|----:|-----:|
| A `8968356` | interchange, 1920×1080 | 31 | 8 | — |
| B `5382494` | rural highway, 4K | 50 | 13 | — |
| C `8457857` | top-down highway, 4K | 27 | 7 | — |
| D `3405804` | urban intersection, 4K | 39 | 10 | — |
| E `13722965` | city highway, portrait | — | — | 153 |

Teacher: YOLO-World with prompts `car, truck, bus, van, vehicle`, merged to one class. Clip C is nadir; a COCO-only YOLO often misses that view, which is why I did not start there. Fallback if World is painful on MPS: YOLO11l, keep `car/motorcycle/bus/truck`. Same post-process for train and eval: class-agnostic NMS, drop boxes with min side &lt; 8 px or broken aspect ratios, optional ByteTrack to kill one-frame flicker.

I have not run the teacher yet. Cleanup will be a short OpenCV pass: delete obvious false positives, leave misses unless they are cheap to add. I will log `n_reviewed / n_deleted / n_added`. Eval GT gets more of that pass than train, because the table is only as honest as those boxes. Eval auto-label happens after the student and `conf` are frozen.

## Distance Estimation Model

Range is inferred from each GT box, not from telemetry. Pinhole, object length `L_ref` subtending `s_px` on image height `H`:

```
range_m ≈ (L_ref * H) / (2 * s_px * tan(FOV_v / 2))
```

Assumptions I am using unless I change them and say so:

- `L_ref = 4.5 m` (passenger car). `s_px = max(box_w, box_h)`.
- Vertical FOV `70°`. The Pexels cameras are unknown.
- This is nadir-ish scale. Clips A/B/D are oblique; the number is a proxy, not a survey.

Bands: 0–200 m and 200–400 m. Boxes past 400 m or with a failed estimate stay in a third bucket in the log; I am not dropping them to dress the table. If eval has almost no GT in one band I will add a second held-out clip and record it here.

I will also publish a 3-row FOV sensitivity (40° / 70° / 90°) so band assignment is visible as an assumption, not a measurement.

## Evaluation & Metrics

IoU ≥ 0.5, greedy one-pred-per-GT. `conf` and NMS come from the train-only val split, then freeze. Eval is scored once.

False alarms / min is `FP × 60 / N_frames` as specified (per 60 frames, not fps-corrected). Time to first detection is seconds from eval start until the first TP in that band (`n/a` if none).

| Metric | 0–200 m | 200–400 m |
|--------|---------|-----------|
| Detection rate TP / (TP + FN) | — | — |
| Precision TP / (TP + FP) | — | — |
| False alarms / min FP × 60 / N_frames | — | — |
| Time to first detection (s) | — | — |

mAP@0.5 across both bands if the eval script already has it. Overlay stills and a short eval demo go in `results/` after freeze. Weights will be a link, not a git blob.

## Quickstart

Python 3.10–3.12. Videos in `train_val_data/`.

```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python src/extract_frames.py --config configs/data.yaml
# later: python src/auto_label.py && python src/train.py && python src/eval.py
```

## Failure Modes & Trade-offs

2 fps on train undersamples fast motion and short occlusions; 5 fps eval is denser than the labels the student saw. I accepted that to stay inside the time box.

Distance from box size will mis-bin trucks, occluded cars, and oblique views. Far-band recall will likely be the weak number: 4K boxes become a few pixels after 1280 letterbox.

Train is landscape 1080p/4K; eval is portrait. That domain gap is real. Top-down clip C may still be sparsely labeled if the teacher fails, and the student will inherit those holes.

I am not claiming a Pi-ready detector. YOLO11n at 1280 is a training choice; onboard would be a smaller input, INT8, and a tracker, on a board I did not run here.
