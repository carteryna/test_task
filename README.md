# Aerial vehicle detection

End-to-end vehicle detection pipeline trained on pseudo-labeled aerial footage, evaluated across 0–200 m and 200–400 m distance bands.

## Overview & Strategy

I scoped this to one teacher, two nano students, one class (`vehicle`), and an 8-hour wall. The score is secondary; the labeling/eval path has to be repeatable and not leak.

Four Pexels clips (A–D) are the train pool. Hold-out eval is Clip E (`13722965`, portrait 2160×3840) plus Clip F (`14693572`, landscape 4K) added for far-band coverage after E audited at only one 200–400 m box. The brief named eval as `32179597`; the file I have for E is `13722965`. I did not use E or F for training, thresholding, or model selection.

I sample train frames at 2 fps (147 train / 38 val) and eval at 5 fps (**153 E + 61 F = 214** frames). Val is the last 20% of each train clip, time-ordered. Random 80/20 on frames would copy-paste near-duplicates across the split. Frames stay native resolution on disk; the student letterboxes to 1280 at train/infer.

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
| F `14693572` | high aerial highway, 4K | — | — | 61 |

Teacher: YOLO-World `yolov8s-worldv2`, prompts `car, truck, bus, van, automobile`, merged to class `0`. Motorcycle / bike prompts omitted. Fallback COCO ids are `[2, 5, 7]` (car/bus/truck) — **not** id 3. That matches cleanup and the `distance:` block in `configs/data.yaml` (`w_ref_m: 2.0`, `size_side: min`). Clip C is nadir; a COCO-only YOLO often misses that view. Full-frame `imgsz=1280`, `conf=0.15`, CPU (MPS is present but `torchvision.nms` has no MPS kernel on this torch). I did not slice frames. On this machine a 4K image at 20% overlap is about six 640 tiles; 185 frames would have been hours. Low conf is cheaper: keep faint far boxes, delete extras in cleanup. I skipped ByteTrack — labels are 2 fps, and a highway car moves too far in 0.5 s for IoU tracks to mean much. Post-process: class-agnostic NMS, min side 8 px, max aspect 8. `src/auto_label.py` reads `train.txt`+`val.txt` by default and refuses `--splits eval`. Hold-out proxy GT is a later, explicit pass: `--allow-eval` writes `data/labels/eval/` only after the student is frozen. Same teacher, same motorcycle filter (COCO id 3 never in `coco_vehicle_ids`).

Teacher run (train+val only; 153 eval frames skipped):

| Clip | Images | Boxes | Empty |
|------|-------:|------:|------:|
| A | 39 | 315 | 0 |
| B | 63 | 184 | 3 |
| C | 34 | 763 | 1 |
| D | 49 | 657 | 0 |
| **Total** | **185** | **1919** | **4** |

Clip C is dense and mostly vehicles. Clip B is sparse and has a few junk boxes on buildings/trees — cleanup targets. Previews: `results/auto_label/{A,B,C,D}.jpg`. Summary JSON: `data/splits/label_summary.json`.

Eval proxy GT after freeze (`--allow-eval`, same teacher, `data/labels/eval/`):

| Clip | Frames | Boxes | Empty | Notes |
|------|------:|------:|------:|-------|
| E | 153 | 227 | 39 | portrait; many misses → QA |
| F | 61 | 211 | 0 | landscape 4K; far cars still under-called |
| **Teacher total** | **214** | **438** | **39** | E folder left at cleaned GT after F-only re-run |

F was labeled with `--clips F` so cleaned E labels were not overwritten. Teacher dump for F also sits under `data/labels/eval_raw/F/`. Previews: `results/auto_label_eval/{E,F}.jpg`. Summary: `data/splits/eval_label_summary.json`.

Eval cleanup (same OpenCV tool): teacher dump frozen under `data/labels/eval_raw/`; edits write back to `data/labels/eval/`. Heuristic auto-filter deleted **0** on both clips. Manual rules: merge cab+trailer, drop infrastructure FPs, add misses; pedestrians/motorcycles stay unlabeled. F was QA’d with `--clips F` so cleaned E was not reopened.

| Clip | Reviewed | Auto-del | Manual-del | Manual-add | Clean boxes |
|------|--------:|---------:|-----------:|-----------:|------------:|
| E | 153 | 0 | 12 | 164 | **379** (227 → 379) |
| F | 61 | 0 | 54 | 447 | **604** (211 → 604) |
| **Total** | **214** | **0** | **66** | **611** | **983** |

Log: `data/splits/eval_cleanup_log.json`.

```bash
python src/cleanup_labels.py --allow-eval --splits eval --auto-only
python src/cleanup_labels.py --allow-eval --splits eval
# Clip F only (keeps cleaned E): --clips F --force-auto --auto-only, then --clips F
python src/dataset_statistics.py --allow-eval
```

| Stat (eval cleaned, E+F) | Value |
|------|------:|
| Frames | 214 |
| Boxes | 983 |
| Boxes/frame (mean / min / max) | 4.59 / 1 / 13 |
| Mean box (px) | 107.0 × 177.3 |
| Aspect median (max/min) | 1.34 |

Per clip: E 2.48 boxes/frame; F 9.90. Aspect mix: ~31% squarish (&lt;1.2), ~39% car-like (1.2–2.0). JSON: `data/splits/eval_dataset_statistics.json`.

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

Aspect mix: ~29% squarish (&lt;1.2), ~50% car-like (1.2–2.0), ~18% van/SUV (2–3.5), ~3% truck-long (≥3.5). Per-clip and full JSON: `data/splits/dataset_statistics.json`.

## Distance Estimation Model

Range is inferred from each GT box, not from telemetry (`src/estimate_distance.py`). Short side is width; focal length in pixels from vertical FOV:

```
f_px = H / (2 * tan(FOV_v / 2))
distance_m = f_px * W_ref / s_px
s_px = min(box_w, box_h)   # pixels
```

(Same model as `(W_ref * H) / (2 * s_px * tan(FOV_v/2))`.)

Assumptions in `configs/data.yaml` → `distance:`:

- `W_ref = 2.0 m`. Motorcycles removed so that prior holds.
- `FOV_v = 40°` — synthetic prior, not EXIF (see below).
- Nadir-ish scale; oblique clips are approximate.

### Synthetic FOV prior (40°)

A first pass at 70° vertical FOV mapped **100%** of the 2207 train/val boxes into 0–200 m (max 199.5 m). That is far-band collapse: a student trained on those tags has no 200–400 m GT and will treat range as a single bin.

I locked `fov_v_deg: 40.0` in `configs/data.yaml`. This is a prior, not a camera calibration. Highway drone monitoring often uses a narrower vertical FOV or a telephoto so the aircraft can stay high; 40° is that assumption written into config so a re-run cannot silently revert to 70°.

Same boxes, same `W_ref`: **1756** near / **451** far (**20.4%**). Those 451 targets are the far-band supervised signal. Without them there is nothing for the student to correlate with small pixel scale, and the eval table’s 200–400 m columns would be scoring a model that never saw that band in training.

Clean train/val (2207 boxes) at FOV **40°**. Distance min / median / mean / max: **7.0 / 112.0 / 135.7 / 383.7 m**. Short-side `s_px`: **8.0 / 52.9 / 65.2 / 846.0**.

| Band | Count | Share | Dist median | `s_px` median (min) |
|------|------:|------:|------------:|--------------------:|
| 0–200 m | 1756 | 79.6% | 93.7 m | 63.4 (14.9) |
| 200–400 m | 451 | 20.4% | 276.4 m | 10.9 (8.0) |
| >400 m / failed | 0 | 0% | — | — |

Far boxes are 10.9 px on the short side at native resolution. After 1280 letterbox on 4K that is a few pixels; that is the far-band recall problem, not a labeling bug.

Per clip (far-band signal is not uniform — **335 / 451** far boxes are clip A):

| Clip | Near | Far | Far share | `f_px` (H, 40°) |
|------|-----:|----:|----------:|----------------:|
| A 1080p | 163 | 335 | 67.3% | 1484 |
| B 4K | 200 | 45 | 18.4% | 2967 |
| C 4K | 669 | 69 | 9.3% | 2967 |
| D 4K | 724 | 2 | 0.3% | 2967 |

Train vs val (time-split, not band-stratified): train **1368 / 372** near/far (1740), val **388 / 79** (467). Far share 21.4% train / 16.9% val.

FOV sensitivity (same boxes, alternate FOV):

| FOV_v | 0–200 m | 200–400 m |
|------:|--------:|----------:|
| 40° (locked) | 1756 | 451 |
| 70° | 2207 | 0 |
| 90° | 2207 | 0 |

CSV + JSON: `data/splits/distance_boxes.csv`, `data/splits/distance_bands.json`.

### Eval distance audit (Clips E+F)

Same priors on cleaned eval GT (`--allow-eval`). E portrait H=3840 → `f_px` ≈ **5275**; F landscape H=2160 → `f_px` ≈ **2967** at 40°. Audit hard-fails if the far band is empty.

| Band | Count | Share | Dist median | `s_px` median |
|------|------:|------:|------------:|--------------:|
| 0–200 m | 904 | 92.0% | 94.2 m | 63.0 |
| 200–400 m | 79 | 8.0% | 247.3 m | 24.0 |
| >400 m / failed | 0 | 0% | — | — |

| Clip | Near | Far | Far % |
|------|-----:|----:|------:|
| E | 378 | 1 | 0.3% |
| F | 526 | 78 | 12.9% |

Audit **passed**. Clip F supplies essentially all far-band hold-out GT (**78 / 79**); E remains a near-band portrait stress test. JSON: `data/splits/eval_distance_bands.json`.

```bash
python src/estimate_distance.py --allow-eval
```

## Evaluation & Metrics

`src/evaluate_custom.py`: predict once at `conf≥0.01`, sweep `conf` on **val only** (mean of near/far F1), freeze thresholds, then score hold-out **once**. IoU ≥ 0.5, greedy one-pred-per-GT. TP/FN by GT distance band; unmatched preds are FP in the pred box’s band (same `W_ref` / FOV prior). False alarms / min is `FP × 60 / N_frames` (per 60 frames, not fps-corrected). Time to first detection is the earliest in-clip `t_sec` with a TP in that band (`n/a` if none); with E+F the headline number is the min across clips.

Val sweep chose **`conf=0.20`**, `nms_iou=0.5` (val mean-band F1 **0.649**). Eval unused for selection. YOLOv8n not trained on this CPU — its columns stay empty.

Hold-out (YOLO11n `best.pt`, 214 frames, conf=0.20):

| Metric | YOLOv8n 0–200 m | YOLO11n 0–200 m | YOLOv8n 200–400 m | YOLO11n 200–400 m |
|--------|-----------------|----------------:|-------------------|------------------:|
| Detection rate TP / (TP + FN) | — | 0.602 | — | 0.063 |
| Precision TP / (TP + FP) | — | 0.605 | — | 0.172 |
| False alarms / min FP × 60 / N_frames | — | 99.5 | — | 6.7 |
| Time to first detection (s) | — | 0.0 | — | 0.4 |

| Clip | Near Det / Prec | Far Det / Prec | Notes |
|------|----------------:|---------------:|-------|
| E | 0.997 / 0.818 | 1.0 / 0.333 (n_gt=1) | portrait; near almost saturated |
| F | 0.318 / 0.381 | 0.051 / 0.154 | high aerial; far and small cars dominate misses/FPs |

Far-band hold-out recall is the weak number (**5 / 79** TP). That matches the train story: 10 px boxes after 1280 letterbox, plus F’s altitude/domain shift vs A–D. Near-band is carried by E; F pulls overall near Det down and FA/min up. JSON: `data/splits/eval_thresholds.json`, `data/splits/eval_metrics.json`.

### Hold-out A/B: clean labels vs DINO+SAM student

Same protocol, separate outputs so the baseline JSON above is untouched. Val still uses **cleaned** labels for conf selection (identical selection target); hold-out still uses the frozen proxy GT. Factory student: `runs/train/yolo11n_dinosam/weights/best.pt`.

Val sweep for the factory student chose **`conf=0.50`** (val mean-band F1 **0.557**) — higher than the clean student’s 0.20, because factory training produces denser high-confidence clutter against the cleaner val GT. Hold-out once at that frozen conf:

| Metric | Clean 0–200 m | DINO+SAM 0–200 m | Clean 200–400 m | DINO+SAM 200–400 m |
|--------|--------------:|-----------------:|----------------:|-------------------:|
| Detection rate | 0.602 | **0.619** | 0.063 | **0.114** |
| Precision | **0.605** | 0.587 | 0.172 | **0.281** |
| F1 | 0.603 | 0.603 | 0.093 | **0.162** |
| False alarms / min | **99.5** | 110.5 | 6.7 | **6.4** |
| Time to first (s) | 0.0 | 0.0 | 0.4 | **0.2** |

| Clip | Clean Near Det / Prec | DINO+SAM Near Det / Prec | Clean Far Det | DINO+SAM Far Det |
|------|----------------------:|-------------------------:|--------------:|-----------------:|
| E | **0.997 / 0.818** | 0.897 / 0.610 | 1.0 (n=1) | 1.0 (n=1) |
| F | 0.318 / 0.381 | **0.420 / 0.555** | 0.051 | **0.103** |

Mean-band selection score: **0.348 → 0.383**. The factory student buys far-band recall (**5 → 9** TP of 79) and F near-band Det, at the cost of Clip E’s near-perfect Det under the higher conf and a small near-band precision drop. Far band is still weak in absolute terms — 11% Det is not solved — but the A/B moves the right way on the metric that was the bottleneck. Artifacts: `data/splits/eval_thresholds_dinosam.json`, `data/splits/eval_metrics_dinosam.json`, `data/splits/eval_ab_clean_vs_dinosam.json`, overlays in `results/eval_overlays_dinosam/`.

```bash
python src/evaluate_custom.py \
  --weights runs/train/yolo11n_dinosam/weights/best.pt \
  --thresholds-path data/splits/eval_thresholds_dinosam.json \
  --metrics-path data/splits/eval_metrics_dinosam.json \
  --overlay-dir results/eval_overlays_dinosam \
  --examples-dir outputs/examples_dinosam \
  --tune-val --score-eval
python src/evaluate_custom.py --weights runs/train/yolo11n_dinosam/weights/best.pt \
  --thresholds-path data/splits/eval_thresholds_dinosam.json \
  --metrics-path data/splits/eval_metrics_dinosam.json \
  --examples-dir outputs/examples_dinosam --export-examples
```

### Example predictions (eval)

Curated stills for the **clean-label** student live under **`outputs/examples/`** (3 frames × clips E and F). The factory-student set is the same layout under **`outputs/examples_dinosam/`**. Each frame has:

- `*_side_by_side.jpg` — left **GT** (green), right **student preds** (blue + conf)
- `*_combined.jpg` — match view: GT matched green, GT miss yellow, pred TP blue, pred FP red

Tags: `success` (clean TPs), `far` (best far-band coverage on that clip), `hard` (error-heavy). Index: `manifest.json` in each folder. Debug dumps (error-sorted) stay in `results/eval_overlays/` / `results/eval_overlays_dinosam/`.

### Detector video (eval)

`src/export_eval_video.py` draws the match view on every hold-out sample and encodes H.264 via ffmpeg at the eval sample rate (5 fps). Factory student, conf frozen at **0.50**:

| Clip | File | Duration | Size |
|------|------|----------:|-----:|
| E | [`outputs/videos/eval_E_dinosam.mp4`](outputs/videos/eval_E_dinosam.mp4) | ~30.6 s | 16 MB |
| F | [`outputs/videos/eval_F_dinosam.mp4`](outputs/videos/eval_F_dinosam.mp4) | ~12.2 s | 11 MB |

Legend on every frame: green = GT matched, yellow = GT miss, blue = pred TP, red = pred FP. Banner shows `t_sec`, conf, TP/FP/FN, far-GT count. Manifest: `outputs/videos/manifest_dinosam.json`. Intermediate JPEGs under `outputs/videos/frames_*` are gitignored and regenerable.

```bash
python src/evaluate_custom.py --tune-val --score-eval
python src/evaluate_custom.py --export-examples   # re-score + rewrite outputs/examples/
# Factory-student stills + video (baseline examples/ untouched):
python src/evaluate_custom.py --weights runs/train/yolo11n_dinosam/weights/best.pt \
  --thresholds-path data/splits/eval_thresholds_dinosam.json \
  --metrics-path data/splits/eval_metrics_dinosam.json \
  --examples-dir outputs/examples_dinosam --export-examples
python src/export_eval_video.py \
  --weights runs/train/yolo11n_dinosam/weights/best.pt \
  --thresholds-path data/splits/eval_thresholds_dinosam.json \
  --tag dinosam
# Short cut of F only: add --clips F
# Clean-label baseline video: --weights runs/train/yolo11n/weights/best.pt --tag clean \
#   --thresholds-path data/splits/eval_thresholds.json
```

## Error taxonomy & diagnostics

`src/error_analysis.py` re-reads the frozen hold-out predictions (`conf=0.20` from `eval_thresholds.json`) and buckets every FP/FN into named failure modes, so the next iteration is data work instead of guesswork. Nothing here re-tunes the model.

Tracking is a two-stage IoU tracker I wrote in-repo (ByteTrack's high-then-low association, `max_age=3` sampled frames). Full ByteTrack/DeepSORT would add an assignment-solver dependency and re-ID weights for 5 fps frames that a greedy IoU pass already links; the point is persistent IDs for Rule 4, not MOTA. Over 214 frames it produced **361** tracks.

| FP rule | Count | % of 379 FP | Reading |
|---------|------:|------:|---------|
| `gt_omission` (conf > 0.50, no GT) | 93 | 24.5% | proxy-GT debt, not model error — 71 of them on F |
| `suspected_motorcycle` (aspect > 3.0) | 4 | 1.1% | all on E; long thin boxes against the 2.0 m width prior |
| `fractured_truck` (large FN split by ≥2 contained FPs) | 2 | 0.5% | one articulated vehicle on F |
| `static_hallucination` (centroid < 5 px for > 30 frames) | 0 | 0.0% | see below |
| unclassified | 280 | 73.9% | 257 near / 23 far, conf median **0.30**, p90 **0.45** |

FN side: **1** of 434 misses is a `fractured_truck`; the rest stay unclassified, which is consistent with the band table — most misses are small far vehicles the head never fired on, not mislocalised ones.

Rule 4 never fires strictly: both hold-out clips are moving drone shots, so a static object still translates across the frame. The longest strict run was **5** frames (track F/45). Rather than ship an empty mining list, a relaxed pass (drift < 1% of image width for ≥ 8 frames, tagged `relaxed` in the manifest) picked up **2** tracks → **6** crops in `data/hard_negatives/`. Strict Rule 4 becomes meaningful on a hovering or gimbal-locked clip; on translating footage it needs ego-motion compensation (homography per frame pair), which I did not build here.

The honest headline is the residual bucket: three quarters of FPs are mid-confidence boxes on vehicle-like clutter, mostly near-band on F. That points at more F-like training data, not at another rule.

Outputs: `data/splits/error_taxonomy.json` (counts, per-clip splits, per-instance rows, top static tracks), `data/hard_negatives/` (crops + `manifest.json`), `outputs/diagnostics/error_taxonomy_grid.jpg` plus one annotated still per rule.

```bash
python src/error_analysis.py                 # all eval clips
python src/error_analysis.py --clips E       # Clip E only
python src/error_analysis.py --refresh       # ignore the prediction cache in runs/cache
```

## Automated data factory (Grounding DINO → SAM)

The taxonomy above says the label pipeline, not the head, is the bottleneck. `src/auto_label_dino_sam.py` replaces the teacher + human cleanup loop with an offline foundation-model factory: **Grounding DINO** (`"vehicle."`) → **SAM** masks → tight boxes → schema and kinematic rules → YOLO labels in `data/labels/auto_generated/`. Nothing here ships to the Pi; it is a data engine that runs once, offline.

Both models are CPU-heavy (**84 s/frame**, 3.6 h for the 185 train/val frames), so stage 1 memoises per-frame boxes in `runs/cache/dino_sam` (gitignored). Stage 2 re-runs every rule over the cache in ~4 s, which is how the thresholds below were tuned.

**Tiling is required, not an optimisation.** Grounding DINO letterboxes to ~800 px, so a 10 px vehicle in a 4K frame lands at ~3 px. Each frame is prompted as 2×2 overlapping tiles *plus* the full frame (the full pass recovers objects straddling seams), then class-agnostic NMS at 0.55 merges the five box sets.

**SAM is the geometry fix.** 92.6% of final boxes come from a mask; the rest keep the DINO box when the mask fails an area or IoU guard (SAM bleeding into road or shadow). On boxes matched to the human-cleaned labels, median IoU is **0.799** and the median area ratio is **0.955** — DINO+SAM boxes are ~4.5% tighter on the same vehicle, which is what the 2.0 m width prior in the distance model wants.

Three rules that were tuned, not assumed:

| Decision | Evidence |
|---|---|
| `box_threshold` 0.25 → **0.15** | probe on A/C/D: recall vs cleaned labels 44–71% → 89–100% |
| prompt stays `"vehicle."` | `car. truck. bus. van. vehicle.` gave 3× the boxes for **no** extra recall |
| merging constrained to truck geometry | unconstrained overlap-merge fired 55× on 7 frames and **cost 8 pts of recall**; crops showed it was consolidating FP clusters on gantries and pylons, not trucks |

Merging now needs the union elongated (aspect 1.8–6.0), one part ≥ 24 px, and union ≤ 1.6× the larger part — a fragment sits inside its parent, two neighbouring cars would balloon the union. That halved the merges and kept the real articulated trucks and buses.

Full run: **7016 raw → 5886 final boxes** over 185 frames, **796** truck merges.

| Drop rule | Boxes |
|---|---:|
| `suspected_motorcycle` (aspect ≥ 3.0 **and** short side ≤ 28 px) | 192 |
| `static_track` (centroid drift < 0.5% of width over ≥ 12 frames) | 74 |
| `max_dim` / `min_side` / `max_aspect` | 32 / 24 / 12 |

Rule 4 finally fires here: 74 boxes on 5 static tracks, all in Clip A (the one near-hovering shot). B/C/D are translating drone footage where a static object still moves across the frame — the same ego-motion limit the hold-out taxonomy hit.

Agreement with the human-cleaned teacher labels (IoU 0.4). This measures **divergence, not accuracy** — the reference is itself pseudo-GT, so "extra" mixes new FPs with vehicles the teacher missed:

| Clip | Reference | Auto | Recall vs ref | Extra |
|---|---:|---:|---:|---:|
| A | 498 | 1471 | 0.797 | 1074 |
| B | 245 | 921 | 0.861 | 710 |
| C | 738 | 822 | **0.660** | 335 |
| D | 726 | 2672 | 0.864 | 2045 |
| **All** | 2207 | 5886 | **0.780** | 22.5/frame |

Clip C is the honest failure: it is the only clip where the factory produces *fewer* boxes than the teacher, and the previews show why — isolated dark cars on open asphalt at mid range are missed by `grounding-dino-tiny`. The factory buys geometry and rule-enforced cleanliness; on this checkpoint it gives back recall on low-contrast mid-range vehicles. `grounding-dino-base` or a 3×3 tiling would likely close it at 2–4× the runtime, which I did not spend.

The headline hold-out band table still reports the clean-label student (`runs/train/yolo11n`). The factory student (`yolo11n_dinosam`) has its own hold-out A/B in [Hold-out A/B](#hold-out-ab-clean-labels-vs-dinosam-student): far Det 0.063 → 0.114, mean-band score 0.348 → 0.383.

```bash
python src/auto_label_dino_sam.py                          # all train/val (A–D), ~3.6 h CPU
python src/auto_label_dino_sam.py --clips C --limit 4      # smoke; eval clips are refused
python src/auto_label_dino_sam.py --stage rules            # re-apply rules from cache (~4 s)
python src/auto_label_dino_sam.py --refresh                # ignore the inference cache
python src/train.py --labels-dir data/labels/auto_generated --run-suffix _dinosam
```

### Hold-out audit: is `gt_omission` really labeling debt?

The taxonomy claimed 93 FPs (24.5%) were `gt_omission` — confident student boxes with no proxy GT — on the argument that they are labeling debt, not model error. That is a hypothesis one labeler cannot test on itself. Running the factory over the frozen hold-out gives an **independent second opinion**: if a student FP overlaps a DINO+SAM box, two unrelated models agree a vehicle is there and the proxy GT is what's missing.

This is leakage-sensitive, so it sits behind `--allow-eval --splits eval` (the same gate `auto_label.py` uses) and writes to a **separate** tree — `data/labels/eval_dino_sam/`, not `data/labels/eval/`. The proxy GT that scored the student is untouched, every metric above still stands, and the summary is tagged `role: eval_proxy_audit`, `used_for_training: false`. `train.py` builds only from `train.txt`/`val.txt` and rejects eval paths, so these labels cannot reach training. Four guards are enforced: eval without the flag, an eval clip id without the flag, `--allow-eval` with a non-eval split, and `--allow-eval` with a train clip.

Sampled student FPs from `error_taxonomy.json`, scored against the DINO+SAM labels at IoU 0.4:

| Student FP bucket | Confirmed as vehicle | Reading |
|---|---:|---|
| `gt_omission` | **36 / 60 (60%)** | the debt hypothesis holds for most, but not all, of the bucket |
| `unclassified` | **23 / 60 (38%)** | a third of the biggest bucket is also proxy-GT debt |
| `fractured_truck` | 2 / 2 | both confirmed |
| `suspected_motorcycle` | 2 / 4 | ambiguous — motorcycles are out of the ontology by design |

Extrapolating those rates across all 379 hold-out FPs puts **~167 (44%)** on a box a second model calls a vehicle. As an upper bound, overall precision would read **0.59 → 0.77** if every one were true. Treat that as a ceiling, not a correction: the factory has its own false positives (8.1 extra boxes/frame here), so agreement between two models is evidence, not ground truth. The honest conclusion is narrower and still useful — a large share of what the metrics call false alarms is the proxy GT's debt, so the reported precision understates the student, and the far-band recall gap (0.063) remains the real problem.

Run: 214 frames in ~4.5 h, **3519 raw → 2542 final** boxes, 89.3% from a mask, median matched IoU **0.841**, median area ratio **0.977**.

| Clip | Proxy GT | Auto | Recall vs proxy | Extra |
|---|---:|---:|---:|---:|
| E (portrait, near) | 379 | 791 | **0.987** | 417 |
| F (high aerial, far) | 604 | 1751 | 0.733 | 1308 |

E is near-total agreement: the factory independently reproduces 374 of 379 manually QA'd boxes, which is the strongest evidence in this repo that the pipeline is sound. F is the harder clip and matches the train-pool pattern — high-altitude vehicles are where `grounding-dino-tiny` and the student both struggle.

Two rule notes. `max_dim` dropped **499** boxes, nearly all on E, where DINO repeatedly proposes the whole roadway as one object in the portrait framing — a pathology the size guard catches cleanly. And Rule 4 fired on F (2 static tracks, 29 boxes) where it never fired strictly during the student diagnostics, because 5 fps eval sampling packs 12 consecutive frames into 2.4 s of drone drift.

```bash
python src/auto_label_dino_sam.py --allow-eval                    # E+F audit, ~4.5 h CPU
python src/auto_label_dino_sam.py --allow-eval --clips F          # one hold-out clip
python src/auto_label_dino_sam.py --allow-eval --stage rules      # re-audit from cache (~5 s)
```

## Student training

`configs/train.yaml` + `src/train.py`: build `data/yolo/` from train/val splits and `data/labels/clean` (symlinks; eval refused), then fine-tune from COCO with **`freeze=10`**, `imgsz=1280`, `batch=4`. Dataset scan: **147 train / 38 val** images, **1740 / 467** boxes, **0** empty frames. YOLOv8n is wired but not trained on this CPU.

Three epochs proved the loader, 1280 letterbox, and freeze path. They are not enough for the head to lock onto the **451** far-band boxes (median short side 10.9 px). Overall val recall after that smoke was **0.276** — Ultralytics all-band, not a far-band score (band metrics are in `evaluate_custom.py`). Ten epochs is the CPU budget that still leaves a visible gradient trajectory without the 50-epoch GPU recipe. `runs/` is gitignored; numbers live in `data/splits/train_smoke.json`.

Smoke (3 epochs, ~13.4 min). `runs/train/yolo11n_e3/weights/best.pt`.

| Epoch | box | cls | P | R | mAP50 | mAP50-95 |
|------:|----:|----:|--:|--:|------:|---------:|
| 1 | 1.54 | 3.02 | 0.002 | 0.045 | 0.016 | 0.007 |
| 2 | 1.56 | 2.18 | 0.052 | 0.013 | 0.002 | 0.000 |
| 3 (best) | 1.54 | 1.92 | 0.775 | 0.276 | 0.449 | 0.217 |

CPU validation run (YOLO11n × 10 epochs, freeze=10, ~40 min / 0.666 h). Mosaic closed for the whole run (`close_mosaic=10`). Val-only; eval clip unused. `runs/train/yolo11n/weights/best.pt` is Ultralytics fitness (epoch 9), not argmax mAP50.

| Epoch | P | R | mAP50 | mAP50-95 |
|------:|--:|--:|------:|---------:|
| 3 | 0.943 | 0.106 | 0.381 | 0.181 |
| 4 | 0.663 | 0.572 | 0.629 | 0.278 |
| 6 (peak mAP50) | 0.693 | 0.625 | 0.675 | 0.314 |
| 9 (`best.pt`) | 0.646 | 0.660 | 0.657 | 0.322 |
| 10 | 0.635 | 0.651 | 0.651 | 0.322 |

Fused `best.pt` val (38 images, 467 instances): **P 0.646 · R 0.660 · mAP50 0.657**. Recall 0.276 → 0.660 vs the 3-epoch smoke. That is still all-band; far-band 10 px boxes remain the likely miss. Peak mAP50 was 0.675 at epoch 6; fitness kept epoch 9 for the extra mAP50-95. CPU infer ~288 ms/image at 1280.

### Student on DINO+SAM labels (A/B)

Same recipe (`freeze=10`, `imgsz=1280`, CPU), but labels from `data/labels/auto_generated` via `--labels-dir` / `--run-suffix _dinosam`. Eval still unused. The val set is denser — **1190** instances vs **467** on the clean teacher labels — so Ultralytics mAP is **not** an apples-to-apples A/B; a fair comparison needs the frozen hold-out band metrics (`evaluate_custom.py`).

Smoke (3 epochs, ~11 min / 0.180 h). `runs/train/yolo11n_dinosam_e3/weights/best.pt`.

| Epoch | P | R | mAP50 | mAP50-95 |
|------:|--:|--:|------:|---------:|
| 1 | 0.033 | 0.057 | 0.020 | 0.010 |
| 2 | 0.127 | 0.197 | 0.051 | 0.026 |
| 3 (best) | 0.521 | 0.402 | 0.386 | 0.206 |

10 epochs (~35 min / 0.580 h). `runs/train/yolo11n_dinosam/weights/best.pt` is Ultralytics fitness (epoch 10).

| Epoch | P | R | mAP50 | mAP50-95 |
|------:|--:|--:|------:|---------:|
| 4 | 0.560 | 0.500 | 0.499 | 0.272 |
| 6 | 0.624 | 0.516 | 0.546 | 0.307 |
| 8 | 0.630 | 0.550 | 0.570 | 0.330 |
| 10 (`best.pt`) | 0.643 | 0.564 | **0.579** | **0.341** |

Fused `best.pt` val (38 images, 1190 instances): **P 0.644 · R 0.565 · mAP50 0.579**. Steady climb, no collapse — the denser factory labels are trainable. Relative to the clean-label student (mAP50 0.657 on 467 instances) the number is lower, as expected when the same detector is scored against ~2.5× more boxes that include factory extras. The fair A/B is the hold-out band table above (far Det ↑, near F1 flat). Snapshots live in `data/splits/train_smoke.json`.

```bash
python src/estimate_distance.py --config configs/data.yaml
python src/train.py --config configs/train.yaml --prepare-only
python src/train.py --config configs/train.yaml --models yolo11n --epochs 3    # loader / freeze smoke
python src/train.py --config configs/train.yaml --models yolo11n --epochs 10   # CPU val trajectory
# DINO+SAM label A/B (writes runs/train/yolo11n_dinosam[_e3]/):
python src/train.py --models yolo11n --epochs 3 --labels-dir data/labels/auto_generated --run-suffix _dinosam_e3
python src/train.py --models yolo11n --epochs 10 --labels-dir data/labels/auto_generated --run-suffix _dinosam
python src/train_statistics.py
# after freeze: python src/auto_label.py --allow-eval --splits eval
# GPU / 50 epochs: python src/train.py --config configs/train.yaml
# hold-out A/B (writes *_dinosam.json; baseline metrics untouched):
python src/evaluate_custom.py --weights runs/train/yolo11n_dinosam/weights/best.pt \
  --thresholds-path data/splits/eval_thresholds_dinosam.json \
  --metrics-path data/splits/eval_metrics_dinosam.json --tune-val --score-eval
```

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
python src/estimate_distance.py --config configs/data.yaml
python src/train.py --config configs/train.yaml
python src/train_statistics.py
python src/auto_label.py --allow-eval --splits eval            # after freeze; all eval clips
# python src/auto_label.py --allow-eval --splits eval --clips F  # F only; keeps other eval folders
python src/cleanup_labels.py --allow-eval --splits eval --auto-only
python src/cleanup_labels.py --allow-eval --splits eval
python src/dataset_statistics.py --allow-eval
python src/estimate_distance.py --allow-eval
python src/evaluate_custom.py --tune-val --score-eval
# regenerates outputs/examples/ (GT|pred side-by-side + combined)
python src/evaluate_custom.py --export-examples
python src/error_analysis.py
python src/auto_label_dino_sam.py                              # DINO+SAM factory (~3.6 h CPU)
python src/auto_label_dino_sam.py --allow-eval                  # hold-out proxy-GT audit (~4.5 h)
python src/train.py --models yolo11n --epochs 10 --labels-dir data/labels/auto_generated --run-suffix _dinosam
python src/evaluate_custom.py --weights runs/train/yolo11n_dinosam/weights/best.pt \
  --thresholds-path data/splits/eval_thresholds_dinosam.json \
  --metrics-path data/splits/eval_metrics_dinosam.json --tune-val --score-eval
python src/evaluate_custom.py --weights runs/train/yolo11n_dinosam/weights/best.pt \
  --thresholds-path data/splits/eval_thresholds_dinosam.json \
  --metrics-path data/splits/eval_metrics_dinosam.json \
  --examples-dir outputs/examples_dinosam --export-examples
python src/export_eval_video.py --tag dinosam    # outputs/videos/eval_{E,F}_dinosam.mp4
```

## Submission checklist

What the brief asks for, mapped to this repo:

| Ask | Where |
|-----|--------|
| Runnable code | `src/`, `configs/`, `requirements.txt`, Quickstart above |
| Strategy + key decisions | README sections above |
| Metrics table | Evaluation & Metrics |
| Example predictions with GT overlaid | `outputs/examples/` (clean student) and `outputs/examples_dinosam/` (factory student) |
| Short detector video on eval | `outputs/videos/eval_E_dinosam.mp4`, `outputs/videos/eval_F_dinosam.mp4` |
| Failure analysis | `data/splits/error_taxonomy.json`, `outputs/diagnostics/`, `data/hard_negatives/` |
| Automated labeling pipeline | `src/auto_label_dino_sam.py`, `data/labels/auto_generated/`, `data/splits/auto_label_dino_sam.json`, `results/auto_label_dino_sam/` |
| Hold-out proxy-GT audit | `data/labels/eval_dino_sam/`, `data/splits/auto_label_dino_sam_eval.json`, `results/auto_label_dino_sam_eval/` |
| Clean vs DINO+SAM hold-out A/B | `data/splits/eval_metrics_dinosam.json`, `data/splits/eval_ab_clean_vs_dinosam.json` |
| Trained weights or a link | `runs/train/yolo11n/weights/best.pt` (clean labels) and `runs/train/yolo11n_dinosam/weights/best.pt` (factory labels) are gitignored (`*.pt`); upload to Drive/S3/HF and put the URL in the README or repo description |

Ship a **public** GitHub repo, or private with reviewer access. Do **not** commit raw videos, `data/frames/`, or `.pt` blobs if the host is picky about size — keep cleaned eval labels (`data/labels/eval/`), splits/metrics JSON, and `outputs/examples/`.

## Failure Modes & Trade-offs

2 fps on train undersamples fast motion and short occlusions; 5 fps eval is denser than the labels the student saw. I accepted that to stay inside the time box.

Distance from box size will mis-bin trucks, occluded cars, and oblique views. Far-band recall will likely be the weak number: 4K boxes become a few pixels after 1280 letterbox.

Train is landscape 1080p/4K; hold-out mixes E (portrait, near) and F (higher altitude, far). F is where Det and FA/min break. Teacher boxes on B include a few non-vehicles at `conf=0.15`; that noise goes into the students unless cleanup removes it.

The hold-out audit says ~44% of the student's FPs land on a box a second model calls a vehicle, so reported precision (0.605 near-band) understates the model. I did not restate the metrics on that basis: the audit samples 126 of 379 FPs and the auditor has its own error rate, so it bounds the bias rather than removing it.

The DINO+SAM factory trades recall for geometry on this checkpoint: tighter boxes and rule-enforced schema, but 66% agreement on Clip C's low-contrast mid-range cars. Its 22.5 extra boxes/frame are unaudited — part real vehicles the teacher missed, part new FPs. On the hold-out A/B the factory student improves far Det (0.063 → 0.114) and F near Det, with near F1 flat; Clip E near Det drops under the higher frozen conf (0.50 vs 0.20). Absolute far-band performance is still the unsolved problem.

I am not claiming a Pi-ready detector. YOLO11n at 1280 is a training choice; onboard would be a smaller input, INT8, and a tracker, on a board I did not run here.
