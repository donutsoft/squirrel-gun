## YOLO bounding box detector

The active model is YOLO26n, replacing YOLOv8n while retaining the same 320px
input, batch size, dataset split, fixed-scene augmentation, and confidence
thresholds. Keeping those controls fixed makes the first deployed run a useful
latency and accuracy comparison. Training and device inference are pinned to
the same YOLO26-capable Ultralytics release.

Train the full fixed-scene model with the values in `settings.conf`:

```bash
uv run yolo_bbox_detector.py --conf settings.conf train
```

This produces the named run `runs/detect/yolo26n_fixed_scene_full`. The
pretrained `yolo26n.pt` checkpoint is downloaded automatically on first use,
then all 50 configured epochs run against the complete prepared training split.
Do not tune confidence thresholds from desktop `.pt` results; first export and
deploy the Edge TPU model, compare its inference timing with the 21.5ms YOLOv8n
baseline, and then evaluate thresholds on the device.

`fixed_scene` is the normal training profile. It disables mosaic, flips,
translation, scaling, rotation, shear, perspective, and image compositing.
Only brightness augmentation remains, since camera geometry is invariant and
the relevant scene change is day versus night.

To run a paired augmentation A/B test using the existing images:

```bash
uv run yolo_bbox_detector.py --conf settings.conf train-ab
```

The command prepares the YOLO dataset once, then starts both models from the
same base checkpoint with the same train/validation membership and random seed:

- `control`: current Ultralytics default augmentations.
- `fixed-scene`: invariant geometry plus brightness variation only.

After training, both `best.pt` files are evaluated against the same validation
images at confidence thresholds 0.4, 0.5, 0.6, 0.7, and 0.8. The comparison
JSON reports positive-image recall, empty-background false-positive rate, and
image-level precision, plus the fixed-scene-minus-control delta at each
threshold. Use `--name EXPERIMENT_NAME` to set the paired run prefix,
`--thresholds ...` to change thresholds, or `--report PATH` to select the JSON
location.

After exporting and deploying the fixed-scene model, stop the Flask daemon so
it releases the Edge TPU, then evaluate the deployed model on the existing
labeled recordings plus a deterministic sample of additional known-negative
backyard images:

```bash
cd /home/pi/squirrel-daemon
pkill -f 'python -m flask run' || true
uv run python -B evaluate_thresholds.py \
  --extra-positives-dir /home/pi/squirrel-training-data/threshold_holdout/positives \
  --extra-negatives-dir /home/pi/squirrel-training-data/threshold_holdout/negatives \
  --extra-negative-limit 100 \
  --extra-negative-seed 42 \
  --thresholds 0.2 0.25 0.3 0.4 0.5 0.6 0.7 0.8 \
  --output-dir threshold_evaluation_fixed_scene
```

The holdout contains complete recording bursts removed before training, with
both day and night examples. Keeping neighboring frames together avoids
train/test leakage from nearly identical frames. The evaluator also excludes
images already referenced by the recording manifest, so additional images are
not double-counted. The console summary and
`threshold_evaluation_fixed_scene/report.json` show true-positive recall and
false-positive rate at each threshold. This existing holdout is useful for
initial calibration, but it does not replace a later check on recordings
captured after deployment.

Run the trained model over videos that are expected to be negative, and save
frames where the model still detects something:

```bash
uv run yolo_bbox_detector.py extract-false-positives
```

The script loads `settings.conf` automatically when it is present. For inference
commands, `--weights` is optional; by default the newest
`runs/detect/*/weights/best.pt` is used, with `last.pt` as fallback.
The false-positive extractor defaults to `conf = 0.05` in `settings.conf` so it
captures weak detections that are still useful for review.

Or override the video source and output directory:

```bash
uv run yolo_bbox_detector.py --conf settings.conf extract-false-positives \
  --source /path/to/video-or-directory \
  --output_dir false_positives/run_001
```

The extractor writes raw candidate frames to `OUTPUT_DIR/frames`, optional boxed
frames to `OUTPUT_DIR/annotated`, and detection metadata to
`OUTPUT_DIR/detections.csv`.
Saved frames are 320x320 gray-letterboxed JPEGs, matching `dataset/negatives`
and the daemon's YOLO preprocessing.

Useful options:

- `--frame_stride 5`: run inference every fifth frame.
- `--min_gap_frames 30`: avoid saving many adjacent frames from the same event.
- `--save_annotated`: save a boxed copy next to the raw frame.
- `--limit 100`: stop after saving 100 candidate frames.
