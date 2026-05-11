## YOLO bounding box detector

Train with the values in `settings.conf`:

```bash
uv run yolo_bbox_detector.py --conf settings.conf train
```

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
