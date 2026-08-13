# Squirrel Gun – Agent Notes

This document captures the working knowledge added while extending the project: motion detection, event-bus driven reactions, motion-triggered recording, and a recordings UI with deletion.

## Overview
- App: Flask server in `squirrel-daemon/app.py` with hardware controllers under `squirrel-daemon/hardware_controllers`.
- Live video: MJPEG stream from `WebcamController.mjpeg()` exposed at `/webcam/stream`.
- Motion: Detected inside `WebcamController` and published onto a lightweight in-process `EventBus`.
- Reactions: Subscribers (in `app.py`) listen for `'motion'` events to aim the servos and to start/extend video recording for 30s.
- Recordings UI: New page `/recordings` lists and plays saved `.mp4` clips; provides delete-one and delete-all actions.
- Recording labels: `/recordings` can mine false-positive and hard-positive frames with the daemon's Edge-TPU squirrel detector. `/training-frames` reviews and deletes mined images.

## Key Components
- `squirrel-daemon/hardware_controllers/WebcamController.py`
  - Produces JPEG frames for MJPEG streaming.
  - Motion detection (MOG2/KNN/AVG background models) with persistence, size threshold, frame skipping, and downscale for performance.
  - Publishes motion events through a callback (if set via `set_motion_publisher`).
  - Recording support using OpenCV `VideoWriter` (`mp4v`); writes into `static/recordings/rec_YYYYmmDD_HHMMSS.mp4`.
  - No longer self-triggers recording internally; instead it subscribes to the app-wide event bus via `set_event_bus(bus, record_on_motion=True, duration_sec=30.0)`.

- `squirrel-daemon/event_bus.py`
  - Minimal thread-based pub/sub with `publish(topic, data)` and `subscribe(topic, handler)`.

- `squirrel-daemon/app.py`
  - Wires controllers and the `EventBus`.
  - Publishes motion from `WebcamController` to the bus: `webcam.set_motion_publisher(bus.publish)`.
  - Subscribes to `'motion'` to perform two reactions:
    - Servo aiming: `_on_motion(evt)` throttles to ~5Hz and uses `LinearAimer` to point servos.
    - Recording: `webcam.set_event_bus(bus, record_on_motion=True, duration_sec=30.0)` starts/extends a 30s recording window upon each motion event.
  - Adds endpoints to list, delete one, and delete all recordings; renders recordings page.

## Motion Detection
- Config entry points:
  - GET `/webcam/stream` accepts query params to tweak motion at stream start (e.g., `motion=1`, `motion_min_area`, `bg_mode`, `frame_skip`, `scale`).
  - POST `/api/motion/config` updates runtime settings.
  - GET `/api/motion/config` returns current settings.
  - GET `/api/motion/center` returns the current motion rect and normalized center (u, v).
- Notable internal parameters (WebcamController):
  - `_motion_min_area` (pixels at full-res), `_motion_alpha` (learning rate), `_persist_ms` (stability window), `_bg_mode` (`avg|mog2|knn`), `_motion_frame_skip`, `_motion_scale`.

## Recording
- Start/stop API (internal to controller):
  - `start_recording(duration_sec=30.0, extend=True)` starts a session or extends the end time if one is active.
  - `stop_recording()` ends recording and releases the writer.
  - `is_recording()` indicates if active and not expired.
- Triggering policy:
  - The controller subscribes to the `EventBus` via `set_event_bus(bus, record_on_motion=True, duration_sec=30.0)`.
  - When a `'motion'` event arrives, it starts/extends a recording window; no direct coupling to detection loop.
- Files:
  - Saved under `squirrel-daemon/static/recordings/` as `rec_YYYYmmDD_HHMMSS.mp4`.
  - Served by Flask’s static route at `/recordings/<filename>`.

## Recordings UI and APIs
- UI: `squirrel-daemon/templates/Recordings.html` lists newest clips first, with inline `<video>` playback and Download/Delete buttons. Link added on the main control page header.
- Routes:
  - GET `/recordings` → recordings gallery.
  - GET `/api/recordings` → `{ files: [{name, url, size, mtime, display_time}] }`.
  - POST `/api/recordings/delete` with `{ name }` → deletes a single file. Validated by pattern `^rec_YYYYmmDD_HHMMSS.mp4$` and constrained to the recordings folder.
  - POST `/api/recordings/clear` → deletes all `.mp4` files in the recordings folder.
  - POST `/api/recordings/label` with `{ name, label }` → queues `true_positive` or `false_positive` mining.
  - GET `/api/recordings/label/status?name=...` → reports queued/running/completed labeling progress.
  - GET `/training-frames` → persistent positive/negative frame review UI. Positives use a large annotated canvas with detector confidence, previous/next navigation, image deletion, and one-box replacement drawing.
  - POST `/api/training-frames/delete` with `{ kind, name }` → deletes a frame and its positive bbox rows.
  - POST `/api/training-frames/bbox` with `{ name, box }` → replaces every box for a positive frame with one validated box; send `box: null` to clear all boxes.

## Persistent Training Data
- Code lives entirely under `squirrel-daemon`; generated data does not.
- The data root is `squirrel-training-data` beside `clicks.db` (normally `/home/pi/squirrel-training-data` on the Pi), so `deploy.sh` does not remove it.
- New recordings retain lossless 320x320 detector-ready PNG frames under `static/recordings/.training-source/<recording-stem>/`; labeling uses these instead of confidence-distorting MP4 frames. Deleting recordings also deletes their retained detector frames.
- False-positive labels save only the highest-confidence frame that meets the live detection threshold under `negatives/`, avoiding near-duplicate training images from the same recording.
- True-positive labels run the retained frames at a lower mining threshold and save the highest-confidence detection from the hard-positive band below the live threshold under `positives/`, with at most one box per new image in `bboxes.txt` using `image,label,xmin,ymin,xmax,ymax`.
- `recording_labels.json` tracks completed labels and generated files. Relabeling a clip replaces its earlier derived files and bbox rows.
- The labeling service recreates the dataset directories and empty bbox file when the persistent training-data root is removed while the daemon is running.

## Existing Control UI
- Main page: `squirrel-daemon/templates/PanTiltControl.html`.
  - Pan/Tilt buttons and numeric inputs.
  - Live MJPEG stream (`/webcam/stream`).
  - Left-clicking the stream aims at that point; right-clicking records that pixel with the current pan/tilt angles without moving.
  - Motion controls: enable overlay, set min area, enable “follow motion with laser”.
  - Click logging, model training, and water firing.

## Automatic Aim Calibration
- POST `/api/calibration/automatic` no longer requires four manually entered corner angles.
- The daemon starts from the current learned model (or the built-in approximate field-of-view seed), aims at a normalized screen grid, detects the actual laser location, and records the observed angle-to-pixel samples.
- Each point uses two independent laser-off/on capture cycles. Detection keeps only image changes that repeat in both laser-on frames and ranks candidates by distance from the requested screen target, suppressing wind and exposure changes.
- Each full grid is scored before retraining, producing pixel mean error, RMSE, maximum error, and detection success rate.
- It repeats screen-wide validation/training rounds until the configured RMSE and worst-point thresholds pass, or until the safety round limit is reached.
- Measurements remain in memory until a later full-grid round validates the fitted model. A converged, axis-monotonic run is written and activated. When no prior model exists, a non-converged checkpoint may be activated only if it is monotonic, passes a conservative error ceiling, and has validation evidence spanning both axes and all four screen quadrants; otherwise the prior calibration remains unchanged.
- The calibration UI requires one click. Accuracy and timing controls remain available under an advanced settings disclosure.

## Water Recoil Calibration
- The Water panel exposes a **Calibrate Recoil** button. Aim the laser near the center of a stationary surface before using it; the workflow deliberately fires water for two seconds.
- POST `/api/calibration/recoil` captures the laser before and during the water pulse, measures the pixel displacement, and uses the trained aim model to convert the opposite displacement into local pan/tilt offsets.
- Every successful measurement is appended to the `recoil_calibrations` table in `clicks.db`, including raw positions, pixel and normalized offsets, angle offsets, capture parameters, and debug-image metadata. GET `/api/calibration/recoil` returns the latest record.
- Manual and motion-triggered water firing use the latest recoil record. The daemon prepositions to the compensated angles, keeps motion aiming locked for the valve duration, then restores the normal aim.
- Automatic aim calibration and recoil calibration require exclusive access to the water and pan/tilt workflow so valve, servo, and laser commands cannot interleave.

## Notes, Caveats, and Limits
- Video codec/container: Uses `mp4v` in `.mp4`. If your OpenCV build lacks this codec, consider switching to `.avi` with `XVID` or `MJPG` for broad compatibility.
- Concurrency: Recording writes are guarded by `_recording_lock`. Frames are captured in the streaming thread; no additional thread for recording.
- Deleting active file: Deleting a file currently in use by the writer may fail on some platforms. Current API surfaces the error. We can add logic to avoid deleting the active file or stop it first.
- Storage: No rotation/quota management yet. Consider adding max files/bytes and cleanup policies.
- Performance: Motion is computed on downscaled frames and every Nth frame to keep CPU low. JPEG encoding runs at the configured FPS; ensure device bandwidth is sufficient.
- Safety: Recording deletion validates names and restricts paths to `static/recordings` to avoid traversal.

## Local Development Policy
- No fallback logic: Do not implement silent fallbacks or auto-degradation for any feature or dependency. This project runs on a single, controlled machine.
- Fail fast: If something is missing, unsupported, or version-mismatched, raise a clear error instead of attempting alternatives.
- Visibility over resiliency: Prefer explicit errors and actionable messages so issues surface immediately during development.

## Quick How-Tos
- Start the server: `bash squirrel-daemon/run.sh` (ensure dependencies in `pyproject.toml` are installed via `uv` or equivalent).
- Open the live view: visit `/` and `/webcam/stream` loads automatically in the page.
- Enable motion overlay: check “Enable motion overlay” in the UI or POST `/api/motion/config`.
- Follow motion with laser: toggle “Follow motion with laser” (throttled aiming via event bus).
- View recordings: visit `/recordings`.
- Delete one recording: POST `/api/recordings/delete` with `{ "name": "rec_YYYYmmDD_HHMMSS.mp4" }`.
- Delete all recordings: POST `/api/recordings/clear`.

## Potential Next Steps
- API/UI to toggle “record on motion” and adjust duration without code changes.
- Pre/post-roll buffers and/or “smart” extension while motion persists.
- Thumbnail generation and timeline scrubbing.
- Storage quotas and auto-pruning.
- Export/share bundles of clips.

## File Map (new/modified)
- `squirrel-daemon/hardware_controllers/WebcamController.py` — recording support; event-bus subscription; motion publisher remains.
- `squirrel-daemon/app.py` — routes for recordings; event bus wiring for recording; existing motion/aim logic.
- `squirrel-daemon/templates/Recordings.html` — new gallery page with delete actions.
- `squirrel-daemon/templates/PanTiltControl.html` — link to recordings page.
