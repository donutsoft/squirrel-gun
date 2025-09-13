#!/usr/bin/env python3
"""
Extract every frame from all MP4 files in `recordings/` and write JPGs
into `stills/` (one subfolder per video) under the `image_extractor/` directory.

Usage:
  python3 extract_frames.py

Options:
  --source DIR     Source directory containing videos (default: recordings)
  --dest DIR       Destination directory for frames (default: stills)
  --pattern GLOB   File pattern to match videos (default: *.mp4)
  --method NAME    one of: auto, ffmpeg, opencv (default: auto)
  --overwrite      Overwrite existing frames (ffmpeg: -y) (default: false)
  --seconds N      Only extract the first N seconds (default: 5)

Requirements:
  - Preferably `ffmpeg` installed in PATH (fastest and most robust)
  - Fallback: Python package `opencv-python` if ffmpeg is unavailable
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def detect_method(preferred: str) -> str:
    if preferred == "ffmpeg":
        if shutil.which("ffmpeg"):
            return "ffmpeg"
        print("ffmpeg not found in PATH; falling back to opencv", file=sys.stderr)
        return "opencv"
    if preferred == "opencv":
        return "opencv"
    # auto
    return "ffmpeg" if shutil.which("ffmpeg") else "opencv"


def extract_with_ffmpeg(video: Path, out_dir: Path, prefix: str, overwrite: bool, seconds: float) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Example output: stills/<video>/<prefix>_000001.jpg
    out_pattern = str((out_dir / f"{prefix}_%06d.jpg").as_posix())
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(video),
        "-t",
        f"{seconds}",
        "-an",  # no audio
        "-sn",  # no subtitles
        "-vsync",
        "0",    # keep all frames; avoid dup/drop
        "-q:v",
        "2",     # high quality JPEG
        out_pattern,
    ]
    try:
        subprocess.run(cmd, check=True)
        # We don't know count without re-listing; return -1 as unknown
        return -1
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed for {video}: {e}", file=sys.stderr)
        return 0


def extract_with_opencv(video: Path, out_dir: Path, prefix: str, overwrite: bool, seconds: float) -> int:
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        print(
            "OpenCV (opencv-python) not available and ffmpeg not usable; install it to proceed.",
            file=sys.stderr,
        )
        raise

    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"Could not open video: {video}", file=sys.stderr)
        return 0

    count = 0
    idx = 1
    fps = cap.get(5)  # cv2.CAP_PROP_FPS
    max_frames = int(fps * seconds) if fps and fps > 0 else None
    msec_limit = seconds * 1000.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Stop early if we've reached the requested time window
        pos_msec = cap.get(0)  # cv2.CAP_PROP_POS_MSEC
        if (max_frames is not None and idx > max_frames) or (pos_msec and pos_msec >= msec_limit):
            break
        out_path = out_dir / f"{prefix}_{idx:06d}.jpg"
        if out_path.exists() and not overwrite:
            # Skip writing existing frame
            idx += 1
            count += 1
            continue
        ok = cv2.imwrite(str(out_path), frame)
        if not ok:
            print(f"Failed to write frame {idx} for {video}", file=sys.stderr)
            break
        idx += 1
        count += 1

    cap.release()
    return count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="recordings", help="Source directory with videos")
    parser.add_argument("--dest", default="stills", help="Destination directory for frames")
    parser.add_argument("--pattern", default="*.mp4", help="Glob to select video files")
    parser.add_argument(
        "--method",
        choices=["auto", "ffmpeg", "opencv"],
        default="auto",
        help="Extraction backend",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing frames instead of skipping",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="Only extract the first N seconds of each video",
    )

    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent
    src = (root / args.source).resolve()
    dst = (root / args.dest).resolve()

    if not src.exists() or not src.is_dir():
        print(f"Source directory not found: {src}", file=sys.stderr)
        return 2

    method = detect_method(args.method)
    print(f"Using method: {method}")

    videos = sorted(src.glob(args.pattern))
    if not videos:
        print(f"No videos found in {src} matching {args.pattern}")
        return 0

    total_frames = 0
    for vid in videos:
        base = vid.stem
        out_dir = dst / base
        print(f"Extracting frames: {vid.name} -> {out_dir}")
        if method == "ffmpeg":
            extracted = extract_with_ffmpeg(vid, out_dir, base, args.overwrite, args.seconds)
        else:
            extracted = extract_with_opencv(vid, out_dir, base, args.overwrite, args.seconds)
        if extracted >= 0:
            print(f"  Wrote {extracted} frames")
            total_frames += extracted

    if total_frames > 0:
        print(f"Done. Extracted {total_frames} frames across {len(videos)} videos.")
    else:
        print(f"Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
