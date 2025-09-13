#!/usr/bin/env python3
"""
Delete JPG images that contain motion-detection green rectangles.

Heuristic (robust):
  - HSV-based green mask + RGB-based green mask combined
  - Optional rectangle-shape check if OpenCV is available

Usage examples:
  Dry run (default):
    python3 delete_green_boxes.py

  Actually delete matches:
    python3 delete_green_boxes.py --delete

  Move matches to a quarantine directory instead of deleting:
    python3 delete_green_boxes.py --move-to image_extractor/quarantine

Helpful debugging:
  - Save masks:  --save-mask image_extractor/masks
  - Print stats: --print-stats

Key thresholds (adjustable):
  HSV hue range for green: ~35–95 degrees, with S/V minimums
  RGB fallback: moderately high G relative to R/B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    from PIL import Image
except Exception as e:
    print("This script requires Pillow and numpy.", file=sys.stderr)
    raise


def _green_mask_rgb(arr, *, g_min: int, r_max: int, b_max: int, delta: int):
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    max_rb = np.maximum(r, b)
    return (g >= g_min) & (r <= r_max) & (b <= b_max) & ((g - max_rb) >= delta)


def _green_mask_hsv_pillow(im: Image.Image, *, hue_min_deg: float, hue_max_deg: float,
                           sat_min: int, val_min: int):
    # Convert to HSV using Pillow; H is 0..255 mapped from 0..360 degrees
    hsv = im.convert("HSV")
    arr = np.asarray(hsv)
    H = arr[:, :, 0].astype(np.int16)
    S = arr[:, :, 1].astype(np.int16)
    V = arr[:, :, 2].astype(np.int16)

    def deg_to_h(val_deg: float) -> int:
        return int(np.clip(round((val_deg % 360) * (255.0 / 360.0)), 0, 255))

    h_min = deg_to_h(hue_min_deg)
    h_max = deg_to_h(hue_max_deg)

    if h_min <= h_max:
        h_mask = (H >= h_min) & (H <= h_max)
    else:
        # wrap-around case
        h_mask = (H >= h_min) | (H <= h_max)

    s_mask = S >= sat_min
    v_mask = V >= val_min
    return h_mask & s_mask & v_mask


def is_annotated(
    img_path: Path,
    *,
    g_min: int,
    r_max: int,
    b_max: int,
    delta: int,
    hue_min_deg: float,
    hue_max_deg: float,
    sat_min: int,
    val_min: int,
    min_ratio: float,
    min_pixels: int,
    require_rect: bool,
    save_mask_dir: Optional[Path] = None,
    print_stats: bool = False,
) -> bool:
    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            arr = np.asarray(im)
    except Exception:
        return False

    if arr.ndim != 3 or arr.shape[2] != 3:
        return False

    # Combine HSV and RGB masks to be more tolerant
    mask_rgb = _green_mask_rgb(arr, g_min=g_min, r_max=r_max, b_max=b_max, delta=delta)
    mask_hsv = _green_mask_hsv_pillow(im, hue_min_deg=hue_min_deg, hue_max_deg=hue_max_deg,
                                      sat_min=sat_min, val_min=val_min)
    mask = mask_rgb | mask_hsv

    count = int(mask.sum())
    total = int(arr.shape[0] * arr.shape[1])

    if count < min_pixels:
        if print_stats:
            print(f"{img_path.name}: green_pixels={count} (< min_pixels)")
        return False

    ratio = count / max(total, 1)
    if print_stats:
        print(f"{img_path.name}: green_pixels={count}, ratio={ratio:.6f}")

    if save_mask_dir is not None:
        save_mask_dir.mkdir(parents=True, exist_ok=True)
        from PIL import ImageOps
        # Save binary mask as visible image
        m = (mask.astype(np.uint8) * 255)
        Image.fromarray(m).save(save_mask_dir / f"{img_path.stem}_mask.png")

    if ratio < min_ratio:
        return False

    if not require_rect:
        return True

    # Optional rectangle/edge check using OpenCV if available
    try:
        import cv2  # type: ignore
    except Exception:
        # If cv2 not available, fallback to color-only decision
        return True

    # Use Canny edges on mask and Hough lines to see if there are long straight edges
    mask_u8 = (mask.astype(np.uint8) * 255)
    edges = cv2.Canny(mask_u8, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=12, maxLineGap=6)
    if lines is None or len(lines) < 2:
        return False
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="stills", help="Directory to scan recursively")
    parser.add_argument("--glob", default="**/*.jpg", help="Glob of files to check")
    parser.add_argument("--min-ratio", type=float, default=0.0002, help="Minimum green ratio to flag")
    parser.add_argument("--min-pixels", type=int, default=200, help="Minimum green pixels to flag")
    # RGB gates (looser by default to handle overlays of various brightness)
    parser.add_argument("--g-min", type=int, default=140, help="Minimum G value")
    parser.add_argument("--r-max", type=int, default=140, help="Maximum R value")
    parser.add_argument("--b-max", type=int, default=140, help="Maximum B value")
    parser.add_argument("--delta", type=int, default=40, help="Minimum G - max(R,B)")
    # HSV thresholds
    parser.add_argument("--hue-min", type=float, default=35.0, help="Minimum hue in degrees (0-360)")
    parser.add_argument("--hue-max", type=float, default=95.0, help="Maximum hue in degrees (0-360)")
    parser.add_argument("--sat-min", type=int, default=60, help="Minimum saturation (0-255)")
    parser.add_argument("--val-min", type=int, default=60, help="Minimum value/brightness (0-255)")
    parser.add_argument("--delete", action="store_true", help="Delete matched files")
    parser.add_argument("--move-to", help="Move matched files to this directory")
    parser.add_argument("--require-rect", action="store_true", default=True, help="Require rectangle-like edges (needs OpenCV if available)")
    parser.add_argument("--save-mask", help="Directory to save binary masks for debugging")
    parser.add_argument("--print-stats", action="store_true", help="Print per-image pixel counts/ratios")

    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent / args.dir
    files = sorted(root.glob(args.glob))
    if not files:
        print(f"No files found under {root} matching {args.glob}")
        return 0

    save_mask_dir = Path(args.save_mask) if args.save_mask else None
    matches: list[Path] = []
    for p in files:
        if not p.is_file():
            continue
        if is_annotated(
            p,
            g_min=args.g_min,
            r_max=args.r_max,
            b_max=args.b_max,
            delta=args.delta,
            hue_min_deg=args.hue_min,
            hue_max_deg=args.hue_max,
            sat_min=args.sat_min,
            val_min=args.val_min,
            min_ratio=args.min_ratio,
            min_pixels=args.min_pixels,
            require_rect=args.require_rect,
            save_mask_dir=save_mask_dir,
            print_stats=args.print_stats,
        ):
            matches.append(p)

    if not matches:
        print("No annotated images detected.")
        return 0

    if not args.delete and not args.move_to:
        print("Dry run. Matches:")
        for p in matches:
            print(f"  {p}")
        print(f"Total: {len(matches)}")
        print("Use --delete to remove, or --move-to DIR to quarantine.")
        return 0

    if args.move_to:
        dest_dir = Path(args.move_to)
        dest_dir.mkdir(parents=True, exist_ok=True)
        for p in matches:
            target = dest_dir / p.name
            # Ensure unique name if collision
            i = 1
            while target.exists():
                stem = p.stem
                target = dest_dir / f"{stem}__{i}{p.suffix}"
                i += 1
            p.rename(target)
        print(f"Moved {len(matches)} files to {dest_dir}")
        return 0

    # delete
    for p in matches:
        try:
            p.unlink()
        except Exception as e:
            print(f"Failed to delete {p}: {e}", file=sys.stderr)
    print(f"Deleted {len(matches)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
