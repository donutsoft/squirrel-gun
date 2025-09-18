#!/usr/bin/env python3
"""
Convert bounding boxes from an original image size to a letterboxed target size.

Input format (same as bbox_viewer):
  <image_path> x1 y1 x2 y2

Coordinates are assumed to be absolute pixels in the original image.
Output format is the same, but coordinates are mapped into the target
letterboxed space (e.g., 320x320) using the same scale+pad as in
image_extractor/extract_frames.py.

Example:
  python image_extractor/convert_bboxes_letterbox.py \
      --in model_maker/bboxes.txt \
      --out model_maker/bboxes_320.txt \
      --orig 1280 720 --target 320 320
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_line(line: str):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    key = parts[0]
    try:
        x1, y1, x2, y2 = map(int, parts[1:5])
    except Exception:
        return None
    return key, (x1, y1, x2, y2)


def clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def convert_box(x1: int, y1: int, x2: int, y2: int, W: int, H: int, TW: int, TH: int):
    # Scale preserving aspect ratio, then pad to center
    s = min(TW / float(W), TH / float(H))
    new_w = int(round(W * s))
    new_h = int(round(H * s))
    dx = (TW - new_w) / 2.0
    dy = (TH - new_h) / 2.0

    def map_pt(x: int, y: int):
        xp = int(round(x * s + dx))
        yp = int(round(y * s + dy))
        return clamp(xp, 0, TW - 1), clamp(yp, 0, TH - 1)

    xa, ya = map_pt(x1, y1)
    xb, yb = map_pt(x2, y2)
    x1p, x2p = sorted([xa, xb])
    y1p, y2p = sorted([ya, yb])
    if x2p <= x1p or y2p <= y1p:
        return None
    return x1p, y1p, x2p, y2p


def main():
    ap = argparse.ArgumentParser(description="Convert bbox_viewer boxes to letterboxed target size.")
    ap.add_argument("--in", dest="in_path", type=Path, required=True, help="Input bboxes file")
    ap.add_argument("--out", dest="out_path", type=Path, required=True, help="Output bboxes file")
    ap.add_argument("--orig", nargs=2, type=int, metavar=("W", "H"), required=True, help="Original image size")
    ap.add_argument("--target", nargs=2, type=int, metavar=("W", "H"), default=(320, 320), help="Target letterbox size")
    args = ap.parse_args()

    W, H = args.orig
    TW, TH = args.target

    if not args.in_path.exists():
        raise SystemExit(f"Input file not found: {args.in_path}")

    out_lines = []
    skipped = 0
    total = 0

    with args.in_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            parsed = parse_line(line)
            if not parsed:
                skipped += 1
                continue
            key, (x1, y1, x2, y2) = parsed
            mapped = convert_box(x1, y1, x2, y2, W, H, TW, TH)
            if not mapped:
                skipped += 1
                continue
            X1, Y1, X2, Y2 = mapped
            out_lines.append(f"{key} {X1} {Y1} {X2} {Y2}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("w") as f:
        f.write("\n".join(out_lines) + ("\n" if out_lines else ""))

    print(f"Converted {len(out_lines)} boxes; skipped {skipped} (out of {total}).")


if __name__ == "__main__":
    main()

