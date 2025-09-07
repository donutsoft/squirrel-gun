"""
Minimal annotator: click two corners to label a bounding box for YOLO.

Usage:
  python scratch/annotate_bbox.py \
    --image scratch/dataset/images/train/img_001.jpeg \
    --out scratch/dataset/labels/train/img_001.txt \
    --class-id 0

Controls:
  - Left-click twice: first = top-left (or any corner), second = opposite corner
  - 'r' key: reset current selection
  - 'q' key or close window: save label and exit

Saves one line in YOLO format: "<class> x_center y_center width height" (normalized).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2


def to_yolo(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[float, float, float, float]:
    xa, xb = sorted([x1, x2])
    ya, yb = sorted([y1, y2])
    bw = xb - xa
    bh = yb - ya
    cx = xa + bw / 2
    cy = ya + bh / 2
    return cx / w, cy / h, bw / w, bh / h


def annotate(image_path: Path, out_path: Path, class_id: int) -> None:
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ih, iw = img.shape[:2]

    fig, ax = plt.subplots(1)
    ax.imshow(img)
    ax.set_title("Click two opposite corners; 'q' to save, 'r' to reset")

    clicks: List[Tuple[int, int]] = []
    rect_patch = None

    def on_click(event):
        nonlocal rect_patch
        if event.inaxes != ax:
            return
        if event.button != 1:  # left click only
            return
        x, y = int(event.xdata), int(event.ydata)
        clicks.append((x, y))
        if len(clicks) == 2:
            (x1, y1), (x2, y2) = clicks
            xa, xb = sorted([x1, x2])
            ya, yb = sorted([y1, y2])
            if rect_patch is not None:
                rect_patch.remove()
            rect_patch = patches.Rectangle((xa, ya), xb - xa, yb - ya, linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect_patch)
            fig.canvas.draw()

    def on_key(event):
        nonlocal clicks, rect_patch
        if event.key == 'r':
            clicks = []
            if rect_patch is not None:
                rect_patch.remove()
                rect_patch = None
            fig.canvas.draw()
        if event.key == 'q':
            plt.close(fig)

    cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    # After window closes
    if len(clicks) < 2:
        print("No box confirmed; nothing saved.")
        return
    (x1, y1), (x2, y2) = clicks[:2]
    xc, yc, bw, bh = to_yolo(x1, y1, x2, y2, iw, ih)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
    print(f"Saved label -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Click two corners to annotate one YOLO bbox")
    ap.add_argument("--image", required=True, help="Path to image to annotate")
    ap.add_argument("--out", required=True, help="Path to label .txt to write")
    ap.add_argument("--class-id", type=int, default=0, help="Class id (default 0 for squirrel)")
    args = ap.parse_args()
    annotate(Path(args.image), Path(args.out), args.class_id)


if __name__ == "__main__":
    main()

