"""
Interactive bounding box annotator for images using OpenCV.

Features:
- Click and drag to create rectangles (multiple per image)
- Undo last box ('u'), delete all for image ('d')
- Navigate images: next ('n' or Enter), previous ('p')
- Save annotations to a CSV file ('s') and auto-save on changes
- Resume from existing annotations file

Saved CSV format:
  image,label,xmin,ymin,xmax,ymax

Coordinates are absolute pixels in the original image size (not scaled).

Usage examples:
  python bbox_viewer.py --changed-only

  python image_extractor/bbox_viewer.py \
      --images-dir image_extractor/stills \
      --ext jpg --ext png --out image_extractor/bboxes.txt

  # Only browse images currently changed in git
  python image_extractor/bbox_viewer.py \
      --images-dir dataset/positives \
      --out dataset/bboxes.txt \
      --changed-only
"""

from __future__ import annotations

import argparse
import os
import csv
from dataclasses import dataclass, field
import shutil
from pathlib import Path
import subprocess
from typing import Dict, List, Tuple

import cv2


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGES_DIR = REPO_ROOT / "dataset" / "positives"
DEFAULT_OUT_PATH = REPO_ROOT / "dataset" / "bboxes.txt"


Point = Tuple[int, int]
Box = Tuple[int, int, int, int]  # x1, y1, x2, y2 (absolute in original image)


@dataclass
class State:
    img_index: int
    image_paths: List[Path]
    boxes_by_image: Dict[str, List[Box]]
    labels_by_image: Dict[str, str] = field(default_factory=dict)
    drawing: bool = False
    start_pt: Point | None = None
    current_pt: Point | None = None
    scale: float = 1.0
    display_img: any | None = None
    original_img: any | None = None


def clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def scaled_to_original(pt: Point, scale: float) -> Point:
    x, y = pt
    if scale == 1.0:
        return x, y
    return int(round(x / scale)), int(round(y / scale))


def original_to_scaled(pt: Point, scale: float) -> Point:
    x, y = pt
    if scale == 1.0:
        return x, y
    return int(round(x * scale)), int(round(y * scale))


def normalize_box(x1: int, y1: int, x2: int, y2: int) -> Box:
    xa, xb = sorted([x1, x2])
    ya, yb = sorted([y1, y2])
    return xa, ya, xb, yb


def draw_boxes(img, boxes: List[Box], scale: float, color=(0, 255, 0)):
    if not boxes:
        return
    for (x1, y1, x2, y2) in boxes:
        p1 = original_to_scaled((x1, y1), scale)
        p2 = original_to_scaled((x2, y2), scale)
        cv2.rectangle(img, p1, p2, color, 2)


def put_help_overlay(img):
    lines = [
        "drag: new box",
        "u: undo  d: delete-all",
        "n/Enter: next  p: prev",
        "x: delete image  s: save  q: quit",
    ]
    x, y = 10, 20
    for line in lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


def load_annotations_raw(path: Path) -> List[Tuple[str, str, Box]]:
    """Load entries as (key_str, label, box) without resolving to Paths."""
    entries: List[Tuple[str, str, Box]] = []
    if not path.exists():
        return entries
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                if "," in line:
                    # Also accept the CSV form used by dataset/bboxes.txt:
                    # filename,label,xmin,ymin,xmax,ymax
                    parts = next(csv.reader([line]))
                    img_path_str = parts[0].strip()
                    label = parts[1].strip() or "rat"
                    x1, y1, x2, y2 = map(int, (value.strip() for value in parts[2:6]))
                else:
                    parts = line.split()
                    img_path_str = parts[0]
                    label = "rat"
                    x1, y1, x2, y2 = map(int, parts[1:5])
                entries.append((img_path_str, label, (x1, y1, x2, y2)))
            except Exception:
                # Skip malformed lines
                continue
    return entries


def save_annotations_csv(
    csv_path: Path,
    boxes_by_image: Dict[str, List[Box]],
    labels_by_image: Dict[str, str] | None = None,
    default_label: str = "rat",
):
    """
    Save annotations to CSV with rows:
      image,label,xmin,ymin,xmax,ymax

    - filename: basename of the image file (no directories)
    - label: loaded per-image label, or default_label for new images
    - coordinates: absolute pixel coordinates in the original image
    """
    labels_by_image = labels_by_image or {}
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label", "xmin", "ymin", "xmax", "ymax"])
        for key in sorted(boxes_by_image.keys()):
            fname = os.path.basename(key)
            label = labels_by_image.get(key, default_label)
            for (x1, y1, x2, y2) in boxes_by_image[key]:
                writer.writerow([fname, label, x1, y1, x2, y2])


def collect_images(images_dir: Path | None, images: List[Path], exts: List[str]) -> List[Path]:
    paths: List[Path] = []
    norm_exts = {e.lower().lstrip('.') for e in (exts or [])}
    if images:
        for p in images:
            p = Path(p)
            if p.is_dir():
                for sub in p.rglob("*"):
                    if sub.suffix.lower().lstrip('.') in norm_exts:
                        paths.append(sub)
            else:
                paths.append(p)
    elif images_dir:
        for p in images_dir.rglob("*"):
            if not p.is_file():
                continue
            if norm_exts and p.suffix.lower().lstrip('.') not in norm_exts:
                continue
            paths.append(p)
    # Fallback extensions if none provided
    if not paths and images_dir:
        for p in images_dir.rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                paths.append(p)
    paths = [p for p in paths if p.is_file()]
    paths.sort(key=lambda p: str(p))
    return paths


def get_changed_paths(repo_dir: Path | None = None) -> set[Path]:
    """Return paths Git reports as changed in the working tree."""
    cwd = str(repo_dir or Path.cwd())
    try:
        root_result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        repo_root = Path(root_result.stdout.strip()).resolve()
        status_result = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Unable to inspect changed files; run the viewer inside a git repository") from exc

    # -z keeps filenames containing whitespace unambiguous. A porcelain v1
    # record has a two-character status followed by a space and the path.
    return {
        (repo_root / record[3:].decode("utf-8", errors="surrogateescape")).resolve()
        for record in status_result.stdout.split(b"\0")
        if len(record) >= 4 and record[2:3] == b" "
    }


def filter_changed_images(image_paths: List[Path], repo_dir: Path | None = None) -> List[Path]:
    """Keep only collected images whose exact paths Git reports as changed."""
    changed_paths = get_changed_paths(repo_dir)
    return [path for path in image_paths if path.resolve() in changed_paths]


def load_and_prepare_image(img_path: Path, max_dim: int) -> Tuple[any, float, any]:
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        disp = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        disp = img.copy()
    return disp, scale, img


def run_viewer(state: State, csv_out_path: Path, max_dim: int, key_for_save, resolve_key_to_path, delete_image_fn):
    win = "BBox Annotator"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def refresh_display():
        if state.original_img is None:
            return
        base = state.display_img.copy()
        # Draw saved boxes
        curr_img = state.image_paths[state.img_index]
        key = key_for_save(curr_img)
        draw_boxes(base, state.boxes_by_image.get(key, []), state.scale, (0, 255, 0))
        # Draw current active box
        if state.drawing and state.start_pt and state.current_pt:
            cv2.rectangle(base, state.start_pt, state.current_pt, (0, 0, 255), 2)
        # Overlay help + filename
        put_help_overlay(base)
        name = str(curr_img)
        cv2.putText(base, name, (10, base.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(base, name, (10, base.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, base)

    def load_current_image():
        img_path = state.image_paths[state.img_index]
        disp, scale, orig = load_and_prepare_image(img_path, max_dim)
        state.display_img = disp
        state.scale = scale
        state.original_img = orig
        state.drawing = False
        state.start_pt = None
        state.current_pt = None
        refresh_display()

    def on_mouse(event, x, y, flags, param):
        h, w = state.display_img.shape[:2]
        x = clamp(x, 0, w - 1)
        y = clamp(y, 0, h - 1)
        if event == cv2.EVENT_LBUTTONDOWN:
            state.drawing = True
            state.start_pt = (x, y)
            state.current_pt = (x, y)
            refresh_display()
        elif event == cv2.EVENT_MOUSEMOVE and state.drawing:
            state.current_pt = (x, y)
            refresh_display()
        elif event == cv2.EVENT_LBUTTONUP and state.drawing:
            state.drawing = False
            state.current_pt = (x, y)
            if state.start_pt is None:
                refresh_display()
                return
            # Convert to original coords and store
            p1_o = scaled_to_original(state.start_pt, state.scale)
            p2_o = scaled_to_original(state.current_pt, state.scale)
            x1, y1, x2, y2 = normalize_box(*p1_o, *p2_o)
            if (x2 - x1) >= 2 and (y2 - y1) >= 2:
                img_p = state.image_paths[state.img_index]
                key = key_for_save(img_p)
                state.boxes_by_image.setdefault(key, []).append((x1, y1, x2, y2))
                state.labels_by_image.setdefault(key, "rat")
                save_annotations_csv(csv_out_path, state.boxes_by_image, state.labels_by_image)
            state.start_pt = None
            state.current_pt = None
            refresh_display()

    cv2.setMouseCallback(win, on_mouse)

    load_current_image()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), 27):  # q or Esc
            break
        elif key in (ord('n'), 13):  # n or Enter
            if state.img_index < len(state.image_paths) - 1:
                state.img_index += 1
                load_current_image()
        elif key == ord('p'):
            if state.img_index > 0:
                state.img_index -= 1
                load_current_image()
        elif key == ord('u'):
            img_p = state.image_paths[state.img_index]
            key = key_for_save(img_p)
            lst = state.boxes_by_image.get(key, [])
            if lst:
                lst.pop()
                if not lst:
                    state.boxes_by_image.pop(key, None)
                save_annotations_csv(csv_out_path, state.boxes_by_image, state.labels_by_image)
                refresh_display()
        elif key == ord('d'):
            img_p = state.image_paths[state.img_index]
            k = key_for_save(img_p)
            if k in state.boxes_by_image:
                state.boxes_by_image.pop(k)
                save_annotations_csv(csv_out_path, state.boxes_by_image, state.labels_by_image)
                refresh_display()
        elif key == ord('s'):
            save_annotations_csv(csv_out_path, state.boxes_by_image, state.labels_by_image)
            refresh_display()
        elif key == ord('x'):
            # Delete current image via provided deleter
            if not state.image_paths:
                continue
            curr_img = state.image_paths[state.img_index]
            # Remove boxes for this image key if present
            k = key_for_save(curr_img)
            if k in state.boxes_by_image:
                state.boxes_by_image.pop(k, None)
                save_annotations_csv(csv_out_path, state.boxes_by_image, state.labels_by_image)
            deleted, msg = delete_image_fn(curr_img)
            # Update list and move selection
            if deleted:
                state.image_paths.pop(state.img_index)
                if not state.image_paths:
                    break
                if state.img_index >= len(state.image_paths):
                    state.img_index = len(state.image_paths) - 1
                load_current_image()
            else:
                # Show message briefly
                base = state.display_img.copy()
                cv2.putText(base, msg or "Delete failed", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.putText(base, msg or "Delete failed", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(win, base)

    # Final save on exit
    save_annotations_csv(csv_out_path, state.boxes_by_image, state.labels_by_image)
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Interactive bounding box annotator")
    ap.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="Directory to scan for images (recursive)",
    )
    ap.add_argument("--image", action="append", default=[], help="Specific image path(s) or directory(ies); can repeat")
    ap.add_argument("--ext", action="append", default=["jpg", "jpeg", "png"], help="Image extensions to include (no dot)")
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="CSV annotation file used for input and output",
    )
    ap.add_argument("--max-dim", type=int, default=1280, help="Max display dimension (longer side)")
    ap.add_argument(
        "--changed-only",
        "--staged-only",
        dest="changed_only",
        action="store_true",
        help="Only show images Git reports as changed (including untracked files)",
    )
    ap.add_argument("--delete-mode", choices=["trash", "rm"], default="trash", help="How to delete images: move to trash dir or remove")
    ap.add_argument("--trash-dir", type=Path, default=None, help="Trash directory for deleted images (default: .trash under images-dir)")
    args = ap.parse_args()

    images_list = [Path(p) for p in args.image]
    image_paths = collect_images(args.images_dir, images_list, args.ext)
    if not image_paths:
        raise SystemExit("No images found. Provide --images-dir or --image paths.")

    # Normalize paths to absolute to avoid duplicates from differing relpaths
    image_paths = [p.resolve() for p in image_paths]

    if args.changed_only:
        try:
            image_paths = filter_changed_images(image_paths)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        if not image_paths:
            raise SystemExit("No changed images found among the selected image paths.")

    # Determine key saving strategy: relative to images-dir if given, else basename
    def key_for_save(p: Path) -> str:
        if args.images_dir:
            try:
                return str(p.relative_to(args.images_dir.resolve()))
            except Exception:
                return p.name
        return p.name

    # Resolve saved key back to a Path in image_paths
    # Build lookup maps for speed
    by_rel: Dict[str, Path] = {}
    by_name: Dict[str, List[Path]] = {}
    images_dir_abs = args.images_dir.resolve() if args.images_dir else None
    for p in image_paths:
        if images_dir_abs:
            try:
                by_rel[str(p.relative_to(images_dir_abs))] = p
            except Exception:
                pass
        by_name.setdefault(p.name, []).append(p)

    def resolve_key_to_path(key: str) -> Path | None:
        # Absolute path in file
        kp = Path(key)
        if kp.is_absolute() and kp.exists():
            return kp.resolve()
        # Relative (to images_dir)
        if images_dir_abs:
            cand = (images_dir_abs / key).resolve()
            if cand.exists():
                return cand
            if key in by_rel:
                return by_rel[key]
        # Basename fallback (if unique)
        lst = by_name.get(key)
        if lst and len(lst) == 1:
            return lst[0]
        return None

    # Load existing annotations and map to keys used by key_for_save
    boxes_by_image: Dict[str, List[Box]] = {}
    labels_by_image: Dict[str, str] = {}
    for key_str, label, box in load_annotations_raw(args.out):
        p = resolve_key_to_path(key_str)
        if p is None:
            # Store using original key to avoid data loss across sessions
            key = key_str
        else:
            key = key_for_save(p)
        boxes_by_image.setdefault(key, []).append(box)
        labels_by_image.setdefault(key, label)

    # Configure delete behavior
    if args.trash_dir is not None:
        trash_dir = args.trash_dir.resolve()
    elif args.images_dir is not None:
        trash_dir = (args.images_dir.resolve() / ".trash").resolve()
    else:
        trash_dir = Path("image_extractor/.trash").resolve()

    def unique_path(dst: Path) -> Path:
        if not dst.exists():
            return dst
        stem = dst.stem
        suffix = dst.suffix
        parent = dst.parent
        i = 1
        while True:
            cand = parent / f"{stem}__{i}{suffix}"
            if not cand.exists():
                return cand
            i += 1

    def delete_image_fn(p: Path):
        try:
            if args.delete_mode == "rm":
                p.unlink(missing_ok=False)
                return True, f"Removed {p.name}"
            # trash mode
            if args.images_dir is not None:
                try:
                    rel = p.resolve().relative_to(args.images_dir.resolve())
                except Exception:
                    rel = Path(p.name)
            else:
                rel = Path(p.name)
            dst = unique_path(trash_dir / rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dst))
            return True, f"Moved to trash: {rel}"
        except Exception as e:
            return False, f"Delete failed: {e}"

    # The configured annotation file is the CSV output consumed by training.
    csv_out_path = args.out

    state = State(
        img_index=0,
        image_paths=image_paths,
        boxes_by_image=boxes_by_image,
        labels_by_image=labels_by_image,
    )
    run_viewer(state, csv_out_path, args.max_dim, key_for_save, resolve_key_to_path, delete_image_fn)


if __name__ == "__main__":
    main()
