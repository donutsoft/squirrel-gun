from __future__ import annotations

import csv
import os
import random
import shutil
import platform
import math
from pathlib import Path
import sys
import configparser
from typing import Dict, List, Tuple, Optional, Iterable, Union, Sequence

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None  # Lazy import error handled in train()/predict()

from PIL import Image


_Row = Tuple[str, str, int, int, int, int]  # (image_rel, label, xmin, ymin, xmax, ymax)
_MAC_DEFAULT_DEVICE: Optional[str] = "mps" if platform.system() == "Darwin" else None
_VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
_SCRIPT_DIR = Path(__file__).resolve().parent
_LETTERBOX_PAD_COLOR = 114


class YOLOBBoxDetector:
    """
    Minimal YOLO detector that consumes a CSV of bounding boxes and a positives directory.

    CSV format (pixel coords): image,label,xmin,ymin,xmax,ymax
      - "image" is a path relative to positives_dir (e.g., rat/img001.jpg)

    Usage:
      det = YOLOBBoxDetector(
          positives_dir="positives",
          bbox_file="model_maker/bboxes_resolved.txt",
          yolo_root="yolo_data",
          seed=42,
      )
      det.train(model="yolov8n.pt", epochs=50, imgsz=640)
      results = det.predict("some.jpg")
    """

    def __init__(
        self,
        positives_dir: Union[str, Path],
        bbox_file: Union[str, Path],
        yolo_root: Union[str, Path] = "yolo_data",
        negatives_dir: Optional[Union[str, Path]] = None,
        seed: int = 0,
    ) -> None:
        self.positives_dir = Path(positives_dir)
        self.bbox_file = Path(bbox_file)
        self.yolo_root = Path(yolo_root)
        self.negatives_dir = Path(negatives_dir) if negatives_dir is not None else None
        self.seed = seed

        if not self.positives_dir.exists():
            raise FileNotFoundError(f"positives_dir not found: {self.positives_dir}")
        if not self.bbox_file.exists():
            raise FileNotFoundError(f"bbox_file not found: {self.bbox_file}")

        self._labels: List[str] = []
        self._label_to_id: Dict[str, int] = {}
        self._prepared = False
        self._model = None
        self._default_device = _MAC_DEFAULT_DEVICE

    # --------------- Public API ---------------
    def train(
        self,
        model: str = "yolov8n.pt",
        epochs: int = 50,
        imgsz: int = 320,
        batch: int = 16,
        val_split: float = 0.1,
        device: Optional[Union[int, str]] = None,
        verbose: bool = True,
        workers: Optional[int] = None,
        amp: Optional[bool] = None,
    ):
        """Prepare YOLO dataset and train using ultralytics.

        Returns the underlying ultralytics model instance.
        """
        if YOLO is None:
            raise RuntimeError(
                "ultralytics is not installed. Install it (e.g. pip install ultralytics) to train."
            )

        if not self._prepared:
            self._prepare_yolo(val_split=val_split)

        data_yaml = self.yolo_root / "dataset.yaml"
        if verbose:
            print(f"[YOLO] Training model={model} data={data_yaml} epochs={epochs} imgsz={imgsz} batch={batch}")

        self._model = YOLO(model)
        # On macOS, default to Metal (MPS) for speed if device not specified
        if device is None:
            device = self._default_device

        # Default dataloader workers: use 0 on macOS/MPS to avoid hangs
        if workers is None:
            workers = 0 if (device in ("mps", None) and self._default_device == "mps") else 8
        # Mixed precision sometimes unstable on MPS; default to False there
        if amp is None:
            amp = False if (device in ("mps", None) and self._default_device == "mps") else True

        self._model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            verbose=verbose,
            plots=False,
            workers=workers,
            amp=amp,
        )
        return self._model

    def predict(
        self,
        image: Union[str, Path, 'Image.Image'],
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 320,
        max_det: int = 20,
        classes: Optional[List[int]] = None,
        verbose: bool = False,
    ):
        """Run inference with the trained (or loaded) model.

        Accepts a path or PIL.Image. Returns ultralytics Results list.
        """
        if self._model is None:
            if YOLO is None:
                raise RuntimeError("ultralytics is not installed. Install it to run inference.")
            raise RuntimeError("Model not trained/loaded. Call train() first or load a .pt with load().")
        if classes is None and len(self._labels) == 1:
            classes = [0]
        return self._model.predict(
            source=image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            classes=classes,
            agnostic_nms=False,
            verbose=verbose,
            device=self._default_device,
        )

    def load(self, weights: Union[str, Path]):
        """Load an existing .pt weights file into this detector."""
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed. Install it to load weights.")
        self._model = YOLO(str(weights))
        return self._model

    # --------------- Internal helpers ---------------
    # To keep the class minimal and straightforward, we implement _read_boxes
    # to return mapping grouped by image alongside dimensions, then write YOLO files.

    def _read_boxes(self) -> List[_Row]:
        """Parse CSV rows into _Box entries.

        Note: The CSV header must be: image,label,xmin,ymin,xmax,ymax
        """
        boxes: List[_Row] = []
        with self.bbox_file.open("r", newline="") as f:
            reader = csv.DictReader(f)
            req = ["image", "label", "xmin", "ymin", "xmax", "ymax"]
            for k in req:
                if k not in reader.fieldnames:
                    raise ValueError(f"Missing column '{k}' in {self.bbox_file}")
            for r in reader:
                img_rel = r["image"].strip()
                cls = r["label"].strip()
                xmin = int(float(r["xmin"]))
                ymin = int(float(r["ymin"]))
                xmax = int(float(r["xmax"]))
                ymax = int(float(r["ymax"]))
                boxes.append((img_rel, cls, xmin, ymin, xmax, ymax))
        return boxes

    def _group_by_image(self, boxes: List[_Row]) -> Dict[Path, List[Tuple[str, Tuple[int, int, int, int]]]]:
        by_image: Dict[Path, List[Tuple[str, Tuple[int, int, int, int]]]] = {}
        for img_rel, cls, xmin, ymin, xmax, ymax in boxes:
            img_path = (self.positives_dir / img_rel).resolve()
            by_image.setdefault(img_path, []).append((cls, (xmin, ymin, xmax, ymax)))
        return by_image

    @staticmethod
    def _xyxy_to_yolo(xmin: int, ymin: int, xmax: int, ymax: int, w: int, h: int) -> Tuple[float, float, float, float]:
        xc = (xmin + xmax) / 2.0 / w
        yc = (ymin + ymax) / 2.0 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        return (xc, yc, bw, bh)

    def _ensure_dir(self, p: Path) -> None:
        p.mkdir(parents=True, exist_ok=True)

    def _link_or_copy(self, src: Path, dst: Path) -> None:
        try:
            if dst.exists():
                return
            self._ensure_dir(dst.parent)
            os.symlink(src, dst)
        except Exception:
            shutil.copy2(src, dst)

    def _write_dataset_yaml(self) -> None:
        yaml = [
            f"path: {self.yolo_root}",
            "train: images/train",
            "val: images/val",
            f"nc: {len(self._labels)}",
            f"names: {self._labels}",
            "",
        ]
        (self.yolo_root / "dataset.yaml").write_text("\n".join(yaml))

    def _prepare_yolo(self, val_split: float = 0.1) -> None:  # type: ignore[no-redef]
        boxes = self._read_boxes()
        by_image = self._group_by_image(boxes)
        classes = sorted({c for items in by_image.values() for c, _ in items})
        self._labels = classes
        self._label_to_id = {c: i for i, c in enumerate(classes)}

        imgs = list(by_image.keys())
        # Optionally add negative images (no boxes) from a separate directory
        if self.negatives_dir is not None and self.negatives_dir.exists():
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            for p in self.negatives_dir.rglob("*"):
                if p.suffix.lower() in exts:
                    rp = p.resolve()
                    if rp not in by_image:
                        by_image[rp] = []
                        imgs.append(rp)
        rng = random.Random(self.seed)
        rng.shuffle(imgs)
        n_val = max(1, int(len(imgs) * val_split)) if len(imgs) > 1 else 0
        val_set = set(imgs[:n_val])

        for split in ("train", "val"):
            (self.yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)

        for img_path, items in by_image.items():
            split = "val" if img_path in val_set else "train"
            # Link/copy image
            dst_img = self.yolo_root / "images" / split / img_path.name
            self._link_or_copy(img_path, dst_img)

            # Build label file
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception:
                # Skip unreadable images
                continue

            lines: List[str] = []
            for cls, (xmin, ymin, xmax, ymax) in items:
                # Normalize coordinates to ensure positive width/height
                xa, xb = (xmin, xmax) if xmin <= xmax else (xmax, xmin)
                ya, yb = (ymin, ymax) if ymin <= ymax else (ymax, ymin)
                # Skip degenerate boxes (and warn)
                if xb <= xa or yb <= ya:
                    try:
                        print(f"[WARN] Skipping invalid box (non-positive size) for {img_path.name}: "
                              f"({xmin},{ymin},{xmax},{ymax}) -> ({xa},{ya},{xb},{yb})")
                    except Exception:
                        pass
                    continue
                idx = self._label_to_id[cls]
                x, y, bw, bh = self._xyxy_to_yolo(xa, ya, xb, yb, w, h)
                # Clamp to [0,1] and drop zero-area after normalization
                x = min(max(x, 0.0), 1.0)
                y = min(max(y, 0.0), 1.0)
                bw = min(max(bw, 0.0), 1.0)
                bh = min(max(bh, 0.0), 1.0)
                if bw <= 0.0 or bh <= 0.0:
                    try:
                        print(f"[WARN] Skipping invalid box (zero area after clamp) for {img_path.name}: "
                              f"({xa},{ya},{xb},{yb}) -> (bw={bw:.6f}, bh={bh:.6f})")
                    except Exception:
                        pass
                    continue
                lines.append(f"{idx} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")

            lbl_path = self.yolo_root / "labels" / split / (img_path.with_suffix(".txt").name)
            lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        self._write_dataset_yaml()
        self._prepared = True


def _iter_video_files(source: Path) -> List[Path]:
    """Return video files from a single file or recursively from a directory."""
    if source.is_file():
        if source.suffix.lower() not in _VIDEO_EXTS:
            raise ValueError(f"source is not a supported video file: {source}")
        return [source]
    if source.is_dir():
        videos = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in _VIDEO_EXTS)
        if not videos:
            raise FileNotFoundError(f"no supported video files found under: {source}")
        return videos
    raise FileNotFoundError(f"source not found: {source}")


def _format_seconds(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000.0))
    return f"{millis // 1000:06d}_{millis % 1000:03d}"


def _class_name(names, class_id: int) -> str:
    try:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        return str(names[class_id])
    except Exception:
        return str(class_id)


def letterbox_frame(frame, target: int = 320):
    """Resize a frame into a square gray-letterboxed image."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv-python and numpy are required to letterbox frames.") from exc

    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"invalid frame size: {w}x{h}")

    scale = min(float(target) / float(w), float(target) / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    if len(frame.shape) == 2 or frame.shape[2] == 1:
        canvas = np.full((target, target), _LETTERBOX_PAD_COLOR, dtype=resized.dtype)
    else:
        canvas = np.full((target, target, frame.shape[2]), _LETTERBOX_PAD_COLOR, dtype=resized.dtype)
    top = (target - new_h) // 2
    left = (target - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    unique: List[Path] = []
    seen = set()
    for path in paths:
        try:
            key = path.resolve()
        except Exception:
            key = path.absolute()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _search_roots() -> List[Path]:
    return _unique_paths([Path.cwd(), _SCRIPT_DIR, _SCRIPT_DIR.parent])


def find_latest_yolo_weights(explicit: Optional[Union[str, Path]] = None) -> Path:
    """Resolve explicit weights or find the newest Ultralytics detector weights."""
    roots = _search_roots()

    if explicit is not None:
        explicit_path = Path(explicit).expanduser()
        candidates = [explicit_path] if explicit_path.is_absolute() else [root / explicit_path for root in roots]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        print(f"[WARN] Weights not found at {explicit_path}; searching runs/detect instead.", file=sys.stderr)

    matches: List[Path] = []
    for filename in ("best.pt", "last.pt"):
        for root in roots:
            matches.extend(root.glob(f"runs/detect/*/weights/{filename}"))

    matches = _unique_paths(p for p in matches if p.is_file())
    if not matches:
        searched = ", ".join(str(root / "runs/detect/*/weights/{best,last}.pt") for root in roots)
        raise FileNotFoundError(f"no YOLO weights found. Searched: {searched}")

    def sort_key(path: Path) -> Tuple[int, float]:
        preferred = 1 if path.name == "best.pt" else 0
        return (preferred, path.stat().st_mtime)

    return max(matches, key=sort_key)


def extract_false_positive_frames(
    weights: Union[str, Path],
    source: Union[str, Path],
    output_dir: Union[str, Path],
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 320,
    max_det: int = 20,
    classes: Optional[Sequence[int]] = None,
    device: Optional[Union[int, str]] = None,
    frame_stride: int = 1,
    min_gap_frames: int = 0,
    save_annotated: bool = False,
    jpeg_quality: int = 95,
    limit: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[int, int, Path]:
    """Run a detector over video(s) and save frames that produce detections.

    This is intended for videos known to contain no target object. Any detection
    in those videos is treated as a false positive candidate.

    Returns: (saved_frame_count, processed_frame_count, detections_csv_path)
    """
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed. Install it to run inference.")

    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv-python is required to read video files.") from exc

    frame_stride = max(1, int(frame_stride))
    min_gap_frames = max(0, int(min_gap_frames))
    jpeg_quality = min(max(int(jpeg_quality), 1), 100)

    source_path = Path(source)
    out = Path(output_dir)
    raw_dir = out / "frames"
    annotated_dir = out / "annotated"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if save_annotated:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    videos = _iter_video_files(source_path)
    try:
        if source_path.is_dir() and out.resolve() == source_path.resolve():
            print(
                f"[WARN] output_dir is the same as source; extracted frames will be written inside {out}",
                file=sys.stderr,
            )
    except Exception:
        pass
    csv_path = out / "detections.csv"
    model = YOLO(str(weights))
    names = getattr(model, "names", {})

    saved_total = 0
    processed_total = 0

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "frame",
                "timestamp_sec",
                "saved_frame",
                "annotated_frame",
                "class_id",
                "class_name",
                "confidence",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
            ],
        )
        writer.writeheader()

        for video_number, video_path in enumerate(videos, start=1):
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"[WARN] Could not open video: {video_path}", file=sys.stderr)
                continue

            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_index = -1
            last_saved_frame = -min_gap_frames - 1
            video_processed = 0
            video_saved = 0
            if verbose:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                print(f"[VIDEO] {video_path} frames={total_frames} fps={fps:.3f}")

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index += 1

                if frame_index % frame_stride != 0:
                    continue
                processed_total += 1
                video_processed += 1

                model_frame = letterbox_frame(frame, target=imgsz)
                results = model.predict(
                    source=model_frame,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    max_det=max_det,
                    classes=list(classes) if classes is not None else None,
                    agnostic_nms=False,
                    device=device,
                    verbose=False,
                )
                result = results[0] if results else None
                boxes = None if result is None else getattr(result, "boxes", None)
                try:
                    n_boxes = 0 if boxes is None else int(boxes.shape[0])
                except Exception:
                    n_boxes = 0 if boxes is None else len(boxes)
                if n_boxes <= 0:
                    continue
                if min_gap_frames and (frame_index - last_saved_frame) < min_gap_frames:
                    continue

                seconds = (frame_index / fps) if fps > 0.0 else 0.0
                time_part = _format_seconds(seconds)
                frame_name = f"video{video_number:03d}_{video_path.stem}_frame{frame_index:08d}_t{time_part}.jpg"
                raw_path = raw_dir / frame_name
                cv2.imwrite(str(raw_path), model_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

                annotated_rel = ""
                if save_annotated and result is not None:
                    annotated_path = annotated_dir / frame_name
                    cv2.imwrite(str(annotated_path), result.plot(), [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                    annotated_rel = annotated_path.relative_to(out).as_posix()

                xyxy = boxes.xyxy.detach().cpu().tolist()
                confs = boxes.conf.detach().cpu().tolist()
                clss = boxes.cls.detach().cpu().tolist()
                for coords, score, cls in zip(xyxy, confs, clss):
                    class_id = int(cls)
                    writer.writerow(
                        {
                            "video": video_path.as_posix(),
                            "frame": frame_index,
                            "timestamp_sec": f"{seconds:.3f}",
                            "saved_frame": raw_path.relative_to(out).as_posix(),
                            "annotated_frame": annotated_rel,
                            "class_id": class_id,
                            "class_name": _class_name(names, class_id),
                            "confidence": f"{float(score):.6f}",
                            "xmin": f"{float(coords[0]):.2f}",
                            "ymin": f"{float(coords[1]):.2f}",
                            "xmax": f"{float(coords[2]):.2f}",
                            "ymax": f"{float(coords[3]):.2f}",
                        }
                    )

                saved_total += 1
                video_saved += 1
                last_saved_frame = frame_index
                if verbose:
                    print(f"[FP] {video_path.name} frame={frame_index} detections={n_boxes} -> {raw_path}")
                if limit is not None and saved_total >= limit:
                    break

            cap.release()
            print(f"[VIDEO] {video_path} processed_frames={video_processed} saved_frames={video_saved} conf={conf}")
            if limit is not None and saved_total >= limit:
                break

    return saved_total, processed_total, csv_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Minimal YOLO bbox trainer/predictor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ap.add_argument(
        "--config",
        dest="config",
        type=Path,
        default=None,
        help="Optional .conf (INI) with parameters for selected subcommand; defaults to settings.conf when present",
    )

    p_train = sub.add_parser("train", help="Prepare YOLO data from CSV and train")
    p_train.add_argument("--positives_dir", type=Path, default=None, help="Directory containing images referenced by CSV")
    p_train.add_argument("--bbox_file", type=Path, default=None, help="CSV with image,label,xmin,ymin,xmax,ymax")
    p_train.add_argument("--yolo_root", type=Path, default=Path("yolo_data"), help="Output YOLO dataset root")
    p_train.add_argument("--negatives_dir", type=Path, default=None, help="Optional directory of negative images (no boxes)")
    p_train.add_argument("--model", type=str, default="yolov8n.pt", help="Ultralytics model or config (e.g., yolov8n.pt)")
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--imgsz", type=int, default=320)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--val_split", type=float, default=0.2)
    p_train.add_argument("--device", type=str, default=_MAC_DEFAULT_DEVICE, help="Device id/name, e.g. '0', 'cpu', or 'mps'")
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--workers", type=int, default=None, help="Dataloader workers (default: 0 on mps, else 8)")
    p_train.add_argument("--amp", type=lambda v: v.lower() in ("1","true","yes"), default=None, help="Use AMP mixed precision (default: False on mps, else True)")

    p_pred = sub.add_parser("predict", help="Run inference using trained weights")
    p_pred.add_argument("--weights", type=Path, default=None, help="Path to .pt weights (default: newest runs/detect weights)")
    p_pred.add_argument("--source", type=Path, default=None, help="Image file or directory to run inference on")
    p_pred.add_argument("--conf", type=float, default=0.25)
    p_pred.add_argument("--iou", type=float, default=0.45)
    p_pred.add_argument("--save", action="store_true", help="Let ultralytics save visualized predictions")
    p_pred.add_argument("--device", type=str, default=_MAC_DEFAULT_DEVICE, help="Device id/name, e.g. '0', 'cpu', or 'mps'")
    p_pred.add_argument("--imgsz", type=int, default=320, help="Inference size (frames already 320)")
    p_pred.add_argument("--max_det", type=int, default=20, help="Max detections per image")
    p_pred.add_argument("--classes", type=int, nargs='*', default=None, help="Restrict to class ids, e.g. 0")

    p_fp = sub.add_parser(
        "extract-false-positives",
        aliases=["false-positives"],
        help="Run trained weights over video(s) and save frames that produce detections",
    )
    p_fp.add_argument("--weights", type=Path, default=None, help="Path to .pt weights (default: newest runs/detect weights)")
    p_fp.add_argument("--source", type=Path, default=None, help="MP4/video file or directory of videos")
    p_fp.add_argument("--output_dir", type=Path, default=Path("false_positives"), help="Directory for extracted frames and detections.csv")
    p_fp.add_argument("--conf", type=float, default=0.25)
    p_fp.add_argument("--iou", type=float, default=0.45)
    p_fp.add_argument("--device", type=str, default=_MAC_DEFAULT_DEVICE, help="Device id/name, e.g. '0', 'cpu', or 'mps'")
    p_fp.add_argument("--imgsz", type=int, default=320, help="Inference size")
    p_fp.add_argument("--max_det", type=int, default=20, help="Max detections per frame")
    p_fp.add_argument("--classes", type=int, nargs='*', default=None, help="Restrict to class ids, e.g. 0")
    p_fp.add_argument("--frame_stride", type=int, default=1, help="Run inference every N frames")
    p_fp.add_argument("--min_gap_frames", type=int, default=0, help="Minimum frame gap between saved detections")
    p_fp.add_argument("--save_annotated", action="store_true", help="Also save frames with predicted boxes drawn")
    p_fp.add_argument("--jpeg_quality", type=int, default=95, help="JPEG quality for saved frames (1-100)")
    p_fp.add_argument("--limit", type=int, default=None, help="Stop after saving this many frames")
    p_fp.add_argument("--verbose", action="store_true", help="Print per-detection progress")

    args = ap.parse_args()

    if args.config is None:
        for candidate in _unique_paths([Path.cwd() / "settings.conf", _SCRIPT_DIR / "settings.conf"]):
            if candidate.exists():
                args.config = candidate
                print(f"[CONFIG] Using {candidate}")
                break

    # Helper: check if a CLI flag was supplied (so CLI overrides conf)
    def _cli_supplied(flag: str) -> bool:
        return any(tok == flag for tok in sys.argv[1:])

    def _attr_supplied(attr: str) -> bool:
        return _cli_supplied(f"--{attr}") or _cli_supplied(f"--{attr.replace('_','-')}")

    # Load and apply config values if provided
    if args.config:
        cfg = configparser.ConfigParser()
        cfg.read(args.config)
        sect = args.cmd
        if sect not in cfg and sect.replace("-", "_") in cfg:
            sect = sect.replace("-", "_")
        if sect not in cfg and args.cmd in ("extract-false-positives", "false-positives") and "false_positives" in cfg:
            sect = "false_positives"
        if sect in cfg:
            c = cfg[sect]
            def set_if(flag: str, attr: str, cast):
                if flag in c and not _attr_supplied(attr):
                    setattr(args, attr, cast(c[flag]))

            if args.cmd == "train":
                set_if("positives_dir", "positives_dir", Path)
                set_if("bbox_file", "bbox_file", Path)
                set_if("yolo_root", "yolo_root", Path)
                if "negatives_dir" in c and not _attr_supplied("negatives_dir"):
                    nd = c["negatives_dir"].strip()
                    setattr(args, "negatives_dir", Path(nd) if nd else None)
                set_if("model", "model", str)
                set_if("epochs", "epochs", int)
                set_if("imgsz", "imgsz", int)
                set_if("batch", "batch", int)
                set_if("val_split", "val_split", float)
                set_if("device", "device", str)
                set_if("seed", "seed", int)
                if "workers" in c and not _attr_supplied("workers"):
                    w = c["workers"].strip()
                    setattr(args, "workers", int(w) if w else None)
                if "amp" in c and not _attr_supplied("amp"):
                    a = c["amp"].strip().lower()
                    setattr(args, "amp", a in ("1","true","yes"))
            elif args.cmd == "predict":
                set_if("weights", "weights", Path)
                set_if("source", "source", Path)
                set_if("conf", "conf", float)
                set_if("iou", "iou", float)
                set_if("device", "device", str)
                set_if("imgsz", "imgsz", int)
                set_if("max_det", "max_det", int)
                if "classes" in c and not _attr_supplied("classes"):
                    cls_vals = [int(x) for x in c["classes"].replace(","," ").split() if x.strip()]
                    setattr(args, "classes", cls_vals)
                if "save" in c and not _attr_supplied("save"):
                    setattr(args, "save", c.getboolean("save"))
            elif args.cmd in ("extract-false-positives", "false-positives"):
                set_if("weights", "weights", Path)
                set_if("source", "source", Path)
                set_if("output_dir", "output_dir", Path)
                set_if("conf", "conf", float)
                set_if("iou", "iou", float)
                set_if("device", "device", str)
                set_if("imgsz", "imgsz", int)
                set_if("max_det", "max_det", int)
                set_if("frame_stride", "frame_stride", int)
                set_if("min_gap_frames", "min_gap_frames", int)
                set_if("jpeg_quality", "jpeg_quality", int)
                if "classes" in c and not _attr_supplied("classes"):
                    cls_vals = [int(x) for x in c["classes"].replace(","," ").split() if x.strip()]
                    setattr(args, "classes", cls_vals)
                if "save_annotated" in c and not _attr_supplied("save_annotated"):
                    setattr(args, "save_annotated", c.getboolean("save_annotated"))
                if "verbose" in c and not _attr_supplied("verbose"):
                    setattr(args, "verbose", c.getboolean("verbose"))
                if "limit" in c and not _attr_supplied("limit"):
                    lim = c["limit"].strip()
                    setattr(args, "limit", int(lim) if lim else None)

    # Validate required args after applying config
    if args.cmd == "train":
        if args.positives_dir is None or args.bbox_file is None:
            ap.error("--positives_dir and --bbox_file are required (via CLI or --conf [train] section)")
    elif args.cmd == "predict":
        if not _attr_supplied("weights"):
            args.weights = find_latest_yolo_weights()
            print(f"[WEIGHTS] Using latest YOLO weights: {args.weights}")
        else:
            args.weights = find_latest_yolo_weights(args.weights)
        if args.source is None:
            ap.error("--source is required (via CLI or --conf [predict] section)")
    elif args.cmd in ("extract-false-positives", "false-positives"):
        if not _attr_supplied("weights"):
            args.weights = find_latest_yolo_weights()
            print(f"[WEIGHTS] Using latest YOLO weights: {args.weights}")
        else:
            args.weights = find_latest_yolo_weights(args.weights)
        if args.source is None:
            ap.error("--source is required (via CLI or --conf [extract_false_positives] section)")
    
    if args.cmd == "train":
        det = YOLOBBoxDetector(
            positives_dir=args.positives_dir,
            bbox_file=args.bbox_file,
            yolo_root=args.yolo_root,
            negatives_dir=args.negatives_dir,
            seed=args.seed,
        )
        model = det.train(
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            val_split=args.val_split,
            device=args.device,
            verbose=True,
            workers=args.workers,
            amp=args.amp,
        )
        print("[DONE] Training complete. Best weights saved under ultralytics runs directory.")

    elif args.cmd == "predict":
        if YOLO is None:
            raise SystemExit("ultralytics not installed. pip install ultralytics")
        model = YOLO(str(args.weights))
        results = model.predict(
            source=str(args.source),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
            classes=args.classes,
            agnostic_nms=False,
            save=args.save,
            device=args.device,
            verbose=False,
        )
        # Print a brief summary: number of detections per image
        for i, r in enumerate(results):
            n = 0
            try:
                n = 0 if r.boxes is None else int(r.boxes.shape[0])
            except Exception:
                try:
                    n = len(r.boxes)
                except Exception:
                    n = 0
            print(f"[PRED] item={i} detections={n}")

    elif args.cmd in ("extract-false-positives", "false-positives"):
        saved, processed, csv_path = extract_false_positive_frames(
            weights=args.weights,
            source=args.source,
            output_dir=args.output_dir,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
            classes=args.classes,
            device=args.device,
            frame_stride=args.frame_stride,
            min_gap_frames=args.min_gap_frames,
            save_annotated=args.save_annotated,
            jpeg_quality=args.jpeg_quality,
            limit=args.limit,
            verbose=args.verbose,
        )
        print(f"[DONE] Processed frames={processed}; saved false-positive candidates={saved}; metadata={csv_path}")
