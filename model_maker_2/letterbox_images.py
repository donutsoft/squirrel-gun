#!/usr/bin/env python3
"""Convert image files to the 320x320 gray-letterboxed training format.

By default this reads full-frame detection JPGs from the daemon detections
directory, skips annotated ``*_bbox`` copies and macOS ``._`` sidecar files,
and writes model-ready JPGs into ``dataset/detections_320``.
"""

from __future__ import annotations

import argparse
import configparser
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_SOURCE = Path("~/squirrel/squirrel-daemon/detections").expanduser()
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "dataset" / "detections_320"
DEFAULT_CONF = Path(__file__).with_name("settings.conf")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_imgsz(conf_path: Path) -> int:
    parser = configparser.ConfigParser()
    parser.read(conf_path)
    for section in ("train", "predict", "extract_false_positives"):
        if parser.has_option(section, "imgsz"):
            return parser.getint(section, "imgsz")
    return 320


def iter_images(source: Path, recursive: bool) -> Iterable[Path]:
    paths = source.rglob("*") if recursive else source.iterdir()
    for path in sorted(paths):
        if not path.is_file():
            continue
        if path.name.startswith("._"):
            continue
        if path.stem.endswith("_bbox"):
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        yield path


def unique_output_path(out_dir: Path, source_root: Path, image_path: Path, overwrite: bool) -> Path:
    if image_path.parent == source_root:
        name = image_path.with_suffix(".jpg").name
    else:
        rel = image_path.relative_to(source_root).with_suffix(".jpg")
        name = "__".join(rel.parts)

    out_path = out_dir / name
    if overwrite or not out_path.exists():
        return out_path

    stem = out_path.stem
    suffix = out_path.suffix
    idx = 1
    while True:
        candidate = out_dir / f"{stem}_{idx:03d}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def convert_images(
    source: Path,
    output: Path,
    target: int,
    quality: int,
    overwrite: bool,
    recursive: bool,
    dry_run: bool,
) -> int:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv-python is required; run from model_maker_2 with `uv run`.") from exc

    try:
        from yolo_bbox_detector import letterbox_frame
    except Exception as exc:
        raise RuntimeError("Could not import model_maker_2.yolo_bbox_detector letterbox helper.") from exc

    if not source.exists():
        raise FileNotFoundError(source)
    if not source.is_dir():
        raise NotADirectoryError(source)

    output.mkdir(parents=True, exist_ok=True)
    written = 0
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    for image_path in iter_images(source, recursive=recursive):
        out_path = unique_output_path(output, source, image_path, overwrite=overwrite)
        if out_path.exists() and not overwrite:
            continue

        if dry_run:
            print(f"{image_path} -> {out_path}")
            written += 1
            continue

        frame = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if frame is None:
            print(f"Skipping unreadable image: {image_path}", file=sys.stderr)
            continue

        model_frame = letterbox_frame(frame, target=target)
        ok = cv2.imwrite(str(out_path), model_frame, params)
        if not ok:
            print(f"Failed to write: {out_path}", file=sys.stderr)
            continue
        written += 1

    return written


def main(argv: list[str] | None = None) -> int:
    default_target = read_imgsz(DEFAULT_CONF)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help=f"Input image directory (default: {DEFAULT_SOURCE})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--target", type=int, default=default_target, help=f"Square output size from settings.conf (default: {default_target})")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality, 1-100 (default: 95)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--recursive", action="store_true", help="Search source recursively")
    parser.add_argument("--dry-run", action="store_true", help="Print conversions without writing images")
    args = parser.parse_args(argv)

    if args.target <= 0:
        parser.error("--target must be positive")
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be between 1 and 100")

    count = convert_images(
        source=args.source.expanduser(),
        output=args.output.expanduser(),
        target=args.target,
        quality=args.quality,
        overwrite=args.overwrite,
        recursive=args.recursive,
        dry_run=args.dry_run,
    )
    action = "Would write" if args.dry_run else "Wrote"
    print(f"{action} {count} image(s) at {args.target}x{args.target}: {args.output.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
