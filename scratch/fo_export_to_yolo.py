"""
Export a FiftyOne dataset folder to YOLOv5/YOLOv8 format and print a ready data.yaml.

Requirements:
  pip install fiftyone

Example:
  python scratch/fo_export_to_yolo.py \
    --src fiftyone \
    --out scratch/dataset_yolo \
    --val-frac 0.2

Inspect available label fields:
  python scratch/fo_export_to_yolo.py --src fiftyone --list-fields

Then train with:
  python scratch/train_squirrel.py --data scratch/dataset_yolo/data.yaml --epochs 100 --imgsz 640 --batch 16 --device 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional


def _list_detection_fields(dataset) -> List[str]:
    """Return names of fields that are Detections on samples."""
    try:
        schema = dataset.get_field_schema()
    except Exception:
        # Fallback: inspect first sample
        sample = dataset.first()
        if not sample:
            return []
        return [
            name
            for name in sample.field_names
            if sample[name].__class__.__module__.startswith("fiftyone.core.labels")
            and sample[name].__class__.__name__ == "Detections"
        ]
    out = []
    for name, ftype in schema.items():
        mod = getattr(ftype, "__module__", "")
        cls = getattr(ftype, "__name__", "")
        if mod.startswith("fiftyone.core.labels") and cls == "Detections":
            out.append(name)
    return out


def _choose_field(cands: List[str]) -> Optional[str]:
    """Pick a reasonable default detections field given candidates."""
    pref = ["detections", "ground_truth", "annotations", "labels"]
    for p in pref:
        for c in cands:
            if c.lower() == p:
                return c
    return cands[0] if cands else None


def export_to_yolo(src: Path, out: Path, val_frac: float = 0.2, label_field: Optional[str] = None) -> Path:
    try:
        import fiftyone as fo
        from fiftyone.core.labels import Detections, Detection  # noqa: F401
        from fiftyone import types as fot
    except Exception as e:
        raise RuntimeError("FiftyOne not installed. Run: pip install fiftyone") from e

    # Load dataset from a FiftyOne dataset directory
    dataset = fo.Dataset.from_dir(dataset_dir=str(src), dataset_type=fot.FiftyOneDataset)

    # Auto-detect a Detections field if not provided
    if label_field is None:
        cands = _list_detection_fields(dataset)
        chosen = _choose_field(cands)
        if not chosen:
            raise ValueError(
                "Could not find a Detections field; run with --list-fields or specify --label-field"
            )
        label_field = chosen

    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Create train/val splits
    val_frac = max(0.0, min(0.9, val_frac))
    if val_frac == 0:
        train_view = dataset.view()
        val_view = None
    else:
        train_view, val_view = dataset.random_split([1 - val_frac, val_frac], seed=42)

    # Export train split
    train_view.export(
        export_dir=str(out),
        dataset_type=fot.YOLOv5Dataset,
        label_field=label_field,
        split="train",
        classes=None,  # export all classes present
    )
    # Export val split (if any)
    if val_view is not None:
        val_view.export(
            export_dir=str(out),
            dataset_type=fot.YOLOv5Dataset,
            label_field=label_field,
            split="val",
            classes=None,
        )

    # Return path to generated data.yaml in export dir
    yaml_path = out / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found in {yaml_path.parent}")
    return yaml_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export FiftyOne dataset folder to YOLOv5/8 format")
    p.add_argument("--src", default="fiftyone", help="Path to the FiftyOne dataset folder")
    p.add_argument("--out", default="scratch/dataset_yolo", help="Output directory for YOLO dataset")
    p.add_argument("--val-frac", type=float, default=0.2, help="Validation split fraction (0-0.9)")
    p.add_argument(
        "--label-field",
        default=None,
        help="Name of the Detections field (auto-detected if omitted)",
    )
    p.add_argument("--list-fields", action="store_true", help="Only list available detection fields and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # If src already looks like a YOLO dataset (has data.yaml), use it directly
    src_path = Path(args.src)
    yolo_yaml = src_path / "data.yaml"
    if yolo_yaml.exists():
        print(f"Detected existing YOLO dataset. Using {yolo_yaml}")
        print(f"Exported YOLO dataset. data.yaml -> {yolo_yaml}")
        return
    if args.list_fields:
        try:
            import fiftyone as fo
            from fiftyone import types as fot  # noqa: F401
        except Exception:
            raise SystemExit("FiftyOne not installed. Run: pip install fiftyone")
        dataset = fo.Dataset.from_dir(dataset_dir=args.src, dataset_type=fot.FiftyOneDataset)
        cands = _list_detection_fields(dataset)
        if not cands:
            print("No Detections fields found on samples.")
        else:
            print("Detections fields:")
            for c in cands:
                print("-", c)
        return

    yaml = export_to_yolo(Path(args.src), Path(args.out), args.val_frac, args.label_field)
    print(f"Exported YOLO dataset. data.yaml -> {yaml}")


if __name__ == "__main__":
    main()
