"""
Train a YOLOv8 model to detect squirrels.

Usage (GPU):
  python scratch/train_squirrel.py --data data/squirrel.yaml --epochs 100 --imgsz 640 --batch 16 --device 0

Usage (CPU/Mac CPU):
  python scratch/train_squirrel.py --data data/squirrel.yaml --device cpu --epochs 50 --batch 8 --amp False

Dataset format (YOLO):
  dataset/
    images/
      train/  # jpg/png images
      val/
    labels/
      train/  # .txt files with: class x_center y_center width height (0-1 normalized)
      val/

Example data YAML (single class):
  path: dataset
  train: images/train
  val: images/val
  names: [squirrel]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def ensure_ultralytics() -> None:
    try:
        import ultralytics  # noqa: F401
    except Exception:
        print(
            "Ultralytics not installed. Install with: pip install -U ultralytics",
            file=sys.stderr,
        )
        raise


def maybe_write_minimal_yaml(yaml_path: Path, dataset_root: Path) -> None:
    """If `yaml_path` does not exist but `dataset_root` is given, write a minimal single-class YAML.

    The YAML assumes standard YOLO directory structure under dataset_root.
    """
    if yaml_path.exists():
        return
    if not dataset_root:
        return
    content = (
        f"path: {dataset_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names: [squirrel]\n"
    )
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(content)
    print(f"Wrote minimal data YAML at {yaml_path}")


def train(
    data_yaml: str,
    model: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "",
    project: str = "runs/train",
    name: str = "squirrel_yolov8n",
    amp: bool | str = True,
) -> None:
    """Run YOLOv8 training with sensible defaults for a single-class dataset."""
    ensure_ultralytics()
    from ultralytics import YOLO  # type: ignore

    model_obj = YOLO(model)
    print(
        f"Starting training -> model={model} data={data_yaml} epochs={epochs} imgsz={imgsz} batch={batch} device={device}"
    )
    model_obj.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        amp=amp,
    )
    # Optional: quick validation summary
    metrics = model_obj.val()
    print("Validation metrics:", metrics)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 to detect squirrels")
    p.add_argument("--data", required=True, help="Path to data YAML (e.g., data/squirrel.yaml)")
    p.add_argument("--model", default="yolov8n.pt", help="Base model or YAML (e.g., yolov8n.pt)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="", help="CUDA device id like 0, or 'cpu'")
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="squirrel_yolov8n")
    p.add_argument(
        "--amp",
        default=True,
        type=lambda x: str(x).lower() not in {"0", "false", "no"},
        help="Use mixed precision (set False on some CPUs/MPS)",
    )
    p.add_argument(
        "--dataset-root",
        default="",
        help="Optional dataset root to auto-write a minimal YAML if --data is missing",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    yaml_path = Path(args.data)
    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    if dataset_root is not None:
        maybe_write_minimal_yaml(yaml_path, dataset_root)
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Data YAML not found at {yaml_path}. Provide --dataset-root to auto-create or supply a valid YAML."
        )
    train(
        data_yaml=str(yaml_path),
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        amp=args.amp,
    )

if __name__ == "__main__":
    main()

