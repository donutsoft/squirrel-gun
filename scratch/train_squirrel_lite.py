"""
Train a lighter YOLOv8 model and optionally export for edge (Pi).

Examples:
  # Train small model at lower resolution
  python scratch/train_squirrel_lite.py --data data/squirrel.yaml --model yolov8n.pt --epochs 50 --imgsz 320 --batch 16 --device 0

  # Export to INT8 TFLite for fast CPU inference on Raspberry Pi
  python scratch/train_squirrel_lite.py --data data/squirrel.yaml \
    --model runs/train/squirrel_yolov8n/weights/best.pt --export --export-format tflite --int8 --export-imgsz 320
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
    # Export options
    do_export: bool = False,
    export_format: str = "tflite",
    export_int8: bool = True,
    export_imgsz: int | None = None,
) -> None:
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
    metrics = model_obj.val()
    print("Validation metrics:", metrics)

    if do_export:
        try:
            ex_imgsz = export_imgsz or imgsz
            print(
                f"Exporting -> format={export_format} int8={export_int8} imgsz={ex_imgsz} (data for PTQ: {data_yaml})"
            )
            export_path = model_obj.export(
                format=export_format,
                int8=export_int8,
                imgsz=ex_imgsz,
                data=data_yaml,
            )
            print(f"Exported model: {export_path}")
        except Exception as e:
            print(f"Export failed: {e}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 (lite) and export for Pi")
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
    # Export flags
    p.add_argument("--export", action="store_true", help="After training, export a deployable model")
    p.add_argument(
        "--export-format",
        default="tflite",
        choices=["tflite", "onnx", "openvino", "torchscript", "ncnn"],
        help="Export format for deployment",
    )
    p.add_argument("--int8", dest="export_int8", action="store_true", help="Use INT8 PTQ when supported")
    p.add_argument("--no-int8", dest="export_int8", action="store_false", help="Disable INT8 quantization")
    p.set_defaults(export_int8=True)
    p.add_argument("--export-imgsz", type=int, default=None, help="Override export image size")
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
        do_export=args.export,
        export_format=args.export_format,
        export_int8=args.export_int8,
        export_imgsz=args.export_imgsz,
    )


if __name__ == "__main__":
    main()

