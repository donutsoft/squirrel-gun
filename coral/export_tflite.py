import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import tensorflow as tf
import numpy as np


EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Keras/SavedModel to TFLite (float32 or INT8).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--saved_model", type=str, help="Path to SavedModel directory.")
    src.add_argument("--keras", type=str, help="Path to .keras model file.")

    p.add_argument("--export_float32", action="store_true", help="Export float32 TFLite.")
    p.add_argument("--export_int8", action="store_true", help="Export per-tensor INT8 TFLite.")

    p.add_argument("--calib_dir", type=str, default=None, help="Directory of images for INT8 calibration.")
    p.add_argument("--rep_samples", type=int, default=100, help="Number of calibration samples to use.")
    p.add_argument("--image_size", type=int, nargs=2, default=None, help="Override model input size H W.")
    p.add_argument("--out_dir", type=str, default=".", help="Output directory.")
    return p.parse_args()


def detect_input_size_from_keras(path: Path) -> Optional[Tuple[int, int]]:
    try:
        model = tf.keras.models.load_model(path)
        ishape = model.input_shape
        if isinstance(ishape, list):
            ishape = ishape[0]
        H, W = int(ishape[1]), int(ishape[2])
        return H, W
    except Exception:
        return None


def detect_input_size_from_saved_model(path: Path) -> Optional[Tuple[int, int]]:
    try:
        sm = tf.saved_model.load(str(path))
        infer = sm.signatures.get("serving_default", None)
        if infer is None:
            return None
        _, kwargs = infer.structured_input_signature
        if not kwargs:
            return None
        spec = list(kwargs.values())[0]
        shape = spec.shape
        H, W = int(shape[1]), int(shape[2])
        return H, W
    except Exception:
        return None


def representative_dataset(calib_dir: Path, H: int, W: int, sample_count: int) -> Iterable[List[np.ndarray]]:
    paths = [p for p in calib_dir.rglob("*") if p.suffix.lower() in EXTS]
    paths.sort()
    if not paths:
        raise ValueError(f"No images found in calibration dir: {calib_dir}")
    from PIL import Image

    for p in paths[:sample_count]:
        im = Image.open(p).convert("RGB").resize((W, H))
        arr = np.array(im, dtype=np.float32)  # 0..255, model has Rescaling(1/255)
        arr = np.expand_dims(arr, 0)
        yield [arr]


def export_float32(converter: tf.lite.TFLiteConverter, out_path: Path):
    tflite = converter.convert()
    out_path.write_bytes(tflite)
    print(f"[INFO] Wrote float32 TFLite -> {out_path}")


def export_int8(converter: tf.lite.TFLiteConverter, calib_dir: Path, H: int, W: int, rep_samples: int, out_path: Path):
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(calib_dir, H, W, rep_samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite = converter.convert()
    out_path.write_bytes(tflite)
    print(f"[INFO] Wrote INT8 TFLite -> {out_path}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.export_float32 and not args.export_int8:
        print("[INFO] No export type specified; defaulting to --export_float32")
        args.export_float32 = True

    converter = None
    input_size: Optional[Tuple[int, int]] = tuple(args.image_size) if args.image_size else None

    if args.keras:
        kpath = Path(args.keras)
        model = tf.keras.models.load_model(kpath)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        if input_size is None:
            input_size = detect_input_size_from_keras(kpath)
    else:
        smdir = Path(args.saved_model)
        converter = tf.lite.TFLiteConverter.from_saved_model(str(smdir))
        if input_size is None:
            input_size = detect_input_size_from_saved_model(smdir)

    if input_size is None:
        print("[WARN] Could not determine input size; defaulting to 224x224. Use --image_size H W to override.")
        input_size = (224, 224)
    H, W = input_size
    print(f"[INFO] Using input size: {H}x{W}")

    if args.export_float32:
        out_path = out_dir / "model_float32.tflite"
        export_float32(converter, out_path)

    if args.export_int8:
        if not args.calib_dir:
            raise SystemExit("--calib_dir is required for --export_int8")
        out_path = out_dir / "model_int8.tflite"
        export_int8(converter, Path(args.calib_dir), H, W, args.rep_samples, out_path)


if __name__ == "__main__":
    main()

