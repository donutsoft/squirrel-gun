'''
python coral/coral_run.py 
--model coral/runs_tf/crops_flat/model_float32.tflite 
--labels coral/runs_tf/crops_flat/labels.txt 
--backend tflite 
--image coral/stills_cropped/images/rec_20250911_104731_000022.jpg

OR 

python coral/coral_run.py 
--model runs_tf/crops_flat/model_int8_edgetpu.tflite 
--labels runs_tf/crops_flat/labels.txt 
--backend edgetpu 
--dir path/to/images

'''

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference on images using a trained classifier with TF or TFLite backends.")
    p.add_argument("--model", type=str, required=True, help="Path to model (.keras, SavedModel dir, or .tflite)")
    p.add_argument("--labels", type=str, default=None, help="Path to labels.txt (one label per line)")
    p.add_argument("--image", type=str, default=None, help="Single image path")
    p.add_argument("--dir", type=str, default=None, help="Directory of images (recursive)")
    p.add_argument("--image_size", type=int, nargs=2, default=(224, 224), help="Input size H W (default 224 224)")
    p.add_argument("--backend", type=str, choices=["auto", "keras", "savedmodel", "tflite", "edgetpu"], default="auto",
                   help="Inference backend: auto-detect or force a specific backend")
    p.add_argument("--topk", type=int, default=3, help="Show top-K predictions")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold for POSITIVE/NEGATIVE decision")
    p.add_argument("--positive_label", type=str, default=None,
                   help="If set, decision is based on this label's probability; otherwise uses top-1")
    return p.parse_args()


def load_labels(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    fp = Path(path)
    if not fp.exists():
        print(f"[WARN] labels file not found: {fp}")
        return None
    return [ln.strip() for ln in fp.read_text().splitlines() if ln.strip()]


def detect_backend(model_path: Path) -> str:
    if model_path.suffix == ".tflite":
        return "tflite"
    if model_path.is_file() and model_path.suffix == ".keras":
        return "keras"
    if model_path.is_dir() and (model_path / "saved_model.pb").exists():
        return "savedmodel"
    return "unknown"


def load_backend(backend: str, model_path: Path):
    if backend in ("keras", "savedmodel"):
        # Lazy import TF to allow tflite-only environments
        import tensorflow as tf
        if backend == "keras":
            model = tf.keras.models.load_model(model_path)
            return (backend, model)
        else:
            sm = tf.saved_model.load(str(model_path))
            infer = sm.signatures.get("serving_default", None)
            if infer is None:
                raise ValueError("SavedModel has no 'serving_default' signature")
            return (backend, infer)
    elif backend in ("tflite", "edgetpu"):
        try:
            import tflite_runtime.interpreter as tflite
            from tflite_runtime.interpreter import load_delegate
        except Exception:
            # Fall back to TF Lite from TF package
            from tensorflow.lite import Interpreter as _Interpreter
            class _Wrap:
                def __init__(self, model_path, experimental_delegates=None):
                    self._interp = _Interpreter(model_path=model_path)
                def allocate_tensors(self):
                    return self._interp.allocate_tensors()
                def get_input_details(self):
                    return self._interp.get_input_details()
                def get_output_details(self):
                    return self._interp.get_output_details()
                def set_tensor(self, idx, val):
                    return self._interp.set_tensor(idx, val)
                def invoke(self):
                    return self._interp.invoke()
                def get_tensor(self, idx):
                    return self._interp.get_tensor(idx)
            tflite = type("_tfl_mod", (), {"Interpreter": _Wrap})
            def load_delegate(name):
                raise RuntimeError("Delegates not available in TensorFlow-lite shim")

        delegates = []
        if backend == "edgetpu":
            try:
                delegates = [load_delegate("libedgetpu.so.1")]
            except Exception as e:
                raise SystemExit(f"Failed to load Edge TPU delegate: {e}")
        interpreter = tflite.Interpreter(model_path=str(model_path), experimental_delegates=delegates)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return (backend, (interpreter, input_details, output_details))
    else:
        raise ValueError(f"Unknown backend: {backend}")


def load_image(path: Path, size: Tuple[int, int]) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = im.resize((size[1], size[0]))
        return np.asarray(im)


def predict_batch(backend, handle, batch: np.ndarray) -> np.ndarray:
    # Keras/SavedModel expect float32 0..255 since model has internal Rescaling(1/255)
    if backend == "keras":
        import tensorflow as tf
        return handle.predict(batch.astype(np.float32), verbose=0)
    if backend == "savedmodel":
        import tensorflow as tf
        outputs = handle(tf.constant(batch.astype(np.float32)))
        if isinstance(outputs, dict):
            key = sorted(outputs.keys())[0]
            return outputs[key].numpy()
        return outputs.numpy()
    # TFLite/EdgeTPU branch
    interpreter, input_details, output_details = handle
    inp = input_details[0]
    out = output_details[0]
    # Match dtype: uint8 models expect 0..255 uint8; float32 expects 0..255 float32 (rescaling in graph)
    if inp["dtype"].__name__ == "uint8":
        data = batch.astype(np.uint8)
    else:
        data = batch.astype(np.float32)
    interpreter.set_tensor(inp["index"], data)
    interpreter.invoke()
    probs = interpreter.get_tensor(out["index"])
    return probs


def decide_positive(pr: np.ndarray, labels: Optional[List[str]], threshold: float, positive_label: Optional[str]) -> Tuple[str, float, int]:
    if positive_label and labels:
        try:
            idx = labels.index(positive_label)
            prob = float(pr[idx])
            return ("POSITIVE" if prob >= threshold else "NEGATIVE", prob, idx)
        except ValueError:
            pass  # fallback to top-1
    top_idx = int(np.argmax(pr))
    top_prob = float(pr[top_idx])
    return ("POSITIVE" if top_prob >= threshold else "NEGATIVE", top_prob, top_idx)


def main():
    args = parse_args()
    model_path = Path(args.model)
    labels = load_labels(args.labels)

    backend = args.backend
    if backend == "auto":
        backend = detect_backend(model_path)
        if backend == "unknown":
            raise SystemExit("Could not auto-detect backend. Specify --backend explicitly.")

    kind, handle = load_backend(backend, model_path)

    # Collect images
    image_paths: List[Path] = []
    if args.image:
        image_paths.append(Path(args.image))
    if args.dir:
        for p in Path(args.dir).rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                image_paths.append(p)
    if not image_paths:
        raise SystemExit("Provide --image or --dir with images to run inference.")

    H, W = args.image_size
    batch = []
    batch_paths = []

    def flush():
        nonlocal batch, batch_paths
        if not batch:
            return
        arr = np.stack(batch, axis=0)
        probs = predict_batch(kind, handle, arr)
        for i, pth in enumerate(batch_paths):
            pr = probs[i]
            decision, prob, idx = decide_positive(pr, labels, args.threshold, args.positive_label)
            label_name = labels[idx] if labels and 0 <= idx < len(labels) else f"class_{idx}"
            # Compose top-K for context
            k = min(args.topk, pr.shape[-1])
            top_idx = np.argsort(-pr)[:k]
            top_pairs = [(labels[j] if labels and j < len(labels) else f"class_{j}", float(pr[j])) for j in top_idx]
            print(f"{pth} -> {decision} ({label_name} {prob:.3f}) top{k}={top_pairs}")
        batch = []
        batch_paths = []

    for p in image_paths:
        img = load_image(p, (H, W))
        batch.append(img)
        batch_paths.append(p)
        if len(batch) >= 16:
            flush()
    flush()


if __name__ == "__main__":
    main()
