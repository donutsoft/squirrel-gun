import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import tensorflow as tf




def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a TensorFlow/Keras detector (RetinaNet via KerasCV) on a YOLO-format dataset.")

    # Dataset
    p.add_argument("--yolo_root", type=str, default="coral/squirrel_yolo",
                   help="Path to YOLO dataset root containing images/{train,val} and labels/{train,val}.")
    p.add_argument("--splits", type=str, nargs="*", default=["train", "val"], help="Dataset splits to use.")
    p.add_argument("--image_size", type=int, nargs=2, default=(512, 512), help="Model input size H W.")

    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=0.001)
    p.add_argument("--mixed_precision", action="store_true", help="Enable mixed precision if supported.")

    # Output
    p.add_argument("--out_dir", type=str, default="runs_det", help="Output directory.")
    p.add_argument("--name", type=str, default="retinanet", help="Run name.")

    return p.parse_args()


def enable_mixed_precision():
    try:
        from tensorflow.keras import mixed_precision as mp
        mp.set_global_policy("mixed_float16")
        print("[INFO] Mixed precision enabled (mixed_float16).")
    except Exception as e:
        print(f"[WARN] Failed to enable mixed precision: {e}")


def try_read_class_names(yolo_root: Path) -> List[str]:
    for fn in [yolo_root / "dataset.yaml", yolo_root / "data.yaml"]:
        if not fn.exists():
            continue
        try:
            text = fn.read_text()
        except Exception:
            continue
        # Inline list style: names: [A, B]
        for line in text.splitlines():
            ls = line.strip()
            if ls.startswith("names:") and "[" in ls and "]" in ls:
                inside = ls.split("[", 1)[1].split("]", 1)[0]
                vals = [x.strip().strip("'\"") for x in inside.split(",") if x.strip()]
                if vals:
                    return vals
        # Mapping style:
        names: Dict[int, str] = {}
        in_names = False
        for line in text.splitlines():
            if line.strip().startswith("names:") and ": [" not in line:
                in_names = True
                continue
            if in_names:
                if not line.startswith(" ") and not line.startswith("\t"):
                    break
                parts = line.strip().split(":", 1)
                if len(parts) == 2:
                    try:
                        idx = int(parts[0])
                        val = parts[1].strip().strip("'\"")
                        names[idx] = val
                    except ValueError:
                        pass
        if names:
            return [names[i] for i in sorted(names.keys())]
    return []


def yolo_label_path_for_image(yolo_root: Path, split: str, img_path: Path) -> Path:
    rel = img_path.relative_to(yolo_root / "images" / split)
    return yolo_root / "labels" / split / (rel.with_suffix(".txt").name)


def list_images(yolo_root: Path, split: str) -> List[Path]:
    img_dir = yolo_root / "images" / split
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Missing images dir: {img_dir}")
    out: List[Path] = []
    for p in img_dir.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            out.append(p)
    return out


def parse_yolo_file(path: Path) -> Tuple[List[int], List[List[float]]]:
    classes: List[int] = []
    boxes: List[List[float]] = []  # xywh normalized, x,y = top-left
    if not path.exists():
        return classes, boxes
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cid = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
        except ValueError:
            continue
        x = cx - w / 2.0
        y = cy - h / 2.0
        classes.append(cid)
        boxes.append([x, y, w, h])
    return classes, boxes


def make_dataset(yolo_root: Path, split: str, image_size: Tuple[int, int]):
    H, W = image_size
    imgs = list_images(yolo_root, split)

    def gen():
        for img_path in imgs:
            lbl_path = yolo_label_path_for_image(yolo_root, split, img_path)
            classes, boxes = parse_yolo_file(lbl_path)
            yield str(img_path), (classes, boxes)

    output_signature = (
        tf.TensorSpec(shape=(), dtype=tf.string),
        (
            tf.RaggedTensorSpec(shape=(None,), dtype=tf.int32),
            tf.RaggedTensorSpec(shape=(None, 4), dtype=tf.float32),
        ),
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    def load_and_pack(path, labels):
        classes_rt, boxes_rt = labels
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        img = tf.image.resize(img, (H, W))
        # Boxes are normalized; resizing not required for boxes.
        bboxes = {
            "classes": classes_rt,
            "boxes": boxes_rt,
        }
        return {"images": img, "bounding_boxes": bboxes}

    ds = ds.map(load_and_pack, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def build_model(num_classes: int, bbox_format: str = "xywh"):
    try:
        import keras_cv
    except Exception as e:
        raise SystemExit(
            "KerasCV is required for this detector scaffold. Install with: pip install keras-cv"
        ) from e

    try:
        # Prefer preset (may download weights)
        model = keras_cv.models.RetinaNet.from_preset(
            "resnet50_imagenet",
            num_classes=num_classes,
            bounding_box_format=bbox_format,
        )
    except Exception:
        # Fallback basic constructor
        model = keras_cv.models.RetinaNet(
            classes=num_classes,
            bounding_box_format=bbox_format,
            backbone="resnet50",
        )

    # Compile with reasonable defaults
    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, global_clipnorm=10.0)
    try:
        model.compile(
            optimizer=opt,
            classification_loss="focal",
            box_loss="smoothl1",
        )
    except Exception:
        # Older KerasCV API
        try:
            import keras_cv
            loss = keras_cv.losses.RetinaNetLoss(bounding_box_format=bbox_format)
            model.compile(optimizer=opt, loss=loss)
        except Exception as e:
            raise SystemExit(
                "Failed to compile RetinaNet. Please ensure a recent keras-cv is installed."
            ) from e

    return model


def main():
    args = parse_args()
    if args.mixed_precision:
        enable_mixed_precision()

    yolo_root = Path(args.yolo_root)
    class_names = try_read_class_names(yolo_root)
    if not class_names:
        raise SystemExit("Could not determine class names from dataset.yaml/data.yaml")
    num_classes = len(class_names)

    print(f"[INFO] Classes: {class_names}")
    print("[INFO] Building datasets...")
    train_ds = make_dataset(yolo_root, "train", tuple(args.image_size))
    val_ds = make_dataset(yolo_root, "val", tuple(args.image_size))

    train_ds = train_ds.shuffle(512).padded_batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.padded_batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    print("[INFO] Building model (RetinaNet)...")
    model = build_model(num_classes=num_classes, bbox_format="xywh")

    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(out_dir / "best.keras"),
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1,
    )

    print("[INFO] Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[ckpt],
    )

    # Save last
    model.save(out_dir / "last.keras")

    # Save labels
    (out_dir / "labels.txt").write_text("\n".join(class_names))

    # Save history
    try:
        import csv
        hist_file = out_dir / "history.csv"
        keys = list(history.history.keys())
        with hist_file.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch"] + keys)
            for i in range(len(history.history[keys[0]])):
                row = [i] + [history.history[k][i] for k in keys]
                w.writerow(row)
        print(f"[INFO] Wrote training history -> {hist_file}")
    except Exception as e:
        print(f"[WARN] Could not write history CSV: {e}")


if __name__ == "__main__":
    main()

