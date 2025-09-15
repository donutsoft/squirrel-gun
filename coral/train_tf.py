import argparse
import os
from pathlib import Path
from typing import Optional, Iterable

import tensorflow as tf



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an image classification model with TensorFlow/Keras.")

    # Dataset
    p.add_argument("--dataset", type=str, default=None,
                   help="Path to dataset root. If it contains subfolders 'train' and 'val', those are used."
                        " Otherwise, a validation split is created from this folder.")
    p.add_argument("--train_dir", type=str, default=None, help="Override: path to training images directory.")
    p.add_argument("--val_dir", type=str, default=None, help="Override: path to validation images directory.")
    p.add_argument("--image_size", type=int, nargs=2, default=(224, 224), help="Image size H W.")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    p.add_argument("--val_split", type=float, default=0.1,
                   help="Validation split used only when a single dataset directory is provided.")
    p.add_argument("--single_class_label", type=str, default=None,
                   help="Treat provided directory(ies) as a flat single-class dataset with this label."
                        " Use with --dataset (flat), or with --train_dir/--val_dir (both flat).")

    # Model
    p.add_argument("--base", type=str, default="none",
                   choices=["none", "mobilenet_v2", "efficientnet_b0"],
                   help="Backbone architecture. 'none' builds a small CNN from scratch.")
    p.add_argument("--freeze_base", action="store_true", help="Freeze base (transfer learning).")
    p.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for classifier head.")

    # Training
    p.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    p.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    p.add_argument("--early_stopping", type=int, default=5, help="Early stopping patience (0 to disable).")
    p.add_argument("--mixed_precision", action="store_true", help="Enable mixed precision training if supported.")

    # Augmentation
    p.add_argument("--augment", action="store_true", help="Enable simple data augmentation.")

    # Output / Export
    p.add_argument("--out_dir", type=str, default="runs_tf", help="Output directory for artifacts.")
    p.add_argument("--name", type=str, default="exp", help="Run name (subfolder of out_dir).")
    p.add_argument("--export_tflite", action="store_true", help="Export a float32 TFLite model.")
    p.add_argument("--quantize_int8", action="store_true",
                   help="Export an int8 quantized TFLite model (uses representative dataset).")
    p.add_argument("--rep_samples", type=int, default=100,
                   help="Representative samples for int8 calibration (from training set).")

    return p.parse_args()



def enable_mixed_precision():
    try:
        from tensorflow.keras import mixed_precision as mp
        mp.set_global_policy("mixed_float16")
        print("[INFO] Mixed precision enabled (mixed_float16).")
    except Exception as e:
        print(f"[WARN] Could not enable mixed precision: {e}")


def make_datasets(args: argparse.Namespace):
    img_h, img_w = args.image_size
    batch_size = args.batch_size
    autotune = tf.data.AUTOTUNE

    def list_image_files(root: Path) -> list[str]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
        files: list[str] = []
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(str(p))
        if not files:
            raise ValueError(f"No images found in directory {root}. Allowed formats: {sorted(exts)}")
        files.sort()
        return files

    def make_flat_ds(dir_path: Path, label_index: int, shuffle: bool) -> tf.data.Dataset:
        files = list_image_files(dir_path)
        ds = tf.data.Dataset.from_tensor_slices(files)
        if shuffle:
            ds = ds.shuffle(buffer_size=len(files), seed=1337, reshuffle_each_iteration=True)
        def _load(path):
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, (img_h, img_w))
            img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]; model rescales further
            label = tf.cast(label_index, tf.int32)
            return img, label
        ds = ds.map(_load, num_parallel_calls=autotune)
        ds = ds.batch(batch_size)
        return ds

    def split_flat_dir(dir_path: Path, val_split: float) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        files = list_image_files(dir_path)
        n = len(files)
        k = int(round(n * (1.0 - val_split)))
        # deterministic split
        train_files = files[:k]
        val_files = files[k:]
        def ds_from(files_list: list[str], shuffle: bool) -> tf.data.Dataset:
            ds = tf.data.Dataset.from_tensor_slices(files_list)
            if shuffle:
                ds = ds.shuffle(buffer_size=len(files_list), seed=1337, reshuffle_each_iteration=True)
            def _load(path):
                img = tf.io.read_file(path)
                img = tf.image.decode_image(img, channels=3, expand_animations=False)
                img = tf.image.resize(img, (img_h, img_w))
                img = tf.image.convert_image_dtype(img, tf.float32)
                label = tf.cast(0, tf.int32)
                return img, label
            ds = ds.map(_load, num_parallel_calls=autotune).batch(batch_size)
            return ds
        return ds_from(train_files, True), ds_from(val_files, False)

    if args.train_dir and args.val_dir:
        train_dir = Path(args.train_dir)
        val_dir = Path(args.val_dir)
        assert train_dir.is_dir(), f"Train dir not found: {train_dir}"
        assert val_dir.is_dir(), f"Val dir not found: {val_dir}"

        if args.single_class_label:
            train_ds = make_flat_ds(train_dir, label_index=0, shuffle=True)
            val_ds = make_flat_ds(val_dir, label_index=0, shuffle=False)
            class_names = [args.single_class_label]
            num_classes = 1
        else:
            train_ds = tf.keras.utils.image_dataset_from_directory(
                str(train_dir),
                image_size=(img_h, img_w),
                batch_size=batch_size,
                shuffle=True,
            )
            val_ds = tf.keras.utils.image_dataset_from_directory(
                str(val_dir),
                image_size=(img_h, img_w),
                batch_size=batch_size,
                shuffle=False,
            )
            class_names = train_ds.class_names
            num_classes = len(class_names)
    else:
        assert args.dataset, "Provide --dataset or both --train_dir and --val_dir"
        root = Path(args.dataset)
        if (root / "train").is_dir() and (root / "val").is_dir():
            if args.single_class_label:
                train_ds = make_flat_ds(root / "train", label_index=0, shuffle=True)
                val_ds = make_flat_ds(root / "val", label_index=0, shuffle=False)
                class_names = [args.single_class_label]
                num_classes = 1
            else:
                train_ds = tf.keras.utils.image_dataset_from_directory(
                    str(root / "train"), image_size=(img_h, img_w), batch_size=batch_size, shuffle=True
                )
                val_ds = tf.keras.utils.image_dataset_from_directory(
                    str(root / "val"), image_size=(img_h, img_w), batch_size=batch_size, shuffle=False
                )
                class_names = train_ds.class_names
                num_classes = len(class_names)
        else:
            # Single directory, split
            if args.single_class_label:
                train_ds, val_ds = split_flat_dir(root, args.val_split)
                class_names = [args.single_class_label]
                num_classes = 1
            else:
                train_ds = tf.keras.utils.image_dataset_from_directory(
                    str(root),
                    validation_split=args.val_split,
                    subset="training",
                    seed=1337,
                    image_size=(img_h, img_w),
                    batch_size=batch_size,
                    shuffle=True,
                )
                val_ds = tf.keras.utils.image_dataset_from_directory(
                    str(root),
                    validation_split=args.val_split,
                    subset="validation",
                    seed=1337,
                    image_size=(img_h, img_w),
                    batch_size=batch_size,
                    shuffle=False,
                )
                class_names = train_ds.class_names
                num_classes = len(class_names)

    # Cache, shuffle, prefetch
    train_ds = train_ds.cache().shuffle(1024).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    return train_ds, val_ds, class_names, num_classes


def build_model(args: argparse.Namespace, num_classes: int) -> tf.keras.Model:
    img_h, img_w = args.image_size
    inputs = tf.keras.Input(shape=(img_h, img_w, 3))

    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)

    if args.augment:
        x = tf.keras.layers.RandomFlip("horizontal")(x)
        x = tf.keras.layers.RandomRotation(0.05)(x)
        x = tf.keras.layers.RandomZoom(0.1)(x)

    base = None
    base_name = args.base.lower()
    if base_name == "mobilenet_v2":
        try:
            base = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(img_h, img_w, 3), weights="imagenet")
        except Exception as e:
            print(f"[WARN] Could not load pretrained weights for MobileNetV2: {e}. Using random init.")
            base = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(img_h, img_w, 3), weights=None)
    elif base_name == "efficientnet_b0":
        try:
            base = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(img_h, img_w, 3), weights="imagenet")
        except Exception as e:
            print(f"[WARN] Could not load pretrained weights for EfficientNetB0: {e}. Using random init.")
            base = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(img_h, img_w, 3), weights=None)

    if base is None:
        # Small CNN from scratch
        x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    else:
        base.trainable = not args.freeze_base
        x = base(x, training=not args.freeze_base)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

    if args.dropout > 0:
        x = tf.keras.layers.Dropout(args.dropout)(x)
    dtype = "float32"
    if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
        # Classifier must output float32 in mixed precision
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Activation("linear", dtype="float32")(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype=dtype)(x)
    else:
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model


def compile_model(model: tf.keras.Model, args: argparse.Namespace):
    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def make_callbacks(out_dir: Path, patience: int) -> list:
    cbs = []
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(out_dir / "best.keras"),
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
        verbose=1,
    )
    cbs.append(ckpt)
    if patience and patience > 0:
        es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=patience, restore_best_weights=True)
        cbs.append(es)
    return cbs


def export_savedmodel_and_labels(model: tf.keras.Model, class_names: list[str], out_dir: Path):
    sm_dir = out_dir / "saved_model"
    sm_dir.mkdir(parents=True, exist_ok=True)
    # Keras 3 prefers Model.export() for SavedModel
    try:
        if hasattr(model, "export"):
            model.export(str(sm_dir))
        else:
            raise AttributeError("Model has no export method")
    except Exception as e:
        print(f"[WARN] model.export failed ({e}); falling back to tf.saved_model.save")
        tf.saved_model.save(model, str(sm_dir))
    labels_path = out_dir / "labels.txt"
    labels_path.write_text("\n".join(class_names))
    print(f"[INFO] SavedModel -> {sm_dir}")
    print(f"[INFO] Labels -> {labels_path}")


def representative_dataset(ds: tf.data.Dataset, sample_count: int) -> Iterable[dict[str, tf.Tensor]]:
    # Important: our model includes a Rescaling(1/255) layer at the input.
    # For proper INT8 calibration, feed calibration samples in the same scale
    # expected by the model input, i.e., raw 0..255 (float32) without extra division.
    taken = ds.unbatch().take(sample_count)
    for batch in taken:
        image = batch[0] if isinstance(batch, (tuple, list)) else batch
        image = tf.cast(image, tf.float32)  # keep 0..255 range
        image = tf.expand_dims(image, 0)
        yield [image]


def export_tflite(model: tf.keras.Model, out_dir: Path, float32: bool = True, int8: bool = False,
                  rep_ds: Optional[tf.data.Dataset] = None, rep_samples: int = 100):
    out_dir.mkdir(parents=True, exist_ok=True)
    if float32:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = conv.convert()
        (out_dir / "model_float32.tflite").write_bytes(tflite_model)
        print(f"[INFO] Exported TFLite (float32): {(out_dir / 'model_float32.tflite')}")

    if int8:
        assert rep_ds is not None, "Representative dataset is required for int8 quantization."
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = lambda: representative_dataset(rep_ds, rep_samples)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.uint8
        conv.inference_output_type = tf.uint8
        tflite_int8 = conv.convert()
        (out_dir / "model_int8.tflite").write_bytes(tflite_int8)
        print(f"[INFO] Exported TFLite (int8): {(out_dir / 'model_int8.tflite')}")


def main():
    args = parse_args()

    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mixed_precision:
        enable_mixed_precision()

    print("[INFO] Loading datasets...")
    train_ds, val_ds, class_names, num_classes = make_datasets(args)
    print(f"[INFO] Classes: {class_names}")

    print("[INFO] Building model...")
    model = build_model(args, num_classes)
    compile_model(model, args)
    model.summary(print_fn=lambda s: (out_dir / "model_summary.txt").open("a").write(s + "\n"))

    print("[INFO] Training...")
    cbs = make_callbacks(out_dir, args.early_stopping)
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs)

    # Save final and best
    model.save(out_dir / "last.keras")
    print(f"[INFO] Saved last model -> {out_dir / 'last.keras'}")

    # Reload best if it exists
    best_path = out_dir / "best.keras"
    if best_path.exists():
        try:
            model = tf.keras.models.load_model(best_path)
            print("[INFO] Loaded best checkpoint for export.")
        except Exception as e:
            print(f"[WARN] Could not load best model, using last: {e}")

    export_savedmodel_and_labels(model, class_names, out_dir)

    if args.export_tflite or args.quantize_int8:
        print("[INFO] Exporting TFLite models...")
        # Representative dataset from training set (preprocessed already)
        rep_ds = train_ds.take(max(1, args.rep_samples // max(1, args.batch_size)))
        export_tflite(
            model,
            out_dir,
            float32=args.export_tflite,
            int8=args.quantize_int8,
            rep_ds=rep_ds,
            rep_samples=args.rep_samples,
        )

    # Save training curves
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
