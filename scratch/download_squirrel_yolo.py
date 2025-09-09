"""
download_squirrel_yolo.py

Download the 'Squirrel' class from Open Images and export in YOLO format.

Usage:
    python download_squirrel_yolo.py --out ./squirrel_yolo --splits train val
    # Optional:
    python download_squirrel_yolo.py --out ./squirrel_yolo --max-samples 2000 --seed 42
    # If a dataset of the same name already exists and lacks detections, use:
    python download_squirrel_yolo.py --out ./squirrel_yolo --overwrite
"""

import argparse
import os
import fiftyone as fo
import fiftyone.zoo as foz


def parse_args():
    p = argparse.ArgumentParser(description="Download Open Images 'Squirrel' and export to YOLO format")
    p.add_argument("--out", "--output-dir", dest="out_dir", required=True, help="Export directory for YOLO dataset")
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        help="Which splits to download. Accepts train/validation/test and aliases val/valid",
    )
    p.add_argument("--max-samples", type=int, default=None, help="Max total samples (across splits)")
    p.add_argument("--seed", type=int, default=51, help="Random seed when sub-sampling")
    p.add_argument("--dataset-name", default="open-images-squirrel", help="FiftyOne dataset name to create/use")
    p.add_argument("--label-field", default=None, help="Detections field name (auto-detected if omitted)")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, delete any existing FiftyOne dataset with the same name before creating",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.overwrite and fo.dataset_exists(args.dataset_name):
        print(f"[info] Deleting existing dataset: {args.dataset_name}")
        fo.delete_dataset(args.dataset_name)

    os.makedirs(args.out_dir, exist_ok=True)

    # Normalize split aliases
    alias = {"val": "validation", "valid": "validation", "validation": "validation", "train": "train", "test": "test"}
    try:
        splits = [alias[s.lower()] for s in args.splits]
    except KeyError as e:
        valid = ", ".join(sorted(set(alias.values())))
        raise SystemExit(f"Invalid split '{e.args[0]}'. Use one of: {valid} (aliases: val, valid)")

    print("[info] Downloading Open Images with class 'Squirrel'… (this may take a while)")
    ds = foz.load_zoo_dataset(
        "open-images-v6",               # FiftyOne's maintained Open Images loader
        splits=splits,
        label_types=["detections"],     # get bounding boxes
        classes=["Squirrel"],           # only squirrel images
        only_matching=True,             # keep only images that actually have squirrels
        max_samples=args.max_samples,   # optional sub-sampling
        shuffle=args.max_samples is not None,
        seed=args.seed,
        dataset_name=args.dataset_name, # store in a named FO dataset for reuse
        overwrite=args.overwrite,       # ensure schema matches current request when set
    )

    print(ds)  # quick summary

    # Determine detections field robustly
    label_field = args.label_field
    if label_field is None:
        try:
            from fiftyone.core.labels import Detections as FODetections
            # Prefer official schema query for embedded Detections
            det_schema = ds.get_field_schema(embedded_doc_type=FODetections)
            det_fields = list(det_schema.keys())
        except Exception:
            det_fields = []

        # If none found via schema, fall back to inspecting a sample
        if not det_fields:
            sample = ds.first()
            if sample is not None:
                try:
                    from fiftyone.core.labels import Detections as FODetections
                    det_fields = [
                        name for name in sample.field_names if isinstance(sample[name], FODetections)
                    ]
                except Exception:
                    det_fields = []

        # Preferred names
        prefs = ["detections", "ground_truth", "annotations", "labels"]
        label_field = next((f for p in prefs for f in det_fields if f and f.lower() == p), None)
        if label_field is None:
            # Heuristic: ground_truth appears in schema text output
            schema_all = ds.get_field_schema()
            if "ground_truth" in schema_all:
                label_field = "ground_truth"

        if label_field is None and det_fields:
            label_field = det_fields[0]
        if label_field is None:
            available = ", ".join(schema_all.keys()) if 'schema_all' in locals() else "(unknown)"
            raise KeyError(
                "No Detections field found on samples. Pass --label-field explicitly (e.g., --label-field ground_truth).\n"
                f"Available top-level fields: {available}"
            )

    print(f"[info] Using detections field: {label_field}")

    # Export to YOLO format (works for YOLOv5/YOLOv8/YOLOv11)
    # Folder structure: out_dir/{images, labels} with txt label files per image + data.yaml
    print(f"[info] Exporting to YOLO format at: {args.out_dir}")
    # Export per split so the folder structure has images/{train,val,test}
    exported_splits = set()
    for s in splits:
        view = ds.match_tags(s)
        # Map 'validation' tag -> 'val' export split
        export_split = "val" if s == "validation" else s
        if len(view) == 0:
            print(f"[warn] No samples tagged '{s}', skipping export for this split")
            continue
        view.export(
            export_dir=args.out_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            split=export_split,
            classes=["Squirrel"],
        )
        exported_splits.add(export_split)

    # If no val split exported, fall back to using train for val so Ultralytics won't fail
    if "train" in exported_splits and "val" not in exported_splits:
        print("[warn] No validation split exported; will point val to train in data.yaml")
        exported_splits.add("val")

    # If you prefer separate YOLO folders per split, uncomment below:
    # for split in args.splits:
    #     view = ds.match_tags(split)  # tag per split is added by zoo
    #     split_out = os.path.join(args.out_dir, split)
    #     view.export(
    #         export_dir=split_out,
    #         dataset_type=fo.types.YOLOv5Dataset,
    #         label_field="detections",
    #         classes=["Squirrel"],
    #     )

    # Write a proper Ultralytics data.yaml referencing the exported splits
    train_rel = "images/train" if "train" in exported_splits else "images/val" if "val" in exported_splits else "images/test"
    val_rel = "images/val" if "val" in exported_splits else train_rel
    yaml_text = (
        f"train: {train_rel}\n"
        f"val: {val_rel}\n"
        f"names: [Squirrel]\n"
    )
    with open(os.path.join(args.out_dir, "data.yaml"), "w") as f:
        f.write(yaml_text)

    print("[done] YOLO dataset ready.")
    print("Tip: Train with Ultralytics:")
    print(f"  python scratch/train_squirrel.py --data {os.path.join(args.out_dir, 'data.yaml')} --epochs 50 --imgsz 640 --batch 16 --device 0")


if __name__ == "__main__":
    main()
